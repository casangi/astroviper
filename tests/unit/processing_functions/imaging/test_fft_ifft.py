# run using eg
# python -m pytest ../astroviper/tests/domain/imaging/test_fft_ifft.py

import sys
import unittest
from types import SimpleNamespace

import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
    fft_lm_to_uv,
    fft_norm_img_xds,
    ifft_uv_to_lm,
)


class FFTTest(unittest.TestCase):
    def test_fft_ifft_round_trip(self):
        sky_lm = np.zeros((128, 128))
        sky_lm[64, 64] = 1
        axes = (0, 1)
        aperture_uv = fft_lm_to_uv(sky_lm, axes)
        sky_lm_round_trip = ifft_uv_to_lm(aperture_uv, axes)
        self.assertTrue(
            np.max(np.abs(sky_lm - sky_lm_round_trip)) < 1e-10, "Round trip failed"
        )
        sky_lm[50, 40] = 1
        aperture_uv = fft_lm_to_uv(sky_lm, axes)
        sky_lm_round_trip = ifft_uv_to_lm(aperture_uv, axes)
        self.assertTrue(
            np.max(np.abs(sky_lm - sky_lm_round_trip)) < 1e-10, "Round trip failed"
        )


def test_forward_image_fft_does_not_open_a_matplotlib_window(monkeypatch):
    """The worker-side FFT path has no interactive plotting side effect."""

    def fail_show():
        raise AssertionError("fft_norm_img_xds called matplotlib.pyplot.show()")

    monkeypatch.setitem(
        sys.modules,
        "matplotlib.pyplot",
        SimpleNamespace(show=fail_show),
    )
    image = xr.Dataset(
        {
            "SKY_MODEL": (
                ("time", "frequency", "polarization", "l", "m"),
                np.ones((1, 1, 1, 4, 4)),
            )
        },
        coords={
            "time": [0],
            "frequency": [1.5e9],
            "polarization": ["I"],
            "l": np.arange(4),
            "m": np.arange(4),
        },
        attrs={
            "type": "image_dataset",
            "data_groups": {"model": {"sky": "SKY_MODEL"}},
        },
    )

    result = fft_norm_img_xds(
        image,
        {"image_size": [4, 4], "fft_padding": 1.0},
        fft_backend="scipy",
    )

    assert result["VISIBILITY_MODEL"].shape == (1, 1, 1, 4, 4)


class FFTDtypeAndOverwriteTest(unittest.TestCase):
    """Guards for the 2026-08-16 multi-cycle OOM regression: the degrid path
    must stay in the image's complex precision (no float64-kernel promotion to
    complex128) and must be able to transform a plane in its own buffer."""

    def test_fft_preserves_single_precision(self):
        plane = np.zeros((64, 64), dtype=np.complex64)
        plane[32, 32] = 1
        uv = fft_lm_to_uv(plane, complex_dtype=np.complex64)
        self.assertEqual(uv.dtype, np.complex64)

    def test_inplace_kernel_division_does_not_promote(self):
        # The fft_norm_img_xds inner loop divides the complex64 plane by
        # float64 kernel arrays IN PLACE; the dtype must not change (the
        # out-of-place spelling promotes to complex128 and allocates two
        # full-grid temporaries).
        plane = np.ones((16, 16), dtype=np.complex64)
        kernel = np.linspace(1.0, 2.0, 16)  # float64, as produced by the GCF
        plane /= kernel[:, None]
        plane /= kernel[None, :]
        self.assertEqual(plane.dtype, np.complex64)

    def test_overwrite_input_matches_default(self):
        rng = np.random.default_rng(7)
        plane = (
            rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64))
        ).astype(np.complex64)
        expected = fft_lm_to_uv(plane.copy(), complex_dtype=np.complex64)
        got = fft_lm_to_uv(plane, complex_dtype=np.complex64, overwrite_input=True)
        np.testing.assert_array_equal(got, expected)

    def test_default_preserves_input(self):
        plane = np.ones((32, 32), dtype=np.complex64)
        original = plane.copy()
        fft_lm_to_uv(plane, complex_dtype=np.complex64)
        np.testing.assert_array_equal(plane, original)


class FourCorrelationSkyTest(unittest.TestCase):
    """The sky of four correlations stays complex through the inverse FFT (a cross
    hand is not conjugate symmetric on its own); two hands and the PSF are real."""

    def _image(self, labels, seed=0):
        import xarray as xr

        n_uv, n_pol = 20, len(labels)
        rng = np.random.default_rng(seed)
        # four planes drawn once, so that the two-hand grids are the first two of them
        grid = rng.normal(size=(1, 1, 4, n_uv, n_uv)) + 1j * rng.normal(
            size=(1, 1, 4, n_uv, n_uv)
        )
        grid = np.ascontiguousarray(grid[:, :, :n_pol])
        img_xds = xr.Dataset(
            {
                "VISIBILITY": (("time", "frequency", "polarization", "u", "v"), grid),
                "VISIBILITY_NORMALIZATION": (
                    ("time", "frequency", "polarization"),
                    np.full((1, 1, n_pol), 2.0),
                ),
                "UV_SAMPLING": (
                    ("time", "frequency", "polarization", "u", "v"),
                    grid.copy(),
                ),
                "UV_SAMPLING_NORMALIZATION": (
                    ("time", "frequency", "polarization"),
                    np.full((1, 1, n_pol), 2.0),
                ),
            },
            coords={"polarization": labels},
        )
        img_xds.attrs["type"] = "image_dataset"  # required by the xradio accessor
        img_xds.attrs["data_groups"] = {
            "residual": {
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
                "uv_sampling": "UV_SAMPLING",
                "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
            }
        }
        return img_xds

    def _transform(self, labels):
        from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
            ifft_norm_img_xds,
        )

        return ifft_norm_img_xds(
            self._image(labels),
            image_params={"image_size": [16, 16]},
            image_data_group_in_name="residual",
            image_data_group_out_name="residual",
            image_data_group_out_modified={
                "sky": "SKY_RESIDUAL",
                "point_spread_function": "POINT_SPREAD_FUNCTION",
            },
            fft_backend="scipy",
        )

    def test_sky_is_complex_for_four_correlations_only(self):
        four = self._transform(["XX", "XY", "YX", "YY"])
        two = self._transform(["XX", "YY"])
        self.assertEqual(four["SKY_RESIDUAL"].dtype, np.complex128)
        self.assertEqual(four["POINT_SPREAD_FUNCTION"].dtype, np.float64)
        self.assertEqual(two["SKY_RESIDUAL"].dtype, np.float64)
        self.assertEqual(two["POINT_SPREAD_FUNCTION"].dtype, np.float64)
        self.assertEqual(four["SKY_RESIDUAL"].shape, (1, 1, 4, 16, 16))
        # the same grids: the two-hand image is the real part of the first planes
        np.testing.assert_array_equal(
            two["SKY_RESIDUAL"].values, four["SKY_RESIDUAL"].values[:, :, :2].real
        )
        self.assertGreater(np.abs(four["SKY_RESIDUAL"].values.imag).max(), 0.0)
        np.testing.assert_array_equal(
            two["POINT_SPREAD_FUNCTION"].values,
            four["POINT_SPREAD_FUNCTION"].values[:, :, :2],
        )
