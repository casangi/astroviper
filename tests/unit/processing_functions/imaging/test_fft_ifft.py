# run using eg
# python -m pytest ../astroviper/tests/domain/imaging/test_fft_ifft.py

import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.fft
import xarray as xr

from astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term import (
    CASA_GRIDDING_KERNEL_PARAMS,
    correction_1D,
    make_gridding_kernel,
)
from astroviper.processing_functions.imaging.normalization.normalize_image import (
    fft_lm_to_uv,
    fft_norm_img_xds,
    ifft_norm_img_xds,
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

    monkeypatch.setattr(plt, "show", fail_show)
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
        gcf_xds=make_gridding_kernel(CASA_GRIDDING_KERNEL_PARAMS),
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
        from astroviper.processing_functions.imaging.normalization.normalize_image import (
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
            gcf_xds=make_gridding_kernel(CASA_GRIDDING_KERNEL_PARAMS),
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


def _shift_transform(x, axes, inverse, complex_dtype):
    """The definition the transforms implement, without the checkerboard fold:
    ``fftshift(F(ifftshift(x)))``."""
    transform = scipy.fft.ifft2 if inverse else scipy.fft.fft2
    x = np.asarray(x, dtype=complex_dtype)
    return scipy.fft.fftshift(
        transform(scipy.fft.ifftshift(x, axes=axes), axes=axes), axes=axes
    )


# Every parity of n0/2 and n1/2 on two even axes (the checkerboard fold), square
# planes, and odd axes (the rolls).
PLANE_SHAPES = [
    (8, 12),  # n0/2 and n1/2 even
    (10, 12),  # n0/2 odd: n0/2 + n1/2 odd
    (12, 10),  # n1/2 odd: n0/2 + n1/2 odd
    (10, 14),  # both odd
    (90, 120),  # the padded grid of a 74 x 100 image at fft_padding 1.2 (45 + 60)
    (16, 16),
    (10, 10),
    (9, 12),
    (12, 9),
    (11, 13),
]


@pytest.mark.parametrize("backend", ["scipy", "pyfftw"])
@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("shape", PLANE_SHAPES, ids=str)
def test_transforms_are_the_shifted_transforms(shape, complex_dtype, backend):
    """ifft_uv_to_lm and fft_lm_to_uv equal fftshift(F(ifftshift(x))) for every
    plane shape, on one plane and on a stack of planes, with and without
    overwrite_input: the checkerboard fold of two even axes carries no factor
    (-1)**(n0/2 + n1/2) (it negated planes where that sum is odd)."""
    rng = np.random.default_rng(sum(shape))
    tolerance = 1e-5 if complex_dtype == np.complex64 else 1e-12
    for stack, axes in (((), (0, 1)), ((3, 2), (-2, -1))):
        full_shape = stack + shape
        x = (
            rng.standard_normal(full_shape) + 1j * rng.standard_normal(full_shape)
        ).astype(complex_dtype)
        for inverse, transform, kwargs in (
            (True, ifft_uv_to_lm, {}),
            (False, fft_lm_to_uv, {"complex_dtype": complex_dtype}),
        ):
            expected = _shift_transform(x, axes, inverse, complex_dtype)
            for overwrite_input in (False, True):
                given = x.copy()
                got = transform(
                    given,
                    axes,
                    fft_backend=backend,
                    overwrite_input=overwrite_input,
                    **kwargs,
                )
                assert got.dtype == complex_dtype
                assert got.shape == full_shape
                np.testing.assert_allclose(
                    got, expected, rtol=0, atol=tolerance * np.abs(expected).max()
                )
                if not overwrite_input:
                    np.testing.assert_array_equal(given, x)


@pytest.mark.parametrize("backend", ["scipy", "pyfftw"])
@pytest.mark.parametrize("shape", [(90, 120), (120, 90), (10, 14), (16, 16)], ids=str)
def test_centre_point_transforms_to_plus_one(shape, backend):
    """A unit point at the centre pixel (n0/2, n1/2), the phase centre of an
    image and the uv origin of a grid, transforms to +1 everywhere, and +1
    everywhere back to the point, in both directions (the fold gave -1 on a
    90 x 120 plane)."""
    point = np.zeros(shape, dtype=np.complex128)
    point[shape[0] // 2, shape[1] // 2] = 1.0
    ones = np.ones(shape, dtype=np.complex128)
    n = point.size
    for got, expected in (
        (fft_lm_to_uv(point, fft_backend=backend), ones),
        (ifft_uv_to_lm(point, fft_backend=backend) * n, ones),
        (ifft_uv_to_lm(ones, fft_backend=backend), point),
        (fft_lm_to_uv(ones, fft_backend=backend) / n, point),
    ):
        np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("shape", [(16, 16), (10, 10), (90, 90)], ids=str)
def test_square_planes_fold_as_before(shape):
    """On a square plane the two centring signs (-1)**(n/2) cancel: both
    transforms are, bit for bit, the uncentred cb * F(cb * x) the fold computed
    before its sign was corrected."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    cb = np.ones(shape[0])
    cb[1::2] = -1.0

    def uncentred(transform):
        work = x * cb[:, None]
        work *= cb[None, :]
        out = transform(work, axes=(-2, -1), workers=1, overwrite_x=True)
        out *= cb[:, None]
        out *= cb[None, :]
        return out

    np.testing.assert_array_equal(
        ifft_uv_to_lm(x, fft_backend="scipy"), uncentred(scipy.fft.ifft2)
    )
    np.testing.assert_array_equal(
        fft_lm_to_uv(x, fft_backend="scipy"), uncentred(scipy.fft.fft2)
    )


@pytest.mark.parametrize("backend", ["scipy", "pyfftw"])
def test_rectangular_point_spread_function_and_dirty_image_are_positive(backend):
    """ifft_norm_img_xds on the 90 x 120 padded grid of a 74 x 100 image
    (fft_padding 1.2, n0/2 + n1/2 = 105 odd): positive uv weights give a point
    spread function, and the visibilities of a 1 Jy point source at the phase
    centre a dirty image, that peak at +1 at the centre pixel (both were
    negated)."""
    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    image_size = [74, 100]
    n_u, n_v = (int(n) for n in padded_grid_size(image_size, 1.2))
    assert (n_u, n_v) == (90, 120)
    u = np.arange(n_u)[:, None] - n_u // 2
    v = np.arange(n_v)[None, :] - n_v // 2
    weights = np.exp(-(u**2 + v**2) / 50.0)  # a smooth, positive sampling
    grid = np.broadcast_to(weights, (1, 1, 2, n_u, n_v)).astype(np.complex128)
    total = np.full((1, 1, 2), weights.sum())
    dims = ("time", "frequency", "polarization", "u", "v")
    img_xds = xr.Dataset(
        {
            "VISIBILITY": (dims, grid.copy()),
            "VISIBILITY_NORMALIZATION": (dims[:3], total),
            "UV_SAMPLING": (dims, grid.copy()),
            "UV_SAMPLING_NORMALIZATION": (dims[:3], total.copy()),
        },
        coords={"polarization": ["XX", "YY"]},
        attrs={
            "type": "image_dataset",
            "data_groups": {
                "residual": {
                    "visibility": "VISIBILITY",
                    "visibility_normalization": "VISIBILITY_NORMALIZATION",
                    "uv_sampling": "UV_SAMPLING",
                    "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
                }
            },
        },
    )
    kernel = make_gridding_kernel(CASA_GRIDDING_KERNEL_PARAMS)
    result = ifft_norm_img_xds(
        img_xds,
        image_params={"image_size": image_size},
        image_data_group_in_name="residual",
        image_data_group_out_name="residual",
        fft_backend=backend,
        gcf_xds=kernel,
    )
    centre = correction_1D(kernel, n_u)[n_u // 2] * correction_1D(kernel, n_v)[n_v // 2]
    for name in ("POINT_SPREAD_FUNCTION", "SKY_RESIDUAL"):
        image = result[name].values[0, 0]
        assert image.shape == (2, 74, 100)
        np.testing.assert_allclose(image[:, 37, 50], 1.0 / centre, rtol=1e-12)
        assert (image[:, 37, 50] == image.max(axis=(-2, -1))).all()
