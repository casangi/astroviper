# run using eg
# python -m pytest ../astroviper/tests/domain/imaging/test_fft_ifft.py

import unittest

import numpy as np

from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
    fft_lm_to_uv,
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
