# run using eg
# python -m pytest ../astroviper/tests/domain/imaging/test_fft_ifft.py

from astroviper.core.imaging.fft import fft_lm_to_uv
from astroviper.core.imaging.ifft import ifft_uv_to_lm
import numpy as np
import unittest


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
