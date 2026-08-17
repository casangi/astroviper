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
