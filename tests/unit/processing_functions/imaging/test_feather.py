"""Unit tests for the feather science function
(``astroviper.processing_functions.imaging.feather.feather_core``).

These are fast, self-contained tests on small hand-built images.  Per the
processing-function convention the inputs are NumPy-backed ``xarray`` datasets
(no Dask arrays); the end-to-end graph/I/O behaviour is covered separately by
``tests/unit/distributed_graphs/imaging/test_feather.py``.
"""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.feather import feather_core


def _make_image(nl=32, nm=32, beam=(0.02, 0.02, 0.0), amp=1.0, dtype=np.float32):
    """Build a small numpy-backed single-plane image with a centred source.

    Parameters
    ----------
    beam : tuple
        ``(major, minor, pa)`` in radians, stored in ``BEAM_FIT_PARAMS_SKY``.
    """
    cell = 1.0e-3  # radians per pixel
    l = (np.arange(nl) - nl // 2) * cell
    m = (np.arange(nm) - nm // 2) * cell
    ll, mm = np.meshgrid(l, m, indexing="ij")
    sky = amp * np.exp(-(ll**2 + mm**2) / (2.0 * (5 * cell) ** 2))
    sky = sky[np.newaxis, np.newaxis, np.newaxis, :, :].astype(dtype)
    beam_arr = np.array(beam, dtype=np.float64)[np.newaxis, np.newaxis, np.newaxis, :]
    ds = xr.Dataset(
        {
            "SKY": (["time", "frequency", "polarization", "l", "m"], sky),
            "BEAM_FIT_PARAMS_SKY": (
                ["time", "frequency", "polarization", "beam_params_label"],
                beam_arr,
            ),
        },
        coords={
            "time": [0.0],
            "frequency": [1.0e9],
            "polarization": ["I"],
            "l": l,
            "m": m,
            "beam_params_label": ["major", "minor", "pa"],
        },
    )
    ds["BEAM_FIT_PARAMS_SKY"].attrs["units"] = "rad"
    return ds


def _make_pair(dtype=np.float32):
    """A single-dish (broad beam) and interferometer (narrow beam) image pair."""
    sd = _make_image(beam=(0.04, 0.04, 0.0), amp=1.0, dtype=dtype)  # low resolution
    intf = _make_image(beam=(0.01, 0.01, 0.0), amp=1.0, dtype=dtype)  # high resolution
    return sd, intf


def test_feather_core_output_structure():
    sd, intf = _make_pair()
    out = feather_core(sd, intf)
    assert "SKY" in out
    assert out["SKY"].dims == ("time", "frequency", "polarization", "l", "m")
    assert out["SKY"].shape == (1, 1, 1, 32, 32)
    # Output is a finite, real sky image.
    assert np.all(np.isfinite(out["SKY"].values))
    assert np.isrealobj(out["SKY"].values)
    # Coordinates follow the interferometer image.
    assert np.array_equal(out["l"].values, intf["l"].values)


def test_feather_core_returns_numpy_not_dask():
    """The science layer operates on (and returns) NumPy arrays, not Dask."""
    sd, intf = _make_pair()
    out = feather_core(sd, intf)
    assert isinstance(out["SKY"].data, np.ndarray)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_feather_core_preserves_dtype(dtype):
    sd, intf = _make_pair(dtype=dtype)
    out = feather_core(sd, intf)
    assert out["SKY"].dtype == dtype


def test_feather_core_scipy_matches_pyfftw():
    """The two FFT backends must agree to FFT precision."""
    sd, intf = _make_pair(dtype=np.float64)
    a = feather_core(sd, intf, fft_backend="pyfftw")
    b = feather_core(sd, intf, fft_backend="scipy")
    np.testing.assert_allclose(a["SKY"].values, b["SKY"].values, atol=1e-9, rtol=1e-7)


def test_feather_core_missing_interferometer_beam_raises():
    sd, intf = _make_pair()
    intf = intf.drop_vars("BEAM_FIT_PARAMS_SKY")
    with pytest.raises(RuntimeError, match="interferometer"):
        feather_core(sd, intf)


def test_feather_core_missing_single_dish_beam_raises():
    sd, intf = _make_pair()
    sd = sd.drop_vars("BEAM_FIT_PARAMS_SKY")
    with pytest.raises(RuntimeError, match="single dish"):
        feather_core(sd, intf)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
