"""Unit tests for the feather science function
(``astroviper.processing_functions.imaging.feather.feather_core``).

These are fast, self-contained tests on small hand-built images.  Per the
processing-function convention the inputs are NumPy-backed ``xarray`` datasets
(no Dask arrays); the end-to-end graph/I/O behaviour is covered separately by
``tests/unit/distributed_applications/imaging/test_feather.py``.
"""

import importlib

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


def test_feather_core_numpy_matches_scipy(monkeypatch):
    """numpy.fft and scipy.fft agree to FFT precision; under "numpy" the three
    transforms of the FFT module run in numpy."""
    from astroviper.processing_functions import fft

    seen = []
    for name in ("fft2", "ifft2"):
        real = getattr(fft, name)

        def spy(*args, _real=real, **kwargs):
            seen.append(kwargs["fft_backend"])
            return _real(*args, **kwargs)

        monkeypatch.setattr(fft, name, spy)
    sd, intf = _make_pair(dtype=np.float64)
    a = feather_core(sd, intf, fft_backend="numpy")
    assert seen == ["numpy"] * 3
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


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"sdfactor": "1"}, TypeError),
        ({"axes": ("l",)}, ValueError),
        ({"fft_backend": "fftw"}, ValueError),
    ],
)
def test_feather_core_refuses_invalid_configuration(monkeypatch, kwargs, error):
    """The configuration is refused before the first transform."""
    feather = importlib.import_module("astroviper.processing_functions.imaging.feather")
    monkeypatch.setattr(
        feather, "fft_lm_to_uv", lambda *args, **kwargs: pytest.fail("transformed")
    )
    sd, intf = _make_pair()
    with pytest.raises(error):
        feather_core(sd, intf, **kwargs)


def test_feather_core_checked_configuration_is_not_checked_again(monkeypatch):
    from astroviper.processing_functions.imaging.config_check import (
        param_check_feather,
    )

    monkeypatch.setattr(
        param_check_feather,
        "check_feather_combination",
        lambda *args: pytest.fail("checked again"),
    )
    sd, intf = _make_pair()
    out = feather_core(sd, intf, input_checked=True)
    assert out["SKY"].shape == intf["SKY"].shape


def test_feather_application_hands_input_checked_to_its_node_tasks(
    monkeypatch, tmp_path
):
    """The application's node tasks get input_checked=True in their input_params."""
    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.feather"
    )
    graph_map = importlib.import_module("graphviper.graph_tools.map")
    opened = iter(_make_pair())
    monkeypatch.setattr(driver, "_open_input_image", lambda *args: next(opened))

    class Mapped(Exception):
        pass

    seen = {}

    def fake_map(**kwargs):
        seen.update(kwargs["input_params"])
        raise Mapped

    monkeypatch.setattr(graph_map, "map", fake_map)
    with pytest.raises(Mapped):
        driver.feather(
            {"name": str(tmp_path / "out.img.zarr")},
            "hi",
            "lo",
            thread_info={"n_threads": 1, "memory_per_thread": 8.0},
        )
    assert seen["input_checked"] is True


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
