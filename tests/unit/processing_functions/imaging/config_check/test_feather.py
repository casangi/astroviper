"""Feather checks reject malformed configuration before image I/O."""

import importlib
from types import MappingProxyType

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    check_feather_combination,
    check_feather_config,
    check_feather_inputs,
    check_feather_output,
)


@pytest.mark.parametrize(
    "outim, highres, lowres, error",
    [
        (None, "hi", "lo", TypeError),
        ({}, "hi", "lo", ValueError),
        ({"name": ""}, "hi", "lo", ValueError),
        ({"name": 1}, "hi", "lo", TypeError),
        ({"name": "out", "overwrite": "false"}, "hi", "lo", TypeError),
        ({"name": "out"}, [], "lo", TypeError),
        ({"name": "out"}, "hi", "", ValueError),
    ],
)
def test_invalid_configuration_precedes_opening(
    monkeypatch, outim, highres, lowres, error
):
    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.feather"
    )
    monkeypatch.setattr(
        driver, "_open_input_image", lambda *args: pytest.fail("opened image")
    )
    with pytest.raises(error):
        driver.feather(outim, highres, lowres)


def test_valid_configuration_does_not_insert_overwrite():
    outim = MappingProxyType({"name": "out"})
    check_feather_config(outim, "hi", "lo")
    assert dict(outim) == {"name": "out"}


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_supported_mixed_precision_images(dtype):
    high = xr.Dataset({"SKY": ("pixel", np.zeros(2, dtype=dtype))})
    low = xr.Dataset({"SKY": ("pixel", np.zeros(2, dtype=np.float64))})
    check_feather_inputs(high, low)
    assert high.SKY.dtype == dtype


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.complex64, bool])
@pytest.mark.parametrize("bad_highres", [True, False])
def test_each_input_dtype_is_validated(dtype, bad_highres):
    good = xr.Dataset({"SKY": ("pixel", np.zeros(2, dtype=np.float32))})
    bad = xr.Dataset({"SKY": ("pixel", np.zeros(2, dtype=dtype))})
    with pytest.raises(ValueError, match="Unsupported image dtype"):
        check_feather_inputs(bad if bad_highres else good, good if bad_highres else bad)


@pytest.mark.parametrize("input_checked", [False, True])
def test_opened_images_checked_without_mutating_output(
    monkeypatch, tmp_path, input_checked
):
    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.feather"
    )
    images = iter(
        [
            xr.Dataset({"SKY": ("pixel", np.zeros(2, dtype=np.float32))}),
            xr.Dataset({"SKY": ("pixel", np.zeros(3, dtype=np.float32))}),
        ]
    )
    monkeypatch.setattr(driver, "_open_input_image", lambda *args: next(images))
    outim = {"name": str(tmp_path / "out")}
    with pytest.raises(RuntimeError, match="Image shapes differ"):
        driver.feather(outim, "hi", "lo", input_checked=input_checked)
    assert outim == {"name": str(tmp_path / "out")}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", ["highres", "lowres"])
def test_feather_requires_sky_in_each_opened_image(missing):
    from astroviper.processing_functions.imaging.config_check.param_check_feather import (
        check_feather_inputs,
    )

    good = xr.Dataset({"SKY": ("pixel", np.ones(2, dtype=np.float32))})
    bad = xr.Dataset()
    with pytest.raises(ValueError, match=f"{missing} must contain a SKY"):
        check_feather_inputs(
            bad if missing == "highres" else good, bad if missing == "lowres" else good
        )


@pytest.mark.parametrize(
    "sdfactor, axes, fft_backend, error",
    [
        (True, ("l", "m"), "scipy", TypeError),
        ("1", ("l", "m"), "scipy", TypeError),
        (np.array([1.0, 2.0]), ("l", "m"), "scipy", TypeError),
        (np.nan, ("l", "m"), "scipy", ValueError),
        (np.inf, ("l", "m"), "scipy", ValueError),
        (1.0, "lm", "scipy", TypeError),
        (1.0, ("l", 1), "scipy", TypeError),
        (1.0, ("l",), "scipy", ValueError),
        (1.0, ("l", "l"), "scipy", ValueError),
        (1.0, ("l", "m", "n"), "scipy", ValueError),
        (1.0, ("l", "m"), "fftw", ValueError),
    ],
)
def test_invalid_combination_is_refused(sdfactor, axes, fft_backend, error):
    with pytest.raises(error):
        check_feather_combination(sdfactor, axes, fft_backend)


@pytest.mark.parametrize("sdfactor", [1, 0.5, np.float32(2.0), np.int64(3)])
def test_valid_combination_is_accepted_unchanged(sdfactor):
    axes = ["l", "m"]
    check_feather_combination(sdfactor, axes, "numpy")
    assert axes == ["l", "m"]


@pytest.mark.parametrize("fmt, engine", [("png", "chunk"), ("fits", "xradio")])
def test_invalid_output_is_refused(fmt, engine):
    with pytest.raises(ValueError):
        check_feather_output(fmt, engine)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"sdfactor": "1"}, TypeError),
        ({"sdfactor": np.nan}, ValueError),
        ({"fft_backend": "fftw"}, ValueError),
        ({"output_image_format": "png"}, ValueError),
    ],
)
def test_invalid_settings_precede_opening(monkeypatch, kwargs, error):
    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.feather"
    )
    monkeypatch.setattr(
        driver, "_open_input_image", lambda *args: pytest.fail("opened image")
    )
    with pytest.raises(error):
        driver.feather({"name": "out"}, "hi", "lo", **kwargs)


def test_checked_configuration_skips_the_combination_check(monkeypatch, tmp_path):
    """With input_checked=True the driver reaches the image opening."""

    class Opened(Exception):
        pass

    def opened(*args):
        raise Opened

    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.feather"
    )
    monkeypatch.setattr(driver, "_open_input_image", opened)
    with pytest.raises(Opened):
        driver.feather(
            {"name": str(tmp_path / "out")},
            "hi",
            "lo",
            sdfactor="1",
            input_checked=True,
        )
