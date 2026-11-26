"""The feather node task checks its configuration unless the driver checked it."""

import numpy as np
import pytest
import xarray as xr

import astroviper.processing_functions as pf
import astroviper.utils.node_task_io as node_task_io
from astroviper.node_tasks.imaging.feather import feather
from astroviper.processing_functions.imaging.config_check import param_check_feather


def _input_params(**changes):
    params = {
        "input_data_store": {"sd": "sd.img.zarr", "int": "int.img.zarr"},
        "data_selection": {"sd": {}, "int": {}},
        "sdfactor": 1.0,
        "axes": ("l", "m"),
        "fft_backend": "scipy",
        "io_engine": "chunk",
        "output_image_format": "zarr",
    }
    params.update(changes)
    return params


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"sdfactor": "1"}, TypeError),
        ({"axes": ("l", "l")}, ValueError),
        ({"fft_backend": "fftw"}, ValueError),
        ({"io_engine": "zarr"}, ValueError),
        ({"output_image_format": "fits", "io_engine": "xradio"}, ValueError),
    ],
)
def test_invalid_configuration_precedes_loading(monkeypatch, changes, error):
    monkeypatch.setattr(
        node_task_io, "load_image", lambda *args: pytest.fail("loaded image")
    )
    with pytest.raises(error):
        feather(_input_params(**changes))


def test_checked_configuration_is_not_checked_again(monkeypatch):
    """With input_checked the node task skips the checks and tells feather_core so."""

    class Called(Exception):
        pass

    calls = {}

    def feather_core(*args, **kwargs):
        calls.update(kwargs)
        raise Called

    image = xr.Dataset({"SKY": (("l", "m"), np.zeros((2, 2)))})

    def checked_again(*args):
        pytest.fail("checked again")

    monkeypatch.setattr(param_check_feather, "check_feather_combination", checked_again)
    monkeypatch.setattr(param_check_feather, "check_feather_output", checked_again)
    monkeypatch.setattr(node_task_io, "check_io_engine", checked_again)
    monkeypatch.setattr(node_task_io, "load_image", lambda *args: image)
    monkeypatch.setattr(pf.imaging, "feather_core", feather_core)
    with pytest.raises(Called):
        feather(_input_params(sdfactor="1", input_checked=True))
    assert calls["input_checked"] is True
