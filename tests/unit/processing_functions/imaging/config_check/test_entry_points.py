"""Configuration checks precede I/O and can be skipped by a checked caller."""

import importlib
import inspect
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging import config_check


@pytest.mark.parametrize("kind", ["cube", "continuum"])
def test_public_driver_rejects_numerics_before_opening_or_writing(
    monkeypatch, tmp_path, kind
):
    module = importlib.import_module(
        f"astroviper.distributed_applications.imaging.image_{kind}_single_field"
    )
    public = getattr(module, f"image_{kind}_single_field")
    open_store = Mock(side_effect=AssertionError("must not open input"))
    monkeypatch.setattr("xradio.measurement_set.open_processing_set", open_store)
    image_params = {
        "frequency_coords": np.array([np.nan]),
        "n_taylor_terms": 1,
        "reference_frequency": 1e9,
        "image_size": [8, 8],
        "cell_size": [-1e-5, 1e-5],
        "phase_direction": [0.0, 0.0],
        "time_coords": [0.0],
        "polarization_coords": ["I", "Q"],
    }
    output = tmp_path / "image.zarr"
    with pytest.raises(ValueError, match="frequency_coords"):
        public("missing-input", str(output), image_params, {}, {})
    open_store.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize("layer", ["node_tasks", "processing_functions"])
@pytest.mark.parametrize("kind", ["cube", "continuum"])
def test_direct_residual_entries_validate_before_reading_data(layer, kind):
    if kind == "cube" and layer == "processing_functions":
        path = "astroviper.processing_functions.imaging.residual_update"
        name = "residual_update_cube_single_field"
    else:
        path = f"astroviper.{layer}.imaging.image_{kind}_single_field"
        name = (
            f"image_{kind}_single_field"
            if kind == "cube"
            else "residual_update_continuum_single_field"
        )
    function = getattr(importlib.import_module(path), name)
    # Supply inert values for required data arguments: validation must run first.
    arguments = {
        n: None
        for n, p in inspect.signature(function).parameters.items()
        if p.default is inspect.Parameter.empty
    }
    arguments["image_params"] = {"frequency_coords": [0.0]}
    if "imaging_weights_params" in arguments:
        arguments["imaging_weights_params"] = {}
    if "cycle_control_params" in arguments:
        arguments["cycle_control_params"] = {}
    with pytest.raises(ValueError, match="frequency_coords"):
        function(**arguments)


class RuntimeReached(Exception):
    """Stop before science, after configuration validation."""


@pytest.mark.parametrize("checked", [False, True])
def test_processing_function_skips_only_config_checks(monkeypatch, checked):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.model_update"
    )
    check = Mock(wraps=config_check.check_model_update)
    monkeypatch.setattr(config_check, "check_model_update", check)
    # Derived-product checks remain active even when configuration was checked.
    runtime = Mock(side_effect=RuntimeReached)
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.config_check.param_check_model_update.check_model_products",
        runtime,
    )
    with pytest.raises(RuntimeReached):
        module.model_update_cube_single_field(
            xr.Dataset(), "hogbom", {}, False, input_checked=checked
        )
    assert check.call_count == (0 if checked else 1)
    runtime.assert_called_once()


@pytest.mark.parametrize("checked", [False, True])
def test_append_passes_checked_flag_to_model_update_node(monkeypatch, checked):
    module = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    check = Mock(wraps=config_check.check_model_update)
    monkeypatch.setattr(config_check, "check_model_update", check)
    model_node = Mock(side_effect=RuntimeReached)
    monkeypatch.setattr(module, "model_update_continuum_single_field", model_node)
    monkeypatch.setattr(
        module, "_prepare_cached_mfs_residual_grid", lambda data, params: data
    )
    monkeypatch.setattr(
        module,
        "_prepare_continuum_image",
        lambda data, params, **kwargs: (data, xr.Dataset(), None),
    )
    params = {
        "input_checked": checked,
        "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
        "deconvolver": "hogbom",
        "cycle_control_params": {},
    }
    before = dict(params)
    with pytest.raises(RuntimeReached):
        module.continuum_cycle_append_node({"image": xr.Dataset()}, params)
    assert check.call_count == (0 if checked else 1)
    assert model_node.call_args.kwargs["input_checked"] is True
    assert params == before


@pytest.mark.parametrize("checked", [False, True])
def test_model_update_node_checks_once_then_calls_checked_processing(
    monkeypatch, checked
):
    from tests.utils.continuum_images import make_runaway_taylor_image

    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    processing = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    check = Mock(wraps=config_check.check_model_update)
    monkeypatch.setattr(config_check, "check_model_update", check)
    native = Mock(side_effect=RuntimeReached)
    monkeypatch.setattr(processing, "_model_update_native_mtmfs", native)
    params = {
        "input_checked": checked,
        "deconvolver": "mtmfs",
        "cycle_control_params": {
            "max_iter": 10,
            "max_cycles": 2,
            "max_iter_per_cycle": 5,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.0,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
    }
    with pytest.raises(RuntimeReached):
        node.model_update_continuum_single_field(
            {"image": make_runaway_taylor_image()}, params
        )
    assert check.call_count == (0 if checked else 1)
    native.assert_called_once()


def test_checked_processing_call_still_rejects_missing_runtime_products():
    processing = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    with pytest.raises(KeyError, match="SKY_RESIDUAL"):
        processing.model_update_mtmfs_single_field(
            xr.Dataset(), "hogbom", {}, input_checked=True
        )
