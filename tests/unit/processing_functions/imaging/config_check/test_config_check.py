"""Read-only configuration validation and explicit imaging-mode coverage."""

from copy import deepcopy
from types import MappingProxyType

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    ImagingMode,
    check_config,
    check_driver_workflow,
    check_model_update,
    check_residual_update,
)

CHECKS = [
    check_residual_update,
    check_model_update,
    check_driver_workflow,
    check_config,
]


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("kind", ["cube", "continuum"])
def test_each_validator_reserves_both_multifield_modes(check, kind):
    with pytest.raises(NotImplementedError, match=f"{kind} multifield"):
        check({}, mode=ImagingMode(kind, "multifield"))


@pytest.mark.parametrize(
    "kind,layout,message",
    [("mfs", "single_field", "imaging_kind"), ("cube", "single", "field_layout")],
)
def test_unknown_mode_axes_are_not_normalized(kind, layout, message):
    with pytest.raises(ValueError, match=message):
        ImagingMode(kind, layout)


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
def test_validators_accept_read_only_config_and_preserve_values(check, mode):
    frequency = np.array([1e9, 1.1e9])
    frequency.flags.writeable = False
    image = {
        "frequency_coords": frequency,
        "reference_frequency": 1e9,
        "n_taylor_terms": np.int64(2) if mode.imaging_kind == "continuum" else 1,
    }
    controls = {"threshold": 0.0}
    config = MappingProxyType(
        {
            "image_params": MappingProxyType(image),
            "cycle_control_params": MappingProxyType(controls),
            "deconvolver": "mtmfs" if mode.imaging_kind == "continuum" else "asp",
            "restore": True,
            "primary_beam_correction": True,
            "primary_beam_limit": 0.2,
            "memory_mode": "in_memory",
        }
    )
    before = deepcopy(image)
    assert check(config, mode=mode) is None
    assert config["image_params"]["frequency_coords"] is frequency
    assert type(image["n_taylor_terms"]) is type(before["n_taylor_terms"])
    for key in image:
        np.testing.assert_array_equal(image[key], before[key])
    assert controls == {"threshold": 0.0}


@pytest.mark.parametrize("check", CHECKS)
def test_absent_parameters_are_not_filled(check):
    config = {}
    check(config, mode=CUBE_SINGLE_FIELD)
    assert config == {}


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("frequency_coords", [], ValueError),
        ("frequency_coords", [0.0], ValueError),
        ("frequency_coords", [-1.0], ValueError),
        ("frequency_coords", [np.nan], ValueError),
        ("frequency_coords", [np.inf], ValueError),
        ("frequency_coords", ["100"], TypeError),
        ("frequency_coords", [1 + 2j], TypeError),
        ("reference_frequency", [1e9], TypeError),
        ("reference_frequency", True, TypeError),
    ],
)
@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
def test_residual_metadata_rejects_invalid_values(field, value, error, mode):
    with pytest.raises(error, match=field):
        check_residual_update({"image_params": {field: value}}, mode=mode)


def test_mode_specific_residual_checks():
    with pytest.raises(ValueError, match="continuum_mode"):
        check_residual_update({"continuum_mode": "MFS"}, mode=CONTINUUM_SINGLE_FIELD)
    config = {"imaging_weights_params": {"weighting_scope": "global"}}
    with pytest.raises(ValueError, match="Cube"):
        check_residual_update(config, mode=CUBE_SINGLE_FIELD)
    check_residual_update(config, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize(
    "deconvolver,nterms,error",
    [
        ("mtmfs", 1, ValueError),
        ("mtmfs", 5, ValueError),
        ("hogbom", 2, ValueError),
        ("mtmfs", 0, ValueError),
        ("mtmfs", 2.0, TypeError),
        ("mtmfs", True, TypeError),
        ("asp", 2, ValueError),
        (None, 2, ValueError),
    ],
)
def test_invalid_model_config_is_not_replaced(deconvolver, nterms, error):
    config = {"deconvolver": deconvolver, "image_params": {"n_taylor_terms": nterms}}
    before = deepcopy(config)
    with pytest.raises(error):
        check_model_update(config, mode=CONTINUUM_SINGLE_FIELD)
    assert config == before


@pytest.mark.parametrize(
    "deconvolver,nterms", [("hogbom", 1), ("mtmfs", 2), ("mtmfs", 3), ("mtmfs", 4)]
)
def test_supported_continuum_model_configs(deconvolver, nterms):
    check_model_update(
        {"deconvolver": deconvolver, "image_params": {"n_taylor_terms": nterms}},
        mode=CONTINUUM_SINGLE_FIELD,
    )


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("threshold", -1, ValueError),
        ("threshold", np.inf, ValueError),
        ("threshold", "0", TypeError),
        ("threshold", None, TypeError),
        ("threshold_per_cycle", None, TypeError),
        ("threshold_per_cycle", [0, -1], ValueError),
    ],
)
def test_model_thresholds(name, value, error):
    with pytest.raises(error, match=name):
        check_model_update({"deconvolve_params": {name: value}}, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"primary_beam_correction": True, "restore": False}, "requires restore"),
        ({"primary_beam_limit": 1.0}, "primary_beam_limit"),
        ({"primary_beam_limit": -0.1}, "primary_beam_limit"),
        ({"primary_beam_limit": np.nan}, "primary_beam_limit"),
        ({"memory_mode": "cache"}, "memory_mode"),
        ({"compute_backend": "mpi"}, "compute_backend"),
    ],
)
def test_workflow_relationships(options, message):
    with pytest.raises(ValueError, match=message):
        check_driver_workflow(options, mode=CONTINUUM_SINGLE_FIELD)


def test_cube_backend_is_not_restricted_to_continuum_capabilities():
    check_driver_workflow({"compute_backend": "mpi"}, mode=CUBE_SINGLE_FIELD)


def test_explicit_mtmfs_config_requires_taylor_order():
    with pytest.raises(KeyError, match="n_taylor_terms"):
        check_model_update(
            {"deconvolver": "mtmfs", "image_params": {}}, mode=CONTINUUM_SINGLE_FIELD
        )


def test_workflow_check_does_not_validate_residual_or_model_settings():
    config = {
        "image_params": {"frequency_coords": [0.0], "n_taylor_terms": 2},
        "deconvolver": "unsupported",
        "restore": True,
        "primary_beam_correction": True,
    }
    check_driver_workflow(config, mode=CONTINUUM_SINGLE_FIELD)
    with pytest.raises(ValueError, match="frequency_coords"):
        check_config(config, mode=CONTINUUM_SINGLE_FIELD)


def test_coordinator_passes_same_inputs_to_each_independent_check(monkeypatch):
    import importlib

    module = importlib.import_module(
        "astroviper.processing_functions.imaging.config_check.check_config"
    )
    config = {}
    calls = []

    def record(stage):
        def check(received, *, mode, require_complete=False):
            assert received is config
            assert mode is CONTINUUM_SINGLE_FIELD
            calls.append(stage)

        return check

    for stage in ("residual_update", "model_update", "driver_workflow"):
        monkeypatch.setattr(module, f"check_{stage}", record(stage))
    module.check_config(config, mode=CONTINUUM_SINGLE_FIELD)
    assert calls == ["residual_update", "model_update", "driver_workflow"]
    assert config == {}
