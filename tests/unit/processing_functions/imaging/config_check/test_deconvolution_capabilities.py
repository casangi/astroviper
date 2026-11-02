"""Deconvolution boundaries delegate capability checks without rewriting names."""

import importlib
from unittest.mock import Mock

import pytest

from astroviper.processing_functions.imaging.config_check import (
    CUBE_SINGLE_FIELD,
    check_model_update,
)


@pytest.mark.parametrize("algorithm", ["ASP", "HOGBOM"])
def test_cube_rejects_noncanonical_deconvolver_names(algorithm):
    with pytest.raises(ValueError, match="Deconvolution algorithm"):
        check_model_update({"deconvolver": algorithm}, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize("algorithm", ["hogbom", "hogbom_many_threads", "asp", "mtmfs"])
def test_history_wrapper_accepts_implemented_algorithms(algorithm):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.deconvolution"
    )
    module.validate_record_iteration_history(algorithm, True)


@pytest.mark.parametrize(
    "wrapper, algorithm",
    [
        ("validate_record_iteration_history", "ASP"),
        ("validate_exact_psf_subtraction", "HOGBOM"),
        ("validate_exact_psf_subtraction", "asp"),
    ],
)
def test_capability_wrappers_reject_unsupported_names(wrapper, algorithm):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.deconvolution"
    )
    with pytest.raises(ValueError, match="supported only"):
        getattr(module, wrapper)(algorithm, True)


@pytest.mark.parametrize("algorithm", ["ASP"])
def test_deconvolve_rejects_invalid_algorithm_before_accessing_image(algorithm):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.deconvolution"
    )
    with pytest.raises(ValueError, match="Deconvolution algorithm"):
        module.deconvolve(None, algorithm=algorithm)


@pytest.mark.parametrize("checked", [False, True])
def test_deconvolve_skips_only_configuration_validation(monkeypatch, checked):
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.deconvolution"
    )
    from astroviper.processing_functions.imaging import config_check

    check = Mock(wraps=config_check.check_model_update)
    monkeypatch.setattr(config_check, "check_model_update", check)
    # Stop at the first runtime data operation, before creating or changing a model.
    runtime = Mock(side_effect=RuntimeError("runtime reached"))
    monkeypatch.setattr(module, "create_data_groups_in_and_out", runtime)
    with pytest.raises(RuntimeError, match="runtime reached"):
        module.deconvolve(None, algorithm="asp", input_checked=checked)
    assert check.call_count == (0 if checked else 1)
    runtime.assert_called_once()
