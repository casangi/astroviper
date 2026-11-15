"""Trust flags reach the real append preparation path without changing inputs."""

import importlib
from copy import deepcopy
from unittest.mock import Mock

import pytest
import xarray as xr

from astroviper.processing_functions.imaging import config_check


class RuntimeReached(Exception):
    """Stop at product processing, after the real configuration helper ran."""


@pytest.mark.parametrize(
    "keyword_flag,dictionary_flag",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_append_checks_only_unchecked_configuration_before_beam_fitting(
    monkeypatch, keyword_flag, dictionary_flag
):
    """Append validates its own inputs before consuming reduced PB-fit products."""
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    processing = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    checks = {}
    for name in (
        "check_residual_update",
        "check_model_update",
        "check_driver_workflow",
    ):
        checks[name] = Mock(wraps=getattr(config_check, name))
        monkeypatch.setattr(config_check, name, checks[name])
    runtime = Mock(side_effect=RuntimeReached)
    # Primary-beam Taylor fitting is now the first runtime product consumer.
    monkeypatch.setattr(processing, "fit_mvc_primary_beam_taylor", runtime)
    params = {
        "continuum_mode": "mvc",
        "deconvolver": "hogbom",
        "input_checked": dictionary_flag,
        "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
    }
    before = deepcopy(params)
    with pytest.raises(RuntimeReached):
        node.continuum_cycle_append_node(
            {"image": xr.Dataset()}, params, input_checked=keyword_flag
        )
    unchecked = not (keyword_flag or dictionary_flag)
    # Residual validation belongs to preparation, which this runtime stop
    # deliberately does not reach.
    assert checks["check_residual_update"].call_count == 0
    assert checks["check_model_update"].call_count == int(unchecked)
    assert checks["check_driver_workflow"].call_count == int(unchecked)
    runtime.assert_called_once()
    assert params == before


@pytest.mark.parametrize(
    "keyword_flag,dictionary_flag",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_preparation_checks_only_unchecked_configuration_before_finalization(
    monkeypatch, keyword_flag, dictionary_flag
):
    """Preparation validates residual inputs before finalizing MVC equations."""
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    processing = importlib.import_module(
        "astroviper.processing_functions.imaging.image_continuum_single_field"
    )
    checks = {}
    for name in (
        "check_residual_update",
        "check_model_update",
        "check_driver_workflow",
    ):
        checks[name] = Mock(wraps=getattr(config_check, name))
        monkeypatch.setattr(config_check, name, checks[name])
    runtime = Mock(side_effect=RuntimeReached)
    monkeypatch.setattr(processing, "finalize_mvc_taylor_normal_equations", runtime)
    params = {
        "continuum_mode": "mvc",
        "deconvolver": "hogbom",
        "input_checked": dictionary_flag,
        "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
    }
    before = deepcopy(params)
    reduced = xr.Dataset({"MVC_RESIDUAL_BEAM_CHANNEL_COUNT": xr.DataArray(1)})
    with pytest.raises(RuntimeReached):
        node._prepare_continuum_image(
            reduced,
            params,
            initialize_static_products=True,
            input_checked=keyword_flag,
        )
    unchecked = not (keyword_flag or dictionary_flag)
    assert checks["check_residual_update"].call_count == int(unchecked)
    assert checks["check_model_update"].call_count == 0
    assert checks["check_driver_workflow"].call_count == 0
    runtime.assert_called_once()
    assert params == before


def test_checked_append_still_requires_actual_mvc_primary_beam_products():
    """Skipping parameter checks must not skip append's PB-fit product checks."""
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    params = {
        "continuum_mode": "mvc",
        "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
    }
    with pytest.raises(KeyError, match="MVC primary-beam Taylor fit is missing"):
        node.continuum_cycle_append_node(
            {"image": xr.Dataset()}, params, input_checked=True
        )


def test_checked_preparation_still_requires_actual_mvc_reduction_products():
    """Skipping parameter checks must not skip reduced MVC product checks."""
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    params = {
        "continuum_mode": "mvc",
        "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
    }
    with pytest.raises(KeyError, match="MVC_RESIDUAL_BEAM_CHANNEL_COUNT"):
        node._prepare_continuum_image(
            xr.Dataset(),
            params,
            initialize_static_products=True,
            input_checked=True,
        )
