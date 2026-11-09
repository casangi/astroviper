"""Trust flags reach the real append preparation path without changing inputs."""

import importlib
from copy import deepcopy
from unittest.mock import Mock

import pytest
import xarray as xr

from astroviper.processing_functions.imaging import config_check


class RuntimeReached(Exception):
    """Stop at product processing, after the real configuration helper ran."""


@pytest.mark.parametrize("entry", ["append", "preparation"])
@pytest.mark.parametrize(
    "keyword_flag,dictionary_flag",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_actual_preparation_checks_only_unchecked_configuration(
    monkeypatch, entry, keyword_flag, dictionary_flag
):
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
        if entry == "append":
            node.continuum_cycle_append_node(
                {"image": xr.Dataset()}, params, input_checked=keyword_flag
            )
        else:
            node._prepare_continuum_image(
                xr.Dataset(),
                params,
                initialize_static_products=True,
                input_checked=keyword_flag,
            )
    unchecked = not (keyword_flag or dictionary_flag)
    assert checks["check_residual_update"].call_count == int(unchecked)
    assert checks["check_model_update"].call_count == int(
        unchecked and entry == "append"
    )
    assert checks["check_driver_workflow"].call_count == int(
        unchecked and entry == "append"
    )
    runtime.assert_called_once()
    assert params == before


@pytest.mark.parametrize("entry", ["append", "preparation"])
def test_checked_configuration_still_requires_actual_mvc_products(entry):
    node = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    params = {
        "continuum_mode": "mvc",
        "image_params": {"n_taylor_terms": 1, "reference_frequency": 1e9},
    }
    with pytest.raises(KeyError, match="MVC primary-beam Taylor fit is missing"):
        if entry == "append":
            node.continuum_cycle_append_node(
                {"image": xr.Dataset()}, params, input_checked=True
            )
        else:
            node._prepare_continuum_image(
                xr.Dataset(),
                params,
                initialize_static_products=True,
                input_checked=True,
            )
