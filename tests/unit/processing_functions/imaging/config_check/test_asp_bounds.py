"""Reject contradictory Asp fit bounds before any image or kernel is used."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_model_update,
)
from astroviper.processing_functions.imaging.deconvolution import asp_clean


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize(
    "name,value",
    [
        ("asp_significance_gain", 1.01),
        ("asp_residual_component_min_fraction", 1.0),
        ("asp_residual_component_max_epsilon_fraction", 0.0),
        ("asp_residual_component_max_epsilon_fraction", 0.0009),
    ],
)
def test_invalid_asp_bounds_are_rejected_without_modifying_controls(mode, name, value):
    config = {"deconvolver": "asp", "deconvolve_params": {name: value}}
    before = deepcopy(config)
    with pytest.raises(ValueError, match=name):
        check_model_update(config, mode=mode)
    assert config == before


@pytest.mark.parametrize(
    "name,value",
    [
        ("asp_significance_gain", 2),
        ("asp_residual_component_min_fraction", 2),
        ("asp_residual_component_max_epsilon_fraction", 0),
    ],
)
def test_direct_asp_calls_reject_invalid_bounds_before_accessing_arrays(name, value):
    with pytest.raises(ValueError, match=name):
        asp_clean(None, None, None, {name: value})


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize("significance_gain", [-1.0, 0.0, 1.0])
def test_asp_boundary_values_and_estimation_sentinels_are_preserved(
    mode, significance_gain
):
    controls = {
        "asp_significance_gain": significance_gain,
        "asp_residual_component_min_fraction": 0.99,
        "asp_residual_component_max_epsilon_fraction": 1e-3,
        "asp_fused_threshold": -1.0,
        "asp_psf_width": -1.0,
        "asp_largest_scale": -1,
        "asp_stop_point_mode": -1,
    }
    before = deepcopy(controls)
    check_model_update({"deconvolver": "asp", "deconvolve_params": controls}, mode=mode)
    assert controls == before


@pytest.mark.parametrize("value", [True, "0.5", np.nan, np.inf])
@pytest.mark.parametrize(
    "name",
    [
        "asp_significance_gain",
        "asp_residual_component_min_fraction",
        "asp_residual_component_max_epsilon_fraction",
    ],
)
def test_asp_bounds_still_require_finite_real_scalars(name, value):
    with pytest.raises((TypeError, ValueError), match=name):
        check_model_update(
            {"deconvolver": "asp", "deconvolve_params": {name: value}},
            mode=CUBE_SINGLE_FIELD,
        )
