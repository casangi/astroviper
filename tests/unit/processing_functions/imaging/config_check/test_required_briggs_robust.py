"""Briggs robustness is an explicit weighting choice, not an inferred default."""

from types import MappingProxyType

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_residual_update,
)


def _weights(weighting):
    params = {"weighting": weighting}
    if weighting == "briggs_abs":
        params["briggs_abs_noise"] = 1.0
    return params


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize("weighting", ["briggs", "briggs_abs"])
def test_briggs_requires_robust_without_inserting_a_default(mode, weighting):
    params = _weights(weighting)
    original = dict(params)
    with pytest.raises(KeyError, match="robust"):
        check_residual_update(
            {"imaging_weights_params": MappingProxyType(params)}, mode=mode
        )
    assert params == original
    assert "robust" not in params


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize("weighting", ["briggs", "briggs_abs"])
@pytest.mark.parametrize("robust", [-2.0, 0.5, 2.0])
def test_explicit_briggs_robust_is_accepted_and_preserved(mode, weighting, robust):
    params = {**_weights(weighting), "robust": robust}
    original = dict(params)
    check_residual_update(
        {"imaging_weights_params": MappingProxyType(params)}, mode=mode
    )
    assert params == original


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize("weighting", ["briggs", "briggs_abs"])
@pytest.mark.parametrize(
    "robust,error",
    [
        (-2.01, ValueError),
        (2.01, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (None, TypeError),
        (True, TypeError),
        ("0.5", TypeError),
    ],
)
def test_invalid_explicit_robust_is_rejected_instead_of_replaced(
    mode, weighting, robust, error
):
    params = {**_weights(weighting), "robust": robust}
    with pytest.raises(error, match="robust"):
        check_residual_update(
            {"imaging_weights_params": MappingProxyType(params)}, mode=mode
        )
    assert params["robust"] is robust
