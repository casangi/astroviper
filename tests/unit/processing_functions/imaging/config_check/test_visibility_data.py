"""Loaded-array predicates validate data independently of coordinate preflight."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    absent_visibility_rows,
    check_effective_weights,
    check_statistical_weights,
    check_visibility_uvw,
)


@pytest.mark.parametrize("weights", ["nan", "zero", "mixed"])
def test_padding_requires_agreeing_absence_markers(weights):
    values = np.zeros((2, 1, 2, 2))
    if weights == "nan":
        values[1] = np.nan
    elif weights == "mixed":
        values[1, 0, 0, 0] = np.nan
    uvw = np.ones((2, 1, 3))
    uvw[1] = np.nan
    centroid = np.array([[1.0], [np.nan]])
    np.testing.assert_array_equal(
        absent_visibility_rows(values, uvw, time_centroid=centroid), [[False], [True]]
    )
    assert not absent_visibility_rows(values, uvw).any()
    centroid[1] = 1.0
    assert not absent_visibility_rows(values, uvw, time_centroid=centroid).any()


@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf])
@pytest.mark.parametrize("check", [check_statistical_weights, check_effective_weights])
def test_invalid_active_weights_are_rejected_but_inactive_values_are_ignored(
    invalid, check
):
    weights = np.array([1.0, invalid])
    context = {
        "measurement_set": "ms",
        "channel_selection": [0, 1],
        "polarization": ["XX"],
    }
    check(weights, np.array([True, False]), context=context)
    with pytest.raises(ValueError, match="weights"):
        check(weights, np.array([True, True]), context=context)


def test_uvw_predicate_requires_finite_loaded_coordinates():
    uvw = np.ones((2, 1, 3))
    check_visibility_uvw(uvw, measurement_set="ms")
    uvw[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite UVW.*ms"):
        check_visibility_uvw(uvw, measurement_set="ms")
