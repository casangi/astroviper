"""Independent configuration checkers apply the same shared numerical rules."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_driver_workflow,
    check_model_update,
    check_residual_update,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_annulus_limits,
)
from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_delphi_parameters,
)


@pytest.mark.parametrize(
    "checker", [check_residual_update, check_model_update, check_driver_workflow]
)
@pytest.mark.parametrize("value", [0.0, np.float64(0.5), np.nextafter(1.0, 0.0), 1.0])
@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
def test_beam_limit_boundaries_are_shared(checker, value, mode):
    controls = {"primary_beam_limit": value}
    config = (
        {"cycle_control_params": controls}
        if checker is check_model_update
        else controls
    )
    checker(config, mode=mode)
    assert controls["primary_beam_limit"] is value


@pytest.mark.parametrize(
    "checker", [check_residual_update, check_model_update, check_driver_workflow]
)
@pytest.mark.parametrize(
    "value", [-0.1, np.nextafter(1.0, np.inf), np.nan, np.inf, True, "0.5", [0.5]]
)
def test_beam_limits_reject_malformed_original_values(checker, value):
    controls = {"primary_beam_limit": value}
    config = (
        {"cycle_control_params": controls}
        if checker is check_model_update
        else controls
    )
    with pytest.raises((TypeError, ValueError), match="primary_beam_limit"):
        checker(config, mode=CUBE_SINGLE_FIELD)
    assert controls["primary_beam_limit"] is value


@pytest.mark.parametrize("delphi", [False, True])
@pytest.mark.parametrize("value", [-2.0, 0, np.float64(2.0)])
def test_briggs_robust_boundaries_are_shared(delphi, value):
    if delphi:
        check_delphi_parameters({"robust": value})
    else:
        check_residual_update(
            {"imaging_weights_params": {"weighting": "briggs", "robust": value}},
            mode=CUBE_SINGLE_FIELD,
        )


@pytest.mark.parametrize("delphi", [False, True])
@pytest.mark.parametrize("value", [-2.01, 2.01, np.nan, np.inf, True, "0", [0]])
def test_briggs_robust_rejection_is_shared(delphi, value):
    with pytest.raises((TypeError, ValueError), match="robust"):
        if delphi:
            check_delphi_parameters({"robust": value})
        else:
            check_residual_update(
                {"imaging_weights_params": {"weighting": "briggs", "robust": value}},
                mode=CUBE_SINGLE_FIELD,
            )


@pytest.mark.parametrize("checker", [check_residual_update, check_model_update])
@pytest.mark.parametrize(
    "value", [0, -1, True, np.bool_(False), 1.5, "2", np.array([2])]
)
def test_taylor_count_type_and_positivity_are_shared(checker, value):
    with pytest.raises((TypeError, ValueError), match="n_taylor_terms"):
        checker(
            {"image_params": {"n_taylor_terms": value}}, mode=CONTINUUM_SINGLE_FIELD
        )


@pytest.mark.parametrize("checker", [check_residual_update, check_model_update])
@pytest.mark.parametrize("value", [1, np.int64(4)])
def test_taylor_count_accepts_original_integer_scalars(checker, value):
    params = {"n_taylor_terms": value}
    checker({"image_params": params}, mode=CONTINUUM_SINGLE_FIELD)
    assert params["n_taylor_terms"] is value


@pytest.mark.parametrize(
    "value",
    [(0.0, 1.0), (0.25, 0.25), [0, 1], np.array([0.2, 0.3], dtype=np.float32)],
)
def test_annulus_limits_accept_ordered_levels_of_the_unit_interval(value):
    check_annulus_limits(value)


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ((0.3, 0.2), ValueError),
        ((-0.1, 0.5), ValueError),
        ((0.5, 1.1), ValueError),
        ((0.2, np.inf), ValueError),
        ((0.2,), ValueError),
        (0.2, ValueError),
        (("0.2", "0.3"), TypeError),
        ((True, True), TypeError),
    ],
)
def test_annulus_limits_reject_malformed_values(value, error):
    with pytest.raises(error, match="annulus_limits"):
        check_annulus_limits(value)
