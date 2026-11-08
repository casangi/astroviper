"""The model-update checker owns ``least_squares_params``."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CUBE_SINGLE_FIELD,
    check_model_update,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_least_squares_params,
)

GOOD = {"least_squares_params": {"max_steps": 5}, "auto_mask_params": {}}


@pytest.mark.parametrize(
    "value, error, match",
    [
        (["max_steps"], TypeError, "must be a mapping"),
        ({}, ValueError, "needs max_steps"),
        ({"steps": 5}, ValueError, "Unknown least_squares_params keys"),
        ({"max_steps": 0}, ValueError, ">= 1"),
        ({"max_steps": 1.5}, TypeError, "integer"),
        ({"max_steps": 5, "smooth_fwhm_pixels": 0.0}, ValueError, "positive"),
        ({"max_steps": 5, "smooth_fwhm_pixels": "64"}, TypeError, "real"),
        ({"max_steps": 5, "start_growth": 0.0}, ValueError, "positive"),
        ({"max_steps": 5, "start_growth": "0.1"}, TypeError, "real"),
        ({"max_steps": 5, "start_updates": 2}, ValueError, "needs start_growth"),
        ({"max_steps": 5, "start_growth": 0.1, "start_updates": 0}, ValueError, ">= 1"),
        (
            {"max_steps": 5, "start_growth": 0.1, "start_updates": 1.5},
            TypeError,
            "integer",
        ),
    ],
)
def test_refuses_a_bad_value(value, error, match):
    with pytest.raises(error, match=match):
        check_least_squares_params(value)


def test_accepts_a_smooth_basis():
    check_least_squares_params({"max_steps": 5, "smooth_fwhm_pixels": 32})


def test_the_start_rule_needs_an_automatic_mask():
    rule = {"max_steps": 5, "start_growth": 0.1, "start_updates": 3}
    check_model_update({**GOOD, "least_squares_params": rule}, mode=CUBE_SINGLE_FIELD)
    with pytest.raises(ValueError, match="start_growth needs auto_mask_params"):
        check_model_update(
            {"least_squares_params": rule, "clean_mask": np.ones((8, 8), bool)},
            mode=CUBE_SINGLE_FIELD,
        )


def test_needs_a_mask_a_gridded_image_and_no_clean_box():
    check_model_update(GOOD, mode=CUBE_SINGLE_FIELD)
    with pytest.raises(ValueError, match="clean_mask or auto_mask_params"):
        check_model_update(
            {"least_squares_params": {"max_steps": 5}}, mode=CUBE_SINGLE_FIELD
        )
    with pytest.raises(ValueError, match="gridded image"):
        check_model_update(
            {**GOOD, "gridding_kernel_params": {"type": "dft"}}, mode=CUBE_SINGLE_FIELD
        )
    with pytest.raises(ValueError, match="clean_box"):
        check_model_update(
            {**GOOD, "cycle_control_params": {"clean_box": (2, 10, 2, 10)}},
            mode=CUBE_SINGLE_FIELD,
        )


def test_the_checker_reaches_the_keys():
    with pytest.raises(ValueError, match=">= 1"):
        check_model_update(
            {**GOOD, "least_squares_params": {"max_steps": 0}}, mode=CUBE_SINGLE_FIELD
        )


def test_least_squares_is_no_public_deconvolver():
    with pytest.raises(ValueError):
        check_model_update({"deconvolver": "least_squares"}, mode=CUBE_SINGLE_FIELD)
