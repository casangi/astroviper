"""The model-update checker owns ``auto_mask_params``."""

import copy
import math

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_model_update,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_auto_mask_params,
)


def test_a_full_set_of_values_is_accepted():
    check_auto_mask_params(
        {
            "sidelobe_threshold": 2,
            "noise_threshold": np.float32(4.25),
            "low_noise_threshold": 1.5,
            "negative_threshold": 0.0,
            "smooth_factor": 1.0,
            "min_beam_fraction": 0.0,
            "cut_threshold": 0.01,
            "grow_iterations": np.int64(0),
            "grow_prune": np.True_,
            "min_percent_change": -1.0,
            "fast_noise": False,
        }
    )


def test_nothing_is_inserted():
    params = {"noise_threshold": 4.25}
    given = copy.deepcopy(params)
    check_auto_mask_params(params)
    assert params == given


def test_refuses_what_is_no_mapping():
    with pytest.raises(TypeError, match="auto_mask_params must be a mapping"):
        check_auto_mask_params(["noise_threshold"])


def test_refuses_an_unknown_key():
    with pytest.raises(ValueError, match="sidelobethreshold"):
        check_auto_mask_params({"sidelobethreshold": 3.0})


@pytest.mark.parametrize(
    "key, value, error, message",
    [
        ("sidelobe_threshold", -1.0, ValueError, "must not be negative"),
        ("noise_threshold", "5", TypeError, "real"),
        ("low_noise_threshold", math.nan, ValueError, "finite"),
        ("negative_threshold", math.inf, ValueError, "finite"),
        ("negative_threshold", -1.0, ValueError, "must not be negative"),
        ("min_beam_fraction", True, TypeError, "real"),
        ("min_beam_fraction", [0.3], TypeError, "real"),
        ("smooth_factor", 0.0, ValueError, "strictly positive"),
        ("cut_threshold", 0.0, ValueError, "must lie in"),
        ("cut_threshold", 1.0, ValueError, "must lie in"),
        ("grow_iterations", -1, ValueError, ">= 0"),
        ("grow_iterations", 1.5, TypeError, "integer"),
        ("grow_iterations", True, TypeError, "integer"),
        ("grow_prune", 1, TypeError, "must be a bool"),
        ("fast_noise", "yes", TypeError, "must be a bool"),
        ("min_percent_change", math.inf, ValueError, "finite"),
    ],
)
def test_refuses_a_bad_value(key, value, error, message):
    with pytest.raises(error, match=message):
        check_auto_mask_params({key: value})


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
class TestModelUpdateChecker:
    def test_no_automatic_mask(self, mode):
        check_model_update({"auto_mask_params": None, "clean_mask": "m"}, mode=mode)

    def test_an_automatic_mask(self, mode):
        check_model_update({"auto_mask_params": {}, "clean_mask": None}, mode=mode)

    @pytest.mark.parametrize("clean_mask", ["mask.npy", np.ones((4, 4), dtype=bool)])
    def test_refuses_a_clean_mask_beside_it(self, mode, clean_mask):
        with pytest.raises(ValueError, match="exclude each other"):
            check_model_update(
                {"auto_mask_params": {}, "clean_mask": clean_mask}, mode=mode
            )

    def test_refuses_a_bad_value(self, mode):
        with pytest.raises(ValueError, match="Unknown auto_mask_params keys"):
            check_model_update({"auto_mask_params": {"noise": 5.0}}, mode=mode)
