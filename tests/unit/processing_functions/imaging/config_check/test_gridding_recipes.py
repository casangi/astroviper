"""The shared checker and kernel builder agree on defaults and explicit overrides."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check._gridding import (
    check_gridding_kernel_parameters,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
    CASA_GRIDDING_KERNEL_PARAMS,
    DEFAULT_GRIDDING_KERNEL_PARAMS,
    resolve_gridding_kernel_params,
)


@pytest.mark.parametrize(
    "recipe, expected",
    [
        (None, DEFAULT_GRIDDING_KERNEL_PARAMS),
        ({}, DEFAULT_GRIDDING_KERNEL_PARAMS),
        ({"type": "prolate_spheroidal_casa"}, CASA_GRIDDING_KERNEL_PARAMS),
        (
            {"type": "prolate_spheroidal_casa", "oversampling": np.int64(200)},
            {**CASA_GRIDDING_KERNEL_PARAMS, "oversampling": 200},
        ),
        (
            {"type": "prolate_spheroidal_casa", "interpolation": "linear"},
            {**CASA_GRIDDING_KERNEL_PARAMS, "interpolation": "linear"},
        ),
        (
            {"type": "exponential_semicircle"},
            {**DEFAULT_GRIDDING_KERNEL_PARAMS, "type": "exponential_semicircle"},
        ),
        ({"type": "dft"}, {"type": "dft"}),
    ],
)
def test_recipe_defaults_and_overrides_preserve_caller_input(recipe, expected):
    before = deepcopy(recipe)
    check_gridding_kernel_parameters(recipe)
    resolved = resolve_gridding_kernel_params(recipe)
    assert resolved == expected
    assert recipe == before
    assert resolved is not recipe
    assert resolved is not expected
    if "support" in resolved:
        assert type(resolved["support"]) is int
        assert type(resolved["oversampling"]) is int


@pytest.mark.parametrize(
    "recipe",
    [
        {"type": "prolate_spheroidal_casa", "support": 9},
        {"type": "prolate_spheroidal_casa", "oversampling": 0},
        {"type": "prolate_spheroidal_casa", "interpolation": "cubic"},
        {"type": "dft", "support": 7},
        {"type": "dft", "oversampling": 100},
        {"type": "dft", "interpolation": "nearest"},
        {"type": "dft", "unknown": True},
        {"support": False},
        {"oversampling": False},
    ],
)
def test_explicit_invalid_values_are_rejected_by_both_boundaries(recipe):
    for check in (check_gridding_kernel_parameters, resolve_gridding_kernel_params):
        with pytest.raises((TypeError, ValueError)):
            check(recipe)


@pytest.mark.parametrize("recipe", [False, [], ""])
def test_false_like_non_mappings_do_not_select_defaults(recipe):
    with pytest.raises(TypeError):
        check_gridding_kernel_parameters(recipe)
    with pytest.raises(ValueError):
        resolve_gridding_kernel_params(recipe)
