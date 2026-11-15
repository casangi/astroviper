"""Gridding recipe validation and cached kernel-design constraints."""

from copy import deepcopy

import pytest

from astroviper.processing_functions.imaging.config_check import (
    CUBE_SINGLE_FIELD,
    check_residual_update,
)


@pytest.mark.parametrize(
    "params,padding",
    [
        ([], 1.2),
        ({"unknown": 1}, 1.2),
        ({"type": "bad"}, 1.2),
        ({"support": 2}, 1.2),
        ({"support": True}, 1.2),
        ({"type": "prolate_spheroidal_casa", "support": 5}, 1.2),
        ({"oversampling": 0}, 1.2),
        ({"oversampling": True}, 1.2),
        ({"interpolation": "cubic"}, 1.2),
        ({"type": "least_misfit"}, None),
        ({"type": "least_misfit"}, 1.01),
    ],
)
def test_kernel_recipe_rejected_before_grid(params, padding):
    config = {"gridding_kernel_params": params, "image_params": {}}
    if padding is not None:
        config["image_params"]["fft_padding"] = padding
    with pytest.raises((ValueError, TypeError)):
        check_residual_update(config, mode=CUBE_SINGLE_FIELD)


def test_kernel_design_reuses_production_cache():
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel import (
        _make_gridding_kernel,
        make_gridding_kernel,
    )

    params = {"type": "exponential_semicircle", "support": 5, "oversampling": 12}
    before = deepcopy(params)
    check_residual_update(
        {"gridding_kernel_params": params, "image_params": {"fft_padding": 1.5}},
        mode=CUBE_SINGLE_FIELD,
    )
    hits = _make_gridding_kernel.cache_info().hits
    make_gridding_kernel(params, fft_padding=1.5)
    assert _make_gridding_kernel.cache_info().hits == hits + 1
    assert params == before


def test_kernel_edge_correction_is_rejected_during_parameter_check():
    # This stencil/padding combination is known to fall below the 1e-6 floor.
    with pytest.raises(ValueError, match="below the floor"):
        check_residual_update(
            {
                "gridding_kernel_params": {
                    "type": "exponential_semicircle",
                    "support": 17,
                },
                "image_params": {"fft_padding": 1.05},
            },
            mode=CUBE_SINGLE_FIELD,
        )


def test_recipe_schema_requires_odd_support_before_constructing_a_kernel():
    from astroviper.processing_functions.imaging.config_check._gridding import (
        check_gridding_kernel_parameters,
    )

    with pytest.raises(ValueError, match="support must be odd"):
        check_gridding_kernel_parameters({"type": "least_misfit", "support": 4})
    check_gridding_kernel_parameters({"type": "least_misfit", "support": 3})
