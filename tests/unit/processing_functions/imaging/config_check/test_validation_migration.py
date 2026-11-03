"""Regression tests for validation shared with direct processing-function calls."""

import pytest


@pytest.mark.parametrize(
    "name",
    [
        "add_dft_image_continuum_single_field",
        "make_dft_undeconvolved_image_continuum_single_field",
        "make_dft_point_spread_function_continuum_single_field",
        "make_dft_visibility_model_continuum_single_field",
    ],
)
@pytest.mark.parametrize(
    "count,frequency",
    [(1.5, 1e9), (True, 1e9), ("2", 1e9), (1, "1e9"), (1, 0), (1, float("nan"))],
)
def test_dft_rejects_original_parameters_before_accessing_data(name, count, frequency):
    from astroviper.processing_functions.imaging import direct_fourier_transform

    function = getattr(direct_fourier_transform, name)
    with pytest.raises((TypeError, ValueError)):
        function(None, None, count, frequency)


@pytest.mark.parametrize(
    "params",
    [
        {"weighting": "natural"},
        {"weighting": "briggs_abs", "robust": 0, "briggs_abs_noise": 1},
        {"weighting": "Briggs", "robust": 0},
        {"weighting": "briggs", "robust": 0, "casa_weighting_implementation": "false"},
    ],
)
@pytest.mark.parametrize(
    "name",
    ["grid_imaging_weight_density_continuum", "degrid_imaging_weights_continuum"],
)
def test_density_operations_reject_unsupported_configuration_before_data(name, params):
    from astroviper.processing_functions.imaging import calculate_imaging_weights

    function = getattr(calculate_imaging_weights, name)
    with pytest.raises((ValueError, TypeError)):
        if name.startswith("degrid"):
            function(None, None, None, params)
        else:
            function(None, None, params)


@pytest.mark.parametrize("coefficient", [True, "0.5", float("nan"), -0.1, 1.0])
def test_momentum_checks_configuration_before_accessing_arrays(coefficient):
    from astroviper.processing_functions.imaging.cycle_acceleration import (
        nesterov_momentum_model_step,
    )

    with pytest.raises(ValueError, match="momentum_coefficient"):
        nesterov_momentum_model_step(None, None, momentum_coefficient=coefficient)
