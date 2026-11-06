"""Density operations reject unsupported weighting before data access."""

import pytest


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
