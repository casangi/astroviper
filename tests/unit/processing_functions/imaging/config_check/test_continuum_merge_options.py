"""New continuum options retain strict validation across the merged workflows."""

from copy import deepcopy

import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    check_driver_workflow,
    check_residual_update,
)
from astroviper.processing_functions.imaging.image_continuum_single_field import (
    make_mvc_taylor_normal_equation_contributions,
    restore_continuum_image,
)


@pytest.mark.parametrize("approximation", ["channel_psf", "local_beam_moments"])
@pytest.mark.parametrize("include_static", [True, False])
def test_mvc_options_preserve_supplied_configuration(approximation, include_static):
    config = {
        "mvc_model_update_approximation": approximation,
        "include_local_beam_static": include_static,
    }
    before = deepcopy(config)
    check_residual_update(config, mode=CONTINUUM_SINGLE_FIELD)
    assert config == before


@pytest.mark.parametrize(
    "value", ["CHANNEL_PSF", "Local_Beam_Moments", "invalid", None, 1, ["channel_psf"]]
)
def test_mvc_approximation_rejected_before_accessing_arrays(value):
    with pytest.raises(ValueError, match="mvc_model_update_approximation"):
        make_mvc_taylor_normal_equation_contributions(
            None,
            None,
            None,
            None,
            None,
            n_taylor_terms=2,
            reference_frequency=1e9,
            mvc_model_update_approximation=value,
        )


@pytest.mark.parametrize("value", ["False", 0, None])
def test_static_moment_switch_requires_boolean(value):
    with pytest.raises(TypeError, match="include_local_beam_static"):
        check_residual_update(
            {"include_local_beam_static": value}, mode=CONTINUUM_SINGLE_FIELD
        )


@pytest.mark.parametrize("output", ["apparent", "primary_beam_corrected", "both", None])
def test_restoration_output_is_validated_without_mutation(output):
    config = {
        "restore": True,
        "primary_beam_correction": True,
        "restoration_output": output,
    }
    before = deepcopy(config)
    check_driver_workflow(config, mode=CONTINUUM_SINGLE_FIELD)
    assert config == before


@pytest.mark.parametrize("output", ["APPARENT", "invalid", 1, ["both"]])
def test_restoration_output_rejected_before_accessing_image(output):
    with pytest.raises(ValueError, match="restoration_output"):
        restore_continuum_image(None, restoration_output=output)


@pytest.mark.parametrize("output", ["primary_beam_corrected", "both"])
def test_corrected_output_requires_primary_beam_correction(output):
    with pytest.raises(ValueError, match="requires primary_beam_correction"):
        restore_continuum_image(None, restoration_output=output)
