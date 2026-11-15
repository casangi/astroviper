"""Shared configuration checks reflect the continuum capabilities from main."""

from copy import deepcopy

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_driver_workflow,
    check_model_update,
)
from astroviper.processing_functions.imaging.config_check._gridding import (
    check_gridding_kernel,
    check_gridding_kernel_parameters,
)
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_model_products,
)
from tests.utils.continuum_images import make_runaway_taylor_image


@pytest.mark.parametrize(
    "deconvolver,nterms", [("hogbom", 1), ("asp", 1), ("mtmfs", 2)]
)
def test_each_continuum_deconvolver_can_record_history(deconvolver, nterms):
    config = {
        "deconvolver": deconvolver,
        "image_params": {"n_taylor_terms": nterms},
        "cycle_control_params": {"record_iteration_history": True},
    }
    before = deepcopy(config)

    check_model_update(config, mode=CONTINUUM_SINGLE_FIELD)

    assert config == before


@pytest.mark.parametrize("nterms", [0, 2, 3, 4])
def test_continuum_asp_requires_one_taylor_term(nterms):
    with pytest.raises(ValueError, match="n_taylor_terms"):
        check_model_update(
            {"deconvolver": "asp", "image_params": {"n_taylor_terms": nterms}},
            mode=CONTINUUM_SINGLE_FIELD,
        )


def test_continuum_asp_requires_an_explicit_order_in_image_configuration():
    with pytest.raises(KeyError, match="n_taylor_terms"):
        check_model_update(
            {"deconvolver": "asp", "image_params": {}}, mode=CONTINUUM_SINGLE_FIELD
        )


def test_actual_multi_term_products_remain_invalid_for_asp():
    image = make_runaway_taylor_image()

    with pytest.raises(ValueError, match="Continuum asp requires n_taylor_terms=1"):
        check_model_products(image, {"deconvolver": "asp"}, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize("deconvolver", ["hogbom", "asp", "mtmfs"])
def test_continuum_graph_cannot_supply_exact_subtraction_psf(deconvolver):
    with pytest.raises(ValueError, match="not supported by continuum imaging"):
        check_model_update(
            {
                "deconvolver": deconvolver,
                "cycle_control_params": {"exact_psf_subtraction": True},
            },
            mode=CONTINUUM_SINGLE_FIELD,
        )


def test_current_asp_control_names_are_accepted_without_rewriting():
    controls = {
        "asp_fused_threshold": 0.0,
        "asp_psf_width": -1.0,
        "asp_largest_scale": -1,
        "asp_stop_point_mode": -1,
        "asp_scale_normalization_method": 1,
        "asp_residual_component_max_epsilon_fraction": 0.2,
        "asp_residual_component_min_fraction": 0.01,
        "asp_residual_component_noise_floor": 0.0,
        "asp_residual_component_switch_factor": 1.0,
        "asp_residual_component_floor_at_threshold": True,
        "asp_residual_component_positive_only": True,
        "asp_no_change_tolerance": 0.0,
    }
    before = controls.copy()

    check_model_update(
        {"deconvolver": "asp", "cycle_control_params": controls},
        mode=CONTINUUM_SINGLE_FIELD,
    )

    assert controls == before


@pytest.mark.parametrize("name", ["fusedthreshold", "psf_width", "asp_rc_noise_floor"])
def test_obsolete_asp_control_names_are_not_silently_rewritten(name):
    with pytest.raises(ValueError, match="Unknown cycle/deconvolution controls"):
        check_model_update(
            {"deconvolver": "asp", "cycle_control_params": {name: 0}},
            mode=CONTINUUM_SINGLE_FIELD,
        )


@pytest.mark.parametrize("deconvolver", ["hogbom", "mtmfs"])
def test_asp_options_are_not_ignored_by_other_deconvolvers(deconvolver):
    with pytest.raises(ValueError, match="Asp controls require"):
        check_model_update(
            {
                "deconvolver": deconvolver,
                "cycle_control_params": {"asp_residual_component_positive_only": True},
            },
            mode=CONTINUUM_SINGLE_FIELD,
        )


@pytest.mark.parametrize("extra", [{"support": 7}, {"oversampling": 100}])
def test_dft_rejects_irrelevant_kernel_overrides(extra):
    with pytest.raises(ValueError, match="DFT accepts only"):
        check_gridding_kernel_parameters({"type": "dft", **extra})


def test_dft_needs_no_kernel_table_or_fft_design_padding():
    params = {"type": "dft"}

    check_gridding_kernel_parameters(params)
    check_gridding_kernel(params, fft_padding=None)

    assert params == {"type": "dft"}


@pytest.mark.parametrize(
    "options,message",
    [
        ({"continuum_mode": "mvc"}, "continuum_mode='mfs'"),
        ({"visibility_memory_mode": "in_memory"}, "visibility_memory_mode='recompute'"),
        ({"visibility_memory_mode": "in_place"}, "visibility_memory_mode='recompute'"),
        ({"image_data_variables_keep": ["visibility"]}, "uv grids"),
        ({"image_data_variables_keep": ["uv_sampling"]}, "uv grids"),
    ],
)
def test_dft_workflow_rejects_uv_grid_dependencies(options, message):
    with pytest.raises(ValueError, match=message):
        check_driver_workflow(
            {"gridding_kernel_params": {"type": "dft"}, **options},
            mode=CONTINUUM_SINGLE_FIELD,
        )


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
def test_dft_accepts_image_products_as_a_numpy_name_array(mode):
    kept = np.asarray(["sky_residual", "point_spread_function"])
    config = {
        "gridding_kernel_params": {"type": "dft"},
        "image_data_variables_keep": kept,
    }

    check_driver_workflow(config, mode=mode)

    assert config["image_data_variables_keep"] is kept


@pytest.mark.parametrize(
    "kept",
    [["visibility"], ["uv_sampling"], np.asarray(["sky_residual", "visibility"])],
)
def test_cube_dft_refuses_uv_outputs_at_the_shared_workflow_boundary(kept):
    with pytest.raises(ValueError, match="uv grids"):
        check_driver_workflow(
            {
                "gridding_kernel_params": {"type": "dft"},
                "image_data_variables_keep": kept,
            },
            mode=CUBE_SINGLE_FIELD,
        )


def test_cube_threshold_shape_is_checked_against_requested_channel_planes():
    with pytest.raises(ValueError, match="per channel threshold"):
        check_model_update(
            {
                "image_params": {
                    "frequency_coords": [1e9, 1.1e9],
                    "polarization_coords": ["I", "Q"],
                },
                "cycle_control_params": {"threshold": np.zeros((1, 2))},
            },
            mode=CUBE_SINGLE_FIELD,
        )
