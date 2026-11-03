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


@pytest.mark.parametrize("deconvolver", ["MTMFS", "unknown", None])
def test_restoration_rejects_algorithm_before_accessing_products(deconvolver):
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        restore_continuum_image,
    )

    with pytest.raises(ValueError, match="deconvolver"):
        restore_continuum_image(None, deconvolver=deconvolver)


@pytest.mark.parametrize("node,params", [(None, {}), (lambda: None, None)])
def test_graph_pair_rejected_before_graph_construction(node, params):
    from astroviper.distributed_applications.imaging.image_continuum_single_field import (
        compute_continuum_graph,
    )

    with pytest.raises(ValueError, match="append_"):
        compute_continuum_graph(
            ps_xdt=None,
            node_task_data_mapping=None,
            cycle_input_params=None,
            reduce_input_params=None,
            disk_chunk_sizes=None,
            processing_set_data_group_name=None,
            monitor_resources_seconds=None,
            task_priorities=None,
            reduce_mode=None,
            reduce_n_batch=None,
            append_node=node,
            append_input_params=params,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"skunk_works": False},
        {"skunk_works": True, "write_visibility_model_to_ps": True},
        {"skunk_works": True, "write_imaging_weights_to_ps": True},
    ],
)
def test_msv2_storage_restrictions(options):
    from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
        check_input_storage,
    )

    original = options.copy()
    with pytest.raises(ValueError, match="Measurement Set"):
        check_input_storage(options, input_is_msv2=True)
    check_input_storage(options, input_is_msv2=False)
    assert options == original


def test_msv2_read_only_supported_path():
    from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
        check_input_storage,
    )

    check_input_storage({"skunk_works": True}, input_is_msv2=True)
    with pytest.raises(ValueError, match="convert"):
        check_input_storage({"skunk_works": True}, input_is_msv2=True, allow_msv2=False)


def test_psf_driver_checks_nested_configuration_before_io(tmp_path):
    from astroviper.distributed_applications.imaging.point_spread_function_cube_single_field import (
        point_spread_function_cube_single_field,
    )

    image_params = {
        "image_size": [0, 16],
        "cell_size": [-1e-5, 1e-5],
        "phase_direction": [0, 0],
        "frequency_coords": [1e9],
        "time_coords": [0],
        "polarization_coords": ["I", "Q"],
    }
    output = tmp_path / "never-created.zarr"
    with pytest.raises(ValueError, match="image_size"):
        point_spread_function_cube_single_field(
            "missing.ps.zarr",
            image_params,
            {"weighting": "natural"},
            image_store=str(output),
        )
    assert not output.exists()


def test_psf_node_checks_configuration_before_loading():
    from astroviper.node_tasks.imaging.point_spread_function_cube_single_field import (
        point_spread_function_cube_single_field,
    )

    with pytest.raises(ValueError, match="image_size"):
        point_spread_function_cube_single_field(
            {"image_size": [0, 16]},
            {"weighting": "natural"},
            None,
            None,
            "missing.ps.zarr",
        )
