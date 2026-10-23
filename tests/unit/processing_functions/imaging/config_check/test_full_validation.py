"""Numerical, product, partition, and full-configuration preflight checks."""

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    check_config,
    check_driver_workflow,
    check_model_update,
    check_residual_update,
)
from astroviper.processing_functions.imaging.config_check.param_check_driver_workflow import (
    check_channel_partitions,
    check_configuration_consistency,
)
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_model_products,
    check_scales,
)
from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_spectral_sampling,
)
from tests.utils.continuum_images import make_runaway_taylor_image


def configuration():
    return {
        "image_params": {
            "image_size": [32, 32],
            "cell_size": [-1e-5, 1e-5],
            "phase_direction": [0.0, 0.0],
            "time_coords": [0.0],
            "polarization_coords": ["I", "Q"],
            "frequency_coords": [1e9, 1.2e9],
            "n_taylor_terms": 2,
            "reference_frequency": 1.1e9,
        },
        "imaging_weights_params": {"weighting": "natural"},
        "cycle_control_params": {
            "max_iter": 10,
            "max_cycles": 2,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.0,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
            "max_iter_per_cycle": -1,
        },
        "deconvolver": "mtmfs",
        "single_precision_image": False,
    }


def test_complete_configuration_is_read_only():
    config = configuration()
    snapshot = deepcopy(config)
    check_config(config, mode=CONTINUUM_SINGLE_FIELD, require_complete=True)
    assert config == snapshot


@pytest.mark.parametrize(
    "group,key",
    [
        ("image_params", "frequency_coords"),
        ("image_params", "reference_frequency"),
        ("image_params", "n_taylor_terms"),
        ("cycle_control_params", "gain"),
    ],
)
def test_complete_configuration_requires_inputs(group, key):
    config = configuration()
    del config[group][key]
    with pytest.raises(KeyError, match=key):
        check_config(config, mode=CONTINUUM_SINGLE_FIELD, require_complete=True)


@pytest.mark.parametrize(
    "params,error",
    [
        ({"weighting": "bad"}, ValueError),
        ({"weighting_scope": "bad"}, ValueError),
        ({"weighting": "briggs"}, KeyError),
        ({"weighting": "briggs", "robust": np.nan}, ValueError),
        ({"weighting": "briggs", "robust": 3}, ValueError),
        ({"weighting": "briggs", "robust": "0"}, TypeError),
        ({"weighting": "briggs_abs", "robust": 0}, KeyError),
        ({"weighting": "briggs_abs", "robust": 0, "briggs_abs_noise": 0}, ValueError),
        ({"casa_weighting_implementation": "False"}, TypeError),
        (
            {
                "weighting": "briggs_abs",
                "robust": 0,
                "briggs_abs_noise": 1,
                "weighting_scope": "global",
            },
            ValueError,
        ),
    ],
)
def test_weighting_configuration(params, error):
    before = dict(params)
    with pytest.raises(error):
        check_residual_update(
            {"imaging_weights_params": params}, mode=CONTINUUM_SINGLE_FIELD
        )
    assert params == before


@pytest.mark.parametrize(
    "controls,error",
    [
        ({"gain": 0}, ValueError),
        ({"gain": 1.1}, ValueError),
        ({"gain": np.nan}, ValueError),
        ({"max_iter": -1}, ValueError),
        ({"max_iter": True}, ValueError),
        ({"max_iter": 1.2}, ValueError),
        ({"max_iter_per_cycle": -2}, ValueError),
        ({"max_cycles": 2.0}, TypeError),
        ({"max_cycles": "auto", "max_psf_fraction": 1}, ValueError),
        ({"max_iter_divergence": 0}, ValueError),
        ({"threshold_tolerance": -1}, ValueError),
        ({"min_psf_fraction": 0.9, "max_psf_fraction": 0.8}, ValueError),
        ({"small_scale_bias": 1.1}, ValueError),
        ({"max_iter_typo": 1}, ValueError),
        ({"record_iteration_history": True}, ValueError),
        ({"entropy_spatial_bins": 0}, ValueError),
        ({"entropy_max_snr": np.inf}, ValueError),
    ],
)
def test_model_control_checks(controls, error):
    with pytest.raises(error):
        check_model_update(
            {"deconvolver": "mtmfs", "cycle_control_params": controls},
            mode=CONTINUUM_SINGLE_FIELD,
        )


@pytest.mark.parametrize(
    "scales", [[], [4, 10], [0, -1], [0, np.nan], [0, 4, 4], [0, 4, 2], [0, 17]]
)
def test_scales_are_rejected_instead_of_sorted_or_dropped(scales):
    with pytest.raises(ValueError):
        check_scales(scales, image_size=[32, 32])


@pytest.mark.parametrize(
    "options",
    [
        {"field_name": "field-0"},
        {"vizualize_graph": True},
        {"compute_backend": "mpi"},
        {"mpi_cluster_setup": {}},
        {"write_visibility_model_to_ps": True},
        {"weight_memory_mode": "cache"},
        {"visibility_memory_mode": "cache"},
        {"wideband_primary_beam_memory_mode": "cache"},
        {"image_chunking": {}},
        {"write_imaging_weights_to_ps": True, "weight_memory_mode": "in_memory"},
        {"weight_memory_mode": "in_place", "skunk_works": True},
        {"n_mapping_parallelism": {"time": 2}},
        {"processing_function_threads": 0},
        {"reduce_n_batch": 1},
        {"disk_chunk_sizes": "bad"},
        {"primary_beam_correction": True, "restore": False},
    ],
)
def test_workflow_options_do_not_silently_fall_back(options):
    with pytest.raises((ValueError, TypeError)):
        check_driver_workflow(options, mode=CONTINUUM_SINGLE_FIELD)


def test_spectral_design_reports_rank_and_counts():
    diagnostics = check_spectral_sampling(
        [1e9, 1.1e9, 1.2e9, 1.2e9],
        [1, 0, 2, 1],
        n_taylor_terms=2,
        reference_frequency=1.1e9,
    )
    assert diagnostics["usable_channel_count"] == 3
    assert diagnostics["distinct_frequency_count"] == 2
    assert diagnostics["spectral_design_rank"] == 2
    assert diagnostics["spectral_design_condition_number"] > 1
    assert diagnostics["spectral_design_reciprocal_condition_number"] > 0


@pytest.mark.parametrize(
    "frequencies,weights",
    [
        ([1e9, 1e9], [1, 1]),
        ([1e9, 1.2e9], [0, 0]),
        ([1e9, 1.2e9], [1, -1]),
        ([1e9, 1.2e9], [1, np.nan]),
        ([1e9, 1.2e9], [1, np.inf]),
    ],
)
def test_spectral_errors_include_actionable_context(frequencies, weights):
    with pytest.raises(
        ValueError,
        match="Spectral-design preflight.*field.*spectral_window.*polarization.*usable_channel_count.*spectral_design_rank",
    ):
        check_spectral_sampling(
            frequencies,
            weights,
            n_taylor_terms=2,
            reference_frequency=1.1e9,
            context={
                "field": "target",
                "spectral_window": "spw0",
                "polarization": ["I"],
            },
        )


def test_spectral_condition_is_invariant_to_common_weight_scale():
    args = dict(n_taylor_terms=2, reference_frequency=1.1e9)
    first = check_spectral_sampling([1e9, 1.2e9], [1, 3], **args)
    second = check_spectral_sampling([1e9, 1.2e9], [1e200, 3e200], **args)
    assert (
        first["spectral_design_condition_number"]
        == second["spectral_design_condition_number"]
    )


def test_full_rank_poor_condition_and_extrapolation_warn():
    with pytest.warns(RuntimeWarning, match="Poorly conditioned spectral design"):
        diagnostics = check_spectral_sampling(
            [1e9, 1e9 + 1e4],
            [1, 1],
            n_taylor_terms=2,
            reference_frequency=1e9,
            precision="float32",
        )
    assert diagnostics["spectral_design_rank"] == 2
    with pytest.warns(RuntimeWarning, match="outside the usable frequency range"):
        check_spectral_sampling(
            [1e9, 1.2e9], [1, 1], n_taylor_terms=2, reference_frequency=1.3e9
        )


def test_hessian_preflight_records_distinct_diagnostic_without_mutation():
    image = make_runaway_taylor_image()
    snapshot = image.copy(deep=True)
    result = check_model_products(
        image,
        {"deconvolver": "mtmfs", "cycle_control_params": {"scales": [0.0]}},
        mode=CONTINUUM_SINGLE_FIELD,
    )
    assert result["effective_scales"] == [0.0]
    assert result["taylor_hessian_condition_number"][0]["per_scale"][0] > 0
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize(
    "fault",
    ["psf_count", "psf_precision", "model_dimensions", "mask_precision", "hessian"],
)
def test_product_checks_reject_invalid_inputs(fault):
    image = make_runaway_taylor_image()
    if fault == "psf_count":
        image = image.isel(psf_taylor_order=slice(0, 2))
    elif fault == "psf_precision":
        image["POINT_SPREAD_FUNCTION"] = image.POINT_SPREAD_FUNCTION.astype(np.float32)
    elif fault == "model_dimensions":
        image["SKY_MODEL"] = image.SKY_RESIDUAL.isel(taylor_term=0, drop=True)
    elif fault == "mask_precision":
        image["MASK"] = xr.zeros_like(image.SKY_RESIDUAL, dtype=np.int32)
        image.attrs["data_groups"]["residual"]["mask"] = "MASK"
    else:
        image.POINT_SPREAD_FUNCTION.values[:, 2] = 0
    with pytest.raises(
        (ValueError, TypeError), match="Taylor|precision|dimensions|planes"
    ):
        check_model_products(
            image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
        )


def test_channel_partitions_are_exclusive_and_complete():
    processing_set = {"ms": xr.Dataset(coords={"frequency": [1e9, 1.1e9, 1.2e9]})}
    image_params = {"frequency_coords": [1e9, 1.1e9, 1.2e9]}
    mapping = {
        0: {"data_selection": {"ms": {"frequency": slice(0, 2)}}},
        1: {"data_selection": {"ms": {"frequency": slice(2, 3)}}},
    }
    check_channel_partitions(mapping, processing_set, image_params=image_params)
    mapping[1]["data_selection"]["ms"]["frequency"] = slice(1, 3)
    with pytest.raises(ValueError, match="repeated=\\[1\\]"):
        check_channel_partitions(mapping, processing_set, image_params=image_params)
    mapping[1]["data_selection"]["ms"]["frequency"] = slice(3, 3)
    with pytest.raises(ValueError, match="missing=\\[2\\]"):
        check_channel_partitions(mapping, processing_set, image_params=image_params)


def test_shared_basis_cannot_change_between_stages():
    first = configuration()
    second = deepcopy(first)
    second["image_params"]["reference_frequency"] *= 2
    with pytest.raises(ValueError, match="reference_frequency"):
        check_configuration_consistency(first, second)


def test_later_plane_hessian_failure_precedes_every_clean_update():
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        model_update_mtmfs_single_field,
    )

    first = make_runaway_taylor_image()
    second = first.copy(deep=True).assign_coords(time=[1.0])
    second.POINT_SPREAD_FUNCTION.values[:, 2] = 0.0
    image = xr.concat([first, second], dim="time")
    snapshot = image.copy(deep=True)
    with pytest.raises(ValueError, match="Taylor-Hessian preflight.*time=1"):
        model_update_mtmfs_single_field(
            image,
            "mtmfs",
            {"max_iter": 1, "gain": 0.1, "threshold": 0.0, "scales": [0.0]},
            input_checked=True,
        )
    xr.testing.assert_identical(image, snapshot)


def test_frequency_lookup_matches_dense_comparison_without_changing_coordinates():
    from astroviper.processing_functions.imaging.config_check._common import (
        selected_frequency_mask,
    )

    frequencies = np.array([1.2e9, 1e9, 1.1e9 + 1e-4, 1.3e9])
    requested = np.array([1.1e9, 1e9, 1.1e9])
    original = requested.copy()
    expected = np.isclose(
        frequencies[:, None], requested[None, :], rtol=1e-12, atol=0
    ).any(axis=1)
    np.testing.assert_array_equal(
        selected_frequency_mask(frequencies, requested), expected
    )
    np.testing.assert_array_equal(requested, original)
    assert not selected_frequency_mask(frequencies, []).any()


def test_invertible_ill_conditioned_hessian_warns_and_records_metric():
    image = make_runaway_taylor_image()
    image.POINT_SPREAD_FUNCTION.values[:, 2] *= 4e-12
    with pytest.warns(RuntimeWarning, match="Poorly conditioned Taylor Hessian"):
        diagnostics = check_model_products(
            image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
        )
    condition = diagnostics["taylor_hessian_condition_number"][0]["per_scale"][0]
    assert condition == pytest.approx(1e12)
