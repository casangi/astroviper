"""Numerical, product, partition, and full-configuration preflight checks."""

import warnings
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
    assert first["taylor_zero_noise_amplification"] == pytest.approx(
        second["taylor_zero_noise_amplification"]
    )


def test_weak_slope_does_not_warn_when_order_zero_noise_is_small():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = check_spectral_sampling(
            [1e9, 1e9 + 1e4],
            [1, 1],
            n_taylor_terms=2,
            reference_frequency=1e9,
            precision="float32",
        )
    assert diagnostics["spectral_design_rank"] == 2
    assert diagnostics["spectral_design_reciprocal_condition_number"] < np.sqrt(
        np.finfo(np.float32).eps
    )
    assert diagnostics["taylor_zero_noise_amplification"] == pytest.approx(np.sqrt(2))
    assert not caught


def test_reference_frequency_extrapolation_still_warns():
    with pytest.warns(RuntimeWarning, match="outside the usable frequency range"):
        check_spectral_sampling(
            [1e9, 1.2e9], [1, 1], n_taylor_terms=2, reference_frequency=1.3e9
        )


@pytest.mark.parametrize("nterms", [1, 2, 3, 4])
def test_order_zero_noise_matches_propagated_channel_covariance(nterms):
    # Propagate unit channel perturbations through an independent least-squares
    # solve, including unequal weights and a channel excluded by zero weight.
    frequencies = np.linspace(1e9, 1.2e9, 7)
    weights = np.array([1.0, 0.0, 2.0, 0.5, 4.0, 1.5, 3.0])
    original_weights = weights.copy()
    original_frequencies = frequencies.copy()
    diagnostics = check_spectral_sampling(
        frequencies,
        weights,
        n_taylor_terms=nterms,
        reference_frequency=1.1e9,
    )
    selected = weights > 0
    q = weights[selected] / weights.max()
    x = (frequencies[selected] - 1.1e9) / 1.1e9
    design = np.polynomial.polynomial.polyvander(x, nterms - 1) * np.sqrt(q)[:, None]
    responses = np.linalg.lstsq(design, np.diag(np.sqrt(q)), rcond=None)[0]
    order_zero_variance = np.sum(responses[0] ** 2)
    constant_variance = np.sum(q**2) / np.sum(q) ** 2
    assert diagnostics["taylor_zero_noise_amplification"] == pytest.approx(
        np.sqrt(order_zero_variance / constant_variance)
    )
    assert diagnostics["taylor_zero_noise_amplification_limit"] == 3.0
    np.testing.assert_array_equal(weights, original_weights)
    np.testing.assert_array_equal(frequencies, original_frequencies)


@pytest.mark.parametrize("nterms", [1, 2])
def test_centered_constant_and_slope_do_not_amplify_order_zero_noise(nterms):
    diagnostics = check_spectral_sampling(
        [1e9 - 1e4, 1e9, 1e9 + 1e4],
        [1, 1, 1],
        n_taylor_terms=nterms,
        reference_frequency=1e9,
    )
    assert diagnostics["taylor_zero_noise_amplification"] == pytest.approx(1.0)


def test_curvature_coupling_warns_about_order_zero_noise_without_rejecting():
    with pytest.warns(
        RuntimeWarning, match="Taylor order-zero noise amplification exceeds 3.0"
    ):
        diagnostics = check_spectral_sampling(
            [1e9, 1.001e9, 1.2e9],
            [1, 1, 1],
            n_taylor_terms=3,
            reference_frequency=1.1e9,
        )
    assert diagnostics["spectral_design_rank"] == 3
    assert diagnostics["taylor_zero_noise_amplification"] > 3


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
    amplification = result["taylor_hessian_order_zero_noise_amplification"][0]
    assert amplification["per_scale"] == pytest.approx([1.0])
    assert amplification["limit"] == 3.0
    assert amplification["covariance_assumption"] == "proportional_to_hessian"
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


def test_small_uncoupled_taylor_moment_does_not_warn_about_order_zero():
    image = make_runaway_taylor_image()
    image.POINT_SPREAD_FUNCTION.values[:, 2] *= 4e-12
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = check_model_products(
            image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
        )
    condition = diagnostics["taylor_hessian_condition_number"][0]["per_scale"][0]
    assert condition == pytest.approx(1e12)
    assert diagnostics["taylor_hessian_order_zero_noise_amplification"][0][
        "per_scale"
    ] == pytest.approx([1.0])
    # This deliberately nonphysical divergence fixture has negative spatial
    # transfer modes, which are independent of local order-zero conditioning.
    assert all(
        "Spatial Taylor-Hessian diagnostic" in str(item.message) for item in caught
    )
    assert (
        diagnostics["taylor_hessian_spatial_blocks"][0]["non_positive_mode_count"] > 0
    )


@pytest.mark.parametrize("rho", [0.0, 0.9, 0.95, 0.99])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hessian_order_zero_warning_tracks_coupling_on_every_scale(rho, dtype):
    # H = a * [[1, rho], [rho, 1]] has G_H0 = 1 / sqrt(1 - rho**2),
    # independently of the common scale-filter amplitude a.
    image = make_runaway_taylor_image(dtype=dtype)
    image.POINT_SPREAD_FUNCTION.values[...] = 0
    image.POINT_SPREAD_FUNCTION.values[0, :, 0, 16, 16] = [1, rho, 1]
    later = image.copy(deep=True).assign_coords(time=[1.0])
    image = xr.concat([image, later], dim="time")
    snapshot = image.copy(deep=True)
    expected = 1 / np.sqrt(1 - rho**2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = check_model_products(
            image,
            {"deconvolver": "mtmfs", "cycle_control_params": {"scales": [0.0, 2.0]}},
            mode=CONTINUUM_SINGLE_FIELD,
        )
    warnings_about_noise = [
        warning
        for warning in caught
        if "Taylor-Hessian order-zero noise amplification exceeds 3.0"
        in str(warning.message)
    ]
    assert len(warnings_about_noise) == (2 if expected > 3 else 0)
    spatial_warnings = [
        item
        for item in caught
        if "Spatial Taylor-Hessian diagnostic" in str(item.message)
    ]
    assert len(spatial_warnings) == (2 if expected > 3 else 0)
    assert len(caught) == len(warnings_about_noise) + len(spatial_warnings)
    entries = diagnostics["taylor_hessian_order_zero_noise_amplification"]
    assert [entry["time_index"] for entry in entries] == [0, 1]
    for entry in entries:
        assert entry["per_scale"] == pytest.approx([expected, expected], rel=5e-5)
        assert entry["polarization_index"] == 0
    xr.testing.assert_identical(image, snapshot)


@pytest.mark.parametrize("factor", [1e-6, 1.0, 1e6])
def test_hessian_order_zero_noise_is_invariant_to_taylor_and_flux_units(factor):
    image = make_runaway_taylor_image()
    # H = factor * D [[1, .9], [.9, 1]] D with D = diag(1, 1e-6).
    # Both common flux scaling and higher-order coefficient units must cancel.
    image.POINT_SPREAD_FUNCTION.values[...] = 0
    image.POINT_SPREAD_FUNCTION.values[0, :, 0, 16, 16] = factor * np.array(
        [1.0, 0.9e-6, 1e-12]
    )
    diagnostics = check_model_products(
        image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
    )
    assert diagnostics["taylor_hessian_order_zero_noise_amplification"][0][
        "per_scale"
    ] == pytest.approx([1 / np.sqrt(1 - 0.9**2)])


@pytest.mark.parametrize("reference", [np.float32(100.0), xr.DataArray(100.0)])
def test_model_product_metadata_uses_serializable_scalars(reference):
    import json

    config = {
        "deconvolver": "mtmfs",
        "image_params": {"reference_frequency": reference},
    }
    diagnostics = check_model_products(
        make_runaway_taylor_image(), config, mode=CONTINUUM_SINGLE_FIELD
    )
    restored = json.loads(json.dumps(diagnostics))
    assert restored["reference_frequency"] == 100.0
    assert type(diagnostics["reference_frequency"]) is float
    assert config["image_params"]["reference_frequency"] is reference


@pytest.mark.parametrize("nterms", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_narrow_band_hessian_is_checked_independently_of_taylor_units(nterms, dtype):
    # Narrow-band moments are small even when their constraints are independent.
    # The three-term Hessian has equilibrated eigenvalues 1 and 1 +/- 1/sqrt(3).
    moments = [1.0, 0.0, 1e-12] if nterms == 2 else [1.0, 0.0, 1e-12, 0.0, 3e-24]
    psf = np.zeros((1, len(moments), 1, 8, 8), dtype=dtype)
    psf[0, :, 0, 4, 4] = moments
    image = xr.Dataset(
        {
            "SKY_RESIDUAL": (
                ("time", "taylor_term", "polarization", "l", "m"),
                np.ones((1, nterms, 1, 8, 8), dtype=dtype),
            ),
            "POINT_SPREAD_FUNCTION": (
                ("time", "psf_taylor_order", "polarization", "l", "m"),
                psf,
            ),
        },
        attrs={
            "data_groups": {
                "residual": {
                    "sky": "SKY_RESIDUAL",
                    "point_spread_function": "POINT_SPREAD_FUNCTION",
                }
            }
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = check_model_products(
            image, {"deconvolver": "mtmfs"}, mode=CONTINUUM_SINGLE_FIELD
        )
    assert diagnostics["n_taylor_terms"] == nterms
    assert diagnostics["taylor_hessian_condition_number"][0]["per_scale"][0] > 1e11
    expected = 1.0 if nterms == 2 else np.sqrt(1.5)
    assert diagnostics["taylor_hessian_order_zero_noise_amplification"][0][
        "per_scale"
    ] == pytest.approx([expected])
    assert not caught


def test_spectral_sampling_accepts_xarray_scalar_without_changing_it():
    reference = xr.DataArray(1.1e9, attrs={"units": "Hz"})
    original = reference.copy(deep=True)
    diagnostics = check_spectral_sampling(
        [1.0e9, 1.2e9], [1.0, 1.0], n_taylor_terms=2, reference_frequency=reference
    )
    assert diagnostics["spectral_design_rank"] == 2
    assert diagnostics["reference_frequency"] == 1.1e9
    xr.testing.assert_identical(reference, original)
