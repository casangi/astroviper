"""Streaming preflight uses selected flags and the production weight kernels."""

import numpy as np
import pytest
import xarray as xr

from astroviper.node_tasks.imaging.config_check import preflight_imaging_inputs
from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
)
from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_input_metadata,
)


def inputs():
    dims = ("time", "baseline_id", "frequency", "polarization")
    shape = (2, 1, 2, 2)
    dataset = xr.Dataset(
        {
            "VISIBILITY": (dims, np.zeros(shape, dtype=np.complex128)),
            "WEIGHT": (dims, np.ones(shape, dtype=np.float64)),
            "FLAG": (dims, np.zeros(shape, dtype=bool)),
            "UVW": (
                ("time", "baseline_id", "uvw_label"),
                np.tile([10.0, 5.0, 0.0], (2, 1, 1)),
            ),
        },
        coords={
            "time": [0.0, 1.0],
            "baseline_id": [0],
            "frequency": [1e9, 1.2e9],
            "polarization": ["XX", "YY"],
            "uvw_label": ["u", "v", "w"],
        },
        attrs={
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "weight": "WEIGHT",
                    "flag": "FLAG",
                    "uvw": "UVW",
                }
            },
            "partition_info": {
                "field_name": ["target"],
                "spectral_window_name": "spw0",
            },
        },
    )
    config = {
        "deconvolver": "mtmfs",
        "processing_set_data_group_name": "base",
        "instrument_polarization_basis": "linear",
        "single_precision_image": False,
        "image_params": {
            "frequency_coords": [1e9, 1.2e9],
            "reference_frequency": 1.1e9,
            "n_taylor_terms": 2,
            "polarization_coords": ["I", "Q"],
            "image_size": [8, 8],
            "cell_size": [-1e-5, 1e-5],
        },
        "imaging_weights_params": {"weighting": "natural"},
    }
    return {"ms": dataset}, config


@pytest.mark.parametrize("role", ["correlated_data", "weight", "flag", "uvw"])
def test_role_targets_are_checked_from_metadata(role):
    processing_set, config = inputs()
    source = processing_set["ms"]
    source.attrs["data_groups"]["base"][role] = "MISSING_VARIABLE"
    with pytest.raises(KeyError, match=role):
        check_input_metadata(source, config, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize("variable", ["VISIBILITY", "WEIGHT", "FLAG", "UVW"])
def test_input_axes_are_checked_before_array_reads(variable):
    processing_set, config = inputs()
    source = processing_set["ms"]
    source[variable] = source[variable].rename({"time": "wrong_time"})
    with pytest.raises(ValueError, match="dimensions"):
        check_input_metadata(source, config, mode=CUBE_SINGLE_FIELD)


def test_structure_preflight_does_not_load_visibility_or_weight_arrays():
    import dask.array as da
    from dask import delayed

    processing_set, config = inputs()
    source = processing_set["ms"]

    def forbidden_read():
        raise AssertionError("Metadata check loaded a sample array")

    for name in ("VISIBILITY", "WEIGHT", "FLAG", "UVW"):
        variable = source[name]
        source[name] = xr.DataArray(
            da.from_delayed(
                delayed(forbidden_read)(), shape=variable.shape, dtype=variable.dtype
            ),
            dims=variable.dims,
        )
    check_input_metadata(source, config, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize(
    "weighting,scope",
    [
        ("natural", "local"),
        ("natural", "global"),
        ("briggs", "local"),
        ("briggs", "global"),
        ("uniform", "global"),
        ("briggs_abs", "local"),
    ],
)
def test_preflight_uses_actual_weighting_and_preserves_input(weighting, scope):
    processing_set, config = inputs()
    config["imaging_weights_params"] = {
        "weighting": weighting,
        "weighting_scope": scope,
        "robust": 0.5,
        "briggs_abs_noise": 1.0,
    }
    snapshot = processing_set["ms"].copy(deep=True)
    result = preflight_imaging_inputs(
        processing_set, config, mode=CONTINUUM_SINGLE_FIELD
    )
    assert result["spectral_design_rank"] == 2
    assert result["distinct_frequency_count"] == 2
    approximation = result["taylor_hessian_approximation"]
    assert approximation["approximation"] == "frequency_moment_gram"
    assert approximation["order_zero_noise_amplification"] == pytest.approx(1.0)
    assert approximation["field"] == ["target"]
    assert (
        approximation["spatial_blocks"]["representation"]
        == "periodic_spatial_fourier_blocks"
    )
    assert (
        approximation["spatial_blocks"]["resolution_recipe"]
        == "wavelength_over_longest_usable_projected_baseline"
    )
    xr.testing.assert_identical(processing_set["ms"], snapshot)


@pytest.mark.parametrize("excluded", ["flagged", "zero_weight"])
def test_gaussian_resolution_ignores_unusable_long_baselines(excluded):
    processing_set, config = inputs()
    source = processing_set["ms"]
    source.UVW.values[1, 0] = [1e9, 0, 0]
    if excluded == "flagged":
        source.FLAG.values[1] = True
    else:
        source.WEIGHT.values[1] = 0
    result = preflight_imaging_inputs(
        processing_set, config, mode=CONTINUUM_SINGLE_FIELD
    )
    widths = result["taylor_hessian_approximation"]["spatial_blocks"][
        "gaussian_fwhm_radians"
    ]
    assert widths == pytest.approx(
        [299792458 / (1.2e9 * np.hypot(10, 5)), 299792458 / (1e9 * np.hypot(10, 5))]
    )


def test_flagged_channel_does_not_count_toward_taylor_rank():
    processing_set, config = inputs()
    processing_set["ms"].FLAG.values[:, :, 1, 0] = True
    with pytest.raises(ValueError, match="distinct_frequency_count.*1"):
        preflight_imaging_inputs(processing_set, config, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
def test_single_term_workflows_do_not_request_multiterm_hessian_approximation(mode):
    processing_set, config = inputs()
    config["deconvolver"] = "hogbom"
    config["image_params"]["n_taylor_terms"] = 1
    result = preflight_imaging_inputs(processing_set, config, mode=mode)
    if mode == CUBE_SINGLE_FIELD:
        assert len(result["cube_channel_sampling"]) == 2
    assert "taylor_hessian_approximation" not in result


def test_zero_weights_are_not_nan_samples():
    processing_set, config = inputs()
    processing_set["ms"].WEIGHT.values[:, :, 1, :] = 0
    with pytest.raises(ValueError, match="fewer distinct"):
        preflight_imaging_inputs(processing_set, config, mode=CONTINUUM_SINGLE_FIELD)


def test_cube_rejects_a_channel_without_usable_weight():
    processing_set, config = inputs()
    processing_set["ms"].WEIGHT.values[:, :, 1, :] = 0
    with pytest.raises(ValueError, match="no selected channel has positive"):
        preflight_imaging_inputs(processing_set, config, mode=CUBE_SINGLE_FIELD)


def test_nonfinite_unflagged_statistical_weights_fail():
    processing_set, config = inputs()
    processing_set["ms"].WEIGHT.values[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="unflagged weights"):
        preflight_imaging_inputs(processing_set, config, mode=CONTINUUM_SINGLE_FIELD)


def test_sparse_padding_is_excluded_without_modifying_source_arrays():
    processing_set, config = inputs()
    dataset = processing_set["ms"]
    dataset["TIME_CENTROID"] = (("time", "baseline_id"), [[0.0], [np.nan]])
    dataset.WEIGHT.values[1] = np.nan
    dataset.UVW.values[1] = np.nan
    snapshot = dataset.copy(deep=True)
    result = preflight_imaging_inputs(
        processing_set, config, mode=CONTINUUM_SINGLE_FIELD
    )
    assert result["distinct_frequency_count"] == 2
    xr.testing.assert_identical(dataset, snapshot)


def test_real_row_with_nan_weights_is_not_treated_as_padding():
    processing_set, config = inputs()
    dataset = processing_set["ms"]
    dataset["TIME_CENTROID"] = (("time", "baseline_id"), [[0.0], [1.0]])
    dataset.WEIGHT.values[1] = np.nan
    dataset.UVW.values[1] = np.nan
    with pytest.raises(ValueError, match="unflagged weights"):
        preflight_imaging_inputs(processing_set, config, mode=CONTINUUM_SINGLE_FIELD)
