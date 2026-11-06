"""Coordinate preflight leaves every bulk input array lazy."""

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


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
@pytest.mark.parametrize(
    "weighting,scope",
    [
        ("natural", "local"),
        ("natural", "global"),
        ("briggs", "local"),
        ("briggs", "global"),
        ("uniform", "local"),
        ("uniform", "global"),
        ("briggs_abs", "local"),
    ],
)
def test_full_preflight_uses_only_metadata(mode, weighting, scope, monkeypatch):
    import dask.array as da
    from dask import delayed

    processing_set, config = inputs()
    config["imaging_weights_params"] = {
        "weighting": weighting,
        "weighting_scope": scope,
        "robust": 0.5,
        "briggs_abs_noise": 1.0,
    }
    if mode == CUBE_SINGLE_FIELD:
        config["deconvolver"] = "hogbom"
        config["image_params"]["n_taylor_terms"] = 1
    source = processing_set["ms"]
    source["TIME_CENTROID"] = (("time", "baseline_id"), np.zeros((2, 1)))

    def forbidden_work(*args, **kwargs):
        raise AssertionError("Coordinate preflight performed bulk imaging work")

    for name in ("VISIBILITY", "WEIGHT", "FLAG", "UVW", "TIME_CENTROID"):
        variable = source[name]
        source[name] = xr.DataArray(
            da.from_delayed(
                delayed(forbidden_work)(), shape=variable.shape, dtype=variable.dtype
            ),
            dims=variable.dims,
        )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights.grid_imaging_weights",
        forbidden_work,
    )
    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.config_check.param_check_model_update.check_taylor_hessian_approximation",
        forbidden_work,
    )
    snapshot = source.copy(deep=False)
    result = preflight_imaging_inputs(processing_set, config, mode=mode)
    channels = (
        result["cube_channel_sampling"] if mode == CUBE_SINGLE_FIELD else [result]
    )
    for channel in channels:
        assert channel["weight_assumption"] == "equal"
        assert channel["support_assumption"] == "selected_frequency_coordinates"
        assert channel["spectral_design_rank_precision"] == "float64"
        assert "taylor_hessian_approximation" not in channel
    # Comparing lazy graphs/metadata must not evaluate the deliberately unreadable data.
    assert source.identical(snapshot)


@pytest.mark.parametrize("mode", [CONTINUUM_SINGLE_FIELD, CUBE_SINGLE_FIELD])
@pytest.mark.parametrize("fault", ["flagged", "zero_weight", "nan_weight", "nan_uvw"])
def test_nominal_support_does_not_depend_on_bulk_sample_values(mode, fault):
    processing_set, config = inputs()
    source = processing_set["ms"]
    if fault == "flagged":
        source.FLAG.values[:, :, 1] = True
    elif fault == "zero_weight":
        source.WEIGHT.values[:, :, 1] = 0
    elif fault == "nan_weight":
        source.WEIGHT.values[:, :, 1] = np.nan
    else:
        source.UVW.values[:] = np.nan
    snapshot = source.copy(deep=True)
    if mode == CUBE_SINGLE_FIELD:
        config["deconvolver"] = "hogbom"
        config["image_params"]["n_taylor_terms"] = 1
    result = preflight_imaging_inputs(processing_set, config, mode=mode)
    if mode == CUBE_SINGLE_FIELD:
        assert len(result["cube_channel_sampling"]) == 2
    else:
        assert result["distinct_frequency_count"] == 2
        assert result["spectral_design_rank"] == 2
    xr.testing.assert_identical(source, snapshot)


@pytest.mark.parametrize("reverse", [False, True])
def test_cube_preflight_preserves_nearest_channel_assignment(reverse):
    processing_set, config = inputs()
    config["deconvolver"] = "hogbom"
    config["image_params"]["n_taylor_terms"] = 1
    if reverse:
        config["image_params"]["frequency_coords"] = [1.2e9, 1e9]
    processing_set["ms"] = processing_set["ms"].assign_coords(
        frequency=processing_set["ms"].frequency.values + 5000
    )
    result = preflight_imaging_inputs(processing_set, config, mode=CUBE_SINGLE_FIELD)
    assert len(result["cube_channel_sampling"]) == 2
    assert all(
        channel["distinct_frequency_count"] == 1
        for channel in result["cube_channel_sampling"]
    )


def test_repeated_measurement_sets_preserve_rank_decision_and_threshold():
    processing_set, config = inputs()
    first = preflight_imaging_inputs(
        processing_set, config, mode=CONTINUUM_SINGLE_FIELD
    )
    processing_set["other"] = processing_set["ms"].copy(deep=True)
    repeated = preflight_imaging_inputs(
        processing_set, config, mode=CONTINUUM_SINGLE_FIELD
    )
    assert repeated["spectral_design_rank"] == first["spectral_design_rank"]
    assert (
        repeated["spectral_design_rank_tolerance"]
        == first["spectral_design_rank_tolerance"]
    )
    assert repeated["distinct_frequency_count"] == first["distinct_frequency_count"]


def test_insufficient_nominal_frequency_coverage_is_rejected():
    processing_set, config = inputs()
    config["image_params"]["n_taylor_terms"] = 3
    with pytest.raises(ValueError, match="fewer distinct"):
        preflight_imaging_inputs(processing_set, config, mode=CONTINUUM_SINGLE_FIELD)


def test_multiple_selected_fields_are_rejected():
    processing_set, config = inputs()
    processing_set["ms"].attrs["partition_info"]["field_name"] = ["one", "two"]
    with pytest.raises(ValueError, match="multiple fields"):
        preflight_imaging_inputs(processing_set, config, mode=CONTINUUM_SINGLE_FIELD)


@pytest.mark.parametrize("mode", [CUBE_SINGLE_FIELD, CONTINUUM_SINGLE_FIELD])
@pytest.mark.parametrize("layout", ["one_partition", "separate_partitions"])
def test_three_selected_fields_fail_without_loading_samples(mode, layout):
    import dask.array as da
    from dask import delayed

    processing_set, config = inputs()
    if mode == CUBE_SINGLE_FIELD:
        config["image_params"]["n_taylor_terms"] = 1
        config["deconvolver"] = "hogbom"

    def forbidden_read():
        pytest.fail("A multi-field selection loaded bulk samples before rejection")

    source = processing_set["ms"]
    for name in ("VISIBILITY", "WEIGHT", "FLAG", "UVW"):
        variable = source[name]
        source[name] = xr.DataArray(
            da.from_delayed(
                delayed(forbidden_read)(), shape=variable.shape, dtype=variable.dtype
            ),
            dims=variable.dims,
        )
    fields = [
        "NGC4038 - Antennae North_0",
        "NGC4038 - Antennae North_1",
        "NGC4038 - Antennae North_2",
    ]
    if layout == "one_partition":
        source.attrs["partition_info"]["field_name"] = fields
    else:
        processing_set = {}
        for index, field in enumerate(fields):
            child = source.copy(deep=True)
            child.attrs["partition_info"] = {
                **source.attrs["partition_info"],
                "field_name": [field],
            }
            processing_set[f"ms{index}"] = child
    snapshots = {
        name: dataset.copy(deep=True) for name, dataset in processing_set.items()
    }
    with pytest.raises(
        ValueError, match="Single-field imaging selected multiple fields"
    ) as failure:
        preflight_imaging_inputs(processing_set, config, mode=mode)
    for field in fields:
        assert field in str(failure.value)
    for name, dataset in processing_set.items():
        assert dataset.identical(snapshots[name])


@pytest.mark.parametrize("role", ["correlated_data", "weight", "flag", "uvw"])
def test_invalid_sample_storage_types_rejected_from_metadata(role):
    processing_set, config = inputs()
    source = processing_set["ms"]
    name = source.attrs["data_groups"]["base"][role]
    source[name] = source[name].astype(str)
    with pytest.raises(TypeError, match=f"unsupported {role} dtype"):
        check_input_metadata(source, config, mode=CUBE_SINGLE_FIELD)


def test_missing_data_group_is_rejected_before_sample_reads():
    processing_set, config = inputs()
    source = processing_set["ms"]
    source.attrs["data_groups"] = {}
    with pytest.raises(KeyError, match="data group.*missing"):
        check_input_metadata(source, config, mode=CUBE_SINGLE_FIELD)


@pytest.mark.parametrize("dimension", ["frequency", "polarization"])
def test_coordinate_metadata_must_be_one_dimensional(dimension):
    processing_set, config = inputs()
    source = processing_set["ms"]
    values = source[dimension].values[:, None]
    source = source.assign_coords({dimension: ((dimension, "extra"), values)})
    with pytest.raises(ValueError, match="one-dimensional coordinate"):
        check_input_metadata(source, config, mode=CUBE_SINGLE_FIELD)
