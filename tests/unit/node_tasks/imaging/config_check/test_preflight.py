"""Streaming preflight uses selected flags and the production weight kernels."""

import numpy as np
import pytest
import xarray as xr

from astroviper.node_tasks.imaging.config_check import preflight_imaging_inputs
from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
)


def inputs():
    dims = ("time", "baseline_id", "frequency", "polarization")
    shape = (2, 1, 2, 2)
    dataset = xr.Dataset(
        {
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
            "data_groups": {"base": {"weight": "WEIGHT", "flag": "FLAG", "uvw": "UVW"}},
            "partition_info": {
                "field_name": ["target"],
                "spectral_window_name": "spw0",
            },
        },
    )
    config = {
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
    xr.testing.assert_identical(processing_set["ms"], snapshot)


def test_flagged_channel_does_not_count_toward_taylor_rank():
    processing_set, config = inputs()
    processing_set["ms"].FLAG.values[:, :, 1, 0] = True
    with pytest.raises(ValueError, match="distinct_frequency_count.*1"):
        preflight_imaging_inputs(processing_set, config, mode=CONTINUUM_SINGLE_FIELD)


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
