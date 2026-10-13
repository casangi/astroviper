"""Invalid and degenerate distributed continuum reducer inputs."""

import numpy as np
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _resolve_continuum_deconvolver,
    combine_continuum_imaging_weight_chunks,
    combine_continuum_weight_density_chunks,
)


def test_public_api_resolves_single_term_mtmfs_to_hogbom():
    with pytest.warns(UserWarning, match="nterms=1.*falling back.*hogbom"):
        resolved = _resolve_continuum_deconvolver("mtmfs", {"nterms": 1})

    assert resolved == "hogbom"


@pytest.mark.parametrize("nterms", [2, 3, 4])
def test_public_api_preserves_supported_mtmfs_taylor_orders(nterms):
    assert _resolve_continuum_deconvolver("mtmfs", {"nterms": nterms}) == "mtmfs"


@pytest.mark.parametrize(
    "params,error,message",
    [
        ({}, KeyError, "must contain 'nterms'"),
        ({"nterms": 1.0}, TypeError, "must be an integer"),
        ({"nterms": 0}, ValueError, "between 2 and 4"),
        ({"nterms": 5}, ValueError, "between 2 and 4"),
    ],
)
def test_public_api_rejects_invalid_mtmfs_taylor_orders(params, error, message):
    with pytest.raises(error, match=message):
        _resolve_continuum_deconvolver("mtmfs", params)


def test_non_mtmfs_deconvolver_does_not_require_nterms():
    assert _resolve_continuum_deconvolver("hogbom", {}) == "hogbom"


@pytest.mark.parametrize(
    "inputs,error,message",
    [
        ([], ValueError, "no inputs"),
        ([None], TypeError, "dictionary"),
        ([{}], KeyError, "neither"),
        ([{"task_id": 0}], KeyError, "weight_datasets"),
        (
            [
                {"task_id": 0, "weight_datasets": {}},
                {"task_id": 0, "weight_datasets": {}},
            ],
            ValueError,
            "Duplicate",
        ),
        (
            [{"weight_cache_mapping": {0: {}}}, {"weight_cache_mapping": {0: {}}}],
            ValueError,
            "Duplicate",
        ),
    ],
)
def test_weight_cache_reducer_rejects_missing_or_duplicate_task_results(
    inputs, error, message
):
    with pytest.raises(error, match=message):
        combine_continuum_imaging_weight_chunks(inputs, {})


def _density():
    return xr.Dataset(
        {
            "WEIGHT_DENSITY_GRID": (
                ("frequency", "weight_polarization", "u", "v"),
                np.ones((2, 1, 2, 2)),
            ),
            "SUM_WEIGHT": (("frequency", "weight_polarization"), np.ones((2, 1))),
        },
        coords={
            "frequency": [100.0, 110.0],
            "weight_polarization": [0],
            "u": [0, 1],
            "v": [0, 1],
        },
    )


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("empty", ValueError, "no inputs"),
        ("result_type", TypeError, "dictionary"),
        ("missing_density", KeyError, "weight_density"),
        ("dataset_type", TypeError, "Dataset"),
        ("missing_variable", KeyError, "missing"),
        ("missing_frequency", KeyError, "frequency coordinate"),
        ("empty_frequency", ValueError, "no frequency"),
        ("nan_frequency", ValueError, "non-finite"),
        ("duplicate_frequency", ValueError, "duplicate"),
        ("density_dimensions", ValueError, "dimensions"),
        ("sum_dimensions", ValueError, "dimensions"),
    ],
)
def test_density_reducer_rejects_invalid_channels_and_arrays(fault, error, message):
    density = _density()
    inputs = [{"weight_density": density}]
    if fault == "empty":
        inputs = []
    elif fault == "result_type":
        inputs = [None]
    elif fault == "missing_density":
        inputs = [{}]
    elif fault == "dataset_type":
        inputs = [{"weight_density": None}]
    else:
        if fault == "missing_variable":
            density = density.drop_vars("SUM_WEIGHT")
        elif fault == "missing_frequency":
            density = density.drop_vars("frequency")
        elif fault == "empty_frequency":
            density = density.isel(frequency=slice(0, 0))
        elif fault == "nan_frequency":
            density = density.assign_coords(frequency=[100.0, np.nan])
        elif fault == "duplicate_frequency":
            density = density.assign_coords(frequency=[100.0, 100.0])
        elif fault == "density_dimensions":
            density["WEIGHT_DENSITY_GRID"] = density.WEIGHT_DENSITY_GRID.transpose(
                "weight_polarization", "frequency", "u", "v"
            )
        else:
            density["SUM_WEIGHT"] = density.SUM_WEIGHT.transpose(
                "weight_polarization", "frequency"
            )
        inputs = [{"weight_density": density}]
    with pytest.raises(error, match=message):
        combine_continuum_weight_density_chunks(inputs, {})
