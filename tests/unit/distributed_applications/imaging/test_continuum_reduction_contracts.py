"""Graph reductions must reject incompatible chunks and preserve input ownership."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    combine_continuum_chunks,
)


def _leaf(task_id):
    image = xr.Dataset(
        {
            "VISIBILITY": (
                ("taylor_term", "u"),
                np.full((2, 2), task_id + 1.0, dtype=complex),
            )
        },
        coords={"taylor_term": [0, 1], "u": [-1.0, 1.0]},
        attrs={
            "continuum_imaging": {
                "nterms": 2,
                "n_psf_taylor_terms": 3,
                "reference_frequency_hz": 100.0,
            }
        },
    )
    image.VISIBILITY.values.flags.writeable = False
    return {"image": image, "timing_node_tasks": pd.DataFrame({"task_id": [task_id]})}


PARAMS = {"specmode": "mfs", "additive_variables": ["VISIBILITY"]}


@pytest.mark.parametrize("deep_copy", [True, False])
def test_reduction_is_associative_and_does_not_mutate_readonly_leaves(deep_copy):
    leaves = [_leaf(i) for i in range(4)]
    snapshots = [leaf["image"].copy(deep=True) for leaf in leaves]
    params = dict(PARAMS, copy_image_deep=deep_copy)
    flat = combine_continuum_chunks(leaves, params)
    tree = combine_continuum_chunks(
        [
            combine_continuum_chunks(leaves[:2], params),
            combine_continuum_chunks(leaves[2:], params),
        ],
        params,
    )
    xr.testing.assert_identical(tree["image"], flat["image"])
    pd.testing.assert_frame_equal(tree["timing_node_tasks"], flat["timing_node_tasks"])
    np.testing.assert_array_equal(flat["image"].VISIBILITY, 10.0)
    assert flat["image"].attrs["n_continuum_chunks_combined"] == 4
    for leaf, snapshot in zip(leaves, snapshots, strict=True):
        xr.testing.assert_identical(leaf["image"], snapshot)
        assert not np.shares_memory(leaf["image"].VISIBILITY, flat["image"].VISIBILITY)


@pytest.mark.parametrize(
    ("key", "value"),
    [("nterms", 3), ("n_psf_taylor_terms", 5), ("reference_frequency_hz", 101.0)],
)
def test_reduction_rejects_incompatible_taylor_definitions(key, value):
    leaves = [_leaf(0), _leaf(1)]
    leaves[1]["image"].attrs["continuum_imaging"][key] = value
    with pytest.raises(ValueError, match="metadata mismatch|same reference frequency"):
        combine_continuum_chunks(leaves, PARAMS)


@pytest.mark.parametrize(
    "fault", ["coordinates", "dimensions", "shape", "missing_variable", "missing_image"]
)
def test_reduction_rejects_invalid_chunks(fault):
    leaves = [_leaf(0), _leaf(1)]
    image = leaves[1]["image"]
    if fault == "coordinates":
        leaves[1]["image"] = image.assign_coords(u=[0.0, 2.0])
    elif fault == "dimensions":
        leaves[1]["image"] = image.rename({"u": "v"})
    elif fault == "shape":
        leaves[1]["image"] = image.isel(u=slice(0, 1))
    elif fault == "missing_variable":
        leaves[1]["image"] = image.drop_vars("VISIBILITY")
    else:
        del leaves[1]["image"]
    with pytest.raises(KeyError if fault.startswith("missing") else ValueError):
        combine_continuum_chunks(leaves, PARAMS)


def test_reduction_rejects_empty_input():
    with pytest.raises(ValueError, match="no inputs"):
        combine_continuum_chunks([], PARAMS)


def test_tree_reduction_preserves_task_local_observed_grids():
    """Port the harness regression: channel caches must never be summed."""
    leaves = [_leaf(i) for i in range(3)]
    for index, leaf in enumerate(leaves):
        leaf["task_id"] = index
        leaf["observed_visibility_grid_xds"] = xr.Dataset(
            {"GRID": ("frequency", [complex(index + 1)])}
        )
    params = dict(PARAMS, specmode="mvc")
    result = combine_continuum_chunks(
        [combine_continuum_chunks(leaves[:2], params), leaves[2]], params
    )
    cache = result["observed_visibility_grid_mapping"]
    assert set(cache) == {0, 1, 2}
    for index, leaf in enumerate(leaves):
        xr.testing.assert_identical(cache[index], leaf["observed_visibility_grid_xds"])


@pytest.mark.parametrize("partial", [False, True])
def test_reducer_rejects_duplicate_observed_grid_ownership(partial):
    leaves = [_leaf(i) for i in range(2)]
    for leaf in leaves:
        if partial:
            leaf["observed_visibility_grid_mapping"] = {0: xr.Dataset()}
        else:
            leaf["task_id"] = 0
            leaf["observed_visibility_grid_xds"] = xr.Dataset()
    with pytest.raises(ValueError, match="Duplicate MVC visibility-grid cache"):
        combine_continuum_chunks(leaves, dict(PARAMS, specmode="mvc"))
