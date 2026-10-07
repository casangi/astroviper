"""Malformed map results are rejected before continuum reduction."""

import pandas as pd
import pytest
import xarray as xr

from astroviper.distributed_applications.imaging.image_continuum_single_field import (
    _apply_exact_frequency_selection_to_continuum_mapping,
    combine_continuum_chunks,
)


def _leaf():
    return {"image": xr.Dataset(), "timing_node_tasks": pd.DataFrame({"task_id": [0]})}


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("type", TypeError, "dictionary"),
        ("timing", KeyError, "timing_node_tasks"),
        ("image", KeyError, "image"),
        ("image_type", TypeError, "Dataset"),
        ("weight_task", KeyError, "task_id"),
        ("weight_mapping", TypeError, "mapping"),
        ("weight_child", TypeError, "dictionary"),
        ("empty_weights", ValueError, "empty"),
        ("pb_mapping", TypeError, "pb_cache_mapping"),
        ("grid_mapping", TypeError, "observed_visibility_grid_mapping"),
    ],
)
def test_reducer_rejects_incomplete_or_malformed_results(fault, error, message):
    leaf = _leaf()
    if fault == "type":
        leaf = None
    elif fault == "timing":
        del leaf["timing_node_tasks"]
    elif fault == "image":
        del leaf["image"]
    elif fault == "image_type":
        leaf["image"] = None
    elif fault == "weight_task":
        leaf["weight_datasets"] = {}
    elif fault == "weight_mapping":
        leaf["weight_cache_mapping"] = []
    elif fault == "weight_child":
        leaf["weight_cache_mapping"] = {0: []}
    elif fault == "empty_weights":
        leaf["weight_cache_mapping"] = {0: {}}
    elif fault == "pb_mapping":
        leaf["pb_cache_mapping"] = []
    else:
        leaf["observed_visibility_grid_mapping"] = []
    with pytest.raises(error, match=message):
        combine_continuum_chunks([leaf], {"specmode": "mfs", "additive_variables": []})


@pytest.mark.parametrize("kind", ["weight_cache_mapping", "pb_cache_mapping", "pb_xds"])
def test_reducer_rejects_duplicate_weight_and_beam_cache_ownership(kind):
    leaves = [_leaf(), _leaf()]
    for leaf in leaves:
        leaf["task_id"] = 0
        if kind == "weight_cache_mapping":
            leaf[kind] = {0: {"child": xr.Dataset()}}
        elif kind == "pb_cache_mapping":
            leaf[kind] = {0: xr.Dataset()}
        else:
            leaf[kind] = xr.Dataset()
    with pytest.raises(ValueError, match="Duplicate"):
        combine_continuum_chunks(leaves, {"specmode": "mvc", "additive_variables": []})


@pytest.mark.parametrize(
    "fault,message",
    [
        ("child_frequency", "no frequency"),
        ("child_dimensions", "one-dimensional"),
        ("task_frequency", "no frequency"),
        ("task_dimensions", "one-dimensional"),
        ("duplicate_child", "more than one channel"),
        ("duplicate_task", "more than one channel"),
        ("unmatched_task", "absent"),
    ],
)
def test_partition_selection_rejects_ambiguous_or_incomplete_frequency_metadata(
    fault, message
):
    child = xr.Dataset(coords={"frequency": [100.0, 110.0]})
    mapping = {
        0: {
            "task_coords": {"frequency": {"data": [100.0, 110.0]}},
            "data_selection": {},
        }
    }
    if fault == "child_frequency":
        child = xr.Dataset()
    elif fault == "child_dimensions":
        child = xr.Dataset(coords={"frequency": (("a", "b"), [[100.0, 110.0]])})
    elif fault == "task_frequency":
        mapping[0]["task_coords"] = {}
    elif fault == "task_dimensions":
        mapping[0]["task_coords"]["frequency"]["data"] = [[100.0, 110.0]]
    elif fault == "duplicate_child":
        child = child.assign_coords(frequency=[100.0, 100.0])
    elif fault == "duplicate_task":
        mapping[0]["task_coords"]["frequency"]["data"] = [100.0, 100.0]
    else:
        mapping[0]["task_coords"]["frequency"]["data"] = [90.0, 100.0, 110.0]
    ps = xr.DataTree.from_dict({"child": child})
    with pytest.raises(ValueError, match=message):
        _apply_exact_frequency_selection_to_continuum_mapping(mapping, ps)
