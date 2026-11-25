"""Direct staged PS writes using the proposed XRADIO #661 API."""

import numpy as np
import pytest
import xarray as xr
import zarr

from astroviper.node_tasks.imaging.image_continuum_single_field import (
    _load_continuum_weights_in_place,
    _write_continuum_weights_in_place,
)
from astroviper.utils.imaging_weight_persistence import (
    discard_imaging_weights,
    prepare_imaging_weights,
    publish_imaging_weights,
)


@pytest.fixture(params=[2, 3])
def weight_store(tmp_path, request):
    store = str(tmp_path / "input.zarr")
    zarr.open_group(store, mode="w", zarr_format=request.param)
    ds = xr.Dataset(
        {
            "WEIGHT": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.ones((2, 3, 4, 2)),
            )
        },
        coords={
            "time": [0.0, 1.0],
            "baseline_id": [0, 1, 2],
            "frequency": [1e9, 1.1e9, 1.2e9, 1.3e9],
            "polarization": ["XX", "YY"],
        },
        attrs={
            "data_groups": {
                "base": {
                    "weight": "WEIGHT",
                    "description": "Original weights",
                    "date": "original",
                }
            }
        },
    )
    ds.to_zarr(
        store, group="ms", mode="a", consolidated=False, zarr_format=request.param
    )
    return store, xr.DataTree.from_dict({"ms": ds})


def test_disjoint_task_regions_are_read_before_registration_and_published_without_copy(
    weight_store, monkeypatch
):
    store, tree = weight_store
    handles = prepare_imaging_weights(tree, store, "base")
    disk = zarr.open_group(store, path="ms", mode="r", use_consolidated=False)
    assert "WEIGHT_IMAGING" not in disk
    assert "weight_imaging" not in disk.attrs["data_groups"]["base"]
    for channels in ([2, 0], [3, 1]):
        selected = (
            tree["ms"].ds.isel(frequency=channels, polarization=[1, 0]).copy(deep=True)
        )
        selected["WEIGHT_IMAGING"] = selected.WEIGHT * (selected.frequency / 1e9)
        selected.attrs = {
            "data_groups": {
                "base": {"weight": "WEIGHT", "weight_imaging": "WEIGHT_IMAGING"}
            }
        }
        task = xr.DataTree.from_dict({"ms": selected})
        _write_continuum_weights_in_place(task, "unused", "chunk", handles)
        actual = _load_continuum_weights_in_place(task, "unused", "base", handles)
        np.testing.assert_array_equal(
            actual["ms"].ds.WEIGHT_IMAGING, selected.WEIGHT_IMAGING
        )
    # Publishing must never read a full source weight array or copy the cache.
    import xradio

    monkeypatch.setattr(
        xradio,
        "read_data_variable_region",
        lambda *a, **k: pytest.fail("publication read array data"),
    )
    publish_imaging_weights(handles, "base", {"weighting": "natural"})
    discard_imaging_weights(handles)  # Repeated cleanup must retain valid results.
    disk = zarr.open_group(store, path="ms", mode="r", use_consolidated=False)
    np.testing.assert_array_equal(
        disk["WEIGHT_IMAGING"][:],
        np.broadcast_to(np.arange(4)[None, None, :, None] * 0.1 + 1, (2, 3, 4, 2)),
    )
    assert disk.attrs["data_groups"]["base"]["weight_imaging"] == "WEIGHT_IMAGING"
    assert "Original weights" in disk.attrs["data_groups"]["base"]["description"]
    assert disk["WEIGHT_IMAGING"].attrs["provenance"] == {"weighting": "natural"}
    with pytest.raises(FileExistsError):
        prepare_imaging_weights(tree, store, "base")


def test_failed_run_cleans_own_staging_and_preserves_input(weight_store):
    store, tree = weight_store
    handles = prepare_imaging_weights(tree, store, "base")
    with pytest.raises(
        FileExistsError
    ):  # Second run cannot take the first run's target.
        prepare_imaging_weights(tree, store, "base")
    discard_imaging_weights(handles)
    discard_imaging_weights(handles)
    disk = zarr.open_group(store, path="ms", mode="r", use_consolidated=False)
    assert set(disk.array_keys()) == set(tree["ms"].ds.variables)
    assert "weight_imaging" not in disk.attrs["data_groups"]["base"]
    np.testing.assert_array_equal(disk["WEIGHT"][:], 1)


def test_persistence_leaves_unselected_children_untouched(weight_store):
    store, tree = weight_store
    unrelated = tree["ms"].ds.copy(deep=True)
    unrelated.attrs = {
        "data_groups": {"base": {"weight": "WEIGHT", "weight_imaging": "EXISTING"}}
    }
    tree = xr.DataTree.from_dict({"ms": tree["ms"].ds, "unselected": unrelated})
    mapping = {
        0: {"data_selection": {"ms": {"frequency": slice(None)}, "unselected": None}}
    }
    # No unselected child is opened or changed, even when it registers a previous product.
    handles = prepare_imaging_weights(tree, store, "base", mapping)
    assert set(handles) == {"ms"}
    discard_imaging_weights(handles)


@pytest.mark.parametrize(
    "node_name",
    [
        "residual_update_continuum_single_field",
        "degrid_imaging_weights_continuum_node",
    ],
)
@pytest.mark.parametrize(
    "mode,targets",
    [
        ("in_processing_set", None),
        ("in_processing_set", {}),
        ("in_memory", {"ms": {"target": "staged"}}),
        ("in_place", {"ms": {"target": "staged"}}),
    ],
)
def test_nodes_reject_inconsistent_storage_targets_before_io(node_name, mode, targets):
    import importlib
    import inspect

    module = importlib.import_module(
        "astroviper.node_tasks.imaging.image_continuum_single_field"
    )
    node = getattr(module, node_name)
    arguments = {
        name: None
        for name, parameter in inspect.signature(node).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    with pytest.raises(ValueError, match="weight_memory_mode"):
        node(
            **arguments,
            input_checked=True,
            weight_memory_mode=mode,
            persisted_imaging_weights=targets,
        )


@pytest.mark.parametrize(
    "mode,targets",
    [
        ("in_memory", None),
        ("in_place", None),
        ("in_memory", {}),
        ("in_place", {}),
        ("in_processing_set", {"ms": {"target": "staged"}}),
    ],
)
def test_consistent_storage_targets(mode, targets):
    from astroviper.utils.imaging_weight_persistence import check_weight_storage_targets

    check_weight_storage_targets(mode, targets)
