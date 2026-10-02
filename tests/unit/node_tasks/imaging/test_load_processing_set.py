"""The imaging node task's loader (``_load_processing_set``): the same
processing set as xradio's ``load_processing_set(..., load_sub_datasets=False)``,
but every tree is held in a local while its ``xr_ms`` accessor runs, so a
garbage collection inside the accessor call cannot free it (xradio 1.2.3
raised ``ReferenceError`` there), and nothing it creates survives it as
cyclic garbage."""

from __future__ import annotations

import gc

import numpy as np
import pytest
import xarray as xr

from astroviper.node_tasks.imaging.image_cube_single_field import (
    _load_processing_set,
)
from astroviper.utils.data_tree import release_data_tree


def _write_processing_set(ps_store):
    """A processing set of one MSv4 with two sub-dataset children (as a
    converted MS has), which the loader leaves out; a tree of them would be
    kept alive only by its parent<->child cycle."""
    shape = (2, 3, 4, 2)
    dims = ("time", "baseline_id", "frequency", "polarization")
    ms_xds = xr.Dataset(
        {
            "VISIBILITY": (dims, np.arange(np.prod(shape)).reshape(shape) + 1j),
            "WEIGHT": (dims, np.ones(shape)),
            "FLAG": (dims, np.zeros(shape, dtype=bool)),
            "VISIBILITY_CORRECTED": (dims, np.zeros(shape, dtype=complex)),
        },
        coords={
            "time": [0.0, 1.0],
            "baseline_id": [0, 1, 2],
            "frequency": [1.0e9, 1.1e9, 1.2e9, 1.3e9],
            "polarization": ["XX", "YY"],
        },
        attrs={
            "type": "visibility",
            "data_groups": {
                "base": {
                    "correlated_data": "VISIBILITY",
                    "weight": "WEIGHT",
                    "flag": "FLAG",
                    "field_and_source": "field_and_source_base_xds",
                    "description": "",
                    "date": "",
                },
                "corrected": {
                    "correlated_data": "VISIBILITY_CORRECTED",
                    "weight": "WEIGHT",
                    "flag": "FLAG",
                    "field_and_source": "field_and_source_base_xds",
                    "description": "",
                    "date": "",
                },
            },
        },
    )
    ms_xdt = xr.DataTree(
        dataset=ms_xds,
        children={
            "antenna_xds": xr.DataTree(
                xr.Dataset({"ANTENNA_POSITION": ("antenna_name", [0.0, 1.0, 2.0])})
            ),
            "field_and_source_base_xds": xr.DataTree(
                xr.Dataset({"FIELD_PHASE_CENTER": ("sky_dir_label", [0.0, 1.0])})
            ),
        },
    )
    ps_xdt = xr.DataTree(children={"ms_0": ms_xdt})
    ps_xdt.attrs["type"] = "processing_set"
    ps_xdt.to_zarr(ps_store, consolidated=False)
    return ps_store


SELECTIONS = [
    {"ms_0": {"frequency": slice(1, 3)}},
    {"ms_0": {"frequency": slice(0, 4), "polarization": [1]}},
    {"ms_0": {}},
    {},
]


@pytest.mark.parametrize("sel_parms", SELECTIONS)
@pytest.mark.parametrize("data_group_name", ["base", "corrected"])
def test_same_processing_set_as_xradio(tmp_path, sel_parms, data_group_name):
    from xradio.measurement_set.load_processing_set import load_processing_set

    store = _write_processing_set(str(tmp_path / "small.ps.zarr"))
    expected = load_processing_set(
        store,
        sel_parms=sel_parms,
        data_group_name=data_group_name,
        load_sub_datasets=False,
    )
    loaded = _load_processing_set(store, sel_parms, data_group_name)
    xr.testing.assert_identical(loaded, expected)
    assert all(not ms_xdt.children for ms_xdt in loaded.children.values())


@pytest.mark.parametrize("sel_parms", SELECTIONS[:3])
def test_survives_garbage_collection_inside_the_accessor_call(
    tmp_path, monkeypatch, sel_parms
):
    """A collection at the entry of ``MeasurementSetXdt.sel`` (another thread's
    allocations can start one there at any time) must not free the tree."""
    from xradio.measurement_set.measurement_set_xdt import MeasurementSetXdt

    store = _write_processing_set(str(tmp_path / "small.ps.zarr"))
    original_sel = MeasurementSetXdt.sel
    calls = []

    def sel_after_gc(self, *args, **kwargs):
        calls.append(gc.collect())
        return original_sel(self, *args, **kwargs)

    monkeypatch.setattr(MeasurementSetXdt, "sel", sel_after_gc)
    loaded = _load_processing_set(store, sel_parms, "base")
    assert len(calls) == 1
    assert loaded["ms_0"].attrs["type"] == "visibility"


@pytest.mark.parametrize("sel_parms", SELECTIONS[:3])
def test_leaves_no_cyclic_garbage(tmp_path, sel_parms):
    """Everything the loader creates dies by reference counting once the
    caller releases the returned processing set: with the garbage collector
    off, a collection afterwards finds no xarray or zarr object.
    ``xr.open_datatree`` would leave one behind: it drops the backend's tree
    of all groups with its parent<->child links intact, out of reach of
    ``release_data_tree``."""
    store = _write_processing_set(str(tmp_path / "small.ps.zarr"))
    # first use (imports, caches) outside the measured window
    release_data_tree(_load_processing_set(store, sel_parms, "base"))
    gc_was_enabled = gc.isenabled()
    gc.disable()  # reference counting alone
    # everything alive now, garbage of earlier tests included, is set aside
    # (not collected): the collection below sees only what the loader leaves
    gc.freeze()
    try:
        loaded = _load_processing_set(store, sel_parms, "base")
        assert loaded["ms_0"].attrs["type"] == "visibility"
        release_data_tree(loaded)
        del loaded
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        left = sorted(
            f"{type(obj).__module__}.{type(obj).__qualname__}"
            for obj in gc.garbage
            if type(obj).__module__.split(".")[0] in ("xarray", "zarr")
        )
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.unfreeze()
        if gc_was_enabled:
            gc.enable()
    assert left == []


def test_s3_branch_opens_every_measurement_set_through_an_s3_map(tmp_path, monkeypatch):
    """The S3 branch, without an S3 server: xradio's file system helper is
    replaced by one that returns an S3 file system, and ``xr.open_dataset``
    by one that serves the S3 map from the local copy of the store. Every
    measurement set is opened through an S3 map (``s3fs.S3Map``, an fsspec
    ``FSMap`` of the S3 file system) of its own group, and the result is the
    same as from the local store. The branch relies on a private xradio
    helper (``_get_file_system_and_items``); this test fails with an
    ImportError if xradio renames it."""
    import fsspec.mapping
    import s3fs
    import xradio._utils.zarr.common as zarr_common

    store = _write_processing_set(str(tmp_path / "small.ps.zarr"))
    sel_parms = {"ms_0": {"frequency": slice(1, 3)}}
    expected = _load_processing_set(store, sel_parms, "base")
    file_system = s3fs.S3FileSystem(anon=True)
    monkeypatch.setattr(
        zarr_common,
        "_get_file_system_and_items",
        lambda path: (file_system, ["ms_0"]),
    )
    open_dataset = xr.open_dataset
    roots = []

    def open_s3_map_from_local_copy(s3_map, **kwargs):
        assert isinstance(s3_map, fsspec.mapping.FSMap)
        assert s3_map.fs is file_system
        roots.append(s3_map.root)
        local = s3_map.root.replace("bucket/small.ps.zarr", store, 1)
        return open_dataset(local, **kwargs)

    monkeypatch.setattr(xr, "open_dataset", open_s3_map_from_local_copy)
    loaded = _load_processing_set("s3://bucket/small.ps.zarr", sel_parms, "base")
    monkeypatch.undo()
    assert roots == ["bucket/small.ps.zarr/ms_0"]
    xr.testing.assert_identical(loaded, expected)
