"""The cyclic garbage check of the memory tests leaves out the lazy tree
``xr.open_datatree`` drops, and nothing else."""

from __future__ import annotations

import gc

import numpy as np
import pytest
import xarray as xr

from tests.unit.node_tasks.imaging.cycle_test_utils import cyclic_garbage


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """A zarr store of a root group and a sub-group, as a measurement set."""
    path = str(tmp_path_factory.mktemp("store") / "two_groups.zarr")
    root = xr.Dataset(
        {"DATA": (("x", "y"), np.ones((4, 3)))}, coords={"x": np.arange(4)}
    )
    root.to_zarr(path, mode="w", consolidated=False)
    child = xr.Dataset({"POSITION": (("antenna",), np.zeros(2))})
    child.to_zarr(path, group="antenna_xds", mode="a", consolidated=False)
    return path


def _garbage_of(make):
    """What a collection finds after ``make()``, with the collector off;
    a first call (imports, caches) is made outside."""
    make()
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    gc.freeze()
    try:
        make()
        gc.set_debug(gc.DEBUG_SAVEALL)
        gc.collect()
        garbage = gc.garbage[:]
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
        gc.unfreeze()
        if gc_was_enabled:
            gc.enable()
    return garbage


def test_the_dropped_lazy_tree_is_left_out(store):
    def make():
        # dropped at once: xarray's backend tree is left behind
        xr.open_datatree(store, engine="zarr", consolidated=False)

    garbage = _garbage_of(make)
    assert any(type(obj).__qualname__ == "DataTree" for obj in garbage)
    assert not cyclic_garbage(garbage)


def test_a_dropped_loaded_tree_is_counted(store):
    def make():
        # loaded in place, then dropped without release_data_tree
        xr.open_datatree(store, engine="zarr", consolidated=False).load()

    left = cyclic_garbage(_garbage_of(make))
    assert left["xarray.core.datatree.DataTree"] >= 1, left


def test_a_plain_cycle_is_counted():
    def make():
        holder = {"data": np.ones(10)}
        holder["self"] = holder

    left = cyclic_garbage(_garbage_of(make))
    assert left["builtins.dict"] == 1, left
