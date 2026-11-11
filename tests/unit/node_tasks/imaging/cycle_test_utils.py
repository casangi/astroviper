"""Helpers for tests that check that objects die by reference counting."""

from __future__ import annotations

import collections
import gc


def _is_lazy_tree(obj):
    """True for an ``xarray.DataTree`` node none of whose data variables is in
    memory: a node still backed by the store it was opened from."""
    if type(obj).__module__ != "xarray.core.datatree":
        return False
    if type(obj).__qualname__ != "DataTree":
        return False
    data_variables = getattr(obj, "_data_variables", None)
    if data_variables is None:
        return False
    return all(not variable._in_memory for variable in data_variables.values())


def cyclic_garbage(garbage):
    """Types of the objects in ``garbage`` (what a collection found, kept with
    ``gc.DEBUG_SAVEALL``) that are not part of a lazy tree xarray dropped,
    counted.

    ``xr.open_datatree`` builds the backend's tree of every group of a zarr
    store and, while creating the default indexes, maps it onto a new tree
    (``_datatree_from_backend_datatree``, xarray 2026.9) and drops the
    backend's tree with its parent<->child links intact: its nodes, their lazy
    variables and the zarr arrays, codecs and stores they wrap are cyclic
    garbage (about 0.12 MB per processing set) that no ``release_data_tree``
    of the returned tree can reach. xradio's ``load_processing_set`` and
    ``open_processing_set`` open their stores that way. Only that garbage is
    left out: every ``DataTree`` node in the garbage none of whose data
    variables is in memory, and everything it reaches through the garbage.
    Any other object is counted, a node holding loaded data (a tree a node
    task dropped without releasing it) and everything below it included. Drop
    the exception once xradio opens the groups it loads one by one, or xarray
    releases its backend tree.
    """
    in_garbage = {id(obj) for obj in garbage}
    explained = set()
    for root in garbage:
        if not _is_lazy_tree(root):
            continue
        explained.add(id(root))
        stack = [root]
        while stack:
            for referent in gc.get_referents(stack.pop()):
                if id(referent) not in in_garbage or id(referent) in explained:
                    continue
                if type(referent).__qualname__ == "DataTree" and not _is_lazy_tree(
                    referent
                ):
                    continue
                explained.add(id(referent))
                stack.append(referent)
    return collections.Counter(
        f"{type(obj).__module__}.{type(obj).__qualname__}"
        for obj in garbage
        if id(obj) not in explained
    )
