"""Lifecycle helpers for xarray DataTree objects."""


def release_data_tree(xdt) -> None:
    """Break a DataTree's parent<->child reference cycles so the tree -- and
    every array attached to it -- is freed immediately by refcounting.

    xarray DataTree nodes hold strong references in BOTH directions (parent
    keeps ``children``, child keeps its parent), so any dropped tree is
    cyclic garbage: it survives until a full garbage-collection pass instead
    of dying when the last reference goes away. At processing-set scale that
    is catastrophic: the 2026-08-12 Frontera diagnosis measured ~2.9 GB of
    cyclic garbage per mapping task (the loaded chunk's visibilities/weights
    pinned by the index machinery of the dead tree), ratcheting worker RSS
    by one task's working set per task and OOMing 14-worker nodes whenever
    per-task garbage collection was disabled.

    Call this on a task-owned tree right before dropping the last reference
    to it. Severing the child links (bottom-up) is sufficient to break every
    parent<->child cycle; empirically it reduces the tree's unreachable-
    object count to zero, restoring deterministic refcount lifetimes.

    No-op for ``None`` and for non-DataTree inputs (plain Datasets and the
    dict-of-datasets used by the data-loading layer -- the latter may be
    SHARED across tasks and must not be dismantled).
    """
    import xarray as xr

    if not isinstance(xdt, xr.DataTree):
        return
    for node in list(xdt.subtree)[::-1]:
        node.children = {}
