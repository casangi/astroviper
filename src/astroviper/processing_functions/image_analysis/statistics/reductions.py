"""Mergeable image-statistics processing functions.

This module contains no I/O and no graph construction.  A node task passes an
already loaded, NumPy-backed image selection to :func:`create_statistics_state`.
The resulting compact state can either be finalized immediately or combined
associatively by a distributed reduction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr

_STATE_NAMES = ("min", "max", "sum", "npts")
_PUBLIC_NAMES = (*_STATE_NAMES, "mean")


def _normalize_dims(data: xr.DataArray, dims: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(dims, str):
        dims = (dims,)
    normalized = tuple(dims)
    unknown = set(normalized) - set(data.dims)
    if unknown:
        raise ValueError(f"Unknown reduction dimensions: {sorted(unknown)}")
    return normalized


def create_statistics_state(
    data: xr.DataArray, dims: Sequence[str] | str
) -> xr.Dataset:
    """Create a mergeable ``min/max/sum/count`` state for loaded image data.

    Parameters
    ----------
    data : xarray.DataArray
        NumPy-backed, already selected image pixels. NaNs are excluded.
    dims : sequence of str or str
        Named dimensions reduced by the statistics.

    Returns
    -------
    xarray.Dataset
        Compact state containing ``min``, ``max``, ``sum`` and ``npts``.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("data must be an xarray.DataArray")
    if not isinstance(data.data, np.ndarray):
        raise TypeError(
            "statistics processing functions require loaded NumPy data; "
            "load the selected chunk in the node-task layer"
        )
    reduction_dims = _normalize_dims(data, dims)
    valid = data.notnull()
    npts = valid.sum(dim=reduction_dims).astype(np.int64)
    state = xr.Dataset(
        {
            "min": data.min(dim=reduction_dims, skipna=True),
            "max": data.max(dim=reduction_dims, skipna=True),
            "sum": data.sum(dim=reduction_dims, skipna=True),
            "npts": npts,
        }
    )
    state.attrs["reduction_dims"] = list(reduction_dims)
    return state


def merge_statistics_states(
    states: Iterable[xr.Dataset],
    *,
    partition_dim: str | None = None,
    reduction_dims: Sequence[str] = (),
) -> xr.Dataset:
    """Associatively combine partial image-statistics states.

    If ``partition_dim`` was reduced locally, partial values describe the same
    output coordinates and are numerically merged. If it was retained, the
    partial outputs are disjoint coordinate tiles and are concatenated.
    """
    states = list(states)
    if not states:
        raise ValueError("At least one statistics state is required")
    for state in states:
        missing = set(_STATE_NAMES) - set(state.data_vars)
        if missing:
            raise ValueError(f"Statistics state is missing {sorted(missing)}")

    if partition_dim is not None and partition_dim not in set(reduction_dims):
        result = xr.concat(states, dim=partition_dim)
        if partition_dim in result.coords:
            result = result.sortby(partition_dim)
        result.attrs["reduction_dims"] = list(reduction_dims)
        return result

    aligned = xr.align(*states, join="exact", copy=False)
    partial_dim = "__statistics_partial__"
    mins = xr.concat([state["min"] for state in aligned], dim=partial_dim)
    maxs = xr.concat([state["max"] for state in aligned], dim=partial_dim)
    sums = xr.concat([state["sum"] for state in aligned], dim=partial_dim)
    counts = xr.concat([state["npts"] for state in aligned], dim=partial_dim)
    result = xr.Dataset(
        {
            "min": mins.min(dim=partial_dim, skipna=True),
            "max": maxs.max(dim=partial_dim, skipna=True),
            "sum": sums.sum(dim=partial_dim, skipna=True),
            "npts": counts.sum(dim=partial_dim).astype(np.int64),
        }
    )
    result.attrs["reduction_dims"] = list(reduction_dims)
    return result


def statistics_min(state: xr.Dataset) -> xr.DataArray:
    """Return the valid minimum from a mergeable statistics state."""
    return state["min"].where(state["npts"] > 0)


def statistics_max(state: xr.Dataset) -> xr.DataArray:
    """Return the valid maximum from a mergeable statistics state."""
    return state["max"].where(state["npts"] > 0)


def statistics_sum(state: xr.Dataset) -> xr.DataArray:
    """Return the valid sum from a mergeable statistics state."""
    return state["sum"].where(state["npts"] > 0)


def statistics_npts(state: xr.Dataset) -> xr.DataArray:
    """Return the number of finite/unmasked samples in a statistics state."""
    return state["npts"]


def statistics_mean(state: xr.Dataset) -> xr.DataArray:
    """Return ``sum / npts`` from a mergeable statistics state."""
    return (state["sum"] / state["npts"]).where(state["npts"] > 0)


STATISTIC_FUNCTIONS = {
    "min": statistics_min,
    "max": statistics_max,
    "sum": statistics_sum,
    "mean": statistics_mean,
    "npts": statistics_npts,
}


def finalize_statistics_state(
    state: xr.Dataset, statistics: Sequence[str] = _PUBLIC_NAMES
) -> xr.Dataset:
    """Finalize a partial/merged state into requested public statistics."""
    requested = tuple(statistics)
    unknown = set(requested) - set(STATISTIC_FUNCTIONS)
    if unknown:
        raise ValueError(f"Unknown statistics: {sorted(unknown)}")
    result = xr.Dataset(
        {name: STATISTIC_FUNCTIONS[name](state).rename(name) for name in requested}
    )
    result.attrs["reduction_dims"] = list(state.attrs.get("reduction_dims", ()))
    return result
