"""Pure, mergeable numerical reductions for ``imstatistics``.

This module contains no I/O and no graph construction.  A node task passes an
already loaded, NumPy-backed image selection to :func:`create_statistics_state`.
The resulting compact state can either be finalized immediately or combined
associatively by a distributed reduction.

The internal state stores ``min``, ``max``, ``sum``, ``sumsq``, and ``npts``.
Derived statistics are not stored because partial means, RMS values, and
standard deviations cannot be combined directly when partitions contain
different valid sample counts. NaNs represent invalid or masked pixels and are
excluded.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr

_STATE_NAMES = ("min", "max", "sum", "sumsq", "npts")
_POSITION_NAMES = ("minpos", "maxpos")
_SAMPLE_NAME = "__statistics_samples__"
_SAMPLE_DIM = "__statistics_sample__"
_AXIS_DIM = "statistics_axis"
_DEFAULT_PUBLIC_NAMES = (
    *_STATE_NAMES,
    "mean",
    "rms",
    "sigma",
    *_POSITION_NAMES,
)


def _normalize_dims(data: xr.DataArray, dims: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(dims, str):
        dims = (dims,)
    normalized = tuple(dims)
    unknown = set(normalized) - set(data.dims)
    if unknown:
        raise ValueError(f"Unknown reduction dimensions: {sorted(unknown)}")
    return normalized


def create_statistics_state(
    data: xr.DataArray,
    dims: Sequence[str] | str,
    *,
    statistics: Sequence[str] = (),
    positions: dict[str, Sequence[int]] | None = None,
) -> xr.Dataset:
    """Create a mergeable numerical state for loaded image data.

    Parameters
    ----------
    data : xarray.DataArray
        NumPy-backed, already selected image pixels. NaNs are excluded.
    dims : sequence of str or str
        Named dimensions reduced by the statistics.
    statistics : sequence of str, optional
        Requested public statistics. Exact median statistics retain selected
        samples in the state only when needed.
    positions : dict of str to sequence of int, optional
        Absolute pixel positions for each reduced dimension. By default,
        positions are relative to the supplied array.

    Returns
    -------
    xarray.Dataset
        Compact state containing ``min``, ``max``, ``sum``, ``sumsq`` and
        ``npts``.

    Notes
    -----
    Dimensions not named in ``dims`` are retained, including singleton
    dimensions and coordinates. The state is independent of the requested
    public statistics so every distributed node emits the same associative
    payload.
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
    squared = data * data
    state = xr.Dataset(
        {
            "min": data.min(dim=reduction_dims, skipna=True),
            "max": data.max(dim=reduction_dims, skipna=True),
            "sum": data.sum(dim=reduction_dims, skipna=True),
            "sumsq": squared.sum(dim=reduction_dims, skipna=True),
            "npts": npts,
        }
    )
    retained_dims = tuple(dim for dim in data.dims if dim not in reduction_dims)
    state.update(_extrema_positions(data, reduction_dims, retained_dims, positions))
    if {"median", "medabsdevmed", "mad"} & set(statistics):
        transposed = data.transpose(*retained_dims, *reduction_dims)
        sample_count = int(np.prod([data.sizes[dim] for dim in reduction_dims]))
        sample_shape = tuple(data.sizes[dim] for dim in retained_dims) + (sample_count,)
        sample_coords = {
            dim: data.coords[dim] for dim in retained_dims if dim in data.coords
        }
        state[_SAMPLE_NAME] = xr.DataArray(
            np.asarray(transposed).reshape(sample_shape),
            dims=(*retained_dims, _SAMPLE_DIM),
            coords=sample_coords,
        )
    state.attrs["reduction_dims"] = list(reduction_dims)
    state.attrs["data_attrs"] = dict(data.attrs)
    return state


def _extrema_positions(data, reduction_dims, retained_dims, positions):
    """Return absolute, lexicographically tie-broken extrema positions."""
    transposed = data.transpose(*retained_dims, *reduction_dims)
    retained_shape = tuple(data.sizes[dim] for dim in retained_dims)
    values = np.asarray(transposed).reshape((*retained_shape, -1))
    valid = ~np.isnan(values)
    min_indices = np.argmin(np.where(valid, values, np.inf), axis=-1)
    max_indices = np.argmax(np.where(valid, values, -np.inf), axis=-1)

    position_axes = []
    for dim in reduction_dims:
        axis = np.asarray((positions or {}).get(dim, np.arange(data.sizes[dim])))
        if axis.shape != (data.sizes[dim],):
            raise ValueError(f"Positions for {dim!r} have the wrong length")
        position_axes.append(axis.astype(np.int64, copy=False))
    if position_axes:
        grids = np.meshgrid(*position_axes, indexing="ij")
        position_table = np.stack([grid.ravel() for grid in grids], axis=-1)
    else:
        position_table = np.empty((1, 0), dtype=np.int64)

    any_valid = valid.any(axis=-1)
    min_positions = position_table[min_indices]
    max_positions = position_table[max_indices]
    min_positions[~any_valid] = -1
    max_positions[~any_valid] = -1
    coords = {dim: data.coords[dim] for dim in retained_dims if dim in data.coords}
    coords[_AXIS_DIM] = list(reduction_dims)
    dims = (*retained_dims, _AXIS_DIM)
    return {
        "minpos": xr.DataArray(min_positions, dims=dims, coords=coords),
        "maxpos": xr.DataArray(max_positions, dims=dims, coords=coords),
    }


def merge_statistics_states(
    states: Iterable[xr.Dataset],
    *,
    partition_dim: str | None = None,
    reduction_dims: Sequence[str] = (),
) -> xr.Dataset:
    """Associatively combine partial image-statistics states.

    Parameters
    ----------
    states : iterable of xarray.Dataset
        Partial states from :func:`create_statistics_state` or an earlier
        reduction level.
    partition_dim : str, optional
        Dimension across which the source image was partitioned.
    reduction_dims : sequence of str, optional
        Dimensions reduced within each partial state.

    Returns
    -------
    xarray.Dataset
        Merged state containing ``min``, ``max``, ``sum``, ``sumsq``, and
        ``npts``.

    If ``partition_dim`` was reduced locally, partial values describe the same
    output coordinates and are numerically merged. If it was retained, the
    partial outputs are disjoint coordinate tiles and are concatenated.

    Numerical merging requires exact coordinate alignment. This detects
    mismatched node outputs instead of silently broadcasting them.
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
        result.attrs["data_attrs"] = dict(states[0].attrs.get("data_attrs", {}))
        return result

    aligned = xr.align(
        *(state[list(_STATE_NAMES)] for state in states), join="exact", copy=False
    )
    partial_dim = "__statistics_partial__"
    mins = xr.concat([state["min"] for state in aligned], dim=partial_dim)
    maxs = xr.concat([state["max"] for state in aligned], dim=partial_dim)
    sums = xr.concat([state["sum"] for state in aligned], dim=partial_dim)
    sumsqs = xr.concat([state["sumsq"] for state in aligned], dim=partial_dim)
    counts = xr.concat([state["npts"] for state in aligned], dim=partial_dim)
    result = xr.Dataset(
        {
            "min": mins.min(dim=partial_dim, skipna=True),
            "max": maxs.max(dim=partial_dim, skipna=True),
            "sum": sums.sum(dim=partial_dim, skipna=True),
            "sumsq": sumsqs.sum(dim=partial_dim, skipna=True),
            "npts": counts.sum(dim=partial_dim).astype(np.int64),
        }
    )
    result.update(_merge_extrema_positions(states, result))
    sample_presence = [_SAMPLE_NAME in state for state in states]
    if any(sample_presence):
        if not all(sample_presence):
            raise ValueError("Only some statistics states contain median samples")
        result[_SAMPLE_NAME] = xr.concat(
            [state[_SAMPLE_NAME] for state in states], dim=_SAMPLE_DIM
        )
    result.attrs["reduction_dims"] = list(reduction_dims)
    result.attrs["data_attrs"] = dict(states[0].attrs.get("data_attrs", {}))
    return result


def _merge_extrema_positions(states, merged):
    """Select the lexicographically first position for each merged extreme."""
    output = {}
    for position_name, value_name in (("minpos", "min"), ("maxpos", "max")):
        candidates = np.stack([state[position_name].values for state in states])
        values = np.stack([state[value_name].values for state in states])
        counts = np.stack([state["npts"].values for state in states])
        target = merged[value_name].values
        result = np.full(candidates.shape[1:], -1, dtype=np.int64)
        output_shape = target.shape
        for index in np.ndindex(output_shape):
            eligible = [
                partial
                for partial in range(len(states))
                if counts[(partial, *index)] > 0
                and values[(partial, *index)] == target[index]
            ]
            if eligible:
                winner = min(
                    eligible,
                    key=lambda partial: tuple(candidates[(partial, *index)]),
                )
                result[index] = candidates[(winner, *index)]
        template = states[0][position_name]
        output[position_name] = xr.DataArray(
            result, dims=template.dims, coords=template.coords
        )
    return output


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


def statistics_sumsq(state: xr.Dataset) -> xr.DataArray:
    """Return the sum of squared valid samples."""
    return state["sumsq"].where(state["npts"] > 0)


def statistics_mean(state: xr.Dataset) -> xr.DataArray:
    """Return ``sum / npts`` from a mergeable statistics state."""
    return (state["sum"] / state["npts"]).where(state["npts"] > 0)


def statistics_rms(state: xr.Dataset) -> xr.DataArray:
    """Return the root mean square of valid samples."""
    return np.sqrt(state["sumsq"] / state["npts"]).where(state["npts"] > 0)


def statistics_sigma(state: xr.Dataset) -> xr.DataArray:
    """Return the sample standard deviation of valid samples.

    The centered sum of squares is divided by ``npts - 1``. A single valid
    sample has zero spread. The maximum guards against a tiny negative
    numerator caused by floating-point roundoff.
    """
    count = state["npts"]
    centered_sumsq = state["sumsq"] - state["sum"] * state["sum"] / count
    variance = centered_sumsq.clip(min=0) / (count - 1)
    return xr.where(count > 1, np.sqrt(variance), 0.0).where(count > 0)


def _require_samples(state: xr.Dataset, statistic: str) -> xr.DataArray:
    if _SAMPLE_NAME not in state:
        raise ValueError(
            f"The state does not contain samples required for {statistic!r}; "
            "pass the requested statistics to create_statistics_state"
        )
    return state[_SAMPLE_NAME]


def statistics_median(state: xr.Dataset) -> xr.DataArray:
    """Return the exact median of valid samples."""
    return _require_samples(state, "median").median(dim=_SAMPLE_DIM, skipna=True)


def statistics_medabsdevmed(state: xr.Dataset) -> xr.DataArray:
    """Return the exact median absolute deviation from the median."""
    samples = _require_samples(state, "medabsdevmed")
    median = samples.median(dim=_SAMPLE_DIM, skipna=True)
    return abs(samples - median).median(dim=_SAMPLE_DIM, skipna=True)


def statistics_minpos(state: xr.Dataset) -> xr.DataArray:
    """Return absolute pixel positions of the valid minimum."""
    return state["minpos"]


def statistics_maxpos(state: xr.Dataset) -> xr.DataArray:
    """Return absolute pixel positions of the valid maximum."""
    return state["maxpos"]


STATISTIC_FUNCTIONS = {
    "min": statistics_min,
    "max": statistics_max,
    "sum": statistics_sum,
    "sumsq": statistics_sumsq,
    "mean": statistics_mean,
    "rms": statistics_rms,
    "sigma": statistics_sigma,
    "median": statistics_median,
    "medabsdevmed": statistics_medabsdevmed,
    "mad": statistics_medabsdevmed,
    "minpos": statistics_minpos,
    "maxpos": statistics_maxpos,
    "npts": statistics_npts,
}


def finalize_statistics_state(
    state: xr.Dataset, statistics: Sequence[str] = _DEFAULT_PUBLIC_NAMES
) -> xr.Dataset:
    """Finalize a partial/merged state into requested public statistics.

    Empty outputs (``npts == 0``) use NaN for ``min``, ``max``, ``sum``, and
    ``mean`` while retaining integer zero for ``npts``. Requested names control
    both membership and ordering of returned variables.
    """
    requested = tuple(statistics)
    unknown = set(requested) - set(STATISTIC_FUNCTIONS)
    if unknown:
        raise ValueError(f"Unknown statistics: {sorted(unknown)}")
    result = xr.Dataset(
        {name: STATISTIC_FUNCTIONS[name](state).rename(name) for name in requested}
    )
    result.attrs["reduction_dims"] = list(state.attrs.get("reduction_dims", ()))
    data_attrs = dict(state.attrs.get("data_attrs", {}))
    same_unit = {
        "min",
        "max",
        "sum",
        "mean",
        "rms",
        "sigma",
        "median",
        "medabsdevmed",
        "mad",
    }
    for name in requested:
        if name in same_unit:
            result[name].attrs.update(data_attrs)
        elif name == "sumsq" and "units" in data_attrs:
            result[name].attrs["units"] = f"({data_attrs['units']})^2"
    return result
