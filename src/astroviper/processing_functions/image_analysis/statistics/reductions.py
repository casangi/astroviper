"""Pure, mergeable numerical reductions for ``imstatistics``.

This module contains no I/O and no graph construction.  A node task passes an
already loaded, NumPy-backed image selection to :func:`create_statistics_state`.
The resulting compact state can either be finalized immediately or combined
associatively by a distributed reduction.

The internal state stores ``min``, ``max``, ``sum``, ``sumsq``, and ``n_pixels``.
The state also stores a mean and centered squared deviations for stable
variance merging. Shared NumPy routines in ``_array_statistics`` perform the
calculations; this module handles xarray metadata and partition coordinates.
Public derived statistics are finalized after merging because partial means,
RMS values, and standard deviations cannot be combined directly when partitions
contain different valid sample counts. NaNs represent invalid or masked pixels and are
excluded.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr

from astroviper.processing_functions.image_analysis.statistics import _array_statistics

_MEAN_NAME = _array_statistics.MEAN_NAME
_M2_NAME = _array_statistics.M2_NAME
_STATE_NAMES = _array_statistics.STATE_NAMES
_POSITION_NAMES = ("minpos", "maxpos")
_SAMPLE_NAME = "__statistics_samples__"
_SAMPLE_DIM = "__statistics_sample__"
_AXIS_DIM = "statistics_axis"
DEFAULT_STATISTICS = (
    "min",
    "max",
    "peak",
    "sum",
    "sumsq",
    "n_pixels",
    "mean",
    "rms",
    "std",
    "minpos",
    "maxpos",
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
        Calculations and retained median samples use float64 precision.
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
        ``n_pixels``, plus internal mean and centered squared deviations.

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
    retained_dims = tuple(dim for dim in data.dims if dim not in reduction_dims)
    sample_count = int(np.prod([data.sizes[dim] for dim in reduction_dims]))
    sample_shape = tuple(data.sizes[dim] for dim in retained_dims) + (sample_count,)
    samples = np.asarray(
        data.transpose(*retained_dims, *reduction_dims).data, dtype=np.float64
    ).reshape(sample_shape)
    summary = _array_statistics.summarize_samples(samples)
    coords = {
        name: coord
        for name, coord in data.coords.items()
        if set(coord.dims).issubset(retained_dims)
    }
    state = xr.Dataset(
        {name: (retained_dims, summary[name]) for name in _STATE_NAMES}, coords=coords
    )
    state.update(
        _extrema_positions(data, reduction_dims, retained_dims, positions, summary)
    )
    if {"median", "mad_sigma"} & set(statistics):
        state[_SAMPLE_NAME] = xr.DataArray(
            samples, dims=(*retained_dims, _SAMPLE_DIM), coords=coords
        )
    state.attrs["reduction_dims"] = list(reduction_dims)
    state.attrs["data_attrs"] = dict(data.attrs)
    return state


def _extrema_positions(data, reduction_dims, retained_dims, positions, summary):
    """Map backend extrema indices to absolute image positions."""
    reduction_shape = tuple(data.sizes[dim] for dim in reduction_dims)
    min_indices = summary["min_index"]
    max_indices = summary["max_index"]

    def absolute_positions(flat_indices):
        """Map only selected flat extrema indices to absolute source positions."""
        if not reduction_dims:
            return np.empty((*np.shape(flat_indices), 0), dtype=np.int64)
        if not all(reduction_shape):
            return np.full(
                (*np.shape(flat_indices), len(reduction_dims)), -1, dtype=np.int64
            )
        local_indices = np.unravel_index(np.maximum(flat_indices, 0), reduction_shape)
        absolute_axes = []
        for dim, local in zip(reduction_dims, local_indices, strict=True):
            indexer = (positions or {}).get(dim)
            if indexer is None:
                absolute = np.asarray(local, dtype=np.int64)
            elif isinstance(indexer, slice):
                if indexer.start is None or indexer.stop is None:
                    raise ValueError(
                        f"Positions slice for {dim!r} needs absolute start and stop"
                    )
                step = indexer.step or 1
                if len(range(indexer.start, indexer.stop, step)) != data.sizes[dim]:
                    raise ValueError(f"Positions for {dim!r} have the wrong length")
                absolute = indexer.start + np.asarray(local) * step
            else:
                indexer = np.asarray(indexer, dtype=np.int64)
                if indexer.shape != (data.sizes[dim],):
                    raise ValueError(f"Positions for {dim!r} have the wrong length")
                absolute = indexer[local]
            absolute_axes.append(np.asarray(absolute, dtype=np.int64))
        return np.stack(absolute_axes, axis=-1)

    any_valid = summary["n_pixels"] > 0
    min_positions = absolute_positions(min_indices)
    max_positions = absolute_positions(max_indices)
    min_positions = np.where(np.expand_dims(any_valid, -1), min_positions, -1)
    max_positions = np.where(np.expand_dims(any_valid, -1), max_positions, -1)
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
        ``n_pixels``, plus internal mean and centered squared deviations.

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
    template = aligned[0]["n_pixels"]
    combined = _array_statistics.merge_summaries(
        [
            {
                name: partial[name].transpose(*template.dims).values
                for name in _STATE_NAMES
            }
            for partial in aligned
        ]
    )
    result = xr.Dataset(
        {name: (template.dims, combined[name]) for name in _STATE_NAMES},
        coords=template.coords,
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
        counts = np.stack([state["n_pixels"].values for state in states])
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


def _value_statistics(state, names):
    """Evaluate shared NumPy routines and attach the state's output coordinates."""
    if not names:
        return {}
    template = state["min"]
    summary = {
        name: state[name].transpose(*template.dims).values for name in _STATE_NAMES
    }
    samples = None
    if {"median", "mad_sigma"}.intersection(names):
        required = "median" if "median" in names else "mad_sigma"
        samples = (
            _require_samples(state, required)
            .transpose(*template.dims, _SAMPLE_DIM)
            .values
        )
    minimum_first = None
    if "peak" in names:
        minimum_first = np.zeros(template.shape, dtype=bool)
        tied = np.ones(template.shape, dtype=bool)
        min_positions = state["minpos"].transpose(*template.dims, _AXIS_DIM).values
        max_positions = state["maxpos"].transpose(*template.dims, _AXIS_DIM).values
        for axis in range(state.sizes[_AXIS_DIM]):
            minimum_first |= tied & (
                min_positions[..., axis] < max_positions[..., axis]
            )
            tied &= min_positions[..., axis] == max_positions[..., axis]
    values = _array_statistics.finalize_summary(
        summary, names, samples=samples, minimum_first=minimum_first
    )
    return {
        name: xr.DataArray(value, dims=template.dims, coords=template.coords, name=name)
        for name, value in values.items()
    }


def statistics_min(state: xr.Dataset) -> xr.DataArray:
    """Return the valid minimum from a mergeable statistics state."""
    return _value_statistics(state, ("min",))["min"]


def statistics_max(state: xr.Dataset) -> xr.DataArray:
    """Return the valid maximum from a mergeable statistics state."""
    return _value_statistics(state, ("max",))["max"]


def statistics_peak(state: xr.Dataset) -> xr.DataArray:
    """Return the signed value with the largest absolute magnitude.

    Equal-magnitude extrema use the first absolute position in reduction-axis
    order, matching plane statistics for spatial reductions. Empty selections
    return NaN. Existing extrema and positions suffice even after merging.
    """
    return _value_statistics(state, ("peak",))["peak"]


def statistics_sum(state: xr.Dataset) -> xr.DataArray:
    """Return the valid sum from a mergeable statistics state."""
    return _value_statistics(state, ("sum",))["sum"]


def statistics_n_pixels(state: xr.Dataset) -> xr.DataArray:
    """Return the valid pixel count as float64, matching plane statistics."""
    return _value_statistics(state, ("n_pixels",))["n_pixels"]


def statistics_sumsq(state: xr.Dataset) -> xr.DataArray:
    """Return the sum of squared valid samples."""
    return _value_statistics(state, ("sumsq",))["sumsq"]


def statistics_mean(state: xr.Dataset) -> xr.DataArray:
    """Return ``sum / n_pixels`` from a mergeable statistics state."""
    return _value_statistics(state, ("mean",))["mean"]


def statistics_rms(state: xr.Dataset) -> xr.DataArray:
    """Return the root mean square of valid samples."""
    return _value_statistics(state, ("rms",))["rms"]


def statistics_std(state: xr.Dataset) -> xr.DataArray:
    """Return the population standard deviation of valid samples.

    Float64 centered squared deviations are accumulated locally and merged
    using the parallel variance formula. Dividing by ``n_pixels`` gives
    population variance (``ddof=0``) without subtracting large raw moments.
    """
    return _value_statistics(state, ("std",))["std"]


def _require_samples(state: xr.Dataset, statistic: str) -> xr.DataArray:
    if _SAMPLE_NAME not in state:
        raise ValueError(
            f"The state does not contain samples required for {statistic!r}; "
            "pass the requested statistics to create_statistics_state"
        )
    return state[_SAMPLE_NAME]


def statistics_median(state: xr.Dataset) -> xr.DataArray:
    """Return the exact median of valid samples."""
    return _value_statistics(state, ("median",))["median"]


def statistics_mad_sigma(state: xr.Dataset) -> xr.DataArray:
    """Return 1.4826 times the median absolute deviation from the median.

    The Gaussian scaling matches the robust noise estimate in plane statistics.
    """
    return _value_statistics(state, ("mad_sigma",))["mad_sigma"]


def statistics_minpos(state: xr.Dataset) -> xr.DataArray:
    """Return absolute pixel positions of the valid minimum."""
    return state["minpos"]


def statistics_maxpos(state: xr.Dataset) -> xr.DataArray:
    """Return absolute pixel positions of the valid maximum."""
    return state["maxpos"]


STATISTIC_FUNCTIONS = {
    "min": statistics_min,
    "max": statistics_max,
    "peak": statistics_peak,
    "sum": statistics_sum,
    "sumsq": statistics_sumsq,
    "mean": statistics_mean,
    "rms": statistics_rms,
    "std": statistics_std,
    "median": statistics_median,
    "mad_sigma": statistics_mad_sigma,
    "minpos": statistics_minpos,
    "maxpos": statistics_maxpos,
    "n_pixels": statistics_n_pixels,
}


def finalize_statistics_state(
    state: xr.Dataset, statistics: Sequence[str] = DEFAULT_STATISTICS
) -> xr.Dataset:
    """Finalize a partial/merged state into requested public statistics.

    Defaults to :data:`DEFAULT_STATISTICS`, shared by the application and node
    task. Pass explicit names to request additional statistics.

    Empty outputs (``n_pixels == 0``) use NaN for ``min``, ``max``, ``sum``, and
    ``mean`` while returning floating-point zero for ``n_pixels``. Requested names control
    both membership and ordering of returned variables.
    """
    requested = tuple(statistics)
    unknown = set(requested) - set(STATISTIC_FUNCTIONS)
    if unknown:
        raise ValueError(f"Unknown statistics: {sorted(unknown)}")
    values = _value_statistics(
        state, tuple(name for name in requested if name not in _POSITION_NAMES)
    )
    result = xr.Dataset(
        {
            name: state[name] if name in _POSITION_NAMES else values[name]
            for name in requested
        }
    )
    result.attrs["reduction_dims"] = list(state.attrs.get("reduction_dims", ()))
    data_attrs = dict(state.attrs.get("data_attrs", {}))
    same_unit = {
        "min",
        "max",
        "peak",
        "sum",
        "mean",
        "rms",
        "std",
        "median",
        "mad_sigma",
    }
    for name in requested:
        if name in same_unit:
            result[name].attrs.update(data_attrs)
        elif name == "sumsq" and "units" in data_attrs:
            result[name].attrs["units"] = f"({data_attrs['units']})^2"
    return result
