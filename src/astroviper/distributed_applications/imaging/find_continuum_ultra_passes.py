"""Passes of find_continuum_ultra over the image store, and their layout.

A pass maps one of the frequency tasks of
:mod:`astroviper.node_tasks.imaging.find_continuum_ultra` over the frequency
chunks of the spectral window (:func:`plan_find_continuum_ultra_layout`),
leaving out the chunks without a channel it needs, and reduces their
per-channel rows in GraphVIPER's binary tree. A pass that makes moments then
maps the merge tasks over the tiles of the maps, which merge the moment
partials of the frequency tasks in the same tree order, and the driver reads
the merged maps. Each pass plans its planes per batch within the memory
budget of a task (:func:`plan_planes_per_batch`), keeps its stores in its own
directory, which it removes when it ends or fails, and returns its results
with the record of its graphs: one row of ``passes``, the timing rows of its
tasks and the records of its reduce nodes.
"""

import math
import os
import shutil
import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
import toolviper.utils.logger as logger
import xarray as xr

from astroviper.distributed_applications.imaging.find_continuum_first_round import (
    find_continuum_merge_tile_counts,
)
from astroviper.node_tasks.imaging.find_continuum_ultra import (
    find_continuum_first_round_task,
    find_continuum_selection_moments_task,
    find_continuum_spectrum_task,
    find_continuum_ultra_merge_task_bytes,
    find_continuum_ultra_task_bytes,
    merge_find_continuum_moment_partials,
    read_moment_maps,
    reduce_find_continuum_channel_rows,
    with_resource_usage,
    write_region_masks,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    channels_of_ranges,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    FindContinuumSettings,
    FirstRoundRows,
    SpectrumRegions,
)
from astroviper.utils.param_docs import shares_param_docs

#: Frequency tasks per thread of the cluster in the default layout.
FREQUENCY_TASKS_PER_THREAD = 4

_ROLES = ("sky", "primary_beam")
# Planes a batch spans at most, unless a read unit within a task holds more.
_PLANES_PER_BATCH = 4
# The driver's peak per pixel: the maps and masks of a later round, 57 B
# (regions and joint masks 9, the moment difference 8, the selection maximum in
# float64 8, round original's moment difference 8, the amended difference map
# 8, a statistics gather and its ddof-1 copy 16; round 0 holds 50 B), and a
# float64 map of NumPy temporaries.
_DRIVER_BYTES_PER_PIXEL = 57 + 8
# The columns of a pass row for the driver's work inside the pass.
_DRIVER_COLUMNS = (
    "T_write_regions",
    "T_first_round_rows",
    "T_read_maps",
    "driver_bytes_read",
    "driver_bytes_written",
)


class FindContinuumUltraLayout(NamedTuple):
    """How the passes over a spectral window read the image store and split
    their work."""

    n_frequency_tasks: int  # frequency chunks of the window, slots of a pass
    channels_per_task: int
    frequency_mapping: dict  # mapping of the chunks, in channels of the cube
    merge_tile_shape: tuple  # (l, m)
    tile_mapping: dict  # mapping of the merge tiles
    budget_bytes: int  # memory budget of a frequency task
    memory_per_thread_bytes: int
    read_unit_bytes: int  # decoded bytes of the largest read unit
    shard_length: int  # channels per frequency shard (chunk when not sharded)
    n_threads: int
    plane_bytes: int  # bytes of a plane of the residual
    input_image_store: str
    variables: dict  # {"sky": ..., "primary_beam": ...}
    polarization_index: int  # of Stokes I
    channel_range: list  # [start, stop) of the window in channels of the cube
    tile_shape: tuple  # (l, m) of the residual's read units
    value_dtypes: dict  # {"sky": ..., "primary_beam": ...} dtype names
    read_unit_planes: int  # channels of a read unit within a task


class PassRecord(NamedTuple):
    """The record of one pass over the image store."""

    row: dict  # one row of "passes"
    timing_node_tasks: pd.DataFrame  # its frequency and merge tasks
    timing_reduce_nodes: list  # its reduce nodes


class FirstRoundReductions(NamedTuple):
    """The first round of a spectral window."""

    rows: FirstRoundRows
    integral: np.ndarray | None  # (l, m) float64
    maximum: np.ndarray | None  # (l, m) cube dtype
    first_round_store: str


class SpectrumReductions(NamedTuple):
    """The per-channel spectra of a spectrum pass, float64 (n_channels,)."""

    mean_spectrum: np.ndarray
    scaled_mad_spectrum: np.ndarray | None


class SelectionReductions(NamedTuple):
    """The maps of a channel selection and the noise of its channels."""

    integral: np.ndarray  # (l, m) float64
    maximum: np.ndarray  # (l, m) cube dtype
    selected_channels: np.ndarray  # sorted unique channels of the ranges
    noise_medians: np.ndarray | None
    noise_mads: np.ndarray | None


def plan_find_continuum_ultra_layout(
    window_xds: xr.Dataset,
    data_group: dict,
    channel_range: list,
    n_mapping_parallelism: dict | None,
    thread_info: dict,
    memory_budget_gb: float | None,
) -> FindContinuumUltraLayout:
    """The frequency chunks, merge tiles and task budget of the passes over a
    spectral window.

    The window is split into ``n_mapping_parallelism["frequency"]`` chunks
    (GraphVIPER's ``make_parallel_coord``, whose chunks have one length but
    the last), or into as many as about ``FREQUENCY_TASKS_PER_THREAD`` tasks
    per thread of whole shards need; the maps into merge tiles of about two
    read units per axis
    (:func:`~astroviper.distributed_applications.imaging.find_continuum_first_round.find_continuum_merge_tile_counts`).
    The memory budget of a task is 0.8 x ``memory_per_thread`` less the task
    baseline of ``min(1.5, 0.2 x memory_per_thread)`` GiB, clamped by
    ``memory_budget_gb``. Reads the metadata of the residual and of its
    primary beam only.

    Parameters
    ----------
    window_xds : xarray.Dataset
        The spectral window, opened lazily from the image store
        (``xarray.open_zarr``, whose ``encoding["source"]`` is the store).
    data_group : dict
        Data group of the residual, with the roles ``sky`` and
        ``primary_beam``.
    channel_range : list of int
        ``[start, stop)`` of the window in channels of the cube.
    n_mapping_parallelism : dict or None
        ``{"frequency": n}``, the frequency chunks of the window; ``None``:
        about ``FREQUENCY_TASKS_PER_THREAD`` per thread.
    thread_info : dict
        ``n_threads`` and ``memory_per_thread`` (GiB) of the cluster.
    memory_budget_gb : float or None
        Largest memory budget of a frequency task (GiB).

    Returns
    -------
    FindContinuumUltraLayout

    Raises
    ------
    ValueError
        If ``window_xds`` was not opened from an image store, or the smallest
        task of any pass (a spectrum task at one plane per batch) needs more
        than the memory of a thread.
    """
    import zarr
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )

    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_ultra_task_memory,
    )

    input_image_store = window_xds.encoding.get("source")
    if input_image_store is None:
        raise ValueError(
            "window_xds has no encoding['source']: open the image store with "
            "xarray.open_zarr"
        )
    variables = {role: data_group[role] for role in _ROLES}
    arrays = {
        role: zarr.open_array(os.path.join(input_image_store, variable), mode="r")
        for role, variable in variables.items()
    }
    sky = arrays["sky"]
    start, stop = (int(channel) for channel in channel_range)
    n_l, n_m = sky.shape[-2:]
    shard_length = (sky.shards or sky.chunks)[1]
    n_threads = int(thread_info["n_threads"])
    memory_per_thread = float(thread_info["memory_per_thread"])
    if n_mapping_parallelism is None:
        channels_per_task = shard_length * max(
            1,
            round(
                (stop - start) / (FREQUENCY_TASKS_PER_THREAD * n_threads * shard_length)
            ),
        )
        n_chunks = -(-(stop - start) // channels_per_task)
    else:
        n_chunks = n_mapping_parallelism["frequency"]
    # Positions are channels of the cube, so that data selections are too.
    index_xds = xr.Dataset(
        coords={
            dim: np.arange(n, dtype=np.float64)
            for dim, n in (("frequency", stop), ("l", n_l), ("m", n_m))
        }
    )
    chunks = make_parallel_coord(
        coord=index_xds.frequency[start:stop], n_chunks=n_chunks
    )
    tiles = {
        dim: make_parallel_coord(coord=index_xds[dim], n_chunks=n_tiles)
        for dim, n_tiles in zip(
            ("l", "m"),
            find_continuum_merge_tile_counts((n_l, n_m), sky.chunks[-2:]),
            strict=True,
        )
    }
    budget = 0.8 * memory_per_thread - min(1.5, 0.2 * memory_per_thread)
    if memory_budget_gb is not None:
        budget = min(memory_budget_gb, budget)
    layout = FindContinuumUltraLayout(
        n_frequency_tasks=len(chunks["data_chunks"]),
        channels_per_task=len(chunks["data_chunks"][0]),
        frequency_mapping=interpolate_data_coords_onto_parallel_coords(
            {"frequency": chunks}, {"img": index_xds}
        ),
        # GraphVIPER's chunks have one length but the last, so the tiles are
        # whole chunks of the maps.
        merge_tile_shape=tuple(len(tiles[dim]["data_chunks"][0]) for dim in ("l", "m")),
        tile_mapping=interpolate_data_coords_onto_parallel_coords(
            tiles, {"img": index_xds}
        ),
        budget_bytes=int(budget * 2**30),
        memory_per_thread_bytes=int(memory_per_thread * 2**30),
        read_unit_bytes=max(
            math.prod(array.chunks) * array.dtype.itemsize for array in arrays.values()
        ),
        shard_length=shard_length,
        n_threads=n_threads,
        plane_bytes=n_l * n_m * sky.dtype.itemsize,
        input_image_store=input_image_store,
        variables=variables,
        polarization_index=window_xds.polarization.values.tolist().index("I"),
        channel_range=[start, stop],
        tile_shape=tuple(sky.chunks[-2:]),
        value_dtypes={role: array.dtype.name for role, array in arrays.items()},
        read_unit_planes=min(
            sky.chunks[1], max(map(len, chunks["data_chunks"].values()))
        ),
    )
    # Every run but a constant cube's makes a spectrum pass.
    check_find_continuum_ultra_task_memory(
        find_continuum_ultra_task_bytes(
            "spectrum",
            n_l,
            n_m,
            1,
            sky.dtype.itemsize,
            arrays["primary_beam"].dtype.itemsize,
            False,
            1,
            False,
            0.0,
            layout.read_unit_bytes,
        ),
        layout.memory_per_thread_bytes,
    )
    return layout


def plan_planes_per_batch(
    task_bytes: Callable[[int], int], layout: FindContinuumUltraLayout
) -> tuple[int, int]:
    """Planes per batch of the frequency tasks of a pass, and their memory
    bound.

    At most 4 planes and at most a shard, or the planes of a read unit within
    a task when it holds more (each read unit is then decoded once per task),
    halved while a task needs more than its budget. A task over its budget at
    one plane per batch runs with a warning, up to the memory of a thread.

    Parameters
    ----------
    task_bytes : callable
        Memory bound of a frequency task of the pass at a number of planes
        per batch
        (:func:`~astroviper.node_tasks.imaging.find_continuum_ultra.find_continuum_ultra_task_bytes`).
    layout : FindContinuumUltraLayout
        Layout of the window.

    Returns
    -------
    planes_per_batch : int
    task_bytes : int
        Memory bound of a task at ``planes_per_batch``.

    Raises
    ------
    ValueError
        If a task needs more than the memory of a thread at one plane per
        batch.
    """
    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_ultra_task_memory,
    )

    smallest = task_bytes(1)
    check_find_continuum_ultra_task_memory(smallest, layout.memory_per_thread_bytes)
    if smallest > layout.budget_bytes:
        logger.warning(
            f"A find_continuum_ultra task needs {smallest / 2**30:.2f} GiB at one "
            f"plane per batch, more than its budget of "
            f"{layout.budget_bytes / 2**30:.2f} GiB: it runs at one plane per batch."
        )
    planes_per_batch = max(
        min(_PLANES_PER_BATCH, layout.shard_length), layout.read_unit_planes
    )
    while planes_per_batch > 1 and task_bytes(planes_per_batch) > layout.budget_bytes:
        planes_per_batch //= 2
    return planes_per_batch, task_bytes(planes_per_batch)


@shares_param_docs
def run_first_round_pass(
    window_xds: xr.Dataset,
    layout: FindContinuumUltraLayout,
    settings: FindContinuumSettings,
    first_round_store: str,
    compute_masks: bool,
    compute_moments: bool,
    pass_directory: str,
    round_index: int | None,
    monitor_resources_seconds: float | None,
    pass_index: int,
) -> tuple[FirstRoundReductions, list[PassRecord]]:
    """findContinuum's first round of a spectral window in one pass.

    With ``compute_masks`` the frequency tasks make the primary-beam masks of
    every channel into the first-round store and return the per-channel rows,
    which the driver writes to the store; otherwise the rows are read from
    the store, whose limit masks the tasks read. With ``compute_moments`` the
    integrated and maximum moments of the valid, nonzero voxels are merged and
    read (findContinuum's first-round moments, fC 7867-7876).

    Parameters
    ----------
    window_xds : xarray.Dataset
        The spectral window, opened lazily.
    layout : FindContinuumUltraLayout
        Layout of the window (:func:`plan_find_continuum_ultra_layout`).
    settings : FindContinuumSettings
        Settings of the window; its ``moment_channel_weights`` are read.
    first_round_store : str
        Path of the window's first-round store, whose attributes give the
        primary-beam limit: with ``compute_masks`` an empty one of the window
        and of the tile shapes of ``layout``
        (:func:`~astroviper.distributed_applications.imaging.find_continuum_first_round.create_find_continuum_first_round_store`),
        otherwise a complete one.
    compute_masks : bool
        Make the primary-beam masks and the per-channel rows.
    compute_moments : bool
        Make the moments.
    pass_directory : str
        New directory of the pass's stores, removed when the pass ends or
        fails.
    round_index : int or None
        Round of findContinuum the pass belongs to, recorded.
    monitor_resources_seconds : float or None
        Sampling interval (s) of GraphVIPER's resource monitor in the tasks;
        ``None``: no monitor.
    pass_index : int
        Index of the pass in the run, recorded.

    Returns
    -------
    reductions : FirstRoundReductions
        The window's rows and, with ``compute_moments``, its maps.
    records : list of PassRecord
        The record of the pass.

    Raises
    ------
    ValueError
        If neither ``compute_masks`` nor ``compute_moments`` is set, or a task
        needs more than the memory of a thread.
    RuntimeError
        If the rows do not hold every channel of the window once.
    """
    import zarr

    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        read_first_round_rows,
        write_first_round_rows,
    )

    if not (compute_masks or compute_moments):
        raise ValueError("A first-round pass makes the masks, the moments or both")
    group = zarr.open_group(first_round_store, mode="r")
    pass_kind = "first_round"
    planes_per_batch, task_bytes = plan_planes_per_batch(
        lambda planes: find_continuum_ultra_task_bytes(
            pass_kind,
            window_xds.sizes["l"],
            window_xds.sizes["m"],
            planes,
            *_itemsizes(layout),
            compute_masks,
            0,
            compute_moments,
            0.0,
            layout.read_unit_bytes,
        ),
        layout,
    )
    input_params = {
        **_task_inputs(layout),
        "primary_beam_limit": group.attrs["find_continuum_first_round"][
            "primary_beam_limit"
        ],
        "moment_channel_weights": settings.moment_channel_weights,
        "first_round_store": first_round_store,
        "compute_masks": compute_masks,
        "compute_moments": compute_moments,
        "partials_store": None,
        "planes_per_batch": planes_per_batch,
    }
    os.makedirs(pass_directory)
    try:
        output, maps, frames, columns = _run_graphs(
            window_xds,
            layout,
            find_continuum_first_round_task,
            input_params,
            _task_priorities(layout, None),
            pass_directory,
            compute_moments,
            monitor_resources_seconds,
        )
    finally:
        shutil.rmtree(pass_directory, ignore_errors=True)
    start = time.time()
    if compute_masks:
        written = write_first_round_rows(
            first_round_store,
            dict(output["rows"], write_failed=False),
            group["SKY_VALID_COUNT"].shape[0],
        )
        if not written["complete"]:
            raise RuntimeError(
                f"The first round of channels {layout.channel_range} is incomplete "
                f"in {first_round_store}"
            )
        rows = FirstRoundRows(
            **{field: output["rows"][field] for field in FirstRoundRows._fields}
        )
    else:
        rows, counters = read_first_round_rows(first_round_store, layout.channel_range)
        columns["driver_bytes_read"] += counters["bytes"] + counters["index_bytes"]
    columns["T_first_round_rows"] = time.time() - start
    work = ["primary-beam masks", "per-channel rows"] if compute_masks else []
    record = _pass_record(
        layout,
        {
            "pass_index": pass_index,
            "pass_kind": pass_kind,
            "round_index": round_index,
            "round_kind": None,
            "purpose": "first round: "
            + ", ".join(work + ["moments"] * compute_moments),
            "data_variables_read": _variables_read(layout, compute_masks),
            "computes_first_round_masks": compute_masks,
            "n_moment_sets": int(compute_moments),
        },
        output,
        frames,
        planes_per_batch,
        task_bytes,
        columns,
    )
    integral, maximum = maps or (None, None)
    return FirstRoundReductions(rows, integral, maximum, first_round_store), [record]


@shares_param_docs
def run_spectrum_pass(
    window_xds: xr.Dataset,
    layout: FindContinuumUltraLayout,
    first_round_store: str,
    spectrum_regions: SpectrumRegions,
    pass_directory: str,
    round_index: int | None,
    round_kind: str,
    monitor_resources_seconds: float | None,
    pass_index: int,
) -> tuple[SpectrumReductions, list[PassRecord]]:
    """The per-channel mean (and scaled median absolute deviation) spectra of
    a spectral window over two-dimensional regions in one pass: the spectrum of
    a computed round or of its fallback to a primary-beam-based joint mask, or
    an extra-mask spectrum.

    The regions are written to the pass's regions store; the frequency tasks
    take the mean of every channel over ``MEAN_REGION`` and, with a scaled-MAD
    region, the scaled median absolute deviation over ``SCALED_MAD_REGION``
    (:func:`~astroviper.node_tasks.imaging.find_continuum_ultra.find_continuum_spectrum_task`).

    Parameters
    ----------
    window_xds : xarray.Dataset
        The spectral window, opened lazily.
    layout : FindContinuumUltraLayout
        Layout of the window (:func:`plan_find_continuum_ultra_layout`).
    first_round_store : str
        Path of the window's complete first-round store, whose primary-beam
        masks the tasks read.
    spectrum_regions : SpectrumRegions
        The regions of the spectra and their per-plane tests.
    pass_directory : str
        New directory of the pass's stores, removed when the pass ends or
        fails.
    round_index : int or None
        Round of findContinuum the pass belongs to, recorded.
    round_kind : str
        Kind of that round, recorded.
    monitor_resources_seconds : float or None
        Sampling interval (s) of GraphVIPER's resource monitor in the tasks;
        ``None``: no monitor.
    pass_index : int
        Index of the pass in the run, recorded.

    Returns
    -------
    reductions : SpectrumReductions
        The spectra, NaN in a channel without a voxel; no scaled-MAD spectrum
        without a scaled-MAD region.
    records : list of PassRecord
        The record of the pass.

    Raises
    ------
    ValueError
        If a task needs more than the memory of a thread.
    """
    regions = {"MEAN_REGION": spectrum_regions.mean_region}
    scaled_mad = spectrum_regions.scaled_mad_region is not None
    if scaled_mad:
        regions["SCALED_MAD_REGION"] = spectrum_regions.scaled_mad_region
    levels = spectrum_regions.scaled_mad_primary_beam_levels
    read_primary_beam = scaled_mad and levels is not None
    pass_kind = "spectrum"
    planes_per_batch, task_bytes = plan_planes_per_batch(
        lambda planes: find_continuum_ultra_task_bytes(
            pass_kind,
            window_xds.sizes["l"],
            window_xds.sizes["m"],
            planes,
            *_itemsizes(layout),
            read_primary_beam,
            len(regions),
            False,
            _gather_fraction(regions),
            layout.read_unit_bytes,
        ),
        layout,
    )
    os.makedirs(pass_directory)
    try:
        regions_store, tiles, driver = _write_regions(pass_directory, regions, layout)
        input_params = {
            **_task_inputs(layout),
            "first_round_store": first_round_store,
            "regions_store": regions_store,
            "use_spectrum_mask": spectrum_regions.use_spectrum_mask,
            "scaled_mad": scaled_mad,
            "scaled_mad_primary_beam_levels": None if levels is None else tuple(levels),
            "region_tiles": tiles,
            "planes_per_batch": planes_per_batch,
        }
        output, _, frames, columns = _run_graphs(
            window_xds,
            layout,
            find_continuum_spectrum_task,
            input_params,
            _task_priorities(layout, None),
            pass_directory,
            False,
            monitor_resources_seconds,
        )
    finally:
        shutil.rmtree(pass_directory, ignore_errors=True)
    columns.update(driver)
    spectra = "mean and scaled MAD spectra" if scaled_mad else "mean spectrum"
    record = _pass_record(
        layout,
        {
            "pass_index": pass_index,
            "pass_kind": pass_kind,
            "round_index": round_index,
            "round_kind": round_kind,
            "purpose": spectra,
            "data_variables_read": _variables_read(layout, read_primary_beam),
            "computes_first_round_masks": False,
            "n_moment_sets": 0,
        },
        output,
        frames,
        planes_per_batch,
        task_bytes,
        columns,
    )
    rows = output["rows"]
    return SpectrumReductions(rows["mean"], rows.get("scaled_mad")), [record]


@shares_param_docs
def run_selection_moments_pass(
    window_xds: xr.Dataset,
    layout: FindContinuumUltraLayout,
    settings: FindContinuumSettings,
    first_round_store: str,
    ranges: list,
    noise_region: xr.DataArray | None,
    pass_directory: str,
    round_index: int | None,
    round_kind: str,
    monitor_resources_seconds: float | None,
    pass_index: int,
) -> tuple[SelectionReductions, list[PassRecord]]:
    """The maps of a channel selection of a spectral window, and the noise of
    its channels, in one pass.

    The frequency tasks of the chunks holding a selected channel add the
    selected channels to the integrated and maximum moments of the valid
    voxels, zeros included (fC 1741), merged and read as the selection's maps;
    with a noise region every selected channel also gets the median and the
    unscaled median absolute deviation of its valid voxels in it (fC 3732).

    Parameters
    ----------
    window_xds : xarray.Dataset
        The spectral window, opened lazily.
    layout : FindContinuumUltraLayout
        Layout of the window (:func:`plan_find_continuum_ultra_layout`).
    settings : FindContinuumSettings
        Settings of the window; its ``moment_channel_weights`` are read.
    first_round_store : str
        Path of the window's complete first-round store, whose limit masks
        the tasks read.
    ranges : list
        ``[start, stop)`` channel ranges of the selection in the window, in
        any order, possibly overlapping.
    noise_region : xarray.DataArray or None
        Boolean region of the noise (findContinuum's noise annulus outside the
        joint mask), image-shaped with one time, frequency and polarization
        or ``(l, m)``; ``None``: no noise.
    pass_directory : str
        New directory of the pass's stores, removed when the pass ends or
        fails.
    round_index : int or None
        Round of findContinuum the pass belongs to, recorded.
    round_kind : str
        Kind of that round, recorded.
    monitor_resources_seconds : float or None
        Sampling interval (s) of GraphVIPER's resource monitor in the tasks;
        ``None``: no monitor.
    pass_index : int
        Index of the pass in the run, recorded.

    Returns
    -------
    reductions : SelectionReductions
        The maps, the selected channels of the window in increasing order and,
        with a noise region, their medians and median absolute deviations
        (NaN for a channel without a voxel).
    records : list of PassRecord
        The record of the pass.

    Raises
    ------
    ValueError
        If the ranges select no channel or a channel outside the window, or a
        task needs more than the memory of a thread.
    """
    start, stop = layout.channel_range
    selected = np.unique(channels_of_ranges(ranges))
    if not (selected.size and selected[0] >= 0 and selected[-1] < stop - start):
        raise ValueError(
            f"The ranges {ranges} select no channel, or a channel outside the "
            f"{stop - start} channels of the window"
        )
    noise = noise_region is not None
    regions = {"NOISE_REGION": noise_region} if noise else {}
    pass_kind = "selection_moments"
    planes_per_batch, task_bytes = plan_planes_per_batch(
        lambda planes: find_continuum_ultra_task_bytes(
            pass_kind,
            window_xds.sizes["l"],
            window_xds.sizes["m"],
            planes,
            *_itemsizes(layout),
            False,
            len(regions),
            True,
            _gather_fraction(regions),
            layout.read_unit_bytes,
        ),
        layout,
    )
    channels = start + selected
    os.makedirs(pass_directory)
    try:
        regions_store, _, driver = _write_regions(pass_directory, regions, layout)
        input_params = {
            **_task_inputs(layout),
            "first_round_store": first_round_store,
            "channels": channels,
            "moment_channel_weights": settings.moment_channel_weights,
            "regions_store": regions_store,
            "noise": noise,
            "planes_per_batch": planes_per_batch,
        }
        output, maps, frames, columns = _run_graphs(
            window_xds,
            layout,
            find_continuum_selection_moments_task,
            input_params,
            _task_priorities(layout, channels),
            pass_directory,
            True,
            monitor_resources_seconds,
        )
    finally:
        shutil.rmtree(pass_directory, ignore_errors=True)
    columns.update(driver)
    record = _pass_record(
        layout,
        {
            "pass_index": pass_index,
            "pass_kind": pass_kind,
            "round_index": round_index,
            "round_kind": round_kind,
            "purpose": "moments of the selection"
            + (", noise of its channels" if noise else ""),
            "data_variables_read": _variables_read(layout, False),
            "computes_first_round_masks": False,
            "n_moment_sets": 1,
        },
        output,
        frames,
        planes_per_batch,
        task_bytes,
        columns,
    )
    rows = output["rows"]
    reductions = SelectionReductions(
        *maps,
        rows["channels"] - start,
        rows.get("noise_median"),
        rows.get("noise_mad"),
    )
    return reductions, [record]


def find_continuum_ultra_driver_bytes(n_l: int, n_m: int) -> int:
    """Memory estimate of find_continuum_ultra's driver at its peak.

    The maps and masks of a later round that the driver holds at once and a
    float64 map of NumPy temporaries, for a float32 cube.

    Parameters
    ----------
    n_l, n_m : int
        Plane shape.

    Returns
    -------
    int
        Bytes.
    """
    return _DRIVER_BYTES_PER_PIXEL * n_l * n_m


def _run_graphs(
    window_xds,
    layout,
    node_task,
    input_params,
    priorities,
    pass_directory,
    moments,
    monitor_resources_seconds,
):
    """The frequency map and reduce of a pass over the tasks of
    ``priorities`` and, with ``moments``, the merge map of its moment
    partials and the read of the maps. Returns the reduce's output, the maps
    (or ``None``), the timing frames of the frequency and merge tasks, and
    the pass row's columns of the graphs and of the maps."""
    import dask
    from graphviper.graph_tools import generate_dask_workflow, map, reduce

    from astroviper.utils.chunk_io.zarr_chunks import _retry_transient_io

    graph_start = time.time()
    if moments:
        # A transient I/O error makes the stores again.
        partials_store, maps_store = _retry_transient_io(
            f"stores of {pass_directory}",
            lambda: _moment_stores(
                pass_directory, layout, (window_xds.sizes["l"], window_xds.sizes["m"])
            ),
        )
        input_params = dict(input_params, partials_store=partials_store)
    viper_graph = map(
        input_data={"img": window_xds},
        node_task_data_mapping={
            task: layout.frequency_mapping[task] for task in priorities
        },
        node_task=node_task,
        input_params=input_params,
        in_memory_compute=False,
        monitor_resources_seconds=monitor_resources_seconds,
        task_priorities=priorities,
    )
    viper_graph = reduce(
        viper_graph, reduce_find_continuum_channel_rows, {}, mode="tree"
    )
    graph_end = time.time()
    dask_graph = generate_dask_workflow(viper_graph)
    compute_start = time.time()
    output = dask.compute(dask_graph)[0]
    compute_end = time.time()
    frames = [_with_usage(output)]
    columns = {
        "merge_task_memory_plan_bytes": 0,
        "task_rss_peak_bytes": _rss_peak(frames[0]),
        "graph_start_unixtime": graph_start,
        "start_unixtime": compute_start,
        "compute_end_unixtime": compute_end,
        "merge_start_unixtime": np.nan,
        "merge_end_unixtime": np.nan,
        "T_create_map_reduce_graph": graph_end - graph_start,
        "T_generate_dask_graph": compute_start - graph_end,
        "T_compute_dask_graph": compute_end - compute_start,
        "T_merge_create_graph": 0.0,
        "T_merge_generate_dask_graph": 0.0,
        "T_merge_compute_dask_graph": 0.0,
        "bytes_partials_written": 0,
        "bytes_partials_read": 0,
        "bytes_maps_written": 0,
        **dict.fromkeys(_DRIVER_COLUMNS, 0),
    }
    if not moments:
        return output, None, frames, columns

    merge_bytes = find_continuum_ultra_merge_task_bytes(
        math.prod(layout.merge_tile_shape), layout.n_frequency_tasks
    )
    if merge_bytes > layout.budget_bytes:
        logger.warning(
            f"A find_continuum_ultra merge task needs {merge_bytes / 2**30:.2f} GiB, "
            f"more than the task budget of {layout.budget_bytes / 2**30:.2f} GiB."
        )
    merge_start = time.time()
    viper_graph = map(
        input_data={"img": window_xds},
        node_task_data_mapping=layout.tile_mapping,
        node_task=merge_find_continuum_moment_partials,
        input_params={
            "partials_store": partials_store,
            "maps_store": maps_store,
            "written_slots": output["written_slots"],
            "n_slots": layout.n_frequency_tasks,
            "value_dtype": layout.value_dtypes["sky"],
        },
        in_memory_compute=False,
        monitor_resources_seconds=monitor_resources_seconds,
    )
    merge_graph_end = time.time()
    dask_graph = generate_dask_workflow(viper_graph)
    merge_compute_start = time.time()
    merged = dask.compute(dask_graph)[0]
    merge_end = time.time()
    merge_frame = pd.concat([_with_usage(item) for item in merged], ignore_index=True)
    frames.append(merge_frame)
    read_start = time.time()
    integral, maximum, counters = read_moment_maps(maps_store)
    columns.update(
        merge_task_memory_plan_bytes=merge_bytes,
        merge_start_unixtime=merge_start,
        merge_end_unixtime=merge_end,
        T_merge_create_graph=merge_graph_end - merge_start,
        T_merge_generate_dask_graph=merge_compute_start - merge_graph_end,
        T_merge_compute_dask_graph=merge_end - merge_compute_start,
        bytes_partials_written=_stored_bytes(partials_store),
        bytes_partials_read=int(merge_frame["bytes_read_partials"].sum()),
        bytes_maps_written=int(merge_frame["bytes_written"].sum()),
        T_read_maps=time.time() - read_start,
        driver_bytes_read=counters["bytes"] + counters["index_bytes"],
    )
    return output, (integral, maximum), frames, columns


def _pass_record(
    layout, identity, output, frames, planes_per_batch, task_bytes, columns
):
    """The record of a pass: its row, its task rows and its reduce records,
    labelled with the pass's index and kinds."""
    labels = {key: identity[key] for key in ("pass_index", "pass_kind", "round_kind")}
    frequency = frames[0]
    logger.info(
        f"find_continuum_ultra pass {identity['pass_index']} ({identity['purpose']}): "
        f"{len(frequency)} of {layout.n_frequency_tasks} frequency tasks, "
        f"{planes_per_batch} planes per batch, {task_bytes / 2**30:.2f} of "
        f"{layout.budget_bytes / 2**30:.2f} GiB per task"
    )
    row = {
        **identity,
        "n_tasks": len(frequency),
        "n_channels_read": int(frequency["n_channels"].sum()),
        **{
            key: int(frequency[key].sum())
            for key in (
                "n_planes_read",
                "n_chunks_read",
                "n_chunks_skipped",
                "bytes_read_sky",
                "bytes_read_primary_beam",
            )
        },
        "planes_per_batch": planes_per_batch,
        "task_memory_plan_bytes": task_bytes,
        **columns,
    }
    return PassRecord(
        row,
        pd.concat([frame.assign(**labels) for frame in frames], ignore_index=True),
        [dict(record, **labels) for record in output.get("timing_reduce_nodes", [])],
    )


def _task_inputs(layout):
    """The inputs every frequency task of the window takes."""
    return {
        "input_image_store": layout.input_image_store,
        "variables": layout.variables,
        "polarization_index": layout.polarization_index,
        "channel_range": layout.channel_range,
        "input_checked": True,
    }


def _itemsizes(layout):
    """Bytes per voxel of the residual and of its primary beam."""
    return tuple(np.dtype(layout.value_dtypes[role]).itemsize for role in _ROLES)


def _variables_read(layout, read_primary_beam):
    """The data variables a pass reads."""
    roles = _ROLES if read_primary_beam else _ROLES[:1]
    return ", ".join(sorted(layout.variables[role] for role in roles))


def _gather_fraction(regions):
    """The largest fraction of a plane that a region holds."""
    return max(
        (np.count_nonzero(region) / region.size for region in regions.values()),
        default=0.0,
    )


def _write_regions(pass_directory, regions, layout):
    """Write the regions of a pass to its regions store (none without
    regions); returns the store, the tiles of the regions and the pass row's
    columns of the write."""
    if not regions:
        return None, {}, {"T_write_regions": 0.0, "driver_bytes_written": 0}
    start = time.time()
    regions_store = os.path.join(pass_directory, "regions.zarr")
    tiles, counters = write_region_masks(regions_store, regions, layout.tile_shape)
    return (
        regions_store,
        tiles,
        {
            "T_write_regions": time.time() - start,
            "driver_bytes_written": counters["bytes"],
        },
    )


def _task_priorities(layout, channels):
    """Task id to the planes its chunk reads, for the chunks holding a needed
    channel of the cube (every channel when ``channels`` is ``None``); the
    tasks of more planes start first."""
    priorities = {}
    for task, item in layout.frequency_mapping.items():
        frequency = item["data_selection"]["img"]["frequency"]
        n_planes = (
            frequency.stop - frequency.start
            if channels is None
            else int(
                np.count_nonzero(
                    (channels >= frequency.start) & (channels < frequency.stop)
                )
            )
        )
        if n_planes:
            priorities[task] = n_planes
    return priorities


def _moment_stores(pass_directory, layout, map_shape):
    """The partials and maps stores of a pass (Zarr v3): per moment, the
    partials ``(slot, l, m)`` chunked by one slot and a merge tile, sharded by
    one slot, and the maps ``(l, m)`` chunked by merge tiles, NaN until
    written."""
    import zarr

    stores = [
        os.path.join(pass_directory, name) for name in ("partials.zarr", "maps.zarr")
    ]
    partials, maps = (
        zarr.open_group(store, mode="w", zarr_format=3) for store in stores
    )
    # A shard holds whole tiles, so it may reach beyond the map.
    shard = tuple(
        -(-n // tile) * tile
        for n, tile in zip(map_shape, layout.merge_tile_shape, strict=True)
    )
    for name, dtype in (
        ("SKY_MOMENT_INTEGRATED", np.float64),
        ("SKY_MOMENT_MAXIMUM", layout.value_dtypes["sky"]),
    ):
        partials.create_array(
            name,
            shape=(layout.n_frequency_tasks, *map_shape),
            chunks=(1, *layout.merge_tile_shape),
            shards=(1, *shard),
            dtype=dtype,
            fill_value=0,
            compressors=None,
            dimension_names=("slot", "l", "m"),
        )
        maps.create_array(
            name,
            shape=map_shape,
            chunks=layout.merge_tile_shape,
            dtype=dtype,
            fill_value=np.nan,
            compressors=None,
            dimension_names=("l", "m"),
        )
    return stores


def _with_usage(output):
    """A task's timing rows with its resource series folded in."""
    frame, usage = output["timing_node_tasks"], output.get("resource_usage")
    return frame if usage is None else with_resource_usage(frame, usage)


def _rss_peak(frame):
    """Largest sampled RSS of the tasks of a frame, NaN without samples."""
    if "memory_rss_bytes" not in frame:
        return np.nan
    peaks = [
        max(series)
        for series in frame["memory_rss_bytes"]
        if isinstance(series, list) and series
    ]
    return max(peaks) if peaks else np.nan


def _stored_bytes(store):
    """Bytes of the chunk and shard files of a Zarr store, counted again after
    a transient I/O error."""
    from astroviper.utils.chunk_io.zarr_chunks import _retry_transient_io

    return _retry_transient_io(
        f"sizes of {store}",
        lambda: sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(store)
            for name in names
            if name != "zarr.json"
        ),
    )
