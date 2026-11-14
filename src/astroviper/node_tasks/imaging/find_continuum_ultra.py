"""Node tasks of find_continuum_ultra.

Each pass over the image store maps one of three frequency tasks over chunks
of the spectral window's channels: :func:`find_continuum_first_round_task`
(the first round), :func:`find_continuum_spectrum_task` (per-channel means and
scaled median absolute deviations over regions) and
:func:`find_continuum_selection_moments_task` (the moments of a channel
selection and the noise of its channels). A task reads the planes it needs at
the Stokes I index in batches, the next batch read ahead in a background
thread, and only the read units that can hold a voxel it uses. It returns
per-channel rows, which :func:`reduce_find_continuum_channel_rows`
concatenates, and writes its moment partials to its slot of the pass's
partials store, which :func:`merge_find_continuum_moment_partials` merges tile
by tile in GraphVIPER's binary-tree order into the pass's maps store. The
driver reads primary-beam planes, edge lines and merged maps and writes regions
and the joint mask with the functions here, and plans with the memory bounds
of :func:`find_continuum_ultra_task_bytes` and
:func:`find_continuum_ultra_merge_task_bytes`.

The stores of a pass (Zarr v3): the partials store, with
``SKY_MOMENT_INTEGRATED`` (float64) and ``SKY_MOMENT_MAXIMUM`` (the cube's
dtype) of dimensions ``(slot, l, m)``, one slot per frequency chunk, chunked by
one slot and a merge tile; the maps store, with the same arrays of dimensions
``(l, m)``, chunked by merge tiles, NaN where no channel contributed; both
created by the distributed application. The regions store is written by
:func:`write_region_masks`.
"""

import contextlib
import math
import os
import time

import numpy as np
import xarray as xr

from astroviper.utils.param_docs import shares_param_docs

_IMAGE_DIMS = ["time", "frequency", "polarization", "l", "m"]
_COUNTERS = ("bytes", "index_bytes", "units_read", "units_skipped", "shard_opens")
_LIMIT_MASK, _SPECTRUM_MASK = "PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_SPECTRUM_MASK"
_MASK_TILES = {
    _LIMIT_MASK: "PRIMARY_BEAM_LIMIT_TILES",
    _SPECTRUM_MASK: "PRIMARY_BEAM_SPECTRUM_TILES",
}
# The arrays of a partials store and of a maps store, per moment partial.
_MOMENT_ARRAYS = {
    "integrated": "SKY_MOMENT_INTEGRATED",
    "maximum": "SKY_MOMENT_MAXIMUM",
}
# T_read and T_read_masks are the reader thread's, whose work the task's
# thread waits for in T_read_wait.
_FREQUENCY_SECONDS = (
    "T_read",
    "T_read_masks",
    "T_read_wait",
    "T_masks",
    "T_moments",
    "T_spectrum",
    "T_noise",
    "T_first_round",
    "T_write_partials",
)
_PASS_KINDS = ("first_round", "spectrum", "selection_moments")
# Node-task baseline of the moments application (interpreter, worker, zarr).
_TASK_BASELINE_BYTES = 1.5 * 2**30
# Rows of the blocks that the reductions of a plane work in.
_BLOCK_ROWS = 625


@shares_param_docs
def find_continuum_first_round_task(
    input_image_store: str,
    variables: dict,
    polarization_index: int,
    primary_beam_limit: float,
    channel_range: list,
    moment_channel_weights: np.ndarray,
    first_round_store: str,
    compute_masks: bool,
    compute_moments: bool,
    partials_store: str | None,
    planes_per_batch: int,
    data_selection: dict | None = None,
    chunk_indices: tuple | None = None,
    task_id: int | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """findContinuum's first round of one chunk of a spectral window.

    Every channel of the chunk is read. With ``compute_masks`` the primary
    beam is read first, its limit and spectrum masks are made plane by plane
    and written to the first-round store
    (:func:`~astroviper.node_tasks.imaging.find_continuum_first_round.write_primary_beam_masks`),
    the residual is read in the read units inside the limit, and the
    per-channel rows of the residual and of the primary beam are returned
    with the tile occupancy of the masks; otherwise the limit masks are read
    from the store. With ``compute_moments`` the integrated and maximum
    moments of the valid, nonzero voxels (findContinuum's first-round
    moments, fC 7867-7876) go to slot ``chunk_indices[0]`` of the partials
    store.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store (Zarr) of the residual and its primary
        beam.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the
        residual and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    primary_beam_limit : float
        Primary-beam level below which a voxel of the residual and of the
        primary beam is invalid [tclean ``pblimit``]; used with
        ``compute_masks``.
    channel_range : list of int
        ``[start, stop)``, the channels of the cube in the spectral window.
    moment_channel_weights : numpy.ndarray
        float64 moment-0 weight (km/s) of every channel of the window, the
        weight of cube channel ``c`` at ``c - start``.
    first_round_store : str
        Path of the first-round store whose primary-beam masks are read, or
        written with ``compute_masks``.
    compute_masks : bool
        Make the primary-beam masks and the per-channel rows.
    compute_moments : bool
        Make the moment partials.
    partials_store : str or None
        Path of the pass's partials store (needed with ``compute_moments``).
    planes_per_batch : int
        Planes a read spans at most.
    data_selection : dict, optional
        ``{"img": {"frequency": slice}}``, the chunk of this task in channels
        of the cube; the whole window if not given.
    chunk_indices : tuple, optional
        Graph chunk index; ``chunk_indices[0]`` is the slot of the moment
        partials (needed with them).
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    dict
        ``rows``: with ``compute_masks``, ``channels`` (int32 ``(n,)``,
        channels of the cube), the rows of
        :func:`~astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions.first_round_channel_rows`
        (``sky_valid_count``, ``sky_minimum``, ``sky_maximum``,
        ``primary_beam_valid_count``, ``primary_beam_minimum``, each
        ``(n,)``) and ``tiles`` (bool ``(n, 2, n_tiles_l, n_tiles_m)``, the
        limit and the spectrum mask), as
        :func:`~astroviper.node_tasks.imaging.find_continuum_first_round.write_first_round_rows`
        reads them without ``write_failed``; empty otherwise.
        ``written_slots``: ``[chunk_indices[0]]`` with ``compute_moments``,
        else ``[]``. ``timing_node_tasks``: one-row ``pandas.DataFrame``.

    Raises
    ------
    ValueError
        If the residual is not one time of ``(time, frequency, polarization,
        l, m)``, or the task's channels are not in ``channel_range``.
    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
            check_find_continuum_ultra_task_config,
        )

        check_find_continuum_ultra_task_config(
            planes_per_batch, polarization_index, channel_range
        )

    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        write_primary_beam_masks,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        accumulate_moments,
        first_round_channel_rows,
        new_moment_partials,
        pack_plane_masks,
        primary_beam_masks,
        tile_occupancy,
        unpack_plane_masks,
        valid_voxels,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
        FirstRoundRows,
    )

    start_unixtime, cpu_start = time.time(), time.process_time()
    stored_masks = () if compute_masks else (_LIMIT_MASK,)
    # The primary beam is read first: its limit masks select the residual's units.
    role_tiles = {"primary_beam": lambda tiles: None} if compute_masks else {}
    role_tiles["sky"] = lambda tiles: tiles[_LIMIT_MASK]
    paths, cube = _open_cube(input_image_store, variables, role_tiles)
    n_l, n_m = cube.shape[-2:]
    tile_shape = cube.chunks[-2:]
    chunk = _task_channels(cube.shape[1], channel_range, data_selection)
    needed = np.full(chunk.size, compute_masks or compute_moments)
    counters = _new_counters()
    seconds = dict.fromkeys(_FREQUENCY_SECONDS, 0.0)
    reader_seconds = dict.fromkeys(_FREQUENCY_SECONDS, 0.0)
    bytes_written = {"masks": 0, "partials": 0}

    def make_masks(primary_beam, channels):
        """The masks of a batch, written to the store (reader thread)."""
        n_tiles = [
            -(-n // size) for n, size in zip((n_l, n_m), tile_shape, strict=True)
        ]
        packed = np.empty((channels.size, n_l, -(-n_m // 8)), dtype=np.uint8)
        tiles = {
            name: np.empty((channels.size, *n_tiles), dtype=bool)
            for name in _MASK_TILES
        }
        for p in range(channels.size):
            masks = primary_beam_masks(primary_beam[:, p : p + 1], primary_beam_limit)
            for name, mask in zip(_MASK_TILES, masks, strict=True):
                tiles[name][p] = tile_occupancy(mask, tile_shape)[0, 0, 0]
            packed[p] = pack_plane_masks(masks[0][0, 0, 0])
            bytes_written["masks"] += write_primary_beam_masks(
                first_round_store, channels[p : p + 1], *masks
            )
        return {_LIMIT_MASK: packed}, tiles

    map_shape = (1, 1, 1, n_l, n_m)
    partials = new_moment_partials(map_shape, cube.dtype) if compute_moments else None
    limit_mask = np.empty(map_shape, dtype=bool)
    rows = {key: [] for key in ("channels", *FirstRoundRows._fields, "tiles")}
    stream = _stream_batches(
        paths,
        polarization_index,
        chunk,
        needed,
        planes_per_batch,
        role_tiles,
        first_round_store,
        stored_masks,
        make_masks if compute_masks else None,
        counters,
        seconds,
        reader_seconds,
    )
    with contextlib.closing(stream):
        for data in stream:
            sky = data["sky"]
            if compute_masks:
                rows["channels"].append(data["channels"])
                rows["tiles"].append(
                    np.stack([data["tiles"][name] for name in _MASK_TILES], axis=1)[
                        data["offsets"]
                    ]
                )
            for channel, offset in zip(
                data["channels"].tolist(), data["offsets"].tolist(), strict=True
            ):
                plane = (slice(None), slice(offset, offset + 1))
                start = time.perf_counter()
                unpack_plane_masks(
                    data["packed"][_LIMIT_MASK][offset], n_m, out=limit_mask[0, 0, 0]
                )
                seconds["T_masks"] += time.perf_counter() - start
                if compute_masks:
                    start = time.perf_counter()
                    plane_rows = first_round_channel_rows(
                        sky[plane], data["primary_beam"][plane], limit_mask
                    )
                    for field, values in plane_rows.items():
                        rows[field].append(values[0, :, 0])
                    seconds["T_first_round"] += time.perf_counter() - start
                if compute_moments:
                    start = time.perf_counter()
                    valid = valid_voxels(sky[plane], limit_mask)
                    valid &= sky[plane] != 0
                    seconds["T_masks"] += time.perf_counter() - start
                    start = time.perf_counter()
                    index = channel - channel_range[0]
                    accumulate_moments(
                        partials,
                        sky[plane],
                        valid,
                        moment_channel_weights[index : index + 1],
                    )
                    seconds["T_moments"] += time.perf_counter() - start
                    # Freed before the next plane's.
                    valid = None

    written_slots = []
    if compute_moments:
        start = time.perf_counter()
        written_slots = [int(chunk_indices[0])]
        bytes_written["partials"] = _write_slot(
            partials_store, written_slots[0], partials
        )
        seconds["T_write_partials"] = time.perf_counter() - start
    if compute_masks:
        rows = {key: np.concatenate(values) for key, values in rows.items()}
        rows["channels"] = rows["channels"].astype(np.int32)
    else:
        rows = {}
    return {
        "rows": rows,
        "written_slots": written_slots,
        "timing_node_tasks": _frequency_row(
            "first_round",
            task_id,
            chunk_indices,
            chunk[needed],
            start_unixtime,
            cpu_start,
            counters,
            seconds,
            reader_seconds,
            sum(bytes_written.values()),
        ),
    }


@shares_param_docs
def find_continuum_spectrum_task(
    input_image_store: str,
    variables: dict,
    polarization_index: int,
    channel_range: list,
    first_round_store: str,
    regions_store: str,
    use_spectrum_mask: bool,
    scaled_mad: bool,
    scaled_mad_primary_beam_levels: tuple | None,
    region_tiles: dict,
    planes_per_batch: int,
    data_selection: dict | None = None,
    chunk_indices: tuple | None = None,
    task_id: int | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """The per-channel mean (and scaled median absolute deviation) of the
    residual over regions, for one chunk of a spectral window.

    Every channel of the chunk gets the mean of its valid voxels in
    ``MEAN_REGION``, with ``use_spectrum_mask`` only where the stored
    spectrum mask holds (findContinuum's mean spectrum, fC 8442-8459, and its
    extra-mask spectra), and with ``scaled_mad`` the scaled median absolute
    deviation of its valid voxels in ``SCALED_MAD_REGION``, between the
    primary-beam levels when they are given (fC 8502-8508). The residual is
    read in the read units holding a tile of a region inside the limit (and
    of the spectrum mask for the mean); the primary beam only with the levels,
    in the tiles of the scaled-MAD region inside the limit.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store (Zarr) of the residual and its primary
        beam.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the
        residual and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    channel_range : list of int
        ``[start, stop)``, the channels of the cube in the spectral window.
    first_round_store : str
        Path of the first-round store whose primary-beam masks are read.
    regions_store : str
        Path of the pass's regions store (:func:`write_region_masks`) with
        ``MEAN_REGION`` and, with ``scaled_mad``, ``SCALED_MAD_REGION``.
    use_spectrum_mask : bool
        Take the mean only where the stored spectrum mask holds.
    scaled_mad : bool
        Make the scaled median absolute deviation as well.
    scaled_mad_primary_beam_levels : tuple of float or None
        ``(lower, higher)``, already rounded as findContinuum writes them,
        between which the primary beam of a voxel of the scaled median
        absolute deviation must be; ``None``: no primary-beam test.
    region_tiles : dict
        Tile occupancy of the regions (:func:`write_region_masks`), on the
        grid of the residual's read units along ``l`` and ``m``.
    planes_per_batch : int
        Planes a read spans at most.
    data_selection : dict, optional
        ``{"img": {"frequency": slice}}``, the chunk of this task in channels
        of the cube; the whole window if not given.
    chunk_indices : tuple, optional
        Graph chunk index (recorded as the task's slot).
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    dict
        ``rows``: ``channels`` (int32 ``(n,)``, every channel of the chunk in
        the cube), ``mean`` and, with ``scaled_mad``, ``scaled_mad`` (float64
        ``(n,)``, NaN for a channel without a voxel); ``written_slots``:
        ``[]``; ``timing_node_tasks``: one-row ``pandas.DataFrame``.

    Raises
    ------
    ValueError
        If the residual is not one time of ``(time, frequency, polarization,
        l, m)``, or the task's channels are not in ``channel_range``.
    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
            check_find_continuum_ultra_task_config,
        )

        check_find_continuum_ultra_task_config(
            planes_per_batch, polarization_index, channel_range
        )

    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        masked_channel_means,
        masked_channel_scaled_mads,
        unpack_plane_masks,
        valid_voxels,
    )

    start_unixtime, cpu_start = time.time(), time.process_time()
    read_primary_beam = scaled_mad and scaled_mad_primary_beam_levels is not None
    mean_tiles = region_tiles["MEAN_REGION"]
    scaled_mad_tiles = region_tiles["SCALED_MAD_REGION"] if scaled_mad else False

    def sky_tiles(tiles):
        mean = mean_tiles & tiles[_SPECTRUM_MASK] if use_spectrum_mask else mean_tiles
        return (mean | scaled_mad_tiles) & tiles[_LIMIT_MASK]

    role_tiles = {"sky": sky_tiles}
    if read_primary_beam:
        role_tiles["primary_beam"] = lambda tiles: scaled_mad_tiles & tiles[_LIMIT_MASK]
    stored_masks = (
        (_LIMIT_MASK, _SPECTRUM_MASK) if use_spectrum_mask else (_LIMIT_MASK,)
    )
    paths, cube = _open_cube(input_image_store, variables, role_tiles)
    n_m = cube.shape[-1]
    chunk = _task_channels(cube.shape[1], channel_range, data_selection)
    counters = _new_counters()
    seconds = dict.fromkeys(_FREQUENCY_SECONDS, 0.0)
    reader_seconds = dict.fromkeys(_FREQUENCY_SECONDS, 0.0)
    names = ("MEAN_REGION", "SCALED_MAD_REGION") if scaled_mad else ("MEAN_REGION",)
    regions = {
        name: _read_region(regions_store, name, n_m, counters["masks"])
        for name in names
    }
    map_shape = (1, 1, 1, *cube.shape[-2:])
    masks = {name: np.empty(map_shape, dtype=bool) for name in stored_masks}
    spectra = {"mean": np.full(chunk.size, np.nan)}
    if scaled_mad:
        spectra["scaled_mad"] = np.full(chunk.size, np.nan)
    stream = _stream_batches(
        paths,
        polarization_index,
        chunk,
        np.ones(chunk.size, dtype=bool),
        planes_per_batch,
        role_tiles,
        first_round_store,
        stored_masks,
        None,
        counters,
        seconds,
        reader_seconds,
    )
    position = 0
    with contextlib.closing(stream):
        for data in stream:
            sky = data["sky"]
            for offset in data["offsets"].tolist():
                plane = (slice(None), slice(offset, offset + 1))
                start = time.perf_counter()
                for name, mask in masks.items():
                    unpack_plane_masks(
                        data["packed"][name][offset], n_m, out=mask[0, 0, 0]
                    )
                valid = valid_voxels(sky[plane], masks[_LIMIT_MASK])
                seconds["T_masks"] += time.perf_counter() - start
                start = time.perf_counter()
                spectra["mean"][position] = masked_channel_means(
                    sky[plane], valid, regions["MEAN_REGION"], masks.get(_SPECTRUM_MASK)
                )[0, 0, 0]
                if scaled_mad:
                    spectra["scaled_mad"][position] = masked_channel_scaled_mads(
                        sky[plane],
                        valid,
                        regions["SCALED_MAD_REGION"],
                        data["primary_beam"][plane] if read_primary_beam else None,
                        scaled_mad_primary_beam_levels if read_primary_beam else None,
                    )[0, 0, 0]
                seconds["T_spectrum"] += time.perf_counter() - start
                # Freed before the next plane's.
                valid = None
                position += 1
    return {
        "rows": {"channels": chunk.astype(np.int32), **spectra},
        "written_slots": [],
        "timing_node_tasks": _frequency_row(
            "spectrum",
            task_id,
            chunk_indices,
            chunk,
            start_unixtime,
            cpu_start,
            counters,
            seconds,
            reader_seconds,
            0,
        ),
    }


@shares_param_docs
def find_continuum_selection_moments_task(
    input_image_store: str,
    variables: dict,
    polarization_index: int,
    channel_range: list,
    first_round_store: str,
    channels: np.ndarray,
    moment_channel_weights: np.ndarray,
    regions_store: str | None,
    noise: bool,
    partials_store: str,
    planes_per_batch: int,
    data_selection: dict | None = None,
    chunk_indices: tuple | None = None,
    task_id: int | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """The moments of a channel selection, and the noise of its channels, for
    one chunk of a spectral window.

    The selected channels of the chunk are added, in increasing order, to the
    integrated and maximum moments of the valid voxels, zeros included (the
    maps of a selection, fC 1741), which go to slot ``chunk_indices[0]`` of
    the partials store; with ``noise``, every selected channel also gets the
    median and the unscaled median absolute deviation of its valid voxels in
    ``NOISE_REGION`` (fC 3732). The residual is read in the read units inside
    the limit; the channels between two selected ones of a read unit are read
    with them, and left out.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store (Zarr) of the residual and its primary
        beam.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the
        residual and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    channel_range : list of int
        ``[start, stop)``, the channels of the cube in the spectral window.
    first_round_store : str
        Path of the first-round store whose primary-beam masks are read.
    channels : numpy.ndarray
        Selected channels of the cube (integers), each once.
    moment_channel_weights : numpy.ndarray
        float64 moment-0 weight (km/s) of every channel of the window, the
        weight of cube channel ``c`` at ``c - start``.
    regions_store : str or None
        Path of the pass's regions store (:func:`write_region_masks`) with
        ``NOISE_REGION`` (needed with ``noise``).
    noise : bool
        Make the median and median absolute deviation of every selected
        channel in ``NOISE_REGION``.
    partials_store : str
        Path of the pass's partials store.
    planes_per_batch : int
        Planes a read spans at most.
    data_selection : dict, optional
        ``{"img": {"frequency": slice}}``, the chunk of this task in channels
        of the cube; the whole window if not given.
    chunk_indices : tuple, optional
        Graph chunk index; ``chunk_indices[0]`` is the slot of the moment
        partials (needed).
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    dict
        ``rows``: ``channels`` (int32 ``(n,)``, the selected channels of the
        chunk in increasing order) and, with ``noise``, ``noise_median`` and
        ``noise_mad`` (float64 ``(n,)``, NaN for a channel without a voxel);
        ``written_slots``: ``[chunk_indices[0]]``; ``timing_node_tasks``:
        one-row ``pandas.DataFrame``.

    Raises
    ------
    ValueError
        If the residual is not one time of ``(time, frequency, polarization,
        l, m)``, or the task's channels are not in ``channel_range``.
    """
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
            check_find_continuum_ultra_task_config,
        )

        check_find_continuum_ultra_task_config(
            planes_per_batch, polarization_index, channel_range
        )

    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        accumulate_moments,
        masked_channel_medians_and_mads,
        new_moment_partials,
        unpack_plane_masks,
        valid_voxels,
    )

    start_unixtime, cpu_start = time.time(), time.process_time()
    role_tiles = {"sky": lambda tiles: tiles[_LIMIT_MASK]}
    paths, cube = _open_cube(input_image_store, variables, role_tiles)
    n_m = cube.shape[-1]
    chunk = _task_channels(cube.shape[1], channel_range, data_selection)
    needed = np.isin(chunk, channels)
    selected = chunk[needed]
    counters = _new_counters()
    seconds = dict.fromkeys(_FREQUENCY_SECONDS, 0.0)
    reader_seconds = dict.fromkeys(_FREQUENCY_SECONDS, 0.0)
    noise_region = (
        _read_region(regions_store, "NOISE_REGION", n_m, counters["masks"])
        if noise
        else None
    )
    map_shape = (1, 1, 1, *cube.shape[-2:])
    partials = new_moment_partials(map_shape, cube.dtype)
    limit_mask = np.empty(map_shape, dtype=bool)
    noise_rows = {}
    if noise:
        noise_rows = {
            name: np.full(selected.size, np.nan)
            for name in ("noise_median", "noise_mad")
        }
    stream = _stream_batches(
        paths,
        polarization_index,
        chunk,
        needed,
        planes_per_batch,
        role_tiles,
        first_round_store,
        (_LIMIT_MASK,),
        None,
        counters,
        seconds,
        reader_seconds,
    )
    position = 0
    with contextlib.closing(stream):
        for data in stream:
            sky = data["sky"]
            for channel, offset in zip(
                data["channels"].tolist(), data["offsets"].tolist(), strict=True
            ):
                plane = (slice(None), slice(offset, offset + 1))
                start = time.perf_counter()
                unpack_plane_masks(
                    data["packed"][_LIMIT_MASK][offset], n_m, out=limit_mask[0, 0, 0]
                )
                valid = valid_voxels(sky[plane], limit_mask)
                seconds["T_masks"] += time.perf_counter() - start
                start = time.perf_counter()
                index = channel - channel_range[0]
                accumulate_moments(
                    partials,
                    sky[plane],
                    valid,
                    moment_channel_weights[index : index + 1],
                )
                seconds["T_moments"] += time.perf_counter() - start
                if noise:
                    start = time.perf_counter()
                    median, mad = masked_channel_medians_and_mads(
                        sky[plane], valid, noise_region
                    )
                    noise_rows["noise_median"][position] = median[0, 0, 0]
                    noise_rows["noise_mad"][position] = mad[0, 0, 0]
                    seconds["T_noise"] += time.perf_counter() - start
                # Freed before the next plane's.
                valid = None
                position += 1
    start = time.perf_counter()
    slot = int(chunk_indices[0])
    bytes_written = _write_slot(partials_store, slot, partials)
    seconds["T_write_partials"] = time.perf_counter() - start
    return {
        "rows": {"channels": selected.astype(np.int32), **noise_rows},
        "written_slots": [slot],
        "timing_node_tasks": _frequency_row(
            "selection_moments",
            task_id,
            chunk_indices,
            selected,
            start_unixtime,
            cpu_start,
            counters,
            seconds,
            reader_seconds,
            bytes_written,
        ),
    }


def reduce_find_continuum_channel_rows(input_data: list, input_params: dict) -> dict:
    """Reduce node of the frequency tasks of a pass.

    Parameters
    ----------
    input_data : list of dict
        Outputs of a frequency task (or of this reduce), in chunk order.
    input_params : dict
        Unused.

    Returns
    -------
    dict
        ``rows`` (every row concatenated in the order of ``input_data``),
        ``written_slots`` (sorted), ``timing_node_tasks`` (the frames
        concatenated, with the series of GraphVIPER's resource monitor, when
        it ran, as columns) and ``timing_reduce_nodes`` (one record per reduce
        node so far, ``family`` ``"frequency"``).
    """
    start_unixtime = time.time()
    rows = {
        key: np.concatenate([item["rows"][key] for item in input_data])
        for key in input_data[0]["rows"]
    }
    combined, records = _gather_timing(input_data)
    records.append(
        dict(_reduce_record(start_unixtime, input_data, combined), family="frequency")
    )
    return {
        "rows": rows,
        "written_slots": sorted(
            slot for item in input_data for slot in item["written_slots"]
        ),
        "timing_node_tasks": combined,
        "timing_reduce_nodes": records,
    }


@shares_param_docs
def merge_find_continuum_moment_partials(
    partials_store: str,
    maps_store: str,
    written_slots: list,
    n_slots: int,
    value_dtype: str,
    data_selection: dict | None = None,
    task_id: int | None = None,
) -> dict:
    """Merge the moment partials of a pass's slots over one tile and write its
    maps.

    The slots are read over the tile in slot order, a slot no task wrote
    being the identity, and merged in the pairs and order of GraphVIPER's
    binary tree
    (:func:`~astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions.merge_moment_partials_in_tree_order`),
    so the maps equal those of a tree reduce of the frequency tasks bit for
    bit. The finalized tile goes to ``SKY_MOMENT_INTEGRATED`` (float64) and
    ``SKY_MOMENT_MAXIMUM`` of the maps store.

    Parameters
    ----------
    partials_store : str
        Path of the pass's partials store.
    maps_store : str
        Path of the pass's maps store.
    written_slots : list of int
        Slots holding moment partials.
    n_slots : int
        Slots of the partials store (frequency chunks of the pass).
    value_dtype : str
        dtype of the residual, of ``SKY_MOMENT_MAXIMUM``.
    data_selection : dict, optional
        ``{"img": {"l": slice, "m": slice}}``, the tile of this task; the
        whole map if not given.
    task_id : int, optional
        Identifier of the parallel chunk being processed.

    Returns
    -------
    dict
        ``timing_node_tasks``: one-row ``pandas.DataFrame``.

    Raises
    ------
    FileNotFoundError
        If a slot of ``written_slots`` has no file for the tile.
    """
    import pandas as pd

    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        _execution_identity,
        _write_array_region,
    )
    from astroviper.node_tasks.imaging.utils.skunk_works import (
        _retry_transient_io,
        read_array_region,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        finalize_moments,
        merge_moment_partials_in_tree_order,
    )

    start_unixtime, cpu_start = time.time(), time.process_time()
    selection = (data_selection or {"img": {}})["img"]
    paths = {
        field: os.path.join(partials_store, name)
        for field, name in _MOMENT_ARRAYS.items()
    }
    arrays = {field: _open_array(path) for field, path in paths.items()}
    _, n_l, n_m = arrays["integrated"].shape
    rows = slice(*selection.get("l", slice(None)).indices(n_l)[:2])
    columns = slice(*selection.get("m", slice(None)).indices(n_m)[:2])
    tile_shape = (1, 1, 1, rows.stop - rows.start, columns.stop - columns.start)
    counters = dict.fromkeys(_COUNTERS, 0)
    seconds = dict.fromkeys(("T_read_partials", "T_merge", "T_write_maps"), 0.0)
    written = set(written_slots)

    def leaves():
        for slot in range(n_slots):
            if slot not in written:
                yield None
                continue
            start = time.perf_counter()
            partials = {}
            for field, path in paths.items():
                grid = arrays[field].shards or arrays[field].chunks
                key = arrays[field].metadata.encode_chunk_key(
                    (slot, rows.start // grid[1], columns.start // grid[2])
                )
                # A transient error of the stat is retried, not taken for absence.
                shard = os.path.join(path, key)
                try:
                    _retry_transient_io(
                        f"stat of {shard}", lambda shard=shard: os.stat(shard)
                    )
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"Slot {slot} is listed as written, but {shard} is absent"
                    ) from None
                partials[field] = read_array_region(
                    path,
                    {"slot": slice(slot, slot + 1), "l": rows, "m": columns},
                    counters=counters,
                )[0].reshape(tile_shape)
            seconds["T_read_partials"] += time.perf_counter() - start
            yield partials

    start = time.perf_counter()
    maps = finalize_moments(
        merge_moment_partials_in_tree_order(leaves()), tile_shape, np.dtype(value_dtype)
    )
    seconds["T_merge"] = time.perf_counter() - start - seconds["T_read_partials"]
    start = time.perf_counter()
    bytes_written = sum(
        _write_array_region(
            os.path.join(maps_store, name), (rows, columns), values[0, 0, 0]
        )
        for name, values in zip(_MOMENT_ARRAYS.values(), maps, strict=True)
    )
    seconds["T_write_maps"] = time.perf_counter() - start
    end_unixtime = time.time()
    row = {
        "family": "merge",
        "task_id": task_id,
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        **_execution_identity(),
        "n_chunks_read": counters["units_read"],
        "n_chunks_skipped": counters["units_skipped"],
        "n_shard_opens": counters["shard_opens"],
        "bytes_read_partials": counters["bytes"] + counters["index_bytes"],
        "bytes_written": bytes_written,
        **seconds,
        "T_cpu": time.process_time() - cpu_start,
        "T_find_continuum_task": end_unixtime - start_unixtime,
    }
    return {"timing_node_tasks": pd.DataFrame([row])}


def read_primary_beam_plane(
    input_image_store: str,
    primary_beam_variable: str,
    polarization_index: int,
    channel: int,
    primary_beam_limit: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """One plane of the primary beam, read in the driver.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store (Zarr).
    primary_beam_variable : str
        Data variable of the primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    channel : int
        Channel of the cube.
    primary_beam_limit : float
        Primary-beam level below which a voxel is invalid [tclean
        ``pblimit``].

    Returns
    -------
    plane : numpy.ndarray
        ``(1, 1, 1, l, m)`` in the store's dtype.
    limit_mask : numpy.ndarray
        Boolean limit mask of the plane
        (:func:`~astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions.primary_beam_masks`).
    counters : dict
        The read counters of
        :func:`~astroviper.node_tasks.imaging.utils.skunk_works.read_array_region`.
    """
    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        primary_beam_masks,
    )

    counters = {}
    plane = read_array_region(
        os.path.join(input_image_store, primary_beam_variable),
        {
            "frequency": slice(channel, channel + 1),
            "polarization": slice(polarization_index, polarization_index + 1),
        },
        counters=counters,
    )[0]
    return plane, primary_beam_masks(plane, primary_beam_limit)[0], counters


def read_primary_beam_edge_lines(
    input_image_store: str,
    primary_beam_variable: str,
    polarization_index: int,
    channel: int,
    primary_beam_limit: float,
) -> tuple[dict, dict, dict]:
    """The four edge lines of one plane of the primary beam, read in the
    driver (the corners and edges of findContinuum's annulus levels, fC
    8341-8368, 8389-8402).

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store (Zarr).
    primary_beam_variable : str
        Data variable of the primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    channel : int
        Channel of the cube.
    primary_beam_limit : float
        Primary-beam level below which a voxel is invalid [tclean
        ``pblimit``].

    Returns
    -------
    edge_lines : dict
        ``first_l`` and ``last_l`` (the lines ``l = 0`` and ``l = n_l - 1``,
        along ``m``), ``first_m`` and ``last_m`` (the lines ``m = 0`` and
        ``m = n_m - 1``, along ``l``), one-dimensional in the store's dtype.
    edge_lines_valid : dict
        Their boolean limit masks.
    counters : dict
        The read counters of
        :func:`~astroviper.node_tasks.imaging.utils.skunk_works.read_array_region`.
    """
    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        primary_beam_masks,
    )

    path = os.path.join(input_image_store, primary_beam_variable)
    plane = {
        "frequency": slice(channel, channel + 1),
        "polarization": slice(polarization_index, polarization_index + 1),
    }
    counters, edge_lines, edge_lines_valid = {}, {}, {}
    for name, line in (
        ("first_l", {"l": slice(0, 1)}),
        ("last_l", {"l": slice(-1, None)}),
        ("first_m", {"m": slice(0, 1)}),
        ("last_m", {"m": slice(-1, None)}),
    ):
        values = read_array_region(path, plane | line, counters=counters)[0].reshape(-1)
        edge_lines[name] = values
        edge_lines_valid[name] = primary_beam_masks(values, primary_beam_limit)[0]
    return edge_lines, edge_lines_valid, counters


def read_moment_maps(maps_store: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """The merged maps of a pass, read in the driver.

    Parameters
    ----------
    maps_store : str
        Path of the pass's maps store.

    Returns
    -------
    integral : numpy.ndarray
        float64 ``(l, m)`` ``SKY_MOMENT_INTEGRATED``, NaN where no channel
        contributed.
    maximum : numpy.ndarray
        ``(l, m)`` ``SKY_MOMENT_MAXIMUM`` in the residual's dtype, NaN where no
        channel contributed.
    counters : dict
        The read counters of
        :func:`~astroviper.node_tasks.imaging.utils.skunk_works.read_array_region`.
    """
    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region

    counters = {}
    integral, maximum = (
        read_array_region(os.path.join(maps_store, name), {}, counters=counters)[0]
        for name in _MOMENT_ARRAYS.values()
    )
    return integral, maximum, counters


def write_region_masks(
    regions_store: str, regions: dict, tile_shape: tuple
) -> tuple[dict, dict]:
    """Write the two-dimensional regions of a pass to a new regions store
    (Zarr v3), in the driver.

    Each region ``<NAME>`` is stored bit-packed along ``m`` (uint8 ``(l,
    ceil(m / 8))``, dimensions ``("l", "m_packed")``) and its tile occupancy
    as ``<NAME>_TILES`` (bool, dimensions ``("l_tile", "m_tile")``).

    Parameters
    ----------
    regions_store : str
        Path of the store; one of that path is replaced.
    regions : dict
        Name (``MEAN_REGION``, ``SCALED_MAD_REGION``, ``NOISE_REGION``) to a
        boolean region: an ``(l, m)`` array, or an image-shaped one with one
        time, frequency and polarization.
    tile_shape : tuple of int
        ``(l, m)`` of the tiles: the read units of the residual.

    Returns
    -------
    tiles : dict
        Name to the tile occupancy of the region, bool ``(n_tiles_l,
        n_tiles_m)``.
    counters : dict
        ``bytes``: the bytes stored.
    """
    import zarr

    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        _write_array_region,
    )
    from astroviper.node_tasks.imaging.utils.skunk_works import _retry_transient_io
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        pack_plane_masks,
        tile_occupancy,
    )

    def write():
        group = zarr.open_group(regions_store, mode="w", zarr_format=3)
        tiles, written = {}, 0
        for name, region in regions.items():
            plane = np.asarray(region)
            plane = plane.reshape(plane.shape[-2:])
            tiles[name] = tile_occupancy(plane[None, None, None], tile_shape)[0, 0, 0]
            for array_name, values, dims in (
                (name, pack_plane_masks(plane), ("l", "m_packed")),
                (f"{name}_TILES", tiles[name], ("l_tile", "m_tile")),
            ):
                group.create_array(
                    array_name,
                    shape=values.shape,
                    chunks=values.shape,
                    dtype=values.dtype,
                    fill_value=0,
                    compressors=zarr.codecs.ZstdCodec(level=1),
                    dimension_names=dims,
                )
                written += _write_array_region(
                    os.path.join(regions_store, array_name),
                    tuple(slice(0, size) for size in values.shape),
                    values,
                )
        return tiles, written

    # A transient I/O error makes the store again.
    tiles, written = _retry_transient_io(f"write of {regions_store}", write)
    return tiles, {"bytes": written}


def write_joint_mask_store(path: str, joint_mask: xr.Dataset) -> dict:
    """Write a joint mask to a new Zarr v3 store, in the driver.

    Parameters
    ----------
    path : str
        Path of the store; one of that path is replaced.
    joint_mask : xarray.Dataset
        ``JOINT_MASK`` and ``JOINT_MASK_FOOTPRINT`` with their coordinates and
        attributes.

    Returns
    -------
    dict
        ``bytes``: the bytes stored in the chunks.
    """
    from astroviper.node_tasks.imaging.utils.skunk_works import _retry_transient_io

    _retry_transient_io(
        f"write of {path}",
        lambda: joint_mask.to_zarr(path, mode="w", zarr_format=3, consolidated=False),
    )
    return {
        "bytes": sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(path)
            for name in names
            if name != "zarr.json"
        )
    }


def find_continuum_ultra_task_bytes(
    pass_kind: str,
    n_l: int,
    n_m: int,
    planes_per_batch: int,
    sky_itemsize: int,
    primary_beam_itemsize: int,
    read_primary_beam: bool,
    n_regions: int,
    moments: bool,
    gather_fraction: float,
    read_unit_bytes: int,
) -> int:
    """Memory bound of a frequency task of find_continuum_ultra.

    The baseline of a node task; two buffers of ``planes_per_batch`` planes
    (the batch reduced and the one read ahead) per array read, with the
    bit-packed primary-beam masks of their planes (the limit mask, and the
    spectrum mask in a spectrum pass); the boolean planes of the plane being
    reduced (the limit mask and the valid voxels; the spectrum mask in a
    spectrum pass; in the first round the valid voxels and their nonzero test
    only with moments, and the two masks made from a primary-beam plane when
    it reads the primary beam) and one per region; the moment partials (a
    float64 integral and a maximum in the residual's dtype); the values a
    masked reduction gathers, in the residual's dtype and in float64; two
    float64 blocks of rows; and a read unit being decoded with its stored
    bytes.

    Parameters
    ----------
    pass_kind : str
        ``"first_round"``, ``"spectrum"`` or ``"selection_moments"``.
    n_l, n_m : int
        Plane shape.
    planes_per_batch : int
        Planes a read spans at most.
    sky_itemsize, primary_beam_itemsize : int
        Bytes per voxel of the residual and of the primary beam.
    read_primary_beam : bool
        The task reads the primary beam (the first round without a store, a
        spectrum pass with primary-beam levels).
    n_regions : int
        Two-dimensional regions of the task.
    moments : bool
        The task makes moment partials.
    gather_fraction : float
        Largest fraction of a plane that a masked reduction gathers.
    read_unit_bytes : int
        Decoded bytes of the largest read unit of the arrays the task reads.

    Returns
    -------
    int
        Bytes.

    Raises
    ------
    ValueError
        If ``pass_kind`` is not one of the three.
    """
    if pass_kind not in _PASS_KINDS:
        raise ValueError(f"pass_kind is one of {_PASS_KINDS}, not {pass_kind!r}")
    pixels = n_l * n_m
    plane_bytes = sky_itemsize + (primary_beam_itemsize if read_primary_beam else 0)
    spectrum = pass_kind == "spectrum"
    packed_bytes = (1 + spectrum) * n_l * -(-n_m // 8)
    if pass_kind == "first_round":
        bool_planes = 1 + 2 * moments + 2 * read_primary_beam
    else:
        bool_planes = 2 + spectrum
    value_bytes = 8 + sky_itemsize
    return int(
        _TASK_BASELINE_BYTES
        + 2 * planes_per_batch * (plane_bytes * pixels + packed_bytes)
        + (bool_planes + n_regions) * pixels
        + value_bytes * pixels * moments
        + value_bytes * gather_fraction * pixels
        + 2 * 8 * _BLOCK_ROWS * n_m
        + 2 * read_unit_bytes
    )


def find_continuum_ultra_merge_task_bytes(tile_pixels: int, n_slots: int) -> int:
    """Memory bound of a merge task of find_continuum_ultra.

    The baseline of a node task, the moment partials of a tile that
    GraphVIPER's tree order holds at once (at most ``ceil(log2(n_slots)) +
    1``, each at most 16 bytes per pixel: a float64 integral and a maximum in
    the residual's dtype), and the finalized maps with the mask of their empty
    pixels.

    Parameters
    ----------
    tile_pixels : int
        Pixels of a merge tile.
    n_slots : int
        Slots merged.

    Returns
    -------
    int
        Bytes.
    """
    partials = math.ceil(math.log2(max(n_slots, 1))) + 1
    return int(_TASK_BASELINE_BYTES + (16 * partials + 17) * tile_pixels)


def with_resource_usage(timing, usage):
    """A copy of a node task's one-row ``timing`` frame with the resource
    monitor's ``usage`` series as columns (a list becomes one cell; scalars
    broadcast)."""
    timing = timing.copy()
    for key, value in usage.items():
        timing[key] = [value] if isinstance(value, list) else value
    return timing


def _stream_batches(
    paths,
    polarization_index,
    chunk,
    needed,
    planes_per_batch,
    role_tiles,
    first_round_store,
    stored_masks,
    make_masks,
    counters,
    seconds,
    reader_seconds,
):
    """Yield the needed planes of a frequency task in batches, each read by a
    background thread while the one before is reduced.

    The positions of ``chunk`` where ``needed`` holds are grouped into
    batches spanning at most ``planes_per_batch`` planes within one shard.
    The needed channels of a read unit (an inner chunk of a sharded array, a
    chunk otherwise) are read in one batch when they span at most
    ``planes_per_batch`` planes, the channels between them included, so that
    the unit is decoded once per task. Each batch reads, at the Stokes I
    index and into the other of two buffers per array, the bit-packed masks
    ``stored_masks`` of the first-round store at its needed channels and,
    per role in the order of ``role_tiles``, the read units holding a tile to
    read of a needed plane: ``role_tiles[role]`` gives these tiles,
    ``(span, n_tiles_l, n_tiles_m)`` or ``None`` for every tile, from the
    batch's tile occupancy per mask name, the stored one of ``stored_masks``.
    With ``make_masks``, the masks and their tile occupancy are instead
    ``make_masks(primary_beam, channels)`` of the primary beam, read before
    the residual.

    Yields
    ------
    dict
        ``channels`` (the needed channels of the batch), ``offsets`` (their
        planes in the buffers), the planes of each role ``(time, span, 1, l,
        m)``, ``packed`` (mask name to ``(span, l, ceil(m / 8))``) and, with
        ``make_masks``, ``tiles``.
    """
    import itertools
    from concurrent.futures import ThreadPoolExecutor

    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region

    arrays = {role: _open_array(paths[role]) for role in role_tiles}
    cube = arrays["sky"]
    n_time, _, _, n_l, n_m = cube.shape
    unit = cube.chunks[1]
    shard = (cube.shards or cube.chunks)[1]
    # Positions of the needed channels of each batch; a batch reads the planes
    # from its first to its last position.
    batches = []
    extendable = False
    for _, group in itertools.groupby(
        np.flatnonzero(needed).tolist(), key=lambda position: chunk[position] // unit
    ):
        group = list(group)
        if group[-1] - group[0] >= planes_per_batch:
            # A read unit of more needed planes than a batch, read per batch.
            batches.append([group[0]])
            for position in group[1:]:
                if position - batches[-1][0] < planes_per_batch:
                    batches[-1].append(position)
                else:
                    batches.append([position])
            extendable = False
        elif (
            extendable
            and group[0] == batches[-1][-1] + 1
            and group[-1] - batches[-1][0] < planes_per_batch
            and chunk[group[0]] // shard == chunk[batches[-1][0]] // shard
        ):
            batches[-1].extend(group)
        else:
            batches.append(group)
            extendable = True
    frequency = slice(int(chunk[0]), int(chunk[-1]) + 1)
    stored_tiles = {
        name: read_array_region(
            os.path.join(first_round_store, _MASK_TILES[name]),
            {"frequency": frequency},
            counters=counters["masks"],
        )[0]
        for name in stored_masks
    }
    span = max((batch[-1] - batch[0] + 1 for batch in batches), default=0)
    buffers = {
        role: [
            np.empty((n_time, span, 1, n_l, n_m), dtype=array.dtype)
            for _ in range(min(2, len(batches)))
        ]
        for role, array in arrays.items()
    }

    def read(batch, index):
        start = time.perf_counter()
        first, stop = batch[0], batch[-1] + 1
        channels, wanted = chunk[first:stop], needed[first:stop, None, None]
        frequency = slice(int(channels[0]), int(channels[-1]) + 1)
        region = {
            "frequency": frequency,
            "polarization": slice(polarization_index, polarization_index + 1),
        }
        data = {
            "channels": chunk[batch],
            "offsets": np.subtract(batch, first),
            "packed": {
                name: read_array_region(
                    os.path.join(first_round_store, name),
                    {"frequency": frequency},
                    wanted,
                    counters["masks"],
                )[0]
                for name in stored_masks
            },
        }
        tiles = {name: values[first:stop] for name, values in stored_tiles.items()}
        starts = np.flatnonzero(np.diff(channels // unit, prepend=-1))
        masks_seconds = 0.0
        for role, tiles_of in role_tiles.items():
            if role == "sky" and make_masks is not None:
                masks_start = time.perf_counter()
                data["packed"], data["tiles"] = make_masks(
                    data["primary_beam"], channels
                )
                tiles = data["tiles"]
                masks_seconds = time.perf_counter() - masks_start
            used = tiles_of(tiles)
            used = wanted if used is None else used & wanted
            data[role] = read_array_region(
                paths[role],
                region,
                np.logical_or.reduceat(used, starts, axis=0)[None, :, None],
                counters[role],
                buffers[role][index][:, : channels.size],
            )[0]
            counters[role]["planes"] += channels.size
        reader_seconds["T_read"] += time.perf_counter() - start - masks_seconds
        reader_seconds["T_read_masks"] += masks_seconds
        return data

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read, batches[0], 0) if batches else None
        for k in range(len(batches)):
            start = time.perf_counter()
            data = future.result()
            seconds["T_read_wait"] += time.perf_counter() - start
            if k + 1 < len(batches):
                future = executor.submit(read, batches[k + 1], (k + 1) % 2)
            yield data


def _open_cube(input_image_store, variables, roles):
    """The paths of the data variables of ``roles``, in their order, and the
    residual's array, required to be one time of (time, frequency,
    polarization, l, m)."""
    paths = {role: os.path.join(input_image_store, variables[role]) for role in roles}
    cube = _open_array(paths["sky"])
    if list(cube.metadata.dimension_names) != _IMAGE_DIMS or cube.shape[0] != 1:
        raise ValueError(
            f"{paths['sky']} has dimensions {cube.metadata.dimension_names} and "
            f"shape {cube.shape}, not one time of {_IMAGE_DIMS}"
        )
    return paths, cube


def _open_array(path):
    """A Zarr array opened for reading; a transient I/O error of its metadata
    read is retried."""
    import zarr

    from astroviper.node_tasks.imaging.utils.skunk_works import _retry_transient_io

    return _retry_transient_io(
        f"metadata of {path}", lambda: zarr.open_array(path, mode="r")
    )


def _task_channels(n_channels, channel_range, data_selection):
    """The channels of the cube in a frequency task's chunk, all in the
    window."""
    start, stop = channel_range
    frequency = (data_selection or {"img": {}})["img"].get(
        "frequency", slice(start, stop)
    )
    chunk = np.arange(n_channels)[frequency]
    if not (chunk.size and start <= chunk[0] and chunk[-1] < stop):
        raise ValueError(
            f"The task's channels {frequency} are not in the window {channel_range}"
        )
    return chunk


def _new_counters():
    """Read counters of the residual, the primary beam and the masks, with the
    planes read."""
    return {
        role: dict.fromkeys((*_COUNTERS, "planes"), 0)
        for role in ("sky", "primary_beam", "masks")
    }


def _read_region(regions_store, name, n_m, counters):
    """A boolean ``(l, m)`` region of a regions store."""
    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        unpack_plane_masks,
    )

    packed = read_array_region(
        os.path.join(regions_store, name), {}, counters=counters
    )[0]
    return unpack_plane_masks(packed, n_m)


def _write_slot(partials_store, slot, moment_partials):
    """Write moment partials ``(1, 1, 1, l, m)`` to a slot of a partials
    store, every chunk even if empty; returns the bytes stored."""
    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        _write_array_region,
    )

    n_l, n_m = moment_partials["integrated"].shape[-2:]
    return sum(
        _write_array_region(
            os.path.join(partials_store, name),
            (slice(slot, slot + 1), slice(0, n_l), slice(0, n_m)),
            moment_partials[field][0, 0, 0][None],
            {"write_empty_chunks": True},
        )
        for field, name in _MOMENT_ARRAYS.items()
    )


def _frequency_row(
    pass_kind,
    task_id,
    chunk_indices,
    processed,
    start_unixtime,
    cpu_start,
    counters,
    seconds,
    reader_seconds,
    bytes_written,
):
    """The one-row timing frame of a frequency task."""
    import pandas as pd

    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        _execution_identity,
    )

    cube = (counters["sky"], counters["primary_beam"])
    end_unixtime = time.time()
    row = {
        "family": "frequency",
        "pass_kind": pass_kind,
        "task_id": task_id,
        "slot": None if chunk_indices is None else int(chunk_indices[0]),
        "first_channel": int(processed[0]) if processed.size else None,
        "last_channel": int(processed[-1]) if processed.size else None,
        "n_channels": int(processed.size),
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        **_execution_identity(),
        "n_planes_read": sum(read["planes"] for read in cube),
        "n_chunks_read": sum(read["units_read"] for read in cube),
        "n_chunks_skipped": sum(read["units_skipped"] for read in cube),
        **{
            f"bytes_read_{role}": read["bytes"] + read["index_bytes"]
            for role, read in counters.items()
        },
        "n_shard_opens": sum(read["shard_opens"] for read in cube),
        "bytes_written": bytes_written,
        **{key: seconds[key] + reader_seconds[key] for key in _FREQUENCY_SECONDS},
        "T_cpu": time.process_time() - cpu_start,
        "T_find_continuum_task": end_unixtime - start_unixtime,
    }
    return pd.DataFrame([row])


def _gather_timing(input_data):
    """The timing rows of the inputs of a reduce node (with the series of
    GraphVIPER's resource monitor folded in) and their reduce records."""
    import pandas as pd

    timing, records = [], []
    for item in input_data:
        usage = item.get("resource_usage")
        frame = item["timing_node_tasks"]
        timing.append(frame if usage is None else with_resource_usage(frame, usage))
        records.extend(item.get("timing_reduce_nodes", []))
    return pd.concat(timing, ignore_index=True), records


def _reduce_record(start_unixtime, input_data, combined):
    """The timing record of a reduce node."""
    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        _execution_identity,
    )

    end_unixtime = time.time()
    return {
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        "T_reduce_node": end_unixtime - start_unixtime,
        **_execution_identity(),
        "n_inputs": len(input_data),
        "n_rows_out": int(len(combined)),
    }
