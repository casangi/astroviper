"""Node tasks of the find_continuum distributed applications.

``find_continuum`` and ``find_continuum_fused`` map
:func:`find_continuum_cube_reductions` over chunks of channels and merge the
partial reductions with :func:`merge_find_continuum_cube_reductions`.
``find_continuum_ultra`` streams batches of planes
(:func:`find_continuum_streamed_reductions`): each task writes its moment maps
to its slot of a scratch store, the reduce gathers the per-channel rows
(:func:`merge_find_continuum_streamed_reductions`) and tile tasks merge the
slots (:func:`merge_find_continuum_partial_maps`). The imager's first round
writes the primary-beam masks and statistics of every chunk it images
(:func:`find_continuum_first_round_chunk`) and keeps moments 0 and 8 in a
registry of the worker process (:func:`add_first_round_moments`), drained after
the imaging compute. The two-dimensional images that masks refer to are the
small Zarr images find_continuum writes (:func:`write_small_image`).
"""

import bisect
import contextlib
import itertools
import json
import os
import threading
import time

import numpy as np

# Windows of the imager's first round whose moments a worker process holds;
# the least recently used one goes to a slot before another is allocated.
FIRST_ROUND_WINDOWS_IN_MEMORY = 2

_IMAGE_DIMS = ["time", "frequency", "polarization", "l", "m"]
_COUNTERS = ("bytes", "index_bytes", "units_read", "units_skipped", "shard_opens")
_MASKS = ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_PROFILE_MASK")

# Run id to the first-round moments of this process: "counts" (additions per
# channel), "moments" (window to its FindContinuumMoments), "recent" (held
# windows, least recently used first), "slots", "sequence" and "slot_bytes".
_FIRST_ROUND_MOMENTS = {}
_FIRST_ROUND_LOCK = threading.Lock()


def write_small_image(path, values, mask, metadata):
    """Write a small image (CASA axis order) as a Zarr store.

    Each variable is one uncompressed chunk, written and read whole. The store
    gets a new write id, so that a reader can tell a rewritten image from the
    one it read before, and records whether every pixel is valid.

    Parameters
    ----------
    path : str
        Store path.
    values : numpy.ndarray
        Pixel values, shape ``(l, m, polarization, frequency)``.
    mask : numpy.ndarray
        Pixel mask of the same shape (``True`` = valid).
    metadata : dict
        CASA image metadata (coordinates, beam, units), stored as JSON.

    Returns
    -------
    str
        The write id of the store.
    """
    import uuid

    import xarray as xr

    dims = ["time", "frequency", "polarization", "l", "m"]
    write_id = uuid.uuid4().hex
    xds = xr.Dataset(
        {
            "SKY": (dims, np.transpose(values, (3, 2, 0, 1))[None]),
            "MASK": (dims, np.transpose(mask, (3, 2, 0, 1))[None]),
        },
        attrs={
            "casa_metadata": json.dumps(metadata),
            "write_id": write_id,
            "all_valid": bool(np.all(mask)),
        },
    )
    encoding = {
        name: {"chunks": xds[name].shape, "compressors": None} for name in xds.data_vars
    }
    xds.to_zarr(path, mode="w", consolidated=False, encoding=encoding)
    return write_id


def small_image_attributes(path):
    """The attributes of a small image, read from its ``zarr.json`` only.

    Parameters
    ----------
    path : str
        Store path.

    Returns
    -------
    dict
        ``write_id`` (str), ``all_valid`` (bool, no masked pixel) and
        ``casa_metadata`` (dict).
    """
    with open(os.path.join(path, "zarr.json")) as stream:
        attributes = json.load(stream)["attributes"]
    return {
        "write_id": attributes["write_id"],
        "all_valid": attributes["all_valid"],
        "casa_metadata": json.loads(attributes["casa_metadata"]),
    }


def read_small_image(path):
    """Read a small image written by :func:`write_small_image`.

    Parameters
    ----------
    path : str
        Store path.

    Returns
    -------
    values : numpy.ndarray
        Pixel values in CASA axis order ``(l, m, polarization, frequency)``.
    mask : numpy.ndarray
        Pixel mask (``True`` = valid).
    metadata : dict
        CASA image metadata.
    """
    import xarray as xr

    xds = xr.open_zarr(path, chunks=None, consolidated=False)
    values = np.transpose(xds["SKY"].values[0], (2, 3, 1, 0))
    mask = np.transpose(xds["MASK"].values[0], (2, 3, 1, 0)).astype(bool)
    return values, mask, json.loads(xds.attrs["casa_metadata"])


def find_continuum_cube_reductions(
    input_image_store,
    variables,
    image_paths,
    requests,
    primary_beam=None,
    primary_beam_limit=None,
    channel_offset=0,
    data_selection=None,
    task_id=None,
):
    """Partial reductions of one chunk of channels of the dirty cube.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store holding the cube.
    variables : dict
        Cube name (as the lattice expressions write it) to the data variable of
        the store that holds it.
    image_paths : dict
        Name of every small image the masks refer to (as the lattice
        expressions write it) to its absolute path.
    requests : list of dict
        See :func:`astroviper.processing_functions.imaging.find_continuum.find_continuum_cube_reductions`.
    primary_beam : str, optional
        Data variable of the primary beam, read with the chunk when
        ``primary_beam_limit`` is given.
    primary_beam_limit : float, optional
        Pixels where the primary beam is below this level are masked in every
        cube. ``None`` (default): the cubes as they are.
    channel_offset : int, default 0
        Channel of the store that is channel 0 of the requests: the first
        channel of the spectral window findContinuum sees.
    data_selection : dict, optional
        ``{"img": {"frequency": slice}}``, the chunk of this task in channels
        of the store.
    task_id : int, optional
        Graph node id.

    Returns
    -------
    dict
        ``reductions`` (the partial results) and ``timing_node_tasks`` (a
        one-row ``pandas.DataFrame``).
    """
    import pandas as pd
    import xarray as xr

    from astroviper.processing_functions.imaging.find_continuum import (
        find_continuum_cube_reductions as reduce_chunk,
    )

    start_unixtime = time.time()
    selection = (data_selection or {"img": {}})["img"]
    lazy_xds = xr.open_zarr(input_image_store, chunks=None)
    names = set(variables.values())
    if primary_beam_limit is not None:
        names.add(primary_beam)
    img_xds = lazy_xds[sorted(names)].isel(selection).sel(polarization=["I"]).load()
    frequency_selection = selection.get("frequency", slice(None))
    channel_index = (
        np.arange(lazy_xds.sizes["frequency"])[frequency_selection] - channel_offset
    )
    images = {}
    for name, path in image_paths.items():
        values, mask, _ = read_small_image(path)
        images[name] = np.where(mask, values, np.nan)
    T_load = time.time() - start_unixtime

    start = time.time()
    partials = reduce_chunk(
        img_xds,
        requests,
        variables,
        images,
        channel_index,
        primary_beam,
        primary_beam_limit,
    )
    T_reduce = time.time() - start

    timing_df = pd.DataFrame(
        [
            {
                "task_id": task_id,
                "start_unixtime": start_unixtime,
                "n_channels": channel_index.size,
                "T_load": T_load,
                "T_reduce": T_reduce,
                "T_find_continuum_task": time.time() - start_unixtime,
                **_execution_identity(),
            }
        ]
    )
    return {"reductions": partials, "timing_node_tasks": timing_df}


def with_resource_usage(timing, usage):
    """A copy of a node task's one-row ``timing`` frame with the resource
    monitor's ``usage`` series as columns (a list becomes one cell; scalars
    broadcast)."""
    timing = timing.copy()
    for key, value in usage.items():
        timing[key] = [value] if isinstance(value, list) else value
    return timing


def merge_find_continuum_cube_reductions(input_data, input_params):
    """Reduce node task: merge the outputs of map or reduce nodes.

    Parameters
    ----------
    input_data : list of dict
        Outputs of :func:`find_continuum_cube_reductions` (or of this task).
    input_params : dict
        Unused.

    Returns
    -------
    dict
        Merged ``reductions``, the concatenated ``timing_node_tasks`` (with
        the series of graphviper's resource monitor, when it ran, as columns)
        and ``timing_reduce_nodes``, one record per reduce node so far.
    """
    from astroviper.processing_functions.imaging.find_continuum import (
        merge_find_continuum_reductions,
    )

    start_unixtime = time.time()
    merged = input_data[0]["reductions"]
    for other in input_data[1:]:
        merged = merge_find_continuum_reductions(merged, other["reductions"])
    combined, records = _gather_timing(input_data)
    records.append(_reduce_record(start_unixtime, input_data, combined))
    return {
        "reductions": merged,
        "timing_node_tasks": combined,
        "timing_reduce_nodes": records,
    }


def find_continuum_streamed_reductions(
    input_image_store,
    variables,
    polarization_index,
    primary_beam_limit,
    requests,
    frequency_axis,
    rest_frequency,
    first_round_store,
    compute_first_round,
    read_primary_beam,
    two_d_mask_files,
    image_paths,
    partials_store,
    planes_per_batch,
    data_selection=None,
    chunk_indices=None,
    task_id=None,
):
    """Reductions of one chunk of channels of the dirty cube, streamed in batches of planes.

    The channels that a request selects are read at the Stokes I index in
    batches of at most ``planes_per_batch`` planes within one shard, one read
    per batch and variable, and only the inner chunks that a request can use:
    inside the primary-beam limit, in the footprint of its two-dimensional
    mask and, with the stored ``"pb">0.21`` mask, where that mask holds a
    pixel. A background thread reads the next batch into a second buffer while
    the planes of a batch are added to the partial results one by one
    (:func:`~astroviper.processing_functions.imaging.find_continuum.find_continuum_batch_reductions`).
    The moment maps go to slot ``chunk_indices[0]`` of the scratch store; only
    per-channel rows are returned.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store holding the cube.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the cube
        and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    primary_beam_limit : float or None
        Pixels where the primary beam is below this level are masked in every
        cube; ``None``: no limit.
    requests : list of dict
        Encoded requests (see
        :func:`~astroviper.processing_functions.imaging.find_continuum.find_continuum_batch_reductions`),
        their channels in channels of the store; ``footprint`` is the tile
        occupancy of the two-dimensional mask (``None``: every tile). Moments
        requests carry ``channel_offset``, the channel of the store that is
        channel 0 of ``frequency_axis``.
    frequency_axis : sequence
        Linear frequency axis of the spectral window findContinuum sees, given
        to every moments request.
    rest_frequency : float
        Rest frequency of the moment axis (Hz), given to every moments request.
    first_round_store : str
        First-round store whose primary-beam masks and tile occupancy are
        read, or written with ``compute_first_round``.
    compute_first_round : bool
        Make the first round of every channel of the chunk: the primary-beam
        masks, from the primary beam (needs ``read_primary_beam``), written to
        ``first_round_store``, and the rows of its statistics and tile
        occupancy.
    read_primary_beam : bool
        Read the primary beam (for ``compute_first_round`` or a request that
        uses it).
    two_d_mask_files : list of str
        ``.npy`` files of the two-dimensional masks that requests index, each
        ``pack_plane_masks`` of an ``(l, m)`` mask.
    image_paths : dict
        Name of every two-dimensional image that a lattice expression of a
        request names, to its path (:func:`write_small_image`).
    partials_store : str or None
        Scratch store with an array ``R<r>_<FIELD>`` of dimensions ``(slot, l,
        m)``, one shard per slot, per moments request ``r`` and field of its
        partial (``FindContinuumMoments.state``); ``None`` without moments
        requests.
    planes_per_batch : int
        Planes per read.
    data_selection : dict, optional
        ``{"img": {"frequency": slice}}``, the chunk of this task in channels
        of the store; the whole store if not given.
    chunk_indices : tuple, optional
        Graph chunk index; ``chunk_indices[0]`` is the slot written.
    task_id : int, optional
        Graph node id.

    Returns
    -------
    dict
        ``reductions``: per request, the rows of ``new_streamed_partials``
        (``None`` for moments); ``written``: ``{request index: [slot]}`` of the
        moments requests written; ``first_round``: with
        ``compute_first_round``, ``{"channels": int32 (n,), "statistics":
        float64 (n, 2, 5), "tiles": bool (n, 2, n_tiles_l, n_tiles_m),
        "write_failed": False}``, else ``None``; ``timing_node_tasks``: a
        one-row ``pandas.DataFrame``.

    Raises
    ------
    ValueError
        If the cube's dimensions are not ``(time, frequency, polarization, l,
        m)``.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    import zarr

    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region
    from astroviper.processing_functions.imaging.find_continuum import (
        find_continuum_batch_reductions,
        first_round_statistics,
        new_streamed_partials,
        pack_plane_masks,
        primary_beam_masks,
        tile_occupancy,
        unpack_plane_masks,
    )

    start_unixtime, cpu_start = time.time(), time.process_time()
    requests = [
        dict(request, frequency_axis=frequency_axis, rest_frequency=rest_frequency)
        if request["kind"] == "moments"
        else request
        for request in requests
    ]
    # The primary beam is read first: the first round's masks select the sky's chunks.
    roles = ["primary_beam", "sky"] if read_primary_beam else ["sky"]
    paths = {role: os.path.join(input_image_store, variables[role]) for role in roles}
    arrays = {role: zarr.open_array(paths[role], mode="r") for role in roles}
    cube = arrays["sky"]
    if list(cube.metadata.dimension_names) != _IMAGE_DIMS:
        raise ValueError(
            f"{paths['sky']} has dimensions {cube.metadata.dimension_names}, "
            f"not {_IMAGE_DIMS}"
        )
    n_l, n_m = cube.shape[-2:]
    unit = cube.chunks
    shard = (cube.shards or cube.chunks)[1]
    n_tiles = (-(-n_l // unit[3]), -(-n_m // unit[4]))
    selection = (data_selection or {"img": {}})["img"]
    chunk = np.arange(cube.shape[1])[selection.get("frequency", slice(None))]
    selected = [
        np.ones(chunk.size, dtype=bool)
        if request["channels"] is None
        else np.isin(chunk, request["channels"])
        for request in requests
    ]
    needed = np.full(chunk.size, compute_first_round)
    for chosen in selected:
        needed |= chosen
    batches = []
    for position in np.flatnonzero(needed):
        if (
            batches
            and position == batches[-1][-1] + 1
            and len(batches[-1]) < planes_per_batch
            and chunk[position] % shard
        ):
            batches[-1].append(position)
        else:
            batches.append([position])
    uses_profile = [
        any(term[0] == "profile" for term in request["cube_terms"])
        for request in requests
    ]
    read_profile = any(uses_profile)
    stored_tiles = {}
    if batches and not compute_first_round:
        channel_range = slice(chunk[0], chunk[-1] + 1)
        for key, name, wanted in (
            ("inside", "PRIMARY_BEAM_LIMIT_TILES", primary_beam_limit is not None),
            ("profile", "PRIMARY_BEAM_PROFILE_TILES", read_profile),
        ):
            if wanted:
                stored_tiles[key] = zarr.open_array(
                    os.path.join(first_round_store, name), mode="r"
                )[channel_range]

    def needed_tiles(batch):
        tiles = np.zeros((len(batch), *n_tiles), dtype=bool)
        for k, request in enumerate(requests):
            chosen = selected[k][batch]
            if not chosen.any():
                continue
            footprint = request["footprint"]
            used = np.ones(n_tiles, dtype=bool) if footprint is None else footprint
            used = used & chosen[:, None, None]
            if uses_profile[k]:
                used &= stored_tiles["profile"][batch]
            tiles |= used
        if "inside" in stored_tiles:
            tiles &= stored_tiles["inside"][batch]
        return tiles

    def read_units(tiles, channels):
        # The grid of read units the batch overlaps: (time, frequency, polarization, l, m).
        starts = np.flatnonzero(np.diff(channels // unit[1], prepend=-1))
        return np.logical_or.reduceat(tiles, starts, axis=0)[None, :, None]

    def first_round_masks(primary_beam):
        packed = np.empty((2, primary_beam.shape[1], n_l, -(-n_m // 8)), np.uint8)
        tiles = np.empty((primary_beam.shape[1], 2, *n_tiles), dtype=bool)
        for p in range(primary_beam.shape[1]):
            masks = primary_beam_masks(primary_beam[:, p : p + 1], primary_beam_limit)
            for k, mask in enumerate(masks):
                packed[k, p] = pack_plane_masks(mask[0, 0, 0])
                tiles[p, k] = tile_occupancy(mask, unit[3:])[0, 0, 0]
        return packed, tiles

    size = max(map(len, batches), default=0)
    buffers = {
        role: [
            np.empty((cube.shape[0], size, 1, n_l, n_m), dtype=arrays[role].dtype)
            for _ in range(min(2, len(batches)))
        ]
        for role in roles
    }
    counters = {role: dict.fromkeys(_COUNTERS, 0) for role in roles}
    reader_seconds = {"T_read": 0.0, "T_first_round": 0.0}

    def read(batch, index):
        start = time.perf_counter()
        channels = chunk[batch]
        frequency = slice(int(channels[0]), int(channels[-1]) + 1)
        region = {
            "frequency": frequency,
            "polarization": slice(polarization_index, polarization_index + 1),
        }
        data = {"channels": channels}
        tiles = None if compute_first_round else needed_tiles(batch)
        masks_seconds = 0.0
        for role in roles:
            if role == "sky" and compute_first_round:
                masks_start = time.perf_counter()
                data["packed"], data["tiles"] = first_round_masks(data["primary_beam"])
                masks_seconds = time.perf_counter() - masks_start
                tiles = data["tiles"][:, 0]
            data[role] = read_array_region(
                paths[role],
                region,
                None if tiles is None else read_units(tiles, channels),
                counters[role],
                buffers[role][index][:, : len(batch)],
            )[0]
        if not compute_first_round:
            data["packed"] = [
                read_array_region(
                    os.path.join(first_round_store, name), {"frequency": frequency}
                )[0]
                if wanted
                else None
                for name, wanted in zip(
                    _MASKS, (primary_beam_limit is not None, read_profile), strict=True
                )
            ]
        reader_seconds["T_first_round"] += masks_seconds
        reader_seconds["T_read"] += time.perf_counter() - start - masks_seconds
        return data

    partials = new_streamed_partials(requests)
    two_d_masks = [unpack_plane_masks(np.load(path), n_m) for path in two_d_mask_files]
    images = {}
    for name, path in image_paths.items():
        values, mask, _ = read_small_image(path)
        images[name] = np.where(mask, values, np.nan)
    planes = np.empty((2, 1, 1, 1, n_l, n_m), dtype=bool)
    work = {}
    seconds = {"T_read_wait": 0.0, "T_unpack": 0.0, "T_first_round": 0.0}
    rows = {
        "channels": [np.empty(0, dtype=np.int32)],
        "statistics": [np.empty((0, 2, 5))],
        "tiles": [np.empty((0, 2, *n_tiles), dtype=bool)],
    }
    bytes_written = 0

    def reduce_batch(data):
        channels, sky = data["channels"], data["sky"]
        primary_beam = data.get("primary_beam")
        written = 0
        if compute_first_round:
            start = time.perf_counter()
            written = _write_packed_masks(first_round_store, channels, *data["packed"])
            rows["channels"].append(channels)
            rows["tiles"].append(data["tiles"])
            seconds["T_first_round"] += time.perf_counter() - start
        for p in range(channels.size):
            start = time.perf_counter()
            inside = profile = None
            if data["packed"][0] is not None:
                inside = planes[0]
                unpack_plane_masks(data["packed"][0][p], n_m, out=inside[0, 0, 0])
            if read_profile:
                profile = planes[1]
                unpack_plane_masks(data["packed"][1][p], n_m, out=profile[0, 0, 0])
            seconds["T_unpack"] += time.perf_counter() - start
            if compute_first_round:
                start = time.perf_counter()
                rows["statistics"].append(
                    first_round_statistics(
                        sky[:, p : p + 1], primary_beam[:, p : p + 1], inside
                    )
                )
                seconds["T_first_round"] += time.perf_counter() - start
            find_continuum_batch_reductions(
                sky[:, p : p + 1],
                channels[p : p + 1],
                requests,
                partials,
                inside=None if primary_beam_limit is None else inside,
                profile=profile,
                two_d_masks=two_d_masks,
                primary_beam=None
                if primary_beam is None
                else primary_beam[:, p : p + 1],
                images=images,
                buffers=work,
            )
        return written

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read, batches[0], 0) if batches else None
        for k in range(len(batches)):
            start = time.perf_counter()
            data = future.result()
            seconds["T_read_wait"] += time.perf_counter() - start
            if k + 1 < len(batches):
                future = executor.submit(read, batches[k + 1], (k + 1) % 2)
            bytes_written += reduce_batch(data)
    timing = {
        key: work.get(key, 0.0)
        for key in ("T_mask", "T_moments", "T_statistics", "T_profile")
    }
    timing["T_mask"] += seconds["T_unpack"]
    data = future = two_d_masks = images = planes = None
    buffers.clear()
    work.clear()

    start = time.perf_counter()
    written = {}
    for k, partial in enumerate(partials):
        if requests[k]["kind"] != "moments" or partial is None:
            continue
        slot = int(chunk_indices[0])
        for field, values in partial.state().items():
            path = os.path.join(partials_store, f"R{k}_{field}")
            array = zarr.open_array(
                path, mode="r+", config={"write_empty_chunks": True}
            )
            array[slot] = values
            bytes_written += _stored_bytes(
                array, path, (slice(slot, slot + 1), slice(0, n_l), slice(0, n_m))
            )
        written[k] = [slot]
        partials[k] = None
    T_write_partials = time.perf_counter() - start

    first_round = None
    if compute_first_round:
        first_round = {key: np.concatenate(values) for key, values in rows.items()}
        first_round["channels"] = first_round["channels"].astype(np.int32)
        first_round["write_failed"] = False
    processed = chunk[needed]
    read_bytes = {role: c["bytes"] + c["index_bytes"] for role, c in counters.items()}
    end_unixtime = time.time()
    row = {
        "family": "frequency",
        "task_id": task_id,
        "slot": None if chunk_indices is None else int(chunk_indices[0]),
        "first_channel": int(processed[0]) if processed.size else None,
        "last_channel": int(processed[-1]) if processed.size else None,
        "n_channels": int(processed.size),
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        **_execution_identity(),
        "n_planes_read": int(processed.size) * len(roles),
        "n_chunks_read": sum(c["units_read"] for c in counters.values()),
        "n_chunks_skipped": sum(c["units_skipped"] for c in counters.values()),
        "bytes_read_sky": read_bytes["sky"],
        "bytes_read_primary_beam": read_bytes.get("primary_beam", 0),
        "n_shard_opens": sum(c["shard_opens"] for c in counters.values()),
        "bytes_written": bytes_written,
        "T_read": reader_seconds["T_read"],
        "T_read_wait": seconds["T_read_wait"],
        **timing,
        "T_first_round": seconds["T_first_round"] + reader_seconds["T_first_round"],
        "T_write_partials": T_write_partials,
        "T_cpu": time.process_time() - cpu_start,
        "T_find_continuum_task": end_unixtime - start_unixtime,
    }
    return {
        "reductions": partials,
        "written": written,
        "first_round": first_round,
        "timing_node_tasks": pd.DataFrame([row]),
    }


def merge_find_continuum_streamed_reductions(input_data, input_params):
    """Reduce node task of :func:`find_continuum_streamed_reductions`.

    Parameters
    ----------
    input_data : list of dict
        Outputs of :func:`find_continuum_streamed_reductions` (or of this
        task).
    input_params : dict
        Unused.

    Returns
    -------
    dict
        ``reductions`` (every request's rows concatenated, ``None`` for
        moments), ``written`` (the union of the slots written, ``{request
        index: sorted slots}``), ``first_round`` (the rows concatenated with
        ``write_failed`` or-ed, ``None`` without), the concatenated
        ``timing_node_tasks`` (with the series of graphviper's resource
        monitor, when it ran, as columns) and ``timing_reduce_nodes``, one
        record per reduce node so far.
    """
    start_unixtime = time.time()
    reductions = list(input_data[0]["reductions"])
    for item in input_data[1:]:
        reductions = [
            _concatenate_rows(mine, theirs)
            for mine, theirs in zip(reductions, item["reductions"], strict=True)
        ]
    slots = {}
    for item in input_data:
        for index, written in item["written"].items():
            slots.setdefault(index, set()).update(written)
    first_rounds = [item["first_round"] for item in input_data]
    first_rounds = [rows for rows in first_rounds if rows is not None]
    first_round = None
    if first_rounds:
        first_round = {
            key: np.concatenate([rows[key] for rows in first_rounds])
            for key in ("channels", "statistics", "tiles")
        }
        first_round["write_failed"] = any(rows["write_failed"] for rows in first_rounds)
    combined, records = _gather_timing(input_data)
    records.append(
        dict(_reduce_record(start_unixtime, input_data, combined), family="frequency")
    )
    return {
        "reductions": reductions,
        "written": {index: sorted(written) for index, written in slots.items()},
        "first_round": first_round,
        "timing_node_tasks": combined,
        "timing_reduce_nodes": records,
    }


def merge_find_continuum_partial_maps(
    partials_store,
    results_store,
    moments_requests,
    written,
    n_slots,
    data_selection=None,
    task_id=None,
):
    """Merge the moment partials of the slots over one tile and write the maps.

    For each moments request, the partials are read slot by slot over the tile
    (a slot no task wrote is the identity) and merged in the pairs and order of
    GraphVIPER's binary tree
    (:func:`~astroviper.processing_functions.imaging.find_continuum.merge_in_tree_order`),
    so the maps equal those of a reduce over the frequency tasks bit for bit.
    The finalized maps are written to ``R<r>_MOMENT_<m>`` of the results
    store.

    Parameters
    ----------
    partials_store : str
        Scratch store written by :func:`find_continuum_streamed_reductions`.
    results_store : str
        Store with a float64 array ``R<r>_MOMENT_<m>`` ``(l, m)`` per moments
        request ``r`` and moment ``m``, chunked by tiles.
    moments_requests : list of tuple
        ``(request index, moments, value dtype)`` of every moments request.
    written : dict
        Request index to the slots that hold its partials.
    n_slots : int
        Slots of the scratch store.
    data_selection : dict, optional
        ``{"img": {"l": slice, "m": slice}}``, the tile of this task; the whole
        map if not given.
    task_id : int, optional
        Graph node id.

    Returns
    -------
    dict
        ``timing_node_tasks``, a one-row ``pandas.DataFrame``.

    Raises
    ------
    FileNotFoundError
        If the shard of a slot listed in ``written`` is absent.
    """
    import pandas as pd
    import zarr

    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region
    from astroviper.processing_functions.imaging.find_continuum import (
        FindContinuumMoments,
        merge_in_tree_order,
    )

    start_unixtime, cpu_start = time.time(), time.process_time()
    selection = (data_selection or {"img": {}})["img"]
    counters = dict.fromkeys(_COUNTERS, 0)
    seconds = dict.fromkeys(("T_read_partials", "T_merge", "T_write_results"), 0.0)

    def merge(index, moments):
        slots = set(written.get(index, ()))
        # Merges and finalize use no velocity widths.
        fields = FindContinuumMoments(moments, (), (0, 0))._fields()
        paths = {
            field: os.path.join(partials_store, f"R{index}_{field}") for field in fields
        }
        arrays = {
            field: zarr.open_array(path, mode="r") for field, path in paths.items()
        }
        n_l, n_m = arrays[fields[0]].shape[1:]
        rows = slice(*selection.get("l", slice(None)).indices(n_l)[:2])
        columns = slice(*selection.get("m", slice(None)).indices(n_m)[:2])
        shape = (rows.stop - rows.start, columns.stop - columns.start)

        def leaves():
            for slot in range(n_slots):
                if slot not in slots:
                    yield None
                    continue
                start = time.perf_counter()
                state = {}
                for field, path in paths.items():
                    key = arrays[field].metadata.encode_chunk_key((slot, 0, 0))
                    if not os.path.exists(os.path.join(path, key)):
                        raise FileNotFoundError(
                            f"Slot {slot} of request {index} is listed as written, "
                            f"but {os.path.join(path, key)} is absent"
                        )
                    state[field] = read_array_region(
                        path,
                        {"slot": slice(slot, slot + 1), "l": rows, "m": columns},
                        counters=counters,
                    )[0][0]
                seconds["T_read_partials"] += time.perf_counter() - start
                yield FindContinuumMoments.from_state(moments, (), shape, state)

        start, read_before = time.perf_counter(), seconds["T_read_partials"]
        merged = merge_in_tree_order(leaves())
        if merged is None:
            merged = FindContinuumMoments(moments, (), shape)
        maps = merged.finalize()
        seconds["T_merge"] += (
            time.perf_counter() - start - (seconds["T_read_partials"] - read_before)
        )
        start, bytes_written = time.perf_counter(), 0
        for moment, values in maps.items():
            path = os.path.join(results_store, f"R{index}_MOMENT_{moment}")
            array = zarr.open_array(path, mode="r+")
            array[rows, columns] = values
            bytes_written += _stored_bytes(array, path, (rows, columns))
        seconds["T_write_results"] += time.perf_counter() - start
        return bytes_written

    bytes_written = sum(merge(index, moments) for index, moments, _ in moments_requests)
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


def find_continuum_first_round_chunk(
    chunk_xds,
    channels,
    first_round_store,
    variables,
    polarization_index,
    primary_beam_limit,
    tile_shape,
    value_dtypes,
    run_id,
    windows,
    partials_directory,
    rest_frequency,
):
    """findContinuum's first round of one chunk the imager has written.

    Plane by plane, at the Stokes I index and in the store dtypes: the
    primary-beam masks (written to the first-round store), the statistics of
    the cube and of the primary beam inside the limit, the tile occupancy of
    the masks and, with a ``run_id``, moments 0 and 8 in the registry of the
    worker process (:func:`add_first_round_moments`).

    Parameters
    ----------
    chunk_xds : xarray.Dataset
        The chunk, with the data variables of ``variables``.
    channels : slice
        Channels of the chunk in the cube.
    first_round_store : str
        Path of the first-round store.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the cube
        and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    primary_beam_limit : float or None
        Level of the primary-beam limit mask; ``None``: every pixel is inside.
    tile_shape : tuple of int
        ``(l, m)`` tiles of the occupancy: the inner chunks of the cube.
    value_dtypes : dict
        ``{"sky": ..., "primary_beam": ...}``, the dtypes of the store.
    run_id : str or None
        Imaging run of the moments registry; ``None``: no moments.
    windows : list of dict
        Spectral windows of the cube (see :func:`add_first_round_moments`).
    partials_directory : str
        Directory of the registry's slots.
    rest_frequency : float
        Rest frequency of the cube (Hz), with which the windows' velocity
        widths were computed.

    Returns
    -------
    dict
        ``channels`` (int32, ``(n,)``), ``statistics`` (float64, ``(n, 2,
        5)``: cube and primary beam x npts, sum, sumsq, min, max) and
        ``tiles`` (bool, ``(n, 2, n_tiles_l, n_tiles_m)``: limit and spectrum
        masks).
    """
    from astroviper.processing_functions.imaging.find_continuum import (
        first_round_statistics,
        primary_beam_masks,
        tile_occupancy,
    )

    stokes_i = slice(polarization_index, polarization_index + 1)
    sky, primary_beam = (
        chunk_xds[variables[role]]
        .isel(polarization=stokes_i)
        .transpose(*_IMAGE_DIMS)
        .values.astype(value_dtypes[role], copy=False)
        for role in ("sky", "primary_beam")
    )
    channel_indices = np.arange(channels.start, channels.stop)
    statistics, tiles = [], []
    for k, channel in enumerate(channel_indices):
        plane = (slice(None), slice(k, k + 1))
        inside, profile = primary_beam_masks(primary_beam[plane], primary_beam_limit)
        write_primary_beam_masks(first_round_store, [channel], inside, profile)
        statistics.append(
            first_round_statistics(sky[plane], primary_beam[plane], inside)
        )
        tiles.append(
            [tile_occupancy(mask, tile_shape)[0, 0, 0] for mask in (inside, profile)]
        )
        if run_id is not None:
            add_first_round_moments(
                run_id,
                windows,
                sky.shape[-2:],
                partials_directory,
                [channel],
                sky[plane],
                inside,
            )
    return {
        "channels": channel_indices.astype(np.int32),
        "statistics": np.concatenate(statistics),
        "tiles": np.array(tiles, dtype=bool),
    }


def write_primary_beam_masks(first_round_store, channels, inside, profile):
    """Write the primary-beam masks of planes, bit-packed, to a first-round store.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    channels : sequence of int
        Channel of every plane in the cube.
    inside, profile : numpy.ndarray
        Boolean ``(time, frequency, polarization, l, m)`` primary-beam limit
        and spectrum masks, one time and polarization.

    Returns
    -------
    int
        Bytes written.
    """
    from astroviper.processing_functions.imaging.find_continuum import (
        pack_plane_masks,
    )

    return _write_packed_masks(
        first_round_store,
        channels,
        pack_plane_masks(inside[0, :, 0]),
        pack_plane_masks(profile[0, :, 0]),
    )


def write_find_continuum_first_round_rows(first_round_store, rows, n_channels):
    """Write the per-channel rows of a first round to its store.

    The tile occupancy of the two masks goes to ``PRIMARY_BEAM_LIMIT_TILES``
    and ``PRIMARY_BEAM_PROFILE_TILES`` and the statistics to ``STATISTICS``, at
    the channels of the rows. The first round is complete when no chunk failed
    to be written and the rows hold every channel of the store's windows once
    and no other channel; ``complete`` is set in the store's attributes.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    rows : dict
        ``channels``, ``statistics``, ``tiles`` and ``write_failed``: the rows
        of every chunk (:func:`find_continuum_first_round_chunk`,
        :func:`find_continuum_streamed_reductions`) concatenated.
    n_channels : int
        Channels of the cube.

    Returns
    -------
    dict
        ``complete`` (bool) and ``written_channels`` (sorted list of the
        channels with a row).
    """
    import zarr

    group = zarr.open_group(first_round_store, mode="r+")
    attributes = dict(group.attrs["find_continuum_first_round"])
    channels = np.asarray(rows["channels"], dtype=np.int64)
    expected = np.zeros(n_channels, dtype=np.int64)
    for start, stop in (window["channel_range"] for window in attributes["windows"]):
        expected[start:stop] = 1
    complete = bool(
        not rows["write_failed"]
        and np.array_equal(np.bincount(channels, minlength=n_channels), expected)
    )
    written, first = np.unique(channels, return_index=True)
    if written.size:
        for index, name in enumerate(
            ("PRIMARY_BEAM_LIMIT_TILES", "PRIMARY_BEAM_PROFILE_TILES")
        ):
            group[name].oindex[written] = rows["tiles"][first, index]
        group["STATISTICS"].oindex[written] = rows["statistics"][first]
    group.attrs["find_continuum_first_round"] = dict(attributes, complete=complete)
    return {"complete": complete, "written_channels": written.tolist()}


def add_first_round_moments(
    run_id, windows, map_shape, partials_directory, channels, sky, inside
):
    """Add planes of the imager's first round to the moments 0 and 8 of this process.

    Each channel goes to the accumulator of its spectral window, where the
    plane is finite, nonzero and inside the primary-beam limit (findContinuum's
    ``"cube"!=0`` on the blanked cube), a block of rows at a time. At most
    ``FIRST_ROUND_WINDOWS_IN_MEMORY`` windows are held: before another is
    allocated, the least recently used one is saved to a slot of
    ``partials_directory`` and dropped. A new ``run_id`` drops the entries of
    earlier runs. Threads of a process add one at a time.

    Parameters
    ----------
    run_id : str
        Imaging run.
    windows : list of dict
        Spectral windows of the cube, in channel order: ``channel_range``
        (``[start, stop)``) and ``velocity_widths`` (float64, one per channel
        of the window).
    map_shape : tuple of int
        ``(l, m)``.
    partials_directory : str
        Directory of the slots, saved as
        ``W<window>/<pid>-<sequence>.<FIELD>.npy``.
    channels : sequence of int
        Channel of every plane in the cube.
    sky : numpy.ndarray
        ``(time, frequency, polarization, l, m)`` planes in the dtype of the
        store, one time and polarization.
    inside : numpy.ndarray
        Boolean primary-beam limit mask, shaped as ``sky``.
    """
    from astroviper.processing_functions.imaging.find_continuum import (
        _BLOCK_ROWS,
        FindContinuumMoments,
    )

    starts = [window["channel_range"][0] for window in windows]
    scratch = np.empty(_BLOCK_ROWS * map_shape[1])
    with _FIRST_ROUND_LOCK:
        entry = _FIRST_ROUND_MOMENTS.get(run_id)
        if entry is None:
            _FIRST_ROUND_MOMENTS.clear()
            entry = _FIRST_ROUND_MOMENTS[run_id] = {
                "counts": np.zeros(windows[-1]["channel_range"][1], dtype=np.uint16),
                "moments": {},
                "recent": [],
                "slots": [],
                "sequence": 0,
                "slot_bytes": 0,
            }
        for position, channel in enumerate(channels):
            window = bisect.bisect_right(starts, channel) - 1
            moments = entry["moments"].get(window)
            if moments is None:
                while len(entry["moments"]) >= FIRST_ROUND_WINDOWS_IN_MEMORY:
                    _flush_first_round_window(
                        entry, entry["recent"][0], partials_directory
                    )
                moments = entry["moments"][window] = FindContinuumMoments(
                    [0, 8], windows[window]["velocity_widths"], map_shape
                )
            else:
                entry["recent"].remove(window)
            entry["recent"].append(window)
            plane, plane_inside = sky[0, position, 0], inside[0, position, 0]
            for start in range(0, map_shape[0], _BLOCK_ROWS):
                rows = slice(start, start + _BLOCK_ROWS)
                block = plane[rows]
                valid = np.isfinite(block)
                valid &= plane_inside[rows]
                valid &= block != 0
                moments.add_plane(
                    channel - starts[window], block, valid, rows=rows, scratch=scratch
                )
            entry["counts"][channel] += 1


def pop_first_round_moments(run_id, address, partials_directory):
    """Take the first-round moments of a run off the registry of this process.

    The windows still held are saved to slots first, so all of the process's
    moments are in ``partials_directory``.

    Parameters
    ----------
    run_id : str
        Imaging run.
    address : str or None
        Address of the Dask worker the pop was sent to; ``None`` in a process
        that is not a worker (the driver).
    partials_directory : str
        Directory of the slots.

    Returns
    -------
    dict
        ``address``, ``placement_ok`` (the pop ran on ``address``), ``counts``
        (uint16 additions per channel; ``None`` when the process holds no
        entry of the run), ``slots`` (``[[window, slot name], ...]`` in the
        order saved), ``slot_bytes``, ``start_unixtime``, ``end_unixtime``,
        ``T_reduce_node`` and where the pop ran (``hostname``,
        ``process_pid``, ``thread_native_id``, ``worker_name``).
    """
    start_unixtime = time.time()
    try:
        from dask.distributed import get_worker

        current = get_worker().address
    except Exception:
        current = None
    with _FIRST_ROUND_LOCK:
        entry = _FIRST_ROUND_MOMENTS.pop(run_id, None)
        if entry is not None:
            while entry["recent"]:
                _flush_first_round_window(entry, entry["recent"][0], partials_directory)
    end_unixtime = time.time()
    return {
        "address": address,
        "placement_ok": current == address,
        "counts": None if entry is None else entry["counts"],
        "slots": [] if entry is None else entry["slots"],
        "slot_bytes": 0 if entry is None else entry["slot_bytes"],
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        "T_reduce_node": end_unixtime - start_unixtime,
        **_execution_identity(),
    }


def merge_first_round_moment_tile(
    first_round_store, partials_directory, window, slots, tile
):
    """Merge the slots of one spectral window over one tile and write moments 0 and 8.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    partials_directory : str
        Directory of the slots.
    window : int
        Index of the window along the first axis of ``MOMENT_0`` and
        ``MOMENT_8``.
    slots : list of str
        Slot names of the window (:func:`pop_first_round_moments`), in merge
        order.
    tile : sequence
        ``[[l0, l1], [m0, m1]]``.

    Returns
    -------
    dict
        Timing record: ``stage`` ("first_round_merge"), ``window``,
        ``bytes_read``, ``start_unixtime``, ``end_unixtime``,
        ``T_reduce_node`` and where the task ran.
    """
    import zarr

    from astroviper.processing_functions.imaging.find_continuum import (
        FindContinuumMoments,
        merge_in_tree_order,
    )

    start_unixtime = time.time()
    (l0, l1), (m0, m1) = tile
    shape = (l1 - l0, m1 - m0)
    directory = os.path.join(partials_directory, f"W{window}")
    bytes_read = 0

    def leaves():
        nonlocal bytes_read
        for name in slots:
            state = {}
            for field in ("INTEGRATED", "MAXIMUM"):
                stored = np.load(
                    os.path.join(directory, f"{name}.{field}.npy"), mmap_mode="r"
                )
                state[field] = np.array(stored[l0:l1, m0:m1])
                bytes_read += state[field].nbytes
            # Merges and finalize use no velocity widths.
            yield FindContinuumMoments.from_state([0, 8], (), shape, state)

    merged = merge_in_tree_order(leaves())
    if merged is None:
        merged = FindContinuumMoments([0, 8], (), shape)
    for moment, values in merged.finalize().items():
        array = zarr.open_array(
            os.path.join(first_round_store, f"MOMENT_{moment}"), mode="r+"
        )
        array[window, l0:l1, m0:m1] = values.astype(array.dtype)
    end_unixtime = time.time()
    return {
        "stage": "first_round_merge",
        "window": window,
        "bytes_read": bytes_read,
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        "T_reduce_node": end_unixtime - start_unixtime,
        **_execution_identity(),
    }


def primary_beam_channel_mean(
    input_image_store,
    primary_beam,
    polarization_index,
    channels,
    primary_beam_limit,
    counters=None,
):
    """Mean primary beam over a few channels, as findContinuum's first round asks for it.

    Moment -1 of the primary beam where it is finite and not below the limit,
    the channels added in channel order, each read alone.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store.
    primary_beam : str
        Data variable of the primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    channels : sequence of int
        Channels in the cube.
    primary_beam_limit : float or None
        Level of the limit; ``None``: no limit.
    counters : dict, optional
        Accumulates the read counters of ``read_array_region``.

    Returns
    -------
    numpy.ndarray
        float64 ``(l, m)``, NaN where no channel is valid.
    """
    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region
    from astroviper.processing_functions.imaging.find_continuum import (
        FindContinuumMoments,
        primary_beam_masks,
    )

    path = os.path.join(input_image_store, primary_beam)
    channels = np.unique(channels)
    mean = None
    for k, channel in enumerate(channels.tolist()):
        plane = read_array_region(
            path,
            {
                "frequency": slice(channel, channel + 1),
                "polarization": slice(polarization_index, polarization_index + 1),
            },
            counters=counters,
        )[0]
        if mean is None:
            mean = FindContinuumMoments([-1], np.ones(channels.size), plane.shape[-2:])
        valid = primary_beam_masks(plane, primary_beam_limit)[0]
        valid &= np.isfinite(plane)
        mean.add_plane(k, plane[0, 0, 0], valid[0, 0, 0])
    return mean.finalize()[-1]


def _execution_identity():
    """Where the calling task runs: host, process, thread and Dask worker."""
    import socket

    try:
        from dask.distributed import get_worker

        worker_name = str(get_worker().name)
    except Exception:
        worker_name = None
    return {
        "hostname": socket.gethostname(),
        "process_pid": os.getpid(),
        "thread_native_id": threading.get_native_id(),
        "worker_name": worker_name,
    }


def _gather_timing(input_data):
    """The timing rows of the inputs of a reduce node (with the series of
    graphviper's resource monitor folded in) and their reduce records."""
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
    end_unixtime = time.time()
    return {
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        "T_reduce_node": end_unixtime - start_unixtime,
        **_execution_identity(),
        "n_inputs": len(input_data),
        "n_rows_out": int(len(combined)),
    }


def _concatenate_rows(mine, theirs):
    """The per-channel rows of two tasks; ``None`` (moments) stays ``None``."""
    if mine is None:
        return None
    merged = {"channels": np.concatenate([mine["channels"], theirs["channels"]])}
    key = "rows" if "rows" in mine else "values"
    values = mine[key]
    if isinstance(values, dict):
        merged[key] = {
            name: np.concatenate([column, theirs[key][name]])
            for name, column in values.items()
        }
    else:
        merged[key] = np.concatenate([values, theirs[key]])
    return merged


def _write_packed_masks(first_round_store, channels, inside, profile):
    """Write packed masks ``(n, l, ceil(m / 8))`` one channel (chunk) at a
    time; returns the bytes stored."""
    import zarr

    written = 0
    for name, packed in zip(_MASKS, (inside, profile), strict=True):
        path = os.path.join(first_round_store, name)
        array = zarr.open_array(path, mode="r+")
        for channel, plane in zip(channels, packed, strict=True):
            channel = int(channel)
            array[channel] = plane
            written += _stored_bytes(
                array,
                path,
                (
                    slice(channel, channel + 1),
                    slice(0, plane.shape[0]),
                    slice(0, plane.shape[1]),
                ),
            )
    return written


def _stored_bytes(array, path, region):
    """Bytes on disk of the chunks (shards) of a Zarr array that overlap
    ``region``, a slice with start and stop per dimension."""
    grid = array.shards or array.chunks
    indices = itertools.product(
        *(
            range(part.start // size, (part.stop - 1) // size + 1)
            for part, size in zip(region, grid, strict=True)
        )
    )
    total = 0
    for index in indices:
        with contextlib.suppress(FileNotFoundError):
            total += os.path.getsize(
                os.path.join(path, array.metadata.encode_chunk_key(index))
            )
    return total


def _flush_first_round_window(entry, window, partials_directory):
    """Save the moments of a window of a registry entry to a slot and drop them."""
    moments = entry["moments"].pop(window)
    entry["recent"].remove(window)
    name = f"{os.getpid()}-{entry['sequence']}"
    entry["sequence"] += 1
    directory = os.path.join(partials_directory, f"W{window}")
    os.makedirs(directory, exist_ok=True)
    for field, values in moments.state().items():
        path = os.path.join(directory, f"{name}.{field}.npy")
        with open(f"{path}.tmp", "wb") as stream:
            np.save(stream, values)
        os.replace(f"{path}.tmp", path)
        entry["slot_bytes"] += os.path.getsize(path)
    entry["slots"].append([window, name])
