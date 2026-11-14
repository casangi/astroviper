"""Node tasks of findContinuum's first round, made by the imager.

``image_cube_single_field`` writes the primary-beam masks and statistics of
every chunk it images to the first-round store
(:func:`find_continuum_first_round_chunk`) and keeps moments 0 and 8 in a
registry of the worker process (:func:`add_first_round_moments`), drained after
the imaging compute (:func:`pop_first_round_moments`,
:func:`merge_first_round_moment_tile`).
"""

import bisect
import contextlib
import itertools
import os
import threading
import time

import numpy as np

# Windows of the imager's first round whose moments a worker process holds;
# the least recently used one goes to a slot before another is allocated.
FIRST_ROUND_WINDOWS_IN_MEMORY = 2

_IMAGE_DIMS = ["time", "frequency", "polarization", "l", "m"]
_MASKS = ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_PROFILE_MASK")

# Run id to the first-round moments of this process: "counts" (additions per
# channel), "moments" (window to its FindContinuumMoments), "recent" (held
# windows, least recently used first), "slots", "lost" (windows of which a
# slot could not be saved), "token" (unique to the entry, so that slot names
# differ across hosts), "sequence" and "slot_bytes".
_FIRST_ROUND_MOMENTS = {}
# Runs whose moments were discarded: tasks still running add nothing more.
_DISCARDED_RUNS = set()
_FIRST_ROUND_LOCK = threading.Lock()


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
        of every chunk (:func:`find_continuum_first_round_chunk`)
        concatenated.
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
    ``partials_directory`` and dropped (a window that cannot be saved, after
    the retries of transient I/O errors, is recorded as lost). A new
    ``run_id`` drops the entries of earlier runs. Threads of a process add one
    at a time.

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
        ``W<window>/<pid>-<token>-<sequence>.<FIELD>.npy``, ``token`` a UUID
        of the process's entry.
    channels : sequence of int
        Channel of every plane in the cube.
    sky : numpy.ndarray
        ``(time, frequency, polarization, l, m)`` planes in the dtype of the
        store, one time and polarization.
    inside : numpy.ndarray
        Boolean primary-beam limit mask, shaped as ``sky``.
    """
    import uuid

    from astroviper.processing_functions.imaging.find_continuum import (
        _BLOCK_ROWS,
        FindContinuumMoments,
    )

    starts = [window["channel_range"][0] for window in windows]
    scratch = np.empty(_BLOCK_ROWS * map_shape[1])
    with _FIRST_ROUND_LOCK:
        if run_id in _DISCARDED_RUNS:
            return
        entry = _FIRST_ROUND_MOMENTS.get(run_id)
        if entry is None:
            _FIRST_ROUND_MOMENTS.clear()
            entry = _FIRST_ROUND_MOMENTS[run_id] = {
                "counts": np.zeros(windows[-1]["channel_range"][1], dtype=np.uint16),
                "moments": {},
                "recent": [],
                "slots": [],
                "lost": set(),
                "token": uuid.uuid4().hex,
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
        order saved), ``lost_windows`` (sorted windows of which a slot could
        not be saved), ``slot_bytes``, ``start_unixtime``, ``end_unixtime``,
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
        "lost_windows": [] if entry is None else sorted(entry["lost"]),
        "slot_bytes": 0 if entry is None else entry["slot_bytes"],
        "start_unixtime": start_unixtime,
        "end_unixtime": end_unixtime,
        "T_reduce_node": end_unixtime - start_unixtime,
        **_execution_identity(),
    }


def discard_first_round_moments(run_id):
    """Drop the first-round moments of a run from the registry of this
    process without saving them, and ignore its later additions (tasks of a
    failed imaging compute may still run).

    Parameters
    ----------
    run_id : str
        Imaging run.

    Returns
    -------
    bool
        Whether the process held moments of the run.
    """
    with _FIRST_ROUND_LOCK:
        _DISCARDED_RUNS.add(run_id)
        return _FIRST_ROUND_MOMENTS.pop(run_id, None) is not None


def merge_first_round_moment_tile(
    first_round_store, partials_directory, window, slots, tile
):
    """Merge the slots of one spectral window over one tile and write moments 0 and 8.

    A transient I/O error repeats the whole merge of the tile.

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

    from astroviper.node_tasks.imaging.utils.skunk_works import _retry_transient_io
    from astroviper.processing_functions.imaging.find_continuum import (
        FindContinuumMoments,
        merge_in_tree_order,
    )

    start_unixtime = time.time()
    (l0, l1), (m0, m1) = tile
    shape = (l1 - l0, m1 - m0)
    directory = os.path.join(partials_directory, f"W{window}")

    def merge():
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
        return bytes_read

    bytes_read = _retry_transient_io(f"merge of window {window} tile {tile}", merge)
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


def _write_packed_masks(first_round_store, channels, inside, profile):
    """Write packed masks ``(n, l, ceil(m / 8))`` one channel (chunk) at a
    time; returns the bytes stored."""
    written = 0
    for name, packed in zip(_MASKS, (inside, profile), strict=True):
        for channel, plane in zip(channels, packed, strict=True):
            channel = int(channel)
            written += _write_array_region(
                os.path.join(first_round_store, name),
                (
                    slice(channel, channel + 1),
                    slice(0, plane.shape[0]),
                    slice(0, plane.shape[1]),
                ),
                plane[None],
            )
    return written


def _write_array_region(path, region, values, config=None):
    """Write ``values`` to ``region`` (a slice with start and stop per
    dimension) of a Zarr array, repeated on a fresh open after a transient
    I/O error; returns the bytes stored in the chunks (shards) it overlaps."""
    import zarr

    from astroviper.node_tasks.imaging.utils.skunk_works import _retry_transient_io

    def write():
        array = zarr.open_array(path, mode="r+", config=config)
        array[region] = values
        return _stored_bytes(array, path, region)

    return _retry_transient_io(f"write of {path}", write)


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
    """Save the moments of a window of a registry entry to a slot and drop
    them; a window whose slot cannot be saved is recorded as lost."""
    import toolviper.utils.logger as logger

    from astroviper.node_tasks.imaging.utils.skunk_works import _retry_transient_io

    name = f"{os.getpid()}-{entry['token']}-{entry['sequence']}"
    entry["sequence"] += 1
    directory = os.path.join(partials_directory, f"W{window}")

    def save(field, values):
        path = os.path.join(directory, f"{name}.{field}.npy")
        with open(f"{path}.tmp", "wb") as stream:
            np.save(stream, values)
        os.replace(f"{path}.tmp", path)
        return os.path.getsize(path)

    try:
        os.makedirs(directory, exist_ok=True)
        slot_bytes = sum(
            _retry_transient_io(
                f"slot {name} of window {window}",
                lambda field=field, values=values: save(field, values),
            )
            for field, values in entry["moments"][window].state().items()
        )
    except OSError as exc:
        logger.warning(
            f"The imager's moments of window {window} could not be saved to "
            f"{directory} ({exc!r}): find_continuum computes them."
        )
        entry["lost"].add(window)
    else:
        entry["slot_bytes"] += slot_bytes
        entry["slots"].append([window, name])
    del entry["moments"][window]
    entry["recent"].remove(window)
