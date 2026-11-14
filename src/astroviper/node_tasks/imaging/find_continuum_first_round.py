"""Node tasks of findContinuum's first round, made by the imager.

``image_cube_single_field`` writes the primary-beam masks of every chunk it
images to the first-round store and returns their tile occupancy and the
per-channel rows (:func:`find_continuum_first_round_chunk`,
:func:`write_first_round_rows`), and keeps the integrated and maximum moments
of each spectral window in a registry of the worker process
(:func:`add_first_round_moments`), drained after the imaging compute
(:func:`pop_first_round_moments`, :func:`merge_first_round_moment_tile`).
find_continuum_ultra reads a window's rows and moments from the store
(:func:`read_first_round_rows`, :func:`read_first_round_moment_maps`).
"""

import bisect
import contextlib
import itertools
import os
import threading
import time

import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    FirstRoundRows,
)

# Windows of the imager's first round whose moments a worker process holds;
# the least recently used one goes to a slot before another is allocated.
FIRST_ROUND_WINDOWS_IN_MEMORY = 2

_IMAGE_DIMS = ["time", "frequency", "polarization", "l", "m"]
_MASKS = ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_SPECTRUM_MASK")
_TILES = ("PRIMARY_BEAM_LIMIT_TILES", "PRIMARY_BEAM_SPECTRUM_TILES")
_MOMENT_MAPS = {"integrated": "SKY_MOMENT_INTEGRATED", "maximum": "SKY_MOMENT_MAXIMUM"}

# Run id to the first-round moments of this process: "counts" (additions per
# channel), "moments" (window to its moment partials), "recent" (held windows,
# least recently used first), "slots", "lost" (windows of which a slot could
# not be saved), "token" (unique to the entry, so that slot names differ
# across hosts), "sequence" and "slot_bytes".
_FIRST_ROUND_MOMENTS = {}
# Runs whose moments were discarded: tasks still running add nothing more.
_DISCARDED_RUNS = set()
_FIRST_ROUND_LOCK = threading.Lock()


def find_continuum_first_round_chunk(
    chunk_xds: xr.Dataset,
    channels: np.ndarray,
    first_round_store: str,
    variables: dict,
    polarization_index: int,
    primary_beam_limit: float,
    tile_shape: tuple,
    value_dtypes: dict,
    run_id: str | None,
    windows: list,
    partials_directory: str,
    rest_frequency: float,
) -> dict:
    """findContinuum's first round of one chunk the imager has written.

    Plane by plane, at the Stokes I index and in the store dtypes: the
    primary-beam limit and spectrum masks (written to the first-round store),
    their tile occupancy, the per-channel rows of the residual and of the
    primary beam inside the limit and, with a ``run_id``, the moments of the
    finite, nonzero residual inside the limit in the registry of the worker
    process (:func:`add_first_round_moments`).

    Parameters
    ----------
    chunk_xds : xarray.Dataset
        The chunk, with the data variables of ``variables`` and one time.
    channels : numpy.ndarray
        Channel of every plane of the chunk in the cube.
    first_round_store : str
        Path of the first-round store.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the cube
        and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    primary_beam_limit : float
        Level of the primary-beam limit mask.
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
        Rest frequency of the cube (Hz); the windows' moment channel weights
        were computed with their own
        (:func:`~astroviper.processing_functions.imaging.utils.spectral_windows.window_rest_frequency`).

    Returns
    -------
    dict
        ``channels`` (int32, ``(n,)``), the rows of
        :func:`~astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions.first_round_channel_rows`
        (``sky_valid_count``, ``sky_minimum``, ``sky_maximum``,
        ``primary_beam_valid_count``, ``primary_beam_minimum``, each
        ``(n,)``) and ``tiles`` (bool, ``(n, 2, n_tiles_l, n_tiles_m)``:
        limit and spectrum masks).
    """
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        first_round_channel_rows,
        primary_beam_masks,
        tile_occupancy,
        valid_voxels,
    )

    stokes_i = slice(polarization_index, polarization_index + 1)
    sky, primary_beam = (
        chunk_xds[variables[role]]
        .isel(polarization=stokes_i)
        .transpose(*_IMAGE_DIMS)
        .values.astype(value_dtypes[role], copy=False)
        for role in ("sky", "primary_beam")
    )
    channels = np.asarray(channels)
    map_shape = (1, 1, 1, *sky.shape[-2:])
    rows, tiles = [], []
    for k in range(channels.size):
        plane = (slice(None), slice(k, k + 1))
        limit_mask, spectrum_mask = primary_beam_masks(
            primary_beam[plane], primary_beam_limit
        )
        write_primary_beam_masks(
            first_round_store, channels[k : k + 1], limit_mask, spectrum_mask
        )
        rows.append(
            first_round_channel_rows(sky[plane], primary_beam[plane], limit_mask)
        )
        tiles.append(
            [
                tile_occupancy(mask, tile_shape)[0, 0, 0]
                for mask in (limit_mask, spectrum_mask)
            ]
        )
        if run_id is not None:
            valid = valid_voxels(sky[plane], limit_mask)
            valid &= sky[plane] != 0
            add_first_round_moments(
                run_id,
                windows,
                map_shape,
                partials_directory,
                channels[k : k + 1],
                sky[plane],
                valid,
            )
    return {
        "channels": channels.astype(np.int32),
        **{
            field: np.array([row[field][0, 0, 0] for row in rows])
            for field in FirstRoundRows._fields
        },
        "tiles": np.array(tiles, dtype=bool),
    }


def write_primary_beam_masks(
    first_round_store: str,
    channels: np.ndarray,
    limit_mask: np.ndarray,
    spectrum_mask: np.ndarray,
) -> int:
    """Write the primary-beam masks of planes, bit-packed, to a first-round store.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    channels : numpy.ndarray
        Channel of every plane in the cube.
    limit_mask, spectrum_mask : numpy.ndarray
        Boolean ``(time, frequency, polarization, l, m)`` primary-beam limit
        and spectrum masks, one time and polarization.

    Returns
    -------
    int
        Bytes written.
    """
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        pack_plane_masks,
    )

    return _write_packed_masks(
        first_round_store,
        channels,
        pack_plane_masks(limit_mask[0, :, 0]),
        pack_plane_masks(spectrum_mask[0, :, 0]),
    )


def write_first_round_rows(first_round_store: str, rows: dict, n_channels: int) -> dict:
    """Write the per-channel rows of a first round to its store.

    The tile occupancy of the two masks goes to ``PRIMARY_BEAM_LIMIT_TILES``
    and ``PRIMARY_BEAM_SPECTRUM_TILES`` and each row to the array of its name
    in upper case (``SKY_VALID_COUNT``, ...), at the channels of the rows. The
    first round is complete when no chunk failed to be written and the rows
    hold every channel of the store's windows once and no other channel;
    ``complete`` is set in the store's attributes.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    rows : dict
        ``channels``, the rows, ``tiles`` and ``write_failed``: the rows of
        every chunk (:func:`find_continuum_first_round_chunk`) concatenated.
    n_channels : int
        Channels of the cube.

    Returns
    -------
    dict
        ``complete`` (bool) and ``written_channels`` (sorted list of the
        channels with a row).
    """
    import zarr

    from astroviper.node_tasks.imaging.utils.skunk_works import _retry_transient_io

    def write():
        group = zarr.open_group(first_round_store, mode="r+")
        attributes = dict(group.attrs["find_continuum_first_round"])
        channels = np.asarray(rows["channels"], dtype=np.int64)
        expected = np.zeros(n_channels, dtype=np.int64)
        for start, stop in (
            window["channel_range"] for window in attributes["windows"]
        ):
            expected[start:stop] = 1
        complete = bool(
            not rows["write_failed"]
            and np.array_equal(np.bincount(channels, minlength=n_channels), expected)
        )
        written, first = np.unique(channels, return_index=True)
        if written.size:
            for index, name in enumerate(_TILES):
                group[name].oindex[written] = rows["tiles"][first, index]
            for field in FirstRoundRows._fields:
                group[field.upper()].oindex[written] = rows[field][first]
        group.attrs["find_continuum_first_round"] = dict(attributes, complete=complete)
        return {"complete": complete, "written_channels": written.tolist()}

    # A transient I/O error writes the rows again on a fresh open.
    return _retry_transient_io(f"rows of {first_round_store}", write)


def add_first_round_moments(
    run_id: str,
    windows: list,
    map_shape: tuple,
    partials_directory: str,
    channels: np.ndarray,
    sky: np.ndarray,
    valid: np.ndarray,
) -> None:
    """Add planes of the imager's first round to the moments of this process.

    Each channel goes to the moment partials of its spectral window
    (:func:`~astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions.accumulate_moments`).
    At most ``FIRST_ROUND_WINDOWS_IN_MEMORY`` windows are held: before another
    is allocated, the least recently used one is saved to a slot of
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
        (``[start, stop)``) and ``moment_channel_weights`` (float64, one per
        channel of the window).
    map_shape : tuple of int
        ``(time, 1, polarization, l, m)`` of the moment partials.
    partials_directory : str
        Directory of the slots, saved as
        ``W<window>/<pid>-<token>-<sequence>.<field>.npy``, ``token`` a UUID
        of the process's entry and ``field`` ``integrated`` or ``maximum``.
    channels : numpy.ndarray
        Channel of every plane in the cube.
    sky : numpy.ndarray
        ``(time, frequency, polarization, l, m)`` planes in the dtype of the
        store.
    valid : numpy.ndarray
        Boolean voxels to add, shaped as ``sky``: finite, nonzero and inside
        the primary-beam limit (findContinuum's ``"cube"!=0`` on the blanked
        cube).
    """
    import uuid

    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        accumulate_moments,
        new_moment_partials,
    )

    starts = [window["channel_range"][0] for window in windows]
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
            partials = entry["moments"].get(window)
            if partials is None:
                while len(entry["moments"]) >= FIRST_ROUND_WINDOWS_IN_MEMORY:
                    _flush_first_round_window(
                        entry, entry["recent"][0], partials_directory
                    )
                partials = entry["moments"][window] = new_moment_partials(
                    map_shape, sky.dtype
                )
            else:
                entry["recent"].remove(window)
            entry["recent"].append(window)
            index = channel - starts[window]
            planes = slice(position, position + 1)
            accumulate_moments(
                partials,
                sky[:, planes],
                valid[:, planes],
                windows[window]["moment_channel_weights"][index : index + 1],
            )
            entry["counts"][channel] += 1


def pop_first_round_moments(
    run_id: str, address: str | None, partials_directory: str
) -> dict:
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


def discard_first_round_moments(run_id: str) -> bool:
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
    first_round_store: str,
    partials_directory: str,
    window: int,
    slots: list,
    tile: list,
) -> dict:
    """Merge the slots of one spectral window over one tile and write its
    integrated and maximum moments.

    The slots' moment partials are merged in GraphVIPER's binary-tree order
    (:func:`~astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions.merge_moment_partials_in_tree_order`)
    and finalized into ``SKY_MOMENT_INTEGRATED`` and ``SKY_MOMENT_MAXIMUM``. A
    transient I/O error repeats the whole merge of the tile.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    partials_directory : str
        Directory of the slots.
    window : int
        Index of the window along the first axis of ``SKY_MOMENT_INTEGRATED``
        and ``SKY_MOMENT_MAXIMUM``.
    slots : list of str
        Slot names of the window (:func:`pop_first_round_moments`), in merge
        order.
    tile : list
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
    from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
        finalize_moments,
        merge_moment_partials_in_tree_order,
    )

    start_unixtime = time.time()
    (l0, l1), (m0, m1) = tile
    directory = os.path.join(partials_directory, f"W{window}")

    def merge():
        bytes_read = 0

        def leaves():
            nonlocal bytes_read
            for name in slots:
                partials = {}
                for field in _MOMENT_MAPS:
                    stored = np.load(
                        os.path.join(directory, f"{name}.{field}.npy"), mmap_mode="r"
                    )
                    partials[field] = np.array(stored[..., l0:l1, m0:m1])
                    bytes_read += partials[field].nbytes
                yield partials

        arrays = {
            field: zarr.open_array(os.path.join(first_round_store, name), mode="r+")
            for field, name in _MOMENT_MAPS.items()
        }
        maps = finalize_moments(
            merge_moment_partials_in_tree_order(leaves()),
            (1, 1, 1, l1 - l0, m1 - m0),
            arrays["maximum"].dtype,
        )
        for field, values in zip(_MOMENT_MAPS, maps, strict=True):
            arrays[field][window, l0:l1, m0:m1] = values[0, 0, 0]
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


def read_first_round_rows(
    first_round_store: str, channel_range: list
) -> tuple[FirstRoundRows, dict]:
    """The per-channel rows of a spectral window, read from a first-round store.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    channel_range : list of int
        ``[start, stop)``, channels of the cube.

    Returns
    -------
    rows : FirstRoundRows
        ``(stop - start,)`` arrays in channel order: int64 counts, float64
        minima and maxima (NaN for a channel without a valid voxel).
    counters : dict
        The read counters of
        :func:`~astroviper.node_tasks.imaging.utils.skunk_works.read_array_region`.
    """
    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region

    start, stop = channel_range
    counters = {}
    rows = FirstRoundRows(
        **{
            field: read_array_region(
                os.path.join(first_round_store, field.upper()),
                {"frequency": slice(start, stop)},
                counters=counters,
            )[0]
            for field in FirstRoundRows._fields
        }
    )
    return rows, counters


def read_first_round_moment_maps(
    first_round_store: str, window_index: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """The integrated and maximum moments of one spectral window, read from a
    first-round store.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    window_index : int
        Index of the window in the store's ``windows``.

    Returns
    -------
    integral : numpy.ndarray
        float64 ``(l, m)`` ``SKY_MOMENT_INTEGRATED``, NaN where no channel
        contributed.
    maximum : numpy.ndarray
        ``(l, m)`` ``SKY_MOMENT_MAXIMUM`` in the cube's dtype, NaN where no
        channel contributed.
    counters : dict
        The read counters of
        :func:`~astroviper.node_tasks.imaging.utils.skunk_works.read_array_region`.
    """
    from astroviper.node_tasks.imaging.utils.skunk_works import read_array_region

    counters = {}
    integral, maximum = (
        read_array_region(
            os.path.join(first_round_store, name),
            {"window": slice(window_index, window_index + 1)},
            counters=counters,
        )[0][0]
        for name in _MOMENT_MAPS.values()
    )
    return integral, maximum, counters


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


def _write_packed_masks(first_round_store, channels, limit_mask, spectrum_mask):
    """Write packed masks ``(n, l, ceil(m / 8))`` one channel (chunk) at a
    time; returns the bytes stored."""
    written = 0
    for name, packed in zip(_MASKS, (limit_mask, spectrum_mask), strict=True):
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
            for field, values in entry["moments"][window].items()
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
