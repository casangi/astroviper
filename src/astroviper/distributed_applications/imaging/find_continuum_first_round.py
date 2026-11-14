"""findContinuum's first round, kept in a Zarr store.

``image_cube_single_field`` (with ``find_continuum_first_round``) makes
findContinuum's first round while it images the cube, in a store next to the
cube (:func:`find_continuum_first_round_store_path`); ``find_continuum_ultra``
makes it in its first pass, in its work directory, when it is given none.
The store (:func:`create_find_continuum_first_round_store`) holds the
bit-packed primary-beam masks of every channel, their tile occupancy, the
per-channel statistics of the cube and of the primary beam, and moments 0 and
8 of every spectral window. The imager's worker processes accumulate the
moments; :func:`drain_first_round_moments` merges them after the imaging
compute, for the windows whose every channel was added exactly once. The store
is bound to the cube it was made from
(:func:`bind_find_continuum_first_round_store`) and is used only on that cube
(:func:`validate_find_continuum_first_round`).
"""

import contextlib
import hashlib
import os
import shutil

import numpy as np
import toolviper.utils.logger as logger
import zarr

from astroviper.node_tasks.imaging.find_continuum_first_round import (
    discard_first_round_moments,
    merge_first_round_moment_tile,
    pop_first_round_moments,
)
from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
    FIND_CONTINUUM_FIRST_ROUND_VERSION,
    check_find_continuum_first_round_store,
)
from astroviper.processing_functions.imaging.find_continuum import PROFILE_LEVEL

#: Inner chunk (px) of which merge tiles are made along an axis that one chunk
#: of the cube spans.
MERGE_TILE_INNER_CHUNK = 625

_IMAGE_DIMS = ("time", "frequency", "polarization", "l", "m")
# Shortest wait for the result of a finished pop.
_POP_FETCH_SECONDS = 30.0


def find_continuum_first_round_store_path(image_store):
    """Path of the first-round store of an image store.

    Parameters
    ----------
    image_store : str
        Path of the xradio image store.

    Returns
    -------
    str
        ``<image_store without .img.zarr or .zarr>_find_continuum_first_round.zarr``.
    """
    path = os.path.normpath(image_store)
    for suffix in (".img.zarr", ".zarr"):
        if path.endswith(suffix):
            path = path.removesuffix(suffix)
            break
    return f"{path}_find_continuum_first_round.zarr"


def find_continuum_first_round_partials_path(first_round_store):
    """Directory of the slots of the imager's first-round moments.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.

    Returns
    -------
    str
        ``<first_round_store without .zarr>_moments_partials``, next to the
        store.
    """
    return (
        f"{os.path.normpath(first_round_store).removesuffix('.zarr')}_moments_partials"
    )


def find_continuum_merge_tile_counts(map_shape: tuple, tile_shape: tuple) -> list[int]:
    """Merge tiles of the moment maps along ``l`` and ``m``.

    About two inner chunks of the cube per tile, or two of
    ``MERGE_TILE_INNER_CHUNK`` pixels along an axis that one chunk spans (an
    image without ``l`` and ``m`` chunking), so that a merge task never holds
    whole planes.

    Parameters
    ----------
    map_shape : sequence of int
        ``(l, m)`` pixels of the cube.
    tile_shape : sequence of int
        ``(l, m)`` inner chunks of the cube.

    Returns
    -------
    list of int
        ``n_chunks`` of ``make_parallel_coord`` along ``l`` and ``m``.
    """
    return [
        -(-n // (2 * (chunk if chunk < n else MERGE_TILE_INNER_CHUNK)))
        for n, chunk in zip(map_shape, tile_shape, strict=True)
    ]


def create_find_continuum_first_round_store(
    first_round_store,
    n_channels,
    image_shape,
    tile_shape,
    merge_tile_shape,
    primary_beam_limit,
    variables,
    polarization_index,
    value_dtypes,
    run_id,
    written_by,
    windows,
    rest_frequency,
):
    """Create an empty first-round store (Zarr v3), replacing one of that path.

    Parameters
    ----------
    first_round_store : str
        Path of the store.
    n_channels : int
        Channels of the cube.
    image_shape : sequence of int
        Shape of the cube; its last two axes are ``l`` and ``m``.
    tile_shape : sequence of int
        ``(l, m)`` inner chunks of the cube, the tiles of the occupancy arrays.
    merge_tile_shape : sequence of int
        ``(l, m)`` chunks of ``MOMENT_0`` and ``MOMENT_8``, the tiles the
        moments are merged in.
    primary_beam_limit : float or None
        Level of the primary-beam limit mask.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the cube
        and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    value_dtypes : dict
        ``{"sky": ..., "primary_beam": ...}``, the dtypes of the cube's data
        variables; ``MOMENT_8`` has the sky dtype, in which it is exact.
    run_id : str or None
        Imaging run of the worker-local moments; ``None``: no moments.
    written_by : str
        ``"image_cube_single_field"`` or ``"find_continuum_ultra"``.
    windows : list of dict
        Spectral windows of the store in channel order: ``channel_range``
        (``[start, stop)``) and ``frequency_axis``
        (:func:`~astroviper.processing_functions.imaging.find_continuum.linear_frequency_axis`
        of the window's frequencies); other keys are not stored.
    rest_frequency : float
        Rest frequency of the cube (Hz).

    Notes
    -----
    Arrays: ``PRIMARY_BEAM_LIMIT_MASK`` and ``PRIMARY_BEAM_PROFILE_MASK``
    (uint8 ``(n_channels, l, ceil(m / 8))``, one chunk per channel),
    ``PRIMARY_BEAM_LIMIT_TILES`` and ``PRIMARY_BEAM_PROFILE_TILES`` (bool
    ``(n_channels, n_tiles_l, n_tiles_m)``), ``STATISTICS`` (float64
    ``(n_channels, 2, 5)``, NaN until written), ``MOMENT_0`` (float64) and
    ``MOMENT_8`` (``(n_windows, l, m)``, NaN until drained). Attributes
    ``find_continuum_first_round``: ``version``, ``written_by``, ``run_id``,
    ``primary_beam_limit``, ``profile_level``, ``variables``,
    ``polarization_index``, ``value_dtypes``, ``tile_shape``, ``n_channels``,
    ``windows``, ``rest_frequency``, ``complete`` and ``moments_complete`` (per
    window) false, and ``n_missing`` and ``n_duplicated`` ``None`` until
    drained.
    """
    n_l, n_m = (int(n) for n in image_shape[-2:])
    n_packed = -(-n_m // 8)
    n_tiles = (-(-n_l // tile_shape[0]), -(-n_m // tile_shape[1]))
    zstd = zarr.codecs.ZstdCodec(level=1)
    group = zarr.open_group(first_round_store, mode="w", zarr_format=3)
    for name in ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_PROFILE_MASK"):
        group.create_array(
            name,
            shape=(n_channels, n_l, n_packed),
            chunks=(1, n_l, n_packed),
            dtype=np.uint8,
            fill_value=0,
            compressors=zstd,
            dimension_names=("frequency", "l", "m_packed"),
        )
    for name in ("PRIMARY_BEAM_LIMIT_TILES", "PRIMARY_BEAM_PROFILE_TILES"):
        group.create_array(
            name,
            shape=(n_channels, *n_tiles),
            chunks=(n_channels, *n_tiles),
            dtype=bool,
            fill_value=False,
            compressors=zstd,
        )
    group.create_array(
        "STATISTICS",
        shape=(n_channels, 2, 5),
        chunks=(n_channels, 2, 5),
        dtype=np.float64,
        fill_value=np.nan,
        compressors=zstd,
    )
    for moment, dtype in ((0, np.float64), (8, value_dtypes["sky"])):
        group.create_array(
            f"MOMENT_{moment}",
            shape=(len(windows), n_l, n_m),
            chunks=(1, *merge_tile_shape),
            dtype=dtype,
            fill_value=np.nan,
            compressors=None,
        )
    group.attrs["find_continuum_first_round"] = {
        "version": FIND_CONTINUUM_FIRST_ROUND_VERSION,
        "written_by": written_by,
        "run_id": run_id,
        "primary_beam_limit": None
        if primary_beam_limit is None
        else float(primary_beam_limit),
        "profile_level": PROFILE_LEVEL,
        "variables": dict(variables),
        "polarization_index": int(polarization_index),
        "value_dtypes": {
            role: np.dtype(dtype).name for role, dtype in value_dtypes.items()
        },
        "tile_shape": [int(size) for size in tile_shape],
        "n_channels": int(n_channels),
        "windows": [
            {
                "channel_range": [int(channel) for channel in window["channel_range"]],
                "frequency_axis": [
                    *(float(value) for value in window["frequency_axis"][:3]),
                    int(window["frequency_axis"][3]),
                ],
            }
            for window in windows
        ],
        "rest_frequency": float(rest_frequency),
        "complete": False,
        "moments_complete": [False] * len(windows),
        "n_missing": None,
        "n_duplicated": None,
    }


def drain_first_round_moments(
    run_id, first_round_store, windows, timeout_seconds=600.0
):
    """Merge the imager's worker-local moments into the first-round store.

    Every process that ran imaging tasks takes its moments of ``run_id`` off
    its registry
    (:func:`~astroviper.node_tasks.imaging.find_continuum_first_round.pop_first_round_moments`):
    with a distributed client, one pop per worker, pinned to it, in sorted
    address order (a pop not finished after ``timeout_seconds``, or whose
    result cannot be fetched within 30 s after that, is cancelled, so the
    channels of its worker count as missing); otherwise the driver's own
    process. A window is complete when every pop ran where it was sent, the
    pops added each of its channels exactly once, listed each of its slots
    once and saved all of them. The slots of a complete
    window, sorted by worker address, process and sequence, are merged tile by
    tile into ``MOMENT_0`` and ``MOMENT_8``
    (:func:`~astroviper.node_tasks.imaging.find_continuum_first_round.merge_first_round_moment_tile`,
    a task each with a distributed client); a window with a tile that fails to
    merge is left incomplete. The partials directory is then removed, also
    when the drain fails, and ``moments_complete``, ``n_missing`` and
    ``n_duplicated`` (per window) are set in the store's attributes.

    Parameters
    ----------
    run_id : str
        Imaging run.
    first_round_store : str
        Path of the first-round store.
    windows : list of dict
        Spectral windows of the store, in its order, with ``channel_range``
        (``[start, stop)``).
    timeout_seconds : float, default 600.0
        Longest wait for the pops.

    Returns
    -------
    result : dict
        ``moments_complete``, ``n_missing`` and ``n_duplicated`` (lists per
        window), ``slot_bytes`` and ``n_slots``.
    records : list of dict
        Timing records: one per pop (stage ``"first_round_pop"``, the pop's
        result without its counts and slots) and one per merge task (stage
        ``"first_round_merge"``).
    """
    partials_directory = find_continuum_first_round_partials_path(first_round_store)
    client = _distributed_client()
    try:
        if client is None:
            pops = [pop_first_round_moments(run_id, None, partials_directory)]
        else:
            pops = _pop_on_workers(client, run_id, partials_directory, timeout_seconds)
        ranges = [window["channel_range"] for window in windows]
        counts = np.zeros(ranges[-1][1], dtype=np.int64)
        slots = [[] for _ in windows]
        lost = set()
        for pop in pops:
            if pop["counts"] is not None:
                counts += pop["counts"]
            for window, name in pop["slots"]:
                pid, _, sequence = name.split("-")
                slots[window].append(
                    (pop["address"] or "", int(pid), int(sequence), name)
                )
            lost.update(pop["lost_windows"])
        placement_ok = all(pop["placement_ok"] for pop in pops)
        n_missing = [int(np.sum(counts[start:stop] == 0)) for start, stop in ranges]
        n_duplicated = [int(np.sum(counts[start:stop] > 1)) for start, stop in ranges]
        names = [[name for *_, name in sorted(window_slots)] for window_slots in slots]
        reasons = [[] for _ in windows]
        for window, window_names in enumerate(names):
            if n_missing[window]:
                reasons[window].append(f"{n_missing[window]} channels missing")
            if n_duplicated[window]:
                reasons[window].append(
                    f"{n_duplicated[window]} channels added twice or more"
                )
            if not placement_ok:
                reasons[window].append("a pop ran on another worker")
            # Two pops listing one slot would merge one file twice.
            if len(set(window_names)) != len(window_names):
                reasons[window].append("a slot listed twice")
            if window in lost:
                reasons[window].append("a slot could not be saved")
        records = [
            {
                "stage": "first_round_pop",
                "n_slots": len(pop["slots"]),
                **{
                    key: value
                    for key, value in pop.items()
                    if key not in ("counts", "slots")
                },
            }
            for pop in pops
        ]
        group = zarr.open_group(first_round_store, mode="r+")
        tiles = _merge_tiles(group["MOMENT_0"])
        tasks = [
            (first_round_store, partials_directory, window, names[window], tile)
            for window, window_reasons in enumerate(reasons)
            if not window_reasons
            for tile in tiles
        ]
        if client is None:
            merges = [
                (task, _call(merge_first_round_moment_tile, *task)) for task in tasks
            ]
        else:
            merges = [
                (task, client.submit(merge_first_round_moment_tile, *task, pure=False))
                for task in tasks
            ]
        for task, merge in merges:
            try:
                records.append(merge.result())
            except Exception as exc:
                if not reasons[task[2]]:
                    reasons[task[2]].append(f"a tile failed to merge ({exc!r})")
    finally:
        # A cancelled pop may still be writing slots.
        shutil.rmtree(partials_directory, ignore_errors=True)
    moments_complete = [not window_reasons for window_reasons in reasons]
    group.attrs["find_continuum_first_round"] = dict(
        group.attrs["find_continuum_first_round"],
        moments_complete=moments_complete,
        n_missing=n_missing,
        n_duplicated=n_duplicated,
    )
    for window, window_reasons in enumerate(reasons):
        if window_reasons:
            logger.warning(
                f"The imager's moments of channels {ranges[window]} are incomplete "
                f"({', '.join(window_reasons)}): find_continuum computes them."
            )
    result = {
        "moments_complete": moments_complete,
        "n_missing": n_missing,
        "n_duplicated": n_duplicated,
        "slot_bytes": int(sum(pop["slot_bytes"] for pop in pops)),
        "n_slots": sum(len(pop["slots"]) for pop in pops),
    }
    return result, records


def clear_first_round_moments(run_id, first_round_store):
    """Drop the imager's worker-local moments of a run without merging them,
    and remove the directory of their slots (after a failed imaging compute
    or drain).

    Every worker process of the distributed client (otherwise this process)
    discards its moments of the run
    (:func:`~astroviper.node_tasks.imaging.find_continuum_first_round.discard_first_round_moments`).
    An error is logged, not raised.

    Parameters
    ----------
    run_id : str or None
        Imaging run; ``None``: no moments (only the directory is removed).
    first_round_store : str
        Path of the first-round store.
    """
    try:
        if run_id is not None:
            client = _distributed_client()
            if client is None:
                discard_first_round_moments(run_id)
            else:
                client.run(discard_first_round_moments, run_id, on_error="ignore")
    except Exception as exc:
        logger.error(f"The imager's find_continuum moments were not cleared: {exc!r}")
    finally:
        shutil.rmtree(
            find_continuum_first_round_partials_path(first_round_store),
            ignore_errors=True,
        )


def bind_find_continuum_first_round_store(first_round_store, input_image_store):
    """Bind a first-round store to its cube as the cube is now.

    Sets ``binding`` in the store's attributes: the modification time of the
    image store's ``zarr.json`` (ns), the sha1 digests of the float64
    ``frequency``, ``l`` and ``m`` coordinates, and the image shape. Call it
    after the cube's metadata is consolidated.

    Parameters
    ----------
    first_round_store : str
        Path of the first-round store.
    input_image_store : str
        Path of the xradio image store it was made from.
    """
    group = zarr.open_group(first_round_store, mode="r+")
    group.attrs["find_continuum_first_round"] = dict(
        group.attrs["find_continuum_first_round"],
        binding=_image_binding(input_image_store),
    )


def validate_find_continuum_first_round(
    first_round_reductions, input_image_store, primary_beam_limit
):
    """Require a complete first-round store made from the cube as it is now.

    Reads the store's attributes and the cube's metadata only.

    Parameters
    ----------
    first_round_reductions : dict
        ``find_continuum_first_round`` of ``image_cube_single_field``: the path
        of the first-round store as ``store``.
    input_image_store : str
        Path of the xradio image store of the cube.
    primary_beam_limit : float or None
        Primary beam limit of the find_continuum call.

    Raises
    ------
    ValueError
        The reason the store cannot be used
        (:func:`~astroviper.processing_functions.imaging.config_check.param_check_find_continuum.check_find_continuum_first_round_store`).
    """
    try:
        attributes = dict(
            zarr.open_group(first_round_reductions["store"], mode="r").attrs
        )
    except FileNotFoundError:
        attributes = None
    first_round = (attributes or {}).get("find_continuum_first_round")
    cube = zarr.open_group(input_image_store, mode="r")
    check_find_continuum_first_round_store(
        attributes,
        _image_binding(input_image_store),
        primary_beam_limit,
        cube["polarization"][...].tolist().index("I"),
        ()
        if first_round is None
        else cube[first_round["variables"]["sky"]].chunks[-2:],
    )


def _call(function, *args):
    """A call made in this process, as a finished future."""
    import concurrent.futures

    future = concurrent.futures.Future()
    try:
        future.set_result(function(*args))
    except Exception as exc:
        future.set_exception(exc)
    return future


def _distributed_client():
    """The distributed client that ``dask.compute`` uses, or ``None``."""
    from dask.base import get_scheduler
    from distributed import Client

    client = getattr(get_scheduler(), "__self__", None)
    return client if isinstance(client, Client) else None


def _pop_on_workers(client, run_id, partials_directory, timeout_seconds):
    """The pops of every worker that finished and could be fetched; the
    others are cancelled."""
    import time

    import distributed

    deadline = time.time() + timeout_seconds
    futures = [
        client.submit(
            pop_first_round_moments,
            run_id,
            address,
            partials_directory,
            workers=[address],
            allow_other_workers=False,
            pure=False,
        )
        for address in sorted(client.nthreads())
    ]
    with contextlib.suppress(TimeoutError):
        distributed.wait(futures, timeout=timeout_seconds)
    pops, unfinished = [], []
    for future in futures:
        # A pop is pinned to its worker: the result of a worker lost after
        # the pop finished can never be fetched again.
        try:
            if future.status != "finished":
                raise TimeoutError
            pops.append(
                future.result(timeout=max(_POP_FETCH_SECONDS, deadline - time.time()))
            )
        except Exception:
            unfinished.append(future)
    if unfinished:
        client.cancel(unfinished)
        logger.warning(
            f"The pops of the imager's moments on {len(unfinished)} of "
            f"{len(futures)} workers did not finish: the channels of those "
            "workers count as missing."
        )
    return pops


def _merge_tiles(moments):
    """``[[l0, l1], [m0, m1]]`` of every chunk of a ``MOMENT`` array."""
    _, n_l, n_m = moments.shape
    _, tile_l, tile_m = moments.chunks
    return [
        [[l0, min(l0 + tile_l, n_l)], [m0, min(m0 + tile_m, n_m)]]
        for l0 in range(0, n_l, tile_l)
        for m0 in range(0, n_m, tile_m)
    ]


def _image_binding(input_image_store):
    """The binding of an image store: ``zarr.json`` mtime, coordinate digests
    and shape."""
    cube = zarr.open_group(input_image_store, mode="r")
    return {
        "image_store_zarr_json_mtime_ns": os.stat(
            os.path.join(input_image_store, "zarr.json")
        ).st_mtime_ns,
        **{
            f"{name}_sha1": hashlib.sha1(
                np.ascontiguousarray(cube[name][...], dtype=np.float64)
            ).hexdigest()
            for name in ("frequency", "l", "m")
        },
        "shape": [cube[name].shape[0] for name in _IMAGE_DIMS],
    }
