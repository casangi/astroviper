"""Pass planner of ``find_continuum_ultra``.

:class:`UltraReductions` makes the passes over the image store that
:class:`~astroviper.distributed_applications.imaging.find_continuum.fused_reductions.FusedReductions`
asks for. The first-round reductions are answered without a pass where they can
be: the whole-cube statistics from the first-round store, the mean primary beam
in the driver and the imager's moments 0 and 8 from the store. The other
requests are encoded for the frequency tasks
(:func:`~astroviper.node_tasks.imaging.find_continuum.find_continuum_streamed_reductions`):
channels of the store, and masks split into a two-dimensional part, evaluated
here and written bit-packed, and cube terms. A pass maps the frequency tasks
over chunks of the spectral window and reduces their per-channel rows; the
tasks write their moment partials to slots of a scratch store, which a map of
tile tasks merges in GraphVIPER's tree order
(:func:`~astroviper.node_tasks.imaging.find_continuum.merge_find_continuum_partial_maps`)
into a results store, read through :class:`StoredMap`. Without a first-round
store the first pass makes the first round, into a store in the work
directory. :meth:`UltraReductions.memory_plan` keeps every task within its
memory budget.
"""

import dataclasses
import hashlib
import os
import re
import shutil
import time

import numpy as np
import toolviper.utils.logger as logger

from astroviper.distributed_applications.imaging.find_continuum import (
    casa_image_tasks,
)
from astroviper.node_tasks.imaging.find_continuum import (
    find_continuum_streamed_reductions,
    merge_find_continuum_partial_maps,
    merge_find_continuum_streamed_reductions,
    primary_beam_channel_mean,
    with_resource_usage,
    write_find_continuum_first_round_rows,
)
from astroviper.processing_functions.imaging.find_continuum import (
    _BLOCK_ROWS,
    PROFILE_LEVEL,
    FindContinuumMoments,
    _request_cubes,
    evaluate_lattice_expression,
    finalize_streamed_reductions,
    find_continuum_first_round_requests,
    lattice_expression_names,
    new_streamed_partials,
    pack_plane_masks,
    split_lattice_conjuncts,
    tile_occupancy,
)

ULTRA_TASKS_PER_THREAD = 4
#: Inner chunk (px) of which merge tiles are made along an axis that one chunk
#: of the cube spans.
MERGE_TILE_INNER_CHUNK = 625

# Node-task baseline of the moments application (interpreter, worker, zarr).
_TASK_BASELINE_BYTES = 1.5 * 2**30
# The first round's transients beside its packed masks, per pixel (traced at
# 512 x 512 and 1024 x 1024, sharded and unsharded, float32 and float64).
_FIRST_ROUND_BYTES_PER_PIXEL = 8
_ROLES = ("sky", "primary_beam")
# Request fields that FusedReductions leaves out of its cache keys.
_AXIS_FIELDS = ("frequency_axis", "rest_frequency")
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_NONZERO = re.compile(r'"(sky|primary_beam)"\s*!=\s*0')
_PROFILE = re.compile(
    r'"primary_beam"\s*>\s*(\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?)'
)


@dataclasses.dataclass(frozen=True)
class StoredMap:
    """A map in a Zarr store, read when it is used.

    Parameters
    ----------
    store : str
        Path of the Zarr group.
    array : str
        Name of the array in the group.
    index : int, optional
        Index along the first axis of the array (a spectral window of a
        first-round store); ``None``: the whole array.
    """

    store: str
    array: str
    index: int | None = None

    def load(self):
        """The map.

        Returns
        -------
        numpy.ndarray
            A new float64 array.
        """
        import zarr

        array = zarr.open_array(os.path.join(self.store, self.array), mode="r")
        values = array[...] if self.index is None else array[self.index]
        return np.asarray(values, dtype=np.float64)


def ultra_task_memory_bytes(
    n_l,
    n_m,
    planes_per_batch,
    moments_bytes_per_pixel,
    n_two_d_masks,
    gather_fraction,
    read_primary_beam,
    sky_itemsize=4,
    primary_beam_itemsize=4,
    read_unit_bytes=0,
    compute_first_round=False,
):
    """Memory of a find_continuum_ultra frequency task.

    The baseline of a node task, two buffers of ``planes_per_batch`` planes
    (the batch and the one read ahead) per cube read, in its dtype, four
    boolean planes (primary-beam masks and validities) and one per
    two-dimensional mask, the moment accumulators, the values gathered for
    per-channel statistics and spectra (12 bytes per pixel), two float64
    blocks of rows, the read unit being decoded with its stored bytes and,
    for the first round, its transients.

    Parameters
    ----------
    n_l, n_m : int
        Plane shape.
    planes_per_batch : int
        Planes per read.
    moments_bytes_per_pixel : int
        Bytes per pixel of the moment accumulators of the task's requests.
    n_two_d_masks : int
        Two-dimensional masks of the task's requests.
    gather_fraction : float
        Largest fraction of a plane that a request gathers in a channel.
    read_primary_beam : bool
        The task reads the primary beam as well as the cube.
    sky_itemsize, primary_beam_itemsize : int, default 4
        Bytes per pixel of the cube and of the primary beam in the store.
    read_unit_bytes : int, default 0
        Decoded bytes of the largest read unit (an inner chunk of a sharded
        array, a chunk otherwise, all of its channels and polarizations) of
        the arrays the task reads; a read holds one with its stored bytes.
    compute_first_round : bool, default False
        The task makes the first round: its transients (8 bytes per pixel,
        measured) and the bit-packed masks of two batches.

    Returns
    -------
    int
        Bytes.
    """
    pixels = n_l * n_m
    plane_bytes = sky_itemsize + (primary_beam_itemsize if read_primary_beam else 0)
    first_round = (
        (_FIRST_ROUND_BYTES_PER_PIXEL + planes_per_batch / 2) * pixels
        if compute_first_round
        else 0
    )
    return int(
        _TASK_BASELINE_BYTES
        + 2 * planes_per_batch * plane_bytes * pixels
        + (4 + n_two_d_masks) * pixels
        + moments_bytes_per_pixel * pixels
        + 12 * gather_fraction * pixels
        + 2 * 8 * _BLOCK_ROWS * n_m
        + 2 * read_unit_bytes
        + first_round
    )


def ultra_merge_tile_counts(map_shape, tile_shape):
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


def ultra_merge_task_memory_bytes(
    tile_pixels, n_slots, state_bytes_per_pixel, n_moments
):
    """Memory of a merge task of find_continuum_ultra or of the imager's drain.

    The baseline of a node task, the partial states of a tile that GraphVIPER's
    tree order holds at once (at most ``ceil(log2(n_slots)) + 1``), and the
    float64 maps it finalizes.

    Parameters
    ----------
    tile_pixels : int
        Pixels of a merge tile.
    n_slots : int
        Slots merged.
    state_bytes_per_pixel : int
        Bytes per pixel of a slot's state (the fields of its moments).
    n_moments : int
        Moments finalized.

    Returns
    -------
    int
        Bytes.
    """
    states = int(np.ceil(np.log2(max(n_slots, 1)))) + 1
    return int(
        _TASK_BASELINE_BYTES
        + (states * state_bytes_per_pixel + 8 * n_moments + 1) * tile_pixels
    )


def ultra_frequency_tasks(n_channels, shard_length, n_threads):
    """Frequency tasks of a find_continuum_ultra pass.

    About ``ULTRA_TASKS_PER_THREAD`` tasks per thread, each of whole shards.

    Parameters
    ----------
    n_channels : int
        Channels of the spectral window.
    shard_length : int
        Channels per frequency shard of the cube (per chunk when it is not
        sharded, 1 when it is not chunked).
    n_threads : int
        Threads of the cluster.

    Returns
    -------
    int
        Number of frequency chunks.
    """
    channels_per_task = shard_length * max(
        1, round(n_channels / (ULTRA_TASKS_PER_THREAD * n_threads * shard_length))
    )
    return -(-n_channels // channels_per_task)


class UltraReductions:
    """Answers findContinuum's reduction requests in streamed passes.

    Parameters
    ----------
    cubes : dict
        The cubes registered with
        :func:`~astroviper.distributed_applications.imaging.find_continuum.casa_image_tasks.cube_session`;
        the entries of the cube and of its primary beam are those whose
        ``variable`` is in ``variables``, and the cube's ``channel_offset`` is
        the first channel of the spectral window.
    input_image_store : str
        Path of the xradio image store.
    variables : dict
        ``{"sky": ..., "primary_beam": ...}``, the data variables of the cube
        and of its primary beam.
    polarization_index : int
        Index of Stokes I along ``polarization``.
    work_directory : str
        findContinuum's work directory; the stores and mask files of the passes
        go to its subdirectory ``.find_continuum_ultra``.
    primary_beam_limit : float or None
        Pixels where the primary beam is below this level are masked in both
        cubes; ``None``: no limit.
    first_round : str or None
        Path of a complete first-round store of the cube, validated against
        it; ``None``: the first pass makes the first round of the window.
    n_tasks : int
        Frequency chunks of the window (``n_chunks`` of ``make_parallel_coord``).
    budget_bytes : int
        Memory budget of a frequency task.
    frequency_axis : sequence
        ``(reference_value, increment, reference_pixel, n_channels)`` of the
        spectral window.
    rest_frequency : float
        Rest frequency (Hz).
    monitor_resources_seconds : float or None
        Sampling interval of GraphVIPER's resource monitor in every map.
    memory_per_thread_bytes : int, optional
        Memory of a thread: a task over its budget at one plane per batch runs
        up to it, with a warning, and is refused above it. ``None`` (default):
        refused above the budget.

    Attributes
    ----------
    passes : list of dict
        One row per pass over the image store.
    timing : list of pandas.DataFrame
        The node-task rows of every pass's frequency and merge maps.
    reduce_records : list of dict
        One record per reduce node.
    cube_reads : dict
        Data variable to the number of passes that read it.
    layout : dict
        ``budget_bytes``, ``n_frequency_tasks``, ``channels_per_task``,
        ``merge_tile_shape``, ``n_merge_tiles`` and ``plane_bytes``.
    T_primary_beam_mean : float
        Seconds spent on the mean primary beam in the driver.
    """

    def __init__(
        self,
        cubes,
        input_image_store,
        variables,
        polarization_index,
        work_directory,
        primary_beam_limit,
        first_round,
        n_tasks,
        budget_bytes,
        frequency_axis,
        rest_frequency,
        monitor_resources_seconds,
        memory_per_thread_bytes=None,
    ):
        import xarray as xr
        import zarr
        from graphviper.graph_tools.coordinate_utils import (
            interpolate_data_coords_onto_parallel_coords,
            make_parallel_coord,
        )

        self.input_image_store = input_image_store
        self.variables = dict(variables)
        self.polarization_index = polarization_index
        self.directory = os.path.join(work_directory, ".find_continuum_ultra")
        self.primary_beam_limit = primary_beam_limit
        self.first_round = first_round
        self.budget_bytes = budget_bytes
        self.limit_bytes = (
            budget_bytes if memory_per_thread_bytes is None else memory_per_thread_bytes
        )
        self.frequency_axis = tuple(frequency_axis)
        self.rest_frequency = rest_frequency
        self.monitor_resources_seconds = monitor_resources_seconds
        self.passes, self.timing, self.reduce_records, self.cube_reads = [], [], [], {}
        self.T_primary_beam_mean = 0.0

        self.roles = {
            path.rstrip("/"): role
            for path, cube in cubes.items()
            for role, variable in self.variables.items()
            if cube["variable"] == variable
        }
        paths = {role: path for path, role in self.roles.items()}
        self.start = int(cubes[paths["sky"]]["channel_offset"])
        self.stop = self.start + int(self.frequency_axis[3])
        self.first_round_requests = find_continuum_first_round_requests(
            self.frequency_axis, rest_frequency, paths["sky"], paths["primary_beam"]
        )

        arrays = {
            role: zarr.open_array(os.path.join(input_image_store, variable), mode="r")
            for role, variable in self.variables.items()
        }
        sky = arrays["sky"]
        self.dtypes = {role: array.dtype for role, array in arrays.items()}
        self.read_unit_bytes = {
            role: int(np.prod(array.chunks)) * array.dtype.itemsize
            for role, array in arrays.items()
        }
        self.n_channels = sky.shape[1]
        self.map_shape = tuple(sky.shape[-2:])
        self.tile_shape = tuple(sky.chunks[-2:])
        self.shard_length = (sky.shards or sky.chunks)[1]

        self.index_xds = xr.Dataset(
            coords={
                dim: np.arange(n, dtype=np.float64)
                for dim, n in zip(
                    ("frequency", "l", "m"),
                    (self.n_channels, *self.map_shape),
                    strict=True,
                )
            }
        )
        chunks = make_parallel_coord(
            coord=self.index_xds.frequency[self.start : self.stop], n_chunks=n_tasks
        )
        self.frequency_mapping = interpolate_data_coords_onto_parallel_coords(
            {"frequency": chunks}, {"img": self.index_xds}
        )
        self.n_slots = len(chunks["data_chunks"])
        # Planes of a read unit within a task, which a batch reads together.
        self.read_unit_planes = min(
            sky.chunks[1], max(map(len, chunks["data_chunks"].values()))
        )
        tiles = {
            dim: make_parallel_coord(coord=self.index_xds[dim], n_chunks=n_chunks)
            for dim, n_chunks in zip(
                ("l", "m"),
                ultra_merge_tile_counts(self.map_shape, self.tile_shape),
                strict=True,
            )
        }
        self.tile_mapping = interpolate_data_coords_onto_parallel_coords(
            tiles, {"img": self.index_xds}
        )
        self.merge_tile_shape = tuple(
            len(tiles[dim]["data_chunks"][0]) for dim in ("l", "m")
        )
        self.layout = {
            "budget_bytes": int(budget_bytes),
            "n_frequency_tasks": self.n_slots,
            "channels_per_task": len(chunks["data_chunks"][0]),
            "merge_tile_shape": list(self.merge_tile_shape),
            "n_merge_tiles": len(self.tile_mapping),
            "plane_bytes": int(np.prod(self.map_shape)) * self.dtypes["sky"].itemsize,
        }

    def run_pass(self, requests, n_requested):
        """Results of findContinuum's reduction requests.

        Parameters
        ----------
        requests : list of dict
            Requests (of
            :func:`~astroviper.distributed_applications.imaging.find_continuum.casa_image_tasks.cube_session`'s
            ``reduce``), the requested ones first, then the predicted ones.
        n_requested : int
            Number of requested requests.

        Returns
        -------
        list
            One result per request, as
            :func:`~astroviper.processing_functions.imaging.find_continuum.finalize_streamed_reductions`
            gives it, moment maps as :class:`StoredMap` (the mean primary
            beam as an array); ``None`` for a prediction the memory plan
            dropped.

        Raises
        ------
        ValueError
            If a moments request selects no channel, or a requested request
            does not fit the memory budget at one plane per batch.
        """
        results = [None] * len(requests)
        mean, moments, *statistics_requests = self.first_round_requests
        statistics, stream = [], []
        for k, request in enumerate(requests):
            if _same_reduction(request, mean):
                start = time.time()
                results[k] = {-1: self._primary_beam_mean(request["channels"])}
                self.T_primary_beam_mean += time.time() - start
            elif any(_same_reduction(request, other) for other in statistics_requests):
                statistics.append(k)
            elif _same_reduction(request, moments) and (
                stored := self._stored_moments()
            ):
                results[k] = stored
            elif request["kind"] == "moments" and request["channels"] == []:
                raise ValueError(f"No channel of the cube is selected by {request}")
            else:
                stream.append(k)

        first_round_needed = bool(statistics) and self.first_round is None
        if stream or first_round_needed:
            start = time.time()
            encoded, masks = self._encode([requests[k] for k in stream])
            plan = self._plan(encoded, masks, sum(k < n_requested for k in stream))
            sub_passes = zip(
                plan["sub_passes"],
                plan["planes_per_batch"],
                plan["task_bytes"],
                strict=True,
            )
            if plan["sub_passes"] == [[]] and not first_round_needed:
                sub_passes = ()
            n_dropped = len(plan["dropped"])
            for indices, planes_per_batch, task_bytes in sub_passes:
                answers = self._run_pass(
                    [requests[stream[j]] for j in indices],
                    [encoded[j] for j in indices],
                    masks,
                    planes_per_batch,
                    task_bytes,
                    n_dropped,
                    start,
                )
                for j, answer in zip(indices, answers, strict=True):
                    results[stream[j]] = answer
                n_dropped, start = 0, time.time()
        for k in statistics:
            results[k] = self._first_round_statistics(requests[k]["image"])
        return results

    def memory_plan(self, requests, n_requested):
        """Plan a pass within the memory budget of its tasks.

        At most 4 planes per batch (and at most a shard), or the planes of a
        read unit within a task when it holds more channels (each unit is then
        decoded once per task), halved while a task is over budget (a unit is
        then decoded once per batch); then predicted moments requests are
        dropped, the last first; then the requested requests are split into
        sub-passes of whole requests, each predicted request joining the first
        sub-pass it fits (or dropped), each sub-pass with its own planes per
        batch. The first sub-pass makes the first round when there is no
        first-round store.

        Parameters
        ----------
        requests : list of dict
            Requests of the pass, the requested ones first.
        n_requested : int
            Number of requested requests.

        Returns
        -------
        dict
            Per sub-pass ``planes_per_batch``, ``task_bytes`` and
            ``sub_passes`` (indices into ``requests``), and ``dropped``
            (indices of the predictions left out).

        Raises
        ------
        ValueError
            If a requested request (or the first round) does not fit the
            memory of a thread at one plane per batch.
        """
        return self._plan(*self._encode(requests), n_requested)

    def check_smallest_task(self):
        """Refuse a run whose smallest frequency task cannot fit the memory of
        a thread: the first round at one plane per batch, or with a
        first-round store, a read of the cube at one plane per batch.

        Raises
        ------
        ValueError
            If that task needs more than the memory of a thread.
        """
        from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
            check_find_continuum_ultra_task_memory,
        )

        first_round = self.first_round is None
        check_find_continuum_ultra_task_memory(
            ultra_task_memory_bytes(
                *self.map_shape,
                1,
                0,
                0,
                0.0,
                first_round,
                self.dtypes["sky"].itemsize,
                self.dtypes["primary_beam"].itemsize,
                max(
                    self.read_unit_bytes[role]
                    for role in _ROLES[: 2 if first_round else 1]
                ),
                first_round,
            ),
            self.limit_bytes,
        )

    def _plan(self, encoded, masks, n_requested):
        from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
            check_find_continuum_ultra_task_memory,
        )

        costs = [self._costs(request, masks) for request in encoded]
        first_round = self.first_round is None

        def task_bytes(indices, planes_per_batch, first):
            read_primary_beam = (first and first_round) or any(
                costs[k][2] for k in indices
            )
            return ultra_task_memory_bytes(
                *self.map_shape,
                planes_per_batch,
                sum(costs[k][0] for k in indices),
                len({encoded[k]["two_d_mask"] for k in indices} - {None}),
                max((costs[k][1] for k in indices), default=0.0),
                read_primary_beam,
                self.dtypes["sky"].itemsize,
                self.dtypes["primary_beam"].itemsize,
                max(
                    self.read_unit_bytes[role]
                    for role in _ROLES[: 2 if read_primary_beam else 1]
                ),
                first and first_round,
            )

        def fits(indices, first, planes_per_batch=1):
            return task_bytes(indices, planes_per_batch, first) <= self.budget_bytes

        def check(indices, first, label):
            smallest = task_bytes(indices, 1, first)
            check_find_continuum_ultra_task_memory(smallest, self.limit_bytes)
            if smallest > self.budget_bytes:
                logger.warning(
                    f"A find_continuum_ultra task needs {smallest / 2**30:.2f} GiB "
                    f"for {label} at one plane per batch, more than its budget of "
                    f"{self.budget_bytes / 2**30:.2f} GiB: it runs alone at one "
                    "plane per batch."
                )

        if first_round:
            check([], True, "the first round")
        for k in range(n_requested):
            check([k], False, f"the {encoded[k]['kind']} of {encoded[k]['image']}")
        kept, dropped = list(range(len(encoded))), []
        for k in reversed(range(n_requested, len(encoded))):
            if fits(kept, True):
                break
            if encoded[k]["kind"] == "moments":
                kept.remove(k)
                dropped.append(k)
        if fits(kept, True):
            sub_passes = [kept]
        else:
            sub_passes = [[]]
            for k in range(n_requested):
                if not fits([*sub_passes[-1], k], len(sub_passes) == 1):
                    sub_passes.append([])
                sub_passes[-1].append(k)
            for k in kept[n_requested:]:
                for i, indices in enumerate(sub_passes):
                    if fits([*indices, k], i == 0):
                        indices.append(k)
                        break
                else:
                    dropped.append(k)
        planes = []
        for i, indices in enumerate(sub_passes):
            planes_per_batch = max(min(4, self.shard_length), self.read_unit_planes)
            while planes_per_batch > 1 and not fits(indices, i == 0, planes_per_batch):
                planes_per_batch //= 2
            planes.append(planes_per_batch)
        return {
            "planes_per_batch": planes,
            "task_bytes": [
                task_bytes(indices, b, i == 0)
                for i, (indices, b) in enumerate(zip(sub_passes, planes, strict=True))
            ],
            "sub_passes": sub_passes,
            "dropped": sorted(dropped),
        }

    def _costs(self, request, masks):
        """Moments bytes per pixel, gather fraction and primary-beam read of an
        encoded request."""
        moments = 0
        if request["kind"] == "moments":
            moments = sum(
                dtype.itemsize for dtype in self._field_dtypes(request).values()
            )
        gather = 0.0
        if request["kind"] == "profile" or request.get("per_channel"):
            index = request["two_d_mask"]
            gather = 1.0 if index is None else masks[index]["fraction"]
        return moments, gather, "primary_beam" in _request_cubes(request)

    def _field_dtypes(self, request):
        """The scratch arrays of an encoded moments request: field to dtype."""
        count = np.uint16 if self.frequency_axis[3] < 2**16 else np.uint32
        value = self.dtypes[request["image"]]
        dtypes = {
            "INTEGRATED": np.float64,
            "SUM": np.float64,
            "MAXIMUM": value,
            "MINIMUM": value,
            "COUNT": count,
        }
        fields = FindContinuumMoments(request["moments"], (), (0, 0))._fields()
        return {field: np.dtype(dtypes[field]) for field in fields}

    def _encode(self, requests):
        """Encoded requests and the distinct two-dimensional masks they index."""
        n_l, n_m = self.map_shape
        encoded, masks, index = [], [], {}
        for request in requests:
            mask = self._rename(request["mask"])
            split = split_lattice_conjuncts(mask, _ROLES)
            two_d, terms = (None, []) if split is None else split
            item = {
                key: request[key]
                for key in ("kind", "moments", "function", "per_channel")
                if key in request
            }
            item.update(
                image=self.roles[request["image"].rstrip("/")],
                channels=None
                if request["channels"] is None
                else np.array(
                    [self.start + c for c in sorted(request["channels"])],
                    dtype=np.int32,
                ),
                two_d_mask=None,
                cube_terms=[_cube_term(term) for term in terms],
                expression=mask if split is None else None,
                footprint=None,
            )
            if request["kind"] == "moments":
                item["channel_offset"] = self.start
            if two_d is not None:
                arrays = casa_image_tasks._expression_arrays(two_d)
                plane = np.broadcast_to(
                    np.asarray(evaluate_lattice_expression(two_d, arrays), dtype=bool),
                    (n_l, n_m, 1, 1),
                )[..., 0, 0]
                packed = pack_plane_masks(plane)
                digest = hashlib.sha1(packed).hexdigest()
                if digest not in index:
                    index[digest] = len(masks)
                    masks.append(
                        {
                            "digest": digest,
                            "packed": packed,
                            "fraction": np.count_nonzero(plane) / plane.size,
                            "footprint": tile_occupancy(
                                plane[None, None, None], self.tile_shape
                            )[0, 0, 0],
                        }
                    )
                item["two_d_mask"] = index[digest]
                item["footprint"] = masks[index[digest]]["footprint"]
            encoded.append(item)
        return encoded, masks

    def _rename(self, expression):
        """A lattice expression with the cubes named by their roles."""

        def rename(match):
            name = match.group()[1:-1].rstrip("/")
            if name in self.roles:
                return f'"{self.roles[name]}"'
            if name in _ROLES:
                raise ValueError(
                    f"The image {match.group()} has the name of a cube role in {expression!r}"
                )
            return match.group()

        return _QUOTED.sub(rename, expression)

    def _run_pass(
        self, requests, encoded, masks, planes_per_batch, task_bytes, n_dropped, start
    ):
        """One pass over the image store; the results of its requests."""
        import dask
        import pandas as pd
        from graphviper.graph_tools import generate_dask_workflow, map, reduce

        index = len(self.passes)
        compute_first_round = self.first_round is None
        read_primary_beam = compute_first_round or any(
            "primary_beam" in _request_cubes(request) for request in encoded
        )
        priorities = self._task_priorities(
            encoded, compute_first_round, read_primary_beam
        )
        if not priorities:
            return finalize_streamed_reductions(new_streamed_partials(encoded), encoded)
        first_round_store = (
            self._create_first_round_store()
            if compute_first_round
            else self.first_round
        )
        used = sorted({request["two_d_mask"] for request in encoded} - {None})
        encoded = [
            dict(
                request,
                two_d_mask=None
                if request["two_d_mask"] is None
                else used.index(request["two_d_mask"]),
            )
            for request in encoded
        ]
        mask_files = [self._mask_file(masks[i]) for i in used]
        moments = [
            r for r, request in enumerate(encoded) if request["kind"] == "moments"
        ]
        partials_store = results_store = None
        if moments:
            partials_store, results_store = self._create_scratch(
                os.path.join(self.directory, f"pass_{index}"), encoded
            )
        names = set()
        for request in encoded:
            texts = [
                term[1] for term in request["cube_terms"] if term[0] == "expression"
            ]
            if request["expression"] is not None:
                texts.append(request["expression"])
            for text in texts:
                names.update(lattice_expression_names(text))
        image_paths = {
            name: os.path.abspath(name) for name in sorted(names - set(_ROLES))
        }
        read = {self.variables["sky"]}
        if read_primary_beam:
            read.add(self.variables["primary_beam"])
        for variable in read:
            self.cube_reads[variable] = self.cube_reads.get(variable, 0) + 1
        labels = ["first round"] if compute_first_round else []
        labels += [
            f"{request['kind']} of {os.path.basename(request['image'])}"
            for request in requests
        ]
        logger.info(
            f"find_continuum_ultra pass {index}: {len(priorities)} frequency tasks of "
            f"{self.layout['channels_per_task']} channels, {planes_per_batch} planes "
            f"per batch, {task_bytes / 2**30:.2f} of {self.budget_bytes / 2**30:.2f} "
            f"GiB per task; {len(moments)} moments requests merged in "
            f"{len(self.tile_mapping)} tiles ({', '.join(labels)})"
        )

        viper_graph = map(
            input_data={"img": self.index_xds},
            node_task_data_mapping={
                task: self.frequency_mapping[task] for task in priorities
            },
            node_task=find_continuum_streamed_reductions,
            input_params={
                "input_image_store": self.input_image_store,
                "variables": self.variables,
                "polarization_index": self.polarization_index,
                "primary_beam_limit": self.primary_beam_limit,
                "requests": encoded,
                "frequency_axis": self.frequency_axis,
                "rest_frequency": self.rest_frequency,
                "first_round_store": first_round_store,
                "compute_first_round": compute_first_round,
                "read_primary_beam": read_primary_beam,
                "two_d_mask_files": mask_files,
                "image_paths": image_paths,
                "partials_store": partials_store,
                "planes_per_batch": planes_per_batch,
            },
            in_memory_compute=False,
            monitor_resources_seconds=self.monitor_resources_seconds,
            task_priorities=priorities,
        )
        viper_graph = reduce(
            viper_graph, merge_find_continuum_streamed_reductions, {}, mode="tree"
        )
        graph_end = time.time()
        dask_graph = generate_dask_workflow(viper_graph)
        compute_start = time.time()
        output = dask.compute(dask_graph)[0]
        compute_end = time.time()
        frame = _with_usage(output)
        self.timing.append(frame.assign(reduction=index))
        self.reduce_records.extend(
            dict(record, reduction=index)
            for record in output.get("timing_reduce_nodes", [])
        )
        if compute_first_round:
            written = write_find_continuum_first_round_rows(
                first_round_store, output["first_round"], self.n_channels
            )
            if not written["complete"]:
                raise RuntimeError(
                    f"The first round of channels [{self.start}, {self.stop}) is "
                    f"incomplete in {first_round_store}"
                )
            self.first_round = first_round_store
        results = finalize_streamed_reductions(output["reductions"], encoded)

        row = {
            "merge_task_memory_plan_bytes": 0,
            "merge_start_unixtime": np.nan,
            "merge_end_unixtime": np.nan,
            "T_merge_create_graph": 0.0,
            "T_merge_generate_dask_graph": 0.0,
            "T_merge_compute_dask_graph": 0.0,
            "bytes_partials_written": 0,
            "bytes_partials_read": 0,
            "bytes_results_written": 0,
        }
        if moments:
            row["merge_task_memory_plan_bytes"] = max(
                ultra_merge_task_memory_bytes(
                    int(np.prod(self.merge_tile_shape)),
                    self.n_slots,
                    sum(d.itemsize for d in self._field_dtypes(encoded[r]).values()),
                    len(encoded[r]["moments"]),
                )
                for r in moments
            )
            if row["merge_task_memory_plan_bytes"] > self.budget_bytes:
                logger.warning(
                    "A find_continuum_ultra merge task needs "
                    f"{row['merge_task_memory_plan_bytes'] / 2**30:.2f} GiB, more "
                    f"than the task budget of {self.budget_bytes / 2**30:.2f} GiB."
                )
            merge_start = time.time()
            viper_graph = map(
                input_data={"img": self.index_xds},
                node_task_data_mapping=self.tile_mapping,
                node_task=merge_find_continuum_partial_maps,
                input_params={
                    "partials_store": partials_store,
                    "results_store": results_store,
                    "moments_requests": [
                        (
                            r,
                            encoded[r]["moments"],
                            str(self.dtypes[encoded[r]["image"]]),
                        )
                        for r in moments
                    ],
                    "written": output["written"],
                    "n_slots": self.n_slots,
                },
                in_memory_compute=False,
                monitor_resources_seconds=self.monitor_resources_seconds,
            )
            merge_graph_end = time.time()
            dask_graph = generate_dask_workflow(viper_graph)
            merge_compute_start = time.time()
            merged = dask.compute(dask_graph)[0]
            merge_end = time.time()
            merge_frame = pd.concat(
                [_with_usage(item) for item in merged], ignore_index=True
            )
            self.timing.append(merge_frame.assign(reduction=index))
            row.update(
                merge_start_unixtime=merge_start,
                merge_end_unixtime=merge_end,
                T_merge_create_graph=merge_graph_end - merge_start,
                T_merge_generate_dask_graph=merge_compute_start - merge_graph_end,
                T_merge_compute_dask_graph=merge_end - merge_compute_start,
                bytes_partials_written=_stored_bytes(partials_store),
                bytes_partials_read=int(merge_frame["bytes_read_partials"].sum()),
                bytes_results_written=int(merge_frame["bytes_written"].sum()),
            )
            shutil.rmtree(partials_store)
            for r in moments:
                results[r] = {
                    moment: StoredMap(results_store, f"R{r}_MOMENT_{moment}")
                    for moment in encoded[r]["moments"]
                }

        self.passes.append(
            {
                "pass": index,
                "reductions": ", ".join(labels),
                "data_variables_read": ", ".join(sorted(read)),
                "compute_first_round": compute_first_round,
                "n_tasks": len(frame),
                "n_channels_read": int(frame["n_channels"].sum()),
                **{
                    key: int(frame[key].sum())
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
                "n_moments_requests": len(moments),
                "n_dropped_predictions": n_dropped,
                "task_rss_peak_bytes": _rss_peak(frame),
                "graph_start_unixtime": start,
                "start_unixtime": compute_start,
                "compute_end_unixtime": compute_end,
                "T_create_map_reduce_graph": graph_end - start,
                "T_generate_dask_graph": compute_start - graph_end,
                "T_compute_dask_graph": compute_end - compute_start,
                **row,
            }
        )
        logger.debug(
            f"find_continuum_ultra pass {index} in {time.time() - start:.2f} s"
        )
        return results

    def _task_priorities(self, encoded, compute_first_round, read_primary_beam):
        """Task id to planes x tiles to read, for the chunks holding a channel
        that a request selects (every chunk for the first round)."""
        n_tiles = tuple(
            -(-n // t) for n, t in zip(self.map_shape, self.tile_shape, strict=True)
        )
        priorities = {}
        for task, item in self.frequency_mapping.items():
            frequency = item["data_selection"]["img"]["frequency"]
            chunk = np.arange(frequency.start, frequency.stop)
            needed = np.full(chunk.size, compute_first_round)
            tiles = np.full(n_tiles, compute_first_round)
            for request in encoded:
                chosen = (
                    np.ones(chunk.size, dtype=bool)
                    if request["channels"] is None
                    else np.isin(chunk, request["channels"])
                )
                if chosen.any():
                    needed |= chosen
                    tiles |= (
                        True if request["footprint"] is None else request["footprint"]
                    )
            if needed.any():
                planes = int(needed.sum()) * (2 if read_primary_beam else 1)
                priorities[task] = planes * int(tiles.sum())
        return priorities

    def _create_scratch(self, directory, encoded):
        """The scratch store of the moment partials and the results store."""
        import zarr

        os.makedirs(directory, exist_ok=True)
        stores = [
            os.path.join(directory, name) for name in ("partials.zarr", "results.zarr")
        ]
        partials, results = (zarr.open_group(store, mode="w") for store in stores)
        # A shard holds whole tiles, so it may reach beyond the map.
        shard = tuple(
            -(-n // t) * t
            for n, t in zip(self.map_shape, self.merge_tile_shape, strict=True)
        )
        for r, request in enumerate(encoded):
            if request["kind"] != "moments":
                continue
            for field, dtype in self._field_dtypes(request).items():
                partials.create_array(
                    f"R{r}_{field}",
                    shape=(self.n_slots, *self.map_shape),
                    chunks=(1, *self.merge_tile_shape),
                    shards=(1, *shard),
                    dtype=dtype,
                    fill_value=0,
                    compressors=None,
                    dimension_names=("slot", "l", "m"),
                )
            for moment in request["moments"]:
                results.create_array(
                    f"R{r}_MOMENT_{moment}",
                    shape=self.map_shape,
                    chunks=self.merge_tile_shape,
                    dtype=np.float64,
                    fill_value=np.nan,
                    compressors=None,
                )
        return stores

    def _create_first_round_store(self):
        """An empty first-round store of the cube in the work directory, for
        the window's channels."""
        from astroviper.distributed_applications.imaging.find_continuum.first_round import (
            create_find_continuum_first_round_store,
        )

        store = os.path.join(self.directory, "first_round.zarr")
        shutil.rmtree(store, ignore_errors=True)
        create_find_continuum_first_round_store(
            first_round_store=store,
            n_channels=self.n_channels,
            image_shape=self.map_shape,
            tile_shape=self.tile_shape,
            merge_tile_shape=self.merge_tile_shape,
            primary_beam_limit=self.primary_beam_limit,
            variables=self.variables,
            polarization_index=self.polarization_index,
            value_dtypes={role: str(dtype) for role, dtype in self.dtypes.items()},
            run_id=None,
            written_by="find_continuum_ultra",
            windows=[
                {
                    "channel_range": [self.start, self.stop],
                    "frequency_axis": list(self.frequency_axis),
                }
            ],
            rest_frequency=self.rest_frequency,
        )
        return store

    def _mask_file(self, mask):
        """The ``.npy`` file of a bit-packed two-dimensional mask."""
        path = os.path.join(self.directory, "masks", f"{mask['digest']}.npy")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, mask["packed"])
        return path

    def _primary_beam_mean(self, channels):
        """Moment -1 of the primary beam at window channels (all if ``None``)."""
        channels = (
            range(self.start, self.stop)
            if channels is None
            else [self.start + c for c in channels]
        )
        session = casa_image_tasks._session
        return primary_beam_channel_mean(
            self.input_image_store,
            self.variables["primary_beam"],
            self.polarization_index,
            channels,
            self.primary_beam_limit,
            None if session is None else session.counters,
        )

    def _first_round_statistics(self, image):
        """Whole-cube statistics of a cube over the window, from the first-round store."""
        import zarr

        rows = zarr.open_array(os.path.join(self.first_round, "STATISTICS"), mode="r")
        partial = {
            "channels": np.arange(self.start, self.stop),
            "rows": rows[self.start : self.stop, _ROLES.index(self.roles[image])],
        }
        return finalize_streamed_reductions(
            [partial], [{"kind": "statistics", "per_channel": False}]
        )[0]

    def _stored_moments(self):
        """The imager's moments 0 and 8 of the window, or ``None``."""
        if self.first_round is None:
            return None
        import zarr

        attributes = zarr.open_group(self.first_round, mode="r").attrs[
            "find_continuum_first_round"
        ]
        for w, window in enumerate(attributes["windows"]):
            if (
                window["channel_range"] == [self.start, self.stop]
                and attributes["moments_complete"][w]
            ):
                return {
                    m: StoredMap(self.first_round, f"MOMENT_{m}", index=w)
                    for m in (0, 8)
                }
        return None


def _same_reduction(request, other):
    """Whether two requests ask for the same reduction."""
    return {k: v for k, v in request.items() if k not in _AXIS_FIELDS} == {
        k: v for k, v in other.items() if k not in _AXIS_FIELDS
    }


def _cube_term(term):
    """The encoding of a cube term of a mask (see find_continuum_batch_reductions)."""
    nonzero = _NONZERO.fullmatch(term)
    if nonzero:
        return ("nonzero", nonzero.group(1))
    level = _PROFILE.fullmatch(term)
    if level and float(level.group(1)) == PROFILE_LEVEL:
        return ("profile",)
    return ("expression", term)


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
    """Bytes of the chunk and shard files of a Zarr store."""
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(store)
        for name in names
        if name != "zarr.json"
    )
