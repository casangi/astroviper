"""Tests of the find_continuum first-round store and the drain of the imager's moments."""

import errno
import os
import shutil
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
import zarr
from xradio.image import make_empty_sky_image

import astroviper.node_tasks.imaging.find_continuum_first_round as node_tasks
from astroviper.distributed_applications.imaging import find_continuum_first_round
from astroviper.distributed_applications.imaging.find_continuum_first_round import (
    FirstRoundStoreInputs,
    bind_find_continuum_first_round_store,
    create_find_continuum_first_round_store,
    drain_first_round_moments,
    find_continuum_first_round_partials_path,
    find_continuum_first_round_store_path,
    find_continuum_first_round_task_priorities,
    find_continuum_merge_tile_counts,
    validate_find_continuum_first_round,
)
from astroviper.node_tasks.imaging.find_continuum_first_round import (
    find_continuum_first_round_chunk,
    write_first_round_rows,
)
from astroviper.node_tasks.imaging.utils import (
    write_result_chunk_to_disk_sharded_skunk_works,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
    accumulate_moments,
    finalize_moments,
    merge_moment_partials_in_tree_order,
    new_moment_partials,
    primary_beam_masks,
    valid_voxels,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    linear_spectral_axis,
    moment_channel_weights,
)
from astroviper.processing_functions.imaging.utils.spectral_windows import (
    linear_channel_ranges,
)
from astroviper.utils.data_group_tools import modify_data_groups_xds
from astroviper.utils.io import (
    create_empty_data_variables_on_disk,
    write_zarr_image_store,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    RAD_PER_ARCSEC,
    make_find_continuum_image_xds,
)

# Two spectral windows of 16 and 18 channels with a gap; 20 px planes in
# chunks of 8 px (ragged edge tiles), moments merged in tiles of 12 px.
N_CHANNELS, N_PIXELS, TILE, MERGE_TILE = 34, 20, (8, 8), (12, 12)
WINDOW_RANGES = [[0, 16], [16, 34]]
LIMIT = 0.2
DIMS = ["time", "frequency", "polarization", "l", "m"]
VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
VALUE_DTYPES = {"sky": "float32", "primary_beam": "float32"}
IN_ORDER = [np.arange(start, min(start + 5, N_CHANNELS)) for start in range(0, 34, 5)]
MAP_SHAPE = (1, 1, 1, N_PIXELS, N_PIXELS)
PACKED_DIMS, TILE_DIMS = (
    ("frequency", "l", "m_packed"),
    ("frequency", "l_tile", "m_tile"),
)
MAP_DIMS = ("window", "l", "m")


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    """A Stokes I and Q cube of two windows, written as the imager writes it."""
    source = make_find_continuum_image_xds(
        n_pixels=N_PIXELS,
        n_channels=N_CHANNELS,
        primary_beam_fwhm_arcsec=3.0,
        lines=(
            dict(
                channel=5,
                fwhm_channels=2,
                peak_jy=0.05,
                offset_pixels=(2, -3),
                size_fwhm_pixels=3,
            ),
        ),
    )
    sky = source["SKY_RESIDUAL"].values.copy()
    primary_beam = source["PRIMARY_BEAM"].values.copy()
    sky[0, 3, 0, :4] = 0.0
    sky[0, 17, 0, 9, 9] = np.nan
    sky[0, 18, 0, 10, 10] = -0.0
    primary_beam[0, 2, 0, 10, 11] = np.nan
    primary_beam[0, 25, 0, 11, 10] = np.nan
    frequency = source.frequency.values.copy()
    frequency[16:] += 40 * (frequency[1] - frequency[0])
    stokes_q = np.random.default_rng(3).standard_normal(sky.shape).astype(np.float32)
    xds = xr.Dataset(
        {
            "SKY_RESIDUAL": (DIMS, np.concatenate([sky, stokes_q], axis=2)),
            "PRIMARY_BEAM": (DIMS, np.concatenate([primary_beam] * 2, axis=2)),
        },
        coords={"polarization": ["I", "Q"]},
    )
    coords = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[N_PIXELS, N_PIXELS],
        cell_size=[0.2 * RAD_PER_ARCSEC] * 2,
        frequency_coords=frequency,
        pol_coords=["I", "Q"],
        time_coords=[0.0],
    )
    modify_data_groups_xds(coords, "residual", dict(VARIABLES), "The residual.")
    # Data variables of the cube other than those the stores are made from.
    modify_data_groups_xds(
        coords,
        "swapped",
        {"sky": "PRIMARY_BEAM", "primary_beam": "SKY_RESIDUAL"},
        "The residual's roles swapped.",
    )
    store = write_zarr_image_store(
        coords, str(tmp_path_factory.mktemp("cube") / "cube.img.zarr")
    )
    create_empty_data_variables_on_disk(
        store,
        ["sky_residual", "primary_beam"],
        shape_dict=dict(coords.sizes),
        parallel_coords={
            "frequency": {"data_chunks": [[c] for c in range(N_CHANNELS)]}
        },
        compressor=None,
        double_precision=False,
        data_variable_definitions="imaging",
        image_chunking={"polarization": 1, "l": TILE[0], "m": TILE[1]},
        image_sharding={"frequency": 4},
    )
    for channel in range(N_CHANNELS):
        write_result_chunk_to_disk_sharded_skunk_works(
            store,
            ["sky_residual", "primary_beam"],
            {"frequency": {"slice": slice(channel, channel + 1)}},
            xds.isel(frequency=slice(channel, channel + 1)),
        )
    zarr.consolidate_metadata(store)
    rest_frequency = float(source.frequency.attrs["rest_frequency"]["data"])
    windows = []
    for start, stop in linear_channel_ranges(frequency):
        n_channels, first_frequency, _, channel_increment = linear_spectral_axis(
            frequency[start:stop]
        )
        windows.append(
            {
                "channel_range": [start, stop],
                "first_frequency": first_frequency,
                "channel_increment": channel_increment,
                "n_channels": n_channels,
                "moment_channel_weights": moment_channel_weights(
                    first_frequency, channel_increment, n_channels, rest_frequency
                ),
            }
        )
    return {
        "store": store,
        "xds": xds,
        "sky": sky,
        "primary_beam": primary_beam,
        "windows": windows,
        "rest_frequency": rest_frequency,
    }


@pytest.fixture(autouse=True)
def empty_registry():
    node_tasks._FIRST_ROUND_MOMENTS.clear()
    yield
    node_tasks._FIRST_ROUND_MOMENTS.clear()


def _create(
    cube, path, run_id="run", tile_shape=TILE, polarization_index=0, windows=None
):
    store = str(path)
    create_find_continuum_first_round_store(
        store,
        N_CHANNELS,
        (1, N_CHANNELS, 2, N_PIXELS, N_PIXELS),
        tile_shape,
        MERGE_TILE,
        LIMIT,
        VARIABLES,
        polarization_index,
        VALUE_DTYPES,
        run_id,
        "image_cube_single_field",
        cube["windows"] if windows is None else windows,
        cube["rest_frequency"],
    )
    return store


def _chunk(cube, store, channels, run_id, tile_shape=TILE, polarization_index=0):
    """The arguments of the imager's first round of one written chunk."""
    return (
        cube["xds"].isel(frequency=slice(channels[0], channels[-1] + 1)),
        channels,
        store,
        VARIABLES,
        polarization_index,
        LIMIT,
        tile_shape,
        VALUE_DTYPES,
        run_id,
        cube["windows"],
        find_continuum_first_round_partials_path(store),
        cube["rest_frequency"],
    )


def _write_rows(store, chunks):
    rows = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    written = write_first_round_rows(store, dict(rows, write_failed=False), N_CHANNELS)
    assert written["complete"]


def _first_round(cube, path, chunks=IN_ORDER, run_id="run", windows=None, **options):
    """A first-round store made in this process, as the imager's tasks make it."""
    store = _create(cube, path, run_id, windows=windows, **options)
    _write_rows(
        store,
        [
            find_continuum_first_round_chunk(
                *_chunk(cube, store, channels, run_id, **options)
            )
            for channels in chunks
        ],
    )
    return store


def _moments(cube, window, channels):
    """Moment partials of channels of a window where the cube is finite,
    nonzero and inside the limit, added in the order given."""
    start = cube["windows"][window]["channel_range"][0]
    weights = cube["windows"][window]["moment_channel_weights"]
    partials = new_moment_partials(MAP_SHAPE, np.float32)
    for channel in channels:
        sky = cube["sky"][:, channel : channel + 1]
        beam = cube["primary_beam"][:, channel : channel + 1]
        valid = valid_voxels(sky, primary_beam_masks(beam, LIMIT)[0]) & (sky != 0)
        index = channel - start
        accumulate_moments(partials, sky, valid, weights[index : index + 1])
    return partials


def _finalized(partials):
    """The integrated and maximum ``(l, m)`` maps of moment partials."""
    return [
        values[0, 0, 0] for values in finalize_moments(partials, MAP_SHAPE, np.float32)
    ]


def _direct(cube, window):
    return _finalized(_moments(cube, window, range(*WINDOW_RANGES[window])))


def _assert_reordered_sum(actual, expected):
    """Moment 0 summed in another order: equal to rounding, relative to the
    largest value (a sum that cancels has no relative precision)."""
    atol = 1e-14 * np.nanmax(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=atol)


def _stored_moments(store):
    return [
        zarr.open_array(os.path.join(store, name), mode="r")[:]
        for name in ("SKY_MOMENT_INTEGRATED", "SKY_MOMENT_MAXIMUM")
    ]


def _attributes(store):
    return zarr.open_group(store, mode="r").attrs["find_continuum_first_round"]


class _InProcessClient:
    """A distributed client that runs what is submitted at once."""

    def submit(self, function, *args, pure=True):
        try:
            value = SimpleNamespace(value=function(*args), error=None)
        except Exception as exc:
            value = SimpleNamespace(value=None, error=exc)

        def result(timeout=None):
            if value.error is not None:
                raise value.error
            return value.value

        return SimpleNamespace(result=result)

    def gather(self, futures):
        return [future.result() for future in futures]


def _drain_pops(monkeypatch, store, pops, windows):
    """The drain of a distributed client whose workers popped ``pops``."""
    monkeypatch.setattr(
        find_continuum_first_round, "_distributed_client", _InProcessClient
    )
    monkeypatch.setattr(
        find_continuum_first_round, "_pop_on_workers", lambda *args: pops
    )
    return drain_first_round_moments("run", store, windows)


def test_store_paths():
    assert (
        find_continuum_first_round_store_path("/d/2019.1.01463.S_v3_medium.img.zarr/")
        == "/d/2019.1.01463.S_v3_medium_find_continuum_first_round.zarr"
    )
    assert (
        find_continuum_first_round_store_path("cube.zarr")
        == "cube_find_continuum_first_round.zarr"
    )
    assert (
        find_continuum_first_round_partials_path(
            "/d/cube_find_continuum_first_round.zarr"
        )
        == "/d/cube_find_continuum_first_round_moments_partials"
    )


def test_merge_tiles_of_a_cube_whose_chunks_span_the_plane():
    """Without l and m chunking a chunk spans the plane: merge tiles are made
    of 625 px chunks, so that a merge task does not hold whole planes."""
    assert find_continuum_merge_tile_counts((11250, 11250), (11250, 11250)) == [9, 9]
    assert find_continuum_merge_tile_counts((11250, 11250), (625, 625)) == [9, 9]
    assert find_continuum_merge_tile_counts((11250, 11250), (5000, 11250)) == [2, 9]
    assert find_continuum_merge_tile_counts((64, 64), (64, 64)) == [1, 1]


def test_create_store(cube, tmp_path):
    store = _create(cube, tmp_path / "fr.zarr")
    group = zarr.open_group(store, mode="r")
    assert group.metadata.zarr_format == 3
    packed, tiles = (N_CHANNELS, N_PIXELS, 3), (N_CHANNELS, 3, 3)
    rows, maps = (N_CHANNELS,), (2, N_PIXELS, N_PIXELS)
    assert {
        name: (array.shape, array.chunks, array.dtype, array.metadata.dimension_names)
        for name, array in group.arrays()
    } == {
        "PRIMARY_BEAM_LIMIT_MASK": (packed, (1, N_PIXELS, 3), np.uint8, PACKED_DIMS),
        "PRIMARY_BEAM_SPECTRUM_MASK": (
            packed,
            (1, N_PIXELS, 3),
            np.uint8,
            PACKED_DIMS,
        ),
        "PRIMARY_BEAM_LIMIT_TILES": (tiles, tiles, bool, TILE_DIMS),
        "PRIMARY_BEAM_SPECTRUM_TILES": (tiles, tiles, bool, TILE_DIMS),
        "SKY_VALID_COUNT": (rows, rows, np.int64, ("frequency",)),
        "SKY_MINIMUM": (rows, rows, np.float64, ("frequency",)),
        "SKY_MAXIMUM": (rows, rows, np.float64, ("frequency",)),
        "PRIMARY_BEAM_VALID_COUNT": (rows, rows, np.int64, ("frequency",)),
        "PRIMARY_BEAM_MINIMUM": (rows, rows, np.float64, ("frequency",)),
        "SKY_MOMENT_INTEGRATED": (maps, (1, *MERGE_TILE), np.float64, MAP_DIMS),
        "SKY_MOMENT_MAXIMUM": (maps, (1, *MERGE_TILE), np.float32, MAP_DIMS),
    }
    for name in ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_SPECTRUM_MASK"):
        assert group[name].fill_value == 0
    for name in ("SKY_VALID_COUNT", "PRIMARY_BEAM_VALID_COUNT"):
        assert group[name].fill_value == 0
    for name in (
        "SKY_MINIMUM",
        "SKY_MAXIMUM",
        "PRIMARY_BEAM_MINIMUM",
        "SKY_MOMENT_INTEGRATED",
        "SKY_MOMENT_MAXIMUM",
    ):
        assert np.isnan(group[name].fill_value)
    assert group["SKY_MOMENT_INTEGRATED"].compressors == ()
    assert group["SKY_MOMENT_MAXIMUM"].compressors == ()
    assert dict(group.attrs) == {"find_continuum_first_round": _attributes(store)}
    assert _attributes(store) == {
        "version": 3,
        "run_id": "run",
        "written_by": "image_cube_single_field",
        "primary_beam_limit": LIMIT,
        "spectrum_primary_beam_level": 0.21,
        "polarization_index": 0,
        "variables": VARIABLES,
        "tile_shape": list(TILE),
        "merge_tile_shape": list(MERGE_TILE),
        "value_dtypes": VALUE_DTYPES,
        "windows": [
            {
                "channel_range": window["channel_range"],
                "first_frequency": window["first_frequency"],
                "channel_increment": window["channel_increment"],
                "n_channels": window["n_channels"],
            }
            for window in cube["windows"]
        ],
        "rest_frequency": cube["rest_frequency"],
        "complete": False,
        "moments_complete": [False, False],
        "n_missing": None,
        "n_duplicated": None,
    }
    assert [window["channel_range"] for window in cube["windows"]] == WINDOW_RANGES
    assert [window["n_channels"] for window in cube["windows"]] == [16, 18]


def test_drain_in_process(cube, tmp_path):
    store = _first_round(cube, tmp_path / "fr.zarr")
    result, records = drain_first_round_moments("run", store, cube["windows"])

    assert result["slot_bytes"] > 0
    assert result == {
        "moments_complete": [True, True],
        "n_missing": [0, 0],
        "n_duplicated": [0, 0],
        "slot_bytes": result["slot_bytes"],
        "n_slots": 2,
    }
    attributes = _attributes(store)
    assert attributes["moments_complete"] == [True, True]
    assert attributes["n_missing"] == attributes["n_duplicated"] == [0, 0]
    assert not os.path.exists(find_continuum_first_round_partials_path(store))
    # One slot per window, filled in channel order: the maps of a direct sum.
    integral, maximum = _stored_moments(store)
    for window in (0, 1):
        np.testing.assert_array_equal(integral[window], _direct(cube, window)[0])
        np.testing.assert_array_equal(maximum[window], _direct(cube, window)[1])
    pop = {
        "stage": "first_round_pop",
        "n_slots": 2,
        "address": None,
        "process_pid": os.getpid(),
        "thread_native_id": threading.get_native_id(),
        "worker_name": None,
        "placement_ok": True,
        "slot_bytes": result["slot_bytes"],
    }
    assert {key: records[0][key] for key in pop} == pop
    assert records[0]["hostname"] and records[0]["T_reduce_node"] >= 0
    assert not {"counts", "slots"} & set(records[0])
    assert {record["stage"] for record in records[1:]} == {"first_round_merge"}
    assert [record["window"] for record in records[1:]] == [0] * 4 + [1] * 4


def test_drain_merges_the_slots_in_sequence_order(cube, tmp_path, monkeypatch):
    """With one window in memory, alternating channels flush a slot each
    (sequences beyond 9); the slots merge in numeric sequence order."""
    monkeypatch.setattr(node_tasks, "FIRST_ROUND_WINDOWS_IN_MEMORY", 1)
    slot_names = {}

    def merge(store, partials_directory, window, slots, tile):
        slot_names[window] = list(slots)
        return node_tasks.merge_first_round_moment_tile(
            store, partials_directory, window, slots, tile
        )

    monkeypatch.setattr(
        find_continuum_first_round, "merge_first_round_moment_tile", merge
    )
    pairs = zip(range(6), range(16, 22), strict=True)
    order = [channel for pair in pairs for channel in pair]
    order += [*range(6, 16), *range(22, 34)]
    store = _first_round(cube, tmp_path / "fr.zarr", [[channel] for channel in order])
    result, _ = drain_first_round_moments("run", store, cube["windows"])

    assert result["moments_complete"] == [True, True]
    assert result["n_slots"] == 14
    sequences = {0: [0, 2, 4, 6, 8, 10, 12], 1: [1, 3, 5, 7, 9, 11, 13]}
    names = [name.split("-") for window in (0, 1) for name in slot_names[window]]
    tokens = {token for _, token, _ in names}
    assert len(tokens) == 1 and len(tokens.pop()) == 32
    assert {
        window: [(int(pid), int(sequence)) for pid, _, sequence in parts]
        for window, parts in (
            (w, [n.split("-") for n in slot_names[w]]) for w in (0, 1)
        )
    } == {
        window: [(os.getpid(), sequence) for sequence in numbers]
        for window, numbers in sequences.items()
    }
    leaves = {
        0: [[channel] for channel in range(6)] + [range(6, 16)],
        1: [[channel] for channel in range(16, 22)] + [range(22, 34)],
    }
    integral, maximum = _stored_moments(store)
    for window, channel_lists in leaves.items():
        merged = _finalized(
            merge_moment_partials_in_tree_order(
                _moments(cube, window, channels) for channels in channel_lists
            )
        )
        np.testing.assert_array_equal(integral[window], merged[0])
        np.testing.assert_array_equal(maximum[window], _direct(cube, window)[1])
        _assert_reordered_sum(integral[window], _direct(cube, window)[0])


def test_task_priorities_run_the_shard_waves_one_window_at_a_time(
    cube, tmp_path, monkeypatch
):
    """The shard waves of the first window come before those of the second:
    a process holding one window that takes the tasks in priority order saves
    one slot per window, where the waves over the whole cube make it save one
    at every change of window."""
    from astroviper.node_tasks.imaging.utils import compute_shard_task_priorities

    monkeypatch.setattr(node_tasks, "FIRST_ROUND_WINDOWS_IN_MEMORY", 1)
    mapping = {
        channel: {"task_coords": {"frequency": {"slice": slice(channel, channel + 1)}}}
        for channel in range(N_CHANNELS)
    }
    waves = compute_shard_task_priorities(cube["store"], "sky_residual", mapping)
    priorities = find_continuum_first_round_task_priorities(
        waves, mapping, cube["windows"]
    )

    def dispatched(task_priorities):
        return sorted(task_priorities, key=lambda task: (-task_priorities[task], task))

    # Shards of four channels: the waves of [0, 16), then those of [16, 34).
    assert dispatched(priorities) == [
        channel
        for start, stop in WINDOW_RANGES
        for within in range(4)
        for channel in range(start + within, stop, 4)
    ]
    n_slots = {}
    for run_id, task_priorities in (("waves", waves), ("windows", priorities)):
        store = _first_round(
            cube,
            tmp_path / f"{run_id}.zarr",
            [[channel] for channel in dispatched(task_priorities)],
            run_id=run_id,
        )
        result, _ = drain_first_round_moments(run_id, store, cube["windows"])
        assert result["moments_complete"] == [True, True]
        n_slots[run_id] = result["n_slots"]
        integral, maximum = _stored_moments(store)
        for window in (0, 1):
            np.testing.assert_array_equal(maximum[window], _direct(cube, window)[1])
            _assert_reordered_sum(integral[window], _direct(cube, window)[0])
    assert n_slots == {"waves": 8, "windows": 2}


@pytest.mark.parametrize("change", ["missing", "duplicated"])
def test_drain_leaves_a_window_with_a_coverage_error_incomplete(cube, tmp_path, change):
    """Channel 3 imaged where its moments were lost, or channel 18 added to
    the moments twice (a task rerun after its result was used): only that
    window is left incomplete."""
    store = _create(cube, tmp_path / "fr.zarr")
    chunks = [np.arange(0, 3), [3], np.arange(4, 16), np.arange(16, 34)]
    run_ids = ["run", None if change == "missing" else "run", "run", "run"]
    _write_rows(
        store,
        [
            find_continuum_first_round_chunk(*_chunk(cube, store, channels, run_id))
            for channels, run_id in zip(chunks, run_ids, strict=True)
        ],
    )
    if change == "duplicated":
        find_continuum_first_round_chunk(*_chunk(cube, store, [18], "run"))
    result, records = drain_first_round_moments("run", store, cube["windows"])

    coverage = {
        "missing": {"n_missing": [1, 0], "n_duplicated": [0, 0]},
        "duplicated": {"n_missing": [0, 0], "n_duplicated": [0, 1]},
    }[change]
    good = 1 if change == "missing" else 0
    complete = [window == good for window in (0, 1)]
    attributes = _attributes(store)
    for drained in (result, attributes):
        assert drained["moments_complete"] == complete
        assert {key: drained[key] for key in coverage} == coverage
    integral, maximum = _stored_moments(store)
    np.testing.assert_array_equal(integral[good], _direct(cube, good)[0])
    np.testing.assert_array_equal(maximum[good], _direct(cube, good)[1])
    assert np.isnan(integral[1 - good]).all() and np.isnan(maximum[1 - good]).all()
    assert {record["window"] for record in records[1:]} == {good}
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_drain_merges_the_slots_of_processes_with_the_same_pid(
    cube, tmp_path, monkeypatch
):
    """Worker processes on two hosts (or in two PID namespaces) can have the
    same pid; their slots in the shared directory have different names, and
    the slots of both are merged in address order."""
    import dask.distributed

    store = _create(cube, tmp_path / "fr.zarr")
    partials_directory = find_continuum_first_round_partials_path(store)
    os.makedirs(partials_directory)
    monkeypatch.setattr(node_tasks.os, "getpid", lambda: 4242)
    halves = [(range(0, 8), range(8, 16)), (range(16, 25), range(25, 34))]
    chunks, pops = [], []
    for k, address in enumerate(("tcp://a", "tcp://b")):
        worker = SimpleNamespace(address=address, name=address)
        monkeypatch.setattr(
            dask.distributed, "get_worker", lambda worker=worker: worker
        )
        for channels in (halves[0][k], halves[1][k]):
            chunks.append(
                find_continuum_first_round_chunk(
                    *_chunk(cube, store, np.asarray(channels), "run")
                )
            )
        pops.append(
            node_tasks.pop_first_round_moments("run", address, partials_directory)
        )
    _write_rows(store, chunks)
    result, _ = _drain_pops(monkeypatch, store, pops, cube["windows"])

    assert result["moments_complete"] == [True, True]
    names = [{name for _, name in pop["slots"]} for pop in pops]
    assert len(names[0]) == len(names[1]) == 2 and not names[0] & names[1]
    integral, maximum = _stored_moments(store)
    for window, channel_lists in enumerate(halves):
        merged = _finalized(
            merge_moment_partials_in_tree_order(
                _moments(cube, window, channels) for channels in channel_lists
            )
        )
        np.testing.assert_array_equal(integral[window], merged[0])
        np.testing.assert_array_equal(maximum[window], _direct(cube, window)[1])


def test_drain_leaves_a_window_with_a_slot_listed_twice_incomplete(
    cube, tmp_path, monkeypatch
):
    """A slot that two pops list is not merged twice: its window is left
    incomplete."""
    store = _first_round(cube, tmp_path / "fr.zarr")
    pop = node_tasks.pop_first_round_moments(
        "run", None, find_continuum_first_round_partials_path(store)
    )
    copy = dict(pop, counts=np.zeros_like(pop["counts"]), slots=pop["slots"][:1])
    result, records = _drain_pops(monkeypatch, store, [pop, copy], cube["windows"])

    assert result["moments_complete"] == [False, True]
    assert result["n_missing"] == result["n_duplicated"] == [0, 0]
    assert {record["window"] for record in records[2:]} == {1}
    integral, maximum = _stored_moments(store)
    assert np.isnan(integral[0]).all() and np.isnan(maximum[0]).all()
    np.testing.assert_array_equal(maximum[1], _direct(cube, 1)[1])


def test_drain_leaves_a_window_whose_slot_was_not_saved_incomplete(
    cube, tmp_path, monkeypatch
):
    save = np.save

    def save_or_fail(stream, values):
        if f"{os.sep}W0{os.sep}" in stream.name:
            raise OSError(errno.ENOSPC, "No space left on device")
        save(stream, values)

    monkeypatch.setattr(node_tasks.np, "save", save_or_fail)
    store = _first_round(cube, tmp_path / "fr.zarr")
    result, records = drain_first_round_moments("run", store, cube["windows"])

    assert result["moments_complete"] == [False, True]
    assert result["n_missing"] == result["n_duplicated"] == [0, 0]
    assert records[0]["lost_windows"] == [0]
    assert _attributes(store)["moments_complete"] == [False, True]
    integral, maximum = _stored_moments(store)
    assert np.isnan(integral[0]).all() and np.isnan(maximum[0]).all()
    np.testing.assert_array_equal(integral[1], _direct(cube, 1)[0])
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


@pytest.mark.parametrize("with_client", [False, True])
def test_drain_leaves_a_window_with_a_tile_that_fails_to_merge_incomplete(
    cube, tmp_path, monkeypatch, with_client
):
    """A Lustre error in one merge task of window 1: the other window is
    merged, window 1 is left incomplete, the slots are removed."""

    def merge(store, partials_directory, window, slots, tile):
        if window == 1 and tile == [[12, 20], [0, 12]]:
            raise OSError(errno.EIO, "Input/output error")
        return node_tasks.merge_first_round_moment_tile(
            store, partials_directory, window, slots, tile
        )

    monkeypatch.setattr(
        find_continuum_first_round, "merge_first_round_moment_tile", merge
    )
    store = _first_round(cube, tmp_path / "fr.zarr")
    if with_client:
        pop = node_tasks.pop_first_round_moments(
            "run", None, find_continuum_first_round_partials_path(store)
        )
        result, records = _drain_pops(monkeypatch, store, [pop], cube["windows"])
    else:
        result, records = drain_first_round_moments("run", store, cube["windows"])

    assert result["moments_complete"] == [True, False]
    assert _attributes(store)["moments_complete"] == [True, False]
    assert [record["window"] for record in records[1:]] == [0] * 4 + [1] * 3
    integral, maximum = _stored_moments(store)
    np.testing.assert_array_equal(integral[0], _direct(cube, 0)[0])
    np.testing.assert_array_equal(maximum[0], _direct(cube, 0)[1])
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_drain_refuses_pops_that_ran_elsewhere(cube, tmp_path, monkeypatch):
    def misplaced(run_id, address, partials_directory):
        pop = node_tasks.pop_first_round_moments(run_id, address, partials_directory)
        return dict(pop, placement_ok=False)

    monkeypatch.setattr(
        find_continuum_first_round, "pop_first_round_moments", misplaced
    )
    store = _first_round(cube, tmp_path / "fr.zarr")
    result, records = drain_first_round_moments("run", store, cube["windows"])

    assert result["moments_complete"] == [False, False]
    assert result["n_missing"] == result["n_duplicated"] == [0, 0]
    assert [record["stage"] for record in records] == ["first_round_pop"]
    assert all(np.isnan(maps).all() for maps in _stored_moments(store))
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_a_pop_whose_result_cannot_be_fetched_counts_as_missing(monkeypatch):
    """A worker lost after its pop finished: its result can never be
    fetched (the pop is pinned to it), so it is cancelled, not waited for."""
    import distributed

    class Future:
        status = "finished"

        def __init__(self, value):
            self.value = value

        def result(self, timeout=None):
            if self.value is None:
                raise TimeoutError("lost")
            return self.value

    class Client:
        cancelled = []

        def nthreads(self):
            return {"tcp://b": 1, "tcp://a": 1}

        def submit(self, function, run_id, address, partials_directory, **kwargs):
            return Future(None if address == "tcp://b" else {"address": address})

        def gather(self, futures):
            return [future.result() for future in futures]

        def cancel(self, futures):
            self.cancelled.extend(futures)

    monkeypatch.setattr(distributed, "wait", lambda futures, timeout: None)
    client = Client()

    assert find_continuum_first_round._pop_on_workers(
        client, "run", "partials", 600.0
    ) == [{"address": "tcp://a"}]
    assert [future.value for future in client.cancelled] == [None]


def test_drain_on_a_local_cluster(cube, tmp_path):
    """Pops run on their workers and the slots of two workers merge in
    address order; a restarted or busy worker leaves its window incomplete."""
    from distributed import Client, Event, LocalCluster

    def wait(name):
        from distributed import Event

        return Event(name).wait(60)

    def image(client, store, assignment, run_id):
        futures = [
            client.submit(
                find_continuum_first_round_chunk,
                *_chunk(cube, store, channels, run_id),
                workers=[address],
                allow_other_workers=False,
                pure=False,
            )
            for address, channels in assignment
        ]
        _write_rows(store, client.gather(futures))

    with (
        LocalCluster(
            n_workers=2,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GiB",
            dashboard_address=None,
        ) as cluster,
        Client(cluster) as client,
    ):
        first, second = sorted(client.nthreads())
        pids = client.run(os.getpid)
        store = _create(cube, tmp_path / "fr.zarr")
        assignment = [
            (first, np.arange(0, 8)),
            (second, np.arange(8, 16)),
            (first, np.arange(16, 25)),
            (second, np.arange(25, 34)),
        ]
        image(client, store, assignment, "run")
        result, records = drain_first_round_moments("run", store, cube["windows"])

        assert result["moments_complete"] == [True, True]
        assert result["n_slots"] == 4
        pops = [record for record in records if record["stage"] == "first_round_pop"]
        assert [(pop["address"], pop["process_pid"]) for pop in pops] == [
            (first, pids[first]),
            (second, pids[second]),
        ]
        assert all(pop["placement_ok"] and pop["n_slots"] == 2 for pop in pops)
        merges = [record for record in records if record["stage"] != "first_round_pop"]
        assert len(merges) == 8
        assert {merge["process_pid"] for merge in merges} <= set(pids.values())
        integral, maximum = _stored_moments(store)
        halves = [(range(0, 8), range(8, 16)), (range(16, 25), range(25, 34))]
        for window, channel_lists in enumerate(halves):
            merged = _finalized(
                merge_moment_partials_in_tree_order(
                    _moments(cube, window, channels) for channels in channel_lists
                )
            )
            np.testing.assert_array_equal(integral[window], merged[0])
            np.testing.assert_array_equal(maximum[window], _direct(cube, window)[1])
            _assert_reordered_sum(integral[window], _direct(cube, window)[0])

        # The moments of a restarted worker are lost; the pop of a busy worker
        # times out and is cancelled.
        for case in ("restarted", "busy"):
            idle, other = sorted(client.nthreads())
            store = _create(cube, tmp_path / f"{case}.zarr", case)
            assignment = [(idle, np.arange(0, 16)), (idle, np.arange(16, 25))]
            image(client, store, [*assignment, (other, np.arange(25, 34))], case)
            if case == "restarted":
                client.restart_workers([other])
                result, _ = drain_first_round_moments(case, store, cube["windows"])
            else:
                release = Event("release")
                blocker = client.submit(wait, "release", workers=[other], pure=False)
                result, _ = drain_first_round_moments(
                    case, store, cube["windows"], timeout_seconds=3.0
                )
                assert blocker.status == "pending"
                release.set()
                assert blocker.result()
            assert result["moments_complete"] == [True, False]
            assert result["n_missing"] == [0, 9] and result["n_duplicated"] == [0, 0]
            integral, maximum = _stored_moments(store)
            np.testing.assert_array_equal(integral[0], _direct(cube, 0)[0])
            np.testing.assert_array_equal(maximum[0], _direct(cube, 0)[1])
            assert np.isnan(integral[1]).all() and np.isnan(maximum[1]).all()


def test_binding_and_validation(cube, tmp_path):
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None)
    with pytest.raises(ValueError, match="not made from the cube"):
        validate_find_continuum_first_round(store, cube["store"], LIMIT, [0, 16])
    bind_find_continuum_first_round_store(store, cube["store"])

    binding = zarr.open_group(store, mode="r").attrs["image_binding"]
    assert binding["image_store_zarr_json_mtime_ns"] == (
        os.stat(os.path.join(cube["store"], "zarr.json")).st_mtime_ns
    )
    assert binding["shape"] == [1, N_CHANNELS, 2, N_PIXELS, N_PIXELS]
    assert sorted(binding) == [
        "frequency_sha1",
        "image_store_zarr_json_mtime_ns",
        "l_sha1",
        "m_sha1",
        "shape",
    ]
    for index, window in enumerate(WINDOW_RANGES):
        assert validate_find_continuum_first_round(
            store, cube["store"], LIMIT, window
        ) == FirstRoundStoreInputs(store, index, False, "image_cube_single_field")
    # A part of a window: its rows, not the window's moments.
    assert validate_find_continuum_first_round(
        store, cube["store"], LIMIT, np.array([18, 30])
    ) == FirstRoundStoreInputs(store, None, False, "image_cube_single_field")
    with pytest.raises(ValueError, match=r"holds channels \[\[0, 16\], \[16, 34\]\]"):
        validate_find_continuum_first_round(store, cube["store"], LIMIT, [8, 24])
    with pytest.raises(ValueError, match="primary_beam_limit"):
        validate_find_continuum_first_round(store, cube["store"], 0.3, [0, 16])
    with pytest.raises(ValueError, match="missing"):
        validate_find_continuum_first_round(
            str(tmp_path / "absent.zarr"), cube["store"], LIMIT, [0, 16]
        )
    incomplete = _create(cube, tmp_path / "incomplete.zarr", None)
    with pytest.raises(ValueError, match="incomplete"):
        validate_find_continuum_first_round(incomplete, cube["store"], LIMIT, [0, 16])


def test_validation_tells_which_window_has_its_moments(cube, tmp_path):
    """The drain completed the moments of window 1 only: the store serves the
    rows of both windows and the moments of window 1."""
    store = _create(cube, tmp_path / "fr.zarr")
    chunks = [np.arange(0, 16), np.arange(16, 34)]
    run_ids = [None, "run"]
    _write_rows(
        store,
        [
            find_continuum_first_round_chunk(*_chunk(cube, store, channels, run_id))
            for channels, run_id in zip(chunks, run_ids, strict=True)
        ],
    )
    result, _ = drain_first_round_moments("run", store, cube["windows"])
    assert result["moments_complete"] == [False, True]
    bind_find_continuum_first_round_store(store, cube["store"])

    assert [
        validate_find_continuum_first_round(store, cube["store"], LIMIT, window)
        for window in WINDOW_RANGES
    ] == [
        FirstRoundStoreInputs(store, 0, False, "image_cube_single_field"),
        FirstRoundStoreInputs(store, 1, True, "image_cube_single_field"),
    ]


def test_validation_refuses_a_store_of_version_1(cube, tmp_path):
    """A complete store of the earlier format, bound to the cube, is refused
    with the reason."""
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None)
    bind_find_continuum_first_round_store(store, cube["store"])
    group = zarr.open_group(store, mode="r+")
    group.attrs["find_continuum_first_round"] = dict(_attributes(store), version=1)
    with pytest.raises(
        ValueError, match="has version 1, not 3: make the first round again"
    ):
        validate_find_continuum_first_round(store, cube["store"], LIMIT, [0, 16])


@pytest.mark.parametrize(
    "options", [{"tile_shape": (4, 4)}, {"polarization_index": 1}], ids=str
)
def test_validation_refuses_a_store_of_other_settings(cube, tmp_path, options):
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None, **options)
    bind_find_continuum_first_round_store(store, cube["store"])
    with pytest.raises(ValueError, match=next(iter(options))):
        validate_find_continuum_first_round(store, cube["store"], LIMIT, [0, 16])


def test_validation_refuses_a_store_of_other_data_variables(cube, tmp_path):
    """The store must have been made from the data variables of the call's
    data group, as find_continuum_ultra requires, so that a caller choosing
    the store by the validation is not refused later."""
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None)
    bind_find_continuum_first_round_store(store, cube["store"])
    assert validate_find_continuum_first_round(
        store, cube["store"], LIMIT, [0, 16], "residual"
    ) == FirstRoundStoreInputs(store, 0, False, "image_cube_single_field")
    with pytest.raises(ValueError, match="made with variables"):
        validate_find_continuum_first_round(
            store, cube["store"], LIMIT, [0, 16], "swapped"
        )
    with pytest.raises(ValueError, match="no sky or no primary_beam"):
        validate_find_continuum_first_round(
            store, cube["store"], LIMIT, [0, 16], "absent"
        )


def test_validation_refuses_a_rewritten_cube(cube, tmp_path):
    image_store = str(tmp_path / "cube.img.zarr")
    shutil.copytree(cube["store"], image_store)
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None)
    bind_find_continuum_first_round_store(store, image_store)
    validate_find_continuum_first_round(store, image_store, LIMIT, [0, 16])
    zarr_json = os.path.join(image_store, "zarr.json")
    stat = os.stat(zarr_json)
    os.utime(zarr_json, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    with pytest.raises(ValueError, match="image_store_zarr_json_mtime_ns differs"):
        validate_find_continuum_first_round(store, image_store, LIMIT, [0, 16])
