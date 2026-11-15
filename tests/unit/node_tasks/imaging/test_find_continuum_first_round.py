"""Tests of the node tasks of findContinuum's first round, made by the imager."""

import errno
import itertools
import os
import threading

import numpy as np
import pytest
import xarray as xr
import zarr

import astroviper.node_tasks.imaging.find_continuum_first_round as node_tasks
from astroviper.node_tasks.imaging.find_continuum_first_round import (
    add_first_round_moments,
    find_continuum_first_round_chunk,
    merge_first_round_moment_tile,
    pop_first_round_moments,
    read_first_round_moment_maps,
    read_first_round_rows,
    write_first_round_rows,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
    accumulate_moments,
    finalize_moments,
    first_round_channel_rows,
    new_moment_partials,
    primary_beam_masks,
    tile_occupancy,
    unpack_plane_masks,
    valid_voxels,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    FirstRoundRows,
)

VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
DIMS = ["time", "frequency", "polarization", "l", "m"]
MASKS = ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_SPECTRUM_MASK")
ROWS = {
    "SKY_VALID_COUNT": (np.int64, 0),
    "SKY_MINIMUM": (np.float64, np.nan),
    "SKY_MAXIMUM": (np.float64, np.nan),
    "PRIMARY_BEAM_VALID_COUNT": (np.int64, 0),
    "PRIMARY_BEAM_MINIMUM": (np.float64, np.nan),
}


def _first_round_store(path, n_channels, shape, tile, windows):
    """An empty first-round store in the layout the node tasks read and write."""
    group = zarr.open_group(str(path), mode="w")
    n_packed = -(-shape[1] // 8)
    n_tiles = tuple(-(-n // t) for n, t in zip(shape, tile, strict=True))
    for name in MASKS:
        group.create_array(
            name,
            shape=(n_channels, shape[0], n_packed),
            chunks=(1, shape[0], n_packed),
            dtype=np.uint8,
            fill_value=0,
            compressors=zarr.codecs.ZstdCodec(level=1),
            dimension_names=("frequency", "l", "m_packed"),
        )
    for name in ("PRIMARY_BEAM_LIMIT_TILES", "PRIMARY_BEAM_SPECTRUM_TILES"):
        group.create_array(
            name,
            shape=(n_channels, *n_tiles),
            chunks=(n_channels, *n_tiles),
            dtype=bool,
            fill_value=False,
            compressors=zarr.codecs.ZstdCodec(level=1),
            dimension_names=("frequency", "l_tile", "m_tile"),
        )
    for name, (dtype, fill_value) in ROWS.items():
        group.create_array(
            name,
            shape=(n_channels,),
            chunks=(n_channels,),
            dtype=dtype,
            fill_value=fill_value,
            compressors=zarr.codecs.ZstdCodec(level=1),
            dimension_names=("frequency",),
        )
    for name, dtype in (
        ("SKY_MOMENT_INTEGRATED", np.float64),
        ("SKY_MOMENT_MAXIMUM", np.float32),
    ):
        group.create_array(
            name,
            shape=(len(windows), *shape),
            chunks=(1, *tile),
            dtype=dtype,
            fill_value=np.nan,
            compressors=None,
            dimension_names=("window", "l", "m"),
        )
    group.attrs["find_continuum_first_round"] = {
        "version": 3,
        "windows": [{"channel_range": list(window)} for window in windows],
        "complete": False,
    }
    return str(path)


def _random_rows(rng, channels):
    n = len(channels)
    return {
        "channels": np.asarray(channels, dtype=np.int32),
        "sky_valid_count": rng.integers(0, 100, n),
        "sky_minimum": rng.standard_normal(n),
        "sky_maximum": rng.standard_normal(n),
        "primary_beam_valid_count": rng.integers(0, 100, n),
        "primary_beam_minimum": rng.standard_normal(n),
        "tiles": rng.random((n, 2, 2, 2)) > 0.5,
    }


@pytest.mark.parametrize("change", [None, "duplicate", "missing", "failed"])
def test_first_round_rows_are_complete_with_every_window_channel_once(tmp_path, change):
    store = _first_round_store(tmp_path / "fr.zarr", 30, (16, 16), (8, 8), [[10, 20]])
    rows = _random_rows(np.random.default_rng(23), range(10, 20))
    if change == "duplicate":
        rows = {key: np.concatenate([value, value[3:4]]) for key, value in rows.items()}
    elif change == "missing":
        rows = {key: value[1:] for key, value in rows.items()}
    rows["write_failed"] = change == "failed"
    result = write_first_round_rows(store, rows, 30)
    assert result["complete"] is (change is None)
    attributes = zarr.open_group(store, mode="r").attrs["find_continuum_first_round"]
    assert attributes["complete"] is (change is None)
    assert result["written_channels"] == sorted(set(rows["channels"].tolist()))
    written = np.array(result["written_channels"])
    first = np.unique(rows["channels"], return_index=True)[1]
    for field in FirstRoundRows._fields:
        stored = zarr.open_array(os.path.join(store, field.upper()), mode="r")[:]
        np.testing.assert_array_equal(stored[written], rows[field][first])
        unwritten = np.delete(stored, written)
        if field.endswith("_count"):
            assert stored.dtype == np.int64 and not unwritten.any()
        else:
            assert stored.dtype == np.float64 and np.isnan(unwritten).all()
    spectrum = zarr.open_array(
        os.path.join(store, "PRIMARY_BEAM_SPECTRUM_TILES"), mode="r"
    )[:]
    np.testing.assert_array_equal(spectrum[written], rows["tiles"][first, 1])
    assert not np.delete(spectrum, written, axis=0).any()


def test_read_first_round_rows_of_a_window(tmp_path):
    store = _first_round_store(
        tmp_path / "fr.zarr", 30, (16, 16), (8, 8), [[0, 10], [10, 30]]
    )
    rows = _random_rows(np.random.default_rng(5), range(30))
    rows["sky_minimum"][12] = rows["sky_maximum"][12] = np.nan
    write_first_round_rows(store, dict(rows, write_failed=False), 30)

    window_rows, counters = read_first_round_rows(store, [10, 30])
    assert isinstance(window_rows, FirstRoundRows)
    for field in FirstRoundRows._fields:
        values = getattr(window_rows, field)
        assert values.shape == (20,) and values.dtype == rows[field].dtype
        np.testing.assert_array_equal(values, rows[field][10:30])
    assert counters["units_read"] == 5 and counters["bytes"] > 0


# --------------------------------------------------------------------------
# The imager's first round: chunks, worker-local moments per window
# --------------------------------------------------------------------------
N_REGISTRY, REGISTRY_SHAPE, REGISTRY_TILE = 12, (40, 24), (20, 12)
MAP_SHAPE = (1, 1, 1, *REGISTRY_SHAPE)
REGISTRY_TILES = [
    [[l0, l0 + 20], [m0, m0 + 12]] for l0, m0 in itertools.product((0, 20), (0, 12))
]


@pytest.fixture
def registry_cube():
    rng = np.random.default_rng(13)
    sky = rng.standard_normal((1, N_REGISTRY, 1, *REGISTRY_SHAPE)).astype(np.float32)
    sky[0, 2, 0, :5] = 0.0
    sky[0, 3, 0, 7, 7] = np.nan
    sky[0, 4, 0, 9, 9] = -0.0
    valid = valid_voxels(sky, rng.random(sky.shape) > 0.2) & (sky != 0)
    # A corner outside the limit in every channel.
    valid[..., 35:, 20:] = False
    windows = [
        {
            "channel_range": [start, start + 6],
            "moment_channel_weights": rng.uniform(0.5, 1.5, 6),
        }
        for start in (0, 6)
    ]
    yield sky, valid, windows
    node_tasks._FIRST_ROUND_MOMENTS.clear()


def _window_moments(sky, valid, window):
    """The integrated and maximum moments of a window's channels at the
    voxels of ``valid``, added in channel order."""
    start, stop = window["channel_range"]
    partials = new_moment_partials(MAP_SHAPE, sky.dtype)
    accumulate_moments(
        partials,
        sky[:, start:stop],
        valid[:, start:stop],
        window["moment_channel_weights"],
    )
    return [
        values[0, 0, 0] for values in finalize_moments(partials, MAP_SHAPE, sky.dtype)
    ]


def _add(run_id, windows, partials, sky, valid, channels):
    planes = slice(channels[0], channels[-1] + 1)
    add_first_round_moments(
        run_id,
        windows,
        MAP_SHAPE,
        partials,
        np.asarray(channels),
        sky[:, planes],
        valid[:, planes],
    )


def _drain(pops, path, partials_directory, windows):
    """Merge the slots of every window tile by tile, as the driver's drain does."""
    store = _first_round_store(
        path,
        N_REGISTRY,
        REGISTRY_SHAPE,
        REGISTRY_TILE,
        [window["channel_range"] for window in windows],
    )
    slots = {index: [] for index in range(len(windows))}
    for pop in pops:
        for window, name in pop["slots"]:
            slots[window].append(name)
    records = [
        merge_first_round_moment_tile(store, partials_directory, window, names, tile)
        for window, names in slots.items()
        for tile in REGISTRY_TILES
    ]
    maps = [
        zarr.open_array(os.path.join(store, name), mode="r")[:]
        for name in ("SKY_MOMENT_INTEGRATED", "SKY_MOMENT_MAXIMUM")
    ]
    return maps, records


@pytest.mark.parametrize("windows_in_memory", [2, 1])
def test_registry_holds_windows_and_flushes_the_least_recent(
    registry_cube, tmp_path, monkeypatch, windows_in_memory
):
    sky, valid, windows = registry_cube
    monkeypatch.setattr(node_tasks, "FIRST_ROUND_WINDOWS_IN_MEMORY", windows_in_memory)
    partials = str(tmp_path / "partials")
    for start in (0, 6, 2, 8, 4, 10):
        _add("run", windows, partials, sky, valid, [start, start + 1])
    pop = pop_first_round_moments("run", None, partials)
    pid = os.getpid()
    assert pop["placement_ok"] and pop["process_pid"] == pid
    assert pop["thread_native_id"] == threading.get_native_id()
    assert pop["hostname"] and pop["worker_name"] is None
    assert 0 <= pop["T_reduce_node"] == pop["end_unixtime"] - pop["start_unixtime"]
    np.testing.assert_array_equal(pop["counts"], np.ones(N_REGISTRY, np.uint16))
    expected_windows = [0, 1] if windows_in_memory == 2 else [0, 1, 0, 1, 0, 1]
    token = pop["slots"][0][1].split("-")[1]
    assert len(token) == 32
    assert pop["slots"] == [
        [window, f"{pid}-{token}-{sequence}"]
        for sequence, window in enumerate(expected_windows)
    ]
    files = [
        os.path.join(partials, f"W{window}", f"{name}.{field}.npy")
        for window, name in pop["slots"]
        for field in ("integrated", "maximum")
    ]
    assert pop["slot_bytes"] == sum(os.path.getsize(path) for path in files)
    assert "run" not in node_tasks._FIRST_ROUND_MOMENTS

    (integral, maximum), records = _drain(
        [pop], tmp_path / "fr.zarr", partials, windows
    )
    for index, window in enumerate(windows):
        expected = _window_moments(sky, valid, window)
        assert np.isfinite(expected[0]).any() and np.isnan(expected[0]).any()
        np.testing.assert_array_equal(maximum[index], expected[1])
        if windows_in_memory == 2:
            np.testing.assert_array_equal(integral[index], expected[0])
        else:
            np.testing.assert_allclose(integral[index], expected[0], rtol=1e-12)
            np.testing.assert_array_equal(
                np.isnan(integral[index]), np.isnan(expected[0])
            )
    assert {record["stage"] for record in records} == {"first_round_merge"}
    assert all(record["bytes_read"] > 0 for record in records)


def test_merge_without_slots_writes_empty_maps(registry_cube, tmp_path):
    _, _, windows = registry_cube
    (integral, maximum), records = _drain(
        [], tmp_path / "fr.zarr", str(tmp_path / "partials"), windows
    )
    assert np.isnan(integral).all() and np.isnan(maximum).all()
    assert all(record["bytes_read"] == 0 for record in records)


def test_registry_counts_duplicated_and_missing_channels(registry_cube, tmp_path):
    sky, valid, windows = registry_cube
    partials = str(tmp_path / "partials")
    for channel in [0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11]:
        _add("run", windows, partials, sky, valid, [channel])
    counts = pop_first_round_moments("run", None, partials)["counts"]
    assert counts[3] == 2 and counts[4] == 0
    assert (np.delete(counts, [3, 4]) == 1).all()


def test_a_new_run_drops_the_old_entry(registry_cube, tmp_path):
    sky, valid, windows = registry_cube
    partials = str(tmp_path / "partials")
    for run, channel in (("old", 0), ("new", 1)):
        _add(run, windows, partials, sky, valid, [channel])
    old = pop_first_round_moments("old", "tcp://127.0.0.1:1", partials)
    assert old["counts"] is None and old["slots"] == [] and old["slot_bytes"] == 0
    assert not old["placement_ok"]
    new = pop_first_round_moments("new", None, partials)
    assert new["counts"].tolist() == [0, 1] + [0] * (N_REGISTRY - 2)


def test_concurrent_additions_from_threads(registry_cube, tmp_path):
    sky, valid, windows = registry_cube
    partials = str(tmp_path / "partials")

    def add(first):
        for channel in range(first, N_REGISTRY, 4):
            _add("run", windows, partials, sky, valid, [channel])

    threads = [threading.Thread(target=add, args=(first,)) for first in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    pop = pop_first_round_moments("run", None, partials)
    np.testing.assert_array_equal(pop["counts"], np.ones(N_REGISTRY, np.uint16))
    (integral, maximum), _ = _drain([pop], tmp_path / "fr.zarr", partials, windows)
    for index, window in enumerate(windows):
        expected = _window_moments(sky, valid, window)
        np.testing.assert_array_equal(maximum[index], expected[1])
        np.testing.assert_allclose(integral[index], expected[0], rtol=1e-12)


def test_first_round_chunks_write_masks_rows_and_moments(registry_cube, tmp_path):
    """The imager's chunks of two channels, Stokes I at index 1 of float64
    planes, give the first round of the whole cube, and the store's rows and
    moments read back."""
    sky, _, windows = registry_cube
    l_index, m_index = np.indices(REGISTRY_SHAPE)
    radius2 = (l_index - 19.5) ** 2 + (m_index - 11.5) ** 2
    beam = np.exp(-radius2 / 300.0)[None, None, None] * np.ones(sky.shape)
    beam[0, 5, 0, 3, 3] = np.nan
    values = sky.astype(np.float64) * (1 + 1e-9)
    chunk_xds = xr.Dataset(
        {
            "SKY_RESIDUAL": (DIMS, np.concatenate([2 * values, values], axis=2)),
            "PRIMARY_BEAM": (DIMS, np.concatenate([beam, beam], axis=2)),
        },
        coords={"polarization": ["Q", "I"]},
    )
    store = _first_round_store(
        tmp_path / "fr.zarr",
        N_REGISTRY,
        REGISTRY_SHAPE,
        REGISTRY_TILE,
        [window["channel_range"] for window in windows],
    )
    partials = str(tmp_path / "partials")
    chunks = [
        find_continuum_first_round_chunk(
            chunk_xds.isel(frequency=slice(start, start + 2)),
            np.arange(start, start + 2),
            store,
            VARIABLES,
            1,
            0.2,
            REGISTRY_TILE,
            {"sky": "float32", "primary_beam": "float32"},
            "imager",
            windows,
            partials,
            230.0e9,
        )
        for start in range(0, N_REGISTRY, 2)
    ]
    rows = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    stokes_i, beam_i = values.astype(np.float32), beam.astype(np.float32)
    limit_mask, spectrum_mask = primary_beam_masks(beam_i, 0.2)
    assert rows["channels"].dtype == np.int32
    np.testing.assert_array_equal(rows["channels"], np.arange(N_REGISTRY))
    expected_rows = first_round_channel_rows(stokes_i, beam_i, limit_mask)
    for field in FirstRoundRows._fields:
        assert rows[field].dtype == expected_rows[field].dtype
        np.testing.assert_array_equal(rows[field], expected_rows[field][0, :, 0])
    tiles = [
        tile_occupancy(mask, REGISTRY_TILE)[0, :, 0]
        for mask in (limit_mask, spectrum_mask)
    ]
    np.testing.assert_array_equal(rows["tiles"], np.stack(tiles, axis=1))
    for name, mask in zip(MASKS, (limit_mask, spectrum_mask), strict=True):
        stored = zarr.open_array(os.path.join(store, name), mode="r")[:]
        np.testing.assert_array_equal(
            unpack_plane_masks(stored, REGISTRY_SHAPE[1]), mask[0, :, 0]
        )
    assert write_first_round_rows(store, dict(rows, write_failed=False), N_REGISTRY)[
        "complete"
    ]
    stored_rows, _ = read_first_round_rows(store, [0, N_REGISTRY])
    for field in FirstRoundRows._fields:
        np.testing.assert_array_equal(getattr(stored_rows, field), rows[field])

    pop = pop_first_round_moments("imager", None, partials)
    np.testing.assert_array_equal(pop["counts"], np.ones(N_REGISTRY, np.uint16))
    (integral, maximum), _ = _drain([pop], tmp_path / "drain.zarr", partials, windows)
    valid = valid_voxels(stokes_i, limit_mask) & (stokes_i != 0)
    for index, window in enumerate(windows):
        expected = _window_moments(stokes_i, valid, window)
        np.testing.assert_array_equal(integral[index], expected[0])
        np.testing.assert_array_equal(maximum[index], expected[1])
        read_integral, read_maximum, counters = read_first_round_moment_maps(
            str(tmp_path / "drain.zarr"), index
        )
        assert read_integral.shape == read_maximum.shape == REGISTRY_SHAPE
        assert read_maximum.dtype == np.float32
        np.testing.assert_array_equal(read_integral, expected[0])
        np.testing.assert_array_equal(read_maximum, expected[1])
        assert counters["units_read"] == 2 * len(REGISTRY_TILES)
        assert counters["bytes"] == read_integral.nbytes + read_maximum.nbytes


def test_a_transient_error_of_a_mask_write_is_retried(
    registry_cube, tmp_path, monkeypatch
):
    """A Lustre eviction fails one write of a mask with EIO; the write is
    repeated on a fresh open of the array."""
    from astroviper.utils.chunk_io import zarr_chunks

    monkeypatch.setattr(zarr_chunks, "_EIO_RETRY_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    _, valid, windows = registry_cube
    store = _first_round_store(
        tmp_path / "fr.zarr",
        N_REGISTRY,
        REGISTRY_SHAPE,
        REGISTRY_TILE,
        [window["channel_range"] for window in windows],
    )
    open_array, failures = zarr.open_array, []

    def evicted_once(path, *args, **kwargs):
        if kwargs.get("mode") == "r+" and not failures:
            failures.append(path)
            raise OSError(errno.EIO, "Input/output error")
        return open_array(path, *args, **kwargs)

    monkeypatch.setattr(zarr, "open_array", evicted_once)
    planes = valid[:, 2:4]
    assert node_tasks.write_primary_beam_masks(store, [2, 3], planes, planes) > 0
    assert len(failures) == 1
    for name in MASKS:
        stored = open_array(os.path.join(store, name), mode="r")[2:4]
        np.testing.assert_array_equal(
            unpack_plane_masks(stored, REGISTRY_SHAPE[1]), planes[0, :, 0]
        )


def test_a_window_whose_slot_cannot_be_saved_is_lost(
    registry_cube, tmp_path, monkeypatch
):
    """No space for a slot of window 0: the window is dropped and reported
    lost, and the other window is saved."""
    sky, valid, windows = registry_cube
    monkeypatch.setattr(node_tasks, "FIRST_ROUND_WINDOWS_IN_MEMORY", 1)
    save = np.save

    def save_or_fail(stream, values):
        if f"{os.sep}W0{os.sep}" in stream.name:
            raise OSError(errno.ENOSPC, "No space left on device")
        save(stream, values)

    monkeypatch.setattr(node_tasks.np, "save", save_or_fail)
    partials = str(tmp_path / "partials")
    for start in (0, 6, 2, 8):
        _add("run", windows, partials, sky, valid, [start, start + 1])
    pop = pop_first_round_moments("run", None, partials)
    assert pop["lost_windows"] == [0]
    assert [window for window, _ in pop["slots"]] == [1, 1]
    np.testing.assert_array_equal(pop["counts"][:4], 1)
    files = [
        os.path.join(partials, "W1", f"{name}.{field}.npy")
        for _, name in pop["slots"]
        for field in ("integrated", "maximum")
    ]
    assert pop["slot_bytes"] == sum(os.path.getsize(path) for path in files)


def test_a_discarded_run_adds_nothing_more(registry_cube, tmp_path):
    sky, valid, windows = registry_cube
    partials = str(tmp_path / "partials")
    _add("failed", windows, partials, sky, valid, [0])
    assert node_tasks.discard_first_round_moments("failed")
    _add("failed", windows, partials, sky, valid, [1])
    assert "failed" not in node_tasks._FIRST_ROUND_MOMENTS
    assert not node_tasks.discard_first_round_moments("failed")
    assert not os.path.exists(partials)
