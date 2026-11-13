"""Tests of the find_continuum first-round store and the drain of the imager's moments."""

import os
import shutil

import numpy as np
import pytest
import xarray as xr
import zarr
from xradio.image import make_empty_sky_image

import astroviper.node_tasks.imaging.find_continuum as node_tasks
from astroviper.distributed_applications.imaging.find_continuum import first_round
from astroviper.distributed_applications.imaging.find_continuum.first_round import (
    bind_find_continuum_first_round_store,
    create_find_continuum_first_round_store,
    drain_first_round_moments,
    find_continuum_first_round_partials_path,
    find_continuum_first_round_store_path,
    load_find_continuum_first_round,
    validate_find_continuum_first_round,
)
from astroviper.node_tasks.imaging.find_continuum import (
    find_continuum_first_round_chunk,
    write_find_continuum_first_round_rows,
)
from astroviper.node_tasks.imaging.utils import (
    write_result_chunk_to_disk_sharded_skunk_works,
)
from astroviper.processing_functions.imaging.find_continuum import (
    PROFILE_LEVEL,
    FindContinuumMoments,
    finalize_streamed_reductions,
    first_round_statistics,
    linear_channel_ranges,
    linear_frequency_axis,
    merge_in_tree_order,
    moment_velocity_widths,
    primary_beam_masks,
)
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
IN_ORDER = [slice(start, min(start + 5, N_CHANNELS)) for start in range(0, 34, 5)]


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
        axis = linear_frequency_axis(frequency[start:stop])
        windows.append(
            {
                "channel_range": [start, stop],
                "frequency_axis": axis,
                "velocity_widths": moment_velocity_widths(axis, rest_frequency),
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
        cube["xds"].isel(frequency=channels),
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
    written = write_find_continuum_first_round_rows(
        store, dict(rows, write_failed=False), N_CHANNELS
    )
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
    """Moments 0 and 8 of channels of a window where the cube is finite,
    nonzero and inside the limit, added in the order given."""
    start = cube["windows"][window]["channel_range"][0]
    moments = FindContinuumMoments(
        [0, 8], cube["windows"][window]["velocity_widths"], (N_PIXELS, N_PIXELS)
    )
    for channel in channels:
        plane = cube["sky"][0, channel, 0]
        beam = cube["primary_beam"][:, channel : channel + 1]
        valid = primary_beam_masks(beam, LIMIT)[0][0, 0, 0]
        valid &= np.isfinite(plane) & (plane != 0)
        moments.add_plane(channel - start, plane, valid)
    return moments


def _direct(cube, window):
    return _moments(cube, window, range(*WINDOW_RANGES[window])).finalize()


def _assert_reordered_sum(actual, expected):
    """Moment 0 summed in another order: equal to rounding, relative to the
    largest value (a sum that cancels has no relative precision)."""
    atol = 1e-14 * np.nanmax(np.abs(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=atol)


def _stored_moments(store):
    return [
        zarr.open_array(os.path.join(store, f"MOMENT_{moment}"), mode="r")[:]
        for moment in (0, 8)
    ]


def _attributes(store):
    return zarr.open_group(store, mode="r").attrs["find_continuum_first_round"]


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


def test_create_store(cube, tmp_path):
    store = _create(cube, tmp_path / "fr.zarr")
    group = zarr.open_group(store, mode="r")
    assert group.metadata.zarr_format == 3
    packed, tiles = (N_CHANNELS, N_PIXELS, 3), (N_CHANNELS, 3, 3)
    statistics, maps = (N_CHANNELS, 2, 5), (2, N_PIXELS, N_PIXELS)
    assert {
        name: (array.shape, array.chunks, array.dtype) for name, array in group.arrays()
    } == {
        "PRIMARY_BEAM_LIMIT_MASK": (packed, (1, N_PIXELS, 3), np.uint8),
        "PRIMARY_BEAM_PROFILE_MASK": (packed, (1, N_PIXELS, 3), np.uint8),
        "PRIMARY_BEAM_LIMIT_TILES": (tiles, tiles, bool),
        "PRIMARY_BEAM_PROFILE_TILES": (tiles, tiles, bool),
        "STATISTICS": (statistics, statistics, np.float64),
        "MOMENT_0": (maps, (1, *MERGE_TILE), np.float64),
        "MOMENT_8": (maps, (1, *MERGE_TILE), np.float32),
    }
    for name in ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_PROFILE_MASK"):
        assert group[name].fill_value == 0
        assert group[name].metadata.dimension_names == ("frequency", "l", "m_packed")
    for name in ("STATISTICS", "MOMENT_0", "MOMENT_8"):
        assert np.isnan(group[name].fill_value)
    assert group["MOMENT_0"].compressors == group["MOMENT_8"].compressors == ()
    assert _attributes(store) == {
        "version": 1,
        "written_by": "image_cube_single_field",
        "run_id": "run",
        "primary_beam_limit": LIMIT,
        "profile_level": PROFILE_LEVEL,
        "variables": VARIABLES,
        "polarization_index": 0,
        "value_dtypes": VALUE_DTYPES,
        "tile_shape": list(TILE),
        "n_channels": N_CHANNELS,
        "windows": [
            {
                "channel_range": window["channel_range"],
                "frequency_axis": list(window["frequency_axis"]),
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
    moment_0, moment_8 = _stored_moments(store)
    for window in (0, 1):
        np.testing.assert_array_equal(moment_0[window], _direct(cube, window)[0])
        np.testing.assert_array_equal(moment_8[window], _direct(cube, window)[8])
    pop = {
        "stage": "first_round_pop",
        "n_slots": 2,
        "address": None,
        "process_pid": os.getpid(),
        "placement_ok": True,
        "slot_bytes": result["slot_bytes"],
    }
    assert {key: records[0][key] for key in pop} == pop
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

    monkeypatch.setattr(first_round, "merge_first_round_moment_tile", merge)
    pairs = zip(range(6), range(16, 22), strict=True)
    order = [channel for pair in pairs for channel in pair]
    order += [*range(6, 16), *range(22, 34)]
    store = _first_round(cube, tmp_path / "fr.zarr", [slice(c, c + 1) for c in order])
    result, _ = drain_first_round_moments("run", store, cube["windows"])

    assert result["moments_complete"] == [True, True]
    assert result["n_slots"] == 14
    sequences = {0: [0, 2, 4, 6, 8, 10, 12], 1: [1, 3, 5, 7, 9, 11, 13]}
    assert slot_names == {
        window: [f"{os.getpid()}-{sequence}" for sequence in numbers]
        for window, numbers in sequences.items()
    }
    leaves = {
        0: [[channel] for channel in range(6)] + [range(6, 16)],
        1: [[channel] for channel in range(16, 22)] + [range(22, 34)],
    }
    moment_0, moment_8 = _stored_moments(store)
    for window, channel_lists in leaves.items():
        merged = merge_in_tree_order(
            _moments(cube, window, channels) for channels in channel_lists
        ).finalize()
        np.testing.assert_array_equal(moment_0[window], merged[0])
        np.testing.assert_array_equal(moment_8[window], _direct(cube, window)[8])
        _assert_reordered_sum(moment_0[window], _direct(cube, window)[0])


@pytest.mark.parametrize("change", ["missing", "duplicated"])
def test_drain_leaves_a_window_with_a_coverage_error_incomplete(cube, tmp_path, change):
    """Channel 3 imaged where its moments were lost, or channel 18 added to
    the moments twice (a task rerun after its result was used): only that
    window is left incomplete."""
    store = _create(cube, tmp_path / "fr.zarr")
    chunks = [slice(0, 3), slice(3, 4), slice(4, 16), slice(16, 34)]
    run_ids = ["run", None if change == "missing" else "run", "run", "run"]
    _write_rows(
        store,
        [
            find_continuum_first_round_chunk(*_chunk(cube, store, channels, run_id))
            for channels, run_id in zip(chunks, run_ids, strict=True)
        ],
    )
    if change == "duplicated":
        find_continuum_first_round_chunk(*_chunk(cube, store, slice(18, 19), "run"))
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
    moment_0, moment_8 = _stored_moments(store)
    np.testing.assert_array_equal(moment_0[good], _direct(cube, good)[0])
    np.testing.assert_array_equal(moment_8[good], _direct(cube, good)[8])
    assert np.isnan(moment_0[1 - good]).all() and np.isnan(moment_8[1 - good]).all()
    assert {record["window"] for record in records[1:]} == {good}
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


def test_drain_refuses_pops_that_ran_elsewhere(cube, tmp_path, monkeypatch):
    def misplaced(run_id, address, partials_directory):
        pop = node_tasks.pop_first_round_moments(run_id, address, partials_directory)
        return dict(pop, placement_ok=False)

    monkeypatch.setattr(first_round, "pop_first_round_moments", misplaced)
    store = _first_round(cube, tmp_path / "fr.zarr")
    result, records = drain_first_round_moments("run", store, cube["windows"])

    assert result["moments_complete"] == [False, False]
    assert result["n_missing"] == result["n_duplicated"] == [0, 0]
    assert [record["stage"] for record in records] == ["first_round_pop"]
    assert all(np.isnan(maps).all() for maps in _stored_moments(store))
    assert not os.path.exists(find_continuum_first_round_partials_path(store))


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
            (first, slice(0, 8)),
            (second, slice(8, 16)),
            (first, slice(16, 25)),
            (second, slice(25, 34)),
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
        moment_0, moment_8 = _stored_moments(store)
        halves = [(range(0, 8), range(8, 16)), (range(16, 25), range(25, 34))]
        for window, channel_lists in enumerate(halves):
            merged = merge_in_tree_order(
                _moments(cube, window, channels) for channels in channel_lists
            ).finalize()
            np.testing.assert_array_equal(moment_0[window], merged[0])
            np.testing.assert_array_equal(moment_8[window], _direct(cube, window)[8])
            _assert_reordered_sum(moment_0[window], _direct(cube, window)[0])

        # The moments of a restarted worker are lost; the pop of a busy worker
        # times out and is cancelled.
        for case in ("restarted", "busy"):
            idle, other = sorted(client.nthreads())
            store = _create(cube, tmp_path / f"{case}.zarr", case)
            assignment = [(idle, slice(0, 16)), (idle, slice(16, 25))]
            image(client, store, [*assignment, (other, slice(25, 34))], case)
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
            moment_0, moment_8 = _stored_moments(store)
            np.testing.assert_array_equal(moment_0[0], _direct(cube, 0)[0])
            np.testing.assert_array_equal(moment_8[0], _direct(cube, 0)[8])
            assert np.isnan(moment_0[1]).all() and np.isnan(moment_8[1]).all()


def test_binding_and_validation(cube, tmp_path):
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None)
    with pytest.raises(ValueError, match="not made from the cube"):
        validate_find_continuum_first_round({"store": store}, cube["store"], LIMIT)
    bind_find_continuum_first_round_store(store, cube["store"])

    binding = _attributes(store)["binding"]
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
    validate_find_continuum_first_round({"store": store}, cube["store"], LIMIT)
    with pytest.raises(ValueError, match="primary_beam_limit"):
        validate_find_continuum_first_round({"store": store}, cube["store"], 0.3)
    with pytest.raises(ValueError, match="missing"):
        validate_find_continuum_first_round(
            {"store": str(tmp_path / "absent.zarr")}, cube["store"], LIMIT
        )
    incomplete = _create(cube, tmp_path / "incomplete.zarr", None)
    with pytest.raises(ValueError, match="incomplete"):
        validate_find_continuum_first_round({"store": incomplete}, cube["store"], LIMIT)


@pytest.mark.parametrize(
    "options", [{"tile_shape": (4, 4)}, {"polarization_index": 1}], ids=str
)
def test_validation_refuses_a_store_of_other_settings(cube, tmp_path, options):
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None, **options)
    bind_find_continuum_first_round_store(store, cube["store"])
    with pytest.raises(ValueError, match=next(iter(options))):
        validate_find_continuum_first_round({"store": store}, cube["store"], LIMIT)


def test_validation_refuses_a_rewritten_cube(cube, tmp_path):
    image_store = str(tmp_path / "cube.img.zarr")
    shutil.copytree(cube["store"], image_store)
    store = _first_round(cube, tmp_path / "fr.zarr", run_id=None)
    bind_find_continuum_first_round_store(store, image_store)
    validate_find_continuum_first_round({"store": store}, image_store, LIMIT)
    zarr_json = os.path.join(image_store, "zarr.json")
    stat = os.stat(zarr_json)
    os.utime(zarr_json, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    with pytest.raises(ValueError, match="image_store_zarr_json_mtime_ns differs"):
        validate_find_continuum_first_round({"store": store}, image_store, LIMIT)


def _statistics(cube, start, stop):
    """The whole-cube statistics of P1: per-channel rows summed in channel order."""
    sky = cube["sky"][:, start:stop]
    primary_beam = cube["primary_beam"][:, start:stop]
    inside = primary_beam_masks(primary_beam, LIMIT)[0]
    rows = first_round_statistics(sky, primary_beam, inside)
    request = {"kind": "statistics", "per_channel": False}
    return finalize_streamed_reductions(
        [{"channels": np.arange(start, stop), "rows": rows[:, i]} for i in (0, 1)],
        [request, request],
    )


def _primary_beam_mean(cube, channels):
    planes = cube["primary_beam"][0, channels, 0]
    valid = ~(planes < LIMIT) & np.isfinite(planes)
    count = valid.sum(axis=0)
    total = np.where(valid, planes.astype(np.float64), 0.0).sum(axis=0)
    with np.errstate(invalid="ignore"):
        return np.where(count > 0, total / count, np.nan)


@pytest.fixture
def drained_store(cube, tmp_path):
    store = _first_round(cube, tmp_path / "fr.zarr")
    drain_first_round_moments("run", store, cube["windows"])
    bind_find_continuum_first_round_store(store, cube["store"])
    return store


@pytest.mark.parametrize(
    "window, primary_beam_channels",
    [(0, [2, 4, 8, 12, 14]), (1, [18, 20, 25, 29, 31])],
)
def test_load_a_window(cube, drained_store, window, primary_beam_channels):
    channel_range = WINDOW_RANGES[window]
    lazy_xds = xr.open_zarr(cube["store"], chunks=None)
    counters = {}
    entries = load_find_continuum_first_round(
        {"store": drained_store},
        cube["store"],
        lazy_xds,
        LIMIT,
        channel_range,
        counters,
    )

    assert len(entries) == 4 and list(entries[0]) == [-1]
    # The mean primary beam reads 5 planes of 9 chunks.
    assert counters["units_read"] == 5 * 9 and counters["bytes"] > 0
    np.testing.assert_array_equal(
        entries[0][-1], _primary_beam_mean(cube, primary_beam_channels)
    )
    assert {m: (stored.array, stored.index) for m, stored in entries[1].items()} == {
        0: ("MOMENT_0", window),
        8: ("MOMENT_8", window),
    }
    for moment in (0, 8):
        loaded = entries[1][moment].load()
        assert loaded.dtype == np.float64
        np.testing.assert_array_equal(loaded, _direct(cube, window)[moment])
    for loaded, expected in zip(
        entries[2:], _statistics(cube, *channel_range), strict=True
    ):
        assert loaded["npts"] > 0
        np.testing.assert_equal(loaded, expected)
    windowed = load_find_continuum_first_round(
        {"store": drained_store},
        cube["store"],
        lazy_xds.isel(frequency=slice(*channel_range)),
        LIMIT,
        channel_range,
    )
    np.testing.assert_array_equal(windowed[0][-1], entries[0][-1])
    assert windowed[1] == entries[1]
    np.testing.assert_equal(windowed[2:], entries[2:])


def test_load_without_moments(cube, drained_store):
    """A range that is not a window of the store, or a window without
    complete moments, is answered without moments."""
    lazy_xds = xr.open_zarr(cube["store"], chunks=None)
    entries = load_find_continuum_first_round(
        {"store": drained_store}, cube["store"], lazy_xds, LIMIT, [17, 22]
    )
    assert entries[1] is None
    np.testing.assert_array_equal(
        entries[0][-1], _primary_beam_mean(cube, list(range(17, 22)))
    )
    for loaded, expected in zip(entries[2:], _statistics(cube, 17, 22), strict=True):
        np.testing.assert_equal(loaded, expected)

    group = zarr.open_group(drained_store, mode="r+")
    group.attrs["find_continuum_first_round"] = dict(
        group.attrs["find_continuum_first_round"], moments_complete=[True, False]
    )
    entries = load_find_continuum_first_round(
        {"store": drained_store}, cube["store"], lazy_xds, LIMIT, [16, 34]
    )
    assert entries[1] is None


def test_load_refuses_channels_the_store_does_not_hold(cube, tmp_path):
    """A store of one window, as find_continuum_ultra makes it, holds no first
    round of another window."""
    chunks = [slice(0, 8), slice(8, 16)]
    store = _first_round(cube, tmp_path / "fr.zarr", chunks, None, cube["windows"][:1])
    bind_find_continuum_first_round_store(store, cube["store"])
    lazy_xds = xr.open_zarr(cube["store"], chunks=None)
    with pytest.raises(ValueError, match=r"not \[16, 34\)"):
        load_find_continuum_first_round(
            {"store": store}, cube["store"], lazy_xds, LIMIT, [16, 34]
        )
    with pytest.raises(ValueError, match="primary_beam_limit"):
        load_find_continuum_first_round(
            {"store": store}, cube["store"], lazy_xds, 0.3, [0, 16]
        )
