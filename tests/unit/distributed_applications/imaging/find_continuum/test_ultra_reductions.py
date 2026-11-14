"""Tests of the pass planner of find_continuum_ultra."""

import errno
import os
import shutil

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from xradio.image import make_empty_sky_image

from astroviper.distributed_applications.imaging.find_continuum import (
    ultra_reductions,
)
from astroviper.distributed_applications.imaging.find_continuum.casa_image_tasks import (
    cube_session,
)
from astroviper.distributed_applications.imaging.find_continuum.ultra_reductions import (
    StoredMap,
    UltraReductions,
    ultra_frequency_tasks,
    ultra_merge_task_memory_bytes,
    ultra_merge_tile_counts,
    ultra_task_memory_bytes,
)
from astroviper.node_tasks.imaging.find_continuum import (
    find_continuum_cube_reductions,
    merge_find_continuum_cube_reductions,
    write_small_image,
)
from astroviper.node_tasks.imaging.utils import (
    write_result_chunk_to_disk_sharded_skunk_works,
)
from astroviper.processing_functions.imaging.find_continuum import (
    CHANNEL_STATISTICS,
    finalize_find_continuum_reductions,
    find_continuum_first_round_requests,
    linear_frequency_axis,
    tile_occupancy,
)
from astroviper.utils.io import (
    create_empty_data_variables_on_disk,
    write_zarr_image_store,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    RAD_PER_ARCSEC,
    graphviper_tree,
    make_find_continuum_image_xds,
)

GiB = 2**30
DIMS = ["time", "frequency", "polarization", "l", "m"]
# 48 channels in shards of 16, 64 px in chunks of 16 px; findContinuum sees
# channels [8, 40) in 3 frequency chunks, merged in 2 x 2 tiles.
N_CHANNELS, N_PIXELS, TILE, WINDOW, N_TASKS = 48, 64, (16, 16), (8, 40), 3
VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
WHOLE_CUBE_SUMS = ("sum", "sumsq", "mean", "sigma", "rms")
PASS_COLUMNS = {
    "pass",
    "reductions",
    "data_variables_read",
    "compute_first_round",
    "n_tasks",
    "n_channels_read",
    "n_planes_read",
    "n_chunks_read",
    "n_chunks_skipped",
    "bytes_read_sky",
    "bytes_read_primary_beam",
    "planes_per_batch",
    "task_memory_plan_bytes",
    "n_moments_requests",
    "n_dropped_predictions",
    "task_rss_peak_bytes",
    "merge_task_memory_plan_bytes",
    "graph_start_unixtime",
    "start_unixtime",
    "compute_end_unixtime",
    "T_create_map_reduce_graph",
    "T_generate_dask_graph",
    "T_compute_dask_graph",
    "merge_start_unixtime",
    "merge_end_unixtime",
    "T_merge_create_graph",
    "T_merge_generate_dask_graph",
    "T_merge_compute_dask_graph",
    "bytes_partials_written",
    "bytes_partials_read",
    "bytes_results_written",
}


def _sharded_store(directory, source, sky, primary_beam, tile, shard):
    """Stokes I (``sky``, ``primary_beam``) and Q written channel by channel
    by the sharded skunk_works writer, Stokes in separate chunks."""
    n_channels, n_pixels = sky.shape[1], sky.shape[-1]
    stokes_q = np.random.default_rng(3).standard_normal(sky.shape).astype(np.float32)
    values = {
        "SKY_RESIDUAL": np.concatenate([sky, stokes_q], axis=2),
        "PRIMARY_BEAM": np.concatenate([primary_beam, primary_beam], axis=2),
    }
    coords = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[n_pixels, n_pixels],
        cell_size=[0.2 * RAD_PER_ARCSEC] * 2,
        frequency_coords=source.frequency.values,
        pol_coords=["I", "Q"],
        time_coords=[0.0],
    )
    store = write_zarr_image_store(coords, str(directory / "cube.img.zarr"))
    create_empty_data_variables_on_disk(
        store,
        ["sky_residual", "primary_beam"],
        shape_dict=dict(coords.sizes),
        parallel_coords={
            "frequency": {"data_chunks": [[c] for c in range(n_channels)]}
        },
        compressor=zarr.codecs.BloscCodec(cname="lz4", clevel=5),
        double_precision=False,
        data_variable_definitions="imaging",
        image_chunking={"polarization": 1, "l": tile[0], "m": tile[1]},
        image_sharding={"frequency": shard},
    )
    for channel in range(n_channels):
        chunk = xr.Dataset(
            {
                name: (DIMS, cube[:, channel : channel + 1])
                for name, cube in values.items()
            }
        )
        write_result_chunk_to_disk_sharded_skunk_works(
            store,
            ["sky_residual", "primary_beam"],
            {"frequency": {"slice": slice(channel, channel + 1)}},
            chunk,
        )
    return store


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    """Stokes I and Q written channel by channel by the sharded skunk_works
    writer, the primary beam below 0.2 in the corner chunks, two small images
    and the requests of a first pass on the window."""
    directory = tmp_path_factory.mktemp("ultra_cube")
    line = dict(
        channel=20,
        fwhm_channels=4,
        peak_jy=0.04,
        offset_pixels=(4, -4),
        size_fwhm_pixels=6,
    )
    source = make_find_continuum_image_xds(
        n_pixels=N_PIXELS,
        n_channels=N_CHANNELS,
        primary_beam_fwhm_arcsec=6.0,
        lines=(line,),
    )
    sky = source["SKY_RESIDUAL"].values.copy()
    primary_beam = source["PRIMARY_BEAM"].values.copy()
    sky[0, 12, 0, 30:34, 30:34] = 0.0
    sky[0, 13, 0, 31, 31] = np.nan
    store = _sharded_store(directory, source, sky, primary_beam, TILE, 16)
    l_index, m_index = np.indices((N_PIXELS, N_PIXELS))
    joint = (((l_index - 36) ** 2 + (m_index - 28) ** 2) < 64).astype(float)
    pbmom = primary_beam[0, slice(*WINDOW), 0].mean(axis=0).astype(np.float64)
    image_paths = {}
    for name, image in (("joint", joint), ("pbmom", pbmom)):
        image_paths[name] = str(directory / f"cube.{name}")
        valid = np.ones(image.shape, dtype=bool)
        write_small_image(
            image_paths[name], image[..., None, None], valid[..., None, None], {}
        )
    sky_path, pb_path = str(directory / "cube.residual"), str(directory / "cube.pb")
    frequency_axis = linear_frequency_axis(source.frequency.values[slice(*WINDOW)])
    rest_frequency = float(source.frequency.attrs["rest_frequency"]["data"])
    joint_path, pbmom_path = image_paths["joint"], image_paths["pbmom"]
    selection = sorted(
        np.random.default_rng(5)
        .choice(WINDOW[1] - WINDOW[0], 20, replace=False)
        .tolist()
    )
    moments = dict(
        kind="moments", frequency_axis=frequency_axis, rest_frequency=rest_frequency
    )
    requests = [
        *find_continuum_first_round_requests(
            frequency_axis, rest_frequency, sky_path, pb_path
        ),
        dict(
            kind="profile",
            image=sky_path,
            mask=f'"{joint_path}">0 && "{pb_path}">0.21',
            channels=None,
            function="mean",
        ),
        dict(
            kind="profile",
            image=sky_path,
            mask=f'"{pb_path}">0.25 && "{pb_path}"<0.5 && "{joint_path}"<0.5',
            channels=None,
            function="xmadm",
        ),
        dict(
            kind="statistics",
            image=sky_path,
            mask=f'"{pbmom_path}">0.3 && "{pbmom_path}"<0.6 && "{joint_path}"==0',
            channels=selection,
            per_channel=True,
        ),
        dict(
            kind="profile",
            image=sky_path,
            mask=f'"{sky_path}">0.001 || "{joint_path}">0',
            channels=None,
            function="max",
        ),
        dict(moments, image=sky_path, mask="", channels=selection, moments=[0, 8]),
        dict(
            moments,
            image=sky_path,
            mask=f'"{sky_path}"<0 && "{joint_path}"<0.5',
            channels=selection[::2],
            moments=[-1, 0, 10],
        ),
    ]
    return {
        "store": store,
        "directory": directory,
        "cubes": {
            sky_path: {"variable": "SKY_RESIDUAL", "channel_offset": WINDOW[0]},
            pb_path: {"variable": "PRIMARY_BEAM", "channel_offset": WINDOW[0]},
            str(directory / "cube.psf"): {
                "variable": "POINT_SPREAD_FUNCTION",
                "channel_offset": WINDOW[0],
            },
        },
        "sky_path": sky_path,
        "pb_path": pb_path,
        "image_paths": {path: path for path in image_paths.values()},
        "joint": joint,
        "joint_path": joint_path,
        "frequency_axis": frequency_axis,
        "rest_frequency": rest_frequency,
        "requests": requests,
    }


def _ultra(cube, work_directory, first_round=None, budget_bytes=64 * GiB):
    return UltraReductions(
        cube["cubes"],
        cube["store"],
        VARIABLES,
        0,
        str(work_directory),
        0.2,
        first_round,
        N_TASKS,
        budget_bytes,
        cube["frequency_axis"],
        cube["rest_frequency"],
        None,
    )


def _port(cube, requests, mapping, channel_offset=WINDOW[0]):
    """find_continuum_fused's tasks, binary tree reduce and finalize on the
    same chunks."""
    outputs = [
        find_continuum_cube_reductions(
            cube["store"],
            {cube["sky_path"]: "SKY_RESIDUAL", cube["pb_path"]: "PRIMARY_BEAM"},
            cube["image_paths"],
            requests,
            "PRIMARY_BEAM",
            0.2,
            channel_offset=channel_offset,
            data_selection=item["data_selection"],
            task_id=task,
        )
        for task, item in mapping.items()
    ]
    merged = graphviper_tree(
        outputs,
        lambda left, right: merge_find_continuum_cube_reductions([left, right], {}),
    )
    return finalize_find_continuum_reductions(merged["reductions"], requests)


def _loaded(result):
    if isinstance(result, dict) and result and isinstance(next(iter(result)), int):
        return {
            moment: values.load() if isinstance(values, StoredMap) else values
            for moment, values in result.items()
        }
    return result


def _assert_results_equal(results, expected, requests, exact=False):
    """Bitwise, but for whole-cube sums (rel 1e-12 unless ``exact``)."""
    for result, reference, request in zip(results, expected, requests, strict=True):
        result, reference = _loaded(result), _loaded(reference)
        if request["kind"] == "moments":
            assert sorted(result) == sorted(request["moments"])
            for moment in request["moments"]:
                np.testing.assert_array_equal(result[moment], reference[moment])
        elif request["kind"] == "profile":
            np.testing.assert_array_equal(result, reference)
        else:
            for key in CHANNEL_STATISTICS:
                if not (exact or request["per_channel"]) and key in WHOLE_CUBE_SUMS:
                    np.testing.assert_allclose(result[key], reference[key], rtol=1e-12)
                else:
                    np.testing.assert_array_equal(result[key], reference[key])


@pytest.fixture(scope="module")
def first_pass(cube, tmp_path_factory):
    """A first pass without a first-round store (threaded scheduler) and the
    port's results on the same chunks."""
    ultra = _ultra(cube, tmp_path_factory.mktemp("work"))
    with dask.config.set(scheduler="threads"):
        results = ultra.run_pass(cube["requests"], len(cube["requests"]))
    return {
        "ultra": ultra,
        "results": results,
        "expected": _port(cube, cube["requests"], ultra.frequency_mapping),
    }


def test_ultra_frequency_tasks():
    assert ultra_frequency_tasks(15360, 32, 24) == 96
    assert ultra_frequency_tasks(3840, 32, 24) == 120
    assert ultra_frequency_tasks(480, 1, 4) == 16
    assert ultra_frequency_tasks(100, 10, 2) == 10
    assert ultra_frequency_tasks(20, 32, 4) == 1


@pytest.mark.parametrize(
    "arguments, gib",
    [
        ((4, 12, 0, 0.0, True), 11.0),  # P1 without a store
        ((4, 12, 0, 0.0, False), 7.3),  # P1 with a store, moments only
        ((4, 0, 1, 0.30, False), 6.4),  # P2 profile
        ((4, 12, 2, 0.19, False), 7.8),  # P3, k = 0
        ((4, 36, 2, 0.19, False), 10.6),  # P3, k = 2
        ((4, 60, 2, 0.19, False), 13.4),  # P3, k = 4
        ((4, 84, 2, 0.19, False), 16.3),  # P3, k = 6 at B = 4
        ((1, 84, 2, 0.19, False), 13.4),  # P3, k = 6 at B = 1
        # P1 without a store, with the first round's transients and the
        # medium cube's inner chunks
        ((4, 12, 0, 0.0, True, 4, 4, 625 * 625 * 4, True), 12.2),
        ((4, 16, 0, 0.0, True, 8, 8, 625 * 625 * 8, True), 20.2),  # float64
    ],
)
def test_task_memory_bounds_of_the_design(arguments, gib):
    assert round(ultra_task_memory_bytes(11250, 11250, *arguments) / GiB, 1) == gib


def test_task_memory_counts_dtypes_read_units_and_the_first_round():
    n = 1000
    base = ultra_task_memory_bytes(n, n, 4, 12, 0, 0.0, True)
    float64 = ultra_task_memory_bytes(n, n, 4, 12, 0, 0.0, True, 8, 8)
    assert float64 - base == 2 * 4 * (4 + 4) * n * n
    sky_only = ultra_task_memory_bytes(n, n, 4, 12, 0, 0.0, False, 8, 8)
    assert sky_only - ultra_task_memory_bytes(n, n, 4, 12, 0, 0.0, False) == (
        2 * 4 * 4 * n * n
    )
    unit = ultra_task_memory_bytes(n, n, 4, 12, 0, 0.0, True, read_unit_bytes=10**6)
    assert unit - base == 2 * 10**6
    first_round = ultra_task_memory_bytes(
        n, n, 4, 12, 0, 0.0, True, compute_first_round=True
    )
    assert first_round - base == (8 + 4 / 2) * n * n


def test_the_plan_takes_the_store_layout(cube, tmp_path, monkeypatch):
    """The plan of a pass gives the task memory the dtypes and the read unit
    of the arrays it reads, and the first round to the first sub-pass."""
    calls = []

    def recorded(*arguments):
        calls.append(arguments)
        return ultra_task_memory_bytes(*arguments)

    monkeypatch.setattr(ultra_reductions, "ultra_task_memory_bytes", recorded)
    ultra = _ultra(cube, tmp_path / "work")
    ultra.memory_plan(cube["requests"], len(cube["requests"]))

    assert calls and {arguments[7:9] for arguments in calls} == {(4, 4)}
    unit_bytes = TILE[0] * TILE[1] * 4
    assert {arguments[9] for arguments in calls} == {unit_bytes}
    assert any(arguments[10] for arguments in calls)
    assert all(arguments[6] for arguments in calls if arguments[10])


def test_merge_tiles_of_a_cube_whose_chunks_span_the_plane(tmp_path):
    """Without l and m chunking a chunk spans the plane: merge tiles are made
    of 625 px chunks, so that a merge task does not hold whole planes."""
    assert ultra_merge_tile_counts((11250, 11250), (11250, 11250)) == [9, 9]
    assert ultra_merge_tile_counts((11250, 11250), (625, 625)) == [9, 9]
    assert ultra_merge_tile_counts((11250, 11250), (5000, 11250)) == [2, 9]
    assert ultra_merge_tile_counts((64, 64), (64, 64)) == [1, 1]
    n_pixels = 1500
    coords = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[n_pixels, n_pixels],
        cell_size=[0.1 * RAD_PER_ARCSEC] * 2,
        frequency_coords=230.0e9 + 1.0e6 * np.arange(4),
        pol_coords=["I"],
        time_coords=[0.0],
    )
    store = write_zarr_image_store(coords, str(tmp_path / "cube.img.zarr"))
    create_empty_data_variables_on_disk(
        store,
        ["sky_residual", "primary_beam"],
        shape_dict=dict(coords.sizes),
        parallel_coords={"frequency": {"data_chunks": [[0, 1], [2, 3]]}},
        compressor=None,
        double_precision=False,
        data_variable_definitions="imaging",
    )
    assert zarr.open_array(os.path.join(store, "SKY_RESIDUAL")).chunks[-2:] == (
        n_pixels,
        n_pixels,
    )
    ultra = UltraReductions(
        {
            str(tmp_path / f"cube.{role}"): {"variable": variable, "channel_offset": 0}
            for role, variable in VARIABLES.items()
        },
        store,
        VARIABLES,
        0,
        str(tmp_path / "work"),
        0.2,
        None,
        1,
        64 * GiB,
        linear_frequency_axis(coords.frequency.values),
        230.0e9,
        None,
    )
    assert ultra.layout["merge_tile_shape"] == [750, 750]
    assert ultra.layout["n_merge_tiles"] == 4


def test_stored_map_loads_a_new_float64_array(tmp_path):
    group = zarr.open_group(str(tmp_path / "maps.zarr"), mode="w")
    values = np.random.default_rng(1).standard_normal((2, 5, 4)).astype(np.float32)
    group.create_array("MOMENT_8", shape=values.shape, dtype=np.float32)[...] = values
    stored = StoredMap(str(tmp_path / "maps.zarr"), "MOMENT_8", index=1)
    loaded = stored.load()
    assert loaded.dtype == np.float64
    np.testing.assert_array_equal(loaded, values[1])
    loaded[...] = 0.0
    np.testing.assert_array_equal(stored.load(), values[1])
    np.testing.assert_array_equal(StoredMap(stored.store, "MOMENT_8").load(), values)


def test_requests_are_encoded_at_store_channels(cube, tmp_path):
    ultra = _ultra(cube, tmp_path)
    requests = cube["requests"]
    encoded, masks = ultra._encode([requests[1], requests[4], requests[7], requests[9]])
    assert [request["image"] for request in encoded] == ["sky"] * 4
    assert encoded[0]["channels"] is None
    assert encoded[0]["cube_terms"] == [("nonzero", "sky")]
    assert encoded[0]["channel_offset"] == WINDOW[0]
    assert encoded[1]["cube_terms"] == [("profile",)]
    joint = cube["joint"] > 0
    assert encoded[1]["two_d_mask"] == 0
    np.testing.assert_array_equal(
        encoded[1]["footprint"], tile_occupancy(joint[None, None, None], TILE)[0, 0, 0]
    )
    assert masks[0]["fraction"] == joint.mean()
    assert encoded[2]["expression"] == f'"sky">0.001 || "{cube["joint_path"]}">0'
    assert encoded[2]["two_d_mask"] is None and encoded[2]["footprint"] is None
    assert encoded[3]["channels"].dtype == np.int32
    np.testing.assert_array_equal(
        encoded[3]["channels"], np.asarray(requests[9]["channels"]) + WINDOW[0]
    )
    assert encoded[3]["cube_terms"] == [("expression", '"sky"<0')]
    assert encoded[3]["two_d_mask"] == 1


def test_first_pass_equals_the_port(cube, first_pass):
    ultra = first_pass["ultra"]
    _assert_results_equal(
        first_pass["results"], first_pass["expected"], cube["requests"]
    )
    assert ultra.cube_reads == {"SKY_RESIDUAL": 1, "PRIMARY_BEAM": 1}
    (row,) = ultra.passes
    assert set(row) == PASS_COLUMNS
    assert row["compute_first_round"]
    assert row["data_variables_read"] == "PRIMARY_BEAM, SKY_RESIDUAL"
    assert row["n_tasks"] == N_TASKS
    assert row["n_channels_read"] == WINDOW[1] - WINDOW[0]
    assert row["planes_per_batch"] == 4 and row["n_moments_requests"] == 3
    # The merge of moments [-1, 0, 10] holds the most: 22 B/px per state.
    assert row["merge_task_memory_plan_bytes"] == ultra_merge_task_memory_bytes(
        32 * 32, N_TASKS, 22, 3
    )
    assert row["n_chunks_skipped"] > 0
    assert row["bytes_partials_written"] > 0 and row["bytes_results_written"] > 0
    assert ultra.T_primary_beam_mean > 0
    timing = pd.concat(ultra.timing, ignore_index=True)
    assert timing["family"].value_counts().to_dict() == {"frequency": 3, "merge": 4}
    assert (timing["reduction"] == 0).all()
    assert [record["reduction"] for record in ultra.reduce_records] == [0, 0]
    assert ultra.layout == {
        "budget_bytes": 64 * GiB,
        "n_frequency_tasks": N_TASKS,
        "channels_per_task": 11,
        "merge_tile_shape": [32, 32],
        "n_merge_tiles": 4,
        "plane_bytes": N_PIXELS**2 * 4,
    }
    pass_directory = os.path.join(ultra.directory, "pass_0")
    assert not os.path.exists(os.path.join(pass_directory, "partials.zarr"))
    stored = first_pass["results"][1][0]
    assert stored == StoredMap(
        os.path.join(pass_directory, "results.zarr"), "R0_MOMENT_0"
    )
    attributes = zarr.open_group(ultra.first_round, mode="r").attrs[
        "find_continuum_first_round"
    ]
    assert attributes["complete"] and attributes["written_by"] == "find_continuum_ultra"
    assert attributes["windows"][0]["channel_range"] == list(WINDOW)


@pytest.fixture(scope="module")
def unsharded_cube(cube, tmp_path_factory):
    """The cube in an unsharded store whose chunks hold 12 channels of one
    Stokes and 16 x 16 px, as stores converted from CASA images hold many."""
    directory = tmp_path_factory.mktemp("unsharded_cube")
    store = str(directory / "cube.img.zarr")
    shutil.copytree(cube["store"], store)
    for name in VARIABLES.values():
        stored = zarr.open_array(os.path.join(store, name), mode="r")
        values, attributes = stored[...], dict(stored.attrs)
        dimension_names = stored.metadata.dimension_names
        shutil.rmtree(os.path.join(store, name))
        array = zarr.create_array(
            os.path.join(store, name),
            shape=values.shape,
            chunks=(1, 12, 1, *TILE),
            dtype=values.dtype,
            fill_value=np.nan,
            compressors=zarr.codecs.BloscCodec(cname="lz4", clevel=5),
            dimension_names=dimension_names,
            attributes=attributes,
        )
        array[...] = values
    zarr.consolidate_metadata(store)
    return dict(cube, store=store)


def _read_units(ultra, frame, channels):
    """Per frequency task, the read units (12-channel chunk x tile) holding a
    needed channel, each counted once."""
    n_tiles = (N_PIXELS // TILE[0]) * (N_PIXELS // TILE[1])
    units = {}
    for task in frame["task_id"]:
        frequency = ultra.frequency_mapping[task]["data_selection"]["img"]["frequency"]
        needed = [c for c in range(frequency.start, frequency.stop) if c in channels]
        units[task] = len({c // 12 for c in needed}) * n_tiles
    return units


def test_a_read_unit_of_many_channels_is_read_once_per_task(unsharded_cube, tmp_path):
    """On a store whose chunks hold 12 channels, a batch reads the needed
    channels of a chunk at once, the ones it skips between them included: a
    task decodes each chunk once, and the results equal the port's."""
    cube = unsharded_cube
    ultra = _ultra(cube, tmp_path / "work")
    with dask.config.set(scheduler="threads"):
        results = ultra.run_pass(cube["requests"], len(cube["requests"]))
    _assert_results_equal(
        results,
        _port(cube, cube["requests"], ultra.frequency_mapping),
        cube["requests"],
    )
    assert ultra.passes[0]["planes_per_batch"] == 11
    frame = ultra.timing[0]
    frame = frame[frame["family"] == "frequency"].set_index("task_id", drop=False)
    units = _read_units(ultra, frame, range(*WINDOW))
    for task, expected in units.items():
        # Both cubes are read in the first pass.
        row = frame.loc[task]
        assert row["n_chunks_read"] + row["n_chunks_skipped"] == 2 * expected

    # Moments and statistics of selected channels, with gaps in the chunks.
    requests = cube["requests"][6:7] + cube["requests"][-2:]
    store_pass = _ultra(cube, tmp_path / "selected", first_round=ultra.first_round)
    with dask.config.set(scheduler="threads"):
        results = store_pass.run_pass(requests, len(requests))
    _assert_results_equal(
        results, _port(cube, requests, store_pass.frequency_mapping), requests
    )
    frame = store_pass.timing[0]
    frame = frame[frame["family"] == "frequency"].set_index("task_id", drop=False)
    selected = {WINDOW[0] + c for request in requests for c in request["channels"]}
    for task, expected in _read_units(store_pass, frame, selected).items():
        row = frame.loc[task]
        assert row["n_chunks_read"] + row["n_chunks_skipped"] == expected
    first, last = frame["first_channel"], frame["last_channel"]
    assert (frame["n_planes_read"] > frame["n_channels"]).any()
    assert (frame["n_planes_read"] <= last - first + 1).all()


def test_a_pass_with_the_store_equals_the_first_pass(cube, first_pass, tmp_path):
    requests = cube["requests"]
    ultra = _ultra(cube, tmp_path, first_round=first_pass["ultra"].first_round)
    with dask.config.set(scheduler="synchronous"):
        results = ultra.run_pass(requests, len(requests))
    _assert_results_equal(results, first_pass["results"], requests, exact=True)
    (row,) = ultra.passes
    assert not row["compute_first_round"]
    assert row["n_moments_requests"] == 3

    # The cube terms "sky"!=0 and the stored "pb">0.21 need no primary beam.
    ultra = _ultra(cube, tmp_path / "second", first_round=ultra.first_round)
    with dask.config.set(scheduler="synchronous"):
        results = ultra.run_pass([requests[1], requests[4]], 2)
    _assert_results_equal(
        results, [first_pass["results"][k] for k in (1, 4)], [requests[1], requests[4]]
    )
    assert ultra.cube_reads == {"SKY_RESIDUAL": 1}
    assert ultra.passes[0]["bytes_read_primary_beam"] == 0


def test_first_round_requests_are_answered_without_a_pass(cube, first_pass, tmp_path):
    requests = cube["requests"]
    store = str(tmp_path / "first_round.zarr")
    shutil.copytree(first_pass["ultra"].first_round, store)
    ultra = _ultra(cube, tmp_path, first_round=store)
    asked = [requests[k] for k in (0, 2, 3)]
    with cube_session(cube["cubes"], ultra.run_pass) as session:
        results = ultra.run_pass(asked, len(asked))
    _assert_results_equal(
        results, [first_pass["results"][k] for k in (0, 2, 3)], asked, exact=True
    )
    # The mean primary beam reads 5 planes of 16 chunks in the driver.
    assert session.counters["units_read"] == 5 * 16

    # The imager's moments of the window, once complete.
    group = zarr.open_group(store, mode="r+")
    maps = _loaded(first_pass["results"][1])
    group["MOMENT_0"][0] = maps[0]
    group["MOMENT_8"][0] = maps[8].astype(np.float32)
    attributes = dict(group.attrs["find_continuum_first_round"])
    group.attrs["find_continuum_first_round"] = dict(
        attributes, moments_complete=[True]
    )
    (result,) = ultra.run_pass([requests[1]], 1)
    assert result == {
        moment: StoredMap(store, f"MOMENT_{moment}", index=0) for moment in (0, 8)
    }
    _assert_results_equal([result], [maps], [requests[1]])
    assert ultra.passes == [] and ultra.cube_reads == {}


def test_memory_plan_with_a_small_budget(cube, first_pass, tmp_path):
    requests = cube["requests"]
    intersection = sorted(set(requests[8]["channels"]) & set(requests[9]["channels"]))
    planned = [
        requests[8],
        requests[9],
        dict(requests[8], channels=intersection),
        requests[4],
    ]
    store = first_pass["ultra"].first_round
    default = _ultra(cube, tmp_path / "default", first_round=store)
    plan = default.memory_plan(planned, 2)
    assert plan["planes_per_batch"] == [4]
    assert plan["sub_passes"] == [[0, 1, 2, 3]] and plan["dropped"] == []
    budget = plan["task_bytes"][0] - 1
    plan = _ultra(cube, tmp_path / "b", store, budget).memory_plan(planned, 2)
    assert plan["planes_per_batch"] == [2]
    assert plan["sub_passes"] == [[0, 1, 2, 3]] and plan["dropped"] == []

    # Request 1 alone at one plane per batch: [-1, 0, 10] (22 B/px), its mask
    # and a decoded inner chunk.
    budget = ultra_task_memory_bytes(
        N_PIXELS, N_PIXELS, 1, 22, 1, 0.0, False, read_unit_bytes=TILE[0] * TILE[1] * 4
    )
    small = _ultra(cube, tmp_path / "small", store, budget)
    plan = small.memory_plan(planned, 2)
    assert plan["sub_passes"] == [[0, 3], [1]] and plan["dropped"] == [2]
    assert plan["planes_per_batch"][1] == 1 and max(plan["planes_per_batch"]) < 4
    assert max(plan["task_bytes"]) <= budget

    with dask.config.set(scheduler="synchronous"):
        expected = default.run_pass(planned, 2)
        results = small.run_pass(planned, 2)
    assert results[2] is None
    del results[2], expected[2]
    _assert_results_equal(
        results, expected, [planned[k] for k in (0, 1, 3)], exact=True
    )
    assert [row["planes_per_batch"] for row in small.passes] == plan["planes_per_batch"]
    assert [row["n_dropped_predictions"] for row in small.passes] == [1, 0]
    assert [row["n_moments_requests"] for row in small.passes] == [1, 1]
    assert small.cube_reads == {"SKY_RESIDUAL": 2}

    with pytest.raises(ValueError, match="one reduction at one plane per batch"):
        _ultra(cube, tmp_path / "refused", store, budget - 1).memory_plan(planned, 2)


def test_a_request_over_the_budget_runs_within_the_memory_of_a_thread(
    cube, first_pass, tmp_path, monkeypatch
):
    """Request 1 needs more than the budget at one plane per batch but no more
    than a thread has: it runs alone at one plane per batch, with a warning,
    and gives the results of the default plan."""
    warnings = []
    monkeypatch.setattr(ultra_reductions.logger, "warning", warnings.append)
    requests = cube["requests"][-2:]
    store = first_pass["ultra"].first_round
    default = _ultra(cube, tmp_path / "default", first_round=store)
    needed = ultra_task_memory_bytes(
        N_PIXELS, N_PIXELS, 1, 22, 1, 0.0, False, read_unit_bytes=TILE[0] * TILE[1] * 4
    )
    tight = UltraReductions(
        cube["cubes"],
        cube["store"],
        VARIABLES,
        0,
        str(tmp_path / "tight"),
        0.2,
        store,
        N_TASKS,
        needed - 1,
        cube["frequency_axis"],
        cube["rest_frequency"],
        None,
        memory_per_thread_bytes=needed,
    )
    plan = tight.memory_plan(requests, 2)
    assert plan["sub_passes"] == [[0], [1]]
    assert plan["planes_per_batch"][1] == 1 and plan["task_bytes"][1] == needed
    assert len(warnings) == 1 and "runs alone at one plane per batch" in warnings[0]

    with dask.config.set(scheduler="synchronous"):
        expected = default.run_pass(requests, 2)
        results = tight.run_pass(requests, 2)
    _assert_results_equal(results, expected, requests, exact=True)
    with pytest.raises(ValueError, match="memory per thread"):
        UltraReductions(
            cube["cubes"],
            cube["store"],
            VARIABLES,
            0,
            str(tmp_path / "refused"),
            0.2,
            store,
            N_TASKS,
            needed - 1,
            cube["frequency_axis"],
            cube["rest_frequency"],
            None,
            memory_per_thread_bytes=needed - 1,
        ).memory_plan(requests, 2)


@pytest.mark.parametrize(
    "failing",
    ["find_continuum_streamed_reductions", "merge_find_continuum_partial_maps"],
)
def test_a_failed_pass_removes_its_scratch(
    cube, first_pass, tmp_path, monkeypatch, failing
):
    """A frequency or merge task that fails leaves no scratch slots (up to
    terabytes at full scale) and no results store in the work directory."""

    def fail(**kwargs):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(ultra_reductions, failing, fail)
    ultra = _ultra(cube, tmp_path / "work", first_round=first_pass["ultra"].first_round)
    requests = cube["requests"][-2:]
    with dask.config.set(scheduler="synchronous"), pytest.raises(OSError):
        ultra.run_pass(requests, len(requests))
    pass_directory = os.path.join(ultra.directory, "pass_0")
    assert os.path.isdir(pass_directory) and os.listdir(pass_directory) == []


def test_a_transient_error_of_a_scratch_write_is_retried(
    cube, first_pass, tmp_path, monkeypatch
):
    """A Lustre eviction fails one write of a slot with EIO: the write is
    repeated on a fresh open, and the results are those of a pass without
    the error."""
    from astroviper.node_tasks.imaging.utils import skunk_works

    requests = cube["requests"][-2:]
    store = first_pass["ultra"].first_round
    with dask.config.set(scheduler="synchronous"):
        expected = _ultra(cube, tmp_path / "expected", store).run_pass(requests, 2)
    monkeypatch.setattr(skunk_works, "_EIO_RETRY_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    open_array, failures = zarr.open_array, []

    def evicted_once(path, *args, **kwargs):
        if kwargs.get("mode") == "r+" and "partials.zarr" in str(path) and not failures:
            failures.append(path)
            raise OSError(errno.EIO, "Input/output error")
        return open_array(path, *args, **kwargs)

    monkeypatch.setattr(zarr, "open_array", evicted_once)
    with dask.config.set(scheduler="synchronous"):
        results = _ultra(cube, tmp_path / "evicted", store).run_pass(requests, 2)
    assert len(failures) == 1
    _assert_results_equal(results, expected, requests, exact=True)


def test_a_moments_request_without_channels_is_refused(cube, first_pass, tmp_path):
    ultra = _ultra(cube, tmp_path, first_round=first_pass["ultra"].first_round)
    request = dict(cube["requests"][8], channels=[])
    with pytest.raises(ValueError, match="No channel of the cube is selected"):
        ultra.run_pass([request], 1)
    assert ultra.passes == []


def test_one_task_pass_with_edge_tiles_and_monitoring(tmp_path):
    # 70 px in chunks of 16: merge tiles of 24, 24 and 22 px, shards of 72 px.
    source = make_find_continuum_image_xds(
        n_pixels=70, n_channels=12, primary_beam_fwhm_arcsec=6.0
    )
    store = _sharded_store(
        tmp_path,
        source,
        source["SKY_RESIDUAL"].values,
        source["PRIMARY_BEAM"].values,
        TILE,
        4,
    )
    sky_path, pb_path = str(tmp_path / "cube.residual"), str(tmp_path / "cube.pb")
    frequency_axis = linear_frequency_axis(source.frequency.values)
    rest_frequency = float(source.frequency.attrs["rest_frequency"]["data"])
    requests = find_continuum_first_round_requests(
        frequency_axis, rest_frequency, sky_path, pb_path
    )
    requests.append(
        dict(kind="profile", image=sky_path, mask="", channels=None, function="mean")
    )
    ultra = UltraReductions(
        {
            sky_path: {"variable": "SKY_RESIDUAL", "channel_offset": 0},
            pb_path: {"variable": "PRIMARY_BEAM", "channel_offset": 0},
        },
        store,
        VARIABLES,
        0,
        str(tmp_path),
        0.2,
        None,
        1,
        64 * GiB,
        frequency_axis,
        rest_frequency,
        0.01,
    )
    assert ultra.layout["merge_tile_shape"] == [24, 24]
    assert ultra.layout["n_merge_tiles"] == 9
    with dask.config.set(scheduler="synchronous"):
        results = ultra.run_pass(requests, len(requests))
    port = {"store": store, "sky_path": sky_path, "pb_path": pb_path, "image_paths": {}}
    _assert_results_equal(
        results, _port(port, requests, ultra.frequency_mapping, 0), requests
    )
    results_store = zarr.open_group(
        os.path.join(ultra.directory, "pass_0", "results.zarr"), mode="r"
    )
    assert results_store["R0_MOMENT_0"].chunks == (24, 24)
    assert ultra.reduce_records == []
    timing = pd.concat(ultra.timing, ignore_index=True)
    assert timing["family"].value_counts().to_dict() == {"frequency": 1, "merge": 9}
    assert all(isinstance(series, list) for series in timing["memory_rss_bytes"])
    assert ultra.passes[0]["task_rss_peak_bytes"] > 0
