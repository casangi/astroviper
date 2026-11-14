"""Tests of the find_continuum node tasks."""

import itertools
import json
import os
import threading

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from xradio.image import make_empty_sky_image, write_image

import astroviper.node_tasks.imaging.find_continuum as node_tasks
from astroviper.node_tasks.imaging.find_continuum import (
    add_first_round_moments,
    find_continuum_cube_reductions,
    find_continuum_first_round_chunk,
    find_continuum_streamed_reductions,
    merge_find_continuum_cube_reductions,
    merge_find_continuum_partial_maps,
    merge_find_continuum_streamed_reductions,
    merge_first_round_moment_tile,
    pop_first_round_moments,
    primary_beam_channel_mean,
    read_small_image,
    small_image_attributes,
    write_find_continuum_first_round_rows,
    write_small_image,
)
from astroviper.node_tasks.imaging.utils import (
    write_result_chunk_to_disk_sharded_skunk_works,
)
from astroviper.processing_functions.imaging.find_continuum import (
    CHANNEL_STATISTICS,
    FindContinuumMoments,
    finalize_find_continuum_reductions,
    finalize_streamed_reductions,
    first_round_statistics,
    linear_frequency_axis,
    merge_in_tree_order,
    pack_plane_masks,
    primary_beam_masks,
    tile_occupancy,
    unpack_plane_masks,
)
from astroviper.processing_functions.imaging.find_continuum import (
    find_continuum_cube_reductions as reduce_chunk,
)
from astroviper.utils.io import (
    create_empty_data_variables_on_disk,
    write_zarr_image_store,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    RAD_PER_ARCSEC,
    encode_streamed_requests,
    graphviper_tree,
    make_find_continuum_image_xds,
)

N_CHANNELS = 20
VARIABLES = {"cube": "SKY_RESIDUAL", "pb": "PRIMARY_BEAM"}
DIMS = ["time", "frequency", "polarization", "l", "m"]
MASKS = ("PRIMARY_BEAM_LIMIT_MASK", "PRIMARY_BEAM_PROFILE_MASK")
WHOLE_CUBE_SUMS = ("sum", "sumsq", "mean", "sigma", "rms")
PORT_VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
# The sharded cube: 80 channels in shards of 32, 128 px in chunks of 32 px,
# mapped in three frequency chunks and merged in four tiles.
N_SHARDED, N_PIXELS, TILE = 80, 128, (32, 32)
CHUNKS = [slice(0, 27), slice(27, 54), slice(54, N_SHARDED)]
MERGE_TILES = list(itertools.product([slice(0, 64), slice(64, 128)], repeat=2))
FREQUENCY_TASK_COLUMNS = {
    "family",
    "task_id",
    "slot",
    "first_channel",
    "last_channel",
    "n_channels",
    "start_unixtime",
    "end_unixtime",
    "hostname",
    "process_pid",
    "thread_native_id",
    "worker_name",
    "n_planes_read",
    "n_chunks_read",
    "n_chunks_skipped",
    "bytes_read_sky",
    "bytes_read_primary_beam",
    "n_shard_opens",
    "bytes_written",
    "T_read",
    "T_read_wait",
    "T_mask",
    "T_moments",
    "T_statistics",
    "T_profile",
    "T_first_round",
    "T_write_partials",
    "T_cpu",
    "T_find_continuum_task",
}


@pytest.fixture
def cube_store(tmp_path):
    line = dict(
        channel=10,
        fwhm_channels=3,
        peak_jy=0.05,
        offset_pixels=(0, 0),
        size_fwhm_pixels=3,
    )
    img_xds = make_find_continuum_image_xds(
        n_pixels=16, n_channels=N_CHANNELS, lines=(line,)
    )
    path = str(tmp_path / "cube.img.zarr")
    write_image(img_xds, imagename=path, out_format="zarr", overwrite=True)
    return path, img_xds


def test_small_image_round_trip(tmp_path):
    values = np.arange(24.0).reshape(4, 3, 1, 2)
    mask = values % 5 != 0
    metadata = {"shape": [4, 3, 1, 2], "brightness_unit": "Jy/beam", "beams": None}
    path = str(tmp_path / "small.zarr")
    write_small_image(path, values, mask, metadata)
    read_values, read_mask, read_metadata = read_small_image(path)
    np.testing.assert_array_equal(read_values, values)
    np.testing.assert_array_equal(read_mask, mask)
    assert read_metadata == metadata


def test_small_image_is_one_chunk_per_variable_with_its_attributes(tmp_path):
    values = np.arange(24.0).reshape(4, 3, 1, 2)
    path = str(tmp_path / "small.zarr")
    write_id = write_small_image(path, values, values % 5 != 0, {"shape": [4, 3]})
    for name in ("SKY", "MASK"):
        with open(os.path.join(path, name, "zarr.json")) as stream:
            metadata = json.load(stream)
        assert (
            metadata["chunk_grid"]["configuration"]["chunk_shape"]
            == (metadata["shape"])
        )
        assert [codec["name"] for codec in metadata["codecs"]] == ["bytes"]
        files = [f for _, _, f in os.walk(os.path.join(path, name, "c"))]
        assert sum(map(len, files)) == 1
    assert small_image_attributes(path) == {
        "write_id": write_id,
        "all_valid": False,
        "casa_metadata": {"shape": [4, 3]},
    }
    rewritten = write_small_image(path, values, np.ones(values.shape, bool), {})
    attributes = small_image_attributes(path)
    assert attributes["write_id"] == rewritten != write_id
    assert attributes["all_valid"] is True


def test_map_and_reduce_tasks_match_one_reduction(cube_store, tmp_path):
    store, img_xds = cube_store
    mask_path = str(tmp_path / "mask.zarr")
    mask_values = np.zeros((16, 16, 1, 1))
    mask_values[4:12, 4:12] = 1.0
    write_small_image(
        mask_path, mask_values, np.ones(mask_values.shape, dtype=bool), {}
    )
    requests = [
        {
            "kind": "moments",
            "image": "cube",
            "mask": '"mask_image">0.5 && "pb">0.2',
            "channels": list(range(5, 16)),
            "moments": [0, 8],
            "frequency_axis": [
                float(img_xds.frequency[0]),
                float(img_xds.frequency[1] - img_xds.frequency[0]),
                0.0,
                N_CHANNELS,
            ],
            "rest_frequency": float(img_xds.frequency.attrs["rest_frequency"]["data"]),
        },
        {
            "kind": "profile",
            "image": "cube",
            "mask": '"mask_image">0.5',
            "channels": None,
            "function": "max",
        },
    ]
    chunks = [slice(0, 7), slice(7, 14), slice(14, N_CHANNELS)]
    outputs = [
        find_continuum_cube_reductions(
            store,
            VARIABLES,
            {"mask_image": mask_path},
            requests,
            data_selection={"img": {"frequency": chunk}},
            task_id=index,
        )
        for index, chunk in enumerate(chunks)
    ]
    merged = merge_find_continuum_cube_reductions(
        [outputs[0], merge_find_continuum_cube_reductions(outputs[1:], {})], {}
    )
    result = finalize_find_continuum_reductions(merged["reductions"], requests)

    images = {"mask_image": mask_values}
    expected = finalize_find_continuum_reductions(
        reduce_chunk(
            img_xds.sel(polarization=["I"]),
            requests,
            VARIABLES,
            images,
            np.arange(N_CHANNELS),
        ),
        requests,
    )
    for moment in (0, 8):
        np.testing.assert_allclose(
            result[0][moment], expected[0][moment], equal_nan=True
        )
    np.testing.assert_allclose(result[1], expected[1])

    timing = merged["timing_node_tasks"]
    assert list(timing["task_id"]) == [0, 1, 2]
    assert list(timing["n_channels"]) == [7, 7, 6]
    assert {
        "T_load",
        "T_reduce",
        "T_find_continuum_task",
        "hostname",
        "worker_name",
    } <= set(timing)


def test_cube_reductions_of_a_window_read_the_store_at_its_channels(
    cube_store, tmp_path
):
    """Window-relative requests on channels [6, 20) of the cube give the
    results of the same requests on a cube of those channels alone."""
    store, img_xds = cube_store
    start = 6
    window = img_xds.isel(frequency=slice(start, None))
    alone = write_zarr_image_store(window, str(tmp_path / "window.img.zarr"))
    requests = [
        {
            "kind": "moments",
            "image": "cube",
            "mask": '"cube"!=0',
            "channels": list(range(6)),
            "moments": [0, 8],
            "frequency_axis": linear_frequency_axis(window.frequency.values),
            "rest_frequency": float(img_xds.frequency.attrs["rest_frequency"]["data"]),
        },
        {
            "kind": "statistics",
            "image": "cube",
            "mask": '"pb">0.5',
            "channels": [1, 3, 8],
            "per_channel": True,
        },
    ]
    shifted = find_continuum_cube_reductions(
        store,
        VARIABLES,
        {},
        requests,
        channel_offset=start,
        data_selection={"img": {"frequency": slice(start, N_CHANNELS)}},
    )
    reference = find_continuum_cube_reductions(alone, VARIABLES, {}, requests)
    result, expected = (
        finalize_find_continuum_reductions(output["reductions"], requests)
        for output in (shifted, reference)
    )
    for moment in (0, 8):
        np.testing.assert_array_equal(result[0][moment], expected[0][moment])
    for key in CHANNEL_STATISTICS:
        np.testing.assert_array_equal(result[1][key], expected[1][key])


def test_primary_beam_limit_reads_the_primary_beam(cube_store):
    store, img_xds = cube_store
    request = {
        "kind": "statistics",
        "image": "cube",
        "mask": "",
        "channels": None,
        "per_channel": False,
    }
    output = find_continuum_cube_reductions(
        store,
        {"cube": "SKY_RESIDUAL"},
        {},
        [request],
        primary_beam="PRIMARY_BEAM",
        primary_beam_limit=0.9999,
    )
    beam = img_xds["PRIMARY_BEAM"].values
    assert output["reductions"][0]["npts"] == (beam >= 0.9999).sum()


def test_whole_store_without_selection(cube_store):
    store, _ = cube_store
    request = {
        "kind": "statistics",
        "image": "cube",
        "mask": "",
        "channels": [3],
        "per_channel": False,
    }
    output = find_continuum_cube_reductions(
        store, {"cube": "SKY_RESIDUAL"}, {}, [request]
    )
    assert output["reductions"][0]["npts"] == 16 * 16
    assert output["timing_node_tasks"]["n_channels"].item() == N_CHANNELS


def test_merge_folds_resource_series_and_records_reduce_nodes():
    def leaf(task_id):
        return {
            "reductions": [
                {"npts": 1.0, "sum": 1.0, "sumsq": 1.0, "min": 0.0, "max": 1.0}
            ],
            "timing_node_tasks": pd.DataFrame({"task_id": [task_id], "T_load": [0.1]}),
            "resource_usage": {
                "time_seconds": [0.0, 0.1],
                "cpu_percent": [50.0, 90.0],
                "sample_interval_seconds": 0.1,
            },
        }

    first = merge_find_continuum_cube_reductions([leaf(0), leaf(1)], {})
    merged = merge_find_continuum_cube_reductions([first, leaf(2)], {})
    timing = merged["timing_node_tasks"]
    assert list(timing["task_id"]) == [0, 1, 2]
    assert timing["cpu_percent"].iloc[2] == [50.0, 90.0]
    assert merged["reductions"][0]["npts"] == 3.0
    assert [record["n_inputs"] for record in merged["timing_reduce_nodes"]] == [2, 2]


# --------------------------------------------------------------------------
# find_continuum_ultra: streamed frequency tasks and the tile merge
# --------------------------------------------------------------------------
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
    for name in ("PRIMARY_BEAM_LIMIT_TILES", "PRIMARY_BEAM_PROFILE_TILES"):
        group.create_array(
            name,
            shape=(n_channels, *n_tiles),
            chunks=(n_channels, *n_tiles),
            dtype=bool,
            fill_value=False,
            compressors=zarr.codecs.ZstdCodec(level=1),
        )
    group.create_array(
        "STATISTICS",
        shape=(n_channels, 2, 5),
        chunks=(n_channels, 2, 5),
        dtype=np.float64,
        fill_value=np.nan,
    )
    for moment, dtype in ((0, np.float64), (8, np.float32)):
        group.create_array(
            f"MOMENT_{moment}",
            shape=(len(windows), *shape),
            chunks=(1, *tile),
            dtype=dtype,
            fill_value=np.nan,
            compressors=None,
        )
    group.attrs["find_continuum_first_round"] = {
        "version": 1,
        "windows": [{"channel_range": list(window)} for window in windows],
        "complete": False,
    }
    return str(path)


def _scratch_stores(directory, requests, n_slots, shape, tile):
    """The scratch and results stores of a pass's moments requests."""
    os.makedirs(directory, exist_ok=True)
    partials_store = os.path.join(directory, "partials.zarr")
    results_store = os.path.join(directory, "results.zarr")
    partials = zarr.open_group(partials_store, mode="w")
    results = zarr.open_group(results_store, mode="w")
    dtypes = {
        "INTEGRATED": np.float64,
        "SUM": np.float64,
        "MAXIMUM": np.float32,
        "MINIMUM": np.float32,
        "COUNT": np.uint16,
    }
    for index, request in enumerate(requests):
        if request["kind"] != "moments":
            continue
        for field in FindContinuumMoments(request["moments"], (), shape)._fields():
            partials.create_array(
                f"R{index}_{field}",
                shape=(n_slots, *shape),
                chunks=(1, *tile),
                shards=(1, *shape),
                dtype=dtypes[field],
                fill_value=0,
                compressors=None,
                dimension_names=("slot", "l", "m"),
            )
        for moment in request["moments"]:
            results.create_array(
                f"R{index}_MOMENT_{moment}",
                shape=shape,
                chunks=tile,
                dtype=np.float64,
                fill_value=np.nan,
                compressors=None,
            )
    return partials_store, results_store


@pytest.fixture(
    scope="module", params=[1, 2], ids=["polarization_chunk_1", "polarization_chunk_2"]
)
def sharded_cube(request, tmp_path_factory):
    """Stokes I and Q written channel by channel by the sharded skunk_works
    writer; the primary beam is below 0.2 in the corner chunks."""
    directory = tmp_path_factory.mktemp("sharded_cube")
    lines = (
        dict(
            channel=30,
            fwhm_channels=6,
            peak_jy=0.04,
            offset_pixels=(6, -6),
            size_fwhm_pixels=8,
        ),
        dict(
            channel=60,
            fwhm_channels=3,
            peak_jy=0.02,
            offset_pixels=(-10, 12),
            size_fwhm_pixels=5,
        ),
    )
    source = make_find_continuum_image_xds(
        n_pixels=N_PIXELS,
        n_channels=N_SHARDED,
        primary_beam_fwhm_arcsec=10.0,
        lines=lines,
    )
    sky = source["SKY_RESIDUAL"].values.copy()
    primary_beam = source["PRIMARY_BEAM"].values.copy()
    sky[0, 5, 0, 60:64, 60:64] = 0.0
    sky[0, 6, 0, 62, 62] = -0.0
    sky[0, 7, 0, 50, 50] = np.nan
    primary_beam[0, 9, 0, 64, 65] = np.nan
    stokes_q = np.random.default_rng(3).standard_normal(sky.shape).astype(np.float32)
    values = {
        "SKY_RESIDUAL": np.concatenate([sky, stokes_q], axis=2),
        "PRIMARY_BEAM": np.concatenate([primary_beam, primary_beam], axis=2),
    }
    coords = make_empty_sky_image(
        phase_center=[1.2, -0.4],
        image_size=[N_PIXELS, N_PIXELS],
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
        parallel_coords={"frequency": {"data_chunks": [[c] for c in range(N_SHARDED)]}},
        compressor=zarr.codecs.BloscCodec(cname="lz4", clevel=5),
        double_precision=False,
        data_variable_definitions="imaging",
        image_chunking={"polarization": request.param, "l": TILE[0], "m": TILE[1]},
        image_sharding={"frequency": 32},
    )
    for channel in range(N_SHARDED):
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
    l_index, m_index = np.indices((N_PIXELS, N_PIXELS))
    joint = (((l_index - 70) ** 2 + (m_index - 58) ** 2) < 144).astype(float)
    pbmom = primary_beam[0, :, 0].mean(axis=0).astype(np.float64)
    images, image_paths = {}, {}
    for name, image, valid in (
        ("joint", joint, np.ones(joint.shape, dtype=bool)),
        ("pbmom", pbmom, np.arange(pbmom.size).reshape(pbmom.shape) != 0),
    ):
        image_paths[name] = str(directory / f"{name}.zarr")
        write_small_image(
            image_paths[name], image[..., None, None], valid[..., None, None], {}
        )
        images[name] = np.where(valid, image, np.nan)[..., None, None]
    return {
        "store": store,
        "directory": directory,
        "sky": sky,
        "primary_beam": primary_beam,
        "frequency_axis": linear_frequency_axis(source.frequency.values),
        "rest_frequency": float(source.frequency.attrs["rest_frequency"]["data"]),
        "images": images,
        "image_paths": image_paths,
    }


def _requests(cube):
    moments = dict(
        kind="moments",
        frequency_axis=cube["frequency_axis"],
        rest_frequency=cube["rest_frequency"],
    )
    statistics = dict(kind="statistics", image="sky", mask="", per_channel=False)
    rng = np.random.default_rng(5)
    selection = sorted(rng.choice(N_SHARDED, 50, replace=False).tolist())
    return [
        dict(moments, image="sky", mask='"sky"!=0', channels=None, moments=[0, 8]),
        dict(
            moments,
            image="sky",
            mask='"sky"<0 && "joint"<0.5',
            channels=selection[::2],
            moments=[-1, 0, 10],
        ),
        dict(statistics, channels=None),
        dict(statistics, image="primary_beam", channels=None),
        dict(
            kind="profile",
            image="sky",
            mask='"joint">0 && "primary_beam">0.21',
            channels=None,
            function="mean",
        ),
        dict(
            kind="profile",
            image="sky",
            mask='"primary_beam">0.25 && "primary_beam"<0.5 && "joint"<0.5',
            channels=None,
            function="xmadm",
        ),
        dict(
            statistics,
            mask='"pbmom">0.3 && "pbmom"<0.6 && "joint"==0',
            channels=selection,
            per_channel=True,
        ),
        dict(
            kind="profile",
            image="sky",
            mask='"sky">0.001 || "joint">0',
            channels=None,
            function="max",
        ),
        dict(moments, image="sky", mask="", channels=selection, moments=[0, 8]),
    ]


def _encode(cube, requests, directory):
    """Encoded requests with the footprints of their two-dimensional masks,
    and the mask files."""
    encoded, two_d_masks = encode_streamed_requests(
        requests, cube["images"], (N_PIXELS, N_PIXELS)
    )
    files = []
    for index, mask in enumerate(two_d_masks):
        files.append(str(directory / f"mask_{index}.npy"))
        np.save(files[-1], pack_plane_masks(mask))
    for request in encoded:
        if request["two_d_mask"] is not None:
            mask = two_d_masks[request["two_d_mask"]]
            request["footprint"] = tile_occupancy(mask[None, None, None], TILE)[0, 0, 0]
    return encoded, files


def _port(cube, requests):
    outputs = [
        find_continuum_cube_reductions(
            cube["store"],
            PORT_VARIABLES,
            cube["image_paths"],
            requests,
            "PRIMARY_BEAM",
            0.2,
            data_selection={"img": {"frequency": chunk}},
            task_id=index,
        )
        for index, chunk in enumerate(CHUNKS)
    ]
    merged = graphviper_tree(
        outputs,
        lambda left, right: merge_find_continuum_cube_reductions([left, right], {}),
    )
    return finalize_find_continuum_reductions(merged["reductions"], requests)


def _ultra(
    cube,
    encoded,
    files,
    directory,
    first_round_store,
    compute_first_round,
    read_primary_beam=True,
):
    """find_continuum_ultra's frequency tasks, binary reduce and tile merge."""
    partials_store, results_store = _scratch_stores(
        directory, encoded, len(CHUNKS), (N_PIXELS, N_PIXELS), (64, 64)
    )
    outputs = [
        find_continuum_streamed_reductions(
            cube["store"],
            PORT_VARIABLES,
            0,
            0.2,
            encoded,
            cube["frequency_axis"],
            cube["rest_frequency"],
            first_round_store,
            compute_first_round,
            read_primary_beam,
            files,
            cube["image_paths"],
            partials_store,
            5,
            data_selection={"img": {"frequency": chunk}},
            chunk_indices=(index,),
            task_id=index,
        )
        for index, chunk in enumerate(CHUNKS)
    ]
    merged = graphviper_tree(
        outputs,
        lambda left, right: merge_find_continuum_streamed_reductions([left, right], {}),
    )
    moments_requests = [
        (index, request["moments"], "float32")
        for index, request in enumerate(encoded)
        if request["kind"] == "moments"
    ]
    merges = [
        merge_find_continuum_partial_maps(
            partials_store,
            results_store,
            moments_requests,
            merged["written"],
            len(CHUNKS),
            data_selection={"img": {"l": rows, "m": columns}},
            task_id=index,
        )
        for index, (rows, columns) in enumerate(MERGE_TILES)
    ]
    results = finalize_streamed_reductions(merged["reductions"], encoded)
    for index, moments, _ in moments_requests:
        results[index] = {
            moment: zarr.open_array(
                os.path.join(results_store, f"R{index}_MOMENT_{moment}"), mode="r"
            )[:]
            for moment in moments
        }
    timing = pd.concat([merge["timing_node_tasks"] for merge in merges])
    return results, merged, timing


def _assert_results_equal(results, expected, requests, exact=False):
    """Bitwise, but for whole-cube sums (rel 1e-12 unless ``exact``)."""
    for result, reference, request in zip(results, expected, requests, strict=True):
        if request["kind"] == "moments":
            assert list(result) == list(request["moments"])
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
def first_round(sharded_cube, tmp_path_factory):
    """A pass that computes the first round (no store yet) and the port's results."""
    cube = sharded_cube
    directory = tmp_path_factory.mktemp("first_round")
    requests = _requests(cube)
    encoded, files = _encode(cube, requests, directory)
    store = _first_round_store(
        directory / "first_round.zarr",
        N_SHARDED,
        (N_PIXELS, N_PIXELS),
        TILE,
        [[0, N_SHARDED]],
    )
    results, merged, merge_timing = _ultra(
        cube, encoded, files, directory / "pass_0", store, True
    )
    return {
        "store": store,
        "requests": requests,
        "encoded": encoded,
        "files": files,
        "results": results,
        "merged": merged,
        "merge_timing": merge_timing,
        "expected": _port(cube, requests),
    }


def test_streamed_reductions_equal_the_port(first_round):
    _assert_results_equal(
        first_round["results"], first_round["expected"], first_round["requests"]
    )


def test_first_round_masks_rows_and_counters(sharded_cube, first_round):
    cube, store = sharded_cube, first_round["store"]
    inside, profile = primary_beam_masks(cube["primary_beam"], 0.2)
    for name, mask in zip(MASKS, (inside, profile), strict=True):
        stored = zarr.open_array(os.path.join(store, name), mode="r")[:]
        np.testing.assert_array_equal(
            unpack_plane_masks(stored, N_PIXELS), mask[0, :, 0]
        )
    rows = first_round["merged"]["first_round"]
    np.testing.assert_array_equal(rows["channels"], np.arange(N_SHARDED))
    assert rows["channels"].dtype == np.int32 and not rows["write_failed"]
    np.testing.assert_array_equal(
        rows["statistics"],
        first_round_statistics(cube["sky"], cube["primary_beam"], inside),
    )
    tiles = [tile_occupancy(mask, TILE)[0, :, 0] for mask in (inside, profile)]
    np.testing.assert_array_equal(rows["tiles"], np.stack(tiles, axis=1))

    timing = first_round["merged"]["timing_node_tasks"]
    assert FREQUENCY_TASK_COLUMNS <= set(timing)
    assert list(timing["family"]) == ["frequency"] * 3
    assert list(timing["slot"]) == [0, 1, 2]
    assert list(timing["first_channel"]) == [0, 27, 54]
    assert list(timing["last_channel"]) == [26, 53, 79]
    # Batches of 5 planes split at the shards (32, 64): 6 per task, one shard
    # opened per batch and variable.
    assert timing["n_shard_opens"].sum() == 3 * 6 * 2
    outside = int((~tiles[0]).sum())
    assert outside > 0
    assert timing["n_chunks_skipped"].sum() == outside
    assert timing["n_chunks_read"].sum() == 2 * N_SHARDED * 16 - outside
    assert (timing["bytes_read_sky"] > 0).all()
    assert (timing["bytes_read_primary_beam"] > 0).all()
    assert (timing["bytes_written"] > 0).all()
    assert (timing["T_first_round"] > 0).all()
    assert set(first_round["merged"]["written"]) == {0, 1, 8}
    assert first_round["merged"]["written"][0] == [0, 1, 2]
    merge_timing = first_round["merge_timing"]
    assert list(merge_timing["family"]) == ["merge"] * 4
    assert (merge_timing["bytes_read_partials"] > 0).all()
    assert (merge_timing["bytes_written"] > 0).all()

    written = write_find_continuum_first_round_rows(store, rows, N_SHARDED)
    assert written == {"complete": True, "written_channels": list(range(N_SHARDED))}
    for index, name in enumerate(
        ("PRIMARY_BEAM_LIMIT_TILES", "PRIMARY_BEAM_PROFILE_TILES")
    ):
        stored = zarr.open_array(os.path.join(store, name), mode="r")[:]
        np.testing.assert_array_equal(stored, tiles[index])
    statistics = zarr.open_array(os.path.join(store, "STATISTICS"), mode="r")[:]
    np.testing.assert_array_equal(statistics, rows["statistics"])


def test_a_pass_with_the_first_round_store_equals_the_first_pass(
    sharded_cube, first_round, tmp_path
):
    store = first_round["store"]
    write_find_continuum_first_round_rows(
        store, first_round["merged"]["first_round"], N_SHARDED
    )
    results, merged, _ = _ultra(
        sharded_cube,
        first_round["encoded"],
        first_round["files"],
        tmp_path,
        store,
        False,
    )
    _assert_results_equal(
        results, first_round["results"], first_round["requests"], exact=True
    )
    assert merged["first_round"] is None
    assert (merged["timing_node_tasks"]["T_first_round"] == 0).all()


def test_reads_only_the_chunks_that_a_request_can_use(
    sharded_cube, first_round, tmp_path
):
    cube, store = sharded_cube, first_round["store"]
    write_find_continuum_first_round_rows(
        store, first_round["merged"]["first_round"], N_SHARDED
    )
    channels = list(range(10, 50))
    request = dict(first_round["requests"][4], channels=channels)
    encoded, files = _encode(cube, [request], tmp_path)
    results, merged, _ = _ultra(
        cube, encoded, files, tmp_path, store, False, read_primary_beam=False
    )
    np.testing.assert_array_equal(results[0], first_round["expected"][4][10:50])
    inside, profile = primary_beam_masks(cube["primary_beam"][:, channels], 0.2)
    used = tile_occupancy(inside, TILE) & tile_occupancy(profile, TILE)
    used &= encoded[0]["footprint"]
    timing = merged["timing_node_tasks"]
    assert timing["n_chunks_read"].sum() == used.sum()
    assert timing["n_chunks_skipped"].sum() == len(channels) * 16 - used.sum()
    assert timing["n_planes_read"].sum() == len(channels)
    assert (timing["bytes_read_primary_beam"] == 0).all()
    assert merged["written"] == {}


def test_primary_beam_channel_mean_equals_the_port(sharded_cube):
    cube = sharded_cube
    channels = [10, 20, 40, 60, 70]
    request = {
        "kind": "moments",
        "image": "primary_beam",
        "mask": "",
        "channels": channels,
        "moments": [-1],
        "frequency_axis": cube["frequency_axis"],
        "rest_frequency": cube["rest_frequency"],
    }
    port = find_continuum_cube_reductions(
        cube["store"], PORT_VARIABLES, {}, [request], "PRIMARY_BEAM", 0.2
    )
    expected = finalize_find_continuum_reductions(port["reductions"], [request])[0]
    counters = {}
    mean = primary_beam_channel_mean(
        cube["store"], "PRIMARY_BEAM", 0, channels[::-1], 0.2, counters
    )
    np.testing.assert_array_equal(mean, expected[-1])
    assert counters["shard_opens"] == len(channels)
    assert counters["units_read"] == len(channels) * 16


def test_partial_maps_merge_slots_in_tree_order(tmp_path):
    rng = np.random.default_rng(17)
    shape, tile, n_slots = (24, 20), (12, 10), 7
    requests = [
        {"kind": "moments", "moments": [0, 8]},
        {"kind": "moments", "moments": [-1, 10]},
    ]
    partials_store, results_store = _scratch_stores(
        tmp_path, requests, n_slots, shape, tile
    )
    written = {0: [0, 2, 3, 6], 1: [1, 2, 5]}
    widths = rng.uniform(0.5, 1.5, 20)
    leaves = {index: [None] * n_slots for index in written}
    for index, request in enumerate(requests):
        for slot in written[index]:
            moments = FindContinuumMoments(request["moments"], widths, shape)
            for channel in sorted(rng.choice(20, 3, replace=False)):
                plane = rng.standard_normal(shape).astype(np.float32)
                moments.add_plane(channel, plane, rng.random(shape) > 0.3)
            state = moments.state()
            for field, values in state.items():
                array = zarr.open_array(
                    os.path.join(partials_store, f"R{index}_{field}"), mode="r+"
                )
                array[slot] = values
            leaves[index][slot] = FindContinuumMoments.from_state(
                request["moments"],
                widths,
                shape,
                {field: values.copy() for field, values in state.items()},
            )
    moments_requests = [(0, [0, 8], "float32"), (1, [-1, 10], "float32")]
    tiles = list(
        itertools.product([slice(0, 12), slice(12, 24)], [slice(0, 10), slice(10, 20)])
    )
    timing = pd.concat(
        [
            merge_find_continuum_partial_maps(
                partials_store,
                results_store,
                moments_requests,
                written,
                n_slots,
                data_selection={"img": {"l": rows, "m": columns}},
                task_id=index,
            )["timing_node_tasks"]
            for index, (rows, columns) in enumerate(tiles)
        ]
    )
    for index, request in enumerate(requests):
        expected = merge_in_tree_order(leaves[index]).finalize()
        for moment in request["moments"]:
            stored = zarr.open_array(
                os.path.join(results_store, f"R{index}_MOMENT_{moment}"), mode="r"
            )[:]
            np.testing.assert_array_equal(stored, expected[moment])
    # Each tile reads one inner chunk per written slot and field.
    assert list(timing["n_chunks_read"]) == [4 * 2 + 3 * 3] * 4
    assert list(timing["family"]) == ["merge"] * 4
    with pytest.raises(FileNotFoundError, match="listed as written"):
        merge_find_continuum_partial_maps(
            partials_store, results_store, moments_requests[:1], {0: [0, 1]}, n_slots
        )
    # No slot written: every slot is the identity and the maps are NaN.
    merge_find_continuum_partial_maps(
        partials_store, results_store, moments_requests[1:], {}, n_slots
    )
    for moment in (-1, 10):
        stored = zarr.open_array(
            os.path.join(results_store, f"R1_MOMENT_{moment}"), mode="r"
        )[:]
        assert np.isnan(stored).all()


def test_streamed_reduce_concatenates_rows_and_unites_slots():
    def leaf(task_id, channels, slot):
        channels = np.array(channels)
        return {
            "reductions": [
                None,
                {"channels": channels, "values": channels * 1.0},
                {"channels": channels, "rows": np.ones((channels.size, 5))},
                {
                    "channels": channels,
                    "values": {key: channels * 2.0 for key in CHANNEL_STATISTICS},
                },
            ],
            "written": {} if slot is None else {0: [slot]},
            "first_round": None
            if slot is None
            else {
                "channels": channels.astype(np.int32),
                "statistics": np.zeros((channels.size, 2, 5)),
                "tiles": np.zeros((channels.size, 2, 1, 1), dtype=bool),
                "write_failed": task_id == 2,
            },
            "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
            "resource_usage": {
                "time_seconds": [0.0, 0.1],
                "cpu_percent": [50.0, 90.0],
                "sample_interval_seconds": 0.1,
            },
        }

    first = merge_find_continuum_streamed_reductions(
        [leaf(0, [0, 1], 0), leaf(1, [2], None)], {}
    )
    merged = merge_find_continuum_streamed_reductions([first, leaf(2, [3, 4], 2)], {})
    assert merged["written"] == {0: [0, 2]}
    reductions = merged["reductions"]
    assert reductions[0] is None
    np.testing.assert_array_equal(reductions[1]["channels"], np.arange(5))
    np.testing.assert_array_equal(reductions[1]["values"], np.arange(5.0))
    assert reductions[2]["rows"].shape == (5, 5)
    np.testing.assert_array_equal(reductions[3]["values"]["median"], 2.0 * np.arange(5))
    np.testing.assert_array_equal(merged["first_round"]["channels"], [0, 1, 3, 4])
    assert merged["first_round"]["write_failed"]
    timing = merged["timing_node_tasks"]
    assert list(timing["task_id"]) == [0, 1, 2]
    assert timing["cpu_percent"].iloc[2] == [50.0, 90.0]
    records = merged["timing_reduce_nodes"]
    assert [record["n_inputs"] for record in records] == [2, 2]
    assert {record["family"] for record in records} == {"frequency"}


@pytest.mark.parametrize("change", [None, "duplicate", "missing", "failed"])
def test_first_round_rows_are_complete_with_every_window_channel_once(tmp_path, change):
    store = _first_round_store(tmp_path / "fr.zarr", 30, (16, 16), (8, 8), [[10, 20]])
    rng = np.random.default_rng(23)
    rows = {
        "channels": np.arange(10, 20, dtype=np.int32),
        "statistics": rng.standard_normal((10, 2, 5)),
        "tiles": rng.random((10, 2, 2, 2)) > 0.5,
        "write_failed": change == "failed",
    }
    if change == "duplicate":
        rows = {
            key: value if key == "write_failed" else np.concatenate([value, value[3:4]])
            for key, value in rows.items()
        }
    elif change == "missing":
        rows = {
            key: value if key == "write_failed" else value[1:]
            for key, value in rows.items()
        }
    result = write_find_continuum_first_round_rows(store, rows, 30)
    assert result["complete"] is (change is None)
    attributes = zarr.open_group(store, mode="r").attrs["find_continuum_first_round"]
    assert attributes["complete"] is (change is None)
    assert result["written_channels"] == sorted(set(rows["channels"].tolist()))
    statistics = zarr.open_array(os.path.join(store, "STATISTICS"), mode="r")[:]
    profile = zarr.open_array(
        os.path.join(store, "PRIMARY_BEAM_PROFILE_TILES"), mode="r"
    )[:]
    written = np.array(result["written_channels"])
    first = np.unique(rows["channels"], return_index=True)[1]
    np.testing.assert_array_equal(statistics[written], rows["statistics"][first])
    np.testing.assert_array_equal(profile[written], rows["tiles"][first, 1])
    assert np.isnan(np.delete(statistics, written, axis=0)).all()
    assert not np.delete(profile, written, axis=0).any()


# --------------------------------------------------------------------------
# The imager's first round: chunks, worker-local moments per window
# --------------------------------------------------------------------------
N_REGISTRY, REGISTRY_SHAPE, REGISTRY_TILE = 12, (40, 24), (20, 12)
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
    inside = rng.random(sky.shape) > 0.2
    windows = [
        {
            "channel_range": [start, start + 6],
            "velocity_widths": rng.uniform(0.5, 1.5, 6),
        }
        for start in (0, 6)
    ]
    yield sky, inside, windows
    node_tasks._FIRST_ROUND_MOMENTS.clear()


def _window_moments(sky, inside, window):
    """Moments 0 and 8 of a window's channels where the cube is finite,
    nonzero and inside the limit, added in channel order."""
    start, stop = window["channel_range"]
    moments = FindContinuumMoments([0, 8], window["velocity_widths"], sky.shape[-2:])
    for channel in range(start, stop):
        plane = sky[0, channel, 0]
        valid = inside[0, channel, 0] & np.isfinite(plane) & (plane != 0)
        moments.add_plane(channel - start, plane, valid)
    return moments.finalize()


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
        zarr.open_array(os.path.join(store, f"MOMENT_{moment}"), mode="r")[:]
        for moment in (0, 8)
    ]
    return maps, records


@pytest.mark.parametrize("windows_in_memory", [2, 1])
def test_registry_holds_windows_and_flushes_the_least_recent(
    registry_cube, tmp_path, monkeypatch, windows_in_memory
):
    sky, inside, windows = registry_cube
    monkeypatch.setattr(node_tasks, "FIRST_ROUND_WINDOWS_IN_MEMORY", windows_in_memory)
    partials = str(tmp_path / "partials")
    for start in (0, 6, 2, 8, 4, 10):
        planes = slice(start, start + 2)
        add_first_round_moments(
            "run",
            windows,
            REGISTRY_SHAPE,
            partials,
            range(start, start + 2),
            sky[:, planes],
            inside[:, planes],
        )
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
        for field in ("INTEGRATED", "MAXIMUM")
    ]
    assert pop["slot_bytes"] == sum(os.path.getsize(path) for path in files)
    assert "run" not in node_tasks._FIRST_ROUND_MOMENTS

    (moment_0, moment_8), records = _drain(
        [pop], tmp_path / "fr.zarr", partials, windows
    )
    for index, window in enumerate(windows):
        expected = _window_moments(sky, inside, window)
        np.testing.assert_array_equal(moment_8[index], expected[8].astype(np.float32))
        if windows_in_memory == 2:
            np.testing.assert_array_equal(moment_0[index], expected[0])
        else:
            np.testing.assert_allclose(moment_0[index], expected[0], rtol=1e-12)
            np.testing.assert_array_equal(
                np.isnan(moment_0[index]), np.isnan(expected[0])
            )
    assert {record["stage"] for record in records} == {"first_round_merge"}
    assert all(record["bytes_read"] > 0 for record in records)


def test_registry_counts_duplicated_and_missing_channels(registry_cube, tmp_path):
    sky, inside, windows = registry_cube
    partials = str(tmp_path / "partials")
    for channel in [0, 1, 2, 3, 3, 5, 6, 7, 8, 9, 10, 11]:
        planes = slice(channel, channel + 1)
        add_first_round_moments(
            "run",
            windows,
            REGISTRY_SHAPE,
            partials,
            [channel],
            sky[:, planes],
            inside[:, planes],
        )
    counts = pop_first_round_moments("run", None, partials)["counts"]
    assert counts[3] == 2 and counts[4] == 0
    assert (np.delete(counts, [3, 4]) == 1).all()


def test_a_new_run_drops_the_old_entry(registry_cube, tmp_path):
    sky, inside, windows = registry_cube
    partials = str(tmp_path / "partials")
    for run, channel in (("old", 0), ("new", 1)):
        planes = slice(channel, channel + 1)
        add_first_round_moments(
            run,
            windows,
            REGISTRY_SHAPE,
            partials,
            [channel],
            sky[:, planes],
            inside[:, planes],
        )
    old = pop_first_round_moments("old", "tcp://127.0.0.1:1", partials)
    assert old["counts"] is None and old["slots"] == [] and old["slot_bytes"] == 0
    assert not old["placement_ok"]
    new = pop_first_round_moments("new", None, partials)
    assert new["counts"].tolist() == [0, 1] + [0] * (N_REGISTRY - 2)


def test_concurrent_additions_from_threads(registry_cube, tmp_path):
    sky, inside, windows = registry_cube
    partials = str(tmp_path / "partials")

    def add(first):
        for channel in range(first, N_REGISTRY, 4):
            planes = slice(channel, channel + 1)
            add_first_round_moments(
                "run",
                windows,
                REGISTRY_SHAPE,
                partials,
                [channel],
                sky[:, planes],
                inside[:, planes],
            )

    threads = [threading.Thread(target=add, args=(first,)) for first in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    pop = pop_first_round_moments("run", None, partials)
    np.testing.assert_array_equal(pop["counts"], np.ones(N_REGISTRY, np.uint16))
    (moment_0, moment_8), _ = _drain([pop], tmp_path / "fr.zarr", partials, windows)
    for index, window in enumerate(windows):
        expected = _window_moments(sky, inside, window)
        np.testing.assert_array_equal(moment_8[index], expected[8].astype(np.float32))
        np.testing.assert_allclose(moment_0[index], expected[0], rtol=1e-12)


def test_first_round_chunks_write_masks_rows_and_moments(registry_cube, tmp_path):
    """The imager's chunks of two channels, Stokes I at index 1 of float64
    planes, give the first round of the whole cube."""
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
            slice(start, start + 2),
            store,
            PORT_VARIABLES,
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
    rows["write_failed"] = False
    stokes_i, beam_i = values.astype(np.float32), beam.astype(np.float32)
    inside, profile = primary_beam_masks(beam_i, 0.2)
    np.testing.assert_array_equal(rows["channels"], np.arange(N_REGISTRY))
    np.testing.assert_array_equal(
        rows["statistics"], first_round_statistics(stokes_i, beam_i, inside)
    )
    tiles = [tile_occupancy(mask, REGISTRY_TILE)[0, :, 0] for mask in (inside, profile)]
    np.testing.assert_array_equal(rows["tiles"], np.stack(tiles, axis=1))
    for name, mask in zip(MASKS, (inside, profile), strict=True):
        stored = zarr.open_array(os.path.join(store, name), mode="r")[:]
        np.testing.assert_array_equal(
            unpack_plane_masks(stored, REGISTRY_SHAPE[1]), mask[0, :, 0]
        )
    assert write_find_continuum_first_round_rows(store, rows, N_REGISTRY)["complete"]

    pop = pop_first_round_moments("imager", None, partials)
    np.testing.assert_array_equal(pop["counts"], np.ones(N_REGISTRY, np.uint16))
    (moment_0, moment_8), _ = _drain([pop], tmp_path / "drain.zarr", partials, windows)
    for index, window in enumerate(windows):
        expected = _window_moments(stokes_i, inside, window)
        np.testing.assert_array_equal(moment_0[index], expected[0])
        np.testing.assert_array_equal(moment_8[index], expected[8].astype(np.float32))
