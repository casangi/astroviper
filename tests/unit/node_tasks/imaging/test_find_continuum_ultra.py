"""Tests of the node tasks of find_continuum_ultra.

The tasks run on a cube written channel by channel by the sharded skunk_works
writer and are compared with the processing functions of
``find_continuum_ultra.cube_reductions`` applied to the whole cube.
"""

import errno
import itertools
import os
import shutil

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from graphviper import graph_tools
from graphviper.graph_tools.coordinate_utils import (
    interpolate_data_coords_onto_parallel_coords,
    make_parallel_coord,
)
from xradio.image import make_empty_sky_image

from astroviper.distributed_applications.imaging.find_continuum_first_round import (
    create_find_continuum_first_round_store,
)
from astroviper.node_tasks.imaging.find_continuum_first_round import (
    write_first_round_rows,
)
from astroviper.node_tasks.imaging.find_continuum_ultra import (
    find_continuum_first_round_task,
    find_continuum_selection_moments_task,
    find_continuum_spectrum_task,
    find_continuum_ultra_merge_task_bytes,
    find_continuum_ultra_task_bytes,
    merge_find_continuum_moment_partials,
    read_moment_maps,
    read_primary_beam_edge_lines,
    read_primary_beam_plane,
    reduce_find_continuum_channel_rows,
    with_resource_usage,
    write_joint_mask_store,
    write_region_masks,
)
from astroviper.node_tasks.imaging.utils import (
    write_result_chunk_to_disk_sharded_skunk_works,
)
from astroviper.processing_functions.imaging.config_check import (
    param_check_find_continuum,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.cube_reductions import (
    accumulate_moments,
    finalize_moments,
    first_round_channel_rows,
    masked_channel_means,
    masked_channel_medians_and_mads,
    masked_channel_scaled_mads,
    merge_moment_partials_in_tree_order,
    new_moment_partials,
    primary_beam_masks,
    tile_occupancy,
    unpack_plane_masks,
    valid_voxels,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    linear_spectral_axis,
    moment_channel_weights,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    FirstRoundRows,
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
VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
# 70 channels in shards of 32, Stokes I (index 0) and Q, 64 px in chunks of
# 16 px; the window [3, 67) in GraphVIPER's three frequency chunks, merged in
# 2 x 2 tiles.
N_CHANNELS, N_PIXELS, TILE, SHARD, STOKES_I, LIMIT = 70, 64, (16, 16), 32, 0, 0.2
WINDOW = (3, 67)
CHUNKS = [slice(3, 25), slice(25, 47), slice(47, 67)]
MERGE_TILE = (32, 32)
MERGE_TILES = list(itertools.product([slice(0, 32), slice(32, 64)], repeat=2))
MAP_SHAPE = (1, 1, 1, N_PIXELS, N_PIXELS)
N_TILES = (N_PIXELS // TILE[0]) * (N_PIXELS // TILE[1])
FREQUENCY_COLUMNS = {
    "family",
    "pass_kind",
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
    "bytes_read_masks",
    "n_shard_opens",
    "bytes_written",
    "T_read",
    "T_read_wait",
    "T_masks",
    "T_moments",
    "T_spectrum",
    "T_noise",
    "T_first_round",
    "T_write_partials",
    "T_cpu",
    "T_find_continuum_task",
}
MERGE_COLUMNS = {
    "family",
    "task_id",
    "start_unixtime",
    "end_unixtime",
    "hostname",
    "process_pid",
    "thread_native_id",
    "worker_name",
    "n_chunks_read",
    "n_chunks_skipped",
    "n_shard_opens",
    "bytes_read_partials",
    "bytes_written",
    "T_read_partials",
    "T_merge",
    "T_write_maps",
    "T_cpu",
    "T_find_continuum_task",
}


@pytest.fixture(
    scope="module", params=[1, 2], ids=["polarization_chunk_1", "polarization_chunk_2"]
)
def cube(request, tmp_path_factory):
    """Stokes I and Q written channel by channel by the sharded skunk_works
    writer; the primary beam is below the limit in the corner chunks."""
    directory = tmp_path_factory.mktemp("cube")
    line = dict(
        channel=30,
        fwhm_channels=4,
        peak_jy=0.04,
        offset_pixels=(4, -4),
        size_fwhm_pixels=6,
    )
    source = make_find_continuum_image_xds(
        n_pixels=N_PIXELS,
        n_channels=N_CHANNELS,
        primary_beam_fwhm_arcsec=5.0,
        lines=(line,),
    )
    sky = source["SKY_RESIDUAL"].values.copy()
    primary_beam = source["PRIMARY_BEAM"].values.copy()
    sky[0, 10, 0, 28:34, 28:34] = 0.0
    sky[0, 11, 0, 30, 30] = -0.0
    sky[0, 12, 0, 31, 31] = np.nan
    primary_beam[0, 14, 0, 32, 33] = np.nan
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
        parallel_coords={
            "frequency": {"data_chunks": [[c] for c in range(N_CHANNELS)]}
        },
        compressor=zarr.codecs.BloscCodec(cname="lz4", clevel=5),
        double_precision=False,
        data_variable_definitions="imaging",
        image_chunking={"polarization": request.param, "l": TILE[0], "m": TILE[1]},
        image_sharding={"frequency": SHARD},
    )
    for channel in range(N_CHANNELS):
        write_result_chunk_to_disk_sharded_skunk_works(
            store,
            ["sky_residual", "primary_beam"],
            {"frequency": {"slice": slice(channel, channel + 1)}},
            xr.Dataset(
                {
                    name: (DIMS, planes[:, channel : channel + 1])
                    for name, planes in values.items()
                }
            ),
        )
    rest_frequency = float(source.frequency.attrs["rest_frequency"]["data"])
    n, first_frequency, _, increment = linear_spectral_axis(
        source.frequency.values[slice(*WINDOW)]
    )
    limit_mask, spectrum_mask = primary_beam_masks(primary_beam, LIMIT)
    return {
        "store": store,
        "directory": directory,
        "sky": sky,
        "primary_beam": primary_beam,
        "limit_mask": limit_mask,
        "spectrum_mask": spectrum_mask,
        "valid": valid_voxels(sky, limit_mask),
        "rest_frequency": rest_frequency,
        "window": {
            "channel_range": list(WINDOW),
            "first_frequency": first_frequency,
            "channel_increment": increment,
            "n_channels": n,
        },
        "weights": moment_channel_weights(
            first_frequency, increment, n, rest_frequency
        ),
    }


def _moment_stores(directory):
    """The partials and maps stores of a pass, as the passes create them."""
    os.makedirs(directory, exist_ok=True)
    stores = [str(directory / "partials.zarr"), str(directory / "maps.zarr")]
    partials, maps = (
        zarr.open_group(store, mode="w", zarr_format=3) for store in stores
    )
    for name, dtype in (
        ("SKY_MOMENT_INTEGRATED", np.float64),
        ("SKY_MOMENT_MAXIMUM", np.float32),
    ):
        partials.create_array(
            name,
            shape=(len(CHUNKS), N_PIXELS, N_PIXELS),
            chunks=(1, *MERGE_TILE),
            shards=(1, N_PIXELS, N_PIXELS),
            dtype=dtype,
            fill_value=0,
            compressors=None,
            dimension_names=("slot", "l", "m"),
        )
        maps.create_array(
            name,
            shape=(N_PIXELS, N_PIXELS),
            chunks=MERGE_TILE,
            dtype=dtype,
            fill_value=np.nan,
            compressors=None,
            dimension_names=("l", "m"),
        )
    return stores


def _frequency_tasks(task, store, chunks=CHUNKS, **params):
    """A frequency task over the chunks of the window, as GraphVIPER maps it."""
    return [
        task(
            input_image_store=store,
            variables=VARIABLES,
            polarization_index=STOKES_I,
            channel_range=list(WINDOW),
            data_selection={"img": {"frequency": chunk}},
            chunk_indices=(np.int64(index),),
            task_id=index,
            input_checked=True,
            **params,
        )
        for index, chunk in enumerate(chunks)
    ]


def _reduce(outputs):
    """GraphVIPER's binary tree reduce of the frequency tasks."""
    return graphviper_tree(
        outputs,
        lambda left, right: reduce_find_continuum_channel_rows([left, right], {}),
    )


def _merge(partials_store, maps_store, written_slots):
    """The merge tasks of the tiles; the maps read back and the merge rows."""
    frames = [
        merge_find_continuum_moment_partials(
            partials_store,
            maps_store,
            written_slots,
            len(CHUNKS),
            "float32",
            data_selection={"img": {"l": rows, "m": columns}},
            task_id=index,
        )["timing_node_tasks"]
        for index, (rows, columns) in enumerate(MERGE_TILES)
    ]
    integral, maximum, _ = read_moment_maps(maps_store)
    return integral, maximum, pd.concat(frames, ignore_index=True)


def _expected_moments(cube, valid, channels=None):
    """The maps of C8 on the whole cube: per frequency chunk the moment
    partials of its (selected) channels in increasing order, merged in
    GraphVIPER's binary tree."""
    leaves = []
    for chunk in CHUNKS:
        selected = [
            c
            for c in range(chunk.start, chunk.stop)
            if channels is None or c in channels
        ]
        partials = new_moment_partials(MAP_SHAPE, np.float32)
        accumulate_moments(
            partials,
            cube["sky"][:, selected],
            valid[:, selected],
            cube["weights"][np.subtract(selected, WINDOW[0])],
        )
        leaves.append(partials)
    return [
        values[0, 0, 0]
        for values in finalize_moments(
            merge_moment_partials_in_tree_order(leaves), MAP_SHAPE, np.float32
        )
    ]


def _units(tiles):
    """The number of (channel, tile) read units of per-channel tile masks."""
    return int(np.count_nonzero(tiles))


@pytest.fixture(scope="module")
def first_round(cube):
    """The first-round pass without a store: masks, rows and moments."""
    directory = cube["directory"] / "first_round"
    store = str(cube["directory"] / "first_round.zarr")
    create_find_continuum_first_round_store(
        store,
        N_CHANNELS,
        (1, N_CHANNELS, 2, N_PIXELS, N_PIXELS),
        TILE,
        MERGE_TILE,
        LIMIT,
        VARIABLES,
        STOKES_I,
        {"sky": "float32", "primary_beam": "float32"},
        None,
        "find_continuum_ultra",
        [cube["window"]],
        cube["rest_frequency"],
    )
    partials_store, maps_store = _moment_stores(directory)
    outputs = _frequency_tasks(
        find_continuum_first_round_task,
        cube["store"],
        primary_beam_limit=LIMIT,
        moment_channel_weights=cube["weights"],
        first_round_store=store,
        compute_masks=True,
        compute_moments=True,
        partials_store=partials_store,
        planes_per_batch=5,
    )
    merged = _reduce(outputs)
    integral, maximum, merge_timing = _merge(
        partials_store, maps_store, merged["written_slots"]
    )
    written = write_first_round_rows(
        store, dict(merged["rows"], write_failed=False), N_CHANNELS
    )
    return {
        "store": store,
        "merged": merged,
        "integral": integral,
        "maximum": maximum,
        "merge_timing": merge_timing,
        "written": written,
    }


def test_first_round_masks_rows_and_moments_equal_the_cube_reductions(
    cube, first_round
):
    window = slice(*WINDOW)
    limit_mask, spectrum_mask = cube["limit_mask"], cube["spectrum_mask"]
    rows = first_round["merged"]["rows"]
    assert rows["channels"].dtype == np.int32
    np.testing.assert_array_equal(rows["channels"], np.arange(*WINDOW))
    expected = first_round_channel_rows(
        cube["sky"][:, window], cube["primary_beam"][:, window], limit_mask[:, window]
    )
    for field in FirstRoundRows._fields:
        assert rows[field].dtype == expected[field].dtype
        np.testing.assert_array_equal(rows[field], expected[field][0, :, 0])
    tiles = [
        tile_occupancy(mask[:, window], TILE)[0, :, 0]
        for mask in (limit_mask, spectrum_mask)
    ]
    np.testing.assert_array_equal(rows["tiles"], np.stack(tiles, axis=1))
    for name, mask in (
        ("PRIMARY_BEAM_LIMIT_MASK", limit_mask),
        ("PRIMARY_BEAM_SPECTRUM_MASK", spectrum_mask),
    ):
        stored = zarr.open_array(os.path.join(first_round["store"], name), mode="r")[:]
        np.testing.assert_array_equal(
            unpack_plane_masks(stored[window], N_PIXELS), mask[0, window, 0]
        )
        assert not np.delete(stored, np.arange(*WINDOW), axis=0).any()
    assert first_round["written"] == {
        "complete": True,
        "written_channels": list(range(*WINDOW)),
    }

    valid = cube["valid"] & (cube["sky"] != 0)
    expected_integral, expected_maximum = _expected_moments(cube, valid)
    assert np.isnan(expected_integral).any() and np.isfinite(expected_integral).any()
    np.testing.assert_array_equal(first_round["integral"], expected_integral)
    np.testing.assert_array_equal(first_round["maximum"], expected_maximum)
    assert first_round["maximum"].dtype == np.float32


def test_first_round_timing_and_counters(cube, first_round):
    merged = first_round["merged"]
    assert merged["written_slots"] == [0, 1, 2]
    timing = merged["timing_node_tasks"]
    assert set(timing) == FREQUENCY_COLUMNS
    assert list(timing["family"]) == ["frequency"] * 3
    assert list(timing["pass_kind"]) == ["first_round"] * 3
    assert list(timing["slot"]) == [0, 1, 2]
    assert list(timing["first_channel"]) == [3, 25, 47]
    assert list(timing["last_channel"]) == [24, 46, 66]
    assert list(timing["n_channels"]) == [22, 22, 20]
    # Both arrays are read in every channel; the residual only inside the limit.
    outside = _units(~merged["rows"]["tiles"][:, 0])
    assert outside > 0
    assert timing["n_planes_read"].sum() == 2 * 64
    assert timing["n_chunks_skipped"].sum() == outside
    assert timing["n_chunks_read"].sum() == 2 * 64 * N_TILES - outside
    # Batches of 5 planes split at the shards (32, 64): 5 per task, a shard
    # opened per batch and array.
    assert timing["n_shard_opens"].sum() == 3 * 5 * 2
    assert (timing["bytes_read_sky"] > 0).all()
    assert (timing["bytes_read_primary_beam"] > 0).all()
    assert (timing["bytes_read_masks"] == 0).all()
    assert (timing["bytes_written"] > 0).all()
    for key in ("T_read", "T_masks", "T_first_round", "T_moments", "T_write_partials"):
        assert (timing[key] > 0).all()
    assert (timing[["T_spectrum", "T_noise"]] == 0).all().all()
    records = merged["timing_reduce_nodes"]
    assert [record["n_inputs"] for record in records] == [2, 2]
    assert [record["n_rows_out"] for record in records] == [2, 3]
    assert {record["family"] for record in records} == {"frequency"}

    merge_timing = first_round["merge_timing"]
    assert set(merge_timing) == MERGE_COLUMNS
    assert list(merge_timing["family"]) == ["merge"] * 4
    # Each tile reads one inner chunk per slot and partial.
    assert list(merge_timing["n_chunks_read"]) == [3 * 2] * 4
    assert (merge_timing["bytes_read_partials"] > 0).all()
    assert (merge_timing["bytes_written"] > 0).all()


def test_moments_with_the_store_equal_those_of_the_first_pass(
    cube, first_round, tmp_path
):
    partials_store, maps_store = _moment_stores(tmp_path)
    outputs = _frequency_tasks(
        find_continuum_first_round_task,
        cube["store"],
        primary_beam_limit=LIMIT,
        moment_channel_weights=cube["weights"],
        first_round_store=first_round["store"],
        compute_masks=False,
        compute_moments=True,
        partials_store=partials_store,
        planes_per_batch=3,
    )
    merged = _reduce(outputs)
    assert merged["rows"] == {} and merged["written_slots"] == [0, 1, 2]
    integral, maximum, _ = _merge(partials_store, maps_store, merged["written_slots"])
    np.testing.assert_array_equal(integral, first_round["integral"])
    np.testing.assert_array_equal(maximum, first_round["maximum"])
    timing = merged["timing_node_tasks"]
    outside = _units(~first_round["merged"]["rows"]["tiles"][:, 0])
    assert timing["n_planes_read"].sum() == 64
    assert timing["n_chunks_skipped"].sum() == outside
    assert timing["n_chunks_read"].sum() == 64 * N_TILES - outside
    assert (timing["bytes_read_primary_beam"] == 0).all()
    assert (timing["bytes_read_masks"] > 0).all()
    assert (timing["T_first_round"] == 0).all()


def _regions():
    l_index, m_index = np.indices((N_PIXELS, N_PIXELS))
    radius2 = (l_index - 36) ** 2 + (m_index - 28) ** 2
    return {
        "MEAN_REGION": radius2 < 64,
        "SCALED_MAD_REGION": (radius2 >= 64) & (radius2 < 300),
        "NOISE_REGION": (radius2 >= 100) & (radius2 < 400),
    }


@pytest.mark.parametrize(
    "use_spectrum_mask, scaled_mad, levels",
    [(True, True, (0.25, 0.6)), (True, False, None), (False, True, None)],
    ids=["normalization", "mean", "only_extra_mask_normalization"],
)
def test_spectrum_equals_the_cube_reductions(
    cube, first_round, tmp_path, use_spectrum_mask, scaled_mad, levels
):
    regions = _regions()
    names = ["MEAN_REGION", "SCALED_MAD_REGION"] if scaled_mad else ["MEAN_REGION"]
    tiles, counters = write_region_masks(
        str(tmp_path / "regions.zarr"), {name: regions[name] for name in names}, TILE
    )
    assert counters["bytes"] > 0
    outputs = _frequency_tasks(
        find_continuum_spectrum_task,
        cube["store"],
        first_round_store=first_round["store"],
        regions_store=str(tmp_path / "regions.zarr"),
        use_spectrum_mask=use_spectrum_mask,
        scaled_mad=scaled_mad,
        scaled_mad_primary_beam_levels=levels,
        region_tiles=tiles,
        planes_per_batch=4,
    )
    merged = _reduce(outputs)
    window = slice(*WINDOW)
    sky, valid = cube["sky"][:, window], cube["valid"][:, window]
    rows = merged["rows"]
    np.testing.assert_array_equal(rows["channels"], np.arange(*WINDOW))
    expected = masked_channel_means(
        sky,
        valid,
        regions["MEAN_REGION"],
        cube["spectrum_mask"][:, window] if use_spectrum_mask else None,
    )[0, :, 0]
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(rows["mean"], expected)
    assert set(rows) == {"channels", "mean"} | ({"scaled_mad"} if scaled_mad else set())
    if scaled_mad:
        expected = masked_channel_scaled_mads(
            sky,
            valid,
            regions["SCALED_MAD_REGION"],
            None if levels is None else cube["primary_beam"][:, window],
            levels,
        )[0, :, 0]
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(rows["scaled_mad"], expected)
    assert merged["written_slots"] == []

    timing = merged["timing_node_tasks"]
    stored_tiles = first_round["merged"]["rows"]["tiles"]
    mean_tiles = tiles["MEAN_REGION"]
    if use_spectrum_mask:
        mean_tiles = mean_tiles & stored_tiles[:, 1]
    scaled_mad_tiles = tiles["SCALED_MAD_REGION"] if scaled_mad else False
    sky_units = _units((mean_tiles | scaled_mad_tiles) & stored_tiles[:, 0])
    beam_units = 0 if levels is None else _units(scaled_mad_tiles & stored_tiles[:, 0])
    assert 0 < sky_units < 64 * N_TILES
    assert timing["n_chunks_read"].sum() == sky_units + beam_units
    assert timing["n_chunks_read"].sum() + timing["n_chunks_skipped"].sum() == (
        64 * N_TILES * (1 if levels is None else 2)
    )
    assert (timing["bytes_read_primary_beam"] > 0).all() == (levels is not None)
    assert (timing["T_spectrum"] > 0).all()
    assert (timing[["T_moments", "T_noise", "T_write_partials"]] == 0).all().all()
    assert set(timing["pass_kind"]) == {"spectrum"}


@pytest.fixture(scope="module")
def selection(cube, first_round):
    """Selected channels of the window, the noise region and its store."""
    rng = np.random.default_rng(11)
    channels = np.sort(rng.choice(np.arange(*WINDOW), 30, replace=False))
    directory = cube["directory"] / "selection"
    os.makedirs(directory, exist_ok=True)
    regions_store = str(directory / "regions.zarr")
    write_region_masks(
        regions_store, {"NOISE_REGION": _regions()["NOISE_REGION"]}, TILE
    )
    return {"channels": channels, "regions_store": regions_store}


def _selection_pass(cube, first_round, selection, store, directory, planes_per_batch):
    partials_store, maps_store = _moment_stores(directory)
    merged = _reduce(
        _frequency_tasks(
            find_continuum_selection_moments_task,
            store,
            first_round_store=first_round["store"],
            channels=selection["channels"],
            moment_channel_weights=cube["weights"],
            regions_store=selection["regions_store"],
            noise=True,
            partials_store=partials_store,
            planes_per_batch=planes_per_batch,
        )
    )
    return merged, _merge(partials_store, maps_store, merged["written_slots"])


@pytest.mark.parametrize("planes_per_batch", [1, 3, 5])
def test_selection_moments_and_noise_equal_the_cube_reductions(
    cube, first_round, selection, tmp_path, planes_per_batch
):
    channels = selection["channels"]
    merged, (integral, maximum, _) = _selection_pass(
        cube, first_round, selection, cube["store"], tmp_path, planes_per_batch
    )
    expected_integral, expected_maximum = _expected_moments(
        cube, cube["valid"], set(channels.tolist())
    )
    np.testing.assert_array_equal(integral, expected_integral)
    np.testing.assert_array_equal(maximum, expected_maximum)
    rows = merged["rows"]
    np.testing.assert_array_equal(rows["channels"], channels)
    medians, mads = masked_channel_medians_and_mads(
        cube["sky"][:, channels],
        cube["valid"][:, channels],
        _regions()["NOISE_REGION"],
    )
    np.testing.assert_array_equal(rows["noise_median"], medians[0, :, 0])
    np.testing.assert_array_equal(rows["noise_mad"], mads[0, :, 0])
    assert merged["written_slots"] == [0, 1, 2]
    timing = merged["timing_node_tasks"]
    outside = _units(~first_round["merged"]["rows"]["tiles"][channels - WINDOW[0], 0])
    assert timing["n_channels"].sum() == channels.size
    assert timing["n_chunks_read"].sum() == channels.size * N_TILES - outside
    assert timing["n_planes_read"].sum() == channels.size
    assert (timing["T_noise"] > 0).all() and (timing["T_moments"] > 0).all()
    assert set(timing["pass_kind"]) == {"selection_moments"}


@pytest.fixture(scope="module")
def unsharded_store(cube):
    """The cube in an unsharded store whose chunks hold 12 channels of one
    Stokes and 16 x 16 px, as stores converted from CASA images hold."""
    store = str(cube["directory"] / "unsharded.img.zarr")
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
    return store


@pytest.mark.parametrize("planes_per_batch", [12, 5])
def test_a_read_unit_of_many_channels(
    cube, first_round, selection, unsharded_store, tmp_path, planes_per_batch
):
    """On a store whose chunks hold 12 channels, a batch reads the selected
    channels of a chunk at once, the ones between them included, when they
    span at most ``planes_per_batch`` planes: a task then decodes each chunk
    once, otherwise once per batch. The results equal those of the sharded
    store."""
    merged, maps = _selection_pass(
        cube,
        first_round,
        selection,
        unsharded_store,
        tmp_path / "unsharded",
        planes_per_batch,
    )
    sharded, sharded_maps = _selection_pass(
        cube, first_round, selection, cube["store"], tmp_path / "sharded", 5
    )
    for values, expected in zip(maps[:2], sharded_maps[:2], strict=True):
        np.testing.assert_array_equal(values, expected)
    for key, values in merged["rows"].items():
        np.testing.assert_array_equal(values, sharded["rows"][key])
    timing = merged["timing_node_tasks"]
    channels = set(selection["channels"].tolist())
    reads = []
    for row, chunk in zip(timing.itertuples(), CHUNKS, strict=True):
        selected = [c for c in range(chunk.start, chunk.stop) if c in channels]
        units = len({c // 12 for c in selected}) * N_TILES
        reads.append(row.n_chunks_read + row.n_chunks_skipped - units)
        assert row.n_planes_read <= selected[-1] - selected[0] + 1
    assert (timing["n_planes_read"] > timing["n_channels"]).any()
    if planes_per_batch == 12:
        assert reads == [0, 0, 0]
    else:
        assert min(reads) >= 0 and max(reads) > 0


def test_merge_of_unwritten_and_missing_slots(cube, tmp_path):
    partials_store, maps_store = _moment_stores(tmp_path)
    integral, maximum, timing = _merge(partials_store, maps_store, [])
    assert np.isnan(integral).all() and np.isnan(maximum).all()
    assert (timing["n_chunks_read"] == 0).all()
    with pytest.raises(FileNotFoundError, match="listed as written"):
        merge_find_continuum_moment_partials(
            partials_store, maps_store, [1], len(CHUNKS), "float32"
        )


def test_a_transient_error_of_a_slot_stat_is_retried(
    cube, first_round, tmp_path, monkeypatch
):
    """A Lustre eviction fails the stat of a listed slot with EIO: the stat is
    repeated instead of taken for a missing slot, and the maps are those of a
    merge without it."""
    from astroviper.node_tasks.imaging.utils import skunk_works

    partials_store, maps_store = _moment_stores(tmp_path)
    outputs = _frequency_tasks(
        find_continuum_first_round_task,
        cube["store"],
        primary_beam_limit=LIMIT,
        moment_channel_weights=cube["weights"],
        first_round_store=first_round["store"],
        compute_masks=False,
        compute_moments=True,
        partials_store=partials_store,
        planes_per_batch=3,
    )
    written_slots = _reduce(outputs)["written_slots"]
    monkeypatch.setattr(skunk_works, "_EIO_RETRY_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    stat, failures = os.stat, []

    def evicted_once(path, *args, **kwargs):
        shard = os.path.join(partials_store, "SKY_MOMENT_INTEGRATED", "c", "")
        if os.fspath(path).startswith(shard) and not failures:
            failures.append(path)
            raise OSError(errno.EIO, "Input/output error")
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", evicted_once)
    integral, maximum, _ = _merge(partials_store, maps_store, written_slots)
    assert len(failures) == 1
    np.testing.assert_array_equal(integral, first_round["integral"])
    np.testing.assert_array_equal(maximum, first_round["maximum"])


def test_reduce_concatenates_rows_and_folds_resource_series():
    def leaf(task_id, channels, slots):
        channels = np.array(channels, dtype=np.int32)
        return {
            "rows": {"channels": channels, "mean": channels * 1.5},
            "written_slots": slots,
            "timing_node_tasks": pd.DataFrame({"task_id": [task_id]}),
            "resource_usage": {
                "time_seconds": [0.0, 0.1],
                "cpu_percent": [50.0, 90.0 + task_id],
                "sample_interval_seconds": 0.1,
            },
        }

    first = reduce_find_continuum_channel_rows(
        [leaf(0, [0, 1], [0]), leaf(1, [2], [])], {}
    )
    merged = reduce_find_continuum_channel_rows([first, leaf(2, [3, 4], [2])], {})
    np.testing.assert_array_equal(merged["rows"]["channels"], np.arange(5))
    np.testing.assert_array_equal(merged["rows"]["mean"], 1.5 * np.arange(5))
    assert merged["written_slots"] == [0, 2]
    timing = merged["timing_node_tasks"]
    assert list(timing["task_id"]) == [0, 1, 2]
    assert timing["cpu_percent"].iloc[2] == [50.0, 92.0]
    assert (timing["sample_interval_seconds"] == 0.1).all()
    records = merged["timing_reduce_nodes"]
    assert [record["n_inputs"] for record in records] == [2, 2]
    assert [record["n_rows_out"] for record in records] == [2, 3]
    assert all(record["T_reduce_node"] >= 0 for record in records)
    frame = with_resource_usage(pd.DataFrame({"task_id": [7]}), {"cpu_percent": [1.0]})
    assert frame["cpu_percent"].iloc[0] == [1.0]


def test_a_pass_in_graphviper_with_the_resource_monitor(cube, first_round, tmp_path):
    """The spectrum task as a GraphVIPER node: the explicit signature takes
    its parameters from the mapping, and the reduce folds in the monitor's
    series."""
    regions = {"MEAN_REGION": _regions()["MEAN_REGION"]}
    tiles, _ = write_region_masks(str(tmp_path / "regions.zarr"), regions, TILE)
    index_xds = xr.Dataset(
        coords={
            dim: np.arange(n, dtype=np.float64)
            for dim, n in zip(
                ("frequency", "l", "m"), (N_CHANNELS, N_PIXELS, N_PIXELS), strict=True
            )
        }
    )
    mapping = interpolate_data_coords_onto_parallel_coords(
        {
            "frequency": make_parallel_coord(
                coord=index_xds.frequency[slice(*WINDOW)], n_chunks=3
            )
        },
        {"img": index_xds},
    )
    graph = graph_tools.map(
        input_data={"img": index_xds},
        node_task_data_mapping=mapping,
        node_task=find_continuum_spectrum_task,
        input_params={
            "input_image_store": cube["store"],
            "variables": VARIABLES,
            "polarization_index": STOKES_I,
            "channel_range": list(WINDOW),
            "first_round_store": first_round["store"],
            "regions_store": str(tmp_path / "regions.zarr"),
            "use_spectrum_mask": True,
            "scaled_mad": False,
            "scaled_mad_primary_beam_levels": None,
            "region_tiles": tiles,
            "planes_per_batch": 4,
            "input_checked": True,
        },
        in_memory_compute=False,
        monitor_resources_seconds=0.01,
    )
    graph = graph_tools.reduce(
        graph, reduce_find_continuum_channel_rows, {}, mode="tree"
    )
    with dask.config.set(scheduler="synchronous"):
        output = dask.compute(graph_tools.generate_dask_workflow(graph))[0]
    direct = _reduce(
        _frequency_tasks(
            find_continuum_spectrum_task,
            cube["store"],
            first_round_store=first_round["store"],
            regions_store=str(tmp_path / "regions.zarr"),
            use_spectrum_mask=True,
            scaled_mad=False,
            scaled_mad_primary_beam_levels=None,
            region_tiles=tiles,
            planes_per_batch=4,
        )
    )
    for key, values in direct["rows"].items():
        np.testing.assert_array_equal(output["rows"][key], values)
    timing = output["timing_node_tasks"]
    assert list(timing["task_id"]) == [0, 1, 2]
    assert all(isinstance(series, list) for series in timing["memory_rss_bytes"])
    assert len(output["timing_reduce_nodes"]) == 2


def test_driver_reads_of_the_primary_beam_and_writes(cube, tmp_path):
    channel = 33
    plane, limit_mask, counters = read_primary_beam_plane(
        cube["store"], "PRIMARY_BEAM", STOKES_I, channel, LIMIT
    )
    expected = cube["primary_beam"][:, channel : channel + 1]
    assert plane.shape == MAP_SHAPE and plane.dtype == np.float32
    np.testing.assert_array_equal(plane, expected)
    np.testing.assert_array_equal(
        limit_mask, cube["limit_mask"][:, channel : channel + 1]
    )
    assert counters["units_read"] == N_TILES and counters["bytes"] > 0

    lines, lines_valid, counters = read_primary_beam_edge_lines(
        cube["store"], "PRIMARY_BEAM", STOKES_I, channel, LIMIT
    )
    expected = expected[0, 0, 0]
    for name, values in (
        ("first_l", expected[0]),
        ("last_l", expected[-1]),
        ("first_m", expected[:, 0]),
        ("last_m", expected[:, -1]),
    ):
        np.testing.assert_array_equal(lines[name], values)
        np.testing.assert_array_equal(lines_valid[name], ~(values < LIMIT))
    # The 12 edge tiles, the 4 corners read twice.
    assert counters["units_read"] == 16

    regions = _regions()
    region = xr.DataArray(regions["NOISE_REGION"][None, None, None], dims=DIMS)
    store = str(tmp_path / "regions.zarr")
    tiles, _ = write_region_masks(store, {"NOISE_REGION": region}, TILE)
    np.testing.assert_array_equal(
        tiles["NOISE_REGION"],
        tile_occupancy(region.values, TILE)[0, 0, 0],
    )
    stored = zarr.open_group(store, mode="r")
    np.testing.assert_array_equal(
        unpack_plane_masks(stored["NOISE_REGION"][...], N_PIXELS),
        regions["NOISE_REGION"],
    )
    assert stored["NOISE_REGION"].metadata.dimension_names == ("l", "m_packed")
    np.testing.assert_array_equal(
        stored["NOISE_REGION_TILES"][...], tiles["NOISE_REGION"]
    )

    joint_mask = xr.Dataset(
        {
            "JOINT_MASK": (DIMS, regions["MEAN_REGION"][None, None, None]),
            "JOINT_MASK_FOOTPRINT": (DIMS, cube["limit_mask"][:, 33:34]),
        },
        coords={"l": np.arange(N_PIXELS) * 1e-6, "polarization": ["I"]},
        attrs={"kind": "phase_1", "primary_beam_based": False},
    )
    counters = write_joint_mask_store(str(tmp_path / "joint_mask.zarr"), joint_mask)
    assert counters["bytes"] > 0
    xr.testing.assert_identical(
        xr.open_zarr(str(tmp_path / "joint_mask.zarr"), consolidated=False).load(),
        joint_mask,
    )


def test_a_transient_error_of_a_slot_write_is_retried(
    cube, first_round, selection, tmp_path, monkeypatch
):
    """A Lustre eviction fails one write of a slot with EIO: the write is
    repeated on a fresh open, and the maps are those of a pass without it."""
    from astroviper.node_tasks.imaging.utils import skunk_works

    _, expected = _selection_pass(
        cube, first_round, selection, cube["store"], tmp_path / "expected", 5
    )
    monkeypatch.setattr(skunk_works, "_EIO_RETRY_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    open_array, failures = zarr.open_array, []

    def evicted_once(path, *args, **kwargs):
        if kwargs.get("mode") == "r+" and "partials.zarr" in str(path) and not failures:
            failures.append(path)
            raise OSError(errno.EIO, "Input/output error")
        return open_array(path, *args, **kwargs)

    monkeypatch.setattr(zarr, "open_array", evicted_once)
    _, maps = _selection_pass(
        cube, first_round, selection, cube["store"], tmp_path / "evicted", 5
    )
    assert len(failures) == 1
    for values, reference in zip(maps[:2], expected[:2], strict=True):
        np.testing.assert_array_equal(values, reference)


def test_the_tasks_check_their_configuration_unless_checked(
    cube, first_round, selection, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        param_check_find_continuum,
        "check_find_continuum_ultra_task_config",
        lambda *arguments: calls.append(arguments),
        raising=False,
    )
    partials_store, _ = _moment_stores(tmp_path)
    tiles, _ = write_region_masks(
        str(tmp_path / "regions.zarr"), {"MEAN_REGION": _regions()["MEAN_REGION"]}, TILE
    )
    common = {
        "input_image_store": cube["store"],
        "variables": VARIABLES,
        "polarization_index": STOKES_I,
        "channel_range": list(WINDOW),
        "first_round_store": first_round["store"],
        "planes_per_batch": 2,
        "data_selection": {"img": {"frequency": slice(5, 9)}},
        "chunk_indices": (0,),
    }
    for checked in (False, True):
        find_continuum_first_round_task(
            **common,
            primary_beam_limit=LIMIT,
            moment_channel_weights=cube["weights"],
            compute_masks=False,
            compute_moments=True,
            partials_store=partials_store,
            input_checked=checked,
        )
        find_continuum_spectrum_task(
            **common,
            regions_store=str(tmp_path / "regions.zarr"),
            use_spectrum_mask=False,
            scaled_mad=False,
            scaled_mad_primary_beam_levels=None,
            region_tiles=tiles,
            input_checked=checked,
        )
        find_continuum_selection_moments_task(
            **common,
            channels=selection["channels"],
            moment_channel_weights=cube["weights"],
            regions_store=None,
            noise=False,
            partials_store=partials_store,
            input_checked=checked,
        )
    assert calls == [(2, STOKES_I, list(WINDOW))] * 3


def test_refusals(cube, first_round, tmp_path):
    arguments = dict(
        input_image_store=cube["store"],
        variables=VARIABLES,
        polarization_index=STOKES_I,
        primary_beam_limit=LIMIT,
        channel_range=list(WINDOW),
        moment_channel_weights=cube["weights"],
        first_round_store=first_round["store"],
        compute_masks=False,
        compute_moments=True,
        partials_store=None,
        planes_per_batch=2,
        input_checked=True,
    )
    with pytest.raises(ValueError, match="not in the window"):
        find_continuum_first_round_task(
            **arguments, data_selection={"img": {"frequency": slice(0, 10)}}
        )
    store = str(tmp_path / "planes.zarr")
    zarr.create_array(
        os.path.join(store, "SKY_RESIDUAL"),
        shape=(2, N_CHANNELS, 1, 4, 4),
        dtype=np.float32,
        dimension_names=DIMS,
    )
    with pytest.raises(ValueError, match="not one time"):
        find_continuum_first_round_task(**dict(arguments, input_image_store=store))
    with pytest.raises(ValueError, match="pass_kind"):
        find_continuum_ultra_task_bytes(
            "moments", 8, 8, 1, 4, 4, False, 0, True, 0.0, 0
        )


@pytest.mark.parametrize(
    "arguments, gib",
    [
        (("first_round", True, 0, True, 0.0), 11.3),  # without a store
        (("first_round", False, 0, True, 0.0), 7.3),  # store without moments
        (("spectrum", False, 1, False, 0.30), 6.5),
        (("spectrum", True, 2, False, 0.30), 10.4),  # with normalization
        (("selection_moments", False, 1, True, 0.19), 7.5),  # round original
        (("selection_moments", False, 0, True, 0.0), 7.1),
    ],
)
def test_task_memory_bounds_at_the_medium_scale(arguments, gib):
    """The bounds at 11250^2, 4 planes per batch and 625^2 read units, within
    0.1 GiB of design section 9.2, below the task budget of 15.6 GiB."""
    pass_kind, read_primary_beam, n_regions, moments, gather_fraction = arguments
    task_bytes = find_continuum_ultra_task_bytes(
        pass_kind,
        11250,
        11250,
        4,
        4,
        4,
        read_primary_beam,
        n_regions,
        moments,
        gather_fraction,
        625 * 625 * 4,
    )
    assert round(task_bytes / GiB, 1) == gib


def test_task_memory_counts_its_terms():
    n = 1000

    def task_bytes(pass_kind="selection_moments", **changes):
        arguments = dict(
            n_l=n,
            n_m=n,
            planes_per_batch=4,
            sky_itemsize=4,
            primary_beam_itemsize=4,
            read_primary_beam=False,
            n_regions=0,
            moments=False,
            gather_fraction=0.0,
            read_unit_bytes=0,
        )
        return find_continuum_ultra_task_bytes(pass_kind, **(arguments | changes))

    base = task_bytes()
    packed = n * 125
    assert base == int(
        1.5 * GiB + 2 * 4 * (4 * n * n + packed) + 2 * n * n + 2 * 8 * 625 * n
    )
    assert task_bytes(sky_itemsize=8) - base == 2 * 4 * 4 * n * n
    assert task_bytes(read_primary_beam=True) - base == 2 * 4 * 4 * n * n
    assert task_bytes(n_regions=2) - base == 2 * n * n
    assert task_bytes(moments=True) - base == 12 * n * n
    assert (
        task_bytes(moments=True, sky_itemsize=8) - task_bytes(sky_itemsize=8)
        == 16 * n * n
    )
    assert task_bytes(gather_fraction=0.5) - base == 6 * n * n
    assert task_bytes(read_unit_bytes=10**6) - base == 2 * 10**6
    assert task_bytes("spectrum") - base == 2 * 4 * packed + n * n
    first_round = task_bytes("first_round", read_primary_beam=True, moments=True)
    assert first_round - task_bytes(read_primary_beam=True, moments=True) == 3 * n * n
    assert task_bytes("first_round") - base == -n * n


def test_merge_task_memory():
    assert find_continuum_ultra_merge_task_bytes(1000, 1) == int(1.5 * GiB + 33 * 1000)
    assert find_continuum_ultra_merge_task_bytes(1000, 120) == int(
        1.5 * GiB + (16 * 8 + 17) * 1000
    )
    assert (
        round(find_continuum_ultra_merge_task_bytes(1250 * 1250, 120) / GiB, 1) == 1.7
    )
