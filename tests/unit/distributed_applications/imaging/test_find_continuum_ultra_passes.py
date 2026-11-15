"""Tests of the passes of find_continuum_ultra and their layout.

The passes run on a cube written channel by channel by the sharded chunk
writer, through GraphVIPER's map and reduce on the synchronous scheduler, and
their results are compared bit for bit with the processing functions of
``find_continuum_ultra.cube_reductions`` applied to the whole cube.
"""

import errno
import math
import os

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from xradio.image import make_empty_sky_image

from astroviper.distributed_applications.imaging import find_continuum_ultra_passes
from astroviper.distributed_applications.imaging.find_continuum_first_round import (
    create_find_continuum_first_round_store,
)
from astroviper.distributed_applications.imaging.find_continuum_ultra_passes import (
    FREQUENCY_TASKS_PER_THREAD,
    FirstRoundReductions,
    SelectionReductions,
    SpectrumReductions,
    find_continuum_ultra_driver_bytes,
    plan_find_continuum_ultra_layout,
    plan_planes_per_batch,
    run_first_round_pass,
    run_selection_moments_pass,
    run_spectrum_pass,
)
from astroviper.node_tasks.imaging.find_continuum_ultra import (
    find_continuum_ultra_merge_task_bytes,
    find_continuum_ultra_task_bytes,
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
    unpack_plane_masks,
    valid_voxels,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    find_continuum_settings,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    AnnulusLevels,
    FirstRoundRows,
    SpectrumRegions,
)
from astroviper.utils.chunk_io import write_zarr_sharded
from astroviper.utils.io import (
    create_empty_data_variables_on_disk,
    write_zarr_image_store,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    RAD_PER_ARCSEC,
    make_find_continuum_image_xds,
)

GiB = 2**30
DIMS = ["time", "frequency", "polarization", "l", "m"]
VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
# 48 channels in shards of 16, Stokes I (index 0) and Q in one chunk, 48 px in
# chunks of 16 px; the window [3, 45) in GraphVIPER's three frequency chunks
# (14 channels each), merged in tiles of 16 px.
N_CHANNELS, N_PIXELS, TILE, SHARD, LIMIT = 48, 48, (16, 16), 16, 0.2
WINDOW = (3, 45)
N_WINDOW = WINDOW[1] - WINDOW[0]
MAP_SHAPE = (1, 1, 1, N_PIXELS, N_PIXELS)
THREADS = {"n_threads": 2, "memory_per_thread": 4.0}
ROW_COLUMNS = [
    "pass_index",
    "pass_kind",
    "round_index",
    "round_kind",
    "purpose",
    "data_variables_read",
    "computes_first_round_masks",
    "n_moment_sets",
    "n_tasks",
    "n_channels_read",
    "n_planes_read",
    "n_chunks_read",
    "n_chunks_skipped",
    "bytes_read_sky",
    "bytes_read_primary_beam",
    "planes_per_batch",
    "task_memory_plan_bytes",
    "merge_task_memory_plan_bytes",
    "task_rss_peak_bytes",
    "graph_start_unixtime",
    "start_unixtime",
    "compute_end_unixtime",
    "merge_start_unixtime",
    "merge_end_unixtime",
    "T_create_map_reduce_graph",
    "T_generate_dask_graph",
    "T_compute_dask_graph",
    "T_merge_create_graph",
    "T_merge_generate_dask_graph",
    "T_merge_compute_dask_graph",
    "bytes_partials_written",
    "bytes_partials_read",
    "bytes_maps_written",
    "T_write_regions",
    "T_first_round_rows",
    "T_read_maps",
    "driver_bytes_read",
    "driver_bytes_written",
]


@pytest.fixture(scope="module")
def cube(tmp_path_factory):
    """Stokes I and Q written channel by channel by the sharded chunk
    writer, metadata consolidated; the primary beam is below the limit in the
    corner chunks."""
    directory = tmp_path_factory.mktemp("cube")
    source = make_find_continuum_image_xds(
        n_pixels=N_PIXELS,
        n_channels=N_CHANNELS,
        primary_beam_fwhm_arcsec=3.0,
        lines=(
            dict(
                channel=20,
                fwhm_channels=3,
                peak_jy=0.04,
                offset_pixels=(3, -3),
                size_fwhm_pixels=5,
            ),
        ),
    )
    sky = source["SKY_RESIDUAL"].values.copy()
    primary_beam = source["PRIMARY_BEAM"].values.copy()
    sky[0, 10, 0, 20:26, 20:26] = 0.0
    sky[0, 11, 0, 22, 22] = -0.0
    sky[0, 12, 0, 23, 23] = np.nan
    primary_beam[0, 14, 0, 24, 25] = np.nan
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
        image_chunking={"polarization": 2, "l": TILE[0], "m": TILE[1]},
        image_sharding={"frequency": SHARD},
    )
    for channel in range(N_CHANNELS):
        write_zarr_sharded(
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
    zarr.consolidate_metadata(store)
    window_xds = xr.open_zarr(store, chunks=None).isel(frequency=slice(*WINDOW))
    n_window = window_xds.sizes["frequency"]
    beams = xr.DataArray(
        np.full((1, n_window, 1, 3), RAD_PER_ARCSEC),
        dims=["time", "frequency", "polarization", "beam_params_label"],
        coords={"polarization": ["I"]},
    )
    settings = find_continuum_settings(
        window_xds.frequency.values,
        window_xds.l.values,
        window_xds.m.values,
        float(window_xds.frequency.attrs["rest_frequency"]["data"]),
        beams,
        np.full(n_window, 0.9),
        np.full(n_window, 20.0),
        "ALMA",
        False,
        1,
        None,
    )
    limit_mask, spectrum_mask = primary_beam_masks(primary_beam, LIMIT)
    return {
        "store": store,
        "directory": directory,
        "window_xds": window_xds,
        "settings": settings,
        "sky": sky,
        "primary_beam": primary_beam,
        "limit_mask": limit_mask,
        "spectrum_mask": spectrum_mask,
        "valid": valid_voxels(sky, limit_mask),
    }


@pytest.fixture(scope="module")
def layout(cube):
    """The layout of the window in three frequency chunks."""
    return plan_find_continuum_ultra_layout(
        cube["window_xds"], VARIABLES, list(WINDOW), {"frequency": 3}, THREADS, None
    )


def _new_store(cube, layout, path, channel_range=WINDOW):
    """An empty first-round store of the window, as the driver makes it."""
    settings = cube["settings"]
    create_find_continuum_first_round_store(
        path,
        N_CHANNELS,
        (1, N_CHANNELS, 2, N_PIXELS, N_PIXELS),
        layout.tile_shape,
        layout.merge_tile_shape,
        LIMIT,
        VARIABLES,
        layout.polarization_index,
        layout.value_dtypes,
        None,
        "find_continuum_ultra",
        [
            {
                "channel_range": list(channel_range),
                "first_frequency": settings.first_frequency,
                "channel_increment": settings.channel_increment,
                "n_channels": settings.n_channels,
            }
        ],
        settings.rest_frequency,
    )
    return path


@pytest.fixture(scope="module")
def first_round(cube, layout):
    """The first-round pass without a store: masks, rows and moments."""
    store = _new_store(cube, layout, str(cube["directory"] / "first_round.zarr"))
    pass_directory = str(cube["directory"] / "pass_0")
    with dask.config.set(scheduler="synchronous"):
        reductions, records = run_first_round_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            store,
            True,
            True,
            pass_directory,
            None,
            None,
            0,
        )
    return {
        "store": store,
        "pass_directory": pass_directory,
        "reductions": reductions,
        "records": records,
    }


def _expected_moments(cube, layout, valid, channels=None):
    """The maps of the processing functions on the whole cube: per frequency
    chunk the moment partials of its (selected) channels, merged in
    GraphVIPER's binary tree."""
    leaves = []
    for item in layout.frequency_mapping.values():
        frequency = item["data_selection"]["img"]["frequency"]
        selected = [
            c
            for c in range(frequency.start, frequency.stop)
            if channels is None or c in channels
        ]
        partials = None
        if selected:
            partials = new_moment_partials(MAP_SHAPE, np.float32)
            accumulate_moments(
                partials,
                cube["sky"][:, selected],
                valid[:, selected],
                cube["settings"].moment_channel_weights[
                    np.subtract(selected, WINDOW[0])
                ],
            )
        leaves.append(partials)
    return [
        values[0, 0, 0]
        for values in finalize_moments(
            merge_moment_partials_in_tree_order(leaves), MAP_SHAPE, np.float32
        )
    ]


def _regions():
    """Regions of the spectrum and selection passes, image-shaped."""
    l_index, m_index = np.indices((N_PIXELS, N_PIXELS))
    radius2 = (l_index - 26) ** 2 + (m_index - 20) ** 2
    regions = {
        "MEAN_REGION": radius2 < 64,
        "SCALED_MAD_REGION": (radius2 >= 64) & (radius2 < 300),
        "NOISE_REGION": (radius2 >= 100) & (radius2 < 400),
    }
    return {
        name: xr.DataArray(region[None, None, None], dims=DIMS, name=name)
        for name, region in regions.items()
    }


def _check_record(record, layout, pass_index, pass_kind, round_kind, n_merges):
    """The row's columns and the labels of the task rows and reduce records."""
    assert list(record.row) == ROW_COLUMNS
    assert record.row["pass_index"] == pass_index
    assert record.row["pass_kind"] == pass_kind
    assert record.row["round_kind"] == round_kind
    timing = record.timing_node_tasks
    assert set(timing["pass_index"]) == {pass_index}
    assert set(timing["pass_kind"]) == {pass_kind}
    assert list(timing["round_kind"]).count(round_kind) == len(timing)
    frequency = timing[timing["family"] == "frequency"]
    assert len(frequency) == record.row["n_tasks"]
    assert (frequency["n_planes_read"] > 0).all()
    assert (timing["family"] == "merge").sum() == n_merges
    assert record.row["n_planes_read"] == frequency["n_planes_read"].sum()
    assert record.row["n_channels_read"] == frequency["n_channels"].sum()
    for reduce_record in record.timing_reduce_nodes:
        assert reduce_record["pass_index"] == pass_index
        assert reduce_record["pass_kind"] == pass_kind
        assert reduce_record["round_kind"] == round_kind
        assert reduce_record["family"] == "frequency"
    assert len(record.timing_reduce_nodes) == record.row["n_tasks"] - 1
    row = record.row
    assert row["graph_start_unixtime"] <= row["start_unixtime"]
    assert row["start_unixtime"] < row["compute_end_unixtime"]
    assert math.isclose(
        row["T_create_map_reduce_graph"]
        + row["T_generate_dask_graph"]
        + row["T_compute_dask_graph"],
        row["compute_end_unixtime"] - row["graph_start_unixtime"],
        abs_tol=1e-6,
    )
    if n_merges:
        assert row["compute_end_unixtime"] <= row["merge_start_unixtime"]
        assert row["merge_start_unixtime"] < row["merge_end_unixtime"]
        assert row["merge_task_memory_plan_bytes"] == (
            find_continuum_ultra_merge_task_bytes(
                math.prod(layout.merge_tile_shape), layout.n_frequency_tasks
            )
        )
        for key in (
            "bytes_partials_written",
            "bytes_partials_read",
            "bytes_maps_written",
            "T_read_maps",
            "driver_bytes_read",
        ):
            assert row[key] > 0, key
    else:
        assert np.isnan(row["merge_start_unixtime"])
        assert np.isnan(row["merge_end_unixtime"])
        assert row["merge_task_memory_plan_bytes"] == 0
        assert row["T_merge_compute_dask_graph"] == 0.0
        assert row["bytes_partials_written"] == row["bytes_maps_written"] == 0


def test_layout_of_a_window(cube, layout):
    store = os.path.abspath(cube["store"])
    assert layout.input_image_store == store
    assert layout.variables == VARIABLES
    assert layout.polarization_index == 0
    assert layout.channel_range == list(WINDOW)
    assert layout.tile_shape == TILE
    assert layout.value_dtypes == {"sky": "float32", "primary_beam": "float32"}
    assert layout.shard_length == SHARD
    # An inner chunk holds one channel of Stokes I and Q.
    assert layout.read_unit_bytes == 2 * TILE[0] * TILE[1] * 4
    assert layout.read_unit_planes == 1
    assert layout.plane_bytes == N_PIXELS * N_PIXELS * 4
    assert layout.n_threads == 2
    # The chunks of GraphVIPER, in channels of the cube.
    assert (layout.n_frequency_tasks, layout.channels_per_task) == (3, 14)
    assert [
        item["data_selection"]["img"]["frequency"]
        for item in layout.frequency_mapping.values()
    ] == [slice(3, 17), slice(17, 31), slice(31, 45)]
    # Merge tiles of two 16 px chunks, whole chunks of the maps.
    assert layout.merge_tile_shape == (24, 24)
    assert [
        tuple(item["data_selection"]["img"].values())
        for item in layout.tile_mapping.values()
    ] == [
        (slice(0, 24), slice(0, 24)),
        (slice(0, 24), slice(24, 48)),
        (slice(24, 48), slice(0, 24)),
        (slice(24, 48), slice(24, 48)),
    ]
    # 0.8 x 4 GiB less a baseline of min(1.5, 0.8) GiB.
    assert layout.budget_bytes == int(2.4 * GiB)
    assert layout.memory_per_thread_bytes == 4 * GiB


@pytest.mark.parametrize(
    "window, n_threads, slices",
    [
        ((3, 35), 1, [slice(3, 19), slice(19, 35)]),
        # ceil(42 / 16) chunks, which GraphVIPER makes of equal length.
        ((3, 45), 1, [slice(3, 17), slice(17, 31), slice(31, 45)]),
        ((3, 45), 16, [slice(3, 17), slice(17, 31), slice(31, 45)]),
    ],
)
def test_layout_defaults_and_budget(cube, window, n_threads, slices):
    """As many chunks as about FREQUENCY_TASKS_PER_THREAD tasks per thread of
    whole shards (one shard at least) need, and a budget clamped by
    memory_budget_gb."""
    window_xds = xr.open_zarr(cube["store"], chunks=None).isel(frequency=slice(*window))
    channels_per_task = SHARD * max(
        1,
        round(
            (window[1] - window[0]) / (FREQUENCY_TASKS_PER_THREAD * n_threads * SHARD)
        ),
    )
    assert channels_per_task == SHARD
    layout = plan_find_continuum_ultra_layout(
        window_xds,
        VARIABLES,
        list(window),
        None,
        {"n_threads": n_threads, "memory_per_thread": 64.0},
        3.0,
    )
    assert [
        item["data_selection"]["img"]["frequency"]
        for item in layout.frequency_mapping.values()
    ] == slices
    assert layout.n_frequency_tasks == len(slices)
    assert layout.channels_per_task == slices[0].stop - slices[0].start
    assert layout.budget_bytes == int(3.0 * GiB)
    assert layout.memory_per_thread_bytes == 64 * GiB
    layout = plan_find_continuum_ultra_layout(
        window_xds,
        VARIABLES,
        list(window),
        None,
        {"n_threads": n_threads, "memory_per_thread": 2.0},
        None,
    )
    assert layout.budget_bytes == int((0.8 * 2.0 - 0.4) * GiB)


def test_layout_refusals(cube):
    window_xds = cube["window_xds"]
    with pytest.raises(ValueError, match="open the image store"):
        plan_find_continuum_ultra_layout(
            window_xds.copy().drop_encoding(),
            VARIABLES,
            list(WINDOW),
            None,
            THREADS,
            None,
        )
    # Even a spectrum task at one plane per batch needs its 1.5 GiB baseline.
    with pytest.raises(ValueError, match="memory per thread"):
        plan_find_continuum_ultra_layout(
            window_xds,
            VARIABLES,
            list(WINDOW),
            None,
            {"n_threads": 1, "memory_per_thread": 1.0},
            None,
        )


def test_planes_per_batch(layout, monkeypatch):
    warnings = []
    monkeypatch.setattr(find_continuum_ultra_passes.logger, "warning", warnings.append)
    calls = []

    def task_bytes(planes_per_batch):
        calls.append(planes_per_batch)
        return 100 + 10 * planes_per_batch

    fits = layout._replace(budget_bytes=140, memory_per_thread_bytes=1000)
    assert plan_planes_per_batch(task_bytes, fits) == (4, 140)
    # Halved while over the budget.
    assert plan_planes_per_batch(task_bytes, fits._replace(budget_bytes=125)) == (
        2,
        120,
    )
    assert plan_planes_per_batch(task_bytes, fits._replace(budget_bytes=111)) == (
        1,
        110,
    )
    # At most a shard; the planes of a read unit within a task when more.
    assert plan_planes_per_batch(task_bytes, fits._replace(shard_length=2)) == (2, 120)
    many = fits._replace(read_unit_planes=12, shard_length=32, budget_bytes=200)
    assert plan_planes_per_batch(task_bytes, many) == (6, 160)
    assert calls[-3:] == [12, 6, 6]
    assert not warnings
    # Over the budget at one plane per batch: a warning; above a thread, refused.
    assert plan_planes_per_batch(task_bytes, fits._replace(budget_bytes=100)) == (
        1,
        110,
    )
    assert len(warnings) == 1 and "more than its budget" in warnings[0]
    with pytest.raises(ValueError, match="memory per thread"):
        plan_planes_per_batch(task_bytes, fits._replace(memory_per_thread_bytes=109))


def test_first_round_pass_without_a_store(cube, layout, first_round):
    reductions, records = first_round["reductions"], first_round["records"]
    assert isinstance(reductions, FirstRoundReductions)
    assert reductions.first_round_store == first_round["store"]
    window = slice(*WINDOW)
    expected = first_round_channel_rows(
        cube["sky"][:, window],
        cube["primary_beam"][:, window],
        cube["limit_mask"][:, window],
    )
    for field in FirstRoundRows._fields:
        values = getattr(reductions.rows, field)
        assert values.dtype == expected[field].dtype
        np.testing.assert_array_equal(values, expected[field][0, :, 0])
    valid = cube["valid"] & (cube["sky"] != 0)
    integral, maximum = _expected_moments(cube, layout, valid)
    assert np.isnan(integral).any() and np.isfinite(integral).any()
    np.testing.assert_array_equal(reductions.integral, integral)
    np.testing.assert_array_equal(reductions.maximum, maximum)
    assert reductions.integral.dtype == np.float64
    assert reductions.maximum.dtype == np.float32

    # The masks and rows are in the store, complete; the pass directory is gone.
    group = zarr.open_group(first_round["store"], mode="r")
    assert group.attrs["find_continuum_first_round"]["complete"]
    np.testing.assert_array_equal(
        unpack_plane_masks(group["PRIMARY_BEAM_LIMIT_MASK"][window], N_PIXELS),
        cube["limit_mask"][0, window, 0],
    )
    np.testing.assert_array_equal(
        group["SKY_VALID_COUNT"][window], expected["sky_valid_count"][0, :, 0]
    )
    assert not os.path.exists(first_round["pass_directory"])

    (record,) = records
    _check_record(record, layout, 0, "first_round", None, len(layout.tile_mapping))
    row = record.row
    assert row["round_index"] is None
    assert (
        row["purpose"] == "first round: primary-beam masks, per-channel rows, moments"
    )
    assert row["data_variables_read"] == "PRIMARY_BEAM, SKY_RESIDUAL"
    assert row["computes_first_round_masks"] is True
    assert row["n_moment_sets"] == 1
    assert row["n_tasks"] == 3 and row["n_channels_read"] == N_WINDOW
    assert row["n_planes_read"] == 2 * N_WINDOW
    assert row["bytes_read_sky"] > 0 and row["bytes_read_primary_beam"] > 0
    assert row["n_chunks_skipped"] > 0
    assert row["planes_per_batch"] == 4
    assert row["task_memory_plan_bytes"] == find_continuum_ultra_task_bytes(
        "first_round", N_PIXELS, N_PIXELS, 4, 4, 4, True, 0, True, 0.0, 2048
    )
    assert np.isnan(row["task_rss_peak_bytes"])
    assert row["T_first_round_rows"] > 0 and row["T_write_regions"] == 0
    assert row["driver_bytes_written"] == 0


def test_first_round_pass_with_the_store(cube, layout, first_round, tmp_path):
    """Moments only: the limit masks are read from the store, the rows too;
    the maps equal those of the pass that made the store."""
    with dask.config.set(scheduler="synchronous"):
        reductions, (record,) = run_first_round_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            first_round["store"],
            False,
            True,
            str(tmp_path / "pass_3"),
            0,
            None,
            3,
        )
    expected = first_round["reductions"]
    for field in FirstRoundRows._fields:
        np.testing.assert_array_equal(
            getattr(reductions.rows, field), getattr(expected.rows, field)
        )
    np.testing.assert_array_equal(reductions.integral, expected.integral)
    np.testing.assert_array_equal(reductions.maximum, expected.maximum)
    _check_record(record, layout, 3, "first_round", None, len(layout.tile_mapping))
    row = record.row
    assert row["round_index"] == 0
    assert row["purpose"] == "first round: moments"
    assert row["data_variables_read"] == "SKY_RESIDUAL"
    assert row["computes_first_round_masks"] is False
    assert row["n_planes_read"] == N_WINDOW and row["bytes_read_primary_beam"] == 0
    timing = record.timing_node_tasks
    assert (timing.loc[timing["family"] == "frequency", "bytes_read_masks"] > 0).all()
    # Maps and rows read by the driver.
    assert row["driver_bytes_read"] > first_round["records"][0].row["driver_bytes_read"]
    assert not os.path.exists(tmp_path / "pass_3")


def test_first_round_pass_of_masks_only(cube, layout, tmp_path):
    store = _new_store(cube, layout, str(tmp_path / "first_round.zarr"))
    with dask.config.set(scheduler="synchronous"):
        reductions, (record,) = run_first_round_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            store,
            True,
            False,
            str(tmp_path / "pass_0"),
            None,
            None,
            0,
        )
    assert reductions.integral is None and reductions.maximum is None
    assert zarr.open_group(store, mode="r").attrs["find_continuum_first_round"][
        "complete"
    ]
    _check_record(record, layout, 0, "first_round", None, 0)
    assert record.row["n_moment_sets"] == 0
    assert record.row["purpose"] == "first round: primary-beam masks, per-channel rows"
    with pytest.raises(ValueError, match="masks, the moments or both"):
        run_first_round_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            store,
            False,
            False,
            str(tmp_path / "pass_1"),
            None,
            None,
            1,
        )


@pytest.mark.parametrize(
    "form", ["normalization", "mean", "only_extra_mask_normalization"]
)
def test_spectrum_pass(cube, layout, first_round, tmp_path, form):
    regions = _regions()
    spectrum_regions = {
        "normalization": SpectrumRegions(
            regions["MEAN_REGION"],
            True,
            regions["SCALED_MAD_REGION"],
            AnnulusLevels(0.25, 0.6),
        ),
        "mean": SpectrumRegions(regions["MEAN_REGION"], True, None, None),
        "only_extra_mask_normalization": SpectrumRegions(
            regions["MEAN_REGION"], False, regions["SCALED_MAD_REGION"], None
        ),
    }[form]
    with dask.config.set(scheduler="synchronous"):
        reductions, (record,) = run_spectrum_pass(
            cube["window_xds"],
            layout,
            first_round["store"],
            spectrum_regions,
            str(tmp_path / "pass_1"),
            0,
            "original",
            None,
            1,
        )
    assert isinstance(reductions, SpectrumReductions)
    window = slice(*WINDOW)
    sky, valid = cube["sky"][:, window], cube["valid"][:, window]
    mean_region = regions["MEAN_REGION"].values[0, 0, 0]
    expected = masked_channel_means(
        sky,
        valid,
        mean_region,
        cube["spectrum_mask"][:, window]
        if spectrum_regions.use_spectrum_mask
        else None,
    )[0, :, 0]
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(reductions.mean_spectrum, expected)
    levels = spectrum_regions.scaled_mad_primary_beam_levels
    if spectrum_regions.scaled_mad_region is None:
        assert reductions.scaled_mad_spectrum is None
    else:
        expected = masked_channel_scaled_mads(
            sky,
            valid,
            regions["SCALED_MAD_REGION"].values[0, 0, 0],
            None if levels is None else cube["primary_beam"][:, window],
            levels,
        )[0, :, 0]
        assert np.isfinite(expected).all()
        np.testing.assert_array_equal(reductions.scaled_mad_spectrum, expected)
    assert not os.path.exists(tmp_path / "pass_1")

    _check_record(record, layout, 1, "spectrum", "original", 0)
    row = record.row
    read_primary_beam = levels is not None
    assert row["data_variables_read"] == (
        "PRIMARY_BEAM, SKY_RESIDUAL" if read_primary_beam else "SKY_RESIDUAL"
    )
    assert (row["bytes_read_primary_beam"] > 0) == read_primary_beam
    assert row["purpose"] == (
        "mean spectrum" if form == "mean" else "mean and scaled MAD spectra"
    )
    assert row["n_moment_sets"] == 0 and row["computes_first_round_masks"] is False
    assert row["n_tasks"] == 3 and row["n_channels_read"] == N_WINDOW
    n_regions = 1 if form == "mean" else 2
    fraction = max(
        np.count_nonzero(regions[name]) / N_PIXELS**2
        for name in ("MEAN_REGION", "SCALED_MAD_REGION")[:n_regions]
    )
    assert row["task_memory_plan_bytes"] == find_continuum_ultra_task_bytes(
        "spectrum",
        N_PIXELS,
        N_PIXELS,
        4,
        4,
        4,
        read_primary_beam,
        n_regions,
        False,
        fraction,
        2048,
    )
    assert row["T_write_regions"] > 0 and row["driver_bytes_written"] > 0
    assert row["T_read_maps"] == 0 and row["driver_bytes_read"] == 0


@pytest.fixture(scope="module")
def selection(cube, layout, first_round):
    """A selection pass with noise over unsorted, overlapping ranges that
    leave the middle chunk [17, 31) without a channel."""
    ranges = [[30, 42], [2, 9], [5, 12], [41, 40], [37, 39]]
    pass_directory = str(cube["directory"] / "pass_2")
    with dask.config.set(scheduler="synchronous"):
        reductions, records = run_selection_moments_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            first_round["store"],
            ranges,
            _regions()["NOISE_REGION"],
            pass_directory,
            0,
            "original",
            None,
            2,
        )
    return {
        "ranges": ranges,
        "channels": np.r_[2:12, 30:42],
        "pass_directory": pass_directory,
        "reductions": reductions,
        "records": records,
    }


def test_selection_moments_pass(cube, layout, selection):
    reductions = selection["reductions"]
    assert isinstance(reductions, SelectionReductions)
    selected = selection["channels"]
    np.testing.assert_array_equal(reductions.selected_channels, selected)
    channels = WINDOW[0] + selected
    integral, maximum = _expected_moments(
        cube, layout, cube["valid"], set(channels.tolist())
    )
    np.testing.assert_array_equal(reductions.integral, integral)
    np.testing.assert_array_equal(reductions.maximum, maximum)
    medians, mads = masked_channel_medians_and_mads(
        cube["sky"][:, channels],
        cube["valid"][:, channels],
        _regions()["NOISE_REGION"].values[0, 0, 0],
    )
    np.testing.assert_array_equal(reductions.noise_medians, medians[0, :, 0])
    np.testing.assert_array_equal(reductions.noise_mads, mads[0, :, 0])
    assert not os.path.exists(selection["pass_directory"])

    (record,) = selection["records"]
    _check_record(
        record, layout, 2, "selection_moments", "original", len(layout.tile_mapping)
    )
    row = record.row
    # The middle chunk holds no selected channel: no task for it.
    assert row["n_tasks"] == 2 and row["n_channels_read"] == selected.size
    timing = record.timing_node_tasks
    assert list(timing.loc[timing["family"] == "frequency", "slot"]) == [0, 2]
    assert row["purpose"] == "moments of the selection, noise of its channels"
    assert row["data_variables_read"] == "SKY_RESIDUAL"
    assert row["n_moment_sets"] == 1
    fraction = np.count_nonzero(_regions()["NOISE_REGION"]) / N_PIXELS**2
    assert row["task_memory_plan_bytes"] == find_continuum_ultra_task_bytes(
        "selection_moments", N_PIXELS, N_PIXELS, 4, 4, 4, False, 1, True, fraction, 2048
    )
    assert row["T_write_regions"] > 0 and row["driver_bytes_written"] > 0


def test_selection_moments_pass_without_noise_on_the_threaded_scheduler(
    cube, layout, first_round, selection, tmp_path
):
    """Without a noise region and on Dask's threaded scheduler, the maps
    equal those of the synchronous pass with noise."""
    reductions, (record,) = run_selection_moments_pass(
        cube["window_xds"],
        layout,
        cube["settings"],
        first_round["store"],
        selection["ranges"],
        None,
        str(tmp_path / "pass_4"),
        1,
        "amended_mask",
        None,
        4,
    )
    expected = selection["reductions"]
    np.testing.assert_array_equal(reductions.integral, expected.integral)
    np.testing.assert_array_equal(reductions.maximum, expected.maximum)
    np.testing.assert_array_equal(
        reductions.selected_channels, expected.selected_channels
    )
    assert reductions.noise_medians is None and reductions.noise_mads is None
    _check_record(
        record, layout, 4, "selection_moments", "amended_mask", len(layout.tile_mapping)
    )
    assert record.row["purpose"] == "moments of the selection"
    assert (
        record.row["T_write_regions"] == 0 and record.row["driver_bytes_written"] == 0
    )
    assert not os.listdir(tmp_path)


def test_selection_moments_pass_refusals(cube, layout, first_round, tmp_path):
    for ranges in ([], [[3, 3]], [[40, N_WINDOW + 1]], [[-1, 2]]):
        with pytest.raises(ValueError, match="select no channel"):
            run_selection_moments_pass(
                cube["window_xds"],
                layout,
                cube["settings"],
                first_round["store"],
                ranges,
                None,
                str(tmp_path / "pass_5"),
                1,
                "extra_mask",
                None,
                5,
            )
    assert not os.path.exists(tmp_path / "pass_5")


def test_an_incomplete_first_round_is_refused(cube, layout, tmp_path):
    """A store whose window holds channels the pass does not make."""
    store = _new_store(
        cube, layout, str(tmp_path / "first_round.zarr"), (0, N_CHANNELS)
    )
    with (
        pytest.raises(RuntimeError, match="incomplete"),
        dask.config.set(scheduler="synchronous"),
    ):
        run_first_round_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            store,
            True,
            False,
            str(tmp_path / "pass_0"),
            None,
            None,
            0,
        )
    assert not os.path.exists(tmp_path / "pass_0")


def test_tasks_over_the_budget_run_at_one_plane_with_warnings(
    cube, layout, first_round, selection, tmp_path, monkeypatch
):
    warnings = []
    monkeypatch.setattr(find_continuum_ultra_passes.logger, "warning", warnings.append)
    with dask.config.set(scheduler="synchronous"):
        reductions, (record,) = run_selection_moments_pass(
            cube["window_xds"],
            layout._replace(budget_bytes=GiB),
            cube["settings"],
            first_round["store"],
            selection["ranges"],
            None,
            str(tmp_path / "pass_8"),
            0,
            "original",
            None,
            8,
        )
    np.testing.assert_array_equal(reductions.integral, selection["reductions"].integral)
    assert record.row["planes_per_batch"] == 1
    assert [message.split(" needs")[0] for message in warnings] == [
        "A find_continuum_ultra task",
        "A find_continuum_ultra merge task",
    ]


def test_the_pass_directory_is_removed_when_a_pass_fails(
    cube, layout, first_round, tmp_path, monkeypatch
):
    regions = SpectrumRegions(_regions()["MEAN_REGION"], True, None, None)
    with (
        pytest.raises(FileNotFoundError),
        dask.config.set(scheduler="synchronous"),
    ):
        run_spectrum_pass(
            cube["window_xds"],
            layout,
            str(tmp_path / "missing_first_round.zarr"),
            regions,
            str(tmp_path / "pass_6"),
            0,
            "original",
            None,
            6,
        )
    assert not os.path.exists(tmp_path / "pass_6")

    # A merge that fails after the frequency tasks wrote their slots.
    def failing_merge(input_params):
        assert os.listdir(os.path.dirname(input_params["partials_store"]))
        raise OSError("merge failed")

    monkeypatch.setattr(
        find_continuum_ultra_passes,
        "merge_find_continuum_moment_partials",
        failing_merge,
    )
    with (
        pytest.raises(OSError, match="merge failed"),
        dask.config.set(scheduler="synchronous"),
    ):
        run_selection_moments_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            first_round["store"],
            [[0, 5]],
            None,
            str(tmp_path / "pass_7"),
            0,
            "original",
            None,
            7,
        )
    assert not os.path.exists(tmp_path / "pass_7")


def test_transient_errors_of_the_drivers_stores_are_retried(
    cube, layout, first_round, tmp_path, monkeypatch
):
    """A Lustre eviction fails with EIO the creation of the pass's partials
    store and the write of the first-round rows, once each: each is
    repeated, and the pass gives the results of one without them."""
    from astroviper.utils.chunk_io import zarr_chunks

    store = _new_store(cube, layout, str(tmp_path / "first_round.zarr"))
    monkeypatch.setattr(zarr_chunks, "_EIO_RETRY_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    targets = {
        "partials store": lambda path, mode: (
            path.endswith("partials.zarr") and mode == "w"
        ),
        "rows": lambda path, mode: path == store and mode == "r+",
    }
    failed = set()
    open_group = zarr.open_group

    def evicted_once(path, *args, **kwargs):
        for name, target in targets.items():
            if name not in failed and target(os.fspath(path), kwargs.get("mode")):
                failed.add(name)
                raise OSError(errno.EIO, "Input/output error")
        return open_group(path, *args, **kwargs)

    monkeypatch.setattr(zarr, "open_group", evicted_once)
    with dask.config.set(scheduler="synchronous"):
        reductions, _ = run_first_round_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            store,
            True,
            True,
            str(tmp_path / "pass_0"),
            None,
            None,
            0,
        )
    assert failed == set(targets)
    expected = first_round["reductions"]
    for field in FirstRoundRows._fields:
        np.testing.assert_array_equal(
            getattr(reductions.rows, field), getattr(expected.rows, field)
        )
    np.testing.assert_array_equal(reductions.integral, expected.integral)
    np.testing.assert_array_equal(reductions.maximum, expected.maximum)


def test_stored_bytes_retries_a_transient_error(tmp_path, monkeypatch):
    """The sizes of a store's files are counted again after a transient EIO."""
    from astroviper.utils.chunk_io import zarr_chunks

    store = tmp_path / "store.zarr"
    (store / "c").mkdir(parents=True)
    (store / "zarr.json").write_bytes(b"{}")
    (store / "c" / "0").write_bytes(b"0" * 10)
    (store / "c" / "1").write_bytes(b"0" * 5)
    monkeypatch.setattr(zarr_chunks, "_EIO_RETRY_BACKOFF_SECONDS", (0.0, 0.0, 0.0))
    getsize, failures = os.path.getsize, []

    def evicted_once(path):
        if not failures:
            failures.append(path)
            raise OSError(errno.EIO, "Input/output error")
        return getsize(path)

    monkeypatch.setattr(os.path, "getsize", evicted_once)
    assert find_continuum_ultra_passes._stored_bytes(str(store)) == 15
    assert len(failures) == 1


def test_a_pass_of_one_task_with_the_resource_monitor(cube, first_round, tmp_path):
    """One frequency task has no reduce node: its resource series are folded
    into its row, as those of the merge tasks are."""
    layout = plan_find_continuum_ultra_layout(
        cube["window_xds"], VARIABLES, list(WINDOW), {"frequency": 1}, THREADS, None
    )
    with dask.config.set(scheduler="synchronous"):
        reductions, (record,) = run_selection_moments_pass(
            cube["window_xds"],
            layout,
            cube["settings"],
            first_round["store"],
            [[0, N_WINDOW]],
            None,
            str(tmp_path / "pass_0"),
            0,
            "single",
            0.01,
            0,
        )
    integral, maximum = _expected_moments(cube, layout, cube["valid"])
    np.testing.assert_array_equal(reductions.integral, integral)
    np.testing.assert_array_equal(reductions.maximum, maximum)
    _check_record(
        record, layout, 0, "selection_moments", "single", len(layout.tile_mapping)
    )
    assert record.timing_reduce_nodes == []
    timing = record.timing_node_tasks
    assert all(isinstance(series, list) for series in timing["memory_rss_bytes"])
    assert record.row["task_rss_peak_bytes"] == max(
        max(series)
        for series in timing.loc[timing["family"] == "frequency", "memory_rss_bytes"]
    )


def test_driver_memory_estimate():
    """About 8 GB at 11250^2 (design section 9.4), in proportion to the
    pixels."""
    assert round(find_continuum_ultra_driver_bytes(11250, 11250) / 1e9, 1) == 8.2
    assert find_continuum_ultra_driver_bytes(100, 200) == 65 * 100 * 200


def test_records_join_into_the_frames_of_a_run(first_round, selection):
    """The records of several passes concatenate into the frames the
    application returns, joined by pass_index."""
    records = first_round["records"] + selection["records"]
    passes = pd.DataFrame([record.row for record in records])
    timing = pd.concat(
        [record.timing_node_tasks for record in records], ignore_index=True
    )
    reduces = pd.DataFrame(
        [r for record in records for r in record.timing_reduce_nodes]
    )
    assert list(passes["pass_index"]) == [0, 2]
    assert set(timing["pass_index"]) == set(reduces["pass_index"]) == {0, 2}
    for _, row in passes.iterrows():
        in_pass = timing[timing["pass_index"] == row["pass_index"]]
        assert (in_pass["family"] == "frequency").sum() == row["n_tasks"]
