"""Tests of find_continuum, the pipeline's findContinuum on AstroVIPER images."""

import os
import sys
import uuid

import numpy as np
import pytest
import xarray as xr
import zarr
from xradio.image import write_image

from astroviper.distributed_applications.imaging import (
    find_continuum,
    find_continuum_fused,
    find_continuum_ultra,
)
from astroviper.distributed_applications.imaging.find_continuum import (
    fused_reductions,
)
from astroviper.distributed_applications.imaging.find_continuum.first_round import (
    bind_find_continuum_first_round_store,
    create_find_continuum_first_round_store,
    drain_first_round_moments,
    find_continuum_first_round_partials_path,
    find_continuum_first_round_store_path,
)
from astroviper.distributed_applications.imaging.find_continuum.ultra_reductions import (
    StoredMap,
    ultra_frequency_tasks,
)
from astroviper.node_tasks.imaging.find_continuum import (
    find_continuum_first_round_chunk,
    read_small_image,
    write_find_continuum_first_round_rows,
)
from astroviper.processing_functions.imaging.find_continuum import (
    CHANNEL_STATISTICS,
    linear_channel_ranges,
    linear_frequency_axis,
    moment_velocity_widths,
)
from astroviper.utils.data_group_tools import modify_data_groups_xds
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    LINE_SET,
    make_find_continuum_image_xds,
)

APPLICATIONS = {
    "separate": find_continuum,
    "fused": find_continuum_fused,
    "ultra": find_continuum_ultra,
}
# Threads with room for find_continuum_ultra's task baseline of 1.5 GiB on
# any machine that runs the tests.
THREAD_INFO = {"n_threads": 4, "memory_per_thread": 4.0}
VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
WHOLE_CUBE_SUMS = ("sum", "sumsq", "mean", "sigma", "rms")
COMPACT_LINE = dict(
    channel=120,
    fwhm_channels=14,
    peak_jy=0.04,
    offset_pixels=(0, 0),
    size_fwhm_pixels=6,
)
# Channel selection, all-continuum flag and moment difference SNR of CASA
# 6.7.6's findContinuum (pipeline 2026.3.0.7) on the same cubes, with CASA's
# atmosphere model. The windows have no atmospheric line, and a flat atmosphere
# selects the same channels. Last, the passes that read the dirty cube in
# find_continuum, find_continuum_fused and find_continuum_ultra.
CASA_RESULTS = {
    "no_lines": (dict(lines=()), "9~464", True, 4.4041, (7, 3, 3)),
    "line_set": (
        dict(lines=LINE_SET),
        "6~85;154~367;433~472",
        False,
        6.2507,
        (7, 3, 3),
    ),
    # The primary beam falls below 0.2 in the corners: masked there, as by
    # tclean's pblimit in the pipeline (and in the CASA reference).
    "small_beam": (
        dict(lines=LINE_SET, primary_beam_fwhm_arcsec=8.0),
        "6~85;154~367;433~467",
        False,
        7.3877,
        (7, 3, 3),
    ),
    # The ripple of the bright line's channels amends the joint mask.
    "sidelobes": (
        dict(
            n_pixels=96,
            lines=(
                dict(COMPACT_LINE, sidelobe_fraction=0.2),
                dict(
                    channel=300,
                    fwhm_channels=20,
                    peak_jy=6e-3,
                    offset_pixels=(22, -18),
                    size_fwhm_pixels=24,
                ),
            ),
        ),
        "3~86;154~255;339~472",
        False,
        5.0069,
        (8, 5, 5),
    ),
}
# The keys and columns of section 7 of the design.
DRIVER_KEYS = {
    "T_total",
    "application_start_unixtime",
    "application_end_unixtime",
    "compute_start_unixtime",
    "compute_end_unixtime",
    "driver_rss_peak_bytes",
    "driver_bytes_read",
}
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
TASK_COLUMNS = {
    "reduction",
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
    "T_read_partials",
    "T_merge",
    "T_write_results",
    "T_cpu",
    "T_find_continuum_task",
}
# The I/O counters of the monitor exist on Linux only.
MONITOR_COLUMNS = {
    "time_seconds",
    "cpu_percent",
    "memory_rss_bytes",
    "sample_interval_seconds",
    "minor_page_faults",
    "major_page_faults",
} | (
    {"read_bytes", "write_bytes", "read_chars", "write_chars"}
    if sys.platform == "linux"
    else set()
)
MEMORY_PLAN_KEYS = {
    "n_threads",
    "memory_per_thread_gib",
    "budget_bytes",
    "n_frequency_tasks",
    "channels_per_task",
    "merge_tile_shape",
    "n_merge_tiles",
    "plane_bytes",
}


def _write_cube(path, img_xds):
    write_image(img_xds, imagename=str(path), out_format="zarr", overwrite=True)
    return str(path)


def _flat_atmosphere(n_channels):
    return {
        "atmospheric_transmission": np.full(n_channels, 0.92),
        "atmospheric_sky_temperature": np.full(n_channels, 27.0),
    }


def _run(backend, store, work_directory, n_channels=480, **kwargs):
    if backend == "ultra":
        kwargs.setdefault("thread_info", THREAD_INFO)
    return APPLICATIONS[backend](
        input_image_store=store,
        work_directory=str(work_directory),
        **_flat_atmosphere(n_channels),
        **kwargs,
    )


def _first_round_store(store, chunk_channels=160):
    """The imager's first round of every spectral window of a cube, made in
    this process with its node tasks (the moments in this process's
    registry), drained and bound."""
    xds = xr.open_zarr(store, chunks=None)
    frequency = xds.frequency.values
    rest_frequency = float(xds.frequency.attrs["rest_frequency"]["data"])
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
    sky = zarr.open_array(os.path.join(store, "SKY_RESIDUAL"), mode="r")
    path = find_continuum_first_round_store_path(store)
    common = dict(
        first_round_store=path,
        variables=VARIABLES,
        polarization_index=0,
        primary_beam_limit=0.2,
        tile_shape=sky.chunks[-2:],
        value_dtypes={"sky": "float32", "primary_beam": "float32"},
        run_id=uuid.uuid4().hex,
        windows=windows,
        rest_frequency=rest_frequency,
    )
    create_find_continuum_first_round_store(
        n_channels=frequency.size,
        image_shape=sky.shape,
        merge_tile_shape=sky.chunks[-2:],
        written_by="image_cube_single_field",
        **common,
    )
    partials = find_continuum_first_round_partials_path(path)
    os.makedirs(partials)
    chunks = [
        find_continuum_first_round_chunk(
            xds.isel(frequency=channels),
            channels,
            partials_directory=partials,
            **common,
        )
        for channels in (
            slice(start, min(start + chunk_channels, frequency.size))
            for start in range(0, frequency.size, chunk_channels)
        )
    ]
    rows = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    assert write_find_continuum_first_round_rows(
        path, dict(rows, write_failed=False), frequency.size
    )["complete"]
    result, _ = drain_first_round_moments(common["run_id"], path, windows)
    assert result["moments_complete"] == [True] * len(windows)
    bind_find_continuum_first_round_store(path, store)
    return {"store": path, "primary_beam_limit": 0.2}


def _small_images(work_directory):
    """Values and masks of every small image findContinuum wrote."""
    return {
        name: read_small_image(os.path.join(work_directory, name))[:2]
        for name in sorted(os.listdir(work_directory))
        if os.path.isfile(os.path.join(work_directory, name, "zarr.json"))
    }


def _assert_same_small_images(work_directory, other):
    images, others = _small_images(work_directory), _small_images(other)
    assert images and sorted(images) == sorted(others)
    for name, (values, mask) in images.items():
        np.testing.assert_array_equal(values, others[name][0], err_msg=name)
        np.testing.assert_array_equal(mask, others[name][1], err_msg=name)


def _assert_same_results(result, other):
    for key in (
        "channel_selection",
        "aggregate_bandwidth",
        "all_continuum",
        "warnings",
        "moment_difference_snr",
    ):
        assert result[key] == other[key], key


def _normalized(value, directory):
    """A cache key with the work directory left out of its paths."""
    if isinstance(value, str):
        return value.replace(directory, "<work>")
    if isinstance(value, tuple):
        return tuple(_normalized(item, directory) for item in value)
    return value


@pytest.fixture(scope="module")
def line_set_store(tmp_path_factory):
    return _write_cube(
        tmp_path_factory.mktemp("line_set") / "line_set.img.zarr",
        make_find_continuum_image_xds(lines=LINE_SET),
    )


@pytest.mark.parametrize("backend", APPLICATIONS)
@pytest.mark.parametrize("case", CASA_RESULTS)
def test_selection_matches_casa(case, backend, tmp_path, monkeypatch):
    kwargs, selection, all_continuum, snr, dirty_cube_reads = CASA_RESULTS[case]
    store = _write_cube(
        tmp_path / f"{case}.img.zarr", make_find_continuum_image_xds(**kwargs)
    )
    work_directory = str(tmp_path / "work")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    monkeypatch.chdir(run_directory)

    result = _run(
        backend, store, work_directory, n_mapping_parallelism={"frequency": 3}
    )

    assert result["channel_selection"] == selection
    assert result["all_continuum"] is all_continuum
    assert result["moment_difference_snr"] == pytest.approx(snr, abs=1e-4)
    assert result["warnings"] == ["", ""]
    assert result["channel_range"] == [0, 480]
    assert os.path.isfile(result["plot"]) and os.path.isdir(result["joint_mask"])
    assert list(run_directory.iterdir()) == []
    assert set(result["cube_reads"]) == {"SKY_RESIDUAL", "PRIMARY_BEAM"}
    reads = dict(zip(APPLICATIONS, dirty_cube_reads, strict=True))
    assert result["cube_reads"]["SKY_RESIDUAL"] == reads[backend]
    timing = result["timing_node_tasks"]
    assert sorted(set(timing["reduction"])) == list(range(result["n_cube_reductions"]))
    if backend == "ultra":
        assert result["cube_reads"]["PRIMARY_BEAM"] == 1
        frequency = timing[timing["family"] == "frequency"]
        assert len(frequency) == result["passes"]["n_tasks"].sum()
    else:
        assert len(timing) == 3 * result["n_cube_reductions"]

    with pytest.raises(ValueError, match="earlier run"):
        find_continuum(
            input_image_store=store,
            work_directory=work_directory,
            **_flat_atmosphere(480),
        )


@pytest.mark.parametrize(
    "ultra_chunks, fused_chunks",
    [({"frequency": 3}, {"frequency": 3}), ({"frequency": 1}, None)],
)
def test_ultra_equals_fused(
    line_set_store, tmp_path, monkeypatch, ultra_chunks, fused_chunks
):
    """The results, reductions and small images of find_continuum_fused with
    the same frequency chunks, but for the last bits of whole-cube sums and
    of the per-channel sumsq and rms."""
    backends = []

    class Recorded(fused_reductions.FusedReductions):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            backends.append(self)

    monkeypatch.setattr(fused_reductions, "FusedReductions", Recorded)
    directories = [str(tmp_path / name) for name in ("ultra", "fused")]
    ultra = _run(
        "ultra", line_set_store, directories[0], n_mapping_parallelism=ultra_chunks
    )
    fused = _run(
        "fused", line_set_store, directories[1], n_mapping_parallelism=fused_chunks
    )

    _assert_same_results(ultra, fused)
    _assert_same_small_images(*directories)
    caches = [
        {_normalized(key, directory): result for key, result in backend.cache.items()}
        for backend, directory in zip(backends, directories, strict=True)
    ]
    assert caches[0].keys() == caches[1].keys()
    for key, expected in caches[1].items():
        request, result = dict(key[0]), caches[0][key]
        if request["kind"] == "moments":
            assert sorted(result) == sorted(expected)
            for moment, values in result.items():
                if isinstance(values, StoredMap):
                    values = values.load()
                np.testing.assert_array_equal(values, expected[moment])
        elif request["kind"] == "profile":
            np.testing.assert_array_equal(result, expected)
        else:
            for name in CHANNEL_STATISTICS:
                if name in ("sumsq", "rms") and request["per_channel"]:
                    np.testing.assert_allclose(result[name], expected[name], rtol=1e-14)
                elif name == "sigma" and not request["per_channel"]:
                    # sumsq - npts mean**2 cancels: the variance is as exact
                    # as sumsq, not sigma relative to itself.
                    np.testing.assert_allclose(
                        (expected["npts"] - 1) * result[name] ** 2,
                        (expected["npts"] - 1) * expected[name] ** 2,
                        rtol=0,
                        atol=1e-12 * expected["sumsq"],
                    )
                elif name in WHOLE_CUBE_SUMS and not request["per_channel"]:
                    np.testing.assert_allclose(result[name], expected[name], rtol=1e-12)
                else:
                    np.testing.assert_array_equal(result[name], expected[name])


def test_ultra_with_a_first_round_store(line_set_store, tmp_path):
    """The imager's first round saves the first pass, its read of the cube and
    its read of the primary beam; without the imager's moments, the first pass
    computes them as without a store."""
    computed = _run(
        "ultra", line_set_store, tmp_path / "computed", memory_budget_gb=2.0
    )
    assert computed["memory_plan"]["budget_bytes"] == 2 * 2**30
    first_round = _first_round_store(line_set_store)
    imager = _run(
        "ultra", line_set_store, tmp_path / "imager", first_round_reductions=first_round
    )

    assert imager["channel_selection"] == computed["channel_selection"]
    assert imager["all_continuum"] is computed["all_continuum"]
    assert imager["warnings"] == computed["warnings"]
    assert imager["moment_difference_snr"] == pytest.approx(
        computed["moment_difference_snr"], rel=1e-12
    )
    assert computed["cube_reads"]["PRIMARY_BEAM"] == 1
    assert imager["cube_reads"] == {
        "SKY_RESIDUAL": computed["cube_reads"]["SKY_RESIDUAL"] - 1
    }
    assert not imager["passes"]["compute_first_round"].any()
    assert imager["first_round"] == {
        "source": "imager",
        "store": first_round["store"],
        "moments": "imager",
        "moments_complete": True,
        "imager": None,
    }
    assert computed["first_round"]["source"] == "find_continuum_ultra"
    assert computed["first_round"]["moments"] == "computed"
    assert os.path.isdir(computed["first_round"]["store"])

    group = zarr.open_group(first_round["store"], mode="r+")
    group.attrs["find_continuum_first_round"] = dict(
        group.attrs["find_continuum_first_round"], moments_complete=[False]
    )
    recomputed = _run(
        "ultra",
        line_set_store,
        tmp_path / "recomputed",
        first_round_reductions=first_round,
    )
    _assert_same_results(recomputed, computed)
    assert recomputed["cube_reads"] == {
        "SKY_RESIDUAL": computed["cube_reads"]["SKY_RESIDUAL"]
    }
    assert recomputed["first_round"]["moments"] == "computed"
    assert not recomputed["first_round"]["moments_complete"]


def test_fused_with_the_first_round_of_the_imager(tmp_path):
    """Given the imager's first-round store, the fused version reads the cube
    once less and finds the same channels."""
    kwargs, selection, _, snr, (_, dirty_cube_reads, _) = CASA_RESULTS["line_set"]
    store = _write_cube(
        tmp_path / "line_set.img.zarr", make_find_continuum_image_xds(**kwargs)
    )
    result = _run(
        "fused",
        store,
        tmp_path / "work",
        first_round_reductions=_first_round_store(store),
    )
    assert result["channel_selection"] == selection
    assert result["moment_difference_snr"] == pytest.approx(snr, abs=1e-4)
    assert result["cube_reads"]["SKY_RESIDUAL"] == dirty_cube_reads - 1
    assert result["first_round"]["source"] == "imager"


def test_first_round_store_of_another_cube_or_data_group_is_refused(tmp_path):
    img_xds = make_find_continuum_image_xds(n_pixels=32, lines=LINE_SET)
    modify_data_groups_xds(
        img_xds,
        "psf",
        {
            "sky": "POINT_SPREAD_FUNCTION",
            "primary_beam": "PRIMARY_BEAM",
            "beam_fit_params_point_spread_function": "BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION",
        },
        "The point spread function as the sky",
    )
    store = _write_cube(tmp_path / "cube.img.zarr", img_xds)
    first_round = _first_round_store(store)
    for backend in ("fused", "ultra"):
        with pytest.raises(ValueError, match="was made from"):
            _run(
                backend,
                store,
                tmp_path / f"psf_{backend}",
                image_data_group_in_name="psf",
                first_round_reductions=first_round,
            )
    zarr_json = os.path.join(store, "zarr.json")
    stat = os.stat(zarr_json)
    os.utime(zarr_json, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    for backend in ("fused", "ultra"):
        with pytest.raises(ValueError, match="as it is now"):
            _run(
                backend,
                store,
                tmp_path / f"stale_{backend}",
                first_round_reductions=first_round,
            )
    with pytest.raises(ValueError, match="'store'"):
        _run(
            "fused",
            store,
            tmp_path / "in_memory",
            first_round_reductions={"primary_beam_limit": 0.2, "reductions": []},
        )


@pytest.fixture(scope="module")
def two_windows(tmp_path_factory):
    """A cube of two spectral windows of 12 channels with a gap, and a cube of
    each window alone (findContinuum needs two edge channels, which 6 channels
    do not give)."""
    directory = tmp_path_factory.mktemp("two_windows")
    line = dict(
        channel=2,
        fwhm_channels=2,
        peak_jy=0.04,
        offset_pixels=(0, 0),
        size_fwhm_pixels=6,
    )
    img_xds = make_find_continuum_image_xds(n_pixels=32, n_channels=24, lines=(line,))
    frequency = img_xds.frequency.values.copy()
    frequency[12:] += 40 * (frequency[1] - frequency[0])
    img_xds = img_xds.assign_coords(frequency=img_xds.frequency.copy(data=frequency))
    windows = [[0, 12], [12, 24]]
    return {
        "store": _write_cube(directory / "both" / "cube.img.zarr", img_xds),
        "windows": windows,
        "single": [
            _write_cube(
                directory / f"window_{w}" / "cube.img.zarr",
                img_xds.isel(frequency=slice(*window)),
            )
            for w, window in enumerate(windows)
        ],
    }


@pytest.mark.parametrize("backend", APPLICATIONS)
def test_each_window_equals_a_cube_of_that_window(two_windows, tmp_path, backend):
    for window, single in zip(
        two_windows["windows"], two_windows["single"], strict=True
    ):
        directories = [
            str(tmp_path / f"{name}_{window[0]}") for name in ("range", "single")
        ]
        result = _run(
            backend,
            two_windows["store"],
            directories[0],
            n_channels=12,
            channel_range=window,
            n_mapping_parallelism={"frequency": 2},
        )
        expected = _run(
            backend,
            single,
            directories[1],
            n_channels=12,
            n_mapping_parallelism={"frequency": 2},
        )
        _assert_same_results(result, expected)
        assert result["channel_range"] == window
        assert result["cube_reads"] == expected["cube_reads"]
        _assert_same_small_images(*directories)


def test_ultra_with_a_first_round_store_on_a_window(two_windows, tmp_path):
    """The imager's first round of the second window saves a read of the cube."""
    window = two_windows["windows"][1]
    first_round = _first_round_store(two_windows["store"], chunk_channels=5)
    results = [
        _run(
            "ultra",
            two_windows["store"],
            tmp_path / name,
            n_channels=12,
            channel_range=window,
            first_round_reductions=reductions,
        )
        for name, reductions in (("computed", None), ("imager", first_round))
    ]
    assert results[1]["channel_selection"] == results[0]["channel_selection"]
    assert results[1]["moment_difference_snr"] == pytest.approx(
        results[0]["moment_difference_snr"], rel=1e-12
    )
    assert results[1]["cube_reads"] == {
        "SKY_RESIDUAL": results[0]["cube_reads"]["SKY_RESIDUAL"] - 1
    }
    assert results[1]["first_round"]["moments"] == "imager"


@pytest.mark.parametrize("backend", APPLICATIONS)
def test_a_window_without_two_edge_channels_is_refused(two_windows, tmp_path, backend):
    """findContinuum finds no result on 7 channels: the reason it gives."""
    with pytest.raises(
        ValueError,
        match=r"no result for channels \[0, 7\) .* \(7 channels\): You must have at "
        r"least 2 edge channels",
    ):
        _run(
            backend, two_windows["store"], tmp_path, n_channels=7, channel_range=[0, 7]
        )


def test_a_cube_of_several_windows_needs_a_channel_range(two_windows, tmp_path):
    with pytest.raises(ValueError, match=r"\[\[0, 12\], \[12, 24\]\]"):
        _run("fused", two_windows["store"], tmp_path / "work", n_channels=24)
    with pytest.raises(ValueError, match="one value per channel"):
        _run(
            "fused",
            two_windows["store"],
            tmp_path / "work",
            n_channels=24,
            channel_range=[0, 12],
        )


@pytest.fixture
def small_cube(tmp_path):
    return make_find_continuum_image_xds(n_pixels=16, n_channels=20)


@pytest.mark.parametrize(
    "arguments, match",
    [
        (dict(atmospheric_transmission=np.ones(3)), "one value per channel"),
        (dict(n_mapping_parallelism={"time": 2}), "frequency"),
        (dict(image_data_group_in_name="sky_only"), "lacks the roles"),
    ],
)
def test_input_checks(small_cube, tmp_path, arguments, match):
    modify_data_groups_xds(small_cube, "sky_only", {"sky": "SKY_RESIDUAL"}, "Sky only")
    store = _write_cube(tmp_path / "cube.img.zarr", small_cube)
    with pytest.raises(ValueError, match=match):
        find_continuum(
            input_image_store=store,
            work_directory=str(tmp_path / "work"),
            **{**_flat_atmosphere(20), **arguments},
        )


def test_needs_stokes_i(small_cube, tmp_path):
    store = _write_cube(
        tmp_path / "cube.img.zarr", small_cube.assign_coords(polarization=["Q"])
    )
    with pytest.raises(ValueError, match="Stokes I"):
        find_continuum(
            input_image_store=store,
            work_directory=str(tmp_path / "work"),
            **_flat_atmosphere(20),
        )


def test_ultra_refuses_a_task_over_the_memory_of_a_thread(
    small_cube, tmp_path, monkeypatch
):
    """By default the threads are the cluster's; 1.5 GiB per thread leaves no
    room for the first round of a task beside its baseline. The call is
    refused before anything is written to the work directory, so that it can
    be repeated there with other settings."""
    from astroviper.utils import data_partitioning

    monkeypatch.setattr(
        data_partitioning,
        "get_thread_info",
        lambda: {"n_threads": 1, "memory_per_thread": 1.5},
    )
    store = _write_cube(tmp_path / "cube.img.zarr", small_cube)
    work = tmp_path / "work"
    with pytest.raises(ValueError, match="than the 1.50 GiB of memory per thread"):
        _run("ultra", store, work, n_channels=20, thread_info=None)
    assert not os.path.exists(work) or os.listdir(work) == []
    with pytest.raises(ValueError, match="memory per thread"):
        _run("ultra", store, work, n_channels=20, thread_info=None)


@pytest.mark.parametrize(
    "thread_info",
    [
        {"n_threads": 0, "memory_per_thread": -1},
        {"n_threads": 4, "memory_per_thread": 0.0},
    ],
)
def test_ultra_refuses_a_cluster_without_workers_or_memory_limit(
    small_cube, tmp_path, monkeypatch, thread_info
):
    from astroviper.utils import data_partitioning

    monkeypatch.setattr(data_partitioning, "get_thread_info", lambda: thread_info)
    store = _write_cube(tmp_path / "cube.img.zarr", small_cube)
    with pytest.raises(ValueError, match="pass thread_info"):
        _run("ultra", store, tmp_path / "work", n_channels=20, thread_info=None)


def test_ultra_runs_a_task_over_its_budget_within_the_memory_of_a_thread(
    two_windows, tmp_path, monkeypatch
):
    """At 2 GiB per thread the budget, 1.2 GiB, is below a task's baseline:
    the tasks run at one plane per batch, with warnings, and find the
    channels of a run with room."""
    from astroviper.distributed_applications.imaging.find_continuum import (
        ultra_reductions,
    )

    warnings = []
    monkeypatch.setattr(ultra_reductions.logger, "warning", warnings.append)
    results = [
        _run(
            "ultra",
            two_windows["store"],
            tmp_path / name,
            n_channels=12,
            channel_range=[0, 12],
            thread_info=thread_info,
        )
        for name, thread_info in (
            ("tight", {"n_threads": 1, "memory_per_thread": 2.0}),
            ("room", THREAD_INFO),
        )
    ]
    _assert_same_results(*results)
    assert results[0]["memory_plan"]["budget_bytes"] == int(1.2 * 2**30)
    assert (results[0]["passes"]["planes_per_batch"] == 1).all()
    assert any("the first round" in warning for warning in warnings)


@pytest.mark.parametrize(
    "backend, n_chunks, monitor, memory_budget_gb",
    [
        ("fused", None, 0.05, None),
        ("fused", 3, 0.05, None),
        ("ultra", None, None, None),
        ("ultra", 3, 0.05, 100.0),
    ],
)
def test_profiling_and_report(
    line_set_store, tmp_path, backend, n_chunks, monitor, memory_budget_gb
):
    """The timing, resource series and report of image_cube_single_field, and
    the keys and columns of section 7 of the design."""
    pytest.importorskip("psutil")
    from astroviper.distributed_applications.imaging.find_continuum.find_continuum import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )

    result = _run(
        backend,
        line_set_store,
        tmp_path / "work",
        n_mapping_parallelism=None if n_chunks is None else {"frequency": n_chunks},
        monitor_resources_seconds=monitor,
        report="html",
        **({} if backend == "fused" else {"memory_budget_gb": memory_budget_gb}),
    )
    assert os.path.isfile(result["report"])
    assert result["report"].startswith(str(tmp_path / "work"))
    timing = result["timing_distributed_application"]
    leaves = [
        key for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES for _, key in steps
    ]
    assert all(timing[key] >= 0.0 for key in leaves)
    assert timing["T_total"] == pytest.approx(
        sum(timing[key] for key in leaves), rel=0.05
    )
    assert DRIVER_KEYS <= set(timing)
    assert ("driver_resource_usage" in timing) == (monitor is not None)
    assert timing["compute_start_unixtime"] <= timing["compute_end_unixtime"]
    assert timing["driver_rss_peak_bytes"] > 0
    assert set(result["first_round"]) == {
        "source",
        "store",
        "moments",
        "moments_complete",
        "imager",
    }
    assert result["channel_range"] == [0, 480]
    passes, tasks = result["passes"], result["timing_node_tasks"]
    assert list(passes["pass"]) == list(range(result["n_cube_reductions"]))
    if monitor is not None:
        assert {"cpu_percent", "memory_rss_bytes"} <= set(
            timing["driver_resource_usage"]
        )
        assert MONITOR_COLUMNS <= set(tasks)
    if backend == "fused":
        assert passes["T_compute_dask_graph"].sum() == pytest.approx(
            timing["T_compute_dask_graph"]
        )
        n_reduce_nodes = 0 if n_chunks is None else 2 * result["n_cube_reductions"]
        assert len(result["timing_reduce_nodes"]) == n_reduce_nodes
        assert result["memory_plan"] is None
        return
    assert set(passes) == PASS_COLUMNS
    assert TASK_COLUMNS <= set(tasks)
    assert set(tasks["family"]) == {"frequency", "merge"}
    assert set(result["memory_plan"]) == MEMORY_PLAN_KEYS
    # 0.8 of 4 GiB less the baseline of 0.8 GiB; 100 GiB is clamped to it.
    assert result["memory_plan"]["budget_bytes"] == int(2.4 * 2**30)
    shard = zarr.open_array(
        os.path.join(line_set_store, "SKY_RESIDUAL"), mode="r"
    ).chunks[1]
    assert result["memory_plan"]["n_frequency_tasks"] == (
        n_chunks or ultra_frequency_tasks(480, shard, THREAD_INFO["n_threads"])
    )
    for key, merge_key in (
        ("T_create_map_reduce_graph", "T_merge_create_graph"),
        ("T_generate_dask_graph", "T_merge_generate_dask_graph"),
        ("T_compute_dask_graph", "T_merge_compute_dask_graph"),
    ):
        assert timing[key] == pytest.approx(
            passes[key].sum() + passes[merge_key].sum(), rel=1e-9
        )
    assert timing["T_primary_beam_mean"] > 0 and timing["driver_bytes_read"] > 0
    records = result["timing_reduce_nodes"]
    assert len(records) == sum(max(n - 1, 0) for n in passes["n_tasks"])
    assert all(record["family"] == "frequency" for record in records)
    assert {record["reduction"] for record in records} <= set(passes["pass"])
