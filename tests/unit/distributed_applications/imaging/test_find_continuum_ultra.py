"""Tests of find_continuum_ultra against the simple port find_continuum and
CASA (design section 14.3).

The port and find_continuum_ultra run at the same frequency chunks, where
their results are equal bit for bit, on the synthetic cubes of
``find_continuum_test_utils``: those with CASA's results, and runs of the
branches no CASA case takes (``PORT_RUNS``). Branches that no cube here takes
run with forced decisions.
"""

import importlib
import os
import sys
import uuid

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
from xradio.image import write_image

from astroviper.distributed_applications.imaging import (
    find_continuum,
    find_continuum_ultra,
)
from astroviper.distributed_applications.imaging.find_continuum_first_round import (
    bind_find_continuum_first_round_store,
    create_find_continuum_first_round_store,
    drain_first_round_moments,
    find_continuum_first_round_partials_path,
    find_continuum_first_round_store_path,
)
from astroviper.node_tasks.imaging.find_continuum_first_round import (
    find_continuum_first_round_chunk,
    write_first_round_rows,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
    linear_spectral_axis,
    moment_channel_weights,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    Decision,
)
from astroviper.processing_functions.imaging.utils.spectral_windows import (
    linear_channel_ranges,
)
from astroviper.utils.data_group_tools import modify_data_groups_xds
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    CASA_RESULTS,
    NATIVE_CASES,
    NATIVE_RESIDUAL_READS,
    make_find_continuum_image_xds,
)

# The modules, not the functions the packages export under their names.
driver = importlib.import_module(
    "astroviper.distributed_applications.imaging.find_continuum_ultra"
)
rounds = importlib.import_module(
    "astroviper.processing_functions.imaging.find_continuum_ultra.rounds"
)
selection_maps = importlib.import_module(
    "astroviper.processing_functions.imaging.find_continuum_ultra.selection_maps"
)

THREAD_INFO = {"n_threads": 4, "memory_per_thread": 4.0}
CHUNKS = {"frequency": 3}
VARIABLES = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
VALUE_DTYPES = {"sky": "float32", "primary_beam": "float32"}
CASA_CASES = {case: values[:4] for case, values in CASA_RESULTS.items()} | NATIVE_CASES
# Runs without a CASA reference, of the branches no CASA case takes (design
# section 14.5): (cube, bad atmosphere, other arguments). findContinuum's
# normalization (a sky temperature with a 60 K line), its primary-beam-based
# first-round mask (no pixel above the thresholds), its fallback for a mean
# spectrum of scaled MAD 0, the smoothing of n_bin channels and its cap, and
# the removal of an initial quadratic.
PORT_RUNS = {
    "line_set_badatm": ("line_set", True, {}),
    "noise_only": ("noise_only", False, {}),
    "constant_spectrum": ("constant_spectrum", False, {}),
    "no_lines_n_bin_4": ("no_lines", False, {"n_bin": 4}),
    "line_set_n_bin_4": ("line_set", False, {"n_bin": 4}),
    "quadratic": ("quadratic", False, {}),
}
# The keys and columns of design section 12.
RESULT_KEYS = {
    "continuum_channel_ranges",
    "channel_range",
    "continuum_frequency_ranges",
    "frequency_frame",
    "aggregate_bandwidth",
    "all_continuum",
    "low_bandwidth",
    "low_spread",
    "single_range_channel_fraction",
    "moment_difference_snr",
    "constant_cube",
    "joint_mask",
    "joint_mask_kind",
    "joint_mask_primary_beam_based",
}
PROFILING_KEYS = {
    "diagnostics",
    "n_passes",
    "cube_reads",
    "passes",
    "timing_node_tasks",
    "timing_reduce_nodes",
    "timing_distributed_application",
    "first_round",
    "memory_plan",
    "report",
}
DIAGNOSTICS_KEYS = {
    "settings",
    "first_round_statistics",
    "rounds",
    "change_code",
    "path_code",
    "reverted",
    "low_spread_fix_applied",
    "mean_spectrum",
    "spectrum",
    "positive_threshold",
    "negative_threshold",
    "corrected_median",
}
PASS_COLUMNS = {
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
}
TASK_COLUMNS = {
    "pass_index",
    "pass_kind",
    "round_kind",
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
    "T_read_partials",
    "T_merge",
    "T_write_maps",
    "T_cpu",
    "T_find_continuum_task",
}
DRIVER_KEYS = {
    "T_total",
    "application_start_unixtime",
    "application_end_unixtime",
    "compute_start_unixtime",
    "compute_end_unixtime",
    "driver_rss_peak_bytes",
    "driver_rss_lifetime_peak_bytes",
    "driver_bytes_read",
    "driver_bytes_written",
}
MEMORY_PLAN_KEYS = {
    "n_threads",
    "memory_per_thread_gib",
    "budget_bytes",
    "n_frequency_tasks",
    "channels_per_task",
    "merge_tile_shape",
    "n_merge_tiles",
    "plane_bytes",
    "driver_memory_estimate_bytes",
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


def _write_cube(path, img_xds):
    write_image(img_xds, imagename=str(path), out_format="zarr", overwrite=True)
    return str(path)


def _atmosphere(n_channels, bad=False):
    sky_temperature = np.full(n_channels, 27.0)
    if bad:
        channel = np.arange(n_channels)
        sky_temperature += 60.0 * np.exp(
            -0.5 * ((channel - n_channels / 2) / (n_channels / 40)) ** 2
        )
    return {
        "atmospheric_transmission": np.full(n_channels, 0.92),
        "atmospheric_sky_temperature": sky_temperature,
    }


def _constant_spectrum_cube():
    """The same planes in every channel in a disk around the continuum source
    and a primary beam constant in frequency: the mean spectrum in the joint
    mask is constant, of scaled MAD 0 (fC 8258)."""
    img_xds = make_find_continuum_image_xds(n_channels=60, continuum_jy=0.05)
    sky = img_xds["SKY_RESIDUAL"].values
    primary_beam = img_xds["PRIMARY_BEAM"].values
    l_index, m_index = np.indices(sky.shape[-2:])
    disk = (l_index - 31.5) ** 2 + (m_index - 31.5) ** 2 < 15**2
    sky[:, :, :, disk] = sky[:, :1, :, disk]
    primary_beam[:] = primary_beam[:, :1]
    return img_xds


def _quadratic_cube():
    """A cube of no line with a quadratic spectral baseline everywhere, which
    findContinuum removes (fC 8278-8282)."""
    img_xds = make_find_continuum_image_xds(lines=())
    channel = np.arange(img_xds.sizes["frequency"])
    baseline = 3e-3 * ((channel - 240) / 240.0) ** 2
    img_xds["SKY_RESIDUAL"].values[:] += baseline[:, None, None, None].astype(
        np.float32
    )
    return img_xds


def _cube(case):
    """The image dataset of a case."""
    if case in CASA_CASES:
        return make_find_continuum_image_xds(**CASA_CASES[case][0])
    if case == "noise_only":
        return make_find_continuum_image_xds(continuum_jy=0.0, n_channels=120)
    if case == "quadratic":
        return _quadratic_cube()
    return _constant_spectrum_cube()


def _ranges(selection):
    """findContinuum's channel selection as ``[start, stop)`` pairs."""
    return [
        [int(start), int(stop) + 1]
        for start, stop in (pair.split("~") for pair in selection.split(";"))
    ]


def _run(store, work_directory, n_channels, **kwargs):
    return find_continuum_ultra(
        input_image_store=store,
        work_directory=str(work_directory),
        **_atmosphere(n_channels),
        **{"thread_info": THREAD_INFO, **kwargs},
    )


def _imager_first_round_store(store, chunk_channels=160):
    """The imager's first round of a cube, made in this process by its node
    tasks (the moments in this process's registry), drained and bound, as
    ``image_cube_single_field`` makes it."""
    xds = xr.open_zarr(store, chunks=None)
    frequency = xds.frequency.values
    rest_frequency = float(xds.frequency.attrs["rest_frequency"]["data"])
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
    sky = zarr.open_array(os.path.join(store, "SKY_RESIDUAL"), mode="r")
    path = find_continuum_first_round_store_path(store)
    run_id = uuid.uuid4().hex
    create_find_continuum_first_round_store(
        path,
        frequency.size,
        sky.shape,
        sky.chunks[-2:],
        sky.chunks[-2:],
        0.2,
        VARIABLES,
        0,
        VALUE_DTYPES,
        run_id,
        "image_cube_single_field",
        windows,
        rest_frequency,
    )
    chunks = [
        find_continuum_first_round_chunk(
            xds.isel(frequency=slice(start, start + chunk_channels)),
            np.arange(start, min(start + chunk_channels, frequency.size)),
            path,
            VARIABLES,
            0,
            0.2,
            sky.chunks[-2:],
            VALUE_DTYPES,
            run_id,
            windows,
            find_continuum_first_round_partials_path(path),
            rest_frequency,
        )
        for start in range(0, frequency.size, chunk_channels)
    ]
    rows = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    assert write_first_round_rows(path, dict(rows, write_failed=False), frequency.size)[
        "complete"
    ]
    drained, _ = drain_first_round_moments(run_id, path, windows)
    assert drained["moments_complete"] == [True] * len(windows)
    bind_find_continuum_first_round_store(path, store)
    return path


def _set_store_attributes(path, **changes):
    group = zarr.open_group(path, mode="r+")
    group.attrs["find_continuum_first_round"] = dict(
        group.attrs["find_continuum_first_round"], **changes
    )


def _assert_same_results(result, expected):
    for key in RESULT_KEYS - {"channel_range", "joint_mask"}:
        assert result[key] == expected[key], key


@pytest.fixture(scope="module")
def cube_stores(tmp_path_factory):
    """The cube store of a case, written once."""
    directory = tmp_path_factory.mktemp("cubes")
    stores = {}

    def store(case):
        if case not in stores:
            stores[case] = _write_cube(
                directory / case / f"{case}.img.zarr", _cube(case)
            )
        return stores[case]

    return store


@pytest.fixture(scope="module")
def runs(cube_stores, tmp_path_factory):
    """The port and find_continuum_ultra on a case at three frequency chunks,
    run once."""
    directory = tmp_path_factory.mktemp("runs")
    results = {}

    def run(case):
        if case not in results:
            cube, bad_atmosphere, arguments = PORT_RUNS.get(case, (case, False, {}))
            store = cube_stores(cube)
            arguments = {
                **_atmosphere(xr.open_zarr(store).sizes["frequency"], bad_atmosphere),
                **arguments,
            }
            port = find_continuum(
                input_image_store=store,
                work_directory=str(directory / case / "port"),
                n_mapping_parallelism=CHUNKS,
                **arguments,
            )
            native = find_continuum_ultra(
                input_image_store=store,
                work_directory=str(directory / case / "native"),
                n_mapping_parallelism=CHUNKS,
                thread_info=THREAD_INFO,
                **arguments,
            )
            results[case] = port, native
        return results[case]

    return run


@pytest.mark.parametrize("case", [*CASA_CASES, *PORT_RUNS])
def test_results_equal_the_port(runs, case):
    """The results of the port at the same frequency chunks; CASA's selection
    for the cubes of CASA's results, and the branch of each run without."""
    port, native = runs(case)
    assert native["continuum_channel_ranges"] == _ranges(port["channel_selection"])
    assert native["all_continuum"] is port["all_continuum"]
    assert [native["low_bandwidth"], native["low_spread"]] == [
        bool(warning) for warning in port["warnings"]
    ]
    assert native["moment_difference_snr"] == port["moment_difference_snr"]
    assert native["aggregate_bandwidth"] * 1e-9 == port["aggregate_bandwidth"]
    assert native["constant_cube"] is False
    assert os.path.isdir(native["joint_mask"])
    if case in CASA_CASES:
        _, selection, all_continuum, snr = CASA_CASES[case]
        assert native["continuum_channel_ranges"] == _ranges(selection)
        assert native["all_continuum"] is all_continuum
        assert native["moment_difference_snr"] == pytest.approx(snr, abs=1e-4)
        assert native["cube_reads"] == {
            "SKY_RESIDUAL": NATIVE_RESIDUAL_READS[case][0],
            "PRIMARY_BEAM": 1,
        }
        assert native["n_passes"] == NATIVE_RESIDUAL_READS[case][0]
    else:
        _assert_branch(case, native)


def _assert_branch(case, result):
    """The branch a run without a CASA reference takes: the normalization
    reads the primary beam once more; the primary-beam-based first-round mask
    reads its plane 1 outside the passes; the fallback of a spectrum of scaled
    MAD 0 makes one spectrum pass more; n_bin 4 smooths the spectrum, capped to
    2 for a peak over MAD above 10; a quadratic baseline is removed."""
    diagnostics = result["diagnostics"]
    first_round, rounds = diagnostics["first_round_statistics"], diagnostics["rounds"]
    if case == "line_set_badatm":
        assert diagnostics["settings"]["bad_atmosphere"]
        assert result["cube_reads"] == {"SKY_RESIDUAL": 3, "PRIMARY_BEAM": 2}
    elif case == "noise_only":
        assert first_round["needs_primary_beam_based_mask"]
        assert result["joint_mask_primary_beam_based"]
        assert result["cube_reads"] == {"SKY_RESIDUAL": 3, "PRIMARY_BEAM": 1}
    elif case == "constant_spectrum":
        assert not first_round["needs_primary_beam_based_mask"]
        assert rounds["joint_mask_primary_beam_based"][0]
        assert result["joint_mask_primary_beam_based"]
        assert list(result["passes"]["pass_kind"]) == [
            "first_round",
            "spectrum",
            "spectrum",
            "selection_moments",
        ]
    elif case == "quadratic":
        assert rounds["initial_quadratic_removed"][0]
    else:
        assert (
            rounds["n_bin"][0] == {"no_lines_n_bin_4": 4, "line_set_n_bin_4": 2}[case]
        )


@pytest.mark.parametrize(
    "case", ["line_set", "tdm", "sidelobes", "only_extra_mask", "eight_channels"]
)
def test_with_the_first_round_of_the_imager(runs, cube_stores, tmp_path, case):
    """The imager's first round saves the first pass, its read of the residual
    and of the primary beam; the selections are the same and the
    signal-to-noise ratio differs in its last digits (the order of the
    moment-0 sums)."""
    _, computed = runs(case)
    store = cube_stores(case)
    first_round_store = _imager_first_round_store(store)
    n_channels = len(computed["diagnostics"]["mean_spectrum"])
    imager = _run(
        store,
        tmp_path / "imager",
        n_channels,
        n_mapping_parallelism=CHUNKS,
        first_round_store=first_round_store,
    )
    _assert_same_results(
        {**imager, "moment_difference_snr": None},
        {**computed, "moment_difference_snr": None},
    )
    assert imager["moment_difference_snr"] == pytest.approx(
        computed["moment_difference_snr"], rel=1e-12
    )
    assert imager["cube_reads"] == {
        "SKY_RESIDUAL": NATIVE_RESIDUAL_READS[case][1],
        "PRIMARY_BEAM": 0,
    }
    assert "first_round" not in set(imager["passes"]["pass_kind"])
    assert imager["first_round"] == {
        "source": "imager",
        "store": first_round_store,
        "moments": "imager",
        "imager": None,
    }
    assert computed["first_round"]["source"] == "find_continuum_ultra"
    assert computed["first_round"]["moments"] == "computed"
    assert os.path.isdir(computed["first_round"]["store"])


def test_with_the_imager_store_without_its_moments(runs, cube_stores, tmp_path):
    """Without the moments the first pass makes them from the store's masks,
    without reading the primary beam: the results of the run without a store,
    bit for bit."""
    _, computed = runs("line_set")
    store = cube_stores("line_set")
    first_round_store = _imager_first_round_store(store)
    _set_store_attributes(first_round_store, moments_complete=[False])
    result = _run(
        store,
        tmp_path / "work",
        480,
        n_mapping_parallelism=CHUNKS,
        first_round_store=first_round_store,
    )
    _assert_same_results(result, computed)
    assert result["cube_reads"] == {"SKY_RESIDUAL": 3, "PRIMARY_BEAM": 0}
    assert result["first_round"]["source"] == "imager"
    assert result["first_round"]["moments"] == "computed"
    first_pass = result["passes"].iloc[0]
    assert first_pass["pass_kind"] == "first_round"
    assert not first_pass["computes_first_round_masks"]


@pytest.fixture(scope="module")
def two_windows(tmp_path_factory):
    """A cube of two spectral windows of 12 channels with a gap, and a cube of
    each window alone."""
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


def _joint_mask(path):
    return xr.open_zarr(path, consolidated=False).load()


def test_each_window_equals_a_cube_of_that_window(two_windows, tmp_path):
    """Each window, run with its channel range, gives the results of a cube
    of that window alone and of the port on it."""
    for window, single in zip(
        two_windows["windows"], two_windows["single"], strict=True
    ):
        chunks = {"frequency": 2}
        result = _run(
            two_windows["store"],
            tmp_path / f"range_{window[0]}",
            12,
            channel_range=window,
            n_mapping_parallelism=chunks,
        )
        expected = _run(
            single, tmp_path / f"single_{window[0]}", 12, n_mapping_parallelism=chunks
        )
        _assert_same_results(result, expected)
        assert result["channel_range"] == window
        assert result["cube_reads"] == expected["cube_reads"]
        assert result["n_passes"] == expected["n_passes"]
        assert os.path.basename(os.path.dirname(result["joint_mask"])) == (
            f"cube.find_continuum_ultra_{window[0]}_{window[1]}"
        )
        xr.testing.assert_identical(
            _joint_mask(result["joint_mask"]), _joint_mask(expected["joint_mask"])
        )
        port = find_continuum(
            input_image_store=two_windows["store"],
            work_directory=str(tmp_path / f"port_{window[0]}"),
            channel_range=window,
            n_mapping_parallelism=chunks,
            **_atmosphere(12),
        )
        assert result["continuum_channel_ranges"] == _ranges(port["channel_selection"])
        assert result["moment_difference_snr"] == port["moment_difference_snr"]
        assert result["all_continuum"] is port["all_continuum"]


@pytest.mark.parametrize("monitor", [None, 0.05])
def test_return_dictionary(two_windows, tmp_path, monitor):
    """Every key and column of design section 12, with and without the
    resource monitor, and the report."""
    pytest.importorskip("psutil")
    result = _run(
        two_windows["single"][0],
        tmp_path / "work",
        12,
        n_mapping_parallelism={"frequency": 2},
        monitor_resources_seconds=monitor,
        report=None if monitor is None else "html",
    )
    assert set(result) == RESULT_KEYS | PROFILING_KEYS
    assert set(result["diagnostics"]) == DIAGNOSTICS_KEYS
    assert result["channel_range"] == [0, 12]
    assert result["frequency_frame"] == "LSRK"
    assert result["joint_mask_kind"] in ("phase_1", "phase_2", "amended")
    joint_mask = _joint_mask(result["joint_mask"])
    assert set(joint_mask) == {"JOINT_MASK", "JOINT_MASK_FOOTPRINT"}
    assert joint_mask.attrs["kind"] == result["joint_mask_kind"]
    assert len(result["continuum_frequency_ranges"]) == len(
        result["continuum_channel_ranges"]
    )
    assert all(low < high for low, high in result["continuum_frequency_ranges"])
    diagnostics = result["diagnostics"]
    assert isinstance(diagnostics["rounds"], pd.DataFrame)
    assert list(diagnostics["rounds"]["round_index"]) == list(
        range(len(diagnostics["rounds"]))
    )
    assert "moment_channel_weights" not in diagnostics["settings"]
    assert list(diagnostics["mean_spectrum"].dims) == ["frequency"]

    passes, tasks = result["passes"], result["timing_node_tasks"]
    assert PASS_COLUMNS <= set(passes)
    assert list(passes["pass_index"]) == list(range(result["n_passes"]))
    assert set(passes["pass_kind"]) <= {"first_round", "spectrum", "selection_moments"}
    assert TASK_COLUMNS <= set(tasks)
    assert set(tasks["pass_index"]) == set(range(result["n_passes"]))
    assert set(tasks["family"]) == {"frequency", "merge"}
    reduces = pd.DataFrame(result["timing_reduce_nodes"])
    assert set(reduces["pass_index"]) <= set(range(result["n_passes"]))
    assert result["cube_reads"] == {
        variable: sum(variable in read for read in passes["data_variables_read"])
        for variable in ("SKY_RESIDUAL", "PRIMARY_BEAM")
    }

    timing = result["timing_distributed_application"]
    leaves = driver._LEAVES
    assert DRIVER_KEYS | set(leaves) <= set(timing)
    assert ("driver_resource_usage" in timing) == (monitor is not None)
    assert all(timing[key] >= 0.0 for key in leaves)
    assert timing["T_total"] == pytest.approx(
        sum(timing[key] for key in leaves), rel=0.05
    )
    assert timing["T_compute_dask_graph"] == pytest.approx(
        (passes["T_compute_dask_graph"] + passes["T_merge_compute_dask_graph"]).sum()
    )
    assert timing["T_read_maps"] >= passes["T_read_maps"].sum()
    assert (
        timing["application_start_unixtime"]
        <= timing["compute_start_unixtime"]
        <= timing["compute_end_unixtime"]
        <= timing["application_end_unixtime"]
    )
    assert timing["driver_bytes_read"] > passes["driver_bytes_read"].sum()
    assert timing["driver_bytes_written"] > passes["driver_bytes_written"].sum()
    assert timing["driver_rss_lifetime_peak_bytes"] > 0
    assert set(result["first_round"]) == {"source", "store", "moments", "imager"}
    assert set(result["memory_plan"]) == MEMORY_PLAN_KEYS
    assert result["memory_plan"]["n_merge_tiles"] == len(
        set(tasks.loc[tasks["family"] == "merge", "task_id"])
    )
    if monitor is None:
        assert result["report"] is None
    else:
        assert MONITOR_COLUMNS <= set(tasks)
        assert {"cpu_percent", "memory_rss_bytes"} <= set(
            timing["driver_resource_usage"]
        )
        assert result["report"] == (
            os.path.dirname(result["joint_mask"]) + "_report.html"
        )
        with open(result["report"]) as report:
            text = report.read()
        assert "find_continuum_ultra report" in text and "Results" in text


def test_a_constant_cube(tmp_path):
    """No continuum channel and no joint mask; with the imager's store no
    pass is made (heur 89-94)."""
    img_xds = make_find_continuum_image_xds(n_pixels=16, n_channels=12)
    img_xds["SKY_RESIDUAL"][:] = 0.5
    store = _write_cube(tmp_path / "constant.img.zarr", img_xds)
    result = _run(store, tmp_path / "work", 12, report="html")
    assert result["constant_cube"]
    assert result["continuum_channel_ranges"] == []
    assert result["moment_difference_snr"] is None
    assert result["joint_mask"] is None
    assert result["n_passes"] == 1
    assert result["cube_reads"] == {"SKY_RESIDUAL": 1, "PRIMARY_BEAM": 1}
    assert os.path.isfile(result["report"])
    imager = _run(
        store,
        tmp_path / "imager",
        12,
        first_round_store=_imager_first_round_store(store, chunk_channels=5),
    )
    assert imager["constant_cube"] and imager["n_passes"] == 0
    assert imager["cube_reads"] == {"SKY_RESIDUAL": 0, "PRIMARY_BEAM": 0}
    assert imager["timing_distributed_application"]["compute_start_unixtime"] is None


@pytest.fixture
def small_store(tmp_path):
    """A cube of 12 channels with the data groups residual and sky_only."""
    img_xds = make_find_continuum_image_xds(n_pixels=16, n_channels=12)
    modify_data_groups_xds(img_xds, "sky_only", {"sky": "SKY_RESIDUAL"}, "Sky only")
    return _write_cube(tmp_path / "small.img.zarr", img_xds)


@pytest.mark.parametrize(
    "arguments, error, match",
    [
        (dict(n_bin=0), ValueError, "n_bin"),
        (dict(primary_beam_limit=0.0), ValueError, r"\(0, 1\)"),
        (dict(spectral_dynamic_range_bandwidth=-1.0), ValueError, "positive"),
        (dict(n_mapping_parallelism={"time": 2}), ValueError, "frequency"),
        (
            dict(thread_info={"n_threads": 0, "memory_per_thread": 4.0}),
            ValueError,
            "n_threads",
        ),
        (dict(memory_budget_gb=0.0), ValueError, "positive"),
        (dict(atmospheric_transmission=np.ones(3)), ValueError, "one value per"),
        (dict(atmospheric_transmission=np.zeros(12)), ValueError, "no zero"),
        (dict(channel_range=[0, 7]), ValueError, "at least 8"),
        (dict(image_data_group_in_name="sky_only"), ValueError, "lacks the roles"),
        (
            dict(thread_info={"n_threads": 1, "memory_per_thread": 1.5}),
            ValueError,
            "memory per thread",
        ),
    ],
)
def test_refusals_before_anything_is_written(
    small_store, tmp_path, arguments, error, match
):
    work_directory = tmp_path / "work"
    with pytest.raises(error, match=match):
        find_continuum_ultra(
            input_image_store=small_store,
            work_directory=str(work_directory),
            **{**_atmosphere(12), "thread_info": THREAD_INFO, **arguments},
        )
    assert not work_directory.exists()


def test_decreasing_frequencies_are_refused(tmp_path):
    img_xds = make_find_continuum_image_xds(
        n_pixels=16, n_channels=12, channel_width=-0.9765625e6
    )
    store = _write_cube(tmp_path / "decreasing.img.zarr", img_xds)
    with pytest.raises(ValueError, match="must increase"):
        _run(store, tmp_path / "work", 12)


def test_refused_first_round_stores_and_an_earlier_run(small_store, tmp_path):
    first_round_store = _imager_first_round_store(small_store, chunk_channels=5)
    result = _run(small_store, tmp_path / "work", 12)
    with pytest.raises(ValueError, match="earlier run"):
        _run(small_store, tmp_path / "work", 12)
    assert result["first_round"]["store"].startswith(str(tmp_path / "work"))
    _set_store_attributes(
        first_round_store,
        variables={"sky": "POINT_SPREAD_FUNCTION", "primary_beam": "PRIMARY_BEAM"},
    )
    with pytest.raises(ValueError, match="was made from"):
        _run(small_store, tmp_path / "other", 12, first_round_store=first_round_store)
    _set_store_attributes(first_round_store, version=1)
    with pytest.raises(
        ValueError, match="version 1, not 2: make the first round again"
    ):
        _run(small_store, tmp_path / "other", 12, first_round_store=first_round_store)
    assert not (tmp_path / "other").exists()


def test_a_window_without_a_valid_voxel(tmp_path):
    img_xds = make_find_continuum_image_xds(n_pixels=16, n_channels=12)
    img_xds["SKY_RESIDUAL"][:] = np.nan
    store = _write_cube(tmp_path / "blank.img.zarr", img_xds)
    with pytest.raises(ValueError, match="no valid voxel"):
        _run(store, tmp_path / "work", 12)


def test_input_checked_skips_only_the_configuration_checks(
    small_store, tmp_path, monkeypatch
):
    config_check = importlib.import_module(
        "astroviper.processing_functions.imaging.config_check.param_check_find_continuum"
    )

    def refuse(*args):
        raise AssertionError("configuration checked")

    monkeypatch.setattr(config_check, "check_find_continuum_ultra_config", refuse)
    result = _run(small_store, tmp_path / "work", 12, input_checked=True)
    assert result["n_passes"] == 3
    with pytest.raises(AssertionError, match="configuration checked"):
        _run(small_store, tmp_path / "again", 12)
    with pytest.raises(ValueError, match="one value per"):
        find_continuum_ultra(
            input_image_store=small_store,
            work_directory=str(tmp_path / "inputs"),
            **{**_atmosphere(12), "atmospheric_transmission": np.ones(3)},
            thread_info=THREAD_INFO,
            input_checked=True,
        )


def test_rounds_after_an_amended_round_and_an_extra_mask(
    cube_stores, tmp_path, monkeypatch
):
    """The amended round's maps of the intersection, its extra-mask spectrum
    and rounds extra_mask, auto_lower and auto_lower_2, with every extra-mask
    decision forced to yes and a significant difference map (no CASA case
    takes them)."""
    extra_mask_decision = rounds.extra_mask_decision

    def yes(bad_atmosphere, median, snr, scaled_mad, ten_event_sigma):
        decision = extra_mask_decision(
            bad_atmosphere, median, snr, scaled_mad, ten_event_sigma
        )
        if decision.kind == "no":
            return Decision("yes_moment", float(median + 2.0 * scaled_mad), 2.0)
        return decision

    monkeypatch.setattr(rounds, "extra_mask_decision", yes)
    monkeypatch.setattr(selection_maps, "difference_map_snr", lambda *args: 100.0)
    result = _run(
        cube_stores("sidelobes"), tmp_path / "work", 480, n_mapping_parallelism=CHUNKS
    )
    table = result["diagnostics"]["rounds"]
    assert list(table["round_kind"]) == [
        "original",
        "amended_mask",
        "extra_mask",
        "auto_lower",
        "auto_lower_2",
    ]
    assert table["threshold_sigma"][2] == 2.5
    assert table["threshold_sigma"][3] == pytest.approx(
        table["final_threshold_sigma"][2] * 6 / 7
    )
    assert table["result"].iloc[-1] == "result_7"
    assert table["intersection_of_selections"][1]
    passes = result["passes"]
    assert list(zip(passes["pass_kind"], passes["round_index"], strict=True)) == [
        ("first_round", 0),
        ("spectrum", 0),
        ("selection_moments", 0),
        ("spectrum", 1),
        ("selection_moments", 1),
        ("selection_moments", 1),
        ("spectrum", 1),
        ("selection_moments", 2),
        ("selection_moments", 3),
        ("selection_moments", 4),
    ]
    assert result["cube_reads"]["SKY_RESIDUAL"] == 10
    assert result["diagnostics"]["path_code"].startswith("ADIE")


def test_reversion_after_the_low_spread_fix(runs, cube_stores, tmp_path, monkeypatch):
    """A reversion after the low-spread fix gives round original's ranges and
    signal-to-noise ratio (fC 3054-3066)."""
    _, fixed = runs("eight_channels")
    assert fixed["diagnostics"]["low_spread_fix_applied"]
    assert not fixed["diagnostics"]["reverted"]
    monkeypatch.setattr(rounds, "after_low_spread_fix", lambda *args: True)
    result = _run(
        cube_stores("eight_channels"),
        tmp_path / "work",
        8,
        n_mapping_parallelism=CHUNKS,
    )
    diagnostics = result["diagnostics"]
    assert diagnostics["reverted"] and diagnostics["low_spread_fix_applied"]
    original = diagnostics["rounds"].iloc[0]
    assert result["continuum_channel_ranges"] == original["continuum_ranges"]
    assert result["moment_difference_snr"] == original["moment_difference_snr"]
    assert result["n_passes"] == fixed["n_passes"]


def test_low_spread_fix_without_a_candidate(cube_stores, tmp_path, monkeypatch):
    """Without channels to add, the fix makes no pass and keeps the ranges of
    the rounds unless both warnings remain (fC 3051-3053)."""
    calls = []

    def after_low_spread_fix(*args):
        calls.append(args)
        return False

    monkeypatch.setattr(rounds, "low_spread_ranges", lambda *args: None)
    monkeypatch.setattr(rounds, "after_low_spread_fix", after_low_spread_fix)
    result = _run(
        cube_stores("eight_channels"),
        tmp_path / "work",
        8,
        n_mapping_parallelism=CHUNKS,
    )
    diagnostics = result["diagnostics"]
    assert not diagnostics["low_spread_fix_applied"]
    assert not diagnostics["reverted"]
    assert [args[0] for args in calls] == [None]
    assert result["n_passes"] == 3
    original = diagnostics["rounds"].iloc[0]
    assert result["continuum_channel_ranges"] == original["continuum_ranges"]
    assert result["moment_difference_snr"] == original["moment_difference_snr"]


def test_threads_of_the_cluster(small_store, tmp_path, monkeypatch):
    """Without thread_info the cluster's threads are taken, refused without a
    memory limit."""
    data_partitioning = importlib.import_module("astroviper.utils.data_partitioning")
    monkeypatch.setattr(data_partitioning, "get_thread_info", lambda: THREAD_INFO)
    result = _run(small_store, tmp_path / "work", 12, thread_info=None)
    assert result["memory_plan"]["n_threads"] == THREAD_INFO["n_threads"]
    monkeypatch.setattr(
        data_partitioning,
        "get_thread_info",
        lambda: {"n_threads": 4, "memory_per_thread": -1},
    )
    with pytest.raises(ValueError, match="pass thread_info"):
        _run(small_store, tmp_path / "other", 12, thread_info=None)
