"""Tests of find_continuum, the pipeline's findContinuum on AstroVIPER images."""

import os
import sys

import numpy as np
import pytest
from xradio.image import write_image

from astroviper.distributed_applications.imaging import find_continuum
from astroviper.node_tasks.imaging.find_continuum import read_small_image
from astroviper.utils.data_group_tools import modify_data_groups_xds
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    CASA_RESULTS,
    LINE_SET,
    make_find_continuum_image_xds,
)

# The keys and columns of section 7 of the design.
DRIVER_KEYS = {
    "T_total",
    "application_start_unixtime",
    "application_end_unixtime",
    "compute_start_unixtime",
    "compute_end_unixtime",
    "driver_rss_peak_bytes",
    "driver_rss_lifetime_peak_bytes",
    "driver_bytes_read",
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


def _flat_atmosphere(n_channels):
    return {
        "atmospheric_transmission": np.full(n_channels, 0.92),
        "atmospheric_sky_temperature": np.full(n_channels, 27.0),
    }


def _run(store, work_directory, n_channels=480, **kwargs):
    return find_continuum(
        input_image_store=store,
        work_directory=str(work_directory),
        **_flat_atmosphere(n_channels),
        **kwargs,
    )


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


@pytest.fixture(scope="module")
def line_set_store(tmp_path_factory):
    return _write_cube(
        tmp_path_factory.mktemp("line_set") / "line_set.img.zarr",
        make_find_continuum_image_xds(lines=LINE_SET),
    )


@pytest.mark.parametrize("case", CASA_RESULTS)
def test_selection_matches_casa(case, tmp_path, monkeypatch):
    kwargs, selection, all_continuum, snr, dirty_cube_reads = CASA_RESULTS[case]
    store = _write_cube(
        tmp_path / f"{case}.img.zarr", make_find_continuum_image_xds(**kwargs)
    )
    work_directory = str(tmp_path / "work")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    monkeypatch.chdir(run_directory)

    result = _run(store, work_directory, n_mapping_parallelism={"frequency": 3})

    assert result["channel_selection"] == selection
    assert result["all_continuum"] is all_continuum
    assert result["moment_difference_snr"] == pytest.approx(snr, abs=1e-4)
    assert result["warnings"] == ["", ""]
    assert result["channel_range"] == [0, 480]
    assert os.path.isfile(result["plot"]) and os.path.isdir(result["joint_mask"])
    assert list(run_directory.iterdir()) == []
    assert set(result["cube_reads"]) == {"SKY_RESIDUAL", "PRIMARY_BEAM"}
    assert result["cube_reads"]["SKY_RESIDUAL"] == dirty_cube_reads
    timing = result["timing_node_tasks"]
    assert sorted(set(timing["reduction"])) == list(range(result["n_cube_reductions"]))
    assert len(timing) == 3 * result["n_cube_reductions"]

    with pytest.raises(ValueError, match="earlier run"):
        find_continuum(
            input_image_store=store,
            work_directory=work_directory,
            **_flat_atmosphere(480),
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


def test_each_window_equals_a_cube_of_that_window(two_windows, tmp_path):
    for window, single in zip(
        two_windows["windows"], two_windows["single"], strict=True
    ):
        directories = [
            str(tmp_path / f"{name}_{window[0]}") for name in ("range", "single")
        ]
        result = _run(
            two_windows["store"],
            directories[0],
            n_channels=12,
            channel_range=window,
            n_mapping_parallelism={"frequency": 2},
        )
        expected = _run(
            single,
            directories[1],
            n_channels=12,
            n_mapping_parallelism={"frequency": 2},
        )
        _assert_same_results(result, expected)
        assert result["channel_range"] == window
        assert result["cube_reads"] == expected["cube_reads"]
        _assert_same_small_images(*directories)


def test_a_window_without_two_edge_channels_is_refused(two_windows, tmp_path):
    """findContinuum finds no result on 7 channels: the reason it gives."""
    with pytest.raises(
        ValueError,
        match=r"no result for channels \[0, 7\) .* \(7 channels\): You must have at "
        r"least 2 edge channels",
    ):
        _run(two_windows["store"], tmp_path, n_channels=7, channel_range=[0, 7])


def test_a_cube_of_several_windows_needs_a_channel_range(two_windows, tmp_path):
    with pytest.raises(ValueError, match=r"\[\[0, 12\], \[12, 24\]\]"):
        _run(two_windows["store"], tmp_path / "work", n_channels=24)
    with pytest.raises(ValueError, match="one value per channel"):
        _run(
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


@pytest.mark.parametrize("monitor", [None, 0.05])
def test_driver_peak_is_the_calls_own(two_windows, tmp_path, monitor):
    """An earlier peak of the driver process is not reported as the call's:
    with the driver sampled, its largest sample; without, ``None``."""
    import mmap

    pytest.importorskip("psutil")
    for name in ("warm", "after_a_transient"):
        if name == "after_a_transient":
            # 256 MiB resident, then returned to the system.
            size = 256 * 2**20
            with mmap.mmap(-1, size) as transient:
                for offset in range(0, size, mmap.PAGESIZE):
                    transient[offset] = 1
        result = _run(
            two_windows["single"][0],
            tmp_path / name,
            n_channels=12,
            monitor_resources_seconds=monitor,
        )
    timing = result["timing_distributed_application"]
    peak, lifetime = (
        timing["driver_rss_peak_bytes"],
        timing["driver_rss_lifetime_peak_bytes"],
    )
    if monitor is None:
        assert peak is None
    else:
        samples = timing["driver_resource_usage"]["memory_rss_bytes"]
        assert peak == max(samples) < lifetime


@pytest.mark.parametrize("n_chunks, monitor", [(None, None), (3, 0.05)])
def test_profiling_and_report(line_set_store, tmp_path, n_chunks, monitor):
    """The timing, resource series and report of image_cube_single_field, and
    the keys and columns of section 7 of the design."""
    pytest.importorskip("psutil")
    from astroviper.distributed_applications.imaging.find_continuum.find_continuum import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )

    result = _run(
        line_set_store,
        tmp_path / "work",
        n_mapping_parallelism=None if n_chunks is None else {"frequency": n_chunks},
        monitor_resources_seconds=monitor,
        report="html",
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
    assert timing["driver_rss_lifetime_peak_bytes"] > 0
    if monitor is not None:
        assert (
            0
            < timing["driver_rss_peak_bytes"]
            <= timing["driver_rss_lifetime_peak_bytes"]
        )
    assert result["channel_range"] == [0, 480]
    passes, tasks = result["passes"], result["timing_node_tasks"]
    assert list(passes["pass"]) == list(range(result["n_cube_reductions"]))
    if monitor is not None:
        assert {"cpu_percent", "memory_rss_bytes"} <= set(
            timing["driver_resource_usage"]
        )
        assert MONITOR_COLUMNS <= set(tasks)
    assert passes["T_compute_dask_graph"].sum() == pytest.approx(
        timing["T_compute_dask_graph"]
    )
    n_reduce_nodes = 0 if n_chunks is None else 2 * result["n_cube_reductions"]
    assert len(result["timing_reduce_nodes"]) == n_reduce_nodes
