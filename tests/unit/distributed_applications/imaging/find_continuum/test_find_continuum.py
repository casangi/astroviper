"""Tests of find_continuum, the pipeline's findContinuum on AstroVIPER images."""

import os

import numpy as np
import pytest
from xradio.image import write_image

from astroviper.distributed_applications.imaging import (
    find_continuum,
    find_continuum_fused,
)
from astroviper.utils.data_group_tools import modify_data_groups_xds
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    LINE_SET,
    make_find_continuum_image_xds,
)

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
# find_continuum and in find_continuum_fused.
CASA_RESULTS = {
    "no_lines": (dict(lines=()), "9~464", True, 4.4041, (7, 3)),
    "line_set": (dict(lines=LINE_SET), "6~85;154~367;433~472", False, 6.2507, (7, 3)),
    # The primary beam falls below 0.2 in the corners: masked there, as by
    # tclean's pblimit in the pipeline (and in the CASA reference).
    "small_beam": (
        dict(lines=LINE_SET, primary_beam_fwhm_arcsec=8.0),
        "6~85;154~367;433~467",
        False,
        7.3877,
        (7, 3),
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
        (8, 5),
    ),
}


def _write_cube(path, img_xds):
    write_image(img_xds, imagename=str(path), out_format="zarr", overwrite=True)
    return str(path)


def _flat_atmosphere(n_channels):
    return {
        "atmospheric_transmission": np.full(n_channels, 0.92),
        "atmospheric_sky_temperature": np.full(n_channels, 27.0),
    }


@pytest.mark.parametrize("fused", [False, True], ids=["separate", "fused"])
@pytest.mark.parametrize("case", CASA_RESULTS)
def test_selection_matches_casa(case, fused, tmp_path, monkeypatch):
    kwargs, selection, all_continuum, snr, dirty_cube_reads = CASA_RESULTS[case]
    store = _write_cube(
        tmp_path / f"{case}.img.zarr", make_find_continuum_image_xds(**kwargs)
    )
    work_directory = str(tmp_path / "work")
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    monkeypatch.chdir(run_directory)

    result = (find_continuum_fused if fused else find_continuum)(
        input_image_store=store,
        work_directory=work_directory,
        n_mapping_parallelism={"frequency": 3},
        **_flat_atmosphere(480),
    )

    assert result["channel_selection"] == selection
    assert result["all_continuum"] is all_continuum
    assert result["moment_difference_snr"] == pytest.approx(snr, abs=1e-4)
    assert result["warnings"] == ["", ""]
    assert os.path.isfile(result["plot"]) and os.path.isdir(result["joint_mask"])
    assert list(run_directory.iterdir()) == []
    assert set(result["cube_reads"]) == {"SKY_RESIDUAL", "PRIMARY_BEAM"}
    assert result["cube_reads"]["SKY_RESIDUAL"] == dirty_cube_reads[fused]
    timing = result["timing_node_tasks"]
    assert len(timing) == 3 * result["n_cube_reductions"]
    assert sorted(set(timing["reduction"])) == list(range(result["n_cube_reductions"]))

    with pytest.raises(ValueError, match="earlier run"):
        find_continuum(
            input_image_store=store,
            work_directory=work_directory,
            **_flat_atmosphere(480),
        )


def test_fused_with_the_first_round_of_the_imager(tmp_path):
    """Given the first round (as image_cube_single_field makes it), the fused
    version reads the cube once less and finds the same channels."""
    import xarray as xr

    from astroviper.processing_functions.imaging.find_continuum import (
        finalize_find_continuum_reductions,
        find_continuum_cube_reductions,
        find_continuum_first_round_requests,
    )

    kwargs, selection, _, snr, (_, dirty_cube_reads) = CASA_RESULTS["line_set"]
    store = _write_cube(
        tmp_path / "line_set.img.zarr", make_find_continuum_image_xds(**kwargs)
    )
    img_xds = xr.open_zarr(store, chunks=None)
    frequency = img_xds.frequency.values
    requests = find_continuum_first_round_requests(
        frequency.size,
        float(frequency[0]),
        float(frequency[1] - frequency[0]),
        float(img_xds.frequency.attrs["rest_frequency"]["data"]),
        "sky",
        "primary_beam",
    )
    variables = {"sky": "SKY_RESIDUAL", "primary_beam": "PRIMARY_BEAM"}
    reductions = finalize_find_continuum_reductions(
        find_continuum_cube_reductions(
            img_xds[list(variables.values())].sel(polarization=["I"]).load(),
            requests,
            variables,
            {},
            np.arange(frequency.size),
            "PRIMARY_BEAM",
            0.2,
        ),
        requests,
    )
    first_round = {"primary_beam_limit": 0.2, "reductions": reductions}

    result = find_continuum_fused(
        input_image_store=store,
        work_directory=str(tmp_path / "work"),
        first_round_reductions=first_round,
        **_flat_atmosphere(480),
    )
    assert result["channel_selection"] == selection
    assert result["moment_difference_snr"] == pytest.approx(snr, abs=1e-4)
    assert result["cube_reads"]["SKY_RESIDUAL"] == dirty_cube_reads - 1


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


@pytest.mark.parametrize("n_chunks", [None, 3])
def test_profiling_and_report(tmp_path, n_chunks):
    """The timing, resource series and report of image_cube_single_field."""
    pytest.importorskip("psutil")
    from astroviper.distributed_applications.imaging.find_continuum.find_continuum import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES,
    )

    store = _write_cube(
        tmp_path / "line_set.img.zarr", make_find_continuum_image_xds(lines=LINE_SET)
    )
    result = find_continuum_fused(
        input_image_store=store,
        work_directory=str(tmp_path / "work"),
        n_mapping_parallelism=None if n_chunks is None else {"frequency": n_chunks},
        monitor_resources_seconds=0.05,
        report="html",
        **_flat_atmosphere(480),
    )
    assert os.path.isfile(result["report"])
    assert result["report"].startswith(str(tmp_path / "work"))
    timing = result["timing_distributed_application"]
    for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES:
        for _, key in steps:
            assert timing[key] >= 0.0
    assert timing["T_total"] == pytest.approx(
        sum(
            timing[key]
            for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES
            for _, key in steps
        ),
        rel=0.05,
    )
    assert timing["compute_start_unixtime"] <= timing["compute_end_unixtime"]
    passes = result["passes"]
    assert list(passes["pass"]) == list(range(result["n_cube_reductions"]))
    assert passes["T_compute_dask_graph"].sum() == pytest.approx(
        timing["T_compute_dask_graph"]
    )
    assert {"cpu_percent", "memory_rss_bytes"} <= set(result["timing_node_tasks"])
    n_reduce_nodes = 0 if n_chunks is None else 2 * result["n_cube_reductions"]
    assert len(result["timing_reduce_nodes"]) == n_reduce_nodes
