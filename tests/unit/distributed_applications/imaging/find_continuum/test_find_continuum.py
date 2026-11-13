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
