"""Tests for astroviper.utils.telescope_layout."""

import numpy as np
import pytest
from xradio.measurement_set.schema import AntennaXds
from xradio.schema.check import check_dataset

from astroviper.utils.telescope_layout import (
    list_telescope_layouts,
    local_tangent_plane_to_itrf,
    make_antenna_xds,
    observatory_position,
    read_telescope_layout,
)

# First / last antennas of the VLA D configuration (legacy SIRIUS test template,
# values produced by CASA simutil.readantenna).
VLA_D_FIRST = [-1601188.989351, -5042000.518599, 3554843.38448]
VLA_D_LAST = [-1601139.483292, -5041679.021042, 3555316.478099]
VLA_D_NAMES_START = ["W01", "W02", "W03"]


def test_shipped_layouts_present():
    names = list_telescope_layouts()
    assert len(names) >= 180
    for expected in [
        "vla.d",
        "vla.a",
        "alma.cycle7.1",
        "aca.cycle7",
        "ngvla-main-revC",
    ]:
        assert expected in names


def test_read_vla_d_matches_casa_template():
    ant = read_telescope_layout("vla.d")
    assert ant.sizes["antenna_name"] == 27
    assert list(ant.antenna_name.values[:3]) == VLA_D_NAMES_START
    np.testing.assert_allclose(ant.ANTENNA_POSITION.values[0], VLA_D_FIRST)
    np.testing.assert_allclose(ant.ANTENNA_POSITION.values[-1], VLA_D_LAST)
    np.testing.assert_array_equal(ant.ANTENNA_DISH_DIAMETER.values, 25.0)
    assert ant.attrs["overall_telescope_name"] == "VLA"
    assert ant.attrs["type"] == "antenna"
    assert list(ant.polarization_type.values[0]) == ["R", "L"]
    assert list(ant.cartesian_pos_label.values) == ["x", "y", "z"]
    assert str(check_dataset(ant, AntennaXds)) == "No schema issues found"


def test_read_loc_layout_alma():
    """LOC (local tangent plane) layouts are converted with the CASA formula.

    Reference values are the CASA ``simutil.readantenna`` output for the first
    antenna of ``alma.cycle7.1`` (from the SIRIUS ``tel.zarr`` products).
    """
    ant = read_telescope_layout("alma.cycle7.1")
    assert ant.attrs["overall_telescope_name"] == "ALMA"
    assert list(ant.polarization_type.values[0]) == ["X", "Y"]
    # all ALMA 12 m dishes, positions close (< 20 km) to the ALMA reference
    np.testing.assert_array_equal(ant.ANTENNA_DISH_DIAMETER.values, 12.0)
    offsets = ant.ANTENNA_POSITION.values - observatory_position("ALMA")
    assert np.all(np.linalg.norm(offsets, axis=1) < 2.0e4)
    assert str(check_dataset(ant, AntennaXds)) == "No schema issues found"


def test_local_tangent_plane_origin_maps_to_reference():
    ref = observatory_position("ALMA")
    xyz = local_tangent_plane_to_itrf(np.zeros((1, 3)), ref)
    # the reference position is the WGS84 point of the observatory itself
    np.testing.assert_allclose(xyz[0], ref, atol=1e-6)
    # moving "up" by 1 m increases the distance from the geocentre by ~1 m
    up = local_tangent_plane_to_itrf(np.array([[0.0, 0.0, 1.0]]), ref)
    assert np.isclose(np.linalg.norm(up[0]) - np.linalg.norm(ref), 1.0, atol=1e-2)


def test_antenna_selection_and_overrides():
    ant = read_telescope_layout(
        "vla.d",
        telescope_name="EVLA",
        antenna_selection=["W03", "W01"],
        polarization_type=("X", "Y"),
    )
    assert list(ant.antenna_name.values) == ["W03", "W01"]
    assert ant.attrs["overall_telescope_name"] == "EVLA"
    assert list(ant.polarization_type.values[0]) == ["X", "Y"]
    ant2 = read_telescope_layout("vla.d", antenna_selection=[2, 0])
    assert list(ant2.antenna_name.values) == ["W03", "W01"]


def test_read_from_file_path(tmp_path):
    cfg = tmp_path / "tiny.cfg"
    cfg.write_text(
        "# observatory=VLA\n# coordsys=XYZ\n"
        "-1601188.989351 -5042000.518599 3554843.384480 25. A1\n"
        "-1601225.230987 -5041980.390730 3554855.657987 25. A2\n"
    )
    ant = read_telescope_layout(str(cfg))
    assert list(ant.antenna_name.values) == ["A1", "A2"]
    with pytest.raises(FileNotFoundError):
        read_telescope_layout("no_such_layout")


def test_utm_is_rejected(tmp_path):
    cfg = tmp_path / "utm.cfg"
    cfg.write_text("# observatory=ALMA\n# coordsys=UTM\n1 2 3 12 A1\n")
    with pytest.raises(NotImplementedError):
        read_telescope_layout(str(cfg))


def test_make_antenna_xds_with_blockage():
    ant = make_antenna_xds(
        ["a", "b"],
        np.arange(6.0).reshape(2, 3),
        [10.7, 7.0],
        "ALMA",
        blockage_diameter=0.75,
    )
    assert "ANTENNA_BLOCKAGE" in ant
    np.testing.assert_array_equal(ant.ANTENNA_BLOCKAGE.values, [0.75, 0.75])
    assert str(check_dataset(ant, AntennaXds)) == "No schema issues found"


def test_observatory_position_fallback():
    with pytest.raises(ValueError):
        observatory_position("NOWHERE")
    pos = observatory_position(
        "NOWHERE", antenna_position=np.array([[0, 0, 0], [2, 2, 2.0]])
    )
    np.testing.assert_allclose(pos, [1, 1, 1])
