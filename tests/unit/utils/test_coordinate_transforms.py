"""Tests for astroviper.utils.coordinate_transforms (ported from SIRIUS)."""

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from astroviper.utils.coordinate_transforms import (
    calculate_uvw_rotation,
    celestial_coord_to_sin_pixel,
    directional_cosines,
    inverse_sin_project,
    make_rotated_grid,
    rotate_coordinates,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    sin_pixel_to_celestial_coord,
    sin_project,
    tan_project,
    wrapped_angle_difference,
)

ARCSEC_TO_RAD = np.pi / 180 / 3600
DEG_TO_RAD = np.pi / 180


def _sin_pixel_to_celestial_coord_astropy(ra_dec_center, image_size, cell_size, pixel):
    rad_to_deg = 180 / np.pi
    w = WCS(naxis=2)
    w.wcs.crpix = np.array(image_size) // 2
    w.wcs.cdelt = np.array(cell_size) * rad_to_deg
    w.wcs.crval = np.array(ra_dec_center) * rad_to_deg
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    ra, dec = w.wcs_pix2world(pixel[:, 0], pixel[:, 1], 1)
    return np.vstack([ra / rad_to_deg, dec / rad_to_deg]).T


def test_sin_pixel_to_celestial_coord_matches_astropy_wcs():
    """Legacy SIRIUS test: analytic SIN de-projection vs astropy WCS (dec=90 excluded)."""
    image_size = np.array([24000, 24000])
    cell_size = np.array([-0.0005, 0.0005]) * ARCSEC_TO_RAD
    centers = (
        np.array(
            [
                [0.0, 0.0],
                [0.0, 8.0],
                [0.0, 22.5],
                [0.0, 45.0],
                [0.0, 67.5],
                [0.0, 81.0],
                [0.0, 89],
            ]
        )
        * DEG_TO_RAD
    )
    pixels = np.array(
        [
            [12000, 12000],
            [7000, 12000],
            [17000, 12000],
            [12000, 17000],
            [12000, 7000],
            [8464, 8464],
            [15536, 15536],
            [8464, 15536],
            [15536, 8464],
            [2000, 12000],
            [22000, 12000],
            [12000, 22000],
            [12000, 2000],
            [4929, 4929],
            [19071, 19071],
            [4929, 19071],
            [19071, 4929],
        ]
    )
    for ra_dec in centers:
        ours = sin_pixel_to_celestial_coord(ra_dec, image_size, cell_size, pixels)
        theirs = _sin_pixel_to_celestial_coord_astropy(
            ra_dec, image_size, cell_size, pixels
        )
        assert np.max(wrapped_angle_difference(ours, theirs)) < 1e-14
        # and back
        back = celestial_coord_to_sin_pixel(ra_dec, image_size, cell_size, ours)
        np.testing.assert_allclose(back, pixels, atol=1e-6)


def test_rotate_coordinates_legacy_values():
    np.testing.assert_allclose(
        rotate_coordinates(1.0, 2.0, 2.0), (1.402448017104221, -1.7415910999199666)
    )


def test_make_rotated_grid_legacy_values():
    x_expected = np.array(
        [
            [-1.97260236, -0.15400751, 1.66458735, 3.4831822, 5.30177705],
            [-2.80489603, -0.98630118, 0.83229367, 2.65088853, 4.46948338],
            [-3.63718971, -1.81859485, 0.0, 1.81859485, 3.63718971],
            [-4.46948338, -2.65088853, -0.83229367, 0.98630118, 2.80489603],
            [-5.30177705, -3.4831822, -1.66458735, 0.15400751, 1.97260236],
        ]
    )
    y_expected = np.array(
        [
            [5.30177705, 4.46948338, 3.63718971, 2.80489603, 1.97260236],
            [3.4831822, 2.65088853, 1.81859485, 0.98630118, 0.15400751],
            [1.66458735, 0.83229367, -0.0, -0.83229367, -1.66458735],
            [-0.15400751, -0.98630118, -1.81859485, -2.65088853, -3.4831822],
            [-1.97260236, -2.80489603, -3.63718971, -4.46948338, -5.30177705],
        ]
    )
    x, y = make_rotated_grid(np.array([5, 5]), np.array([2, 2]), 2)
    np.testing.assert_allclose(x, x_expected, atol=1e-8)
    np.testing.assert_allclose(y, y_expected, atol=1e-8)
    x0, y0 = make_rotated_grid(np.array([5, 5]), np.array([2, 2]), 0)
    np.testing.assert_allclose(x0[:, 0], [-4, -2, 0, 2, 4])
    np.testing.assert_allclose(y0[0, :], [-4, -2, 0, 2, 4])


@pytest.fixture()
def phase_center_and_source():
    pc = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
    src = SkyCoord(ra="19h59m50.51793355s", dec="+40d48m11.3694551s", frame="fk5")
    return np.array([pc.ra.rad, pc.dec.rad]), np.array([src.ra.rad, src.dec.rad])


def test_sin_project_legacy_value(phase_center_and_source):
    pc, src = phase_center_and_source
    lm = sin_project(pc, src[None, :])[0]
    np.testing.assert_allclose(lm, [0.00121203, 0.00121203], atol=1e-8)
    # vectorised over several positions and round trip
    srcs = np.stack([src, pc, src + 1e-3])
    lm = sin_project(pc, srcs)
    assert lm.shape == (3, 2)
    np.testing.assert_allclose(lm[1], [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(inverse_sin_project(pc, lm), srcs, atol=1e-12)


def test_tan_project_close_to_sin_for_small_offsets(phase_center_and_source):
    pc, src = phase_center_and_source
    np.testing.assert_allclose(tan_project(pc, src), sin_project(pc, src), rtol=1e-5)


def test_rotation_matrices_are_orthonormal():
    for mat in (
        rotation_matrix_x(0.3),
        rotation_matrix_y(-1.1),
        rotation_matrix_z(2.0),
    ):
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-15)
        assert np.isclose(np.linalg.det(mat), 1.0)


def test_directional_cosines_unit_vectors():
    ra_dec = np.array([[0.0, 0.0], [np.pi / 2, 0.0], [0.0, np.pi / 2], [1.0, -0.3]])
    lmn = directional_cosines(ra_dec)
    np.testing.assert_allclose(np.linalg.norm(lmn, axis=1), 1.0)
    np.testing.assert_allclose(lmn[1], [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(lmn[2], [0, 0, 1], atol=1e-15)


def test_calculate_uvw_rotation_source_at_phase_center(phase_center_and_source):
    pc, src = phase_center_and_source
    rot, lmn_rot = calculate_uvw_rotation(pc, pc)
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(lmn_rot, 0.0, atol=1e-15)
    rot, lmn_rot = calculate_uvw_rotation(pc, src)
    # small offset: |(l, m)| of lmn_rot equals the angular separation; lmn_rot[2] = 1 - n
    lm = sin_project(pc, src)
    np.testing.assert_allclose(
        np.linalg.norm(lmn_rot[:2]), np.linalg.norm(lm), rtol=1e-6
    )
    np.testing.assert_allclose(lmn_rot[2], 1 - np.sqrt(1 - np.sum(lm**2)), rtol=1e-6)


def test_wrapped_angle_difference():
    np.testing.assert_allclose(wrapped_angle_difference(0.1, 2 * np.pi - 0.1), 0.2)
    np.testing.assert_allclose(wrapped_angle_difference(np.pi, -np.pi), 0.0, atol=1e-15)
