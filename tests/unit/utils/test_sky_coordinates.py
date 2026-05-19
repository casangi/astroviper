"""Tests for public sky-coordinate utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from astroviper.utils.sky_coordinates import (
    coerce_angle_to_radians,
    frame_prefers_hourangle,
    is_scalar_number,
    parse_sky_center_to_radians,
    skycoord_to_lm_from_wcs,
)


def test_is_scalar_number_rejects_strings():
    """String-like coordinate tokens should not be treated as numeric scalars."""
    assert is_scalar_number(3.0)
    assert not is_scalar_number("03:14:15")


def test_frame_prefers_hourangle_for_equatorial_frames_only():
    """Equatorial frames should default sexagesimal longitudes to hour angle."""
    assert frame_prefers_hourangle("icrs")
    assert not frame_prefers_hourangle("galactic")


def test_parse_sky_center_to_radians_is_frame_aware():
    """Sexagesimal longitude parsing should follow the frame convention."""
    eq_lon, eq_lat = parse_sky_center_to_radians("12:00:00", "30:00:00", "icrs")
    gal_lon, gal_lat = parse_sky_center_to_radians("12:00:00", "30:00:00", "galactic")

    np.testing.assert_allclose(eq_lon, np.pi, atol=1e-12)
    np.testing.assert_allclose(eq_lat, np.deg2rad(30.0), atol=1e-12)
    np.testing.assert_allclose(gal_lon, np.deg2rad(12.0), atol=1e-12)
    np.testing.assert_allclose(gal_lat, np.deg2rad(30.0), atol=1e-12)


def test_parse_sky_center_to_radians_preserves_mixed_numeric_radians():
    """Mixed string/numeric pairs should not reinterpret numeric radians."""
    lon_rad, lat_rad = parse_sky_center_to_radians("12:00:00", 0.5, "icrs")
    lon_rad_2, lat_rad_2 = parse_sky_center_to_radians(1.0, "30:00:00", "icrs")

    np.testing.assert_allclose(lon_rad, np.pi, atol=1e-12)
    np.testing.assert_allclose(lat_rad, 0.5, atol=0.0)
    np.testing.assert_allclose(lon_rad_2, 1.0, atol=0.0)
    np.testing.assert_allclose(lat_rad_2, np.deg2rad(30.0), atol=1e-12)


def test_coerce_angle_to_radians_preserves_numeric_radians():
    """Plain numeric values should be treated as radians."""
    np.testing.assert_allclose(coerce_angle_to_radians(0.25), 0.25, atol=0.0)


def test_coerce_angle_to_radians_accepts_explicit_unit_strings():
    """Explicit-unit angle strings should take the direct Astropy parsing path."""
    np.testing.assert_allclose(
        coerce_angle_to_radians("180deg"),
        np.pi,
        atol=1e-12,
    )


def test_coerce_angle_to_radians_falls_back_for_ambiguous_sexagesimal_strings():
    """Ambiguous sexagesimal strings should honor the caller's default unit choice."""
    np.testing.assert_allclose(
        coerce_angle_to_radians("12:00:00", prefer_hourangle=True),
        np.pi,
        atol=1e-12,
    )


def test_parse_sky_center_to_radians_falls_back_after_paired_parse_failure():
    """The independent-angle fallback should run when SkyCoord rejects the pair."""
    lon_rad, lat_rad = parse_sky_center_to_radians("12:00:00", "91deg", "icrs")

    np.testing.assert_allclose(lon_rad, np.pi, atol=1e-12)
    np.testing.assert_allclose(lat_rad, np.deg2rad(91.0), atol=1e-12)


def test_skycoord_to_lm_from_wcs_maps_phase_center_to_origin():
    """The phase center should map to native ``(l, m) = (0, 0)``."""
    l_val, m_val = skycoord_to_lm_from_wcs(1.0, 0.5, (1.0, 0.5), "SIN")

    np.testing.assert_allclose(l_val, 0.0, atol=1e-12)
    np.testing.assert_allclose(m_val, 0.0, atol=1e-12)


def test_skycoord_to_lm_from_wcs_warns_for_non_sin_projection():
    """Unsupported projections should warn while reusing the SIN conversion."""
    with pytest.warns(UserWarning, match="falling back to SIN projection"):
        l_val, m_val = skycoord_to_lm_from_wcs(1.0, 0.5, (1.0, 0.5), "TAN")

    np.testing.assert_allclose(l_val, 0.0, atol=1e-12)
    np.testing.assert_allclose(m_val, 0.0, atol=1e-12)
