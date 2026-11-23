import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import EarthLocation
from astropy.time import Time

from astroviper.utils.direction_frame_conversion import (
    convert_direction_frame,
    interpolate_direction_to_times,
)

ALMA = EarthLocation.from_geodetic(
    lon=-67.7549 * u.deg, lat=-23.0293 * u.deg, height=5058.7 * u.m
)

GC_ICRS = (266.40500, -28.93617)
NGP_ICRS = (192.85948, 27.12825)


def _times(n, step_s=600.0):
    """n UTC timestamps, step_s seconds apart."""
    return Time("2020-01-01T00:00:00", scale="utc") + np.arange(n) * step_s * u.s


def _sample_icrs(n):
    """A spread of ICRS directions avoiding the poles."""
    ra = np.linspace(5.0, 355.0, n)
    dec = np.linspace(-60.0, 70.0, n)
    return ra, dec


def _angle_diff(a, b):
    """Smallest signed difference between two angles in degrees."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def test_simple():
    assert True


class TestInputValidation:
    """Argument checks in convert_direction_frame"""

    def test_unknown_input_frame(self):
        with pytest.raises(ValueError, match="Unknown input frame"):
            convert_direction_frame(
                np.array([0.0]), np.array([0.0]), "FOO", "ICRS", _times(1)
            )

    def test_unknown_output_frame(self):
        with pytest.raises(ValueError, match="Unknown output frame"):
            convert_direction_frame(
                np.array([0.0]), np.array([0.0]), "ICRS", "FOO", _times(1)
            )

    def test_same_frame(self):
        with pytest.raises(ValueError, match="must differ"):
            convert_direction_frame(
                np.array([0.0]), np.array([0.0]), "ICRS", "icrs", _times(1)
            )

    @pytest.mark.parametrize("frame", ["ALTAZ", "HADEC"])
    def test_location_required(self, frame):
        with pytest.raises(ValueError, match="location is required"):
            convert_direction_frame(
                np.array([0.0]), np.array([0.0]), "ICRS", frame, _times(1)
            )

    def test_frame_names_case_insensitive(self):
        _, _, out_frame = convert_direction_frame(
            np.array([GC_ICRS[0]]),
            np.array([GC_ICRS[1]]),
            "icrs",
            "galactic",
            _times(1),
        )
        assert out_frame == "GALACTIC"


LAT = ALMA.lat.deg

# Where the north / south horizon points sit in HADEC depends on the hemisphere:
# the horizon point lies "below" the visible celestial pole (ha = 180)
# and on the near side of the hidden one (ha = 0).
NORTH_HORIZON_HADEC = (180.0 if LAT > 0 else 0.0, 90.0 - abs(LAT))
SOUTH_HORIZON_HADEC = (0.0 if LAT > 0 else 180.0, abs(LAT) - 90.0)

# (description, in_frame, (v0, v1) in, out_frame, (v0, v1) expected, atol [deg])
# Each point is defined exactly in one frame and has a known value in another.
KNOWN_POSITIONS = [
    # Galactic centre / pole, as defined by the IAU (1958) in B1950
    ("GC -> ICRS", "GALACTIC", (0.0, 0.0), "ICRS", GC_ICRS, 1e-3),
    ("GC -> FK5", "GALACTIC", (0.0, 0.0), "FK5", GC_ICRS, 1e-3),
    ("GC -> FK4", "GALACTIC", (0.0, 0.0), "FK4", (265.6108, -28.9168), 2e-3),
    ("ICRS -> GC", "ICRS", GC_ICRS, "GALACTIC", (0.0, 0.0), 1e-3),
    ("NGP -> ICRS", "GALACTIC", (0.0, 90.0), "ICRS", NGP_ICRS, 1e-3),
    ("NGP -> FK4", "GALACTIC", (0.0, 90.0), "FK4", (192.25, 27.4), 2e-3),
    # J2000 origin precessed back 50 years to B1950
    ("ICRS 0,0 -> FK4", "ICRS", (0.0, 0.0), "FK4", (359.36, -0.278), 1e-2),
    # ICRS and FK5 J2000 agree to tens of milliarcseconds
    ("ICRS 0,0 -> FK5", "ICRS", (0.0, 0.0), "FK5", (0.0, 0.0), 1e-4),
    # Horizon geometry: zenith and the four compass points
    ("zenith", "ALTAZ", (0.0, 90.0), "HADEC", (0.0, LAT), 1e-3),
    ("north horizon", "ALTAZ", (0.0, 0.0), "HADEC", NORTH_HORIZON_HADEC, 1e-3),
    ("south horizon", "ALTAZ", (180.0, 0.0), "HADEC", SOUTH_HORIZON_HADEC, 1e-3),
    ("east horizon", "ALTAZ", (90.0, 0.0), "HADEC", (-90.0, 0.0), 1e-3),
    ("west horizon", "ALTAZ", (-90.0, 0.0), "HADEC", (90.0, 0.0), 1e-3),
    ("meridian HA=0", "HADEC", (0.0, LAT), "ALTAZ", (0.0, 90.0), 1e-3),
]


class TestKnownPositions:
    """A point known exactly in one frame must land on its known value in another"""

    @pytest.mark.parametrize(
        "in_frame, v_in, out_frame, expected, atol",
        [case[1:] for case in KNOWN_POSITIONS],
        ids=[case[0] for case in KNOWN_POSITIONS],
    )
    def test_known_position(self, in_frame, v_in, out_frame, expected, atol):
        out0, out1, _ = convert_direction_frame(
            np.array([v_in[0]]),
            np.array([v_in[1]]),
            in_frame,
            out_frame,
            _times(1),
            location=ALMA,
        )
        # Skip the longitude check when the point is at a pole,
        # where longitude / azimuth is undefined.
        if abs(expected[1]) < 90.0 - 1e-6:
            np.testing.assert_allclose(_angle_diff(out0, expected[0]), 0.0, atol=atol)
        np.testing.assert_allclose(out1, expected[1], atol=atol)


class TestCelestialFrames:
    """Conversions that do not need an observer location"""

    def test_icrs_fk5_nearly_identical(self):
        """ICRS and FK5 J2000 differ by only tens of milliarcseconds."""
        ra, dec = _sample_icrs(20)
        ra5, dec5, _ = convert_direction_frame(ra, dec, "ICRS", "FK5", _times(20))
        np.testing.assert_allclose(_angle_diff(ra5, ra), 0.0, atol=1e-4)
        np.testing.assert_allclose(dec5, dec, atol=1e-4)

    def test_icrs_fk4_differs(self):
        """B1950 positions are visibly precessed away from ICRS."""
        ra, dec = _sample_icrs(20)
        ra4, dec4, _ = convert_direction_frame(ra, dec, "ICRS", "FK4", _times(20))
        sep = np.hypot(_angle_diff(ra4, ra) * np.cos(np.deg2rad(dec)), dec4 - dec)
        assert np.all(sep > 0.1)

    @pytest.mark.parametrize("frame", ["FK5", "FK4", "GALACTIC"])
    def test_round_trip(self, frame):
        ra, dec = _sample_icrs(20)
        t = _times(20)
        v0, v1, _ = convert_direction_frame(ra, dec, "ICRS", frame, t)
        ra2, dec2, _ = convert_direction_frame(v0, v1, frame, "ICRS", t)
        np.testing.assert_allclose(_angle_diff(ra2, ra), 0.0, atol=1e-8)
        np.testing.assert_allclose(dec2, dec, atol=1e-8)

    def test_output_shape_and_ranges(self):
        ra, dec = _sample_icrs(15)
        l, b, _ = convert_direction_frame(ra, dec, "ICRS", "GALACTIC", _times(15))
        assert l.shape == b.shape == (15,)
        assert np.all((l >= 0) & (l < 360))
        assert np.all((b >= -90) & (b <= 90))


class TestLocationDependentFrames:
    """ALTAZ and HADEC conversions"""

    @pytest.mark.parametrize("frame", ["ALTAZ", "HADEC"])
    def test_round_trip(self, frame):
        ra, dec = _sample_icrs(20)
        t = _times(20)
        v0, v1, _ = convert_direction_frame(ra, dec, "ICRS", frame, t, ALMA)
        ra2, dec2, _ = convert_direction_frame(v0, v1, frame, "ICRS", t, ALMA)
        # 1e-5 deg = 36 mas
        np.testing.assert_allclose(_angle_diff(ra2, ra), 0.0, atol=1e-5)
        np.testing.assert_allclose(dec2, dec, atol=1e-5)

    @pytest.mark.parametrize("frame", ["ALTAZ", "HADEC"])
    def test_first_coordinate_range(self, frame):
        """az / ha are normalised to (-180, 180] (MSv2 convention)."""
        ra, dec = _sample_icrs(50)
        v0, v1, _ = convert_direction_frame(ra, dec, "ICRS", frame, _times(50), ALMA)
        assert np.all((v0 > -180) & (v0 <= 180))
        assert np.all((v1 >= -90) & (v1 <= 90))

    def test_pole_elevation_equals_latitude(self):
        """The north celestial pole sits at elevation = site latitude
        (below the horizon for a southern site such as ALMA)."""
        n = 10
        _, el, _ = convert_direction_frame(
            np.zeros(n), np.full(n, 90.0), "ICRS", "ALTAZ", _times(n, 3600), ALMA
        )
        # ICRS pole is ~0.3 deg from the true pole of date (precession)
        np.testing.assert_allclose(el, ALMA.lat.deg, atol=0.5)

    def test_altaz_to_hadec_matches_via_icrs(self):
        ra, dec = _sample_icrs(10)
        t = _times(10)
        az, el, _ = convert_direction_frame(ra, dec, "ICRS", "ALTAZ", t, ALMA)
        ha_direct, d_direct, _ = convert_direction_frame(
            az, el, "ALTAZ", "HADEC", t, ALMA
        )
        ha_icrs, d_icrs, _ = convert_direction_frame(ra, dec, "ICRS", "HADEC", t, ALMA)
        np.testing.assert_allclose(_angle_diff(ha_direct, ha_icrs), 0.0, atol=1e-5)
        np.testing.assert_allclose(d_direct, d_icrs, atol=1e-5)

    def test_interpolate_matches_exact(self):
        """ErfaAstromInterpolator should agree with the full calculation."""
        n = 200
        ra, dec = np.full(n, 83.633), np.full(n, 22.0145)
        t = _times(n, step_s=10.0)
        az0, el0, _ = convert_direction_frame(
            ra, dec, "ICRS", "ALTAZ", t, ALMA, interpolate=False
        )
        az1, el1, _ = convert_direction_frame(
            ra, dec, "ICRS", "ALTAZ", t, ALMA, interpolate=True, time_resolution=300.0
        )
        # 1e-6 deg = 3.6 mas, far looser than the expected error
        np.testing.assert_allclose(_angle_diff(az1, az0), 0.0, atol=1e-6)
        np.testing.assert_allclose(el1, el0, atol=1e-6)


class TestInterpolateDirectionToTimes:
    """Tests for interpolate_direction_to_times"""

    def test_linear_motion(self):
        """A cubic spline reproduces linear motion exactly."""
        src = _times(6, 60)
        tgt = src[0] + np.linspace(0, 300, 11) * u.s
        ra = 10.0 + 0.1 * np.arange(6)
        dec = -5.0 + 0.2 * np.arange(6)
        ra_i, dec_i = interpolate_direction_to_times(ra, dec, src, tgt)
        frac = np.linspace(0, 5, 11)
        np.testing.assert_allclose(ra_i, 10.0 + 0.1 * frac, atol=1e-9)
        np.testing.assert_allclose(dec_i, -5.0 + 0.2 * frac, atol=1e-9)

    def test_ra_wraps_across_zero(self):
        """RA crossing 360 -> 0 must not swing back through 180."""
        src = _times(5, 60)
        ra = np.array([358.0, 359.0, 0.0, 1.0, 2.0])
        dec = np.zeros(5)
        tgt = src[0] + np.array([30.0, 90.0, 150.0, 210.0]) * u.s
        ra_i, _ = interpolate_direction_to_times(ra, dec, src, tgt)
        np.testing.assert_allclose(ra_i, [358.5, 359.5, 0.5, 1.5], atol=1e-9)
        assert np.all((ra_i >= 0) & (ra_i < 360))

    def test_same_times_is_identity(self):
        src = _times(5, 60)
        ra = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        dec = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
        ra_i, dec_i = interpolate_direction_to_times(ra, dec, src, src)
        np.testing.assert_allclose(ra_i, ra, atol=1e-9)
        np.testing.assert_allclose(dec_i, dec, atol=1e-9)

    def test_output_length(self):
        src = _times(5, 60)
        tgt = _times(17, 15)
        ra_i, dec_i = interpolate_direction_to_times(
            np.linspace(0, 4, 5), np.linspace(0, 4, 5), src, tgt
        )
        assert ra_i.shape == dec_i.shape == (17,)


if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
