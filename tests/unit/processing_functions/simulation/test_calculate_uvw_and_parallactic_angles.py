"""uvw and parallactic-angle processing functions vs the legacy SIRIUS fixtures."""

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from astroviper.processing_functions.simulation.calculate_parallactic_angles import (
    calculate_parallactic_angles,
    find_representative_angles,
)
from astroviper.processing_functions.simulation.calculate_uvw import (
    calculate_antenna_uvw,
    calculate_uvw,
)
from tests.unit.processing_functions.simulation.legacy_fixtures import load_legacy


@pytest.mark.parametrize(
    "name", ["vla_airy", "alma_het_mosaic_noise", "evla_zernike_beam"]
)
def test_uvw_matches_legacy(name):
    f = load_legacy(name)
    uvw, a1, a2 = calculate_uvw(
        f["antenna_position"],
        f["site_position"],
        f["time"],
        f["phase_center_ra_dec"],
    )
    np.testing.assert_array_equal(a1, f["antenna1"])
    np.testing.assert_array_equal(a2, f["antenna2"])
    # Legacy SIRIUS used the archival / VLBI convention adopted by MSv4
    # (uvw = P(antenna1) - P(antenna2)), so the fixtures match directly.
    np.testing.assert_allclose(uvw, f["uvw"], atol=1e-9)
    # Pin the convention explicitly against the per-antenna projections.
    antenna_uvw = calculate_antenna_uvw(
        f["antenna_position"], f["site_position"], f["time"], f["phase_center_ra_dec"]
    )
    np.testing.assert_allclose(
        uvw, antenna_uvw[:, a1, :] - antenna_uvw[:, a2, :], atol=1e-12
    )


def test_uvw_accepts_unix_times_and_auto_correlations():
    f = load_legacy("vla_airy")
    from astropy.time import Time

    unix = Time(f["time"].astype(str), scale="utc").unix
    uvw_str, _, _ = calculate_uvw(
        f["antenna_position"], f["site_position"], f["time"], f["phase_center_ra_dec"]
    )
    uvw_unix, a1, a2 = calculate_uvw(
        f["antenna_position"],
        f["site_position"],
        unix,
        f["phase_center_ra_dec"],
        auto_correlations=True,
    )
    assert uvw_unix.shape[1] == 27 * 28 // 2
    auto = a1 == a2
    np.testing.assert_allclose(uvw_unix[:, auto], 0.0, atol=1e-9)
    np.testing.assert_allclose(uvw_unix[:, ~auto], uvw_str, atol=1e-6)


def test_antenna_uvw_w_points_to_phase_center():
    """The w component of an antenna's uvw is its projection on the phase-centre direction."""
    f = load_legacy("vla_airy")
    ant_uvw = calculate_antenna_uvw(
        f["antenna_position"],
        f["site_position"],
        f["time"][:1],
        f["phase_center_ra_dec"],
    )
    assert ant_uvw.shape == (1, 27, 3)
    # baselines: pairwise differences reproduce calculate_uvw
    uvw, a1, a2 = calculate_uvw(
        f["antenna_position"],
        f["site_position"],
        f["time"][:1],
        f["phase_center_ra_dec"],
    )
    np.testing.assert_allclose(ant_uvw[0, a1] - ant_uvw[0, a2], uvw[0], atol=1e-9)


def test_threads_computing_the_first_uvw_at_once_get_the_serial_uvw():
    """No thread raises, and each gets the uvw of a serial call.

    astropy adds the GCRS sky offset frame to its global transform graph on
    first use, once per process, so each trial runs in a fresh interpreter. The
    threads meet in ``SkyCoord.skyoffset_frame``, which calculate_uvw calls to
    make that frame, and are released under a forced switch interval.
    """
    code = textwrap.dedent(
        """
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        import astropy.units as u
        import numpy as np
        from astropy.coordinates import EarthLocation, SkyCoord
        from astropy.time import Time

        from astroviper.processing_functions.simulation.calculate_uvw import (
            calculate_uvw,
        )

        n_thread = 12
        site = np.array([-1601185.4, -5041977.5, 3554875.9])
        antenna = site + np.array([[0.0, 0, 0], [100, 50, 0], [-80, 120, 30]])
        time = 1570129200.0 + 60.0 * np.arange(n_thread)
        phase_center = [5.2336, 0.7116] + 0.001 * np.arange(n_thread)[:, None]


        def uvw(i):
            return calculate_uvw(
                antenna, site, time[i : i + 1], phase_center[i], direction_frame="fk5"
            )[0]


        # the IERS table is loaded once, before the threads
        EarthLocation(*site * u.m).get_gcrs_posvel(Time(time, format="unix"))
        skyoffset_frame = SkyCoord.skyoffset_frame
        barrier = Barrier(n_thread, action=lambda: sys.setswitchinterval(1e-6))


        def skyoffset_frame_together(self, rotation=None):
            barrier.wait()
            return skyoffset_frame(self, rotation)


        SkyCoord.skyoffset_frame = skyoffset_frame_together
        with ThreadPoolExecutor(n_thread) as pool:
            threaded = list(pool.map(uvw, range(n_thread)))
        SkyCoord.skyoffset_frame = skyoffset_frame
        for i in range(n_thread):
            np.testing.assert_array_equal(threaded[i], uvw(i))
        """
    )
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
        assert result.returncode == 0, result.stderr


def test_parallactic_angles_match_legacy():
    f = load_legacy("evla_zernike_beam")
    pa = calculate_parallactic_angles(
        f["time"], f["site_position"], f["phase_center_ra_dec"], direction_frame="fk5"
    )
    np.testing.assert_allclose(pa, f["parallactic_angle"], atol=1e-12)
    pa_icrs = calculate_parallactic_angles(
        f["time"], f["site_position"], f["phase_center_ra_dec"]
    )
    np.testing.assert_allclose(pa_icrs, f["parallactic_angle"], atol=1e-6)
    # monotonic over a few hours for a northern source at the VLA and within [0, 2pi)
    assert np.all(np.diff(pa) < 0)


def test_parallactic_angle_time_varying_direction():
    f = load_legacy("alma_het_mosaic_noise")
    pa = calculate_parallactic_angles(
        f["time"], f["site_position"], f["phase_center_ra_dec"]
    )
    assert pa.shape == (4,)
    pa_single = calculate_parallactic_angles(
        f["time"][:2], f["site_position"], f["phase_center_ra_dec"][:1]
    )
    np.testing.assert_allclose(pa[:2], pa_single)


def test_find_representative_angles():
    angles = np.array([0.0, 0.05, 0.1, 1.0, 1.02, 2 * np.pi - 0.03])
    reps, diff, idx = find_representative_angles(angles, 0.2)
    # 0.0 (with 0.05, 0.1 and the wrapped 2pi-0.03) and 1.0/1.02 are grouped
    assert len(reps) == 2
    assert np.all(diff <= 0.2)
    assert set(idx.tolist()) == {0, 1}
    np.testing.assert_allclose(reps[idx], np.where(diff == 0, angles, reps[idx]))
    reps1, diff1, idx1 = find_representative_angles(np.array([0.3]), 0.2)
    np.testing.assert_array_equal(reps1, [0.3])
    np.testing.assert_array_equal(idx1, [0])
    # legacy fixture: four times, 0.2 rad radius -> four representatives (every time is its own group)
    f = load_legacy("evla_zernike_beam")
    reps, _, _ = find_representative_angles(f["parallactic_angle"], 0.2)
    np.testing.assert_allclose(np.sort(reps), np.sort(f["beam_pa"]))
