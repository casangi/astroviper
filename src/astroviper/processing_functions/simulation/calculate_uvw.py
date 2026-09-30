"""Baseline ``uvw`` coordinates of a simulated observation (astropy)."""

from __future__ import annotations

import numpy as np

from astroviper.utils.measurement_set_tools import baseline_antenna_pairs


def calculate_antenna_uvw(
    antenna_position: np.ndarray,
    site_position: np.ndarray,
    time: np.ndarray,
    phase_center_ra_dec: np.ndarray,
    direction_frame: str = "icrs",
) -> np.ndarray:
    """Per-antenna ``(u, v, w)`` (metres) for each time.

    The antenna ITRF positions are transformed to GCRS with the array centre as
    observer and then rotated into the sky-offset frame of the phase centre, so
    that ``w`` points towards the phase centre and ``u``/``v`` lie in the plane
    perpendicular to it (east / north).  This is the SIRIUS ``_calc_uvw_astropy``
    algorithm.

    Parameters
    ----------
    antenna_position : np.ndarray, [n_antenna, 3], metres
        ITRF geocentric antenna positions.
    site_position : np.ndarray, [3], metres
        ITRF geocentric array reference position (observer).
    time : np.ndarray, [n_time]
        UTC times, either ISO strings ``YYYY-MM-DDTHH:MM:SS.SSS`` or unix seconds.
    phase_center_ra_dec : np.ndarray, [n_time | 1, 2], radians
        Phase centre(s); one per time or a single one.
    direction_frame : str
        Astropy frame of ``phase_center_ra_dec`` (``"icrs"`` or ``"fk5"``).

    Returns
    -------
    np.ndarray, [n_time, n_antenna, 3]
        Antenna ``(u, v, w)`` in metres.
    """
    import astropy.coordinates as coord
    import astropy.units as u
    from astropy.time import Time

    antenna_position = np.asarray(antenna_position, dtype=np.float64)
    n_antenna = antenna_position.shape[0]
    time = np.asarray(time)
    n_time = time.shape[0]
    phase_center_ra_dec = np.asarray(phase_center_ra_dec, dtype=np.float64).reshape(
        -1, 2
    )
    phase_center_per_time = np.broadcast_to(phase_center_ra_dec, (n_time, 2))

    if np.issubdtype(time.dtype, np.number):
        astropy_time = Time(time.astype(np.float64), format="unix", scale="utc")
    else:
        astropy_time = Time(time.astype(str), scale="utc")

    site = coord.EarthLocation(
        x=site_position[0] * u.m, y=site_position[1] * u.m, z=site_position[2] * u.m
    )
    antenna_location = coord.EarthLocation(
        x=antenna_position[:, 0] * u.m,
        y=antenna_position[:, 1] * u.m,
        z=antenna_position[:, 2] * u.m,
    )

    antenna_uvw = np.empty((n_time, n_antenna, 3), dtype=np.float64)
    # one astropy transform per distinct phase centre (vectorised over its times)
    unique_pc, inverse = np.unique(phase_center_per_time, axis=0, return_inverse=True)
    inverse = np.ravel(inverse)
    for i_pc, pc in enumerate(unique_pc):
        i_time = np.where(inverse == i_pc)[0]
        obstime = astropy_time[i_time][:, None]  # [n_t, 1] broadcast against antennas
        site_p, site_v = site.get_gcrs_posvel(obstime)
        antenna_gcrs = coord.GCRS(
            antenna_location.get_gcrs_posvel(obstime)[0],
            obstime=obstime,
            obsgeoloc=site_p,
            obsgeovel=site_v,
        )
        phase_center = coord.SkyCoord(
            pc[0] * u.rad, pc[1] * u.rad, frame=direction_frame
        )
        frame_uvw = phase_center.transform_to(antenna_gcrs).skyoffset_frame()
        cart = antenna_gcrs.transform_to(frame_uvw).cartesian
        antenna_uvw[i_time, :, 0] = cart.y.to_value(u.m)
        antenna_uvw[i_time, :, 1] = cart.z.to_value(u.m)
        antenna_uvw[i_time, :, 2] = cart.x.to_value(u.m)
    return antenna_uvw


def calculate_uvw(
    antenna_position: np.ndarray,
    site_position: np.ndarray,
    time: np.ndarray,
    phase_center_ra_dec: np.ndarray,
    auto_correlations: bool = False,
    direction_frame: str = "icrs",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Baseline ``uvw`` coordinates (metres) for every time.

    Follows the archival / VLBI convention adopted by MSv4: the ``uvw`` of the
    baseline ``(antenna1, antenna2)`` is the difference of the per-antenna
    projections ``P(antenna1) - P(antenna2)``, which is what observatory-written
    measurement sets (ALMA/VLA via CASA), VLBI correlators (CALC/DiFX) and AIPS
    contain.  The visibility phase sign pairs with this choice in
    :func:`~astroviper.processing_functions.simulation.calculate_visibilities`.

    Parameters
    ----------
    antenna_position : np.ndarray, [n_antenna, 3], metres
        ITRF geocentric antenna positions.
    site_position : np.ndarray, [3], metres
        ITRF geocentric array reference position.
    time : np.ndarray, [n_time]
        UTC times (ISO strings or unix seconds).
    phase_center_ra_dec : np.ndarray, [n_time | 1, 2], radians
    auto_correlations : bool
        Include autocorrelation baselines.
    direction_frame : str
        Astropy frame of ``phase_center_ra_dec``.

    Returns
    -------
    uvw : np.ndarray, [n_time, n_baseline, 3], metres
    antenna1, antenna2 : np.ndarray, [n_baseline] int
        Antenna indices of each baseline (``antenna1 <= antenna2``).

    Notes
    -----
    The historical MSv2 definition text describes the opposite direction
    (``POSITION2 - POSITION1``); decades of practice -- and therefore archives
    and everything converted from them -- realise ``POSITION1 - POSITION2``
    (see ``experiments/uvw_convention_investigation``).  AstroVIPER and the
    MSv4 schema align with practice.
    """
    antenna1, antenna2 = baseline_antenna_pairs(
        np.shape(antenna_position)[0], auto_correlations
    )
    antenna_uvw = calculate_antenna_uvw(
        antenna_position, site_position, time, phase_center_ra_dec, direction_frame
    )
    uvw = antenna_uvw[:, antenna1, :] - antenna_uvw[:, antenna2, :]
    return np.ascontiguousarray(uvw), antenna1, antenna2
