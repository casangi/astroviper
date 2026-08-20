"""Parallactic angles of a simulated observation (astropy)."""

from __future__ import annotations

import numpy as np

from astroviper.utils.coordinate_transforms import wrapped_angle_difference


def calculate_parallactic_angles(
    time: np.ndarray,
    site_position: np.ndarray,
    direction_ra_dec: np.ndarray,
    direction_frame: str = "icrs",
) -> np.ndarray:
    """Parallactic angle of a sky direction seen from the array reference position.

    The direction and the celestial pole are transformed to a topocentric
    altitude-azimuth frame at ``site_position`` for each time; the parallactic
    angle is the position angle of the pole as seen from the direction
    (SIRIUS ``_calc_parallactic_angles_astropy``).

    Parameters
    ----------
    time : np.ndarray, [n_time]
        UTC times (ISO strings ``YYYY-MM-DDTHH:MM:SS.SSS`` or unix seconds).
    site_position : np.ndarray, [3], metres
        ITRF geocentric observing location.
    direction_ra_dec : np.ndarray, [n_time | 1, 2], radians
        Pointing / phase-centre direction per time (or a single direction).
    direction_frame : str
        Astropy frame of ``direction_ra_dec`` (``"icrs"`` or ``"fk5"``); the pole
        of the same frame is used as the reference.

    Returns
    -------
    np.ndarray, [n_time], radians
    """
    import astropy.coordinates as coord
    import astropy.units as u
    from astropy.time import Time

    time = np.asarray(time)
    n_time = time.shape[0]
    if np.issubdtype(time.dtype, np.number):
        obstime = Time(time.astype(np.float64), format="unix", scale="utc")
    else:
        obstime = Time(time.astype(str), scale="utc")

    direction = np.broadcast_to(
        np.asarray(direction_ra_dec, dtype=np.float64).reshape(-1, 2), (n_time, 2)
    )
    location = coord.EarthLocation.from_geocentric(
        x=site_position[0] * u.m, y=site_position[1] * u.m, z=site_position[2] * u.m
    )
    sky = coord.SkyCoord(
        ra=direction[:, 0] * u.rad, dec=direction[:, 1] * u.rad, frame=direction_frame
    )
    pole = coord.SkyCoord(0, 90, unit=u.deg, frame=direction_frame)
    altaz_frame = coord.AltAz(location=location, obstime=obstime)
    pole_altaz = pole.transform_to(altaz_frame)
    direction_altaz = sky.transform_to(altaz_frame)
    return np.asarray(
        direction_altaz.position_angle(pole_altaz).to_value(u.rad), dtype=np.float64
    )


def find_representative_angles(
    angles: np.ndarray, max_difference: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy subset of angles so every input lies within ``max_difference`` of one member.

    Angles are compared with wrap-around (``2 pi`` periodic).  Port of SIRIUS
    ``_find_optimal_set_angle``: the angle with most neighbours within
    ``max_difference`` is chosen as a representative, its neighbourhood is removed,
    and the procedure repeats.

    Parameters
    ----------
    angles : np.ndarray, [n], radians
    max_difference : float, radians

    Returns
    -------
    representative_angles : np.ndarray, [n_subset], radians
    nearest_difference : np.ndarray, [n], radians
        Wrapped distance of each input angle to its nearest representative.
    nearest_index : np.ndarray, [n] int
        Index into ``representative_angles`` of the nearest representative.
    """
    angles = np.atleast_1d(np.asarray(angles, dtype=np.float64))
    n = angles.shape[0]
    neighbours = (
        wrapped_angle_difference(angles[:, None], angles[None, :]) <= max_difference
    )
    np.fill_diagonal(neighbours, True)
    representatives = []
    while True:
        rank = neighbours.sum(axis=1)
        best = int(np.argmax(rank))
        if rank[best] == 0:
            break
        members = np.where(neighbours[best])[0]
        representatives.append(angles[best])
        neighbours[members, :] = False
        neighbours[:, members] = False
    representatives = np.array(representatives, dtype=np.float64)
    if n == 0:
        return representatives, np.zeros(0), np.zeros(0, dtype=np.int64)
    difference = wrapped_angle_difference(angles[:, None], representatives[None, :])
    nearest_index = np.argmin(difference, axis=1)
    return (
        representatives,
        difference[np.arange(n), nearest_index],
        nearest_index.astype(np.int64),
    )
