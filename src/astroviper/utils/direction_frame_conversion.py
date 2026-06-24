from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates.erfa_astrom import (
    erfa_astrom,
    ErfaAstrom,
    ErfaAstromInterpolator,
)
from astropy.time import Time
import astropy.units as u

import numpy as np
from scipy.interpolate import CubicSpline

_SUPPORTED = ("ICRS", "ALTAZ")


def convert_direction_frame(
    v0: np.ndarray,
    v1: np.ndarray,
    in_frame: str,
    out_frame: str,
    times: Time,
    location: EarthLocation,
    interpolate: bool = False,
    time_resolution: float = 300.0,
) -> tuple:
    """
    Convert between ALTAZ and ICRS.

      ALTAZ → ICRS : v0, v1 = az, el  [deg]  →  returns ra,  dec [deg]
      ICRS → ALTAZ : v0, v1 = ra, dec [deg]  →  returns az,  el  [deg]

    For each sample astropy applies the full ERFA chain (Earth rotation,
    polar motion, precession, nutation, aberration) at times[i] for location.

    Parameters
    ----------
    v0, v1          : np.ndarray (N,)  input coordinates in degrees
                      ALTAZ  → az  (-180°, 180°],  el  [-90°,  90°]
                      ICRS   → ra  [0°,    360°),   dec [-90°,  90°]
    in_frame        : "ALTAZ" or "ICRS"
    out_frame       : "ALTAZ" or "ICRS"
    times           : astropy Time (N,)  one timestamp per sample
    location        : astropy EarthLocation
    interpolate     : if True, use ErfaAstromInterpolator for a large speedup
                      by interpolating Earth orientation parameters
                      on a coarser time grid instead of every sample.
                      Precision remains at micro-arcsecond level for typical
                      time_resolution values.
    time_resolution : interpolation grid spacing in seconds (default 300 s).
                      Smaller = more precise but slower;
                      300 s gives ~0.05 µas error.

    Returns
    -------
    out0, out1 : np.ndarray (N,) in degrees
                  ICRS output → ra [0°, 360°), dec [-90°, 90°]
                  ALTAZ output → az (-180°, 180°], el  [-90°, 90°]
    """
    fin, fout = in_frame.upper(), out_frame.upper()
    if fin not in _SUPPORTED or fout not in _SUPPORTED:
        raise ValueError(f"Only {_SUPPORTED} are supported for now.")
    if fin == fout:
        raise ValueError("in_frame and out_frame must differ.")

    # Build one AltAz frame with the full time array — astropy broadcasts
    # each (v0[i], v1[i]) against times[i] internally via ERFA
    altaz_frame = AltAz(location=location, obstime=times)

    # ErfaAstromInterpolator computes Earth orientation parameters (precession,
    # nutation, polar motion) on a coarser time grid of
    # `time_resolution` seconds and interpolates between those points,
    # instead of calling ERFA at every sample.
    # This gives up to ~100x speedup with micro-arcsecond precision loss.
    ctx = (
        erfa_astrom.set(ErfaAstromInterpolator(time_resolution * u.s))
        if interpolate
        else None
    )
    try:
        if fin == "ALTAZ":
            coord = SkyCoord(az=v0 * u.deg, alt=v1 * u.deg, frame=altaz_frame)
            c = coord.icrs
            return c.ra.deg, c.dec.deg, fout
        else:
            coord = SkyCoord(ra=v0 * u.deg, dec=v1 * u.deg, frame="icrs")
            c = coord.transform_to(altaz_frame)
            # Normalise for consistency;
            # remove once MSv2 compatibility is no longer needed.
            az = (c.az.deg + 180) % 360 - 180
            return az, c.alt.deg, fout
    finally:
        if ctx is not None:
            erfa_astrom.set(ErfaAstrom())  # reset to default


def interpolate_pointing_to_spectral_times(
    ra: np.ndarray,
    dec: np.ndarray,
    pointing_times: Time,
    ms: object,
) -> tuple:
    """
    Interpolate (ra, dec) from pointing timestamps to the spectral timestamps.

    Pointing and spectral data are sampled independently at different rates:
      pointing_times : [t0, t1, ...]   from pointing_xds  (e.g. 5050 samples)
      spectral_times : [T0, T1, ...]   from main.xds time (e.g. 15120 samples)

    For each spectral timestamp Ti we interpolate ra and dec from the pointing
    grid to get ra(Ti) and dec(Ti).

    Parameters
    ----------
    ra, dec         : np.ndarray ICRS coordinates at pointing_times [deg]
    pointing_times  : astropy Time timestamps of the pointing samples
    ms              : msv4 used to extract spectral timestamps from main.xds

    Returns
    -------
    ra_interp, dec_interp : np.ndarray interpolated at spectral times [deg]
    spectral_times        : astropy Time the spectral timestamps used
    """

    # -- extract spectral timestamps from main.xds ----------------------------
    # spectral times live on the 'time' axis of the main dataset
    main_xds = ms["main.xds"] if "main.xds" in ms.children else ms
    t_raw = main_xds["time"].data * u.Unit(main_xds["time"].attrs["units"])
    spec_times = Time(
        t_raw,
        format=main_xds["time"].attrs.get("format", "unix"),
        scale=main_xds["time"].attrs.get("scale", "utc"),
    )

    # -- convert both time axes to unix seconds for interpolation -------------
    # scipy needs plain floats; unix seconds is a natural common reference
    t_point = pointing_times.unix
    t_spec = spec_times.unix

    # -- interpolate ----------------------------------------------------------
    # ra wraps at 0/360 — interpolate on the unwrapped values to avoid
    # discontinuities near 0°/360°, then re-wrap to [0°, 360°)
    ra_unwrapped = np.unwrap(np.deg2rad(ra))  # unwrap in radians
    ra_interp = np.rad2deg(CubicSpline(t_point, ra_unwrapped)(t_spec)) % 360
    dec_interp = CubicSpline(t_point, dec)(t_spec)

    return ra_interp, dec_interp, spec_times
