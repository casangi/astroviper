from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstrom,ErfaAstromInterpolator
from astropy.time import Time
import astropy.units as u

import numpy as np

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
    in_frame        : "ALTAZ" or "ICRS"
    out_frame       : "ALTAZ" or "ICRS"
    times           : astropy Time (N,)  one timestamp per sample
    location        : astropy EarthLocation
    interpolate     : if True, use ErfaAstromInterpolator for a large speedup
                      by interpolating Earth orientation parameters on a coarser
                      time grid instead of computing them at every sample.
                      Precision remains at micro-arcsecond level for typical
                      time_resolution values.
    time_resolution : interpolation grid spacing in seconds (default 300 s).
                      Smaller = more precise but slower; 300 s gives ~0.05 µas error.

    Returns
    -------
    out0, out1 : np.ndarray (N,) in degrees
    """
    fin, fout = in_frame.upper(), out_frame.upper()
    if fin not in _SUPPORTED or fout not in _SUPPORTED:
        raise ValueError(f"Only {_SUPPORTED} are supported for now.")
    if fin == fout:
        raise ValueError("in_frame and out_frame must differ.")

    # Build one AltAz frame with the full time array — astropy broadcasts
    # each (v0[i], v1[i]) against times[i] internally via ERFA, no Python loop needed.
    altaz_frame = AltAz(location=location, obstime=times)

    # ErfaAstromInterpolator computes Earth orientation parameters (precession,
    # nutation, polar motion) on a coarser time grid of `time_resolution` seconds
    # and interpolates between those points, instead of calling ERFA at every sample.
    # This gives up to ~100x speedup with micro-arcsecond precision loss.
    ctx = erfa_astrom.set(ErfaAstromInterpolator(time_resolution * u.s)) if interpolate else None
    try:
        if fin == "ALTAZ":                    # ALTAZ → ICRS
            coord = SkyCoord(az=v0*u.deg, alt=v1*u.deg, frame=altaz_frame)
            c = coord.icrs
            return c.ra.deg, c.dec.deg, fout
        else:                                 # ICRS → ALTAZ
            coord = SkyCoord(ra=v0*u.deg, dec=v1*u.deg, frame="icrs")
            c = coord.transform_to(altaz_frame)
            # Astropy returns az in [0°, 360°) but MSv2 convention is (-180°, 180°].
            # Normalise for consistency; remove once MSv2 compatibility is no longer needed.
            az = (c.az.deg + 180) % 360 - 180
            return az, c.alt.deg, fout
    finally:
        if ctx is not None:
            erfa_astrom.set(ErfaAstrom())  # reset to default