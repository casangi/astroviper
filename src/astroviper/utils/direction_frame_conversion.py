from astropy.coordinates import (
    SkyCoord, EarthLocation, AltAz, HADec, FK4, FK5,
)
from astropy.coordinates.erfa_astrom import (
    erfa_astrom,
    ErfaAstromInterpolator,
)
from astropy.time import Time
import astropy.units as u

import numpy as np
from scipy.interpolate import CubicSpline

_SUPPORTED_FRAMES = (
    "ICRS",
    "FK5",
    "FK4",  # B1950 equatorial
    "GALACTIC",
    "ALTAZ",
    "HADEC",
)

# frames that require observer location + time
_LOCATION_DEPENDENT = ("ALTAZ", "HADEC")


def convert_direction_frame(
    v0: np.ndarray,
    v1: np.ndarray,
    in_frame: str,
    out_frame: str,
    times: Time,
    location: EarthLocation = None,
    target_times: Time = None,
    ephemeris: bool = False,
    interpolate: bool = False,
    time_resolution: float = 300.0,
) -> tuple:
    """
    Convert between direction frames.
 
      ALTAZ -> ICRS : v0, v1 = az, el  [deg]  ->  returns ra,  dec [deg]
      ICRS -> ALTAZ : v0, v1 = ra, dec [deg]  ->  returns az,  el  [deg]
 
    For each sample astropy applies the full ERFA chain (Earth rotation,
    polar motion, precession, nutation, aberration) at times[i] for location.
 
    If ephemeris=True the input (v0, v1) live on a coarse time grid (times)
    and are first interpolated to target_times in ICRS before converting.
    Interpolation must happen in ICRS — not after conversion — because ALTAZ
    changes non-linearly with Earth's rotation even for a fixed source.
    Use is_ephemeris_ms() to determine this automatically from the MS.
 
    Parameters
    ----------
    v0, v1          : np.ndarray (N,)  input coordinates in degrees
                      ALTAZ  -> az  (-180, 180],  el  [-90,  90]
                      ICRS   -> ra  [0,    360),   dec [-90,  90]
    in_frame        : "ALTAZ", "ICRS", "FK5", "FK4", "GALACTIC", "HADEC"
    out_frame       : "ALTAZ", "ICRS", "FK5", "FK4", "GALACTIC", "HADEC"
    times           : astropy Time (N,)  one timestamp per sample
    location        : astropy EarthLocation  required for ALTAZ / HADEC
    target_times    : astropy Time (M,)  dense timestamps to interpolate onto;
                      required when ephemeris=True
    ephemeris       : if True, interpolate (v0, v1) from times to target_times
                      in ICRS before converting; use is_ephemeris_ms() to set
    interpolate     : if True, use ErfaAstromInterpolator for a large speedup
                      by interpolating Earth orientation parameters on a coarser
                      time grid instead of every sample. Precision remains at
                      micro-arcsecond level for typical time_resolution values.
    time_resolution : interpolation grid spacing in seconds (default 300 s).
                      Smaller = more precise but slower;
                      300 s gives ~0.05 µas error.
 
    Supported frames and coordinate ranges:
      ICRS    : ra [0, 360),    dec [-90, 90]  (approx J2000 for most purposes)
      FK5     : ra [0, 360),    dec [-90, 90]  (J2000 equatorial)
      FK4     : ra [0, 360),    dec [-90, 90]  (B1950 equatorial)
      GALACTIC: l  [0, 360),    b   [-90, 90]
      ALTAZ   : az (-180, 180], el  [-90, 90]  requires location + times
      HADEC   : ha (-180, 180], dec [-90, 90]  requires location + times
 
    Returns
    -------
    out0      : np.ndarray in degrees
    out1      : np.ndarray in degrees
    out_frame : str
    """
    fin, fout = in_frame.upper(), out_frame.upper()

    if fin not in _SUPPORTED_FRAMES:
        raise ValueError(f"Unknown input frame '{fin}'. Supported: {_SUPPORTED_FRAMES}")
    if fout not in _SUPPORTED_FRAMES:
        raise ValueError(
            f"Unknown output frame '{fout}'. Supported: {_SUPPORTED_FRAMES}"
        )
    if fin == fout:
        raise ValueError("in_frame and out_frame must differ.")
    loc_needed = fin in _LOCATION_DEPENDENT or fout in _LOCATION_DEPENDENT
    if loc_needed and location is None:
        raise ValueError(f"location is required for {_LOCATION_DEPENDENT} frames.")

    if ephemeris:
        if target_times is None:
            raise ValueError("target_times is required when ephemeris=True.")
        v0, v1 = interpolate_direction_to_times(v0, v1, times, target_times)
        times = target_times

    # Build location-dependent frames once over the full time array.
    # Astropy broadcasts (v0[i], v1[i]) against times[i] internally via ERFA.
    altaz_frame = AltAz(location=location, obstime=times) if "ALTAZ" in (fin, fout) else None
    hadec_frame = HADec(location=location, obstime=times) if "HADEC" in (fin, fout) else None

    def _transform():
        coord = _build_skycoord(v0, v1, fin, altaz_frame, hadec_frame)
        out0, out1 = _extract_coords(coord, fout, altaz_frame, hadec_frame)
        return out0, out1, fout

    # ErfaAstromInterpolator computes Earth orientation parameters (precession,
    # nutation, polar motion) on a coarser time grid of
    # `time_resolution` seconds and interpolates between those points,
    # instead of calling ERFA at every sample.
    # This gives up to ~100x speedup with micro-arcsecond precision loss.
    if interpolate and (fin in _LOCATION_DEPENDENT or fout in _LOCATION_DEPENDENT):
        with erfa_astrom.set(ErfaAstromInterpolator(time_resolution * u.s)):
            return _transform()
    return _transform()


def _build_skycoord(v0, v1, frame, altaz_frame, hadec_frame):
    """
    Build a SkyCoord from (v0, v1) [deg] given the input frame name.
    altaz_frame and hadec_frame are pre-built with location+obstime.
    """
    if frame == "ICRS":
        return SkyCoord(ra=v0*u.deg, dec=v1*u.deg, frame="icrs")
    elif frame == "FK5":
        return SkyCoord(ra=v0*u.deg, dec=v1*u.deg, frame=FK5(equinox="J2000"))
    elif frame == "FK4":
        return SkyCoord(ra=v0*u.deg, dec=v1*u.deg, frame=FK4(equinox="B1950"))
    elif frame == "GALACTIC":
        return SkyCoord(l=v0*u.deg, b=v1*u.deg, frame="galactic")
    elif frame == "ALTAZ":
        return SkyCoord(az=v0*u.deg, alt=v1*u.deg, frame=altaz_frame)
    elif frame == "HADEC":
        return SkyCoord(ha=v0*u.deg, dec=v1*u.deg, frame=hadec_frame)


def _extract_coords(coord, frame, altaz_frame, hadec_frame):
    """
    Extract (v0, v1) [deg] from a SkyCoord in the output frame.
    For ALTAZ/HADEC output, az/ha is normalised to (-180, 180] for MSv2
    consistency.
    """
    if frame == "ICRS":
        c = coord.icrs
        return c.ra.deg, c.dec.deg
    elif frame == "FK5":
        c = coord.transform_to(FK5(equinox="J2000"))
        return c.ra.deg, c.dec.deg
    elif frame == "FK4":
        c = coord.transform_to(FK4(equinox="B1950"))
        return c.ra.deg, c.dec.deg
    elif frame == "GALACTIC":
        c = coord.galactic
        return c.l.deg, c.b.deg
    elif frame == "ALTAZ":
        c = coord.transform_to(altaz_frame)
        # Astropy returns az in [0°, 360°) but MSv2 convention is (-180°, 180°]
        # Normalise for consistency;
        # remove once MSv2 compatibility is no longer needed.
        return (c.az.deg + 180) % 360 - 180, c.alt.deg
    elif frame == "HADEC":
        c = coord.transform_to(hadec_frame)
        return (c.ha.deg + 180) % 360 - 180, c.dec.deg


def interpolate_direction_to_times(
    ra: np.ndarray,
    dec: np.ndarray,
    source_times: Time,
    target_times: Time,
) -> tuple:
    """
    Interpolate (ra, dec) from source_times onto target_times.
 
    Used by both the pointing and ephemeris pipelines:
      - Pointing  : source_times = time_pointing,  target_times = visibility time
      - Ephemeris : source_times = time_ephemeris, target_times = visibility time
 
    RA is unwrapped before interpolation to avoid discontinuities at 0/360,
    then re-wrapped to [0, 360) after.
 
    Parameters
    ----------
    ra, dec      : np.ndarray (N,)  ICRS coordinates at source_times [deg]
                   ra [0, 360),  dec [-90, 90]
    source_times : astropy Time (N,)  timestamps of the input samples
    target_times : astropy Time (M,)  timestamps to interpolate onto
 
    Returns
    -------
    ra_interp  : np.ndarray (M,)  ra [0, 360) [deg]
    dec_interp : np.ndarray (M,)  dec [-90, 90] [deg]
    """

    t_src = source_times.unix
    t_tgt = target_times.unix
 
    ra_unwrapped = np.unwrap(np.deg2rad(ra))
    ra_interp = np.rad2deg(CubicSpline(t_src, ra_unwrapped)(t_tgt)) % 360
    dec_interp = CubicSpline(t_src, dec)(t_tgt)
 
    return ra_interp, dec_interp


def is_ephemeris_ms(ms) -> bool:
    """
    Return True if the MS contains ephemeris data.
 
    Detection is based on field_and_source_xds.attrs["type"]:
      "field_and_source_ephemeris" -> True
      anything else                -> False
 
    Parameters
    ----------
    ms : msv4 DataTree
 
    Returns
    -------
    bool
    """
    fxds = ms.xr_ms.get_field_and_source_xds()
    return fxds.attrs.get("type") == "field_and_source_ephemeris"
