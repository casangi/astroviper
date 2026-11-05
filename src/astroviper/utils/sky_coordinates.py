"""Utilities for frame-aware angular parsing and sky-to-native conversions.

This module centralizes small but reusable coordinate helpers that appear in
multiple astronomy workflows:
- deciding whether a frame's longitude is naturally expressed in hours or degrees
- coercing mixed numeric, quantity, and sexagesimal inputs into radians
- parsing longitude/latitude pairs in a frame-aware way
- converting sky-frame coordinates into native image ``(l, m)`` direction cosines

The helpers are intentionally generic and do not assume any specific dataset
schema or fitter API.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, SkyCoord


def is_scalar_number(value: Any) -> bool:
    """Identify scalar numeric inputs that can be treated as already-typed values.

    Parameters
    ----------
    value : Any
        Candidate scalar value.

    Returns
    -------
    bool
        ``True`` when *value* is a scalar numeric quantity and not a string-like
        token.

    Notes
    -----
    Coordinate-parsing code often needs to distinguish machine-readable numeric
    values from human-oriented string tokens such as ``"12:34:56"`` or
    ``"10deg"``. This helper exists so callers can keep numeric inputs on the
    direct float path while routing string-like values through unit-aware parsers.
    """
    return np.isscalar(value) and not isinstance(value, str | bytes)


def frame_prefers_hourangle(frame: str) -> bool:
    """Report the default longitude convention for sexagesimal strings in a frame.

    Parameters
    ----------
    frame : str
        Celestial frame name.

    Returns
    -------
    bool
        ``True`` for common equatorial frames whose longitude axis is normally
        expressed as right ascension, otherwise ``False``.

    Notes
    -----
    Sexagesimal strings such as ``"12:00:00"`` are ambiguous without units. In
    equatorial frames they are usually interpreted as hour angle for the
    longitude coordinate, while in Galactic or Supergalactic-style frames the
    longitude is more naturally interpreted in degrees. This helper captures that
    convention choice in one place so callers do not need to hard-code frame
    lists repeatedly.
    """
    return str(frame).lower() in {
        "icrs",
        "fk5",
        "fk4",
        "fk4noeterms",
        "cirs",
        "hcrs",
        "gcrs",
    }


def coerce_angle_to_radians(
    value: Any,
    *,
    prefer_hourangle: bool = False,
) -> float:
    """Normalize an angular input into radians across numeric and string forms.

    Parameters
    ----------
    value : Any
        Angle value to convert. Supported inputs are numeric radians, Astropy angle
        or quantity objects, and strings with explicit angular units or sexagesimal
        notation.
    prefer_hourangle : bool, optional
        When ``True``, ambiguous unit-less sexagesimal strings are interpreted as
        hour angle instead of degrees.

    Returns
    -------
    float
        Angle value in radians.

    Notes
    -----
    The conversion order is:
    1. Plain numeric scalar: treat it as already being in radians.
    2. Let ``astropy.coordinates.Angle`` parse the value directly. This handles
       explicit-unit strings and Astropy angle/quantity objects.
    3. If direct parsing fails, retry using a caller-selected default unit of
       either hour angle or degrees for ambiguous sexagesimal strings.

    This makes the helper usable in public APIs that accept mixed machine and
    human input forms without forcing every caller to reimplement the same
    parsing cascade.
    """
    if is_scalar_number(value):
        return float(value)
    try:
        return Angle(value).to_value(u.rad)
    except Exception:
        unit = u.hourangle if prefer_hourangle else u.deg
        return Angle(value, unit=unit).to_value(u.rad)


def parse_sky_center_to_radians(
    lon_value: Any,
    lat_value: Any,
    frame: str,
) -> tuple[float, float]:
    """Parse a longitude/latitude pair into radians using frame-aware defaults.

    Parameters
    ----------
    lon_value : Any
        Longitude-like value in the target sky frame. Numeric inputs are
        interpreted as radians; sexagesimal strings are accepted.
    lat_value : Any
        Latitude-like value in the target sky frame. Numeric inputs are
        interpreted as radians; sexagesimal strings are accepted.
    frame : str
        Celestial frame name.

    Returns
    -------
    tuple[float, float]
        Parsed ``(lon_rad, lat_rad)`` pair in radians.

    Notes
    -----
    This helper first tries to parse string inputs as a coupled sky coordinate
    using ``astropy.coordinates.SkyCoord`` so frame-aware conventions and
    pairwise validation are handled together. For equatorial frames, ambiguous
    sexagesimal longitudes default to hour angle; for non-equatorial frames they
    default to degrees. If that coupled parse fails, the helper falls back to
    parsing longitude and latitude independently via :func:`coerce_angle_to_radians`.

    Numeric inputs are always treated as radians. The function therefore provides
    a consistent boundary between human-facing angular input and internal
    radian-based numerical code.
    """
    prefer_hourangle = frame_prefers_hourangle(frame)
    if isinstance(lon_value, str) and isinstance(lat_value, str):
        try:
            unit = (u.hourangle, u.deg) if prefer_hourangle else (u.deg, u.deg)
            sc = SkyCoord(lon_value, lat_value, unit=unit, frame=frame)
            return (
                sc.spherical.lon.to_value(u.rad),
                sc.spherical.lat.to_value(u.rad),
            )
        except Exception:
            pass
    return (
        coerce_angle_to_radians(lon_value, prefer_hourangle=prefer_hourangle),
        coerce_angle_to_radians(lat_value, prefer_hourangle=False),
    )


def skycoord_to_lm_from_wcs(
    lon_rad: float,
    lat_rad: float,
    phase_center: Sequence[float],
    projection: str,
) -> tuple[float, float]:
    """Convert sky longitude/latitude into native direction-cosine coordinates.

    Parameters
    ----------
    lon_rad : float
        Sky longitude in radians.
    lat_rad : float
        Sky latitude in radians.
    phase_center : sequence[float]
        Reference ``[lon0, lat0]`` in radians.
    projection : str
        WCS projection code. Currently only ``"SIN"`` is modeled explicitly;
        other values warn and reuse the SIN small-field relation.

    Returns
    -------
    tuple[float, float]
        Native ``(l, m)`` coordinates in radians.

    Notes
    -----
    The returned coordinates are the standard direction cosines used by radio
    imaging, relative to the supplied phase center. The current implementation
    explicitly models the SIN projection and reuses that relation as a fallback
    for unsupported projection labels while emitting a warning. This keeps the
    function useful for small-field workflows without silently pretending that an
    arbitrary projection has exact support.
    """
    if projection != "SIN":
        warnings.warn(
            f"Projection {projection!r} not directly supported for sky -> "
            f"l,m conversion; falling back to SIN projection.",
            stacklevel=2,
        )
    lon0, lat0 = float(phase_center[0]), float(phase_center[1])
    dlon = float(lon_rad) - lon0
    lat = float(lat_rad)
    l_val = np.cos(lat) * np.sin(dlon)
    m_val = np.sin(lat) * np.cos(lat0) - np.cos(lat) * np.sin(lat0) * np.cos(dlon)
    return float(l_val), float(m_val)
