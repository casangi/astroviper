"""Utilities for validating monotonic coordinate axes and mapping them to pixels.

These helpers are intended for code that needs to move between explicit
world-coordinate axes and zero-based pixel-center coordinates without making
assumptions about whether an axis is ascending or descending. The functions keep
the logic for monotonicity checks, interpolation preparation, and representative
pixel-scale estimation in one place so image-analysis code can reuse the same
rules consistently.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def prepare_world_to_pixel_interp(
    axis_coord: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a world axis into interpolation inputs compatible with ``np.interp``.

    Parameters
    ----------
    axis_coord : np.ndarray
        One-dimensional world-coordinate axis associated with zero-based pixel
        centers.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Ascending ``(xp, fp)`` arrays suitable for ``np.interp`` where ``xp`` is the
        world axis and ``fp`` is the matching pixel-center axis.

    Raises
    ------
    ValueError
        If the axis is not finite, one-dimensional, or strictly monotonic.

    Notes
    -----
    ``np.interp`` requires its sample axis to be strictly increasing. Real image
    axes are often descending, for example when a longitude-like coordinate
    increases to the left on the screen. This helper validates the axis, creates
    the matching zero-based pixel-center vector, and reverses both arrays
    together when needed so the physical mapping is preserved while satisfying
    ``np.interp``'s preconditions.
    """
    coord = np.asarray(axis_coord, dtype=float)
    if coord.ndim != 1 or coord.size == 0 or not np.all(np.isfinite(coord)):
        raise ValueError("Image coordinate axes must be finite 1-D arrays.")
    diff = np.diff(coord)
    if not (np.all(diff > 0) or np.all(diff < 0)):
        raise ValueError("Image coordinate axes must be strictly monotonic.")
    pixels = np.arange(coord.size, dtype=float)
    if np.all(diff < 0):
        coord = coord[::-1]
        pixels = pixels[::-1]
    return coord, pixels


def world_value_to_pixel(
    value: float,
    axis_coord: np.ndarray,
    axis_name: str,
) -> float:
    """Map one world-coordinate value onto the corresponding pixel-center index.

    Parameters
    ----------
    value : float
        World-coordinate value to convert.
    axis_coord : np.ndarray
        One-dimensional world-coordinate axis for the corresponding dimension.
    axis_name : str
        Axis label used in any validation error.

    Returns
    -------
    float
        Pixel-center coordinate in the zero-based frame.

    Raises
    ------
    ValueError
        If the requested coordinate lies outside the axis span.

    Notes
    -----
    The interpolation is defined on zero-based pixel centers, not pixel edges.
    The helper therefore returns a floating-point pixel coordinate suitable for
    downstream fitting or resampling code. Values outside the world-axis span are
    rejected explicitly rather than clipped so callers do not accidentally turn an
    out-of-bounds physical coordinate into an apparently valid edge pixel.
    """
    xp, fp = prepare_world_to_pixel_interp(axis_coord)
    if value < xp[0] or value > xp[-1]:
        raise ValueError(
            f"Initial guess {axis_name}={value!r} lies outside the image coordinate range."
        )
    return float(np.interp(float(value), xp, fp))


def representative_pixel_scale(
    axis_coord: np.ndarray,
    axis_name: str,
) -> float:
    """Estimate a robust world-units-per-pixel scale from a monotonic axis.

    Parameters
    ----------
    axis_coord : np.ndarray
        One-dimensional monotonic world-coordinate axis.
    axis_name : str
        Axis label used in validation errors.

    Returns
    -------
    float
        Positive world-units-per-pixel scale derived from the median axis spacing.

    Raises
    ------
    ValueError
        If the axis is invalid or degenerate for scale conversion.

    Notes
    -----
    The scale is computed from the median absolute spacing between neighboring
    coordinate values. Using the median keeps the estimate stable in the presence
    of small floating-point irregularities while still rejecting axes that are
    degenerate or non-finite. The result is a convenient summary scale for code
    that needs a local pixel metric without carrying the full coordinate axis
    through every interface.
    """
    coord = np.asarray(axis_coord, dtype=float)
    prepare_world_to_pixel_interp(coord)
    spacing = np.abs(np.diff(coord))
    scale = float(np.median(spacing))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Image {axis_name}-axis spacing must be positive and finite.")
    return scale
