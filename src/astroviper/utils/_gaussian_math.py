"""Shared Gaussian math constants and conversion helpers.

All Gaussian-related code in the package should import from here rather than
inlining conversion factors or angle formulas.
"""

from __future__ import annotations

import numpy as np

# Conversion factors between Gaussian sigma and full width at half maximum.
# FWHM = SIG2FWHM * sigma,  sigma = FWHM2SIG * FWHM
SIG2FWHM: float = 2.0 * np.sqrt(2.0 * np.log(2.0))
FWHM2SIG: float = 1.0 / SIG2FWHM


def fwhm_from_sigma(sigma) -> np.ndarray:
    """Convert Gaussian sigma values to full width at half maximum.

    Parameters
    ----------
    sigma : array-like
        Scalar or array of Gaussian sigma values.

    Returns
    -------
    np.ndarray
        Values converted to FWHM using the standard Gaussian relation
        ``FWHM = 2 * sqrt(2 * ln(2)) * sigma``.

    Notes
    -----
    No positivity check is applied; callers are expected to pass physically
    meaningful widths.
    """
    return SIG2FWHM * np.asarray(sigma)


def sigma_from_fwhm(fwhm) -> np.ndarray:
    """Convert full width at half maximum values to Gaussian sigma.

    Parameters
    ----------
    fwhm : array-like
        Scalar or array of FWHM values.

    Returns
    -------
    np.ndarray
        Values converted to sigma using the standard Gaussian relation
        ``sigma = FWHM / (2 * sqrt(2 * ln(2)))``.

    Notes
    -----
    No positivity check is applied; callers are expected to pass physically
    meaningful widths.
    """
    return FWHM2SIG * np.asarray(fwhm)


# ---------------------------------------------------------------------------
# PA ↔ math angle conversion
#
# "math" convention: measured from +x toward +y.
# "pa"   convention: astronomical position angle, measured from +y toward +x.
# Relation: theta_math = pi/2 - PA  (and its inverse is identical in form).
# ---------------------------------------------------------------------------


def theta_pa_to_math(pa: np.ndarray) -> np.ndarray:
    """Convert position-angle values to the internal math-angle convention.

    Parameters
    ----------
    pa : array-like
        Angle values in radians measured from +y toward +x.

    Returns
    -------
    np.ndarray
        Equivalent angles in radians measured from +x toward +y, wrapped
        into ``[0, 2*pi)``.

    Notes
    -----
    This conversion is purely geometric and does not apply handedness flips.
    """
    return (np.pi / 2.0 - np.asarray(pa)) % (2.0 * np.pi)


def theta_math_to_pa(theta_math: np.ndarray) -> np.ndarray:
    """Convert internal math-angle values to position-angle values.

    Parameters
    ----------
    theta_math : array-like
        Angle values in radians measured from +x toward +y.

    Returns
    -------
    np.ndarray
        Equivalent PA angles in radians measured from +y toward +x, wrapped
        into ``[0, 2*pi)``.

    Notes
    -----
    This is the inverse mapping of :func:`theta_pa_to_math`.
    """
    return (np.pi / 2.0 - np.asarray(theta_math)) % (2.0 * np.pi)


def normalize_angle(angle_value: float, *, angle: str, degrees: bool) -> float:
    """Convert a user-provided angle to the internal math convention in radians.

    Parameters
    ----------
    angle_value : float
        Input angle, either in radians or degrees depending on ``degrees``.
    angle : {"math", "pa"}
        Convention of the input angle.
        Supported choices are:

        - ``"math"``: measured from +x toward +y; returned unchanged.
        - ``"pa"``: astronomical position angle measured from +y toward +x;
          converted via ``theta_math = pi/2 - PA``.
    degrees : bool
        If ``True``, ``angle_value`` is in degrees and is converted to radians
        before any convention mapping.

    Returns
    -------
    float
        Angle in radians in the internal math convention. No wrapping is
        applied; the raw formula result is returned.

    Notes
    -----
    Unlike :func:`theta_pa_to_math`, this function operates on a single scalar
    and does not wrap the result modulo ``2*pi``.
    """
    theta = float(np.deg2rad(angle_value) if degrees else angle_value)
    return (np.pi / 2.0) - theta if angle == "pa" else theta
