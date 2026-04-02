"""Shared Gaussian math constants and FWHM/sigma conversion helpers.

All Gaussian-related code in the package should import from here rather than
inlining the conversion factor.
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
