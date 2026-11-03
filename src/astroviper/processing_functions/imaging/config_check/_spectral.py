"""Shared spectral inputs and weighted Taylor design for preflight diagnostics.

These utilities prepare small spectral arrays only. Numerical rank, Hessian
safety, noise assumptions, and caller-specific error context stay in the checkers.
"""

import numpy as np

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_taylor_count,
)


def prepare_spectral_samples(
    frequencies, effective_weights, *, n_taylor_terms, reference_frequency, precision
):
    """Validate spectral metadata and identify positive-weight channels.

    Parameters
    ----------
    frequencies, effective_weights : array-like
        Matching one-dimensional arrays of positive frequencies and finite,
        nonnegative effective imaging weights. Repeated frequencies are allowed.
    n_taylor_terms : int
        Positive integer Taylor count, independent of the selected deconvolver.
    reference_frequency : float
        Finite positive scalar reference in Hz.
    precision : {"float32", "float64"}
        Working precision used by the caller's numerical diagnostics.

    Returns
    -------
    frequencies, weights : numpy.ndarray
        Validated original arrays; no weights or frequencies are changed.
    reference : float
        Scalar reference for arithmetic without xarray scalar dispatch.
    usable : numpy.ndarray of bool
        Positive-weight channel mask. Callers report empty or insufficient
        independent support in the context of their respective diagnostics.
    """
    # --- Physical metadata and Taylor basis ---
    frequencies = require_finite_real(
        "selected frequencies", frequencies, positive=True
    )
    weights = require_finite_real("effective imaging weights", effective_weights)
    reference = require_finite_real(
        "reference_frequency", reference_frequency, positive=True, scalar=True
    )
    check_taylor_count(n_taylor_terms)

    # --- Channel correspondence and working precision ---
    if frequencies.ndim != 1 or weights.shape != frequencies.shape:
        raise ValueError(
            "frequencies and effective weights must be matching one-dimensional arrays"
        )
    if np.any(weights < 0):
        raise ValueError("effective imaging weights must be non-negative")
    if np.dtype(precision) not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("working precision must be float32 or float64")

    return frequencies, weights, float(reference), weights > 0


def make_weighted_taylor_design(
    frequencies, weights, *, n_taylor_terms, reference_frequency
):
    """Construct the common weighted design from validated usable samples.

    Parameters
    ----------
    frequencies, weights : numpy.ndarray
        Nonempty usable channels selected from ``prepare_spectral_samples``;
        weights must be strictly positive. Caller inputs are never modified.
    n_taylor_terms : int
        Validated positive Taylor count.
    reference_frequency : float
        Validated positive scalar reference in Hz.

    Returns
    -------
    design : numpy.ndarray
        Matrix with entries sqrt(q_i) * ((nu_i - nu_0) / nu_0)**k.
    normalized_weights, sqrt_weights : numpy.ndarray
        q_i = weight_i / max(weight) and their square roots. A common weight
        scale cancels from both diagnostics and avoids overflowing sqrt(weight).
    """
    # --- Common weight units and frequency offsets ---
    normalized_weights = weights / weights.max()
    sqrt_weights = np.sqrt(normalized_weights)
    x = (frequencies.astype(np.float64) - reference_frequency) / reference_frequency

    # --- Weighted polynomial columns ---
    # Build the design directly; the spectral check uses its SVD, while the
    # approximate Hessian check equilibrates its columns before forming A.T @ A.
    with np.errstate(over="ignore", invalid="ignore"):
        design = np.vander(x, N=n_taylor_terms, increasing=True)
        design *= sqrt_weights[:, None]
    if not np.all(np.isfinite(design)):
        raise ValueError(
            "the weighted Taylor design overflows at this reference frequency"
        )
    return design, normalized_weights, sqrt_weights
