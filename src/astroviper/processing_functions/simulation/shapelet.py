"""Cartesian (Gauss-Hermite) shapelet sky component, with the w term.

Shapelets (Refregier 2003) expand a compact source in the orthonormal basis

    phi_n(x; beta) = [2^n n! sqrt(pi) beta]^(-1/2) H_n(x / beta) exp(-x^2 / 2 beta^2)

(``H_n`` the physicists' Hermite polynomials), one factor per axis:
``b(x, y) = sum c_{n1 n2} phi_{n1}(x; beta_1) phi_{n2}(y; beta_2) / N_c`` with
``N_c`` the integral of the raw expansion, so that the component has unit
flux whatever the coefficients (the coefficients must not integrate to zero).
The first axis ``x`` is along position angle ``pa`` (``+m`` for ``pa = 0``),
the second axis ``y`` is perpendicular (``+l`` for ``pa = 0``).  This is the
generic "arbitrary compact morphology" component: a few coefficients describe
lopsided, boxy or multi-lobed sources that no other analytic component fits,
and image fits with shapelet codes can be imported directly.

The Hermite functions are eigenfunctions of the Fourier transform,

    int H_n(x/beta) e^{-x^2/2 beta^2} e^{2 pi i u x} dx
        = sqrt(2 pi) beta i^n H_n(2 pi u beta) e^{-2 pi^2 u^2 beta^2},

so the tangent-plane response is again a shapelet series.  The w term
multiplies the Gaussian by ``exp(-i pi w x^2)``, turning its width into the
complex ``beta_c^2 = beta^2 g``, ``g = 1 / (1 + 2 pi i w beta^2)``, while the
polynomial keeps the real scale; the Hermite multiplication theorem

    H_n(lambda t) = sum_k n! / (k! (n - 2k)!) (lambda^2 - 1)^k lambda^(n-2k) H_{n-2k}(t),
    lambda = beta_c / beta = sqrt(g),

re-expands the polynomial at the complex scale, after which each term
transforms exactly (analytic continuation of the identity above to complex
``beta_c`` with ``Re(1/beta_c^2) > 0``).  The w term is therefore exact and
finite for shapelets: ``floor(n/2) + 1`` terms per order.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.component_series import (
    deprojected_baseline,
    deprojected_sky,
    gaussian_width_factor,
)

MAX_ORDER = 40


def _check_shapelet(scale, coefficients):
    scale = np.atleast_1d(np.asarray(scale, dtype=np.float64)).ravel()
    if scale.size == 1:
        scale = np.repeat(scale, 2)
    if scale.size != 2 or np.any(scale <= 0):
        raise ValueError(
            f"shapelet scale must be a positive width or a pair [beta_1, beta_2]; got {scale}."
        )
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=np.float64))
    if (
        coefficients.ndim != 2
        or coefficients.size == 0
        or not np.all(np.isfinite(coefficients))
    ):
        raise ValueError(
            "shapelet coefficients must be a finite 2-D array [n_1 + 1, n_2 + 1]."
        )
    if max(coefficients.shape) - 1 > MAX_ORDER:
        raise ValueError(f"shapelet orders above {MAX_ORDER} are not supported.")
    return scale, coefficients


def _hermite_transform_1d(k, w, beta, n_max):
    """``F_n(k) = int phi_n(x) e^{-i pi w x^2} e^{2 pi i k x} dx`` for ``n = 0 .. n_max``.

    Returns an array ``[n_max + 1, *k.shape]`` (complex).  ``w`` broadcastable
    with ``k``.
    """
    from math import factorial

    from numpy.polynomial.hermite import hermval
    from scipy.special import gamma

    k = np.asarray(k, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    shape = np.broadcast(k, w).shape
    k = np.broadcast_to(k, shape)
    w = np.broadcast_to(w, shape)
    g = gaussian_width_factor(w, beta**2)  # complex, [shape]
    beta_c = beta * np.sqrt(g)
    lam = np.sqrt(g)  # beta_c / beta
    arg = 2.0 * np.pi * k * beta_c  # complex Hermite argument
    envelope = np.sqrt(2.0 * np.pi) * beta_c * np.exp(-0.5 * arg**2)
    # H_j(arg) for all needed orders, by the three-term recurrence
    hermite = np.empty((n_max + 1,) + shape, dtype=np.complex128)
    hermite[0] = 1.0
    if n_max >= 1:
        hermite[1] = 2.0 * arg
    for j in range(2, n_max + 1):
        hermite[j] = 2.0 * arg * hermite[j - 1] - 2.0 * (j - 1) * hermite[j - 2]
    del hermval  # (only the recurrence is used; hermval kept importable for tests)
    out = np.empty((n_max + 1,) + shape, dtype=np.complex128)
    lam2m1 = lam**2 - 1.0
    for n in range(n_max + 1):
        norm = 1.0 / np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi) * beta)
        total = np.zeros(shape, dtype=np.complex128)
        for kk in range(n // 2 + 1):
            order = n - 2 * kk
            coefficient = factorial(n) / (factorial(kk) * factorial(order))
            total += (
                coefficient * lam2m1**kk * lam**order * (1j**order) * hermite[order]
            )
        out[n] = norm * envelope * total
    _ = gamma  # noqa: F841  (scipy import kept for parity with the image routine)
    return out


def shapelet_uv_response(u, v, scale, coefficients, pa=0.0, w=0.0) -> np.ndarray:
    """Normalised visibility of a unit-flux Cartesian shapelet expansion, with the w term.

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component; ``w = 0``
        (default) is the tangent-plane approximation.
    scale : float or [2] floats, radians
        Shapelet scale ``beta`` (one value for both axes, or ``[beta_1, beta_2]``).
    coefficients : array_like, [n_1 + 1, n_2 + 1]
        Real shapelet coefficients ``c_{n1 n2}``; the expansion is normalised
        to unit flux, so only their ratios matter.
    pa : float, radians
        Position angle of the first axis, from ``+m`` towards ``+l``.

    Returns
    -------
    numpy.ndarray of complex, broadcast shape of ``u``, ``v`` and ``w``.
    """
    scale, coefficients = _check_shapelet(scale, coefficients)
    w = np.asarray(w, dtype=np.float64)
    k1, k2 = deprojected_baseline(u, v, pa, 1.0)
    shape = np.broadcast(k1, k2, w).shape
    n1, n2 = coefficients.shape
    f1 = _hermite_transform_1d(np.broadcast_to(k1, shape), w, scale[0], n1 - 1)
    f2 = _hermite_transform_1d(np.broadcast_to(k2, shape), w, scale[1], n2 - 1)
    raw = np.einsum("ij,i...,j...->...", coefficients, f1, f2)
    flux = _raw_flux(scale, coefficients)
    return raw / flux


def _raw_flux(scale, coefficients):
    """Integral of the raw expansion (its response at the origin)."""
    zero = np.zeros(())
    f1 = _hermite_transform_1d(zero, zero, scale[0], coefficients.shape[0] - 1)
    f2 = _hermite_transform_1d(zero, zero, scale[1], coefficients.shape[1] - 1)
    flux = complex(np.einsum("ij,i,j->", coefficients, f1, f2)).real
    if abs(flux) < 1e-300:
        raise ValueError(
            "shapelet coefficients integrate to zero flux; the component cannot be normalised."
        )
    return flux


def _hermite_functions_1d(x, beta, n_max):
    """``phi_n(x; beta)`` for ``n = 0 .. n_max``: array ``[n_max + 1, *x.shape]``."""
    from math import factorial

    x = np.asarray(x, dtype=np.float64)
    t = x / beta
    hermite = np.empty((n_max + 1,) + x.shape, dtype=np.float64)
    hermite[0] = 1.0
    if n_max >= 1:
        hermite[1] = 2.0 * t
    for j in range(2, n_max + 1):
        hermite[j] = 2.0 * t * hermite[j - 1] - 2.0 * (j - 1) * hermite[j - 2]
    envelope = np.exp(-0.5 * t**2)
    out = np.empty_like(hermite)
    for n in range(n_max + 1):
        out[n] = (
            hermite[n]
            * envelope
            / np.sqrt(2.0**n * factorial(n) * np.sqrt(np.pi) * beta)
        )
    return out


def shapelet_image(l, m, scale, coefficients, pa=0.0) -> np.ndarray:  # noqa: E741
    """Unit-total-flux surface brightness of the shapelet expansion (per steradian)."""
    scale, coefficients = _check_shapelet(scale, coefficients)
    x, y = deprojected_sky(l, m, pa, 1.0)
    shape = np.broadcast(x, y).shape
    p1 = _hermite_functions_1d(
        np.broadcast_to(x, shape), scale[0], coefficients.shape[0] - 1
    )
    p2 = _hermite_functions_1d(
        np.broadcast_to(y, shape), scale[1], coefficients.shape[1] - 1
    )
    return np.einsum("ij,i...,j...->...", coefficients, p1, p2) / _raw_flux(
        scale, coefficients
    )
