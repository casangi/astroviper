"""Gaussian-tapered power-law sky component, with the w term.

The unit-flux profile ``r^(-gamma) exp(-r^2 / 2 r_c^2)`` (``0 <= gamma < 2``)
in its face-on frame, inclined by ``inclination`` (axis ratio ``q = cos i``)
with the major axis at position angle ``pa``: the cuspy cores of galaxies,
jets' unresolved bases, or any power law that must be tapered to keep its
flux finite.  ``gamma = 0`` is the circular Gaussian of width ``r_c``.

In the tangent plane the visibility is Kummer's confluent hypergeometric
function (Weber's integral),

    T = 1F1(1 - gamma/2; 1; -2 pi^2 r_c^2 rho^2),

``rho`` the deprojected baseline length.  With the w term the profile is
written as a *scale mixture of Gaussians*,

    r^-gamma = int_0^inf t^(gamma/2 - 1) e^{-t r^2} dt / Gamma(gamma/2),

i.e. Gaussians of variance ``sigma^2 = r_c^2 x``, ``x in (0, 1]``, with the
Beta weight ``(1 - x)^(gamma/2 - 1) x^(-gamma/2) / B(gamma/2, 1 - gamma/2)``;
the response is the same mixture of the closed-form elliptical-Gaussian
responses, integrated by Gauss-Jacobi (and, for strongly resolved baselines,
generalised Gauss-Laguerre) quadrature
(:func:`~astroviper.processing_functions.simulation.component_series.tapered_power_law_series`).
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.component_series import (
    deprojected_baseline,
    deprojected_sky,
    tapered_power_law_series,
)


def _check_power_law(cutoff_radius, index, inclination):
    if cutoff_radius <= 0:
        raise ValueError(
            f"tapered power law cutoff_radius must be positive; got {cutoff_radius}."
        )
    if not (0.0 <= index < 2.0):
        raise ValueError(f"tapered power law index must be in [0, 2); got {index}.")
    if not (0.0 <= inclination < np.pi / 2):
        raise ValueError(
            f"tapered power law inclination must be in [0, pi/2) radians; got {inclination}."
        )


def tapered_power_law_uv_response(
    u, v, cutoff_radius, index, inclination=0.0, pa=0.0, w=0.0
) -> np.ndarray:
    """Normalised visibility of a unit-flux inclined Gaussian-tapered power law, with the w term.

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component; ``w = 0``
        (default) is the tangent-plane approximation.
    cutoff_radius : float, radians
        Width ``r_c`` of the Gaussian taper.
    index : float
        Power-law index ``gamma`` in ``[0, 2)``.
    inclination : float, radians
        ``0`` face-on; the minor axis is ``cos(inclination)`` times the major axis.
    pa : float, radians
        Position angle of the major axis, from ``+m`` towards ``+l``.

    Returns
    -------
    numpy.ndarray
        Broadcast shape of ``u``, ``v`` and ``w``; real when ``w`` is
        identically zero, complex otherwise.
    """
    from scipy.special import hyp1f1

    cutoff_radius = float(cutoff_radius)
    index = float(index)
    inclination = float(inclination)
    _check_power_law(cutoff_radius, index, inclination)
    q = float(np.cos(inclination))
    w = np.asarray(w, dtype=np.float64)
    k1, k2 = deprojected_baseline(u, v, pa, q)
    if index == 0.0:  # a circular Gaussian of width r_c
        from astroviper.processing_functions.simulation.component_series import (
            gaussian_width_factor,
        )

        sigma2 = cutoff_radius**2
        if not np.any(w):
            return np.exp(-2.0 * np.pi**2 * sigma2 * (k1**2 + k2**2))
        g1 = gaussian_width_factor(w, sigma2)
        g2 = gaussian_width_factor(q * q * w, sigma2)
        return np.sqrt(g1 * g2) * np.exp(
            -2.0 * np.pi**2 * sigma2 * (g1 * k1**2 + g2 * k2**2)
        )
    if not np.any(w):
        return hyp1f1(
            1.0 - index / 2.0, 1.0, -2.0 * np.pi**2 * cutoff_radius**2 * (k1**2 + k2**2)
        )
    return tapered_power_law_series(k1, k2, w, q * q * w, cutoff_radius, index)


def tapered_power_law_image(
    l,
    m,
    cutoff_radius,
    index,
    inclination=0.0,
    pa=0.0,
    pixel_area=None,  # noqa: E741
) -> np.ndarray:
    """Unit-total-flux surface brightness ``C r^-gamma exp(-r^2 / 2 r_c^2) / q`` (per steradian).

    ``C = 1 / (pi (2 r_c^2)^(1 - gamma/2) Gamma(1 - gamma/2))``.  The profile
    diverges (integrably) at the centre; when ``pixel_area`` is given the
    radius is floored at the value that makes the central pixel carry its
    pixel-averaged flux (``r_eff = ((2 - gamma) / 2)^(1/gamma) r_p`` with
    ``pi r_p^2 = pixel_area``), otherwise the centre evaluates to ``inf``.
    """
    from scipy.special import gamma as gamma_function

    cutoff_radius = float(cutoff_radius)
    index = float(index)
    inclination = float(inclination)
    _check_power_law(cutoff_radius, index, inclination)
    q = float(np.cos(inclination))
    x, y = deprojected_sky(l, m, pa, q)
    r = np.sqrt(x**2 + y**2)
    norm = (
        np.pi
        * (2.0 * cutoff_radius**2) ** (1.0 - index / 2.0)
        * gamma_function(1.0 - index / 2.0)
    )
    if index > 0.0 and pixel_area is not None:
        r_pixel = np.sqrt(float(pixel_area) / np.pi)
        r = np.maximum(r, ((2.0 - index) / 2.0) ** (1.0 / index) * r_pixel)
    with np.errstate(divide="ignore"):
        radial = r ** (-index) if index > 0.0 else np.ones_like(r)
    return radial * np.exp(-0.5 * r**2 / cutoff_radius**2) / (norm * q)
