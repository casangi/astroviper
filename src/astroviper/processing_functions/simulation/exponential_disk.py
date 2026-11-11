"""Exponential disk sky component (Sersic n = 1), with the w term.

The unit-flux profile ``exp(-r / r_s) / (2 pi r_s^2)`` of a galaxy disk, in
its face-on frame, inclined by ``inclination`` (axis ratio ``q = cos i``)
with the major axis at position angle ``pa``.  In the tangent plane the
visibility is the closed form

    T = (1 + 4 pi^2 r_s^2 rho^2)^(-3/2),

``rho`` the deprojected baseline length.  With the w term there is no closed
form, but ``exp(-r / r_s)`` is a *scale mixture of Gaussians*,

    exp(-r / r_s) / (2 pi r_s^2) = int_0^inf (2 / sqrt(pi)) s^(1/2) e^(-s) N(r; 2 r_s^2 s) ds,

so the response is the same mixture of the closed-form elliptical-Gaussian
responses (complex widths ``g_j = 1 / (1 + 2 pi i w_j sigma^2)`` with the
anisotropic ``w_1 = w``, ``w_2 = q^2 w`` of the face-on frame), integrated
by generalised Gauss-Laguerre quadrature
(:func:`~astroviper.processing_functions.simulation.component_series.exponential_disk_series`).
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.component_series import (
    deprojected_baseline,
    deprojected_sky,
    exponential_disk_series,
)


def _check_exponential(scale_radius, inclination):
    if scale_radius <= 0:
        raise ValueError(
            f"exponential disk scale_radius must be positive; got {scale_radius}."
        )
    if not (0.0 <= inclination < np.pi / 2):
        raise ValueError(
            f"exponential disk inclination must be in [0, pi/2) radians; got {inclination}."
        )


def exponential_disk_uv_response(
    u, v, scale_radius, inclination=0.0, pa=0.0, w=0.0
) -> np.ndarray:
    """Normalised visibility of a unit-flux inclined exponential disk, with the w term.

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component; ``w = 0``
        (default) is the tangent-plane approximation.
    scale_radius : float, radians
        Exponential scale length ``r_s`` (the half-light radius is ``1.678 r_s``).
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
    scale_radius = float(scale_radius)
    inclination = float(inclination)
    _check_exponential(scale_radius, inclination)
    q = float(np.cos(inclination))
    w = np.asarray(w, dtype=np.float64)
    k1, k2 = deprojected_baseline(u, v, pa, q)
    if not np.any(w):
        return (1.0 + 4.0 * np.pi**2 * scale_radius**2 * (k1**2 + k2**2)) ** -1.5
    return exponential_disk_series(k1, k2, w, q * q * w, scale_radius)


def exponential_disk_image(l, m, scale_radius, inclination=0.0, pa=0.0) -> np.ndarray:  # noqa: E741
    """Unit-total-flux surface brightness ``exp(-r / r_s) / (2 pi r_s^2 q)`` (per steradian)."""
    scale_radius = float(scale_radius)
    inclination = float(inclination)
    _check_exponential(scale_radius, inclination)
    q = float(np.cos(inclination))
    x, y = deprojected_sky(l, m, pa, q)
    r = np.sqrt(x**2 + y**2)
    return np.exp(-r / scale_radius) / (2.0 * np.pi * scale_radius**2 * q)
