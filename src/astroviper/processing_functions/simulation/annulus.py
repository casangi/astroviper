"""Uniform annulus sky component (a ring of finite width), with the w term.

A uniform ring between the radii ``inner_radius`` and ``radius`` in its
face-on frame, inclined by ``inclination`` (axis ratio ``q = cos i``) with the
major axis at position angle ``pa`` and optionally convolved with a circular
Gaussian of FWHM ``fwhm`` (in the face-on frame; the whole blurred image is
stretched, as for the Gaussian ring).  ``inner_radius = 0`` is the uniform
disk; a narrow annulus approaches the thin ring.  Shells of supernova
remnants and planetary nebulae, the "uniform annulus" black-hole models and
finite-width protoplanetary rings are the intended uses.

The annulus is the difference of two concentric uniform disks, so its w-term
response is ``[R^2 T_disk(R) - R_in^2 T_disk(R_in)] / (R^2 - R_in^2)`` with
both disks evaluated by the same
:func:`~astroviper.processing_functions.simulation.component_series.uniform_disk_series`
(same face-on baseline, same anisotropic quadratic phase ``(w, q^2 w)``, same
:func:`~astroviper.processing_functions.simulation.component_series.blur_transform`).
In the tangent plane: ``[R^2 2J_1(2 pi R rho)/(2 pi R rho) - R_in^2 2J_1(2 pi
R_in rho)/(2 pi R_in rho)] / (R^2 - R_in^2)`` with ``rho`` the deprojected
baseline length.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.component_series import (
    blur_quadrature_nodes,
    blur_transform,
    blurred_radial_profile_image,
    deprojected_baseline,
    deprojected_sky,
    uniform_disk_series,
)
from astroviper.processing_functions.simulation.gaussian_source import FWHM_TO_SIGMA


def _check_annulus(radius, inner_radius, inclination, fwhm):
    if radius <= 0 or inner_radius < 0 or inner_radius >= radius:
        raise ValueError(
            "annulus needs 0 <= inner_radius < radius; "
            f"got radius={radius}, inner_radius={inner_radius}."
        )
    if not (0.0 <= inclination < np.pi / 2):
        raise ValueError(
            f"annulus inclination must be in [0, pi/2) radians; got {inclination}."
        )
    if fwhm < 0:
        raise ValueError(f"annulus fwhm must be non-negative; got {fwhm}.")


def annulus_uv_response(
    u,
    v,
    radius,
    inner_radius,
    inclination=0.0,
    pa=0.0,
    fwhm=0.0,
    w=0.0,
    tolerance=1e-14,
) -> np.ndarray:
    """Normalised visibility of a unit-flux (blurred, inclined) uniform annulus, with the w term.

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component; ``w = 0``
        (default) is the tangent-plane approximation.
    radius, inner_radius : float, radians
        Outer and inner radii in the face-on frame (``0 <= inner < outer``).
    inclination : float, radians
        ``0`` face-on; the minor axis is ``cos(inclination)`` times the major axis.
    pa : float, radians
        Position angle of the major axis, from ``+m`` towards ``+l``.
    fwhm : float, radians
        FWHM of the circular Gaussian the annulus is convolved with.
    tolerance : float
        Truncation tolerance of the disk series.

    Returns
    -------
    numpy.ndarray of complex, broadcast shape of ``u``, ``v`` and ``w``.
    """
    radius = float(radius)
    inner_radius = float(inner_radius)
    inclination = float(inclination)
    fwhm = float(fwhm)
    _check_annulus(radius, inner_radius, inclination, fwhm)
    q = float(np.cos(inclination))
    w = np.asarray(w, dtype=np.float64)
    k1, k2 = deprojected_baseline(u, v, pa, q)
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    k1, k2, w1, w2, prefactor = blur_transform(k1, k2, w, q * q * w, sigma2)
    response = radius**2 * uniform_disk_series(k1, k2, w1, w2, radius, 1.0, tolerance)
    if inner_radius > 0.0:
        response = response - inner_radius**2 * uniform_disk_series(
            k1, k2, w1, w2, inner_radius, 1.0, tolerance
        )
    return prefactor * response / (radius**2 - inner_radius**2)


def annulus_image(l, m, radius, inner_radius, inclination=0.0, pa=0.0, fwhm=0.0):  # noqa: E741
    """Unit-total-flux surface brightness of a (blurred, inclined) uniform annulus (per steradian)."""
    radius = float(radius)
    inner_radius = float(inner_radius)
    inclination = float(inclination)
    fwhm = float(fwhm)
    _check_annulus(radius, inner_radius, inclination, fwhm)
    q = float(np.cos(inclination))
    x, y = deprojected_sky(l, m, pa, q)
    r = np.sqrt(x**2 + y**2)
    norm = np.pi * (radius**2 - inner_radius**2) * q
    if fwhm == 0.0:
        return ((r >= inner_radius) & (r < radius)).astype(np.float64) / norm
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    nodes, weights = np.polynomial.legendre.leggauss(
        blur_quadrature_nodes(radius - inner_radius, sigma2)
    )
    half = (radius - inner_radius) / 2.0
    nodes = inner_radius + half * (nodes + 1.0)
    return blurred_radial_profile_image(r, nodes, weights * half, sigma2) / norm
