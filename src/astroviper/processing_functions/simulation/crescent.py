"""Crescent sky component (Kamruddin & Dexter 2013), with the w term.

A uniform disk of radius ``R`` from which a smaller uniform disk of radius
``R_in``, displaced by ``offset`` from the centre, is removed (or dimmed to a
``floor`` fraction of the crescent brightness), optionally convolved with a
circular Gaussian of FWHM ``fwhm``: the geometric model of the Event Horizon
Telescope's black-hole shadow images (EHT Collaboration 2019, Paper IV,
"crescent" / ``eht-imaging``'s ``crescent`` and ``blurred_crescent``).  The
brightness asymmetry points at position angle ``pa`` (the thick, bright side
of the crescent; the hole is displaced towards ``pa + 180 deg``).

Unit flux: ``b = [Theta(R - |r|) - (1 - f) Theta(R_in - |r - a|)] /
(pi (R^2 - (1 - f) R_in^2))`` with ``a`` the displacement vector.  Because
the displaced disk is a translate, its w-term response follows exactly from
the centred one,

    int b(r - a) e^{-i pi w r^2} e^{2 pi i k.r} dA
        = e^{2 pi i k.a - i pi w a^2} T_disk(k - w a; w),

so the crescent is two evaluations of the uniform-disk series
(:func:`~astroviper.processing_functions.simulation.component_series.uniform_disk_series`)
and the blur is the exact :func:`~astroviper.processing_functions.simulation.component_series.blur_transform`.
In the tangent plane this reduces to the familiar difference of two Airy
patterns, ``[R^2 2J_1(x)/x - (1 - f) R_in^2 e^{2 pi i k.a} 2J_1(x_in)/x_in] /
(R^2 - (1 - f) R_in^2)``.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.component_series import (
    blur_transform,
    blurred_radial_profile_image,
    uniform_disk_profile_rule,
    uniform_disk_series,
)
from astroviper.processing_functions.simulation.gaussian_source import FWHM_TO_SIGMA


def _check_crescent(radius, inner_radius, offset, floor, fwhm):
    if radius <= 0 or inner_radius < 0 or inner_radius >= radius:
        raise ValueError(
            "crescent needs 0 <= inner_radius < radius; "
            f"got radius={radius}, inner_radius={inner_radius}."
        )
    if offset < 0 or offset > radius - inner_radius:
        raise ValueError(
            "crescent offset must be in [0, radius - inner_radius] (the hole stays "
            f"inside the disk); got {offset}."
        )
    if not (0.0 <= floor <= 1.0):
        raise ValueError(f"crescent floor must be in [0, 1]; got {floor}.")
    if fwhm < 0:
        raise ValueError(f"crescent fwhm must be non-negative; got {fwhm}.")


def _displacement(offset, pa):
    """Displacement ``(a_l, a_m)`` of the hole: towards ``pa + pi`` (away from the bright side)."""
    return -offset * np.sin(pa), -offset * np.cos(pa)


def crescent_uv_response(
    u,
    v,
    radius,
    inner_radius,
    offset,
    pa=0.0,
    floor=0.0,
    fwhm=0.0,
    w=0.0,
    tolerance=1e-14,
) -> np.ndarray:
    """Normalised visibility of a unit-flux (blurred) crescent, with the w term.

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component; ``w = 0``
        (default) is the tangent-plane approximation.
    radius : float, radians
        Radius of the outer uniform disk.
    inner_radius : float, radians
        Radius of the removed (or dimmed) inner disk, ``< radius``.
    offset : float, radians
        Displacement of the inner disk from the centre,
        ``0 <= offset <= radius - inner_radius``.
    pa : float, radians
        Position angle of the bright (thick) side of the crescent, from ``+m``
        towards ``+l``.
    floor : float
        Brightness inside the inner disk as a fraction of the crescent
        brightness (``0``: empty hole, ``1``: a uniform disk).
    fwhm : float, radians
        FWHM of the circular Gaussian the crescent is convolved with.
    tolerance : float
        Truncation tolerance of the disk series.

    Returns
    -------
    numpy.ndarray of complex, broadcast shape of ``u``, ``v`` and ``w``.
    """
    radius = float(radius)
    inner_radius = float(inner_radius)
    offset = float(offset)
    floor = float(floor)
    fwhm = float(fwhm)
    _check_crescent(radius, inner_radius, offset, floor, fwhm)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    k1, k2, w1, w2, prefactor = blur_transform(u, v, w, w, sigma2)
    outer = radius**2 * uniform_disk_series(k1, k2, w1, w2, radius, 1.0, tolerance)
    weight_inner = (1.0 - floor) * inner_radius**2
    if weight_inner > 0.0:
        a_l, a_m = _displacement(offset, pa)
        shift = np.exp(
            2j * np.pi * (k1 * a_l + k2 * a_m) - 1j * np.pi * w1 * (a_l**2 + a_m**2)
        )
        inner = shift * uniform_disk_series(
            k1 - w1 * a_l, k2 - w2 * a_m, w1, w2, inner_radius, 1.0, tolerance
        )
        outer = outer - weight_inner * inner
    return prefactor * outer / (radius**2 - weight_inner)


def crescent_image(l, m, radius, inner_radius, offset, pa=0.0, floor=0.0, fwhm=0.0):  # noqa: E741
    """Unit-total-flux surface brightness of a (blurred) crescent (per steradian).

    Parameters as in :func:`crescent_uv_response`; ``l, m`` are sky offsets
    from the crescent centre (the centre of the outer disk).
    """
    radius = float(radius)
    inner_radius = float(inner_radius)
    offset = float(offset)
    floor = float(floor)
    fwhm = float(fwhm)
    _check_crescent(radius, inner_radius, offset, floor, fwhm)
    l = np.asarray(l, dtype=np.float64)  # noqa: E741
    m = np.asarray(m, dtype=np.float64)
    a_l, a_m = _displacement(offset, pa)
    r_outer = np.sqrt(l**2 + m**2)
    r_inner = np.sqrt((l - a_l) ** 2 + (m - a_m) ** 2)
    weight_inner = (1.0 - floor) * inner_radius**2
    norm = np.pi * (radius**2 - weight_inner)
    if fwhm == 0.0:
        image = (r_outer < radius).astype(np.float64)
        if weight_inner > 0.0:
            image = image - (1.0 - floor) * (r_inner < inner_radius)
        return image / norm
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    nodes, values = uniform_disk_profile_rule(radius, sigma2)
    image = blurred_radial_profile_image(r_outer, nodes, values, sigma2)
    if weight_inner > 0.0:
        nodes, values = uniform_disk_profile_rule(inner_radius, sigma2)
        image = image - (1.0 - floor) * blurred_radial_profile_image(
            r_inner, nodes, values, sigma2
        )
    return image / norm
