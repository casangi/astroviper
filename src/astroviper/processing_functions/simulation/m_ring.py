"""m-ring sky component (a thin ring with azimuthal Fourier modes), with the w term.

The *m-ring* of the Event Horizon Telescope literature (Johnson et al. 2020;
EHT Collaboration 2019, Paper IV) is an infinitesimally thin ring of radius
``R`` whose brightness around the ring is a truncated Fourier series in the
azimuth ``phi`` (measured from ``+m``, north, towards ``+l``, east)::

    I(phi) = F / (2 pi R) * [1 + sum_{m=1}^{M} 2 Re(beta_m exp(i m phi))]

``beta_1`` sets the brightness asymmetry, ``beta_2`` an elliptical modulation,
and so on.  Since ``2 Re(beta_m exp(i m phi)) = 2 |beta_m| cos(m phi + arg
beta_m)``, the maximum of the ``m``-th mode lies at position angle
``-arg(beta_m) / m``; in particular the bright side of the ring is at ``phi =
-arg(beta_1)`` (from ``+m`` towards ``+l``, in the unstretched image).  This is
the convention of ``eht-imaging`` (its ``beta_list``), so published m-ring fits
can be used unchanged.  The ring
is optionally convolved with a circular Gaussian of FWHM ``fwhm`` (the
"thick m-ring") and inclined by ``inclination`` with the major axis at
position angle ``pa`` (the "stretched" m-ring; the whole blurred image is
stretched, as in ``eht-imaging``).  ``beta = []`` is the (thick) ring.

In the tangent plane and face on the visibility is the classic

    V = F exp(-2 pi^2 sigma^2 rho^2) sum_m beta_m i^m J_m(2 pi R rho) exp(i m phi_uv),

``beta_0 = 1``, ``beta_{-m} = conj(beta_m)`` and ``phi_uv`` the azimuth of
``(u, v)`` (from ``+v`` towards ``+u``).  With the w term (paraxial phase
``exp(-i pi (w_1 x^2 + w_2 y^2))`` in the face-on frame, ``w_1 = w``,
``w_2 = q^2 w``) and the blur (:func:`component_series.blur_transform`) the
azimuthal integral over the ring becomes a Neumann series::

    T = sqrt(g_1 g_2) exp(D) sum_m beta_m sum_n I_n(C) i^(m+2n) J_{m+2n}(X) E^(m+2n)

    D = -i pi R^2 (w_1 g_1 + w_2 g_2) / 2 - 2 pi^2 sigma^2 (g_1 k_1^2 + g_2 k_2^2)
    C = -i pi R^2 (w_1 g_1 - w_2 g_2) / 2
    X = 2 pi R sqrt(g_1^2 k_1^2 + g_2^2 k_2^2),   E = 2 pi R (g_1 k_1 + i g_2 k_2) / X

with ``(k_1, k_2) = (u_maj, q u_min)`` and ``g_j = 1 / (1 + 2 pi i w_j
sigma^2)``; face on (``C = 0``) only ``n = 0`` survives and the series is the
closed form above with complex ``R g`` and ``sigma^2 g``; for ``beta = []``
it is the Gaussian ring of :mod:`~astroviper.processing_functions.simulation.gaussian_ring`.
The unit tests pin the series to brute-force integration.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.component_series import (
    bessel_over_power,
    deprojected_baseline,
    deprojected_sky,
    gaussian_width_factor,
)
from astroviper.processing_functions.simulation.gaussian_source import FWHM_TO_SIGMA

# Terms of the Neumann series in I_n(C): O((|C|/2)^n / n!).
_MAX_N_TERMS = 40


def _check_m_ring(radius, fwhm, inclination, beta):
    if radius < 0 or fwhm < 0:
        raise ValueError(
            f"m-ring radius and fwhm must be non-negative; got radius={radius}, fwhm={fwhm}."
        )
    if not (0.0 <= inclination < np.pi / 2):
        raise ValueError(
            f"m-ring inclination must be in [0, pi/2) radians; got {inclination}."
        )
    beta = np.atleast_1d(np.asarray(beta, dtype=np.complex128)).ravel()
    if not np.all(np.isfinite(beta)):
        raise ValueError("m-ring beta coefficients must be finite.")
    return beta


def m_ring_uv_response(
    u,
    v,
    radius,
    beta,
    fwhm=0.0,
    inclination=0.0,
    pa=0.0,
    w=0.0,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Normalised visibility of a unit-flux (thick, stretched) m-ring, with the w term.

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component (see
        :func:`~astroviper.processing_functions.simulation.calculate_visibilities.source_frame_uvw`);
        ``w = 0`` (default) is the tangent-plane approximation.
    radius : float, radians
        Radius of the thin ring (in the face-on frame; the projected major
        axis has this radius).
    beta : array_like of complex, [M]
        Azimuthal Fourier coefficients ``beta_1 .. beta_M`` of the ring
        brightness ``1 + sum 2 Re(beta_m exp(i m phi))`` (the bright side is
        at position angle ``-arg(beta_1)``); ``[]`` is a plain ring.
        ``|beta_1| < 1/2`` keeps the ring brightness positive.
    fwhm : float, radians
        FWHM of the circular Gaussian the ring is convolved with (``0``: thin).
    inclination : float, radians
        ``0`` face-on; the minor axis is ``cos(inclination)`` times the major axis.
    pa : float, radians
        Position angle of the major axis, from ``+m`` towards ``+l``.
    tolerance : float
        Truncation tolerance of the inclination series.

    Returns
    -------
    numpy.ndarray of complex, broadcast shape of ``u``, ``v`` and ``w``.
    """
    from scipy.special import iv

    radius = float(radius)
    fwhm = float(fwhm)
    inclination = float(inclination)
    beta = _check_m_ring(radius, fwhm, inclination, beta)
    q = float(np.cos(inclination))
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    shape = np.broadcast(u, v, w).shape
    w = np.broadcast_to(np.asarray(w, dtype=np.float64), shape)
    k1, k2 = deprojected_baseline(u, v, pa, q)
    k1 = np.broadcast_to(k1, shape)
    k2 = np.broadcast_to(k2, shape)
    w1 = w
    w2 = q * q * w
    g1 = gaussian_width_factor(w1, sigma2)
    g2 = gaussian_width_factor(w2, sigma2)
    two_pi2 = 2.0 * np.pi**2
    d_exponent = -0.5j * np.pi * radius**2 * (w1 * g1 + w2 * g2) - two_pi2 * sigma2 * (
        g1 * k1**2 + g2 * k2**2
    )
    coupling = -0.5j * np.pi * radius**2 * (w1 * g1 - w2 * g2)  # C
    z_plus = 2.0 * np.pi * radius * (g1 * k1 + 1j * g2 * k2)  # X E
    z_minus = 2.0 * np.pi * radius * (g1 * k1 - 1j * g2 * k2)  # X / E
    x_arg = np.sqrt(z_plus * z_minus + 0j)  # X
    prefactor = np.sqrt(g1 * g2) * np.exp(d_exponent)

    # The mode phases are defined in the sky frame (azimuth from +m towards +l)
    # while the series runs in the frame of the major axis: rotate by pa.  Full
    # coefficient list beta_m for m = -M .. M.
    beta = beta * np.exp(1j * np.arange(1, len(beta) + 1) * float(pa))
    orders = np.arange(-len(beta), len(beta) + 1)
    coefficients = np.concatenate([np.conj(beta[::-1]), [1.0 + 0j], beta])

    max_coupling = float(np.max(np.abs(coupling))) if coupling.size else 0.0
    n_max = 0
    if max_coupling > 0.0:
        from math import factorial

        while (
            n_max < _MAX_N_TERMS
            and (max_coupling / 2.0) ** (n_max + 1) / factorial(n_max + 1) > tolerance
        ):
            n_max += 1

    def harmonic(order):
        """i^order J_order(X) E^order, finite at X = 0 (J_p(X) E^p = J_p(X)/X^p * (X E)^p)."""
        p = abs(order)
        ratio = bessel_over_power(p, x_arg, p)  # J_p(X) / X^p
        z = z_plus if order >= 0 else z_minus
        return (1j**order) * ratio * z**p * ((-1) ** p if order < 0 else 1.0)

    series = np.zeros(shape, dtype=np.complex128)
    for n in range(-n_max, n_max + 1):
        i_n = iv(abs(n), coupling) if n_max > 0 else np.ones(shape)
        for m, coefficient in zip(orders, coefficients, strict=True):
            series += coefficient * i_n * harmonic(m + 2 * n)
    return prefactor * series


def m_ring_image(l, m, radius, beta, fwhm, inclination=0.0, pa=0.0) -> np.ndarray:  # noqa: E741
    """Unit-total-flux surface brightness of a thick (stretched) m-ring.

    ``I = exp(-(r - R)^2 / 2 sigma^2) / (2 pi sigma^2 q) [ive(0, z) + sum_m 2
    Re(beta_m exp(i m phi)) ive(m, z)]`` with ``z = r R / sigma^2`` and
    ``(r, phi)`` the polar coordinates in the face-on frame (``phi`` from the
    ``+m`` axis towards ``+l`` when ``pa = 0``).  ``fwhm`` must be positive.
    """
    from scipy.special import ive

    radius = float(radius)
    fwhm = float(fwhm)
    inclination = float(inclination)
    beta = _check_m_ring(radius, fwhm, inclination, beta)
    if fwhm == 0.0:
        raise ValueError(
            "m_ring_image needs fwhm > 0 (a thin ring has no finite surface brightness)."
        )
    q = float(np.cos(inclination))
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    x, y = deprojected_sky(l, m, pa, q)  # x along the major axis (pa direction)
    r = np.sqrt(x**2 + y**2)
    # azimuth from the +m axis towards +l when pa = 0: pa rotates the frame, so
    # measure phi from the major axis (x) towards the direction of -y ... i.e.
    # phi = pa + atan2(y_sky, x_sky) with (x, y) = (l_maj, l_min): l = x sin pa + y cos pa,
    # m = x cos pa - y sin pa -> atan2(l, m) = pa + atan2(y, x)
    phi = float(pa) + np.arctan2(y, x)
    z = r * radius / sigma2
    azimuthal = ive(0, z)
    for order, coefficient in enumerate(beta, start=1):
        azimuthal = azimuthal + 2.0 * np.real(
            coefficient * np.exp(1j * order * phi)
        ) * ive(order, z)
    return (
        np.exp(-0.5 * (r - radius) ** 2 / sigma2)
        / (2.0 * np.pi * sigma2 * q)
        * azimuthal
    )
