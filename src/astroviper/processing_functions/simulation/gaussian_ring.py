"""Gaussian-broadened ring ("Gaussian disk") sky component, with the w term.

The component of the simulation memo *Analytic visibility model for
Gaussian-broadened rings* (version 2): an infinitesimally thin circular ring of
radius ``R`` and total flux ``F``, convolved with a circular Gaussian of width
``sigma`` (``fwhm = 2 sqrt(2 ln 2) sigma``), inclined by ``i`` (``q = cos i``)
and rotated to position angle ``pa``.  ``R = 0`` is a circular Gaussian; a set
of nested rings with independent radii, widths, inclinations and position
angles models a protoplanetary disk.

Face-on and in the tangent plane (``w = 0``) the visibility is
``F J_0(2 pi R rho) exp(-2 pi^2 sigma^2 rho^2)``; inclination replaces ``rho``
by the deprojected ``rho_ell = sqrt(u_maj^2 + q^2 u_min^2)``.  With the w term
(paraxial expansion ``w (n - 1) ~ -w r^2 / 2`` in the component frame, the
``exp(+2 pi i)`` sign convention of ``calculate_visibilities``) the quadratic
phase becomes anisotropic in the face-on frame, ``exp(-i pi (w_1 x^2 + w_2
y^2))`` with ``w_1 = w`` and ``w_2 = q^2 w``, and the azimuthal integral over
the ring is a Neumann series::

    T = sqrt(g_1 g_2) exp(D) [ I_0(C) J_0(X)
                               + 2 sum_{n>=1} (-1)^n I_n(C) J_{2n}(X) T_n(c) ]

    g_j = 1 / (1 + 2 pi i w_j sigma^2)               complex Gaussian widths
    D   = -i pi R^2 (w_1 g_1 + w_2 g_2) / 2 - 2 pi^2 sigma^2 (g_1 k_1^2 + g_2 k_2^2)
    C   = -i pi R^2 (w_1 g_1 - w_2 g_2) / 2          inclination coupling
    X   = 2 pi R sqrt(g_1^2 k_1^2 + g_2^2 k_2^2)
    c   = (g_1^2 k_1^2 - g_2^2 k_2^2) / (g_1^2 k_1^2 + g_2^2 k_2^2)

with ``(k_1, k_2) = (u_maj, q u_min)``, ``T_n`` the Chebyshev polynomials
(``cos 2n phi_0``) and ``I_n`` modified Bessel functions.  Face on ``C = 0``
and the series collapses to the closed form
``g exp(-i pi w R^2 g) exp(-2 pi^2 sigma^2 g rho^2) J_0(2 pi R g rho)``; at
``w = 0`` every ``g = 1`` and the memo's tangent-plane formula is recovered.
The unit tests pin this series to brute-force integration of the brightness
with the quadratic phase.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.simulation.gaussian_source import FWHM_TO_SIGMA

# Terms of the Neumann series in I_n(C): the n-th term is O((|C|/2)^n / n!), so
# this many terms keep the truncation below double precision for |C| < ~15.
_MAX_TERMS = 40


def _check_ring_shape(radius, fwhm, inclination):
    if radius < 0 or fwhm < 0:
        raise ValueError(
            f"ring radius and fwhm must be non-negative; got radius={radius}, fwhm={fwhm}."
        )
    if not (0.0 <= inclination < np.pi / 2):
        raise ValueError(
            f"ring inclination must be in [0, pi/2) radians (face-on to edge-on); got {inclination}."
        )


def gaussian_ring_uv_response(
    u, v, radius, fwhm, inclination, pa, w=0.0, tolerance: float = 1e-14
) -> np.ndarray:
    """Normalised visibility of a unit-flux Gaussian-broadened ring, with the w term.

    Multiplying the visibilities of a point source by this response turns the
    point source into the ring (the response is ``1`` at ``u = v = w = 0``).

    Parameters
    ----------
    u, v, w : numpy.ndarray (broadcastable), wavelengths
        Baseline coordinates in the frame of the component (see
        :func:`~astroviper.processing_functions.simulation.calculate_visibilities.source_frame_uvw`).
        ``w = 0`` (default) is the tangent-plane approximation.
    radius : float, radians
        Radius of the thin ring before broadening; ``0`` gives a circular Gaussian.
    fwhm : float, radians
        FWHM of the circular Gaussian the ring is convolved with (the FWHM of
        the radial cross-section of a narrow ring).
    inclination : float, radians
        Inclination, ``0`` face-on; the minor axis is ``cos(inclination)`` times
        the major axis.
    pa : float, radians
        Position angle of the major axis, measured from the ``+m`` axis towards
        the ``+l`` axis (the clean-beam / Gaussian-source convention).
    tolerance : float, optional
        Truncation tolerance of the inclination series.

    Returns
    -------
    numpy.ndarray
        Broadcast shape of ``u``, ``v`` and ``w``; real when ``w`` is
        identically zero, complex otherwise.

    See Also
    --------
    gaussian_ring_image : the image-plane form (unit total flux).
    """
    from scipy.special import iv, jv

    radius = float(radius)
    fwhm = float(fwhm)
    inclination = float(inclination)
    _check_ring_shape(radius, fwhm, inclination)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    q = float(np.cos(inclination))
    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    k1 = u * sin_pa + v * cos_pa  # along the major axis
    k2 = q * (u * cos_pa - v * sin_pa)  # deprojected minor-axis coordinate
    two_pi2 = 2.0 * np.pi**2

    if not np.any(w):  # tangent plane: the memo's closed form
        rho2 = k1**2 + k2**2
        return np.exp(-two_pi2 * sigma2 * rho2) * jv(
            0, 2.0 * np.pi * radius * np.sqrt(rho2)
        )

    w1 = w
    w2 = q * q * w
    g1 = 1.0 / (1.0 + 2j * np.pi * w1 * sigma2)
    g2 = 1.0 / (1.0 + 2j * np.pi * w2 * sigma2)
    d_exponent = -0.5j * np.pi * radius**2 * (w1 * g1 + w2 * g2) - two_pi2 * sigma2 * (
        g1 * k1**2 + g2 * k2**2
    )
    coupling = -0.5j * np.pi * radius**2 * (w1 * g1 - w2 * g2)  # C
    s1 = (g1 * k1) ** 2
    s2 = (g2 * k2) ** 2
    total = s1 + s2
    x_arg = 2.0 * np.pi * radius * np.sqrt(total + 0j)
    prefactor = np.sqrt(g1 * g2) * np.exp(d_exponent)

    series = iv(0, coupling) * jv(0, x_arg)
    max_coupling = float(np.max(np.abs(coupling))) if coupling.size else 0.0
    if max_coupling > 0.0:  # inclined ring: add the cos(2 n phi) harmonics
        with np.errstate(divide="ignore", invalid="ignore"):
            cos2phi = np.where(
                np.abs(total) > 0.0, (s1 - s2) / np.where(total == 0, 1, total), 0.0
            )
        t_prev = np.ones_like(cos2phi)  # T_0
        t_curr = cos2phi  # T_1
        for n in range(1, _MAX_TERMS + 1):
            term = 2.0 * (-1) ** n * iv(n, coupling) * jv(2 * n, x_arg) * t_curr
            series = series + term
            if float(np.max(np.abs(term))) < tolerance:
                break
            t_prev, t_curr = t_curr, 2.0 * cos2phi * t_curr - t_prev
    return prefactor * series


def gaussian_ring_image(l, m, radius, fwhm, inclination, pa) -> np.ndarray:
    """Unit-total-flux surface brightness of a Gaussian-broadened ring.

    Evaluates ``I(l, m)`` (per steradian) at sky offsets ``(l, m)`` from the
    ring centre: the memo's radial profile
    ``exp(-(r^2 + R^2) / (2 sigma^2)) I_0(r R / sigma^2) / (2 pi sigma^2)`` in
    the deprojected radius ``r = sqrt(l_maj^2 + (l_min / q)^2)``, divided by
    ``q = cos(inclination)`` so the projected component keeps unit flux.

    Parameters
    ----------
    l, m : numpy.ndarray (broadcastable), radians
        Direction cosines (sky offsets) relative to the ring centre.
    radius, fwhm, inclination, pa : float, radians
        As in :func:`gaussian_ring_uv_response` (``fwhm > 0``).

    Returns
    -------
    numpy.ndarray
        Broadcast shape of ``l`` and ``m``.
    """
    from scipy.special import i0e

    radius = float(radius)
    fwhm = float(fwhm)
    inclination = float(inclination)
    _check_ring_shape(radius, fwhm, inclination)
    if fwhm == 0.0:
        raise ValueError(
            "gaussian_ring_image needs fwhm > 0 (a thin ring has no finite surface brightness)."
        )
    l = np.asarray(l, dtype=np.float64)  # noqa: E741
    m = np.asarray(m, dtype=np.float64)
    sigma2 = (fwhm * FWHM_TO_SIGMA) ** 2
    q = float(np.cos(inclination))
    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    l_major = l * sin_pa + m * cos_pa
    l_minor = l * cos_pa - m * sin_pa
    r = np.sqrt(l_major**2 + (l_minor / q) ** 2)
    # exp(-(r^2 + R^2) / 2 sigma^2) I0(r R / sigma^2) = exp(-(r - R)^2 / 2 sigma^2) i0e(r R / sigma^2)
    radial = np.exp(-0.5 * (r - radius) ** 2 / sigma2) * i0e(r * radius / sigma2)
    return radial / (2.0 * np.pi * sigma2 * q)
