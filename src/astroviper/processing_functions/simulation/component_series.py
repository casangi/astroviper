"""Shared numerical kernels of the analytic sky components (w-term series).

Every extended component of the simulator is a unit-flux brightness ``b(l, m)``
whose visibility in the frame of the component, to second order in the
direction cosines, is

.. math::

    T(u, v, w) = \\int b(l, m)\\, e^{-i \\pi w (l^2 + m^2)}\\,
                 e^{2 \\pi i (u l + v m)}\\, dl\\, dm

(the ``exp(+2 pi i)`` convention of ``calculate_visibilities``).  Two building
blocks recur in the closed forms of the simulation memo (version 2) and are
implemented here once, for **complex** arguments so that Gaussian blurring
(which turns baselines and ``w`` into complex quantities, see
:func:`blur_transform`) and inclination (which makes the quadratic phase
anisotropic in the face-on frame) can be composed freely:

* :func:`uniform_disk_series` -- the limb-darkened elliptical disk of the
  memo (Sonine / Jacobi-Anger triple series in the isotropic Fresnel
  parameter ``beta`` and the inclination coupling ``delta``), used by the
  disk, annulus and crescent components;
* :func:`blur_transform` -- the exact effect of convolving any component with
  a circular Gaussian: the response of the blurred component equals the
  response of the sharp one evaluated at the complex baseline ``g k`` and the
  complex ``w' = g w``, times ``g exp(-2 pi^2 sigma^2 g rho^2)`` with
  ``g = 1 / (1 + 2 pi i w sigma^2)`` (per axis when the phase is anisotropic).

The Gaussian scale mixtures (:func:`gaussian_mixture_response`) evaluate the
w-term response of profiles that are superpositions of Gaussians of different
widths (exponential, tapered power law) as the same superposition of the
closed-form elliptical-Gaussian responses, with Gauss-Jacobi / Gauss-Laguerre
quadrature over the width.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Below this |argument| the Bessel ratios are replaced by their leading power
# series term (exact to double precision there); also handles x == 0.
SERIES_X_MAX = 1e-4
# Caps on the adaptive truncation of the disk series (see uniform_disk_series).
MAX_N_TERMS = 60
MAX_P_TERMS = 12
MAX_S_TERMS = 12


def gaussian_width_factor(w, sigma2):
    """``g = 1 / (1 + 2 pi i w sigma^2)``: the complex-width factor of a Gaussian with the w term."""
    return 1.0 / (1.0 + 2j * np.pi * np.asarray(w) * sigma2)


def bessel_over_power(order, x, power):
    """``J_order(x) / x**power`` for complex ``x`` with the power-series limit where ``|x|`` is tiny.

    ``order - power`` is a non-negative even integer in every use here, so the
    small-``x`` limit ``(x/2)**order / (Gamma(order + 1) x**power)`` is finite.
    """
    from scipy.special import gamma, jv

    x = np.asarray(x)
    out = np.empty(x.shape, dtype=np.complex128)
    small = np.abs(x) < SERIES_X_MAX
    large = ~small
    if large.any():
        out[large] = jv(order, x[large]) / x[large] ** power
    if small.any():
        xs = x[small]
        out[small] = xs ** (order - power) / (2.0**order * gamma(order + 1.0))
    return out


def uniform_disk_series(k1, k2, w1, w2, radius, nu, tolerance=1e-14):
    """Response of a unit-flux limb-darkened circular disk with an anisotropic quadratic phase.

    Evaluates, in the face-on frame of the disk (radius ``radius``, brightness
    ``(1 - r^2 / a^2)^(nu - 1)``),

    .. math::

        T = \\int b(x, y)\\, e^{-i \\pi (w_1 x^2 + w_2 y^2)}\\,
            e^{2 \\pi i (k_1 x + k_2 y)}\\, dx\\, dy

    by the series of the simulation memo (version 2, appendix C)::

        T = sum_p eps_p (-i)^p (-1)^p cos(2 p phi_k)
            sum_s (-1)^s (delta/2)^(p+2s) / (s! (p+s)!)
            sum_n (-i beta)^n / n!
            sum_j C(N-p, j) (-1)^j 2^(nu+j) Gamma(nu+1) (nu)_j J_{2p+nu+j}(x) / x^(nu+j)

    with ``beta = pi a^2 (w_1 + w_2) / 2``, ``delta = pi a^2 (w_1 - w_2) / 2``,
    ``N = n + p + 2 s``, ``x = 2 pi a sqrt(k_1^2 + k_2^2)`` and ``phi_k`` the
    azimuth of ``(k_1, k_2)``.  All inputs may be complex (Gaussian blurring,
    :func:`blur_transform`); the series is truncated adaptively.

    Parameters
    ----------
    k1, k2 : array_like, wavelengths (real or complex, broadcastable)
        Baseline coordinates along the face-on axes.
    w1, w2 : array_like (real or complex, broadcastable)
        Quadratic-phase parameters along the face-on axes (``w`` and
        ``q^2 w`` for an inclined disk of axis ratio ``q``).
    radius : float, radians
        Radius of the disk in the face-on frame.
    nu : float
        ``limb_darkening / 2 + 1`` (``nu = 1`` uniform disk).
    tolerance : float
        Truncation tolerance of the series.

    Returns
    -------
    numpy.ndarray of complex, broadcast shape of the inputs.
    """
    from math import comb, factorial

    from scipy.special import gamma, poch

    shape = np.broadcast(k1, k2, w1, w2).shape
    k1 = np.broadcast_to(np.asarray(k1), shape).astype(np.complex128).ravel()
    k2 = np.broadcast_to(np.asarray(k2), shape).astype(np.complex128).ravel()
    w1 = np.broadcast_to(np.asarray(w1), shape).astype(np.complex128).ravel()
    w2 = np.broadcast_to(np.asarray(w2), shape).astype(np.complex128).ravel()
    a = float(radius)
    if a == 0.0:  # a point: no extent, no phase
        return np.ones(shape, dtype=np.complex128)

    kappa2 = k1**2 + k2**2
    x = 2.0 * np.pi * a * np.sqrt(kappa2)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos2phi = np.where(np.abs(kappa2) > 0.0, (k1**2 - k2**2) / np.where(kappa2 == 0, 1, kappa2), 0.0)  # fmt: skip
    beta = np.pi * a**2 * (w1 + w2) / 2.0  # isotropic Fresnel parameter
    delta = np.pi * a**2 * (w1 - w2) / 2.0  # anisotropy (inclination) coupling

    beta_max = float(np.max(np.abs(beta)))
    delta_max = float(np.max(np.abs(delta)))
    n_max = 0
    while n_max < MAX_N_TERMS and beta_max ** (n_max + 1) / factorial(n_max + 1) > tolerance:  # fmt: skip
        n_max += 1
    p_max = 0
    while p_max < MAX_P_TERMS and (delta_max / 2.0) ** (p_max + 1) / factorial(p_max + 1) > tolerance:  # fmt: skip
        p_max += 1
    s_max = 0
    while s_max < MAX_S_TERMS and (delta_max / 2.0) ** (2 * s_max + 2) / factorial(s_max + 1) ** 2 > tolerance:  # fmt: skip
        s_max += 1
    j_max = n_max + 2 * s_max

    # B_n = (-i beta)^n / n! and the binomial sums S[s][j] = sum_n B_n C(n + 2s, j)
    b_n = [np.ones(x.shape, dtype=np.complex128)]
    for n in range(1, n_max + 1):
        b_n.append(b_n[-1] * (-1j * beta) / n)
    s_sums = np.zeros((s_max + 1, j_max + 1) + x.shape, dtype=np.complex128)
    for s_idx in range(s_max + 1):
        for n in range(n_max + 1):
            for j in range(min(j_max, n + 2 * s_idx) + 1):
                s_sums[s_idx, j] += comb(n + 2 * s_idx, j) * b_n[n]

    # Chebyshev T_p(cos 2 phi_k): (t_prev, t_curr) = (T_{p-1}, T_p), seeded with
    # T_{-1} = T_1 so that the recurrence produces T_1 after p = 0.
    t_prev = cos2phi.copy()
    t_curr = np.ones_like(cos2phi)
    gamma_nu1 = gamma(nu + 1.0)
    response = np.zeros(x.shape, dtype=np.complex128)
    for p in range(p_max + 1):
        harmonic = (1.0 if p == 0 else 2.0) * ((-1j) ** p) * ((-1) ** p) * t_curr
        a_ps = [
            (-1) ** s_idx * (delta / 2.0) ** (p + 2 * s_idx) / (factorial(s_idx) * factorial(p + s_idx))
            for s_idx in range(s_max + 1)
        ]  # fmt: skip
        radial = np.zeros(x.shape, dtype=np.complex128)
        for j in range(j_max + 1):
            weight = np.zeros(x.shape, dtype=np.complex128)
            for s_idx in range(s_max + 1):
                weight += a_ps[s_idx] * s_sums[s_idx, j]
            if not np.any(weight):
                continue
            weight *= (-1) ** j
            order = 2 * p + nu + j
            radial += weight * (
                2.0 ** (nu + j) * gamma_nu1 * poch(nu, j) * bessel_over_power(order, x, nu + j)
            )  # fmt: skip
        response += harmonic * radial
        t_prev, t_curr = t_curr, 2.0 * cos2phi * t_curr - t_prev
    return response.reshape(shape)


def blur_transform(k1, k2, w1, w2, sigma2):
    """Complex baseline, ``w`` and prefactor that turn a sharp response into a Gaussian-blurred one.

    Convolving a component with a circular Gaussian of variance ``sigma2`` (in
    its face-on frame) and keeping the quadratic phase
    ``exp(-i pi (w_1 x^2 + w_2 y^2))`` gives, exactly,

    .. math::

        T_{\\rm blurred}(k_1, k_2; w_1, w_2)
        = \\sqrt{g_1 g_2}\\, e^{-2\\pi^2\\sigma^2 (g_1 k_1^2 + g_2 k_2^2)}\\;
          T_{\\rm sharp}(g_1 k_1, g_2 k_2;\\, g_1 w_1, g_2 w_2),
        \\qquad g_j = \\frac{1}{1 + 2\\pi i w_j \\sigma^2}

    (complete the square in the convolution integral; the identity
    ``1 - 2 pi i sigma^2 w g = g`` collapses the cross terms).  With
    ``sigma2 = 0`` the transform is the identity.

    Returns
    -------
    k1_blur, k2_blur, w1_blur, w2_blur, prefactor : numpy.ndarray of complex
    """
    g1 = gaussian_width_factor(w1, sigma2)
    g2 = gaussian_width_factor(w2, sigma2)
    prefactor = np.sqrt(g1 * g2) * np.exp(
        -2.0 * np.pi**2 * sigma2 * (g1 * np.asarray(k1) ** 2 + g2 * np.asarray(k2) ** 2)
    )
    return g1 * k1, g2 * k2, g1 * w1, g2 * w2, prefactor


@lru_cache(maxsize=32)
def _gauss_jacobi(n, alpha, beta):
    """Nodes and weights of ``int_0^1 (1 - x)^alpha x^beta f(x) dx``."""
    from scipy.special import roots_jacobi

    with np.errstate(
        invalid="ignore", divide="ignore"
    ):  # 0/0 guarded inside scipy for a + b = -1
        nodes, weights = roots_jacobi(n, alpha, beta)  # on [-1, 1] with (1-t)^a (1+t)^b
    # t = 2 x - 1: (1 - t)^a (1 + t)^b dt = 2^(a + b + 1) (1 - x)^a x^b dx
    return (nodes + 1.0) / 2.0, weights / 2.0 ** (alpha + beta + 1)


@lru_cache(maxsize=32)
def _gauss_laguerre(n, alpha):
    """Nodes and weights of ``int_0^inf x^alpha e^-x f(x) dx``."""
    from scipy.special import roots_genlaguerre

    return roots_genlaguerre(n, alpha)


def gaussian_mixture_response(k1, k2, w1, w2, sigma2_of_x, nodes, weights):
    """Superposition of elliptical-Gaussian responses over a width-mixture quadrature.

    Evaluates ``sum_i weights_i sqrt(g_1 g_2) exp(-2 pi^2 sigma_i^2 (g_1 k_1^2
    + g_2 k_2^2))`` with ``sigma_i^2 = sigma2_of_x(nodes_i)`` and
    ``g_j = 1 / (1 + 2 pi i w_j sigma_i^2)``; the face-on-frame axis ratio is
    already folded into ``k_2`` and ``w_2``.

    Parameters
    ----------
    k1, k2, w1, w2 : numpy.ndarray (broadcastable)
    sigma2_of_x : callable
        Variance of the mixture Gaussian as a function of the quadrature node.
    nodes, weights : numpy.ndarray, [n_nodes]

    Returns
    -------
    numpy.ndarray of complex, broadcast shape of the baseline inputs.
    """
    shape = np.broadcast(k1, k2, w1, w2).shape
    k1 = np.broadcast_to(np.asarray(k1, dtype=np.float64), shape)[..., None]
    k2 = np.broadcast_to(np.asarray(k2, dtype=np.float64), shape)[..., None]
    w1 = np.broadcast_to(np.asarray(w1, dtype=np.float64), shape)[..., None]
    w2 = np.broadcast_to(np.asarray(w2, dtype=np.float64), shape)[..., None]
    sigma2 = sigma2_of_x(np.asarray(nodes, dtype=np.float64))  # [n_nodes]
    g1 = gaussian_width_factor(w1, sigma2)
    g2 = gaussian_width_factor(w2, sigma2)
    terms = np.sqrt(g1 * g2) * np.exp(
        -2.0 * np.pi**2 * sigma2 * (g1 * k1**2 + g2 * k2**2)
    )
    return np.sum(terms * np.asarray(weights), axis=-1)


def exponential_mixture_quadrature(n_nodes: int = 160):
    """Nodes/weights of the Gaussian scale mixture of ``exp(-r / r_s)``.

    ``exp(-r / r_s) / (2 pi r_s^2) = int_0^inf (2 / sqrt(pi)) s^(1/2) e^-s
    N(r; sigma^2 = 2 r_s^2 s) ds`` (``N`` a unit-flux circular Gaussian), so
    the response is the generalised Gauss-Laguerre (``alpha = 1/2``) sum of
    Gaussian responses; the ``2 / sqrt(pi)`` normalisation is folded into the
    weights.  Used with the scaled variable ``t = (1 + K) s`` in
    :func:`exponential_disk_series` so that the tangent-plane limit is exact.
    """
    nodes, weights = _gauss_laguerre(n_nodes, 0.5)
    return nodes, weights * 2.0 / np.sqrt(np.pi)


def exponential_disk_series(k1, k2, w1, w2, scale_radius, n_nodes: int = 160):
    """Response of the unit-flux exponential profile ``exp(-r / r_s)`` with the w term (face-on frame).

    The scale mixture of :func:`exponential_mixture_quadrature` is integrated in
    the variable ``t = (1 + K) s`` with ``K = 4 pi^2 r_s^2 (k_1^2 + k_2^2)``
    the tangent-plane decay rate, so that for ``w = 0`` the quadrature is
    exact (``(1 + K)^(-3/2)``) for every baseline and the w-term factor is a
    smooth function of ``t`` (verified against adaptive quadrature to
    ``< 1e-10`` for Fresnel parameters ``pi w r_s^2`` up to a few).
    """
    shape = np.broadcast(k1, k2, w1, w2).shape
    k1 = np.broadcast_to(np.asarray(k1, dtype=np.float64), shape)
    k2 = np.broadcast_to(np.asarray(k2, dtype=np.float64), shape)
    w1 = np.broadcast_to(np.asarray(w1, dtype=np.float64), shape)
    w2 = np.broadcast_to(np.asarray(w2, dtype=np.float64), shape)
    r_s2 = float(scale_radius) ** 2
    nodes, weights = exponential_mixture_quadrature(n_nodes)
    scale = 1.0 + 4.0 * np.pi**2 * r_s2 * (k1**2 + k2**2)  # 1 + K, [shape]
    s = nodes / scale[..., None]  # mixture variable per baseline
    sigma2 = 2.0 * r_s2 * s
    g1 = gaussian_width_factor(w1[..., None], sigma2)
    g2 = gaussian_width_factor(w2[..., None], sigma2)
    # weight e^{-t} = e^{-s (1+K)}: e^{-s} is in the Laguerre weight, the rest
    # (e^{-Ks} at w = 0) is the Gaussian response itself, so multiply back e^{sK}
    terms = np.sqrt(g1 * g2) * np.exp(
        -2.0 * np.pi**2 * sigma2 * (g1 * k1[..., None] ** 2 + g2 * k2[..., None] ** 2)
        + s * (scale[..., None] - 1.0)
    )
    return np.sum(terms * weights, axis=-1) / scale**1.5


def tapered_power_law_series(k1, k2, w1, w2, cutoff_radius, index, n_nodes: int = 160):
    """Response of the unit-flux profile ``r^-gamma exp(-r^2 / 2 r_c^2)`` with the w term (face-on frame).

    ``r^-gamma = int_0^inf t^(gamma/2 - 1) e^{-t r^2} dt / Gamma(gamma/2)`` makes
    the profile a scale mixture of Gaussians with ``sigma^2 = r_c^2 x``,
    ``x in (0, 1]`` and weight ``(1 - x)^(gamma/2 - 1) x^(-gamma/2) /
    B(gamma/2, 1 - gamma/2)`` (a Gauss-Jacobi integral).  For ``0 < gamma < 2``.
    Two quadratures cover the tangent-plane decay rate ``K = 2 pi^2 r_c^2
    (k_1^2 + k_2^2)``: Gauss-Jacobi on ``[0, 1]`` where ``K`` is moderate and,
    for strongly resolved baselines, generalised Gauss-Laguerre in ``y = K x``
    (the mixture then lives at ``x << 1``).  At ``w = 0`` the result is the
    Kummer function ``1F1(1 - gamma/2; 1; -K)`` of the closed form.
    """
    from scipy.special import beta as beta_function

    gamma_index = float(index)
    if not (0.0 < gamma_index < 2.0):
        raise ValueError(
            f"tapered power-law index must be in (0, 2); got {gamma_index}."
        )
    shape = np.broadcast(k1, k2, w1, w2).shape
    k1 = np.broadcast_to(np.asarray(k1, dtype=np.float64), shape).ravel()
    k2 = np.broadcast_to(np.asarray(k2, dtype=np.float64), shape).ravel()
    w1 = np.broadcast_to(np.asarray(w1, dtype=np.float64), shape).ravel()
    w2 = np.broadcast_to(np.asarray(w2, dtype=np.float64), shape).ravel()
    r_c2 = float(cutoff_radius) ** 2
    norm = beta_function(gamma_index / 2.0, 1.0 - gamma_index / 2.0)
    decay = 2.0 * np.pi**2 * r_c2 * (k1**2 + k2**2)  # K
    out = np.empty(k1.shape, dtype=np.complex128)

    moderate = decay <= 40.0
    if moderate.any():
        nodes, weights = _gauss_jacobi(
            n_nodes, gamma_index / 2.0 - 1.0, -gamma_index / 2.0
        )
        out[moderate] = gaussian_mixture_response(
            k1[moderate], k2[moderate], w1[moderate], w2[moderate],
            lambda x: r_c2 * x, nodes, weights / norm,
        )  # fmt: skip
    strong = ~moderate
    if strong.any():
        # y = K x: (1 - x)^(g/2-1) x^(-g/2) dx = K^(g/2 - 1) (1 - y/K)^(g/2-1) y^(-g/2) dy,
        # and the Gaussian response carries e^{-y} (the Laguerre weight) at w = 0.
        nodes, weights = _gauss_laguerre(n_nodes, -gamma_index / 2.0)
        ks = decay[strong][:, None]
        x_nodes = nodes[None, :] / ks  # [n_strong, n_nodes]
        inside = x_nodes < 1.0
        sigma2 = r_c2 * x_nodes
        g1 = gaussian_width_factor(w1[strong][:, None], sigma2)
        g2 = gaussian_width_factor(w2[strong][:, None], sigma2)
        with np.errstate(invalid="ignore", divide="ignore"):
            jacobi_factor = np.where(
                inside, (1.0 - x_nodes) ** (gamma_index / 2.0 - 1.0), 0.0
            )
        terms = np.sqrt(g1 * g2) * np.exp(
            -2.0 * np.pi**2 * sigma2 * (g1 * k1[strong][:, None] ** 2 + g2 * k2[strong][:, None] ** 2)
            + nodes[None, :]
        )  # fmt: skip
        out[strong] = (
            np.sum(np.where(inside, terms * jacobi_factor, 0.0) * weights, axis=-1)
            * ks[:, 0] ** (gamma_index / 2.0 - 1.0)
            / norm
        )
    return out.reshape(shape)


def deprojected_baseline(u, v, pa, axis_ratio):
    """Face-on-frame baseline coordinates ``(k_1, k_2) = (u_maj, q u_min)`` of an inclined component.

    ``pa`` is the position angle of the major axis from ``+m`` towards ``+l``
    (the clean-beam convention); ``axis_ratio = cos(inclination)``.
    """
    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return u * sin_pa + v * cos_pa, axis_ratio * (u * cos_pa - v * sin_pa)


def deprojected_sky(l, m, pa, axis_ratio):  # noqa: E741
    """Face-on-frame sky coordinates ``(x, y) = (l_maj, l_min / q)`` of an inclined component."""
    sin_pa = float(np.sin(pa))
    cos_pa = float(np.cos(pa))
    l = np.asarray(l, dtype=np.float64)  # noqa: E741
    m = np.asarray(m, dtype=np.float64)
    return l * sin_pa + m * cos_pa, (l * cos_pa - m * sin_pa) / axis_ratio


def blurred_radial_profile_image(r, profile_nodes, profile_values, sigma2):
    """Radial brightness of a circular profile convolved with a circular Gaussian.

    ``I(r) = int b(r') (r' / sigma^2) exp(-(r^2 + r'^2) / 2 sigma^2) I_0(r r' / sigma^2) dr'``
    evaluated with a quadrature rule ``(profile_nodes, profile_values)`` whose
    values already include the rule's weights, i.e. ``sum_i values_i f(r_i)``
    approximates ``int b(r') f(r') dr'``.
    """
    from scipy.special import i0e

    r = np.asarray(r, dtype=np.float64)[..., None]
    nodes = np.asarray(profile_nodes, dtype=np.float64)
    z = r * nodes / sigma2
    kernel = (nodes / sigma2) * np.exp(-0.5 * (r - nodes) ** 2 / sigma2) * i0e(z)
    return np.sum(kernel * np.asarray(profile_values), axis=-1)


def blur_quadrature_nodes(
    width, sigma2, per_sigma: int = 8, minimum: int = 48, maximum: int = 400
):
    """Number of radial quadrature nodes that resolve a Gaussian blur of variance ``sigma2`` over ``width``."""
    return int(np.clip(per_sigma * width / np.sqrt(sigma2) + minimum, minimum, maximum))


def uniform_disk_profile_rule(radius, sigma2=None, n_nodes: int | None = None):
    """Gauss-Legendre rule for ``int_0^a f(r') dr'`` (weights included in the values)."""
    if n_nodes is None:
        n_nodes = 400 if sigma2 is None else blur_quadrature_nodes(radius, sigma2)
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    nodes = radius * (nodes + 1.0) / 2.0
    return nodes, weights * radius / 2.0
