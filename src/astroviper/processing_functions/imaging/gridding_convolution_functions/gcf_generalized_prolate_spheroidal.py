"""The generalised prolate spheroidal gridding kernel.

The gridding convolution function of Schwab (1984, in Indirect Imaging, ed.
J. A. Roberts, Cambridge University Press, p. 333):

.. math::

    C(z) = (1 - z^2)^\\alpha \\, \\psi_{\\alpha 0}(c, z), \\qquad |z| \\le 1,

zero outside, with ``z`` the distance from the kernel centre in units of the
half width ``W / 2`` (``W`` the support, the number of grid points the kernel
spans) and :math:`\\psi_{\\alpha 0}(c, \\eta)` the generalised prolate
spheroidal function of order zero, the even eigenfunction of largest
eigenvalue of the weighted finite Fourier transform

.. math::

    \\lambda \\, \\psi(\\eta) = \\int_{-1}^{1} e^{\\imath c \\eta t}
    (1 - t^2)^\\alpha \\, \\psi(t) \\, dt.

The kernel's Fourier transform is therefore the function itself,
:math:`\\hat C(x) = (W / 2) \\, \\lambda \\, \\psi_{\\alpha 0}(c, \\pi W x / c)`
(``x`` in units of the padded image width): the bandwidth ``c`` sets where
the transform falls off, and ``alpha`` how the kernel goes to zero at its
edge. For every support and padding factor the pair :math:`(c, \\alpha)` is
chosen to minimise the largest aliasing error over the image kept,
:func:`optimize_generalized_prolate_spheroidal`, with the objective and the
error measure of the exponential of a semicircle kernel
(:mod:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle`).
Schwab's kernel, the ``prolate_spheroidal_casa`` type
(:mod:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal`),
is the case :math:`c = 3 \\pi`, :math:`\\alpha = 1` with a half width of 3
cells, the bandwidth of an image without padding.

:math:`\\psi_{\\alpha 0}` is computed from its expansion in the polynomials
orthonormal for the weight :math:`(1 - t^2)^\\alpha` (Gegenbauer
polynomials): the spheroidal differential operator, which commutes with the
transform above, is tridiagonal in that basis on the even functions, with
eigenvalues well separated for any ``c``. (The leading eigenvalues of the
transform itself, of the orders 0, 4, 8, ..., crowd together exponentially in
``c`` at :math:`\\alpha = 0`: its eigenvector of largest eigenvalue is off
by about 1e-3 at ``c = 25``.) The transform beyond the band, where the aliases
fall, is the integral above by Gauss-Jacobi quadrature, which takes the
weight and its edge exactly.
"""

from __future__ import annotations

import functools
import threading

import numpy as np

from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle import (
    ALIASING_ERROR_FLOOR,
    aliasing_error,
)


def _recurrence(k, alpha):
    """The coefficients ``a_k`` of ``t p_k = a_{k+1} p_{k+1} + a_k p_{k-1}``.

    For the polynomials ``p_k`` orthonormal on ``[-1, 1]`` for the weight
    ``(1 - t^2)^alpha``, ``k >= 1``.
    """
    k = np.asarray(k, dtype=np.float64)
    return np.sqrt(
        k
        * (k + 2.0 * alpha)
        / ((2.0 * k + 2.0 * alpha + 1.0) * (2.0 * k + 2.0 * alpha - 1.0))
    )


def _n_terms(c):
    """Even polynomial degrees kept in the expansion of the function.

    The expansion coefficients fall faster than geometrically beyond a degree
    of about ``c``; those dropped are far below the rounding of double
    precision.
    """
    return int(c) + 40


def _spheroidal_coefficients(c, alpha, n_terms):
    """Expansion of :math:`\\psi_{\\alpha 0}(c, \\cdot)` in the orthonormal polynomials.

    The function is the eigenfunction of smallest eigenvalue :math:`\\chi` of

    .. math::

        (1 - \\eta^2) \\psi'' - 2 (\\alpha + 1) \\eta \\psi' - c^2 \\eta^2 \\psi
        = -\\chi \\psi,

    whose first two terms take ``p_k`` to ``-k (k + 2 alpha + 1) p_k`` and
    whose last couples ``p_k`` to ``p_{k-2}``, ``p_k`` and ``p_{k+2}``: a
    symmetric tridiagonal matrix on the even degrees.

    Returns
    -------
    coefficients : numpy.ndarray
        Of the degrees ``0 .. 2 n_terms - 1`` (the odd ones zero), up to sign
        and scale.
    a : numpy.ndarray
        The recurrence coefficients, ``a[0] = 0``.
    """
    from scipy.linalg import eigh_tridiagonal

    degree = 2 * np.arange(int(n_terms))
    a = np.zeros(2 * int(n_terms) + 2)
    a[1:] = _recurrence(np.arange(1, a.size), alpha)
    diagonal = degree * (degree + 2.0 * alpha + 1.0) + c * c * (
        a[degree] ** 2 + a[degree + 1] ** 2
    )
    off_diagonal = c * c * a[degree[:-1] + 1] * a[degree[:-1] + 2]
    _, vectors = eigh_tridiagonal(
        diagonal, off_diagonal, select="i", select_range=(0, 0)
    )
    coefficients = np.zeros(2 * int(n_terms))
    coefficients[::2] = vectors[:, 0]
    return coefficients, a


def _series(eta, coefficients, a):
    """Sum of the expansion at ``eta`` (forward recurrence, stable on [-1, 1])."""
    p_previous = np.zeros_like(eta)
    p = np.ones_like(eta)
    total = coefficients[0] * p
    for k in range(1, coefficients.size):
        p, p_previous = (eta * p - a[k - 1] * p_previous) / a[k], p
        if k % 2 == 0:
            total += coefficients[k] * p
    return total


def generalized_prolate_spheroidal(eta, c, alpha):
    """The generalised prolate spheroidal function of order zero.

    :math:`\\psi_{\\alpha 0}(c, \\eta)` on ``|eta| <= 1``, normalised to 1 at
    ``eta = 0`` (beyond, it is the kernel's transform, see
    :func:`generalized_prolate_spheroidal_transform`).

    Parameters
    ----------
    eta : array_like
        Any shape, ``|eta| <= 1``.
    c : float
        The bandwidth (positive).
    alpha : float
        The exponent of the weight (at least 0).

    Returns
    -------
    numpy.ndarray
        float64, the shape of ``eta``.

    Raises
    ------
    ValueError
        For ``|eta| > 1``.
    """
    eta = np.asarray(eta, dtype=np.float64)
    if np.any(np.abs(eta) > 1.0):
        raise ValueError(
            "the expansion of the prolate spheroidal function holds for "
            "|eta| <= 1; its continuation is the kernel's transform"
        )
    coefficients, a = _spheroidal_coefficients(float(c), float(alpha), _n_terms(c))
    centre = _series(np.zeros(1), coefficients, a)[0]
    return _series(eta, coefficients, a) / centre


def generalized_prolate_spheroidal_kernel(z, c, alpha):
    """The kernel at ``z`` (distance from the centre over the half width).

    Parameters
    ----------
    z : array_like
        Any shape.
    c, alpha : float
        See :func:`generalized_prolate_spheroidal`.

    Returns
    -------
    numpy.ndarray
        float64, 1 at ``z = 0`` and zero where ``|z| > 1``.
    """
    z = np.asarray(z, dtype=np.float64)
    inside = np.abs(z) <= 1.0
    z_inside = np.where(inside, z, 0.0)
    values = (1.0 - z_inside * z_inside) ** alpha * generalized_prolate_spheroidal(
        z_inside, c, alpha
    )
    return np.where(inside, values, 0.0)


@functools.lru_cache(maxsize=32)
def _half_jacobi_nodes(n_quadrature, alpha):
    """The positive Gauss-Jacobi nodes and weights of ``(1 - t^2)^alpha``."""
    from scipy.special import roots_jacobi

    nodes, weights = roots_jacobi(2 * ((n_quadrature + 1) // 2), alpha, alpha)
    positive = nodes > 0.0
    nodes, weights = nodes[positive], weights[positive]
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def generalized_prolate_spheroidal_transform(x, support, c, alpha, n_quadrature=None):
    """Fourier transform of the continuous kernel.

    :math:`\\hat C(x) = W \\int_0^1 C(z) \\cos(\\pi W z x) \\, dz` for the
    kernel in uv cells (``z = 2 d / W``), by Gauss-Jacobi quadrature of the
    weight :math:`(1 - z^2)^\\alpha`; it equals
    :math:`(W / 2) \\, \\lambda \\, \\psi_{\\alpha 0}(c, \\pi W x / c)`.

    Parameters
    ----------
    x : array_like
        Image coordinates in units of the padded image width, any shape.
    support : int
        The support ``W`` (the half width ``W / 2`` in cells).
    c, alpha : float
        See :func:`generalized_prolate_spheroidal`.
    n_quadrature : int, optional
        Quadrature nodes on ``[-1, 1]``; the default resolves the oscillation
        of the integrand out to ``|x| = 12``.

    Returns
    -------
    numpy.ndarray
        float64, the shape of ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    if n_quadrature is None:
        n_quadrature = 40 + 20 * int(support) + 2 * _n_terms(c)
    nodes, weights = _half_jacobi_nodes(int(n_quadrature), float(alpha))
    weighted = weights * generalized_prolate_spheroidal(nodes, c, alpha)
    phase = np.pi * support * np.multiply.outer(x, nodes)
    return support * (np.cos(phase) @ weighted)


def _max_aliasing_error(params, support, x):
    c, alpha = params

    def transform(pos):
        return generalized_prolate_spheroidal_transform(pos, support, c, alpha)

    return float(np.max(aliasing_error(transform, x)))


def nominal_bandwidth(support, x0):
    """The bandwidth that puts the first alias at the edge of the band.

    ``c = pi W (1 - x0)``: the alias of the edge of the image kept,
    ``x0 - 1``, falls at :math:`\\eta = 1`. With ``W = 6`` and no padding
    (``x0 = 1/2``) it is Schwab's :math:`3 \\pi`.

    Parameters
    ----------
    support : int
        The support ``W``.
    x0 : float
        Half width of the image kept in units of the padded image width.

    Returns
    -------
    float
    """
    return float(np.pi * support * (1.0 - x0))


# one optimisation at a time per process, so that every thread gets the
# same parameters (see gcf_least_misfit._SOLVE_LOCK)
_OPTIMIZE_LOCK = threading.Lock()


def optimize_generalized_prolate_spheroidal(support, x0, n_x=129):
    """The ``(c, alpha)`` that minimise the largest aliasing error.

    A grid search over ``c`` in ``[0.6, 1.4]`` times
    :func:`nominal_bandwidth` and ``alpha`` in ``[0, 3]``, followed by a
    Nelder-Mead polish of the best grid point (the error has a sharp valley
    where its maximum moves from the edge of the image kept to within it;
    the polish is kept only if it improves on the grid point); a few seconds
    per support.

    Parameters
    ----------
    support : int
        The support ``W`` (odd).
    x0 : float
        Half width of the image kept in units of the padded image width,
        ``1 / (2 * fft_padding)``.
    n_x : int
        Positions in ``[0, x0]`` the final error is evaluated at (the grid
        search uses 65, the polish 129).

    Returns
    -------
    c, alpha, error : float
        The parameters and the largest aliasing error over ``[0, x0]``
        (not below :data:`ALIASING_ERROR_FLOOR`).
    """
    with _OPTIMIZE_LOCK:
        return _optimize_generalized_prolate_spheroidal(
            int(support), float(x0), int(n_x)
        )


@functools.cache
def _optimize_generalized_prolate_spheroidal(support, x0, n_x):
    from scipy.optimize import minimize

    c0 = nominal_bandwidth(support, x0)

    def objective(params, x):
        ratio, alpha = params
        if not (0.5 <= ratio <= 1.5 and 0.0 <= alpha <= 3.0):
            return np.log(10.0)
        error = _max_aliasing_error((ratio * c0, alpha), support, x)
        return np.log(max(error, ALIASING_ERROR_FLOOR))

    coarse_x = np.linspace(0.0, x0, 65)
    grid = [
        (ratio, alpha)
        for alpha in np.arange(0.0, 3.01, 0.25)
        for ratio in np.arange(0.6, 1.401, 0.02)
    ]
    start = min(grid, key=lambda p: objective(p, coarse_x))
    fine_x = np.linspace(0.0, x0, 129)
    result = minimize(
        objective,
        np.array(start),
        args=(fine_x,),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300},
    )
    x = np.linspace(0.0, x0, int(n_x))
    candidates = [
        (float(ratio) * c0, float(alpha)) for ratio, alpha in (start, result.x)
    ]
    errors = [
        max(_max_aliasing_error(params, support, x), ALIASING_ERROR_FLOOR)
        for params in candidates
    ]
    best = int(np.argmin(errors))
    c, alpha = candidates[best]
    return c, alpha, float(errors[best])


# (c, alpha) of optimize_generalized_prolate_spheroidal for the usual supports
# and padding factors, rounded to 4 decimals, generated (in about 90 s) by
#   for p in (1.15, 1.2, 1.25, 1.5, 2.0):
#       for w in range(3, 18, 2):
#           if p != 1.15 or w == 7:
#               c, alpha, _ = optimize_generalized_prolate_spheroidal(w, 0.5 / p)
#               print((w, p), (round(c, 4), round(alpha, 4)))
# The largest aliasing error over the image kept follows each. Every optimum
# has alpha below 5e-4: the prolate spheroidal function itself (alpha = 0)
# with c a little below nominal_bandwidth (anywhere in a range of c where the
# error is at the floor of 1.0e-12).
SHAPE_TABLE = {
    (7, 1.15): (12.2556, 0.0000),  # 1.1e-03
    (3, 1.2): (4.9167, 0.0001),  # 5.1e-02
    (5, 1.2): (8.9033, 0.0000),  # 4.8e-03
    (7, 1.2): (12.6665, 0.0000),  # 4.2e-04
    (9, 1.2): (16.3763, 0.0001),  # 3.8e-05
    (11, 1.2): (19.4708, 0.0000),  # 4.2e-06
    (13, 1.2): (23.7479, 0.0000),  # 2.6e-07
    (15, 1.2): (27.0110, 0.0001),  # 2.4e-08
    (17, 1.2): (30.9996, 0.0002),  # 1.8e-09
    (3, 1.25): (5.1367, 0.0002),  # 3.7e-02
    (5, 1.25): (9.1784, 0.0000),  # 2.8e-03
    (7, 1.25): (13.0425, 0.0000),  # 1.9e-04
    (9, 1.25): (16.8525, 0.0000),  # 1.4e-05
    (11, 1.25): (20.0842, 0.0001),  # 1.1e-06
    (13, 1.25): (24.4320, 0.0001),  # 6.0e-08
    (15, 1.25): (27.8217, 0.0001),  # 4.0e-09
    (17, 1.25): (31.8916, 0.0003),  # 2.5e-10
    (3, 1.5): (5.8763, 0.0001),  # 1.4e-02
    (5, 1.5): (10.2817, 0.0003),  # 4.5e-04
    (7, 1.5): (14.5350, 0.0000),  # 1.4e-05
    (9, 1.5): (18.1399, 0.0000),  # 5.5e-07
    (11, 1.5): (22.9651, 0.0001),  # 1.3e-08
    (13, 1.5): (26.7727, 0.0001),  # 3.7e-10
    (15, 1.5): (31.0269, 0.0002),  # 1.0e-11
    (17, 1.5): (34.8926, 0.0000),  # 1.0e-12
    (3, 2.0): (6.7635, 0.0001),  # 5.0e-03
    (5, 2.0): (11.6277, 0.0000),  # 7.1e-05
    (7, 2.0): (16.3903, 0.0000),  # 9.7e-07
    (9, 2.0): (20.6113, 0.0001),  # 1.5e-08
    (11, 2.0): (25.4455, 0.0001),  # 1.8e-10
    (13, 2.0): (30.2403, 0.0004),  # 2.1e-12
    (15, 2.0): (31.8086, 0.0000),  # 1.0e-12
    (17, 2.0): (32.0442, 0.0000),  # 1.0e-12
}


def generalized_prolate_spheroidal_parameters(support, x0):
    """The ``(c, alpha)`` of a support and image region, and the aliasing error.

    From :data:`SHAPE_TABLE` when the support and the padding factor
    ``1 / (2 * x0)`` are listed (the same kernel on every platform), else
    from :func:`optimize_generalized_prolate_spheroidal`, rounded to 4
    decimals.

    Parameters
    ----------
    support : int
        The support ``W`` (odd).
    x0 : float
        Half width of the image kept in units of the padded image width.

    Returns
    -------
    c, alpha, error : float
    """
    support = int(support)
    key = (support, round(0.5 / x0, 6))
    if key in SHAPE_TABLE:
        c, alpha = SHAPE_TABLE[key]
    else:
        c, alpha, _ = optimize_generalized_prolate_spheroidal(support, float(x0))
        c, alpha = round(c, 4), round(alpha, 4)
    x = np.linspace(0.0, x0, 129)
    error = max(_max_aliasing_error((c, alpha), support, x), ALIASING_ERROR_FLOOR)
    return c, alpha, float(error)


def create_generalized_prolate_spheroidal_kernel_1D(oversampling, support, c, alpha):
    """The half kernel table of the kernel (the layout of ``kernel_lookup``).

    Entry ``k`` is the kernel ``k / oversampling`` uv cells from its centre;
    the kernel spans ``support / 2`` cells, and the table runs to
    ``support // 2 + 1`` cells (zero beyond the half width).

    Parameters
    ----------
    oversampling : int
        Table samples per uv cell.
    support : int
        The support ``W`` (odd).
    c, alpha : float
        See :func:`generalized_prolate_spheroidal`.

    Returns
    -------
    numpy.ndarray
        float64 of ``oversampling * (support // 2 + 1)`` entries.
    """
    support = int(support)
    oversampling = int(oversampling)
    cells = np.arange(oversampling * (support // 2 + 1)) / oversampling
    return generalized_prolate_spheroidal_kernel(2.0 * cells / support, c, alpha)
