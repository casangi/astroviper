"""The modified exponential of a semicircle gridding kernel.

The kernel of Barnett, Magland and af Klinteberg (2019, SIAM J. Sci. Comput.
41, C479) in the two parameter form of Arras et al. (2021, A&A 646, A58):

.. math::

    \\phi(z) = \\exp\\Big(\\beta W \\big[(1 - z^2)^\\mu - 1\\big]\\Big),
    \\qquad |z| \\le 1,

zero outside, with ``z`` the distance from the kernel centre in units of the
half width ``W / 2`` (``W`` the support, the number of grid points the kernel
spans). Its Fourier transform falls off exponentially beyond the image region
the kernel is designed for, so the aliases of the periodic FFT image are
suppressed there. For every support and padding factor the pair
:math:`(\\beta, \\mu)` is chosen to minimise the largest aliasing error over
the image kept, :func:`optimize_exponential_semicircle`, which reproduces the
tables of Arras et al. (their appendix A) to the accuracy of the error
measure.
"""

from __future__ import annotations

import functools
import threading

import numpy as np

# Barnett et al. (2019): beta = 2.30 W, mu = 1/2 for a padding factor of 2
BARNETT_BETA = 2.30
BARNETT_MU = 0.5


def exponential_semicircle(z, support, beta, mu):
    """The kernel at ``z`` (distance from the centre over the half width).

    Parameters
    ----------
    z : array_like
        Any shape.
    support : int
        The support ``W``.
    beta : float
        Shape parameter per unit support (the exponent is ``beta * W``).
    mu : float
        Exponent of the semicircle (``1/2`` is the exponential of a
        semicircle proper).

    Returns
    -------
    numpy.ndarray
        float64, zero where ``|z| > 1``.
    """
    z = np.asarray(z, dtype=np.float64)
    inside = np.abs(z) <= 1.0
    base = np.where(inside, 1.0 - z * z, 0.0)
    return np.where(inside, np.exp(beta * support * (base**mu - 1.0)), 0.0)


def exponential_semicircle_transform(x, support, beta, mu, n_quadrature=None):
    """Fourier transform of the continuous kernel.

    :math:`\\hat\\phi(x) = W \\int_0^1 \\phi(z) \\cos(\\pi W z x) \\, dz`
    for the kernel in uv cells (``z = 2 d / W``), by Gauss-Legendre
    quadrature. Double precision rounding of the quadrature limits the
    transform's far tail (and so the aliasing error below) to about
    ``1e-12`` of its peak.

    Parameters
    ----------
    x : array_like
        Image coordinates in units of the padded image width, any shape.
    support, beta, mu
        See :func:`exponential_semicircle`.
    n_quadrature : int, optional
        Quadrature nodes; the default resolves the oscillation of the
        integrand out to ``|x| = 12``.

    Returns
    -------
    numpy.ndarray
        float64, the shape of ``x``.
    """
    from scipy.special import roots_legendre

    x = np.asarray(x, dtype=np.float64)
    if n_quadrature is None:
        n_quadrature = 40 + 20 * int(support)
    nodes, weights = roots_legendre(int(n_quadrature))
    z = 0.5 * (nodes + 1.0)
    w = 0.5 * weights * exponential_semicircle(z, support, beta, mu)
    phase = np.pi * support * np.multiply.outer(x, z)
    return support * np.sum(np.cos(phase) * w, axis=-1)


def aliasing_error(transform, x, n_alias=10):
    """The aliasing error of a kernel at image positions ``x``.

    The rms over the position of a visibility within a uv cell of the map
    error function (the gridded response of a unit visibility minus the
    ideal fringe, in units of the fringe) is

    .. math::

        \\epsilon(x) = \\sqrt{\\sum_{k \\ne 0} \\hat c(x - k)^2} \\, / \\,
        |\\hat c(x)|,

    the kernel's transform at the aliases of ``x`` (the periodic images of
    the FFT grid) over its transform at ``x``.

    Parameters
    ----------
    transform : callable
        The kernel's Fourier transform of an array of image coordinates.
    x : array_like
        Image coordinates in units of the padded image width.
    n_alias : int
        Aliases ``k = 1 .. n_alias`` on either side are summed.

    Returns
    -------
    numpy.ndarray
        float64, the shape of ``x``.
    """
    x = np.asarray(x, dtype=np.float64)
    k = np.arange(-n_alias, n_alias + 1)
    values = transform(np.add.outer(x, -k.astype(np.float64)))
    centre = values[..., n_alias]
    others = np.delete(values, n_alias, axis=-1)
    return np.sqrt(np.sum(others * others, axis=-1)) / np.abs(centre)


# the aliasing error the quadrature of the transform resolves
ALIASING_ERROR_FLOOR = 1e-12


def _max_aliasing_error(params, support, x):
    beta, mu = params

    def transform(pos):
        return exponential_semicircle_transform(pos, support, beta, mu)

    return float(np.max(aliasing_error(transform, x)))


# one optimisation at a time per process, so that every thread gets the
# same parameters (see gcf_least_misfit._SOLVE_LOCK)
_OPTIMIZE_LOCK = threading.Lock()


def optimize_exponential_semicircle(support, x0, n_x=129):
    """The ``(beta, mu)`` that minimise the largest aliasing error.

    A coarse grid search over ``beta`` and ``mu`` followed by a Nelder-Mead
    polish of the best grid point; about a second per support.

    Parameters
    ----------
    support : int
        The support ``W`` (odd).
    x0 : float
        Half width of the image kept in units of the padded image width,
        ``1 / (2 * fft_padding)``.
    n_x : int
        Positions in ``[0, x0]`` the final error is evaluated at (the search
        uses a coarser grid).

    Returns
    -------
    beta, mu, error : float
        The parameters and the largest aliasing error over ``[0, x0]``
        (not below :data:`ALIASING_ERROR_FLOOR`, which the quadrature of the
        transform does not resolve).
    """
    with _OPTIMIZE_LOCK:
        return _optimize_exponential_semicircle(int(support), float(x0), int(n_x))


@functools.cache
def _optimize_exponential_semicircle(support, x0, n_x):
    from scipy.optimize import minimize

    def objective(params, x):
        beta, mu = params
        if not (0.1 <= beta <= 6.0 and 0.25 <= mu <= 1.0):
            return np.log(10.0)
        return np.log(
            max(_max_aliasing_error(params, support, x), ALIASING_ERROR_FLOOR)
        )

    coarse_x = np.linspace(0.0, x0, 17)
    grid = [
        (beta, mu)
        for beta in np.arange(0.6, 3.21, 0.2)
        for mu in np.arange(0.4, 0.71, 0.05)
    ]
    start = min(grid, key=lambda p: objective(p, coarse_x))
    fine_x = np.linspace(0.0, x0, 33)
    result = minimize(
        objective,
        np.array(start),
        args=(fine_x,),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 300},
    )
    beta, mu = (float(v) for v in result.x)
    x = np.linspace(0.0, x0, int(n_x))
    error = max(_max_aliasing_error((beta, mu), support, x), ALIASING_ERROR_FLOOR)
    return beta, mu, float(error)


# (beta, mu) of optimize_exponential_semicircle for the usual supports and
# padding factors, rounded to 4 decimals (generated by
# dev/imaging/gridding_kernel/exponential_semicircle_table.py); the largest
# aliasing error over the image kept follows each
SHAPE_TABLE = {
    (7, 1.15): (1.5211, 0.5294),  # 7.7e-04
    (3, 1.2): (1.1053, 0.6295),  # 5.1e-02
    (5, 1.2): (1.4592, 0.5506),  # 4.5e-03
    (7, 1.2): (1.5719, 0.5289),  # 2.7e-04
    (9, 1.2): (1.6695, 0.5157),  # 2.6e-05
    (11, 1.2): (1.7426, 0.5074),  # 3.3e-06
    (13, 1.2): (1.7500, 0.5068),  # 1.9e-07
    (15, 1.2): (1.7313, 0.5078),  # 1.8e-08
    (17, 1.2): (1.7678, 0.5050),  # 1.2e-09
    (3, 1.25): (1.0761, 0.6761),  # 3.8e-02
    (5, 1.25): (1.5136, 0.5480),  # 2.5e-03
    (7, 1.25): (1.6217, 0.5282),  # 1.3e-04
    (9, 1.25): (1.7304, 0.5143),  # 9.1e-06
    (11, 1.25): (1.7403, 0.5130),  # 5.9e-07
    (13, 1.25): (1.8012, 0.5067),  # 3.8e-08
    (15, 1.25): (1.7828, 0.5080),  # 2.7e-09
    (17, 1.25): (1.8409, 0.4996),  # 8.4e-10
    (3, 1.5): (1.3372, 0.6314),  # 1.5e-02
    (5, 1.5): (1.7145, 0.5427),  # 3.1e-04
    (7, 1.5): (1.8791, 0.5195),  # 8.9e-06
    (9, 1.5): (1.9547, 0.5112),  # 2.9e-07
    (11, 1.5): (1.9480, 0.5116),  # 9.0e-09
    (13, 1.5): (1.9658, 0.5094),  # 2.3e-10
    (15, 1.5): (2.0866, 0.4993),  # 5.6e-11
    (17, 1.5): (2.1564, 0.4933),  # 1.3e-11
    (3, 2.0): (1.6561, 0.5956),  # 4.4e-03
    (5, 2.0): (2.0654, 0.5250),  # 5.8e-05
    (7, 2.0): (2.0940, 0.5218),  # 7.8e-07
    (9, 2.0): (2.1898, 0.5122),  # 9.3e-09
    (11, 2.0): (2.2130, 0.5097),  # 1.1e-10
    (13, 2.0): (2.2380, 0.5075),  # 1.5e-12
    (15, 2.0): (2.4368, 0.4924),  # 3.3e-12
    (17, 2.0): (2.4000, 0.5000),  # 1.0e-12
}


def exponential_semicircle_parameters(support, x0):
    """The ``(beta, mu)`` of a support and image region, and the aliasing error.

    From :data:`SHAPE_TABLE` when the support and the padding factor
    ``1 / (2 * x0)`` are listed (the same kernel on every platform), else
    from :func:`optimize_exponential_semicircle`, rounded to 4 decimals.

    Parameters
    ----------
    support : int
        The support ``W`` (odd).
    x0 : float
        Half width of the image kept in units of the padded image width.

    Returns
    -------
    beta, mu, error : float
    """
    support = int(support)
    key = (support, round(0.5 / x0, 6))
    if key in SHAPE_TABLE:
        beta, mu = SHAPE_TABLE[key]
    else:
        beta, mu, _ = optimize_exponential_semicircle(support, float(x0))
        beta, mu = round(beta, 4), round(mu, 4)
    x = np.linspace(0.0, x0, 129)
    error = max(_max_aliasing_error((beta, mu), support, x), ALIASING_ERROR_FLOOR)
    return beta, mu, float(error)


def create_exponential_semicircle_kernel_1D(oversampling, support, beta, mu):
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
    beta, mu : float
        See :func:`exponential_semicircle`.

    Returns
    -------
    numpy.ndarray
        float64 of ``oversampling * (support // 2 + 1)`` entries.
    """
    support = int(support)
    oversampling = int(oversampling)
    cells = np.arange(oversampling * (support // 2 + 1)) / oversampling
    return exponential_semicircle(2.0 * cells / support, support, beta, mu)
