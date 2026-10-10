"""The least misfit gridding kernel (Ye, Gull, Tan and Nikolic 2020, MNRAS 491,
1146).

The kernel table is not sampled from a function: its entries are the least
squares solution that makes the gridded response of a visibility, over all
positions of the visibility within a uv cell and over the image kept, as
close as possible to the ideal fringe. The design includes exactly what the
gridders apply (the table entries they combine at each stencil point, with
``nearest`` or ``linear`` lookup), so the within cell lookup error is part of
what is minimised, and the image plane correction is the transform of the
kernel applied (:func:`~.kernel_lookup.table_transform`).

For a visibility whose nearest grid point is ``offset`` cells away, the
gridders put the stencil values :math:`C_i` on the grid points
:math:`i - W // 2` cells from that grid point, which after the FFT and the
correction :math:`h(x)` gives the response
:math:`A(x) / h(x) = \\sum_i C_i e^{2 \\pi \\imath (i - W//2) x} / h(x)` to
compare with the fringe :math:`e^{-2 \\pi \\imath \\, \\mathrm{offset} \\, x}`
of the visibility itself (``x`` in units of the padded image width). The
misfit

.. math::

    \\int_{-1/2}^{1/2} d\\,\\mathrm{offset} \\int_{-x_0}^{x_0} dx \\,
    \\Big| \\frac{A(x)}{h(x)} - e^{-2 \\pi \\imath \\, \\mathrm{offset} \\, x}
    \\Big|^2

is quadratic in the table for a given correction; the solver alternates
the least squares solution of the table with the update of the correction
from the table until the misfit stops decreasing.
"""

from __future__ import annotations

import functools
import threading

import numpy as np

from astroviper.processing_functions.imaging.gridding_convolution_functions.kernel_lookup import (
    stencil_lookup,
    table_size,
    table_transform,
)


def _quadrature_x(x0, n_x):
    """Gauss-Legendre nodes and weights on ``[0, x0]`` (the misfit is even in x)."""
    from scipy.special import roots_legendre

    nodes, weights = roots_legendre(int(n_x))
    return 0.5 * x0 * (nodes + 1.0), 0.5 * x0 * weights


def misfit(table, support, oversampling, interpolation, x0, offsets=None, n_x=96):
    """The rms map error of a kernel table over the image kept.

    The root of the misfit of the module docstring (per unit offset range
    and per unit image width), with the correction the transform of the
    kernel applied.

    Parameters
    ----------
    table : numpy.ndarray
        The half kernel table.
    support, oversampling, interpolation
        The stencil (see :func:`~.kernel_lookup.stencil_lookup`).
    x0 : float
        Half width of the image kept in units of the padded image width.
    offsets : array_like, optional
        Visibility offsets to average over; default 4 per table cell.
    n_x : int
        Quadrature nodes over ``[0, x0]``.

    Returns
    -------
    float
    """
    if offsets is None:
        n_offsets = 4 * int(oversampling)
        offsets = -0.5 + (np.arange(n_offsets) + 0.5) / n_offsets
    offsets = np.asarray(offsets, dtype=np.float64)
    x, w_x = _quadrature_x(x0, n_x)
    rel = np.arange(int(support)) - int(support) // 2
    indices, weights = stencil_lookup(support, oversampling, interpolation, offsets)
    values = np.sum(np.asarray(table)[indices] * weights, axis=-2)  # (n_off, W)
    # the response in units of the kernel's transform (the correction, with
    # the kernel's scale): A(x) / c(x), independent of the table's scale
    transform = table_transform(table, oversampling, interpolation, x)
    basis = np.exp(2j * np.pi * np.multiply.outer(rel, x))  # (W, n_x)
    response = values @ basis / transform  # (n_off, n_x)
    target = np.exp(-2j * np.pi * np.multiply.outer(offsets, x))
    error = np.abs(response - target) ** 2
    return float(np.sqrt(np.sum(error @ w_x) / offsets.size / x0))


# the largest table solved for directly; finer tables are resampled from it
MAX_SOLVED_TABLE = 2400
# One solve at a time per process: concurrent solves (node tasks sharing a
# process) land at different points of the flat valley of the misfit, since
# the BLAS rounding differs under contention, and a table that differs
# between the point spread function and the residual updates of one imaging
# run makes the model updates inconsistent. The lock plus the cache give
# every thread of a process the same table.
_SOLVE_LOCK = threading.Lock()


class _MisfitProblem:
    """The misfit of a kernel table and its Gauss-Newton normal equations.

    Everything that does not depend on the table is precomputed: the lookup
    matrix ``P`` (stencil values ``= P @ table``), the fringe and response
    phases on the quadrature nodes, and the rows ``G`` of the table
    transform at the nodes (``transform = table @ G``).
    """

    def __init__(self, support, oversampling, interpolation, x0, offsets_per_cell, n_x):
        import scipy.sparse

        self.support = support
        self.oversampling = oversampling
        self.interpolation = interpolation
        self.x0 = x0
        n_table = table_size(support, oversampling)
        rel = (np.arange(support) - support // 2).astype(np.float64)
        n_offsets = int(offsets_per_cell) * oversampling
        self.offsets = -0.5 + (np.arange(n_offsets) + 0.5) / n_offsets
        self.omega = 1.0 / n_offsets
        indices, weights = stencil_lookup(
            support, oversampling, interpolation, self.offsets
        )
        rows = np.repeat(np.arange(n_offsets * support), 2)
        P = scipy.sparse.csr_matrix(
            (
                weights.transpose(0, 2, 1).reshape(-1),
                (rows, indices.transpose(0, 2, 1).reshape(-1)),
            ),
            shape=(n_offsets * support, n_table),
        )
        self.used = np.flatnonzero(np.asarray(abs(P).sum(axis=0)).ravel() > 0.0)
        self.P = P[:, self.used].tocsr()
        self.n_table = n_table
        self.n_offsets = n_offsets

        self.x, self.w_x = _quadrature_x(x0, n_x)
        x = self.x
        self.phase = np.exp(2j * np.pi * np.multiply.outer(rel, x))  # (W, n_x)
        self.fringe = np.exp(
            -2j * np.pi * np.multiply.outer(self.offsets, x)
        )  # (n_off, n_x)
        # cos(2 pi (rel_i - rel_j) x) (W, W, n_x); cos(2 pi (rel_i + offset) x) (n_off W, n_x)
        self.cos_pair = np.cos(
            2.0 * np.pi * np.multiply.outer(np.subtract.outer(rel, rel), x)
        )
        cos_target = np.cos(
            2.0 * np.pi * np.multiply.outer(np.add.outer(self.offsets, rel), x)
        )
        self.U = (
            np.asarray(self.P.T @ cos_target.reshape(n_offsets * support, -1))
            * self.omega
        )
        # the transform of the used entries at the nodes and at the centre
        k = self.used.astype(np.float64)
        factor = np.where(self.used == 0, 1.0, 2.0) / oversampling
        G = (
            np.cos(2.0 * np.pi * np.multiply.outer(k / oversampling, x))
            * factor[:, None]
        )
        s = np.sinc(x / oversampling)
        G *= (s * s if interpolation == "linear" else s)[None, :]
        self.G = G  # (n_used, n_x)
        self.g0 = factor

    def full_table(self, solution):
        table = np.zeros(self.n_table)
        table[self.used] = solution
        return table

    def stencil(self, solution):
        return np.asarray(self.P @ solution).reshape(self.n_offsets, self.support)

    def misfit(self, solution):
        response = self.stencil(solution) @ self.phase / (solution @ self.G)
        error = np.abs(response - self.fringe) ** 2
        return float(np.sqrt((error @ self.w_x).sum() * self.omega / self.x0))

    def gauss_newton_step(self, solution, damping):
        """The constrained, damped Gauss-Newton step from ``solution``.

        The misfit residual ``E = A / c - t`` (response over transform minus
        fringe) has the Jacobian ``J = (B - (A / c) g) / c`` with respect to
        the table (``B`` the response, ``g`` the transform rows); its normal
        equations are assembled from the sparse response term and rank
        ``n_x`` transform terms, and solved with the constraint that the
        transform at the image centre does not change.
        """
        import scipy.sparse

        W = self.support
        C = self.stencil(solution)  # (n_off, W)
        c = solution @ self.G  # (n_x,)
        w = self.w_x / c**2
        # S_q T and T^T S_q T for every node q
        CQ = np.einsum("si,ijq->sjq", C, self.cos_pair).reshape(self.n_offsets * W, -1)
        ST = np.asarray(self.P.T @ CQ) * self.omega  # (n_used, n_x)
        a = np.einsum("sjq,sj->q", CQ.reshape(self.n_offsets, W, -1), C) * self.omega
        Q = self.cos_pair @ w  # (W, W)
        Q_block = scipy.sparse.kron(
            scipy.sparse.identity(self.n_offsets, format="csr"),
            scipy.sparse.csr_matrix(Q),
            format="csr",
        )
        normal = (self.P.T @ Q_block @ self.P).toarray() * self.omega
        cross = (ST * (w / c)) @ self.G.T
        normal -= cross + cross.T
        normal += (self.G * (w * a / c**2)) @ self.G.T
        uT = solution @ self.U  # (n_x,)
        gradient = ST @ w - self.U @ (self.w_x / c)
        gradient -= self.G @ ((self.w_x / c**2) * (a / c - uT))
        n = solution.size
        scale = np.diag(normal).copy()
        scale[scale <= 0.0] = np.max(scale)
        kkt = np.zeros((n + 1, n + 1))
        kkt[:n, :n] = normal
        kkt[np.diag_indices(n)] += damping * scale
        kkt[:n, n] = self.g0
        kkt[n, :n] = self.g0
        rhs = np.zeros(n + 1)
        rhs[:n] = -gradient
        step = np.linalg.solve(kkt, rhs)[:n]
        return step


def _solve_least_misfit(problem, start, max_iterations, tolerance):
    """Levenberg-Marquardt iterations of :class:`_MisfitProblem` from ``start``."""
    solution = start / (start @ problem.g0)
    value = problem.misfit(solution)
    history = [value]
    damping = 1e-3
    for _ in range(int(max_iterations)):
        accepted = False
        for _attempt in range(12):
            candidate = solution + problem.gauss_newton_step(solution, damping)
            candidate /= candidate @ problem.g0
            new_value = problem.misfit(candidate)
            if np.isfinite(new_value) and new_value < value:
                accepted = True
                break
            damping *= 10.0
        if not accepted:
            break
        converged = value - new_value < tolerance * value
        solution, value = candidate, new_value
        history.append(value)
        damping = max(damping / 3.0, 1e-12)
        if converged:
            break
    return solution, history


def create_least_misfit_kernel_1D(
    oversampling,
    support,
    interpolation,
    x0,
    offsets_per_cell=4,
    n_x=96,
    max_iterations=30,
    tolerance=1e-5,
):
    """The half kernel table of the least misfit kernel.

    The misfit of the module docstring is minimised over the table entries
    the gridders can reach, with the image plane correction the transform of
    the table, by damped Gauss-Newton (Levenberg-Marquardt) iterations from
    the exponential of a semicircle kernel of the same support, under the
    constraint that the transform is 1 at the image centre. Tables beyond
    :data:`MAX_SOLVED_TABLE` entries are cubic spline resamplings of the
    table solved at the largest oversampling within the limit (the lookup
    error is then negligible and the kernel is a smooth function).

    Parameters
    ----------
    oversampling : int
        Table samples per uv cell.
    support : int
        Grid points the kernel spans (odd).
    interpolation : str
        ``"nearest"`` or ``"linear"``, the lookup the table is designed for.
    x0 : float
        Half width of the image kept in units of the padded image width,
        ``1 / (2 * fft_padding)``.
    offsets_per_cell : int
        Visibility offsets per table cell the misfit is averaged over.
    n_x : int
        Gauss-Legendre nodes over ``[0, x0]``.
    max_iterations : int
        Gauss-Newton iterations at most.
    tolerance : float
        Stop when the misfit decreases by less than this fraction.

    Returns
    -------
    table : numpy.ndarray
        float64 of ``oversampling * (support // 2 + 1)`` entries, normalised
        so that the transform of the kernel applied is 1 at the image centre.
    info : dict
        ``misfit`` (the final rms map error), ``iterations``, the
        ``history`` of the misfit and ``solved_oversampling``.
    """
    with _SOLVE_LOCK:
        return _create_least_misfit_kernel_1D(
            int(oversampling),
            int(support),
            interpolation,
            float(x0),
            int(offsets_per_cell),
            int(n_x),
            int(max_iterations),
            float(tolerance),
        )


@functools.cache
def _create_least_misfit_kernel_1D(
    oversampling,
    support,
    interpolation,
    x0,
    offsets_per_cell,
    n_x,
    max_iterations,
    tolerance,
):
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle import (
        create_exponential_semicircle_kernel_1D,
        optimize_exponential_semicircle,
    )

    cells = support // 2 + 1
    if table_size(support, oversampling) > MAX_SOLVED_TABLE:
        from scipy.interpolate import CubicSpline

        solved_oversampling = MAX_SOLVED_TABLE // cells
        coarse, info = _create_least_misfit_kernel_1D(
            solved_oversampling,
            support,
            interpolation,
            x0,
            offsets_per_cell,
            n_x,
            max_iterations,
            tolerance,
        )
        distance = np.arange(coarse.size) / solved_oversampling
        spline = CubicSpline(distance, coarse, bc_type=((1, 0.0), "not-a-knot"))
        fine_distance = np.arange(table_size(support, oversampling)) / oversampling
        table = np.where(fine_distance <= 0.5 * support, spline(fine_distance), 0.0)
        table = table / table_transform(table, oversampling, interpolation, 0.0)
        info = dict(info)
        info["misfit"] = misfit(
            table, support, oversampling, interpolation, x0, n_x=n_x
        )
        return table, info

    problem = _MisfitProblem(
        support, oversampling, interpolation, x0, offsets_per_cell, n_x
    )
    beta, mu, _ = optimize_exponential_semicircle(support, x0)
    start = create_exponential_semicircle_kernel_1D(oversampling, support, beta, mu)
    solution, history = _solve_least_misfit(
        problem, start[problem.used], max_iterations, tolerance
    )
    table = problem.full_table(solution)
    table = table / table_transform(table, oversampling, interpolation, 0.0)
    return table, {
        "misfit": history[-1],
        "iterations": len(history) - 1,
        "history": history,
        "solved_oversampling": oversampling,
    }
