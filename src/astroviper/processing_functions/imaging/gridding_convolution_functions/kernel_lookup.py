"""How the gridders read the sampled kernel table (the Python twin of
``kernel_lookup.hpp``).

The table holds the half kernel: entry ``k`` is the kernel ``k / oversampling``
uv cells from its centre, ``oversampling * (support // 2 + 1)`` entries in
all. A visibility is gridded onto the ``support`` grid points centred on the
grid point nearest to it; the stencil point ``i`` (``0 <= i < support``) is
``i - support // 2`` cells from that grid point and ``offset`` is the grid
point minus the visibility position (cells, in ``(-0.5, 0.5]``), so the
stencil point lies ``|i - support // 2 + offset|`` cells from the visibility.

The functions here reproduce the arithmetic of the C++ lookup exactly, for
the kernel designs that must know what the gridders apply (the least misfit
kernel, the table consistent image plane correction) and for tests.
"""

from __future__ import annotations

import numpy as np

# the C++ codes of kernel_lookup.hpp
INTERPOLATIONS = {"nearest": 0, "linear": 1}


def table_size(support, oversampling):
    """Entries of the half kernel table of a kernel of this support."""
    return int(oversampling) * (int(support) // 2 + 1)


def stencil_lookup(support, oversampling, interpolation, offsets):
    """Table entries and weights the gridders combine at each stencil point.

    Parameters
    ----------
    support : int
        Grid points of the stencil along one axis (odd).
    oversampling : int
        Table samples per uv cell.
    interpolation : str
        ``"nearest"`` or ``"linear"`` (a key of :data:`INTERPOLATIONS`).
    offsets : array_like
        Nearest grid point minus the visibility position, cells, any shape.

    Returns
    -------
    indices : numpy.ndarray of int
        Shape ``offsets.shape + (2, support)``: the two table entries combined
        at each stencil point (``nearest`` uses the first only).
    weights : numpy.ndarray
        Same shape, the weight of each entry (they sum to 1 at every stencil
        point).
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    support = int(support)
    oversampling = int(oversampling)
    center = support // 2
    rel = (np.arange(support) - center).astype(np.float64)
    last = table_size(support, oversampling) - 1
    shape = offsets.shape + (2, support)
    indices = np.zeros(shape, dtype=np.int64)
    weights = np.zeros(shape, dtype=np.float64)
    if interpolation == "linear":
        pos = np.abs(oversampling * rel + (offsets * oversampling)[..., None])
        k = np.minimum(pos.astype(np.int64), last)
        k1 = np.minimum(k + 1, last)
        w = pos - k
        indices[..., 0, :] = k
        indices[..., 1, :] = k1
        weights[..., 0, :] = 1.0 - w
        weights[..., 1, :] = w
    elif interpolation == "nearest":
        center_index = np.floor(offsets * oversampling + 0.5).astype(np.int64)
        indices[..., 0, :] = np.abs(
            oversampling * rel.astype(np.int64) + center_index[..., None]
        )
        weights[..., 0, :] = 1.0
    else:
        raise ValueError(f"unknown kernel interpolation {interpolation!r}")
    return indices, weights


def stencil_values(table, support, oversampling, interpolation, offsets):
    """The kernel values the gridders apply at the stencil points.

    Parameters
    ----------
    table : numpy.ndarray
        The half kernel table (see the module docstring).
    support, oversampling, interpolation, offsets
        See :func:`stencil_lookup`.

    Returns
    -------
    numpy.ndarray
        Shape ``offsets.shape + (support,)``.
    """
    table = np.asarray(table, dtype=np.float64)
    indices, weights = stencil_lookup(support, oversampling, interpolation, offsets)
    return np.sum(table[indices] * weights, axis=-2)


def table_transform(table, oversampling, interpolation, x):
    """Fourier transform of the kernel the gridders apply.

    With ``nearest`` lookup the kernel applied is piecewise constant over the
    table cells (``[(k - 1/2) / M, (k + 1/2) / M]`` takes entry ``k``, ``M``
    the oversampling), with ``linear`` lookup it is the piecewise linear
    interpolant of the entries. Both have a closed form transform:

    .. math::

        \\tilde c(x) = \\frac{s(x)}{M} \\Big[T_0 + 2 \\sum_{k \\ge 1} T_k
        \\cos(2 \\pi k x / M)\\Big]

    with ``s`` the sinc of ``x / M`` (nearest) or its square (linear). The
    image plane correction of a kernel is this transform normalised to 1 at
    the image centre, so that the gridded image is corrected for exactly the
    kernel applied (sampling the continuous kernel's transform instead leaves
    a position dependent flux scale error of the order of the table's
    interpolation error).

    Parameters
    ----------
    table : numpy.ndarray
        The half kernel table.
    oversampling : int
        Table samples per uv cell.
    interpolation : str
        ``"nearest"`` or ``"linear"``.
    x : array_like
        Image coordinates in units of the padded image width (``[-0.5, 0.5)``
        spans the padded image), any shape.

    Returns
    -------
    numpy.ndarray
        The transform at ``x``, same shape (float64).
    """
    table = np.asarray(table, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    oversampling = int(oversampling)
    # the cosine series in blocks of table entries, so that a long table
    # (a large oversampling) against a long image axis stays within memory
    series = np.full(x.shape, table[0], dtype=np.float64)
    block = max(1, 2**22 // max(x.size, 1))
    for start in range(1, table.size, block):
        k = np.arange(start, min(start + block, table.size))
        phase = 2.0 * np.pi * np.multiply.outer(x, k / oversampling)
        series += 2.0 * np.sum(table[k] * np.cos(phase), axis=-1)
    s = np.sinc(x / oversampling)
    if interpolation == "linear":
        s = s * s
    elif interpolation != "nearest":
        raise ValueError(f"unknown kernel interpolation {interpolation!r}")
    return s * series / oversampling
