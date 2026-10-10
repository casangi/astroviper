"""The gridding convolution kernel of the single field gridder.

The gridder and the degridder convolve every visibility with a separable
kernel they read from a sampled table (``oversampling`` samples per uv cell
over the half width of the kernel), and the image is divided by the kernel's
Fourier transform afterwards (the correction, see
:func:`~astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder.ifft_norm_img_xds`).
This module holds the parameters that select the kernel, the
:class:`GriddingKernel` that carries the table and makes the correction, and
the factory the imaging functions use.
"""

from __future__ import annotations

import numpy as np

from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    coordinates,
    create_prolate_spheroidal_kernel_1D,
    prolate_spheroidal_function,
)

KERNEL_TYPES = ("prolate_spheroidal",)
# how the gridders read the kernel table: the C++ codes of kernel_lookup.hpp
INTERPOLATIONS = {"nearest": 0, "linear": 1}
# the support (number of grid points the kernel spans) each kernel type accepts
PROLATE_SPHEROIDAL_SUPPORT = 7
DEFAULT_GRIDDING_KERNEL_PARAMS = {
    "type": "prolate_spheroidal",
    "support": PROLATE_SPHEROIDAL_SUPPORT,
    "oversampling": 100,
    "interpolation": "nearest",
}


def resolve_gridding_kernel_params(gridding_kernel_params=None):
    """The gridding kernel parameters with the defaults filled in and checked.

    Parameters
    ----------
    gridding_kernel_params : dict, optional
        Keys ``type`` (``"prolate_spheroidal"``), ``support`` (the odd number
        of grid points the kernel spans along one axis; the prolate spheroidal
        kernel exists for 7 only, Schwab's rational approximation of the
        ``m = 6``, ``alpha = 1`` function), ``oversampling`` (samples of the
        kernel table per uv cell, an integer of at least 1) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, ``"linear"``
        interpolates the two neighbouring samples). Missing keys take the
        defaults of :data:`DEFAULT_GRIDDING_KERNEL_PARAMS`.

    Returns
    -------
    dict
        A new dict with every key.

    Raises
    ------
    ValueError
        For an unknown key or kernel type, a support the kernel type does not
        have, or an oversampling that is not a positive integer.
    """
    params = dict(DEFAULT_GRIDDING_KERNEL_PARAMS)
    if gridding_kernel_params:
        unknown = set(gridding_kernel_params) - set(params)
        if unknown:
            raise ValueError(
                f"unknown gridding_kernel_params key(s) {sorted(unknown)}; "
                f"expected {sorted(params)}"
            )
        params.update(gridding_kernel_params)
    if params["type"] not in KERNEL_TYPES:
        raise ValueError(
            f"gridding_kernel_params['type'] {params['type']!r} is not one of "
            f"{KERNEL_TYPES}"
        )
    support = params["support"]
    if not isinstance(support, int | np.integer) or support < 1 or support % 2 == 0:
        raise ValueError(
            f"gridding_kernel_params['support'] must be an odd positive integer, "
            f"got {support!r}"
        )
    if params["type"] == "prolate_spheroidal" and support != PROLATE_SPHEROIDAL_SUPPORT:
        raise ValueError(
            "the prolate spheroidal kernel is Schwab's rational approximation of "
            f"the m = 6, alpha = 1 function and spans {PROLATE_SPHEROIDAL_SUPPORT} "
            f"grid points; got support {support}"
        )
    oversampling = params["oversampling"]
    if not isinstance(oversampling, int | np.integer) or oversampling < 1:
        raise ValueError(
            "gridding_kernel_params['oversampling'] must be a positive integer, "
            f"got {oversampling!r}"
        )
    if params["interpolation"] not in INTERPOLATIONS:
        raise ValueError(
            f"gridding_kernel_params['interpolation'] {params['interpolation']!r} "
            f"is not one of {tuple(INTERPOLATIONS)}"
        )
    params["support"] = int(support)
    params["oversampling"] = int(oversampling)
    return params


class GriddingKernel:
    """A sampled gridding kernel and its image plane correction.

    Parameters
    ----------
    table : numpy.ndarray
        The half kernel, float64, ``oversampling * (support // 2 + 1)`` values:
        entry ``k`` is the kernel at ``k / oversampling`` uv cells from its
        centre, and the last ``oversampling`` entries are zero (the kernel is
        zero beyond its half width, which the gridder's stencil may reach).
        This is the layout the C++ gridder and degridder read.
    support : int
        Number of grid points the kernel spans along one axis (odd).
    oversampling : int
        Samples of the table per uv cell.
    kernel_type : str
        The kernel type the table holds (a key of :data:`KERNEL_TYPES`).
    interpolation : str
        How the gridders read the table (a key of :data:`INTERPOLATIONS`).
    """

    def __init__(
        self,
        table,
        support,
        oversampling,
        kernel_type="prolate_spheroidal",
        interpolation="nearest",
    ):
        table = np.ascontiguousarray(table, dtype=np.float64)
        expected = oversampling * (support // 2 + 1)
        if table.ndim != 1 or table.size != expected:
            raise ValueError(
                f"a kernel table of support {support} and oversampling "
                f"{oversampling} has {expected} entries, got shape {table.shape}"
            )
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"unknown kernel interpolation {interpolation!r}")
        self.table = table
        self.support = int(support)
        self.oversampling = int(oversampling)
        self.kernel_type = kernel_type
        self.interpolation = interpolation

    def __repr__(self):
        return (
            f"GriddingKernel({self.kernel_type}, support={self.support}, "
            f"oversampling={self.oversampling}, interpolation={self.interpolation})"
        )

    @property
    def params(self):
        """The parameters that make this kernel (a ``gridding_kernel_params`` dict)."""
        return {
            "type": self.kernel_type,
            "support": self.support,
            "oversampling": self.oversampling,
            "interpolation": self.interpolation,
        }

    @property
    def interpolation_code(self):
        """The ``interpolation`` argument of the C++ gridders."""
        return INTERPOLATIONS[self.interpolation]

    def correction_1D(self, n_pixels):
        """The image plane correction along one axis of the padded image.

        The gridded image is divided by this: the kernel's Fourier transform
        at the ``n_pixels`` pixel positions of the padded FFT image (the
        padding factor of the image is what makes ``n_pixels`` larger than the
        image kept).

        Parameters
        ----------
        n_pixels : int
            Pixels of the padded image along the axis.

        Returns
        -------
        numpy.ndarray
            float64 of length ``n_pixels``.
        """
        if self.kernel_type == "prolate_spheroidal":
            # Schwab's transform of the kernel, over the whole padded image
            return prolate_spheroidal_function(np.abs(2.0 * coordinates(n_pixels)))[0]
        raise ValueError(f"no correction for kernel type {self.kernel_type!r}")


def make_gridding_kernel(gridding_kernel_params=None):
    """Make the kernel the parameters select.

    Parameters
    ----------
    gridding_kernel_params : dict, optional
        See :func:`resolve_gridding_kernel_params`.

    Returns
    -------
    GriddingKernel
    """
    params = resolve_gridding_kernel_params(gridding_kernel_params)
    if params["type"] == "prolate_spheroidal":
        table = create_prolate_spheroidal_kernel_1D(
            params["oversampling"], params["support"]
        )
        return GriddingKernel(
            table,
            params["support"],
            params["oversampling"],
            params["type"],
            params["interpolation"],
        )
    raise ValueError(f"no builder for kernel type {params['type']!r}")


def as_gridding_kernel(cgk_1D):
    """The :class:`GriddingKernel` of a kernel argument.

    The imaging functions take either a :class:`GriddingKernel` or, as they
    always did, the bare prolate spheroidal half kernel table of support 7
    (whose oversampling is its length over 4).

    Parameters
    ----------
    cgk_1D : GriddingKernel or numpy.ndarray
        The kernel, or the table of the default prolate spheroidal kernel.

    Returns
    -------
    GriddingKernel
    """
    if isinstance(cgk_1D, GriddingKernel):
        return cgk_1D
    table = np.asarray(cgk_1D)
    cells = PROLATE_SPHEROIDAL_SUPPORT // 2 + 1
    if table.ndim != 1 or table.size % cells != 0:
        raise ValueError(
            "a bare kernel table must be the prolate spheroidal half kernel of "
            f"support {PROLATE_SPHEROIDAL_SUPPORT} (a multiple of {cells} entries), "
            f"got shape {table.shape}"
        )
    return GriddingKernel(table, PROLATE_SPHEROIDAL_SUPPORT, table.size // cells)
