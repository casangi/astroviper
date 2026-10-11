"""The gridding convolution kernel of the single field gridder.

The gridder and the degridder convolve every visibility with a separable
kernel they read from a sampled table (``oversampling`` samples per uv cell
over the half width of the kernel), and the image is divided by the kernel's
Fourier transform afterwards (the correction, see
:func:`~astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder.ifft_norm_img_xds`).
This module holds the parameters that select the kernel, the
:class:`GriddingKernel` that carries the table and makes the correction, and
the factory the imaging functions use.

Three kernel types exist:

``prolate_spheroidal``
    Schwab's rational approximation of the ``m = 6``, ``alpha = 1``
    spheroidal function (support 7 only); the kernel CASA uses. The default
    sampling (oversampling 100, nearest lookup) keeps Schwab's analytic
    correction; any other sampling or lookup is corrected by the transform of
    the table applied. It concentrates the kernel's transform within the
    unpadded image, so it does not profit from the FFT padding.
``exponential_semicircle``
    The modified exponential of a semicircle of Barnett et al. (2019) and
    Arras et al. (2021), any odd support, with its shape chosen for the
    image region the padding keeps
    (:mod:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle`).
``least_misfit``
    The table that minimises the map error over the image region kept and
    over the positions of a visibility within a uv cell, for the lookup the
    gridders apply, after Ye et al. (2020)
    (:mod:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_least_misfit`).

The corrections of the last two are the transforms of the kernels the
gridders apply, the sampled tables with their lookup
(:func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.kernel_lookup.table_transform`).
"""

from __future__ import annotations

import functools

import numpy as np

from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    coordinates,
    create_prolate_spheroidal_kernel_1D,
    prolate_spheroidal_function,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.kernel_lookup import (
    INTERPOLATIONS,
    stencil_values,
    table_size,
    table_transform,
)

KERNEL_TYPES = ("prolate_spheroidal", "exponential_semicircle", "least_misfit")
# the support (number of grid points the kernel spans) each kernel type accepts
PROLATE_SPHEROIDAL_SUPPORT = 7
MIN_SUPPORT = 3
MIN_DESIGN_PADDING = 1.05
DEFAULT_GRIDDING_KERNEL_PARAMS = {
    "type": "prolate_spheroidal",
    "support": PROLATE_SPHEROIDAL_SUPPORT,
    "oversampling": 100,
    "interpolation": "nearest",
}
# the numerical corrections are kept above this fraction of their centre
# value, so that the division of the padded image stays finite; a sensible
# kernel is far above it over the image kept (see GriddingKernel.correction_1D)
CORRECTION_FLOOR = 1e-6

__all__ = [
    "DEFAULT_GRIDDING_KERNEL_PARAMS",
    "INTERPOLATIONS",
    "KERNEL_TYPES",
    "GriddingKernel",
    "as_gridding_kernel",
    "kept_half_width",
    "make_gridding_kernel",
    "resolve_gridding_kernel_params",
]


def resolve_gridding_kernel_params(gridding_kernel_params=None):
    """The gridding kernel parameters with the defaults filled in and checked.

    Parameters
    ----------
    gridding_kernel_params : dict, optional
        Keys ``type`` (one of :data:`KERNEL_TYPES`), ``support`` (the odd
        number of grid points the kernel spans along one axis; the prolate
        spheroidal kernel exists for 7 only, Schwab's rational approximation
        of the ``m = 6``, ``alpha = 1`` function, the others for any odd
        support of at least 3), ``oversampling`` (samples of the kernel table
        per uv cell, an integer of at least 1) and ``interpolation`` (how the
        gridders read the table at a stencil point: ``"nearest"`` takes the
        nearest sample, ``"linear"`` interpolates the two neighbouring
        samples). Missing keys take the defaults of
        :data:`DEFAULT_GRIDDING_KERNEL_PARAMS`.

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
    if (
        not isinstance(support, int | np.integer)
        or support < MIN_SUPPORT
        or support % 2 == 0
    ):
        raise ValueError(
            "gridding_kernel_params['support'] must be an odd integer of at least "
            f"{MIN_SUPPORT}, got {support!r}"
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


def kept_half_width(fft_padding):
    """Half width of the image kept, in units of the padded image width.

    The kernels designed for the padding (``exponential_semicircle`` and
    ``least_misfit``) suppress the aliases of the periodic FFT image over
    ``|x| <= 1 / (2 * fft_padding)``, the image that remains after the
    padding is cropped.

    Parameters
    ----------
    fft_padding : float
        The padding factor of the FFT grid (at least 1).

    Returns
    -------
    float
    """
    if fft_padding is None:
        raise ValueError(
            "the exponential_semicircle and least_misfit gridding kernels are "
            "designed for the image region the padding keeps: pass fft_padding"
        )
    fft_padding = float(fft_padding)
    if fft_padding < MIN_DESIGN_PADDING:
        raise ValueError(
            "the exponential_semicircle and least_misfit gridding kernels need "
            f"fft_padding >= {MIN_DESIGN_PADDING} to suppress the aliases at the "
            f"edge of the image kept; got {fft_padding}"
        )
    return 0.5 / fft_padding


class GriddingKernel:
    """A sampled gridding kernel and its image plane correction.

    Parameters
    ----------
    table : numpy.ndarray
        The half kernel, float64, ``oversampling * (support // 2 + 1)`` values:
        entry ``k`` is the kernel at ``k / oversampling`` uv cells from its
        centre, out to ``support // 2 + 1`` cells (the kernel is zero beyond
        its half width ``support / 2``). This is the layout the C++ gridder
        and degridder read (``kernel_lookup.hpp``).
    support : int
        Number of grid points the kernel spans along one axis (odd).
    oversampling : int
        Samples of the table per uv cell.
    kernel_type : str
        The kernel type the table holds (one of :data:`KERNEL_TYPES`).
    interpolation : str
        How the gridders read the table (a key of :data:`INTERPOLATIONS`).
    correction : callable, optional
        The image plane correction at image coordinates (an array in units
        of the padded image width, ``[-0.5, 0.5)`` spanning the padded
        image), normalised to 1 at the image centre. The default is Schwab's
        correction of the prolate spheroidal kernel; the other kernel types
        pass the transform of the kernel applied.
    shape : dict, optional
        The numbers that define the kernel beyond the four parameters (the
        shape parameters of the exponential of a semicircle, the aliasing
        error or misfit, the padding designed for), for the record.
    """

    def __init__(
        self,
        table,
        support,
        oversampling,
        kernel_type="prolate_spheroidal",
        interpolation="nearest",
        correction=None,
        shape=None,
    ):
        table = np.ascontiguousarray(table, dtype=np.float64)
        expected = table_size(support, oversampling)
        if table.ndim != 1 or table.size != expected:
            raise ValueError(
                f"a kernel table of support {support} and oversampling "
                f"{oversampling} has {expected} entries, got shape {table.shape}"
            )
        if interpolation not in INTERPOLATIONS:
            raise ValueError(f"unknown kernel interpolation {interpolation!r}")
        if kernel_type not in KERNEL_TYPES:
            raise ValueError(f"unknown kernel type {kernel_type!r}")
        self.table = table
        self.support = int(support)
        self.oversampling = int(oversampling)
        self.kernel_type = kernel_type
        self.interpolation = interpolation
        self.shape = dict(shape or {})
        self._correction = correction
        self._correction_1D = {}

    def __repr__(self):
        shape = "".join(
            f", {key}={value:.4g}" if isinstance(value, float) else f", {key}={value!r}"
            for key, value in self.shape.items()
        )
        return (
            f"GriddingKernel({self.kernel_type}, support={self.support}, "
            f"oversampling={self.oversampling}, interpolation={self.interpolation}"
            f"{shape})"
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

    def stencil_values(self, offsets):
        """The kernel values the gridders apply at the stencil points.

        Parameters
        ----------
        offsets : array_like
            Nearest grid point minus the visibility position, cells, any
            shape.

        Returns
        -------
        numpy.ndarray
            Shape ``offsets.shape + (support,)``, stencil point ``i`` being
            ``i - support // 2`` cells from the nearest grid point.
        """
        return stencil_values(
            self.table, self.support, self.oversampling, self.interpolation, offsets
        )

    def transform(self, x):
        """Fourier transform of the kernel the gridders apply.

        Parameters
        ----------
        x : array_like
            Image coordinates in units of the padded image width.

        Returns
        -------
        numpy.ndarray
            See :func:`~.kernel_lookup.table_transform`.
        """
        return table_transform(self.table, self.oversampling, self.interpolation, x)

    def correction(self, x):
        """The image plane correction at image coordinates ``x``.

        The gridded image is divided by it: the transform of the kernel at
        the image coordinates (units of the padded image width), normalised
        to 1 at the image centre. Beyond the image kept the numerical
        corrections are floored at :data:`CORRECTION_FLOOR` so that the
        division of the padded image stays finite (the padding is cropped,
        and the model is zero there).

        Parameters
        ----------
        x : array_like
            Any shape.

        Returns
        -------
        numpy.ndarray
            float64, the shape of ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._correction is None:
            # Schwab's transform of the prolate spheroidal kernel
            return prolate_spheroidal_function(np.abs(2.0 * x))[0]
        return np.maximum(self._correction(x), CORRECTION_FLOOR)

    def correction_1D(self, n_pixels):
        """The image plane correction along one axis of the padded image.

        Parameters
        ----------
        n_pixels : int
            Pixels of the padded image along the axis (the padding factor of
            the image is what makes it larger than the image kept).

        Returns
        -------
        numpy.ndarray
            float64 of length ``n_pixels``, :meth:`correction` at the pixel
            positions (computed once per axis length and kept, read only).
        """
        n_pixels = int(n_pixels)
        if n_pixels not in self._correction_1D:
            values = self.correction(coordinates(n_pixels))
            values.setflags(write=False)
            self._correction_1D[n_pixels] = values
        return self._correction_1D[n_pixels]

    def map_error(self, x, offsets_per_cell=4):
        """The rms map error of the kernel at image positions ``x``.

        The degridded visibility of a unit source at ``x`` over the ideal
        fringe, minus 1, as the rms over the position of the visibility
        within a uv cell: the kernel's aliasing plus the error of its table
        lookup, with the degridder's division by the stencil sum and the
        correction applied. It is what limits the residual of a self
        consistent imaging cycle (one axis; the two axes multiply).

        Parameters
        ----------
        x : array_like
            Image coordinates in units of the padded image width.
        offsets_per_cell : int
            Visibility offsets per table cell the rms is taken over.

        Returns
        -------
        numpy.ndarray
            float64, the shape of ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        n_offsets = int(offsets_per_cell) * self.oversampling
        offsets = -0.5 + (np.arange(n_offsets) + 0.5) / n_offsets
        values = self.stencil_values(offsets)  # (n_off, W)
        rel = np.arange(self.support) - self.support // 2
        phase = np.exp(2j * np.pi * np.multiply.outer(rel, x.ravel()))  # (W, n_x)
        response = values @ phase / values.sum(axis=1)[:, None]
        response /= self.correction(x.ravel())[None, :]
        fringe = np.exp(-2j * np.pi * np.multiply.outer(offsets, x.ravel()))
        error = np.sqrt(np.mean(np.abs(response - fringe) ** 2, axis=0))
        return error.reshape(x.shape)


def _table_correction(table, oversampling, interpolation):
    """The correction of a table: its transform normalised at the centre."""
    centre = table_transform(table, oversampling, interpolation, 0.0)

    def correction(x):
        return table_transform(table, oversampling, interpolation, x) / centre

    return correction


def make_gridding_kernel(gridding_kernel_params=None, fft_padding=None):
    """Make the kernel the parameters select.

    The kernel of a parameter set and padding is made once per process and
    returned again on every later call (the imaging functions ask for it at
    every residual update); the kernel, its table and the corrections it
    keeps per image axis are read only.

    Parameters
    ----------
    gridding_kernel_params : dict, optional
        See :func:`resolve_gridding_kernel_params`.
    fft_padding : float, optional
        The padding factor of the FFT grid, which sets the image region the
        ``exponential_semicircle`` and ``least_misfit`` kernels are designed
        for (:func:`kept_half_width`); the prolate spheroidal kernel does
        not depend on it.

    Returns
    -------
    GriddingKernel
    """
    params = resolve_gridding_kernel_params(gridding_kernel_params)
    if params["type"] == "prolate_spheroidal":
        fft_padding = None  # the kernel does not depend on it
    elif fft_padding is not None:
        fft_padding = float(fft_padding)
    return _make_gridding_kernel(
        params["type"],
        params["support"],
        params["oversampling"],
        params["interpolation"],
        fft_padding,
    )


@functools.cache
def _make_gridding_kernel(
    kernel_type, support, oversampling, interpolation, fft_padding
):
    params = {
        "type": kernel_type,
        "support": support,
        "oversampling": oversampling,
        "interpolation": interpolation,
    }
    if params["type"] == "prolate_spheroidal":
        table = create_prolate_spheroidal_kernel_1D(oversampling, support)
        correction = None  # Schwab's analytic correction, the default bit for bit
        if (oversampling, interpolation) != (
            DEFAULT_GRIDDING_KERNEL_PARAMS["oversampling"],
            DEFAULT_GRIDDING_KERNEL_PARAMS["interpolation"],
        ):
            correction = _table_correction(table, oversampling, interpolation)
        kernel = GriddingKernel(
            table,
            support,
            oversampling,
            params["type"],
            interpolation,
            correction=correction,
        )
        kernel.table.setflags(write=False)
        return kernel
    x0 = kept_half_width(fft_padding)
    if params["type"] == "exponential_semicircle":
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_exponential_semicircle import (
            create_exponential_semicircle_kernel_1D,
            exponential_semicircle_parameters,
        )

        beta, mu, error = exponential_semicircle_parameters(support, x0)
        table = create_exponential_semicircle_kernel_1D(oversampling, support, beta, mu)
        shape = {
            "beta": beta,
            "mu": mu,
            "aliasing_error": error,
            "fft_padding": float(fft_padding),
        }
    elif params["type"] == "least_misfit":
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_least_misfit import (
            create_least_misfit_kernel_1D,
        )

        table, info = create_least_misfit_kernel_1D(
            oversampling, support, interpolation, x0
        )
        shape = {"misfit": info["misfit"], "fft_padding": float(fft_padding)}
    else:  # pragma: no cover - resolve_gridding_kernel_params checks the type
        raise ValueError(f"no builder for kernel type {params['type']!r}")
    kernel = GriddingKernel(
        table,
        support,
        oversampling,
        params["type"],
        interpolation,
        correction=_table_correction(table, oversampling, interpolation),
        shape=shape,
    )
    kernel.table.setflags(write=False)
    return kernel


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
