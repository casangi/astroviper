"""The gridding convolution kernel of the single field gridder.

The gridder and the degridder convolve every visibility with a separable
kernel they read from a sampled table (``oversampling`` samples per uv cell
over the half width of the kernel), and the image is divided by the kernel's
Fourier transform afterwards (the correction, see
:func:`~astroviper.processing_functions.imaging.fft_normalize_anti_alias_gridder.ifft_norm_img_xds`).
This module holds the parameters that select the kernel, the factory the
imaging functions use, which makes the kernel's ``gcf_xds`` (see Schema
below), and the functions that make the correction from it.

Four kernel types exist:

``prolate_spheroidal_casa``
    Schwab's rational approximation of the ``m = 6``, ``alpha = 1``
    spheroidal function (support 7 only); the kernel CASA uses. CASA's
    sampling (oversampling 100, nearest lookup,
    :data:`CASA_GRIDDING_KERNEL_PARAMS`) keeps Schwab's analytic
    correction; any other sampling or lookup is corrected by the transform of
    the table applied. Keys left out with this type take CASA's values, so
    ``{"type": "prolate_spheroidal_casa"}`` alone is CASA's kernel and
    sampling. It concentrates the kernel's transform within the unpadded
    image, so it does not profit from the FFT padding.
``prolate_spheroidal``
    The generalised prolate spheroidal function of Schwab (1984), any odd
    support, with its bandwidth and weight exponent chosen for the image
    region the padding keeps
    (:mod:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_generalized_prolate_spheroidal`).
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

The corrections of the last three are the transforms of the kernels the
gridders apply, the sampled tables with their lookup
(:func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.kernel_lookup.table_transform`).

The default (:data:`DEFAULT_GRIDDING_KERNEL_PARAMS`) is the ``least_misfit``
kernel of support 9, oversampling 300 and linear lookup, the kernel RADPS
Memo 13a selects. It is designed for the padding, so it needs ``fft_padding``;
for 1.2 its solved table ships with the package. A bare kernel table
(:func:`as_gridding_kernel`) is CASA's kernel
(:data:`CASA_GRIDDING_KERNEL_PARAMS`). The FFT normalization has no default
kernel: it is given the kernel the visibilities are gridded or degridded
with.

The gridder ``"direct_fourier_transform"`` uses no kernel: it replaces
gridding and FFT by the exact direct Fourier transform with the w term
(:mod:`~astroviper.processing_functions.imaging.direct_fourier_transform`).

Schema
------
A kernel is the anti-alias term of a gridding convolution function dataset,
``gcf_xds``: an :class:`xarray.Dataset` with the data variable ``ANTI_ALIAS``
on the dimension ``anti_alias_sample``, whose coordinate is
``k / oversampling``. ``ANTI_ALIAS`` holds the half kernel, float64 and
C-contiguous, ``oversampling * (support // 2 + 1)`` values: entry ``k`` is the
kernel at ``k / oversampling`` uv cells from its centre, out to
``support // 2 + 1`` cells (the kernel is zero beyond its half width
``support / 2``). This is the layout the C++ gridder and degridder read
(``kernel_lookup.hpp``). The attributes of ``ANTI_ALIAS`` are:

``kernel_type``
    The kernel type the table holds (one of :data:`KERNEL_TYPES`).
``support``
    Number of grid points the kernel spans along one axis (odd), an int.
``oversampling``
    Samples of the table per uv cell, an int.
``interpolation``
    How the gridders read the table (a key of :data:`INTERPOLATIONS`).
``correction``
    The image plane correction (:func:`correction`): ``"analytic"``, Schwab's
    correction of the ``prolate_spheroidal_casa`` kernel, or
    ``"table_transform"``, the transform of the kernel the gridders apply.
``shape``
    A dict of the numbers that define the kernel beyond the four parameters
    (the shape parameters of the prolate spheroidal function or of the
    exponential of a semicircle, the aliasing error or misfit, the padding
    designed for, the file of a shipped least misfit table), for the record;
    empty for the ``prolate_spheroidal_casa`` kernel.
"""

from __future__ import annotations

import functools

import numpy as np
import xarray as xr

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

KERNEL_TYPES = (
    "prolate_spheroidal_casa",
    "prolate_spheroidal",
    "exponential_semicircle",
    "least_misfit",
)
# the support (number of grid points the kernel spans) each kernel type accepts
PROLATE_SPHEROIDAL_SUPPORT = 7
MIN_SUPPORT = 3
MIN_DESIGN_PADDING = 1.05
# CASA's kernel and sampling, the only one with Schwab's analytic correction; a bare
# kernel table is this kernel
CASA_GRIDDING_KERNEL_PARAMS = {
    "type": "prolate_spheroidal_casa",
    "support": PROLATE_SPHEROIDAL_SUPPORT,
    "oversampling": 100,
    "interpolation": "nearest",
}
# The kernel RADPS Memo 13a selects by its dynamic range rule (kernel errors at most
# 0.1 sigma at a dynamic range of 1e4) at the grid memory of fft_padding 1.2, the
# kernel delphi uses (GRIDDING_KERNEL_PARAMS of utils/imaging_parameter_heuristics.py,
# which gives the measurements); CASA's kernel misses that rule by a factor 13.
DEFAULT_GRIDDING_KERNEL_PARAMS = {
    "type": "least_misfit",
    "support": 9,
    "oversampling": 300,
    "interpolation": "linear",
}
# the numerical corrections are kept above this fraction of their centre
# value, so that the division of the padded image stays finite; a sensible
# kernel is far above it over the image kept (see correction_1D)
CORRECTION_FLOOR = 1e-6

__all__ = [
    "CASA_GRIDDING_KERNEL_PARAMS",
    "DEFAULT_GRIDDING_KERNEL_PARAMS",
    "INTERPOLATIONS",
    "KERNEL_TYPES",
    "anti_alias_kernel_arguments",
    "as_gridding_kernel",
    "correction",
    "correction_1D",
    "kept_half_width",
    "make_gridding_kernel",
    "map_error",
    "resolve_gridding_kernel_params",
]


def resolve_gridding_kernel_params(gridding_kernel_params=None):
    """The gridding kernel parameters with the defaults filled in and checked.

    Parameters
    ----------
    gridding_kernel_params : dict, optional
        Keys ``type`` (one of :data:`KERNEL_TYPES`), ``support`` (the odd
        number of grid points the kernel spans along one axis; the
        ``prolate_spheroidal_casa`` kernel exists for 7 only, Schwab's
        rational approximation of the ``m = 6``, ``alpha = 1`` function, the
        others for any odd support of at least 3), ``oversampling`` (samples
        of the kernel table per uv cell, an integer of at least 1) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, ``"linear"``
        interpolates the two neighbouring samples). Missing keys take the
        defaults of :data:`DEFAULT_GRIDDING_KERNEL_PARAMS`, or, for the type
        ``"prolate_spheroidal_casa"``, those of
        :data:`CASA_GRIDDING_KERNEL_PARAMS` (support 7, oversampling 100,
        nearest lookup), so that type alone is CASA's kernel and sampling.

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
    from astroviper.processing_functions.imaging.config_check._gridding import (
        check_gridding_kernel_parameters,
    )

    try:
        check_gridding_kernel_parameters(gridding_kernel_params)
    except TypeError as error:
        raise ValueError(str(error)) from error
    # Build effective parameters without changing the caller's recipe.
    if gridding_kernel_params is not None:
        if gridding_kernel_params.get("type") == "prolate_spheroidal_casa":
            defaults = CASA_GRIDDING_KERNEL_PARAMS
        else:
            defaults = DEFAULT_GRIDDING_KERNEL_PARAMS
    else:
        defaults = DEFAULT_GRIDDING_KERNEL_PARAMS
    params = {**defaults, **(gridding_kernel_params or {})}
    params["support"] = int(params["support"])
    params["oversampling"] = int(params["oversampling"])
    return params


def kept_half_width(fft_padding):
    """Half width of the image kept, in units of the padded image width.

    The kernels designed for the padding (``prolate_spheroidal``,
    ``exponential_semicircle`` and ``least_misfit``) suppress the aliases
    of the periodic FFT image over ``|x| <= 1 / (2 * fft_padding)``, the
    image that remains after the padding is cropped.

    Parameters
    ----------
    fft_padding : float
        The padding factor of the FFT grid (at least 1).

    Returns
    -------
    float
    """
    from astroviper.processing_functions.imaging.config_check._gridding import (
        check_kernel_padding,
    )

    check_kernel_padding(fft_padding)
    return 0.5 / float(fft_padding)


def _gcf_xds_from_table(
    table,
    support,
    oversampling,
    kernel_type="prolate_spheroidal_casa",
    interpolation="nearest",
    correction="analytic",
    shape=None,
):
    """The ``gcf_xds`` of a sampled gridding kernel (see the module docstring).

    Parameters
    ----------
    table : numpy.ndarray
        The half kernel, ``oversampling * (support // 2 + 1)`` values; the
        dataset holds the table itself when it is float64 and C-contiguous.
    support : int
        Number of grid points the kernel spans along one axis (odd).
    oversampling : int
        Samples of the table per uv cell.
    kernel_type : str
        The kernel type the table holds (one of :data:`KERNEL_TYPES`).
    interpolation : str
        How the gridders read the table (a key of :data:`INTERPOLATIONS`).
    correction : str
        ``"analytic"`` or ``"table_transform"`` (see :func:`correction`).
    shape : dict, optional
        The numbers that define the kernel beyond the four parameters, for
        the record.

    Returns
    -------
    xarray.Dataset
    """
    table = np.ascontiguousarray(table, dtype=np.float64)
    _check_table_size(table, support, oversampling)
    from astroviper.processing_functions.imaging.config_check._gridding import (
        check_kernel_options,
    )

    check_kernel_options(kernel_type, interpolation)
    attrs = {
        "kernel_type": kernel_type,
        "support": int(support),
        "oversampling": int(oversampling),
        "interpolation": interpolation,
        "correction": correction,
        "shape": dict(shape or {}),
    }
    return xr.Dataset(
        {"ANTI_ALIAS": (("anti_alias_sample",), table, attrs)},
        coords={"anti_alias_sample": np.arange(table.size) / int(oversampling)},
    )


def _check_table_size(table, support, oversampling):
    """Refuse a table that is not the half kernel of this support and oversampling."""
    expected = table_size(support, oversampling)
    if table.ndim != 1 or table.size != expected:
        raise ValueError(
            f"a kernel table of support {support} and oversampling "
            f"{oversampling} has {expected} entries, got shape {table.shape}"
        )


def anti_alias_kernel_arguments(gcf_xds):
    """The kernel arguments of the C++ gridders and degridder.

    ``anti_alias_grid``, ``anti_alias_grid_uv_sampling`` and
    ``anti_alias_degrid`` take the table as their positional argument
    ``cgk_1D`` and ``support``, ``oversampling`` and ``interpolation`` as
    keywords.

    Parameters
    ----------
    gcf_xds : xarray.Dataset
        The kernel's ``gcf_xds`` (see the module docstring).

    Returns
    -------
    table : numpy.ndarray
        ``gcf_xds["ANTI_ALIAS"].values``, the half kernel table.
    support : int
    oversampling : int
    interpolation : int
        The C++ code of the lookup (a value of :data:`INTERPOLATIONS`).

    Raises
    ------
    ValueError
        If the table is not the half kernel of its ``support`` and
        ``oversampling``: the C++ lookup does not check its size.
    """
    term = gcf_xds["ANTI_ALIAS"]
    table = term.values
    _check_table_size(table, term.attrs["support"], term.attrs["oversampling"])
    return (
        table,
        term.attrs["support"],
        term.attrs["oversampling"],
        INTERPOLATIONS[term.attrs["interpolation"]],
    )


def correction(gcf_xds, x):
    """The image plane correction at image coordinates ``x``.

    The gridded image is divided by it: the transform of the kernel at
    the image coordinates (units of the padded image width), normalised
    to 1 at the image centre. Beyond the image kept the numerical
    corrections are floored at :data:`CORRECTION_FLOOR` so that the
    division of the padded image stays finite (the padding is cropped,
    and the model is zero there).

    Parameters
    ----------
    gcf_xds : xarray.Dataset
        The kernel's ``gcf_xds`` (see the module docstring).
    x : array_like
        Any shape.

    Returns
    -------
    numpy.ndarray
        float64, the shape of ``x``.

    Raises
    ------
    ValueError
        For a ``correction`` attribute other than ``"analytic"`` and
        ``"table_transform"``.
    """
    term = gcf_xds["ANTI_ALIAS"]
    return _correction(
        term.attrs["correction"],
        term.values,
        term.attrs["oversampling"],
        term.attrs["interpolation"],
        x,
    )


def _correction(correction, table, oversampling, interpolation, x):
    """:func:`correction` of a table and its attributes."""
    x = np.asarray(x, dtype=np.float64)
    if correction == "analytic":
        # Schwab's transform of the prolate spheroidal kernel
        return prolate_spheroidal_function(np.abs(2.0 * x))[0]
    if correction != "table_transform":
        raise ValueError(f"unknown kernel correction {correction!r}")
    return np.maximum(
        _table_correction(table, oversampling, interpolation)(x), CORRECTION_FLOOR
    )


def correction_1D(gcf_xds, n_pixels):
    """The image plane correction along one axis of the padded image.

    Parameters
    ----------
    gcf_xds : xarray.Dataset
        The kernel's ``gcf_xds`` (see the module docstring).
    n_pixels : int
        Pixels of the padded image along the axis (the padding factor of
        the image is what makes it larger than the image kept).

    Returns
    -------
    numpy.ndarray
        float64 of length ``n_pixels``, :func:`correction` at the pixel
        positions (computed once per table, correction and axis length and
        kept, read only).
    """
    term = gcf_xds["ANTI_ALIAS"]
    return _correction_1D(
        term.attrs["correction"],
        term.attrs["oversampling"],
        term.attrs["interpolation"],
        np.ascontiguousarray(term.values, dtype=np.float64).tobytes(),
        int(n_pixels),
    )


@functools.cache
def _correction_1D(correction, oversampling, interpolation, table_bytes, n_pixels):
    # the key holds the table's bytes, so that a table with the same
    # attributes cannot get another table's correction
    table = np.frombuffer(table_bytes, dtype=np.float64)
    values = _correction(
        correction, table, oversampling, interpolation, coordinates(n_pixels)
    )
    values.setflags(write=False)
    return values


def map_error(gcf_xds, x, offsets_per_cell=4):
    """The rms map error of the kernel at image positions ``x``.

    The degridded visibility of a unit source at ``x`` over the ideal
    fringe, minus 1, as the rms over the position of the visibility
    within a uv cell: the kernel's aliasing plus the error of its table
    lookup, with the degridder's division by the stencil sum and the
    correction applied. It is what limits the residual of a self
    consistent imaging cycle (one axis; the two axes multiply).

    Parameters
    ----------
    gcf_xds : xarray.Dataset
        The kernel's ``gcf_xds`` (see the module docstring).
    x : array_like
        Image coordinates in units of the padded image width.
    offsets_per_cell : int
        Visibility offsets per table cell the rms is taken over.

    Returns
    -------
    numpy.ndarray
        float64, the shape of ``x``.
    """
    term = gcf_xds["ANTI_ALIAS"]
    support = term.attrs["support"]
    oversampling = term.attrs["oversampling"]
    x = np.asarray(x, dtype=np.float64)
    n_offsets = int(offsets_per_cell) * oversampling
    offsets = -0.5 + (np.arange(n_offsets) + 0.5) / n_offsets
    values = stencil_values(
        term.values, support, oversampling, term.attrs["interpolation"], offsets
    )  # (n_off, W)
    rel = np.arange(support) - support // 2
    phase = np.exp(2j * np.pi * np.multiply.outer(rel, x.ravel()))  # (W, n_x)
    response = values @ phase / values.sum(axis=1)[:, None]
    response /= correction(gcf_xds, x.ravel())[None, :]
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
    """Make the ``gcf_xds`` of the kernel the parameters select.

    The ``gcf_xds`` of a parameter set and padding is made once per process
    and returned again on every later call (the imaging functions ask for it
    at every residual update). The returned dataset is shared: do not modify
    it. Its table and the corrections :func:`correction_1D` keeps per image
    axis are read only.

    Parameters
    ----------
    gridding_kernel_params : dict, optional
        See :func:`resolve_gridding_kernel_params`.
    fft_padding : float, optional
        The padding factor of the FFT grid, which sets the image region the
        ``prolate_spheroidal``, ``exponential_semicircle`` and
        ``least_misfit`` kernels (the default among them) are designed for
        (:func:`kept_half_width`); the ``prolate_spheroidal_casa`` kernel does
        not depend on it.

    Returns
    -------
    xarray.Dataset
        The kernel's ``gcf_xds`` (see the module docstring).

    Raises
    ------
    ValueError
        For parameters :func:`resolve_gridding_kernel_params` refuses, a
        padding :func:`kept_half_width` refuses, or a kernel designed for the
        padding whose correction at the edge of the image kept is below
        :data:`CORRECTION_FLOOR` (a support too large for the padding).
    """
    params = resolve_gridding_kernel_params(gridding_kernel_params)
    if params["type"] == "prolate_spheroidal_casa":
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
    if params["type"] == "prolate_spheroidal_casa":
        table = create_prolate_spheroidal_kernel_1D(oversampling, support)
        correction = "analytic"  # Schwab's analytic correction, CASA's bit for bit
        if (oversampling, interpolation) != (
            CASA_GRIDDING_KERNEL_PARAMS["oversampling"],
            CASA_GRIDDING_KERNEL_PARAMS["interpolation"],
        ):
            correction = "table_transform"
        gcf_xds = _gcf_xds_from_table(
            table,
            support,
            oversampling,
            params["type"],
            interpolation,
            correction=correction,
        )
        gcf_xds["ANTI_ALIAS"].values.setflags(write=False)
        return gcf_xds
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
        if "shipped_table" in info:
            shape["shipped_table"] = info["shipped_table"]
    elif params["type"] == "prolate_spheroidal":
        from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_generalized_prolate_spheroidal import (
            create_generalized_prolate_spheroidal_kernel_1D,
            generalized_prolate_spheroidal_parameters,
        )

        c, alpha, error = generalized_prolate_spheroidal_parameters(support, x0)
        table = create_generalized_prolate_spheroidal_kernel_1D(
            oversampling, support, c, alpha
        )
        shape = {
            "c": c,
            "alpha": alpha,
            "aliasing_error": error,
            "fft_padding": float(fft_padding),
        }
    else:  # pragma: no cover - resolve_gridding_kernel_params checks the type
        raise ValueError(f"no builder for kernel type {params['type']!r}")
    correction = _table_correction(table, oversampling, interpolation)
    edge = float(correction(x0))
    if not edge >= CORRECTION_FLOOR:  # refuses a NaN too
        raise ValueError(
            f"the {params['type']} gridding kernel of support {support} designed "
            f"for fft_padding {fft_padding} has a correction of {edge:.2g} at the "
            f"edge of the image kept, below the floor of {CORRECTION_FLOOR:g} the "
            "correction is clamped at, so the image near that edge would be scaled "
            "wrongly: use a larger fft_padding or a smaller support"
        )
    gcf_xds = _gcf_xds_from_table(
        table,
        support,
        oversampling,
        params["type"],
        interpolation,
        correction="table_transform",
        shape=shape,
    )
    gcf_xds["ANTI_ALIAS"].values.setflags(write=False)
    return gcf_xds


def as_gridding_kernel(cgk_1D):
    """The ``gcf_xds`` of a bare kernel table.

    A bare table is the prolate spheroidal half kernel of support 7 (whose
    oversampling is its length over 4): the ``prolate_spheroidal_casa``
    kernel with nearest lookup and Schwab's analytic correction, whatever the
    oversampling.

    Parameters
    ----------
    cgk_1D : numpy.ndarray
        A table of the ``prolate_spheroidal_casa`` kernel.

    Returns
    -------
    xarray.Dataset
        The kernel's ``gcf_xds`` (see the module docstring).
    """
    table = np.asarray(cgk_1D)
    cells = PROLATE_SPHEROIDAL_SUPPORT // 2 + 1
    if table.ndim != 1 or table.size % cells != 0:
        raise ValueError(
            "a bare kernel table must be the prolate spheroidal half kernel of "
            f"support {PROLATE_SPHEROIDAL_SUPPORT} (a multiple of {cells} entries), "
            f"got shape {table.shape}"
        )
    return _gcf_xds_from_table(table, PROLATE_SPHEROIDAL_SUPPORT, table.size // cells)
