"""
Transform image data between polarization bases.

Supports 4-pol (XX,XY,YX,YY or RR,RL,LR,LL ↔ I,Q,U,V) and
2-pol (XX,YY ↔ I,Q  or  RR,LL ↔ I,V) conversions.

The workhorse is :func:`_select_transform_matrix`, which is the single
source of truth for every supported (input basis, output basis) pair.
:func:`transform_polarization_basis` and
:func:`transform_polarization_basis_image_data_variable` both delegate to it
so that the output polarization labels are never computed in two places.
"""

import copy
from typing import Optional

import numpy as np
import toolviper.utils.logger as logger
import xarray as xr

# ── 4-pol transformation matrices ────────────────────────────────────────────

# correlation → Stokes  (columns: XX, XY, YX, YY)
LINEAR_CORR_TO_STOKES = np.array(
    [
        [0.5, 0, 0, 0.5],  # I = (XX + YY) / 2
        [0.5, 0, 0, -0.5],  # Q = (XX - YY) / 2
        [0, 0.5, 0.5, 0],  # U = (XY + YX) / 2
        [0, -0.5j, 0.5j, 0],  # V = i(YX - XY) / 2
    ],
    dtype=complex,
)

# correlation → Stokes  (columns: RR, RL, LR, LL)
CIRCULAR_CORR_TO_STOKES = np.array(
    [
        [0.5, 0, 0, 0.5],  # I = (RR + LL) / 2
        [0, 0.5, 0.5, 0],  # Q = (RL + LR) / 2
        [0, -0.5j, 0.5j, 0],  # U = i(LR - RL) / 2
        [0.5, 0, 0, -0.5],  # V = (RR - LL) / 2
    ],
    dtype=complex,
)

# Stokes → correlation  (columns: I, Q, U, V)
LINEAR_STOKES_TO_CORR = np.array(
    [
        [0.5, 0.5, 0, 0],  # XX = (I + Q) / 2
        [0, 0, 0.5, 0.5j],  # XY = (U + iV) / 2
        [0, 0, 0.5, -0.5j],  # YX = (U - iV) / 2
        [0.5, -0.5, 0, 0],  # YY = (I - Q) / 2
    ],
    dtype=complex,
)

# Stokes → correlation  (columns: I, Q, U, V)
CIRCULAR_STOKES_TO_CORR = np.array(
    [
        [0.5, 0, 0, 0.5],  # RR = (I + V) / 2
        [0, 0.5, 0.5j, 0],  # RL = (Q + iU) / 2
        [0, 0.5, -0.5j, 0],  # LR = (Q - iU) / 2
        [0.5, 0, 0, -0.5],  # LL = (I - V) / 2
    ],
    dtype=complex,
)

# ── 2-pol transformation matrices ────────────────────────────────────────────

# correlation → Stokes  (columns: XX, YY)
LINEAR_2POL_CORR_TO_STOKES = np.array(
    [
        [0.5, 0.5],  # I = (XX + YY) / 2
        [0.5, -0.5],  # Q = (XX - YY) / 2
    ],
    dtype=complex,
)

# correlation → Stokes  (columns: RR, LL)
CIRCULAR_2POL_CORR_TO_STOKES = np.array(
    [
        [0.5, 0.5],  # I = (RR + LL) / 2
        [0.5, -0.5],  # V = (RR - LL) / 2
    ],
    dtype=complex,
)

# Stokes → correlation  (columns: I, Q)
LINEAR_2POL_STOKES_TO_CORR = np.array(
    [
        [0.5, 0.5],  # XX = (I + Q) / 2
        [0.5, -0.5],  # YY = (I - Q) / 2
    ],
    dtype=complex,
)

# Stokes → correlation  (columns: I, V)
CIRCULAR_2POL_STOKES_TO_CORR = np.array(
    [
        [0.5, 0.5],  # RR = (I + V) / 2
        [0.5, -0.5],  # LL = (I - V) / 2
    ],
    dtype=complex,
)

# ── Fallback output labels for custom transformation matrices ─────────────────
# Used when the caller supplies an explicit matrix and a standard basis name.
# Keyed by (new_polarization_basis, n_out).  The 2-pol stokes case is
# ambiguous (IQ vs IV depends on the input) so it is intentionally absent.
_CUSTOM_MATRIX_OUTPUT_LABELS: dict[tuple[str, int], list[str]] = {
    ("stokes", 4): ["I", "Q", "U", "V"],
    ("linear", 4): ["XX", "XY", "YX", "YY"],
    ("linear", 2): ["XX", "YY"],
    ("circular", 4): ["RR", "RL", "LR", "LL"],
    ("circular", 2): ["RR", "LL"],
}


def _select_transform_matrix(
    pol_set: frozenset,
    new_polarization_basis: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return the transformation matrix and ordered polarization labels.

    This is the single source of truth for every supported conversion.
    Both :func:`transform_polarization_basis` and
    :func:`transform_polarization_basis_image_data_variable` call this
    function so that input/output labels and the matrix are always derived
    from the same place.

    Parameters
    ----------
    pol_set : frozenset
        Frozenset of the current polarization coordinate values.
    new_polarization_basis : str
        Target basis: ``'stokes'``, ``'linear'``, or ``'circular'``.

    Returns
    -------
    matrix : np.ndarray, shape (n_out, n_in)
        Transformation matrix where
        ``result[new_pol] = sum_i matrix[new_pol, i] * data[i]``.
    in_pol_labels : list[str]
        Input polarization labels in column order of *matrix*.
    out_pol_labels : list[str]
        Output polarization labels in row order of *matrix*.

    Raises
    ------
    ValueError
        If no transformation is defined for the given combination.
    """
    # ── 4-pol correlations → Stokes ──────────────────────────────────────────
    if pol_set == frozenset({"XX", "XY", "YX", "YY"}):
        if new_polarization_basis == "stokes":
            return LINEAR_CORR_TO_STOKES, ["XX", "XY", "YX", "YY"], ["I", "Q", "U", "V"]

    elif pol_set == frozenset({"RR", "RL", "LR", "LL"}):
        if new_polarization_basis == "stokes":
            return (
                CIRCULAR_CORR_TO_STOKES,
                ["RR", "RL", "LR", "LL"],
                ["I", "Q", "U", "V"],
            )

    # ── 4-pol Stokes → correlations ──────────────────────────────────────────
    elif pol_set == frozenset({"I", "Q", "U", "V"}):
        if new_polarization_basis == "linear":
            return LINEAR_STOKES_TO_CORR, ["I", "Q", "U", "V"], ["XX", "XY", "YX", "YY"]
        elif new_polarization_basis == "circular":
            return (
                CIRCULAR_STOKES_TO_CORR,
                ["I", "Q", "U", "V"],
                ["RR", "RL", "LR", "LL"],
            )

    # ── 2-pol correlations → Stokes ──────────────────────────────────────────
    elif pol_set == frozenset({"XX", "YY"}):
        if new_polarization_basis == "stokes":
            return LINEAR_2POL_CORR_TO_STOKES, ["XX", "YY"], ["I", "Q"]

    elif pol_set == frozenset({"RR", "LL"}):
        if new_polarization_basis == "stokes":
            return CIRCULAR_2POL_CORR_TO_STOKES, ["RR", "LL"], ["I", "V"]

    # ── 2-pol Stokes → correlations ──────────────────────────────────────────
    elif pol_set == frozenset({"I", "Q"}):
        if new_polarization_basis == "linear":
            return LINEAR_2POL_STOKES_TO_CORR, ["I", "Q"], ["XX", "YY"]

    elif pol_set == frozenset({"I", "V"}):
        if new_polarization_basis == "circular":
            return CIRCULAR_2POL_STOKES_TO_CORR, ["I", "V"], ["RR", "LL"]

    raise ValueError(
        f"No transformation defined from polarization set {pol_set!r} "
        f"to basis '{new_polarization_basis}'."
    )


def transform_polarization_basis(
    img_xds: xr.Dataset,
    new_polarization_basis: str,
    transformation_matrix: Optional[np.ndarray] = None,
    overwrite: bool = True,
) -> xr.Dataset:
    """Transform the polarization basis of every data variable in an image dataset.

    Output polarization labels are determined by :func:`_select_transform_matrix`
    for all standard conversions.  When a custom *transformation_matrix* is
    supplied the labels are looked up from ``_CUSTOM_MATRIX_OUTPUT_LABELS``
    using *new_polarization_basis* and the matrix output size; if no match is
    found, integer indices ``[0, 1, …, n_out-1]`` are used.

    Parameters
    ----------
    img_xds : xr.Dataset
        Image dataset with a ``polarization`` dimension of size 2 or 4.
        All data variables must share the same polarization axis.
    new_polarization_basis : str
        Target basis.  One of ``'stokes'``, ``'linear'``, or ``'circular'``
        for built-in conversions; any string is accepted when
        *transformation_matrix* is provided.
    transformation_matrix : np.ndarray of shape (n_out, n_in), optional
        Custom transformation matrix.  When provided, *new_polarization_basis*
        is only used for the output-label fallback lookup and is otherwise
        ignored.
    overwrite : bool, default True
        If ``True`` the input dataset is modified in place.
        If ``False`` a new dataset is returned with copied coordinates and
        attributes but transformed data variables.

    Returns
    -------
    xr.Dataset
        Dataset with transformed data variables and an updated
        ``polarization`` coordinate.
    """
    if overwrite:
        img_transformed_xds = img_xds
    else:
        img_transformed_xds = xr.Dataset()
        # Copy attributes and all coordinates except the polarization values,
        # which will be overwritten below.
        img_transformed_xds.attrs = copy.deepcopy(img_xds.attrs)
        for coord_name in img_xds.coords:
            img_transformed_xds.coords[coord_name] = copy.deepcopy(
                img_xds.coords[coord_name]
            )

    # Determine the output polarization labels via the single source of truth.
    if transformation_matrix is not None:
        n_out = np.asarray(transformation_matrix).shape[0]
        key = (new_polarization_basis, n_out)
        new_pol_labels: list = (
            _CUSTOM_MATRIX_OUTPUT_LABELS[key]
            if key in _CUSTOM_MATRIX_OUTPUT_LABELS
            else list(range(n_out))
        )
    else:
        _, _, new_pol_labels = _select_transform_matrix(
            frozenset(img_xds.polarization.values), new_polarization_basis
        )

    img_transformed_xds = img_transformed_xds.assign_coords(polarization=new_pol_labels)

    for var_name in img_xds.data_vars:
        img_transformed_xds[var_name] = (
            transform_polarization_basis_image_data_variable(
                img_xds[var_name],
                new_polarization_basis=new_polarization_basis,
                transformation_matrix=transformation_matrix,
            )
        )

    return img_transformed_xds


def transform_polarization_basis_image_data_variable(
    data_var: xr.DataArray,
    new_polarization_basis: Optional[str] = None,
    transformation_matrix: Optional[np.ndarray] = None,
) -> xr.DataArray:
    """Apply a polarization basis transformation to a single image data variable.

    The contraction is performed with :func:`xarray.dot` using ``optimize=True``,
    which delegates to *opt_einsum* when it is installed.  xarray's
    label-based alignment ensures that the polarization axis is matched by
    coordinate value, so the input data does not need to be in any particular
    polarization order.

    Parameters
    ----------
    data_var : xr.DataArray
        Image with dimensions ``(time, frequency, polarization, l, m)``.
        The ``polarization`` coordinate must contain recognized labels
        (e.g. ``["XX", "XY", "YX", "YY"]`` or ``["I", "Q", "U", "V"]``)
        unless *transformation_matrix* is supplied, in which case any labels
        are accepted.
    new_polarization_basis : str, optional
        Target basis: ``'stokes'``, ``'linear'``, or ``'circular'``.
        Drives the automatic matrix selection via :func:`_select_transform_matrix`.
        Ignored when *transformation_matrix* is provided, except as a hint
        for the output polarization labels (see *transformation_matrix*).
    transformation_matrix : np.ndarray of shape (n_out, n_in), optional
        Explicit transformation matrix.  When provided *new_polarization_basis*
        is only used to look up standard output labels from
        ``_CUSTOM_MATRIX_OUTPUT_LABELS``; if no match is found, integer
        indices ``[0, 1, …, n_out-1]`` are used as the output
        ``polarization`` coordinate.

    Returns
    -------
    xr.DataArray
        Transformed array with the same dimension order as *data_var* and an
        updated ``polarization`` coordinate.  All other coordinates and
        attributes are preserved.
    """
    pol_values = list(data_var.polarization.values)
    original_dims = list(data_var.dims)

    if transformation_matrix is not None:
        matrix = np.asarray(transformation_matrix, dtype=complex)
        n_out = matrix.shape[0]
        in_pol_labels = pol_values
        key = (
            (new_polarization_basis, n_out)
            if new_polarization_basis is not None
            else None
        )
        out_pol_labels = (
            _CUSTOM_MATRIX_OUTPUT_LABELS[key]
            if key in _CUSTOM_MATRIX_OUTPUT_LABELS
            else list(range(n_out))
        )
        logger.debug(
            f"transform_polarization_basis_image_data_variable: "
            f"custom matrix {matrix.shape}, output labels {out_pol_labels}"
        )
    else:
        if new_polarization_basis is None:
            raise ValueError(
                "new_polarization_basis must be provided when transformation_matrix is None."
            )
        matrix, in_pol_labels, out_pol_labels = _select_transform_matrix(
            frozenset(pol_values), new_polarization_basis
        )
        logger.debug(
            f"transform_polarization_basis_image_data_variable: "
            f"{pol_values} -> {out_pol_labels}"
        )

    # Rename the input polarization dim to avoid a name clash with the output
    # "polarization" dim that xr.dot will produce from the transform DataArray.
    data_renamed = data_var.rename({"polarization": "pol_in"})

    transform_da = xr.DataArray(
        matrix,
        dims=["polarization", "pol_in"],
        coords={"polarization": out_pol_labels, "pol_in": in_pol_labels},
    )

    # xr.dot contracts over "pol_in", aligning on coordinate labels.
    # optimize=True enables opt_einsum path optimization when available.
    result = xr.dot(transform_da, data_renamed, dim="pol_in", optimize=True)

    # Restore the original dimension order: (time, frequency, polarization, l, m)
    result = result.transpose(*original_dims)

    result.attrs = data_var.attrs.copy()
    return result
