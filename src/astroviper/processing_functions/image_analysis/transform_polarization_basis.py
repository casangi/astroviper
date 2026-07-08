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
        [1.0, 1.0, 0, 0],  # XX = (I + Q)
        [0, 0, 1.0, 1.0j],  # XY = (U + iV)
        [0, 0, 1.0, -1.0j],  # YX = (U - iV)
        [1.0, -1.0, 0, 0],  # YY = (I - Q)
    ],
    dtype=complex,
)

# Stokes → correlation  (columns: I, Q, U, V)
CIRCULAR_STOKES_TO_CORR = np.array(
    [
        [1.0, 0, 0, 1.0],  # RR = (I + V)
        [0, 1.0, 1.0j, 0],  # RL = (Q + iU)
        [0, 1.0, -1.0j, 0],  # LR = (Q - iU)
        [1.0, 0, 0, -1.0],  # LL = (I - V)
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
    dtype=np.float64,
)

# correlation → Stokes  (columns: RR, LL)
CIRCULAR_2POL_CORR_TO_STOKES = np.array(
    [
        [0.5, 0.5],  # I = (RR + LL) / 2
        [0.5, -0.5],  # V = (RR - LL) / 2
    ],
    dtype=np.float64,
)

# Stokes → correlation  (columns: I, Q)
LINEAR_2POL_STOKES_TO_CORR = np.array(
    [
        [1.0, 1.0],  # XX = (I + Q)
        [1.0, -1.0],  # YY = (I - Q)
    ],
    dtype=np.float64,
)

# Stokes → correlation  (columns: I, V)
CIRCULAR_2POL_STOKES_TO_CORR = np.array(
    [
        [1.0, 1.0],  # RR = (I + V)
        [1.0, -1.0],  # LL = (I - V)
    ],
    dtype=np.float64,
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
    transformation_matrix: np.ndarray | None = None,
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
        If ``False`` a deep copy is returned with its data variables
        transformed and the input left untouched.

    Returns
    -------
    xr.Dataset
        Dataset with transformed data variables and an updated
        ``polarization`` coordinate.
    """
    if overwrite:
        img_transformed_xds = img_xds
    else:
        # Independent deep copy so the loop below can transform its data
        # variables in place without mutating the caller's dataset. A deep copy
        # (rather than an empty Dataset) guarantees every data variable is
        # present in the output -- including the pass-through variables (PSF,
        # airy-disk primary beam, uv-domain grids) that the loop skips.
        img_transformed_xds = img_xds.copy(deep=True)

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

    matrix, in_pol_labels, out_pol_labels = get_transformation_matrix(
        list(img_transformed_xds.polarization.values),
        new_polarization_basis,
        transformation_matrix,
    )

    from toolviper.utils.memory_management import free_memory, get_rss_gb, memory_setup

    for var_name in img_xds.data_vars:
        # if "type" in img_xds[var_name].attrs:
        #     print("###### The type of the variable is ", var_name, img_xds[var_name].attrs["type"])
        if (
            "type" in img_xds[var_name].attrs
            and img_xds[var_name].attrs["type"] == "point_spread_function"
        ):
            # Skip PSF variables
            continue

        if (
            "type" in img_xds[var_name].attrs
            and img_xds[var_name].attrs["type"] == "primary_beam"
        ):
            if img_xds["PRIMARY_BEAM"].attrs["method"] == "airy_disk":
                continue

        # Skip uv-domain grids (e.g. VISIBILITY_MODEL). The residual update grids
        # and degrids in the telescope's NATIVE polarization basis while the model
        # update works in Stokes, so a model-visibility uv grid is never the thing
        # being basis-transformed. It also persists across major cycles (it is
        # rebuilt by fft_norm each cycle), so it must not be deleted -- only
        # skipped here. Transforming it would be wasteful (large complex grid,
        # padded 1.2x) and pointless.
        if "u" in img_xds[var_name].dims or "v" in img_xds[var_name].dims:
            continue

        if ("polarization" in img_xds[var_name].dims) and ("BEAM_FIT" not in var_name):
            # print("###### The type of the variable is ", var_name, img_xds[var_name].attrs.keys())
            # if not img_xds[var_name].attrs["type"] == "point_spread_function":
            original_dims = list(img_transformed_xds[var_name].dims)

            # Cast the mixing matrix to the image's precision so xr.dot keeps the
            # result in the image dtype instead of upcasting single-precision
            # images/grids to float64 / complex128 (the matrices are declared
            # float64 / complex128). Only the precision is matched; the matrix
            # stays real or complex as appropriate.
            img_dtype = img_transformed_xds[var_name].dtype
            single = (img_dtype == np.float32) or (img_dtype == np.complex64)
            if np.iscomplexobj(matrix):
                matrix_dtype = np.complex64 if single else np.complex128
            else:
                matrix_dtype = np.float32 if single else np.float64
            transform_da = xr.DataArray(
                matrix.astype(matrix_dtype),
                dims=["polarization", "pol_in"],
                coords={"polarization": out_pol_labels, "pol_in": in_pol_labels},
            )

            # Use [:] slice assignment so the transposed result is copied
            # into the DataArray's existing C-contiguous buffer. Writing
            # `.values = rhs` instead would *replace* the buffer with the
            # transposed view, which is strided and thus neither C- nor
            # F-contiguous.
            img_transformed_xds[var_name].values[:] = (
                xr.dot(
                    transform_da,
                    img_transformed_xds[var_name].rename({"polarization": "pol_in"}),
                    dim="pol_in",
                    optimize=True,
                )
                .transpose(*original_dims)
                .values
            )

    # Update the polarization labels in place on the (possibly copied) dataset
    # and return it. ``assign_coords`` would return a *new* object and leave the
    # caller's coordinate stale; doing it in place keeps overwrite=True a true
    # in-place operation -- the returned dataset is the input, now fully
    # consistent (transformed data *and* the new labels).
    img_transformed_xds.coords["polarization"] = new_pol_labels

    return img_transformed_xds


def get_transformation_matrix(
    in_pol_labels,
    new_polarization_basis: str | None = None,
    transformation_matrix: np.ndarray | None = None,
):

    if transformation_matrix is not None:
        matrix = np.asarray(transformation_matrix, dtype=complex)
        n_out = matrix.shape[0]
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
            frozenset(in_pol_labels), new_polarization_basis
        )
        logger.debug(
            f"transform_polarization_basis_image_data_variable: "
            f"{in_pol_labels} -> {out_pol_labels}"
        )

    return matrix, in_pol_labels, out_pol_labels


# def transform_polarization_basis_image_data_variable(
#     data_var: xr.DataArray,
#     new_polarization_basis: Optional[str] = None,
#     transformation_matrix: Optional[np.ndarray] = None,
# ) -> xr.DataArray:
#     """Apply a polarization basis transformation to a single image data variable.

#     The contraction is performed with :func:`xarray.dot` using ``optimize=True``,
#     which delegates to *opt_einsum* when it is installed.  xarray's
#     label-based alignment ensures that the polarization axis is matched by
#     coordinate value, so the input data does not need to be in any particular
#     polarization order.

#     Parameters
#     ----------
#     data_var : xr.DataArray
#         Image with dimensions ``(time, frequency, polarization, l, m)``.
#         The ``polarization`` coordinate must contain recognized labels
#         (e.g. ``["XX", "XY", "YX", "YY"]`` or ``["I", "Q", "U", "V"]``)
#         unless *transformation_matrix* is supplied, in which case any labels
#         are accepted.
#     new_polarization_basis : str, optional
#         Target basis: ``'stokes'``, ``'linear'``, or ``'circular'``.
#         Drives the automatic matrix selection via :func:`_select_transform_matrix`.
#         Ignored when *transformation_matrix* is provided, except as a hint
#         for the output polarization labels (see *transformation_matrix*).
#     transformation_matrix : np.ndarray of shape (n_out, n_in), optional
#         Explicit transformation matrix.  When provided *new_polarization_basis*
#         is only used to look up standard output labels from
#         ``_CUSTOM_MATRIX_OUTPUT_LABELS``; if no match is found, integer
#         indices ``[0, 1, …, n_out-1]`` are used as the output
#         ``polarization`` coordinate.

#     Returns
#     -------
#     xr.DataArray
#         Transformed array with the same dimension order as *data_var* and an
#         updated ``polarization`` coordinate.  All other coordinates and
#         attributes are preserved.
#     """

#     matrix, in_pol_labels, out_pol_labels = get_transformation_matrix(list(data_var.polarization.values), new_polarization_basis, transformation_matrix)


#     # Rename the input polarization dim to avoid a name clash with the output
#     # "polarization" dim that xr.dot will produce from the transform DataArray.
#     data_renamed = data_var.rename({"polarization": "pol_in"})

#     original_dims = list(data_var.dims)

#     transform_da = xr.DataArray(
#         matrix,
#         dims=["polarization", "pol_in"],
#         coords={"polarization": out_pol_labels, "pol_in": in_pol_labels},
#     )

#     # xr.dot contracts over "pol_in", aligning on coordinate labels.
#     # optimize=True enables opt_einsum path optimization when available.
#     result = xr.dot(transform_da, data_renamed, dim="pol_in", optimize=True)

#     # Restore the original dimension order: (time, frequency, polarization, l, m)
#     result = result.transpose(*original_dims)

#     result.attrs = data_var.attrs.copy()
#     return result
