"""
Convert correlation products to Stokes parameters.

Supports 4-pol (XX,XY,YX,YY or RR,RL,LR,LL → I,Q,U,V) and
2-pol (XX,YY → I,Q or RR,LL → I,V).
"""

from typing import Optional, Union, List

import numpy as np
import xarray as xr

# Valid Stokes parameters
VALID_STOKES = {"I", "Q", "U", "V"}

# Stokes available from 2-pol data
STOKES_FROM_2POL_LINEAR = {"I", "Q"}  # XX, YY → I, Q
STOKES_FROM_2POL_CIRCULAR = {"I", "V"}  # RR, LL → I, V

# Map polarization coordinate values to corr_type
CORR_TYPE_MAP = {
    frozenset({"XX", "XY", "YX", "YY"}): "linear",
    frozenset({"RR", "RL", "LR", "LL"}): "circular",
    frozenset({"XX", "YY"}): "linear",
    frozenset({"RR", "LL"}): "circular",
}

# Transformation matrices: correlation → Stokes
LINEAR_CORR_TO_STOKES = np.array(
    [
        [1, 0, 0, 1],  # I = XX + YY
        [1, 0, 0, -1],  # Q = XX - YY
        [0, 1, 1, 0],  # U = XY + YX
        [0, -1j, 1j, 0],  # V = i(YX - XY)
    ],
    dtype=complex,
)

CIRCULAR_CORR_TO_STOKES = np.array(
    [
        [1, 0, 0, 1],  # I = RR + LL
        [0, 1, 1, 0],  # Q = RL + LR
        [0, -1j, 1j, 0],  # U = i(LR - RL)
        [1, 0, 0, -1],  # V = RR - LL
    ],
    dtype=complex,
)

# Transformation matrices: Stokes → correlation
LINEAR_STOKES_TO_CORR = np.array(
    [
        [0.5, 0.5, 0, 0],  # XX = (I + Q)/2
        [0, 0, 0.5, 0.5j],  # XY = (U + iV)/2
        [0, 0, 0.5, -0.5j],  # YX = (U - iV)/2
        [0.5, -0.5, 0, 0],  # YY = (I - Q)/2
    ],
    dtype=complex,
)

CIRCULAR_STOKES_TO_CORR = np.array(
    [
        [0.5, 0, 0, 0.5],  # RR = (I + V)/2
        [0, 0.5, 0.5j, 0],  # RL = (Q + iU)/2
        [0, 0.5, -0.5j, 0],  # LR = (Q - iU)/2
        [0.5, 0, 0, -0.5],  # LL = (I - V)/2
    ],
    dtype=complex,
)


def corr_to_stokes(
    data: np.ndarray,
    corr_type: str = "linear",
    transformation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert correlation products to Stokes parameters.

    Parameters
    ----------
    data : np.ndarray
        Array with correlation products as the final dimension, shape (..., 4).
    corr_type : str
        'linear', 'circular', or 'custom'.
    transformation_matrix : np.ndarray, optional
        Custom transformation matrix. Required if corr_type='custom'.

    Returns
    -------
    np.ndarray
        Array with Stokes parameters as the final dimension.
    """
    if transformation_matrix is None:
        if corr_type == "linear":
            transformation_matrix = LINEAR_CORR_TO_STOKES
        elif corr_type == "circular":
            transformation_matrix = CIRCULAR_CORR_TO_STOKES
        else:
            raise ValueError(
                "For 'custom' corr_type, a transformation_matrix must be provided."
            )

    return data @ transformation_matrix.T


def stokes_to_corr(
    data: np.ndarray,
    corr_type: str = "linear",
    transformation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert Stokes parameters to correlation products.

    Parameters
    ----------
    data : np.ndarray
        Array with Stokes parameters as the final dimension, shape (..., 4).
    corr_type : str
        'linear', 'circular', or 'custom'.
    transformation_matrix : np.ndarray, optional
        Custom transformation matrix. Required if corr_type='custom'.

    Returns
    -------
    np.ndarray
        Array with correlation products as the final dimension.
    """
    if transformation_matrix is None:
        if corr_type == "linear":
            transformation_matrix = LINEAR_STOKES_TO_CORR
        elif corr_type == "circular":
            transformation_matrix = CIRCULAR_STOKES_TO_CORR
        else:
            raise ValueError(
                "For 'custom' corr_type, a transformation_matrix must be provided."
            )

    return data @ transformation_matrix.T


def image_corr_to_stokes(
    image_data: Union[np.ndarray, xr.DataArray],
    corr_type: Optional[str] = None,
    pol_axis: int = 2,
    stokes_out: Optional[List[str]] = None,
    transformation_matrix: Optional[np.ndarray] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert image data from correlation products to Stokes parameters.

    Handles image cubes where polarization is not the last dimension. Moves pol axis
    to end, applies conversion, then moves it back.

    Parameters
    ----------
    image_data : np.ndarray or xr.DataArray
        Image with polarization dimension. Shape typically (time, freq, pol, l, m).
        Correlations: [XX, XY, YX, YY] or [RR, RL, LR, LL] for 4-pol,
        [XX, YY] or [RR, LL] for 2-pol.
    corr_type : str, optional
        'linear', 'circular', or 'custom'. Auto-detected for xarray input.
    pol_axis : int
        Polarization axis index. Default 2.
    stokes_out : list of str, optional
        Output Stokes, e.g., ['I'], ['I', 'Q']. Default: all available.
    transformation_matrix : np.ndarray, optional
        Custom matrix. Required if corr_type='custom'.

    Returns
    -------
    np.ndarray or xr.DataArray
        Converted data with Stokes in polarization dimension.

    Examples
    --------
    >>> stokes = image_corr_to_stokes(corr_image, corr_type='linear')
    >>> stokes_i = image_corr_to_stokes(corr_image, stokes_out=['I'])
    >>> stokes = image_corr_to_stokes(xr_data)  # auto-detects corr_type
    """
    # Check if input is xarray
    is_xarray = isinstance(image_data, xr.DataArray)

    # Extract numpy array if xarray
    if is_xarray:
        data = image_data.values
        pol_dim_name = image_data.dims[pol_axis]
    else:
        data = image_data

    # Auto-detect corr_type if input is xarray and corr_type not specified
    if corr_type is None:
        if is_xarray:
            corr_type = _detect_corr_type(image_data, pol_axis)
        else:
            raise ValueError(
                "corr_type is required for numpy input. "
                "Use xarray DataArray with labeled polarization coordinates for auto-detection."
            )

    # Normalize pol_axis to positive index
    ndim = data.ndim
    if pol_axis < 0:
        pol_axis_pos = ndim + pol_axis
    else:
        pol_axis_pos = pol_axis

    # Get number of input correlations
    n_corr = data.shape[pol_axis_pos]

    # Validate minimum correlations
    if n_corr < 2:
        raise ValueError(
            f"Minimum 2 correlations required for Stokes conversion, got {n_corr}"
        )

    # Determine default stokes_out based on input
    if stokes_out is None:
        if n_corr == 2:
            if corr_type == "linear":
                stokes_out = ["I", "Q"]
            elif corr_type == "circular":
                stokes_out = ["I", "V"]
            else:
                raise ValueError(
                    f"2-pol conversion requires corr_type 'linear' or 'circular', got '{corr_type}'"
                )
        else:
            stokes_out = ["I", "Q", "U", "V"]

    # Validate stokes_out
    stokes_set = set(stokes_out)
    invalid_stokes = stokes_set - VALID_STOKES
    if invalid_stokes:
        raise ValueError(
            f"Invalid Stokes parameters: {invalid_stokes}. Valid: {VALID_STOKES}"
        )

    # Validate stokes_out against available correlations for 2-pol
    if n_corr == 2:
        if corr_type == "linear":
            unavailable = stokes_set - STOKES_FROM_2POL_LINEAR
            if unavailable:
                raise ValueError(
                    f"2-pol linear data (XX, YY) cannot produce Stokes {unavailable}. "
                    f"Only {STOKES_FROM_2POL_LINEAR} available."
                )
        elif corr_type == "circular":
            unavailable = stokes_set - STOKES_FROM_2POL_CIRCULAR
            if unavailable:
                raise ValueError(
                    f"2-pol circular data (RR, LL) cannot produce Stokes {unavailable}. "
                    f"Only {STOKES_FROM_2POL_CIRCULAR} available."
                )

    # Move polarization axis to end (creates view, no copy)
    data_moved = np.moveaxis(data, pol_axis_pos, -1)

    # Handle 2-pol case: expand to 4-corr, use matrix, then slice
    if n_corr == 2:
        data_4corr = _expand_to_4corr(data_moved, corr_type)
        full_stokes = corr_to_stokes(
            data_4corr, corr_type=corr_type, transformation_matrix=transformation_matrix
        )
    else:
        # 4-pol case: use existing matrix multiplication directly
        full_stokes = corr_to_stokes(
            data_moved, corr_type=corr_type, transformation_matrix=transformation_matrix
        )

    # If requesting all 4 Stokes, return in standard order regardless of input order
    if set(stokes_out) == {"I", "Q", "U", "V"}:
        converted = full_stokes
        stokes_out = ["I", "Q", "U", "V"]  # Normalize to standard order
    else:
        # Slice to requested Stokes
        stokes_indices = {"I": 0, "Q": 1, "U": 2, "V": 3}
        indices = [stokes_indices[s] for s in stokes_out]
        converted = np.take(full_stokes, indices, axis=-1)

    # Move polarization axis back to original position
    result = np.moveaxis(converted, -1, pol_axis_pos)

    # If input was xarray, create new DataArray with correct shape
    if is_xarray:
        # Build new coordinates
        new_coords = dict(image_data.coords)
        new_coords[pol_dim_name] = stokes_out

        # Create new DataArray with potentially different shape
        result_da = xr.DataArray(
            result,
            dims=image_data.dims,
            coords=new_coords,
            attrs=image_data.attrs,
        )
        return result_da
    else:
        return result


def image_stokes_to_corr(
    image_data: Union[np.ndarray, xr.DataArray],
    corr_type: str = "linear",
    pol_axis: int = 2,
    stokes_in: Optional[List[str]] = None,
    corr_out: Optional[int] = None,
    transformation_matrix: Optional[np.ndarray] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert image data from Stokes parameters to correlation products.

    Handles image cubes where polarization is not the last dimension. Moves pol axis
    to end, applies conversion, then moves it back.

    Parameters
    ----------
    image_data : np.ndarray or xr.DataArray
        Image with polarization dimension containing Stokes parameters.
    corr_type : str
        'linear', 'circular', or 'custom'. Default 'linear'.
    pol_axis : int
        Polarization axis index. Default 2.
    stokes_in : list of str, optional
        Input Stokes labels. Inferred from size if None:
        4 → [I,Q,U,V], 2 → [I,Q] or [I,V], 1 → [I].
    corr_out : int, optional
        Output correlations (2 or 4). Default: 4 for 4-Stokes, 2 otherwise.
    transformation_matrix : np.ndarray, optional
        Custom matrix. Required if corr_type='custom'.

    Returns
    -------
    np.ndarray or xr.DataArray
        Converted data with correlations in polarization dimension.

    Examples
    --------
    >>> corr = image_stokes_to_corr(stokes_image, corr_type='linear')
    >>> corr = image_stokes_to_corr(stokes_iq, stokes_in=['I', 'Q'])
    >>> corr = image_stokes_to_corr(stokes_i, stokes_in=['I'])  # XX=YY=I/2
    """
    # Check if input is xarray
    is_xarray = isinstance(image_data, xr.DataArray)

    # Extract numpy array if xarray
    if is_xarray:
        data = image_data.values
        pol_dim_name = image_data.dims[pol_axis]
    else:
        data = image_data

    # Normalize pol_axis to positive index
    ndim = data.ndim
    if pol_axis < 0:
        pol_axis_pos = ndim + pol_axis
    else:
        pol_axis_pos = pol_axis

    # Get number of input Stokes
    n_stokes = data.shape[pol_axis_pos]

    # Determine stokes_in if not provided
    if stokes_in is None:
        if n_stokes == 4:
            stokes_in = ["I", "Q", "U", "V"]
        elif n_stokes == 2:
            if corr_type == "linear":
                stokes_in = ["I", "Q"]
            elif corr_type == "circular":
                stokes_in = ["I", "V"]
            else:
                raise ValueError(
                    f"2-Stokes conversion requires corr_type 'linear' or 'circular', got '{corr_type}'"
                )
        elif n_stokes == 1:
            stokes_in = ["I"]
        else:
            raise ValueError(f"Unexpected number of Stokes parameters: {n_stokes}")
    else:
        # Validate provided stokes_in
        if len(stokes_in) != n_stokes:
            raise ValueError(
                f"stokes_in length ({len(stokes_in)}) doesn't match input polarization "
                f"dimension size ({n_stokes})"
            )

    # Validate stokes_in values
    stokes_set = set(stokes_in)
    invalid_stokes = stokes_set - VALID_STOKES
    if invalid_stokes:
        raise ValueError(
            f"Invalid Stokes parameters: {invalid_stokes}. Valid: {VALID_STOKES}"
        )

    # Determine corr_out
    if corr_out is None:
        if n_stokes == 4:
            corr_out = 4
        else:
            corr_out = 2

    # Validate corr_out
    if corr_out not in [2, 4]:
        raise ValueError(f"corr_out must be 2 or 4, got {corr_out}")

    # Validate that we can produce the requested correlations
    if corr_out == 4 and n_stokes < 4:
        raise ValueError(
            f"Cannot produce 4-pol output from {n_stokes}-Stokes input. "
            "Need U, V for cross-polarizations (XY, YX or RL, LR)."
        )

    # Move polarization axis to end (creates view, no copy)
    data_moved = np.moveaxis(data, pol_axis_pos, -1)

    # Handle 2-pol output with direct formulas
    if corr_out == 2:
        if corr_type not in ("linear", "circular"):
            raise ValueError(
                f"2-pol output requires corr_type 'linear' or 'circular', got '{corr_type}'"
            )

        stokes_idx = {s: i for i, s in enumerate(stokes_in)}
        if "I" not in stokes_idx:
            raise ValueError(
                f"Cannot convert to {corr_type} correlations without Stokes I"
            )

        # Linear: XX=(I+Q)/2, YY=(I-Q)/2; Circular: RR=(I+V)/2, LL=(I-V)/2
        paired_stokes = "Q" if corr_type == "linear" else "V"
        result_shape = list(data_moved.shape[:-1]) + [2]
        converted = np.zeros(result_shape, dtype=data_moved.dtype)

        I = data_moved[..., stokes_idx["I"]]
        if paired_stokes in stokes_idx:
            P = data_moved[..., stokes_idx[paired_stokes]]
            converted[..., 0] = (I + P) / 2
            converted[..., 1] = (I - P) / 2
        else:
            # Only Stokes I: unpolarized
            converted[..., 0] = I / 2
            converted[..., 1] = I / 2

    else:
        # Full 4-Stokes to 4-pol: use existing matrix multiplication
        converted = stokes_to_corr(
            data_moved, corr_type=corr_type, transformation_matrix=transformation_matrix
        )

    # Move polarization axis back to original position
    result = np.moveaxis(converted, -1, pol_axis_pos)

    # Determine output correlation labels
    if corr_type == "linear":
        if corr_out == 4:
            corr_labels = ["XX", "XY", "YX", "YY"]
        else:
            corr_labels = ["XX", "YY"]
    elif corr_type == "circular":
        if corr_out == 4:
            corr_labels = ["RR", "RL", "LR", "LL"]
        else:
            corr_labels = ["RR", "LL"]
    else:
        # Custom: keep generic labels
        corr_labels = [f"CORR{i}" for i in range(corr_out)]

    # If input was xarray, create new DataArray with correct shape
    if is_xarray:
        new_coords = dict(image_data.coords)
        new_coords[pol_dim_name] = corr_labels

        result_da = xr.DataArray(
            result,
            dims=image_data.dims,
            coords=new_coords,
            attrs=image_data.attrs,
        )
        return result_da
    else:
        return result


def _expand_to_4stokes(data: np.ndarray, stokes_in: List[str]) -> np.ndarray:
    """Expand partial Stokes to full [I,Q,U,V] array. Missing values filled with zeros."""
    result_shape = list(data.shape[:-1]) + [4]
    result = np.zeros(result_shape, dtype=data.dtype)

    stokes_order = ["I", "Q", "U", "V"]
    input_idx = {s: i for i, s in enumerate(stokes_in)}

    for out_idx, stokes in enumerate(stokes_order):
        if stokes in input_idx:
            result[..., out_idx] = data[..., input_idx[stokes]]

    return result


def _expand_to_4corr(data: np.ndarray, corr_type: str) -> np.ndarray:
    """Expand 2-pol to 4-pol: [XX,YY]→[XX,0,0,YY] or [RR,LL]→[RR,0,0,LL]."""
    result_shape = list(data.shape[:-1]) + [4]
    result = np.zeros(result_shape, dtype=data.dtype)

    # Indices: XX/RR=0, XY/RL=1, YX/LR=2, YY/LL=3
    result[..., 0] = data[..., 0]  # XX or RR
    result[..., 3] = data[..., 1]  # YY or LL
    # Cross-pols (indices 1, 2) remain zero

    return result


def _detect_corr_type(image_data: xr.DataArray, pol_axis: int) -> str:
    """Detect 'linear' or 'circular' from xarray polarization coordinate labels."""

    pol_dim_name = image_data.dims[pol_axis]
    pol_values = frozenset(str(v) for v in image_data.coords[pol_dim_name].values)

    corr_type = CORR_TYPE_MAP.get(pol_values)
    if corr_type is None:
        raise ValueError(
            f"Cannot detect corr_type from polarization values: {set(pol_values)}. "
            f"Expected one of: {[set(k) for k in CORR_TYPE_MAP.keys()]}"
        )
    return corr_type
