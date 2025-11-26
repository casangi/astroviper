"""
Convert correlation products to Stokes parameters, classes to go both ways.
"""

import numpy as np
import xarray as xr


def corr_to_stokes(
    data: np.ndarray,
    corr_type="linear",
    transformation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert linear correlation products to Stokes parameters.

    Parameters
    ----------
    data : np.ndarray
        Numpy array with correlation products as the final dimension.
        Typically of shape (..., 4) for 'XX', ''XY', 'YX', 'YY' for e.g.
    corr_type : str, optional
        Type of correlation products, e.g. 'linear', 'circular', or 'custom'.
    transformation_matrix : np.ndarray, optional
        Transformation matrix from correlation products to Stokes parameters.
        If None, a default matrix will be used for corr_type 'linear' or 'circular'.
        An error will be thrown if no matrix is provided for 'custom' corr_type.

    Returns
    -------
    np.ndarray
        Numpy array with Stokes parameters as the final dimension.

    Raises
    ------
    ValueError
        If corr_type is 'custom' and no transformation_matrix is provided.

    Example
    -------
    >>> data = xr.DataArray(
        np.random.rand(10, 4),
        dims=['time', 'corr_product'],
        coords={'corr_product': ['XX', 'YY', 'XY', 'YX']}
    )
    >>> stokes_data = correlation_to_stokes(data, corr_type='linear')
    """

    if corr_type == "custom" and transformation_matrix is None:
        raise ValueError(
            "For 'custom' corr_type, a transformation_matrix must be provided."
        )

    if corr_type == "linear" and transformation_matrix is None:
        transformation_matrix = [
            [0.5, 0, 0, 0.5],  # I
            [0.5, 0, 0, -0.5],  # Q
            [0, 0.5, 0.5, 0],  # U
            [0, -0.5j, 0.5j, 0],  # V
        ]
    elif corr_type == "circular" and transformation_matrix is None:
        transformation_matrix = [
            [0.5, 0, 0, 0.5],  # I
            [0, 0.5, 0.5, 0],  # Q
            [0, -0.5j, 0.5j, 0],  # U
            [0.5, 0, 0, -0.5],  # V
        ]

    # Copy data to avoid modifying original
    feed_data = data.values.copy()
    stokes_data = feed_data @ np.array(transformation_matrix)

    return stokes_data


def stokes_to_corr(
    data: np.ndarray,
    corr_type="linear",
    transformation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert linear correlation products to Stokes parameters.

    Parameters
    ----------
    data : np.ndarray
        Numpy array with Stokes products as the final dimension.
        Typically of shape (..., 4) for 'XX', ''XY', 'YX', 'YY' for e.g.
    corr_type : str, optional
        Type of correlation products, e.g. 'linear', 'circular', or 'custom'.
    transformation_matrix : np.ndarray, optional
        Transformation matrix from correlation products to Stokes parameters.
        If None, a default matrix will be used for corr_type 'linear' or 'circular'.
        An error will be thrown if no matrix is provided for 'custom' corr_type.

    Returns
    -------
    np.ndarray
        Numpy array with Stokes parameters as the final dimension.

    Raises
    ------
    ValueError
        If corr_type is 'custom' and no transformation_matrix is provided.

    Example
    -------
    >>> data = xr.DataArray(
        np.random.rand(10, 4),
        dims=['time', 'corr_product'],
        coords={'corr_product': ['XX', 'YY', 'XY', 'YX']}
    )
    >>> stokes_data = correlation_to_stokes(data, corr_type='linear')
    """

    if corr_type == "custom" and transformation_matrix is None:
        raise ValueError(
            "For 'custom' corr_type, a transformation_matrix must be provided."
        )

    if corr_type == "linear" and transformation_matrix is None:
        transformation_matrix = [
            [0.5, 0.5, 0, 0],  # XX
            [0.5, -0.5, 0, 0],  # YY
            [0, 0, 0.5, -0.5j],  # XY
            [0, 0, 0.5, 0.5j],  # YX
        ]
    elif corr_type == "circular" and transformation_matrix is None:
        transformation_matrix = [
            [0.5, 0, 0, 0.5],  # RR
            [0.5, 0, 0, -0.5],  # LL
            [0, 0.5, -0.5j, 0],  # RL
            [0, 0.5, 0.5j, 0],  # LR
        ]

    # Copy data to avoid modifying original
    feed_data = data.values.copy()
    stokes_data = feed_data @ np.array(transformation_matrix)

    return stokes_data
