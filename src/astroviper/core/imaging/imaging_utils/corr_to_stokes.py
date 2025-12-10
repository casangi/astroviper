"""
Convert correlation products to Stokes parameters, classes to go both ways.
"""

from typing import Optional, Union

import numpy as np
import xarray as xr


def corr_to_stokes(
    data: Union[np.ndarray, xr.DataArray],
    corr_type: str = "linear",
    transformation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert linear correlation products to Stokes parameters.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Numpy array or xarray DataArray with correlation products as the final dimension.
        Typically of shape (..., 4) for 'XX', 'XY', 'YX', 'YY' for e.g.
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
    >>> stokes_data = corr_to_stokes(data, corr_type='linear')
    """

    if corr_type == "custom" and transformation_matrix is None:
        raise ValueError(
            "For 'custom' corr_type, a transformation_matrix must be provided."
        )

    if corr_type == "linear" and transformation_matrix is None:
        transformation_matrix = [
            [1, 0, 0, 1],  # I = XX + YY
            [1, 0, 0, -1],  # Q = XX - YY
            [0, 1, 1, 0],  # U = XY + YX
            [0, -1j, 1j, 0],  # V = i(YX - XY)
        ]
    elif corr_type == "circular" and transformation_matrix is None:
        transformation_matrix = [
            [1, 0, 0, 1],  # I = RR + LL
            [0, 1, 1, 0],  # Q = RL + LR
            [0, -1j, 1j, 0],  # U = i(LR - RL)
            [1, 0, 0, -1],  # V = RR - LL
        ]

    # Copy data to avoid modifying original
    if isinstance(data, xr.DataArray):
        feed_data = data.values.copy()
    else:
        feed_data = data.copy()
    stokes_data = feed_data @ np.array(transformation_matrix).T

    return stokes_data


def stokes_to_corr(
    data: Union[np.ndarray, xr.DataArray],
    corr_type: str = "linear",
    transformation_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert Stokes parameters to correlation products.

    Parameters
    ----------
    data : np.ndarray or xr.DataArray
        Numpy array or xarray DataArray with Stokes parameters as the final dimension.
        Typically of shape (..., 4) for 'I', 'Q', 'U', 'V'.
    corr_type : str, optional
        Type of correlation products, e.g. 'linear', 'circular', or 'custom'.
    transformation_matrix : np.ndarray, optional
        Transformation matrix from Stokes parameters to correlation products.
        If None, a default matrix will be used for corr_type 'linear' or 'circular'.
        An error will be thrown if no matrix is provided for 'custom' corr_type.

    Returns
    -------
    np.ndarray
        Numpy array with correlation products as the final dimension.

    Raises
    ------
    ValueError
        If corr_type is 'custom' and no transformation_matrix is provided.

    Example
    -------
    >>> stokes_data = xr.DataArray(
        np.random.rand(10, 4),
        dims=['time', 'stokes'],
        coords={'stokes': ['I', 'Q', 'U', 'V']}
    )
    >>> corr_data = stokes_to_corr(stokes_data, corr_type='linear')
    """

    if corr_type == "custom" and transformation_matrix is None:
        raise ValueError(
            "For 'custom' corr_type, a transformation_matrix must be provided."
        )

    if corr_type == "linear" and transformation_matrix is None:
        transformation_matrix = [
            [0.5, 0.5, 0, 0],  # XX = (I + Q)/2
            [0, 0, 0.5, 0.5j],  # XY = (U + iV)/2
            [0, 0, 0.5, -0.5j],  # YX = (U - iV)/2
            [0.5, -0.5, 0, 0],  # YY = (I - Q)/2
        ]
    elif corr_type == "circular" and transformation_matrix is None:
        transformation_matrix = [
            [0.5, 0, 0, 0.5],  # RR = (I + V)/2
            [0, 0.5, 0.5j, 0],  # RL = (Q + iU)/2
            [0, 0.5, -0.5j, 0],  # LR = (Q - iU)/2
            [0.5, 0, 0, -0.5],  # LL = (I - V)/2
        ]

    # Copy data to avoid modifying original
    if isinstance(data, xr.DataArray):
        feed_data = data.values.copy()
    else:
        feed_data = data.copy()
    corr_data = feed_data @ np.array(transformation_matrix).T

    return corr_data
