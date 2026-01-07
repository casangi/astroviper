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


def image_corr_to_stokes(
    image_data: Union[np.ndarray, xr.DataArray],
    corr_type: str = "linear",
    pol_axis: int = 2,
    transformation_matrix: Optional[np.ndarray] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert image data from correlation products to Stokes parameters.

    This is a high-level function for converting polarization basis in image cubes.
    It handles the typical image_xds structure where polarization is not the last
    dimension. The function moves the polarization axis to the last position,
    applies the conversion, then moves it back.

    Parameters
    ----------
    image_data : np.ndarray or xr.DataArray
        Image data with polarization dimension. For standard image_xds, expected
        shape is (time, frequency, polarization, l, m). The polarization dimension
        should contain correlation products in order [XX, XY, YX, YY] for linear
        or [RR, RL, LR, LL] for circular.
    corr_type : str, optional
        Type of correlation products: 'linear', 'circular', or 'custom'.
        Default is 'linear'.
    pol_axis : int, optional
        Which axis contains the polarization dimension. Default is 2 (for standard
        image_xds with shape [time, frequency, polarization, l, m]). Can be
        negative to count from the end.
    transformation_matrix : np.ndarray, optional
        Custom transformation matrix. Required if corr_type='custom'.

    Returns
    -------
    np.ndarray or xr.DataArray
        Converted image data with same shape and type as input, but with Stokes
        parameters [I, Q, U, V] instead of correlation products in the polarization
        dimension. If input is xarray, returns xarray with updated polarization
        coordinates.

    Notes
    -----
    For large images, this function uses np.moveaxis which creates views (no copy)
    to efficiently rearrange dimensions. The actual conversion is done via matrix
    multiplication in the corr_to_stokes function.

    This function is optimized for image cubes where you need to convert the
    polarization basis while preserving the spatial and spectral structure.
    For visibility data, use the low-level corr_to_stokes function directly.

    If input is an xarray DataArray, the output will preserve all coordinates
    and attributes, with the polarization coordinate updated to ['I', 'Q', 'U', 'V'].

    Examples
    --------
    Convert correlation image to Stokes (linear polarization):

    >>> # Input shape: (1, 64, 4, 512, 512) = (time, freq, pol, l, m)
    >>> # Polarization: [XX, XY, YX, YY]
    >>> stokes_image = image_corr_to_stokes(corr_image, corr_type='linear')
    >>> # Output shape: (1, 64, 4, 512, 512) with polarization [I, Q, U, V]

    Convert circular correlation image to Stokes:

    >>> # Input polarization: [RR, RL, LR, LL]
    >>> stokes_image = image_corr_to_stokes(corr_image, corr_type='circular')
    >>> # Output polarization: [I, Q, U, V]

    Handle image with polarization in different position:

    >>> # Shape: (4, 64, 512, 512) = (pol, freq, l, m)
    >>> stokes_image = image_corr_to_stokes(corr_image, pol_axis=0)

    With xarray DataArray (preserves structure):

    >>> # Input: xarray DataArray with dims ['time', 'frequency', 'polarization', 'l', 'm']
    >>> sky_image = image_xds["SKY"]  # DataArray with correlation products
    >>> stokes_sky = image_corr_to_stokes(sky_image, corr_type='linear')
    >>> # Output: xarray DataArray with same dims, polarization coord = ['I', 'Q', 'U', 'V']
    """
    # Check if input is xarray
    is_xarray = isinstance(image_data, xr.DataArray)

    # Extract numpy array if xarray
    if is_xarray:
        data = image_data.values
        pol_dim_name = image_data.dims[pol_axis]
    else:
        data = image_data

    # Move polarization axis to end (creates view, no copy)
    data_moved = np.moveaxis(data, pol_axis, -1)

    # Apply conversion using low-level function (operates on last axis)
    converted = corr_to_stokes(
        data_moved, corr_type=corr_type, transformation_matrix=transformation_matrix
    )

    # Move polarization axis back to original position
    result = np.moveaxis(converted, -1, pol_axis)

    # If input was xarray, create copy and update in-place
    if is_xarray:
        result_da = image_data.copy()
        result_da.values[:] = result
        result_da.coords[pol_dim_name] = ["I", "Q", "U", "V"]
        return result_da
    else:
        return result


def image_stokes_to_corr(
    image_data: Union[np.ndarray, xr.DataArray],
    corr_type: str = "linear",
    pol_axis: int = 2,
    transformation_matrix: Optional[np.ndarray] = None,
) -> Union[np.ndarray, xr.DataArray]:
    """
    Convert image data from Stokes parameters to correlation products.

    This is a high-level function for converting polarization basis in image cubes.
    It handles the typical image_xds structure where polarization is not the last
    dimension. The function moves the polarization axis to the last position,
    applies the conversion, then moves it back.

    Parameters
    ----------
    image_data : np.ndarray or xr.DataArray
        Image data with polarization dimension. For standard image_xds, expected
        shape is (time, frequency, polarization, l, m). The polarization dimension
        should contain Stokes parameters in order [I, Q, U, V].
    corr_type : str, optional
        Type of correlation products to output: 'linear', 'circular', or 'custom'.
        Default is 'linear'.
    pol_axis : int, optional
        Which axis contains the polarization dimension. Default is 2 (for standard
        image_xds with shape (time, frequency, polarization, l, m)). Can be
        negative to count from the end.
    transformation_matrix : np.ndarray, optional
        Custom transformation matrix. Required if corr_type='custom'.

    Returns
    -------
    np.ndarray or xr.DataArray
        Converted image data with same shape and type as input, but with correlation
        products [XX, XY, YX, YY] for linear or [RR, RL, LR, LL] for circular
        instead of Stokes parameters in the polarization dimension. If input is
        xarray, returns xarray with updated polarization coordinates.

    Notes
    -----
    For large images, this function uses np.moveaxis which creates views (no copy)
    to efficiently rearrange dimensions. The actual conversion is done via matrix
    multiplication in the stokes_to_corr function.

    This function is optimized for image cubes where you need to convert the
    polarization basis while preserving the spatial and spectral structure.
    For visibility data, use the low-level stokes_to_corr function directly.

    If input is an xarray DataArray, the output will preserve all coordinates
    and attributes, with the polarization coordinate updated to correlation
    product labels.

    Examples
    --------
    Convert Stokes image to linear correlation products:

    >>> # Input shape: (1, 64, 4, 512, 512) = (time, freq, pol, l, m)
    >>> # Polarization: [I, Q, U, V]
    >>> corr_image = image_stokes_to_corr(stokes_image, corr_type='linear')
    >>> # Output shape: (1, 64, 4, 512, 512) with polarization [XX, XY, YX, YY]

    Convert Stokes image to circular correlation products:

    >>> # Input polarization: [I, Q, U, V]
    >>> corr_image = image_stokes_to_corr(stokes_image, corr_type='circular')
    >>> # Output polarization: [RR, RL, LR, LL]

    Handle image with polarization in different position:

    >>> # Shape: (4, 64, 512, 512) = (pol, freq, l, m)
    >>> corr_image = image_stokes_to_corr(stokes_image, pol_axis=0)

    With xarray DataArray (preserves structure):

    >>> # Input: xarray DataArray with dims ['time', 'frequency', 'polarization', 'l', 'm']
    >>> stokes_sky = image_xds["SKY"]  # DataArray with Stokes parameters
    >>> corr_sky = image_stokes_to_corr(stokes_sky, corr_type='linear')
    >>> # Output: xarray DataArray with same dims, polarization coord = ['XX', 'XY', 'YX', 'YY']
    """
    # Check if input is xarray
    is_xarray = isinstance(image_data, xr.DataArray)

    # Extract numpy array if xarray
    if is_xarray:
        data = image_data.values
        pol_dim_name = image_data.dims[pol_axis]
    else:
        data = image_data

    # Move polarization axis to end (creates view, no copy)
    data_moved = np.moveaxis(data, pol_axis, -1)

    # Apply conversion using low-level function (operates on last axis)
    converted = stokes_to_corr(
        data_moved, corr_type=corr_type, transformation_matrix=transformation_matrix
    )

    # Move polarization axis back to original position
    result = np.moveaxis(converted, -1, pol_axis)

    # If input was xarray, create copy and update in-place
    if is_xarray:
        result_da = image_data.copy()
        result_da.values[:] = result
        # Update polarization coordinate based on output type
        if corr_type == "linear":
            result_da.coords[pol_dim_name] = ["XX", "XY", "YX", "YY"]
        elif corr_type == "circular":
            result_da.coords[pol_dim_name] = ["RR", "RL", "LR", "LL"]
        # For custom, keep original coordinate labels
        # (user is responsible for knowing their custom correlation products)
        return result_da
    else:
        return result
