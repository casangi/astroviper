"""
Methods and utilities for computing statistics on image data, cubes and single
plane images.
"""

import numpy as np


def get_image_masksum(image_xds, dv="SKY"):
    """
    Compute the sum of the mask in an image dataset.

    Parameters:
    -----------
    image_xds: xarray.Dataset
        The image dataset with dimensions (y, x) or (time, frequency, polarization, y, x).
    dv: str
        The data variable in the xarray.Dataset to get the mask from.
        Default is 'SKY'.

    Returns:
    --------
    mask_sum: int
        The sum of the mask values in the image.
    """

    maskname = image_xds[dv].active_mask
    mask_xds = image_xds[maskname]
    if mask_xds is not None:
        mask_sum = int(mask_xds.sum().values)
    else:
        mask_sum = 0.0

    return mask_sum


def image_peak_residual(image_xds, per_plane_stats=False, use_mask=True, dv="SKY"):
    """
    Compute the peak residual of an image, optionally per plane.

    Parameters:
    -----------
    image_xds: xarray.Dataset
        The image dataset with dimensions (y, x) or (time, frequency, polarization, y, x).
    per_plane_stats: bool
        If True, compute peak residual for each (time, frequency, polarization) plane.
        If False, compute peak residual for the entire image across all planes.
    use_mask: bool
        If True, consider only unmasked pixels in the computation.
    dv: str
        The data variable in the xarray.Dataset to compute the peak residual from.
        Default is 'SKY'.

    Returns:
    --------
    peak_residual: float
        The peak residual value of the image.
    """

    # Get location of the peak absolute value
    # Use that to index into the original image to get the signed value

    # Apply the mask if requested
    if use_mask:
        maskname = image_xds[dv].active_mask
        mask_xds = image_xds[maskname]
        if mask_xds is not None:
            image_xds = image_xds.where(mask_xds)

    if per_plane_stats:
        # Compute peak residual for each (time, frequency, polarization) plane
        peak_residual = image_xds[dv].reduce(
            np.vectorize(
                lambda arr: arr[np.unravel_index(np.abs(arr).argmax(), arr.shape)]
            ),
            dim=["y", "x"],
        )
    else:
        # Compute peak residual for the entire image across all planes
        peak_res_idx = np.unravel_index(
            np.abs(image_xds[dv].values).argmax(), image_xds[dv].shape
        )
        peak_residual = image_xds[dv].values[peak_res_idx]

    return peak_residual
