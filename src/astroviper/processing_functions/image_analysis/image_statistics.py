"""
Methods and utilities for computing statistics on image data, cubes and single
plane images.
"""

import numpy as np


def get_image_masksum(image_xds, data_group_name):
    """
    Compute the per-plane sum of the mask in an image dataset.

    Parameters:
    -----------
    image_xds: xarray.Dataset
        The image dataset with dimensions (time, frequency, polarization, l, m).
    data_group_name: str
        Name of the entry in ``image_xds.attrs["data_groups"]`` whose
        ``"sky"`` key gives the data variable to size against and whose
        optional ``"mask"`` key gives the mask data variable.

    Returns:
    --------
    mask_sum: numpy.ndarray
        Sum of valid (unmasked) pixels per ``(time, frequency, polarization)``
        plane.
    """

    data_group = image_xds.attrs["data_groups"][data_group_name]
    sky_name = data_group["sky"]
    spatial_dims = image_xds[sky_name].dims[-2:]
    mask_name = data_group.get("mask", None)

    if mask_name is not None and mask_name in image_xds:
        mask_sum = image_xds[mask_name].sum(dim=spatial_dims).values
    else:
        # No mask present = all pixels valid, on every plane
        n_pixels = int(np.prod(image_xds[sky_name].shape[-2:]))
        plane_shape = image_xds[sky_name].shape[:-2]
        mask_sum = np.full(plane_shape, n_pixels, dtype=int)

    return mask_sum


def image_peak_residual(
    image_xds, data_group_name, per_plane_stats=False, use_mask=True, dv="SKY_RESIDUAL"
):
    """
    Compute the peak residual of an image, optionally per plane.

    Parameters:
    -----------
    image_xds: xarray.Dataset
        The image dataset with dimensions (time, frequency, polarization, l, m).
    data_group_name: str
        Name of the entry in ``image_xds.attrs["data_groups"]`` whose
        optional ``"mask"`` key gives the mask data variable, used when
        ``use_mask`` is True.
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
        mask_name = image_xds.attrs["data_groups"][data_group_name].get("mask", None)
        if mask_name is not None and mask_name in image_xds:
            mask_xds = image_xds[mask_name]
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
