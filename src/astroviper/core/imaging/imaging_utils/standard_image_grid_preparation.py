"""Utilities for preparing standard image grids and applying primary beam."""

from typing import List, Optional, Union

import numpy as np
import xarray as xr
from xradio.image.image import make_empty_lmuv_image


def remove_padding(image: xr.Dataset, image_size: np.ndarray) -> xr.Dataset:
    """Reduce an oversampled image back to ``image_size``.

    Parameters
    ----------
    image:
        An xarray ``Dataset`` containing spatial dims ``l`` and ``m``.
    image_size:
        Sequence-like with two integers for the output ``(l, m)`` size.

    Returns
    -------
    xr.Dataset
        The cropped dataset.
    """

    image_size = np.array(image_size)
    image_size_padded = np.array([image.sizes["l"], image.sizes["m"]])
    start_xy = image_size_padded // 2 - image_size // 2
    end_xy = start_xy + image_size

    return image.isel(l=slice(start_xy[0], end_xy[0]), m=slice(start_xy[1], end_xy[1]))


def make_empty_padded_uv_image(
    image: xr.Dataset,
    image_size: np.ndarray,
    uv_data_array: str = "VISIBILITY_RESIDUAL",
) -> xr.Dataset:
    """Create an empty UV image with padding to ``image_size``.


    Parameters
    ----------
    image:
        Reference xradion image dataset
    image_size:
        Target image size in the l,m or u, v dimensions.
    uv_data_array: optional,
    options are : "VISIBILITY", "VISIBILITY_RESIDUAL", "VISIBILITY_MODEL", "UV_SAMPLING"

    Returns
    -------
    xr.Dataset

    Raises
    ------
    Exception
        If the current image is larger than the requested size.
    """

    image_size = np.array(image_size)
    image_size_current = np.array([image.sizes["l"], image.sizes["m"]])
    if np.any(image_size_current > image_size):
        raise Exception(
            f"current image size {image_size_current} is larger than requested {image_size}"
        )
    # lets get some parameters to make_empty_lmuv_image
    pc = image.direction["reference"]["data"]
    cellsize_l = np.abs(image.l[1].data - image.l[0].data)
    cellsize_m = np.abs(image.m[1].data - image.m[0].data)
    cellsize = np.array([cellsize_l, cellsize_m])
    freq = image.frequency.values
    pol = image.polarization.values
    tim = image.time.values
    # dir_frame = image.coordinate_system_info["reference_direction"]["attrs"]["frame"]
    dir_frame = image.direction["reference"]["attrs"]["frame"]
    freq_frame = image.frequency.observer
    # projection = image.coordinate_system_info["projection"]
    projection = image.direction["projection"]

    out_im = make_empty_lmuv_image(
        phase_center=pc,
        image_size=image_size,
        sky_image_cell_size=cellsize,
        chan_coords=freq,
        pol_coords=pol,
        time_coords=tim,
        direction_reference=dir_frame,
        spectral_reference=freq_frame,
        projection=projection,
    )
    sky_data_dims = ("time", "frequency", "polarization", "l", "m")
    sky_data_shape = (
        len(tim),
        len(freq),
        len(pol),
        image_size[0],
        image_size[1],
    )
    weight_shape = (len(tim), len(freq), len(pol))
    sky_coords = {dim: out_im.coords[dim] for dim in sky_data_dims}
    out_im[uv_data_array] = xr.DataArray(
        np.zeros(sky_data_shape, dtype=np.complex64),
        coords=sky_coords,
        dims=sky_data_dims,
    )
    weight_coords = sky_coords.copy()
    weight_coords.pop("l")
    weight_coords.pop("m")
    out_im[uv_data_array + "_NORMALIZATION"] = xr.DataArray(
        np.zeros(weight_shape, dtype=np.float64),
        coords=weight_coords,
        dims=("time", "frequency", "polarization"),
    )
    return out_im


def apply_pb(
    image: xr.Dataset,
    pb: Optional[xr.Dataset] = None,
    multiply: bool = True,
    data_vars: Union[str, List[str]] = "",
) -> None:
    """Apply (multiply or divide) the primary beam to data variables.

    Parameters
    ----------
    image:
        Dataset containing data variables (e.g. ``SKY``) and optionally
        a ``PRIMARY_BEAM`` variable.
    pb:
        Optional separate dataset containing ``PRIMARY_BEAM``.
    multiply:
        If True multiply, otherwise divide.
    data_vars:
        Variable name or list of variable names to operate on. If empty,
        defaults to ["SKY", "RESIDUAL", "MODEL"].
    """

    if (pb is None) and ("PRIMARY_BEAM" not in image):
        raise Exception(f"No Primary beam information in {image}")

    pb_da = pb["PRIMARY_BEAM"] if pb is not None else image["PRIMARY_BEAM"]

    if not data_vars:
        try_vars = ["SKY", "RESIDUAL", "MODEL"]
    elif isinstance(data_vars, str):
        try_vars = [data_vars]
    else:
        try_vars = data_vars

    if pb_da.shape[-2:] != image[try_vars[0]].shape[-2:]:
        raise Exception("shapes on sky are not the same")

    for sky_var in try_vars:
        _mult_div(image[sky_var], pb_da, multiply=multiply)


def _mult_div(im_da: xr.DataArray, pb_da: xr.DataArray, multiply: bool = True) -> None:
    """Multiply or divide a sky DataArray by a primary-beam DataArray.

    Works with either matching shapes or with broadcasting over the leading
    axes (time/frequency/polarization).
    """

    if im_da.shape == pb_da.shape:
        if multiply:
            im_da *= pb_da
        else:
            im_da /= pb_da
            im_da.data[pb_da == 0.0] = np.nan
        return

    tind_eq = im_da.shape[0] == pb_da.shape[0]
    cind_eq = im_da.shape[1] == pb_da.shape[1]
    pind_eq = im_da.shape[2] == pb_da.shape[2]

    for t in range(im_da.shape[0]):
        tuse = t if tind_eq else 0
        for c in range(im_da.shape[1]):
            cuse = c if cind_eq else 0
            for p in range(im_da.shape[2]):
                puse = p if pind_eq else 0
                if multiply:
                    im_da[t, c, p, :, :] *= pb_da[tuse, cuse, puse, :, :]
                else:
                    im_da[t, c, p] /= pb_da[tuse, cuse, puse, :, :]
                    im_da[t, c, p, :, :].data[
                        pb_da[tuse, cuse, puse, :, :] == 0
                    ] = np.nan
