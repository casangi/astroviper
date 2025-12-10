# File that contains function to prepare grid or images for converting to
# images or uv grid

import numpy as np
import xarray
from typing import Optional, Union, List


def remove_padding(image: xarray.core.dataset.Dataset, image_size: np.ndarray):
    """
    presumably image is the product of fft of gridded vis that have been
    oversampled
    Reduces image back to image_size

    Parameters
    ----------
    image : xarray.core.datatree.DataTree
        DESCRIPTION.
    image_size : np.ndarray
        DESCRIPTION.

    Returns
    -------
    image_dask_array : TYPE
        DESCRIPTION.

    """
    image_size = np.array(image_size)
    image_size_padded = np.array([image.sizes["l"], image.sizes["m"]])
    start_xy = image_size_padded // 2 - image_size // 2
    end_xy = start_xy + image_size

    # print("start end",image_size_padded,start_xy,end_xy)

    image = image.isel(l=slice(start_xy[0], end_xy[0]), m=slice(start_xy[1], end_xy[1]))
    return image


def apply_pb(
    image: xarray.core.dataset.Dataset,
    pb: Optional[xarray.core.dataset.Dataset] = None,
    multiply: bool = True,
    type: str = "flatnoise",
    data_vars: Union[str, List[str]] = "",
):
    """


    Parameters
    ----------
    image : xarray.core.dataset.Dataset
        DESCRIPTION.
    pb : Optional[xarray.core.dataset.Dataset], optional
        DESCRIPTION. The default is None.
    multiply : bool, optional
        DESCRIPTION. The default is True.
    type : TYPE, optional
        DESCRIPTION. The default is str: 'flatnoise'.
    data_vars: str or list[str] Optional; possible values: SKY, RESIDUAL, MODEL

    Returns
    -------
    None.

    """
    if (not pb) and (not ("PRIMARY_BEAM" in image)):
        raise Exception(f"No Primary beam information in {image}")
    if pb:
        pb_da = pb["PRIMARY_BEAM"]
    else:
        pb_da = image["PRIMARY_BEAM"]
    if not data_vars:
        try_vars = ["SKY", "RESIDUAL", "MODEL"]
    elif type(data_vars) == str:
        try_vars = [data_vars]
    else:
        try_vars = data_vars
    if pb.shape[-2:] != image[try_vars[0]].shape[-2:]:
        raise Exception(f"shapes on sky  are not the same ")
    for sky_var in try_vars:
        _mult_div(image[sky_var], pb, multiply=multiply)


def _mult_div(
    im_da: xarray.dataarray.DataArray,
    pb_da: xarray.core.dataarray.DataArray,
    multiply: bool = True,
):
    """

    Parameters
    ----------
    im_da : xarray.dataarray.DataArray
        DESCRIPTION.
    pb_da : xarray.core.dataarray.DataArray
        DESCRIPTION.
    multiply : boo, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

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
        tuse = t if (tind_eq) else 0
        for c in range(im_da.shape[1]):
            cuse = c if (cind_eq) else 0
            for p in range(im_da.shape[2]):
                puse = p if (pind_eq) else 0
                if multiply:
                    im_da[t, c, p, :, :] *= pb_da[tuse, cuse, puse, :, :]
                else:
                    im_da[t, c, p] /= pb_da[tuse, cuse, puse, :, :]
                    im_da[t, c, p, :, :].data[
                        pb_da[tuse, cuse, puse, :, :] == 0
                    ] = np.nan
