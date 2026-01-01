"""Utilities for preparing standard image grids and applying primary beam."""

from typing import List, Optional, Union

import numpy as np
import xarray as xr
from xradio.image.image import make_empty_lmuv_image
from astroviper.core.imaging.fft import fft_lm_to_uv
from astroviper.core.imaging.ifft import ifft_uv_to_lm
from astroviper.core.imaging.imaging_utils.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel,
    create_prolate_spheroidal_kernel_1D,
)
from astroviper.core.imaging.imaging_utils.standard_grid import (
    standard_grid_numpy_wrap_input_checked,
)


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
    insert_res_or_mod=True,
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
    insert_res_or_mod: optional, default True
    copy  model or residual dataArray to the padded output image, depending
    on if VISIBILITY_MODEL or VISIBILITY_RESIDUAL was requested as padded grid
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
    # pc = image.direction["reference"]["data"]
    pc = image.coordinate_system_info["reference_direction"]["data"]
    cellsize_l = np.abs(image.l[1].data - image.l[0].data)
    cellsize_m = np.abs(image.m[1].data - image.m[0].data)
    cellsize = np.array([cellsize_l, cellsize_m])
    freq = image.frequency.values
    pol = image.polarization.values
    tim = image.time.values
    dir_frame = image.coordinate_system_info["reference_direction"]["attrs"]["frame"]
    # dir_frame = image.direction["reference"]["attrs"]["frame"]
    freq_frame = image.frequency.observer
    projection = image.coordinate_system_info["projection"]
    # projection = image.direction["projection"]

    out_im = make_empty_lmuv_image(
        phase_center=pc,
        image_size=image_size,
        sky_image_cell_size=cellsize,
        frequency_coords=freq,
        pol_coords=pol,
        time_coords=tim,
        direction_reference=dir_frame,
        spectral_reference=freq_frame,
        projection=projection,
    )
    vis_data_dims = ("time", "frequency", "polarization", "u", "v")
    sky_data_shape = (
        len(tim),
        len(freq),
        len(pol),
        image_size[0],
        image_size[1],
    )
    weight_shape = (len(tim), len(freq), len(pol))
    vis_coords = {dim: out_im.coords[dim] for dim in vis_data_dims}
    out_im[uv_data_array] = xr.DataArray(
        np.zeros(sky_data_shape, dtype=np.complex64),
        coords=vis_coords,
        dims=vis_data_dims,
    )
    weight_coords = vis_coords.copy()
    weight_coords.pop("u")
    weight_coords.pop("v")
    out_im[uv_data_array + "_NORMALIZATION"] = xr.DataArray(
        np.zeros(weight_shape, dtype=np.float64),
        coords=weight_coords,
        dims=("time", "frequency", "polarization"),
    )
    out_im.attrs["data_groups"]["base"][uv_data_array.lower()] = uv_data_array
    out_im.attrs["data_groups"]["base"][uv_data_array.lower() + "_normalization"] = (
        uv_data_array + "_NORMALIZATION"
    )
    if insert_res_or_mod:
        sky_data_dims = ("time", "frequency", "polarization", "l", "m")
        sky_coords = {dim: out_im.coords[dim] for dim in sky_data_dims}
        dat_name = ""
        if uv_data_array == "VISIBILITY_MODEL":
            dat_name = "MODEL"
        elif uv_data_array == "VISIBILITY_RESIDUAL":
            dat_name = "RESIDUAL"
        if dat_name and dat_name in image.data_vars:
            dat = image[dat_name].values
            blc = (image_size - dat.shape[-2:]) // 2
            trc = blc + dat.shape[-2:]
            out_im[dat_name] = xr.DataArray(
                np.zeros(sky_data_shape, dtype=np.float32),
                coords=sky_coords,
                dims=sky_data_dims,
            )

            out_im[dat_name].values[:, :, :, blc[0] : trc[0], blc[1] : trc[1]] = dat
            out_im.attrs["data_groups"]["base"][dat_name.lower()] = dat_name

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
        if sky_var in image.data_vars:
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


def fft_to_uv(image: xr.Dataset, data_vars: Union[str, List[str]] = "") -> None:
    """


    Parameters
    ----------
    image : xr.Dataset
        xarray dataset that contains at least one image in in l,m.
    data_vars : Union[str, List[str]], optional
        list of image data variables to transform to uv grid.
        options are RESIDUAL, MODEL, SKY and they have to exist in the dataset
        The default is "".

    Returns
    -------
    None

    """
    if not data_vars:
        try_vars = ["SKY", "RESIDUAL", "MODEL"]
    elif isinstance(data_vars, str):
        try_vars = [data_vars]
    else:
        try_vars = data_vars
    for sky_var in try_vars:
        if sky_var in image.data_vars:
            dat_name = "VISIBILITY_" + sky_var
            if dat_name not in image.data_vars:
                raise Exception(f"{dat_name} is not in {image}")
            image[dat_name].data = fft_lm_to_uv(image[sky_var].data, axes=[3, 4])


def ifft_to_lm(image: xr.Dataset, data_vars: Union[str, List[str]] = "") -> None:
    """


    Parameters
    ----------
    image : xr.Dataset
        xarray dataset that contains at least one image in in u,v.
    data_vars : Union[str, List[str]], optional
        list of image data variables to transform to uv grid.
        options are VISIBILITY_RESIDUAL, VISIBILITY_MODEL, VISIBILITY and they have to exist in the dataset
        The default is "".

    Returns
    -------
    None

    """
    if not data_vars:
        try_vars = [
            "VISIBILITY",
            "VISIBILITY_RESIDUAL",
            "VISIBILITY_MODEL",
        ]
    elif isinstance(data_vars, str):
        try_vars = [data_vars]
    else:
        try_vars = data_vars
    for vis_var in try_vars:
        if vis_var in image.data_vars:
            dat_name = (
                "SKY" if vis_var == "VISIBILITY" else vis_var.split("VISIBILITY_", 1)[1]
            )
            image[dat_name].data = ifft_uv_to_lm(image[vis_var].data, axes=[3, 4])


def correct_fft_to_uv(image: xr.Dataset, data_vars: Union[str, List[str]] = "") -> None:
    """
    Function to correct the N factor when transforming to "frequency"

    Parameters
    ----------
    image : xr.Dataset
        xarray dataset that contains at least one image in in u,v.
    data_vars : Union[str, List[str]], optional
        list of image data variables to transform to uv grid.
        options are VISIBILITY_RESIDUAL, VISIBILITY_MODEL, VISIBILITY and they have to exist in the dataset
        The default is "".

    Returns
    -------
    None
        DESCRIPTION.

    """
    if not data_vars:
        try_vars = ["VISIBILITY", "VISIBILITY_RESIDUAL", "VISIBILITY_MODEL"]
    elif isinstance(data_vars, str):
        try_vars = [data_vars]
    else:
        try_vars = data_vars
    u_size = image.sizes["u"]
    v_size = image.sizes["v"]
    for vis_var in try_vars:
        if vis_var in image.data_vars:
            image[vis_var].data /= u_size * v_size


def correct_ifft_to_lm(
    image: xr.Dataset,
    data_vars: Union[str, List[str]] = "",
    doSpheroidCorr: bool = False,
    convsampling: int = 100,
    convsupport: int = 7,
) -> None:
    """
    Function to correct the N factor when transforming from "frequency",
    Can correct for spheroid convolution function used if applicable

    Parameters
    ----------
    image : xr.Dataset
        xarray dataset that contains at least one image in in u,v.
    data_vars : Union[str, List[str]], optional
        list of image data variables to transform to uv grid.
        options are RESIDUAL, SKY and they have to exist in the dataset
        The default is "".
    doSpheroidCorr : bool default False
        Correct for the effect of the spheroidal function used in gridding
    convsampling : int default 100
        oversmapling used in Convolution function used in gridding
    convsupport : int default 7
        support size of the convolution function used

    Returns
    -------
    None
        DESCRIPTION.

    """
    if not data_vars:
        try_vars = [
            "SKY",
            "RESIDUAL",
        ]
    elif isinstance(data_vars, str):
        try_vars = [data_vars]
    else:
        try_vars = data_vars
    nx = image.sizes["l"]
    ny = image.sizes["m"]
    imsize = np.array([nx, ny])
    if doSpheroidCorr:
        kernel, corrTerm = create_prolate_spheroidal_kernel(
            convsampling, convsupport, imsize
        )
    for sky_var in try_vars:
        if sky_var in image.data_vars:
            if not doSpheroidCorr:
                image[sky_var] *= nx * ny
            else:
                sumwt = (
                    "VISIBILITY_NORMALIZATION"
                    if sky_var == "SKY"
                    else "VISIBILITY_" + sky_var + "_NORMALIZATION"
                )
                if sumwt not in image.data_vars:
                    raise Exception(f"Do not have {sumwt} to do image correction")
                dat_array = image[sky_var]
                for t_coord in dat_array.coords["time"].values:
                    for f_coord in dat_array.coords["frequency"].values:
                        for p_coord in dat_array.coords["polarization"].values:
                            # Select the 2D slice for the current combination of t, f, p
                            # Using .sel() with exact coordinate values
                            data_slice = dat_array.sel(
                                time=t_coord,
                                frequency=f_coord,
                                polarization=p_coord,
                            ).data
                            data_slice *= (nx * ny) / (
                                corrTerm
                                * image[sumwt].sel(
                                    frequency=f_coord, polarization=p_coord
                                )
                            )


def grid2xradio_spheroid_ms4(
    vis: Union[xr.core.datatree.DataTree, List[xr.core.datatree.DataTree]],
    image: xr.Dataset,
    support: int = 7,
    sampling: int = 100,
    data_var: str = "VISIBILITY",
    chan_mode: str = "continuum",
):
    """
    Parameters
    ----------
    vis : single xarray.core.datatree.DataTree or a list of them
        an ms v4 compatible xarray
    resid_array : np.ndarray
        an array that defines the image shape
        will contain the image made
    pixelincr : np.ndarray
        pixel increment in the direction coordinate axes
    support : int, optional
        Size of support of the spheroidal convolution function
        The default is 7.
    sampling : int, optional
        oversampling of convolution function
        The default is 100.
    data_var : TYPE, optional
        The default is "VISIBILITY".
        options are : 'VISIBILITY', 'UV_SAMPLING', 'VISIBILITY_RESIDUAL'
    chan_mode : TYPE, optional
        can be continuum or cube
        The default is "continuum".

    Returns
    -------
    None.
    *TODO* option to avoid DC (u=0, v=0)
    For which time coordinate is the gridding supposed to be
    """
    if isinstance(vis, xr.core.datatree.DataTree):
        listvis = [vis]
    else:
        listvis = vis
    nx = image.sizes["u"]
    ny = image.sizes["v"]
    dopsf = False
    if data_var == "UV_SAMPLING":
        dopsf = True
    cellsize_l = np.abs(image.l[1].data - image.l[0].data)
    cellsize_m = np.abs(image.m[1].data - image.m[0].data)
    pixelincr = np.array([cellsize_l, cellsize_m])
    if data_var not in image.data_vars:
        raise Exception("{data_var} is not in {image}")
    sumwt_var = data_var + "_NORMALIZATION"
    dat_array = image[data_var]
    sumwt_array = image[sumwt_var]
    cgk_1D = create_prolate_spheroidal_kernel_1D(sampling, support)
    for t_coord in dat_array.coords["time"].values:
        gridvis = dat_array.sel(time=t_coord).data
        sumwt = sumwt_array.sel(time=t_coord).data
        for elvis in listvis:
            if not isinstance(elvis, xr.core.datatree.DataTree):
                raise TypeError("One of the elements of vis is not an xarray datatree")
            vis_data = elvis[data_var].data
            uvw = elvis.UVW.data
            weight = elvis.WEIGHT.data
            ###Make sure flag data are not used
            flag = elvis.FLAG.data
            weight[flag] = 0.0
            # weight[np.logical_and(uvw[:, :, 0] == 0, uvw[:, :, 1] == 0)] = 0.0
            freq_chan = elvis.coords["frequency"].values

            gridvis, sumwt = standard_grid_numpy_wrap_input_checked(
                vis_data,
                gridvis,
                sumwt,
                uvw,
                weight,
                freq_chan,
                cgk_1D,
                image_size=np.array([nx, ny], dtype=int),
                cell_size=pixelincr,
                oversampling=sampling,
                support=support,
                complex_grid=True,
                do_psf=dopsf,
                chan_mode=chan_mode,
            )
