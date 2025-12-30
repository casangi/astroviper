# generate primary beam image using make_pb_symmetric.casa_airy_disk_rorder
from astroviper.core.imaging.imaging_utils.make_pb_symmetric import (
    airy_disk_rorder,
    casa_airy_disk_rorder,
)
from xradio.image import make_empty_sky_image
import xarray as xr
import numpy as np


def cube_single_field_primary_beam(im_params, telescope, model="casa_airy_disk"):
    """
    Generate a primary beam image for a single field in a measurement set.

    Parameters:
    im_params : dict
        Imaging parameters including 'cell', 'imsize', and 'freq'.

        The name of the telescope (e.g., 'ALMA', 'ACA').

    Returns:
    pb_image : np.ndarray
        The generated primary beam image.
    """

    # ALMA parameters:
    # ALMA 12m: effective diameter 10.7m, blockage 0.75m
    # ACA 7m: effective diameter 6.7m, blockage 0.75m
    match telescope:
        case "ALMA":
            dish_diameter = 10.7
            blockage_diameter = 0.75
        case "ACA":
            dish_diameter = 6.7
            blockage_diameter = 0.75
        case _:
            raise ValueError(f"Unsupported telescope: {telescopes}")

    pb_params = {
        "list_dish_diameters": [10.7],  # in meters
        "list_blockage_diameters": [0.75],  # in meters
        "ipower": 2,
    }
    cell = im_params["cell_size"]
    imsize = im_params["image_size"]
    freq_chan = np.array(im_params["frequency_coords"])
    pol = im_params["polarization"]
    phase_center = im_params["phase_center"]
    print("phase_center:", phase_center)
    print("imsize=", imsize)
    if model == "casa_airy_disk":
        pb_image_data = casa_airy_disk_rorder(freq_chan, pol, pb_params, im_params)
    else:
        pb_image_data = airy_disk_rorder(freq_chan, pol, pb_params, im_params)
    print("pb_image_data.shape=", pb_image_data.shape)
    pb_image = make_empty_sky_image(
        phase_center=phase_center,
        image_size=imsize,
        cell_size=cell,
        frequency_coords=freq_chan,
        pol_coords=pol,
        time_coords=im_params["time_coords"],
        direction_reference="fk5",
        projection="SIN",
        do_sky_coords=True,
    )
    print("pb_image.dims=", tuple(pb_image.dims))
    print("pb_image.sizes.values=", pb_image.sizes.values())
    print("pb_image.coords=", pb_image.coords)
    mypb_dims = ("time", "frequency", "polarization", "l", "m")
    new_dims = tuple(d for d in pb_image.dims if d != "beam_params_label")
    coords = pb_image.drop_vars("beam_params_label").coords
    pb_da = xr.DataArray(
        pb_image_data,
        # dims=pb_image.dims[:-1],
        dims=new_dims,
        coords=coords,
        name="PRIMARY_BEAM",
    )

    pb_image["PRIMARY_BEAM"] = pb_da
    return pb_image
