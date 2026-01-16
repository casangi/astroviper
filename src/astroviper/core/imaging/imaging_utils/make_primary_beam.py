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
        Imaging parameters: must include
        'cell_size': angular size of a pixel 2d tuple (rad, rad),
        'image_size': int 2d tuple (nx, ny),
        'frequency_coords': list of frequencies in Hz,
        'polarization': list of Stokes parameters,
        'image_center': image center pixel coordinates int 2d tuple (x, y),
        'phase_center': phase reference center (RA, Dec) in radians,
        'time_coords': list of time coordinates in seconds.

    telescope : str
        The name of the telescope (e.g., 'ALMA', 'ACA').

    model : str, optional
        The primary beam model to use. Currently only one of the two versions of
        Airy disk model defined in make_pb_symmetric. Default is 'casa_airy_disk'.

    Returns:
    pb_image : xarray.Dataset
        The generated primary beam image as an XRADIO image with a data variable 'PRIMARY_BEAM'.

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
            raise ValueError(f"Unsupported telescope: {telescope}")

    pb_params = {
        "list_dish_diameters": [dish_diameter],  # in meters
        "list_blockage_diameters": [blockage_diameter],  # in meters
        "ipower": 2,
    }
    cell = im_params["cell_size"]
    imsize = im_params["image_size"]
    freq_chan = np.array(im_params["frequency_coords"])
    pol = im_params["polarization"]
    phase_center = im_params["phase_center"]
    if model == "casa_airy_disk":
        pb_image_data = casa_airy_disk_rorder(freq_chan, pol, pb_params, im_params)
    else:
        pb_image_data = airy_disk_rorder(freq_chan, pol, pb_params, im_params)
    # the first dimension for pb_image_data is antenna (for storing multiple dish size models)
    pb_image_data = pb_image_data[0]
    # Asuume here no time dependence for PB so for multiple time coords, just repeat the same PB image
    time_coords_size = len(im_params["time_coords"])
    pb_image_data_repeated = np.repeat(
        pb_image_data[np.newaxis, :, :, :, :], time_coords_size, axis=0
    )

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
    new_dims = tuple(d for d in pb_image.dims if d != "beam_params_label")
    coords = pb_image.drop_vars("beam_params_label").coords
    pb_da = xr.DataArray(
        pb_image_data_repeated,
        dims=new_dims,
        coords=coords,
        name="PRIMARY_BEAM",
    )

    pb_image["PRIMARY_BEAM"] = pb_da
    return pb_image
