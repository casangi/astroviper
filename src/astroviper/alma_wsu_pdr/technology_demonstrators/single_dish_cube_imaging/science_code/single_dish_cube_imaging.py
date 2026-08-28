import os
from pathlib import Path
import math

import xarray as xr
from xarray import DataTree, DataArray

import zarr
import numpy as np

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time as AstroTime
from astropy.units import Unit as AstroUnit
import astropy.units as u
from astropy.units import Quantity
from astropy import wcs

from xradio.measurement_set.open_processing_set import open_processing_set

from astroviper.processing_functions.imaging\
    .gridding_convolution_functions.gcf_prolate_spheroidal import (
        prolate_spheroidal_function
    )

from astroviper.alma_wsu_pdr.technology_demonstrators\
    .single_dish_cube_imaging.science_code.single_dish_gridder import (
        single_dish_gridder_jit
    )


def initialize_output_images(
        ps_store: str = "",
        image_store: str = "",
        channels_chunk: int = 1000,
        msv4_selection: dict = {},
        image_definition: dict = {}):

    ps_xdt: DataTree = open_processing_set(
        ps_store=ps_store,
        array_backend="xarray"
    )

    # Compute single-dish cube image shape
    image_axes = ("n_x", "n_y", "n_channels", "n_polarizations")
    # ---- Frequency axis: pick-up the number of channels from first selected msv4
    antenna_name, msv4_name = next(iter(msv4_selection.items()))
    msv4_dt = ps_xdt[msv4_name]
    n_channels = msv4_dt.dims["frequency"]
    # ---- Polarization axis
    if image_definition["stokes"] == "I":
        n_polarizations = 1
    else:
        raise RuntimeError(
            f"Not implemented: stokes != 'I': {image_definition['stokes']}"
        )
    image_size = {
        "n_x": int(Quantity(image_definition["image_size"][0]).value),
        "n_y": int(Quantity(image_definition["image_size"][1]).value),
        "n_channels": n_channels,
        "n_polarizations": n_polarizations
    }
    image_shape = tuple(image_size[axis] for axis in image_axes)
    # Compute image cube slices/chunks shape
    image_slice_size = image_size.copy()
    image_slice_size["n_channels"] = channels_chunk
    image_slice_shape = tuple(image_slice_size[axis] for axis in image_axes)
    # Compute images paths
    image_kind_suffix = {
        "plain_image": "image",
        "weight_image": "weight_image"
    }
    images_basename = Path(ps_store).name.removesuffix('.ps.zarr')
    image_kind_path = {
        image_kind: (
            f"{Path(image_store) / images_basename}.{image_suffix}"
        )
        for image_kind, image_suffix in image_kind_suffix.items()
    }
    # Create output images, lazily initialized
    for image_path in image_kind_path.values():
        os.makedirs(Path(image_path).parent.as_posix(), exist_ok=True)
        zarr.open_array(
            image_path,
            mode='w',  # Create, overwrite if exists
            shape=image_shape,
            chunks=image_slice_shape,
            dtype='float32'
        )
    # Compute slices info
    slices_info = []
    slice_index = 0
    for slice_start_channel in range(0, n_channels, channels_chunk):
        slice_end_channel = min(
            n_channels - 1,
            slice_start_channel + channels_chunk - 1
        )
        slice_channels_count = slice_end_channel - slice_start_channel + 1
        slice_info = {
            "slice_index": slice_index,
            "start_channel": slice_start_channel,
            "end_channel": slice_end_channel,
            "n_channels": slice_channels_count
        }
        slices_info.append(slice_info)
        slice_index += 1

    ps_xdt.close()

    result = {
        "image_kind_path": image_kind_path,
        "slices_info": slices_info
    }

    return result


def select_msv4s(ps_store, data_selection: dict):
    """Quick and dirty msv4 selection"""

    ps_xdt: DataTree = open_processing_set(
        ps_store=ps_store,
        array_backend="xarray"
    )

    msv2_spw = data_selection["msv2_spw"]
    spectral_window_suffix = f"_{msv2_spw}"
    msv2_field = data_selection["msv2_field"]
    field_name_suffix = f"_{msv2_field}"

    msv4_selection = {}
    for msv4_name, msv4_dt in ps_xdt.children.items():
        # Assuming here the Processing Set is split by antenna
        antenna_name = msv4_dt.antenna_name.values[0]
        partition_info: dict = msv4_dt.xr_ms.get_partition_info()
        spectral_window_name: str = partition_info["spectral_window_name"]
        if not spectral_window_name.endswith(spectral_window_suffix):
            continue
        field_name: str = partition_info["field_name"][0]
        if not field_name.endswith(field_name_suffix):
            continue
        if antenna_name not in msv4_selection:
            msv4_selection[antenna_name] = msv4_name

    ps_xdt.close()

    return msv4_selection


def image_single_dish_cube_slice(
        ps_store: str = "",
        msv4_selection: dict = {},
        image_kind_path: dict = {},
        image_definition: dict = {},
        slice_info: dict = {},
        data_directions: dict = {}):

    ps_xdt: DataTree = open_processing_set(
        ps_store=ps_store,
        # Fix issue: performances do not scale:
        array_backend="xarray"
    )
    # Open output images in read-write mode
    # ---- Plain image
    plain_image_path = image_kind_path["plain_image"]
    plain_image_store = zarr.open_array(plain_image_path, mode="r+")
    # ---- Weight image
    weight_image_path = image_kind_path["weight_image"]
    weight_image_store = zarr.open_array(weight_image_path, mode="r+")
    # Initialize image slice for each image
    slice_channels_range = slice(
        slice_info["start_channel"], slice_info["end_channel"] + 1
    )
    plain_image_store[:, :, slice_channels_range, :] = 0
    weight_image_store[:, :, slice_channels_range, :] = 0

    # Convolution kernel
    # ---- For now, hard-code and re-compute it for each slice
    cut_radius_pixels = 6
    samples_per_pixels = 100
    n_samples = cut_radius_pixels * samples_per_pixels + 1
    sampling_points = np.linspace(
        start=0, stop=1, num=n_samples, endpoint=True
    )
    _, convolution_function_values = (
        prolate_spheroidal_function(sampling_points)
    )
    kernel_1D = convolution_function_values.astype(np.float32)
    # Create image combining data from all antennas
    for antenna_name, msv4_name in msv4_selection.items():
        msv4_dt = ps_xdt[msv4_name]
        # Retrieve coordinates of spectra taken with the current antenna
        antenna_spectra_coordinates = (
            data_directions[antenna_name]["data_coordinates"]
        )
        # Lazily bin spectra data into requested image data shape
        stokes = image_definition["stokes"]
        if stokes != "I":
            raise RuntimeError(
                f"Not implemented: stokes != 'I': {stokes}"
            )
        value_da = (
            msv4_dt.SPECTRUM
            .squeeze("antenna_name")
            .isel(frequency=slice_channels_range)
        )
        weight_da = (
            msv4_dt.WEIGHT
            .squeeze("antenna_name")
            .isel(frequency=slice_channels_range)
        )
        flag_da = (
            msv4_dt.FLAG
            .squeeze("antenna_name")
            .isel(frequency=slice_channels_range)
        )
        stokes_I_da, stokes_I_weight_da, stokes_I_flag_da = (
            to_stokes_I(
                value_da=value_da,
                weight_da=weight_da,
                flag_da=flag_da
            )
        )
        # Lazily flag invalid binned data
        stokes_I_da_masked = stokes_I_da.where(
            ~stokes_I_flag_da,  # Keep value
            np.nan  # Replace value
        )
        # Initialize output images slice arrays
        n_x = int(
            Quantity(image_definition["image_size"][0]).value
        )
        n_y = int(
            Quantity(image_definition["image_size"][1]).value
        )
        stokes_dim_size = stokes_I_da_masked.coords["stokes"].size
        slice_shape = (n_x, n_y, slice_info["n_channels"], stokes_dim_size)
        slice_values = np.zeros(slice_shape, dtype=np.float32)
        slice_weights = np.zeros(slice_shape, dtype=np.float32)
        # Now do the real work:
        # - compute the values of our lazy arrays
        # - grid them
        single_dish_gridder_jit(
            samples_values=stokes_I_da_masked.values,
            samples_weights=stokes_I_weight_da.values,
            samples_coords=antenna_spectra_coordinates,
            grid_values=slice_values,
            grid_weights=slice_weights,
            cut_radius=cut_radius_pixels,
            sampling=samples_per_pixels,
            convolution_array=kernel_1D
        )
        # Update images stores
        # ---- Plain image
        plain_image_store[:, :, slice_channels_range, :] += (
            slice_values
        )
        # ---- Weight image
        weight_image_store[:, :, slice_channels_range, :] += (
            slice_weights
        )

    # Finally, normalize the values of the slice combining
    # contributions from all antennas
    combined_slice_values = plain_image_store[:, :, slice_channels_range, :]
    combined_slice_weights = weight_image_store[:, :, slice_channels_range, :]
    normalized_slice_values = np.where(
        combined_slice_weights > 0,
        combined_slice_values / combined_slice_weights,
        0
    )
    plain_image_store[:, :, slice_channels_range, :] = normalized_slice_values

    ps_xdt.close()


def to_stokes_I(
        value_da: DataArray,
        weight_da: DataArray,
        flag_da: DataArray):
    # Values
    stokes_I = value_da.mean(dim="polarization")
    stokes_I = (
        stokes_I
        .expand_dims(stokes=["I"])
        .transpose("time", "frequency", "stokes")
    )
    # Weights
    w_xx = weight_da.sel(polarization="XX")
    w_yy = weight_da.sel(polarization="YY")
    numerator = 4 * w_xx * w_yy
    denominator = w_xx + w_yy
    stokes_I_weight: DataArray = xr.where(
        denominator > 0,
        numerator / denominator,
        0
    )
    stokes_I_weight = (
        stokes_I_weight
        .expand_dims(stokes=["I"])
        .transpose("time", "frequency", "stokes")
    )
    # Flags
    flag_xx = flag_da.sel(polarization="XX")
    flag_yy = flag_da.sel(polarization="YY")
    stokes_I_flag = flag_xx | flag_yy
    stokes_I_flag = (
        stokes_I_flag
        .expand_dims(stokes=["I"])
        .transpose("time", "frequency", "stokes")
    )

    return stokes_I, stokes_I_weight, stokes_I_flag


def compute_data_coordinates(
            ps_store: str,
            antenna_name: str,
            msv4_name: str,
            image_definition: dict
        ):

    ps_xdt: DataTree = open_processing_set(
        ps_store=ps_store,
        array_backend="xarray"
    )

    msv4_dt = ps_xdt[msv4_name]

    # 1. Convert Antenna Pointing Directions to ICRS Celestial Reference Frame
    # ---- 1.1 Antenna Pointings Directions
    antenna_pointings = msv4_dt.pointing_xds.POINTING_BEAM.sel(
        antenna_name=antenna_name
    )
    # ---- 1.2 Antenna Pointings Times
    antenna_pointings_astro_times = AstroTime(
        antenna_pointings.time.values * AstroUnit(
            antenna_pointings.time.units
        ),
        format=antenna_pointings.time.format,
        scale=antenna_pointings.time.scale
    )
    # ---- 1.3 Antenna Position
    antenna_position: DataArray = msv4_dt.antenna_xds.ANTENNA_POSITION.sel(
        antenna_name=antenna_name
    )
    # ---- 1.4 Convert Antenna Pointings Directions
    # From: Az/Alt Horizontal Reference Frame
    # To:   ICRS Celestial Reference Frame
    antenna_coords = dict(
        zip(
            antenna_position.cartesian_pos_label.values,
            antenna_position.values
        )
    )
    antenna_pointings_coords_icrs = SkyCoord(
        alt=antenna_pointings.sel(local_sky_dir_label='alt'),
        az=antenna_pointings.sel(local_sky_dir_label='az'),
        unit=antenna_pointings.units,
        obstime=antenna_pointings_astro_times,
        frame=antenna_pointings.frame,
        location=EarthLocation.from_geocentric(
            antenna_coords['x'],
            antenna_coords['y'],
            antenna_coords['z'],
            unit=antenna_position.units
        )
    ).icrs
    # 2. Compute Data Directions:
    #    Interpolate Antenna Pointing Directions at Data Acquisition Times
    #    Currently, this is done at MSv2-to-ProcessingSet conversion time
    antenna_data_directions_icrs = antenna_pointings_coords_icrs

    # 3. Compute Data Coordinates:
    #    Perform WCS Spherical Projections of Data Directions
    # ---- 3.1 Create and configure WCS object
    w = wcs.WCS(naxis=2)
    # Image center coordinates: grid center 1-based coordinates
    image_center_coords = [
        (Quantity(axis_size).value + 1) / 2
        for axis_size in image_definition["image_size"]
    ]
    # Reference pixel coordinates:
    # for compatibility with CASA6 we must set:
    w.wcs.crpix = [
        math.ceil(image_center_coord)
        for image_center_coord in image_center_coords
    ]
    # World coordinates at reference pixel
    frame, grid_center_ra_hms, grid_center_dec_dms = (
        image_definition["center_direction"].split(" ")
    )
    world_grid_center = SkyCoord(
        f"{grid_center_ra_hms} {grid_center_dec_dms.replace('.', ':', 2)}",
        unit=(u.hourangle, u.deg),
        frame=frame.lower()
    )
    w.wcs.crval = [world_grid_center.ra.value, world_grid_center.dec.value]
    # Increments per pixel
    ra_size, dec_size = [
        Quantity(s).to(u.degree).value
        for s in image_definition["cell_size"]
    ]
    w.wcs.cdelt = [-ra_size, dec_size]
    # Projection type
    projection = image_definition["projection"]
    w.wcs.ctype = [
        f"RA---{projection}",
        f"DEC--{projection}"
    ]
    # ---- 3.2 Perform the projection
    antenna_data_coords_pixels = w.wcs_world2pix(
        np.column_stack((
            antenna_data_directions_icrs.ra.value,
            antenna_data_directions_icrs.dec.value
        )),
        0  # origin: 0-based/1-based
    )

    return {
        "data_coordinates": antenna_data_coords_pixels,
    }
