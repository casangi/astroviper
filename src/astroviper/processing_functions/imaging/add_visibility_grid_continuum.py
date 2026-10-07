"""Taylor-weighted visibility gridding for MT-MFS continuum imaging."""

import copy

import numpy as np
import xarray as xr


def add_visibility_grid_continuum_single_field(
    ms_xds: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    nterms: int,
    reference_frequency: float,
    ms_data_group_in_name: str = "base",
    image_data_group_in_name: str = "residual",
    image_data_group_out_name: str = "residual",
    image_data_group_out_modified: dict | None = None,
    overwrite: bool = True,
    fft_padding: float = 1.2,
    processing_function_threads: int = 1,
    complex_dtype=None,
):
    """Grid visibilities into MT-MFS residual Taylor uv grids.

    For each Taylor order ``t`` this function grids the visibility data using
    the effective imaging weight

    ``weight_imaging * x**t``

    where

    ``x = (frequency - reference_frequency) / reference_frequency``.

    The generated arrays have dimensions

    ``VISIBILITY(time, taylor_term, polarization, u, v)``

    and

    ``VISIBILITY_NORMALIZATION(time, taylor_term, polarization)``.

    Repeated calls accumulate contributions from multiple measurement sets into
    the same image dataset.

    Parameters
    ----------
    ms_xds : xr.Dataset
        Measurement-set dataset containing correlated data, UVW coordinates,
        imaging weights, and a frequency coordinate.
    cgk_1D : np.ndarray
        One-dimensional prolate-spheroidal convolution kernel.
    img_xds : xr.Dataset
        Image dataset that receives the Taylor uv grids.
    nterms : int
        Number of residual Taylor terms.
    reference_frequency : float
        Common MT-MFS reference frequency in Hz.
    ms_data_group_in_name : str, optional
        Input measurement-set data group.
    image_data_group_in_name : str, optional
        Existing image data group used as the input group.
    image_data_group_out_name : str, optional
        Image data group under which the Taylor grids are registered.
    image_data_group_out_modified : dict, optional
        Output role-to-variable mapping.
    overwrite : bool, optional
        Passed to the data-group helper.
    fft_padding : float, optional
        FFT padding factor.
    processing_function_threads : int, optional
        Number of threads used by the C++ gridder.
    complex_dtype : numpy dtype, optional
        Complex precision of the visibility grid.

    Returns
    -------
    None
        ``img_xds`` is modified in place.
    """

    from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp import (
        prolate_spheroidal_grid,
    )
    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )

    # Read input
    nterms = int(nterms)
    reference_frequency = float(reference_frequency)

    if nterms < 1:
        raise ValueError("nterms must be at least 1.")

    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "reference_frequency must be a positive finite frequency in Hz."
        )

    if complex_dtype is None:
        complex_dtype = np.complex128

    # Prepare image data group
    if image_data_group_out_modified is None:
        image_data_group_out_modified = {
            "visibility": "VISIBILITY",
            "visibility_normalization": "VISIBILITY_NORMALIZATION",
        }
    image_data_group_out_modified = copy.deepcopy(image_data_group_out_modified)

    ms_data_group_in = ms_xds.attrs["data_groups"][ms_data_group_in_name]

    _, image_data_group_out = create_data_groups_in_and_out(
        img_xds,
        data_group_in_name=image_data_group_in_name,
        data_group_out_name=image_data_group_out_name,
        data_group_out_modified=image_data_group_out_modified,
        overwrite=overwrite,
    )

    # Find dimensions
    weight_imaging = ms_xds[ms_data_group_in["weight_imaging"]].values

    if weight_imaging.ndim != 4:
        raise ValueError(
            "The imaging-weight array must have dimensions "
            "(time, baseline, frequency, polarization); received shape "
            f"{weight_imaging.shape}."
        )

    n_chan = weight_imaging.shape[2]
    n_imag_time = 1
    n_time = ms_xds.sizes["time"]
    n_imag_pol = weight_imaging.shape[3]

    time_map = np.zeros(n_time, dtype=int)
    pol_map = np.arange(n_imag_pol, dtype=int)

    # Each gridder invocation writes all input channels into one output plane.
    frequency_map = np.zeros(n_chan, dtype=int)

    n_uv = padded_grid_size(
        [img_xds.sizes["l"], img_xds.sizes["m"]],
        fft_padding,
    )
    delta_lm = img_xds.xr_img.get_lm_cell_size()

    # sanity check on naming of axes
    if "taylor_term" not in img_xds.coords:
        img_xds.coords["taylor_term"] = np.arange(
            nterms,
            dtype=np.int32,
        )
    elif img_xds.sizes["taylor_term"] != nterms:
        raise ValueError(
            "Existing taylor_term coordinate has length "
            f"{img_xds.sizes['taylor_term']}, but nterms={nterms}."
        )

    visibility_name = image_data_group_out["visibility"]
    normalization_name = image_data_group_out["visibility_normalization"]

    expected_grid_shape = (
        n_imag_time,
        nterms,
        n_imag_pol,
        n_uv[0],
        n_uv[1],
    )
    expected_normalization_shape = (
        n_imag_time,
        nterms,
        n_imag_pol,
    )

    # Prepare empty grids with the correct layout
    if visibility_name not in img_xds:
        coords = {
            dim: img_xds.coords[dim]
            for dim in ("time", "taylor_term", "polarization", "u", "v")
            if dim in img_xds.coords
        }
        img_xds[visibility_name] = xr.DataArray(
            np.zeros(expected_grid_shape, dtype=complex_dtype),
            dims=[
                "time",
                "taylor_term",
                "polarization",
                "u",
                "v",
            ],
            coords=coords,
        )

        normalization_coords = {
            dim: img_xds.coords[dim]
            for dim in ("time", "taylor_term", "polarization")
            if dim in img_xds.coords
        }
        img_xds[normalization_name] = xr.DataArray(
            np.zeros(expected_normalization_shape, dtype=np.double),
            dims=["time", "taylor_term", "polarization"],
            coords=normalization_coords,
        )

        modify_data_groups_xds(
            img_xds,
            image_data_group_out_name,
            image_data_group_out,
            description=(
                "Added Taylor-weighted gridded visibilities to img_xds "
                "with add_visibility_grid_continuum_single_field."
            ),
        )
    else:
        if img_xds[visibility_name].shape != expected_grid_shape:
            raise ValueError(
                f"Existing {visibility_name} has shape "
                f"{img_xds[visibility_name].shape}; expected "
                f"{expected_grid_shape}."
            )
        if img_xds[normalization_name].shape != expected_normalization_shape:
            raise ValueError(
                f"Existing {normalization_name} has shape "
                f"{img_xds[normalization_name].shape}; expected "
                f"{expected_normalization_shape}."
            )

    # extract grid, correlated data, uvw and frequencies
    grid = img_xds[visibility_name].values
    normalization = img_xds[normalization_name].values

    vis_data = ms_xds[ms_data_group_in["correlated_data"]].values
    uvw = ms_xds[ms_data_group_in["uvw"]].values
    frequency_coord = np.asarray(ms_xds.frequency.values)

    if frequency_coord.shape[0] != n_chan:
        raise ValueError(
            "The frequency coordinate length does not match the imaging-weight "
            f"channel axis: {frequency_coord.shape[0]} != {n_chan}."
        )

    # Taylor weighting related to reference frequency
    x = (
        frequency_coord.astype(np.float64, copy=False) - reference_frequency
    ) / reference_frequency

    # Common zeroth-order normalization.
    zeroth_normalization = np.zeros(
        (n_imag_time, 1, n_imag_pol),
        dtype=np.double,
    )

    # Gridding per Taylor term
    for taylor_term in range(nterms):
        # taylor weight
        taylor_factor = x**taylor_term

        taylor_imaging_weight = np.ascontiguousarray(
            weight_imaging * taylor_factor[None, None, :, None]
        )

        grid_view = grid[:, taylor_term : taylor_term + 1, :, :, :]

        # to be overwritten by zero-order normalization
        temporary_normalization = np.zeros(
            (n_imag_time, 1, n_imag_pol),
            dtype=np.double,
        )

        # gridding functionality
        prolate_spheroidal_grid(
            grid_view,
            temporary_normalization,
            vis_data,
            uvw,
            frequency_coord,
            frequency_map,
            time_map,
            pol_map,
            taylor_imaging_weight,
            cgk_1D,
            n_uv,
            delta_lm,
            support=7,
            oversampling=100,
            processing_function_threads=processing_function_threads,
        )

        #
        # Keep only the ordinary sum of imaging weights.
        #
        if taylor_term == 0:
            zeroth_normalization += temporary_normalization

    #
    # Every residual Taylor image is normalized by the same
    # zeroth-order sum of imaging weights.
    #
    # Repeated calls add separate processing-set children to the same Taylor
    # grid, so their normalization sums must be accumulated in the same way.
    normalization[...] += zeroth_normalization

    # Format output
    img_xds[visibility_name].attrs.update(
        {
            "description": "MT-MFS Taylor-weighted visibility uv grids.",
            "nterms": nterms,
            "reference_frequency": reference_frequency,
        }
    )
    img_xds[normalization_name].attrs.update(
        {
            "description": (
                "Normalization sums for MT-MFS Taylor-weighted visibility uv grids."
            ),
            "nterms": nterms,
            "reference_frequency": reference_frequency,
        }
    )
