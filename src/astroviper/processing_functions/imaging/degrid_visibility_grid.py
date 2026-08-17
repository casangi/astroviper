"""Shared standard-gridder visibility prediction primitive."""

import copy

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)


def degrid_visibility_grid_single_field(
    ms_xds: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    grid: np.ndarray,
    frequency_map: np.ndarray,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "model",
    ms_data_group_out_modified: dict | None = None,
    overwrite: bool = True,
    fft_padding: float = 1.2,
    processing_function_threads: int = 1,
    description: str = "Degridded standard-gridder model visibilities.",
):
    """Sample a prepared UV grid directly onto visibility coordinates.

    This is the shared numerical primitive beneath cube and continuum model
    prediction.  Callers own the spectral representation of ``grid`` and supply
    the mapping from each visibility channel to its corresponding grid plane.

    Parameters
    ----------
    ms_xds : xarray.Dataset
        Measurement-set partition modified in place with model visibilities.
    cgk_1D : numpy.ndarray
        One-dimensional prolate-spheroidal convolution kernel.
    img_xds : xarray.Dataset
        Image geometry supplying direction-cell sizes and unpadded image shape.
    grid : numpy.ndarray
        C-contiguous UV grid with dimensions
        ``(time, spectral_plane, polarization, u, v)``.
    frequency_map : numpy.ndarray
        Integer grid-plane index for every visibility frequency channel.
    ms_data_group_in_name, ms_data_group_out_name : str
        Input and output measurement-set data-group names.
    ms_data_group_out_modified : dict, optional
        Output role-to-variable mapping. Defaults to
        ``{"correlated_data": "VISIBILITY_MODEL"}``.
    overwrite : bool
        Whether an existing output data group may be replaced.
    fft_padding : float
        FFT padding used to derive the UV-grid shape from the image geometry.
    processing_function_threads : int
        Threads supplied to the C++ degridder.
    description : str
        Description stored with a newly registered output data group.

    Returns
    -------
    None
        ``ms_xds`` is modified in place.
    """
    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {"correlated_data": "VISIBILITY_MODEL"}

    output_mapping = copy.deepcopy(ms_data_group_out_modified)
    ms_data_group_in, ms_data_group_out = create_data_groups_in_and_out(
        ms_xds,
        data_group_in_name=ms_data_group_in_name,
        data_group_out_name=ms_data_group_out_name,
        data_group_out_modified=output_mapping,
        overwrite=overwrite,
    )

    frequency_map = np.ascontiguousarray(frequency_map, dtype=np.int64)
    n_chan = ms_xds.sizes["frequency"]
    if frequency_map.shape != (n_chan,):
        raise ValueError(
            "frequency_map must contain one grid-plane index per visibility "
            f"channel; received shape {frequency_map.shape} for {n_chan} channels."
        )

    grid = np.ascontiguousarray(grid)
    if grid.ndim != 5:
        raise ValueError(
            "grid must have dimensions (time, spectral_plane, polarization, u, v)."
        )
    if np.any(frequency_map < 0) or np.any(frequency_map >= grid.shape[1]):
        raise ValueError("frequency_map contains an index outside the UV grid.")

    output_name = ms_data_group_out["correlated_data"]
    if output_name not in ms_xds:
        input_visibility = ms_xds[ms_data_group_in["correlated_data"]]
        ms_xds[output_name] = xr.DataArray(
            np.zeros(input_visibility.shape, dtype=np.complex128),
            dims=input_visibility.dims,
        )
        modify_data_groups_xds(
            ms_xds,
            ms_data_group_out_name,
            ms_data_group_out,
            description=description,
        )

    vis_data = ms_xds[output_name].values
    uvw = np.ascontiguousarray(ms_xds[ms_data_group_in["uvw"]].values)
    frequency_coord = np.ascontiguousarray(ms_xds.frequency.values, dtype=np.float64)
    time_map = np.zeros(ms_xds.sizes["time"], dtype=np.int64)
    pol_map = np.arange(ms_xds.sizes["polarization"], dtype=np.int64)

    from astroviper.processing_functions.imaging.utils.fft_sizing import (
        padded_grid_size,
    )

    n_uv = padded_grid_size([img_xds.sizes["l"], img_xds.sizes["m"]], fft_padding)
    delta_lm = img_xds.xr_img.get_lm_cell_size()

    from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp import (
        prolate_spheroidal_degrid,
    )

    prolate_spheroidal_degrid(
        grid,
        vis_data,
        uvw,
        frequency_coord,
        frequency_map,
        time_map,
        pol_map,
        cgk_1D,
        n_uv,
        delta_lm,
        support=7,
        oversampling=100,
        processing_function_threads=processing_function_threads,
    )
