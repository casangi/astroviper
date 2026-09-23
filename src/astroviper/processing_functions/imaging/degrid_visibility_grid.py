"""Shared prolate spheroidal gridder visibility prediction primitive."""

import copy

import numpy as np
import xarray as xr

from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
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
    description: str = "Degridded prolate spheroidal gridder model visibilities.",
):
    """Sample a prepared UV grid directly onto visibility coordinates.

    This is the shared numerical primitive beneath cube and continuum model
    prediction with the prolate spheroidal gridder (the standard gridder in
    CASA). Callers own the spectral representation of ``grid`` and supply the
    mapping from each visibility channel to its corresponding grid plane.

    Parameters
    ----------
    ms_xds : xarray.Dataset
        Measurement-set partition modified in place with model visibilities.
        Must expose the ``correlated_data`` and ``uvw`` roles of
        ``ms_data_group_in_name`` and a ``frequency`` coordinate. When allocating
        model visibilities for a cached-grid cycle, the observed visibility
        variable may be absent if a registered ``weight_imaging`` variable is
        available as the shape and dimension template.
    cgk_1D : numpy.ndarray
        One-dimensional prolate spheroidal convolution kernel, shape
        ``(oversampling * (support // 2 + 1),)``.
    img_xds : xarray.Dataset
        Image geometry supplying direction-cell sizes and unpadded image shape.
    grid : numpy.ndarray
        C-contiguous ``complex64`` or ``complex128`` UV grid with dimensions
        ``(time, spectral_plane, polarization, u, v)``. The polarization axis
        must match ``ms_xds`` (an identity polarization map is used) and only
        time plane ``0`` is sampled (time mapping is not implemented).
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
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    description : str
        Description stored with a newly registered output data group.

    Returns
    -------
    None
        ``ms_xds`` is modified in place.

    Raises
    ------
    ValueError
        If ``frequency_map`` does not hold one index per visibility channel,
        if ``grid`` is not five-dimensional, if a ``frequency_map`` entry
        does not name a grid plane, or if the polarization axes of ``grid``
        and ``ms_xds`` differ in length.
    KeyError
        If model allocation is needed and neither observed visibilities nor a
        registered imaging-weight template is available.
    AssertionError
        If the output data group exists and ``overwrite`` is ``False`` (raised
        by :func:`~astroviper.utils.data_group_tools.create_data_groups_in_and_out`).

    Notes
    -----
    ``processing_function_threads <= 0`` lets the C++ degridder fall back to
    the hardware concurrency; ``1`` runs serially.
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

    # The C++ binding rejects a non-contiguous grid; make it contiguous here
    # (a no-op, no copy, for the contiguous grids the imaging loop produces).
    grid = np.ascontiguousarray(grid)
    if grid.ndim != 5:
        raise ValueError(
            "grid must have dimensions (time, spectral_plane, polarization, u, v)."
        )
    if np.any(frequency_map < 0) or np.any(frequency_map >= grid.shape[1]):
        raise ValueError("frequency_map contains an index outside the UV grid.")

    # The C++ kernel does not bounds-check the polarization map, so enforce
    # the identity-map assumption (visibility and grid polarization axes agree)
    # before handing over raw pointers.
    n_pol = ms_xds.sizes["polarization"]
    if grid.shape[2] != n_pol:
        raise ValueError(
            "grid polarization axis must match the measurement set; received "
            f"{grid.shape[2]} grid polarizations for {n_pol} visibility "
            "polarizations."
        )

    # Initialise the output visibility array on the first call; subsequent
    # calls reuse (and overwrite) the existing data variable in place.
    # The model visibilities are kept double precision (complex128): the
    # visibilities stay double even when the image-domain model grid is single
    # precision, and the residual = observed - model is formed in double
    # precision. The C++ degridder widens each (possibly complex64) model-grid
    # cell to complex128 for the accumulation, so it can write complex128 here.
    output_name = ms_data_group_out["correlated_data"]
    if output_name not in ms_xds:
        input_visibility_name = ms_data_group_in["correlated_data"]
        if input_visibility_name in ms_xds:
            input_visibility = ms_xds[input_visibility_name]
        else:
            # Cached-grid continuum cycles deliberately avoid loading the
            # observed visibility values. Imaging weights have the identical
            # visibility layout and therefore provide a zero-allocation shape
            # template for the degridded model array.
            template_name = ms_data_group_in.get("weight_imaging")
            if template_name is None or template_name not in ms_xds:
                raise KeyError(
                    f"Neither input visibility {input_visibility_name!r} nor a "
                    "registered imaging-weight template is available for model "
                    "visibility allocation."
                )
            input_visibility = ms_xds[template_name]
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
    # Time mapping is not implemented: every visibility time samples plane 0.
    time_map = np.zeros(ms_xds.sizes["time"], dtype=np.int64)
    pol_map = np.arange(n_pol, dtype=np.int64)

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
