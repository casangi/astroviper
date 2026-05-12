import numpy as np
import scipy
from scipy import constants
from numba import jit
import numba
import xarray as xr

from astroviper.processing_functions.imaging.gridders.mosaic_grid import mosaic_grid_jit
from astroviper.utils.data_group_tools import (
    create_data_groups_in_and_out,
    modify_data_groups_xds,
)
import copy

def get_visibility_grid_single_field(
    ms_xds: xr.Dataset,
    cgk_1D: np.ndarray,
    img_xds: xr.Dataset,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "model",
    ms_data_group_out_modified: dict = {
        "correlated_data": "VISIBILITY_MODEL",
    },
    img_data_group_in_name: str = "model",
    overwrite: bool = True,
    chan_mode: str = "cube",
    fft_padding: float = 1.2,
    num_threads: int = 1,
):
    """Degrid a UV-domain model onto measurement-set visibility coordinates.

    Samples a model UV grid stored in ``img_xds`` at each visibility's
    ``(u, v)`` coordinate using a separable 1-D convolutional gridding kernel
    (``cgk_1D``), and writes the resulting predicted visibilities into
    ``ms_xds`` under the data variable named by
    ``ms_data_group_out_modified["correlated_data"]`` (default
    ``"VISIBILITY_MODEL"``).  A corresponding output data group is registered
    on ``ms_xds`` via
    :func:`~astroviper.utils.data_group_tools.modify_data_groups_xds`.

    This function is the inverse of
    :func:`~astroviper.processing_functions.imaging.add_visibility_grid.add_visibility_grid_single_field`:
    it predicts visibilities *from* a UV-plane model rather than gridding
    observed visibilities onto a UV plane.  It uses the standard separable
    prolate-spheroidal degridder (:func:`prolate_spheroidal_degrid_jit`) and
    therefore does not require a GCF dataset.

    Parameters
    ----------
    ms_xds : xr.Dataset
        Measurement set dataset.  Must contain the data variables referenced
        by ``ms_data_group_in_name`` (``correlated_data``, ``uvw``,
        ``weight_imaging``) and a ``frequency`` coordinate.  On the first call
        the output visibility data variable is created on ``ms_xds`` by
        allocating an array shaped and dtyped like the input ``correlated_data``.
    cgk_1D : np.ndarray
        Oversampled 1-D prolate spheroidal wave function (PSWF) kernel.
        Shape ``(oversampling * (support // 2 + 1),)``; passed directly to
        :func:`prolate_spheroidal_degrid_jit`.
    img_xds : xr.Dataset
        Image dataset holding the model UV grid.  Must expose an image data
        group under ``img_data_group_in_name`` whose ``"SKY"`` role names a
        data variable shaped ``(time, frequency, polarization, u, v)``, the
        same ``(u, v)`` axis order used by
        :func:`~astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid.prolate_spheroidal_grid_jit`.
        Cell size and image dimensions are read directly from ``img_xds`` via
        ``img_xds.xr_img.get_lm_cell_size()`` and ``img_xds.sizes``.
    ms_data_group_out_name : str, default ``"model"``
        Key under which the output data group is registered in
        ``ms_xds.attrs["data_groups"]``.
    ms_data_group_out_modified : dict, default ``{"correlated_data": "VISIBILITY_MODEL"}``
        Mapping of role keys to the data-variable names written into
        ``ms_xds``.  ``"correlated_data"`` stores the complex predicted
        visibilities.
    img_data_group_in_name : str, default ``"model_visibility_grid"``
        Key of the image input data group in ``img_xds.attrs["data_groups"]``
        whose ``"SKY"`` role names the model UV grid.
    overwrite : bool, default ``True``
        If ``True``, an existing output data group or output data variable is
        silently overwritten.  Defaults to ``True`` because this function is
        typically called repeatedly in an iterative cycle.
    chan_mode : str, default ``"cube"``
        Channel mapping mode.  ``"cube"`` maps each visibility channel to its
        own image channel; ``"continuum"`` sources every visibility channel
        from image channel 0.
    num_threads : int, default ``1``
        Reserved for a future threaded C++ degridder; currently unused by the
        Numba implementation.

    Returns
    -------
    None
        Modifies ``ms_xds`` in place (output data variable and ``data_groups``
        attribute); no return value.

    Notes
    -----
    - Flags must be applied by the caller before invoking this function.  The
      underlying :func:`prolate_spheroidal_degrid_jit` has no flag argument
      and only skips samples whose ``vis_data`` value is ``NaN``.  Newly
      allocated output arrays are zero-initialised (not ``NaN``), so every
      visibility whose ``uvw`` is finite and whose support falls inside the
      grid will receive a prediction.
    - Time mapping is not currently implemented: every visibility time index
      is mapped to image time index 0.

    See Also
    --------
    astroviper.processing_functions.imaging.add_visibility_grid.add_visibility_grid_single_field :
        Forward (gridding) counterpart.
    prolate_spheroidal_degrid_jit :
        Numba-compiled standard separable degridding kernel.
    """
    _ms_data_group_out_modified = copy.deepcopy(ms_data_group_out_modified)

    # Resolve the image input and output data groups, guarding against
    # accidental overwrites according to the overwrite flag.
    ms_data_group_in, ms_data_group_out = create_data_groups_in_and_out(
        ms_xds,
        data_group_in_name=ms_data_group_in_name,
        data_group_out_name=ms_data_group_out_name,
        data_group_out_modified=_ms_data_group_out_modified,
        overwrite=overwrite,
    )
    
    model_name = img_xds.attrs["data_groups"][img_data_group_in_name]["visibility"]

    n_chan = img_xds.sizes["frequency"]
    if chan_mode == "cube":
        n_imag_chan = n_chan
        frequency_map = (np.arange(0, n_chan)).astype(int)
    else:  # continuum
        n_imag_chan = 1  # Single continuum image collapsed across all channels.
        frequency_map = (np.zeros(n_chan)).astype(int)

    # Time Map #Currently not implemented.
    n_imag_time = 1
    n_time = ms_xds.sizes["time"]
    time_map = (np.zeros(n_time)).astype(int)

    n_imag_pol = img_xds.sizes["polarization"]
    pol_map = (np.arange(0, n_imag_pol)).astype(int)

    # NBNB To do: create UV  and allow for padding
    n_uv = (fft_padding * np.array([img_xds.sizes["l"], img_xds.sizes["m"]])).astype(
        int
    )
    # n_uv = np.array([300,300])

    delta_lm = img_xds.xr_img.get_lm_cell_size()
    print("!!!!!!!!get_visibility_grid:", "n_uv:", n_uv, "delta_lm:", delta_lm)

    # Initialise the output visibility array on the first call; subsequent
    # calls reuse (and overwrite) the existing data variable in place.
    if ms_data_group_out["correlated_data"] not in ms_xds:
        ms_xds[ms_data_group_out["correlated_data"]] = xr.DataArray(
            np.zeros((ms_xds.sizes["time"], ms_xds.sizes["baseline_id"], ms_xds.sizes["frequency"], ms_xds.sizes["polarization"]), dtype=np.complex64),
            dims=["time", "baseline_id", "frequency", "polarization"])
        
        modify_data_groups_xds(
            ms_xds,
            ms_data_group_out_name,
            ms_data_group_out,
            description="Degridded visibilities from img_xds " +  img_data_group_in_name + " to ms_xds " + ms_data_group_out_name + " with get_visibility_grid_single_field.",
        )

    grid = img_xds[model_name].values
    vis_data = ms_xds[ms_data_group_out["correlated_data"]].values
    uvw = ms_xds[ms_data_group_in["uvw"]].values
    frequency_coord = ms_xds.frequency.values

    from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_degrid import (
        prolate_spheroidal_degrid_jit,
    )

    cpp_gridder = True
    if cpp_gridder:
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
            num_threads=num_threads,
        )
        
        print("222222!@@@@@@@@@@@@@@@@@@@@ vis_data:", np.sum(np.abs(vis_data)))    
    else:
        prolate_spheroidal_degrid_jit(
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
        )
        print("!@@@@@@@@@@@@@@@@@@@@ vis_data:", np.sum(np.abs(vis_data)))    