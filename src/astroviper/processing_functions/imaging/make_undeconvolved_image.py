"""Gridding of the (residual) visibilities into the undeconvolved image grid."""

import time

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def make_undeconvolved_image_single_field(
    ps_xdt,
    img_xds,
    image_params,
    cgk_1D,
    is_n_iter_0,
    ms_data_group_in_name="corrected",
    image_data_group_out_name="residual",
    num_threads=1,
    complex_dtype=None,
):
    """Grid the (residual) visibilities of every measurement set onto the uv grid.

    Drops auto-correlations and grids the visibilities (``VISIBILITY`` /
    ``VISIBILITY_NORMALIZATION``).  Together with the inverse FFT
    (:func:`~astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder.ifft_norm_img_xds`)
    this produces the undeconvolved (dirty/residual) image.  In the single-field
    imager the UV-sampling grid that becomes the PSF is built once, separately,
    by
    :func:`~astroviper.processing_functions.imaging.make_point_spread_function.make_point_spread_function_single_field`.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; auto-correlations are dropped in place before gridding.
    img_xds : xarray.Dataset
        Image dataset that the uv grid is written into.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    cgk_1D : numpy.ndarray
        1-D prolate-spheroidal gridding convolution kernel.
    is_n_iter_0 : bool
        Accepted for call-site compatibility with the major-cycle loop;
        currently unused (the UV-sampling/PSF grid is built once in
        ``make_point_spread_function_single_field``).
    ms_data_group_in_name : str, optional
        Measurement-set data group to grid from.  Default ``"corrected"``.
    image_data_group_out_name : str, optional
        Image data group the uv grid is written under.  Default ``"residual"``.
    num_threads : int, optional
        Threads handed to the gridding kernel.  Default ``1``.
    complex_dtype : numpy.dtype, optional
        Complex precision of the gridded visibility grid (``complex64`` for a
        single-precision image, ``complex128`` otherwise).  Defaults to
        ``numpy.complex128``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with the visibility uv grid added.
    return_df : pandas.DataFrame
        One-row timing frame with ``T_vis_mask``, ``T_uv_sampling_grid`` and
        ``T_vis_grid``.
    """
    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.add_visibility_grid import (
        add_visibility_grid_single_field,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    T_vis_grid = 0.0

    for ms_name, ms_xdt in ps_xdt.items():
        T_start_vis_mask = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask = T_vis_mask + time.time() - T_start_vis_mask

        T_start_vis = time.time()
        add_visibility_grid_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified={
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
            },
            overwrite=True,
            chan_mode="cube",
            fft_padding=image_params["fft_padding"],
            num_threads=num_threads,
            complex_dtype=complex_dtype,
        )
        T_vis_grid = T_vis_grid + time.time() - T_start_vis

    return_df = pd.DataFrame(
        {
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
            "T_vis_grid": [T_vis_grid],
        }
    )

    return img_xds, return_df
