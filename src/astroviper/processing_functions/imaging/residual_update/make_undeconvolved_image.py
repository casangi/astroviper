"""Gridding of the (residual) visibilities into the undeconvolved image grid."""

import time

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def make_undeconvolved_image_single_field(
    ps_xdt,
    img_xds,
    image_params,
    gcf_xds,
    model_exists,
    ms_data_group_in_name="corrected",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
    gridder="anti_alias",
):
    """Grid the (residual) visibilities of every measurement set onto the uv grid.

    Drops auto-correlations and grids the visibilities (``VISIBILITY`` /
    ``VISIBILITY_NORMALIZATION``).  Together with the inverse FFT
    (:func:`~astroviper.processing_functions.imaging.normalization.normalize_image.ifft_norm_img_xds`)
    this produces the undeconvolved (dirty/residual) image.  A gridder that
    accumulates into the image itself (``"direct_fourier_transform"``) makes
    the ``SKY_RESIDUAL`` and its ``VISIBILITY_NORMALIZATION`` instead, after
    deleting an existing residual image, and its ``uv_to_image`` divides the
    one by the other.  In the single-field
    imager the UV-sampling grid that becomes the PSF is built once, separately,
    by
    :func:`~astroviper.processing_functions.imaging.imaging_setup.make_point_spread_function.make_point_spread_function_single_field`.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set; auto-correlations are dropped in place before gridding.
    img_xds : xarray.Dataset
        Image dataset that the uv grid is written into.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``). The labels must be a
        nonempty one-dimensional vector. An xarray DataArray must have exactly
        the dimension ``polarization``; other dimension names are rejected.
        ``time_coords`` accepts XRADIO image-factory time formats: numeric MJD
        (UTC) days, numeric strings, ISO time strings, datetime64 values, Python
        datetime objects, or Astropy Time, as scalars or one-dimensional sequences.
        Times must be finite and nonempty; validation preserves the supplied values.
        Distributed single-field imagers require exactly one image time plane and
        an explicit ``fft_padding``. Designed kernels, including the default,
        require padding at least 1.05; CASA's fixed kernel permits 1.0.
    gcf_xds : xarray.Dataset
        The ``gcf_xds`` of the gridding kernel, from
        :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.make_gridding_kernel`,
        or from
        :func:`~astroviper.processing_functions.imaging.gridding_convolution_functions.anti_alias_term.as_gridding_kernel`
        for a bare table of CASA's prolate spheroidal kernel of support 7.
    model_exists : bool
        Accepted for call-site compatibility with the imaging cycle loop;
        currently unused (the UV-sampling/PSF grid is built once in
        ``make_point_spread_function_single_field``).
    ms_data_group_in_name : str, optional
        Measurement-set data group to grid from.  Default ``"corrected"``.
    image_data_group_out_name : str, optional
        Image data group the uv grid is written under.  Default ``"residual"``.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels. A distributed application resolves ``0`` (auto) once, before
        its checks, to the CPUs per Dask worker thread of the host with the
        fewest, at least 1 (1 where these are not known, as under MPI), and
        warns when a host's worker threads times the count exceed its CPUs
        (see
        :func:`~astroviper.utils.data_partitioning.resolve_processing_function_threads`).
        Node tasks and processing functions do not resolve ``0``; those that
        check the count refuse it.
    complex_dtype : numpy.dtype, optional
        Complex precision of the gridded visibility grid (``complex64`` for a
        single-precision image, ``complex128`` otherwise).  Defaults to
        ``numpy.complex128``.
    gridder : str, optional
        Registry name of the gridder (a key of
        :data:`~astroviper.processing_functions.imaging.gridders.gridder_registry.GRIDDERS`):
        ``"anti_alias"`` (the default) or ``"direct_fourier_transform"``,
        which uses neither ``gcf_xds`` nor ``image_params["fft_padding"]``.

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
    from astroviper.processing_functions.imaging.gridders.gridder_registry import (
        GRIDDERS,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128

    if "sky" in GRIDDERS[gridder].grid_roles:
        # The gridder accumulates into the residual image itself.
        data_group = img_xds.attrs["data_groups"].get(image_data_group_out_name, {})
        image_name = data_group.get("sky", GRIDDERS[gridder].grid_roles["sky"])
        if image_name in img_xds:
            img_xds.xr_img.delete_data_variables(variables=[image_name])

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    T_vis_grid = 0.0

    for ms_xdt in ps_xdt.values():
        T_start_vis_mask = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask = T_vis_mask + time.time() - T_start_vis_mask

        T_start_vis = time.time()
        add_visibility_grid_single_field(
            ms_xdt,
            gcf_xds,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified=dict(GRIDDERS[gridder].grid_roles),
            overwrite=True,
            chan_mode="cube",
            fft_padding=(
                image_params["fft_padding"]
                if "anti_alias" in GRIDDERS[gridder].gcf_terms
                else None
            ),
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
            gridder=gridder,
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
