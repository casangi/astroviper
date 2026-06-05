"""Point-spread-function creation for the single-field cube imager."""


def _drop_auto_correlations(ms_xdt):
    """Drop auto-correlation baselines from a measurement set in place.

    ``where(mask, drop=True)`` indexes ``baseline_id`` with an integer array,
    which leaves the (numpy-backed) data variables as transposed,
    non-C-contiguous views.  Downstream C++ kernels (e.g. the degridder in
    :func:`~astroviper.processing_functions.imaging.get_visibility_grid.get_visibility_grid_single_field`,
    reused across major cycles) require C-contiguous input, so C-order is
    restored before storing the dataset back.

    Idempotent: re-running on already-masked data keeps every baseline and is a
    no-op, which is why the setup and each residual cycle can both call it.

    Parameters
    ----------
    ms_xdt : xarray.DataTree
        Measurement-set node whose ``.ds`` is filtered in place.
    """
    import numpy as np

    # Keep only cross-correlations (antenna1 != antenna2).
    mask = ms_xdt["baseline_antenna1_name"] != ms_xdt["baseline_antenna2_name"]
    masked_ds = ms_xdt.ds.where(mask, drop=True)
    for var_name, var in masked_ds.data_vars.items():
        data = var.data
        if isinstance(data, np.ndarray) and not data.flags["C_CONTIGUOUS"]:
            masked_ds[var_name].data = np.ascontiguousarray(data)
    ms_xdt.ds = masked_ds


def make_single_field_point_spread_function(
    ps_xdt,
    img_xds,
    image_params,
    ms_data_group_in_name="base",
    image_data_group_name="residual",
    image_data_variables_keep=None,
    gcf_oversampling=100,
    gcf_support=7,
    num_threads=1,
    fft_backend="pyfftw",
    complex_dtype=None,
):
    """Build the point spread function for a single-field image chunk.

    Grids the UV-sampling function of every measurement set in ``ps_xdt`` onto
    the ``UV_SAMPLING`` grid and inverse-transforms it (with prolate-spheroidal
    normalization) to the ``POINT_SPREAD_FUNCTION`` image.  Auto-correlations are
    dropped in place before gridding.  The resulting PSF is produced in whatever
    polarization basis ``img_xds`` currently carries (the correlation basis for
    the single-field imager); it is stamped with
    ``type="point_spread_function"`` by the inverse transform so a later
    :func:`~astroviper.processing_functions.image_analysis.transform_polarization_basis.transform_polarization_basis`
    leaves it untouched.

    This is the once-per-chunk PSF builder; the per-cycle visibility grid is made
    separately by
    :func:`~astroviper.processing_functions.imaging.residual_cycle.make_uv_images_single_field`.
    The Gaussian beam fit of the PSF is performed afterwards by
    :func:`~astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit.point_spread_function_gaussian_fit`.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing set whose measurement sets are gridded.  Auto-correlations
        are dropped in place.
    img_xds : xarray.Dataset
        Image dataset to populate.  The data group ``image_data_group_name``
        must already exist.  Modified in place.
    image_params : dict
        Image geometry; must contain ``"fft_padding"`` (and the usual
        ``image_size`` / ``cell_size`` used by the gridders and FFT).
    ms_data_group_in_name : str, optional
        Measurement-set data group that supplies the weights/uvw used for the
        UV-sampling grid.  Default ``"base"``.
    image_data_group_name : str, optional
        Image data group that the ``UV_SAMPLING`` and ``POINT_SPREAD_FUNCTION``
        variables are registered under.  Default ``"residual"``.
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk; forwarded to
        :func:`~astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder.ifft_norm_img_xds`.
        Default ``[]``.
    gcf_oversampling : int, optional
        Oversampling of the prolate-spheroidal gridding convolution kernel.
        Default ``100``.
    gcf_support : int, optional
        Support (in pixels) of the gridding convolution kernel.  Default ``7``.
    num_threads : int, optional
        Number of threads handed to the gridding and FFT kernels.  Default ``1``.
    fft_backend : str, optional
        FFT backend forwarded to the inverse transform.  Default ``"pyfftw"``.
    complex_dtype : numpy.dtype, optional
        Complex precision of the gridded data.  Defaults to
        ``numpy.complex128``.

    Returns
    -------
    img_xds : xarray.Dataset
        The input dataset with ``UV_SAMPLING`` and ``POINT_SPREAD_FUNCTION``
        added.
    return_df : pandas.DataFrame
        One-row timing frame with the ``T_gcf``, ``T_vis_mask``,
        ``T_uv_sampling_grid`` and ``T_fft_norm`` columns.

    See Also
    --------
    astroviper.processing_functions.imaging.primary_beam.make_primary_beam.make_single_field_primary_beam
    """
    import time
    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.add_uv_sampling_grid import (
        add_uv_sampling_grid_single_field,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        ifft_norm_img_xds,
    )

    if complex_dtype is None:
        complex_dtype = np.complex128
    if image_data_variables_keep is None:
        image_data_variables_keep = []

    T_start_gcf = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(gcf_oversampling, gcf_support)
    T_gcf = time.time() - T_start_gcf

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    for ms_name, ms_xdt in ps_xdt.items():
        T_start_vis_mask = time.time()
        _drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - T_start_vis_mask

        T_start_uv = time.time()
        add_uv_sampling_grid_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            img_data_group_in_name=image_data_group_name,
            img_data_group_out_name=image_data_group_name,
            img_data_group_out_modified={
                "uv_sampling": "UV_SAMPLING",
                "uv_sampling_normalization": "UV_SAMPLING_NORMALIZATION",
            },
            overwrite=True,
            chan_mode="cube",
            fft_padding=image_params["fft_padding"],
            num_threads=num_threads,
        )  # Will become the PSF.
        T_uv_sampling_grid += time.time() - T_start_uv

    T_start_fft_norm = time.time()
    img_xds = ifft_norm_img_xds(
        img_xds,
        image_params=image_params,
        image_data_group_in_name=image_data_group_name,
        image_data_group_out_name=image_data_group_name,
        image_data_group_out_modified={
            "point_spread_function": "POINT_SPREAD_FUNCTION",
        },
        image_data_variables_keep=image_data_variables_keep,
        num_threads=num_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )
    T_fft_norm = time.time() - T_start_fft_norm

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
            "T_fft_norm": [T_fft_norm],
        }
    )

    return img_xds, return_df
