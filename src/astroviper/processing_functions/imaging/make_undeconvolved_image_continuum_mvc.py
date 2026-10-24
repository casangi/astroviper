"""Gridding of residual visibilities into an MVC frequency cube."""

import time

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def make_undeconvolved_image_mvc_single_field(
    ps_xdt,
    img_xds,
    image_params,
    cgk_1D,
    ms_data_group_in_name="corrected",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
):
    """Grid residual visibilities into a frequency-resolved MVC UV cube.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing Set partition containing visibilities and imaging weights.
    img_xds : xarray.Dataset
        MVC image dataset updated with channelized residual UV grids.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
        Continuum imaging also requires ``frequency_coords``,
        ``n_taylor_terms`` (CASA ``nterms``), and ``reference_frequency`` in Hz;
        ``reference_frequency_hz`` is a deprecated alias. ``spectral_reference``
        sets the spectral frame (default ``"lsrk"``). ``primary_beam_model``
        selects ``auto``, ``airy`` or ``casa_airy``; ``list_dish_diameters`` and
        ``list_blockage_diameters`` are arrays in metres,
        ``primary_beam_max_radius_1ghz`` is a radius in radians at 1 GHz, and
        ``primary_beam_ipower`` is the exponent applied to the voltage pattern
        (default 2, the power beam). These primary-beam keys override the inferred
        beam prescription.
    cgk_1D : GriddingKernel or numpy.ndarray
        The gridding kernel
        (:class:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel.GriddingKernel`),
        or the bare table of the default prolate spheroidal kernel of support 7.
    ms_data_group_in_name : str, optional
        Measurement Set data group registering residual visibilities and weights.
    image_data_group_out_name : str, optional
        Output image data-group name.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    complex_dtype : numpy.dtype or None, optional
        Complex dtype of the channelized residual UV grids.

    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset containing the accumulated MVC residual UV cube.
    return_df : pandas.DataFrame
        One-row timing dataframe for filtering and visibility gridding.
    """
    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.add_visibility_grid_continuum_mvc import (
        add_visibility_grid_mvc_single_field,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128

    T_vis_mask = 0.0
    T_vis_grid = 0.0

    for _, ms_xdt in ps_xdt.items():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_visibility_grid_mvc_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified={
                "visibility": "VISIBILITY",
                "visibility_normalization": ("VISIBILITY_NORMALIZATION"),
            },
            overwrite=True,
            fft_padding=image_params["fft_padding"],
            processing_function_threads=(processing_function_threads),
            complex_dtype=complex_dtype,
        )
        T_vis_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_vis_mask": [T_vis_mask],
            "T_vis_grid": [T_vis_grid],
        }
    )

    return img_xds, return_df
