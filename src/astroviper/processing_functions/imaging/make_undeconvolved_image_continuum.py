"""Gridding of residual visibilities into continuum Taylor uv grids."""

import time

from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def make_undeconvolved_image_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    cgk_1D,
    n_taylor_terms=None,
    reference_frequency=None,
    ms_data_group_in_name="corrected",
    image_data_group_out_name="residual",
    processing_function_threads=1,
    complex_dtype=None,
    *,
    input_checked: bool = False,
):
    """Grid residual visibilities into MT-MFS residual Taylor uv grids.

    For Taylor order ``t``, the gridder accumulates visibility contributions
    weighted by

    ``((frequency - reference_frequency) / reference_frequency)**t``.

    The resulting variables are expected to have dimensions

    ``VISIBILITY(time, taylor_term, polarization, u, v)``

    and

    ``VISIBILITY_NORMALIZATION(time, taylor_term, polarization)``.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing Set partition containing visibilities and imaging weights.
    img_xds : xarray.Dataset
        Continuum image dataset updated with residual Taylor UV grids.
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
    cgk_1D : numpy.ndarray
        The gridding kernel
        (:class:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel.GriddingKernel`),
        or the bare table of the default prolate spheroidal kernel of support 7.
    n_taylor_terms : int or None, optional
        Number of residual Taylor orders; defaults to ``image_params``.
    reference_frequency : float or None, optional
        Taylor reference frequency in Hz; defaults to ``image_params``.
    ms_data_group_in_name : str, optional
        Measurement Set data group registering residual visibilities and weights.
    image_data_group_out_name : str, optional
        Output image data-group name.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    complex_dtype : numpy.dtype or None, optional
        Complex dtype of the residual UV grids.

    input_checked : bool, optional
        Skip configuration checks already performed by the calling layer.
        Runtime checks on newly created data always remain active. For graph nodes
        using an input_params dictionary, its input_checked entry also carries this
        flag. Set it only for an unchanged configuration in the same imaging mode.

    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset containing the accumulated residual Taylor UV grids.
    return_df : pandas.DataFrame
        One-row timing dataframe for filtering and visibility gridding.
    """

    if not input_checked:
        from astroviper.processing_functions.imaging.config_check import (
            CONTINUUM_SINGLE_FIELD,
            check_residual_update,
        )

        check_residual_update(locals(), mode=CONTINUUM_SINGLE_FIELD)

    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.add_visibility_grid_continuum import (
        add_visibility_grid_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.utils import drop_auto_correlations

    if complex_dtype is None:
        complex_dtype = np.complex128

    if n_taylor_terms is None:
        n_taylor_terms = int(image_params.get("n_taylor_terms", 2))
    else:
        n_taylor_terms = int(n_taylor_terms)

    if reference_frequency is None:
        reference_frequency = float(image_params["reference_frequency"])
    else:
        reference_frequency = float(reference_frequency)

    T_vis_mask = 0.0
    T_uv_sampling_grid = 0.0
    T_vis_grid = 0.0

    for _, ms_xdt in ps_xdt.items():
        start = time.time()
        drop_auto_correlations(ms_xdt)
        T_vis_mask += time.time() - start

        start = time.time()
        add_visibility_grid_continuum_single_field(
            ms_xdt,
            cgk_1D,
            img_xds,
            n_taylor_terms=n_taylor_terms,
            reference_frequency=reference_frequency,
            ms_data_group_in_name=ms_data_group_in_name,
            image_data_group_in_name=image_data_group_out_name,
            image_data_group_out_name=image_data_group_out_name,
            image_data_group_out_modified={
                "visibility": "VISIBILITY",
                "visibility_normalization": "VISIBILITY_NORMALIZATION",
            },
            overwrite=True,
            fft_padding=image_params["fft_padding"],
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
            input_checked=True,
        )
        T_vis_grid += time.time() - start

    return_df = pd.DataFrame(
        {
            "T_vis_mask": [T_vis_mask],
            "T_uv_sampling_grid": [T_uv_sampling_grid],
            "T_vis_grid": [T_vis_grid],
        }
    )

    return img_xds, return_df
