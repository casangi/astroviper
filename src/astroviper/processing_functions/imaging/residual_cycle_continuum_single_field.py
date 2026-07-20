from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def residual_cycle_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    is_n_iter_0,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    image_data_group_in_name="model",
    image_data_group_out_name="residual",
    last_residual_cycle=False,
):
    """Create continuum residual Taylor products for the initial dirty-image cycle.

    This implementation supports the current ``niter=0`` path. It grids the
    observed visibilities into MT-MFS residual Taylor uv grids, inverse-transforms
    them into ``SKY_RESIDUAL``, and transforms the result back to Stokes.

    Model degridding and residual-visibility formation for later major cycles are
    intentionally not implemented yet.
    """
    import time

    import numpy as np
    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.fft_normalize_prolate_spheriodal_gridder import (
        ifft_norm_img_xds,
    )
    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.make_undeconvolved_image_continuum import (
        make_undeconvolved_image_continuum_single_field,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    nterms = int(image_params.get("nterms", 2))
    reference_frequency = float(image_params["reference_frequency_hz"])

    if nterms < 1:
        raise ValueError("image_params['nterms'] must be at least 1.")

    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "image_params['reference_frequency'] must be a positive finite "
            "frequency in Hz."
        )

    complex_dtype = np.complex64 if single_precision_image else np.complex128

    ps_data_group_name = processing_set_data_group_name

    start = time.time()
    cgk_1D = create_prolate_spheroidal_kernel_1D(100, 7)
    T_gcf = time.time() - start

    T_transform_pol = 0.0
    T_fft_degrid = 0.0
    T_degrid = 0.0
    T_residual_vis = 0.0

    start = time.time()
    img_xds = transform_polarization_basis(
        img_xds,
        new_polarization_basis=instrument_polarization_basis,
        overwrite=True,
    )
    T_transform_pol += time.time() - start

    start = time.time()
    (
        img_xds,
        make_undeconvolved_image_return_df,
    ) = make_undeconvolved_image_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        cgk_1D,
        nterms=nterms,
        reference_frequency=reference_frequency,
        ms_data_group_in_name=ps_data_group_name,
        image_data_group_out_name=image_data_group_out_name,
        processing_function_threads=processing_function_threads,
        complex_dtype=complex_dtype,
    )
    T_grid = time.time() - start

    start = time.time()
    img_xds = ifft_norm_img_xds(
        img_xds,
        image_params=image_params,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_group_out_modified={
            "sky": "SKY_RESIDUAL",
        },
        image_data_variables_keep=image_data_variables_keep,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )
    T_fft_grid = time.time() - start

    if "SKY_RESIDUAL" not in img_xds:
        raise RuntimeError("ifft_norm_img_xds did not create SKY_RESIDUAL.")

    if "taylor_term" not in img_xds["SKY_RESIDUAL"].dims:
        raise RuntimeError(
            "Continuum inverse FFT did not preserve the taylor_term dimension."
        )

    img_xds["SKY_RESIDUAL"].attrs.update(
        {
            "description": "Continuum residual Taylor products.",
            "nterms": nterms,
            "reference_frequency": reference_frequency,
            "placeholder": False,
        }
    )

    start = time.time()
    img_xds = transform_polarization_basis(
        img_xds,
        new_polarization_basis="stokes",
        overwrite=True,
    )
    T_transform_pol += time.time() - start

    logger.debug(
        "Created continuum residual Taylor products with dimensions "
        f"{img_xds['SKY_RESIDUAL'].dims} and shape "
        f"{img_xds['SKY_RESIDUAL'].shape}."
    )

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_degrid": [T_degrid],
            "T_fft_degrid": [T_fft_degrid],
            "T_residual_vis": [T_residual_vis],
            "T_grid": [T_grid],
            "T_fft_grid": [T_fft_grid],
            "T_transform_pol": [T_transform_pol],
            "nterms": [nterms],
            "is_n_iter_0": [bool(is_n_iter_0)],
            "last_residual_cycle": [bool(last_residual_cycle)],
        }
    )

    return_df = pd.concat(
        [return_df, make_undeconvolved_image_return_df],
        axis=1,
    )

    return img_xds, return_df
