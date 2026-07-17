from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def make_point_spread_function_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    nterms=2,
    reference_frequency=None,
    ms_data_group_in_name="corrected",
    image_data_group_in_name="residual",
    image_data_group_out_name="residual",
    image_data_variables_keep=None,
    processing_function_threads=1,
    fft_backend="pyfftw",
    complex_dtype=None,
):
    """Create zero-valued continuum Taylor PSF terms for testing.

    This is a temporary scaffolding implementation of the continuum MT-MFS
    point-spread-function calculation. It does not yet grid the weighted
    UV-sampling function or perform an inverse FFT.

    Instead, it creates a zero-valued ``POINT_SPREAD_FUNCTION`` array with
    dimensions

    ``(time, psf_taylor_order, polarization, l, m)``.

    For ``nterms=N``, MT-MFS requires ``2*N - 1`` PSF/Hessian Taylor orders.
    Thus, for ``nterms=2``, this function creates three zero-valued terms:

    ``H_0``, ``H_1``, and ``H_2``.

    The function retains the intended final interface so that the zero-filled
    placeholder can later be replaced by the real weighted gridding
    implementation without changing its callers.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Visibility data for this frequency chunk. Currently unused.
    img_xds : xarray.Dataset
        Image dataset to which the zero-valued Taylor PSFs are added.
    image_params : dict
        Image geometry and continuum parameters. Currently used only for
        metadata.
    nterms : int, optional
        Number of sky-model Taylor terms. The number of PSF Taylor orders is
        ``2*nterms - 1``.
    reference_frequency : float, optional
        Common MT-MFS Taylor expansion frequency in Hz.
    ms_data_group_in_name : str, optional
        Input processing-set data group. Currently unused.
    image_data_group_in_name : str, optional
        Input image data group. Retained for interface compatibility.
    image_data_group_out_name : str, optional
        Output image data group under which the PSF is registered.
    image_data_variables_keep : list of str, optional
        Retained for interface compatibility.
    processing_function_threads : int, optional
        Retained for interface compatibility.
    fft_backend : str, optional
        Retained for interface compatibility.
    complex_dtype : numpy dtype, optional
        Complex grid dtype that the eventual implementation will use. The
        zero-valued image-domain PSF is real-valued; its float dtype is inferred
        from this argument when supplied.

    Returns
    -------
    img_xds : xarray.Dataset
        Dataset containing zero-valued Taylor PSF terms.
    return_df : pandas.DataFrame
        One-row timing dataframe compatible with the existing PSF setup timing
        accumulation.
    """
    import time

    import numpy as np
    import pandas as pd
    import toolviper.utils.logger as logger
    import xarray as xr

    from astroviper.utils.data_group_tools import modify_data_groups_xds

    start_total = time.time()

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    nterms = int(nterms)

    if nterms < 1:
        raise ValueError("nterms must be at least 1.")

    n_psf_taylor_terms = 2 * nterms - 1

    if complex_dtype is None:
        float_dtype = np.float32
    else:
        complex_dtype = np.dtype(complex_dtype)

        if complex_dtype == np.dtype(np.complex64):
            float_dtype = np.float32
        elif complex_dtype == np.dtype(np.complex128):
            float_dtype = np.float64
        else:
            raise TypeError(
                "complex_dtype must be complex64 or complex128 when supplied; "
                f"received {complex_dtype}."
            )

    required_dimensions = ("time", "polarization", "l", "m")
    missing_dimensions = [
        dimension for dimension in required_dimensions if dimension not in img_xds.sizes
    ]

    if missing_dimensions:
        raise ValueError(
            "img_xds is missing dimensions required for continuum Taylor PSFs: "
            f"{missing_dimensions}. Available dimensions are "
            f"{list(img_xds.sizes)}."
        )

    # Add or replace the PSF Taylor-order coordinate.
    img_xds = img_xds.assign_coords(
        psf_taylor_order=np.arange(
            n_psf_taylor_terms,
            dtype=np.int32,
        )
    )

    psf_dims = (
        "time",
        "psf_taylor_order",
        "polarization",
        "l",
        "m",
    )

    psf_shape = (
        img_xds.sizes["time"],
        n_psf_taylor_terms,
        img_xds.sizes["polarization"],
        img_xds.sizes["l"],
        img_xds.sizes["m"],
    )

    psf_coords = {
        dimension: img_xds.coords[dimension]
        for dimension in psf_dims
        if dimension in img_xds.coords
    }

    psf_name = "POINT_SPREAD_FUNCTION"

    img_xds[psf_name] = xr.DataArray(
        np.zeros(psf_shape, dtype=float_dtype),
        dims=psf_dims,
        coords=psf_coords,
        attrs={
            "description": (
                "Temporary zero-valued continuum PSF/Hessian Taylor terms."
            ),
            "units": "dimensionless",
            "nterms": nterms,
            "n_psf_taylor_terms": n_psf_taylor_terms,
            "reference_frequency_hz": (
                None if reference_frequency is None else float(reference_frequency)
            ),
            "placeholder": True,
        },
    )

    # Ensure that the requested output data group exists.
    data_groups = img_xds.attrs.setdefault("data_groups", {})

    if image_data_group_out_name not in data_groups:
        img_xds.attrs["type"] = "image_dataset"
        img_xds = img_xds.xr_img.add_data_group(
            new_data_group_name=image_data_group_out_name,
            new_data_group={
                "description": ("Continuum Taylor PSF and residual products."),
                "date": "2026",
            },
        )

    # Register the Taylor PSF under the same logical role used by the existing
    # cube imaging data groups.
    output_data_group = dict(img_xds.attrs["data_groups"][image_data_group_out_name])
    output_data_group["point_spread_function"] = psf_name

    modify_data_groups_xds(
        img_xds,
        data_group_out_name=image_data_group_out_name,
        data_group_out=output_data_group,
        description=(
            "Created zero-valued continuum PSF/Hessian Taylor terms for " "testing."
        ),
    )

    logger.debug(
        "Created zero-valued continuum Taylor PSF with shape "
        f"{psf_shape}, dtype {np.dtype(float_dtype).name}, and "
        f"{n_psf_taylor_terms} Taylor orders."
    )

    # Preserve the timing fields expected by imaging setup and
    # accumulate_timing(). Fine-grained gridding and FFT operations are zero
    # because this is only a placeholder.
    return_df = pd.DataFrame(
        {
            "T_gcf": [0.0],
            "T_vis_mask": [0.0],
            "T_uv_sampling_grid": [0.0],
            "T_fft_norm": [0.0],
            "T_make_point_spread_function_placeholder": [time.time() - start_total],
            "nterms": [nterms],
            "n_psf_taylor_terms": [n_psf_taylor_terms],
            "reference_frequency_hz": [
                np.nan if reference_frequency is None else float(reference_frequency)
            ],
        }
    )

    return img_xds, return_df
