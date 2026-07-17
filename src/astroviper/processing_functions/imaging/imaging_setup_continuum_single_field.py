import time

import numpy as np

from astroviper.utils.param_docs import shares_param_docs


def _get_reference_frequency_hz(image_params, img_xds):
    """Resolve the MT-MFS reference frequency in Hz.

    Resolution order:

    1. ``image_params["reference_frequency"]``;
    2. ``image_params["reference_frequency_hz"]``;
    3. weighted/unweighted mean is intentionally *not* guessed here;
    4. if no explicit value is supplied, use the arithmetic mean of the image
       frequency coordinate as a temporary fallback.

    The fallback keeps the first implementation usable, but callers should
    normally provide an explicit reference frequency that is common to all map
    tasks. Otherwise every frequency chunk would choose a different Taylor
    expansion point, and the chunk-local Taylor products could not be reduced
    consistently.
    """
    if "reference_frequency" in image_params:
        reference_frequency = image_params["reference_frequency"]
    elif "reference_frequency_hz" in image_params:
        reference_frequency = image_params["reference_frequency_hz"]
    else:
        if "frequency" not in img_xds.coords:
            raise ValueError(
                "Continuum imaging requires an explicit reference frequency "
                "or an image frequency coordinate."
            )

        frequency = np.asarray(img_xds.coords["frequency"].data, dtype=np.float64)

        if frequency.size == 0:
            raise ValueError(
                "Cannot infer a continuum reference frequency from an empty "
                "frequency coordinate."
            )

        reference_frequency = float(np.mean(frequency))

    # Permit a scalar xarray/numpy object while rejecting a vector.
    reference_frequency = np.asarray(reference_frequency)

    if reference_frequency.size != 1:
        raise ValueError(
            "image_params['reference_frequency'] must be a scalar frequency "
            "shared by all continuum map tasks."
        )

    reference_frequency_hz = float(reference_frequency.reshape(()))

    if not np.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0.0:
        raise ValueError(
            "The continuum reference frequency must be finite and positive."
        )

    return reference_frequency_hz


def _validate_continuum_parameters(image_params, img_xds):
    """Validate and return ``(nterms, reference_frequency_hz)``."""
    nterms = int(image_params.get("nterms", 2))

    if nterms < 1:
        raise ValueError("image_params['nterms'] must be at least 1.")

    reference_frequency_hz = _get_reference_frequency_hz(
        image_params,
        img_xds,
    )

    return nterms, reference_frequency_hz


def _attach_continuum_metadata(
    img_xds,
    *,
    nterms,
    reference_frequency_hz,
):
    """Attach metadata needed by the map/reduce continuum workflow.

    The original input frequency coordinate is preserved. The actual Taylor
    arrays are created by
    ``make_point_spread_function_continuum_single_field``.
    """
    img_xds.attrs["continuum_imaging"] = {
        "nterms": int(nterms),
        "reference_frequency_hz": float(reference_frequency_hz),
        "n_residual_taylor_terms": int(nterms),
        "n_psf_taylor_terms": int(2 * nterms - 1),
    }

    return img_xds


@shares_param_docs
def imaging_setup_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    image_data_group_out_name="residual",
):
    """Perform the once-per-chunk setup for continuum MT-MFS imaging.

    This is the continuum counterpart of
    :func:`imaging_setup_single_field`. It performs the work that is
    independent of the current sky model and therefore needs to run only once
    for one frequency-chunk map task:

    * validate the continuum Taylor expansion parameters;
    * create the output image data group;
    * calculate imaging weights;
    * construct this chunk's local Taylor PSF/Hessian products;
    * construct the primary beam;
    * transform the image products from the instrument correlation basis to
      Stokes.

    The dirty/residual Taylor images are not calculated here. They are created
    later by :func:`residual_cycle_continuum_single_field`.

    For ``nterms = N``, the local PSF/Hessian contribution contains
    ``2*N - 1`` Taylor orders. In particular, ``nterms=2`` produces local
    orders 0, 1, and 2.

    The PSF Gaussian fit and maximum-sidelobe calculation are intentionally
    deferred until after the map-task products have been globally reduced.
    Fitting every chunk-local zeroth-order PSF is not equivalent to fitting the
    globally summed zeroth-order PSF.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Visibility data for this frequency chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this frequency chunk, initially in the
        instrument correlation basis.
    image_params : dict
        Image geometry and continuum settings. The following continuum entries
        are used:

        ``nterms``
            Number of MT-MFS sky Taylor terms. Defaults to 2.

        ``reference_frequency`` or ``reference_frequency_hz``
            Scalar Taylor expansion reference frequency, common to all map
            tasks. Supplying this explicitly is strongly recommended.

        The usual geometry entries such as ``image_size``, ``cell_size``,
        ``phase_direction`` and ``fft_padding`` are forwarded to lower-level
        functions.
    imaging_weights_params : dict
        Imaging-weight configuration.
    processing_set_data_group_name : str, optional
        Processing-set data group to image.
    single_precision_image : bool, optional
        If true, image-domain arrays use ``float32`` / ``complex64``.
    processing_function_threads : int, optional
        Threads supplied to processing kernels.
    fft_backend : str, optional
        FFT backend used during PSF normalization.
    image_data_variables_keep : list of str, optional
        Logical image variables retained in the returned dataset.
    image_data_group_out_name : str, optional
        Output image data group. Defaults to ``"residual"``.

    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset in the Stokes basis containing the chunk-local Taylor
        PSF/Hessian products and primary beam.
    return_df : pandas.DataFrame
        One-row setup timing dataframe.

    Notes
    -----
    This function depends on the new lower-level processing function

    ``make_point_spread_function_continuum_single_field(...)``.

    That function is responsible for creating the xarray Taylor-order
    representation. A recommended layout is:

    ``POINT_SPREAD_FUNCTION``
        dimensions ``(time, psf_taylor_order, polarization, l, m)``

    with ``psf_taylor_order = range(2*nterms - 1)``.

    The input frequency channels remain available as coordinates or metadata,
    but the PSF output itself should be collapsed over frequency into Taylor
    orders.
    """
    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.image_analysis.transform_polarization_basis import (
        transform_polarization_basis,
    )
    from astroviper.processing_functions.imaging.calculate_imaging_weights import (
        calculate_imaging_weights,
    )
    from astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field_zero import (
        make_point_spread_function_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.primary_beam.make_primary_beam import (
        make_primary_beam_single_field,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    if single_precision_image:
        float_dtype = np.float32
        complex_dtype = np.complex64
    else:
        float_dtype = np.float64
        complex_dtype = np.complex128

    ps_data_group_name = processing_set_data_group_name

    nterms, reference_frequency_hz = _validate_continuum_parameters(
        image_params,
        img_xds,
    )

    img_xds = _attach_continuum_metadata(
        img_xds,
        nterms=nterms,
        reference_frequency_hz=reference_frequency_hz,
    )

    # Create the residual group up front. The local Taylor PSFs, primary beam,
    # and later residual Taylor products are registered under this group.
    data_groups = img_xds.attrs.setdefault("data_groups", {})

    if image_data_group_out_name not in data_groups:
        img_xds.attrs["type"] = "image_dataset"
        img_xds = img_xds.xr_img.add_data_group(
            new_data_group_name=image_data_group_out_name,
            new_data_group={
                "description": (
                    "Chunk-local continuum Taylor PSF and residual products."
                ),
                "date": "2026",
            },
        )

        logger.debug("continuum img_xds size " + str(img_xds.nbytes / 1.0e9) + " GB")

    # -------------------------------------------------------------
    # Imaging weights
    # -------------------------------------------------------------
    start = time.time()

    calculate_imaging_weights(
        ps_xdt,
        img_xds,
        imaging_weights_params=imaging_weights_params,
        return_weight_density_grid=False,
        ms_data_group_in_name=ps_data_group_name,
        ms_data_group_out_name=ps_data_group_name,
        ms_data_group_out_modified={
            "weight_imaging": "WEIGHT_IMAGING",
        },
        processing_function_threads=processing_function_threads,
    )

    T_weights = time.time() - start

    # -------------------------------------------------------------
    # Chunk-local Taylor PSF/Hessian products
    # -------------------------------------------------------------
    start = time.time()

    (
        img_xds,
        point_spread_function_return_df,
    ) = make_point_spread_function_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        nterms=nterms,
        reference_frequency=reference_frequency_hz,
        ms_data_group_in_name=ps_data_group_name,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        image_data_variables_keep=image_data_variables_keep,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        complex_dtype=complex_dtype,
    )

    T_make_point_spread_function = time.time() - start

    # -------------------------------------------------------------
    # Primary beam
    # -------------------------------------------------------------
    #
    # This currently reuses the cube primary-beam routine. It therefore retains
    # the chunk frequency coordinate. A later continuum-specific implementation
    # may choose to evaluate the primary beam only at the common reference
    # frequency or to form Taylor PB terms.
    (img_xds, primary_beam_return_df,) = make_primary_beam_single_field(
        img_xds,
        image_params,
        image_data_group_in_name=image_data_group_out_name,
        image_data_group_out_name=image_data_group_out_name,
        float_dtype=float_dtype,
    )

    T_primary_beam = float(primary_beam_return_df["T_primary_beam"].iloc[0])

    # -------------------------------------------------------------
    # Correlation -> Stokes
    # -------------------------------------------------------------
    start = time.time()

    img_xds = transform_polarization_basis(
        img_xds,
        new_polarization_basis="stokes",
        overwrite=True,
    )

    T_transform_pol = time.time() - start

    # Deliberately no point_spread_function_gaussian_fit here. The correct beam
    # is fitted from the globally reduced zeroth-order Taylor PSF.

    return_df = pd.DataFrame(
        {
            "T_weights": [T_weights],
            "T_make_point_spread_function": [T_make_point_spread_function],
            "T_primary_beam": [T_primary_beam],
            "T_transform_pol": [T_transform_pol],
            # Preserve the cube timing schema while showing that the fit was
            # intentionally deferred.
            "T_psf_fit": [0.0],
            "nterms": [nterms],
            "n_psf_taylor_terms": [2 * nterms - 1],
            "reference_frequency_hz": [reference_frequency_hz],
        }
    )

    # Avoid duplicate columns between the setup-level summary and the
    # fine-grained PSF timing dataframe. Duplicate names cause
    # return_df[column] to return a DataFrame instead of a Series.
    overlapping_columns = return_df.columns.intersection(
        point_spread_function_return_df.columns
    )

    if len(overlapping_columns) > 0:
        point_spread_function_return_df = point_spread_function_return_df.drop(
            columns=list(overlapping_columns)
        )

    return_df = pd.concat(
        [return_df, point_spread_function_return_df],
        axis=1,
    )

    return img_xds, return_df
