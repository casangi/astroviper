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
    """Prepare the partition-local static products for continuum MT-MFS imaging.

    This function performs the setup required once for each visibility partition
    before the first continuum major cycle. It prepares all products that are
    independent of the current sky model and can therefore be reused throughout the
    major-cycle iterations.

    Specifically, this function

    * validates the continuum Taylor-expansion parameters;
    * attaches continuum metadata to the image dataset;
    * creates the continuum residual image data group;
    * verifies that imaging weights have already been prepared and attached to the
      processing-set partition;
    * constructs the partition-local Taylor PSF/Hessian products;
    * constructs the primary beam.

    The dirty/residual Taylor images are intentionally **not** calculated here.
    They are generated during
    :func:`residual_cycle_continuum_single_field`.

    Likewise, the restoring-beam fit is intentionally deferred until after the
    global reduction of the Taylor PSFs. Fitting the beam from a partition-local
    zeroth-order PSF is not equivalent to fitting the globally accumulated PSF.

    For ``nterms = N``, this function generates ``2N-1`` local PSF/Hessian Taylor
    orders. For example, ``nterms=2`` produces the local Taylor orders
    ``H_0``, ``H_1``, and ``H_2``.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing-set partition containing the visibility data and previously
        prepared imaging weights.
    img_xds : xarray.Dataset
        Empty continuum image dataset. On return, the dataset contains the
        partition-local static imaging products.
    image_params : dict
        Image geometry and continuum parameters. The following continuum-specific
        entries are used:

        ``nterms``
            Number of MT-MFS sky Taylor terms.

        ``reference_frequency`` or ``reference_frequency_hz``
            Common Taylor-expansion reference frequency shared by all visibility
            partitions.

        Standard imaging parameters such as image size, cell size, phase center,
        and FFT padding are forwarded to the lower-level processing functions.
    imaging_weights_params : dict
        Imaging-weight configuration. This function does not calculate imaging
        weights but retains this argument for API compatibility.
    processing_set_data_group_name : str, optional
        Processing-set data group containing the prepared imaging weights.
    single_precision_image : bool, optional
        If true, image-domain arrays are allocated using single precision.
    processing_function_threads : int, optional
        Number of threads passed to the lower-level processing functions.
    fft_backend : str, optional
        FFT backend used by the PSF-generation routine.
    image_data_variables_keep : list of str, optional
        Image products that should be retained in the returned dataset.
    image_data_group_out_name : str, optional
        Name of the continuum image data group. Defaults to ``"residual"``.

    Returns
    -------
    img_xds : xarray.Dataset
        Continuum image dataset containing the partition-local Taylor PSF/Hessian
        products, primary beam, and continuum metadata.
    return_df : pandas.DataFrame
        One-row timing dataframe summarizing the setup stage.

    Notes
    -----
    This function prepares only partition-local quantities. The globally reduced
    Taylor PSFs, Gaussian restoring-beam fit, inverse FFT, normalization,
    polarization conversion, minor cycle, and restoration are performed in later
    stages of the continuum imaging workflow after the map-task outputs have been
    combined.
    """
    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.make_point_spread_function_continuum_single_field import (
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
    # Load imaging weights from memory
    # -------------------------------------------------------------
    start = time.time()

    for ms_name, ms_xdt in ps_xdt.items():
        data_groups = ms_xdt.attrs.get("data_groups", {})

        if processing_set_data_group_name not in data_groups:
            raise KeyError(
                f"Data group {processing_set_data_group_name!r} is missing "
                f"from processing-set child {ms_name!r}."
            )

        weight_name = data_groups[processing_set_data_group_name].get("weight_imaging")

        if weight_name is None:
            raise KeyError(
                f"Imaging weights have not been registered for "
                f"processing-set child {ms_name!r}."
            )

        if weight_name not in ms_xdt:
            raise KeyError(
                f"Registered imaging-weight variable {weight_name!r} is absent "
                f"from processing-set child {ms_name!r}."
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

    # Deliberately no point_spread_function_gaussian_fit here. The correct beam
    # is fitted from the globally reduced zeroth-order Taylor PSF.

    return_df = pd.DataFrame(
        {
            "T_weights": [T_weights],
            "T_make_point_spread_function": [T_make_point_spread_function],
            "T_primary_beam": [T_primary_beam],
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
