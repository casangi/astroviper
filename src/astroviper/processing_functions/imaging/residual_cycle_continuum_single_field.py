from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def residual_cycle_continuum_single_field(
    ps_xdt,
    img_xds,
    model_uv_xds,
    image_params,
    is_n_iter_0,
    specmode="mfs",
    model_xds=None,
    primary_beam_xds=None,
    processing_set_data_group_name="corrected",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    visibility_memory_mode="in_place",
    image_data_group_in_name="model",
    image_data_group_out_name="residual",
    last_residual_cycle=False,
):
    """Calculate the partition-local continuum residual Taylor products.

    This function performs one continuum major-cycle residual update for a single
    visibility partition. During the first major cycle it directly grids the
    observed visibilities into Taylor-weighted residual UV grids. During later
    major cycles it first predicts the current sky model from the globally prepared
    Fourier-domain Taylor model, forms residual visibilities, and then grids those
    residuals.

    The processing-set partition is expected to already contain prepared imaging
    weights. These are calculated before entering the continuum major-cycle loop
    and are reused throughout all subsequent major cycles.

    The function performs the following operations:

    * create the gridding convolution kernel;
    * (later major cycles only)
        * reconstruct channel-dependent model visibility grids from the Fourier
          Taylor model;
        * degrid the model into predicted visibilities;
        * form residual visibilities;
    * grid the residual visibilities into partition-local Taylor residual UV grids;
    * inverse Fourier-transform and normalize the local Taylor products.

    The resulting Taylor residual images remain partition-local and are later
    combined by the distributed reduction stage. No global reduction, minor cycle,
    restoration, or Gaussian PSF fitting is performed here.

    Parameters
    ----------
    ps_xdt : xarray.DataTree or mapping
        Processing-set partition containing the observed (or residual)
        visibilities together with previously prepared imaging weights.
    img_xds : xarray.Dataset
        Continuum image dataset containing the partition-local static products.
    model_uv_xds : xarray.Dataset
        Globally prepared Fourier-domain Taylor model. This is only used after the
        first major cycle.
    image_params : dict
        Image geometry and continuum imaging parameters.
    is_n_iter_0 : bool
        True for the first major cycle, in which no model prediction is required.
    processing_set_data_group_name : str, optional
        Processing-set data group containing the observed visibilities.
    instrument_polarization_basis : str, optional
        Instrument correlation basis.
    single_precision_image : bool, optional
        Whether complex image-domain arrays use single precision.
    processing_function_threads : int, optional
        Number of threads supplied to the gridding kernels.
    fft_backend : str, optional
        FFT backend used by the lower-level imaging functions.
    image_data_variables_keep : list of str, optional
        Image products retained in the returned dataset.
    visibility_memory_mode : {"in_memory", "in_place"}, optional
        MFS residual-update storage policy for the observed-data visibility grid.
        ``"in_place"`` reloads the observed visibilities and grids their
        visibility-domain residual during every residual-update cycle.
        ``"in_memory"`` retains the globally reduced observed-data Taylor UV
        grid from the first cycle; later map tasks grid only the predicted-model
        contribution, and the append node subtracts it from the cached observed
        grid before the inverse FFT. The setting currently applies only to MFS;
        MVC requires ``"in_place"``.
    image_data_group_in_name : str, optional
        Image data group containing the current Taylor model.
    image_data_group_out_name : str, optional
        Image data group receiving the residual Taylor products.
    last_residual_cycle : bool, optional
        Indicates whether this is the final residual calculation after the last
        minor cycle.

    Returns
    -------
    img_xds : xarray.Dataset
        Partition-local continuum image dataset containing the residual Taylor
        products.
    return_df : pandas.DataFrame
        Timing summary for the residual-update stage.

    Notes
    -----
    This function operates entirely on one visibility partition. The globally
    reduced Taylor residuals are produced later by the distributed reduction
    stage, after which the inverse FFT, Stokes conversion, normalization, and
    minor-cycle processing are performed."""
    import time

    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
        create_prolate_spheroidal_kernel_1D,
    )
    from astroviper.processing_functions.imaging.image_continuum_single_field import (
        prepare_model_uv_mvc_single_field,
    )
    from astroviper.processing_functions.imaging.residual_cycle import (
        calculate_residual_visibilities,
        make_visibility_model_single_field,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    specmode = str(specmode).lower()
    if specmode not in ("mfs", "mvc"):
        raise ValueError(
            f"specmode must be either 'mfs' or 'mvc'; received {specmode!r}."
        )
    if visibility_memory_mode not in ("in_memory", "in_place"):
        raise ValueError(
            "visibility_memory_mode must be 'in_memory' or 'in_place'; received "
            f"{visibility_memory_mode!r}."
        )
    if specmode == "mvc" and visibility_memory_mode != "in_place":
        raise ValueError(
            "visibility_memory_mode='in_memory' is currently supported only "
            "for specmode='mfs'."
        )

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

    T_degrid = 0.0
    T_residual_vis = 0.0

    # ------------------------------------------------------------
    # Later major cycles:
    # predict the current model and form residual visibilities.
    # ------------------------------------------------------------
    if not is_n_iter_0:
        start = time.time()

        if specmode == "mfs":
            # delete an old residual if present
            residual_data_group = img_xds.attrs.get("data_groups", {}).get(
                image_data_group_out_name,
                {},
            )

            residual_sky_name = residual_data_group.get("sky")

            if residual_sky_name is not None and residual_sky_name in img_xds:
                img_xds.xr_img.delete_data_variables(variables=[residual_sky_name])

            start = time.time()

            make_visibility_model_continuum_single_field(
                ps_xdt,
                model_uv_xds,
                cgk_1D,
                nterms=nterms,
                reference_frequency=reference_frequency,
                ms_data_group_out_name="model",
                ms_data_group_out_modified={
                    "correlated_data": "VISIBILITY_MODEL",
                },
                image_data_group_in_name=image_data_group_in_name,
                processing_function_threads=processing_function_threads,
                fft_padding=image_params["fft_padding"],
            )

            T_degrid += time.time() - start

            if visibility_memory_mode == "in_place":
                start = time.time()

                calculate_residual_visibilities(
                    ps_xdt,
                    ms_data_group_out_residual="residual",
                    ms_data_group_in_model="model",
                    ms_data_group_in_observed=ps_data_group_name,
                )

                T_residual_vis += time.time() - start
                ps_data_group_name = "residual"
            else:
                # The globally reduced observed-data grid is retained by the
                # append node. Grid only the predicted model contribution here;
                # the append node forms observed minus model after reduction.
                ps_data_group_name = "model"

        else:
            if model_xds is None:
                raise ValueError(
                    "MVC requires the accumulated image-domain Taylor model."
                )

            if primary_beam_xds is None:
                raise ValueError(
                    "MVC requires the task-local frequency-dependent primary beam."
                )

            frequency_values = np.asarray(
                img_xds.coords["frequency"].values,
                dtype=np.float64,
            )

            local_model_uv_xds = prepare_model_uv_mvc_single_field(
                model_xds,
                primary_beam_xds,
                frequency_values,
                image_params,
                instrument_polarization_basis=(instrument_polarization_basis),
                single_precision_image=(single_precision_image),
                processing_function_threads=(processing_function_threads),
                fft_backend=fft_backend,
                image_data_group_name=(image_data_group_in_name),
            )

            make_visibility_model_single_field(
                ps_xdt,
                local_model_uv_xds,
                cgk_1D,
                ms_data_group_out_name="model",
                ms_data_group_out_modified={
                    "correlated_data": "VISIBILITY_MODEL",
                },
                image_data_group_in_name=(image_data_group_in_name),
                processing_function_threads=(processing_function_threads),
                fft_padding=image_params["fft_padding"],
            )

            start = time.time()

            calculate_residual_visibilities(
                ps_xdt,
                ms_data_group_out_residual="residual",
                ms_data_group_in_model="model",
                ms_data_group_in_observed=ps_data_group_name,
            )

            T_residual_vis += time.time() - start
            ps_data_group_name = "residual"

        T_degrid += time.time() - start

    # ------------------------------------------------------------
    # Grid residual
    # ------------------------------------------------------------

    start = time.time()

    if specmode == "mfs":
        from astroviper.processing_functions.imaging.make_undeconvolved_image_continuum import (
            make_undeconvolved_image_continuum_single_field,
        )

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

    elif specmode == "mvc":
        from astroviper.processing_functions.imaging.make_undeconvolved_image_continuum_mvc import (
            make_undeconvolved_image_mvc_single_field,
        )

        (
            img_xds,
            make_undeconvolved_image_return_df,
        ) = make_undeconvolved_image_mvc_single_field(
            ps_xdt,
            img_xds,
            image_params,
            cgk_1D,
            ms_data_group_in_name=ps_data_group_name,
            image_data_group_out_name=image_data_group_out_name,
            processing_function_threads=processing_function_threads,
            complex_dtype=complex_dtype,
        )

    else:
        raise ValueError(
            f"specmode must be either 'mfs' or 'mvc'; received {specmode!r}."
        )

    T_grid = time.time() - start

    return_df = pd.DataFrame(
        {
            "T_gcf": [T_gcf],
            "T_degrid": [T_degrid],
            "T_residual_vis": [T_residual_vis],
            "T_grid": [T_grid],
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


def make_visibility_model_continuum_single_field(
    ps_xdt,
    model_xds,
    cgk_1D,
    nterms,
    reference_frequency,
    ms_data_group_out_name="model",
    ms_data_group_out_modified=None,
    image_data_group_in_name="model",
    processing_function_threads=1,
    fft_padding=1.2,
):
    """Predict channel-dependent model visibilities from a Fourier Taylor model.

    The continuum major cycle maintains the sky model as Fourier-transformed Taylor
    coefficients rather than frequency images. This function reconstructs the model
    visibility grid at the actual observing frequencies of one processing-set
    partition and degrids that model into predicted visibilities.

    For each frequency channel ν, the model visibility grid is reconstructed as

        M(u,v,ν) =
            Σ_t M_t(u,v)
                ((ν-ν_ref)/ν_ref)^t,

    where ``M_t`` denotes the Fourier-domain Taylor coefficient and ``ν_ref`` is
    the common continuum reference frequency.

    The reconstructed frequency grid is passed directly to the shared standard
    degridding primitive, which produces predicted model visibilities in the
    measurement set.

    This reconstruction is repeated independently for every visibility partition,
    whereas the Fourier Taylor model itself is prepared only once after each global
    minor-cycle update.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Processing-set partition receiving the predicted model visibilities.
    model_xds : xarray.Dataset
        Globally prepared Fourier-domain Taylor model. The selected image data
        group must register a visibility variable containing the Taylor
        coefficients.
    cgk_1D : numpy.ndarray
        One-dimensional prolate-spheroidal convolution kernel.
    nterms : int
        Number of Taylor terms to reconstruct.
    reference_frequency : float
        Common MT-MFS reference frequency in Hz.
    ms_data_group_out_name : str, optional
        Processing-set data group receiving the predicted model visibilities.
    ms_data_group_out_modified : dict, optional
        Mapping describing the output visibility variable. Defaults to
        ``{"correlated_data": "VISIBILITY_MODEL"}``.
    image_data_group_in_name : str, optional
        Image data group containing the Fourier Taylor model.
    processing_function_threads : int, optional
        Number of threads supplied to the standard degridder.
    fft_padding : float, optional
        FFT padding factor used to interpret the continuum UV-grid geometry.

    Returns
    -------
    None

    Notes
    -----
    For each processing-set partition this continuum implementation reconstructs
    the frequency-dependent UV grid from the stored Taylor coefficients and calls
    the shared standard degridding primitive directly. It does not route model
    prediction through the cube processing API.

    The Fourier-domain Taylor model is prepared only once after each global
    continuum minor cycle. Consequently, this function performs only the spectral
    reconstruction and degridding required for the current visibility partition.
    """
    import numpy as np
    import xarray as xr

    from astroviper.processing_functions.imaging.degrid_visibility_grid import (
        degrid_visibility_grid_single_field,
    )

    # ------------------------------------------------------------
    # Load grids from memory and perform sanity checks
    # ------------------------------------------------------------

    if ms_data_group_out_modified is None:
        ms_data_group_out_modified = {
            "correlated_data": "VISIBILITY_MODEL",
        }

    nterms = int(nterms)
    reference_frequency = float(reference_frequency)

    if nterms < 1:
        raise ValueError("nterms must be at least 1.")

    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError(
            "reference_frequency must be a positive finite frequency in Hz."
        )

    data_groups = model_xds.attrs.get("data_groups", {})

    if image_data_group_in_name not in data_groups:
        raise KeyError(
            f"Image data group {image_data_group_in_name!r} is not present "
            "in model_xds.attrs['data_groups']."
        )

    model_data_group = data_groups[image_data_group_in_name]
    model_visibility_name = model_data_group.get("visibility")

    if model_visibility_name is None:
        raise KeyError(
            f"Image data group {image_data_group_in_name!r} does not define "
            "a 'visibility' variable."
        )

    if model_visibility_name not in model_xds:
        raise KeyError(
            f"Model visibility-grid variable {model_visibility_name!r} is "
            "not present in model_xds."
        )

    model_taylor_grid = model_xds[model_visibility_name]

    if "taylor_term" not in model_taylor_grid.dims:
        raise ValueError(
            f"Model visibility-grid variable {model_visibility_name!r} must "
            "contain a 'taylor_term' dimension."
        )

    available_nterms = model_taylor_grid.sizes["taylor_term"]

    if available_nterms < nterms:
        raise ValueError(
            f"The model contains {available_nterms} Taylor terms, but "
            f"nterms={nterms} was requested."
        )

    # Ignore any extra Taylor terms that may be present.
    model_taylor_grid = model_taylor_grid.isel(taylor_term=slice(0, nterms))

    # ------------------------------------------------------------
    # Wrapper around cube implementation, may need refactoring into a native continuum implementation later
    # ------------------------------------------------------------

    for _, ms_xdt in ps_xdt.items():
        if "frequency" in ms_xdt.coords:
            frequency = ms_xdt.coords["frequency"]
        elif "frequency" in ms_xdt:
            frequency = ms_xdt["frequency"]
        else:
            raise KeyError(
                "The measurement-set dataset does not contain a 'frequency' coordinate."
            )

        if frequency.ndim != 1:
            raise ValueError(
                "The measurement-set frequency coordinate must be "
                f"one-dimensional; received dimensions {frequency.dims}."
            )

        frequency_dimension = frequency.dims[0]

        # Use the canonical name expected by the cube degridder.
        if frequency_dimension != "frequency":
            frequency = frequency.rename({frequency_dimension: "frequency"})

        frequency_values = np.asarray(
            frequency.values,
            dtype=np.float64,
        )

        if not np.all(np.isfinite(frequency_values)):
            raise ValueError(
                "The measurement-set frequency coordinate contains non-finite values."
            )

        taylor_coordinate = np.arange(nterms, dtype=np.int64)

        spectral_coordinate = (
            frequency_values - reference_frequency
        ) / reference_frequency

        # dimensions:
        #
        #     frequency x taylor_term
        #
        taylor_weights = xr.DataArray(
            spectral_coordinate[:, np.newaxis] ** taylor_coordinate[np.newaxis, :],
            dims=("frequency", "taylor_term"),
            coords={
                "frequency": frequency_values,
                "taylor_term": model_taylor_grid.coords.get(
                    "taylor_term",
                    taylor_coordinate,
                ),
            },
        )

        # Reconstruct the uv model for every actual visibility channel:
        #
        #     M(nu) = sum_t M_t x^t.
        #
        frequency_model_grid = (model_taylor_grid * taylor_weights).sum(
            dim="taylor_term",
            keep_attrs=True,
        )

        preferred_dimension_order = [
            dimension
            for dimension in (
                "time",
                "frequency",
                "polarization",
                "u",
                "v",
            )
            if dimension in frequency_model_grid.dims
        ]

        remaining_dimensions = [
            dimension
            for dimension in frequency_model_grid.dims
            if dimension not in preferred_dimension_order
        ]

        frequency_model_grid = frequency_model_grid.transpose(
            *preferred_dimension_order,
            *remaining_dimensions,
        )

        # Degrid the continuum-owned reconstructed grid through the common
        # numerical primitive; no cube processing function is involved.
        degrid_visibility_grid_single_field(
            ms_xdt,
            cgk_1D,
            model_xds,
            frequency_model_grid.values,
            np.arange(frequency_values.size, dtype=np.int64),
            ms_data_group_out_name=ms_data_group_out_name,
            ms_data_group_out_modified=ms_data_group_out_modified,
            overwrite=True,
            fft_padding=fft_padding,
            processing_function_threads=processing_function_threads,
            description=(
                "Continuum Taylor model reconstructed at visibility frequencies "
                "and sampled with the standard degridder."
            ),
        )

    return
