from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def imaging_preparation_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    task_id=0,
):
    """Run the once-per-chunk continuum-imaging setup.

    Everything performed only once for one frequency-chunk map task belongs
    here:

    * construction of the :class:`IterationController`;
    * construction of the initially empty combined deconvolution return dict;
    * imaging-weight calculation;
    * creation of the local Taylor PSF/Hessian products;
    * creation of the primary-beam products, when requested;
    * creation of any continuum/Taylor coordinates and data groups required by
      the returned :class:`xarray.Dataset`.

    The dirty/residual Taylor images are deliberately not calculated here.
    They are produced by the first call to
    :func:`residual_cycle_continuum_single_field`.

    Notes
    -----
    For ``nterms=2`` the setup implementation is expected to create local
    Taylor PSF/Hessian products of orders 0, 1, and 2. The preferred xarray
    representation is one point-spread-function variable with a Taylor-order
    dimension rather than separate unrelated variables.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this frequency chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this chunk.
    image_params : dict
        Image geometry and continuum configuration. In addition to the normal
        image geometry, this should contain or resolve:

        * ``nterms``;
        * ``reference_frequency``;
        * the output polarization and time coordinates;
        * gridding/FFT parameters such as ``fft_padding``.
    imaging_weights_params : dict
        Weighting configuration.
    iteration_control_params : dict
        Major/minor-cycle control configuration.
    processing_set_data_group_name : str, optional
        Processing-set data group to image.
    single_precision_image : bool, optional
        Whether image-domain products use single precision.
    processing_function_threads : int, optional
        Number of threads supplied to lower-level processing kernels.
    fft_backend : str, optional
        FFT backend used during image normalization.
    image_data_variables_keep : list of str, optional
        Logical output variables retained in ``img_xds``.
    task_id : int, optional
        Frequency-chunk identifier.

    Returns
    -------
    controller : IterationController
        Freshly initialized iteration controller.
    img_xds : xarray.Dataset
        Dataset carrying the local Taylor PSF/Hessian and setup products.
    return_df : pandas.DataFrame
        One-row timing frame returned by the setup function.
    combined_deconvolve_dict : ReturnDict
        Initially empty deconvolution-statistics accumulator.
    T_setup : float
        Wall-clock duration of this setup phase.
    """
    import time

    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
        imaging_setup_continuum_single_field,
    )

    logger.debug("Processing continuum chunk " + str(task_id))

    start = time.time()

    img_xds, return_df = imaging_setup_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        processing_set_data_group_name=processing_set_data_group_name,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
    )

    T_setup = time.time() - start

    return (
        img_xds,
        return_df,
        T_setup,
    )


@shares_param_docs
def residual_update_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    restore=False,
    is_n_iter_0=True,
    task_id=0,
):
    """Perform setup and exactly one continuum residual update."""

    import time

    import pandas as pd

    from astroviper.processing_functions.imaging.residual_cycle_continuum_single_field import (
        residual_cycle_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.utils import accumulate_timing

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    timing = {
        "T_prep": 0.0,
        "T_residual_cycle": 0.0,
        "task_id": task_id,
    }

    # -------------------------------------------------------------
    # Setup phase
    # -------------------------------------------------------------
    (img_xds, setup_return_df, T_setup,) = imaging_preparation_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        processing_set_data_group_name=processing_set_data_group_name,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
        task_id=task_id,
    )

    timing["T_prep"] = T_setup
    accumulate_timing(
        timing,
        setup_return_df,
        phase="prep",
    )

    # -------------------------------------------------------------
    # Exactly one residual update
    # -------------------------------------------------------------
    start = time.time()

    img_xds, residual_return_df = residual_cycle_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        is_n_iter_0,
        processing_set_data_group_name=processing_set_data_group_name,
        instrument_polarization_basis=instrument_polarization_basis,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
    )

    timing["T_residual_cycle"] = time.time() - start
    accumulate_timing(timing, residual_return_df)

    timing["n_channels"] = img_xds.sizes.get(
        "frequency",
        len(img_xds.coords.get("frequency", [])),
    )
    timing["nterms"] = image_params.get("nterms", 2)

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df


@shares_param_docs
def model_update_mtmfs_single_field(
    img_xds,
    deconvolver,
    deconvolve_params,
    is_n_iter_0=True,
    processing_function_threads=1,
    image_data_group_in_name="residual",
    image_data_group_out_name="model",
):
    """Run a temporary continuum minor cycle using Taylor order zero.

        This is an interim implementation until a true MT-MFS deconvolution backend
        is available. It performs an ordinary Högbom CLEAN using only

            SKY_RESIDUAL[taylor_term=0]

        and

            POINT_SPREAD_FUNCTION[psf_taylor_order=0].

        The resulting model is copied into

            SKY_MODEL[taylor_term=0].

        Higher-order Taylor residuals, PSFs, and model terms are left unchanged.

        For now, we follow this strategy as a placeholder
        continuum Taylor dataset
            │
            ├── take residual Taylor term 0
            ├── take PSF Taylor order 0
            ├── take model Taylor term 0
            │
            ▼
    build temporary one-channel cube dataset
    (time, frequency=1, polarization, l, m)
            │
            ▼
    call existing cube model-update function
            │
            ├── create primary-beam mask if needed
            ├── run Högbom CLEAN
            └── update the temporary cube model
            │
            ▼
    copy cleaned model plane back into SKY_MODEL[taylor_term=0]

        Parameters
        ----------
        img_xds : xarray.Dataset
            Globally reduced continuum image dataset. Residual/model images are
            expected to use ``taylor_term`` while the PSF uses
            ``psf_taylor_order``.
        deconvolver : str
            Deconvolver name. Currently only ``"hogbom"`` is supported.
        deconvolve_params : dict
            Minor-cycle parameters, including ``cycleniter``,
            ``cyclethreshold``, ``niter_per_plane``, and
            ``cyclethreshold_per_plane``.
        is_n_iter_0 : bool, optional
            Whether this is the first model-update cycle.
        processing_function_threads : int, optional
            Number of threads used by the deconvolution backend.
        image_data_group_in_name : str, optional
            Input image data-group name.
        image_data_group_out_name : str, optional
            Output model data-group name.

        Returns
        -------
        deconvolve_dict : ReturnDict
            Per-plane deconvolution statistics returned by the Högbom backend.
        return_df : pandas.DataFrame
            Timing information for this model-update cycle.

        Notes
        -----
        Despite the function name, this is not yet a true MT-MFS minor cycle.
        It is a compatibility layer around the existing single-frequency Högbom
        backend.
    """
    import copy
    import time

    import numpy as np
    import pandas as pd
    import xarray as xr

    if deconvolver.lower() != "hogbom":
        raise NotImplementedError(
            "Continuum deconvolution currently supports only "
            "'hogbom' cleaning of Taylor term zero."
        )

    if "SKY_RESIDUAL" not in img_xds:
        raise KeyError("The continuum image dataset does not contain SKY_RESIDUAL.")

    if "POINT_SPREAD_FUNCTION" not in img_xds:
        raise KeyError(
            "The continuum image dataset does not contain " "POINT_SPREAD_FUNCTION."
        )

    data_groups = img_xds.attrs.get("data_groups", {})

    if image_data_group_in_name not in data_groups:
        raise KeyError(
            f"Input data group {image_data_group_in_name!r} is not present "
            "in img_xds.attrs['data_groups']."
        )

    residual_data_group = data_groups[image_data_group_in_name]

    residual_name = residual_data_group.get("sky")
    psf_name = residual_data_group.get("point_spread_function")

    residual = img_xds["SKY_RESIDUAL"]
    psf = img_xds["POINT_SPREAD_FUNCTION"]

    if "taylor_term" not in residual.dims:
        raise ValueError("SKY_RESIDUAL must contain a 'taylor_term' dimension.")

    if "psf_taylor_order" not in psf.dims:
        raise ValueError(
            "POINT_SPREAD_FUNCTION must contain a " "'psf_taylor_order' dimension."
        )

    if residual.sizes["taylor_term"] < 1:
        raise ValueError("SKY_RESIDUAL contains no Taylor terms.")

    if psf.sizes["psf_taylor_order"] < 1:
        raise ValueError("POINT_SPREAD_FUNCTION contains no PSF Taylor terms.")

    # make a cube-compatible copy of the per-plane controls
    cube_deconvolve_params = copy.deepcopy(deconvolve_params)

    for parameter_name in (
        "niter_per_plane",
        "cyclethreshold_per_plane",
    ):
        parameter_value = cube_deconvolve_params.get(parameter_name)

        if parameter_value is None:
            continue

        parameter_array = np.asarray(parameter_value)

        # Continuum layout:
        #     time, taylor_term/frequency-like plane, polarization
        #
        # Temporary Högbom layout:
        #     time, frequency=1, polarization
        if parameter_array.ndim != 3:
            raise ValueError(
                f"{parameter_name} must have three dimensions "
                "(time, plane, polarization); "
                f"received shape {parameter_array.shape}."
            )

        if parameter_array.shape[1] < 1:
            raise ValueError(f"{parameter_name} contains no image planes.")

        cube_deconvolve_params[parameter_name] = np.ascontiguousarray(
            parameter_array[:, 0:1, :]
        )

    start = time.time()

    # add sidelobe point spread function
    img_xds = add_max_sidelobe_point_spread_function_continuum_single_field(img_xds)

    # ------------------------------------------------------------------
    # Build a temporary cube-like image dataset.
    #
    # The existing Högbom backend expects
    #
    #     time x frequency x polarization x l x m
    #
    # whereas the continuum products use Taylor dimensions.
    # ------------------------------------------------------------------

    if "frequency" in img_xds.coords and img_xds.sizes.get("frequency", 0) > 0:
        frequency_coord = np.atleast_1d(img_xds.coords["frequency"].values)[:1]
    elif "reference_frequency" in img_xds.attrs:
        frequency_coord = np.atleast_1d(img_xds.attrs["reference_frequency"])
    else:
        # The numerical value is irrelevant to Högbom CLEAN. The singleton
        # dimension exists only to satisfy the cube image interface.
        frequency_coord = np.asarray([0.0], dtype=np.float64)

    hogbom_xds = xr.Dataset(attrs=dict(img_xds.attrs))

    # Preserve coordinates that do not depend on a Taylor-order dimension.
    for coord_name, coord in img_xds.coords.items():
        if (
            "taylor_term" not in coord.dims
            and "psf_taylor_order" not in coord.dims
            and coord_name != "frequency"
        ):
            hogbom_xds = hogbom_xds.assign_coords({coord_name: coord.copy(deep=False)})

    hogbom_xds = hogbom_xds.assign_coords(frequency=("frequency", frequency_coord))

    def _to_single_frequency(data_array, plane_dimension):
        """Select plane zero and insert a singleton frequency dimension."""
        plane = data_array.isel({plane_dimension: 0}, drop=True)

        if "frequency" in plane.dims:
            if plane.sizes["frequency"] != 1:
                plane = plane.isel(frequency=0, drop=False)

            return plane.assign_coords(frequency=frequency_coord)

        # Insert frequency immediately after time to preserve the conventional
        # image dimension ordering.
        if "time" in plane.dims:
            axis = plane.dims.index("time") + 1
        else:
            axis = 0

        return plane.expand_dims(
            frequency=frequency_coord,
            axis=axis,
        )

    # Copy Taylor-dependent variables into the temporary cube layout.
    for variable_name, data_array in img_xds.data_vars.items():
        if "taylor_term" in data_array.dims:
            hogbom_xds[variable_name] = _to_single_frequency(
                data_array,
                "taylor_term",
            )

        elif "psf_taylor_order" in data_array.dims:
            hogbom_xds[variable_name] = _to_single_frequency(
                data_array,
                "psf_taylor_order",
            )

        elif "frequency" in data_array.dims:
            # Primary beam and similar products may retain a frequency
            # dimension. Högbom only needs one plane here.
            if data_array.sizes["frequency"] == 1:
                hogbom_xds[variable_name] = data_array.assign_coords(
                    frequency=frequency_coord
                )
            else:
                hogbom_xds[variable_name] = data_array.isel(
                    frequency=0, drop=False
                ).assign_coords(frequency=frequency_coord)

        else:
            hogbom_xds[variable_name] = data_array.copy(deep=False)

    # Ensure the temporary model variable exists.
    if "SKY_MODEL" not in hogbom_xds:
        hogbom_xds["SKY_MODEL"] = xr.zeros_like(hogbom_xds["SKY_RESIDUAL"])

    # Register the temporary model through the data-group interface expected by
    # deconvolve() and starting_statistics().
    hogbom_xds.attrs.setdefault("data_groups", {})

    hogbom_xds.attrs["data_groups"][image_data_group_out_name] = {
        "sky": "SKY_MODEL",
    }
    model_name = hogbom_xds.attrs["data_groups"][image_data_group_out_name]["sky"]

    T_make_hogbom_view = time.time() - start

    # Convert to real numbers
    hogbom_xds[residual_name] = hogbom_xds[residual_name].real.astype(np.float32)
    hogbom_xds[psf_name] = hogbom_xds[psf_name].real.astype(np.float32)
    hogbom_xds[model_name] = hogbom_xds[model_name].real.astype(np.float32)

    # ------------------------------------------------------------------
    # Run the existing single-frequency Högbom model-update backend.
    # ------------------------------------------------------------------
    start = time.time()

    from astroviper.processing_functions.imaging.model_update_cycle import (
        model_update_cycle_cube_single_field,
    )

    deconvolve_dict, hogbom_return_df = model_update_cycle_cube_single_field(
        hogbom_xds,
        deconvolver,
        cube_deconvolve_params,
        is_n_iter_0,
        processing_function_threads=processing_function_threads,
        image_data_group_in_name=image_data_group_in_name,
        image_data_group_out_name=image_data_group_out_name,
    )

    T_hogbom = time.time() - start

    # ------------------------------------------------------------------
    # Copy the Högbom model back into Taylor term zero.
    # ------------------------------------------------------------------
    start = time.time()

    if "SKY_MODEL" not in hogbom_xds:
        raise RuntimeError("The Högbom model-update backend did not produce SKY_MODEL.")

    model_zero = hogbom_xds["SKY_MODEL"].isel(
        frequency=0,
        drop=True,
    )

    if "SKY_MODEL" not in img_xds:
        # Create all model Taylor terms, initialized to zero.
        img_xds["SKY_MODEL"] = xr.zeros_like(img_xds["SKY_RESIDUAL"])

    if "taylor_term" not in img_xds["SKY_MODEL"].dims:
        raise ValueError(
            "The continuum SKY_MODEL variable must contain a "
            "'taylor_term' dimension."
        )

    # Use positional assignment because the Taylor coordinate need not
    # literally contain the value zero.
    img_xds["SKY_MODEL"].data[:, 0, ...] = model_zero.data

    T_copy_model = time.time() - start

    return_df = pd.DataFrame(
        {
            "T_make_hogbom_view": [T_make_hogbom_view],
            "T_hogbom": [T_hogbom],
            "T_copy_model": [T_copy_model],
            "T_model_update_cycle": [T_make_hogbom_view + T_hogbom + T_copy_model],
        }
    )

    if hogbom_return_df is not None:
        hogbom_return_df = hogbom_return_df.reset_index(drop=True)
        return_df = pd.concat(
            [return_df, hogbom_return_df],
            axis=1,
        )

    return deconvolve_dict, return_df


def add_max_sidelobe_point_spread_function_continuum_single_field(
    img_xds,
    *,
    image_data_group_name="residual",
    output_name="MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
):
    """Measure the maximum sidelobe of each Taylor PSF plane."""

    import numpy as np
    import xarray as xr

    data_group = img_xds.attrs["data_groups"][image_data_group_name]
    psf_name = data_group["point_spread_function"]

    psf = img_xds[psf_name]

    if "psf_taylor_order" not in psf.dims:
        raise ValueError(
            f"{psf_name!r} must contain 'psf_taylor_order'; "
            f"received dimensions {psf.dims}."
        )

    # Replace this block with the existing cube-PSF sidelobe-analysis
    # implementation if one is available.
    psf_values = np.abs(
        psf.transpose(
            "time",
            "psf_taylor_order",
            "polarization",
            "l",
            "m",
        ).values
    )

    n_time, n_order, n_pol, n_l, n_m = psf_values.shape
    max_sidelobe = np.empty(
        (n_time, n_order, n_pol),
        dtype=np.float64,
    )

    for time_index in range(n_time):
        for order_index in range(n_order):
            for pol_index in range(n_pol):
                plane = psf_values[
                    time_index,
                    order_index,
                    pol_index,
                ]

                peak_l, peak_m = np.unravel_index(
                    np.argmax(plane),
                    plane.shape,
                )

                sidelobe_plane = plane.copy()

                # Temporary main-lobe exclusion. Ideally derive this region from
                # the fitted restoring beam or reuse the cube implementation.
                half_width = 3

                l_start = max(0, peak_l - half_width)
                l_stop = min(n_l, peak_l + half_width + 1)
                m_start = max(0, peak_m - half_width)
                m_stop = min(n_m, peak_m + half_width + 1)

                sidelobe_plane[
                    l_start:l_stop,
                    m_start:m_stop,
                ] = 0.0

                max_sidelobe[
                    time_index,
                    order_index,
                    pol_index,
                ] = np.max(sidelobe_plane)

    img_xds[output_name] = xr.DataArray(
        max_sidelobe,
        dims=(
            "time",
            "psf_taylor_order",
            "polarization",
        ),
        coords={
            "time": psf.coords["time"],
            "psf_taylor_order": psf.coords["psf_taylor_order"],
            "polarization": psf.coords["polarization"],
        },
        attrs={
            "type": "max_sidelobe_point_spread_function",
        },
    )

    data_group["max_sidelobe_point_spread_function"] = output_name

    return img_xds
