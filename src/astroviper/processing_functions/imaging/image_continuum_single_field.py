from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def imaging_preparation_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    iteration_control_params,
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
    from astroviper.processing_functions.imaging.utils import (
        IterationController,
        ReturnDict,
    )

    logger.debug("Processing continuum chunk " + str(task_id))

    controller = IterationController(
        niter=iteration_control_params["niter"],
        nmajor=iteration_control_params["nmajor"],
        threshold=iteration_control_params["threshold"],
        gain=iteration_control_params["gain"],
        cyclefactor=iteration_control_params["cyclefactor"],
        minpsffraction=iteration_control_params["minpsffraction"],
        maxpsffraction=iteration_control_params["maxpsffraction"],
        cycleniter=iteration_control_params["cycleniter"],
    )

    combined_deconvolve_dict = ReturnDict()

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
        controller,
        img_xds,
        return_df,
        combined_deconvolve_dict,
        T_setup,
    )


@shares_param_docs
def image_continuum_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    iteration_control_params,
    processing_set_data_group_name="corrected",
    deconvolver="hogbom",
    instrument_polarization_basis="linear",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    restore=False,
    task_id=0,
):
    """Run continuum major/minor cycles for one frequency-chunk map task.

    This function is the processing-level continuum counterpart of
    ``image_cube_single_field``. It keeps the same controller/deconvolver
    structure and the same three-value return interface, but it operates on
    Taylor products instead of independent frequency-plane images.

    In the first implementation the caller may enforce ``niter=0``. In that
    case the function performs one continuum residual/dirty-image cycle and
    returns the chunk-local Taylor products without executing a model update.

    The expected local products for ``nterms=2`` are:

    * residual Taylor terms of orders 0 and 1;
    * PSF/Hessian Taylor terms of orders 0, 1, and 2;
    * normalization or sum-of-weight products required by the global reducer.

    These products should remain in ``img_xds`` and are returned in memory to
    the GraphViper reduce stage.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this frequency chunk.
    img_xds : xarray.Dataset
        Empty image dataset constructed by the node task.
    image_params : dict
        Image geometry and continuum parameters, including ``nterms`` and the
        reference frequency.
    imaging_weights_params : dict
        Imaging-weight configuration.
    iteration_control_params : dict
        CLEAN major/minor-cycle controls.
    processing_set_data_group_name : str, optional
        Processing-set data group to image.
    deconvolver : str, optional
        Deconvolver name. Retained and forwarded so the same function supports
        later MT-MFS model updates.
    instrument_polarization_basis : str, optional
        Instrument correlation basis used by the gridder.
    single_precision_image : bool, optional
        Whether image-domain arrays use single precision.
    processing_function_threads : int, optional
        Threads supplied to lower-level processing kernels.
    fft_backend : str, optional
        FFT backend used during image normalization.
    image_data_variables_keep : list of str, optional
        Logical image products retained in ``img_xds``.
    restore : bool, optional
        Whether to restore after deconvolution. Normally false for the
        map-stage major-cycle graph.
    task_id : int, optional
        Frequency-chunk identifier.

    Returns
    -------
    img_xds : xarray.Dataset
        Dataset containing the local continuum Taylor products.
    timing_df : pandas.DataFrame
        One-row timing frame for this processing function.
    combined_deconvolve_dict : ReturnDict
        Deconvolution/convergence metadata for this chunk.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger

    #    from astroviper.processing_functions.imaging.model_update_cycle import (
    #           model_update_cycle_mtmfs_single_field,
    #           )
    from astroviper.processing_functions.imaging.residual_cycle_continuum_single_field import (
        residual_cycle_continuum_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        ReturnDict,
        accumulate_timing,
        get_calculate_cycle_controls,
        merge_return_dicts,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    (
        controller,
        img_xds,
        setup_return_df,
        combined_deconvolve_dict,
        T_setup,
    ) = imaging_preparation_continuum_single_field(
        ps_xdt,
        img_xds,
        image_params,
        imaging_weights_params,
        iteration_control_params,
        processing_set_data_group_name=processing_set_data_group_name,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
        task_id=task_id,
    )

    timing = {"T_prep": T_setup}
    accumulate_timing(timing, setup_return_df, phase="prep")

    timing["T_residual_cycle"] = 0.0
    timing["T_model_update_cycle"] = 0.0
    timing["T_iteration_control"] = 0.0
    timing["T_convergence"] = 0.0

    is_n_iter_0 = True
    n_major_cycles = 0

    while controller.stopcode.major == 0:
        n_major_cycles += 1

        # -------------------------------------------------------------
        # Residual / major-cycle phase
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

        timing["T_residual_cycle"] += time.time() - start
        accumulate_timing(timing, residual_return_df)

        # -------------------------------------------------------------
        # Model-update / minor-cycle phase
        # -------------------------------------------------------------
        model_phase_start = time.time()

        """if iteration_control_params["niter"] > 0:
            logger.debug("Doing continuum model update")

            # For MT-MFS the frequency channels have already been collapsed into
            # Taylor products. The controller therefore operates on the output
            # continuum image planes rather than on the original channel axis.
            #
            # The expected image layout is:
            #
            #   time x taylor_term x polarization x l x m
            #
            # for residual/model products. Taylor terms are coupled by the
            # deconvolver, so the number of independently controlled planes is
            # normally time x polarization rather than time x frequency x
            # polarization.
            start = time.time()

            controller.ensure_planes(
                img_xds.sizes["time"],
                1,
                img_xds.sizes["polarization"],
            )

            (
                cycle_niter,
                cyclethreshold,
                cyclethreshold_per_plane,
            ) = get_calculate_cycle_controls(
                controller,
                combined_deconvolve_dict,
                img_xds,
                is_n_iter_0,
                iteration_control_params=iteration_control_params,
            )

            timing["T_iteration_control"] += time.time() - start

            deconvolve_params = {
                **iteration_control_params,
                "cycleniter": cycle_niter,
                "cyclethreshold": cyclethreshold,
                "niter_per_plane": controller.niter.clip(max=cycle_niter),
                "cyclethreshold_per_plane": cyclethreshold_per_plane,
            }

            (
                deconvolve_dict,
                model_update_return_df,
            ) = model_update_cycle_mtmfs_single_field(
                img_xds,
                deconvolver,
                deconvolve_params,
                is_n_iter_0=is_n_iter_0,
                processing_function_threads=processing_function_threads,
                image_data_group_in_name="residual",
                image_data_group_out_name="model",
            )

            accumulate_timing(timing, model_update_return_df)

        else:
            # Dirty/Taylor-product-only execution.
            deconvolve_dict = ReturnDict()"""
        deconvolve_dict = ReturnDict()

        is_n_iter_0 = False

        start = time.time()

        controller.update_counts(deconvolve_dict)

        stopcode, stopdesc = controller.check_convergence(deconvolve_dict)

        combined_deconvolve_dict = merge_return_dicts(
            [combined_deconvolve_dict, deconvolve_dict]
        )

        timing["T_convergence"] += time.time() - start
        timing["T_model_update_cycle"] += time.time() - model_phase_start

        if stopcode.major != 0:
            logger.debug(f"  *** CONVERGED: {stopdesc} ***")
            break

        # Defensive guard for the initial niter=0 implementation. The controller
        # is expected to stop after the first residual cycle, but this prevents an
        # accidental endless loop if its behavior changes.
        if iteration_control_params["niter"] == 0:
            logger.debug(
                "niter=0: completed one continuum residual/Taylor-product cycle."
            )
            break

    # -------------------------------------------------------------
    # Final residual cycle after the last model update
    # -------------------------------------------------------------
    if iteration_control_params["niter"] > 0:
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

        timing["T_residual_cycle"] += time.time() - start
        accumulate_timing(timing, residual_return_df)

    # -------------------------------------------------------------
    # Optional restoration
    # -------------------------------------------------------------
    timing["T_restore"] = 0.0

    if restore and iteration_control_params["niter"] > 0:
        from astroviper.processing_functions.imaging.restore import restore_image

        img_xds, restore_return_df = restore_image(
            img_xds,
            image_data_group_in_residual_name="residual",
            image_data_group_in_model_name="model",
            image_data_group_out_restore_name="restored",
            processing_function_threads=processing_function_threads,
        )

        accumulate_timing(timing, restore_return_df)

    timing["task_id"] = task_id
    timing["n_channels"] = img_xds.sizes.get(
        "frequency",
        len(img_xds.coords.get("frequency", [])),
    )
    timing["nterms"] = image_params.get("nterms", 2)
    timing["n_major_cycles"] = n_major_cycles

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df, combined_deconvolve_dict
