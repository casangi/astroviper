from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def imaging_preparation_single_field(
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
    """Run the once-per-chunk imaging setup before the major-cycle loop.

    Everything that is done a single time per chunk happens here:

    * construction of the :class:`IterationController` and the (empty) combined
      return dict, and
    * the imaging weights, the point spread function and the primary beam (via
      :func:`~astroviper.processing_functions.imaging.residual_cycle.imaging_setup_single_field`).

    The dirty image and the first model update are deliberately NOT done here --
    they are the first iteration of the loop in :func:`image_cube_single_field`.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this chunk.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    iteration_control_params : dict
        CLEAN iteration controls: ``niter``, ``nmajor``, ``threshold``, ``gain``,
        ``cyclefactor``, ``cycleniter``, ``minpsffraction`` and
        ``maxpsffraction``.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / Numba /
        FFT) kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    task_id : int, optional
        Identifier of the frequency chunk being imaged.
    Returns
    -------
    controller : IterationController
        Freshly constructed controller (``stopcode.major == 0``).
    img_xds : xarray.Dataset
        Image dataset with the PSF and primary beam, in the Stokes basis.
    return_df : pandas.DataFrame
        One-row timing frame from the setup step.
    combined_deconvolve_dict : ReturnDict
        Empty accumulator for the per-plane convergence statistics.
    T_setup : float
        Wall-clock time of the setup step (seconds).
    """
    import time
    import toolviper.utils.logger as logger
    from astroviper.processing_functions.imaging.residual_cycle import (
        imaging_setup_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        IterationController,
        ReturnDict,
    )

    logger.debug("Processing chunk " + str(task_id))

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

    # Once-only imaging setup: imaging weights, PSF and primary beam. The dirty
    # image and the model update are NOT done here.
    start = time.time()
    img_xds, return_df = imaging_setup_single_field(
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

    return controller, img_xds, return_df, combined_deconvolve_dict, T_setup


@shares_param_docs
def image_cube_single_field(
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
    """Run the major/minor cycle CLEAN loop for one single-field image chunk.

    Performs the once-per-chunk setup (imaging weights, PSF, primary beam), then
    iterates residual (major) and model-update (minor) cycles under the
    :class:`IterationController` until convergence, finishing with a last
    residual cycle that produces the final residual image.  Every processing
    function is timed; the totals are returned as a one-row timing frame.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this chunk.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor.
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    iteration_control_params : dict
        CLEAN iteration controls: ``niter``, ``nmajor``, ``threshold``, ``gain``,
        ``cyclefactor``, ``cycleniter``, ``minpsffraction`` and
        ``maxpsffraction``.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    deconvolver : str, optional
        Deconvolution algorithm for the minor cycle. One of ``"hogbom"`` or
        ``"asp"``.
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` (``XX``/``YY``) or ``"circular"`` (``RR``/``LL``). The
        output image is always produced in the Stokes basis.
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / Numba /
        FFT) kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    restore : bool, optional
        If ``True`` produce a restored image after deconvolution: the model
        convolved with the clean beam (the Gaussian fit to the PSF) plus the
        residual, written to the ``sky_restored`` (``SKY_RESTORED``) variable.
    task_id : int, optional
        Identifier of the frequency chunk being imaged.
    Returns
    -------
    img_xds : xarray.Dataset
        Image dataset with the final residual image, sky model, PSF and primary
        beam (in the Stokes basis), and -- when ``restore`` is ``True`` -- the
        restored image.
    timing_df : pandas.DataFrame
        One-row frame with a ``T_*`` column per processing function plus
        ``task_id``, ``n_channels`` and ``n_major_cycles``.
    combined_deconvolve_dict : ReturnDict
        Per-plane convergence statistics for this chunk.  Channel labels are
        chunk-local (0-based); the node task remaps them to global channel
        numbers before the reduce.
    """
    import time
    import toolviper.utils.logger as logger
    import pandas as pd
    from astroviper.processing_functions.imaging.residual_cycle import (
        residual_cycle_cube_single_field,
    )
    from astroviper.processing_functions.imaging.model_update_cycle import (
        model_update_cycle_cube_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        ReturnDict,
        merge_return_dicts,
        accumulate_timing,
        get_calculate_cycle_controls,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    # All once-only work -- controller setup, imaging weights, PSF and primary
    # beam creation -- happens in the preparation step before the major-cycle
    # loop. The dirty image and the first model update are the first iteration
    # of the loop below.
    (
        controller,
        img_xds,
        setup_return_df,
        combined_deconvolve_dict,
        T_setup,
    ) = imaging_preparation_single_field(
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

    # Per-chunk timing accumulator, grouped by pipeline phase. The preparation
    # (setup) sub-timings are namespaced (``T_prep_*``) because several of them
    # (``T_transform_pol``, ``T_fft_norm``, ``T_gcf``, ...) share names with the
    # residual cycle and would otherwise be summed into one indistinguishable
    # number. Phase totals: T_prep, T_residual_cycle, T_model_update_cycle,
    # T_restore.
    timing = {"T_prep": T_setup}
    accumulate_timing(timing, setup_return_df, phase="prep")

    # Phase totals plus the two model-phase leaves measured here: iteration
    # control and the convergence/merge bookkeeping run in this loop (not inside
    # a processing function), so they are timed inline.
    timing["T_residual_cycle"] = 0.0
    timing["T_model_update_cycle"] = 0.0
    timing["T_iteration_control"] = 0.0
    timing["T_convergence"] = 0.0

    is_n_iter_0 = True
    n_major_cycles = 0
    while controller.stopcode.major == 0:
        n_major_cycles += 1

        # ---- Residual-update phase ----
        start = time.time()
        img_xds, residual_return_df = residual_cycle_cube_single_field(
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

        # ---- Model-update phase (iteration control + deconvolve + convergence) ----
        model_phase_start = time.time()
        if iteration_control_params["niter"] > 0:
            logger.debug("Doing model update")
            # Size the controller's per-plane state to this cube so iteration
            # control (niter and threshold) is tracked independently for every
            # (time, frequency, polarization) plane before the deconvolver runs.
            start = time.time()
            controller.ensure_planes(
                img_xds.sizes["time"],
                img_xds.sizes["frequency"],
                img_xds.sizes["polarization"],
            )
            cycle_niter, cyclethreshold, cyclethreshold_per_plane = (
                get_calculate_cycle_controls(
                    controller,
                    combined_deconvolve_dict,
                    img_xds,
                    is_n_iter_0,
                    iteration_control_params=iteration_control_params,
                )
            )
            timing["T_iteration_control"] += time.time() - start

            # Build the per-cycle deconvolution parameters as a fresh dict (the
            # shared iteration_control_params is never mutated). ``threshold``
            # stays the absolute user stopping threshold (the floor); the
            # adaptive minor-cycle controls are the representative scalar
            # ``cyclethreshold`` plus the per-plane remaining iterations and
            # ``cyclethreshold_per_plane`` arrays that actually drive the
            # deconvolver.
            deconvolve_params = {
                **iteration_control_params,
                "cycleniter": cycle_niter,
                "cyclethreshold": cyclethreshold,
                "niter_per_plane": controller.niter,
                "cyclethreshold_per_plane": cyclethreshold_per_plane,
            }

            deconvolve_dict, model_update_return_df = (
                model_update_cycle_cube_single_field(
                    img_xds,
                    deconvolver,
                    deconvolve_params,
                    is_n_iter_0=is_n_iter_0,
                    num_threads=processing_function_threads,
                    image_data_group_in_name="residual",
                    image_data_group_out_name="model",
                )
            )
            accumulate_timing(timing, model_update_return_df)
        else:
            deconvolve_dict = ReturnDict()

        is_n_iter_0 = False

        start = time.time()
        controller.update_counts(deconvolve_dict)

        # check_convergence stamps the stop code into deconvolve_dict, so run
        # it before the merge to carry that stop code into the combined dict.
        stopcode, stopdesc = controller.check_convergence(deconvolve_dict)
        combined_deconvolve_dict = merge_return_dicts(
            [combined_deconvolve_dict, deconvolve_dict]
        )
        timing["T_convergence"] += time.time() - start

        # Model-update phase total: iteration control + deconvolve +
        # convergence (everything done after each residual cycle).
        timing["T_model_update_cycle"] += time.time() - model_phase_start

        if stopcode.major != 0:
            logger.info(f"  *** CONVERGED: {stopdesc} ***")
            break

    # Last residual cycle to compute the final residual image after the last
    # model-update cycle.
    if iteration_control_params["niter"] > 0:
        start = time.time()
        img_xds, residual_return_df = residual_cycle_cube_single_field(
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

    # Restore: convolve the model with the clean beam and add the residual. Only
    # meaningful once a model exists (niter > 0); the model/residual/beam-fit all
    # live on img_xds at this point. restore_image self-times and returns a
    # one-row timing frame (``T_restore``) folded in like the other steps.
    timing["T_restore"] = 0.0
    if restore and iteration_control_params["niter"] > 0:
        from astroviper.processing_functions.imaging.restore import restore_image

        img_xds, restore_return_df = restore_image(
            img_xds,
            image_data_group_in_residual_name="residual",
            image_data_group_in_model_name="model",
            image_data_group_out_restore_name="restored",
            num_threads=processing_function_threads,
        )
        accumulate_timing(timing, restore_return_df)

    timing["task_id"] = task_id
    timing["n_channels"] = img_xds.sizes["frequency"]
    timing["n_major_cycles"] = n_major_cycles

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df, combined_deconvolve_dict
