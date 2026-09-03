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
        CLEAN iteration controls. Iteration control is performed
        **independently per** ``(time, frequency, polarization)`` **plane**: each
        plane carries its own iteration budget and stopping thresholds, and the
        residual update cycle loop continues until *every* selected plane has
        stopped.

        Terminology: a **residual update cycle** recomputes the residual image
        from the visibilities; a **model update cycle** deconvolves that residual
        into the sky model (CASA calls these the major and minor cycle).

        Keys, with the CASA ``tclean`` equivalent in brackets:

        - ``niter_per_plane`` [CASA ``niter``] : Maximum number of CLEAN
          iterations (flux components) for **one plane**, summed over all
          residual update cycles. A plane stops once it has spent this budget.
          ``niter_per_plane=0`` makes only the dirty image (no deconvolution).
          *Differs from CASA*: CASA's ``niter`` is one budget for the whole
          image; here every plane gets the full value, and no budget is shared
          or split between planes.
        - ``nmajor`` [CASA ``nmajor``] : Maximum number of deconvolving residual
          update cycles. ``nmajor=N`` performs ``N`` deconvolutions -- the dirty
          image is computed inside the first such cycle, matching CASA -- and
          ``nmajor=-1`` removes the limit. Shared across planes (not tracked per
          plane), unlike ``niter_per_plane``.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, given
          as a float in Jy. A plane stops when its peak residual inside the clean
          mask falls to or below ``threshold``; the value is also a hard floor on
          ``cycle_threshold`` (below). ``threshold=0`` disables the absolute stop.
          *Differs from CASA*: a float in Jy only -- no ``'1mJy'`` strings.
        - ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary-beam
          mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.
          Pixels where the primary beam is below this fraction are excluded from
          cleaning. A masking cutoff, distinct from ``threshold``.
        - ``gain`` [CASA ``gain``] : CLEAN loop gain -- the fraction of the
          selected peak flux subtracted from the residual image each iteration
          (``0 < gain <= 1``).
        - ``cycle_factor`` [CASA ``cyclefactor``] : Scaling applied to the
          brightest PSF sidelobe level when setting the model update cycle
          stopping depth (see ``cycle_threshold`` below). Larger values trigger
          the next residual update sooner; smaller values clean deeper first.
        - ``cycle_niter`` [CASA ``cycleniter``] : Maximum number of iterations a
          plane may run in one model update cycle before a residual update is
          triggered. ``cycle_niter=-1`` lets the adaptive ``cycle_threshold``
          govern the depth instead; otherwise the count is clamped to never
          exceed the plane's remaining ``niter_per_plane``.
        - ``minpsffraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF
          fraction used to set ``cycle_threshold = clamp(max_psf_sidelobe *
          cycle_factor, minpsffraction, maxpsffraction) * peak_residual`` (then
          floored at ``threshold``). Raising it limits how deep one model update
          cycle cleans.
        - ``maxpsffraction`` [CASA ``maxpsffraction``] : Upper clamp on that same
          PSF fraction; it guarantees a minimum amount of cleaning per model
          update cycle even when the PSF sidelobe level is high.

        Two derived quantities appear in the deconvolution parameter dict and are
        computed, not set by the caller: ``cycle_niter_cap``
        (``min(cycle_niter, remaining niter_per_plane)``) and its per-plane array
        ``cycle_niter_cap_pp``, alongside ``cycle_threshold_pp``.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the minor
        cycle runs in single precision; the visibilities always stay double
        precision. If ``False`` the image-domain arrays are double precision.
    processing_function_threads : int, optional
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    fft_backend : str, optional
        FFT backend used by the gridder normalization (``"pyfftw"`` or
        ``"scipy"``).
    image_data_variables_keep : list of str, optional
        Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,
        ``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).
    task_id : int, optional
        Identifier of the parallel chunk being processed.
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
        niter_per_plane=iteration_control_params["niter_per_plane"],
        nmajor=iteration_control_params["nmajor"],
        threshold=iteration_control_params["threshold"],
        gain=iteration_control_params["gain"],
        cycle_factor=iteration_control_params["cycle_factor"],
        minpsffraction=iteration_control_params["minpsffraction"],
        maxpsffraction=iteration_control_params["maxpsffraction"],
        cycle_niter=iteration_control_params["cycle_niter"],
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
        CLEAN iteration controls. Iteration control is performed
        **independently per** ``(time, frequency, polarization)`` **plane**: each
        plane carries its own iteration budget and stopping thresholds, and the
        residual update cycle loop continues until *every* selected plane has
        stopped.

        Terminology: a **residual update cycle** recomputes the residual image
        from the visibilities; a **model update cycle** deconvolves that residual
        into the sky model (CASA calls these the major and minor cycle).

        Keys, with the CASA ``tclean`` equivalent in brackets:

        - ``niter_per_plane`` [CASA ``niter``] : Maximum number of CLEAN
          iterations (flux components) for **one plane**, summed over all
          residual update cycles. A plane stops once it has spent this budget.
          ``niter_per_plane=0`` makes only the dirty image (no deconvolution).
          *Differs from CASA*: CASA's ``niter`` is one budget for the whole
          image; here every plane gets the full value, and no budget is shared
          or split between planes.
        - ``nmajor`` [CASA ``nmajor``] : Maximum number of deconvolving residual
          update cycles. ``nmajor=N`` performs ``N`` deconvolutions -- the dirty
          image is computed inside the first such cycle, matching CASA -- and
          ``nmajor=-1`` removes the limit. Shared across planes (not tracked per
          plane), unlike ``niter_per_plane``.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, given
          as a float in Jy. A plane stops when its peak residual inside the clean
          mask falls to or below ``threshold``; the value is also a hard floor on
          ``cycle_threshold`` (below). ``threshold=0`` disables the absolute stop.
          *Differs from CASA*: a float in Jy only -- no ``'1mJy'`` strings.
        - ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary-beam
          mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.
          Pixels where the primary beam is below this fraction are excluded from
          cleaning. A masking cutoff, distinct from ``threshold``.
        - ``gain`` [CASA ``gain``] : CLEAN loop gain -- the fraction of the
          selected peak flux subtracted from the residual image each iteration
          (``0 < gain <= 1``).
        - ``cycle_factor`` [CASA ``cyclefactor``] : Scaling applied to the
          brightest PSF sidelobe level when setting the model update cycle
          stopping depth (see ``cycle_threshold`` below). Larger values trigger
          the next residual update sooner; smaller values clean deeper first.
        - ``cycle_niter`` [CASA ``cycleniter``] : Maximum number of iterations a
          plane may run in one model update cycle before a residual update is
          triggered. ``cycle_niter=-1`` lets the adaptive ``cycle_threshold``
          govern the depth instead; otherwise the count is clamped to never
          exceed the plane's remaining ``niter_per_plane``.
        - ``minpsffraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF
          fraction used to set ``cycle_threshold = clamp(max_psf_sidelobe *
          cycle_factor, minpsffraction, maxpsffraction) * peak_residual`` (then
          floored at ``threshold``). Raising it limits how deep one model update
          cycle cleans.
        - ``maxpsffraction`` [CASA ``maxpsffraction``] : Upper clamp on that same
          PSF fraction; it guarantees a minimum amount of cleaning per model
          update cycle even when the PSF sidelobe level is high.

        Two derived quantities appear in the deconvolution parameter dict and are
        computed, not set by the caller: ``cycle_niter_cap``
        (``min(cycle_niter, remaining niter_per_plane)``) and its per-plane array
        ``cycle_niter_cap_pp``, alongside ``cycle_threshold_pp``.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    deconvolver : str, optional
        Deconvolution algorithm for the minor cycle. One of ``"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``
        (C++, threaded across *and* within planes -- faster when there are
        few planes, e.g. single-channel imaging) or ``"asp"``.
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
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
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
        Identifier of the parallel chunk being processed.
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

    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.model_update_cycle import (
        model_update_cycle_cube_single_field,
    )
    from astroviper.processing_functions.imaging.residual_cycle import (
        residual_cycle_cube_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        ReturnDict,
        accumulate_timing,
        get_calculate_cycle_controls,
        merge_return_dicts,
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
        # print("*********** This is major cycle ", n_major_cycles)

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
        if iteration_control_params["niter_per_plane"] > 0:
            logger.debug("Doing model update")
            # Size the controller's per-plane state to this cube so iteration
            # control (niter_per_plane and threshold) is tracked independently for every
            # (time, frequency, polarization) plane before the deconvolver runs.
            start = time.time()
            controller.ensure_planes(
                img_xds.sizes["time"],
                img_xds.sizes["frequency"],
                img_xds.sizes["polarization"],
            )
            (
                cycle_niter_cap,
                cycle_threshold,
                cycle_threshold_pp,
            ) = get_calculate_cycle_controls(
                controller,
                combined_deconvolve_dict,
                img_xds,
                is_n_iter_0,
                iteration_control_params=iteration_control_params,
            )
            timing["T_iteration_control"] += time.time() - start

            # Build the per-cycle deconvolution parameters as a fresh dict (the
            # shared iteration_control_params is never mutated). ``threshold``
            # stays the absolute user stopping threshold (the floor); the
            # adaptive minor-cycle controls are the representative scalar
            # ``cycle_threshold`` plus the per-plane remaining iterations and
            # ``cycle_threshold_pp`` arrays that actually drive the
            # deconvolver.
            deconvolve_params = {
                **iteration_control_params,
                "cycle_threshold": cycle_threshold,
                # NOTE: no "cycle_niter" key is set here. It used to be, holding
                # the *computed* cycle_niter_cap rather than the user's
                # parameter, and nothing ever read it. The user's own
                # cycle_niter still arrives via **iteration_control_params.
                # Cap this minor cycle at cycle_niter_cap (= min(cycle_niter,
                # remaining)). The deconvolver uses cycle_niter_cap_pp as its
                # per-plane max_iter and never reads "cycle_niter"; without this
                # clamp a single minor cycle is handed the full remaining budget
                # and (when cycle_threshold is 0) consumes all of niter_per_plane at once,
                # collapsing the run to one major cycle. Clamping here makes
                # cycle_niter actually bound each minor cycle so nmajor major
                # cycles run as intended. .clip returns a copy (the controller's
                # own remaining-budget array is decremented later by update_counts).
                "cycle_niter_cap_pp": controller.niter_per_plane.clip(
                    max=cycle_niter_cap
                ),
                "cycle_threshold_pp": cycle_threshold_pp,
            }

            (
                deconvolve_dict,
                model_update_return_df,
            ) = model_update_cycle_cube_single_field(
                img_xds,
                deconvolver,
                deconvolve_params,
                is_n_iter_0=is_n_iter_0,
                processing_function_threads=processing_function_threads,
                image_data_group_in_name="residual",
                image_data_group_out_name="model",
            )
            accumulate_timing(timing, model_update_return_df)
            # print("cycle_niter: ", cycle_niter_cap)
            # print("cycle_threshold: ", cycle_threshold)
            # print("cycle_niter_cap_pp: ", controller.niter_per_plane)
            # print("cycle_threshold_pp", cycle_threshold_pp)
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
            logger.debug(f"  *** CONVERGED: {stopdesc} ***")
            break

    # Last residual cycle to compute the final residual image after the last
    # model-update cycle.
    if iteration_control_params["niter_per_plane"] > 0:
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
    # meaningful once a model exists (niter_per_plane > 0); the model/residual/beam-fit all
    # live on img_xds at this point. restore_image self-times and returns a
    # one-row timing frame (``T_restore``) folded in like the other steps.
    timing["T_restore"] = 0.0
    if restore and iteration_control_params["niter_per_plane"] > 0:
        from astroviper.processing_functions.imaging.restore import restore_image

        img_xds, restore_return_df = restore_image(
            img_xds,
            image_data_group_in_residual_name="residual",
            image_data_group_in_model_name="model",
            image_data_group_out_restore_name="restored",
            processing_function_threads=processing_function_threads,
            # The model cube is dead weight after the restore unless it is
            # written to the output store; let the restore reuse its buffer
            # instead of allocating a fresh restored cube.
            consume_model="sky_model" not in image_data_variables_keep,
        )
        accumulate_timing(timing, restore_return_df)

    timing["task_id"] = task_id
    timing["n_channels"] = img_xds.sizes["frequency"]
    timing["n_major_cycles"] = n_major_cycles

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df, combined_deconvolve_dict
