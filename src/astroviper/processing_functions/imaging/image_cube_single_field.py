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
    psf_fitting_method="astroviper",
    task_id=0,
):
    """Run the once-per-chunk imaging setup before the imaging cycle loop.

    Everything that is done a single time per chunk happens here:

    * construction of the :class:`IterationController` and the (empty) combined
      return dict, and
    * the imaging weights, the point spread function and the primary beam (via
      :func:`~astroviper.processing_functions.imaging.residual_update.imaging_setup_single_field`).

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
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    iteration_control_params : dict
        CLEAN iteration controls. An **imaging cycle** (below simply a cycle)
        is one **residual update** (degrid the model, form residual
        visibilities, grid and inverse FFT them into the residual image)
        followed by one **model update** (deconvolve the residual image into
        the sky model). Every limit and threshold is applied independently to
        each ``(time, frequency, polarization)`` plane: a plane stops when it
        meets its own criterion. The imaging cycle loop runs separately for
        every frequency channel (the node task images one channel at a time),
        so a channel's cycles continue until all of its (time, polarization)
        planes have stopped, and a channel that has stopped does no further
        residual updates while the others carry on. The CASA ``tclean``
        equivalent is given in brackets. Keys:

        - ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution
          iterations (flux components) per plane, summed over all cycles. A
          plane stops once it has spent this budget. ``max_iter = 0`` makes
          only the dirty image (no deconvolution). *Differs from CASA*: CASA's
          ``niter`` is one budget for the whole image; here every plane gets
          the full value, and no budget is shared or split between planes.
        - ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.
          ``max_cycles = N`` performs ``N`` model updates; the dirty image is
          made by the residual update of the first cycle, and a closing
          residual update follows the last model update so that the written
          residual reflects the final model. ``max_cycles = 0`` makes only the
          dirty image; ``max_cycles = -1`` removes the limit. Counted per
          frequency channel: a channel that converges early stops cycling while
          the others continue.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a
          float in Jy. A plane stops when its peak residual inside the clean
          mask falls to or below ``threshold``; the value is also a hard floor
          on ``threshold_per_cycle``. ``threshold = 0`` disables the absolute
          stop. *Differs from CASA*: a float in Jy only, no ``'1mJy'`` strings.
        - ``threshold_sigma`` [CASA ``nsigma``] : Noise based stopping threshold
          per plane, as a multiple of the plane's robust residual rms
          (``1.4826 * MAD``). The effective threshold of a plane is
          ``max(threshold, threshold_sigma * rms)`` and it floors
          ``threshold_per_cycle`` in the same way. ``0`` disables it. Reserved:
          accepted but not yet implemented.
        - ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary beam
          mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.
          Pixels where the primary beam is below this fraction are excluded
          from cleaning. A masking cutoff, distinct from ``threshold``.
        - ``gain`` [CASA ``gain``] : CLEAN loop gain, the fraction of the
          selected peak flux subtracted from the residual image at each
          deconvolution iteration (``0 < gain <= 1``).
        - ``psf_sidelobe_factor`` [CASA ``cyclefactor``] : Multiplier applied to
          the measured peak PSF sidelobe level (``max_psf_sidelobe``) when
          setting how deep one model update cleans (see
          ``threshold_per_cycle``). Larger values trigger the next residual
          update sooner; smaller values clean deeper before each residual
          update.
        - ``max_iter_per_cycle`` [CASA ``cycleniter``] : Maximum number of
          deconvolution iterations a plane may run in one cycle's model update
          before the next residual update is triggered. ``max_iter_per_cycle =
          -1`` lets the adaptive ``threshold_per_cycle`` govern the depth
          instead; otherwise the count is clamped to never exceed the plane's
          remaining ``max_iter``.
        - ``min_psf_fraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF
          fraction defined below. Raising it limits how deep a single model
          update cleans.
        - ``max_psf_fraction`` [CASA ``maxpsffraction``] : Upper clamp on the
          same PSF fraction; it guarantees a minimum amount of cleaning per
          model update even when the PSF sidelobe level is high.
        - ``max_iter_divergence`` : Divergence test of the model update
          (Hogbom). Number of consecutive deconvolution iterations the RMS of
          a plane's residual, taken over the clean mask, may be above
          ``(1 + gain / 10)`` times the lowest RMS it has reached in the model
          update before that model update is stopped as diverged. A peak
          above ``(1 + gain)`` times the peak at the start of the model
          update, or a peak that is not finite, stops it at once. The next
          residual update then recomputes the true residual and the cycles
          go on. Default 1, the first such iteration; ``-1`` disables the
          test. *Differs from CASA*, which tests the peak for a fixed 10
          percent rise once every 2000 iterations.

        A plane whose model updates do no iteration any more (two in a row)
        is stopped with the no progress stop code, so an all-zero plane cannot
        keep its channel cycling.

        Derived per plane before each model update (not set by the caller):
        ``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,
        min_psf_fraction, max_psf_fraction)`` is the fraction of the current
        peak residual down to which one model update cleans, and
        ``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``
        is the stopping threshold of that model update, where
        ``peak_residual`` is the plane's peak residual inside the mask at the
        start of the cycle. The deconvolver also receives the per-plane
        ``max_iter_per_cycle``, ``min(max_iter_per_cycle, remaining max_iter)``.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the model
        update runs in single precision; the visibilities always stay double
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
        Freshly constructed controller (``stopcode.imaging == 0``).
    img_xds : xarray.Dataset
        Image dataset with the PSF and primary beam, in the Stokes basis.
    return_df : pandas.DataFrame
        One-row timing frame from the setup step.
    combined_imaging_dict : ImagingDict
        Empty accumulator for the per-plane convergence statistics.
    T_setup : float
        Wall-clock time of the setup step (seconds).
    """
    import time

    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.residual_update import (
        imaging_setup_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        ImagingDict,
        IterationController,
    )

    logger.debug("Processing chunk " + str(task_id))

    controller = IterationController(
        max_iter=iteration_control_params["max_iter"],
        max_cycles=iteration_control_params["max_cycles"],
        threshold=iteration_control_params["threshold"],
        gain=iteration_control_params["gain"],
        psf_sidelobe_factor=iteration_control_params["psf_sidelobe_factor"],
        min_psf_fraction=iteration_control_params["min_psf_fraction"],
        max_psf_fraction=iteration_control_params["max_psf_fraction"],
        max_iter_per_cycle=iteration_control_params["max_iter_per_cycle"],
    )
    combined_imaging_dict = ImagingDict()

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
        psf_fitting_method=psf_fitting_method,
    )
    T_setup = time.time() - start

    return controller, img_xds, return_df, combined_imaging_dict, T_setup


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
    primary_beam_correction=False,
    psf_fitting_method="astroviper",
    task_id=0,
):
    """Run the imaging cycle CLEAN loop for one single-field image chunk.

    Performs the once-per-chunk setup (imaging weights, PSF, primary beam), then
    iterates imaging cycles (a residual update followed by a model update) under the
    :class:`IterationController` until convergence, finishing with a last
    residual update that produces the final residual image.  Every processing
    function is timed; the totals are returned as a one-row timing frame.

    Operates on the full ``(time, frequency, polarization, l, m)`` cube it is
    given: every plane is controlled independently and the loop runs until all
    planes have stopped. The imaging node task calls this function once per
    frequency channel so that each channel runs its own imaging cycle loop (a
    converged channel then does no further residual updates); nothing here
    assumes a single channel, and multi-channel cubes are imaged in one call.

    Parameters
    ----------
    ps_xdt : xarray.DataTree
        Visibility data for this chunk.
    img_xds : xarray.Dataset
        Empty image dataset for this chunk.
    image_params : dict
        Image geometry and output coordinates: ``image_size``, ``cell_size``,
        ``phase_direction``, ``time_coords``, ``polarization_coords`` and the
        ``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is
        ``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image
        the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four
        correlations (see ``instrument_polarization_basis``).
    imaging_weights_params : dict
        Weighting scheme configuration: ``weighting`` (``"natural"`` or
        ``"briggs"``) and the Briggs ``robust`` parameter.
    iteration_control_params : dict
        CLEAN iteration controls. An **imaging cycle** (below simply a cycle)
        is one **residual update** (degrid the model, form residual
        visibilities, grid and inverse FFT them into the residual image)
        followed by one **model update** (deconvolve the residual image into
        the sky model). Every limit and threshold is applied independently to
        each ``(time, frequency, polarization)`` plane: a plane stops when it
        meets its own criterion. The imaging cycle loop runs separately for
        every frequency channel (the node task images one channel at a time),
        so a channel's cycles continue until all of its (time, polarization)
        planes have stopped, and a channel that has stopped does no further
        residual updates while the others carry on. The CASA ``tclean``
        equivalent is given in brackets. Keys:

        - ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution
          iterations (flux components) per plane, summed over all cycles. A
          plane stops once it has spent this budget. ``max_iter = 0`` makes
          only the dirty image (no deconvolution). *Differs from CASA*: CASA's
          ``niter`` is one budget for the whole image; here every plane gets
          the full value, and no budget is shared or split between planes.
        - ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.
          ``max_cycles = N`` performs ``N`` model updates; the dirty image is
          made by the residual update of the first cycle, and a closing
          residual update follows the last model update so that the written
          residual reflects the final model. ``max_cycles = 0`` makes only the
          dirty image; ``max_cycles = -1`` removes the limit. Counted per
          frequency channel: a channel that converges early stops cycling while
          the others continue.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a
          float in Jy. A plane stops when its peak residual inside the clean
          mask falls to or below ``threshold``; the value is also a hard floor
          on ``threshold_per_cycle``. ``threshold = 0`` disables the absolute
          stop. *Differs from CASA*: a float in Jy only, no ``'1mJy'`` strings.
        - ``threshold_sigma`` [CASA ``nsigma``] : Noise based stopping threshold
          per plane, as a multiple of the plane's robust residual rms
          (``1.4826 * MAD``). The effective threshold of a plane is
          ``max(threshold, threshold_sigma * rms)`` and it floors
          ``threshold_per_cycle`` in the same way. ``0`` disables it. Reserved:
          accepted but not yet implemented.
        - ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary beam
          mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.
          Pixels where the primary beam is below this fraction are excluded
          from cleaning. A masking cutoff, distinct from ``threshold``.
        - ``gain`` [CASA ``gain``] : CLEAN loop gain, the fraction of the
          selected peak flux subtracted from the residual image at each
          deconvolution iteration (``0 < gain <= 1``).
        - ``psf_sidelobe_factor`` [CASA ``cyclefactor``] : Multiplier applied to
          the measured peak PSF sidelobe level (``max_psf_sidelobe``) when
          setting how deep one model update cleans (see
          ``threshold_per_cycle``). Larger values trigger the next residual
          update sooner; smaller values clean deeper before each residual
          update.
        - ``max_iter_per_cycle`` [CASA ``cycleniter``] : Maximum number of
          deconvolution iterations a plane may run in one cycle's model update
          before the next residual update is triggered. ``max_iter_per_cycle =
          -1`` lets the adaptive ``threshold_per_cycle`` govern the depth
          instead; otherwise the count is clamped to never exceed the plane's
          remaining ``max_iter``.
        - ``min_psf_fraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF
          fraction defined below. Raising it limits how deep a single model
          update cleans.
        - ``max_psf_fraction`` [CASA ``maxpsffraction``] : Upper clamp on the
          same PSF fraction; it guarantees a minimum amount of cleaning per
          model update even when the PSF sidelobe level is high.
        - ``max_iter_divergence`` : Divergence test of the model update
          (Hogbom). Number of consecutive deconvolution iterations the RMS of
          a plane's residual, taken over the clean mask, may be above
          ``(1 + gain / 10)`` times the lowest RMS it has reached in the model
          update before that model update is stopped as diverged. A peak
          above ``(1 + gain)`` times the peak at the start of the model
          update, or a peak that is not finite, stops it at once. The next
          residual update then recomputes the true residual and the cycles
          go on. Default 1, the first such iteration; ``-1`` disables the
          test. *Differs from CASA*, which tests the peak for a fixed 10
          percent rise once every 2000 iterations.

        A plane whose model updates do no iteration any more (two in a row)
        is stopped with the no progress stop code, so an all-zero plane cannot
        keep its channel cycling.

        Derived per plane before each model update (not set by the caller):
        ``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,
        min_psf_fraction, max_psf_fraction)`` is the fraction of the current
        peak residual down to which one model update cleans, and
        ``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``
        is the stopping threshold of that model update, where
        ``peak_residual`` is the plane's peak residual inside the mask at the
        start of the cycle. The deconvolver also receives the per-plane
        ``max_iter_per_cycle``, ``min(max_iter_per_cycle, remaining max_iter)``.
    processing_set_data_group_name : str, optional
        Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).
    deconvolver : str, optional
        Deconvolution algorithm for the model update. One of ``"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``
        (C++, threaded across *and* within planes -- faster when there are
        few planes, e.g. single-channel imaging) or ``"asp"``.
    instrument_polarization_basis : str, optional
        Correlation (instrument) polarization basis the gridding is performed in:
        ``"linear"`` or ``"circular"``. The residual update grids and degrids the
        correlations of this basis and the model update deconvolves in the
        Stokes basis, in which the image is written. The Stokes planes requested
        in ``image_params["polarization_coords"]`` fix the correlations that are
        loaded and gridded: the two parallel hands give ``I, Q`` (linear) or
        ``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample
        is used only if none of its loaded correlations is flagged.
    single_precision_image : bool, optional
        If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model
        images) are single precision (``complex64`` / ``float32``) and the model
        update runs in single precision; the visibilities always stay double
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
    primary_beam_correction : bool, optional
        If ``True`` divide the restored sky by the (power) primary beam,
        writing the ``sky_restored_primary_beam_corrected``
        (``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``) variable (CASA ``pbcor``);
        pixels below the primary-beam cutoff are blanked with NaN.  Requires
        ``restore``.
    psf_fitting_method : str, optional
        Beam-fit algorithm for the PSF: ``"astroviper"`` (default) or
        ``"casa"``, the C++ port of CASA's ``StokesImageUtil::FitGaussianPSF``
        (the fit behind ``tclean``'s restoring beam).
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
        ``task_id``, ``n_channels`` and ``n_cycles``.
    combined_imaging_dict : ImagingDict
        Per-plane convergence statistics for this chunk.  Channel labels are
        chunk-local (0-based); the node task remaps them to global channel
        numbers before the reduce.
    """
    import time

    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.model_update import (
        model_update_cube_single_field,
    )
    from astroviper.processing_functions.imaging.residual_update import (
        residual_update_cube_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        accumulate_timing,
        build_residual_imaging_dict,
        get_calculate_cycle_controls,
        merge_imaging_dicts,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []

    # All once-only work -- controller setup, imaging weights, PSF and primary
    # beam creation -- happens in the preparation step before the imaging cycle
    # loop. The dirty image and the first model update are the first iteration
    # of the loop below.
    (
        controller,
        img_xds,
        setup_return_df,
        combined_imaging_dict,
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
        psf_fitting_method=psf_fitting_method,
        task_id=task_id,
    )

    # Per-chunk timing accumulator, grouped by pipeline phase. The preparation
    # (setup) sub-timings are namespaced (``T_prep_*``) because several of them
    # (``T_transform_pol``, ``T_fft_norm``, ``T_gcf``, ...) share names with the
    # residual update and would otherwise be summed into one indistinguishable
    # number. Phase totals: T_prep, T_residual_update, T_model_update,
    # T_restore.
    timing = {"T_prep": T_setup}
    accumulate_timing(timing, setup_return_df, phase="prep")

    # Phase totals plus the two model-phase leaves measured here: iteration
    # control and the convergence/merge bookkeeping run in this loop (not inside
    # a processing function), so they are timed inline.
    timing["T_residual_update"] = 0.0
    timing["T_model_update"] = 0.0
    timing["T_iteration_control"] = 0.0
    timing["T_convergence"] = 0.0

    model_exists = False
    n_cycles = 0
    while controller.stopcode.imaging == 0:
        n_cycles += 1

        # ---- Residual-update phase ----
        start = time.time()
        img_xds, residual_return_df = residual_update_cube_single_field(
            ps_xdt,
            img_xds,
            image_params,
            model_exists,
            processing_set_data_group_name=processing_set_data_group_name,
            instrument_polarization_basis=instrument_polarization_basis,
            single_precision_image=single_precision_image,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            image_data_variables_keep=image_data_variables_keep,
        )
        timing["T_residual_update"] += time.time() - start
        accumulate_timing(timing, residual_return_df)

        # Check convergence against the fresh residual before deconvolving,
        # so e.g. max_cycles=0 stops here without spending any iterations.
        residual_imaging_dict = build_residual_imaging_dict(
            img_xds,
            image_data_group_in_name="residual",
            iteration_control_params=iteration_control_params,
        )
        pre_stopcode, pre_stopdesc = controller.check_convergence(residual_imaging_dict)

        # ---- Model-update phase (iteration control + deconvolve + convergence) ----
        model_phase_start = time.time()
        if pre_stopcode.imaging == 0 and iteration_control_params["max_iter"] > 0:
            logger.debug("Doing model update")
            # Size the controller's per-plane state to this cube so iteration
            # control (max_iter and threshold) is tracked independently for every
            # (time, frequency, polarization) plane before the deconvolver runs.
            start = time.time()
            controller.ensure_planes(
                img_xds.sizes["time"],
                img_xds.sizes["frequency"],
                img_xds.sizes["polarization"],
            )
            max_iter_per_cycle, threshold_per_cycle = get_calculate_cycle_controls(
                controller,
                combined_imaging_dict,
                img_xds,
                model_exists,
                iteration_control_params=iteration_control_params,
                residual_imaging_dict=residual_imaging_dict,
            )
            timing["T_iteration_control"] += time.time() - start

            # Per-cycle deconvolution parameters as a fresh dict (the shared
            # iteration_control_params is never mutated). ``threshold`` stays
            # the absolute user stopping threshold (the floor); the per-plane
            # ``max_iter_per_cycle`` (min(max_iter_per_cycle, remaining max_iter)
            # for every plane) and ``threshold_per_cycle`` arrays drive the
            # deconvolver. Without the per-cycle cap one model update would be
            # handed the whole remaining budget and, with threshold_per_cycle 0,
            # spend all of max_iter at once, collapsing the run to one cycle.
            deconvolve_params = {
                **iteration_control_params,
                "max_iter_per_cycle": max_iter_per_cycle,
                "threshold_per_cycle": threshold_per_cycle,
            }

            (
                imaging_dict,
                model_update_return_df,
            ) = model_update_cube_single_field(
                img_xds,
                deconvolver,
                deconvolve_params,
                model_exists=model_exists,
                processing_function_threads=processing_function_threads,
                image_data_group_in_name="residual",
                image_data_group_out_name="model",
            )
            accumulate_timing(timing, model_update_return_df)

            # Only flip once a deconvolve actually runs: if every cycle is
            # skipped, no model is ever created, and the closing residual update
            # below must not try to subtract one that doesn't exist.
            model_exists = True
            model_update_ran = True
        else:
            if pre_stopcode.imaging != 0:
                logger.debug(f"  *** CONVERGED before model update: {pre_stopdesc} ***")
            imaging_dict = residual_imaging_dict
            model_update_ran = False

        start = time.time()
        controller.update_counts(imaging_dict)

        # check_convergence stamps the stop code into imaging_dict, so run
        # it before the merge to carry that stop code into the combined dict.
        # It also applies the no progress stop: a plane whose model updates do
        # no iteration any more (an all-zero plane, say) cannot change its
        # residual and is stopped instead of cycling for ever.
        stopcode, stopdesc = controller.check_convergence(
            imaging_dict, model_update_ran=model_update_ran
        )
        combined_imaging_dict = merge_imaging_dicts(
            [combined_imaging_dict, imaging_dict]
        )
        timing["T_convergence"] += time.time() - start

        # Model-update phase total: iteration control + deconvolve +
        # convergence (everything done after each residual update).
        timing["T_model_update"] += time.time() - model_phase_start

        if stopcode.imaging != 0:
            logger.debug(f"  *** CONVERGED: {stopdesc} ***")
            break

    # Closing residual update: the final residual image after the last model
    # update. Skipped when no model update ever ran (the residual on img_xds is
    # then already the dirty image).
    if model_exists:
        start = time.time()
        img_xds, residual_return_df = residual_update_cube_single_field(
            ps_xdt,
            img_xds,
            image_params,
            model_exists,
            processing_set_data_group_name=processing_set_data_group_name,
            instrument_polarization_basis=instrument_polarization_basis,
            single_precision_image=single_precision_image,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            image_data_variables_keep=image_data_variables_keep,
        )
        timing["T_residual_update"] += time.time() - start
        accumulate_timing(timing, residual_return_df)

    # Restore: convolve the model with the clean beam and add the residual. Only
    # possible once a model exists; the model/residual/beam-fit all live on
    # img_xds at this point. restore_image self-times and returns a one-row
    # timing frame (``T_restore``) folded in like the other steps.
    timing["T_restore"] = 0.0
    if restore and model_exists:
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

    # Primary-beam correction of the restored sky (CASA pbcor): a single
    # division since PRIMARY_BEAM follows the CASA (power) definition. Uses
    # the deconvolver's primary_beam_limit as the blanking cutoff when set,
    # else the CASA pblimit default of 0.2.
    timing["T_correct_sky_by_primary_beam"] = 0.0
    if primary_beam_correction and restore and iteration_control_params["max_iter"] > 0:
        from astroviper.processing_functions.imaging.correct_sky_by_primary_beam import (
            correct_sky_by_primary_beam,
        )

        img_xds, pb_corr_return_df = correct_sky_by_primary_beam(
            img_xds,
            primary_beam_limit=(
                iteration_control_params.get("primary_beam_limit", 0.0) or 0.2
            ),
        )
        accumulate_timing(timing, pb_corr_return_df)

    timing["task_id"] = task_id
    timing["n_channels"] = img_xds.sizes["frequency"]
    timing["n_cycles"] = n_cycles

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df, combined_imaging_dict
