from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def imaging_preparation_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    cycle_control_params,
    processing_set_data_group_name="corrected",
    single_precision_image=True,
    processing_function_threads=1,
    fft_backend="pyfftw",
    image_data_variables_keep=None,
    psf_fitting_method="astroviper",
    gridding_kernel_params=None,
    primary_beam_params=None,
    task_id=0,
):
    """Run the once-per-chunk imaging setup before the imaging cycle loop.

    Everything that is done a single time per chunk happens here:

    * construction of the :class:`CycleController` and the (empty) combined
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
    cycle_control_params : dict
        CLEAN cycle controls. An **imaging cycle** (below simply a cycle)
        is one **residual update** (degrid the model, form residual
        visibilities, grid and inverse FFT them into the residual image)
        followed by one **model update** (deconvolve the residual image into
        the sky model). Every limit and threshold is applied independently to
        each ``(time, frequency, polarization)`` plane: a plane stops when it
        meets its own criterion. The imaging cycle loop runs separately for
        every frequency channel in cube imaging (one channel per node task),
        so a channel's cycles continue until all of its (time, polarization)
        planes have stopped, and a channel that has stopped does no further
        residual updates while the others carry on. **Cycle control** is the
        stop test between the residual update and the model update, which
        decides whether a plane gets another cycle and sets the limits of its
        next model update. It is made after every residual update, the first
        included, on the residual that update has just made, never on the
        residual a model update ends with, and the residual and restored
        images written are those of the last residual update. **Iteration
        control** is the stop test inside the model update, made by the
        deconvolver at every iteration against those limits. The CASA
        ``tclean`` equivalent is given in brackets.
        Keys:

        - ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution
          iterations (flux components) per plane, summed over all cycles. A
          plane stops once it has spent this budget. ``max_iter = 0`` makes
          only the dirty image (no deconvolution). *Differs from CASA*: CASA's
          ``niter`` is one budget for the whole image; here every plane gets
          the full value, and no budget is shared or split between planes.
        - ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.
          ``max_cycles = N`` performs at most ``N`` model updates and
          ``N + 1`` residual updates: the first makes the dirty image, and
          every model update is followed by a residual update, whose residual
          the stop test checks; the last one's residual is written.
          ``max_cycles = 0`` makes only the dirty image; ``max_cycles = -1``
          removes the limit. Counted per
          frequency channel: a channel that converges early stops cycling while
          the others continue. ``max_cycles = "auto"`` predicts the limit of a
          channel from its dirty image: every plane takes ``P0``, its peak
          residual at the start of the first cycle (the ``peak_residual``
          below), ``f``, its PSF fraction (below), and ``L``, its threshold or,
          when that is 0, its theoretical point source sensitivity, worked out
          from the imaging weights as the ``SENSITIVITY`` of
          ``point_spread_function_cube_single_field``;
          ``n = ceil(ln(P0 / L) / ln(1 / f))`` cycles (at least 1) bring the
          peak down to ``L``, and the plane's limit is ``n + 3`` against a
          threshold and ``3 * n`` against the sensitivity (``EXTRA_CYCLES``
          and ``NOISY_CYCLE_SAFETY`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          The sensitivity is the thermal noise of the plane when the weights
          are absolute (``WEIGHT = 1 / sigma**2``). With unit weights, as a
          simulation without noise writes them, it is not a noise level but
          about 1 over the square root of the number of visibilities (natural
          weighting; 1e-3 for a million) in the units of the data; the limit
          grows only with ``ln(P0 / L)``, and on the same prediction ``3 * n``
          cycles bring the peak down to ``P0 * (L / P0)**3``. The channel's
          ``max_cycles`` is the largest limit of its planes (at least 1; a
          plane without a pixel in the clean mask, with a peak of 0 or without
          data is left out) and then counts down like a number. Every plane
          records its limit as ``max_cycles_auto`` in the returned convergence
          record. Needs ``max_psf_fraction < 1``.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a
          float in Jy, or a list of one float per Stokes plane in the order of
          ``polarization_coords`` (the floors of the Stokes planes of noise free
          data scale with their fluxes). The distributed application and the
          node task also take a nested list of shape (n_frequency,
          n_polarization), one row per image channel in the order of the image
          frequency coordinate; each channel gets its row. A plane stops when the
          peak of its residual inside the clean mask, as the latest residual
          update made it, falls to or below its threshold times
          ``1 + threshold_tolerance``; the threshold is
          also a hard floor on ``threshold_per_cycle``. ``0`` disables the
          absolute stop. *Differs from CASA*: floats in Jy only, no ``'1mJy'``
          strings.
        - ``threshold_tolerance`` : Relative tolerance of the threshold stop,
          a finite number ``>= 0``: a plane whose threshold is above 0 stops
          with imaging stop code 2, and gets no iterations while other planes of
          its channel go on, once the peak of its residual inside the clean mask
          is at most ``threshold * (1 + threshold_tolerance)``; the model update
          of a plane that goes on still cleans down to ``threshold_per_cycle``,
          floored at the threshold itself. Default 0.02 (``THRESHOLD_TOLERANCE`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          No CASA equivalent.
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
          Default 0: without a clean mask, the model update cleans and the
          stop test measures every pixel where the primary beam is above 0.
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
        - ``max_iter_divergence`` : Divergence test of the model update.
          Number of consecutive deconvolution iterations the RMS of
          a plane's residual, taken over the clean mask, may be above
          ``(1 + gain / 10)`` times the lowest RMS it has reached in the model
          update before that model update is stopped as diverged. For Hogbom
          a peak above ``(1 + gain)`` times the peak at the start of the
          model update stops it at once. Asp, whose least-squares components
          can raise the peak of a healthy residual, has no peak limit and
          allows ``(1 + gain)`` times the lowest RMS instead. An infinite peak
          always stops the model update. The next residual update then
          recomputes the true residual, the stop test is made on it, and the
          cycles go on (see ``max_cycles_divergence``). Default 1, the first
          such iteration; ``-1`` disables the two limits.
        - ``max_cycles_divergence`` : Divergence stop of the cycle control. A
          plane whose last ``max_cycles_divergence`` model updates all ended
          diverged (see ``max_iter_divergence``) gets no more iterations and
          is stopped with imaging stop code 5 by the stop test after the next
          residual update (unless a stop that takes precedence applies there:
          an empty mask, ``max_iter`` spent, the threshold or the entropy
          stop, codes 7, 1, 2 and 10), while the other planes of its channel
          carry on.
          Default ``-1``, off.
        - ``entropy_stop`` : If ``True``, a plane stops once the entropy of
          its residual has passed its maximum. The entropy (Homan, Roth and
          Pushkarev 2024, AJ 167, 11) measures how much the residual looks
          like noise everywhere. It rises while the clean removes emission
          and falls once the clean fits noise. It is worked out after every
          residual update, and the plane stops when it is lower than in an
          earlier cycle. The fall is noticed one model update after the
          maximum and the model of that model update is kept, so a small
          ``max_iter_per_cycle`` makes the stop sharper. Default ``False``.
          No CASA equivalent.
        - ``entropy_max_snr`` : The entropy of a plane is followed once the
          peak of its residual inside the clean mask is at most this many
          times the RMS of the residual. Above it the residual is dominated
          by the pattern of the point spread function. Default 6.
        - ``entropy_spatial_bins`` : Number of spatial bins along each of the
          two image axes used for the entropy. Default 7.
        - ``entropy_flux_bins`` : Number of flux bins per unit of RMS used for
          the entropy. Default 10.
        - ``record_iteration_history`` : If ``True``, every model update
          records, for every plane and every deconvolution iteration, the peak
          residual the iteration cleaned (signed, in the units of the residual
          image) and the RMS of the residual inside the clean mask after its
          subtraction, to follow the deconvolution in detail. The returned
          convergence record holds them as ``iteration_peak_residual`` and
          ``iteration_residual_rms``: per plane one array per cycle, with one
          value per iteration of that cycle's model update (empty for a cycle
          without one), so the record grows by two float64 values per
          iteration done. A debugging and tuning tool: the images are the same
          to the bit with and without it, and the RMS is the sum of squares
          the peak search of the next iteration makes anyway, so it costs no
          extra pass over the image per iteration. Hogbom deconvolvers only:
          ``deconvolver="asp"`` raises ``ValueError``. Default ``False``.
          No CASA equivalent.
        - ``exact_psf_subtraction`` : If ``True``, a Hogbom model update
          subtracts the PSF of every component it finds from the whole residual
          image, with a PSF of twice the image size made once per channel from
          the same imaging weights, cell size and gridding kernel. With
          ``False`` it subtracts the image-sized PSF, which reaches about
          ``n/2`` pixels either side of the component along each axis
          (``n`` the image size in pixels), so farther pixels keep the
          sidelobes of the flux cleaned until the next residual update. The
          beam fit, ``max_psf_sidelobe`` and the PSF written use the
          image-sized PSF either way. Costs a second PSF per channel (four
          times the pixels, kept through the channel's cycles) and a
          subtraction over the whole image at every iteration. Hogbom
          deconvolvers only: ``deconvolver="asp"`` raises ``ValueError``.
          Default ``False``. No CASA equivalent.
        - ``asp_stop_after_hogbom_run`` : If ``True``, an Asp model update ends
          after each of its Hogbom runs (51 or 510 point components) unless
          the RMS of the residual has fallen, by less than a factor 1.5, since
          the model update started, as in CASA and LibRA. Their residual drifts
          from the true one during a model update; the Asp residual of
          AstroVIPER is the exact one, so with ``False`` the model update goes
          on to its ``threshold_per_cycle``, its iteration limit or the
          divergence test, which takes fewer imaging cycles. Asp only.
          Default ``False``.
        - ``asp_residual_components`` : If ``True``, each Asp model update
          starts with components shaped by the residual (Ma, Zhang & Zhang
          2026, A&A 713, A164) down to ``threshold_per_cycle``, and the Asp
          iterations go on from there. They remove the ring residuals Asp
          leaves on resolved emission in far fewer iterations, and take no
          residual below ``threshold_per_cycle``. Further keys
          (``asp_rc_*``) are described in
          :func:`~astroviper.processing_functions.imaging.deconvolution.asp_clean`.
          Asp only. Default ``True``. No CASA equivalent.
        - ``asp_scale_significance`` : If ``True``, Asp measures the noise of
          the residual smoothed with each of its initial scales and, once the
          peak is below ``threshold_per_cycle``, goes on cleaning the most
          significant smoothed residual while it is as significant as the
          threshold is at the smallest scale. Asp only. Default ``True``. No
          CASA equivalent.

        A plane whose model updates do no iteration any more (two in a row)
        is stopped with the no progress stop code at the next stop test, so
        an all-zero plane cannot keep its channel cycling.

        Derived per plane before each model update (not set by the caller):
        ``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,
        min_psf_fraction, max_psf_fraction)`` is the fraction of the current
        peak residual down to which one model update cleans, and
        ``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``
        is the stopping threshold of that model update, where
        ``peak_residual`` is the plane's peak residual inside the clean mask
        after the residual update of the same cycle (the peak of the stop
        test), and
        ``max_iter_per_cycle_updated = min(max_iter_per_cycle,
        max_iter_remaining)`` is the iteration limit of that model update (0
        for a plane the entropy or divergence stop has ended), where
        ``max_iter_remaining`` is what is left of ``max_iter``.
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
    gridding_kernel_params : dict, optional
        The gridding convolution kernel the gridder and degridder use:
        ``type`` (``"prolate_spheroidal_casa"``, Schwab's rational
        approximation of the ``m = 6``, ``alpha = 1`` function with its
        analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,
        the generalised prolate spheroidal function of Schwab 1984 with its
        bandwidth and weight exponent chosen for the image region the
        ``fft_padding`` keeps; ``"exponential_semicircle"``, the modified
        exponential of a semicircle of Barnett et al. 2019 and Arras et al.
        2021 shaped for that region; or
        ``"least_misfit"``, the table that minimises the map error over that
        region for the lookup used, after Ye et al. 2020), ``support`` (the
        odd number of grid points the kernel spans along one axis; 7 for the
        ``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),
        ``oversampling`` (samples of the kernel table per uv cell) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, so the lookup error,
        and with it the floor the imaging cycle converges to, falls as 1 over
        the oversampling; ``"linear"`` interpolates the two neighbouring
        samples and the error falls as 1 over its square). Missing keys take
        the defaults ``{"type": "prolate_spheroidal_casa", "support": 7,
        "oversampling": 100, "interpolation": "nearest"}``; ``None`` is all
        defaults. A ``prolate_spheroidal``, ``exponential_semicircle`` or
        ``least_misfit`` kernel whose image plane correction falls below 1e-6
        of its centre value at the edge of the image kept (a support too large
        for the ``fft_padding``, e.g. 17 at 1.05) is refused. The kernel of the
        former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,
        not ``"prolate_spheroidal"``.
    primary_beam_params : dict, optional
        The primary beam of the imager, the power pattern that gives the clean
        mask and the primary beam correction: ``telescope`` (``"alma"``,
        ``"aca"`` or ``"ngvla"``, the textbook obscured Airy disk with the
        diameters of :func:`~astroviper.utils.beam_models.airy_disk_model`, no
        cut; ``"vla"``, the EVLA beam polynomial of CASA's ``PBMath1DEVLA``,
        with the coefficients CASA picks for each channel (of its two 10052 MHz
        rows, the first), at the exact radius and zero beyond 58 arcmin at
        1 GHz, past the 5 percent power level its polynomials were fitted
        to), ``effective_dish_diameter`` and
        ``blockage_diameter`` (metres; refused for ``"vla"``) and
        ``max_rad_1GHz`` (radians at 1 GHz, scaled as 1 / frequency, beyond
        which the beam is zero; ``None`` is no cut for an Airy disk). Numbers
        given with a telescope override that telescope's values;
        no telescope means ``"alma"``. ``None`` is the ALMA 12 m beam (10.7 m,
        0.75 m, no cut). Or ``beam_model`` alone: an analytic dict
        or a beam-polynomial dataset as ``simulate_processing_set`` takes it in
        ``beam_models`` (``"vla"`` is
        ``read_beam_polynomial_coefficients("EVLA_")``). The beam is the square
        of the voltage pattern the simulator applies for the model, computed by
        the same function
        (:func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`).
        The CASA variants, to compare with the ``.pb`` of ``tclean``, are such
        models: ``airy_disk_model(telescope, func="casa_airy")`` and
        ``read_beam_polynomial_coefficients("EVLA_", n_sample=10000)``.
        Zernike aperture models and Jones images are not supported: they are
        not azimuthally symmetric and depend on the parallactic angle.
    task_id : int, optional
        Identifier of the parallel chunk being processed.
    Returns
    -------
    controller : CycleController
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
        CycleController,
        ImagingDict,
        validate_entropy_params,
    )
    from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
        THRESHOLD_TOLERANCE,
    )

    logger.debug("Processing chunk " + str(task_id))

    # The entropy stop is optional: its keys take their defaults when absent.
    entropy_stop, entropy_max_snr, entropy_spatial_bins, entropy_flux_bins = (
        validate_entropy_params(cycle_control_params)
    )
    controller = CycleController(
        max_iter=cycle_control_params["max_iter"],
        max_cycles=cycle_control_params["max_cycles"],
        threshold=cycle_control_params["threshold"],
        gain=cycle_control_params["gain"],
        psf_sidelobe_factor=cycle_control_params["psf_sidelobe_factor"],
        min_psf_fraction=cycle_control_params["min_psf_fraction"],
        max_psf_fraction=cycle_control_params["max_psf_fraction"],
        max_iter_per_cycle=cycle_control_params["max_iter_per_cycle"],
        entropy_stop=entropy_stop,
        entropy_max_snr=entropy_max_snr,
        entropy_spatial_bins=entropy_spatial_bins,
        entropy_flux_bins=entropy_flux_bins,
        max_cycles_divergence=cycle_control_params.get("max_cycles_divergence", -1),
        threshold_tolerance=cycle_control_params.get(
            "threshold_tolerance", THRESHOLD_TOLERANCE
        ),
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
        gridding_kernel_params=gridding_kernel_params,
        primary_beam_params=primary_beam_params,
    )
    T_setup = time.time() - start

    return controller, img_xds, return_df, combined_imaging_dict, T_setup


@shares_param_docs
def image_cube_single_field(
    ps_xdt,
    img_xds,
    image_params,
    imaging_weights_params,
    cycle_control_params,
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
    gridding_kernel_params=None,
    primary_beam_params=None,
    task_id=0,
):
    """Run the imaging cycle CLEAN loop for one single-field image chunk.

    Performs the once-per-chunk setup (imaging weights, PSF, primary beam), then
    iterates imaging cycles (a residual update followed by a model update) under the
    :class:`CycleController` until convergence. The stop test is made after
    every residual update, the first included, on the residual that update has
    just made; the loop ends only there, so ``N`` model updates take ``N + 1``
    residual updates and the residual written is that of the last one.  Every
    processing function is timed; the totals are returned as a one-row timing
    frame.

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
    cycle_control_params : dict
        CLEAN cycle controls. An **imaging cycle** (below simply a cycle)
        is one **residual update** (degrid the model, form residual
        visibilities, grid and inverse FFT them into the residual image)
        followed by one **model update** (deconvolve the residual image into
        the sky model). Every limit and threshold is applied independently to
        each ``(time, frequency, polarization)`` plane: a plane stops when it
        meets its own criterion. The imaging cycle loop runs separately for
        every frequency channel in cube imaging (one channel per node task),
        so a channel's cycles continue until all of its (time, polarization)
        planes have stopped, and a channel that has stopped does no further
        residual updates while the others carry on. **Cycle control** is the
        stop test between the residual update and the model update, which
        decides whether a plane gets another cycle and sets the limits of its
        next model update. It is made after every residual update, the first
        included, on the residual that update has just made, never on the
        residual a model update ends with, and the residual and restored
        images written are those of the last residual update. **Iteration
        control** is the stop test inside the model update, made by the
        deconvolver at every iteration against those limits. The CASA
        ``tclean`` equivalent is given in brackets.
        Keys:

        - ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution
          iterations (flux components) per plane, summed over all cycles. A
          plane stops once it has spent this budget. ``max_iter = 0`` makes
          only the dirty image (no deconvolution). *Differs from CASA*: CASA's
          ``niter`` is one budget for the whole image; here every plane gets
          the full value, and no budget is shared or split between planes.
        - ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.
          ``max_cycles = N`` performs at most ``N`` model updates and
          ``N + 1`` residual updates: the first makes the dirty image, and
          every model update is followed by a residual update, whose residual
          the stop test checks; the last one's residual is written.
          ``max_cycles = 0`` makes only the dirty image; ``max_cycles = -1``
          removes the limit. Counted per
          frequency channel: a channel that converges early stops cycling while
          the others continue. ``max_cycles = "auto"`` predicts the limit of a
          channel from its dirty image: every plane takes ``P0``, its peak
          residual at the start of the first cycle (the ``peak_residual``
          below), ``f``, its PSF fraction (below), and ``L``, its threshold or,
          when that is 0, its theoretical point source sensitivity, worked out
          from the imaging weights as the ``SENSITIVITY`` of
          ``point_spread_function_cube_single_field``;
          ``n = ceil(ln(P0 / L) / ln(1 / f))`` cycles (at least 1) bring the
          peak down to ``L``, and the plane's limit is ``n + 3`` against a
          threshold and ``3 * n`` against the sensitivity (``EXTRA_CYCLES``
          and ``NOISY_CYCLE_SAFETY`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          The sensitivity is the thermal noise of the plane when the weights
          are absolute (``WEIGHT = 1 / sigma**2``). With unit weights, as a
          simulation without noise writes them, it is not a noise level but
          about 1 over the square root of the number of visibilities (natural
          weighting; 1e-3 for a million) in the units of the data; the limit
          grows only with ``ln(P0 / L)``, and on the same prediction ``3 * n``
          cycles bring the peak down to ``P0 * (L / P0)**3``. The channel's
          ``max_cycles`` is the largest limit of its planes (at least 1; a
          plane without a pixel in the clean mask, with a peak of 0 or without
          data is left out) and then counts down like a number. Every plane
          records its limit as ``max_cycles_auto`` in the returned convergence
          record. Needs ``max_psf_fraction < 1``.
        - ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a
          float in Jy, or a list of one float per Stokes plane in the order of
          ``polarization_coords`` (the floors of the Stokes planes of noise free
          data scale with their fluxes). The distributed application and the
          node task also take a nested list of shape (n_frequency,
          n_polarization), one row per image channel in the order of the image
          frequency coordinate; each channel gets its row. A plane stops when the
          peak of its residual inside the clean mask, as the latest residual
          update made it, falls to or below its threshold times
          ``1 + threshold_tolerance``; the threshold is
          also a hard floor on ``threshold_per_cycle``. ``0`` disables the
          absolute stop. *Differs from CASA*: floats in Jy only, no ``'1mJy'``
          strings.
        - ``threshold_tolerance`` : Relative tolerance of the threshold stop,
          a finite number ``>= 0``: a plane whose threshold is above 0 stops
          with imaging stop code 2, and gets no iterations while other planes of
          its channel go on, once the peak of its residual inside the clean mask
          is at most ``threshold * (1 + threshold_tolerance)``; the model update
          of a plane that goes on still cleans down to ``threshold_per_cycle``,
          floored at the threshold itself. Default 0.02 (``THRESHOLD_TOLERANCE`` of
          :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).
          No CASA equivalent.
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
          Default 0: without a clean mask, the model update cleans and the
          stop test measures every pixel where the primary beam is above 0.
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
        - ``max_iter_divergence`` : Divergence test of the model update.
          Number of consecutive deconvolution iterations the RMS of
          a plane's residual, taken over the clean mask, may be above
          ``(1 + gain / 10)`` times the lowest RMS it has reached in the model
          update before that model update is stopped as diverged. For Hogbom
          a peak above ``(1 + gain)`` times the peak at the start of the
          model update stops it at once. Asp, whose least-squares components
          can raise the peak of a healthy residual, has no peak limit and
          allows ``(1 + gain)`` times the lowest RMS instead. An infinite peak
          always stops the model update. The next residual update then
          recomputes the true residual, the stop test is made on it, and the
          cycles go on (see ``max_cycles_divergence``). Default 1, the first
          such iteration; ``-1`` disables the two limits.
        - ``max_cycles_divergence`` : Divergence stop of the cycle control. A
          plane whose last ``max_cycles_divergence`` model updates all ended
          diverged (see ``max_iter_divergence``) gets no more iterations and
          is stopped with imaging stop code 5 by the stop test after the next
          residual update (unless a stop that takes precedence applies there:
          an empty mask, ``max_iter`` spent, the threshold or the entropy
          stop, codes 7, 1, 2 and 10), while the other planes of its channel
          carry on.
          Default ``-1``, off.
        - ``entropy_stop`` : If ``True``, a plane stops once the entropy of
          its residual has passed its maximum. The entropy (Homan, Roth and
          Pushkarev 2024, AJ 167, 11) measures how much the residual looks
          like noise everywhere. It rises while the clean removes emission
          and falls once the clean fits noise. It is worked out after every
          residual update, and the plane stops when it is lower than in an
          earlier cycle. The fall is noticed one model update after the
          maximum and the model of that model update is kept, so a small
          ``max_iter_per_cycle`` makes the stop sharper. Default ``False``.
          No CASA equivalent.
        - ``entropy_max_snr`` : The entropy of a plane is followed once the
          peak of its residual inside the clean mask is at most this many
          times the RMS of the residual. Above it the residual is dominated
          by the pattern of the point spread function. Default 6.
        - ``entropy_spatial_bins`` : Number of spatial bins along each of the
          two image axes used for the entropy. Default 7.
        - ``entropy_flux_bins`` : Number of flux bins per unit of RMS used for
          the entropy. Default 10.
        - ``record_iteration_history`` : If ``True``, every model update
          records, for every plane and every deconvolution iteration, the peak
          residual the iteration cleaned (signed, in the units of the residual
          image) and the RMS of the residual inside the clean mask after its
          subtraction, to follow the deconvolution in detail. The returned
          convergence record holds them as ``iteration_peak_residual`` and
          ``iteration_residual_rms``: per plane one array per cycle, with one
          value per iteration of that cycle's model update (empty for a cycle
          without one), so the record grows by two float64 values per
          iteration done. A debugging and tuning tool: the images are the same
          to the bit with and without it, and the RMS is the sum of squares
          the peak search of the next iteration makes anyway, so it costs no
          extra pass over the image per iteration. Hogbom deconvolvers only:
          ``deconvolver="asp"`` raises ``ValueError``. Default ``False``.
          No CASA equivalent.
        - ``exact_psf_subtraction`` : If ``True``, a Hogbom model update
          subtracts the PSF of every component it finds from the whole residual
          image, with a PSF of twice the image size made once per channel from
          the same imaging weights, cell size and gridding kernel. With
          ``False`` it subtracts the image-sized PSF, which reaches about
          ``n/2`` pixels either side of the component along each axis
          (``n`` the image size in pixels), so farther pixels keep the
          sidelobes of the flux cleaned until the next residual update. The
          beam fit, ``max_psf_sidelobe`` and the PSF written use the
          image-sized PSF either way. Costs a second PSF per channel (four
          times the pixels, kept through the channel's cycles) and a
          subtraction over the whole image at every iteration. Hogbom
          deconvolvers only: ``deconvolver="asp"`` raises ``ValueError``.
          Default ``False``. No CASA equivalent.
        - ``asp_stop_after_hogbom_run`` : If ``True``, an Asp model update ends
          after each of its Hogbom runs (51 or 510 point components) unless
          the RMS of the residual has fallen, by less than a factor 1.5, since
          the model update started, as in CASA and LibRA. Their residual drifts
          from the true one during a model update; the Asp residual of
          AstroVIPER is the exact one, so with ``False`` the model update goes
          on to its ``threshold_per_cycle``, its iteration limit or the
          divergence test, which takes fewer imaging cycles. Asp only.
          Default ``False``.
        - ``asp_residual_components`` : If ``True``, each Asp model update
          starts with components shaped by the residual (Ma, Zhang & Zhang
          2026, A&A 713, A164) down to ``threshold_per_cycle``, and the Asp
          iterations go on from there. They remove the ring residuals Asp
          leaves on resolved emission in far fewer iterations, and take no
          residual below ``threshold_per_cycle``. Further keys
          (``asp_rc_*``) are described in
          :func:`~astroviper.processing_functions.imaging.deconvolution.asp_clean`.
          Asp only. Default ``True``. No CASA equivalent.
        - ``asp_scale_significance`` : If ``True``, Asp measures the noise of
          the residual smoothed with each of its initial scales and, once the
          peak is below ``threshold_per_cycle``, goes on cleaning the most
          significant smoothed residual while it is as significant as the
          threshold is at the smallest scale. Asp only. Default ``True``. No
          CASA equivalent.

        A plane whose model updates do no iteration any more (two in a row)
        is stopped with the no progress stop code at the next stop test, so
        an all-zero plane cannot keep its channel cycling.

        Derived per plane before each model update (not set by the caller):
        ``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,
        min_psf_fraction, max_psf_fraction)`` is the fraction of the current
        peak residual down to which one model update cleans, and
        ``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``
        is the stopping threshold of that model update, where
        ``peak_residual`` is the plane's peak residual inside the clean mask
        after the residual update of the same cycle (the peak of the stop
        test), and
        ``max_iter_per_cycle_updated = min(max_iter_per_cycle,
        max_iter_remaining)`` is the iteration limit of that model update (0
        for a plane the entropy or divergence stop has ended), where
        ``max_iter_remaining`` is what is left of ``max_iter``.
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
        If ``True`` write the primary beam corrected restored sky to the
        ``sky_restored_primary_beam_corrected``
        (``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``) variable: the model divided
        by the (power) primary beam and convolved with the clean beam, plus
        the residual divided by the primary beam; pixels below the primary
        beam cutoff (``primary_beam_limit``) are blanked with NaN.  Requires
        ``restore``.
    psf_fitting_method : str, optional
        Beam-fit algorithm for the PSF: ``"astroviper"`` (default) or
        ``"casa"``, the C++ port of CASA's ``StokesImageUtil::FitGaussianPSF``
        (the fit behind ``tclean``'s restoring beam).
    gridding_kernel_params : dict, optional
        The gridding convolution kernel the gridder and degridder use:
        ``type`` (``"prolate_spheroidal_casa"``, Schwab's rational
        approximation of the ``m = 6``, ``alpha = 1`` function with its
        analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,
        the generalised prolate spheroidal function of Schwab 1984 with its
        bandwidth and weight exponent chosen for the image region the
        ``fft_padding`` keeps; ``"exponential_semicircle"``, the modified
        exponential of a semicircle of Barnett et al. 2019 and Arras et al.
        2021 shaped for that region; or
        ``"least_misfit"``, the table that minimises the map error over that
        region for the lookup used, after Ye et al. 2020), ``support`` (the
        odd number of grid points the kernel spans along one axis; 7 for the
        ``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),
        ``oversampling`` (samples of the kernel table per uv cell) and
        ``interpolation`` (how the gridders read the table at a stencil
        point: ``"nearest"`` takes the nearest sample, so the lookup error,
        and with it the floor the imaging cycle converges to, falls as 1 over
        the oversampling; ``"linear"`` interpolates the two neighbouring
        samples and the error falls as 1 over its square). Missing keys take
        the defaults ``{"type": "prolate_spheroidal_casa", "support": 7,
        "oversampling": 100, "interpolation": "nearest"}``; ``None`` is all
        defaults. A ``prolate_spheroidal``, ``exponential_semicircle`` or
        ``least_misfit`` kernel whose image plane correction falls below 1e-6
        of its centre value at the edge of the image kept (a support too large
        for the ``fft_padding``, e.g. 17 at 1.05) is refused. The kernel of the
        former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,
        not ``"prolate_spheroidal"``.
    primary_beam_params : dict, optional
        The primary beam of the imager, the power pattern that gives the clean
        mask and the primary beam correction: ``telescope`` (``"alma"``,
        ``"aca"`` or ``"ngvla"``, the textbook obscured Airy disk with the
        diameters of :func:`~astroviper.utils.beam_models.airy_disk_model`, no
        cut; ``"vla"``, the EVLA beam polynomial of CASA's ``PBMath1DEVLA``,
        with the coefficients CASA picks for each channel (of its two 10052 MHz
        rows, the first), at the exact radius and zero beyond 58 arcmin at
        1 GHz, past the 5 percent power level its polynomials were fitted
        to), ``effective_dish_diameter`` and
        ``blockage_diameter`` (metres; refused for ``"vla"``) and
        ``max_rad_1GHz`` (radians at 1 GHz, scaled as 1 / frequency, beyond
        which the beam is zero; ``None`` is no cut for an Airy disk). Numbers
        given with a telescope override that telescope's values;
        no telescope means ``"alma"``. ``None`` is the ALMA 12 m beam (10.7 m,
        0.75 m, no cut). Or ``beam_model`` alone: an analytic dict
        or a beam-polynomial dataset as ``simulate_processing_set`` takes it in
        ``beam_models`` (``"vla"`` is
        ``read_beam_polynomial_coefficients("EVLA_")``). The beam is the square
        of the voltage pattern the simulator applies for the model, computed by
        the same function
        (:func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`).
        The CASA variants, to compare with the ``.pb`` of ``tclean``, are such
        models: ``airy_disk_model(telescope, func="casa_airy")`` and
        ``read_beam_polynomial_coefficients("EVLA_", n_sample=10000)``.
        Zernike aperture models and Jones images are not supported: they are
        not azimuthally symmetric and depend on the parallactic angle.
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
        ``task_id``, ``n_channels`` and ``n_cycles`` (the model updates).
        With ``exact_psf_subtraction`` (and ``max_iter > 0``) it also holds
        ``T_prep_subtraction_psf``, the time of the PSF of twice the image
        size, which ``T_prep`` includes.
    combined_imaging_dict : ImagingDict
        Per-plane convergence statistics for this chunk: one entry per model
        update and a closing entry from the stop test that ended the loop
        (``iter_done`` 0, ``peakres`` the peak of the written residual inside
        the clean mask).  Channel labels are chunk-local (0-based); the node
        task remaps them to global channel numbers before the reduce.  With
        ``record_iteration_history`` every plane holds, per entry, the
        ``iteration_peak_residual`` and ``iteration_residual_rms`` of its
        model update (one value per iteration; empty for the closing entry);
        with ``max_cycles="auto"`` it holds ``max_cycles_auto``, the limit
        predicted for the plane (the channel's limit is the largest of its
        planes').
    """
    import time

    import numpy as np
    import pandas as pd
    import toolviper.utils.logger as logger

    from astroviper.processing_functions.imaging.deconvolution import (
        validate_exact_psf_subtraction,
        validate_record_iteration_history,
    )
    from astroviper.processing_functions.imaging.make_point_spread_function import (
        make_subtraction_point_spread_function_single_field,
    )
    from astroviper.processing_functions.imaging.model_update import (
        model_update_cube_single_field,
    )
    from astroviper.processing_functions.imaging.residual_update import (
        residual_update_cube_single_field,
    )
    from astroviper.processing_functions.imaging.utils import (
        accumulate_timing,
        build_residual_imaging_dict,
        copy_residual_entropy,
        get_per_cycle_limits,
        merge_imaging_dicts,
        resolve_auto_max_cycles,
    )

    if image_data_variables_keep is None:
        image_data_variables_keep = []
    # Only the Hogbom deconvolvers record the iteration history: refused
    # before any imaging work is done.
    validate_record_iteration_history(
        deconvolver, cycle_control_params.get("record_iteration_history", False)
    )
    exact_psf_subtraction = cycle_control_params.get("exact_psf_subtraction", False)
    validate_exact_psf_subtraction(deconvolver, exact_psf_subtraction)
    # The polarization basis of the imaging weights, which the preparation
    # turns to Stokes on img_xds.
    correlation_coords = img_xds.polarization.values.copy()

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
        cycle_control_params,
        processing_set_data_group_name=processing_set_data_group_name,
        single_precision_image=single_precision_image,
        processing_function_threads=processing_function_threads,
        fft_backend=fft_backend,
        image_data_variables_keep=image_data_variables_keep,
        psf_fitting_method=psf_fitting_method,
        gridding_kernel_params=gridding_kernel_params,
        primary_beam_params=primary_beam_params,
        task_id=task_id,
    )

    # Per-chunk timing accumulator, grouped by pipeline phase. The preparation
    # (setup) sub-timings are namespaced (``T_prep_*``) because several of them
    # (``T_transform_pol``, ``T_fft_norm``, ``T_gcf``, ...) share names with the
    # residual update and would otherwise be summed into one indistinguishable
    # number. Phase totals: T_prep, T_residual_update, T_stop_test,
    # T_model_update, T_restore.
    timing = {"T_prep": T_setup}
    accumulate_timing(timing, setup_return_df, phase="prep")

    # exact_psf_subtraction: the PSF of twice the image size, made once and
    # used by the model updates only for the subtraction.
    subtraction_psf_cube = None
    if exact_psf_subtraction and cycle_control_params["max_iter"] > 0:
        start = time.time()
        subtraction_psf_cube, _ = make_subtraction_point_spread_function_single_field(
            ps_xdt,
            img_xds,
            image_params,
            correlation_coords,
            ms_data_group_in_name=processing_set_data_group_name,
            gridding_kernel_params=gridding_kernel_params,
            processing_function_threads=processing_function_threads,
            fft_backend=fft_backend,
            complex_dtype=np.complex64 if single_precision_image else np.complex128,
        )
        timing["T_prep_subtraction_psf"] = time.time() - start
        timing["T_prep"] += timing["T_prep_subtraction_psf"]

    # Phase totals plus the two model-phase leaves measured here: cycle
    # control and the convergence/merge bookkeeping run in this loop (not inside
    # a processing function), so they are timed inline.
    timing["T_residual_update"] = 0.0
    timing["T_stop_test"] = 0.0
    timing["T_model_update"] = 0.0
    timing["T_cycle_control"] = 0.0
    timing["T_convergence"] = 0.0

    model_exists = False
    n_cycles = 0
    while True:
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
            gridding_kernel_params=gridding_kernel_params,
        )
        timing["T_residual_update"] += time.time() - start
        accumulate_timing(timing, residual_return_df)

        # ---- Stop-test phase ----
        # Cycle control: the stop test on the residual this residual update
        # has just made, never on the residual a model update ends with. The
        # loop ends only here, so every model update is followed by a
        # residual update and the residual written is the one tested last.
        start = time.time()
        residual_imaging_dict = build_residual_imaging_dict(
            img_xds,
            image_data_group_in_name="residual",
            cycle_control_params=cycle_control_params,
        )
        # max_cycles="auto" becomes a number from the dirty image, the
        # residual of the first cycle, and for a plane with threshold 0 from
        # the sensitivity of the imaging weights (a no-op afterwards, and for
        # a number).
        resolve_auto_max_cycles(
            controller,
            residual_imaging_dict,
            ps_xdt,
            img_xds,
            cycle_control_params,
            processing_set_data_group_name=processing_set_data_group_name,
        )
        stopcode, stopdesc = controller.check_convergence(residual_imaging_dict)
        timing["T_stop_test"] += time.time() - start

        # ---- Model-update phase (cycle control + deconvolve + convergence) ----
        model_phase_start = time.time()
        if stopcode.imaging == 0:
            logger.debug("Doing model update")
            # Size the controller's per-plane state to this cube so cycle
            # control (max_iter and threshold) is tracked independently for every
            # (time, frequency, polarization) plane before the deconvolver runs.
            start = time.time()
            controller.ensure_planes(
                img_xds.sizes["time"],
                img_xds.sizes["frequency"],
                img_xds.sizes["polarization"],
            )
            max_iter_per_cycle_updated, threshold_per_cycle = get_per_cycle_limits(
                controller, residual_imaging_dict
            )
            timing["T_cycle_control"] += time.time() - start

            # Per-cycle deconvolution parameters as a fresh dict (the shared
            # cycle_control_params is never mutated). ``threshold`` stays
            # the absolute user stopping threshold (the floor) and
            # ``max_iter_per_cycle`` the user's cap; the per-plane
            # ``max_iter_per_cycle_updated`` (min(max_iter_per_cycle,
            # max_iter_remaining) for every plane) and ``threshold_per_cycle``
            # arrays drive the deconvolver. Without the per-cycle cap one model
            # update would be handed the whole remaining budget and, with
            # threshold_per_cycle 0, spend all of max_iter at once, collapsing
            # the run to one cycle.
            deconvolve_params = {
                **cycle_control_params,
                "max_iter_per_cycle_updated": max_iter_per_cycle_updated,
                "threshold_per_cycle": threshold_per_cycle,
            }
            if subtraction_psf_cube is not None:
                deconvolve_params["subtraction_psf_cube"] = subtraction_psf_cube

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
            # The entropy belongs to the residual this model update started
            # from (a no-op unless the entropy stop is on).
            copy_residual_entropy(imaging_dict, residual_imaging_dict)
            model_exists = True
            n_cycles += 1

            # The model update decides nothing: its iterations, its stop code
            # and the no progress and divergence runs are booked for the stop
            # test after the next residual update.
            start = time.time()
            controller.update_counts(imaging_dict)
            controller.record_model_update(imaging_dict)
        else:
            # The record ends with the stop test that ended the loop.
            start = time.time()
            imaging_dict = residual_imaging_dict
        combined_imaging_dict = merge_imaging_dicts(
            [combined_imaging_dict, imaging_dict]
        )
        timing["T_convergence"] += time.time() - start

        # Model-update phase total: cycle control + deconvolve +
        # convergence (everything done after each stop test).
        timing["T_model_update"] += time.time() - model_phase_start

        if stopcode.imaging != 0:
            logger.debug(f"  *** CONVERGED: {stopdesc} ***")
            break

    # The PSF of twice the image size is freed before the restore.
    subtraction_psf_cube = deconvolve_params = None

    # Restore: convolve the model with the clean beam and add the residual. Only
    # possible once a model exists; the model/residual/beam-fit all live on
    # img_xds at this point. restore_image self-times and returns a one-row
    # timing frame (``T_restore``) folded in like the other steps.
    timing["T_restore"] = 0.0
    # The primary beam corrected restored image is made in the same pass as
    # the restored image: the model is divided by the primary beam before its
    # convolution with the clean beam, which needs the model plane before the
    # restore may overwrite it. The deconvolver's primary_beam_limit is the
    # blanking cutoff when set, else the CASA pblimit default of 0.2.
    timing["T_correct_sky_by_primary_beam"] = 0.0
    correct = (
        primary_beam_correction and restore and cycle_control_params["max_iter"] > 0
    )
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
            primary_beam_correction=correct,
            primary_beam_limit=(
                cycle_control_params.get("primary_beam_limit", 0.0) or 0.2
            ),
        )
        accumulate_timing(timing, restore_return_df)

    timing["task_id"] = task_id
    timing["n_channels"] = img_xds.sizes["frequency"]
    timing["n_cycles"] = n_cycles

    timing_df = pd.DataFrame({key: [value] for key, value in timing.items()})

    return img_xds, timing_df, combined_imaging_dict
