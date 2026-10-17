"""Single source of truth for shared imaging-parameter descriptions.

Every parameter below is spelled out in more than one layer of the imaging
stack (the distributed-graph driver, the node task and the science processing
functions).  Edit the canonical description **here** and run
``python -m astroviper.utils.param_docs sync`` to propagate it into every
function decorated with
:func:`~astroviper.utils.param_docs.shares_param_docs` (CI verifies it with
``... param_docs check``).

Each value is the NumPy-style *description* body only (no ``name : type`` line):
the per-function ``name : type`` line -- which encodes the type and whether the
parameter is optional -- is preserved by the codegen, so the same description
can be shared by functions that give the parameter different defaults.
"""

IMAGING_PARAM_DOCS = {
    "image_params": (
        "Image geometry and output coordinates: ``image_size``, ``cell_size``,\n"
        "``phase_direction``, ``time_coords``, ``polarization_coords`` and the\n"
        "``fft_padding`` gridding/FFT padding factor. ``polarization_coords`` is\n"
        '``["I", "Q"]`` (linear feeds) or ``["I", "V"]`` (circular feeds) to image\n'
        'the two parallel hands, or ``["I", "Q", "U", "V"]`` to image all four\n'
        "correlations (see ``instrument_polarization_basis``)."
    ),
    "imaging_weights_params": (
        'Weighting scheme configuration: ``weighting`` (``"natural"`` or\n'
        '``"briggs"``) and the Briggs ``robust`` parameter.'
    ),
    "gridding_kernel_params": (
        "The gridding convolution kernel the gridder and degridder use:\n"
        '``type`` (``"prolate_spheroidal_casa"``, Schwab\'s rational\n'
        "approximation of the ``m = 6``, ``alpha = 1`` function with its\n"
        'analytic correction, the kernel CASA uses; ``"prolate_spheroidal"``,\n'
        "the generalised prolate spheroidal function of Schwab 1984 with its\n"
        "bandwidth and weight exponent chosen for the image region the\n"
        '``fft_padding`` keeps; ``"exponential_semicircle"``, the modified\n'
        "exponential of a semicircle of Barnett et al. 2019 and Arras et al.\n"
        "2021 shaped for that region; or\n"
        '``"least_misfit"``, the table that minimises the map error over that\n'
        "region for the lookup used, after Ye et al. 2020), ``support`` (the\n"
        "odd number of grid points the kernel spans along one axis; 7 for the\n"
        "``prolate_spheroidal_casa`` kernel, any odd number from 3 for the others),\n"
        "``oversampling`` (samples of the kernel table per uv cell) and\n"
        "``interpolation`` (how the gridders read the table at a stencil\n"
        'point: ``"nearest"`` takes the nearest sample, so the lookup error,\n'
        "and with it the floor the imaging cycle converges to, falls as 1 over\n"
        'the oversampling; ``"linear"`` interpolates the two neighbouring\n'
        "samples and the error falls as 1 over its square). Missing keys take\n"
        'the defaults ``{"type": "prolate_spheroidal_casa", "support": 7,\n'
        '"oversampling": 100, "interpolation": "nearest"}``; ``None`` is all\n'
        "defaults. A ``prolate_spheroidal``, ``exponential_semicircle`` or\n"
        "``least_misfit`` kernel whose image plane correction falls below 1e-6\n"
        "of its centre value at the edge of the image kept (a support too large\n"
        "for the ``fft_padding``, e.g. 17 at 1.05) is refused. The kernel of the\n"
        'former ``gridder="prolate_spheroidal"`` is ``"prolate_spheroidal_casa"``,\n'
        'not ``"prolate_spheroidal"``.'
    ),
    "cgk_1D": (
        "The gridding kernel\n"
        "(:class:`~astroviper.processing_functions.imaging.gridding_convolution_functions.gridding_kernel.GriddingKernel`),\n"
        "or the bare table of the default prolate spheroidal kernel of support 7."
    ),
    "primary_beam_params": (
        "The primary beam of the imager, the power pattern that gives the clean\n"
        'mask and the primary beam correction: ``telescope`` (``"alma"``,\n'
        '``"aca"`` or ``"ngvla"``, the textbook obscured Airy disk with the\n'
        "diameters of :func:`~astroviper.utils.beam_models.airy_disk_model`, no\n"
        'cut; ``"vla"``, the EVLA beam polynomial of CASA\'s ``PBMath1DEVLA``,\n'
        "with the coefficients CASA picks for each channel (of its two 10052 MHz\n"
        "rows, the first), at the exact radius and zero beyond 58 arcmin at\n"
        "1 GHz, past the 5 percent power level its polynomials were fitted\n"
        "to), ``effective_dish_diameter`` and\n"
        '``blockage_diameter`` (metres; refused for ``"vla"``) and\n'
        "``max_rad_1GHz`` (radians at 1 GHz, scaled as 1 / frequency, beyond\n"
        "which the beam is zero; ``None`` is no cut for an Airy disk). Numbers\n"
        "given with a telescope override that telescope's values;\n"
        'no telescope means ``"alma"``. ``None`` is the ALMA 12 m beam (10.7 m,\n'
        "0.75 m, no cut). Or ``beam_model`` alone: an analytic dict\n"
        "or a beam-polynomial dataset as ``simulate_processing_set`` takes it in\n"
        '``beam_models`` (``"vla"`` is\n'
        '``read_beam_polynomial_coefficients("EVLA_")``). The beam is the square\n'
        "of the voltage pattern the simulator applies for the model, computed by\n"
        "the same function\n"
        "(:func:`~astroviper.processing_functions.simulation.antenna_beams.symmetric_beam_response`).\n"
        "The CASA variants, to compare with the ``.pb`` of ``tclean``, are such\n"
        'models: ``airy_disk_model(telescope, func="casa_airy")`` and\n'
        '``read_beam_polynomial_coefficients("EVLA_", n_sample=10000)``.\n'
        "Zernike aperture models and Jones images are not supported: they are\n"
        "not azimuthally symmetric and depend on the parallactic angle."
    ),
    "cycle_control_params": (
        "CLEAN cycle controls. An **imaging cycle** (below simply a cycle)\n"
        "is one **residual update** (degrid the model, form residual\n"
        "visibilities, grid and inverse FFT them into the residual image)\n"
        "followed by one **model update** (deconvolve the residual image into\n"
        "the sky model). Every limit and threshold is applied independently to\n"
        "each ``(time, frequency, polarization)`` plane: a plane stops when it\n"
        "meets its own criterion. The imaging cycle loop runs separately for\n"
        "every frequency channel (the node task images one channel at a time),\n"
        "so a channel's cycles continue until all of its (time, polarization)\n"
        "planes have stopped, and a channel that has stopped does no further\n"
        "residual updates while the others carry on. **Cycle control** is the\n"
        "stop test between the residual update and the model update, which\n"
        "decides whether a plane gets another cycle and sets the limits of its\n"
        "next model update. It is made after every residual update, the first\n"
        "included, on the residual that update has just made, never on the\n"
        "residual a model update ends with, and the residual and restored\n"
        "images written are those of the last residual update. **Iteration\n"
        "control** is the stop test inside the model update, made by the\n"
        "deconvolver at every iteration against those limits. The CASA\n"
        "``tclean`` equivalent is given in brackets.\n"
        "Keys:\n"
        "\n"
        "- ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution\n"
        "  iterations (flux components) per plane, summed over all cycles. A\n"
        "  plane stops once it has spent this budget. ``max_iter = 0`` makes\n"
        "  only the dirty image (no deconvolution). *Differs from CASA*: CASA's\n"
        "  ``niter`` is one budget for the whole image; here every plane gets\n"
        "  the full value, and no budget is shared or split between planes.\n"
        "- ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.\n"
        "  ``max_cycles = N`` performs at most ``N`` model updates and\n"
        "  ``N + 1`` residual updates: the first makes the dirty image, and\n"
        "  every model update is followed by a residual update, whose residual\n"
        "  the stop test checks; the last one's residual is written.\n"
        "  ``max_cycles = 0`` makes only the dirty image; ``max_cycles = -1``\n"
        "  removes the limit. Counted per\n"
        "  frequency channel: a channel that converges early stops cycling while\n"
        '  the others continue. ``max_cycles = "auto"`` predicts the limit of a\n'
        "  channel from its dirty image: every plane takes ``P0``, its peak\n"
        "  residual at the start of the first cycle (the ``peak_residual``\n"
        "  below), ``f``, its PSF fraction (below), and ``L``, its threshold or,\n"
        "  when that is 0, its theoretical point source sensitivity, worked out\n"
        "  from the imaging weights as the ``SENSITIVITY`` of\n"
        "  ``point_spread_function_cube_single_field``;\n"
        "  ``n = ceil(ln(P0 / L) / ln(1 / f))`` cycles (at least 1) bring the\n"
        "  peak down to ``L``, and the plane's limit is ``n + 3`` against a\n"
        "  threshold and ``3 * n`` against the sensitivity (``EXTRA_CYCLES``\n"
        "  and ``NOISY_CYCLE_SAFETY`` of\n"
        "  :mod:`~astroviper.processing_functions.imaging.utils.cycle_control_heuristics`).\n"
        "  The sensitivity is the thermal noise of the plane when the weights\n"
        "  are absolute (``WEIGHT = 1 / sigma**2``). With unit weights, as a\n"
        "  simulation without noise writes them, it is not a noise level but\n"
        "  about 1 over the square root of the number of visibilities (natural\n"
        "  weighting; 1e-3 for a million) in the units of the data; the limit\n"
        "  grows only with ``ln(P0 / L)``, and on the same prediction ``3 * n``\n"
        "  cycles bring the peak down to ``P0 * (L / P0)**3``. The channel's\n"
        "  ``max_cycles`` is the largest limit of its planes (at least 1; a\n"
        "  plane without a pixel in the clean mask, with a peak of 0 or without\n"
        "  data is left out) and then counts down like a number. Every plane\n"
        "  records its limit as ``max_cycles_auto`` in the returned convergence\n"
        "  record. Needs ``max_psf_fraction < 1``.\n"
        "- ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a\n"
        "  float in Jy, or a list of one float per Stokes plane in the order of\n"
        "  ``polarization_coords`` (the floors of the Stokes planes of noise free\n"
        "  data scale with their fluxes). The distributed application and the\n"
        "  node task also take a nested list of shape (n_frequency,\n"
        "  n_polarization), one row per image channel in the order of the image\n"
        "  frequency coordinate; each channel gets its row. A plane stops when the\n"
        "  peak of its residual inside the clean mask, as the latest residual\n"
        "  update made it, falls to or below its threshold times\n"
        "  ``1 + threshold_tolerance``; the threshold is\n"
        "  also a hard floor on ``threshold_per_cycle``. ``0`` disables the\n"
        "  absolute stop. *Differs from CASA*: floats in Jy only, no ``'1mJy'``\n"
        "  strings.\n"
        "- ``threshold_tolerance`` : Relative tolerance of the threshold stop,\n"
        "  a finite number ``>= 0``: a plane whose threshold is above 0 stops\n"
        "  with imaging stop code 2, and gets no iterations while other planes of\n"
        "  its channel go on, once the peak of its residual inside the clean mask\n"
        "  is at most ``threshold * (1 + threshold_tolerance)``; the model update\n"
        "  of a plane that goes on still cleans down to ``threshold_per_cycle``,\n"
        "  floored at the threshold itself. Default 0. No CASA equivalent.\n"
        "- ``threshold_sigma`` [CASA ``nsigma``] : Noise based stopping threshold\n"
        "  per plane, as a multiple of the plane's robust residual rms\n"
        "  (``1.4826 * MAD``). The effective threshold of a plane is\n"
        "  ``max(threshold, threshold_sigma * rms)`` and it floors\n"
        "  ``threshold_per_cycle`` in the same way. ``0`` disables it. Reserved:\n"
        "  accepted but not yet implemented.\n"
        "- ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary beam\n"
        "  mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.\n"
        "  Pixels where the primary beam is below this fraction are excluded\n"
        "  from cleaning. A masking cutoff, distinct from ``threshold``.\n"
        "- ``gain`` [CASA ``gain``] : CLEAN loop gain, the fraction of the\n"
        "  selected peak flux subtracted from the residual image at each\n"
        "  deconvolution iteration (``0 < gain <= 1``).\n"
        "- ``psf_sidelobe_factor`` [CASA ``cyclefactor``] : Multiplier applied to\n"
        "  the measured peak PSF sidelobe level (``max_psf_sidelobe``) when\n"
        "  setting how deep one model update cleans (see\n"
        "  ``threshold_per_cycle``). Larger values trigger the next residual\n"
        "  update sooner; smaller values clean deeper before each residual\n"
        "  update.\n"
        "- ``max_iter_per_cycle`` [CASA ``cycleniter``] : Maximum number of\n"
        "  deconvolution iterations a plane may run in one cycle's model update\n"
        "  before the next residual update is triggered. ``max_iter_per_cycle =\n"
        "  -1`` lets the adaptive ``threshold_per_cycle`` govern the depth\n"
        "  instead; otherwise the count is clamped to never exceed the plane's\n"
        "  remaining ``max_iter``.\n"
        "- ``min_psf_fraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF\n"
        "  fraction defined below. Raising it limits how deep a single model\n"
        "  update cleans.\n"
        "- ``max_psf_fraction`` [CASA ``maxpsffraction``] : Upper clamp on the\n"
        "  same PSF fraction; it guarantees a minimum amount of cleaning per\n"
        "  model update even when the PSF sidelobe level is high.\n"
        "- ``max_iter_divergence`` : Divergence test of the model update.\n"
        "  Number of consecutive deconvolution iterations the RMS of\n"
        "  a plane's residual, taken over the clean mask, may be above\n"
        "  ``(1 + gain / 10)`` times the lowest RMS it has reached in the model\n"
        "  update before that model update is stopped as diverged. For Hogbom\n"
        "  a peak above ``(1 + gain)`` times the peak at the start of the\n"
        "  model update stops it at once. Asp, whose least-squares components\n"
        "  can raise the peak of a healthy residual, has no peak limit and\n"
        "  allows ``(1 + gain)`` times the lowest RMS instead. An infinite peak\n"
        "  always stops the model update. The next residual update then\n"
        "  recomputes the true residual, the stop test is made on it, and the\n"
        "  cycles go on (see ``max_cycles_divergence``). Default 1, the first\n"
        "  such iteration; ``-1`` disables the two limits.\n"
        "- ``max_cycles_divergence`` : Divergence stop of the cycle control. A\n"
        "  plane whose last ``max_cycles_divergence`` model updates all ended\n"
        "  diverged (see ``max_iter_divergence``) gets no more iterations and\n"
        "  is stopped with imaging stop code 5 by the stop test after the next\n"
        "  residual update (unless a stop that takes precedence applies there:\n"
        "  an empty mask, ``max_iter`` spent, the threshold or the entropy\n"
        "  stop, codes 7, 1, 2 and 10), while the other planes of its channel\n"
        "  carry on.\n"
        "  Default ``-1``, off.\n"
        "- ``entropy_stop`` : If ``True``, a plane stops once the entropy of\n"
        "  its residual has passed its maximum. The entropy (Homan, Roth and\n"
        "  Pushkarev 2024, AJ 167, 11) measures how much the residual looks\n"
        "  like noise everywhere. It rises while the clean removes emission\n"
        "  and falls once the clean fits noise. It is worked out after every\n"
        "  residual update, and the plane stops when it is lower than in an\n"
        "  earlier cycle. The fall is noticed one model update after the\n"
        "  maximum and the model of that model update is kept, so a small\n"
        "  ``max_iter_per_cycle`` makes the stop sharper. Default ``False``.\n"
        "  No CASA equivalent.\n"
        "- ``entropy_max_snr`` : The entropy of a plane is followed once the\n"
        "  peak of its residual inside the clean mask is at most this many\n"
        "  times the RMS of the residual. Above it the residual is dominated\n"
        "  by the pattern of the point spread function. Default 6.\n"
        "- ``entropy_spatial_bins`` : Number of spatial bins along each of the\n"
        "  two image axes used for the entropy. Default 7.\n"
        "- ``entropy_flux_bins`` : Number of flux bins per unit of RMS used for\n"
        "  the entropy. Default 10.\n"
        "- ``record_iteration_history`` : If ``True``, every model update\n"
        "  records, for every plane and every deconvolution iteration, the peak\n"
        "  residual the iteration cleaned (signed, in the units of the residual\n"
        "  image) and the RMS of the residual inside the clean mask after its\n"
        "  subtraction, to follow the deconvolution in detail. The returned\n"
        "  convergence record holds them as ``iteration_peak_residual`` and\n"
        "  ``iteration_residual_rms``: per plane one array per cycle, with one\n"
        "  value per iteration of that cycle's model update (empty for a cycle\n"
        "  without one), so the record grows by two float64 values per\n"
        "  iteration done. A debugging and tuning tool: the images are the same\n"
        "  to the bit with and without it, and the RMS is the sum of squares\n"
        "  the peak search of the next iteration makes anyway, so it costs no\n"
        "  extra pass over the image per iteration. Hogbom deconvolvers only:\n"
        '  ``deconvolver="asp"`` raises ``ValueError``. Default ``False``.\n'
        "  No CASA equivalent.\n"
        "- ``exact_psf_subtraction`` : If ``True``, a Hogbom model update\n"
        "  subtracts the PSF of every component it finds from the whole residual\n"
        "  image, with a PSF of twice the image size made once per channel from\n"
        "  the same imaging weights, cell size and gridding kernel. With\n"
        "  ``False`` it subtracts the image-sized PSF, which reaches about\n"
        "  ``n/2`` pixels either side of the component along each axis\n"
        "  (``n`` the image size in pixels), so farther pixels keep the\n"
        "  sidelobes of the flux cleaned until the next residual update. The\n"
        "  beam fit, ``max_psf_sidelobe`` and the PSF written use the\n"
        "  image-sized PSF either way. Costs a second PSF per channel (four\n"
        "  times the pixels, kept through the channel's cycles) and a\n"
        "  subtraction over the whole image at every iteration. Hogbom\n"
        '  deconvolvers only: ``deconvolver="asp"`` raises ``ValueError``.\n'
        "  Default ``False``. No CASA equivalent.\n"
        "\n"
        "A plane whose model updates do no iteration any more (two in a row)\n"
        "is stopped with the no progress stop code at the next stop test, so\n"
        "an all-zero plane cannot keep its channel cycling.\n"
        "\n"
        "Derived per plane before each model update (not set by the caller):\n"
        "``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,\n"
        "min_psf_fraction, max_psf_fraction)`` is the fraction of the current\n"
        "peak residual down to which one model update cleans, and\n"
        "``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``\n"
        "is the stopping threshold of that model update, where\n"
        "``peak_residual`` is the plane's peak residual inside the clean mask\n"
        "after the residual update of the same cycle (the peak of the stop\n"
        "test), and\n"
        "``max_iter_per_cycle_updated = min(max_iter_per_cycle,\n"
        "max_iter_remaining)`` is the iteration limit of that model update (0\n"
        "for a plane the entropy or divergence stop has ended), where\n"
        "``max_iter_remaining`` is what is left of ``max_iter``.\n"
    ),
    "processing_set_data_group_name": (
        'Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).'
    ),
    "deconvolver": (
        "Deconvolution algorithm for the model update. One of ``"
        '"hogbom"`` (C++, threaded across planes), ``"hogbom_many_threads"``\n'
        "(C++, threaded across *and* within planes -- faster when there are\n"
        'few planes, e.g. single-channel imaging) or ``"asp"``.'
    ),
    "instrument_polarization_basis": (
        "Correlation (instrument) polarization basis the gridding is performed in:\n"
        '``"linear"`` or ``"circular"``. The residual update grids and degrids the\n'
        "correlations of this basis and the model update deconvolves in the\n"
        "Stokes basis, in which the image is written. The Stokes planes requested\n"
        'in ``image_params["polarization_coords"]`` fix the correlations that are\n'
        "loaded and gridded: the two parallel hands give ``I, Q`` (linear) or\n"
        "``I, V`` (circular), all four correlations give ``I, Q, U, V``. A sample\n"
        "is used only if none of its loaded correlations is flagged."
    ),
    "single_precision_image": (
        "If ``True`` the image-domain arrays (gridded uv grids and sky/PSF/model\n"
        "images) are single precision (``complex64`` / ``float32``) and the model\n"
        "update runs in single precision; the visibilities always stay double\n"
        "precision. If ``False`` the image-domain arrays are double precision."
    ),
    "processing_function_threads": (
        "Number of threads handed to the per-processing-function (C++ / FFT)\nkernels."
    ),
    "fft_backend": (
        'FFT backend used by the gridder normalization (``"pyfftw"`` or\n``"scipy"``).'
    ),
    "image_data_variables_keep": (
        'Logical image-variable keys to retain on disk (e.g. ``"sky_residual"``,\n'
        '``"sky_model"``, ``"point_spread_function"``, ``"primary_beam"``).'
    ),
    "primary_beam_correction": (
        "If ``True`` write the primary beam corrected restored sky to the\n"
        "``sky_restored_primary_beam_corrected``\n"
        "(``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``) variable: the model divided\n"
        "by the (power) primary beam and convolved with the clean beam, plus\n"
        "the residual divided by the primary beam; pixels below the primary\n"
        "beam cutoff (``primary_beam_limit``) are blanked with NaN.  Requires\n"
        "``restore``."
    ),
    "psf_fitting_method": (
        'Beam-fit algorithm for the PSF: ``"astroviper"`` (default) or\n'
        '``"casa"``, the C++ port of CASA\'s ``StokesImageUtil::FitGaussianPSF``\n'
        "(the fit behind ``tclean``'s restoring beam)."
    ),
    "restore": (
        "If ``True`` produce a restored image after deconvolution: the model\n"
        "convolved with the clean beam (the Gaussian fit to the PSF) plus the\n"
        "residual, written to the ``sky_restored`` (``SKY_RESTORED``) variable."
    ),
    "image_store": "Path/URL of the on-disk Zarr image cube.",
    # Shared by every distributed application that makes a report.
    "report": (
        'Make a report of the run: ``"html"`` (one self contained file) or\n'
        '``"pdf"``, written next to the output store as\n'
        "``<name>_report.html`` or ``<name>_report.pdf`` (``cube.img.zarr``\n"
        'gives ``cube_report.html``), or ``"notebook"``, displayed in the\n'
        "calling Jupyter notebook. The report has a processing section common\n"
        "to all distributed applications (timing, task stream and, with\n"
        "``monitor_resources_seconds``, the cluster CPU, memory and I/O) and a\n"
        "results section of the application's own; the products it makes with\n"
        "other distributed applications are kept in ``<name>_report_files``\n"
        "next to the output store. It is made after the output is written: a\n"
        "failure while making it is logged and leaves the results intact.\n"
        "``None`` (default): no report."
    ),
    "task_id": "Identifier of the parallel chunk being processed.",
    "task_coords": (
        "Per-chunk coordinate mapping; ``task_coords[<parallel dim>]`` supplies\n"
        'this chunk\'s parallel coordinate values (``"data"``) and its\n'
        '``"slice"`` into the full output array (for cube imaging the\n'
        "parallel dim is ``frequency``)."
    ),
}
