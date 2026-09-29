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
        "``fft_padding`` gridding/FFT padding factor."
    ),
    "imaging_weights_params": (
        'Weighting scheme configuration: ``weighting`` (``"natural"`` or\n'
        '``"briggs"``) and the Briggs ``robust`` parameter.'
    ),
    "iteration_control_params": (
        "CLEAN iteration controls. An **imaging cycle** (below simply a cycle)\n"
        "is one **residual update** (degrid the model, form residual\n"
        "visibilities, grid and inverse FFT them into the residual image)\n"
        "followed by one **model update** (deconvolve the residual image into\n"
        "the sky model). Every limit and threshold is applied independently to\n"
        "each ``(time, frequency, polarization)`` plane: a plane stops when it\n"
        "meets its own criterion. The imaging cycle loop runs separately for\n"
        "every frequency channel (the node task images one channel at a time),\n"
        "so a channel's cycles continue until all of its (time, polarization)\n"
        "planes have stopped, and a channel that has stopped does no further\n"
        "residual updates while the others carry on. The CASA ``tclean``\n"
        "equivalent is given in brackets. Keys:\n"
        "\n"
        "- ``max_iter`` [CASA ``niter``] : Maximum number of deconvolution\n"
        "  iterations (flux components) per plane, summed over all cycles. A\n"
        "  plane stops once it has spent this budget. ``max_iter = 0`` makes\n"
        "  only the dirty image (no deconvolution). *Differs from CASA*: CASA's\n"
        "  ``niter`` is one budget for the whole image; here every plane gets\n"
        "  the full value, and no budget is shared or split between planes.\n"
        "- ``max_cycles`` [CASA ``nmajor``] : Maximum number of cycles.\n"
        "  ``max_cycles = N`` performs ``N`` model updates; the dirty image is\n"
        "  made by the residual update of the first cycle, and a closing\n"
        "  residual update follows the last model update so that the written\n"
        "  residual reflects the final model. ``max_cycles = 0`` makes only the\n"
        "  dirty image; ``max_cycles = -1`` removes the limit. Counted per\n"
        "  frequency channel: a channel that converges early stops cycling while\n"
        "  the others continue.\n"
        "- ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, as a\n"
        "  float in Jy. A plane stops when its peak residual inside the clean\n"
        "  mask falls to or below ``threshold``; the value is also a hard floor\n"
        "  on ``threshold_per_cycle``. ``threshold = 0`` disables the absolute\n"
        "  stop. *Differs from CASA*: a float in Jy only, no ``'1mJy'`` strings.\n"
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
        "- ``max_iter_divergence`` : Number of consecutive deconvolution\n"
        "  iterations a plane's peak residual may stay above ``(1 + gain / 2)``\n"
        "  times the lowest peak it has reached in the model update before that\n"
        "  model update is stopped as diverged (Hogbom). A peak above\n"
        "  ``(1 + gain)`` times the peak at the start of the model update, or a\n"
        "  peak that is not finite, stops it at once. The next residual update\n"
        "  then recomputes the true residual. Default 30; ``-1`` disables the\n"
        "  test. *Differs from CASA*, which tests a fixed 10 percent rise once\n"
        "  every 2000 iterations.\n"
        "\n"
        "A plane whose model updates do no iteration any more (two in a row)\n"
        "is stopped with the no progress stop code, so an all-zero plane cannot\n"
        "keep its channel cycling.\n"
        "\n"
        "Derived per plane before each model update (not set by the caller):\n"
        "``psf_fraction = clamp(max_psf_sidelobe * psf_sidelobe_factor,\n"
        "min_psf_fraction, max_psf_fraction)`` is the fraction of the current\n"
        "peak residual down to which one model update cleans, and\n"
        "``threshold_per_cycle = max(psf_fraction * peak_residual, threshold)``\n"
        "is the stopping threshold of that model update, where\n"
        "``peak_residual`` is the plane's peak residual inside the mask at the\n"
        "start of the cycle. The deconvolver also receives the per-plane\n"
        "``max_iter_per_cycle``, ``min(max_iter_per_cycle, remaining max_iter)``.\n"
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
        '``"linear"`` (``XX``/``YY``) or ``"circular"`` (``RR``/``LL``). The\n'
        "output image is always produced in the Stokes basis."
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
        "If ``True`` divide the restored sky by the (power) primary beam,\n"
        "writing the ``sky_restored_primary_beam_corrected``\n"
        "(``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``) variable (CASA ``pbcor``);\n"
        "pixels below the primary-beam cutoff are blanked with NaN.  Requires\n"
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
    "task_id": "Identifier of the parallel chunk being processed.",
    "task_coords": (
        "Per-chunk coordinate mapping; ``task_coords[<parallel dim>]`` supplies\n"
        'this chunk\'s parallel coordinate values (``"data"``) and its\n'
        '``"slice"`` into the full output array (for cube imaging the\n'
        "parallel dim is ``frequency``)."
    ),
}
