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
        "CLEAN iteration controls. Iteration control is performed\n"
        "**independently per** ``(time, frequency, polarization)`` **plane**: each\n"
        "plane carries its own iteration budget and stopping thresholds, and the\n"
        "residual update cycle loop continues until *every* selected plane has\n"
        "stopped.\n"
        "\n"
        "Terminology: a **residual update cycle** recomputes the residual image\n"
        "from the visibilities; a **model update cycle** deconvolves that residual\n"
        "into the sky model (CASA calls these the major and minor cycle).\n"
        "\n"
        "Keys, with the CASA ``tclean`` equivalent in brackets:\n"
        "\n"
        "- ``niter_per_plane`` [CASA ``niter``] : Maximum number of CLEAN\n"
        "  iterations (flux components) for **one plane**, summed over all\n"
        "  residual update cycles. A plane stops once it has spent this budget.\n"
        "  ``niter_per_plane=0`` makes only the dirty image (no deconvolution).\n"
        "  *Differs from CASA*: CASA's ``niter`` is one budget for the whole\n"
        "  image; here every plane gets the full value, and no budget is shared\n"
        "  or split between planes.\n"
        "- ``nmajor`` [CASA ``nmajor``] : Maximum number of deconvolving residual\n"
        "  update cycles. ``nmajor=N`` performs ``N`` deconvolutions -- the dirty\n"
        "  image is computed inside the first such cycle, matching CASA -- and\n"
        "  ``nmajor=-1`` removes the limit. Shared across planes (not tracked per\n"
        "  plane), unlike ``niter_per_plane``.\n"
        "- ``threshold`` [CASA ``threshold``] : Absolute stopping threshold, given\n"
        "  as a float in Jy. A plane stops when its peak residual inside the clean\n"
        "  mask falls to or below ``threshold``; the value is also a hard floor on\n"
        "  ``cycle_threshold`` (below). ``threshold=0`` disables the absolute stop.\n"
        "  *Differs from CASA*: a float in Jy only -- no ``'1mJy'`` strings.\n"
        "- ``primary_beam_limit`` [CASA ``pblimit`` / ``pbmask``] : Primary-beam\n"
        "  mask cutoff as a fraction of the peak primary beam, in ``[0, 1]``.\n"
        "  Pixels where the primary beam is below this fraction are excluded from\n"
        "  cleaning. A masking cutoff, distinct from ``threshold``.\n"
        "- ``gain`` [CASA ``gain``] : CLEAN loop gain -- the fraction of the\n"
        "  selected peak flux subtracted from the residual image each iteration\n"
        "  (``0 < gain <= 1``).\n"
        "- ``cycle_factor`` [CASA ``cyclefactor``] : Scaling applied to the\n"
        "  brightest PSF sidelobe level when setting the model update cycle\n"
        "  stopping depth (see ``cycle_threshold`` below). Larger values trigger\n"
        "  the next residual update sooner; smaller values clean deeper first.\n"
        "- ``cycle_niter`` [CASA ``cycleniter``] : Maximum number of iterations a\n"
        "  plane may run in one model update cycle before a residual update is\n"
        "  triggered. ``cycle_niter=-1`` lets the adaptive ``cycle_threshold``\n"
        "  govern the depth instead; otherwise the count is clamped to never\n"
        "  exceed the plane's remaining ``niter_per_plane``.\n"
        "- ``minpsffraction`` [CASA ``minpsffraction``] : Lower clamp on the PSF\n"
        "  fraction used to set ``cycle_threshold = clamp(max_psf_sidelobe *\n"
        "  cycle_factor, minpsffraction, maxpsffraction) * peak_residual`` (then\n"
        "  floored at ``threshold``). Raising it limits how deep one model update\n"
        "  cycle cleans.\n"
        "- ``maxpsffraction`` [CASA ``maxpsffraction``] : Upper clamp on that same\n"
        "  PSF fraction; it guarantees a minimum amount of cleaning per model\n"
        "  update cycle even when the PSF sidelobe level is high.\n"
        "\n"
        "Two derived quantities appear in the deconvolution parameter dict and are\n"
        "computed, not set by the caller: ``cycle_niter_cap``\n"
        "(``min(cycle_niter, remaining niter_per_plane)``) and its per-plane array\n"
        "``cycle_niter_cap_pp``, alongside ``cycle_threshold_pp``.\n"
    ),
    "processing_set_data_group_name": (
        'Measurement-set data group to image (e.g. ``"base"`` or ``"corrected"``).'
    ),
    "deconvolver": (
        "Deconvolution algorithm for the minor cycle. One of ``"
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
        "images) are single precision (``complex64`` / ``float32``) and the minor\n"
        "cycle runs in single precision; the visibilities always stay double\n"
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
