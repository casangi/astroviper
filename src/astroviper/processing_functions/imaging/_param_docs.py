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
        "CLEAN minor/major-cycle iteration controls, matching the meaning of the\n"
        "corresponding CASA ``tclean`` parameters. Iteration control is performed\n"
        "**independently per** ``(time, frequency, polarization)`` **plane**: each\n"
        "plane carries its own iteration budget and stopping thresholds, and the\n"
        "major-cycle loop continues until *every* selected plane has stopped --\n"
        "the one deliberate difference from CASA, whose ``niter`` budget is global\n"
        "across the image. Keys:\n"
        "\n"
        "- ``niter`` : Maximum number of minor-cycle CLEAN iterations (flux\n"
        "  components) per plane, summed over all major cycles. A plane stops once\n"
        "  it has spent this budget; ``niter=0`` makes only the dirty image (no\n"
        "  deconvolution).\n"
        "- ``nmajor`` : Maximum number of deconvolving major cycles (each a\n"
        "  residual update followed by a minor cycle). ``nmajor=N`` performs ``N``\n"
        "  deconvolutions -- the dirty image is computed inside the first such\n"
        "  cycle, matching CASA's ``nmajor`` -- and ``nmajor=-1`` removes the\n"
        "  major-cycle limit. Shared across planes (not tracked per plane).\n"
        "- ``threshold`` : Absolute stopping threshold, given as a float in Jy. A\n"
        "  plane stops when its peak residual inside the clean mask falls to or\n"
        "  below ``threshold``; the value is also a hard floor on the\n"
        "  per-minor-cycle ``cyclethreshold`` (below). ``threshold=0`` disables\n"
        "  the absolute stop.\n"
        "- ``primary_beam_limit`` : Primary-beam mask cutoff as a fraction of the\n"
        "  peak primary beam, in ``[0, 1]`` (the analogue of CASA's ``pblimit`` /\n"
        "  ``pbmask``). Pixels where the primary beam is below this fraction are\n"
        "  excluded from cleaning. A masking cutoff, distinct from ``threshold``.\n"
        "- ``gain`` : CLEAN loop gain -- the fraction of the selected peak flux\n"
        "  subtracted from the residual image each minor iteration\n"
        "  (``0 < gain <= 1``).\n"
        "- ``cyclefactor`` : Scaling applied to the brightest PSF sidelobe level\n"
        "  when setting the minor-cycle stopping depth (see ``cyclethreshold``\n"
        "  below). Larger values trigger the next major cycle sooner; smaller\n"
        "  values clean deeper before each residual update.\n"
        "- ``cycleniter`` : Maximum number of minor-cycle iterations a plane may\n"
        "  run before a major cycle is triggered. ``cycleniter=-1`` lets the\n"
        "  adaptive ``cyclethreshold`` govern the depth instead; otherwise the\n"
        "  count is clamped to never exceed the plane's remaining ``niter``.\n"
        "- ``minpsffraction`` : Lower clamp on the PSF fraction used to set the\n"
        "  minor-cycle threshold ``cyclethreshold = clamp(max_psf_sidelobe *\n"
        "  cyclefactor, minpsffraction, maxpsffraction) * peak_residual`` (then\n"
        "  floored at ``threshold``). Raising it limits how deep a single minor\n"
        "  cycle cleans.\n"
        "- ``maxpsffraction`` : Upper clamp on that same PSF fraction; it\n"
        "  guarantees a minimum amount of cleaning per minor cycle even when the\n"
        "  PSF sidelobe level is high.\n"
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
    "primary_beam_correction": (
        "If ``True`` divide the restored sky by the (power) primary beam,\n"
        "writing the ``sky_restored_primary_beam_corrected``\n"
        "(``SKY_RESTORED_PRIMARY_BEAM_CORRECTED``) variable (CASA ``pbcor``);\n"
        "pixels below the primary-beam cutoff are blanked with NaN.  Requires\n"
        "``restore``."
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
