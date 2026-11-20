"""Single source of truth for shared visibility-manipulation parameter descriptions.

Parameters that appear in more than one layer of a visibility-manipulation
application (the distributed application, the node task and the processing
function) are described **here**; ``python -m astroviper.utils.param_docs sync``
propagates the descriptions into every function decorated with
:func:`~astroviper.utils.param_docs.shares_param_docs` (see
``processing_functions/imaging/_param_docs.py`` for the pattern). The names
are shared with the imaging functions, so every key is scoped to its function:
``"<function name>.<parameter name>"``.
"""

_UV_CONTINUUM_SUBTRACTION = {
    "continuum_frequency_ranges": (
        "The continuum ranges, ``[low, high]`` pairs of frequencies (Hz) in\n"
        "``continuum_frequency_frame``; a channel is a continuum channel at the\n"
        "times its frequency, converted to that frame, lies in a range, edges\n"
        "included (the ``continuum_frequency_ranges`` of ``find_continuum_ultra``\n"
        "are the outer edges of its continuum channels)."
    ),
    "continuum_frequency_frame": (
        "Spectral frame of ``continuum_frequency_ranges``, by its XRADIO\n"
        'observer name: the frame of the measurement set, or, for a ``"TOPO"``\n'
        'measurement set, ``"lsrk"``, ``"lsrd"`` or ``"BARY"``, to which the\n'
        "channel frequencies are converted at every time with the Doppler factor\n"
        "of ``ms_spectral_frame_conversion.doppler_factor`` (the telescope site\n"
        "at rest on the Earth, the phase centre of the first field)."
    ),
    "fit_order": (
        "Order of the polynomial fitted to the continuum channels of every\n"
        "spectrum [CASA ``fitorder``]; lowered to ``n - 1`` for a spectrum with\n"
        "``n <= fit_order`` valid continuum channels."
    ),
    "ms_data_group_in_name": (
        "Measurement-set data group whose correlated data are continuum\n"
        "subtracted, with its flags and weights."
    ),
    "ms_data_group_out_name": (
        "Measurement-set data group of the continuum-subtracted visibilities."
    ),
    "ms_data_group_out_modified": (
        "The output variables of the roles ``correlated_data`` and ``flag``;\n"
        '``None`` gives ``{"correlated_data": "VISIBILITY_CONTINUUM_SUBTRACTED",\n'
        '"flag": "FLAG_CONTINUUM_SUBTRACTED"}``. Neither may be a variable of\n'
        "another data group, the input group included."
    ),
    "input_checked": (
        "Skip the configuration checks done by the calling layer. The checks of\n"
        "the measurement sets always run."
    ),
    "report": (
        'Make a report of the run: ``"html"`` (one self contained file) or\n'
        '``"pdf"``, written next to the processing set as\n'
        "``<name>_<spectral_window_name>_report.html`` or ``.pdf``, ``<name>``\n"
        "the processing set without its store suffix (``demo.ps.zarr`` and\n"
        '``spw_0`` give ``demo_spw_0_report.html``), or ``"notebook"``, displayed\n'
        "in the calling Jupyter notebook. The report has a processing section\n"
        "common to all distributed applications (performance measures, timing,\n"
        "task stream, task times and, with ``monitor_resources_seconds``, the\n"
        "CPU, memory and I/O of the cluster, of every worker slot and of the\n"
        "tasks) and a results section of the application's own (the fit, the\n"
        "continuum channels and fit statistics of every MSv4, and the mean\n"
        "spectra before and after). It is made after the output is written: a\n"
        "failure while making it is logged and leaves the results intact.\n"
        "``None`` (default): no report."
    ),
    "task_coords": (
        'GraphVIPER task coordinates; ``task_coords["time"]["slice"]`` is the\n'
        "task's block of times of the MSv4."
    ),
}

VISIBILITY_MANIPULATION_PARAM_DOCS = {
    f"uv_continuum_subtraction.{name}": description
    for name, description in _UV_CONTINUUM_SUBTRACTION.items()
}
