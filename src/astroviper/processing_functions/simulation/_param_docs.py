"""Single source of truth for shared simulation-parameter descriptions.

Parameters that appear in more than one layer of the simulation stack (the
distributed application, the node task and the processing function) are
described **here**; ``python -m astroviper.utils.param_docs sync`` propagates the
descriptions into every function decorated with
:func:`~astroviper.utils.param_docs.shares_param_docs` (see
``processing_functions/imaging/_param_docs.py`` for the pattern).
"""

SIMULATION_PARAM_DOCS = {
    "polarization": (
        "MSv4 polarization labels to simulate, a subset of one instrumental basis\n"
        '(``["RR", "RL", "LR", "LL"]`` or ``["XX", "XY", "YX", "YY"]``).'
    ),
    "point_source_flux": (
        "Flux of every point source in the four instrumental correlations\n"
        "(``RR, RL, LR, LL`` or ``XX, XY, YX, YY``); singleton time/frequency axes broadcast."
    ),
    "point_source_ra_dec": (
        "Right ascension and declination of the point sources (per time or fixed)."
    ),
    "gaussian_source_flux": (
        "Integrated flux of each Gaussian source in the four instrumental\n"
        "correlations; singleton time/frequency axes broadcast.  ``None``\n"
        "(default) simulates no Gaussian sources."
    ),
    "gaussian_source_ra_dec": (
        "Right ascension and declination of the Gaussian sources (per time or fixed)."
    ),
    "gaussian_source_shape": (
        "``[major, minor, position angle]`` FWHM shape of each Gaussian source, in\n"
        "the imaging clean-beam convention\n"
        "(:func:`astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper`)."
    ),
    "phase_center_ra_dec": (
        "Phase centre of the array per time (time-varying for mosaics) or fixed."
    ),
    "beam_models": (
        "Antenna beam models: analytic dicts, aperture (Zernike) coefficient\n"
        "datasets, beam polynomial datasets or Jones image datasets\n"
        "(see ``astroviper.utils.beam_models``)."
    ),
    "beam_model_map": "Index into ``beam_models`` for each antenna.",
    "beam_params": (
        "Beam evaluation parameters: ``mueller_selection`` (row-major indices of the\n"
        "4x4 Mueller elements to apply, default ``[0, 5, 10, 15]``), ``pa_radius``\n"
        "(rad; parallactic-angle spacing of the Zernike beam images, default 0.2),\n"
        "``image_size`` (Zernike beam image size, default ``[1000, 1000]``),\n"
        "``fov_scaling`` (beam image extent in units of the beam cut radius,\n"
        'default 4) and ``zernike_freq_interp`` (default ``"nearest"``).'
    ),
    "pointing_ra_dec": (
        "Antenna pointing directions; ``None`` points every antenna at the phase centre."
    ),
    "uvw_params": (
        "``auto_correlations`` (bool, default False) and ``uvw_convention``\n"
        '(``"msv4"`` = antenna2 - antenna1, default; or ``"sirius"``).'
    ),
    "noise_params": (
        "Thermal-noise system parameters (``casatools.simulator.setnoise`` tsys-manual\n"
        "model): ``t_receiver``, ``t_atmos``, ``tau``, ``ant_efficiency``,\n"
        "``spill_efficiency``, ``corr_efficiency``, ``quantization_efficiency``,\n"
        "``t_cmb`` and ``random_seed``; ``None`` disables noise (unit weights)."
    ),
    "direction_frame": (
        'Astropy frame of all right ascension / declination inputs (``"icrs"`` or ``"fk5"``).'
    ),
    "implementation": (
        'Visibility kernel implementation: ``"cpp"`` (multithreaded C++, default) or\n'
        '``"numpy"`` (vectorised NumPy reference).'
    ),
}
