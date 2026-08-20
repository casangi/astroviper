"""Simulate one in-memory chunk of visibilities (uvw → beams → DFT → noise)."""

from __future__ import annotations

import time as _time

import numpy as np
import xarray as xr

from astroviper.processing_functions.simulation.antenna_beams import (
    dish_diameters_of_beam_models,
    evaluate_beam_models,
    pack_beam_models,
    resolve_beam_params,
)
from astroviper.processing_functions.simulation.calculate_noise import (
    calculate_noise,
    resolve_noise_params,
)
from astroviper.processing_functions.simulation.calculate_uvw import calculate_uvw
from astroviper.processing_functions.simulation.calculate_visibilities import (
    calculate_visibilities,
)
from astroviper.utils.measurement_set_tools import (
    normalize_polarization,
    polarization_index,
)
from astroviper.utils.param_docs import shares_param_docs


@shares_param_docs
def simulate_processing_set(
    time: np.ndarray,
    frequency: np.ndarray,
    polarization: list,
    antenna_position: np.ndarray,
    site_position: np.ndarray,
    point_source_flux: np.ndarray,
    point_source_ra_dec: np.ndarray,
    phase_center_ra_dec: np.ndarray,
    beam_models: list,
    beam_model_map: np.ndarray,
    beam_params: dict | None = None,
    pointing_ra_dec: np.ndarray | None = None,
    uvw_params: dict | None = None,
    noise_params: dict | None = None,
    channel_width: float | None = None,
    integration_time: float | None = None,
    direction_frame: str = "icrs",
    processing_function_threads: int = 1,
    implementation: str = "cpp",
    random_seed: int | None = None,
    gaussian_source_flux: np.ndarray | None = None,
    gaussian_source_ra_dec: np.ndarray | None = None,
    gaussian_source_shape: np.ndarray | None = None,
) -> tuple[xr.Dataset, dict]:
    """Simulate the visibilities of a point- and Gaussian-source sky for a block of times and channels.

    This is the science function behind the ``simulate_processing_set`` node task and
    distributed application: it computes the baseline ``uvw`` coordinates,
    evaluates the antenna beam models (parallactic angles, Zernike → Jones
    images), runs the direct-Fourier-transform visibility kernel and optionally
    adds thermal noise.  It does no I/O.

    Parameters
    ----------
    time : np.ndarray, [n_time]
        UTC times of the chunk as unix seconds (MSv4 ``time`` coordinate) or ISO strings.
    frequency : np.ndarray, [n_frequency], Hz
        Channel frequencies of the chunk.
    polarization : list of str
        MSv4 polarization labels to simulate, a subset of one instrumental basis
        (``["RR", "RL", "LR", "LL"]`` or ``["XX", "XY", "YX", "YY"]``).
    antenna_position : np.ndarray, [n_antenna, 3], metres
        ITRF geocentric antenna positions (``antenna_xds.ANTENNA_POSITION``).
    site_position : np.ndarray, [3], metres
        ITRF geocentric array reference position used for uvw and parallactic angles.
    point_source_flux : np.ndarray, [n_source, n_time | 1, n_frequency | 1, 4], Jy
        Flux of every point source in the four instrumental correlations
        (``RR, RL, LR, LL`` or ``XX, XY, YX, YY``); singleton time/frequency axes broadcast.
    point_source_ra_dec : np.ndarray, [n_time | 1, n_source, 2], radians
        Right ascension and declination of the point sources (per time or fixed).
    gaussian_source_flux : np.ndarray, [n_gaussian, n_time | 1, n_frequency | 1, 4], Jy, optional
        Integrated flux of each Gaussian source in the four instrumental
        correlations; singleton time/frequency axes broadcast.  ``None``
        (default) simulates no Gaussian sources.
    gaussian_source_ra_dec : np.ndarray, [n_time | 1, n_gaussian, 2], radians, optional
        Right ascension and declination of the Gaussian sources (per time or fixed).
    gaussian_source_shape : np.ndarray, [n_gaussian, 3], radians, optional
        ``[major, minor, position angle]`` FWHM shape of each Gaussian source, in
        the imaging clean-beam convention
        (:func:`astroviper.processing_functions.imaging.restore.elliptical_gaussian_uv_taper`).
    phase_center_ra_dec : np.ndarray, [n_time | 1, 2], radians
        Phase centre of the array per time (time-varying for mosaics) or fixed.
    beam_models : list
        Antenna beam models: analytic dicts, aperture (Zernike) coefficient
        datasets, beam polynomial datasets or Jones image datasets
        (see ``astroviper.utils.beam_models``).
    beam_model_map : np.ndarray, [n_antenna] int
        Index into ``beam_models`` for each antenna.
    beam_params : dict, optional
        Beam evaluation parameters: ``mueller_selection`` (row-major indices of the
        4x4 Mueller elements to apply, default ``[0, 5, 10, 15]``), ``pa_radius``
        (rad; parallactic-angle spacing of the Zernike beam images, default 0.2),
        ``image_size`` (Zernike beam image size, default ``[1000, 1000]``),
        ``fov_scaling`` (beam image extent in units of the beam cut radius,
        default 4) and ``zernike_freq_interp`` (default ``"nearest"``).
    pointing_ra_dec : np.ndarray, [n_time | 1, n_antenna | 1, 2], radians, optional
        Antenna pointing directions; ``None`` points every antenna at the phase centre.
    uvw_params : dict, optional
        ``auto_correlations`` (bool, default False) and ``uvw_convention``
        (``"msv4"`` = antenna2 - antenna1, default; or ``"sirius"``).
    noise_params : dict, optional
        Thermal-noise system parameters (``casatools.simulator.setnoise`` tsys-manual
        model): ``t_receiver``, ``t_atmos``, ``tau``, ``ant_efficiency``,
        ``spill_efficiency``, ``corr_efficiency``, ``quantization_efficiency``,
        ``t_cmb`` and ``random_seed``; ``None`` disables noise (unit weights).
    channel_width : float, Hz, optional
        Channel width used for the noise level; defaults to the channel spacing.
    integration_time : float, s, optional
        Integration time used for the noise level; defaults to the time spacing.
    direction_frame : str
        Astropy frame of all right ascension / declination inputs (``"icrs"`` or ``"fk5"``).
    processing_function_threads : int
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    implementation : {"numpy", "cpp"}
        Visibility kernel implementation: ``"cpp"`` (multithreaded C++, default) or
        ``"numpy"`` (vectorised NumPy reference).
    random_seed : int, optional
        Overrides ``noise_params["random_seed"]`` (used by node tasks to derive a
        seed per task).

    Returns
    -------
    xds : xr.Dataset
        In-memory MSv4-like chunk with ``VISIBILITY[time, baseline_id, frequency,
        polarization]`` (complex128), ``UVW[time, baseline_id, uvw_label]`` (m),
        ``WEIGHT`` and ``FLAG`` (all False), coordinates ``time``, ``frequency``,
        ``polarization``, ``baseline_id``, ``baseline_antenna1_id``/``2_id`` and
        ``parallactic_angle``; attrs ``uvw_convention``.
    timing : dict
        ``T_uvw``, ``T_beams``, ``T_visibilities``, ``T_noise`` in seconds.
    """
    time = np.asarray(time)
    frequency = np.asarray(frequency, dtype=np.float64)
    polarization = normalize_polarization(polarization)
    pol_index = polarization_index(polarization)
    antenna_position = np.asarray(antenna_position, dtype=np.float64)
    site_position = np.asarray(site_position, dtype=np.float64)
    beam_model_map = np.asarray(beam_model_map, dtype=np.int64)
    params = resolve_beam_params(beam_params)
    uvw_params = {
        "auto_correlations": False,
        "uvw_convention": "msv4",
        **(uvw_params or {}),
    }
    noise = resolve_noise_params(noise_params)
    timing = {}

    start = _time.time()
    uvw, antenna1, antenna2 = calculate_uvw(
        antenna_position,
        site_position,
        time,
        phase_center_ra_dec,
        auto_correlations=uvw_params["auto_correlations"],
        direction_frame=direction_frame,
        uvw_convention=uvw_params["uvw_convention"],
    )
    timing["T_uvw"] = _time.time() - start

    start = _time.time()
    evaluated_models, parallactic_angle = evaluate_beam_models(
        beam_models,
        time,
        frequency,
        phase_center_ra_dec,
        site_position,
        params,
        direction_frame,
    )
    packed_models = pack_beam_models(evaluated_models)
    timing["T_beams"] = _time.time() - start

    start = _time.time()
    visibility = calculate_visibilities(
        uvw,
        antenna1,
        antenna2,
        frequency,
        pol_index,
        point_source_flux,
        point_source_ra_dec,
        phase_center_ra_dec,
        pointing_ra_dec,
        beam_model_map,
        packed_models,
        parallactic_angle,
        params["mueller_selection"],
        processing_function_threads=processing_function_threads,
        implementation=implementation,
        gaussian_source_flux=gaussian_source_flux,
        gaussian_source_ra_dec=gaussian_source_ra_dec,
        gaussian_source_shape=gaussian_source_shape,
    )
    timing["T_visibilities"] = _time.time() - start

    start = _time.time()
    if noise is not None:
        if channel_width is None:
            channel_width = (
                float(np.abs(frequency[1] - frequency[0]))
                if frequency.size > 1
                else None
            )
        if integration_time is None:
            t_numeric = (
                time.astype(np.float64)
                if np.issubdtype(time.dtype, np.number)
                else None
            )
            integration_time = (
                float(t_numeric[1] - t_numeric[0])
                if t_numeric is not None and t_numeric.size > 1
                else None
            )
        if channel_width is None or integration_time is None:
            raise ValueError(
                "channel_width and integration_time are required for the noise model."
            )
        dish_diameter = dish_diameters_of_beam_models(beam_models)[beam_model_map]
        seed = random_seed if random_seed is not None else noise.get("random_seed")
        noise_values, weight, _ = calculate_noise(
            visibility.shape,
            dish_diameter,
            antenna1,
            antenna2,
            channel_width,
            integration_time,
            noise,
            auto_correlations=uvw_params["auto_correlations"],
            random_seed=seed,
        )
        visibility = visibility + noise_values
    else:
        weight = np.ones(visibility.shape, dtype=np.float64)
    timing["T_noise"] = _time.time() - start

    xds = xr.Dataset(
        {
            "VISIBILITY": (
                ("time", "baseline_id", "frequency", "polarization"),
                visibility,
            ),
            "UVW": (("time", "baseline_id", "uvw_label"), uvw),
            "WEIGHT": (("time", "baseline_id", "frequency", "polarization"), weight),
            "FLAG": (
                ("time", "baseline_id", "frequency", "polarization"),
                np.zeros(visibility.shape, dtype=bool),
            ),
        },
        coords={
            "time": time,
            "baseline_id": np.arange(len(antenna1)),
            "baseline_antenna1_id": ("baseline_id", antenna1),
            "baseline_antenna2_id": ("baseline_id", antenna2),
            "frequency": frequency,
            "polarization": np.array(polarization, dtype=str),
            "uvw_label": ["u", "v", "w"],
            "parallactic_angle": ("time", parallactic_angle),
        },
        attrs={"uvw_convention": uvw_params["uvw_convention"]},
    )
    xds["VISIBILITY"].attrs.update({"type": "quantity", "units": "Jy"})
    xds["UVW"].attrs.update({"type": "uvw", "units": "m", "frame": direction_frame})
    return xds, timing
