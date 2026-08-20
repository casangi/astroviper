"""Node task: simulate one ``(time, frequency)`` chunk and write it into an MSv4 store."""

from __future__ import annotations

import os
import socket
import time as _time

import numpy as np
import toolviper.utils.logger as logger

from astroviper.utils.param_docs import shares_param_docs


def _slice_time_axis(array, axis: int, time_slice: slice):
    """Slice ``axis`` of ``array`` with ``time_slice`` unless that axis is a singleton."""
    if array is None:
        return None
    array = np.asarray(array)
    if array.shape[axis] == 1:
        return array
    index = [slice(None)] * array.ndim
    index[axis] = time_slice
    return array[tuple(index)]


@shares_param_docs
def simulate_processing_set(
    ms_path: str,
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
    task_coords: dict | None = None,
    data_selection: dict | None = None,
    task_id: int = 0,
    graph_mode: bool = True,
):
    """Simulate the visibilities of one time/frequency chunk and write them to disk.

    The chunk is defined by ``task_coords`` (GraphVIPER): ``task_coords["time"]``
    and ``task_coords["frequency"]`` carry the coordinate values and the slices
    into the full axes, which are used to cut the time/frequency dependent
    inputs (``point_source_flux``, ``point_source_ra_dec``,
    ``phase_center_ra_dec``, ``pointing_ra_dec``).  The simulated ``VISIBILITY``,
    ``UVW``, ``WEIGHT`` and ``FLAG`` are region-written into the MSv4 created by the
    distributed application.

    Parameters
    ----------
    ms_path : str
        Path of the (pre-created, empty) MSv4 group ``<ps_store>/<ms_name>``.
    polarization : list of str
        MSv4 polarization labels to simulate, a subset of one instrumental basis
        (``["RR", "RL", "LR", "LL"]`` or ``["XX", "XY", "YX", "YY"]``).
    antenna_position : np.ndarray, [n_antenna, 3], metres
        ITRF geocentric antenna positions.
    site_position : np.ndarray, [3], metres
        ITRF geocentric array reference position.
    point_source_flux : np.ndarray, [n_source, n_time | 1, n_frequency | 1, 4], Jy
        Flux of every point source in the four instrumental correlations
        (``RR, RL, LR, LL`` or ``XX, XY, YX, YY``); singleton time/frequency axes broadcast.
    point_source_ra_dec : np.ndarray, [n_time | 1, n_source, 2], radians
        Right ascension and declination of the point sources (per time or fixed).
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
        Channel width used for the noise level.
    integration_time : float, s, optional
        Integration time used for the noise level.
    direction_frame : str
        Astropy frame of all right ascension / declination inputs (``"icrs"`` or ``"fk5"``).
    processing_function_threads : int
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    implementation : {"numpy", "cpp"}
        Visibility kernel implementation: ``"cpp"`` (multithreaded C++, default) or
        ``"numpy"`` (vectorised NumPy reference).
    task_coords : dict
        Per-chunk coordinate mapping; ``task_coords[<parallel dim>]`` supplies
        this chunk's parallel coordinate values (``"data"``) and its
        ``"slice"`` into the full output array (for cube imaging the
        parallel dim is ``frequency``).
    data_selection : dict
        GraphVIPER data selection (unused: the simulator has no input data).
    task_id : int
        Identifier of the parallel chunk being processed.
    graph_mode : bool
        If ``False`` the simulated chunk dataset is returned instead of being
        written to ``ms_path`` (testing / interactive use).

    Returns
    -------
    pandas.DataFrame or xr.Dataset
        One timing row (``task_id``, ``T_uvw``, ``T_beams``, ``T_visibilities``,
        ``T_noise``, ``T_write``, ``T_simulate_task`` plus host/pid/thread
        provenance) in graph mode; the chunk dataset when ``graph_mode=False``.
    """
    import threading

    import pandas as pd

    from astroviper.processing_functions.simulation.simulate_processing_set import (
        simulate_processing_set as simulate_processing_set_pf,
    )
    from astroviper.utils.measurement_set_tools import write_visibility_chunk_to_disk

    task_start = _time.time()
    time_chunk = np.asarray(task_coords["time"]["data"])
    frequency_chunk = np.asarray(task_coords["frequency"]["data"], dtype=np.float64)
    time_slice = task_coords["time"]["slice"]
    frequency_slice = task_coords["frequency"]["slice"]

    flux_chunk = _slice_time_axis(point_source_flux, 1, time_slice)
    flux_chunk = _slice_time_axis(flux_chunk, 2, frequency_slice)
    source_chunk = _slice_time_axis(point_source_ra_dec, 0, time_slice)
    phase_center_chunk = _slice_time_axis(phase_center_ra_dec, 0, time_slice)
    pointing_chunk = _slice_time_axis(pointing_ra_dec, 0, time_slice)

    noise = None
    if noise_params is not None:
        noise = dict(noise_params)
        base_seed = noise.get("random_seed")
        # a distinct, reproducible stream per task
        noise["random_seed"] = (
            None if base_seed is None else int(base_seed) + int(task_id)
        )

    logger.debug(
        f"simulate_processing_set task {task_id}: {len(time_chunk)} times x "
        f"{len(frequency_chunk)} channels"
    )
    xds, timing = simulate_processing_set_pf(
        time_chunk,
        frequency_chunk,
        polarization,
        antenna_position,
        site_position,
        flux_chunk,
        source_chunk,
        phase_center_chunk,
        beam_models,
        beam_model_map,
        beam_params=beam_params,
        pointing_ra_dec=pointing_chunk,
        uvw_params=uvw_params,
        noise_params=noise,
        channel_width=channel_width,
        integration_time=integration_time,
        direction_frame=direction_frame,
        processing_function_threads=processing_function_threads,
        implementation=implementation,
    )
    if not graph_mode:
        return xds

    start = _time.time()
    # UVW does not depend on frequency: only the first frequency chunk writes it.
    variables = ["VISIBILITY", "WEIGHT", "FLAG"]
    if (frequency_slice.start or 0) == 0:
        variables.append("UVW")
    write_visibility_chunk_to_disk(ms_path, task_coords, xds[variables])
    timing["T_write"] = _time.time() - start
    timing["T_simulate_task"] = _time.time() - task_start

    row = {
        "task_id": task_id,
        "n_time": len(time_chunk),
        "n_frequency": len(frequency_chunk),
        **timing,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "thread": threading.get_ident(),
    }
    return pd.DataFrame([row])
