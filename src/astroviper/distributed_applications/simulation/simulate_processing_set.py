"""Distributed application: simulate an interferometric observation into an MSv4 processing set."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import toolviper.utils.parameter
import xarray as xr
from numcodecs import Blosc

from astroviper.utils.param_docs import shares_param_docs

# The toolviper parameter-check schema lives next to this module.
_PARAM_CONFIG_DIR = os.path.dirname(__file__)

DISTRIBUTED_APPLICATION_TIMING_PHASES = [
    (
        "DISTRIBUTED APPLICATION (driver)",
        None,
        [
            ("build coordinates + MSv4 metadata", "T_make_coordinates"),
            (
                "determine chunks + parallel coords",
                "T_determine_chunks_and_parallel_coords",
            ),
            ("create empty MSv4 on disk", "T_create_empty_measurement_set"),
            ("interpolate data coords", "T_interpolate_data_coords"),
            ("create map/reduce graph", "T_create_map_reduce_graph"),
            ("generate dask graph", "T_generate_dask_graph"),
            ("compute dask graph", "T_compute_dask_graph"),
            ("consolidate metadata + schema check", "T_consolidate_metadata"),
            ("write MSv2 (arcae)", "T_write_ms_v2"),
        ],
    ),
]


def combine_timing_data_frames(input_data, input_params):
    """GraphVIPER reducer: concatenate the per-task timing DataFrames."""
    import pandas as pd

    frames = [df for df in input_data if df is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@shares_param_docs
@toolviper.utils.parameter.validate(
    config_dir=_PARAM_CONFIG_DIR, add_data_type=xr.Dataset
)
def simulate_processing_set(
    ps_store: str,
    antenna_xds: xr.Dataset,
    time_params: dict,
    frequency_params: dict,
    polarization: list,
    point_source_flux: np.ndarray | list,
    point_source_ra_dec: np.ndarray | list,
    phase_center_ra_dec: np.ndarray | list,
    beam_models: list,
    beam_model_map: np.ndarray | list,
    beam_params: dict | None = None,
    field_name: list | str | None = None,
    pointing_ra_dec: np.ndarray | list | None = None,
    uvw_params: dict | None = None,
    noise_params: dict | None = None,
    gaussian_source_flux: np.ndarray | list | None = None,
    gaussian_source_ra_dec: np.ndarray | list | None = None,
    gaussian_source_shape: np.ndarray | list | None = None,
    ms_v2_path: str | None = None,
    direction_frame: str = "icrs",
    ms_name: str | None = None,
    n_time_chunks: int | None = None,
    n_frequency_chunks: int | None = None,
    processing_function_threads: int = 1,
    implementation: str = "cpp",
    compressor: Any = None,
    overwrite: bool = False,
    compute_backend: str = "dask",
    mpi_cluster_setup: dict | None = None,
    thread_info: dict | None = None,
    check_schema: bool = True,
) -> dict:
    """Simulate the visibilities of a point-source sky and write them as an MSv4 processing set.

    Builds the time/frequency axes and all MSv4 metadata, creates the empty
    processing set on disk (one measurement set: one spectral window, one
    polarization setup, one or more fields), maps the
    ``simulate_processing_set`` node task over ``(time, frequency)`` chunks with
    GraphVIPER (each task computes uvw, beams, the visibility DFT and noise for
    its chunk and region-writes ``VISIBILITY``, ``UVW``, ``WEIGHT``, ``FLAG``),
    computes the graph and validates the result against the MSv4 schema.

    Parameters
    ----------
    ps_store : str
        Output processing-set directory (conventionally ``<name>.ps.zarr``).
    antenna_xds : xr.Dataset
        MSv4 antenna dataset describing the array, e.g. from
        :func:`astroviper.utils.telescope_layout.read_telescope_layout`.  Its
        ``overall_telescope_name`` selects the array reference position (uvw,
        parallactic angles) via
        :func:`astroviper.utils.telescope_layout.observatory_position`.
    time_params : dict
        ``time_start`` (``"YYYY-MM-DDTHH:MM:SS.SSS"`` UTC), ``time_delta`` (s,
        integration time) and ``n_samples``.
    frequency_params : dict
        ``freq_start`` (Hz), ``freq_delta`` (Hz), ``n_channels`` and optionally
        ``channel_width`` (Hz), ``spectral_window_name``, ``observer``,
        ``spectral_window_intents``.
    polarization : list of str
        MSv4 polarization labels to simulate, a subset of one instrumental basis
        (``["RR", "RL", "LR", "LL"]`` or ``["XX", "XY", "YX", "YY"]``).
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
    ms_v2_path : str, optional
        Additionally write the simulated MSv4 as a CASA Measurement Set v2 at
        this path via the optional `arcae <https://github.com/ska-sa/arcae>`_
        backend (``utils.measurement_set_v2.write_measurement_set_v2``).
        Default ``None`` (no MSv2 output).
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
    field_name : list of str [n_time | 1] or str, optional
        Field name per time (mosaics) or a single name; ``None`` names distinct
        phase centres ``field_0``, ``field_1``, ...  Written as the MSv4
        ``field_name`` coordinate and ``field_and_source_base_xds``.
    pointing_ra_dec : np.ndarray, [n_time | 1, n_antenna | 1, 2], radians, optional
        Antenna pointing directions; ``None`` points every antenna at the phase centre.
    uvw_params : dict, optional
        ``auto_correlations`` (bool, default False).  The uvw follow the
        archival / VLBI convention adopted by MSv4:
        ``uvw = P(antenna1) - P(antenna2)`` (see
        :func:`~astroviper.processing_functions.simulation.calculate_uvw.calculate_uvw`).
    noise_params : dict, optional
        Thermal-noise system parameters (``casatools.simulator.setnoise`` tsys-manual
        model): ``t_receiver``, ``t_atmos``, ``tau``, ``ant_efficiency``,
        ``spill_efficiency``, ``corr_efficiency``, ``quantization_efficiency``,
        ``t_cmb`` and ``random_seed``; ``None`` disables noise (unit weights).
    direction_frame : str
        Astropy frame of all right ascension / declination inputs (``"icrs"`` or ``"fk5"``).
    ms_name : str, optional
        Name of the MSv4 inside the processing set; default
        ``"<telescope>_<spectral_window_name>"``.
    n_time_chunks, n_frequency_chunks : int, optional
        Number of GraphVIPER chunks along time / frequency (the map has
        ``n_time_chunks * n_frequency_chunks`` tasks).  ``None`` lets
        :func:`astroviper.utils.data_partitioning.calculate_data_chunking`
        choose from the per-chunk memory estimate and ``thread_info``.
    processing_function_threads : int
        Number of threads handed to the per-processing-function (C++ / FFT)
        kernels.
    implementation : {"numpy", "cpp"}
        Visibility kernel implementation: ``"cpp"`` (multithreaded C++, default) or
        ``"numpy"`` (vectorised NumPy reference).
    compressor : numcodecs compressor or None
        Compressor for the output Zarr arrays; ``None`` selects
        ``Blosc(cname="lz4", clevel=5)``.
    overwrite : bool
        Replace an existing ``ps_store``.
    compute_backend : {"dask", "mpi"}
        Execute the GraphVIPER graph with Dask (default) or with
        :func:`graphviper.graph_tools.processes_with_mpi`.
    mpi_cluster_setup : dict, optional
        Options forwarded to ``processes_with_mpi`` when ``compute_backend="mpi"``.
    thread_info : dict, optional
        ``{"n_threads", "memory_per_thread"}`` used for automatic chunking;
        default from :func:`astroviper.utils.data_partitioning.get_thread_info`.
    check_schema : bool
        Validate the written processing set with ``xradio.schema.check.check_datatree``.

    Returns
    -------
    dict
        ``{"timing_node_tasks": pandas.DataFrame (one row per task),
        "timing_distributed_application": dict, "ps_store": str, "ms_name": str}``.

    See Also
    --------
    astroviper.node_tasks.simulation.simulate_processing_set
    astroviper.processing_functions.simulation.simulate_processing_set
    """
    import time as _time

    import dask
    import toolviper.utils.logger as logger
    import zarr
    from graphviper.graph_tools import generate_dask_workflow, map, reduce
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )

    import astroviper.node_tasks as node_tasks
    from astroviper.processing_functions.simulation.antenna_beams import (
        dish_diameters_of_beam_models,
        resolve_beam_params,
    )
    from astroviper.utils.data_partitioning import (
        calculate_data_chunking,
        get_thread_info,
    )
    from astroviper.utils.measurement_set_tools import (
        create_empty_measurement_set_v4_on_disk,
        make_empty_visibility_xds,
        make_field_and_source_xds,
        make_frequency_coordinate,
        make_time_coordinate,
        normalize_polarization,
        number_of_baselines,
        resolve_fields,
    )
    from astroviper.utils.telescope_layout import observatory_position
    from astroviper.utils.timing import format_timing_summary

    application_start = _time.time()
    timing_distributed_application = {}

    # --- coordinates and MSv4 metadata ---------------------------------------
    start = _time.time()
    polarization = normalize_polarization(polarization)
    point_source_flux = np.asarray(point_source_flux, dtype=np.float64)
    point_source_ra_dec = np.asarray(point_source_ra_dec, dtype=np.float64)
    if gaussian_source_flux is not None:
        gaussian_source_flux = np.asarray(gaussian_source_flux, dtype=np.float64)
        gaussian_source_ra_dec = np.asarray(gaussian_source_ra_dec, dtype=np.float64)
        gaussian_source_shape = np.asarray(gaussian_source_shape, dtype=np.float64)
    phase_center_ra_dec = np.asarray(phase_center_ra_dec, dtype=np.float64)
    beam_model_map = np.asarray(beam_model_map, dtype=np.int64)
    if pointing_ra_dec is not None:
        pointing_ra_dec = np.asarray(pointing_ra_dec, dtype=np.float64)
    uvw_params = {
        "auto_correlations": False,
        **(uvw_params or {}),
    }
    beam_params = resolve_beam_params(beam_params)
    if compressor is None:
        compressor = Blosc(cname="lz4", clevel=5)

    time_coord = make_time_coordinate(time_params)
    frequency_coord = make_frequency_coordinate(frequency_params)
    n_time = len(time_coord["data"])
    n_frequency = len(frequency_coord["data"])
    n_antenna = antenna_xds.sizes["antenna_name"]
    n_baseline = number_of_baselines(n_antenna, uvw_params["auto_correlations"])
    n_polarization = len(polarization)
    _check_input_shapes(
        point_source_flux, point_source_ra_dec, phase_center_ra_dec, pointing_ra_dec,
        beam_model_map, len(beam_models), n_time, n_frequency, n_antenna,
    )  # fmt: skip
    _check_gaussian_input_shapes(
        gaussian_source_flux, gaussian_source_ra_dec, gaussian_source_shape,
        n_time, n_frequency,
    )  # fmt: skip

    antenna_position = np.asarray(antenna_xds.ANTENNA_POSITION.values, dtype=np.float64)
    telescope_name = str(antenna_xds.attrs.get("overall_telescope_name", "unknown"))
    site_position = observatory_position(telescope_name, antenna_position)

    field_name_per_time, unique_field_names, unique_phase_centers = resolve_fields(
        phase_center_ra_dec, field_name, n_time
    )
    field_and_source_xds = make_field_and_source_xds(
        unique_field_names, unique_phase_centers, frame=direction_frame
    )
    ms_xds = make_empty_visibility_xds(
        time_coord,
        frequency_coord,
        polarization,
        antenna_xds,
        field_name_per_time,
        auto_correlations=uvw_params["auto_correlations"],
        description=(
            f"Simulated visibilities of {point_source_ra_dec.shape[1]} point source(s)"
            + (
                f" and {gaussian_source_ra_dec.shape[1]} Gaussian source(s)"
                if gaussian_source_ra_dec is not None
                else ""
            )
            + " "
            "with astroviper.distributed_applications.simulation.simulate_processing_set."
        ),
    )
    if ms_name is None:
        ms_name = f"{telescope_name}_{frequency_coord['attrs']['spectral_window_name']}".replace(
            " ", "_"
        )
    timing_distributed_application["T_make_coordinates"] = _time.time() - start

    # --- chunking ---------------------------------------------------------------
    start = _time.time()
    if n_time_chunks is None or n_frequency_chunks is None:
        if thread_info is None:
            thread_info = get_thread_info()
        # memory of a (1 time, 1 channel) chunk: vis + weight + flag + uvw plus the
        # NumPy kernel temporaries (Mueller-scaled flux, fringes) and beam images
        bytes_singleton = n_baseline * n_polarization * (16 + 8 + 1) + n_baseline * 24
        bytes_singleton += (
            6 * n_baseline * 16 * 4
        )  # kernel temporaries [n_baseline, 4] complex
        beam_bytes = 0.0
        for bm in beam_models:
            if isinstance(bm, xr.Dataset) and (
                "ZPC" in bm.data_vars or "JONES" in bm.data_vars
            ):
                beam_bytes += float(np.prod(beam_params["image_size"])) * 16 * 4
        memory_singleton_chunk = bytes_singleton * 1.5 / 1024**3
        suggested = calculate_data_chunking(
            memory_singleton_chunk,
            {"time": n_time, "frequency": n_frequency},
            thread_info,
            constant_memory=beam_bytes / 1024**3,
            tasks_per_thread=2,
        )
        if n_time_chunks is None:
            n_time_chunks = int(suggested.get("time", 1))
        if n_frequency_chunks is None:
            n_frequency_chunks = int(suggested.get("frequency", 1))
    n_time_chunks = int(min(max(n_time_chunks, 1), n_time))
    n_frequency_chunks = int(min(max(n_frequency_chunks, 1), n_frequency))
    parallel_coords = {
        "time": make_parallel_coord(coord=time_coord, n_chunks=n_time_chunks),
        "frequency": make_parallel_coord(
            coord=frequency_coord, n_chunks=n_frequency_chunks
        ),
    }
    logger.info(
        f"simulate_processing_set: {n_time} times x {n_baseline} baselines x {n_frequency} "
        f"channels x {n_polarization} polarizations in {n_time_chunks} x {n_frequency_chunks} chunks."
    )
    timing_distributed_application["T_determine_chunks_and_parallel_coords"] = (
        _time.time() - start
    )

    # --- empty MSv4 on disk ---------------------------------------------------
    start = _time.time()
    ms_path = create_empty_measurement_set_v4_on_disk(
        ps_store,
        ms_name,
        ms_xds,
        antenna_xds,
        field_and_source_xds,
        parallel_coords,
        compressor=compressor,
        double_precision=True,
        overwrite=overwrite,
    )
    timing_distributed_application["T_create_empty_measurement_set"] = (
        _time.time() - start
    )

    # --- graph -----------------------------------------------------------------
    start = _time.time()
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, {}
    )
    timing_distributed_application["T_interpolate_data_coords"] = _time.time() - start

    channel_width = float(frequency_coord["attrs"]["channel_width"]["data"])
    integration_time = float(time_params["time_delta"])
    input_params = {
        "ms_path": ms_path,
        "polarization": polarization,
        "antenna_position": antenna_position,
        "site_position": site_position,
        "point_source_flux": point_source_flux,
        "point_source_ra_dec": point_source_ra_dec,
        "gaussian_source_flux": gaussian_source_flux,
        "gaussian_source_ra_dec": gaussian_source_ra_dec,
        "gaussian_source_shape": gaussian_source_shape,
        "phase_center_ra_dec": phase_center_ra_dec,
        "beam_models": list(beam_models),
        "beam_model_map": beam_model_map,
        "beam_params": beam_params,
        "pointing_ra_dec": pointing_ra_dec,
        "uvw_params": uvw_params,
        "noise_params": noise_params,
        "channel_width": channel_width,
        "integration_time": integration_time,
        "direction_frame": direction_frame,
        "processing_function_threads": processing_function_threads,
        "implementation": implementation,
    }
    # dish diameters are resolved here once so the node tasks never fail late
    dish_diameters_of_beam_models(beam_models)

    start = _time.time()
    viper_graph = map(
        input_data={},
        node_task_data_mapping=node_task_data_mapping,
        node_task=node_tasks.simulation.simulate_processing_set,
        input_params=input_params,
        in_memory_compute=False,
    )
    viper_graph = reduce(viper_graph, combine_timing_data_frames, {}, mode="tree")
    timing_distributed_application["T_create_map_reduce_graph"] = _time.time() - start

    if compute_backend == "mpi":
        from graphviper.graph_tools import processes_with_mpi

        timing_distributed_application["T_generate_dask_graph"] = 0.0
        start = _time.time()
        timing_node_tasks = processes_with_mpi(viper_graph, mpi_cluster_setup)
        timing_distributed_application["T_compute_dask_graph"] = _time.time() - start
    elif compute_backend == "dask":
        start = _time.time()
        dask_graph = generate_dask_workflow(viper_graph)
        timing_distributed_application["T_generate_dask_graph"] = _time.time() - start
        start = _time.time()
        timing_node_tasks = dask.compute(dask_graph)[0]
        timing_distributed_application["T_compute_dask_graph"] = _time.time() - start
    else:
        raise ValueError(
            f"Unknown compute_backend {compute_backend!r}; expected 'dask' or 'mpi'."
        )

    # --- finalize ----------------------------------------------------------------
    start = _time.time()
    zarr.consolidate_metadata(ps_store)
    if check_schema:
        from xradio.measurement_set import open_processing_set
        from xradio.schema.check import check_datatree

        issues = check_datatree(open_processing_set(ps_store))
        if str(issues) != "No schema issues found":
            logger.warning(f"MSv4 schema check of {ps_store}: {issues}")
    timing_distributed_application["T_consolidate_metadata"] = _time.time() - start

    if ms_v2_path is not None:
        from astroviper.utils.measurement_set_v2 import write_measurement_set_v2

        start = _time.time()
        write_measurement_set_v2(
            ps_store, ms_v2_path, ms_name=ms_name, overwrite=overwrite
        )
        timing_distributed_application["T_write_ms_v2"] = _time.time() - start

    timing_distributed_application["T_total"] = _time.time() - application_start
    logger.info(
        format_timing_summary(
            timing_distributed_application,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            total_key="T_total",
            title="simulate_processing_set timing",
        )
    )
    if hasattr(timing_node_tasks, "sort_values") and "task_id" in timing_node_tasks:
        timing_node_tasks = timing_node_tasks.sort_values("task_id", ignore_index=True)
    return {
        "timing_node_tasks": timing_node_tasks,
        "timing_distributed_application": timing_distributed_application,
        "ps_store": ps_store,
        "ms_name": ms_name,
    }


def _check_gaussian_input_shapes(flux, source_ra_dec, shape, n_time, n_frequency):
    if flux is None and source_ra_dec is None and shape is None:
        return
    if flux is None or source_ra_dec is None or shape is None:
        raise ValueError(
            "gaussian_source_flux, gaussian_source_ra_dec and gaussian_source_shape "
            "must be given together (or all omitted)."
        )
    if flux.ndim != 4 or flux.shape[3] != 4:
        raise ValueError(
            f"gaussian_source_flux must have shape [n_gaussian, n_time|1, n_frequency|1, 4]; got {flux.shape}."
        )
    if source_ra_dec.ndim != 3 or source_ra_dec.shape[2] != 2:
        raise ValueError(
            f"gaussian_source_ra_dec must have shape [n_time|1, n_gaussian, 2]; got {source_ra_dec.shape}."
        )
    if flux.shape[0] != source_ra_dec.shape[1]:
        raise ValueError(
            "n_gaussian of gaussian_source_flux and gaussian_source_ra_dec differ."
        )
    if flux.shape[1] not in (1, n_time) or flux.shape[2] not in (1, n_frequency):
        raise ValueError(
            "gaussian_source_flux time/frequency axes must be 1 or match the simulated axes."
        )
    if source_ra_dec.shape[0] not in (1, n_time):
        raise ValueError("gaussian_source_ra_dec time axis must be 1 or n_time.")
    if shape.shape != (flux.shape[0], 3):
        raise ValueError(
            f"gaussian_source_shape must have shape [n_gaussian, 3]; got {shape.shape}."
        )


def _check_input_shapes(
    flux,
    source_ra_dec,
    phase_center,
    pointing,
    beam_model_map,
    n_beam_models,
    n_time,
    n_frequency,
    n_antenna,
):
    if flux.ndim != 4 or flux.shape[3] != 4:
        raise ValueError(
            f"point_source_flux must have shape [n_source, n_time|1, n_frequency|1, 4]; got {flux.shape}."
        )
    if source_ra_dec.ndim != 3 or source_ra_dec.shape[2] != 2:
        raise ValueError(
            f"point_source_ra_dec must have shape [n_time|1, n_source, 2]; got {source_ra_dec.shape}."
        )
    if flux.shape[0] != source_ra_dec.shape[1]:
        raise ValueError(
            "n_source of point_source_flux and point_source_ra_dec differ."
        )
    if flux.shape[1] not in (1, n_time) or flux.shape[2] not in (1, n_frequency):
        raise ValueError(
            "point_source_flux time/frequency axes must be 1 or match the simulated axes."
        )
    if source_ra_dec.shape[0] not in (1, n_time):
        raise ValueError("point_source_ra_dec time axis must be 1 or n_time.")
    if (
        phase_center.ndim != 2
        or phase_center.shape[1] != 2
        or phase_center.shape[0] not in (1, n_time)
    ):
        raise ValueError(
            f"phase_center_ra_dec must have shape [n_time|1, 2]; got {phase_center.shape}."
        )
    if pointing is not None and (
        pointing.ndim != 3
        or pointing.shape[2] != 2
        or pointing.shape[0] not in (1, n_time)
        or pointing.shape[1] not in (1, n_antenna)
    ):
        raise ValueError("pointing_ra_dec must have shape [n_time|1, n_antenna|1, 2].")
    if beam_model_map.shape != (n_antenna,):
        raise ValueError(
            f"beam_model_map must have shape [n_antenna={n_antenna}]; got {beam_model_map.shape}."
        )
    if beam_model_map.min() < 0 or beam_model_map.max() >= n_beam_models:
        raise ValueError("beam_model_map indices must index into beam_models.")
