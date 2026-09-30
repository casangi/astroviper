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
    phase_center_ra_dec: np.ndarray | list,
    beam_models: list,
    beam_model_map: np.ndarray | list,
    sky_components: list | None = None,
    point_source_flux: np.ndarray | list | None = None,
    point_source_ra_dec: np.ndarray | list | None = None,
    beam_params: dict | None = None,
    field_name: list | str | None = None,
    pointing_ra_dec: np.ndarray | list | None = None,
    uvw_params: dict | None = None,
    noise_params: dict | None = None,
    gaussian_source_flux: np.ndarray | list | None = None,
    gaussian_source_ra_dec: np.ndarray | list | None = None,
    gaussian_source_shape: np.ndarray | list | None = None,
    disk_source_flux: np.ndarray | list | None = None,
    disk_source_ra_dec: np.ndarray | list | None = None,
    disk_source_shape: np.ndarray | list | None = None,
    disk_source_limb_darkening: np.ndarray | list | None = None,
    gaussian_ring_source_flux: np.ndarray | list | None = None,
    gaussian_ring_source_ra_dec: np.ndarray | list | None = None,
    gaussian_ring_source_shape: np.ndarray | list | None = None,
    ms_v2_path: str | None = None,
    sky_image_params: dict | None = None,
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
    """Simulate the visibilities of a sky of analytic components and write them as an MSv4 processing set.

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
    sky_components : list of dict, optional
        Sky components of any kind (point, gaussian, disk, gaussian_ring,
        m_ring, crescent, annulus, exponential_disk, tapered_power_law,
        shapelet): ``{"kind", "flux", "ra_dec", <shape parameters>}`` as
        described in :mod:`~astroviper.processing_functions.simulation.sky_components`.
    point_source_flux : np.ndarray, [n_source, n_time | 1, n_frequency | 1, 4], Jy, optional
        Flux of every point source in the four instrumental correlations
        (``RR, RL, LR, LL`` or ``XX, XY, YX, YY``); singleton time/frequency axes
        broadcast.  Real, or complex with conjugate cross hands (``XY = U + iV``,
        ``YX = U - iV``; ``RL = Q + iU``, ``LR = Q - iU``).
    point_source_ra_dec : np.ndarray, [n_time | 1, n_source, 2], radians, optional
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
    disk_source_flux : np.ndarray, [n_disk, n_time | 1, n_frequency | 1, 4], Jy, optional
        Integrated flux of each limb-darkened disk source in the four
        instrumental correlations; singleton time/frequency axes broadcast.
        ``None`` (default) simulates no disk sources.
    disk_source_ra_dec : np.ndarray, [n_time | 1, n_disk, 2], radians, optional
        Right ascension and declination of the disk sources (per time or fixed).
    disk_source_shape : np.ndarray, [n_disk, 3], radians, optional
        ``[major, minor, position angle]`` outer diameters and orientation of
        each (inclined) disk, in the Gaussian-source / clean-beam position-angle
        convention
        (:func:`astroviper.processing_functions.simulation.limb_darkened_disk.limb_darkened_disk_uv_response`).
    disk_source_limb_darkening : np.ndarray, [n_disk] float, optional
        Power-law limb-darkening exponent ``alpha`` of each disk
        (``I ~ mu**alpha``, Hestroffer 1997): ``0`` uniform disk (the default
        when ``None``), ``> 0`` darker towards the limb, ``-2 < alpha < 0`` limb
        brightened, ``-2`` an infinitely thin ring.
    gaussian_ring_source_flux : np.ndarray, [n_ring, n_time | 1, n_frequency | 1, 4], Jy, optional
        Integrated flux of each Gaussian-broadened ring source (a thin ring
        convolved with a circular Gaussian; ``radius = 0`` is a Gaussian, nested
        rings model a protoplanetary disk) in the four instrumental
        correlations; singleton time/frequency axes broadcast.  ``None``
        (default) simulates no ring sources.
    gaussian_ring_source_ra_dec : np.ndarray, [n_time | 1, n_ring, 2], radians, optional
        Right ascension and declination of the ring sources (per time or fixed).
    gaussian_ring_source_shape : np.ndarray, [n_ring, 4], radians, optional
        ``[radius, fwhm, inclination, position angle]`` of each ring: ring radius
        and FWHM of the broadening Gaussian (radians), inclination (radians,
        ``0`` face-on) and the position angle of the major axis in the
        Gaussian-source / clean-beam convention
        (:func:`astroviper.processing_functions.simulation.gaussian_ring.gaussian_ring_uv_response`).
    ms_v2_path : str, optional
        Additionally write the simulated MSv4 as a CASA Measurement Set v2 at
        this path via the optional `arcae <https://github.com/ska-sa/arcae>`_
        backend (``utils.measurement_set_v2.write_measurement_set_v2``).
        Default ``None`` (no MSv2 output).
    sky_image_params : dict, optional
        Also write the simulated sky itself (every component, no primary
        beam) as an XRADIO image (Zarr) with the data variable ``SKY`` in
        Jy/pixel on the imager's grid, so that imaging results can be
        compared with the truth pixel by pixel: ``{"image_store": path,
        "image_size": [n_l, n_m], "cell_size": [dl, dm] (radians, the
        ``image_params`` convention of ``image_cube_single_field``),
        "phase_direction": [ra, dec] radians (default: the phase centre),
        "polarization_coords": Stokes labels (default all four when four
        correlations are simulated, else ``["I"]``), "time_index": which
        time sample of time-dependent fluxes and positions to draw
        (default 0)}``.  Point sources are added to the pixel they fall in;
        extended components are sampled at the pixel centres.  Stokes V
        (linear feeds) or U (circular feeds) needs complex cross-hand
        fluxes and is zero otherwise.  Default ``None`` (no image).
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
    from astroviper.processing_functions.simulation.sky_components import (
        as_correlation_flux,
        describe_sky_components,
        normalize_sky_components,
        sky_components_from_arrays,
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
    if (point_source_flux is None) != (point_source_ra_dec is None):
        raise ValueError(
            "point_source_flux and point_source_ra_dec must be given together (or both omitted)."
        )
    if point_source_flux is not None:
        point_source_flux = as_correlation_flux(point_source_flux, "point_source_flux")
        point_source_ra_dec = np.asarray(point_source_ra_dec, dtype=np.float64)
    if gaussian_source_flux is not None:
        gaussian_source_flux = as_correlation_flux(
            gaussian_source_flux, "gaussian_source_flux"
        )
        gaussian_source_ra_dec = np.asarray(gaussian_source_ra_dec, dtype=np.float64)
        gaussian_source_shape = np.asarray(gaussian_source_shape, dtype=np.float64)
    if disk_source_flux is not None:
        disk_source_flux = as_correlation_flux(disk_source_flux, "disk_source_flux")
    if disk_source_ra_dec is not None:
        disk_source_ra_dec = np.asarray(disk_source_ra_dec, dtype=np.float64)
    if disk_source_shape is not None:
        disk_source_shape = np.asarray(disk_source_shape, dtype=np.float64)
        if disk_source_limb_darkening is None and disk_source_shape.ndim == 2:
            disk_source_limb_darkening = np.zeros(disk_source_shape.shape[0])
    if disk_source_limb_darkening is not None:
        disk_source_limb_darkening = np.asarray(
            disk_source_limb_darkening, dtype=np.float64
        )
    if gaussian_ring_source_flux is not None:
        gaussian_ring_source_flux = as_correlation_flux(
            gaussian_ring_source_flux, "gaussian_ring_source_flux"
        )
    if gaussian_ring_source_ra_dec is not None:
        gaussian_ring_source_ra_dec = np.asarray(
            gaussian_ring_source_ra_dec, dtype=np.float64
        )
    if gaussian_ring_source_shape is not None:
        gaussian_ring_source_shape = np.asarray(
            gaussian_ring_source_shape, dtype=np.float64
        )
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
    sky_components = normalize_sky_components(
        sky_components, n_time, n_frequency, direction_frame
    )
    _check_extended_source_shapes(
        "gaussian", gaussian_source_flux, gaussian_source_ra_dec,
        gaussian_source_shape, n_time, n_frequency,
    )  # fmt: skip
    _check_extended_source_shapes(
        "disk", disk_source_flux, disk_source_ra_dec, disk_source_shape,
        n_time, n_frequency, limb_darkening=disk_source_limb_darkening,
    )  # fmt: skip
    _check_extended_source_shapes(
        "gaussian_ring", gaussian_ring_source_flux, gaussian_ring_source_ra_dec,
        gaussian_ring_source_shape, n_time, n_frequency,
    )  # fmt: skip

    antenna_position = np.asarray(antenna_xds.ANTENNA_POSITION.values, dtype=np.float64)
    telescope_name = str(antenna_xds.attrs.get("overall_telescope_name", "unknown"))
    site_position = observatory_position(telescope_name, antenna_position)

    all_components = []
    if point_source_flux is not None:
        all_components += sky_components_from_arrays(
            "point", point_source_flux, point_source_ra_dec
        )
    for kind, flux, ra_dec, shape, extra in [
        ("gaussian", gaussian_source_flux, gaussian_source_ra_dec, gaussian_source_shape, None),
        ("disk", disk_source_flux, disk_source_ra_dec, disk_source_shape, disk_source_limb_darkening),
        ("gaussian_ring", gaussian_ring_source_flux, gaussian_ring_source_ra_dec, gaussian_ring_source_shape, None),
    ]:  # fmt: skip
        if flux is not None:
            all_components += sky_components_from_arrays(
                kind, flux, ra_dec, shape, extra
            )
    all_components += sky_components
    if not all_components:
        raise ValueError(
            "no sources: give sky_components and/or point_source_flux / point_source_ra_dec."
        )
    normalized_components = normalize_sky_components(
        all_components, n_time, n_frequency, direction_frame
    )
    described = describe_sky_components(normalized_components)
    sky_image = _resolve_sky_image_params(
        sky_image_params, phase_center_ra_dec, polarization, n_time
    )

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
            "Simulated visibilities of " + described + " with "
            "astroviper.distributed_applications.simulation.simulate_processing_set."
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

    # --- optional image of the simulated sky ----------------------------------
    if sky_image is not None:
        start = _time.time()
        _write_sky_image(
            sky_image, normalized_components, time_coord, frequency_coord, overwrite
        )
        timing_distributed_application["T_write_sky_image"] = _time.time() - start

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
        "disk_source_flux": disk_source_flux,
        "disk_source_ra_dec": disk_source_ra_dec,
        "disk_source_shape": disk_source_shape,
        "disk_source_limb_darkening": disk_source_limb_darkening,
        "gaussian_ring_source_flux": gaussian_ring_source_flux,
        "gaussian_ring_source_ra_dec": gaussian_ring_source_ra_dec,
        "gaussian_ring_source_shape": gaussian_ring_source_shape,
        "sky_components": sky_components,
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
        "sky_image_store": None if sky_image is None else sky_image["image_store"],
    }


_EXTENDED_SOURCE_SHAPE_COLUMNS = {"gaussian": 3, "disk": 3, "gaussian_ring": 4}


def _check_extended_source_shapes(
    kind, flux, source_ra_dec, shape, n_time, n_frequency, limb_darkening=None
):
    """Validate the ``<kind>_source_*`` arrays of Gaussian, disk or Gaussian-ring sources."""
    if flux is None and source_ra_dec is None and shape is None:
        return
    if flux is None or source_ra_dec is None or shape is None:
        raise ValueError(
            f"{kind}_source_flux, {kind}_source_ra_dec and {kind}_source_shape "
            "must be given together (or all omitted)."
        )
    if flux.ndim != 4 or flux.shape[3] != 4:
        raise ValueError(
            f"{kind}_source_flux must have shape [n_{kind}, n_time|1, n_frequency|1, 4]; got {flux.shape}."
        )
    if source_ra_dec.ndim != 3 or source_ra_dec.shape[2] != 2:
        raise ValueError(
            f"{kind}_source_ra_dec must have shape [n_time|1, n_{kind}, 2]; got {source_ra_dec.shape}."
        )
    if flux.shape[0] != source_ra_dec.shape[1]:
        raise ValueError(
            f"n_{kind} of {kind}_source_flux and {kind}_source_ra_dec differ."
        )
    if flux.shape[1] not in (1, n_time) or flux.shape[2] not in (1, n_frequency):
        raise ValueError(
            f"{kind}_source_flux time/frequency axes must be 1 or match the simulated axes."
        )
    if source_ra_dec.shape[0] not in (1, n_time):
        raise ValueError(f"{kind}_source_ra_dec time axis must be 1 or n_time.")
    n_columns = _EXTENDED_SOURCE_SHAPE_COLUMNS[kind]
    if shape.shape != (flux.shape[0], n_columns):
        raise ValueError(
            f"{kind}_source_shape must have shape [n_{kind}, {n_columns}]; got {shape.shape}."
        )
    if kind == "gaussian_ring":
        if np.any(shape[:, :2] < 0):
            raise ValueError(
                "gaussian_ring_source_shape radius and fwhm must be non-negative."
            )
        if np.any(shape[:, 2] < 0) or np.any(shape[:, 2] >= np.pi / 2):
            raise ValueError(
                "gaussian_ring_source_shape inclination must be in [0, pi/2) radians."
            )
    if kind == "disk":
        if limb_darkening is None or limb_darkening.shape != (flux.shape[0],):
            raise ValueError(
                "disk_source_limb_darkening must have shape [n_disk]; got "
                f"{None if limb_darkening is None else limb_darkening.shape}."
            )
        if np.any(limb_darkening < -2) or not np.all(np.isfinite(limb_darkening)):
            raise ValueError(
                "disk_source_limb_darkening exponents must be finite and >= -2 "
                "(-2 is the thin-ring limit)."
            )
        if np.any(shape[:, :2] < 0):
            raise ValueError("disk_source_shape diameters must be non-negative.")


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
    if flux is None:
        flux = np.zeros((0, 1, 1, 4))
        source_ra_dec = np.zeros((1, 0, 2))
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


_STOKES_LABELS = ("I", "Q", "U", "V")
_SKY_IMAGE_KEYS = {
    "image_store",
    "image_size",
    "cell_size",
    "phase_direction",
    "polarization_coords",
    "time_index",
}


def _resolve_sky_image_params(
    sky_image_params, phase_center_ra_dec, polarization, n_time
):
    """Validate ``sky_image_params`` and fill in its defaults; ``None`` stays ``None``."""
    if sky_image_params is None:
        return None
    from astroviper.processing_functions.simulation.sky_components import (
        polarization_basis_of,
    )

    params = dict(sky_image_params)
    unknown = sorted(set(params) - _SKY_IMAGE_KEYS)
    if unknown:
        raise ValueError(f"sky_image_params: unknown keys {unknown}.")
    for key in ("image_store", "image_size", "cell_size"):
        if key not in params:
            raise ValueError(f"sky_image_params needs '{key}'.")
    image_size = [int(n) for n in np.asarray(params["image_size"]).ravel()]
    cell_size = [
        float(c) for c in np.asarray(params["cell_size"], dtype=np.float64).ravel()
    ]
    if (
        len(image_size) != 2
        or min(image_size) < 1
        or len(cell_size) != 2
        or 0.0 in cell_size
    ):
        raise ValueError(
            "sky_image_params: image_size must be two positive integers and "
            "cell_size two non-zero angles in radians."
        )
    time_index = int(params.get("time_index", 0))
    if not 0 <= time_index < n_time:
        raise ValueError(
            f"sky_image_params: time_index {time_index} is outside the {n_time} simulated times."
        )
    phase_direction = params.get("phase_direction")
    if phase_direction is None:
        phase_direction = phase_center_ra_dec[
            time_index if phase_center_ra_dec.shape[0] > 1 else 0
        ]
    phase_direction = np.asarray(phase_direction, dtype=np.float64).reshape(2)
    stokes = params.get("polarization_coords")
    if stokes is None:
        stokes = list(_STOKES_LABELS) if len(polarization) == 4 else ["I"]
    stokes = [str(label).upper() for label in stokes]
    bad = [label for label in stokes if label not in _STOKES_LABELS]
    if bad:
        raise ValueError(
            f"sky_image_params: polarization_coords must be Stokes labels (I, Q, U, V); got {bad}."
        )
    return {
        "image_store": str(params["image_store"]),
        "image_size": image_size,
        "cell_size": cell_size,
        "phase_direction": phase_direction,
        "polarization_coords": stokes,
        "time_index": time_index,
        "polarization_basis": polarization_basis_of(polarization),
    }


def _write_sky_image(sky_image, components, time_coord, frequency_coord, overwrite):
    """Rasterise ``components`` on the imager's grid and write the XRADIO image (``SKY``, Jy/pixel)."""
    import xarray as xr
    from xradio.image import make_empty_sky_image, write_image

    from astroviper.processing_functions.simulation.sky_components import (
        stokes_sky_model_images,
    )
    from astroviper.utils.data_group_tools import modify_data_groups_xds

    time_index = sky_image["time_index"]
    unix_seconds = float(np.asarray(time_coord["data"], dtype=np.float64)[time_index])
    img_xds = make_empty_sky_image(
        phase_center=sky_image["phase_direction"],
        image_size=sky_image["image_size"],
        cell_size=sky_image["cell_size"],
        frequency_coords=np.asarray(frequency_coord["data"], dtype=np.float64),
        pol_coords=sky_image["polarization_coords"],
        time_coords=[
            unix_seconds / 86400.0 + 40587.0
        ],  # unix seconds (UTC) -> MJD days
        do_sky_coords=False,
    )
    l_axis, m_axis = img_xds.l.values, img_xds.m.values
    n_frequency = img_xds.sizes["frequency"]
    sky = np.zeros(
        (
            1,
            n_frequency,
            len(sky_image["polarization_coords"]),
            l_axis.size,
            m_axis.size,
        ),
        dtype=np.float64,
    )
    for channel in range(n_frequency):
        sky[0, channel] = stokes_sky_model_images(
            components,
            l_axis,
            m_axis,
            sky_image["phase_direction"],
            sky_image["polarization_basis"],
            sky_image["polarization_coords"],
            time_index=time_index,
            frequency_index=channel,
        )
    img_xds["SKY"] = xr.DataArray(
        sky, dims=("time", "frequency", "polarization", "l", "m")
    )
    img_xds.attrs.get("data_groups", {}).pop("base", None)
    modify_data_groups_xds(
        img_xds,
        data_group_out_name="base",
        data_group_out={"sky": "SKY"},
        description=(
            "Simulated sky model in Jy/pixel (all components, no primary beam), "
            "written by astroviper.distributed_applications.simulation.simulate_processing_set."
        ),
    )
    write_image(
        img_xds, sky_image["image_store"], out_format="zarr", overwrite=overwrite
    )
