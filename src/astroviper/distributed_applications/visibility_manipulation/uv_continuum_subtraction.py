"""uv_continuum_subtraction: continuum subtraction in the uv domain of the
MSv4s of one spectral window of a processing set, the port of the pipeline's
``hif_uvcontsub``.

The driver opens the processing set lazily, checks the MSv4s of the spectral
window, computes the Doppler factors of their times once when the continuum
ranges are in another frame, and creates the output variables in every MSv4.
Each MSv4 has its own map over blocks of its times
(:func:`~astroviper.node_tasks.visibility_manipulation.uv_continuum_subtraction.uv_continuum_subtraction`),
so that a task writes whole on-disk chunks of its MSv4; the graphs of all
MSv4s are computed together. The output data group is registered in the
MSv4s once their visibilities are written.
"""

from __future__ import annotations

import os

import toolviper.utils.parameter

from astroviper.utils.param_docs import shares_param_docs

_PARAM_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

DISTRIBUTED_APPLICATION_TIMING_PHASES = [
    (
        "DISTRIBUTED APPLICATION (driver)",
        None,
        [
            ("open the processing set and check it", "T_open_and_check"),
            ("Doppler factors and continuum channels", "T_continuum_channels"),
            (
                "determine chunks + parallel coords",
                "T_determine_chunks_and_parallel_coords",
            ),
            ("create empty output variables on disk", "T_create_output_variables"),
            ("create map/reduce graphs", "T_create_map_reduce_graph"),
            ("get compute resources", "T_get_compute_resources"),
            ("generate dask graphs", "T_generate_dask_graph"),
            ("compute dask graphs", "T_compute_dask_graph"),
            ("register data group + consolidate metadata", "T_register_data_group"),
        ],
    ),
]
NODE_TASK_TIMING_PHASES = [
    (
        "TIME BLOCK TASK (one block of times of one MSv4)",
        None,
        [
            ("load", "T_load"),
            ("continuum channels", "T_continuum_channel_mask"),
            ("fit and subtract", "T_fit_and_subtract"),
            ("spectrum sums", "T_spectrum_sums"),
            ("write", "T_write"),
        ],
    ),
]
NODE_TASK_TIMING_TOTAL_KEY = "T_uv_continuum_subtraction_task"
TASK_STREAM_COLUMNS = {
    "task": NODE_TASK_TIMING_TOTAL_KEY,
    "load": "T_load",
    "write": "T_write",
}
# Memory a task holds per visibility sample (bytes): the loaded data group,
# its spectra as rows, the fit weights and the residuals in double precision,
# and the outputs.
TASK_BYTES_PER_SAMPLE = 128
# Upper bound of an on-disk chunk of the output variables (bytes).
OUTPUT_CHUNK_BYTES = 64 * 2**20


def combine_uv_continuum_subtraction_results(input_data, input_params):
    """GraphVIPER reducer: concatenate the timing and fit statistics, add the spectra.

    Parameters
    ----------
    input_data : list of dict or None
        Node-task results or partial reductions, with ``timing_node_tasks``,
        ``fit_statistics`` and ``spectrum_sums``; with
        ``monitor_resources_seconds`` GraphVIPER adds to a node task's result
        a ``resource_usage`` dict of series, folded into its one-row timing
        frame as list-valued columns, as the imager does. ``None`` entries
        are skipped.
    input_params : dict
        Unused (the signature of a GraphVIPER reducer).

    Returns
    -------
    dict
        ``timing_node_tasks`` and ``fit_statistics`` (``pandas.DataFrame``,
        concatenated) and ``spectrum_sums`` (``{ms_name: dict}`` of summed
        arrays): the keys of the inputs, so that it composes in a tree
        reduction.
    """
    import pandas as pd

    timing, statistics, spectra = [], [], {}
    for result in input_data:
        if result is None:
            continue
        frame = result["timing_node_tasks"]
        usage = result.get("resource_usage")
        if usage is not None:
            frame = frame.copy()
            for key, value in usage.items():
                frame[key] = [value] if isinstance(value, list) else value
        timing.append(frame)
        statistics.append(result["fit_statistics"])
        for ms_name, sums in result["spectrum_sums"].items():
            if ms_name in spectra:
                spectra[ms_name] = {
                    key: spectra[ms_name][key] + value for key, value in sums.items()
                }
            else:
                spectra[ms_name] = dict(sums)
    return {
        "timing_node_tasks": (
            pd.concat(timing, ignore_index=True) if timing else pd.DataFrame()
        ),
        "fit_statistics": (
            pd.concat(statistics, ignore_index=True) if statistics else pd.DataFrame()
        ),
        "spectrum_sums": spectra,
    }


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def uv_continuum_subtraction(
    ps_store: str,
    spectral_window_name: str,
    continuum_frequency_ranges,
    continuum_frequency_frame: str,
    fit_order: int = 1,
    ms_data_group_in_name: str = "base",
    ms_data_group_out_name: str = "continuum_subtracted",
    ms_data_group_out_modified: dict | None = None,
    n_mapping_parallelism: dict | None = None,
    thread_info: dict | None = None,
    overwrite: bool = False,
    monitor_resources_seconds: float | None = None,
    report: str | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """Subtract the continuum of the visibilities of one spectral window of a processing set.

    The port of the pipeline's ``hif_uvcontsub``: every spectrum (one time,
    baseline and polarization) of the MSv4s of ``spectral_window_name`` has a
    polynomial fitted to its continuum channels and subtracted, as CASA
    ``uvcontsub`` does (see
    :mod:`~astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction`
    for the fit and its departures from CASA). The continuum-subtracted
    visibilities and their flags are written into each MSv4 as the variables
    of a new data group, ``ms_data_group_out_name``, which keeps the other
    roles of the input group.
    :func:`~astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction.continuum_fit_from_find_continuum`
    gives the continuum ranges, their frame and the fit order from a
    ``find_continuum_ultra`` result, as the pipeline takes them from
    ``cont.dat``.

    Parameters
    ----------
    ps_store : str
        Path of the processing set (Zarr). The output variables are written
        into its MSv4s.
    spectral_window_name : str
        The spectral window (``spectral_window_name`` of the frequency axis)
        whose MSv4s are continuum subtracted, all of them.
    continuum_frequency_ranges : list or numpy.ndarray
        The continuum ranges, ``[low, high]`` pairs of frequencies (Hz) in
        ``continuum_frequency_frame``; a channel is a continuum channel at the
        times its frequency, converted to that frame, lies in a range, edges
        included (the ``continuum_frequency_ranges`` of ``find_continuum_ultra``
        are the outer edges of its continuum channels).
    continuum_frequency_frame : str
        Spectral frame of ``continuum_frequency_ranges``, by its XRADIO
        observer name: the frame of the measurement set, or, for a ``"TOPO"``
        measurement set, ``"lsrk"``, ``"lsrd"`` or ``"BARY"``, to which the
        channel frequencies are converted at every time with the Doppler factor
        of ``ms_spectral_frame_conversion.doppler_factor`` (the telescope site
        at rest on the Earth, the phase centre of the first field).
    fit_order : int, default 1
        Order of the polynomial fitted to the continuum channels of every
        spectrum [CASA ``fitorder``]; lowered to ``n - 1`` for a spectrum with
        ``n <= fit_order`` valid continuum channels.
    ms_data_group_in_name : str, default "base"
        Measurement-set data group whose correlated data are continuum
        subtracted, with its flags and weights.
    ms_data_group_out_name : str, default "continuum_subtracted"
        Measurement-set data group of the continuum-subtracted visibilities.
    ms_data_group_out_modified : dict, optional
        The output variables of the roles ``correlated_data`` and ``flag``;
        ``None`` gives ``{"correlated_data": "VISIBILITY_CONTINUUM_SUBTRACTED",
        "flag": "FLAG_CONTINUUM_SUBTRACTED"}``. Neither may be a variable of
        another data group, the input group included.
    n_mapping_parallelism : dict, optional
        ``{"time": n}``, the blocks of times of every MSv4, one node task
        each. ``None`` (default): blocks that fit the memory of a thread
        (:data:`TASK_BYTES_PER_SAMPLE` per sample), up to about two tasks per
        thread of ``thread_info``
        (:func:`~astroviper.utils.data_partitioning.calculate_data_chunking`),
        whose count reads the fewest on-disk time chunks of the input
        correlated data (a task decompresses every chunk its block touches).
    thread_info : dict, optional
        ``n_threads`` and ``memory_per_thread`` (GiB) of the cluster, for the
        default ``n_mapping_parallelism``; ``None`` (default): those of
        :func:`~astroviper.utils.data_partitioning.get_thread_info`.
    overwrite : bool, default False
        Replace the output data group and variables of an earlier run.
    monitor_resources_seconds : float, optional
        If set, sample each node task's worker-process CPU, memory and I/O
        every this many seconds (GraphVIPER's resource monitor) into
        ``timing_node_tasks``. Requires ``psutil``.
    report : str, optional
        Make a report of the run: ``"html"`` (one self contained file) or
        ``"pdf"``, written next to the processing set as
        ``<name>_<spectral_window_name>_report.html`` or ``.pdf``, ``<name>``
        the processing set without its store suffix (``demo.ps.zarr`` and
        ``spw_0`` give ``demo_spw_0_report.html``), or ``"notebook"``, displayed
        in the calling Jupyter notebook. The report has a processing section
        common to all distributed applications (performance measures, timing,
        task stream, task times and, with ``monitor_resources_seconds``, the
        CPU, memory and I/O of the cluster, of every worker slot and of the
        tasks) and a results section of the application's own (the fit, the
        continuum channels and fit statistics of every MSv4, and the mean
        spectra before and after). It is made after the output is written: a
        failure while making it is logged and leaves the results intact.
        ``None`` (default): no report.
    input_checked : bool, default False
        Skip the configuration checks done by the calling layer. The checks of
        the measurement sets always run.

    Returns
    -------
    dict
        ``ps_store``; ``ms_names``, the MSv4s continuum subtracted;
        ``continuum_channels`` (``pandas.DataFrame``, one row per MSv4:
        ``n_time``, ``n_frequency``, the fewest and most continuum channels of
        a time, the channels that are continuum channels at every time and at
        some time, and the fraction of the channels of all times that are);
        ``fit_statistics`` (``pandas.DataFrame``, one row per MSv4 and
        polarization: ``n_spectra``, ``n_fitted``, ``n_order_reduced``,
        ``n_without_continuum``, ``n_flagged_without_continuum`` and the
        average, minimum and maximum over the fitted spectra of the
        chi-squared CASA computes for each fit, of the real and imaginary
        parts); ``mean_spectra`` (``{ms_name: xarray.Dataset}`` over
        ``frequency`` and ``polarization``: ``VISIBILITY_MEAN_INPUT`` and
        ``VISIBILITY_MEAN_OUTPUT``, the weighted means over the times and
        baselines of the valid input and output visibilities, and
        ``CONTINUUM_TIME_FRACTION``, the fraction of the times at which each
        channel is a continuum channel); ``timing_node_tasks``,
        ``timing_distributed_application``, ``compute_resources`` and
        ``report`` (the path of the report written, else ``None``).

    Raises
    ------
    ValueError
        If the configuration fails the checks; the processing set has no MSv4
        of the spectral window; an MSv4 lacks the input data group or one of
        its roles ``correlated_data``, ``flag`` and ``weight``, has other
        dimensions or frequency units, or has a frame the ranges cannot be
        converted to; or the output data group or variables fail
        :func:`~astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction.check_output_data_group`
        (a new group distinct from the input group, outputs that are no other
        group's variables, existing ones replaced only with ``overwrite``).
    AssertionError
        If a parameter fails the schema of ``uv_continuum_subtraction.param.json``,
        or the memory of a thread cannot hold one time of an MSv4
        (:func:`~astroviper.utils.data_partitioning.calculate_data_chunking`).

    See Also
    --------
    astroviper.node_tasks.visibility_manipulation.uv_continuum_subtraction
    astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction
    """
    import time as _time

    import dask
    import numpy as np
    import pandas as pd
    import toolviper.utils.logger as logger
    import xarray as xr
    import zarr
    from graphviper.graph_tools import generate_dask_workflow, map, reduce
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from numcodecs import Blosc
    from xradio.measurement_set import open_processing_set
    from xradio.schema.check import is_extension_type

    from astroviper.node_tasks.visibility_manipulation.uv_continuum_subtraction import (
        uv_continuum_subtraction as uv_continuum_subtraction_task,
    )
    from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
        DEFAULT_MS_DATA_GROUP_OUT_MODIFIED,
        check_continuum_fit,
        continuum_channel_mask,
        measurement_set_doppler_factors,
    )
    from astroviper.utils.data_group_tools import (
        create_data_groups_in_and_out,
        modify_data_groups_xds,
    )
    from astroviper.utils.data_partitioning import (
        calculate_data_chunking,
        get_compute_resources,
        get_thread_info,
    )
    from astroviper.utils.io import create_empty_data_variables_on_disk
    from astroviper.utils.timing import format_timing_summary, node_task_timing_table

    application_start = _time.time()
    if not input_checked:
        check_continuum_fit(
            continuum_frequency_ranges,
            continuum_frequency_frame,
            fit_order,
            ms_data_group_out_modified,
        )
        _check_driver_config(
            n_mapping_parallelism, thread_info, monitor_resources_seconds
        )
    ranges = np.asarray(continuum_frequency_ranges, dtype=np.float64)
    out_modified = dict(
        ms_data_group_out_modified or DEFAULT_MS_DATA_GROUP_OUT_MODIFIED
    )
    ps_store = os.path.abspath(ps_store)
    timing = {}

    start = _time.time()
    ps_xdt = open_processing_set(ps_store)
    ms_names = [
        name
        for name, ms_xdt in ps_xdt.items()
        if not is_extension_type(ms_xdt.attrs.get("type"))
        and ms_xdt.frequency.attrs.get("spectral_window_name") == spectral_window_name
    ]
    if not ms_names:
        raise ValueError(
            f"The processing set {ps_store} has no MSv4 of the spectral window "
            f"{spectral_window_name!r}."
        )
    for name in ms_names:
        _check_measurement_set(
            ps_xdt[name],
            os.path.join(ps_store, name),
            continuum_frequency_frame,
            ms_data_group_in_name,
            ms_data_group_out_name,
            out_modified,
            overwrite,
        )
    timing["T_open_and_check"] = _time.time() - start

    start = _time.time()
    doppler = {}
    continuum_time_fraction = {}
    channel_rows = []
    for name in ms_names:
        doppler[name] = measurement_set_doppler_factors(
            ps_xdt[name], continuum_frequency_frame
        )
        mask = continuum_channel_mask(
            ps_xdt[name].frequency.values, ranges, doppler[name]
        )
        continuum_time_fraction[name] = mask.mean(axis=0)
        n_continuum = mask.sum(axis=1)
        channel_rows.append(
            {
                "ms_name": name,
                "n_time": int(ps_xdt[name].sizes["time"]),
                "n_frequency": int(ps_xdt[name].sizes["frequency"]),
                "n_continuum_channels_min": int(n_continuum.min()),
                "n_continuum_channels_max": int(n_continuum.max()),
                "n_continuum_channels_every_time": int(mask.all(axis=0).sum()),
                "n_continuum_channels_some_time": int(mask.any(axis=0).sum()),
                "continuum_channel_fraction": float(mask.mean()),
            }
        )
    timing["T_continuum_channels"] = _time.time() - start

    start = _time.time()
    if n_mapping_parallelism is None and thread_info is None:
        thread_info = get_thread_info()
    parallel_coords = {}
    for name in ms_names:
        ms_xdt = ps_xdt[name]
        n_time = ms_xdt.sizes["time"]
        if n_mapping_parallelism is not None:
            n_chunks = n_mapping_parallelism["time"]
        else:
            bytes_per_time = TASK_BYTES_PER_SAMPLE * (
                ms_xdt.sizes["baseline_id"]
                * ms_xdt.sizes["frequency"]
                * ms_xdt.sizes["polarization"]
            )
            n_target = calculate_data_chunking(
                bytes_per_time / 1024**3,
                {"time": n_time},
                thread_info,
                tasks_per_thread=2,
            )["time"]
            correlated_data = ms_xdt.attrs["data_groups"][ms_data_group_in_name][
                "correlated_data"
            ]
            n_chunks = _time_blocks(
                n_time,
                n_target,
                max(
                    1,
                    int(thread_info["memory_per_thread"] * 1024**3 // bytes_per_time),
                ),
                zarr.open_array(
                    os.path.join(ps_store, name, correlated_data), mode="r"
                ).chunks[0],
            )
        parallel_coords[name] = {
            "time": make_parallel_coord(
                coord=ms_xdt.time, n_chunks=int(min(max(n_chunks, 1), n_time))
            )
        }
    timing["T_determine_chunks_and_parallel_coords"] = _time.time() - start

    start = _time.time()
    try:
        for name in ms_names:
            _create_output_variables(
                os.path.join(ps_store, name),
                ps_xdt[name],
                ms_data_group_in_name,
                ms_data_group_out_name,
                out_modified,
                parallel_coords[name],
                overwrite,
                create_empty_data_variables_on_disk,
                Blosc(cname="lz4", clevel=5),
            )
    finally:
        # The metadata of the processing set matches its arrays if the run fails.
        zarr.consolidate_metadata(ps_store)
    timing["T_create_output_variables"] = _time.time() - start

    start = _time.time()
    graphs = []
    for name in ms_names:
        input_data = {name: ps_xdt[name].to_dataset()}
        node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
            parallel_coords[name], input_data
        )
        graph = map(
            input_data=input_data,
            node_task_data_mapping=node_task_data_mapping,
            node_task=uv_continuum_subtraction_task,
            input_params={
                "ps_store": ps_store,
                "ms_name": name,
                "continuum_frequency_ranges": ranges.tolist(),
                "continuum_frequency_frame": continuum_frequency_frame,
                "fit_order": fit_order,
                "ms_data_group_in_name": ms_data_group_in_name,
                "ms_data_group_out_name": ms_data_group_out_name,
                "ms_data_group_out_modified": out_modified,
                "doppler_factors": doppler[name],
                "input_checked": True,
            },
            in_memory_compute=False,
            monitor_resources_seconds=monitor_resources_seconds,
        )
        graphs.append(
            reduce(graph, combine_uv_continuum_subtraction_results, {}, mode="tree")
        )
    timing["T_create_map_reduce_graph"] = _time.time() - start

    start = _time.time()
    compute_resources = get_compute_resources("dask", None)
    timing["T_get_compute_resources"] = _time.time() - start
    start = _time.time()
    dask_graphs = [generate_dask_workflow(graph) for graph in graphs]
    timing["T_generate_dask_graph"] = _time.time() - start
    start = _time.time()
    results = dask.compute(*dask_graphs)
    end = _time.time()
    timing["T_compute_dask_graph"] = end - start
    # Absolute anchors of the compute call, which place the compute window on
    # the wall clock of the tasks' start_unixtime (the task stream).
    timing["compute_start_unixtime"] = start
    timing["compute_end_unixtime"] = end
    combined = combine_uv_continuum_subtraction_results(list(results), {})

    start = _time.time()
    description = (
        f"Continuum subtracted from {ms_data_group_in_name} with "
        "astroviper.distributed_applications.visibility_manipulation."
        f"uv_continuum_subtraction: polynomial of order {fit_order} fitted to the "
        f"channels in {len(ranges)} continuum frequency ranges "
        f"({continuum_frequency_frame})."
    )
    for name in ms_names:
        group = zarr.open_group(
            os.path.join(ps_store, name), mode="r+", use_consolidated=False
        )
        stored = xr.Dataset(attrs={"data_groups": dict(group.attrs["data_groups"])})
        _, data_group_out = create_data_groups_in_and_out(
            stored,
            data_group_in_name=ms_data_group_in_name,
            data_group_out_name=ms_data_group_out_name,
            data_group_out_modified=out_modified,
            overwrite=True,
        )
        modify_data_groups_xds(
            stored,
            data_group_out_name=ms_data_group_out_name,
            data_group_out=data_group_out,
            description=description,
        )
        group.attrs["data_groups"] = stored.attrs["data_groups"]
        zarr.consolidate_metadata(ps_store, path=name)
    zarr.consolidate_metadata(ps_store)
    timing["T_register_data_group"] = _time.time() - start

    application_end = _time.time()
    timing["T_total"] = application_end - application_start
    # Absolute anchors of the whole call, which place the driver's own steps on
    # the wall clock of the task stream.
    timing["application_start_unixtime"] = application_start
    timing["application_end_unixtime"] = application_end
    logger.info(
        format_timing_summary(
            timing,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            total_key="T_total",
            title="uv_continuum_subtraction timing",
        )
    )
    timing_node_tasks = combined["timing_node_tasks"]
    if len(timing_node_tasks):
        timing_node_tasks = timing_node_tasks.sort_values(
            ["ms_name", "task_id"], ignore_index=True
        )
        logger.info(
            "uv_continuum_subtraction node-task timing over the "
            f"{len(timing_node_tasks)} blocks of times (seconds)\n"
            + node_task_timing_table(timing_node_tasks).to_string(
                float_format="{:.3f}".format
            )
        )
    result = {
        "ps_store": ps_store,
        "ms_names": ms_names,
        "continuum_channels": pd.DataFrame(channel_rows),
        "fit_statistics": _total_fit_statistics(combined["fit_statistics"]),
        "mean_spectra": {
            name: _mean_spectra(
                ps_xdt[name],
                combined["spectrum_sums"][name],
                continuum_time_fraction[name],
            )
            for name in ms_names
        },
        "timing_node_tasks": timing_node_tasks,
        "timing_distributed_application": timing,
        "compute_resources": compute_resources,
        "report": None,
    }
    if report is not None:
        from astroviper.distributed_applications.visibility_manipulation.uv_continuum_subtraction_report import (
            uv_continuum_subtraction_report,
        )
        from astroviper.utils.report import _report_name, make_report

        result["report"] = make_report(
            uv_continuum_subtraction_report,
            result,
            # One report per spectral window of the processing set.
            report_store=f"{_report_name(ps_store)}_{spectral_window_name}",
            report=report,
            fit={
                "spectral_window_name": spectral_window_name,
                "continuum_frequency_ranges": ranges,
                "continuum_frequency_frame": continuum_frequency_frame,
                "fit_order": fit_order,
                "ms_data_group_in_name": ms_data_group_in_name,
                "ms_data_group_out_name": ms_data_group_out_name,
            },
        )
    return result


def _check_driver_config(n_mapping_parallelism, thread_info, monitor_resources_seconds):
    """The checks of the driver's own settings."""
    if n_mapping_parallelism is not None:
        if set(n_mapping_parallelism) != {"time"}:
            raise ValueError(
                'n_mapping_parallelism must be {"time": n}, not '
                f"{n_mapping_parallelism!r}."
            )
        n = n_mapping_parallelism["time"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(
                f'n_mapping_parallelism["time"] must be a positive integer, not {n!r}.'
            )
    if thread_info is not None and not {"n_threads", "memory_per_thread"} <= set(
        thread_info
    ):
        raise ValueError(
            "thread_info must have the keys 'n_threads' and 'memory_per_thread'."
        )
    if monitor_resources_seconds is not None and not monitor_resources_seconds > 0:
        raise ValueError(
            "monitor_resources_seconds must be positive, not "
            f"{monitor_resources_seconds!r}."
        )


def _stored_metadata(ms_path):
    """The data groups and the ``{name: dimensions}`` of the arrays of an MSv4
    as stored, not as the consolidated metadata of the processing set gives
    them (arrays left by a failed run are only in the store)."""
    import zarr

    group = zarr.open_group(ms_path, mode="r", use_consolidated=False)
    variable_dims = {}
    for name, array in group.arrays():
        dims = (
            array.metadata.dimension_names
            if array.metadata.zarr_format == 3
            else array.attrs.get("_ARRAY_DIMENSIONS")
        )
        variable_dims[name] = tuple(dims or ())
    return dict(group.attrs.get("data_groups", {})), variable_dims


def _check_measurement_set(
    ms_xdt,
    ms_path,
    continuum_frequency_frame,
    ms_data_group_in_name,
    ms_data_group_out_name,
    out_modified,
    overwrite,
):
    """Metadata checks of one MSv4, before anything is written."""
    from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
        FRAMES_FROM_TOPO,
        VISIBILITY_DIMS,
        check_output_data_group,
    )

    ms_name = os.path.basename(ms_path)
    data_groups, variable_dims = _stored_metadata(ms_path)
    if ms_data_group_in_name not in data_groups:
        raise ValueError(
            f"{ms_name} has no data group {ms_data_group_in_name!r} (data groups "
            f"{sorted(data_groups)})."
        )
    data_group_in = data_groups[ms_data_group_in_name]
    for role in ("correlated_data", "flag", "weight"):
        if role not in data_group_in:
            raise ValueError(
                f"Data group {ms_data_group_in_name!r} of {ms_name} has no {role!r} role."
            )
    dims = variable_dims.get(data_group_in["correlated_data"])
    if dims != VISIBILITY_DIMS:
        raise ValueError(
            f"The correlated data of {ms_name} must have the dimensions "
            f"{VISIBILITY_DIMS}, not {dims}."
        )
    units = ms_xdt.frequency.attrs.get("units")
    if units != "Hz":
        raise ValueError(
            f"The frequency axis of {ms_name} must be in Hz, not {units!r}."
        )
    ms_frame = ms_xdt.frequency.attrs.get("observer")
    if continuum_frequency_frame != ms_frame and (
        ms_frame != "TOPO" or continuum_frequency_frame not in FRAMES_FROM_TOPO
    ):
        raise ValueError(
            f"Continuum ranges in {continuum_frequency_frame!r} cannot be converted "
            f"to the {ms_frame!r} frequencies of {ms_name}: use the frame of the "
            f"measurement set, or one of {list(FRAMES_FROM_TOPO)} for a 'TOPO' "
            "measurement set."
        )
    try:
        check_output_data_group(
            data_groups,
            variable_dims,
            ms_data_group_in_name,
            ms_data_group_out_name,
            out_modified,
            overwrite,
        )
    except ValueError as error:
        raise ValueError(f"{ms_name}: {error}") from error


def _time_blocks(n_time, n_target, max_task_length, input_time_chunk):
    """The number of blocks of times of an MSv4 by default.

    A task decompresses every on-disk time chunk of the input its block
    touches. Of the counts from the fewest blocks that fit the memory of a
    task to ``n_target``, the one whose blocks read the fewest input chunks,
    then the one closest to ``n_target``. Blocks have the equal lengths of
    ``make_parallel_coord``.
    """
    import numpy as np

    n_min = -(-n_time // max_task_length)
    best = None
    for n_chunks in range(n_min, max(n_min, n_target) + 1):
        length = -(-n_time // n_chunks)
        if length > 1 and -(-n_time // length) < n_chunks:
            length -= 1
        starts = np.arange(0, n_time, length)
        stops = np.minimum(starts + length, n_time)
        reads = int(np.sum(-(-stops // input_time_chunk) - starts // input_time_chunk))
        key = (reads, abs(n_chunks - n_target))
        if best is None or key < best[0]:
            best = (key, n_chunks)
    return best[1]


def _output_chunking(task_length, sizes, itemsize):
    """On-disk chunks of the output variables: whole frequency and polarization
    axes, at most :data:`OUTPUT_CHUNK_BYTES`, and a time chunk that divides a
    task's block of times, so that every task writes whole chunks."""
    bytes_per_baseline = sizes["frequency"] * sizes["polarization"] * itemsize
    baseline_chunk = int(
        max(1, min(sizes["baseline_id"], OUTPUT_CHUNK_BYTES // bytes_per_baseline))
    )
    bytes_per_time = baseline_chunk * bytes_per_baseline
    time_chunk = max(
        divisor
        for divisor in range(1, task_length + 1)
        if task_length % divisor == 0
        and (divisor == 1 or divisor * bytes_per_time <= OUTPUT_CHUNK_BYTES)
    )
    return {"time": time_chunk, "baseline_id": baseline_chunk}


def _create_output_variables(
    ms_path,
    ms_xdt,
    ms_data_group_in_name,
    ms_data_group_out_name,
    out_modified,
    parallel_coords,
    overwrite,
    create_empty_data_variables_on_disk,
    compressor,
):
    """Create the empty output variables of one MSv4.

    With ``overwrite`` the earlier output data group is unregistered before
    its variables are deleted, so that a run that fails leaves no data group
    over partly written variables."""
    import numpy as np
    import xarray as xr
    import zarr

    from astroviper.utils.data_group_tools import remove_data_group_xds

    data_group_in = ms_xdt.attrs["data_groups"][ms_data_group_in_name]
    visibility = ms_xdt[data_group_in["correlated_data"]]
    flag = ms_xdt[data_group_in["flag"]]
    if overwrite:
        group = zarr.open_group(ms_path, mode="r+", use_consolidated=False)
        stored = xr.Dataset(attrs={"data_groups": dict(group.attrs["data_groups"])})
        if remove_data_group_xds(stored, ms_data_group_out_name) is not None:
            group.attrs["data_groups"] = stored.attrs["data_groups"]
        for name in out_modified.values():
            if name in group:
                del group[name]
        zarr.consolidate_metadata(ms_path)
    definitions = {
        "correlated_data": {
            "dims": list(visibility.dims),
            "dtype": np.dtype(visibility.dtype).str,
            "name": out_modified["correlated_data"],
            "attrs": dict(visibility.attrs),
        },
        "flag": {
            "dims": list(flag.dims),
            "dtype": "|b1",
            "name": out_modified["flag"],
            "attrs": dict(flag.attrs),
        },
    }
    sizes = dict(ms_xdt.sizes)
    create_empty_data_variables_on_disk(
        ms_path,
        ["correlated_data", "flag"],
        shape_dict=sizes,
        parallel_coords=parallel_coords,
        compressor=compressor,
        double_precision=visibility.dtype == np.complex128,
        data_variable_definitions=definitions,
        image_chunking=_output_chunking(
            len(parallel_coords["time"]["data_chunks"][0]),
            sizes,
            np.dtype(visibility.dtype).itemsize,
        ),
    )


def _total_fit_statistics(fit_statistics):
    """The fit statistics of the tasks, per MSv4 and polarization."""
    import numpy as np
    import pandas as pd

    from astroviper.processing_functions.visibility_manipulation.uv_continuum_subtraction import (
        FIT_STATISTICS_MAXIMA,
        FIT_STATISTICS_MINIMA,
        FIT_STATISTICS_SUMS,
    )

    if not len(fit_statistics):
        return pd.DataFrame()
    grouped = fit_statistics.groupby(["ms_name", "polarization"], sort=False)
    total = pd.concat(
        [
            grouped[list(FIT_STATISTICS_SUMS)].sum(),
            grouped[list(FIT_STATISTICS_MINIMA)].min(),
            grouped[list(FIT_STATISTICS_MAXIMA)].max(),
        ],
        axis=1,
    )
    for part in ("real", "imag"):
        total[f"chi_squared_average_{part}"] = total[f"chi_squared_sum_{part}"] / total[
            "n_fitted"
        ].where(total["n_fitted"] > 0, np.nan)
    columns = [
        "n_spectra",
        "n_fitted",
        "n_order_reduced",
        "n_without_continuum",
        "n_flagged_without_continuum",
    ] + [
        f"chi_squared_{statistic}_{part}"
        for statistic in ("average", "min", "max")
        for part in ("real", "imag")
    ]
    return total[columns].reset_index()


def _mean_spectra(ms_xdt, sums, continuum_time_fraction):
    """The weighted mean spectra of the input and output visibilities of one
    MSv4, with the fraction of the times at which each channel is a continuum
    channel."""
    import numpy as np
    import xarray as xr

    def mean(weighted, weight):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight > 0, weighted / weight, np.nan + 0j)

    coords = {
        "frequency": ms_xdt.frequency.values,
        "polarization": ms_xdt.polarization.values,
    }
    dataset = xr.Dataset(
        {
            "VISIBILITY_MEAN_INPUT": (
                ("frequency", "polarization"),
                mean(sums["sum_weighted_visibility_in"], sums["sum_weight_in"]),
            ),
            "VISIBILITY_MEAN_OUTPUT": (
                ("frequency", "polarization"),
                mean(sums["sum_weighted_visibility_out"], sums["sum_weight_out"]),
            ),
            "CONTINUUM_TIME_FRACTION": ("frequency", continuum_time_fraction),
        },
        coords=coords,
    )
    dataset.frequency.attrs.update(ms_xdt.frequency.attrs)
    return dataset
