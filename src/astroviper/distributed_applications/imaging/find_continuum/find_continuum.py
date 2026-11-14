"""Continuum channel finding of the ALMA pipeline (``hif_findcont``).

Runs the pipeline's findContinuum (vendored in ``findContinuum.py``) on one
spectral window of a dirty cube in an xradio image store. findContinuum's CASA
tasks and tools are those of :mod:`casa_image_tasks`. In :func:`find_continuum`
every reduction of the cube is its own map-reduce over frequency chunks, so the
cube is read as often as the pipeline reads it.
"""

import contextlib
import glob
import io
import os
import sys
import time

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astroviper.utils.param_docs import shares_param_docs

_PARAM_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# The layout of ``timing_distributed_application`` (format_timing_summary and
# the processing section of the report). findContinuum runs in the driver and
# asks for one pass over the image store at a time: the graph and compute
# timings are summed over the passes, and the driver's work between them (2-D
# images, spectra, plots) is the last step.
DISTRIBUTED_APPLICATION_TIMING_PHASES = [
    (
        "DISTRIBUTED APPLICATION (driver)",
        None,
        [
            ("open and check the image", "T_open_image"),
            ("register the cubes, parallel coords", "T_register_cubes"),
            ("create map/reduce graphs (all passes)", "T_create_map_reduce_graph"),
            ("generate dask graphs (all passes)", "T_generate_dask_graph"),
            ("compute dask graphs (all passes)", "T_compute_dask_graph"),
            ("findContinuum between the passes", "T_find_continuum_between_passes"),
        ],
    ),
]
DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY = "T_total"
NODE_TASK_TIMING_PHASES = [
    (
        "NODE TASK (one chunk of one pass)",
        None,
        [("load the chunk", "T_load"), ("reductions", "T_reduce")],
    ),
]
NODE_TASK_TIMING_TOTAL_KEY = "T_find_continuum_task"


class _LogStream(io.TextIOBase):
    """Sends findContinuum's printed lines to the logger at debug level and
    keeps the last one."""

    def __init__(self):
        self.pending = ""
        self.last = ""

    def write(self, text):
        *lines, self.pending = (self.pending + text).split("\n")
        for line in lines:
            if line.strip():
                logger.debug(line)
                self.last = line.strip()
        return len(text)


def _cube_metadata(lazy_xds, variable, beam_variable, telescope_name):
    from astroviper.processing_functions.imaging.find_continuum import (
        linear_frequency_axis,
    )

    reference_frequency, frequency_increment, reference_channel, n_channels = (
        linear_frequency_axis(lazy_xds.frequency.values)
    )
    l_values, m_values = lazy_xds.l.values, lazy_xds.m.values
    direction = lazy_xds.attrs["coordinate_system_info"]["reference_direction"]
    beams = None
    if beam_variable is not None:
        beam = lazy_xds[beam_variable]
        if beam.attrs.get("units", "rad") != "rad":
            raise ValueError(f"{beam_variable} must be in radians")
        beams = beam.sel(polarization="I").values[0].tolist()
    return {
        "shape": [l_values.size, m_values.size, 1, n_channels],
        "increment": [
            float(l_values[1] - l_values[0]),
            float(m_values[1] - m_values[0]),
            1.0,
            frequency_increment,
        ],
        "reference_value": [
            float(direction["data"][0]),
            float(direction["data"][1]),
            1.0,
            reference_frequency,
        ],
        "reference_pixel": [
            float(np.argmin(np.abs(l_values))),
            float(np.argmin(np.abs(m_values))),
            0.0,
            reference_channel,
        ],
        "rest_frequency": float(lazy_xds.frequency.attrs["rest_frequency"]["data"]),
        "spectral_frame": lazy_xds.frequency.attrs["frame"],
        "direction_frame": "J2000"
        if direction["attrs"]["frame"].lower() == "fk5"
        else direction["attrs"]["frame"].upper(),
        "beams": beams,
        "brightness_unit": lazy_xds[variable].attrs.get("units", ""),
        "telescope": telescope_name,
        "object": "",
        "epoch_mjd": None,
    }


def _rss_peak_bytes():
    """Peak resident memory of this process (``ru_maxrss`` is in kB on Linux
    and in bytes on macOS)."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def find_continuum(
    input_image_store: str,
    atmospheric_transmission,
    atmospheric_sky_temperature,
    work_directory: str,
    image_data_group_in_name: str = "residual",
    telescope_name: str = "ALMA",
    single_continuum: bool = False,
    n_bin: int = 1,
    spectral_dynamic_range_bandwidth: float | None = None,
    primary_beam_limit: float | None = 0.2,
    n_mapping_parallelism: dict | None = None,
    channel_range: list[int] | None = None,
    monitor_resources_seconds: float | None = None,
    report: str | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """Find the continuum channels of a dirty cube as the ALMA pipeline does.

    Parameters
    ----------
    input_image_store : str
        xradio image store of the dirty cube (Stokes I is used).
    atmospheric_transmission : list or numpy.ndarray
        Atmospheric transmission at every channel of ``channel_range`` (0 to
        1), as CASA's atmosphere model gives it for the observation.
    atmospheric_sky_temperature : list or numpy.ndarray
        Sky brightness temperature at every channel of ``channel_range`` (K).
    work_directory : str
        Directory for findContinuum's plots, spectra and intermediate images.
        It must not hold the products of an earlier run on a cube of the same
        name: findContinuum would reuse them, as in the pipeline.
    image_data_group_in_name : str, default "residual"
        Image data group with the roles ``sky``, ``primary_beam`` and
        ``beam_fit_params_point_spread_function`` (findContinuum reads the
        point spread function through its beams only).
    telescope_name : str, default "ALMA"
        Telescope, as findContinuum reads it from a CASA image header.
    single_continuum : bool, default False
        The spectral window is a single-continuum setup [findContinuum
        ``singleContinuum``].
    n_bin : int, default 1
        Channels to average in the mean spectrum [findContinuum ``nbin``].
    spectral_dynamic_range_bandwidth : float, optional
        The requested spectral dynamic range bandwidth (Hz) [findContinuum
        ``spectralDynamicRangeBandWidth``].
    primary_beam_limit : float, default 0.2
        Pixels where the primary beam is below this level are masked in the
        dirty cube and the primary beam, as in the cube ``hif_findcont`` makes
        with ``tclean(pblimit=0.2)``; ``None`` reads the cubes as they are.
    n_mapping_parallelism : dict, optional
        ``{"frequency": n}``, the number of frequency chunks of every
        map-reduce; one chunk if not given.
    channel_range : list of int, optional
        ``[start, stop)``, the channels of the cube that findContinuum sees as
        its cube: one spectral window, on which the frequencies are linear (a
        run of
        :func:`~astroviper.processing_functions.imaging.utils.spectral_windows.linear_channel_ranges`);
        in a cube of several windows its middle channel is the rest frequency
        (:func:`~astroviper.processing_functions.imaging.utils.spectral_windows.window_rest_frequency`).
        ``None`` (default): every channel, refused for a cube of several
        spectral windows.
    monitor_resources_seconds : float, optional
        If set, sample each node task's worker-process CPU, memory and I/O
        every this many seconds (graphviper's resource monitor) and carry the
        series into ``timing_node_tasks`` as list-valued columns, as
        ``image_cube_single_field`` does; the driver is sampled while
        findContinuum runs. Requires ``psutil``.
    report : str, optional
        Make a report of the run: ``"html"`` or ``"pdf"``, written in
        ``work_directory`` as ``<name>_report.html`` or ``.pdf``, or
        ``"notebook"``, displayed in the calling notebook. It has the
        processing section of every distributed application (timing, task
        stream and, with ``monitor_resources_seconds``, CPU, memory and I/O)
        and the results of findContinuum. A failure while making it is logged
        and leaves the results intact. ``None`` (default): no report.
    input_checked : bool, default False
        Skip the configuration checks already done by the calling layer. The
        checks of the opened image and of the work directory always run.

    Returns
    -------
    dict
        ``channel_selection`` (CASA channel ranges of the spectral window,
        ``"a~b;c~d"``), ``aggregate_bandwidth`` (GHz), ``all_continuum``,
        ``warnings``, ``joint_mask`` (path of the final joint mask),
        ``moment_difference_snr``, ``plot`` (path of the summary plot),
        ``channel_range`` (``[start, stop)`` in channels of the cube),
        ``n_cube_reductions`` (map-reduce passes over the image store),
        ``cube_reads`` (data variable to the number of passes that read it),
        ``passes`` (one row per pass: the reductions asked for, the data
        variables read and the graph and compute times), ``timing_node_tasks``
        (one row per node task of every pass), ``timing_reduce_nodes``,
        ``timing_distributed_application`` (the driver steps of
        :data:`DISTRIBUTED_APPLICATION_TIMING_PHASES`, ``T_total``, the
        wall-clock anchors, ``driver_rss_peak_bytes`` (the peak resident
        memory of the driver during the call: the process's peak when the
        call raised it, else the largest sample of ``driver_resource_usage``,
        else ``None``), ``driver_rss_lifetime_peak_bytes`` (the process's peak
        since it started, ``getrusage``), ``driver_bytes_read`` (the bytes the
        driver read from the image store) and, with
        ``monitor_resources_seconds``, ``driver_resource_usage``) and
        ``report`` (the path of the report written, else ``None``).

    Raises
    ------
    ValueError
        If the configuration or the image fails the checks of
        :mod:`~astroviper.processing_functions.imaging.config_check.param_check_find_continuum`,
        ``work_directory`` holds the products of an earlier run, or
        findContinuum finds no result, with the reason it gave (for example a
        window of 7 channels or fewer, which gives no two edge channels at its
        ``nBaselineChannels`` of 0.19).

    Notes
    -----
    findContinuum runs in ``work_directory`` (the working directory of the
    process changes while it runs) and draws with pyplot, so a process runs
    one call at a time; spectral windows run in parallel in separate
    processes.
    """
    import dask
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    from graphviper.graph_tools import generate_dask_workflow, map, reduce
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )
    from graphviper.graph_tools.map import monitor_node_task

    import astroviper.node_tasks.imaging.find_continuum as node_tasks
    from astroviper.distributed_applications.imaging.find_continuum import (
        casa_image_tasks,
        findContinuum,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_channel_range,
        check_find_continuum_config,
        check_find_continuum_inputs,
    )
    from astroviper.processing_functions.imaging.find_continuum import (
        finalize_find_continuum_reductions,
    )
    from astroviper.processing_functions.imaging.utils.spectral_windows import (
        window_rest_frequency,
    )
    from astroviper.utils.timing import format_timing_summary

    application_start = time.time()
    rss_peak_start = _rss_peak_bytes()
    timing_distributed_application = {
        key: 0.0
        for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES
        for _, key in steps
    }
    if not input_checked:
        check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism)
    # findContinuum runs in its work directory, as in the pipeline, and node
    # tasks may run elsewhere, so every path is made absolute.
    input_image_store = os.path.abspath(input_image_store)
    work_directory = os.path.abspath(work_directory)
    stem = os.path.join(
        work_directory, os.path.basename(input_image_store).split(".")[0]
    )
    if glob.glob(glob.escape(stem) + ".*"):
        raise ValueError(
            f"{work_directory} holds the products of an earlier run on "
            f"{os.path.basename(stem)}, which findContinuum would reuse"
        )
    cube_xds = xr.open_zarr(input_image_store, chunks=None)
    window_start, window_stop = check_find_continuum_channel_range(
        channel_range, cube_xds.frequency.values
    )
    lazy_xds = cube_xds.isel(frequency=slice(window_start, window_stop))
    check_find_continuum_inputs(
        lazy_xds,
        image_data_group_in_name,
        atmospheric_transmission,
        atmospheric_sky_temperature,
    )
    data_group = lazy_xds.attrs["data_groups"][image_data_group_in_name]
    data_variables = {role: data_group[role] for role in ("sky", "primary_beam")}
    timing_distributed_application["T_open_image"] = time.time() - application_start

    step_start = time.time()
    os.makedirs(work_directory, exist_ok=True)
    beam_variable = data_group["beam_fit_params_point_spread_function"]
    polarization_index = lazy_xds.polarization.values.tolist().index("I")
    rest_frequency = window_rest_frequency(
        cube_xds.frequency.values,
        [window_start, window_stop],
        float(cube_xds.frequency.attrs["rest_frequency"]["data"]),
    )
    cubes = {}
    for suffix, variable, beam in (
        (".residual", data_variables["sky"], beam_variable),
        (".pb", data_variables["primary_beam"], None),
        (".psf", data_group.get("point_spread_function"), beam_variable),
    ):
        path = stem + suffix
        cubes[path] = {
            "store": input_image_store,
            "variable": variable,
            "metadata": dict(
                _cube_metadata(lazy_xds, data_variables["sky"], beam, telescope_name),
                rest_frequency=rest_frequency,
            ),
            "channel_offset": window_start,
            "polarization_index": polarization_index,
            "primary_beam": data_variables["primary_beam"],
            "primary_beam_limit": primary_beam_limit,
        }
    cubes[stem + ".residual"]["atmosphere"] = {
        "transmission": np.asarray(atmospheric_transmission, dtype=np.float64),
        "tsky": np.asarray(atmospheric_sky_temperature, dtype=np.float64),
    }
    timing, cube_reads, passes, reduce_records = [], {}, [], []
    primary_beam = data_variables["primary_beam"]

    def run_reductions(requests):
        start = time.time()
        variables, image_paths = {}, {}
        for request in requests:
            variables[request["image"]] = cubes[request["image"]]["variable"]
            small_images, mask_cubes = casa_image_tasks._mask_references(
                request["mask"]
            )
            for name in mask_cubes:
                variables[name] = cubes[name]["variable"]
            image_paths.update({name: os.path.abspath(name) for name in small_images})
        read = set(variables.values())
        if primary_beam_limit is not None:
            read.add(primary_beam)
        for variable in read:
            cube_reads[variable] = cube_reads.get(variable, 0) + 1
        viper_graph = map(
            input_data={"img": index_xds},
            node_task_data_mapping=node_task_data_mapping,
            node_task=node_tasks.find_continuum_cube_reductions,
            input_params={
                "input_image_store": input_image_store,
                "variables": variables,
                "image_paths": image_paths,
                "requests": requests,
                "primary_beam": primary_beam,
                "primary_beam_limit": primary_beam_limit,
                "channel_offset": window_start,
            },
            in_memory_compute=False,
            monitor_resources_seconds=monitor_resources_seconds,
        )
        viper_graph = reduce(
            viper_graph,
            node_tasks.merge_find_continuum_cube_reductions,
            {},
            mode="tree",
        )
        graph_end = time.time()
        dask_graph = generate_dask_workflow(viper_graph)
        compute_start = time.time()
        output = dask.compute(dask_graph)[0]
        compute_end = time.time()
        index = len(passes)
        # A pass of one task has no reduce node to fold the monitor series in.
        frame, usage = output["timing_node_tasks"], output.get("resource_usage")
        if usage is not None:
            frame = node_tasks.with_resource_usage(frame, usage)
        timing.append(frame.assign(reduction=index))
        reduce_records.extend(
            dict(record, reduction=index)
            for record in output.get("timing_reduce_nodes", [])
        )
        passes.append(
            {
                "pass": index,
                "reductions": ", ".join(
                    f"{request['kind']} of {os.path.basename(request['image'])}"
                    for request in requests
                ),
                "data_variables_read": ", ".join(sorted(read)),
                "n_tasks": len(frame),
                "start_unixtime": compute_start,
                "T_create_map_reduce_graph": graph_end - start,
                "T_generate_dask_graph": compute_start - graph_end,
                "T_compute_dask_graph": compute_end - compute_start,
            }
        )
        for key in (
            "T_create_map_reduce_graph",
            "T_generate_dask_graph",
            "T_compute_dask_graph",
        ):
            timing_distributed_application[key] += passes[-1][key]
        timing_distributed_application.setdefault(
            "compute_start_unixtime", compute_start
        )
        timing_distributed_application["compute_end_unixtime"] = compute_end
        logger.debug(
            f"find_continuum pass {index} ({passes[-1]['reductions']}) in "
            f"{compute_end - start:.2f} s"
        )
        return finalize_find_continuum_reductions(output["reductions"], requests)

    index_xds = cube_xds.assign_coords(
        frequency=np.arange(cube_xds.sizes["frequency"], dtype=np.float64)
    )
    n_chunks = (
        1 if n_mapping_parallelism is None else n_mapping_parallelism["frequency"]
    )
    parallel_coords = {
        "frequency": make_parallel_coord(
            coord=index_xds.frequency[window_start:window_stop], n_chunks=n_chunks
        )
    }
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, {"img": index_xds}
    )
    for path in cubes:
        os.makedirs(path, exist_ok=True)
    timing_distributed_application["T_register_cubes"] = time.time() - step_start

    def call_find_continuum(input_params):
        return {
            "result": findContinuum.findContinuum(
                img=stem + ".residual",
                pbcube=stem + ".pb",
                psfcube=stem + ".psf",
                singleContinuum=single_continuum,
                returnAllContinuumBoolean=True,
                returnWarnings=True,
                vis="",
                nbin=n_bin,
                spectralDynamicRangeBandWidth=spectral_dynamic_range_bandwidth,
                returnMomDiffSNR=True,
            )
        }

    if monitor_resources_seconds:
        call_find_continuum = monitor_node_task(
            call_find_continuum, monitor_resources_seconds
        )
    figures = set(plt.get_fignums())
    log_stream = _LogStream()
    find_continuum_start = time.time()
    try:
        with (
            contextlib.chdir(work_directory),
            contextlib.redirect_stdout(log_stream),
            casa_image_tasks.cube_session(cubes, run_reductions) as session,
        ):
            output = call_find_continuum({})
    finally:
        log_stream.write("\n")
        for number in set(plt.get_fignums()) - figures:
            plt.close(number)
    if output["result"] is None:
        # findContinuum prints its reason before it returns None.
        raise ValueError(
            f"findContinuum found no result for channels [{window_start}, "
            f"{window_stop}) of {input_image_store} ({window_stop - window_start} "
            "channels)" + (f": {log_stream.last}" if log_stream.last else "")
        )
    find_continuum_seconds = time.time() - find_continuum_start
    timing_distributed_application["T_find_continuum_between_passes"] = (
        find_continuum_seconds
        - sum(
            timing_distributed_application[key]
            for key in (
                "T_create_map_reduce_graph",
                "T_generate_dask_graph",
                "T_compute_dask_graph",
            )
        )
    )
    application_end = time.time()
    timing_distributed_application["T_total"] = application_end - application_start
    timing_distributed_application["application_start_unixtime"] = application_start
    timing_distributed_application["application_end_unixtime"] = application_end
    rss_peak = _rss_peak_bytes()
    timing_distributed_application["driver_rss_lifetime_peak_bytes"] = rss_peak
    # The process's peak is the call's when the call raised it; otherwise only
    # the driver's samples, if any, tell the call's peak.
    samples = (output.get("resource_usage") or {}).get("memory_rss_bytes") or []
    timing_distributed_application["driver_rss_peak_bytes"] = (
        rss_peak
        if rss_peak > rss_peak_start
        else (int(max(samples)) if samples else None)
    )
    timing_distributed_application["driver_bytes_read"] = sum(
        session.counters.get(key, 0) for key in ("bytes", "index_bytes")
    )
    if monitor_resources_seconds:
        timing_distributed_application["driver_resource_usage"] = output.get(
            "resource_usage"
        )
    (
        selection,
        plot,
        aggregate_bandwidth,
        all_continuum,
        warnings,
        joint_mask,
        moment_difference_snr,
    ) = output["result"]
    timing_node_tasks = (
        pd.concat(timing, ignore_index=True) if timing else pd.DataFrame()
    )
    logger.info(
        format_timing_summary(
            timing_distributed_application,
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            total_key=DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
            title="find_continuum timing (driver, seconds)",
            total_label="TOTAL (driver wall time)",
        )
    )
    if len(timing_node_tasks):
        logger.info(
            format_timing_summary(
                timing_node_tasks.sum(numeric_only=True).to_dict(),
                NODE_TASK_TIMING_PHASES,
                total_key=NODE_TASK_TIMING_TOTAL_KEY,
                title="find_continuum node-task timing: TOTAL over all tasks (seconds)",
            )
        )
    return_dict = {
        "channel_selection": selection,
        "aggregate_bandwidth": aggregate_bandwidth,
        "all_continuum": bool(all_continuum),
        "warnings": list(warnings),
        "joint_mask": joint_mask,
        "moment_difference_snr": moment_difference_snr,
        "plot": plot,
        "channel_range": [window_start, window_stop],
        "n_cube_reductions": len(passes),
        "cube_reads": cube_reads,
        "passes": pd.DataFrame(passes),
        "timing_node_tasks": timing_node_tasks,
        "timing_reduce_nodes": reduce_records,
        "timing_distributed_application": timing_distributed_application,
        "report": None,
    }
    if report is not None:
        from astroviper.distributed_applications.imaging.find_continuum.find_continuum_report import (
            find_continuum_report,
        )
        from astroviper.utils.report import make_report

        return_dict["report"] = make_report(
            find_continuum_report,
            return_dict,
            report_store=stem,
            report=report,
        )
    return return_dict
