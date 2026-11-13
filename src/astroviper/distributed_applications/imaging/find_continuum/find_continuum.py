"""Continuum channel finding of the ALMA pipeline (``hif_findcont``).

Runs the pipeline's findContinuum (vendored in ``findContinuum.py``) on a dirty
cube in an xradio image store. findContinuum's CASA tasks and tools are those of
:mod:`casa_image_tasks`. In :func:`find_continuum` every reduction of the cube
is its own map-reduce over frequency chunks, so the cube is read as often as
the pipeline reads it; :func:`find_continuum_fused` gives the same results in
fewer passes (:mod:`fused_reductions`).
"""

import contextlib
import glob
import io
import os
import time

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter

from astroviper.utils.param_docs import shares_param_docs

_PARAM_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# The layout of ``timing_distributed_application`` (format_timing_summary and
# the processing section of the report). findContinuum runs in the driver and
# asks for one pass over the image store at a time: the graph and compute
# timings are summed over the passes, and the driver's work between them
# (2-D images, spectra, plots) is the last step.
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
    """Sends findContinuum's printed lines to the logger at debug level."""

    def __init__(self):
        self.pending = ""

    def write(self, text):
        *lines, self.pending = (self.pending + text).split("\n")
        for line in lines:
            if line.strip():
                logger.debug(line)
        return len(text)


def _cube_metadata(lazy_xds, variable, beam_variable, telescope_name):
    frequency = lazy_xds.frequency.values
    l_values, m_values = lazy_xds.l.values, lazy_xds.m.values
    direction = lazy_xds.attrs["coordinate_system_info"]["reference_direction"]
    beams = None
    if beam_variable is not None:
        beam = lazy_xds[beam_variable]
        if beam.attrs.get("units", "rad") != "rad":
            raise ValueError(f"{beam_variable} must be in radians")
        beams = beam.sel(polarization="I").values[0].tolist()
    return {
        "shape": [l_values.size, m_values.size, 1, frequency.size],
        "increment": [
            float(l_values[1] - l_values[0]),
            float(m_values[1] - m_values[0]),
            1.0,
            float(frequency[1] - frequency[0]) if frequency.size > 1 else 1.0,
        ],
        "reference_value": [
            float(direction["data"][0]),
            float(direction["data"][1]),
            1.0,
            float(frequency[0]),
        ],
        "reference_pixel": [
            float(np.argmin(np.abs(l_values))),
            float(np.argmin(np.abs(m_values))),
            0.0,
            0.0,
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
        Atmospheric transmission at every channel (0 to 1), as CASA's
        atmosphere model gives it for the observation.
    atmospheric_sky_temperature : list or numpy.ndarray
        Sky brightness temperature at every channel (K).
    work_directory : str
        Directory for findContinuum's plots, spectra and intermediate images.
        It must not hold the products of an earlier run on a cube of the same
        name: findContinuum would reuse them, as in the pipeline.
    image_data_group_in_name : str, default "residual"
        Image data group with the roles ``sky``, ``primary_beam``,
        ``point_spread_function`` and ``beam_fit_params_point_spread_function``.
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
    monitor_resources_seconds : float, optional
        If set, sample each node task's worker-process CPU, memory and I/O
        every this many seconds (graphviper's resource monitor) and carry the
        series into ``timing_node_tasks`` as list-valued columns, as
        ``image_cube_single_field`` does. Requires ``psutil``.
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
        ``channel_selection`` (CASA channel ranges, ``"a~b;c~d"``),
        ``aggregate_bandwidth`` (GHz), ``all_continuum``, ``warnings``,
        ``joint_mask`` (path of the final joint mask), ``moment_difference_snr``,
        ``plot`` (path of the summary plot), ``n_cube_reductions`` (map-reduce
        passes over the image store), ``cube_reads`` (data variable to the
        number of passes that read it), ``passes`` (one row per pass: the
        reductions asked for, the data variables read and the graph and
        compute times), ``timing_node_tasks`` (one row per node task of every
        pass), ``timing_reduce_nodes``, ``timing_distributed_application``
        (the driver steps of :data:`DISTRIBUTED_APPLICATION_TIMING_PHASES`,
        ``T_total`` and the wall-clock anchors) and ``report`` (the path of
        the report written, else ``None``).

    Raises
    ------
    ValueError
        If the configuration or the image fails the checks of
        :mod:`~astroviper.processing_functions.imaging.config_check.param_check_find_continuum`,
        or ``work_directory`` holds the products of an earlier run.

    Notes
    -----
    findContinuum runs in ``work_directory`` (the working directory of the
    process changes while it runs) and draws with pyplot, so a process runs
    one call at a time; spectral windows run in parallel in separate
    processes.
    """
    return _find_continuum(
        input_image_store,
        atmospheric_transmission,
        atmospheric_sky_temperature,
        work_directory,
        image_data_group_in_name,
        telescope_name,
        single_continuum,
        n_bin,
        spectral_dynamic_range_bandwidth,
        primary_beam_limit,
        n_mapping_parallelism,
        input_checked=input_checked,
        fused=False,
        monitor_resources_seconds=monitor_resources_seconds,
        report=report,
    )


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def find_continuum_fused(
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
    first_round_reductions: dict | None = None,
    monitor_resources_seconds: float | None = None,
    report: str | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """Find the continuum channels as :func:`find_continuum` does, in fewer passes.

    findContinuum's reductions of the cube are answered by
    :class:`~astroviper.distributed_applications.imaging.find_continuum.fused_reductions.FusedReductions`:
    a pass over the image store also computes the reductions findContinuum
    asks for next, as far as they are known by then, and later requests are
    answered from them. The results are those of :func:`find_continuum`.

    Parameters
    ----------
    input_image_store : str
        xradio image store of the dirty cube (Stokes I is used).
    atmospheric_transmission : list or numpy.ndarray
        Atmospheric transmission at every channel (0 to 1), as CASA's
        atmosphere model gives it for the observation.
    atmospheric_sky_temperature : list or numpy.ndarray
        Sky brightness temperature at every channel (K).
    work_directory : str
        Directory for findContinuum's plots, spectra and intermediate images.
        It must not hold the products of an earlier run on a cube of the same
        name: findContinuum would reuse them, as in the pipeline.
    image_data_group_in_name : str, default "residual"
        Image data group with the roles ``sky``, ``primary_beam``,
        ``point_spread_function`` and ``beam_fit_params_point_spread_function``.
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
    first_round_reductions : dict, optional
        The first-round reductions of the cube, as
        :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`
        returns them in ``find_continuum_first_round`` when it makes the cube
        (with the same primary beam limit); the pass that would compute them
        is then not made.
    monitor_resources_seconds : float, optional
        If set, sample each node task's worker-process CPU, memory and I/O
        every this many seconds (graphviper's resource monitor) and carry the
        series into ``timing_node_tasks`` as list-valued columns, as
        ``image_cube_single_field`` does. Requires ``psutil``.
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
        As :func:`find_continuum`.

    Raises
    ------
    ValueError
        As :func:`find_continuum`.

    Notes
    -----
    As :func:`find_continuum`, a process runs one call at a time.
    """
    return _find_continuum(
        input_image_store,
        atmospheric_transmission,
        atmospheric_sky_temperature,
        work_directory,
        image_data_group_in_name,
        telescope_name,
        single_continuum,
        n_bin,
        spectral_dynamic_range_bandwidth,
        primary_beam_limit,
        n_mapping_parallelism,
        input_checked=input_checked,
        fused=True,
        first_round_reductions=first_round_reductions,
        monitor_resources_seconds=monitor_resources_seconds,
        report=report,
    )


def _find_continuum(
    input_image_store,
    atmospheric_transmission,
    atmospheric_sky_temperature,
    work_directory,
    image_data_group_in_name,
    telescope_name,
    single_continuum,
    n_bin,
    spectral_dynamic_range_bandwidth,
    primary_beam_limit,
    n_mapping_parallelism,
    *,
    input_checked,
    fused,
    first_round_reductions=None,
    monitor_resources_seconds=None,
    report=None,
):
    import dask
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    from graphviper.graph_tools import generate_dask_workflow, map, reduce
    from graphviper.graph_tools.coordinate_utils import (
        interpolate_data_coords_onto_parallel_coords,
        make_parallel_coord,
    )

    import astroviper.node_tasks.imaging.find_continuum as node_tasks
    from astroviper.distributed_applications.imaging.find_continuum import (
        casa_image_tasks,
        findContinuum,
    )
    from astroviper.distributed_applications.imaging.find_continuum.fused_reductions import (
        FusedReductions,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_config,
        check_find_continuum_inputs,
    )
    from astroviper.processing_functions.imaging.find_continuum import (
        finalize_find_continuum_reductions,
    )
    from astroviper.utils.timing import format_timing_summary

    application_start = time.time()
    timing_distributed_application = {
        key: 0.0
        for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES
        for _, key in steps
    }
    if not input_checked:
        check_find_continuum_config(
            n_bin, primary_beam_limit, n_mapping_parallelism, first_round_reductions
        )
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
    lazy_xds = xr.open_zarr(input_image_store, chunks=None)
    check_find_continuum_inputs(
        lazy_xds,
        image_data_group_in_name,
        atmospheric_transmission,
        atmospheric_sky_temperature,
    )
    data_group = lazy_xds.attrs["data_groups"][image_data_group_in_name]
    nchan = lazy_xds.sizes["frequency"]
    timing_distributed_application["T_open_image"] = time.time() - application_start

    start = time.time()
    os.makedirs(work_directory, exist_ok=True)
    beam_variable = data_group["beam_fit_params_point_spread_function"]
    cubes = {}
    for suffix, role, beam in (
        (".residual", "sky", beam_variable),
        (".pb", "primary_beam", None),
        (".psf", "point_spread_function", beam_variable),
    ):
        path = stem + suffix
        os.makedirs(path, exist_ok=True)
        cubes[path] = {
            "store": input_image_store,
            "variable": data_group[role],
            "metadata": _cube_metadata(
                lazy_xds, data_group["sky"], beam, telescope_name
            ),
        }
    cubes[stem + ".residual"]["atmosphere"] = {
        "transmission": np.asarray(atmospheric_transmission, dtype=np.float64),
        "tsky": np.asarray(atmospheric_sky_temperature, dtype=np.float64),
    }

    index_xds = lazy_xds.assign_coords(frequency=np.arange(nchan, dtype=np.float64))
    n_chunks = (
        1 if n_mapping_parallelism is None else n_mapping_parallelism["frequency"]
    )
    parallel_coords = {
        "frequency": make_parallel_coord(coord=index_xds.frequency, n_chunks=n_chunks)
    }
    node_task_data_mapping = interpolate_data_coords_onto_parallel_coords(
        parallel_coords, {"img": index_xds}
    )
    timing_distributed_application["T_register_cubes"] = time.time() - start
    timing, cube_reads, passes, reduce_records = [], {}, [], []
    primary_beam = data_group["primary_beam"]

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

    figures = set(plt.get_fignums())
    log_stream = _LogStream()
    find_continuum_start = time.time()
    try:
        with (
            contextlib.chdir(work_directory),
            contextlib.redirect_stdout(log_stream),
            casa_image_tasks.cube_session(
                cubes,
                FusedReductions(
                    run_reductions,
                    stem + ".residual",
                    stem + ".pb",
                    None
                    if first_round_reductions is None
                    else first_round_reductions["reductions"],
                )
                if fused
                else run_reductions,
            ),
        ):
            result = findContinuum.findContinuum(
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
    finally:
        log_stream.write("\n")
        for number in set(plt.get_fignums()) - figures:
            plt.close(number)
    timing_distributed_application["T_find_continuum_between_passes"] = (
        time.time()
        - find_continuum_start
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
    (
        selection,
        plot,
        aggregate_bandwidth,
        all_continuum,
        warnings,
        joint_mask,
        moment_difference_snr,
    ) = result
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
        "n_cube_reductions": len(timing),
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
            application="find_continuum_fused" if fused else "find_continuum",
        )
    return return_dict
