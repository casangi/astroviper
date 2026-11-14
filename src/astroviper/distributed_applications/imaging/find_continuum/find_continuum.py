"""Continuum channel finding of the ALMA pipeline (``hif_findcont``).

Runs the pipeline's findContinuum (vendored in ``findContinuum.py``) on one
spectral window of a dirty cube in an xradio image store. findContinuum's CASA
tasks and tools are those of :mod:`casa_image_tasks`. In :func:`find_continuum`
every reduction of the cube is its own map-reduce over frequency chunks, so the
cube is read as often as the pipeline reads it; :func:`find_continuum_fused`
gives the same results in fewer passes (:mod:`fused_reductions`), and
:func:`find_continuum_ultra` streams those passes plane by plane and keeps the
moment maps out of their reduce (:mod:`ultra_reductions`).
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
# timings are summed over the passes (find_continuum_ultra's merges of the
# moment maps included), and the driver's work between them (2-D images,
# spectra, plots) is the last step.
DISTRIBUTED_APPLICATION_TIMING_PHASES = [
    (
        "DISTRIBUTED APPLICATION (driver)",
        None,
        [
            ("open and check the image", "T_open_image"),
            ("load the imager's first round", "T_load_first_round"),
            ("register the cubes, parallel coords", "T_register_cubes"),
            ("create map/reduce graphs (all passes)", "T_create_map_reduce_graph"),
            ("generate dask graphs (all passes)", "T_generate_dask_graph"),
            ("compute dask graphs (all passes)", "T_compute_dask_graph"),
            ("mean primary beam in the driver", "T_primary_beam_mean"),
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
NODE_TASK_TIMING_PHASES_ULTRA = [
    (
        "FREQUENCY TASK (one chunk of one pass)",
        None,
        [
            ("wait for the reads", "T_read_wait"),
            ("masks", "T_mask"),
            ("moments", "T_moments"),
            ("statistics", "T_statistics"),
            ("spectra", "T_profile"),
            ("first round", "T_first_round"),
            ("write the moment partials", "T_write_partials"),
        ],
    ),
    (
        "MERGE TASK (one tile of the moment maps of one pass)",
        None,
        [
            ("read the partials", "T_read_partials"),
            ("merge", "T_merge"),
            ("write the moment maps", "T_write_results"),
        ],
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


def _ultra_budget_bytes(thread_info, memory_budget_gb):
    """Memory budget of a find_continuum_ultra frequency task: 0.8 of a
    thread's memory less the task baseline, as for the moments application;
    an explicit budget is clamped to it."""
    memory_per_thread = float(thread_info["memory_per_thread"])
    budget = 0.8 * memory_per_thread - min(1.5, 0.2 * memory_per_thread)
    if memory_budget_gb is not None:
        budget = min(memory_budget_gb, budget)
    return int(budget * 2**30)


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
        :func:`~astroviper.processing_functions.imaging.find_continuum.linear_channel_ranges`).
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
        ``channel_range`` (``[start, stop]`` in channels of the cube),
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
        ``monitor_resources_seconds``, ``driver_resource_usage``),
        ``first_round`` (``source``, the application that computed
        findContinuum's first round, or ``"imager"``; ``store``, the
        first-round store or ``None``; ``moments``, ``"imager"`` or
        ``"computed"``; ``moments_complete`` and ``imager``, the cost of the
        imager's first round), ``memory_plan`` (``None``; see
        :func:`find_continuum_ultra`) and ``report`` (the path of the report
        written, else ``None``).

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
        backend="separate",
        channel_range=channel_range,
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
    channel_range: list[int] | None = None,
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
    first_round_reductions : dict, optional
        ``find_continuum_first_round`` of
        :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`
        (made with the same primary beam limit): the first-round store it
        made next to the cube while imaging it. The store's primary-beam
        statistics, cube statistics and, when they are complete for the
        spectral window, moments 0 and 8 answer findContinuum's first round,
        and the mean primary beam is read from five planes, so the pass that
        would compute them is not made. A store is used only on the cube and
        data group it was made from, unchanged since.
    channel_range : list of int, optional
        ``[start, stop)``, the channels of the cube that findContinuum sees as
        its cube: one spectral window, on which the frequencies are linear (a
        run of
        :func:`~astroviper.processing_functions.imaging.find_continuum.linear_channel_ranges`).
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
        As :func:`find_continuum`.

    Raises
    ------
    ValueError
        As :func:`find_continuum`, and if the first-round store is incomplete,
        was made from another cube, data group or primary beam limit, or the
        cube changed since.

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
        backend="fused",
        first_round_reductions=first_round_reductions,
        channel_range=channel_range,
        monitor_resources_seconds=monitor_resources_seconds,
        report=report,
    )


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def find_continuum_ultra(
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
    thread_info: dict | None = None,
    memory_budget_gb: float | None = None,
    channel_range: list[int] | None = None,
    monitor_resources_seconds: float | None = None,
    report: str | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """Find the continuum channels as :func:`find_continuum_fused` does, streamed.

    The passes of :func:`find_continuum_fused` are made by
    :class:`~astroviper.distributed_applications.imaging.find_continuum.ultra_reductions.UltraReductions`:
    frequency tasks of whole shards read Stokes I in batches of planes, only
    the channels and chunks a request can use, and write their moment maps
    to slots of a scratch store, which tile tasks merge in the order of
    GraphVIPER's reduce; the reduce carries only per-channel rows. The
    first-round reductions come from the imager's first-round store, or are
    made by the first pass; every pass is planned within the memory budget of
    its tasks. The results are those of :func:`find_continuum_fused` at the
    same ``n_mapping_parallelism``, but for the last bits of the whole-cube
    sums and of the per-channel ``sumsq`` and ``rms``, which decide nothing.

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
        ``{"frequency": n}``, the number of frequency chunks of every pass;
        about 4 x ``n_threads`` chunks of whole shards if not given.
    first_round_reductions : dict, optional
        ``find_continuum_first_round`` of
        :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`
        (made with the same primary beam limit): the first-round store it
        made next to the cube while imaging it. The store's primary-beam
        statistics, cube statistics and, when they are complete for the
        spectral window, moments 0 and 8 answer findContinuum's first round,
        and the mean primary beam is read from five planes, so the pass that
        would compute them is not made. A store is used only on the cube and
        data group it was made from, unchanged since.
    thread_info : dict, optional
        ``n_threads`` and ``memory_per_thread`` (GiB) of the cluster, as
        :func:`~astroviper.utils.data_partitioning.get_thread_info` returns
        them (the default, refused when the cluster reports no thread or no
        memory limit): the frequency chunks and the memory budget of a task
        follow from them.
    memory_budget_gb : float, optional
        Memory budget of a frequency task (GiB). The default, and the largest
        value used, is 0.8 of ``memory_per_thread`` less the task baseline
        (``min(1.5, 0.2 x memory_per_thread)``). A pass whose tasks would
        exceed it reads fewer planes at a time, leaves out predicted moment
        maps or is split into passes. A requested reduction, or the first
        round, that does not fit it at one plane per read runs so, alone, with
        a warning; it is refused only when it needs more than
        ``memory_per_thread``.
    channel_range : list of int, optional
        ``[start, stop)``, the channels of the cube that findContinuum sees as
        its cube: one spectral window, on which the frequencies are linear (a
        run of
        :func:`~astroviper.processing_functions.imaging.find_continuum.linear_channel_ranges`).
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
        As :func:`find_continuum`, with: ``passes`` (one row per pass over the
        image store: its reductions and data variables, the first round, the
        tasks, channels, planes, chunks and bytes read, the memory plan of
        its frequency tasks and of its merge tasks
        (``merge_task_memory_plan_bytes``), the anchors and the graph and
        compute times of the pass and of its merge, and the bytes of the
        scratch stores); ``timing_node_tasks`` (frequency
        and merge tasks, ``family``, with the reads, writes and phase times of
        each); ``timing_reduce_nodes`` (with ``reduction`` and ``family``);
        ``memory_plan`` (``n_threads``, ``memory_per_thread_gib``,
        ``budget_bytes``, ``n_frequency_tasks``, ``channels_per_task``,
        ``merge_tile_shape``, ``n_merge_tiles`` and ``plane_bytes``); the
        graph, generate and compute times of the merges in those of
        ``timing_distributed_application``.

    Raises
    ------
    ValueError
        As :func:`find_continuum_fused`; if the cluster reports no thread or no
        memory limit (without ``thread_info``); and if a requested reduction,
        or the first round, needs more than ``memory_per_thread`` at one plane
        per read (the first round, or a read of the cube with
        ``first_round_reductions``, is checked before anything is written to
        ``work_directory``).

    Notes
    -----
    As :func:`find_continuum`, a process runs one call at a time. The stores
    of the passes, the bit-packed two-dimensional masks and, without
    ``first_round_reductions``, the first-round store of the spectral window
    are kept in the subdirectory ``<cube>.find_continuum_ultra`` of
    ``work_directory``, named after the cube as findContinuum's products; the
    scratch store of a pass is removed after its merge, or when the pass
    fails.
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
        backend="ultra",
        first_round_reductions=first_round_reductions,
        thread_info=thread_info,
        memory_budget_gb=memory_budget_gb,
        channel_range=channel_range,
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
    backend,
    first_round_reductions=None,
    thread_info=None,
    memory_budget_gb=None,
    channel_range=None,
    monitor_resources_seconds=None,
    report=None,
):
    import dask
    import matplotlib.pyplot as plt
    import pandas as pd
    import xarray as xr
    import zarr
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
    from astroviper.distributed_applications.imaging.find_continuum.first_round import (
        load_find_continuum_first_round,
    )
    from astroviper.distributed_applications.imaging.find_continuum.fused_reductions import (
        FusedReductions,
    )
    from astroviper.distributed_applications.imaging.find_continuum.ultra_reductions import (
        UltraReductions,
        ultra_frequency_tasks,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_channel_range,
        check_find_continuum_cluster_threads,
        check_find_continuum_config,
        check_find_continuum_inputs,
    )
    from astroviper.processing_functions.imaging.find_continuum import (
        finalize_find_continuum_reductions,
        linear_frequency_axis,
    )
    from astroviper.utils.data_partitioning import get_thread_info
    from astroviper.utils.timing import format_timing_summary

    application_start = time.time()
    rss_peak_start = _rss_peak_bytes()
    application = {
        "separate": "find_continuum",
        "fused": "find_continuum_fused",
        "ultra": "find_continuum_ultra",
    }[backend]
    timing_distributed_application = {
        key: 0.0
        for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES
        for _, key in steps
    }
    if not input_checked:
        check_find_continuum_config(
            n_bin,
            primary_beam_limit,
            n_mapping_parallelism,
            first_round_reductions,
            thread_info,
            memory_budget_gb,
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
    first_round = None
    first_round_info = {
        "source": application,
        "store": None,
        "moments": "computed",
        "moments_complete": False,
        "imager": None,
    }
    load_counters = {}
    if first_round_reductions is not None:
        store = os.path.abspath(first_round_reductions["store"])
        first_round = load_find_continuum_first_round(
            dict(first_round_reductions, store=store),
            input_image_store,
            lazy_xds,
            primary_beam_limit,
            [window_start, window_stop],
            load_counters,
        )
        attributes = zarr.open_group(store, mode="r").attrs[
            "find_continuum_first_round"
        ]
        if attributes["variables"] != data_variables:
            raise ValueError(
                f"The first-round store {store} was made from {attributes['variables']}, "
                f"not from {data_variables} of data group {image_data_group_in_name!r}."
            )
        first_round_info.update(
            source="imager",
            store=store,
            moments="computed" if first_round[1] is None else "imager",
            moments_complete=first_round[1] is not None,
            imager=attributes.get("imager"),
        )
    timing_distributed_application["T_load_first_round"] = time.time() - step_start

    step_start = time.time()
    os.makedirs(work_directory, exist_ok=True)
    beam_variable = data_group["beam_fit_params_point_spread_function"]
    polarization_index = lazy_xds.polarization.values.tolist().index("I")
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
            "metadata": _cube_metadata(
                lazy_xds, data_variables["sky"], beam, telescope_name
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

    memory_plan = None
    if backend == "ultra":
        if thread_info is None:
            thread_info = get_thread_info()
            check_find_continuum_cluster_threads(thread_info)
        sky = zarr.open_array(
            os.path.join(input_image_store, data_variables["sky"]), mode="r"
        )
        ultra = UltraReductions(
            cubes,
            input_image_store,
            data_variables,
            polarization_index,
            work_directory,
            primary_beam_limit,
            first_round_info["store"],
            ultra_frequency_tasks(
                window_stop - window_start,
                (sky.shards or sky.chunks)[1],
                thread_info["n_threads"],
            )
            if n_mapping_parallelism is None
            else n_mapping_parallelism["frequency"],
            _ultra_budget_bytes(thread_info, memory_budget_gb),
            linear_frequency_axis(lazy_xds.frequency.values),
            cubes[stem + ".residual"]["metadata"]["rest_frequency"],
            monitor_resources_seconds,
            int(thread_info["memory_per_thread"] * 2**30),
        )
        # Before anything is written to the work directory, so that the call
        # can be repeated there with other settings.
        ultra.check_smallest_task()
        reductions = FusedReductions(
            ultra.run_pass, stem + ".residual", stem + ".pb", first_round
        )
    else:
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
        reductions = (
            run_reductions
            if backend == "separate"
            else FusedReductions(
                lambda requests, n_requested: run_reductions(requests),
                stem + ".residual",
                stem + ".pb",
                first_round,
            )
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
            casa_image_tasks.cube_session(cubes, reductions) as session,
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
    if backend == "ultra":
        timing, cube_reads, passes, reduce_records = (
            ultra.timing,
            ultra.cube_reads,
            ultra.passes,
            ultra.reduce_records,
        )
        for key, merge_key in (
            ("T_create_map_reduce_graph", "T_merge_create_graph"),
            ("T_generate_dask_graph", "T_merge_generate_dask_graph"),
            ("T_compute_dask_graph", "T_merge_compute_dask_graph"),
        ):
            timing_distributed_application[key] = sum(
                row[key] + row[merge_key] for row in passes
            )
        timing_distributed_application["T_primary_beam_mean"] = (
            ultra.T_primary_beam_mean
        )
        if passes:
            timing_distributed_application["compute_start_unixtime"] = min(
                row["start_unixtime"] for row in passes
            )
            timing_distributed_application["compute_end_unixtime"] = float(
                np.nanmax(
                    [
                        [row["compute_end_unixtime"], row["merge_end_unixtime"]]
                        for row in passes
                    ]
                )
            )
        first_round_info["store"] = ultra.first_round
        memory_plan = {
            "n_threads": thread_info["n_threads"],
            "memory_per_thread_gib": thread_info["memory_per_thread"],
            **ultra.layout,
        }
    timing_distributed_application["T_find_continuum_between_passes"] = (
        find_continuum_seconds
        - sum(
            timing_distributed_application[key]
            for key in (
                "T_create_map_reduce_graph",
                "T_generate_dask_graph",
                "T_compute_dask_graph",
                "T_primary_beam_mean",
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
        counters.get("bytes", 0) + counters.get("index_bytes", 0)
        for counters in (load_counters, session.counters)
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
            title=f"{application} timing (driver, seconds)",
            total_label="TOTAL (driver wall time)",
        )
    )
    if len(timing_node_tasks):
        logger.info(
            format_timing_summary(
                timing_node_tasks.sum(numeric_only=True).to_dict(),
                NODE_TASK_TIMING_PHASES_ULTRA
                if backend == "ultra"
                else NODE_TASK_TIMING_PHASES,
                total_key=NODE_TASK_TIMING_TOTAL_KEY,
                title=f"{application} node-task timing: TOTAL over all tasks (seconds)",
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
        "first_round": first_round_info,
        "memory_plan": memory_plan,
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
            application=application,
        )
    return return_dict
