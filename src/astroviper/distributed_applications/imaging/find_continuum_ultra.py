"""find_continuum_ultra: the continuum channels of one spectral window of a
dirty cube, as the ALMA pipeline's ``hif_findcont`` finds them with
findContinuum, in streamed passes over the image store.

:func:`find_continuum_ultra` checks its configuration, opens the spectral
window lazily and takes findContinuum's settings from its metadata
(:func:`~astroviper.processing_functions.imaging.find_continuum_ultra.settings.find_continuum_settings`).
It then alternates passes over the window's frequency chunks
(:mod:`~astroviper.distributed_applications.imaging.find_continuum_ultra_passes`)
with the processing functions of
:mod:`~astroviper.processing_functions.imaging.find_continuum_ultra` on
image-shaped maps and one-dimensional spectra: the first round, or the
imager's first-round store; the joint mask; findContinuum's rounds, each with
its spectrum, channel selection and selection maps; the steps after the
rounds; and the products. The values that cross the rounds are local
variables of :func:`_run_rounds`. The driver reads planes of the primary beam
and writes the joint mask through
:mod:`~astroviper.node_tasks.imaging.find_continuum_ultra`. ``fC`` line numbers
are those of the pipeline's ``findContinuum.py`` and ``heur`` those of its
``hif/heuristics/findcont.py`` (pipeline 2026.3.0.7).
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import toolviper.utils.logger as logger
import toolviper.utils.parameter
import xarray as xr

from astroviper.utils.param_docs import shares_param_docs

if TYPE_CHECKING:
    from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
        ChannelSelection,
        ComputedSpectrum,
        FirstRoundRows,
        FirstRoundStatistics,
    )

_PARAM_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# The layout of ``timing_distributed_application`` (format_timing_summary and
# the processing section of the report): the graph and compute times are
# summed over the passes and their merges; the other leaves are the driver's
# own steps.
DISTRIBUTED_APPLICATION_TIMING_PHASES = [
    (
        "DISTRIBUTED APPLICATION (driver)",
        None,
        [
            ("open and check the spectral window", "T_open_image"),
            ("settings", "T_settings"),
            ("first-round store and rows", "T_load_first_round"),
            ("layout and memory plan", "T_plan"),
            ("create map/reduce graphs (all passes)", "T_create_map_reduce_graph"),
            ("generate dask graphs (all passes)", "T_generate_dask_graph"),
            ("compute dask graphs (all passes)", "T_compute_dask_graph"),
            ("read primary-beam planes", "T_primary_beam_planes"),
            ("read the maps of the passes", "T_read_maps"),
            ("write the regions of the passes", "T_write_regions"),
            ("first-round maps and joint mask", "T_first_round_maps"),
            ("spectra", "T_spectrum"),
            ("channel selection", "T_channel_selection"),
            ("selection maps and statistics", "T_selection_maps"),
            ("decisions of the rounds", "T_round_decisions"),
            ("products", "T_products"),
            ("write the joint mask", "T_write_joint_mask"),
        ],
    ),
]
DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY = "T_total"
NODE_TASK_TIMING_PHASES = [
    (
        "FREQUENCY TASK (one chunk of one pass)",
        None,
        [
            ("wait for the reads", "T_read_wait"),
            ("masks", "T_masks"),
            ("moments", "T_moments"),
            ("spectra", "T_spectrum"),
            ("noise", "T_noise"),
            ("first round", "T_first_round"),
            ("write the moment partials", "T_write_partials"),
        ],
    ),
    (
        "MERGE TASK (one tile of the maps of one pass)",
        None,
        [
            ("read the partials", "T_read_partials"),
            ("merge", "T_merge"),
            ("write the maps", "T_write_maps"),
        ],
    ),
]
NODE_TASK_TIMING_TOTAL_KEY = "T_find_continuum_task"

_LEAVES = tuple(
    key for _, _, steps in DISTRIBUTED_APPLICATION_TIMING_PHASES for _, key in steps
)
# Threshold sigma of round extra_mask (fC 491, 1424-1425).
_EXTRA_MASK_THRESHOLD_SIGMA = 2.5
# Threshold of the change code of the low-spread fix (fC 3017, 3029).
_LOW_SPREAD_CHANGE_THRESHOLD = 0.05


class _RoundsOutcome(NamedTuple):
    """What findContinuum leaves of a spectral window, for the products and
    the diagnostics."""

    rows: FirstRoundRows
    first_round_store: str
    constant_cube: bool
    continuum_ranges: list | None = None
    ranges_before_blue_pruning: list | None = None
    aggregate_bandwidth: float | None = None  # Hz, the value set last
    moment_difference_snr: float | None = None
    joint_mask: xr.Dataset | None = None  # of the last computed round
    first_round_statistics: FirstRoundStatistics | None = None
    spectrum: ComputedSpectrum | None = None  # of the last computed round
    selection: ChannelSelection | None = None  # of the last computed round
    rounds: list | None = None  # one record per round
    change_code: str | None = None
    path_code: str | None = None
    reverted: bool = False
    low_spread_fix_applied: bool = False


@shares_param_docs
@toolviper.utils.parameter.validate(config_dir=_PARAM_CONFIG_DIR)
def find_continuum_ultra(
    input_image_store: str,
    atmospheric_transmission,
    atmospheric_sky_temperature,
    work_directory: str,
    image_data_group_in_name: str = "residual",
    channel_range: list[int] | None = None,
    telescope_name: str = "ALMA",
    single_continuum: bool = False,
    n_bin: int = 1,
    spectral_dynamic_range_bandwidth: float | None = None,
    primary_beam_limit: float = 0.2,
    first_round_store: str | None = None,
    n_mapping_parallelism: dict | None = None,
    thread_info: dict | None = None,
    memory_budget_gb: float | None = None,
    monitor_resources_seconds: float | None = None,
    report: str | None = None,
    *,
    input_checked: bool = False,
) -> dict:
    """Find the continuum channels of one spectral window of a dirty cube, as
    the ALMA pipeline's hif_findcont does, in streamed passes over the image
    store.

    The results equal those of
    :func:`~astroviper.distributed_applications.imaging.find_continuum.find_continuum.find_continuum`
    at the same ``n_mapping_parallelism``; at other frequency chunks, or with
    the imager's moments, the selections, flags and decisions are the same
    and the moment-difference signal-to-noise ratio differs in its last
    digits (the order of the moment-0 sums). The passes over the residual are
    3 on findContinuum's common path, 2 without rounds (time-division mode)
    and 5 with an amended or an extra joint mask, one fewer with the imager's
    moments.

    Parameters
    ----------
    input_image_store : str
        Path of the xradio image store (Zarr) of the dirty cube, its primary
        beam and the beam fits of its point spread function; Stokes I is
        used.
    atmospheric_transmission : list or numpy.ndarray
        Atmospheric transmission (0 to 1) at every channel of the spectral
        window, as an atmosphere model gives it for the observation.
    atmospheric_sky_temperature : list or numpy.ndarray
        Sky brightness temperature (K) at every channel of the spectral
        window.
    work_directory : str
        Directory in which the run makes its directory
        ``<cube>.find_continuum_ultra_<start>_<stop>``, ``<cube>`` the name of
        the image store without ``.img.zarr`` or ``.zarr`` (refused if it
        exists: the products of an earlier run), with the stores of the
        passes while they run, the first-round store when none is given and
        the joint-mask store; the report is written next to it.
    image_data_group_in_name : str, default "residual"
        Image data group with the roles ``sky``, ``primary_beam`` and
        ``beam_fit_params_point_spread_function``.
    channel_range : list of int, optional
        ``[start, stop)``, the channels of the cube in the spectral window: a
        run of linear frequencies
        (:func:`~astroviper.processing_functions.imaging.utils.spectral_windows.linear_channel_ranges`),
        increasing, of at least 8 channels; in a cube of several windows its
        middle channel is the rest frequency of moment 0
        (:func:`~astroviper.processing_functions.imaging.utils.spectral_windows.window_rest_frequency`).
        ``None`` (default): every channel, refused for a cube of several
        spectral windows.
    telescope_name : str, default "ALMA"
        ``"ALMA"`` or ``"VLA"``, whose time-division-mode rule sets the number
        of rounds; every other step takes ALMA's rule, as findContinuum does.
    single_continuum : bool, default False
        The spectral window is a single-continuum setup [findContinuum
        ``singleContinuum``].
    n_bin : int, default 1
        Channels to average in the mean spectrum, as the caller resolved them
        [findContinuum ``nbin``].
    spectral_dynamic_range_bandwidth : float, optional
        The requested spectral dynamic range bandwidth (Hz), which sets
        ``n_bin`` when given [findContinuum ``spectralDynamicRangeBandWidth``];
        ``None``: none.
    primary_beam_limit : float, default 0.2
        Primary-beam level in (0, 1) below which a voxel of the dirty cube and
        of the primary beam is invalid, as in the cube ``hif_findcont`` makes
        [tclean ``pblimit``].
    first_round_store : str, optional
        Path of the imager's first-round store of the cube
        (:func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`
        with ``find_continuum_first_round``), made with the same primary-beam
        limit and data variables. Its masks and per-channel rows, and its
        moments when they are complete for the spectral window, replace the
        first pass; a store that does not fit the window is refused
        (:func:`~astroviper.distributed_applications.imaging.find_continuum_first_round.validate_find_continuum_first_round`).
        ``None`` (default): the first pass makes the first round, in a store
        of the run's directory.
    n_mapping_parallelism : dict, optional
        ``{"frequency": n}``, the frequency chunks of every pass; ``None``
        (default): ``ceil(n_channels / k)`` chunks of the window, ``k`` the
        channels of a whole number of shards that gives about
        ``FREQUENCY_TASKS_PER_THREAD`` chunks per thread. GraphVIPER makes the
        chunks of equal length, so they are whole shards only when the window
        starts on a shard and ``k`` divides its channels.
    thread_info : dict, optional
        ``n_threads`` and ``memory_per_thread`` (GiB) of the cluster, which
        set the frequency chunks and the memory budget of a task. ``None``
        (default): those of
        :func:`~astroviper.utils.data_partitioning.get_thread_info`, refused
        when the cluster reports no thread or no memory limit.
    memory_budget_gb : float, optional
        Memory budget of a frequency task (GiB). The default, and the largest
        value used, is 0.8 of ``memory_per_thread`` less the task baseline
        (``min(1.5, 0.2 x memory_per_thread)``). A pass whose tasks would
        exceed it reads fewer planes at a time; a task over it at one plane
        per batch runs so with a warning, and a pass whose task needs more
        than ``memory_per_thread`` is refused.
    monitor_resources_seconds : float, optional
        If set, sample the CPU, memory and I/O of each node task's worker
        process every this many seconds (GraphVIPER's resource monitor) into
        ``timing_node_tasks``, and the driver's process while it works into
        ``driver_resource_usage``. Requires ``psutil``.
    report : str, optional
        Make a report of the run: ``"html"`` or ``"pdf"``, written next to the
        run's directory as ``<run directory>_report.html`` or ``.pdf``, or
        ``"notebook"``, displayed in the calling notebook. It has the
        processing section of every distributed application and the results:
        the ranges, flags, the spectrum of the last computed round with its
        selection, the rounds and the passes. A failure while making it is
        logged and leaves the results intact. ``None`` (default): no report.
    input_checked : bool, default False
        Skip the configuration checks already done by the calling layer. The
        checks of the opened window, of the first-round store and of the
        run's directory, and the runtime checks, always run.

    Returns
    -------
    dict
        Results: ``continuum_channel_ranges`` (``[start, stop)`` channel pairs
        of the spectral window in findContinuum's list order),
        ``channel_range`` (``[start, stop)`` of the window in the cube),
        ``continuum_frequency_ranges`` (per range the outer channel edges in
        Hz, the lower first), ``frequency_frame`` (of the cube's frequency
        coordinate), ``aggregate_bandwidth`` (Hz, the value findContinuum
        sets last), ``all_continuum``, ``low_bandwidth``, ``low_spread``,
        ``single_range_channel_fraction`` (for one range, else ``None``),
        ``moment_difference_snr`` (0.0 without rounds), ``constant_cube``
        (with no range, and ``None`` for the signal-to-noise ratio, the
        fraction and the joint mask), ``joint_mask`` (path of the Zarr store
        of ``JOINT_MASK`` and ``JOINT_MASK_FOOTPRINT`` of the last computed
        round), ``joint_mask_kind`` (``"phase_1"``, ``"phase_2"`` or
        ``"amended"``) and ``joint_mask_primary_beam_based``.

        ``diagnostics``: ``settings`` (the scalars of findContinuum's
        settings), ``first_round_statistics``, ``rounds``
        (``pandas.DataFrame``, one row per round: its kind, threshold sigma in
        and out and mode, signal-ratio tier, spectrum steps, selection,
        thresholds, the ranges of its maps, decisions, levels, statistics and
        the flags of its intersection), ``change_code``, ``path_code``,
        ``reverted``, ``low_spread_fix_applied``, and of the last computed
        round ``mean_spectrum`` and ``spectrum`` (the mean spectrum of its
        pass and the spectrum its selection read, ``xarray.DataArray`` over
        ``frequency``), ``positive_threshold``, ``negative_threshold`` and
        ``corrected_median``.

        Profiling, by the mechanisms of the imaging applications:
        ``n_passes``; ``cube_reads`` (data variable to the passes that read
        it); ``passes`` (``pandas.DataFrame``, one row per pass: its index,
        kind, round, purpose, data variables read, tasks, channels, planes,
        chunks and bytes read, planes per batch, memory plans, anchors, graph
        and compute times of the pass and of its merge, scratch bytes and the
        driver's work inside it); ``timing_node_tasks`` (one row per
        frequency and merge task of every pass, with the series of
        GraphVIPER's resource monitor); ``timing_reduce_nodes``;
        ``timing_distributed_application`` (the leaves of
        :data:`DISTRIBUTED_APPLICATION_TIMING_PHASES`, ``T_total``, the
        application and compute anchors, ``driver_rss_peak_bytes`` (the peak
        resident memory of the driver during the call: the process's peak
        when the call raised it, else the largest sample of
        ``driver_resource_usage``, else ``None``),
        ``driver_rss_lifetime_peak_bytes``, ``driver_bytes_read``,
        ``driver_bytes_written`` and, with ``monitor_resources_seconds``,
        ``driver_resource_usage``); ``first_round`` (``source``,
        ``"imager"`` or ``"find_continuum_ultra"``, ``store``, ``moments``,
        ``"imager"`` or ``"computed"``, and ``imager``, the cost attributes
        of the imager's store); ``memory_plan`` (``n_threads``,
        ``memory_per_thread_gib``, ``budget_bytes``, ``n_frequency_tasks``,
        ``channels_per_task``, ``merge_tile_shape``, ``n_merge_tiles``,
        ``plane_bytes``, ``driver_memory_estimate_bytes``) and ``report``
        (the path of the report written, else ``None``).

    Raises
    ------
    ValueError
        If the configuration, the opened window or the first-round store
        fails the checks of
        :mod:`~astroviper.processing_functions.imaging.config_check.param_check_find_continuum`,
        the run's directory exists, a pass needs more than the memory of a
        thread, the window has no valid voxel, or findContinuum fails on the
        window, with the reason (an empty selection, an empty region outside
        the joint mask or noise annulus, line-full ranges findContinuum
        cannot invert).
    """
    from graphviper.graph_tools.map import monitor_node_task

    from astroviper.distributed_applications.imaging.find_continuum_ultra_passes import (
        plan_find_continuum_ultra_layout,
    )
    from astroviper.node_tasks.imaging.find_continuum_ultra import (
        find_continuum_ultra_task_bytes,
        write_joint_mask_store,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_cluster_threads,
        check_find_continuum_ultra_task_memory,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.settings import (
        find_continuum_settings,
    )
    from astroviper.utils.data_partitioning import get_thread_info
    from astroviper.utils.timing import format_timing_summary

    application_start = time.time()
    rss_peak_start = _rss_peak_bytes()
    if not input_checked:
        from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
            check_find_continuum_ultra_config,
        )

        check_find_continuum_ultra_config(
            n_bin,
            spectral_dynamic_range_bandwidth,
            telescope_name,
            single_continuum,
            primary_beam_limit,
            first_round_store,
            n_mapping_parallelism,
            thread_info,
            memory_budget_gb,
            monitor_resources_seconds,
        )
    # Node tasks may run in other processes and working directories.
    input_image_store = os.path.abspath(input_image_store)
    work_directory = os.path.abspath(work_directory)
    (
        window_xds,
        data_group,
        window_range,
        image_shape,
        map_coords,
        rest_frequency,
    ) = _open_window(
        input_image_store,
        image_data_group_in_name,
        channel_range,
        atmospheric_transmission,
        atmospheric_sky_temperature,
    )
    measures = {"T_open_image": time.time() - application_start}

    start = time.perf_counter()
    settings = find_continuum_settings(
        window_xds.frequency.values,
        window_xds.l.values,
        window_xds.m.values,
        rest_frequency,
        window_xds[data_group["beam_fit_params_point_spread_function"]],
        np.asarray(atmospheric_transmission, dtype=np.float64),
        np.asarray(atmospheric_sky_temperature, dtype=np.float64),
        telescope_name,
        single_continuum,
        n_bin,
        spectral_dynamic_range_bandwidth,
    )
    measures["T_settings"] = time.perf_counter() - start

    start = time.perf_counter()
    store_inputs, imager_costs = None, None
    if first_round_store is not None:
        store_inputs, imager_costs = _first_round_store_inputs(
            os.path.abspath(first_round_store),
            input_image_store,
            primary_beam_limit,
            window_range,
            image_data_group_in_name,
        )
    measures["T_load_first_round"] = time.perf_counter() - start

    start = time.perf_counter()
    if thread_info is None:
        thread_info = get_thread_info()
        check_find_continuum_cluster_threads(thread_info)
    layout = plan_find_continuum_ultra_layout(
        window_xds,
        data_group,
        window_range,
        n_mapping_parallelism,
        thread_info,
        memory_budget_gb,
    )
    if store_inputs is None or not store_inputs.moments_usable:
        # The first pass, refused before anything is written.
        check_find_continuum_ultra_task_memory(
            find_continuum_ultra_task_bytes(
                "first_round",
                settings.n_l,
                settings.n_m,
                1,
                np.dtype(layout.value_dtypes["sky"]).itemsize,
                np.dtype(layout.value_dtypes["primary_beam"]).itemsize,
                store_inputs is None,
                0,
                True,
                0.0,
                layout.read_unit_bytes,
            ),
            layout.memory_per_thread_bytes,
        )
    measures["T_plan"] = time.perf_counter() - start
    run_directory = _new_run_directory(work_directory, input_image_store, window_range)
    frequency_frame = window_xds.frequency.attrs.get("frame")

    def analyse(input_params):
        outcome, records, analysis = _run_rounds(
            window_xds,
            layout,
            settings,
            store_inputs,
            image_shape,
            primary_beam_limit,
            map_coords,
            run_directory,
            monitor_resources_seconds,
        )
        if outcome.constant_cube:
            results = _constant_cube_result(
                outcome.rows, window_range, frequency_frame, input_image_store
            )
        else:
            start = time.perf_counter()
            results = _products(outcome, settings, window_range, frequency_frame)
            analysis = _sum_measures(
                analysis, {"T_products": time.perf_counter() - start}
            )
            start = time.perf_counter()
            results["joint_mask"] = os.path.join(run_directory, "joint_mask.zarr")
            written = write_joint_mask_store(results["joint_mask"], outcome.joint_mask)
            analysis = _sum_measures(
                analysis,
                {
                    "T_write_joint_mask": time.perf_counter() - start,
                    "driver_bytes_written": written["bytes"],
                },
            )
        return {
            "results": results,
            "outcome": outcome,
            "records": records,
            "measures": analysis,
        }

    if monitor_resources_seconds:
        analyse = monitor_node_task(analyse, monitor_resources_seconds)
    output = analyse({})
    return_dict = _return_dict(
        output["results"],
        output["outcome"],
        settings,
        window_xds.frequency.values,
        data_group,
        output["records"],
        _sum_measures(measures, output["measures"]),
        store_inputs,
        imager_costs,
        layout,
        thread_info,
        application_start,
        rss_peak_start,
        output.get("resource_usage"),
        monitor_resources_seconds,
    )
    logger.info(
        format_timing_summary(
            return_dict["timing_distributed_application"],
            DISTRIBUTED_APPLICATION_TIMING_PHASES,
            total_key=DISTRIBUTED_APPLICATION_TIMING_TOTAL_KEY,
            title="find_continuum_ultra timing (driver, seconds)",
            total_label="TOTAL (driver wall time)",
        )
    )
    if len(return_dict["timing_node_tasks"]):
        logger.info(
            format_timing_summary(
                return_dict["timing_node_tasks"].sum(numeric_only=True).to_dict(),
                NODE_TASK_TIMING_PHASES,
                total_key=NODE_TASK_TIMING_TOTAL_KEY,
                title="find_continuum_ultra node-task timing: TOTAL over all tasks "
                "(seconds)",
            )
        )
    if report is not None:
        from astroviper.distributed_applications.imaging.find_continuum_ultra_report import (
            find_continuum_ultra_report,
        )
        from astroviper.utils.report import make_report

        return_dict["report"] = make_report(
            find_continuum_ultra_report,
            return_dict,
            report_store=run_directory,
            report=report,
        )
    return return_dict


def _open_window(
    input_image_store,
    image_data_group_in_name,
    channel_range,
    atmospheric_transmission,
    atmospheric_sky_temperature,
):
    """The spectral window of the image store opened lazily and checked
    (metadata only), its data group, its ``[start, stop)`` in channels of the
    cube, the shape of the cube, the coordinates of the window's maps (one
    time, the mean frequency and Stokes I) and the window's rest frequency."""
    from astroviper.processing_functions.image_analysis.moments import (
        collapsed_moment_axis_coords,
    )
    from astroviper.processing_functions.imaging.config_check.param_check_find_continuum import (
        check_find_continuum_ultra_inputs,
    )
    from astroviper.processing_functions.imaging.utils.spectral_windows import (
        window_rest_frequency,
    )

    cube_xds = xr.open_zarr(input_image_store, chunks=None)
    window_range = check_find_continuum_ultra_inputs(
        cube_xds,
        image_data_group_in_name,
        channel_range,
        atmospheric_transmission,
        atmospheric_sky_temperature,
    )
    data_group = cube_xds.attrs["data_groups"][image_data_group_in_name]
    window_xds = cube_xds.isel(frequency=slice(*window_range))
    stokes_i = window_xds.polarization.values.tolist().index("I")
    map_coords = collapsed_moment_axis_coords(
        window_xds[[data_group["sky"]]].isel(polarization=[stokes_i]), "frequency"
    )
    return (
        window_xds,
        data_group,
        window_range,
        cube_xds[data_group["sky"]].shape,
        map_coords,
        window_rest_frequency(
            cube_xds.frequency.values,
            window_range,
            float(cube_xds.frequency.attrs["rest_frequency"]["data"]),
        ),
    )


def _first_round_store_inputs(
    first_round_store,
    input_image_store,
    primary_beam_limit,
    window_range,
    image_data_group_in_name,
):
    """The first-round store validated for the spectral window and the data
    group, and the imager's cost attributes in it."""
    import zarr

    from astroviper.distributed_applications.imaging.find_continuum_first_round import (
        validate_find_continuum_first_round,
    )

    store_inputs = validate_find_continuum_first_round(
        first_round_store,
        input_image_store,
        primary_beam_limit,
        window_range,
        image_data_group_in_name,
    )
    attributes = zarr.open_group(first_round_store, mode="r").attrs[
        "find_continuum_first_round"
    ]
    return store_inputs, attributes.get("imager")


def _new_run_directory(work_directory, input_image_store, window_range):
    """The new directory of the run in the work directory, named after the
    cube and the spectral window; refused if it exists."""
    start, stop = window_range
    stem = os.path.basename(input_image_store)
    for suffix in (".img.zarr", ".zarr"):
        if stem.endswith(suffix):
            stem = stem.removesuffix(suffix)
            break
    run_directory = os.path.join(
        work_directory, f"{stem}.find_continuum_ultra_{start}_{stop}"
    )
    os.makedirs(work_directory, exist_ok=True)
    try:
        os.mkdir(run_directory)
    except FileExistsError:
        raise ValueError(
            f"{run_directory} holds the products of an earlier run: remove it or "
            "choose another work_directory."
        ) from None
    return run_directory


def _first_round(
    window_xds,
    layout,
    settings,
    store_inputs,
    image_shape,
    primary_beam_limit,
    run_directory,
    monitor_resources_seconds,
):
    """findContinuum's first round of the spectral window (fC 7609-7876): read
    from the imager's store when its moments are usable, else a first-round
    pass, which without a store makes one in the run's directory. Returns the
    reductions, the records of the pass and the driver's measures."""
    from astroviper.distributed_applications.imaging.find_continuum_first_round import (
        create_find_continuum_first_round_store,
    )
    from astroviper.distributed_applications.imaging.find_continuum_ultra_passes import (
        FirstRoundReductions,
        run_first_round_pass,
    )
    from astroviper.node_tasks.imaging.find_continuum_first_round import (
        read_first_round_moment_maps,
        read_first_round_rows,
    )

    start = time.perf_counter()
    if store_inputs is not None and store_inputs.moments_usable:
        rows, counters = read_first_round_rows(store_inputs.store, layout.channel_range)
        integral, maximum, map_counters = read_first_round_moment_maps(
            store_inputs.store, store_inputs.window_index
        )
        return (
            FirstRoundReductions(rows, integral, maximum, store_inputs.store),
            [],
            {
                "T_load_first_round": time.perf_counter() - start,
                "driver_bytes_read": _bytes_read(counters) + _bytes_read(map_counters),
            },
        )
    if store_inputs is None:
        store = os.path.join(run_directory, "first_round.zarr")
        create_find_continuum_first_round_store(
            store,
            image_shape[1],
            image_shape,
            layout.tile_shape,
            layout.merge_tile_shape,
            primary_beam_limit,
            layout.variables,
            layout.polarization_index,
            layout.value_dtypes,
            None,
            "find_continuum_ultra",
            [
                {
                    "channel_range": layout.channel_range,
                    "first_frequency": settings.first_frequency,
                    "channel_increment": settings.channel_increment,
                    "n_channels": settings.n_channels,
                }
            ],
            float(window_xds.frequency.attrs["rest_frequency"]["data"]),
        )
    else:
        store = store_inputs.store
    measures = {"T_load_first_round": time.perf_counter() - start}
    reductions, records = run_first_round_pass(
        window_xds,
        layout,
        settings,
        store,
        store_inputs is None,
        True,
        _pass_directory(run_directory, 0),
        0,
        monitor_resources_seconds,
        0,
    )
    return reductions, records, measures


def _primary_beam_products(layout, settings, rows, primary_beam_limit, map_coords):
    """The mean primary beam [findContinuum ``pbmom``] of the planes of
    ``primary_beam_mean_channels``, read one at a time (fC 7790-7798); the
    annulus levels of the primary-beam cube, from the edge lines of its
    middle plane (fC 8370-8411), and of the mean primary beam (fC 3721); and
    the driver's measures."""
    from astroviper.node_tasks.imaging.find_continuum_ultra import (
        read_primary_beam_edge_lines,
        read_primary_beam_plane,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.first_round_maps import (
        add_primary_beam_planes,
        annulus_levels,
        map_edge_samples,
        plane_edge_samples,
        primary_beam_mean,
    )

    start = time.perf_counter()
    window_start = layout.channel_range[0]
    variable = layout.variables["primary_beam"]
    map_shape = (1, 1, 1, settings.n_l, settings.n_m)
    primary_beam_sum = np.zeros(map_shape)
    # At most 15 planes (fC 7793-7794).
    valid_count = np.zeros(map_shape, dtype=np.uint8)
    bytes_read = 0
    for channel in settings.primary_beam_mean_channels:
        plane, limit_mask, counters = read_primary_beam_plane(
            layout.input_image_store,
            variable,
            layout.polarization_index,
            window_start + channel,
            primary_beam_limit,
        )
        add_primary_beam_planes(primary_beam_sum, valid_count, plane, limit_mask)
        bytes_read += _bytes_read(counters)
    edge_lines, edge_lines_valid, counters = read_primary_beam_edge_lines(
        layout.input_image_store,
        variable,
        layout.polarization_index,
        window_start + settings.n_channels // 2,
        primary_beam_limit,
    )
    bytes_read += _bytes_read(counters)
    read_seconds = time.perf_counter() - start
    start = time.perf_counter()
    primary_beam_mean_map = primary_beam_mean(primary_beam_sum, valid_count, map_coords)
    cube_levels = annulus_levels(
        np.nanmin(rows.primary_beam_minimum),
        plane_edge_samples(edge_lines, edge_lines_valid),
    )
    mean_levels = annulus_levels(
        np.nanmin(primary_beam_mean_map.values),
        map_edge_samples(primary_beam_mean_map),
    )
    return (
        primary_beam_mean_map,
        cube_levels,
        mean_levels,
        {
            "T_primary_beam_planes": read_seconds,
            "T_first_round_maps": time.perf_counter() - start,
            "driver_bytes_read": bytes_read,
        },
    )


def _round_zero(
    integral,
    maximum,
    primary_beam_mean_map,
    n_valid_voxels,
    settings,
    cube_levels,
    layout,
    primary_beam_limit,
    map_coords,
):
    """The joint mask of the first round and its statistics (fC 7889-8186),
    the primary-beam-based joint mask instead when it holds too few pixels
    (fC 8187-8204), plane 1 of the primary beam when read for it, and the
    driver's measures."""
    from astroviper.processing_functions.imaging.find_continuum_ultra.first_round_maps import (
        first_round_joint_mask,
        primary_beam_based_joint_mask,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
        BeamGeometry,
    )

    start = time.perf_counter()
    joint_mask, statistics = first_round_joint_mask(
        _as_map(integral, "SKY_MOMENT_INTEGRATED", map_coords),
        _as_map(maximum, "SKY_MOMENT_MAXIMUM", map_coords),
        primary_beam_mean_map,
        n_valid_voxels,
        BeamGeometry(
            settings.beam_major_arcsec,
            settings.beam_minor_arcsec,
            settings.cell_l_arcsec,
            settings.cell_m_arcsec,
            settings.max_baseline,
        ),
    )
    measures = {"T_first_round_maps": time.perf_counter() - start}
    plane_1 = None
    if statistics.needs_primary_beam_based_mask:
        plane_1, plane_measures = _primary_beam_plane_1(
            layout, primary_beam_limit, map_coords
        )
        start = time.perf_counter()
        joint_mask = primary_beam_based_joint_mask(
            plane_1,
            joint_mask["JOINT_MASK_FOOTPRINT"],
            cube_levels.higher,
            joint_mask.attrs["kind"],
        )
        measures = _sum_measures(
            measures,
            plane_measures,
            {"T_first_round_maps": time.perf_counter() - start},
        )
    return joint_mask, statistics, plane_1, measures


def _primary_beam_plane_1(layout, primary_beam_limit, map_coords):
    """The primary beam of the window's channel 1, NaN where invalid (fC 8197,
    8264), and the driver's measures."""
    from astroviper.node_tasks.imaging.find_continuum_ultra import (
        read_primary_beam_plane,
    )

    start = time.perf_counter()
    plane, limit_mask, counters = read_primary_beam_plane(
        layout.input_image_store,
        layout.variables["primary_beam"],
        layout.polarization_index,
        layout.channel_range[0] + 1,
        primary_beam_limit,
    )
    plane_1 = _as_map(np.where(limit_mask, plane, np.nan), "PRIMARY_BEAM", map_coords)
    return plane_1, {
        "T_primary_beam_planes": time.perf_counter() - start,
        "driver_bytes_read": _bytes_read(counters),
    }


def _computed_round(
    window_xds,
    layout,
    settings,
    first_round_store,
    joint_mask,
    primary_beam_footprint,
    cube_levels,
    n_bin,
    plane_1,
    primary_beam_limit,
    map_coords,
    run_directory,
    round_index,
    round_kind,
    monitor_resources_seconds,
    pass_index,
):
    """The spectrum of a computed round (fC 8255-8300): its regions, a
    spectrum pass and the spectrum steps. A spectrum of scaled MAD 0 replaces
    the joint mask by the primary-beam-based one, in the footprint of the mean
    primary beam, with a second spectrum pass (fC 8258-8277). Returns the
    spectrum, the round's joint mask, plane 1 of the primary beam (read on
    demand), the records of the passes and the driver's measures."""
    from astroviper.distributed_applications.imaging.find_continuum_ultra_passes import (
        run_spectrum_pass,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.first_round_maps import (
        primary_beam_based_joint_mask,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.spectrum import (
        computed_round_spectrum,
        spectrum_regions,
    )

    start = time.perf_counter()
    regions = spectrum_regions(joint_mask, cube_levels, settings.bad_atmosphere)
    measures = {"T_spectrum": time.perf_counter() - start}
    reductions, records = run_spectrum_pass(
        window_xds,
        layout,
        first_round_store,
        regions,
        _pass_directory(run_directory, pass_index),
        round_index,
        round_kind,
        monitor_resources_seconds,
        pass_index,
    )
    start = time.perf_counter()
    spectrum = computed_round_spectrum(
        reductions.mean_spectrum,
        reductions.scaled_mad_spectrum,
        settings.bad_atmosphere,
        n_bin,
        settings.max_baseline,
    )
    measures = _sum_measures(measures, {"T_spectrum": time.perf_counter() - start})
    if not spectrum.needs_primary_beam_based_mask:
        return spectrum, joint_mask, plane_1, records, measures
    if plane_1 is None:
        plane_1, plane_measures = _primary_beam_plane_1(
            layout, primary_beam_limit, map_coords
        )
        measures = _sum_measures(measures, plane_measures)
    start = time.perf_counter()
    joint_mask = primary_beam_based_joint_mask(
        plane_1, primary_beam_footprint, cube_levels.higher, joint_mask.attrs["kind"]
    )
    first_round_maps_seconds = time.perf_counter() - start
    start = time.perf_counter()
    regions = spectrum_regions(joint_mask, cube_levels, settings.bad_atmosphere)
    spectrum_seconds = time.perf_counter() - start
    reductions, fallback_records = run_spectrum_pass(
        window_xds,
        layout,
        first_round_store,
        regions,
        _pass_directory(run_directory, pass_index + 1),
        round_index,
        round_kind,
        monitor_resources_seconds,
        pass_index + 1,
    )
    start = time.perf_counter()
    spectrum = computed_round_spectrum(
        reductions.mean_spectrum,
        reductions.scaled_mad_spectrum,
        settings.bad_atmosphere,
        n_bin,
        settings.max_baseline,
        after_mad_zero_fallback=True,
    )
    measures = _sum_measures(
        measures,
        {
            "T_first_round_maps": first_round_maps_seconds,
            "T_spectrum": spectrum_seconds + time.perf_counter() - start,
        },
    )
    return spectrum, joint_mask, plane_1, records + fallback_records, measures


def _round_maps(
    window_xds,
    layout,
    settings,
    first_round_store,
    ranges,
    noise_region,
    moment_difference_region,
    map_coords,
    run_directory,
    round_index,
    round_kind,
    monitor_resources_seconds,
    pass_index,
):
    """The maximum and moment-difference maps of a channel selection
    (fC 1729-1776) from a selection-moments pass, which with a noise region
    also gives the noise of the selected channels (fC 3732). Returns the maps,
    the reductions of the pass without their maps, its records and the
    driver's measures."""
    from astroviper.distributed_applications.imaging.find_continuum_ultra_passes import (
        run_selection_moments_pass,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
        channels_of_ranges,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.selection_maps import (
        selection_maps,
    )

    reductions, records = run_selection_moments_pass(
        window_xds,
        layout,
        settings,
        first_round_store,
        ranges,
        noise_region,
        _pass_directory(run_directory, pass_index),
        round_index,
        round_kind,
        monitor_resources_seconds,
        pass_index,
    )
    start = time.perf_counter()
    maps = selection_maps(
        reductions.integral,
        reductions.maximum,
        len(channels_of_ranges(ranges)),
        settings.channel_width,
        settings.mean_frequency,
        moment_difference_region,
        map_coords,
    )
    return (
        maps,
        reductions._replace(integral=None, maximum=None),
        records,
        {"T_selection_maps": time.perf_counter() - start},
    )


def _extra_mask_spectrum(
    window_xds,
    layout,
    settings,
    first_round_store,
    request,
    regions,
    cube_levels,
    joint_mask,
    run_directory,
    round_index,
    round_kind,
    monitor_resources_seconds,
    pass_index,
):
    """The spectrum of the rounds after an extra mask (fC 2040-2055,
    2439-2458, 2510-2530): its regions, of which the forms after round
    amended_mask read the deciding round's ``joint_mask``, a spectrum pass and
    the spectrum steps. Returns it, the records of the pass and the driver's
    measures."""
    from astroviper.distributed_applications.imaging.find_continuum_ultra_passes import (
        run_spectrum_pass,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.spectrum import (
        extra_mask_regions,
        extra_mask_spectrum,
    )

    start = time.perf_counter()
    spectrum_regions = extra_mask_regions(
        request.form,
        request.moment_difference,
        request.level,
        regions["MOMENT_DIFFERENCE_REGION"],
        regions["ONLY_EXTRA_MASK_NORMALIZATION_REGION"],
        cube_levels,
        settings.bad_atmosphere,
        joint_mask,
    )
    seconds = time.perf_counter() - start
    reductions, records = run_spectrum_pass(
        window_xds,
        layout,
        first_round_store,
        spectrum_regions,
        _pass_directory(run_directory, pass_index),
        round_index,
        round_kind,
        monitor_resources_seconds,
        pass_index,
    )
    start = time.perf_counter()
    spectrum = extra_mask_spectrum(
        reductions.mean_spectrum,
        reductions.scaled_mad_spectrum,
        settings.bad_atmosphere,
        request.continuum_ranges,
        settings.first_frequency,
        settings.last_frequency,
    )
    return spectrum, records, {"T_spectrum": seconds + time.perf_counter() - start}


def _run_rounds(
    window_xds,
    layout,
    settings,
    store_inputs,
    image_shape,
    primary_beam_limit,
    map_coords,
    run_directory,
    monitor_resources_seconds,
):
    """findContinuum on the spectral window from its first round on: the
    first round, the constant-cube stop (heur 89-94), the joint mask, the
    rounds (fC 1380-2816) and the steps after them (fC 2834-3066).

    The values that cross the rounds are the local variables of the loop
    (spec section 2.2), which only the returned values of the processing
    functions change. The first-round maps are freed after the joint mask,
    the mean primary beam after the regions of round 0.

    Returns
    -------
    outcome : _RoundsOutcome
    records : list of PassRecord
        Of every pass, in order.
    measures : dict
        The driver's leaves of ``timing_distributed_application`` and the
        bytes it read and wrote outside the passes.
    """
    from astroviper.processing_functions.imaging.find_continuum_ultra.channel_selection import (
        select_continuum_channels,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.first_round_maps import (
        amended_round_statistics,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.products import (
        constant_cube,
        selection_bandwidth,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.rounds import (
        COMPUTED_ROUND_KINDS,
        SIGNAL_RATIO_TIER_1,
        after_amended_intersection,
        after_amended_round,
        after_extra_round,
        after_low_spread_fix,
        after_original_round,
        after_rounds,
        change_code,
        low_spread_ranges,
        path_code,
        round_maps_ranges,
    )
    from astroviper.processing_functions.imaging.find_continuum_ultra.selection_maps import (
        difference_map_snr,
        lowered_moment_difference_level,
        moment_difference_counts,
        negative_pixel_fraction,
        original_round_cube_statistics,
        round_zero_regions,
        selection_statistics,
    )

    first_round, records, measures = _first_round(
        window_xds,
        layout,
        settings,
        store_inputs,
        image_shape,
        primary_beam_limit,
        run_directory,
        monitor_resources_seconds,
    )
    rows, first_round_store = first_round.rows, first_round.first_round_store
    n_valid_voxels = int(rows.sky_valid_count.sum())
    if n_valid_voxels == 0:
        raise ValueError(
            f"Channels {layout.channel_range} of {layout.input_image_store} have no "
            "valid voxel: the residual is NaN or below the primary-beam limit "
            f"{primary_beam_limit} everywhere."
        )
    if constant_cube(np.nanmin(rows.sky_minimum), np.nanmax(rows.sky_maximum)):
        return (
            _RoundsOutcome(rows, first_round_store, True, rounds=[]),
            records,
            measures,
        )
    primary_beam_mean_map, cube_levels, mean_levels, step = _primary_beam_products(
        layout, settings, rows, primary_beam_limit, map_coords
    )
    measures = _sum_measures(measures, step)
    joint_mask, statistics_0, plane_1, step = _round_zero(
        first_round.integral,
        first_round.maximum,
        primary_beam_mean_map,
        n_valid_voxels,
        settings,
        cube_levels,
        layout,
        primary_beam_limit,
        map_coords,
    )
    measures = _sum_measures(measures, step)
    # Only scalars of the first-round maps are read later (fC 8205-8246).
    del first_round
    # The valid mean primary beam above 0 (fC 8203, 8267-8271).
    primary_beam_footprint = joint_mask["JOINT_MASK_FOOTPRINT"]

    round_kind = "original" if settings.rounds_enabled else "single"  # fC 1327-1331
    threshold_sigma, threshold_sigma_mode = "auto", "auto"  # fC 450-451
    signal_ratio_tier_1 = SIGNAL_RATIO_TIER_1  # fC 476
    n_bin, nonzero_mad_positions = settings.n_bin, None  # fC 1336
    moment_difference_snr = -1.0  # fC 1161
    moment_difference_level = settings.moment_difference_level  # fC 1364-1370
    round_joint_mask = joint_mask  # of the next computed round
    extra_spectrum = None  # of the next round after an extra mask
    selection_reference = change_code_references = round_change_code = None  # fC 1379
    decisions, round_records = [], []
    for round_index in range(settings.n_rounds_after_original + 1):  # fC 1380
        if round_kind == "extra_mask":
            threshold_sigma = _EXTRA_MASK_THRESHOLD_SIGMA  # fC 1424-1425
        record = {
            "round_index": round_index,
            "round_kind": round_kind,
            "threshold_sigma": threshold_sigma,
            "threshold_sigma_mode": threshold_sigma_mode,
            "signal_ratio_tier_1": signal_ratio_tier_1,
        }
        computed = round_kind in COMPUTED_ROUND_KINDS
        if computed:
            spectrum, round_joint_mask, plane_1, new_records, step = _computed_round(
                window_xds,
                layout,
                settings,
                first_round_store,
                round_joint_mask,
                primary_beam_footprint,
                cube_levels,
                n_bin,
                plane_1,
                primary_beam_limit,
                map_coords,
                run_directory,
                round_index,
                round_kind,
                monitor_resources_seconds,
                len(records),
            )
            records, measures = records + new_records, _sum_measures(measures, step)
            values = spectrum.spectrum
            first_frequency, last_frequency = (
                settings.first_frequency,
                settings.last_frequency,
            )
            start = time.perf_counter()
            peaks = (
                amended_round_statistics(statistics_0)
                if round_kind == "amended_mask"
                else statistics_0
            )
            measures["T_first_round_maps"] += time.perf_counter() - start
            n_pixels_maximum_mask = peaks.n_pixels_maximum_mask
            n_bin = spectrum.n_bin  # fC 1487, 1470
            record.update(
                n_bin=spectrum.n_bin,
                normalized=spectrum.normalized,
                initial_quadratic_removed=spectrum.initial_quadratic_removed,
                quadratic_ratio=spectrum.quadratic_ratio,
                initial_peak_over_mad=spectrum.initial_peak_over_mad,
                joint_mask_kind=round_joint_mask.attrs["kind"],
                joint_mask_primary_beam_based=round_joint_mask.attrs[
                    "primary_beam_based"
                ],
            )
        else:
            values = extra_spectrum.spectrum
            first_frequency = extra_spectrum.first_frequency
            last_frequency = extra_spectrum.last_frequency
            # The first round's peaks are never read here (fC 5118, 5476).
            n_pixels_maximum_mask = 0
        start = time.perf_counter()
        selection = select_continuum_channels(
            values,
            settings.n_baseline_channels,
            settings.channel_width,
            first_frequency,
            last_frequency,
            threshold_sigma,
            threshold_sigma_mode,
            signal_ratio_tier_1,
            round_kind,
            moment_difference_snr,
            n_pixels_maximum_mask,
            peaks.integrated_peak,
            peaks.maximum_peak,
            settings.single_continuum,
            nonzero_mad_positions,
            settings.rounds_enabled,
        )
        ranges = selection.continuum_ranges
        ranges_before_blue_pruning = selection.ranges_before_blue_pruning
        aggregate_bandwidth = selection_bandwidth(ranges, settings.channel_width)
        measures["T_channel_selection"] += time.perf_counter() - start
        if computed:
            nonzero_mad_positions = selection.nonzero_mad_positions  # fC 1481-1487
            computed_spectrum, computed_selection = spectrum, selection
        if round_index == 0:
            original_ranges = ranges  # fC 1694
        record.update(
            final_threshold_sigma=selection.final_threshold_sigma,
            continuum_ranges=ranges,
            ranges_before_blue_pruning=ranges_before_blue_pruning,
            positive_threshold=selection.positive_threshold,
            negative_threshold=selection.negative_threshold,
            corrected_median=selection.corrected_median,
        )
        if not settings.rounds_enabled:  # fC 1859: no maps without rounds
            round_records.append(record)
            continue
        if round_index == 0:
            start = time.perf_counter()
            regions = round_zero_regions(
                round_joint_mask, primary_beam_mean_map, cube_levels, mean_levels
            )  # fC 1709-1723, after a fallback of round original
            measures["T_selection_maps"] += time.perf_counter() - start
            joint_mask_0 = round_joint_mask  # fC 1715, 2096
            primary_beam_mean_map = None  # later uses are its regions

        start = time.perf_counter()
        maps_ranges = round_maps_ranges(
            round_kind, ranges, selection_reference, settings.n_channels
        )
        measures["T_round_decisions"] += time.perf_counter() - start
        maps, noise, new_records, step = _round_maps(
            window_xds,
            layout,
            settings,
            first_round_store,
            maps_ranges.ranges,
            regions["CUBE_NOISE_ANNULUS"] if round_kind == "original" else None,
            regions["MOMENT_DIFFERENCE_REGION"],
            map_coords,
            run_directory,
            round_index,
            round_kind,
            monitor_resources_seconds,
            len(records),
        )
        records, measures = records + new_records, _sum_measures(measures, step)
        record["maps_ranges"] = maps_ranges.ranges
        start = time.perf_counter()
        statistics = selection_statistics(maps, regions)
        moment_difference_snr = statistics.moment_difference_snr
        if round_kind == "original":
            cube_noise = original_round_cube_statistics(
                noise.noise_medians,
                noise.noise_mads,
                noise.selected_channels,
                n_valid_voxels,
                statistics.moment_difference_peak_outside,
                statistics.moment_difference_median,
            )
            moment_difference_level = lowered_moment_difference_level(
                moment_difference_level,
                statistics.moment_difference_snr,
                statistics.maximum_snr_outside,
            )  # fC 1915-1920
            counts = moment_difference_counts(
                maps["MOMENT_DIFFERENCE"],
                statistics.moment_difference_median,
                statistics.moment_difference_scaled_mad,
                cube_noise.cube_scaled_mad_outside,
                moment_difference_level,
                regions["OUTSIDE_JOINT_MASK"],
            )
            negative_fraction = negative_pixel_fraction(
                maps["SKY_MOMENT_MAXIMUM_SELECTION"], regions["OUTSIDE_JOINT_MASK"]
            )
            original_moment_difference = maps["MOMENT_DIFFERENCE"]
            original_snr = moment_difference_snr
            measures["T_selection_maps"] += time.perf_counter() - start
            start = time.perf_counter()
            transition = after_original_round(
                settings.bad_atmosphere,
                statistics,
                cube_noise,
                counts,
                negative_fraction,
                _warnings(ranges, settings).low_bandwidth,
                ranges,
                ranges_before_blue_pruning,
                aggregate_bandwidth,
                selection.final_threshold_sigma,
                threshold_sigma,
                threshold_sigma_mode,
                joint_mask_0,
                maps["SKY_MOMENT_MAXIMUM_SELECTION"],
                maps["MOMENT_DIFFERENCE"],
                regions["PRIMARY_BEAM_ABOVE_SPECTRUM_LEVEL"],
            )
            if transition.next_round_kind != "amended_mask":
                original_moment_difference = None  # read only by fC 2200-2207
            measures["T_round_decisions"] += time.perf_counter() - start
        elif round_kind == "amended_mask":
            difference_snr = difference_map_snr(
                maps["MOMENT_DIFFERENCE"],
                original_moment_difference,
                regions["OUTSIDE_JOINT_MASK"],
                regions["NOISE_ANNULUS"],
            )  # fC 2200-2207
            original_moment_difference = None
            measures["T_selection_maps"] += time.perf_counter() - start
            start = time.perf_counter()
            transition = after_amended_round(
                settings.bad_atmosphere,
                statistics,
                cube_noise.ten_event_sigma,
                _warnings(ranges, settings).low_bandwidth,
                difference_snr,
                ranges,
                ranges_before_blue_pruning,
                aggregate_bandwidth,
                selection_reference,
                change_code_references,
                maps["MOMENT_DIFFERENCE"],
                settings.n_channels,
                threshold_sigma,
                threshold_sigma_mode,
            )
            measures["T_round_decisions"] += time.perf_counter() - start
            if transition.maps_request is not None:  # fC 2266-2344
                decisions.append(transition.decisions)
                record.update(transition.round_record)
                maps, _, new_records, step = _round_maps(
                    window_xds,
                    layout,
                    settings,
                    first_round_store,
                    transition.maps_request,
                    None,
                    regions["MOMENT_DIFFERENCE_REGION"],
                    map_coords,
                    run_directory,
                    round_index,
                    round_kind,
                    monitor_resources_seconds,
                    len(records),
                )
                records, measures = records + new_records, _sum_measures(measures, step)
                start = time.perf_counter()
                statistics = selection_statistics(maps, regions)
                moment_difference_snr = statistics.moment_difference_snr
                measures["T_selection_maps"] += time.perf_counter() - start
                start = time.perf_counter()
                transition = after_amended_intersection(
                    settings.bad_atmosphere,
                    statistics,
                    cube_noise.ten_event_sigma,
                    transition.maps_request,
                    settings.channel_width,
                    change_code_references,
                    maps["MOMENT_DIFFERENCE"],
                    threshold_sigma,
                    threshold_sigma_mode,
                )
                measures["T_round_decisions"] += time.perf_counter() - start
        else:
            measures["T_selection_maps"] += time.perf_counter() - start
            start = time.perf_counter()
            transition = after_extra_round(
                round_kind,
                maps_ranges,
                statistics,
                selection.final_threshold_sigma,
                _warnings(maps_ranges.ranges, settings),
                settings.bad_atmosphere,
                settings.n_rounds_after_original,
                cube_noise.ten_event_sigma,
                change_code_references,
                ranges,
                ranges_before_blue_pruning,
                aggregate_bandwidth,
                selection_reference,
                settings.channel_width,
                threshold_sigma_mode,
            )
            measures["T_round_decisions"] += time.perf_counter() - start
        decisions.append(transition.decisions)
        record.update(transition.round_record)
        round_records.append(record)
        # Freed before the next pass; the transition holds what it reads.
        maps = None
        if transition.extra_mask_request is not None:
            extra_spectrum, new_records, step = _extra_mask_spectrum(
                window_xds,
                layout,
                settings,
                first_round_store,
                transition.extra_mask_request,
                regions,
                cube_levels,
                round_joint_mask,
                run_directory,
                round_index,
                round_kind,
                monitor_resources_seconds,
                len(records),
            )
            records, measures = records + new_records, _sum_measures(measures, step)
        (
            round_kind,
            threshold_sigma,
            threshold_sigma_mode,
            signal_ratio_tier_1,
            selection_reference,
            change_code_references,
            ranges,
            ranges_before_blue_pruning,
            aggregate_bandwidth,
            round_change_code,
        ) = (
            transition.next_round_kind,
            transition.threshold_sigma,
            transition.threshold_sigma_mode,
            transition.signal_ratio_tier_1,
            transition.selection_reference,
            transition.change_code_references,
            transition.continuum_ranges,
            transition.ranges_before_blue_pruning,
            transition.aggregate_bandwidth,
            transition.change_code,
        )
        if transition.amended_joint_mask is not None:
            round_joint_mask = transition.amended_joint_mask  # fC 2083-2099
        if transition.result is not None:
            break

    outcome = _RoundsOutcome(
        rows,
        first_round_store,
        False,
        joint_mask=round_joint_mask,
        first_round_statistics=statistics_0,
        spectrum=computed_spectrum,
        selection=computed_selection,
        rounds=round_records,
    )
    if not settings.rounds_enabled:
        # No reversion; the signal-to-noise ratio is 0 (fC 2834, 3086-3089).
        return (
            outcome._replace(
                continuum_ranges=ranges,
                ranges_before_blue_pruning=ranges_before_blue_pruning,
                aggregate_bandwidth=aggregate_bandwidth,
                moment_difference_snr=0.0,
                path_code=path_code(decisions, False),
            ),
            records,
            measures,
        )

    start = time.perf_counter()
    warnings = _warnings(ranges, settings)
    final = after_rounds(
        round_change_code, moment_difference_snr, moment_difference_level, warnings
    )
    reverted, low_spread_ranges_found = final.revert, None
    if final.run_low_spread_fix:
        low_spread_ranges_found = low_spread_ranges(
            ranges,
            settings.n_channels,
            computed_selection.baseline_channels,
            computed_spectrum.spectrum,
            computed_selection.positive_threshold,
            computed_selection.negative_threshold,
        )  # fC 2856-2919
        if low_spread_ranges_found is None:
            reverted = after_low_spread_fix(
                None, moment_difference_snr, moment_difference_level, warnings
            )  # fC 3051-3053
    measures["T_round_decisions"] += time.perf_counter() - start
    if low_spread_ranges_found is not None:
        if change_code_references is None:  # fC 2922-2932
            change_code_references = _change_values(statistics)
        start = time.perf_counter()
        ranges = low_spread_ranges_found
        aggregate_bandwidth = selection_bandwidth(ranges, settings.channel_width)
        warnings = _warnings(ranges, settings)  # fC 2939-2944
        measures["T_round_decisions"] += time.perf_counter() - start
        maps, _, new_records, step = _round_maps(
            window_xds,
            layout,
            settings,
            first_round_store,
            ranges,
            None,
            regions["MOMENT_DIFFERENCE_REGION"],
            map_coords,
            run_directory,
            None,
            None,
            monitor_resources_seconds,
            len(records),
        )
        records, measures = records + new_records, _sum_measures(measures, step)
        start = time.perf_counter()
        statistics = selection_statistics(maps, regions)
        moment_difference_snr = statistics.moment_difference_snr
        measures["T_selection_maps"] += time.perf_counter() - start
        start = time.perf_counter()
        round_change_code = change_code(
            _change_values(statistics),
            change_code_references,
            threshold=_LOW_SPREAD_CHANGE_THRESHOLD,
        )  # fC 3026-3035
        decisions.append({"added_selection": True})
        reverted = after_low_spread_fix(
            round_change_code, moment_difference_snr, moment_difference_level, warnings
        )  # fC 3040-3053
        measures["T_round_decisions"] += time.perf_counter() - start
    if reverted:  # fC 3054-3066
        ranges = original_ranges
        aggregate_bandwidth = selection_bandwidth(ranges, settings.channel_width)
        moment_difference_snr = original_snr
    return (
        outcome._replace(
            continuum_ranges=ranges,
            ranges_before_blue_pruning=ranges_before_blue_pruning,
            aggregate_bandwidth=aggregate_bandwidth,
            moment_difference_snr=moment_difference_snr,
            change_code=round_change_code,
            path_code=path_code(decisions, True),
            reverted=bool(reverted),
            low_spread_fix_applied=low_spread_ranges_found is not None,
        ),
        records,
        measures,
    )


def _products(outcome, settings, window_range, frequency_frame):
    """The results of the spectral window (fC 3071-3147, heur 109-137), but
    the path of the joint-mask store."""
    from astroviper.processing_functions.imaging.find_continuum_ultra.products import (
        continuum_frequency_ranges,
        is_all_continuum,
        single_range_channel_fraction,
    )

    ranges = [[int(start), int(stop)] for start, stop in outcome.continuum_ranges]
    warnings = _warnings(ranges, settings)
    return {
        "continuum_channel_ranges": ranges,
        "channel_range": list(window_range),
        "continuum_frequency_ranges": continuum_frequency_ranges(
            ranges, settings.first_frequency, settings.channel_increment
        ),
        "frequency_frame": frequency_frame,
        "aggregate_bandwidth": float(outcome.aggregate_bandwidth),
        "all_continuum": bool(
            is_all_continuum(
                ranges,
                outcome.ranges_before_blue_pruning,
                outcome.aggregate_bandwidth,
                settings.channel_width,
                settings.n_channels,
            )
        ),
        "low_bandwidth": bool(warnings.low_bandwidth),
        "low_spread": bool(warnings.low_spread),
        "single_range_channel_fraction": single_range_channel_fraction(
            ranges, settings.n_channels
        ),
        "moment_difference_snr": float(outcome.moment_difference_snr),
        "constant_cube": False,
        "joint_mask": None,
        "joint_mask_kind": outcome.joint_mask.attrs["kind"],
        "joint_mask_primary_beam_based": bool(
            outcome.joint_mask.attrs["primary_beam_based"]
        ),
    }


def _constant_cube_result(rows, window_range, frequency_frame, input_image_store):
    """The results of a spectral window whose valid voxels hold one value:
    no continuum channel, and findContinuum does not run (heur 89-94)."""
    logger.warning(
        f"Channels {list(window_range)} of {input_image_store} are constant at "
        f"{np.nanmax(rows.sky_maximum)}: find_continuum_ultra selects no channel."
    )
    return {
        "continuum_channel_ranges": [],
        "channel_range": list(window_range),
        "continuum_frequency_ranges": [],
        "frequency_frame": frequency_frame,
        "aggregate_bandwidth": 0.0,
        "all_continuum": False,
        "low_bandwidth": False,
        "low_spread": False,
        "single_range_channel_fraction": None,
        "moment_difference_snr": None,
        "constant_cube": True,
        "joint_mask": None,
        "joint_mask_kind": None,
        "joint_mask_primary_beam_based": None,
    }


def _return_dict(
    results,
    outcome,
    settings,
    frequency,
    data_group,
    records,
    measures,
    store_inputs,
    imager_costs,
    layout,
    thread_info,
    application_start,
    rss_peak_start,
    resource_usage,
    monitor_resources_seconds,
):
    """The return dictionary: the results, the diagnostics and the profiling
    of the passes and of the driver."""
    import pandas as pd

    from astroviper.distributed_applications.imaging.find_continuum_ultra_passes import (
        find_continuum_ultra_driver_bytes,
    )

    rows = [record.row for record in records]
    timing = {key: float(measures.get(key, 0.0)) for key in _LEAVES}
    for key, merge_key in (
        ("T_create_map_reduce_graph", "T_merge_create_graph"),
        ("T_generate_dask_graph", "T_merge_generate_dask_graph"),
        ("T_compute_dask_graph", "T_merge_compute_dask_graph"),
    ):
        timing[key] = sum(row[key] + row[merge_key] for row in rows)
    timing["T_write_regions"] += sum(row["T_write_regions"] for row in rows)
    timing["T_read_maps"] += sum(row["T_read_maps"] for row in rows)
    timing["T_load_first_round"] += sum(row["T_first_round_rows"] for row in rows)
    application_end = time.time()
    timing["T_total"] = application_end - application_start
    timing["application_start_unixtime"] = application_start
    timing["application_end_unixtime"] = application_end
    timing["compute_start_unixtime"] = (
        min(row["start_unixtime"] for row in rows) if rows else None
    )
    timing["compute_end_unixtime"] = (
        float(
            np.nanmax(
                [
                    [row["compute_end_unixtime"], row["merge_end_unixtime"]]
                    for row in rows
                ]
            )
        )
        if rows
        else None
    )
    rss_peak = _rss_peak_bytes()
    # The process's peak is the call's when the call raised it; otherwise only
    # the driver's samples, if any, tell the call's peak.
    samples = (resource_usage or {}).get("memory_rss_bytes") or []
    timing["driver_rss_peak_bytes"] = (
        rss_peak
        if rss_peak > rss_peak_start
        else (int(max(samples)) if samples else None)
    )
    timing["driver_rss_lifetime_peak_bytes"] = rss_peak
    for key in ("driver_bytes_read", "driver_bytes_written"):
        timing[key] = int(measures.get(key, 0) + sum(row[key] for row in rows))
    if monitor_resources_seconds:
        timing["driver_resource_usage"] = resource_usage

    spectrum, selection = outcome.spectrum, outcome.selection
    diagnostics = {
        "settings": {
            name: value
            for name, value in settings._asdict().items()
            if name != "moment_channel_weights"
        },
        "first_round_statistics": None
        if outcome.first_round_statistics is None
        else outcome.first_round_statistics._asdict(),
        "rounds": pd.DataFrame(outcome.rounds),
        "change_code": outcome.change_code,
        "path_code": outcome.path_code,
        "reverted": outcome.reverted,
        "low_spread_fix_applied": outcome.low_spread_fix_applied,
        "mean_spectrum": None
        if spectrum is None
        else _spectrum_over_frequency(
            spectrum.mean_spectrum, "MEAN_SPECTRUM", frequency
        ),
        "spectrum": None
        if spectrum is None
        else _spectrum_over_frequency(spectrum.spectrum, "SPECTRUM", frequency),
        "positive_threshold": None
        if selection is None
        else selection.positive_threshold,
        "negative_threshold": None
        if selection is None
        else selection.negative_threshold,
        "corrected_median": None if selection is None else selection.corrected_median,
    }
    moments_from_store = store_inputs is not None and store_inputs.moments_usable
    return {
        **results,
        "diagnostics": diagnostics,
        "n_passes": len(rows),
        "cube_reads": {
            variable: sum(
                variable in row["data_variables_read"].split(", ") for row in rows
            )
            for variable in (data_group["sky"], data_group["primary_beam"])
        },
        "passes": pd.DataFrame(rows),
        "timing_node_tasks": pd.concat(
            [record.timing_node_tasks for record in records], ignore_index=True
        )
        if records
        else pd.DataFrame(),
        "timing_reduce_nodes": [
            reduce_record
            for record in records
            for reduce_record in record.timing_reduce_nodes
        ],
        "timing_distributed_application": timing,
        "first_round": {
            "source": "imager"
            if store_inputs is not None
            and store_inputs.written_by == "image_cube_single_field"
            else "find_continuum_ultra",
            "store": outcome.first_round_store,
            "moments": "imager" if moments_from_store else "computed",
            "imager": imager_costs,
        },
        "memory_plan": {
            "n_threads": layout.n_threads,
            "memory_per_thread_gib": thread_info["memory_per_thread"],
            "budget_bytes": layout.budget_bytes,
            "n_frequency_tasks": layout.n_frequency_tasks,
            "channels_per_task": layout.channels_per_task,
            "merge_tile_shape": layout.merge_tile_shape,
            "n_merge_tiles": len(layout.tile_mapping),
            "plane_bytes": layout.plane_bytes,
            "driver_memory_estimate_bytes": find_continuum_ultra_driver_bytes(
                settings.n_l, settings.n_m
            ),
        },
        "report": None,
    }


def _change_values(statistics):
    """The values of the change code, in its order: the moment difference's
    peak, peak outside the joint mask, sum and scaled MAD (fC 3630-3687)."""
    return (
        statistics.moment_difference_peak,
        statistics.moment_difference_peak_outside,
        statistics.moment_difference_sum,
        statistics.moment_difference_scaled_mad,
    )


def _warnings(ranges, settings):
    """The bandwidth warnings of channel ranges of the window."""
    from astroviper.processing_functions.imaging.find_continuum_ultra.products import (
        bandwidth_warnings,
    )

    return bandwidth_warnings(
        ranges, settings.n_channels, settings.channel_width, settings.channel_increment
    )


def _as_map(values, name, map_coords):
    """An ``(l, m)`` or image-shaped array as the map ``name`` on
    ``map_coords``."""
    from astroviper.processing_functions.imaging.find_continuum_ultra.first_round_maps import (
        MAP_DIMS,
    )

    return map_coords.assign(
        {name: (MAP_DIMS, values.reshape(1, 1, 1, *values.shape[-2:]))}
    )[name]


def _spectrum_over_frequency(values, name, frequency):
    """A spectrum of the window as a ``DataArray`` over ``frequency`` (Hz)."""
    return xr.DataArray(
        values, coords={"frequency": frequency}, dims="frequency", name=name
    )


def _pass_directory(run_directory, pass_index):
    """The directory of a pass's stores in the run's directory."""
    return os.path.join(run_directory, f"pass_{pass_index}")


def _bytes_read(counters):
    """Stored bytes of the read counters of ``read_array_region``."""
    return counters.get("bytes", 0) + counters.get("index_bytes", 0)


def _sum_measures(*measures):
    """The sum of the driver's measures, key by key."""
    total = dict.fromkeys((*_LEAVES, "driver_bytes_read", "driver_bytes_written"), 0)
    for measure in measures:
        for key, value in measure.items():
            total[key] += value
    return total


def _rss_peak_bytes():
    """Peak resident memory of this process (``ru_maxrss`` is in kB on Linux
    and in bytes on macOS)."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)
