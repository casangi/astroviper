"""Plots of the per-task resource-usage series recorded by graphviper's monitor.

When a distributed application is run with ``monitor_resources_seconds`` set
(e.g. :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`),
its ``return_dict["timing_node_tasks"]`` frame carries one sampled series per
node task as list-valued columns: ``time_seconds`` (relative to the task's own
start), ``cpu_percent`` (whole worker process incl. OpenMP threads, so >100%
is normal; newer graphviper also records the cumulative ``cpu_seconds``),
``memory_rss_bytes``, the cumulative syscall-level I/O counters
``read_chars`` / ``write_chars`` (Linux only; network filesystems such as
Lustre included), the scalar ``sample_interval_seconds`` and the wall-clock
anchor ``start_unixtime``.

Four complementary views:

* :func:`plot_task_resource_usage` -- tasks aligned at their OWN start
  (t=0 = task start): a subsample of individual task curves plus the average
  over the tasks. "What does a typical task do?"
* :func:`plot_cluster_resource_usage` -- tasks placed at their wall-clock
  position via ``start_unixtime`` and SUMMED over the tasks, as bin means:
  busy cores, total RSS, aggregate I/O rate. "What was the whole cluster
  doing at each moment of the run?"
* :func:`plot_core_resource_usage` -- the same bins per worker slot (a core
  when workers are single-threaded, one per core): the peak, mean, median
  and minimum over the slots of CPU, RSS and the read/write rates. "How
  evenly were the cores loaded?"
* :func:`plot_task_stream` -- a Dask-dashboard-style per-worker-process
  timeline: one lane per worker slot, each task drawn as load / science /
  write segments, plus a running-task utilization panel, a printed
  efficiency decomposition and -- given the application's own step list
  (``timing_phases``) -- a strip of the distributed application's setup
  (every driver step before the compute call), compute and finalize steps,
  so the timeline accounts for the entire distributed application call.
  Unlike the three views above it needs only the
  wall-clock anchor ``start_unixtime`` (recorded by the node task itself),
  NOT the monitor's sampled series, so it works with
  ``monitor_resources_seconds=None``.

All accept the full return dict or the ``timing_node_tasks`` frame directly,
and skip gracefully (with a printed note) when the needed columns are absent.
:func:`performance_measures` gives the performance measures of a run (the
RADPS benchmarking memo's speedups and parallel efficiencies), which the task
stream prints. The worker slots, hosts and memory come from the run's
``return_dict["compute_resources"]``
(:func:`~astroviper.utils.data_partitioning.get_compute_resources`), or from
a ``compute_resources`` argument, which replaces it.

Over several runs, :func:`plot_scaling` draws the performance measures, the
total time or the throughput, and :func:`plot_efficiency_loss_decomposition`
splits each run's wall clock into useful work and losses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "performance_measures",
    "plot_task_resource_usage",
    "plot_cluster_resource_usage",
    "plot_core_resource_usage",
    "plot_task_stream",
    "plot_scaling",
    "plot_efficiency_loss_decomposition",
    "plot_application_task_stream",
    "assign_task_stream_lanes",
    "IMAGING_TASK_STREAM_COLUMNS",
]

# The node task timing columns plot_task_stream draws by default, those of the
# imaging node task: the task's total time, the columns of its load segment
# and the column of its write segment.
IMAGING_TASK_STREAM_COLUMNS = {
    "task": "T_image_cube_task",
    "load": ("T_make_empty_image", "T_load"),
    "write": "T_write",
}

# Task-stream palette (colorblind-safe Okabe-Ito).
_LOAD_COLOR, _SCIENCE_COLOR, _WRITE_COLOR = "#0072B2", "#009E73", "#E69F00"
_POST_COLOR = "#CC79A7"
_PRE_COLOR = "#7f7f7f"

_DRIVER_SETUP_LABEL = "setup"
_DRIVER_COMPUTE_LABEL = "compute (task stream below)"
_DRIVER_UNTIMED_LABEL = "driver (untimed)"
# Everything before compute is drawn as ONE light-red "setup" segment (its
# per-step breakdown goes to the printed summary and the HTML tooltip); the
# compute window is near-white so it reads as "see the panels below", the
# finalize steps get pale colors from the palette, untimed gaps gray.
_DRIVER_SETUP_COLOR = "#fbb4ae"
_DRIVER_STEP_COLORS = [
    "#8dd3c7",
    "#bebada",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]
_DRIVER_COMPUTE_COLOR = "#eef3fa"
_DRIVER_UNTIMED_COLOR = "#d9d9d9"


def _driver_steps(timing_phases):
    """The ``(label, key)`` steps of a phase layout before and after its
    ``T_compute_dask_graph`` step, in layout order."""
    steps = [step for _, _, phase_steps in timing_phases for step in phase_steps]
    keys = [key for _, key in steps]
    if "T_compute_dask_graph" not in keys:
        raise ValueError(
            "timing_phases has no T_compute_dask_graph step, so its steps "
            "cannot be placed before and after the compute call."
        )
    split = keys.index("T_compute_dask_graph")
    return steps[:split], steps[split + 1 :]


# Look of the performance figures. The two windows of the performance measures
# (RADPS benchmarking memo): map reduce (t3 - t1) and end to end (t3 - t0).
_WINDOW_NAMES = {"map_reduce": "map reduce", "end_to_end": "end to end"}
_WINDOW_STYLES = {
    "map_reduce": {"marker": "o", "color": "C0"},
    "end_to_end": {"marker": "s", "color": "C1"},
}
# The ideal reference (S = p, E = 1) of every speedup and efficiency figure.
_IDEAL_LINE = {"linestyle": "--", "color": "black"}
# Symbols and legend labels of the measures of performance_measures: absolute
# (the run's own work W over its window) and relative (to the baseline run).
_MEASURE_SYMBOLS = {
    "speedup_map_reduce": r"$S^{mr}$",
    "speedup_end_to_end": r"$S^{e2e}$",
    "speedup_map_reduce_relative": r"$S^{mr,rel}$",
    "speedup_end_to_end_relative": r"$S^{e2e,rel}$",
    "parallel_efficiency_map_reduce": r"$E^{mr}$",
    "parallel_efficiency_end_to_end": r"$E^{e2e}$",
    "parallel_efficiency_map_reduce_relative": r"$E^{mr,rel}$",
    "parallel_efficiency_end_to_end_relative": r"$E^{e2e,rel}$",
}
_MEASURE_LABELS = {
    "speedup_map_reduce": r"map reduce: $S^{mr} = W/(t_3-t_1)$",
    "speedup_end_to_end": r"end to end: $S^{e2e} = W/(t_3-t_0)$",
    "speedup_map_reduce_relative": (
        r"map reduce: $S^{mr,rel} = p_{\min}\,(t_3-t_1)_{\min}/(t_3-t_1)$"
    ),
    "speedup_end_to_end_relative": (
        r"end to end: $S^{e2e,rel} = p_{\min}\,(t_3-t_0)_{\min}/(t_3-t_0)$"
    ),
    "parallel_efficiency_map_reduce": r"map reduce: $E^{mr} = S^{mr}/p$",
    "parallel_efficiency_end_to_end": r"end to end: $E^{e2e} = S^{e2e}/p$",
    "parallel_efficiency_map_reduce_relative": (
        r"map reduce: $E^{mr,rel} = S^{mr,rel}/p$"
    ),
    "parallel_efficiency_end_to_end_relative": (
        r"end to end: $E^{e2e,rel} = S^{e2e,rel}/p$"
    ),
}
# Weak scaling: the relative measures scale the baseline's ideal window by the
# work ratio W/W0.
_WEAK_RELATIVE_LABELS = {
    "speedup_map_reduce_relative": (
        r"map reduce: $S^{mr,rel} = p_0\,(W/W_0)\,(t_3-t_1)_0/(t_3-t_1)$"
    ),
    "speedup_end_to_end_relative": (
        r"end to end: $S^{e2e,rel} = p_0\,(W/W_0)\,(t_3-t_0)_0/(t_3-t_0)$"
    ),
    "parallel_efficiency_map_reduce_relative": _MEASURE_LABELS[
        "parallel_efficiency_map_reduce_relative"
    ],
    "parallel_efficiency_end_to_end_relative": _MEASURE_LABELS[
        "parallel_efficiency_end_to_end_relative"
    ],
}
# The parts of the wall clock in the efficiency-loss decomposition, bottom to
# top: name and colour.
_LOSS_PARTS = (
    ("useful work", "#009E73"),
    ("ramp", "tab:blue"),
    ("in-window gaps", "tab:cyan"),
    ("straggler tail", "tab:red"),
    ("pre-map", "tab:gray"),
    ("post-map", "tab:purple"),
    ("fixed costs", "tab:brown"),
)


def _driver_timeline(driver_timing, compute_start, compute_end, t0_abs, timing_phases):
    """Place the driver's setup/finalize steps on the run's wall clock.

    ``driver_timing`` is the application's ``timing_distributed_application``
    dict (or the same keys pulled from a benchmark overall row) and
    ``timing_phases`` its phase layout, the application's
    ``DISTRIBUTED_APPLICATION_TIMING_PHASES``: the steps before
    ``T_compute_dask_graph`` are the setup, those after it the finalize steps
    (:func:`_driver_steps`), listed in layout order. EVERYTHING before the
    measured compute start is drawn as one "setup" segment (starting at the
    absolute ``application_start_unixtime`` anchor when recorded -- since
    2026-08-31 -- else at compute start minus the summed setup steps); the
    finalize steps run forward from the compute end, with any leftover time
    before the ``application_end_unixtime`` anchor shown as an "(untimed)"
    gray segment, so the timeline spans the ENTIRE application call.

    Returns ``(segments, app_start_rel, app_end_rel, color_of, setup_detail)``
    -- segments as ``(label, start, end)`` in seconds relative to ``t0_abs``
    (the first task start), ``color_of`` mapping each label to its color and
    ``setup_detail`` the per-step ``(label, seconds)`` breakdown of the setup
    segment -- or None when ``driver_timing``, ``timing_phases`` or the
    compute anchors are unavailable.
    """
    if (
        not driver_timing
        or timing_phases is None
        or compute_start is None
        or compute_end is None
    ):
        return None

    def get(key):
        value = driver_timing.get(key)
        if value is None:
            return None
        value = float(value)
        return None if np.isnan(value) else value

    pre_steps, post_steps = _driver_steps(timing_phases)
    pre = [(lbl, get(k)) for lbl, k in pre_steps if get(k) is not None]
    post = [(lbl, get(k)) for lbl, k in post_steps if get(k) is not None]
    if not pre and not post:
        return None

    rel_cs, rel_ce = compute_start - t0_abs, compute_end - t0_abs
    setup_start = rel_cs - sum(dur for _, dur in pre)
    app_start = get("application_start_unixtime")
    app_start_rel = setup_start if app_start is None else app_start - t0_abs
    setup_detail = list(pre)
    if app_start is not None and setup_start - app_start_rel > 0.05:
        setup_detail.insert(0, (_DRIVER_UNTIMED_LABEL, setup_start - app_start_rel))
    segments = []
    if rel_cs - app_start_rel > 0:
        segments.append((_DRIVER_SETUP_LABEL, app_start_rel, rel_cs))
    segments.append((_DRIVER_COMPUTE_LABEL, rel_cs, rel_ce))
    cursor = rel_ce
    for lbl, dur in post:
        segments.append((lbl, cursor, cursor + dur))
        cursor += dur
    app_end = get("application_end_unixtime")
    app_end_rel = cursor if app_end is None else app_end - t0_abs
    if app_end is not None and app_end_rel - cursor > 0.05:
        segments.append((_DRIVER_UNTIMED_LABEL, cursor, app_end_rel))

    color_of, palette = {}, iter(_DRIVER_STEP_COLORS * 3)
    for lbl, _, _ in segments:
        if lbl not in color_of:
            if lbl == _DRIVER_SETUP_LABEL:
                color_of[lbl] = _DRIVER_SETUP_COLOR
            elif lbl == _DRIVER_COMPUTE_LABEL:
                color_of[lbl] = _DRIVER_COMPUTE_COLOR
            elif lbl == _DRIVER_UNTIMED_LABEL:
                color_of[lbl] = _DRIVER_UNTIMED_COLOR
            else:
                color_of[lbl] = next(palette)
    return segments, app_start_rel, app_end_rel, color_of, setup_detail


def _task_columns(task_columns):
    """The node task timing columns: ``task_columns`` over the imager's."""
    return {**IMAGING_TASK_STREAM_COLUMNS, **(task_columns or {})}


def _with_imaging_task_columns(tasks, task_columns):
    """A copy of ``tasks`` with the ``task_columns`` timings under the imager's
    column names, which the task stream draws. A load column the frame lacks
    is left out of the load segment, with a printed note."""
    columns = _task_columns(task_columns)
    for column in columns["load"]:
        if column not in tasks.columns:
            print(f"task stream: no {column} column; left out of the load segment")
    load = [column for column in columns["load"] if column in tasks.columns]
    task = tasks[columns["task"]]
    load_time = tasks[load].sum(axis=1) if load else 0.0
    write_time = tasks[columns["write"]] if columns["write"] else 0.0
    tasks = tasks.copy()
    tasks["T_image_cube_task"] = task
    tasks["T_make_empty_image"] = 0.0
    tasks["T_load"] = load_time
    tasks["T_write"] = write_time
    return tasks


def _as_timing_frame(source):
    """Accept an application return dict or the timing_node_tasks frame."""
    if isinstance(source, dict):
        source = source["timing_node_tasks"]
    if not isinstance(source, pd.DataFrame):
        raise TypeError(
            "expected the application return dict or its 'timing_node_tasks' "
            f"DataFrame, got {type(source).__name__}"
        )
    return source.reset_index(drop=True)


def _cpu_seconds(row):
    """Cumulative CPU seconds of a task's worker process at each sample.

    From graphviper's ``cpu_seconds`` series when recorded. Older series only
    have ``cpu_percent``, the average over each sample's interval, so
    cpu_percent[k] * (t[k] - t[k-1]) rebuilds the increments; their sample 0
    covered only the microseconds after psutil's baseline call (0 or up to
    ~15,000 %) and is left out.
    """
    c = row.get("cpu_seconds")
    if isinstance(c, (list, np.ndarray)):
        return np.asarray(c, dtype=float)
    t = np.asarray(row["time_seconds"], dtype=float)
    pct = np.asarray(row["cpu_percent"], dtype=float)
    return np.concatenate(([0.0], np.cumsum(pct[1:] / 100.0 * np.diff(t))))


def _cpu_percent_series(t, c, interval):
    """CPU % over each sample interval from the cumulative CPU seconds ``c``,
    with graphviper's edge rule: a last interval shorter than half the sample
    interval is merged into the one before, and sample 0, which has no
    interval, repeats sample 1."""
    if len(t) > 2 and t[-1] - t[-2] < 0.5 * interval:
        t, c = np.delete(t, -2), np.delete(c, -2)
    if len(t) < 2:
        return t, np.zeros_like(t)
    dt = np.diff(t)
    pct = np.divide(100.0 * np.diff(c), dt, out=np.zeros_like(dt), where=dt > 0)
    return t, np.concatenate((pct[:1], pct))


def _time_integral(t, v):
    """Cumulative trapezoid integral of ``v`` over ``t``, 0 at the first sample."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))))


def _bin_increments(edges, offset, t_rel, cumulative):
    """Increase of a task's cumulative series within each bin of ``edges``
    (run clock); 0 in bins before the task starts and after it ends."""
    return np.diff(np.interp(edges - offset, t_rel, cumulative))


def _display_bins(span, interval, period, bin_seconds=None):
    """Bin edges over the run [0, span]: about 600 bins of a whole number of
    sample intervals. Once a bin would be longer than a quarter of the task
    period it is a whole number of task periods instead, so the bin means do
    not alias the workers' lockstep reads and write bursts. ``bin_seconds``,
    when given, replaces that choice, up to half the run so that there are
    at least two bins. The last bin ends at ``span``; one shorter than half
    a bin is merged into the bin before.
    Returns (bin width, edges)."""
    b = span / 600 if bin_seconds is None else min(bin_seconds, span / 2)
    if bin_seconds is None and period > 0 and b > 0.25 * period:
        b = period * max(1, round(b / period))
    binw = interval * max(1, round(b / interval))
    edges = np.append(np.arange(0.0, span, binw), span)
    if len(edges) > 2 and edges[-1] - edges[-2] < 0.5 * binw:
        edges = np.delete(edges, -2)
    return binw, edges


def _running_task_count(starts, ends):
    """The number of running tasks from their start and end times: returns
    ``(event_times, running)``, the count after each event in time order."""
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    events = np.concatenate([starts, ends])
    deltas = np.concatenate([np.ones(len(starts)), -np.ones(len(ends))])
    order = np.argsort(events, kind="stable")
    return events[order], np.cumsum(deltas[order])


def _non_io_task_time(tasks, task_columns=None):
    """Every task's time outside its load and write segments (seconds): the
    task column minus the load columns present (NaN skipped) and the write
    column (see :data:`IMAGING_TASK_STREAM_COLUMNS`)."""
    columns = _task_columns(task_columns)
    load = [column for column in columns["load"] if column in tasks.columns]
    value = tasks[columns["task"]].to_numpy(dtype=float)
    if load:
        value = value - tasks[load].sum(axis=1).to_numpy(dtype=float)
    if columns["write"]:
        value = value - tasks[columns["write"]].to_numpy(dtype=float)
    return value


def _task_start_bins(tasks, values, n_bins):
    """The median of ``values`` (one per task) in ``n_bins`` quantile bins of
    the tasks' ``start_unixtime``; tasks with a non-finite value or start are
    left out. Returns a DataFrame with one row per bin: ``bin``, ``hours``
    (the mean start in the bin, hours after the first start), ``fraction``
    (the bin's position in the run, ``(bin + 0.5) / n_bins``) and
    ``median``."""
    start = tasks["start_unixtime"].to_numpy(dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(start) & np.isfinite(values)
    if not keep.any():
        return pd.DataFrame(columns=["bin", "hours", "fraction", "median"])
    start, values = start[keep], values[keep]
    binned = (
        pd.DataFrame(
            {
                "bin": pd.qcut(start, n_bins, labels=False, duplicates="drop"),
                "hours": (start - start.min()) / 3600,
                "value": values,
            }
        )
        .groupby("bin")
        .agg(hours=("hours", "mean"), median=("value", "median"))
        .reset_index()
    )
    binned["fraction"] = (binned["bin"] + 0.5) / n_bins
    return binned[["bin", "hours", "fraction", "median"]]


def _is_series(value):
    """Whether a timing-frame cell holds a sampled series."""
    return isinstance(value, (list, np.ndarray))


def _per_task_totals(tasks):
    """Per-task totals of the monitor's series, aligned with ``tasks``: the
    CPU seconds (``cpu_seconds``), the mean CPU over the task (%,
    ``mean_cpu_percent``), the peak resident memory (``peak_rss_gib``) and
    the GiB read and written (``read_gib``, ``written_gib``); NaN where a
    series is absent or has fewer than 2 samples."""
    names = ("cpu_seconds", "mean_cpu_percent", "peak_rss_gib", "read_gib")
    rows = []
    for _, row in tasks.iterrows():
        totals = dict.fromkeys((*names, "written_gib"), np.nan)
        t = row.get("time_seconds")
        cpu = row.get("cpu_seconds")
        if not _is_series(cpu):
            cpu = row.get("cpu_percent")
        if _is_series(t) and len(t) >= 2 and _is_series(cpu) and len(cpu) >= 2:
            c = _cpu_seconds(row)
            totals["cpu_seconds"] = c[-1] - c[0]
            span = float(t[-1] - t[0])
            if span > 0:
                totals["mean_cpu_percent"] = 100.0 * totals["cpu_seconds"] / span
        rss = row.get("memory_rss_bytes")
        if _is_series(rss) and len(rss) >= 2:
            totals["peak_rss_gib"] = float(np.max(rss)) / 2**30
        for column, key in (("read_chars", "read_gib"), ("write_chars", "written_gib")):
            c = row.get(column)
            if _is_series(c) and len(c) >= 2:
                totals[key] = (float(c[-1]) - float(c[0])) / 2**30
        rows.append(totals)
    return pd.DataFrame(rows, index=tasks.index, columns=[*names, "written_gib"])


def _compute_resources(source, compute_resources):
    """The compute resources of a run: ``compute_resources`` when given (it
    replaces the recorded ones), else the return dict's
    ``"compute_resources"``; ``{}`` when neither is known. Checked with
    :func:`~astroviper.utils.data_partitioning.check_compute_resources`."""
    from astroviper.utils.data_partitioning import check_compute_resources

    if compute_resources is not None:
        resources = compute_resources
    elif isinstance(source, dict):
        resources = source.get("compute_resources") or {}
    else:
        resources = {}
    if resources:
        check_compute_resources(resources)
    return resources


def _n_worker_slots(resources, map_tasks):
    """p, the tasks that can run at once, and where it comes from: the
    recorded ``n_worker_slots``, else the lanes of the map tasks
    (:func:`assign_task_stream_lanes` of a frame with ``start_unixtime``,
    ``T_image_cube_task``, ``hostname`` and, when recorded, the execution
    identity). Returns ``(n_worker_slots, "recorded" or "lane packing")``."""
    if resources.get("n_worker_slots") is not None:
        return int(resources["n_worker_slots"]), "recorded"
    return assign_task_stream_lanes(map_tasks)[2], "lane packing"


def _n_hosts(resources):
    """The recorded number of hosts with workers; None when not recorded."""
    return resources.get("n_hosts")


def _host_worker_layout(resources, hostname):
    """``(n_workers, n_worker_slots)`` of the host ``hostname``: from the
    recorded ``hosts``, else the totals spread evenly over ``n_hosts``; None
    when a value is not known."""
    hosts = resources.get("hosts")
    if hosts:
        host = hosts.get(hostname) or {}
        n_workers, n_slots = host.get("n_workers"), host.get("n_worker_slots")
    else:
        n_hosts = resources.get("n_hosts")
        n_workers, n_slots = resources.get("n_workers"), resources.get("n_worker_slots")
        if not n_hosts or n_workers is None or n_slots is None:
            return None
        n_workers, n_slots = n_workers / n_hosts, n_slots / n_hosts
    if not n_workers or n_slots is None:
        return None
    return n_workers, n_slots


def _shared_cumulative(c, d):
    """A cumulative per-process series rebased to 0 at its first sample, each
    increment divided by that sample's divisor ``d`` (a scalar 1.0 when the
    task owns its worker process; see :func:`_worker_share_divisors`)."""
    inc = np.diff(np.asarray(c, dtype=float))
    if not np.isscalar(d):
        inc = inc / np.asarray(d, dtype=float)[1:]
    return np.concatenate(([0.0], np.cumsum(inc)))


def _worker_share_divisors(df, compute_resources):
    """Per-task arrays (aligned with each task's ``time_seconds`` samples) of
    the estimated number of concurrent tasks sharing the task's worker process.

    The monitor samples the WHOLE worker process, so when a worker runs
    several task threads at once every concurrent task in that worker reports
    the same process-wide series: summing over tasks overcounts, and a single
    task's series is not its own. Dividing each sample by this factor makes
    per-task values approximately the task's share and makes cluster-wide
    sums count each process once. With ``(w, s)`` the workers and worker
    slots of the task's host (:func:`_host_worker_layout`) and ``k(t)`` the
    tasks running on that host at wall-clock instant t, the factor assumes
    the scheduler spreads tasks evenly over a host's workers:

        share(t) = clip(k(t) / w, 1, s / w)

    This is exact for W workers x 1 thread (share = 1) and 1 worker x T
    threads (share = k).

    Returns None when no correction is needed (every host has one slot per
    worker) or none is possible (layout of a host unknown, no hostname or no
    ``start_unixtime``), printing a warning when the threads per worker
    exceed 1.
    """
    n_workers = compute_resources.get("n_workers")
    n_slots = compute_resources.get("n_worker_slots")
    threads = n_slots / n_workers if n_workers and n_slots else None
    layouts = {}
    if "hostname" in df.columns:
        layouts = {
            host: _host_worker_layout(compute_resources, host)
            for host in df["hostname"].unique()
        }
    if not layouts or any(layout is None for layout in layouts.values()):
        if threads is not None and threads > 1:
            print(
                f"WARNING: {threads:g} task threads share each worker process, "
                "but the compute resources give no layout of the workers on "
                "every host: the sums count each worker process once per "
                "concurrent task (overcounted)."
            )
        return None
    if all(s / w <= 1 for w, s in layouts.values()):
        return None
    if "start_unixtime" not in df.columns or df["start_unixtime"].isna().any():
        print(
            "WARNING: task threads share each worker process but not every "
            "task has a start_unixtime to estimate its share: the series are "
            "the raw, process-wide (overcounted) ones."
        )
        return None
    starts = df["start_unixtime"].to_numpy(dtype=float)
    ends = starts + np.array([t[-1] for t in df["time_seconds"]], dtype=float)
    hosts = df["hostname"].to_numpy()
    divisors = [None] * len(df)
    for host, (w, s) in layouts.items():
        idx = np.flatnonzero(hosts == host)
        s_sorted = np.sort(starts[idx])
        e_sorted = np.sort(ends[idx])
        for i in idx:
            t_abs = starts[i] + np.asarray(df["time_seconds"].iloc[i], dtype=float)
            k = np.searchsorted(s_sorted, t_abs, side="right") - np.searchsorted(
                e_sorted, t_abs, side="left"
            )
            divisors[i] = np.clip(k / w, 1.0, max(1.0, s / w))
    return divisors


def _binned_maximum(df, edges, column, scale, divisors):
    """The maximum of a task series in each bin of ``edges`` (run clock, 0 at
    the first task start) over the samples in the bin and the values
    interpolated at the edges a task spans (for bins shorter than the sample
    interval); NaN in bins without a task. Each sample is multiplied by
    ``scale`` and divided by its divisor (:func:`_worker_share_divisors`)."""
    starts = df["start_unixtime"].to_numpy(dtype=float)
    run_t0 = starts.min()
    n_bins = len(edges) - 1
    maximum = np.full(n_bins, np.nan)
    for i in range(len(df)):
        t_rel = np.asarray(df["time_seconds"].iloc[i], dtype=float)
        offset = starts[i] - run_t0
        d = divisors[i] if divisors is not None else 1.0
        v = np.asarray(df[column].iloc[i], dtype=float) * scale / d
        k = np.searchsorted(edges, offset + t_rel, side="right") - 1
        ok = (k >= 0) & (k < n_bins)
        np.fmax.at(maximum, k[ok], v[ok])
        at_edges = np.flatnonzero((edges >= offset) & (edges <= offset + t_rel[-1]))
        v_edges = np.interp(edges[at_edges] - offset, t_rel, v)
        for kk in (at_edges, at_edges - 1):
            ok = (kk >= 0) & (kk < n_bins)
            np.fmax.at(maximum, kk[ok], v_edges[ok])
    return maximum


def _performance_figure(figsize, n_rows=1, n_columns=1, **subplot_kwargs):
    """A figure of the performance look and its axes, without pyplot:
    returns ``(Figure, axes)`` from ``Figure.subplots``."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=figsize)
    return figure, figure.subplots(n_rows, n_columns, **subplot_kwargs)


def _wall_clock_figure(n_main_panels, figsize, height_ratios):
    """A figure over the run's wall clock: ``n_main_panels`` panels and below
    them the running-tasks strip, sharing x, gridded. Returns ``(Figure, main
    axes, strip axis)``."""
    figure, axes = _performance_figure(
        figsize, n_main_panels + 1, 1, sharex=True, height_ratios=height_ratios
    )
    for ax in axes:
        ax.grid(True, color="lightgray", alpha=0.5)
    axes[-1].set_xlabel("Wall-clock time since first task start (s)")
    return figure, list(axes[:-1]), axes[-1]


def _figure_save_paths(save_path, keys):
    """``{key: PNG path}`` of the figures ``keys`` for ``save_path`` (``{}``
    for None): the field ``{figure}`` in it is replaced by the key. Raises
    ValueError for several figures and no field, so a call can refuse before
    it draws."""
    if save_path is None:
        return {}
    keys = list(keys)
    if len(keys) > 1 and "{figure}" not in save_path:
        raise ValueError(
            f"save_path {save_path!r} needs the field {{figure}}, replaced by "
            f"each of the figures {keys}."
        )
    return {key: save_path.replace("{figure}", key) for key in keys}


def _save_figures(figures, save_path):
    """Save the figures ``{key: Figure or None}`` as PNG (dpi 150, tight box)
    at the paths of :func:`_figure_save_paths` and print each path."""
    for key, path in _figure_save_paths(save_path, figures).items():
        if figures[key] is not None:
            figures[key].savefig(path, dpi=150, bbox_inches="tight")
            print("wrote", path)


def _add_notes(figure, notes):
    """Write ``notes`` (a str or a sequence of str; empty ones skipped) along
    the bottom of ``figure``."""
    if isinstance(notes, str):
        notes = [notes]
    notes = [note for note in (notes or ()) if note]
    if notes:
        figure.text(
            0.5,
            0.005,
            "\n".join(notes),
            ha="center",
            va="bottom",
            fontsize=9,
            color="dimgray",
        )


def _number_text(value):
    """A tick number: an integer without decimals."""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _parallelism_ticks(ax, x, parallelism, x_label):
    """Two-line x ticks at the unique ``x``: the value over the worker slots p
    of the runs there (``/``-joined when they differ), and the label
    ``x_label + " (total parallel workers)"``. No change when no run has a
    p."""
    slots = {}
    for value, p in zip(x, parallelism, strict=True):
        if p is not None and np.isfinite(p):
            slots.setdefault(value, set()).add(int(p))
    if not slots:
        return
    ticks = sorted(slots)
    ax.set_xticks(
        ticks,
        [
            f"{_number_text(v)}\n{'/'.join(str(p) for p in sorted(slots[v]))}"
            for v in ticks
        ],
    )
    ax.set_xlabel(x_label + " (total parallel workers)")


def _value_ticks(ax, x, x_label):
    """x ticks at exactly the values ``x``, comma-formatted, and the label."""
    ax.set_xticks(list(x), [f"{v:,}" for v in x])
    ax.set_xlabel(x_label)


def _node_units_axis(ax, parallelism, n_nodes):
    """A right-hand axis of a worker-unit speedup in node units (worker units
    over the worker slots per node), added only when the runs share one
    number of worker slots per node; returns it, or None."""
    per_node = np.unique(
        np.round(np.asarray(parallelism, dtype=float) / np.asarray(n_nodes, float), 6)
    )
    if len(per_node) != 1:
        return None
    slots = float(per_node[0])
    secondary = ax.secondary_yaxis(
        "right", functions=(lambda v: v / slots, lambda v: v * slots)
    )
    secondary.set_ylabel(f"Speedup (node units, {slots:g} workers/node)")
    return secondary


def _band_or_auto(values, band, inside=0.95):
    """``band`` if at least ``inside`` of the finite ``values`` fall in it, else
    None (autoscale): a fixed band of one data set leaves others blank."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not band or not len(v):
        return band
    return band if np.mean((v >= band[0]) & (v <= band[1])) >= inside else None


def _band_or_minimum_span(ax, values, band):
    """y-limits of a binned median-time panel: ``band`` when the values fit it
    (:func:`_band_or_auto`), else a range around the median at least as wide,
    relative to it, as the 35-55 s band is around 45 s (+-22 %), so a 1 %
    wobble looks like 1 % in every figure."""
    band = _band_or_auto(values, band)
    if band:
        ax.set_ylim(*band)
        return
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v):
        c = float(np.median(v))
        half = max(0.22 * abs(c), 0.55 * float(v.max() - v.min()))
        ax.set_ylim(c - half, c + half)


def _placed_on_wall_clock(df, caller):
    """Whether the tasks' series can be placed on the run's wall clock; prints
    why not."""
    if "cpu_percent" not in df.columns:
        print(
            f"{caller}: no resource series recorded "
            "(monitoring off or psutil missing); skipping."
        )
        return False
    if "start_unixtime" not in df.columns or df["start_unixtime"].isna().all():
        print(
            f"{caller}: no start_unixtime column (recorded "
            "with an older graphviper); tasks cannot be placed on the run's "
            "wall clock -- use plot_task_resource_usage instead."
        )
        return False
    n_missing = int(df["start_unixtime"].isna().sum())
    if n_missing:
        print(
            f"{caller}: {n_missing} of {len(df)} tasks have no start_unixtime "
            "and cannot be placed on the run's wall clock; skipping."
        )
        return False
    return True


def _binned_task_series(df, slot=None, bin_seconds=None, divisors=None):
    """Bin sums of the tasks' cumulative series on the run's wall clock, on
    the bins of :func:`_display_bins`.

    Returns ``(binw, edges, sums)``. ``sums[key]`` has one row per slot
    (task i adds to row ``slot[i]``; all tasks to row 0 when ``slot`` is
    None) and holds the task-seconds (``running``), CPU seconds (``cpu``),
    the time integral of RSS in GiB s (``rss``) and, when recorded, the GiB
    read and written (``read``, ``write``) in each bin. With ``divisors``
    (:func:`_worker_share_divisors`) every CPU, RSS and I/O sample counts as
    the task's share of its worker process.
    """
    interval = float(df["sample_interval_seconds"].iloc[0])
    starts = df["start_unixtime"].to_numpy(dtype=float)
    rel_end = np.array([t[-1] for t in df["time_seconds"]], dtype=float)
    run_t0 = starts.min()
    span = (starts + rel_end).max() - run_t0
    period = float(np.median([t[-1] - t[0] for t in df["time_seconds"]]))
    binw, edges = _display_bins(span, interval, period, bin_seconds)
    slot = np.zeros(len(df), dtype=int) if slot is None else np.asarray(slot)
    io = {"read": "read_chars", "write": "write_chars"}
    if "read_chars" not in df.columns or df["read_chars"].isna().all():
        io = {}
    sums = {
        key: np.zeros((slot.max() + 1, len(edges) - 1))
        for key in ("running", "cpu", "rss", *io)
    }
    for i, row in df.iterrows():
        t_rel = np.asarray(row["time_seconds"], dtype=float)
        offset = starts[i] - run_t0
        s = slot[i]
        d = 1.0 if divisors is None else divisors[i]
        sums["running"][s] += _bin_increments(edges, offset, t_rel, t_rel - t_rel[0])
        cpu = _cpu_seconds(row)
        cpu = cpu - cpu[0] if divisors is None else _shared_cumulative(cpu, d)
        sums["cpu"][s] += _bin_increments(edges, offset, t_rel, cpu)
        rss = np.asarray(row["memory_rss_bytes"], dtype=float) / 2**30 / d
        sums["rss"][s] += _bin_increments(
            edges, offset, t_rel, _time_integral(t_rel, rss)
        )
        for key, col in io.items():
            c = np.asarray(row[col], dtype=float)
            c = c - c[0] if divisors is None else _shared_cumulative(c, d)
            sums[key][s] += _bin_increments(edges, offset, t_rel, c / 2**30)
    return binw, edges, sums


def _plot_task_series(
    df,
    column,
    ylabel,
    title,
    scale=1.0,
    ax=None,
    color="tab:blue",
    label_prefix="",
    max_task_lines=200,
    subtract_first=False,
):
    """One task-aligned series: a subsample of tasks as faint lines plus the
    average over the tasks.

    ``subtract_first`` rebases each task's series to its first sample -- needed
    for the I/O counters, which are cumulative over the WORKER PROCESS lifetime
    (a task run late in a sweep starts far into its worker's counter). Such
    cumulative series are averaged over ALL tasks at every time, a finished
    task holding its total; instantaneous series (CPU%, RSS) are averaged over
    the tasks still running, drawn only while at least 5 % of the tasks run.
    ``cpu_percent`` is derived from the cumulative CPU seconds
    (:func:`_cpu_seconds`), so the monitor's first and last samples over
    near-zero intervals do not spike.

    The y-axis is clipped to the 99.5th percentile of the plotted values (or
    the average's peak if larger).
    """
    import matplotlib.pyplot as plt

    if column not in df or df[column].isna().all():
        print(
            f"No {column!r} series recorded (monitoring off, psutil missing, "
            "or counter unavailable on this platform)."
        )
        return None
    created_ax = ax is None
    if created_ax:
        _, ax = plt.subplots(figsize=(10, 6))
    interval = float(df["sample_interval_seconds"].iloc[0])
    grid = np.arange(0.0, max(t[-1] for t in df["time_seconds"]) + interval, interval)
    total, count = np.zeros_like(grid), np.zeros_like(grid)
    show = set(range(0, len(df), max(1, len(df) // max_task_lines)))
    shown_values = []
    first = True
    for i, row in df.iterrows():
        t = np.asarray(row["time_seconds"], dtype=float)
        if column == "cpu_percent":
            t, v = _cpu_percent_series(t, _cpu_seconds(row), interval)
            v = v * scale
        else:
            v = np.asarray(row[column], dtype=float) * scale
        if subtract_first:
            v = v - v[0]
        if i in show:
            ax.plot(
                t,
                v,
                color=color,
                alpha=0.15,
                linewidth=0.8,
                label=(f"{label_prefix}node tasks (subsample)" if first else None),
            )
            shown_values.append(v)
            first = False
        if subtract_first:
            total += np.interp(grid, t, v)
            count += 1
        else:
            mask = grid <= t[-1]
            total[mask] += np.interp(grid[mask], t, v)
            count[mask] += 1
    mean = np.divide(
        total,
        count,
        out=np.full_like(total, np.nan),
        where=count >= (1 if subtract_first else 0.05 * len(df)),
    )
    ax.plot(
        grid,
        mean,
        color=color,
        linewidth=2.5,
        label=(
            f"{label_prefix}mean over all {len(df)} tasks "
            "(finished tasks hold their total)"
            if subtract_first
            else f"{label_prefix}mean over the running tasks (of {len(df)}; "
            "drawn while >= 5 % run)"
        ),
    )
    top = 1.1 * float(
        max(np.nanpercentile(np.concatenate(shown_values), 99.5), np.nanmax(mean))
    )
    if not created_ax:  # shared axes (read+write): never shrink the other series
        top = max(top, float(ax.get_ylim()[1]))
    ax.set_ylim(bottom=0.0, top=top)
    ax.set_xlabel("Time since task start (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="lightgray", alpha=0.5)
    ax.legend()
    return ax


def plot_task_resource_usage(source, max_task_lines=200, save_prefix=None):
    """Task-aligned CPU / memory / I/O plots (one figure each).

    Every task's clock starts at its own beginning. The bold CPU and memory
    curves are the average over the tasks still running at each instant, so
    they are not diluted by tasks that already finished; the bold I/O curves
    average every task, a finished task holding its total, so they end at the
    mean per-task volume.

    Parameters
    ----------
    source : dict or pandas.DataFrame
        The application return dict, or its ``timing_node_tasks`` frame.
    max_task_lines : int, optional
        How many individual task curves to draw behind the average.
    save_prefix : str, optional
        If set, save ``<prefix>task_cpu/memory/io.png``.

    Returns
    -------
    dict
        ``{"cpu": ax, "memory": ax, "io": ax}`` (values are None for series
        that were not recorded).
    """

    df = _as_timing_frame(source)
    axes = {}

    axes["cpu"] = _plot_task_series(
        df,
        "cpu_percent",
        "CPU (%)",
        "Node-task CPU usage (per worker process, all threads)",
        max_task_lines=max_task_lines,
    )
    axes["memory"] = _plot_task_series(
        df,
        "memory_rss_bytes",
        "Resident memory (GiB)",
        "Node-task memory usage",
        scale=1 / 2**30,
        max_task_lines=max_task_lines,
    )
    # I/O counters are cumulative over the worker process -> rebase per task.
    io_title = "Node-task I/O (syscall-level, counts network filesystems)"
    axes["io"] = _plot_task_series(
        df,
        "read_chars",
        "I/O since task start (GiB)",
        io_title,
        scale=1 / 2**30,
        color="tab:blue",
        label_prefix="read: ",
        max_task_lines=max_task_lines,
        subtract_first=True,
    )
    if axes["io"] is not None:
        _plot_task_series(
            df,
            "write_chars",
            "I/O since task start (GiB)",
            io_title,
            scale=1 / 2**30,
            ax=axes["io"],
            color="tab:orange",
            label_prefix="write: ",
            max_task_lines=max_task_lines,
            subtract_first=True,
        )

    if save_prefix is not None:
        for name, ax in axes.items():
            if ax is not None:
                ax.figure.savefig(f"{save_prefix}task_{name}.png")
    return axes


def plot_cluster_resource_usage(
    source, core_capacity=None, memory_capacity_gib=None, save_prefix=None
):
    """Cluster-wide CPU / memory / I/O over the RUN's wall clock.

    Each task's series is placed at its actual wall-clock position using
    ``start_unixtime`` and the quantities are SUMMED over the tasks: busy
    cores, total RSS of task-running worker processes (idle workers are not
    counted), and the aggregate read/write rate (GiB/s). Every curve is a
    mean over bins of about 1/600 of the run (a whole number of sample
    intervals; a line at the 1 s sample interval over a long run is a solid
    band of second-long bursts), taken from a cumulative series (CPU
    seconds, bytes, the time integral of RSS), so it conserves that
    quantity; once a bin would exceed a quarter of the task period it spans
    whole task periods (:func:`_display_bins`). Every figure also shows the
    mean number of running tasks in a strip below.

    Parameters
    ----------
    source : dict or pandas.DataFrame
        The application return dict, or its ``timing_node_tasks`` frame.
        Needs the ``start_unixtime`` column (graphviper monitor >= 2026-07-13);
        cross-node placement relies on NTP-synced node clocks.
    core_capacity : int, optional
        Total cores of the allocation; drawn as a dashed capacity line.
    memory_capacity_gib : float, optional
        Total memory of the allocation (GiB); drawn as a dashed capacity line.
    save_prefix : str, optional
        If set, save ``<prefix>cluster_cpu/memory/io.png``.

    Returns
    -------
    dict
        ``{"cpu": ax, "memory": ax, "io": ax}`` (``io`` is None when the I/O
        counters were not recorded), or None when ``start_unixtime`` is absent.
    """
    import matplotlib.pyplot as plt

    df = _as_timing_frame(source)
    if not _placed_on_wall_clock(df, "plot_cluster_resource_usage"):
        return None

    binw, edges, sums = _binned_task_series(df)
    grid = 0.5 * (edges[:-1] + edges[1:])
    acc = {key: v[0] / np.diff(edges) for key, v in sums.items()}
    have_io = "read" in acc
    n_running = acc.pop("running")
    bin_note = f" ({binw:g} s bin means)"

    def _figure(title, ylabel, curves, capacity=None):
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(11, 7), sharex=True, height_ratios=(4, 1)
        )
        for label, y, color in curves:
            ax.plot(grid, y, color=color, linewidth=1.5, label=label)
        top = max(np.nanmax(y) for _, y, _ in curves)
        if capacity is not None:
            ax.axhline(capacity[1], color="gray", linestyle="--", label=capacity[0])
            top = max(top, capacity[1])
        ax.set_ylim(0.0, 1.25 * top)  # headroom for the legend
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=2)
        axr.plot(grid, n_running, color="dimgray", linewidth=1)
        axr.set_ylim(0.0, 1.1 * n_running.max())
        axr.set_ylabel("Running tasks")
        axr.set_xlabel("Wall-clock time since first task start (s)")
        for a in (ax, axr):
            a.grid(True, color="lightgray", alpha=0.5)
        fig.tight_layout()
        return ax

    axes = {}
    axes["cpu"] = _figure(
        "Cluster CPU usage over the run",
        "Busy cores" + bin_note,
        [("busy cores", acc["cpu"], "tab:blue")],
        capacity=(
            None
            if core_capacity is None
            else (f"capacity ({core_capacity} cores)", core_capacity)
        ),
    )
    axes["memory"] = _figure(
        "Cluster memory usage over the run (task-running workers only)",
        "Total resident memory (GiB)" + bin_note,
        [("total RSS", acc["rss"], "tab:blue")],
        capacity=(
            None
            if memory_capacity_gib is None
            else (f"capacity ({memory_capacity_gib:g} GiB)", memory_capacity_gib)
        ),
    )
    if have_io:
        axes["io"] = _figure(
            "Cluster I/O rate over the run (syscall-level, counts network filesystems)",
            "Aggregate I/O rate (GiB/s)" + bin_note,
            [
                ("write", acc["write"], "tab:orange"),
                ("read", acc["read"], "tab:blue"),
            ],
        )
    else:
        axes["io"] = None
        print("(no read_chars/write_chars series; skipping the cluster I/O plot)")

    if save_prefix is not None:
        for name, ax in axes.items():
            if ax is not None:
                ax.figure.savefig(f"{save_prefix}cluster_{name}.png")
    return axes


# Statistics over the worker slots of plot_core_resource_usage: key, legend
# label, line style, line width.
_SLOT_STATISTICS = (
    ("peak", "peak (max over slots)", "--", 1.0),
    ("mean", "mean", "-", 2.0),
    ("median", "median", "-.", 1.2),
    ("min", "min", ":", 1.5),
)


def _over_slots(v):
    """Peak, mean, median and minimum over the rows (worker slots) of ``v``
    in each column, ignoring NaN; NaN in a column without values."""
    ok = ~np.isnan(v).all(axis=0)
    stats = {}
    for key, f in (
        ("peak", np.nanmax),
        ("mean", np.nanmean),
        ("median", np.nanmedian),
        ("min", np.nanmin),
    ):
        stats[key] = np.full(v.shape[1], np.nan)
        stats[key][ok] = f(v[:, ok], axis=0)
    return stats


def plot_core_resource_usage(
    source, memory_per_slot_gib=None, bin_seconds=None, save_prefix=None
):
    """Per-core CPU / memory / I/O over the RUN's wall clock: the peak, mean,
    median and minimum over the worker slots in each bin.

    A worker slot runs one task at a time: a thread of a Dask worker, whose
    tasks are packed into slots within the worker (a worker keeps its name
    when its nanny restarts the process), else the recorded (host, process,
    thread), else interval packing within each host
    (:func:`assign_task_stream_lanes`). With single-threaded workers, one
    per core, a slot is a core; the figures say worker slot because a slot
    whose tasks run several threads uses several cores. Each slot's CPU
    (all threads of its worker process, so OpenMP threads can take it above
    100 %), resident memory and read/write rates are bin means, by default on
    the bins of :func:`plot_cluster_resource_usage`; for CPU and I/O the mean
    over the slots times the number of slots is the cluster total. A slot
    between tasks counts as 0 % CPU and no I/O. Its memory is not monitored
    then, so the memory statistics are over the slots that ran a task in the
    bin, each averaged over its running time. Every figure also shows the
    mean number of running tasks in a strip below.

    Parameters
    ----------
    source : dict or pandas.DataFrame
        The application return dict, or its ``timing_node_tasks`` frame.
        Needs the ``start_unixtime`` column of every task.
    memory_per_slot_gib : float, optional
        Memory available to one worker slot (GiB), e.g. the node's memory
        over its worker slots; drawn as a dashed line.
    bin_seconds : float, optional
        Bin width, rounded to whole sample intervals and at most half the
        run; by default the bins of :func:`plot_cluster_resource_usage`. A
        slot's I/O comes in bursts (each task's reads and writes): in bins of
        at least a task period the peak and minimum differ from the mean by
        about one burst over the bin width; in narrower bins the minimum
        drops towards 0 and the peak towards the rate of a single burst.
    save_prefix : str, optional
        If set, save ``<prefix>core_cpu/memory/io.png``.

    Returns
    -------
    dict
        ``{"cpu": ax, "memory": ax, "io": ax}`` (``io`` is the write panel,
        with the read panel below it, or None when the I/O counters were not
        recorded), or None when the tasks cannot be placed on the wall clock.
    """
    import matplotlib.pyplot as plt

    df = _as_timing_frame(source)
    if not _placed_on_wall_clock(df, "plot_core_resource_usage"):
        return None

    identity = ["hostname", "process_pid", "thread_native_id", "start_unixtime"]
    lanes = df[[c for c in identity if c in df.columns]].assign(
        T_image_cube_task=[t[-1] for t in df["time_seconds"]]
    )
    if "hostname" not in lanes.columns:
        lanes["hostname"] = ""
    if "worker_name" in df.columns and df["worker_name"].notna().all():
        lanes["hostname"] = (
            lanes["hostname"].astype(str) + "/" + df["worker_name"].astype(str)
        )
        lanes = lanes.drop(columns=["process_pid", "thread_native_id"], errors="ignore")
    lanes, _, n_slots = assign_task_stream_lanes(lanes)
    binw, edges, sums = _binned_task_series(df, lanes["row"].to_numpy(), bin_seconds)
    grid = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    running = sums["running"]
    n_running = running.sum(axis=0) / width
    bin_note = f" ({binw:g} s bin means)"

    def _figure(title, panels):
        fig, axs = plt.subplots(
            len(panels) + 1,
            1,
            figsize=(11, 4 + 3 * len(panels)),
            sharex=True,
            height_ratios=(*[4] * len(panels), 1),
        )
        for ax, (v, color, ylabel, reference) in zip(axs, panels, strict=False):
            stats = _over_slots(v)
            ax.fill_between(
                grid, stats["min"], stats["peak"], color=color, alpha=0.15, lw=0
            )
            for key, label, style, lw in _SLOT_STATISTICS:
                ax.plot(grid, stats[key], color=color, ls=style, lw=lw, label=label)
            top = np.nanmax(stats["peak"])
            if reference is not None:
                ax.axhline(reference[1], color="gray", ls="--", label=reference[0])
                top = max(top, reference[1])
            ax.set_ylim(0.0, 1.25 * top)  # headroom for the legend
            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right", ncol=3)
        axs[0].set_title(title)
        axr = axs[-1]
        axr.plot(grid, n_running, color="dimgray", linewidth=1)
        axr.set_ylim(0.0, 1.1 * n_slots)
        axr.set_ylabel("Running tasks")
        axr.set_xlabel("Wall-clock time since first task start (s)")
        for a in axs:
            a.grid(True, color="lightgray", alpha=0.5)
        fig.tight_layout()
        return axs[0]

    axes = {}
    axes["cpu"] = _figure(
        f"CPU usage per worker slot over the run ({n_slots} slots)",
        [
            (
                100.0 * sums["cpu"] / width,
                "tab:blue",
                "CPU per worker slot (%)" + bin_note,
                ("one core (100 %)", 100.0),
            )
        ],
    )
    rss = np.divide(
        sums["rss"],
        running,
        out=np.full_like(running, np.nan),
        where=running > 1e-6 * width,
    )
    axes["memory"] = _figure(
        f"Memory usage per worker slot over the run ({n_slots} slots; "
        "slots running a task)",
        [
            (
                rss,
                "tab:blue",
                "Resident memory per worker slot (GiB)" + bin_note,
                (
                    None
                    if memory_per_slot_gib is None
                    else (
                        f"memory per slot ({memory_per_slot_gib:g} GiB)",
                        memory_per_slot_gib,
                    )
                ),
            )
        ],
    )
    if "read" in sums:
        axes["io"] = _figure(
            f"I/O rate per worker slot over the run ({n_slots} slots; "
            "syscall-level, counts network filesystems)",
            [
                (
                    1024.0 * sums[key] / width,
                    color,
                    f"{key.capitalize()} rate per worker slot (MiB/s)" + bin_note,
                    None,
                )
                for key, color in (("write", "tab:orange"), ("read", "tab:blue"))
            ],
        )
    else:
        axes["io"] = None
        print("(no read_chars/write_chars series; skipping the per-slot I/O plot)")

    if save_prefix is not None:
        for name, ax in axes.items():
            if ax is not None:
                ax.figure.savefig(f"{save_prefix}core_{name}.png")
    return axes


def assign_task_stream_lanes(tasks):
    """Assign each task to a per-host worker lane; returns
    ``(frame, hosts, n_lanes)`` where the frame gains ``start``/``end``
    (seconds since the first task) and ``host_idx``/``lane``/``row`` columns
    (row = global y position in the stream).

    When the frame carries the recorded execution identity
    (``process_pid`` + ``thread_native_id``, written by astroviper's node task
    and reduce since 2026-07-28), a lane IS one (process, thread) slot --
    exact, and reduce rows recorded with the same identity land on the lane
    they truly ran on. Older frames without identity fall back to first-fit
    interval packing within each host (valid because every worker slot runs
    one task at a time).
    """
    t = tasks.copy()
    if "task_failed_phase" in t.columns and t["task_failed_phase"].notna().any():
        print(
            f"note: {t['task_failed_phase'].notna().sum()} failed tasks included "
            "in the stream (their spans are real wall time)"
        )
    t0 = t["start_unixtime"].min()
    t["start"] = t["start_unixtime"] - t0
    t["end"] = t["start"] + t["T_image_cube_task"]
    hosts = sorted(t["hostname"].unique())
    t["host_idx"] = t["hostname"].map(hosts.index)
    t["lane"] = -1
    lane_offset, offsets = {}, 0

    has_identity = (
        {"process_pid", "thread_native_id"} <= set(t.columns)
        and t["process_pid"].notna().all()
        and t["thread_native_id"].notna().all()
    )
    for h in hosts:
        idx = t.index[t["hostname"] == h]
        if has_identity:
            # True lanes: one per (process, thread), ordered by first activity.
            sub = t.loc[idx]
            first = (
                sub.groupby(["process_pid", "thread_native_id"])["start"]
                .min()
                .sort_values()
            )
            lane_of = {key: ln for ln, key in enumerate(first.index)}
            t.loc[idx, "lane"] = [
                lane_of[(p, th)]
                for p, th in zip(
                    sub["process_pid"], sub["thread_native_id"], strict=False
                )
            ]
            n_lanes_h = len(lane_of)
        else:
            busy_until = []  # per-lane end time
            order = t.loc[idx].sort_values("start").index
            for i in order:
                s = t.at[i, "start"]
                for ln, bu in enumerate(busy_until):
                    if bu <= s + 1e-6:
                        t.at[i, "lane"] = ln
                        busy_until[ln] = t.at[i, "end"]
                        break
                else:
                    t.at[i, "lane"] = len(busy_until)
                    busy_until.append(t.at[i, "end"])
            n_lanes_h = len(busy_until)
        lane_offset[h] = offsets
        offsets += n_lanes_h
    t["row"] = [
        lane_offset[h] + ln for h, ln in zip(t["hostname"], t["lane"], strict=False)
    ]
    return t, hosts, offsets


def _performance_measures_of_run(
    source, compute_resources, timing_phases, task_columns, fixed_non_io_compute
):
    """The performance measures of one run; see :func:`performance_measures`."""
    tasks = _as_timing_frame(source)
    columns = _task_columns(task_columns)
    resources = _compute_resources(source, compute_resources)
    timing, reduce_tasks = {}, None
    if isinstance(source, dict):
        timing = source.get("timing_distributed_application") or {}
        records = source.get("timing_reduce_nodes")
        if records is not None and len(records):
            reduce_tasks = pd.DataFrame(records)

    def recorded(key):
        value = timing.get(key)
        return np.nan if value is None else float(value)

    durations = tasks[columns["task"]].to_numpy(dtype=float)
    placed = (
        len(tasks) > 0
        and "start_unixtime" in tasks.columns
        and bool(tasks["start_unixtime"].notna().all())
    )
    p, provenance = np.nan, None
    if resources.get("n_worker_slots") is not None or placed:
        identity = ("start_unixtime", "hostname", "process_pid", "thread_native_id")
        lanes = tasks[[c for c in identity if c in tasks.columns]].assign(
            T_image_cube_task=durations
        )
        p, provenance = _n_worker_slots(resources, lanes)

    work_measured = float(tasks[columns["task"]].sum())
    work_map, fixed_compute_shift = work_measured, 0.0
    if fixed_non_io_compute is not None:
        io_columns = [c for c in columns["load"] if c in tasks.columns]
        if columns["write"]:
            io_columns.append(columns["write"])
        work_map = float(tasks[io_columns].sum(axis=1).sum()) + (
            fixed_non_io_compute * len(tasks)
        )
        fixed_compute_shift = (work_measured - work_map) / p
    work_reduce = 0.0
    if reduce_tasks is not None:
        work_reduce = float(
            (reduce_tasks["end_unixtime"] - reduce_tasks["start_unixtime"]).sum()
        )
    work = work_map + work_reduce

    compute_start = recorded("compute_start_unixtime")
    t3 = recorded("compute_end_unixtime")
    t0, t0_reconstructed = recorded("application_start_unixtime"), False
    if np.isnan(t0) and timing_phases is not None and np.isfinite(compute_start):
        pre_steps, _ = _driver_steps(timing_phases)
        step_times = [recorded(key) for _, key in pre_steps]
        t0 = compute_start - sum(v for v in step_times if np.isfinite(v))
        t0_reconstructed = True
    compute_time = recorded("T_compute_dask_graph")
    total_time = recorded("T_total")

    placed_measures = dict.fromkeys(
        (
            "first_task_start_unixtime",
            "map_end_unixtime",
            "task_window",
            "pre_map",
            "post_map",
            "map_busy_fraction",
            "time_to_95_percent_busy",
            "p95_task_end",
            "idle_ramp",
            "idle_in_window",
            "idle_tail",
        ),
        np.nan,
    )
    if placed:
        starts = tasks["start_unixtime"].to_numpy(dtype=float)
        t1 = float(starts.min())
        t2 = float(np.nanmax(starts + durations))
        rel_starts, rel_ends = starts - t1, starts + durations - t1
        task_window = t2 - t1
        event_times, running = _running_task_count(rel_starts, rel_ends)
        reach = running >= 0.95 * p
        to_busy = float(event_times[np.argmax(reach)]) if reach.any() else np.nan
        p95_task_end = float(np.quantile(rel_ends, 0.95))

        def idle(a, b):
            if b <= a:
                return 0.0
            busy = np.clip(
                np.minimum(rel_ends, b) - np.maximum(rel_starts, a), 0.0, None
            )
            return (b - a) - float(np.nansum(busy)) / p

        ramp_end = 0.0 if np.isnan(to_busy) else to_busy
        tail_start = max(p95_task_end, ramp_end)
        if np.isfinite(compute_start):
            pre_map, post_map = t1 - compute_start, t3 - t2
        else:
            pre_map, post_map = 0.0, compute_time - task_window
        placed_measures = {
            "first_task_start_unixtime": t1,
            "map_end_unixtime": t2,
            "task_window": task_window,
            "pre_map": pre_map,
            "post_map": post_map,
            "map_busy_fraction": work_measured / (p * task_window),
            "time_to_95_percent_busy": to_busy,
            "p95_task_end": p95_task_end,
            "idle_ramp": idle(0.0, ramp_end),
            "idle_in_window": idle(ramp_end, tail_start),
            "idle_tail": idle(tail_start, task_window),
        }
    t1 = placed_measures["first_task_start_unixtime"]
    map_reduce_window = t3 - t1 - fixed_compute_shift
    end_to_end_window = t3 - t0 - fixed_compute_shift
    speedup_map_reduce = work / map_reduce_window
    speedup_end_to_end = work / end_to_end_window
    return {
        "n_worker_slots": p,
        "n_worker_slots_reconstructed": provenance == "lane packing",
        "n_map_tasks": len(tasks),
        "n_reduce_tasks": 0 if reduce_tasks is None else len(reduce_tasks),
        "work_map": work_map,
        "work_reduce": work_reduce,
        "work": work,
        "ideal_time": work / p,
        "task_time_median": float(np.nanmedian(durations)) if len(tasks) else np.nan,
        "task_time_max": float(np.nanmax(durations)) if len(tasks) else np.nan,
        "application_start_unixtime": t0,
        "application_start_reconstructed": t0_reconstructed,
        "first_task_start_unixtime": t1,
        "map_end_unixtime": placed_measures["map_end_unixtime"],
        "compute_start_unixtime": compute_start,
        "compute_end_unixtime": t3,
        "fixed_compute_shift": fixed_compute_shift,
        "task_window": placed_measures["task_window"],
        "map_reduce_window": map_reduce_window,
        "end_to_end_window": end_to_end_window,
        "driver_setup_before_first_task": t1 - t0,
        "pre_map": placed_measures["pre_map"],
        "post_map": placed_measures["post_map"],
        "T_compute_dask_graph": compute_time,
        "T_total": total_time,
        "fixed_costs": total_time - compute_time,
        "map_busy_fraction": placed_measures["map_busy_fraction"],
        "time_to_95_percent_busy": placed_measures["time_to_95_percent_busy"],
        "p95_task_end": placed_measures["p95_task_end"],
        "idle_ramp": placed_measures["idle_ramp"],
        "idle_in_window": placed_measures["idle_in_window"],
        "idle_tail": placed_measures["idle_tail"],
        "speedup_map_reduce": speedup_map_reduce,
        "speedup_end_to_end": speedup_end_to_end,
        "parallel_efficiency_map_reduce": speedup_map_reduce / p,
        "parallel_efficiency_end_to_end": speedup_end_to_end / p,
    }


def performance_measures(
    source,
    compute_resources=None,
    scaling="strong",
    timing_phases=None,
    task_columns=None,
    fixed_non_io_compute=None,
):
    """The performance measures of the RADPS benchmarking memo for one or
    several runs of a distributed application.

    The map tasks run from t1 (the first task start) to t2 (the end of the
    last map task); the compute call, with the reduce, ends at t3; the
    application starts at t0. The work W is the summed time of the map and
    reduce tasks, p the number of worker slots (tasks that can run at once),
    and the speedups over the map reduce window ``W / (t3 - t1)`` and the end
    to end window ``W / (t3 - t0)`` in worker units, with the parallel
    efficiencies ``S / p``.

    Parameters
    ----------
    source : dict, pandas.DataFrame or sequence of dict
        The application's return dict (``"timing_node_tasks"``,
        ``"timing_distributed_application"``, ``"timing_reduce_nodes"`` and
        ``"compute_resources"`` when recorded) or its ``timing_node_tasks``
        frame, for one run; a sequence of return dicts for several runs.
    compute_resources : dict or sequence, optional
        Replaces the recorded ``return_dict["compute_resources"]`` (keys of
        :data:`~astroviper.utils.data_partitioning.COMPUTE_RESOURCES_KEYS`,
        e.g. ``{"n_worker_slots": 112}`` for a run that recorded none); for
        several runs a sequence aligned with ``source``, None entries meaning
        the recorded value. p is its ``n_worker_slots``; without one the map
        tasks are packed into lanes (:func:`assign_task_stream_lanes`).
    scaling : str, optional
        ``"strong"`` (default) or ``"weak"``, for the relative measures of
        several runs: strong scaling compares with the first usable run of
        the lowest p, ``S_rel = p_b (t3 - t1)_b / (t3 - t1)``; weak scaling
        with the first usable run, ``S_rel = p_b (W / W_b) (t3 - t1)_b /
        (t3 - t1)`` (the same for the end to end window).
    timing_phases : list of tuple, optional
        The phase layout of ``"timing_distributed_application"``, the
        application's ``DISTRIBUTED_APPLICATION_TIMING_PHASES``. Used only
        when ``application_start_unixtime`` was not recorded: t0 is then the
        compute start minus the recorded steps before ``T_compute_dask_graph``.
    task_columns : dict, optional
        The node task timing columns, as for :func:`plot_task_stream`
        (``None``: the imager's, :data:`IMAGING_TASK_STREAM_COLUMNS`); load
        columns the frame lacks are left out.
    fixed_non_io_compute : float, optional
        Seconds. Every map task is counted as its load and write time plus
        this fixed non-I/O compute, and both windows are shortened by the
        removed excess compute per worker slot, ``fixed_compute_shift``.

    Returns
    -------
    dict or pandas.DataFrame
        For one run a dict (times in seconds, ``*_unixtime`` absolute):
        ``n_worker_slots`` (p), ``n_worker_slots_reconstructed`` (True when
        lane-packed), ``n_map_tasks``, ``n_reduce_tasks``, ``work_map``,
        ``work_reduce``, ``work`` (W), ``ideal_time`` (W / p),
        ``task_time_median``, ``task_time_max``,
        ``application_start_unixtime`` (t0), ``application_start_reconstructed``,
        ``first_task_start_unixtime`` (t1), ``map_end_unixtime`` (t2),
        ``compute_start_unixtime``, ``compute_end_unixtime`` (t3),
        ``fixed_compute_shift``, ``task_window`` (t2 - t1),
        ``map_reduce_window`` (t3 - t1 - shift), ``end_to_end_window``
        (t3 - t0 - shift), ``driver_setup_before_first_task`` (t1 - t0),
        ``pre_map`` (t1 - compute start; 0 without that anchor), ``post_map``
        (t3 - t2; without the compute start, ``T_compute_dask_graph`` minus
        the task window), ``T_compute_dask_graph``, ``T_total``,
        ``fixed_costs`` (``T_total - T_compute_dask_graph``),
        ``map_busy_fraction`` (measured map work / (p (t2 - t1))),
        ``time_to_95_percent_busy`` (from t1 until 95 % of p run; NaN if
        never), ``p95_task_end`` (95th percentile of the map task ends, from
        t1), the idle worker time per worker during the ramp (until 95 %
        busy), inside the window and in the straggler tail (after
        ``p95_task_end``) ``idle_ramp``, ``idle_in_window``, ``idle_tail``
        (with W of the map tasks over p they sum to the task window), and
        ``speedup_map_reduce``, ``speedup_end_to_end``,
        ``parallel_efficiency_map_reduce``,
        ``parallel_efficiency_end_to_end``. A value that cannot be worked out
        is NaN: without ``start_unixtime`` every time placed on the run's
        wall clock, and p when it was not recorded. For several runs a
        DataFrame, one row per run in input order, with the columns
        ``speedup_map_reduce_relative``, ``speedup_end_to_end_relative``,
        ``parallel_efficiency_map_reduce_relative`` and
        ``parallel_efficiency_end_to_end_relative`` added (NaN for a run
        whose map reduce window is not finite).

    Raises
    ------
    ValueError
        For an unknown ``scaling``, a ``compute_resources`` for several runs
        that is not a sequence aligned with ``source``, or a
        ``timing_phases`` without a ``T_compute_dask_graph`` step.
    """
    if scaling not in ("strong", "weak"):
        raise ValueError(f"scaling must be 'strong' or 'weak'; got {scaling!r}.")
    if isinstance(source, (dict, pd.DataFrame)):
        return _performance_measures_of_run(
            source, compute_resources, timing_phases, task_columns, fixed_non_io_compute
        )
    sources = list(source)
    if compute_resources is None:
        overrides = [None] * len(sources)
    elif isinstance(compute_resources, dict) or len(compute_resources) != len(sources):
        raise ValueError(
            "For several runs compute_resources is a sequence aligned with "
            "source (None entries: the recorded compute resources)."
        )
    else:
        overrides = list(compute_resources)
    measures = pd.DataFrame(
        [
            _performance_measures_of_run(
                run, override, timing_phases, task_columns, fixed_non_io_compute
            )
            for run, override in zip(sources, overrides, strict=True)
        ]
    )
    usable = np.isfinite(measures["map_reduce_window"].to_numpy(dtype=float))
    for window in ("map_reduce", "end_to_end"):
        measures[f"speedup_{window}_relative"] = np.nan
    if usable.any():
        if scaling == "strong":
            base = measures.loc[usable, "n_worker_slots"].astype(float).idxmin()
        else:
            base = measures.index[usable][0]
        ratio = measures["work"] / measures.at[base, "work"] if scaling == "weak" else 1
        for window in ("map_reduce", "end_to_end"):
            relative = (
                measures.at[base, "n_worker_slots"]
                * ratio
                * measures.at[base, f"{window}_window"]
                / measures[f"{window}_window"]
            )
            measures[f"speedup_{window}_relative"] = relative.where(usable)
    for window in ("map_reduce", "end_to_end"):
        measures[f"parallel_efficiency_{window}_relative"] = (
            measures[f"speedup_{window}_relative"] / measures["n_worker_slots"]
        )
    return measures


def plot_application_task_stream(source, title=None, save_path=None):
    """Plot every distributed graph in one vertically stacked task stream.

    Unlike :func:`plot_task_stream`, which deliberately retains the historical
    final-graph view, this diagnostic consumes ``source["timing_graphs"]`` and
    gives each graph its own panel.  Panels are independently time-scaled so
    short weighting graphs remain readable beside longer imaging-cycle graphs.

    Parameters
    ----------
    source : dict
        Continuum application result containing ``timing_graphs``. Each record
        supplies ``stage``, structured ``stage_kind``,
        a ``timing_node_tasks`` or ``timing_finalization`` frame, and
        compute-window anchors.
    title : str, optional
        Overall figure title.
    save_path : str, optional
        PNG path. The figure is returned whether or not it is saved.

    Returns
    -------
    matplotlib.figure.Figure or None
        Stacked task-stream figure, or ``None`` when no graph records can be
        plotted.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not isinstance(source, dict):
        raise TypeError(
            "plot_application_task_stream expects an application result dict"
        )
    graph_records = source.get("timing_graphs")
    if not graph_records:
        print("plot_application_task_stream: no timing_graphs records; skipping.")
        return None

    prepared = []
    required = {"hostname", "start_unixtime", "T_image_cube_task"}
    for record in graph_records:
        tasks = record.get("timing_node_tasks")
        if tasks is None:
            tasks = record.get("timing_finalization")
        if not isinstance(tasks, pd.DataFrame) or tasks.empty:
            continue
        if not required.issubset(tasks.columns):
            print(
                "plot_application_task_stream: graph "
                f"{record.get('stage', '<unnamed>')!r} lacks "
                f"{sorted(required - set(tasks.columns))}; skipping."
            )
            continue
        tasks = tasks.copy()
        for column in ("T_make_empty_image", "T_load", "T_write"):
            if column not in tasks:
                tasks[column] = 0.0
        stream, hosts, n_lanes = assign_task_stream_lanes(tasks)
        prepared.append((record, stream, hosts, n_lanes))

    if not prepared:
        return None

    heights = [max(2.2, 0.42 * n_lanes + 1.2) for _, _, _, n_lanes in prepared]
    fig, axes = plt.subplots(
        len(prepared),
        1,
        figsize=(16, sum(heights)),
        squeeze=False,
        gridspec_kw={"height_ratios": heights, "hspace": 0.55},
    )
    axes = axes[:, 0]

    for graph_index, (record, stream, _hosts, n_lanes) in enumerate(prepared, start=1):
        ax = axes[graph_index - 1]
        first_task = float(stream["start_unixtime"].min())
        compute_start = record.get("compute_start_unixtime")
        compute_end = record.get("compute_end_unixtime")
        offset = max(0.0, first_task - float(compute_start or first_task))
        stream = stream.copy()
        stream["start"] += offset
        stream["end"] += offset
        task_end = float(stream["end"].max())
        graph_duration = record.get("T_compute")
        if (
            graph_duration is None
            and compute_start is not None
            and compute_end is not None
        ):
            graph_duration = float(compute_end) - float(compute_start)
        graph_duration = max(float(graph_duration or task_end), task_end)

        if offset > 0:
            ax.axvspan(0, offset, color=_PRE_COLOR, alpha=0.18, lw=0)
        if graph_duration > task_end:
            ax.axvspan(task_end, graph_duration, color=_POST_COLOR, alpha=0.20, lw=0)

        for _, row in stream.iterrows():
            y = row["row"] - 0.42
            load_end = row["start"] + row["T_make_empty_image"] + row["T_load"]
            write_start = max(load_end, row["end"] - row["T_write"])
            for start, end, color in (
                (row["start"], load_end, _LOAD_COLOR),
                (load_end, write_start, _SCIENCE_COLOR),
                (write_start, row["end"], _WRITE_COLOR),
            ):
                if end > start:
                    ax.add_patch(
                        plt.Rectangle((start, y), end - start, 0.84, color=color, lw=0)
                    )

        stage = str(record.get("stage", f"graph {graph_index}"))
        stage_kind = record.get("stage_kind")
        if stage_kind == "imaging_cycle":
            post_label = "reduce + model update"
        elif stage_kind == "final" and record.get("includes_restoration", False):
            post_label = "reduce + restoration"
        elif stage_kind == "final":
            post_label = "reduce + finalization"
        else:
            post_label = "reduce / gather"
        if graph_duration - task_end > 0.04 * graph_duration:
            ax.text(
                0.5 * (task_end + graph_duration),
                max(n_lanes - 0.5, 0),
                post_label,
                ha="center",
                va="top",
                fontsize=8,
                color="#8a4a6d",
            )

        bounds = stream.groupby("host_idx")["row"].max().sort_index()
        for bound in bounds.values[:-1]:
            ax.axhline(bound + 0.5, color="#dddddd", lw=0.4)
        ax.set_yticks(range(n_lanes), [f"worker {i + 1}" for i in range(n_lanes)])
        ax.set_ylim(-0.6, n_lanes - 0.4)
        ax.set_xlim(0, graph_duration * 1.005 if graph_duration else 1)
        ax.set_title(
            f"Graph {graph_index}: {stage} — {len(stream)} tasks, "
            f"{graph_duration:.1f} s",
            loc="left",
            fontsize=10,
        )
        ax.set_xlabel("time since this graph's compute started (s)")
        ax.grid(axis="x", color="lightgray", alpha=0.4)

    axes[0].legend(
        handles=[
            Patch(color=_LOAD_COLOR, label="load"),
            Patch(color=_SCIENCE_COLOR, label="science"),
            Patch(color=_WRITE_COLOR, label="write"),
            Patch(color=_PRE_COLOR, alpha=0.3, label="before first map task"),
            Patch(color=_POST_COLOR, alpha=0.3, label="reduce / append / gather"),
        ],
        loc="upper right",
        fontsize=8,
        ncols=5,
    )
    fig.suptitle(title or "Continuum application task streams by graph", fontsize=14)
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("wrote", save_path)
    return fig


def _write_task_stream_html(
    t,
    hosts,
    n_lanes,
    n_worker_slots,
    ev_t,
    running,
    pre_rel,
    post_end_rel,
    makespan,
    ideal,
    T_compute,
    pre_label,
    post_label,
    title,
    html_path,
    driver=None,
    add_time_labels=False,
):
    """Write a standalone interactive HTML task stream: the same panels as
    the matplotlib figure (including the driver strip when ``driver`` -- the
    :func:`_driver_timeline` result -- is given), drawn as inline SVG with a
    JavaScript tooltip that shows a bar's timing details on hover. Pure SVG +
    vanilla JS -- no extra dependencies, works offline. One <rect> per task
    segment, so the file grows with the task count (~a few MB for a 15k-task
    run)."""
    import html as _html

    # ---- geometry (pixels) ----
    left, top = 190, 44
    bottom = 166 if add_time_labels else 72  # room for the t_0..t_3 marks
    inner_w = 1500
    util_h, gap = 110, 16
    driver_h, dgap = (36, 14) if driver is not None else (0, 0)
    top_u = top + driver_h + dgap  # top of the utilization panel
    lane_px = max(3.0, min(14.0, 9000.0 / max(n_lanes, 1)))
    stream_h = lane_px * n_lanes
    width = left + inner_w + 20
    height = top_u + util_h + gap + stream_h + bottom

    x0 = min(pre_rel, 0.0) if pre_rel is not None else 0.0
    x1 = (
        post_end_rel
        if post_end_rel is not None
        else max(T_compute or 0.0, makespan, float(t["end"].max()))
    )
    app_total = None
    if driver is not None:
        d_segs, app_start_rel, app_end_rel, d_colors, d_setup = driver
        app_total = app_end_rel - app_start_rel
        x0 = min(x0, app_start_rel)
        x1 = max(x1, app_end_rel)
    x1 *= 1.005
    sx = inner_w / (x1 - x0)

    def X(sec):
        return left + (sec - x0) * sx

    def Y(row):  # stream panel: row 0 at the bottom, like the matplotlib figure
        return top_u + util_h + gap + stream_h - (row + 1) * lane_px

    out = [
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{_html.escape(title or 'task stream')}</title>",
        """<style>
body { font-family: sans-serif; margin: 12px; }
rect[data-i]:hover { stroke: #000; stroke-width: 0.8px; }
#tt { position: fixed; display: none; background: #fffef5; border: 1px solid
      #888; border-radius: 3px; padding: 5px 8px; font-size: 18px;
      pointer-events: none; box-shadow: 2px 2px 5px rgba(0,0,0,0.25);
      white-space: pre; z-index: 10; }
.axis { font-size: 22px; fill: #333; }
.host { font-size: 18px; fill: #333; }
.note { font-size: 22px; fill: #555; }
.tm { font-weight: bold; pointer-events: none; }
.dt { font-weight: bold; pointer-events: none; fill: #fff;
      stroke: #555; stroke-width: 2.5px; paint-order: stroke; }
</style></head><body>""",
        f"<h3 style='margin:4px 0;font-size:30px'>{_html.escape(title or 'task stream')}</h3>",
        f"<div style='font-size:22px;color:#444;margin-bottom:6px'>"
        f"ideal {ideal:.0f} s | task window {makespan:.0f} s"
        + (f" | T_compute {T_compute:.0f} s" if T_compute else "")
        + (f" | application {app_total:.0f} s" if app_total is not None else "")
        + " &mdash; hover a bar for that task's timing</div>",
        f'<svg width="{width}" height="{height:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg">',
    ]

    # ---- driver strip (application setup/finalize steps) ----
    if driver is not None:
        out.append(
            f'<text class="axis" x="{left - 8}" '
            f'y="{top + driver_h / 2 + 8:.0f}" text-anchor="end">driver</text>'
        )
        for lbl, s, e in d_segs:
            text = f"driver: {lbl} · {e - s:.1f} s"
            if lbl == _DRIVER_SETUP_LABEL:  # per-step breakdown on hover
                text += "".join(f"\n  {sl} · {dur:.1f} s" for sl, dur in d_setup)
            info = _html.escape(text).replace("\n", "&#10;")
            out.append(
                f'<rect x="{X(s):.1f}" y="{top}" '
                f'width="{max((e - s) * sx, 0.5):.1f}" height="{driver_h}" '
                f'fill="{d_colors[lbl]}" data-i="{info}"/>'
            )
            if (e - s) * sx > 13 * len(lbl) + 16:  # name the wide segments
                out.append(
                    f'<text class="note" x="{X((s + e) / 2):.1f}" '
                    f'y="{top + driver_h / 2 + 8:.0f}" text-anchor="middle">'
                    f"{_html.escape(lbl)}</text>"
                )

    # Pre/post shaded regions, spanning the utilization + stream panels.
    panels_y0, panels_y1 = top_u, top_u + util_h + gap + stream_h

    def _vspan(a, b, color, opacity):
        out.append(
            f'<rect x="{X(a):.1f}" y="{panels_y0}" '
            f'width="{(b - a) * sx:.1f}" height="{panels_y1 - panels_y0:.1f}" '
            f'fill="{color}" fill-opacity="{opacity}"/>'
        )

    if pre_rel is not None:
        _vspan(pre_rel, 0.0, _PRE_COLOR, 0.20)
        _vspan(makespan, post_end_rel, _POST_COLOR, 0.25)
    elif T_compute:
        _vspan(makespan, T_compute, _POST_COLOR, 0.25)

    # ---- utilization panel ----
    r_max = max(float(np.max(running)), float(n_worker_slots))

    def uy(v):
        return top_u + util_h - v / (1.08 * r_max) * util_h

    pts = [f"{X(ev_t[0]):.1f},{uy(0):.1f}"]
    for i in range(len(ev_t)):  # step-after curve
        y = f"{uy(running[i]):.1f}"
        pts.append(f"{X(ev_t[i]):.1f},{pts[-1].split(',')[1]}")
        pts.append(f"{X(ev_t[i]):.1f},{y}")
    out.append(
        f'<polyline points="{" ".join(pts)}" fill="none" '
        f'stroke="{_LOAD_COLOR}" stroke-width="1"/>'
    )
    out.append(
        f'<line x1="{left}" y1="{uy(n_worker_slots):.1f}" x2="{left + inner_w}" '
        f'y2="{uy(n_worker_slots):.1f}" stroke="#666" stroke-dasharray="5,4"/>'
    )
    out.append(
        f'<text class="note" x="{left + 4}" y="{uy(n_worker_slots) - 4:.1f}">'
        f"p = {n_worker_slots} worker slots</text>"
    )
    out.append(
        f'<text class="axis" x="{left - 8}" y="{top_u + util_h / 2:.0f}" '
        f'text-anchor="end">running<tspan x="{left - 8}" dy="24">tasks'
        f"</tspan></text>"
    )

    # ---- stream panel: one rect per segment, timing in data-i ----
    def _rect(s, e, row, color, info):
        if e <= s:
            return
        h = lane_px * 0.9
        info = _html.escape(info).replace("\n", "&#10;")
        out.append(
            f'<rect x="{X(s):.1f}" y="{Y(row) + lane_px * 0.05:.1f}" '
            f'width="{max((e - s) * sx, 0.5):.1f}" height="{h:.1f}" '
            f'fill="{color}" data-i="{info}"/>'
        )

    def _tau(r):  # Δτ_i duration mark centered on a task bar (map or reduce)
        fs = max(8.0, min(22.0, lane_px + 6.0))
        y = Y(r["row"]) + lane_px / 2 + fs / 3
        y -= 0.30 * lane_px * int(r.get("tau_level", 0))  # de-collision dodge
        out.append(
            f'<text class="dt" x="{X((r["start"] + r["end"]) / 2):.1f}" '
            f'y="{y:.1f}" '
            f'text-anchor="middle" font-size="{fs:.0f}">'
            f'Δτ<tspan dy="{fs / 4:.0f}" font-size="{0.75 * fs:.0f}">'
            f"{int(r['tau_index'])}</tspan></text>"
        )

    for _, r in t.iterrows():
        host = str(r["hostname"]).split(".")[0]
        task = r.get("task_id")
        task = "" if task is None or pd.isna(task) else f"task {int(task)} · "
        failed = ""
        if "task_failed_phase" in t.columns and pd.notna(r.get("task_failed_phase")):
            failed = f" — FAILED in {r['task_failed_phase']}"
        span = (
            f"{task}{host}{failed}\nwindow {r['start']:.1f} – {r['end']:.1f} s"
            f" (task total {r['end'] - r['start']:.1f} s)"
        )
        if r["kind"] == "reduce":
            _rect(
                r["start"],
                r["end"],
                r["row"],
                _POST_COLOR,
                f"reduce node · {r['end'] - r['start']:.1f} s\n{span}",
            )
            if add_time_labels:
                _tau(r)
            continue
        load_end = r["start"] + r["T_make_empty_image"] + r["T_load"]
        write_start = r["end"] - r["T_write"]
        _rect(
            r["start"],
            load_end,
            r["row"],
            _LOAD_COLOR,
            f"load · {load_end - r['start']:.1f} s\n{span}",
        )
        _rect(
            load_end,
            write_start,
            r["row"],
            _SCIENCE_COLOR,
            f"science · {write_start - load_end:.1f} s\n{span}",
        )
        _rect(
            write_start,
            r["end"],
            r["row"],
            _WRITE_COLOR,
            f"write · {r['end'] - write_start:.1f} s\n{span}",
        )
        if add_time_labels:
            _tau(r)

    # Host separators + sparse host labels (same sparsity as the PNG).
    bounds = t.groupby("host_idx")["row"].max().sort_index()
    for b in bounds.values[:-1]:
        y = Y(b + 0.5) + lane_px / 2
        out.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + inner_w}" '
            f'y2="{y:.1f}" stroke="#ddd" stroke-width="0.6"/>'
        )
    step = max(1, len(hosts) // 24)
    for hi in range(0, len(hosts), step):
        rows = t.loc[t["host_idx"] == hi, "row"]
        out.append(
            f'<text class="host" x="{left - 6}" '
            f'y="{Y(rows.mean()) + lane_px / 2 + 6:.1f}" '
            f'text-anchor="end">{_html.escape(hosts[hi].split(".")[0])}'
            f"</text>"
        )

    # X axis: ticks at a "nice" interval targeting ~12 of them.
    raw = (x1 - x0) / 12
    mag = 10 ** np.floor(np.log10(raw))
    tick = float(min((m for m in (1, 2, 5, 10) if m * mag >= raw), default=10) * mag)
    ax_y = panels_y1
    out.append(
        f'<line x1="{left}" y1="{ax_y:.1f}" x2="{left + inner_w}" '
        f'y2="{ax_y:.1f}" stroke="#333"/>'
    )
    v = np.ceil(x0 / tick) * tick
    while v <= x1:
        out.append(
            f'<line x1="{X(v):.1f}" y1="{ax_y:.1f}" x2="{X(v):.1f}" '
            f'y2="{ax_y + 5:.1f}" stroke="#333"/>'
        )
        out.append(
            f'<text class="axis" x="{X(v):.1f}" y="{ax_y + 28:.1f}" '
            f'text-anchor="middle">{v + 0.0:g}</text>'
        )
        v += tick
    if add_time_labels:  # bold t_0..t_3 efficiency-derivation marks
        marks = [(0.0, "1"), (makespan, "2")]
        if driver is not None:
            marks.append((app_start_rel, "0"))
        t3 = post_end_rel if post_end_rel is not None else (T_compute or None)
        if t3 is not None:
            marks.append((t3, "3"))
        marks.sort(key=lambda m: m[0])
        last_x = None  # nearly coincident marks (e.g. t_2/t_3) dodge to a 2nd row
        for x, sub in marks:
            dodged = last_x is not None and (X(x) - X(last_x)) < 44
            out.append(
                f'<text class="axis tm" x="{X(x):.1f}" '
                f'y="{ax_y + (104 if dodged else 66):.1f}" '
                f'text-anchor="middle" font-size="28" font-style="italic">t'
                f'<tspan dy="6" font-size="20">{sub}</tspan></text>'
            )
            last_x = None if dodged else x
    out.append(
        f'<text class="axis" x="{left + inner_w / 2:.0f}" '
        f'y="{ax_y + (150 if add_time_labels else 56):.1f}" '
        f'text-anchor="middle">time since first task start (s)</text>'
    )

    # Legend.
    lx = left
    entries = [
        ("load", _LOAD_COLOR),
        ("science", _SCIENCE_COLOR),
        ("write", _WRITE_COLOR),
        ("reduce node", _POST_COLOR),
        (pre_label, _PRE_COLOR),
        (post_label, _POST_COLOR),
    ]
    for label, color in entries:
        out.append(
            f'<rect x="{lx}" y="12" width="20" height="20" fill="{color}"'
            f' fill-opacity="0.8"/>'
        )
        out.append(
            f'<text class="axis" x="{lx + 26}" y="29">{_html.escape(label)}</text>'
        )
        lx += 26 + 12 * len(label) + 30

    out.append("</svg>")
    out.append("""<div id="tt"></div><script>
const tt = document.getElementById('tt');
document.querySelector('svg').addEventListener('mousemove', e => {
  const d = e.target.dataset && e.target.dataset.i;
  if (!d) { tt.style.display = 'none'; return; }
  tt.textContent = d;
  tt.style.display = 'block';
  tt.style.left = Math.min(e.clientX + 14, window.innerWidth - 320) + 'px';
  tt.style.top = (e.clientY + 14) + 'px';
});
document.querySelector('svg').addEventListener('mouseleave',
  () => { tt.style.display = 'none'; });
</script></body></html>""")

    with open(html_path, "w") as f:
        f.write("\n".join(out))
    print("wrote", html_path)
    return html_path


def plot_task_stream(
    source,
    compute_resources=None,
    timing_phases=None,
    anchor_source=None,
    add_time_labels=False,
    title=None,
    save_path=None,
    html_path=None,
    task_columns=None,
):
    """Dask-dashboard-style task stream: per-worker-process timeline of a run.

    Reconstructs one lane per worker slot from the per-task wall-clock anchor
    ``start_unixtime`` and the ``T_*`` phase durations, draws each task as
    load (blue) / science (green) / write (orange) segments -- white gaps are
    worker slots with nothing to do -- with a running-task utilization panel
    on top, and prints the efficiency decomposition (the performance measures
    of :func:`performance_measures`: ideal time, ramp-up, straggler tail,
    pre/post-map regions, parallel efficiencies). Works for both compute
    backends.

    Given the application's step list (``timing_phases``), a thin DRIVER strip
    is added above the utilization panel: one light-red SETUP segment
    covering everything before the compute call (the steps of
    ``timing_phases`` before ``T_compute_dask_graph`` -- broken down per step
    in the printed summary and in the HTML tooltip), the compute window, and
    the finalize steps after it, placed on the same wall clock, so the plot
    accounts for the ENTIRE time the distributed application was running, not
    just the task window.

    Parameters
    ----------
    source : dict or pandas.DataFrame
        The application return dict, or its ``timing_node_tasks`` frame.
        Needs the ``start_unixtime`` column of every task (recorded by the
        node task since 2026-07-28 -- with or without resource monitoring);
        cross-node placement relies on NTP-synced node clocks. From the full
        return dict the reduce-node rows (``timing_reduce_nodes``, drawn on
        the worker lane they ran on), the compute-window anchors and the
        driver steps (``timing_distributed_application``) and the worker
        slots (``compute_resources``) are taken. A bare frame has no reduce
        rows and no anchors.
    compute_resources : dict, optional
        Replaces the recorded ``return_dict["compute_resources"]``, e.g.
        ``{"n_worker_slots": 112}`` for a run that recorded none. p is its
        ``n_worker_slots``; without one it is reconstructed by packing the
        map tasks into lanes.
    timing_phases : list of tuple, optional
        The phase layout of the driver timing, the application's
        ``DISTRIBUTED_APPLICATION_TIMING_PHASES``: its steps before
        ``T_compute_dask_graph`` make up the setup segment and its per-step
        breakdown, those after it the finalize segments; it also gives t0 of
        a run that did not record ``application_start_unixtime``. ``None``
        (default): no driver strip.
    anchor_source : str, optional
        Provenance note for the compute anchors, echoed in the printout.
    add_time_labels : bool, optional
        Annotate the timeline with the bold time marks used when deriving the
        efficiencies: below the x-axis ``t_0`` (start of the driver setup;
        needs the driver strip), ``t_1`` (first task start, x = 0), ``t_2``
        (end of the mapping stage) and ``t_3`` (end of the reduce / post-map
        region), plus each task's duration label ``Δτ_i`` centered on its bar
        -- map AND measured reduce tasks alike (map tasks keep their recorded
        task number, reduce tasks are numbered on in start order), matching
        the paper's ``W = Σ Δτ_i``. Meant for small pedagogical runs (e.g.
        the tutorial notebook) -- with thousands of tasks the per-task labels
        overlap. Applied to the interactive HTML twin as well. Default False.
    title : str, optional
        Figure title prefix (e.g. a run name); default "task stream".
    save_path : str, optional
        If set, save the figure there (PNG) and print the path.
    html_path : str, optional
        If set, also write a standalone INTERACTIVE version there: the same
        two panels as inline SVG with a hover tooltip giving each bar's
        timing (segment duration, task id, host, task window, total task
        time). Pure SVG + vanilla JavaScript -- no extra dependencies, works
        offline; one SVG element per task segment, so the file grows with the
        task count (~a few MB for a 15k-task run).
    task_columns : dict, optional
        The node task timing columns drawn, for node tasks other than the
        imager's: ``"task"`` (the task's total time), ``"load"`` (the columns
        summed into the load segment) and ``"write"`` (the write segment;
        ``None`` for none), e.g. ``{"task": "T_simulate_task", "load": [],
        "write": "T_write"}``. Missing keys keep the imager's columns
        (:data:`IMAGING_TASK_STREAM_COLUMNS`); ``None`` (default) draws the
        imager's.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None when ``start_unixtime`` is absent (run recorded
        before the anchor existed) or missing for some tasks.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    tasks = _as_timing_frame(source)
    if "start_unixtime" not in tasks.columns or tasks["start_unixtime"].isna().all():
        print(
            "plot_task_stream: no start_unixtime column (run recorded before "
            "the wall-clock anchor was added); cannot place tasks on the run "
            "timeline -- skipping."
        )
        return None
    n_missing = int(tasks["start_unixtime"].isna().sum())
    if n_missing:
        print(
            f"plot_task_stream: {n_missing} of {len(tasks)} tasks have no "
            "start_unixtime and cannot be placed on the run's wall clock; skipping."
        )
        return None
    measures = performance_measures(
        source,
        compute_resources,
        timing_phases=timing_phases,
        task_columns=task_columns,
    )
    if task_columns is not None:
        tasks = _with_imaging_task_columns(tasks, task_columns)
    reduce_frame = compute_start = compute_end = driver_timing = None
    if isinstance(source, dict):
        records = source.get("timing_reduce_nodes")
        if records is not None and len(records):
            reduce_frame = pd.DataFrame(records)
        timing = source.get("timing_distributed_application") or {}
        if timing.get("compute_start_unixtime") is not None:
            compute_start = float(timing["compute_start_unixtime"])
            compute_end = float(timing["compute_end_unixtime"])
            anchor_source = anchor_source or "measured"
        driver_timing = timing or None
    T_compute = measures["T_compute_dask_graph"]
    T_compute = None if np.isnan(T_compute) else T_compute

    pre_label = "pre-map: worker imports + graph submit"
    post_label = "post-map: reduce residue + gather/save"
    # Fold measured reduce nodes (when recorded) into the lane assignment: they
    # run on the same worker processes (dask) / the manager (MPI), one at a
    # time, so they share the per-host lane pool with the map tasks.
    tasks = tasks.copy()
    tasks["kind"] = "map"
    if reduce_frame is not None and len(reduce_frame):
        red_rows = pd.DataFrame(
            {
                "hostname": reduce_frame["hostname"],
                "start_unixtime": reduce_frame["start_unixtime"],
                "T_image_cube_task": (
                    reduce_frame["end_unixtime"] - reduce_frame["start_unixtime"]
                ),
                "T_make_empty_image": 0.0,
                "T_load": 0.0,
                "T_write": 0.0,
                "kind": "reduce",
            }
        )
        # Carry the recorded execution identity through (recorded on BOTH map
        # tasks and reduce nodes since 2026-07-28), so each reduce lands on
        # the worker lane it actually ran on -- including a dedicated lane for
        # the MPI manager, which runs no map tasks.
        for col in ("process_pid", "thread_native_id", "worker_name"):
            if col in reduce_frame.columns:
                red_rows[col] = reduce_frame[col].values
        tasks = pd.concat([tasks, red_rows], ignore_index=True)
    t, hosts, n_lanes = assign_task_stream_lanes(tasks)
    # Δτ index over ALL tasks in START order: map tasks numbered as they
    # started, then the reduce tasks numbered on, also in start order. Used by
    # the add_time_labels marks and consistent with the paper's W = Σ Δτ_i,
    # which includes the reduce tasks. The index is a label, not a channel id:
    # the scheduler does not dispatch tasks in task_id order, so numbering by
    # task_id put Δτ_0 wherever channel 0 happened to run (e.g. in the second
    # wave) instead of on the first task started.
    m = t["kind"] == "map"
    map_order = t.loc[m].sort_values("start_unixtime", kind="stable").index
    t.loc[map_order, "tau_index"] = np.arange(len(map_order))
    nxt = len(map_order)
    red_order = t.loc[~m].sort_values("start_unixtime", kind="stable").index
    t.loc[red_order, "tau_index"] = np.arange(nxt, nxt + len(red_order))
    t["tau_index"] = t["tau_index"].astype(int)
    tm = t[t["kind"] == "map"]
    tr = t[t["kind"] == "reduce"]
    makespan = tm["end"].max()
    # The parallelism p of the paper's Performance Measures section: the
    # recorded worker slots, else a MAP-ONLY packing pass (in the combined
    # packing, reduce nodes borrow idle map lanes and bump map tasks into
    # overflow lanes, so combined lane counts overstate the workers). The
    # ideal time T_ideal = W / p, with W = Σ Δτ_i over ALL tasks (map +
    # measured reduce).
    n_worker_slots = measures["n_worker_slots"]
    ideal = measures["ideal_time"]

    # Utilization curve from start/end events (map tasks only; reduce gets its
    # own curve below when recorded).
    ev_t, running = _running_task_count(tm["start"].values, tm["end"].values)

    t0_abs = tasks["start_unixtime"].min()
    pre_rel = post_end_rel = None
    if compute_start is not None:
        pre_rel = compute_start - t0_abs  # negative seconds
        post_end_rel = compute_end - t0_abs
    driver = _driver_timeline(
        driver_timing, compute_start, compute_end, t0_abs, timing_phases
    )
    # t_3 of the efficiency derivation: end of the reduce / post-map region
    # (falls back to T_compute when the absolute anchors are missing).
    t3_rel = post_end_rel if post_end_rel is not None else (T_compute or None)
    if driver_timing and driver is None:
        if timing_phases is None:
            print(
                "plot_task_stream: no timing_phases given; the driver strip is "
                "left out."
            )
        elif compute_start is None:
            print(
                "plot_task_stream: driver timings given but no absolute compute "
                "anchors; cannot place the driver strip on the run timeline."
            )
    summary = []  # the printed decomposition, echoed at the top of the figure

    def _p(line):
        print(line)
        summary.append(line)

    if title:
        print(f"run: {title}")
    provenance = (
        "lane packing" if measures["n_worker_slots_reconstructed"] else "recorded"
    )
    _p(
        f"  parallelism p              : {n_worker_slots} worker slots "
        f"({provenance}; {len(hosts)} hosts, {n_lanes} reconstructed lanes)"
    )
    _p(
        f"  T_ideal = W / p            : {ideal:8.1f} s "
        f"(W = Σ Δτ_i = {measures['work']:.1f} s over {measures['n_map_tasks']} "
        f"map + {measures['n_reduce_tasks']} reduce tasks)"
    )
    task_window = measures["task_window"]
    _p(
        f"  task-window makespan       : {task_window:8.1f} s "
        f"(+{task_window - measures['work_map'] / n_worker_slots:.1f} s "
        f"ramp/gaps/tail, {100 * measures['map_busy_fraction']:.1f}% busy "
        "inside window)"
    )
    if T_compute:
        if pre_rel is not None:
            _p(
                f"  T_compute (map+reduce)     : {T_compute:8.1f} s = "
                f"{measures['pre_map']:.1f} s PRE first task ({pre_label}) + "
                f"{task_window:.1f} s task window + "
                f"{measures['post_map']:.1f} s POST ({post_label})"
                + (f"   [anchors: {anchor_source}]" if anchor_source else "")
            )
        else:
            _p(
                f"  T_compute (map+reduce)     : {T_compute:8.1f} s "
                f"(+{measures['post_map']:.1f} s outside the task window)"
            )
    if driver is not None:
        d_segs, app_start_rel, app_end_rel, d_colors, d_setup = driver
        app_total = app_end_rel - app_start_rel
        setup = pre_rel - app_start_rel
        finalize = app_end_rel - post_end_rel
        compute_span = post_end_rel - pre_rel
        _p(
            f"  application total          : {app_total:8.1f} s = "
            f"{setup:.1f} s driver setup + {compute_span:.1f} s compute + "
            f"{finalize:.1f} s finalize (entire distributed application call)"
        )
        steps = " | ".join(
            [f"{lbl} {dur:.1f}" for lbl, dur in d_setup if dur >= 0.1]
            + [
                f"{lbl} {e - s:.1f}"
                for lbl, s, e in d_segs
                if lbl not in (_DRIVER_SETUP_LABEL, _DRIVER_COMPUTE_LABEL)
                and e - s >= 0.1
            ]
        )
        if steps:
            _p(f"  driver steps (s)           : {steps}")
    # The two absolute parallel efficiencies of the paper's Performance
    # Measures section, in the t_0..t_3 / Δτ_i notation of add_time_labels:
    # T_ideal = W / p over the map-reduce window (t_1 = first task start = 0
    # .. t_3 = end of reduce) resp. the end-to-end window (t_0 = application
    # start .. t_3). W includes the reduce tasks.
    if np.isfinite(measures["parallel_efficiency_map_reduce"]):
        _p(
            f"  E map-reduce               : "
            f"{100 * measures['parallel_efficiency_map_reduce']:8.1f} %   "
            "[= (W / p) / (t_3 - t_1)]"
        )
    if np.isfinite(measures["parallel_efficiency_end_to_end"]):
        _p(
            f"  E end-to-end               : "
            f"{100 * measures['parallel_efficiency_end_to_end']:8.1f} % "
            f"  [= (W / p) / (t_3 - t_0)]"
        )
    # Ramp and tail: time to reach 95% of the worker slots / time the last 5%
    # of tasks spend after the 95th-percentile end.
    _p(f"  ramp to 95% busy           : {measures['time_to_95_percent_busy']:8.1f} s")
    _p(
        f"  straggler tail (p95->last) : "
        f"{task_window - measures['p95_task_end']:8.1f} s"
    )
    if len(tr):
        busy_r = tr["T_image_cube_task"].sum()
        inside = np.clip(np.minimum(tr["end"], makespan) - tr["start"], 0, None)
        overlap = float(inside.sum() / busy_r) if busy_r else 0.0
        _p(
            f"  reduce nodes (measured)    : {len(tr)} calls, {busy_r:8.1f} s "
            f"busy, window {tr['start'].min():.1f}-{tr['end'].max():.1f} s, "
            f"{100 * overlap:.0f}% of reduce busy inside the map window"
        )

    # ---------------- figure ---------------------------------------------- #
    if driver is not None:
        fig, (ax_d, ax_u, ax) = plt.subplots(
            3,
            1,
            figsize=(15, 13.8),
            sharex=True,
            gridspec_kw={"height_ratios": [0.45, 1, 5], "hspace": 0.05},
        )
        for lbl, s, e in d_segs:
            ax_d.add_patch(plt.Rectangle((s, 0), e - s, 1, color=d_colors[lbl], lw=0))
        d_dur = {}
        for lbl, s, e in d_segs:
            d_dur[lbl] = d_dur.get(lbl, 0.0) + (e - s)
        ax_d.legend(
            handles=[
                Patch(color=d_colors[lbl], label=f"{lbl} ({dur:.1f} s)")
                for lbl, dur in d_dur.items()
            ],
            loc="upper left",
            bbox_to_anchor=(1.005, 1.15),
            fontsize=12,
            frameon=False,
            labelspacing=0.3,
            handlelength=1.4,
        )
        ax_d.set_ylim(0, 1)
        ax_d.set_yticks([])
        ax_d.set_ylabel("driver", fontsize=16)
    else:
        fig, (ax_u, ax) = plt.subplots(
            2,
            1,
            figsize=(15, 13),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 5], "hspace": 0.04},
        )

    ax_u.fill_between(ev_t, running, step="post", color=_LOAD_COLOR, alpha=0.35, lw=0)
    ax_u.plot(
        ev_t,
        running,
        drawstyle="steps-post",
        color=_LOAD_COLOR,
        lw=1,
        label="running map tasks",
    )
    if len(tr):
        rev = np.concatenate([tr["start"].values, tr["end"].values])
        rdl = np.concatenate([np.ones(len(tr)), -np.ones(len(tr))])
        ro = np.argsort(rev, kind="stable")
        ax_u.plot(
            rev[ro],
            np.cumsum(rdl[ro]),
            drawstyle="steps-post",
            color=_POST_COLOR,
            lw=1.2,
            label="running reduce nodes",
        )
    ax_u.axhline(
        n_worker_slots,
        color="#666666",
        ls="--",
        lw=1,
        label=f"p = {n_worker_slots} worker slots",
    )
    ax_u.legend(
        loc="upper left", bbox_to_anchor=(1.005, 1.0), fontsize=14, frameon=False
    )
    if pre_rel is not None:
        ax_u.axvspan(pre_rel, 0, color=_PRE_COLOR, alpha=0.20)
        ax_u.axvspan(makespan, post_end_rel, color=_POST_COLOR, alpha=0.25)
    elif T_compute:
        ax_u.axvspan(makespan, T_compute, color=_POST_COLOR, alpha=0.25)
        ax_u.text(
            0.5 * (makespan + T_compute),
            0.5 * n_worker_slots,
            f"{post_label}\n{measures['post_map']:.0f} s",
            ha="center",
            fontsize=16,
            color="#8a4a6d",
        )
    ax_u.set_ylabel("running tasks", fontsize=20)
    ax_u.tick_params(labelsize=20)
    ax_u.grid(True, alpha=0.25)

    seg_h = 0.9
    for _, r in t.iterrows():
        y = r["row"] - seg_h / 2
        if r["kind"] == "reduce":
            ax.add_patch(
                plt.Rectangle(
                    (r["start"], y),
                    r["end"] - r["start"],
                    seg_h,
                    color=_POST_COLOR,
                    lw=0,
                )
            )
            continue
        load_end = r["start"] + r["T_make_empty_image"] + r["T_load"]
        write_start = r["end"] - r["T_write"]
        ax.add_patch(
            plt.Rectangle(
                (r["start"], y),
                load_end - r["start"],
                seg_h,
                color=_LOAD_COLOR,
                lw=0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (load_end, y),
                max(write_start - load_end, 0),
                seg_h,
                color=_SCIENCE_COLOR,
                lw=0,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (write_start, y), r["T_write"], seg_h, color=_WRITE_COLOR, lw=0
            )
        )
    if pre_rel is not None:
        ax.axvspan(pre_rel, 0, color=_PRE_COLOR, alpha=0.20)
        ax.axvspan(makespan, post_end_rel, color=_POST_COLOR, alpha=0.25)
    elif T_compute:
        ax.axvspan(makespan, T_compute, color=_POST_COLOR, alpha=0.25)
    # Host separators + sparse host-name ticks (real yticks, so matplotlib
    # keeps the axis label clear of them however long the hostnames are).
    bounds = t.groupby("host_idx")["row"].max().sort_index()
    for b in bounds.values[:-1]:
        ax.axhline(b + 0.5, color="#dddddd", lw=0.4)
    step = max(1, len(hosts) // 24)
    shown = range(0, len(hosts), step)
    ax.set_yticks(
        [t.loc[t["host_idx"] == hi, "row"].mean() for hi in shown],
        [hosts[hi].split(".")[0] for hi in shown],
        fontsize=12,
    )
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_ylim(-1, n_lanes)
    left = pre_rel if pre_rel is not None else 0.0
    right = (
        post_end_rel
        if post_end_rel is not None
        else max(T_compute or 0, makespan, t["end"].max())
    )
    if driver is not None:
        left = min(left, app_start_rel)
        right = max(right, app_end_rel)
    ax.set_xlim(left * 1.02 if left < 0 else left, right * 1.005)
    if driver is not None:
        # Name the wide driver segments in place; every step is identified in
        # the legend below the figure (with its duration) regardless of width.
        span = right - left
        for lbl, s, e in d_segs:
            if e - s > 0.04 * span:
                ax_d.text(
                    (s + e) / 2,
                    0.5,
                    lbl,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="#333333",
                )
    ax.set_ylabel(f"worker processes ({n_lanes} lanes, grouped by host)", fontsize=20)
    ax.set_xlabel("time since first task start (s)", fontsize=20)
    if add_time_labels:
        # Bold efficiency-derivation marks: t_0..t_3 under the x-axis and each
        # map task's duration Δτ_i on its bar (i = task number).
        import matplotlib.transforms as mtransforms

        marks = [(0.0, r"$\mathbf{t_1}$"), (makespan, r"$\mathbf{t_2}$")]
        if driver is not None:
            marks.append((app_start_rel, r"$\mathbf{t_0}$"))
        if t3_rel is not None:
            marks.append((t3_rel, r"$\mathbf{t_3}$"))
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        marks.sort(key=lambda m: m[0])
        last_x = None  # nearly coincident marks (e.g. t_2/t_3) dodge to a 2nd row
        for x, lbl in marks:
            dodged = last_x is not None and (x - last_x) < 0.055 * (right - left)
            ax.text(
                x,
                -0.095 if dodged else -0.048,
                lbl,
                transform=trans,
                ha="center",
                va="top",
                fontsize=26,
            )
            last_x = None if dodged else x
        ax.xaxis.labelpad = 96  # keep the axis caption clear of the t marks
        # Δτ_i on every task, map AND reduce (W = Σ Δτ_i over all). The white
        # text gets a thin dark stroke so it stays readable where it spills
        # past a narrow bar, and neighboring labels within a lane closer than
        # the approximate label width dodge up/down within the lane band.
        import matplotlib.patheffects as path_effects

        stroke = [path_effects.withStroke(linewidth=2.5, foreground="#555555")]
        w_est = 0.045 * (right - left)  # ~label width in data units
        t["tau_level"] = 0  # shared with the HTML twin
        last_in_lane = {}  # row -> (center_x, level) of the previous label
        for i, r in t.sort_values(["row", "start"]).iterrows():
            cx = (r["start"] + r["end"]) / 2
            prev = last_in_lane.get(r["row"])
            level = 0
            if prev is not None and cx - prev[0] < w_est:
                level = 1 if prev[1] != 1 else -1
            last_in_lane[r["row"]] = (cx, level)
            t.at[i, "tau_level"] = level
            ax.text(
                cx,
                r["row"] + 0.30 * level,
                rf"$\mathbf{{\Delta\tau_{{{int(r['tau_index'])}}}}}$",
                ha="center",
                va="center",
                fontsize=18,
                color="white",
                path_effects=stroke,
            )
    handles = [
        Patch(color=_LOAD_COLOR, label="load"),
        Patch(color=_SCIENCE_COLOR, label="science"),
        Patch(color=_WRITE_COLOR, label="write"),
        Patch(color=_PRE_COLOR, alpha=0.35, label=pre_label),
        Patch(color=_POST_COLOR, alpha=0.4, label=post_label),
    ]
    if len(tr):
        handles.insert(3, Patch(color=_POST_COLOR, label="reduce node (measured)"))
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        fontsize=16,
        frameon=False,
    )
    fig.suptitle(
        f"{title or 'task stream'}\n"
        f"ideal {ideal:.0f} s | task window {makespan:.0f} s | "
        + (f"T_compute {T_compute:.0f} s" if T_compute else "")
        + (f" | application {app_total:.0f} s" if driver is not None else ""),
        fontsize=24,
    )
    # Efficiency decomposition block above the title; it sits outside the
    # figure edge, so it relies on the tight bounding box used when saving.
    # Long lines (e.g. the driver-step breakdown) are wrapped so the block
    # stays no wider than the figure at this font size.
    import textwrap

    wrapped = [
        w
        for line in summary
        for w in textwrap.wrap(
            line, width=88, subsequent_indent=" " * 31, drop_whitespace=False
        )
    ]
    fig.text(
        0.02,
        1.002,
        "\n".join(wrapped),
        va="bottom",
        ha="left",
        fontsize=18,
        family="monospace",
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("wrote", save_path)
    if html_path is not None:
        _write_task_stream_html(
            t,
            hosts,
            n_lanes,
            n_worker_slots,
            ev_t,
            running,
            pre_rel,
            post_end_rel,
            makespan,
            ideal,
            T_compute,
            pre_label,
            post_label,
            title,
            html_path,
            driver=driver,
            add_time_labels=add_time_labels,
        )
    return fig


_SCALING_TITLES = {
    "total_time": "Total Time",
    "speedup": "Speedup: absolute (work-based)",
    "speedup_relative": "Speedup: relative (baseline-normalized)",
    "parallel_efficiency": "Parallel efficiency: absolute (work-based)",
    "parallel_efficiency_relative": (
        "Parallel efficiency: relative (baseline-normalized)"
    ),
    "throughput": "Delivered Throughput and Allocation Cost",
}
_TIME_UNIT_SECONDS = {"hours": 3600.0, "minutes": 60.0, "seconds": 1.0}


def _aligned_with_runs(values, n_runs, name):
    """``values`` as a list with one entry per run; refuses a dict or another
    length."""
    if isinstance(values, (dict, str)) or len(values) != n_runs:
        raise ValueError(
            f"{name} must be a sequence with one entry per run ({n_runs})."
        )
    return list(values)


def _runs_and_resources(source, compute_resources, caller):
    """The runs of a multi-run figure, each one's compute resources
    (:func:`_compute_resources`) and the per-run overrides passed on to
    :func:`performance_measures`."""
    if isinstance(source, (dict, pd.DataFrame)):
        raise ValueError(
            f"{caller}: source must be a sequence of return dicts, one per run."
        )
    runs = list(source)
    if not runs:
        raise ValueError(f"{caller}: source holds no run.")
    overrides = (
        None
        if compute_resources is None
        else _aligned_with_runs(compute_resources, len(runs), "compute_resources")
    )
    resources = [
        _compute_resources(run, None if overrides is None else overrides[i])
        for i, run in enumerate(runs)
    ]
    return runs, resources, overrides


def _scale_point_label(n_hosts, n_worker_slots):
    """A bar label: the hosts over the worker slots p (the hosts alone when
    p is not known)."""
    if n_worker_slots is None or not np.isfinite(n_worker_slots):
        return str(n_hosts)
    return f"{n_hosts}\n{int(n_worker_slots)}"


def plot_scaling(
    source,
    quantity,
    x=None,
    compute_resources=None,
    sweep_labels=None,
    scaling="strong",
    total_time=None,
    time_unit="hours",
    x_label="Number of Nodes",
    x_tick_labels="n_worker_slots",
    reference_points=None,
    timing_phases=None,
    task_columns=None,
    fixed_non_io_compute=None,
    notes=None,
    title=None,
    save_path=None,
):
    """Plot one scaling quantity of several runs of a distributed application.

    The quantities are the total time, the speedups and parallel efficiencies
    of the RADPS benchmarking memo (:func:`performance_measures`) over the map
    reduce window (t3 - t1) and the end to end window (t3 - t0), absolute or
    relative to a baseline run, and the throughput with the cost in
    node-hours. Runs with equal ``sweep_labels`` form a sweep, drawn as one
    line (one line per window for the speedups and efficiencies) in input
    order; the relative measures are relative to the sweep's baseline. One
    sweep is coloured by window, several sweeps by sweep.

    Parameters
    ----------
    source : sequence of dict
        The return dicts of the runs (``"timing_node_tasks"``,
        ``"timing_distributed_application"``, ``"timing_reduce_nodes"`` and
        ``"compute_resources"``), in plotting order.
    quantity : str
        ``"total_time"``, ``"speedup"``, ``"speedup_relative"``,
        ``"parallel_efficiency"``, ``"parallel_efficiency_relative"`` or
        ``"throughput"`` (channels per second of ``total_time``, the sum of
        the node tasks' ``n_channels``, over the cost ``n_hosts *
        total_time`` in node-hours).
    x : sequence, optional
        The x value of every run, e.g. the number of channels of a weak
        scaling; default the hosts ``n_hosts`` of each run's compute
        resources.
    compute_resources : sequence of dict, optional
        One entry per run replacing its recorded
        ``return_dict["compute_resources"]``; None entries keep the recorded
        ones (see :func:`performance_measures`).
    sweep_labels : sequence of str, optional
        The sweep of every run; default one sweep labelled ``"measured"``.
    scaling : str, optional
        ``"strong"`` (default) or ``"weak"``: the baseline of the relative
        measures (see :func:`performance_measures`). Weak scaling also starts
        the total-time axis at 0 and fits the speedup axis to the data.
    total_time : sequence of float, optional
        The wall clock of every run in seconds, e.g. a benchmark harness's;
        default each run's ``T_total``.
    time_unit : str, optional
        ``"hours"`` (default), ``"minutes"`` or ``"seconds"``, for the total
        time.
    x_label : str, optional
        The x axis label.
    x_tick_labels : str, optional
        ``"n_worker_slots"`` (default): ticks at the x values with the worker
        slots p of the runs there below them; ``"values"``: ticks at the x
        values.
    reference_points : list of dict, optional
        Reference measurements drawn on the total-time figure, each
        ``{"label": str, "x": sequence, "total_time": sequence}`` (seconds),
        e.g. ``[{"label": "CASA Memo 13", "x": [72], "total_time": [11194]}]``.
    timing_phases : list of tuple, optional
        The application's ``DISTRIBUTED_APPLICATION_TIMING_PHASES``, to
        rebuild t0 of runs that did not record ``application_start_unixtime``.
    task_columns : dict, optional
        The node task timing columns, as for :func:`plot_task_stream`.
    fixed_non_io_compute : float, optional
        Seconds of non-I/O compute per map task (see
        :func:`performance_measures`); the total-time figure is shortened by
        the same ``fixed_compute_shift`` as the windows. The throughput is
        always that of the measured total time.
    notes : str or sequence of str, optional
        Notes written along the bottom of the figure.
    title : str, optional
        The figure title; default a title naming the quantity.
    save_path : str, optional
        If set, the figure is saved there (PNG, dpi 150, tight bounding box)
        and the path printed; the field ``{figure}`` in it is replaced by
        ``quantity``.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a printed note) when no run has the
        quantity: a speedup or efficiency needs a finite map reduce window
        (the tasks' ``start_unixtime`` and the ``compute_end_unixtime``
        anchor), the throughput the ``n_channels`` column.

    Raises
    ------
    ValueError
        For an unknown ``quantity``, ``scaling``, ``time_unit`` or
        ``x_tick_labels``, a ``source`` that is not a sequence of runs, a
        per-run argument without one entry per run, or a run without a
        recorded ``n_hosts`` when ``x`` is not given or for the throughput.
    """
    if quantity not in _SCALING_TITLES:
        raise ValueError(
            f"quantity must be one of {list(_SCALING_TITLES)}; got {quantity!r}."
        )
    if time_unit not in _TIME_UNIT_SECONDS:
        raise ValueError(
            f"time_unit must be one of {list(_TIME_UNIT_SECONDS)}; got {time_unit!r}."
        )
    if x_tick_labels not in ("n_worker_slots", "values"):
        raise ValueError(
            "x_tick_labels must be 'n_worker_slots' or 'values'; got "
            f"{x_tick_labels!r}."
        )
    runs, resources, overrides = _runs_and_resources(
        source, compute_resources, "plot_scaling"
    )
    n_runs = len(runs)
    n_hosts = [_n_hosts(r) for r in resources]
    if None in n_hosts and quantity == "throughput":
        raise ValueError(
            "plot_scaling: the node-hours need the n_hosts of every run's "
            "compute resources; pass compute_resources."
        )
    if None in n_hosts and x is None:
        raise ValueError(
            "plot_scaling: x defaults to the n_hosts of every run's compute "
            "resources, which not every run records; pass x or compute_resources."
        )
    x_values = n_hosts if x is None else _aligned_with_runs(x, n_runs, "x")
    x_plot = np.asarray(x_values, dtype=float)
    sweep_labels = (
        ["measured"] * n_runs
        if sweep_labels is None
        else _aligned_with_runs(sweep_labels, n_runs, "sweep_labels")
    )
    sweeps = {}
    for i, label in enumerate(sweep_labels):
        sweeps.setdefault(label, []).append(i)
    frames = []
    for index in sweeps.values():
        measures = performance_measures(
            [runs[i] for i in index],
            None if overrides is None else [overrides[i] for i in index],
            scaling=scaling,
            timing_phases=timing_phases,
            task_columns=task_columns,
            fixed_non_io_compute=fixed_non_io_compute,
        )
        measures.index = index
        frames.append(measures)
    measures = pd.concat(frames).sort_index()
    p = measures["n_worker_slots"].to_numpy(dtype=float)
    seconds = (
        measures["T_total"].to_numpy(dtype=float)
        if total_time is None
        else np.asarray(
            _aligned_with_runs(total_time, n_runs, "total_time"), dtype=float
        )
    )

    if quantity == "throughput":
        n_channels = np.full(n_runs, np.nan)
        for i, run in enumerate(runs):
            tasks = _as_timing_frame(run)
            if "n_channels" in tasks.columns:
                n_channels[i] = float(tasks["n_channels"].sum())
        throughput = n_channels / seconds
        node_hours = np.asarray(n_hosts, dtype=float) * seconds / 3600
        drawn = np.isfinite(throughput) & np.isfinite(node_hours) & np.isfinite(x_plot)
        if not drawn.any():
            print(
                "plot_scaling: no run has the n_channels column and a finite "
                "total time; no throughput figure."
            )
            return None
        figure, (ax_throughput, ax_cost) = _performance_figure(
            (9, 10), 2, 1, sharex=True, gridspec_kw={"hspace": 0.06}
        )
        for sweep, index in sweeps.items():
            index = [i for i in index if drawn[i]]
            if not index:
                continue
            if len(sweeps) == 1:
                ax_throughput.plot(x_plot[index], throughput[index], marker="o")
                ax_cost.plot(
                    x_plot[index], node_hours[index], marker="o", color="tab:orange"
                )
            else:
                (line,) = ax_throughput.plot(
                    x_plot[index], throughput[index], marker="o", label=sweep
                )
                ax_cost.plot(
                    x_plot[index],
                    node_hours[index],
                    marker="o",
                    color=line.get_color(),
                    label=sweep,
                )
        ax_throughput.set_ylabel("Throughput (channels / s)")
        ax_throughput.set_ylim(bottom=0)
        ax_throughput.set_title(title or _SCALING_TITLES[quantity])
        if len(sweeps) > 1:
            ax_throughput.legend()
        ax_cost.set_ylabel("Cost of the run (node-hours)")
        ax_cost.set_ylim(bottom=0)
        for ax in (ax_throughput, ax_cost):
            ax.grid(True, color="lightgray")
        ax = ax_cost
    elif quantity == "total_time":
        shift = (
            measures["fixed_compute_shift"].to_numpy(dtype=float)
            if fixed_non_io_compute is not None
            else 0.0
        )
        y = (seconds - shift) / _TIME_UNIT_SECONDS[time_unit]
        drawn = np.isfinite(y) & np.isfinite(x_plot)
        if not drawn.any():
            print("plot_scaling: no run has a finite total time; no figure.")
            return None
        figure, ax = _performance_figure((8, 8))
        for sweep, index in sweeps.items():
            index = [i for i in index if drawn[i]]
            if index:
                ax.plot(x_plot[index], y[index], marker="o", label=sweep)
        for reference in reference_points or ():
            reference_x = np.asarray(reference["x"], dtype=float)
            ax.plot(
                reference_x,
                np.asarray(reference["total_time"], dtype=float)
                / _TIME_UNIT_SECONDS[time_unit],
                marker="X",
                markersize=10,
                linestyle="--" if len(reference_x) > 1 else "none",
                label=reference["label"],
            )
        ax.set_ylabel(f"Total Time ({time_unit})")
        if scaling == "weak":
            ax.set_ylim(bottom=0)
    else:
        measure = "speedup" if quantity.startswith("speedup") else "parallel_efficiency"
        relative = quantity.endswith("_relative")
        keys = {
            window: f"{measure}_{window}" + ("_relative" if relative else "")
            for window in _WINDOW_NAMES
        }
        values = {
            window: measures[key].to_numpy(dtype=float) for window, key in keys.items()
        }
        usable = np.isfinite(measures["map_reduce_window"].to_numpy(dtype=float))
        usable &= np.isfinite(x_plot)
        if not usable.any():
            print(
                "plot_scaling: no run has a finite map reduce window (tasks "
                "with start_unixtime and the compute_end_unixtime anchor); "
                f"no {quantity} figure."
            )
            return None
        figure, ax = _performance_figure((8, 8))
        if len(sweeps) == 1:
            labels = (
                _WEAK_RELATIVE_LABELS
                if scaling == "weak" and relative
                else _MEASURE_LABELS
            )
            for window, key in keys.items():
                ax.plot(
                    x_plot[usable],
                    values[window][usable],
                    label=labels[key],
                    **_WINDOW_STYLES[window],
                )
        else:
            for sweep, index in sweeps.items():
                index = [i for i in index if usable[i]]
                if not index:
                    continue
                (line,) = ax.plot(
                    x_plot[index],
                    values["map_reduce"][index],
                    marker=_WINDOW_STYLES["map_reduce"]["marker"],
                    label=f"{sweep}, {_WINDOW_NAMES['map_reduce']} "
                    f"({_MEASURE_SYMBOLS[keys['map_reduce']]})",
                )
                ax.plot(
                    x_plot[index],
                    values["end_to_end"][index],
                    marker=_WINDOW_STYLES["end_to_end"]["marker"],
                    linestyle="--",
                    color=line.get_color(),
                    label=f"{sweep}, {_WINDOW_NAMES['end_to_end']} "
                    f"({_MEASURE_SYMBOLS[keys['end_to_end']]})",
                )
        plotted = np.concatenate([values[window][usable] for window in keys])
        if measure == "speedup":
            ideal = sorted(set(zip(x_plot[usable], p[usable], strict=True)))
            ax.plot(
                [value for value, _ in ideal],
                [slots for _, slots in ideal],
                label=r"ideal ($S = p$)",
                **_IDEAL_LINE,
            )
            if scaling == "weak":
                plotted = np.concatenate([plotted, p[usable]])
                ax.set_ylim(0.95 * np.nanmin(plotted), 1.02 * np.nanmax(plotted))
            ax.set_ylabel("Speedup (worker units)")
            hosts = [n_hosts[i] for i in np.flatnonzero(usable)]
            if None not in hosts:
                _node_units_axis(ax, p[usable], hosts)
        else:
            ax.axhline(1.0, label=r"ideal ($E = 1$)", **_IDEAL_LINE)
            ax.set_ylim(
                min(0.75, 0.95 * np.nanmin(plotted)),
                max(1.02, 1.02 * np.nanmax(plotted)),
            )
            ax.set_ylabel("Parallel Efficiency")

    ax.set_xlabel(x_label)
    if x_tick_labels == "n_worker_slots":
        _parallelism_ticks(ax, x_values, p, x_label)
    else:
        _value_ticks(ax, sorted(set(x_values)), x_label)
    if quantity != "throughput":
        ax.set_title(title or _SCALING_TITLES[quantity])
        ax.grid(True, color="lightgray")
        ax.legend()
        ax.set_aspect(1.0 / ax.get_data_ratio())
    _add_notes(figure, notes)
    _save_figures({quantity: figure}, save_path)
    return figure


def plot_efficiency_loss_decomposition(
    source,
    compute_resources=None,
    breakdown="losses",
    timing_phases=None,
    task_columns=None,
    notes=None,
    title=None,
    save_path=None,
):
    """Stacked bars of where the wall clock of every run of a distributed
    application goes.

    ``breakdown="losses"`` splits each run's ``T_total`` into the useful work
    (the map work over the worker slots, ``work_map / n_worker_slots``) and
    the loss sources of :func:`performance_measures`: the idle worker time
    during the ramp (until 95 % of the slots are busy), inside the window and
    in the straggler tail (after 95 % of the map tasks ended), the time before
    the first map task (``pre_map``) and after the last one (``post_map``)
    inside the compute call, and the fixed costs outside it (``T_total -
    T_compute_dask_graph``). Useful work and idle time make up the map window
    exactly. A second panel shows the losses as a percentage of ``T_total``.

    ``breakdown="driver_steps"`` stacks the application's own driver steps
    outside the compute call (the steps of ``timing_phases`` except
    ``T_compute_dask_graph``, recorded in
    ``return_dict["timing_distributed_application"]``), largest mean at the
    bottom; the stack is the fixed costs of the losses breakdown.

    Parameters
    ----------
    source : sequence of dict
        The return dicts of the runs, one bar per run in input order.
    compute_resources : sequence of dict, optional
        One entry per run replacing its recorded
        ``return_dict["compute_resources"]``; None entries keep the recorded
        ones. The bars are labelled with each run's ``n_hosts`` and
        ``n_worker_slots``.
    breakdown : str, optional
        ``"losses"`` (default) or ``"driver_steps"``.
    timing_phases : list of tuple, optional
        The application's ``DISTRIBUTED_APPLICATION_TIMING_PHASES``; needed
        for ``"driver_steps"``.
    task_columns : dict, optional
        The node task timing columns, as for :func:`plot_task_stream`.
    notes : str or sequence of str, optional
        Notes written along the bottom of the figure.
    title : str, optional
        The figure title; default a title naming the breakdown.
    save_path : str, optional
        If set, the figure is saved there (PNG, dpi 150, tight bounding box)
        and the path printed; the field ``{figure}`` in it is replaced by
        ``breakdown``.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None (with a printed note) when no run can be drawn:
        the losses need the tasks' ``start_unixtime``, the driver steps a
        recorded step.

    Raises
    ------
    ValueError
        For an unknown ``breakdown``, ``"driver_steps"`` without
        ``timing_phases``, a ``source`` that is not a sequence of runs, a
        ``compute_resources`` without one entry per run, or a run without a
        recorded ``n_hosts``.
    """
    if breakdown not in ("losses", "driver_steps"):
        raise ValueError(
            f"breakdown must be 'losses' or 'driver_steps'; got {breakdown!r}."
        )
    if breakdown == "driver_steps" and timing_phases is None:
        raise ValueError(
            "breakdown='driver_steps' needs the application's timing_phases."
        )
    runs, resources, overrides = _runs_and_resources(
        source, compute_resources, "plot_efficiency_loss_decomposition"
    )
    n_hosts = [_n_hosts(r) for r in resources]
    if None in n_hosts:
        raise ValueError(
            "plot_efficiency_loss_decomposition: not every run records n_hosts "
            "in its compute resources; pass compute_resources."
        )

    if breakdown == "losses":
        measures = performance_measures(
            runs,
            overrides,
            timing_phases=timing_phases,
            task_columns=task_columns,
        )
        drawn = []
        for i in range(len(runs)):
            if np.isfinite(measures.at[i, "task_window"]):
                drawn.append(i)
            else:
                print(
                    f"plot_efficiency_loss_decomposition: skipping run {i} "
                    f"({n_hosts[i]} hosts): no start_unixtime anchor."
                )
        if not drawn:
            print("plot_efficiency_loss_decomposition: nothing to plot.")
            return None
        m = measures.loc[drawn]
        parts = [
            (m["work_map"] / m["n_worker_slots"]).to_numpy(dtype=float),
            *(
                m[key].to_numpy(dtype=float)
                for key in (
                    "idle_ramp",
                    "idle_in_window",
                    "idle_tail",
                    "pre_map",
                    "post_map",
                    "fixed_costs",
                )
            ),
        ]
        labels = [
            _scale_point_label(n_hosts[i], measures.at[i, "n_worker_slots"])
            for i in drawn
        ]
        x = np.arange(len(drawn))
        figure, (ax_time, ax_percent) = _performance_figure(
            (11, 10), 2, 1, sharex=True, gridspec_kw={"hspace": 0.06}
        )
        bottom = np.zeros(len(drawn))
        for (name, color), seconds in zip(_LOSS_PARTS, parts, strict=True):
            ax_time.bar(
                x, seconds / 3600, bottom=bottom / 3600, color=color, label=name
            )
            bottom += seconds
        ax_time.set_ylabel("Wall-clock time (hours)")
        ax_time.set_title(
            title
            or "Where the Wall Clock Goes: T_total split into useful work and "
            "loss sources"
        )
        ax_time.legend(fontsize=8)
        total = m["T_total"].to_numpy(dtype=float)
        bottom = np.zeros(len(drawn))
        for (name, color), seconds in zip(_LOSS_PARTS[1:], parts[1:], strict=True):
            percent = 100 * seconds / total
            ax_percent.bar(x, percent, bottom=bottom, color=color, label=name)
            bottom += percent
        ax_percent.set_ylabel("Losses (% of T_total; useful work excluded)")
        axes = (ax_time, ax_percent)
    else:
        from matplotlib import colormaps

        keys = [
            key
            for _, _, steps in timing_phases
            for _, key in steps
            if key != "T_compute_dask_graph"
        ]
        timings = [run.get("timing_distributed_application") or {} for run in runs]
        values = np.array(
            [
                [np.nan if t.get(key) is None else float(t[key]) for t in timings]
                for key in keys
            ]
        ).reshape(len(keys), len(runs))
        recorded = [k for k in range(len(keys)) if np.isfinite(values[k]).any()]
        if not recorded:
            print(
                "plot_efficiency_loss_decomposition: no run records a driver "
                "step of timing_phases; nothing to plot."
            )
            return None
        recorded.sort(key=lambda k: -np.nanmean(values[k]))
        labels = [
            _scale_point_label(n_hosts[i], resources[i].get("n_worker_slots"))
            for i in range(len(runs))
        ]
        x = np.arange(len(runs))
        figure, ax_steps = _performance_figure((11, 7))
        tab20 = colormaps["tab20"]
        bottom = np.zeros(len(runs))
        for i, k in enumerate(recorded):
            seconds = np.nan_to_num(values[k])
            ax_steps.bar(x, seconds, bottom=bottom, color=tab20(i % 20), label=keys[k])
            bottom += seconds
        ax_steps.set_ylabel("Time (s)")
        ax_steps.set_title(
            title
            or "Distributed-Application Fixed Costs per Scale Point "
            "(compute call excluded)"
        )
        ax_steps.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
        axes = (ax_steps,)
    axes[-1].set_xticks(x, labels)
    axes[-1].set_xlabel("Number of Nodes (total parallel workers)")
    for ax in axes:
        ax.grid(True, axis="y", color="lightgray")
        ax.set_axisbelow(True)
    _add_notes(figure, notes)
    _save_figures({breakdown: figure}, save_path)
    return figure
