"""Plots of the per-task resource-usage series recorded by graphviper's monitor.

When a distributed application is run with ``monitor_resources_seconds`` set
(e.g. :func:`~astroviper.distributed_applications.imaging.image_cube_single_field.image_cube_single_field`),
its ``return_dict["timing_node_tasks"]`` frame carries one sampled series per
node task as list-valued columns: ``time_seconds`` (relative to the task's own
start), ``cpu_percent`` (whole worker process incl. OpenMP threads, so >100%
is normal), ``memory_rss_bytes``, the cumulative syscall-level I/O counters
``read_chars`` / ``write_chars`` (Linux only; unlike ``read_bytes`` /
``write_bytes`` these count network filesystems such as Lustre), the scalar
``sample_interval_seconds`` and the wall-clock anchor ``start_unixtime``.

Three complementary views:

* :func:`plot_task_resource_usage` -- tasks aligned at their OWN start
  (t=0 = task start): a subsample of individual task curves plus the average
  over the tasks still running at each t. "What does a typical task do?"
* :func:`plot_cluster_resource_usage` -- tasks placed at their wall-clock
  position via ``start_unixtime`` and SUMMED over the tasks running at each
  moment: busy cores, total RSS, aggregate I/O rate. "What was the whole
  cluster doing at each moment of the run?"
* :func:`plot_task_stream` -- a Dask-dashboard-style per-worker-process
  timeline: one lane per worker slot, each task drawn as load / science /
  write segments, plus a running-task utilization panel and a printed
  efficiency decomposition. Unlike the two views above it needs only the
  wall-clock anchor ``start_unixtime`` (recorded by the node task itself),
  NOT the monitor's sampled series, so it works with
  ``monitor_resources_seconds=None``.

All accept the full return dict or the ``timing_node_tasks`` frame directly,
and skip gracefully (with a printed note) when the needed columns are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "plot_task_resource_usage",
    "plot_cluster_resource_usage",
    "plot_task_stream",
    "assign_task_stream_lanes",
]

# Task-stream palette (colorblind-safe Okabe-Ito).
_LOAD_COLOR, _SCIENCE_COLOR, _WRITE_COLOR = "#0072B2", "#009E73", "#E69F00"
_POST_COLOR = "#CC79A7"
_PRE_COLOR = "#7f7f7f"


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
    average over ALL tasks (at each t, over the tasks still running at t).

    ``subtract_first`` rebases each task's series to its first sample -- needed
    for the I/O counters, which are cumulative over the WORKER PROCESS lifetime
    (a task run late in a sweep starts far into its worker's counter).

    The y-axis is clipped to the 99.5th percentile of the plotted values (or
    the average's peak if larger): psutil's first cpu_percent samples over a
    near-zero interval can spike absurdly and would flatten the whole plot.
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
        mask = grid <= t[-1]
        total[mask] += np.interp(grid[mask], t, v)
        count[mask] += 1
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    ax.plot(
        grid,
        mean,
        color=color,
        linewidth=2.5,
        label=f"{label_prefix}average over running tasks ({len(df)} tasks)",
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

    Every task's clock starts at its own beginning; the bold curve is the
    average over the tasks still running at each instant, so it is not diluted
    by tasks that already finished.

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
        "Resident memory (GB)",
        "Node-task memory usage",
        scale=1 / 1e9,
        max_task_lines=max_task_lines,
    )
    # I/O counters are cumulative over the worker process -> rebase per task.
    io_title = "Node-task I/O (syscall-level, counts network filesystems)"
    axes["io"] = _plot_task_series(
        df,
        "read_chars",
        "I/O since task start (GB)",
        io_title,
        scale=1 / 1e9,
        color="tab:blue",
        label_prefix="read: ",
        max_task_lines=max_task_lines,
        subtract_first=True,
    )
    if axes["io"] is not None:
        _plot_task_series(
            df,
            "write_chars",
            "I/O since task start (GB)",
            io_title,
            scale=1 / 1e9,
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
    source, core_capacity=None, memory_capacity_gb=None, save_prefix=None
):
    """Cluster-wide CPU / memory / I/O over the RUN's wall clock.

    Each task's series is placed at its actual wall-clock position using
    ``start_unixtime`` and the quantities are SUMMED over the tasks running at
    each instant: busy cores (sum of cpu_percent/100), total RSS of
    task-running worker processes (idle workers are not counted), and the
    aggregate read/write rate (GB/s, from the per-task derivative of the
    cumulative counters). Every figure also shows the number of concurrently
    running tasks on a right-hand axis.

    Parameters
    ----------
    source : dict or pandas.DataFrame
        The application return dict, or its ``timing_node_tasks`` frame.
        Needs the ``start_unixtime`` column (graphviper monitor >= 2026-07-13);
        cross-node placement relies on NTP-synced node clocks.
    core_capacity : int, optional
        Total cores of the allocation; drawn as a dashed capacity line.
    memory_capacity_gb : float, optional
        Total memory of the allocation (GB); drawn as a dashed capacity line.
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
    if "cpu_percent" not in df.columns:
        print(
            "plot_cluster_resource_usage: no resource series recorded "
            "(monitoring off or psutil missing); skipping."
        )
        return None
    if "start_unixtime" not in df.columns or df["start_unixtime"].isna().all():
        print(
            "plot_cluster_resource_usage: no start_unixtime column (recorded "
            "with an older graphviper); tasks cannot be placed on the run's "
            "wall clock -- use plot_task_resource_usage instead."
        )
        return None

    interval = float(df["sample_interval_seconds"].iloc[0])
    starts = df["start_unixtime"].to_numpy(dtype=float)
    rel_end = np.array([t[-1] for t in df["time_seconds"]], dtype=float)
    run_t0 = starts.min()
    grid = np.arange(0.0, (starts + rel_end).max() - run_t0 + interval, interval)

    have_io = "read_chars" in df.columns and not df["read_chars"].isna().all()
    acc = {k: np.zeros_like(grid) for k in ("cores", "rss", "read_rate", "write_rate")}
    n_running = np.zeros_like(grid)

    for i, row in df.iterrows():
        t_rel = np.asarray(row["time_seconds"], dtype=float)
        offset = starts[i] - run_t0
        mask = (grid >= offset) & (grid <= offset + t_rel[-1])
        g = grid[mask] - offset
        n_running[mask] += 1
        acc["cores"][mask] += (
            np.interp(g, t_rel, np.asarray(row["cpu_percent"], dtype=float)) / 100.0
        )
        acc["rss"][mask] += (
            np.interp(g, t_rel, np.asarray(row["memory_rss_bytes"], dtype=float)) / 1e9
        )
        if have_io:
            for col, key in (
                ("read_chars", "read_rate"),
                ("write_chars", "write_rate"),
            ):
                c = np.asarray(row[col], dtype=float)
                rate = np.gradient(c, t_rel) / 1e9 if len(c) > 1 else np.zeros(1)
                acc[key][mask] += np.interp(g, t_rel, rate)

    def _figure(title, ylabel, curves, capacity=None):
        fig, ax = plt.subplots(figsize=(11, 6))
        for label, y, color in curves:
            ax.plot(grid, y, color=color, linewidth=1.5, label=label)
        if capacity is not None:
            ax.axhline(capacity[1], color="gray", linestyle="--", label=capacity[0])
        ax2 = ax.twinx()
        ax2.plot(grid, n_running, color="lightgray", linewidth=1, zorder=0)
        ax2.set_ylabel("Running tasks", color="gray")
        ax.set_xlabel("Wall-clock time since first task start (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, color="lightgray", alpha=0.5)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.legend(loc="upper right")
        return ax

    axes = {}
    axes["cpu"] = _figure(
        "Cluster CPU usage over the run",
        "Busy cores (sum of task CPU% / 100)",
        [("busy cores", acc["cores"], "tab:blue")],
        capacity=(
            None
            if core_capacity is None
            else (f"capacity ({core_capacity} cores)", core_capacity)
        ),
    )
    axes["memory"] = _figure(
        "Cluster memory usage over the run (task-running workers only)",
        "Total resident memory (GB)",
        [("total RSS", acc["rss"], "tab:blue")],
        capacity=(
            None
            if memory_capacity_gb is None
            else (f"capacity ({memory_capacity_gb:g} GB)", memory_capacity_gb)
        ),
    )
    if have_io:
        axes["io"] = _figure(
            "Cluster I/O rate over the run (syscall-level, counts network filesystems)",
            "Aggregate I/O rate (GB/s)",
            [
                ("read", acc["read_rate"], "tab:blue"),
                ("write", acc["write_rate"], "tab:orange"),
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


def _write_task_stream_html(
    t,
    hosts,
    n_lanes,
    n_workers,
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
):
    """Write a standalone interactive HTML task stream: the same two panels as
    the matplotlib figure, drawn as inline SVG with a JavaScript tooltip that
    shows a bar's timing details on hover. Pure SVG + vanilla JS -- no
    extra dependencies, works offline. One <rect> per task segment, so the
    file grows with the task count (~a few MB for a 15k-task run)."""
    import html as _html

    # ---- geometry (pixels) ----
    left, top, bottom = 95, 30, 42
    inner_w = 1500
    util_h, gap = 110, 16
    lane_px = max(3.0, min(14.0, 9000.0 / max(n_lanes, 1)))
    stream_h = lane_px * n_lanes
    width = left + inner_w + 20
    height = top + util_h + gap + stream_h + bottom

    x0 = min(pre_rel, 0.0) if pre_rel is not None else 0.0
    x1 = (
        post_end_rel
        if post_end_rel is not None
        else max(T_compute or 0.0, makespan, float(t["end"].max()))
    )
    x1 *= 1.005
    sx = inner_w / (x1 - x0)

    def X(sec):
        return left + (sec - x0) * sx

    def Y(row):  # stream panel: row 0 at the bottom, like the matplotlib figure
        return top + util_h + gap + stream_h - (row + 1) * lane_px

    out = [
        f'<!DOCTYPE html><html><head><meta charset="utf-8">'
        f"<title>{_html.escape(title or 'task stream')}</title>",
        """<style>
body { font-family: sans-serif; margin: 12px; }
rect[data-i]:hover { stroke: #000; stroke-width: 0.8px; }
#tt { position: fixed; display: none; background: #fffef5; border: 1px solid
      #888; border-radius: 3px; padding: 5px 8px; font-size: 12px;
      pointer-events: none; box-shadow: 2px 2px 5px rgba(0,0,0,0.25);
      white-space: pre; z-index: 10; }
.axis { font-size: 11px; fill: #333; }
.host { font-size: 9px; fill: #333; }
.note { font-size: 11px; fill: #555; }
</style></head><body>""",
        f"<h3 style='margin:4px 0'>{_html.escape(title or 'task stream')}</h3>",
        f"<div style='font-size:13px;color:#444;margin-bottom:6px'>"
        f"ideal {ideal:.0f} s | task window {makespan:.0f} s"
        + (f" | T_compute {T_compute:.0f} s" if T_compute else "")
        + " &mdash; hover a bar for that task's timing</div>",
        f'<svg width="{width}" height="{height:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg">',
    ]

    # Pre/post shaded regions, spanning both panels.
    panels_y0, panels_y1 = top, top + util_h + gap + stream_h

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
    r_max = max(float(np.max(running)), float(n_workers))

    def uy(v):
        return top + util_h - v / (1.08 * r_max) * util_h

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
        f'<line x1="{left}" y1="{uy(n_workers):.1f}" x2="{left + inner_w}" '
        f'y2="{uy(n_workers):.1f}" stroke="#666" stroke-dasharray="5,4"/>'
    )
    out.append(
        f'<text class="note" x="{left + 4}" y="{uy(n_workers) - 4:.1f}">'
        f"{n_workers} workers</text>"
    )
    out.append(
        f'<text class="axis" x="{left - 8}" y="{top + util_h / 2:.0f}" '
        f'text-anchor="end">running<tspan x="{left - 8}" dy="12">tasks'
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
            f'y="{Y(rows.mean()) + lane_px / 2 + 3:.1f}" '
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
            f'<text class="axis" x="{X(v):.1f}" y="{ax_y + 18:.1f}" '
            f'text-anchor="middle">{v + 0.0:g}</text>'
        )
        v += tick
    out.append(
        f'<text class="axis" x="{left + inner_w / 2:.0f}" '
        f'y="{ax_y + 34:.1f}" text-anchor="middle">'
        f"time since first task start (s)</text>"
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
            f'<rect x="{lx}" y="8" width="12" height="12" fill="{color}"'
            f' fill-opacity="0.8"/>'
        )
        out.append(
            f'<text class="axis" x="{lx + 16}" y="18">{_html.escape(label)}</text>'
        )
        lx += 16 + 7 * len(label) + 22

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
    reduce_frame=None,
    compute_start=None,
    compute_end=None,
    T_compute=None,
    n_workers=None,
    anchor_source=None,
    title=None,
    save_path=None,
    html_path=None,
):
    """Dask-dashboard-style task stream: per-worker-process timeline of a run.

    Reconstructs one lane per worker slot from the per-task wall-clock anchor
    ``start_unixtime`` and the ``T_*`` phase durations, draws each task as
    load (blue) / science (green) / write (orange) segments -- white gaps are
    worker slots with nothing to do -- with a running-task utilization panel
    on top, and prints the efficiency decomposition (ideal time, ramp-up,
    straggler tail, pre/post-map regions). Works for both compute backends.

    Parameters
    ----------
    source : dict or pandas.DataFrame
        The application return dict, or its ``timing_node_tasks`` frame.
        Needs the ``start_unixtime`` column (recorded by the node task since
        2026-07-28 -- with or without resource monitoring); cross-node
        placement relies on NTP-synced node clocks. When the full return dict
        is given, the reduce-node rows (``timing_reduce_nodes``) and the
        compute-window anchors (``timing_distributed_application``'s
        ``compute_start_unixtime`` / ``compute_end_unixtime`` /
        ``T_compute_dask_graph``) are picked up automatically.
    reduce_frame : pandas.DataFrame, optional
        Reduce-node timing rows (``hostname``, ``start_unixtime``,
        ``end_unixtime`` + optional execution identity); overrides the ones
        from the return dict. Reduce nodes are drawn on the worker lane they
        actually ran on.
    compute_start, compute_end : float, optional
        Unixtime of the compute call's start/end; overrides the return dict's
        anchors. These expose the PRE-first-task gap (worker imports + graph
        submission) and the POST-last-task region that a first-task-relative
        plot misses.
    T_compute : float, optional
        Wall time of the compute call (seconds); overrides the return dict's
        ``T_compute_dask_graph``.
    n_workers : int, optional
        Known worker-slot count of the cluster; default reconstructs it by
        packing the map tasks into lanes.
    anchor_source : str, optional
        Provenance note for the compute anchors, echoed in the printout.
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

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None when ``start_unixtime`` is absent (run recorded
        before the anchor existed).
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
    if isinstance(source, dict):
        if reduce_frame is None and source.get("timing_reduce_nodes"):
            reduce_frame = pd.DataFrame(source["timing_reduce_nodes"])
        timing = source.get("timing_distributed_application") or {}
        if compute_start is None and timing.get("compute_start_unixtime") is not None:
            compute_start = float(timing["compute_start_unixtime"])
            compute_end = float(timing["compute_end_unixtime"])
            anchor_source = anchor_source or "measured"
        if T_compute is None:
            T_compute = timing.get("T_compute_dask_graph")

    pre_label = "pre-map: worker imports + graph submit"
    post_label = "post-map: reduce residue + gather/save"
    # Fold measured reduce nodes (when recorded) into the lane assignment: they
    # run on the same worker processes (dask) / the manager (MPI), one at a
    # time, so they share the per-host lane pool with the map tasks.
    tasks = tasks.copy()
    tasks["kind"] = "map"
    map_only = tasks
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
    tm = t[t["kind"] == "map"]
    tr = t[t["kind"] == "reduce"]
    makespan = tm["end"].max()
    # Worker count: recorded geometry when known, else a MAP-ONLY packing pass
    # (in the combined packing, reduce nodes borrow idle map lanes and bump map
    # tasks into overflow lanes, so combined lane counts overstate the workers).
    n_workers = n_workers or assign_task_stream_lanes(map_only)[2]
    ideal = tm["T_image_cube_task"].sum() / n_workers

    # Utilization curve from start/end events (map tasks only; reduce gets its
    # own curve below when recorded).
    events = np.concatenate([tm["start"].values, tm["end"].values])
    deltas = np.concatenate([np.ones(len(tm)), -np.ones(len(tm))])
    order = np.argsort(events, kind="stable")
    ev_t, running = events[order], np.cumsum(deltas[order])

    # Efficiency decomposition. With absolute compute anchors the
    # PRE-first-task and POST-last-task shares of T_compute are measured;
    # otherwise fall back to attributing everything after the window.
    busy = tm["T_image_cube_task"].sum()
    in_window_loss = makespan - ideal
    t0_abs = tasks["start_unixtime"].min()
    pre_rel = post_end_rel = None
    if compute_start is not None:
        pre_rel = compute_start - t0_abs  # negative seconds
        post_end_rel = compute_end - t0_abs
    post = (T_compute - makespan) if T_compute else None
    if title:
        print(f"run: {title}")
    print(
        f"  workers                    : {n_workers} ({len(hosts)} hosts, "
        f"{n_lanes} reconstructed lanes)"
    )
    print(f"  ideal time (sum/workers)   : {ideal:8.1f} s")
    print(
        f"  task-window makespan       : {makespan:8.1f} s "
        f"(+{in_window_loss:.1f} s ramp/gaps/tail, "
        f"{100 * busy / (n_workers * makespan):.1f}% busy inside window)"
    )
    if T_compute:
        if pre_rel is not None:
            print(
                f"  T_compute (map+reduce)     : {T_compute:8.1f} s = "
                f"{-pre_rel:.1f} s PRE first task ({pre_label}) + "
                f"{makespan:.1f} s task window + "
                f"{post_end_rel - makespan:.1f} s POST ({post_label})"
                + (f"   [anchors: {anchor_source}]" if anchor_source else "")
            )
        else:
            print(
                f"  T_compute (map+reduce)     : {T_compute:8.1f} s "
                f"(+{post:.1f} s outside the task window)"
            )
        print(f"  end-to-end efficiency      : {100 * ideal / T_compute:.1f}% of ideal")
    # Ramp and tail: time to reach 95% of workers / time the last 5% of tasks
    # spend after the 95th-percentile end. (argmax = first True; searchsorted
    # is invalid here -- the running count is not monotonic.)
    reach = running >= 0.95 * n_workers
    ramp = float(ev_t[int(np.argmax(reach))]) if reach.any() else np.nan
    tail = makespan - np.quantile(tm["end"], 0.95)
    print(f"  ramp to 95% busy           : {ramp:8.1f} s")
    print(f"  straggler tail (p95->last) : {tail:8.1f} s")
    if len(tr):
        busy_r = tr["T_image_cube_task"].sum()
        inside = np.clip(np.minimum(tr["end"], makespan) - tr["start"], 0, None)
        overlap = float(inside.sum() / busy_r) if busy_r else 0.0
        print(
            f"  reduce nodes (measured)    : {len(tr)} calls, {busy_r:8.1f} s "
            f"busy, window {tr['start'].min():.1f}-{tr['end'].max():.1f} s, "
            f"{100 * overlap:.0f}% of reduce busy inside the map window"
        )

    # ---------------- figure ---------------------------------------------- #
    fig, (ax_u, ax) = plt.subplots(
        2,
        1,
        figsize=(15, 13),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.04},
    )

    ax_u.fill_between(ev_t, running, step="post", color=_LOAD_COLOR, alpha=0.35, lw=0)
    ax_u.plot(ev_t, running, drawstyle="steps-post", color=_LOAD_COLOR, lw=1)
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
        ax_u.legend(loc="upper right", fontsize=8, frameon=False)
    ax_u.axhline(n_workers, color="#666666", ls="--", lw=1)
    ax_u.text(
        0, n_workers, f" {n_workers} workers", va="bottom", fontsize=8, color="#666666"
    )
    if pre_rel is not None:
        ax_u.axvspan(pre_rel, 0, color=_PRE_COLOR, alpha=0.20)
        ax_u.axvspan(makespan, post_end_rel, color=_POST_COLOR, alpha=0.25)
    elif T_compute:
        ax_u.axvspan(makespan, T_compute, color=_POST_COLOR, alpha=0.25)
        ax_u.text(
            0.5 * (makespan + T_compute),
            0.5 * n_workers,
            f"{post_label}\n{post:.0f} s",
            ha="center",
            fontsize=8,
            color="#8a4a6d",
        )
    ax_u.set_ylabel("running tasks")
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
        fontsize=6,
    )
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1, n_lanes)
    left = pre_rel * 1.02 if pre_rel is not None else 0
    right = (
        post_end_rel
        if post_end_rel is not None
        else max(T_compute or 0, makespan, t["end"].max())
    )
    ax.set_xlim(left, right * 1.005)
    ax.set_ylabel(f"worker processes ({n_lanes} lanes, grouped by host)")
    ax.set_xlabel("time since first task start (s)")
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
        handles=handles, loc="lower right", fontsize=8, frameon=True, framealpha=0.9
    )
    fig.suptitle(
        f"{title or 'task stream'}\n"
        f"ideal {ideal:.0f} s | task window {makespan:.0f} s | "
        + (f"T_compute {T_compute:.0f} s" if T_compute else ""),
        fontsize=12,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("wrote", save_path)
    if html_path is not None:
        _write_task_stream_html(
            t,
            hosts,
            n_lanes,
            n_workers,
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
        )
    return fig
