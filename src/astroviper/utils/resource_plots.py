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

Two complementary views:

* :func:`plot_task_resource_usage` -- tasks aligned at their OWN start
  (t=0 = task start): a subsample of individual task curves plus the average
  over the tasks still running at each t. "What does a typical task do?"
* :func:`plot_cluster_resource_usage` -- tasks placed at their wall-clock
  position via ``start_unixtime`` and SUMMED over the tasks running at each
  moment: busy cores, total RSS, aggregate I/O rate. "What was the whole
  cluster doing at each moment of the run?"

Both accept the full return dict or the ``timing_node_tasks`` frame directly,
and skip gracefully (with a printed note) when the needed columns are absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["plot_task_resource_usage", "plot_cluster_resource_usage"]


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
    import matplotlib.pyplot as plt

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
