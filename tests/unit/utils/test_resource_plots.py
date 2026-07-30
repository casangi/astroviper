"""Unit tests for astroviper.utils.resource_plots (synthetic monitored series,
no cluster, headless matplotlib)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from astroviper.utils.resource_plots import (
    plot_cluster_resource_usage,
    plot_task_resource_usage,
)


def _monitored_frame(n_tasks=6, with_io=True, with_start=True, interval=0.5):
    """Frame shaped like timing_node_tasks from a monitored run: staggered
    tasks with list-valued series columns."""
    rng = np.random.default_rng(7)
    rows = []
    for tid in range(n_tasks):
        n = int(rng.integers(20, 40))
        t = np.arange(n) * interval
        row = {
            "task_id": tid,
            "T_image_cube_task": float(t[-1]),
            "time_seconds": t.tolist(),
            "cpu_percent": (150 + 100 * rng.random(n)).tolist(),
            "memory_rss_bytes": (1e9 + 4e9 * rng.random(n)).tolist(),
            "sample_interval_seconds": interval,
        }
        if with_io:
            # cumulative over the worker process: nonzero baseline
            row["read_chars"] = (5e9 + np.cumsum(rng.random(n)) * 1e8).tolist()
            row["write_chars"] = (9e9 + np.cumsum(rng.random(n)) * 5e7).tolist()
        if with_start:
            row["start_unixtime"] = 1.75e9 + tid * 3.0  # staggered starts
        rows.append(row)
    return pd.DataFrame(rows)


def test_task_plots_all_three_series():
    axes = plot_task_resource_usage(_monitored_frame())
    assert all(axes[k] is not None for k in ("cpu", "memory", "io"))
    # I/O is rebased per task: the plotted lines start at ~0, not at the
    # multi-GB process-cumulative baseline.
    io_lines = axes["io"].get_lines()
    assert min(line.get_ydata()[0] for line in io_lines) < 0.1


def test_task_plots_accept_return_dict_and_save(tmp_path):
    ret = {"timing_node_tasks": _monitored_frame(), "deconvolution": {}}
    axes = plot_task_resource_usage(ret, save_prefix=str(tmp_path / "x_"))
    assert axes["cpu"] is not None
    for name in ("cpu", "memory", "io"):
        assert (tmp_path / f"x_task_{name}.png").exists()


def test_task_plots_without_io_columns():
    axes = plot_task_resource_usage(_monitored_frame(with_io=False))
    assert axes["cpu"] is not None and axes["io"] is None


def test_cluster_plots_sum_over_running_tasks():
    df = _monitored_frame()
    axes = plot_cluster_resource_usage(df, core_capacity=8, memory_capacity_gb=64)
    assert all(axes[k] is not None for k in ("cpu", "memory", "io"))
    # x-axis spans the whole run (last task starts at +15s and runs >=9.5s),
    # not just one task's duration.
    x = axes["cpu"].get_lines()[0].get_xdata()
    assert x[-1] >= 15.0
    # summed busy cores exceed any single task's cpu/100 while tasks overlap.
    busy = axes["cpu"].get_lines()[0].get_ydata()
    assert np.nanmax(busy) > 2.6  # > one task's ~2.5 cores -> real summation


def test_cluster_plots_require_start_unixtime(capsys):
    assert plot_cluster_resource_usage(_monitored_frame(with_start=False)) is None
    assert "start_unixtime" in capsys.readouterr().out


def test_cluster_plots_skip_unmonitored(capsys):
    df = pd.DataFrame([{"task_id": 0, "T_image_cube_task": 1.0}])
    assert plot_cluster_resource_usage(df) is None
    assert "no resource series" in capsys.readouterr().out


def test_rejects_wrong_source_type():
    with pytest.raises(TypeError, match="timing_node_tasks"):
        plot_task_resource_usage([1, 2, 3])
