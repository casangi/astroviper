"""Unit tests for astroviper.utils.resource_plots (synthetic monitored series,
no cluster, headless matplotlib)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from astroviper.utils.resource_plots import (
    _with_imaging_task_columns,
    plot_application_task_stream,
    plot_cluster_resource_usage,
    plot_core_resource_usage,
    plot_task_resource_usage,
    plot_task_stream,
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
    axes = plot_cluster_resource_usage(df, core_capacity=8, memory_capacity_gib=64)
    assert all(axes[k] is not None for k in ("cpu", "memory", "io"))
    # x-axis spans the whole run (last task starts at +15s and runs >=9.5s),
    # not just one task's duration.
    x = axes["cpu"].get_lines()[0].get_xdata()
    assert x[-1] >= 15.0
    # summed busy cores exceed any single task's cpu/100 while tasks overlap.
    busy = axes["cpu"].get_lines()[0].get_ydata()
    assert np.nanmax(busy) > 2.6  # > one task's ~2.5 cores -> real summation


def test_cluster_io_bin_means_conserve_bytes():
    df = _monitored_frame()
    axes = plot_cluster_resource_usage(df)
    lines = {line.get_label(): line for line in axes["io"].get_lines()}
    edges = [0.0]  # from the bin centres; the last bin may be shorter
    for centre in lines["write"].get_xdata():
        edges.append(2 * centre - edges[-1])
    width = np.diff(edges)
    for col in ("read_chars", "write_chars"):
        total_gib = sum(c[-1] - c[0] for c in df[col]) / 2**30
        plotted = (lines[col.split("_")[0]].get_ydata() * width).sum()
        assert plotted == pytest.approx(total_gib, rel=1e-9)


def test_cluster_cpu_ignores_the_first_sample_spike():
    """graphviper <= 0.0.58 took the first cpu_percent sample microseconds
    after psutil's baseline: 0 or up to ~15,000 %."""
    df = _monitored_frame()
    df.at[0, "cpu_percent"] = [15000.0] + df.at[0, "cpu_percent"][1:]
    busy = plot_cluster_resource_usage(df)["cpu"].get_lines()[0].get_ydata()
    assert np.nanmax(busy) <= 2.5 * len(df)  # every sample after the first <= 250 %


def test_cpu_seconds_preferred_over_cpu_percent():
    df = _monitored_frame(n_tasks=1)
    t = np.asarray(df.at[0, "time_seconds"])
    df["cpu_seconds"] = [(0.5 * t).tolist()]  # half a core throughout
    df.at[0, "cpu_percent"] = [999.0] * len(t)
    ax = plot_task_resource_usage(df)["cpu"]
    mean = ax.get_lines()[-1].get_ydata()
    assert np.nanmax(np.abs(mean - 50.0)) < 1e-9


def test_task_io_mean_holds_finished_tasks():
    df = _monitored_frame()
    ax = plot_task_resource_usage(df)["io"]
    write_mean = [
        line for line in ax.get_lines() if line.get_label().startswith("write: mean")
    ][0].get_ydata()
    assert np.all(np.diff(write_mean) >= 0)
    expected = np.mean([c[-1] - c[0] for c in df["write_chars"]]) / 2**30
    assert write_mean[-1] == pytest.approx(expected, rel=1e-9)


def test_cluster_plots_require_start_unixtime(capsys):
    assert plot_cluster_resource_usage(_monitored_frame(with_start=False)) is None
    assert "start_unixtime" in capsys.readouterr().out


def test_cluster_plots_skip_unmonitored(capsys):
    df = pd.DataFrame([{"task_id": 0, "T_image_cube_task": 1.0}])
    assert plot_cluster_resource_usage(df) is None
    assert "no resource series" in capsys.readouterr().out


def _two_slot_frame():
    """Two worker slots, 1 s samples: slot 1 runs two 9 s tasks at 100 % CPU,
    2 GiB and 1 MiB/s written, idle from 9 to 10 s; slot 2 runs one 19 s
    task at 50 % CPU, 4 GiB and 2 MiB/s written."""
    rows = []
    for pid, start, n, cores, gib, mib_s in (
        (1, 0.0, 10, 1.0, 2.0, 1.0),
        (1, 10.0, 10, 1.0, 2.0, 1.0),
        (2, 0.0, 20, 0.5, 4.0, 2.0),
    ):
        t = np.arange(float(n))
        rows.append(
            {
                "time_seconds": t.tolist(),
                "cpu_seconds": (cores * t).tolist(),
                "cpu_percent": [100.0 * cores] * n,
                "memory_rss_bytes": [gib * 2**30] * n,
                "read_chars": (3e9 + 0.5 * mib_s * 2**20 * t).tolist(),
                "write_chars": (7e9 + mib_s * 2**20 * t).tolist(),
                "sample_interval_seconds": 1.0,
                "start_unixtime": 1.75e9 + start,
                "hostname": "node",
                "process_pid": pid,
                "thread_native_id": 10 * pid,
            }
        )
    return pd.DataFrame(rows)


def _statistics(ax):
    return {line.get_label(): line.get_ydata() for line in ax.get_lines()}


def test_core_plots_statistics_over_the_slots(tmp_path):
    axes = plot_core_resource_usage(
        _two_slot_frame(), memory_per_core_gib=8, save_prefix=str(tmp_path / "x_")
    )
    for name in ("cpu", "memory", "io"):
        assert (tmp_path / f"x_core_{name}.png").exists()
    assert "2 worker slots" in axes["cpu"].get_title()
    x = axes["cpu"].get_lines()[0].get_xdata()
    gap = (x > 9) & (x < 10)  # slot 1 between its tasks
    assert gap.sum() == 1 and len(x) == 19
    cpu = _statistics(axes["cpu"])
    np.testing.assert_allclose(cpu["peak (max over cores)"], np.where(gap, 50, 100))
    np.testing.assert_allclose(cpu["mean"], np.where(gap, 25, 75))
    np.testing.assert_allclose(cpu["median"], np.where(gap, 25, 75))
    np.testing.assert_allclose(cpu["min"], np.where(gap, 0, 50))
    # memory only over the slots running a task: the idle slot is left out
    memory = _statistics(axes["memory"])
    np.testing.assert_allclose(memory["peak (max over cores)"], 4.0)
    np.testing.assert_allclose(memory["min"], np.where(gap, 4, 2))
    np.testing.assert_allclose(memory["mean"], np.where(gap, 4, 3))
    assert "memory per core (8 GiB)" in memory
    write = _statistics(axes["io"])
    np.testing.assert_allclose(write["min"], np.where(gap, 0, 1))
    np.testing.assert_allclose(write["peak (max over cores)"], 2.0)
    read = _statistics(axes["io"].figure.axes[1])
    np.testing.assert_allclose(read["mean"], np.where(gap, 0.5, 0.75))


def test_core_slots_from_interval_packing():
    df = _two_slot_frame().drop(columns=["process_pid", "thread_native_id"])
    axes = plot_core_resource_usage(df)
    assert "2 worker slots" in axes["cpu"].get_title()


def test_core_mean_times_slots_is_the_cluster_curve():
    df = _monitored_frame()
    core = plot_core_resource_usage(df)
    cluster = plot_cluster_resource_usage(df)
    n_slots = int(core["cpu"].get_title().split("(")[1].split()[0])
    busy = cluster["cpu"].get_lines()[0].get_ydata()
    mean = _statistics(core["cpu"])["mean"]
    np.testing.assert_allclose(mean * n_slots / 100, busy, rtol=1e-12)
    cluster_io = {
        line.get_label(): line.get_ydata() for line in cluster["io"].get_lines()
    }
    read = _statistics(core["io"].figure.axes[1])["mean"]
    np.testing.assert_allclose(read * n_slots / 1024, cluster_io["read"], rtol=1e-12)


def test_core_plots_bin_seconds():
    axes = plot_core_resource_usage(_two_slot_frame(), bin_seconds=4.6)
    x = axes["cpu"].get_lines()[0].get_xdata()
    np.testing.assert_allclose(x, [2.5, 7.5, 12.5, 17.0])  # 5 s bins, last 4 s
    assert "(5 s bin means)" in axes["cpu"].get_ylabel()


def test_core_plots_without_io_or_start(capsys):
    axes = plot_core_resource_usage(_monitored_frame(with_io=False))
    assert axes["cpu"] is not None and axes["io"] is None
    assert plot_core_resource_usage(_monitored_frame(with_start=False)) is None
    assert "plot_core_resource_usage: no start_unixtime" in capsys.readouterr().out


def test_rejects_wrong_source_type():
    with pytest.raises(TypeError, match="timing_node_tasks"):
        plot_task_resource_usage([1, 2, 3])


def test_task_stream_of_other_task_columns():
    """The task stream of a node task with its own timing columns (the
    simulator's): the total, a summed load segment and a write segment."""
    df = pd.DataFrame(
        {
            "task_id": [0, 1],
            "T_simulate_task": [4.0, 5.0],
            "T_uvw": [1.0, 1.5],
            "T_beams": [0.5, 0.5],
            "T_write": [0.5, 1.0],
            "start_unixtime": [1.75e9, 1.75e9 + 1.0],
            "hostname": ["node", "node"],
        }
    )
    columns = {
        "task": "T_simulate_task",
        "load": ["T_uvw", "T_beams"],
        "write": "T_write",
    }
    mapped = _with_imaging_task_columns(df, columns)
    np.testing.assert_allclose(mapped["T_image_cube_task"], [4.0, 5.0])
    np.testing.assert_allclose(mapped["T_load"], [1.5, 2.0])
    np.testing.assert_allclose(mapped["T_make_empty_image"], 0.0)
    np.testing.assert_allclose(mapped["T_write"], [0.5, 1.0])
    assert "T_image_cube_task" not in df  # the caller's frame is not changed
    no_write = _with_imaging_task_columns(df, {**columns, "load": (), "write": None})
    np.testing.assert_allclose(no_write[["T_load", "T_write"]], 0.0)

    figure = plot_task_stream(df, task_columns=columns)
    assert figure is not None
    plt.close(figure)
    with pytest.raises(KeyError):  # the imager's columns by default
        plot_task_stream(df)


def test_application_task_stream_plots_every_graph(tmp_path):
    """The application view retains and labels one worker stream per graph."""
    graph_one = _monitored_frame(n_tasks=4)
    graph_two = _monitored_frame(n_tasks=3)
    graph_one["hostname"] = "worker"
    graph_two["hostname"] = "worker"
    graph_two["start_unixtime"] = graph_two["start_unixtime"].map(
        lambda value: value + 40
    )
    finalization = _monitored_frame(n_tasks=1)
    finalization["hostname"] = "worker"
    finalization["start_unixtime"] += 80
    result = {
        "timing_graphs": [
            {
                "stage": "global weight density",
                "stage_kind": "weighting",
                "timing_node_tasks": graph_one,
                "compute_start_unixtime": graph_one.start_unixtime.min() - 1,
                "compute_end_unixtime": graph_one.start_unixtime.max() + 25,
                "T_compute": 41.0,
            },
            {
                "stage": "imaging cycle 1 (residual + model update)",
                "stage_kind": "imaging_cycle",
                "timing_node_tasks": graph_two,
                "compute_start_unixtime": graph_two.start_unixtime.min() - 2,
                "compute_end_unixtime": graph_two.start_unixtime.max() + 20,
                "T_compute": 28.0,
            },
            {
                "stage": "restoration",
                "stage_kind": "final",
                "includes_restoration": True,
                "timing_finalization": finalization,
                "compute_start_unixtime": finalization.start_unixtime.min() - 1,
                "compute_end_unixtime": finalization.start_unixtime.max() + 10,
                "T_compute": 11.0,
            },
        ]
    }
    output = tmp_path / "application_task_stream.png"
    figure = plot_application_task_stream(result, save_path=output)
    assert figure is not None
    assert len(figure.axes) == 3
    assert "global weight density" in figure.axes[0].get_title(loc="left")
    assert "imaging cycle 1" in figure.axes[1].get_title(loc="left")
    assert "restoration" in figure.axes[2].get_title(loc="left")
    assert "reduce / gather" in {text.get_text() for text in figure.axes[0].texts}
    assert "reduce + model update" in {text.get_text() for text in figure.axes[1].texts}
    assert output.exists()


def test_application_task_stream_keeps_old_results_optional(capsys):
    """Results recorded before timing_graphs fail gracefully."""
    assert (
        plot_application_task_stream({"timing_node_tasks": _monitored_frame()}) is None
    )
    assert "no timing_graphs" in capsys.readouterr().out
