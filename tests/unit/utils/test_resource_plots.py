"""Unit tests for astroviper.utils.resource_plots (synthetic monitored series,
no cluster, headless matplotlib)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from astroviper.utils.resource_plots import (
    _add_notes,
    _band_or_auto,
    _band_or_minimum_span,
    _binned_maximum,
    _binned_task_series,
    _compute_resources,
    _driver_steps,
    _driver_timeline,
    _figure_save_paths,
    _host_worker_layout,
    _n_hosts,
    _n_worker_slots,
    _node_units_axis,
    _non_io_task_time,
    _over_slots,
    _parallelism_ticks,
    _per_task_totals,
    _performance_figure,
    _running_task_count,
    _save_figures,
    _shared_cumulative,
    _task_start_bins,
    _value_ticks,
    _wall_clock_figure,
    _with_imaging_task_columns,
    _worker_share_divisors,
    performance_measures,
    plot_application_task_stream,
    plot_cluster_resource_usage,
    plot_core_resource_usage,
    plot_efficiency_loss_decomposition,
    plot_scaling,
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
    axes = plot_task_resource_usage(
        ret, save_path=str(tmp_path / "x_task_{figure}.png")
    )
    assert axes["cpu"] is not None
    for name in ("cpu", "memory", "io"):
        assert (tmp_path / f"x_task_{name}.png").exists()


def test_task_plots_without_io_columns():
    axes = plot_task_resource_usage(_monitored_frame(with_io=False))
    assert axes["cpu"] is not None and axes["io"] is None


def test_cluster_plots_sum_over_running_tasks():
    df = _monitored_frame()
    axes = plot_cluster_resource_usage(
        df,
        compute_resources={
            "n_hosts": 1,
            "n_logical_cpus": 8,
            "memory_total_bytes": 64 * 2**30,
        },
    )
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


def test_cluster_plots_require_start_unixtime(logged):
    assert plot_cluster_resource_usage(_monitored_frame(with_start=False)) is None
    assert "start_unixtime" in logged.read()


def test_cluster_plots_skip_unmonitored(logged):
    df = pd.DataFrame([{"task_id": 0, "T_image_cube_task": 1.0}])
    assert plot_cluster_resource_usage(df) is None
    assert "no resource series" in logged.read()


def _slot_frame(tasks):
    """One row per task (pid, worker name or None, start s, samples, cores,
    GiB, MiB/s written; half that read), 1 s samples on one host."""
    rows = []
    for pid, worker, start, n, cores, gib, mib_s in tasks:
        t = np.arange(float(n))
        row = {
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
        if worker is not None:
            row["worker_name"] = worker
        rows.append(row)
    return pd.DataFrame(rows)


def _two_slot_frame():
    """Two worker slots: slot 1 runs two 9 s tasks at 100 % CPU, 2 GiB and
    1 MiB/s written, idle from 9 to 10 s; slot 2 runs one 19 s task at 50 %
    CPU, 4 GiB and 2 MiB/s written."""
    return _slot_frame(
        [
            (1, None, 0.0, 10, 1.0, 2.0, 1.0),
            (1, None, 10.0, 10, 1.0, 2.0, 1.0),
            (2, None, 0.0, 20, 0.5, 4.0, 2.0),
        ]
    )


def _statistics(ax):
    return {line.get_label(): line.get_ydata() for line in ax.get_lines()}


def test_over_slots():
    stats = _over_slots(np.array([[1.0, np.nan], [2.0, np.nan], [9.0, np.nan]]))
    for key, expected in (("peak", 9), ("mean", 4), ("median", 2), ("min", 1)):
        np.testing.assert_allclose(stats[key], [expected, np.nan])


def test_core_plots_statistics_over_the_slots(tmp_path):
    axes = plot_core_resource_usage(
        _two_slot_frame(),
        compute_resources={"n_worker_slots": 2, "memory_total_bytes": 16 * 2**30},
        save_path=str(tmp_path / "x_core_{figure}.png"),
    )
    for name in ("cpu", "memory", "io"):
        assert (tmp_path / f"x_core_{name}.png").exists()
    assert "(2 slots)" in axes["cpu"].get_title()
    x = axes["cpu"].get_lines()[0].get_xdata()
    gap = (x > 9) & (x < 10)  # slot 1 between its tasks
    assert gap.sum() == 1 and len(x) == 19
    cpu = _statistics(axes["cpu"])
    np.testing.assert_allclose(cpu["peak (max over slots)"], np.where(gap, 50, 100))
    np.testing.assert_allclose(cpu["mean"], np.where(gap, 25, 75))
    np.testing.assert_allclose(cpu["min"], np.where(gap, 0, 50))
    # memory only over the slots running a task: the idle slot is left out
    memory = _statistics(axes["memory"])
    np.testing.assert_allclose(memory["peak (max over slots)"], 4.0)
    np.testing.assert_allclose(memory["min"], np.where(gap, 4, 2))
    np.testing.assert_allclose(memory["mean"], np.where(gap, 4, 3))
    assert "host memory per worker slot (8 GiB)" in memory
    write = _statistics(axes["io"])
    np.testing.assert_allclose(write["min"], np.where(gap, 0, 1))
    np.testing.assert_allclose(write["peak (max over slots)"], 2.0)
    read = _statistics(axes["io"].figure.axes[1])
    np.testing.assert_allclose(read["mean"], np.where(gap, 0.5, 0.75))


def test_core_slots_from_interval_packing():
    df = _two_slot_frame().drop(columns=["process_pid", "thread_native_id"])
    axes = plot_core_resource_usage(df)
    assert "(2 slots)" in axes["cpu"].get_title()


def test_core_slots_keep_a_restarted_worker():
    """Worker w2's process is restarted at 10 s (new pid): still one slot."""
    df = _slot_frame(
        [
            (1, "w1", 0.0, 10, 1.0, 2.0, 1.0),
            (1, "w1", 10.0, 10, 1.0, 2.0, 1.0),
            (2, "w2", 0.0, 10, 0.5, 4.0, 2.0),
            (3, "w2", 10.0, 10, 0.5, 4.0, 2.0),
        ]
    )
    axes = plot_core_resource_usage(df)
    assert "(2 slots)" in axes["cpu"].get_title()
    x = axes["cpu"].get_lines()[0].get_xdata()
    gap = (x > 9) & (x < 10)  # both workers between their tasks
    np.testing.assert_allclose(_statistics(axes["cpu"])["min"], np.where(gap, 0, 50))
    by_pid = plot_core_resource_usage(df.drop(columns="worker_name"))
    assert "(3 slots)" in by_pid["cpu"].get_title()


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
    # slot 1 runs 4 s of the bin [5, 10): its memory is the mean over those 4 s
    cpu, memory = _statistics(axes["cpu"]), _statistics(axes["memory"])
    np.testing.assert_allclose(cpu["peak (max over slots)"], [100, 80, 100, 100])
    np.testing.assert_allclose(cpu["mean"], [75, 65, 75, 75])
    np.testing.assert_allclose(memory["min"], 2.0)
    np.testing.assert_allclose(memory["mean"], 3.0)
    # a width beyond half the run is capped there: two bins, not one
    axes = plot_core_resource_usage(_two_slot_frame(), bin_seconds=100)
    np.testing.assert_allclose(axes["cpu"].get_lines()[0].get_xdata(), [5.0, 14.5])
    assert "(10 s bin means)" in axes["cpu"].get_ylabel()


def test_core_plots_without_io_or_start(logged):
    axes = plot_core_resource_usage(_monitored_frame(with_io=False))
    assert axes["cpu"] is not None and axes["io"] is None
    assert plot_core_resource_usage(_monitored_frame(with_start=False)) is None
    assert "plot_core_resource_usage: no start_unixtime" in logged.read()


def test_wall_clock_plots_skip_tasks_without_start(logged):
    df = _two_slot_frame()
    df.loc[1, "start_unixtime"] = np.nan
    assert plot_core_resource_usage(df) is None
    assert plot_cluster_resource_usage(df) is None
    assert logged.read().count("1 of 3 tasks have no start_unixtime") == 2


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


def test_application_task_stream_keeps_old_results_optional(logged):
    """Results recorded before timing_graphs fail gracefully."""
    assert (
        plot_application_task_stream({"timing_node_tasks": _monitored_frame()}) is None
    )
    assert "no timing_graphs" in logged.read()


# --- driver strip ---------------------------------------------------------- #

#: A driver layout: two setup steps, the compute call and one finalize step.
DRIVER_PHASES = [
    (
        "DRIVER",
        None,
        [
            ("open input", "T_open"),
            ("build graph", "T_graph"),
            ("compute dask graph", "T_compute_dask_graph"),
            ("consolidate metadata", "T_consolidate_metadata"),
        ],
    )
]


def test_driver_steps_split_at_the_compute_step():
    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES as IMAGER_PHASES,
    )
    from astroviper.distributed_applications.simulation.simulate_processing_set import (
        DISTRIBUTED_APPLICATION_TIMING_PHASES as SIMULATOR_PHASES,
    )

    pre, post = _driver_steps(IMAGER_PHASES)
    labels = [label for label, _ in pre]
    assert {"input preflight", "add lazy input data", "get compute resources"} <= set(
        labels
    )
    assert post == [
        ("write find_continuum first round", "T_find_continuum_first_round_write"),
        ("drain find_continuum moments", "T_find_continuum_first_round_drain"),
        ("consolidate metadata", "T_consolidate_metadata"),
    ]
    pre, post = _driver_steps(SIMULATOR_PHASES)
    assert "write sky image" in [label for label, _ in pre]
    assert [label for label, _ in post] == [
        "consolidate metadata + schema check",
        "write MSv2 (arcae)",
    ]
    with pytest.raises(ValueError, match="no T_compute_dask_graph"):
        _driver_steps([("DRIVER", None, [("open input", "T_open")])])


def _driver_timing():
    """Driver timing whose steps account for the whole call: setup 5 s before
    the compute start at 1000 s, the compute call 1000-1020 s, 2 s after it."""
    return {
        "T_open": 2.0,
        "T_graph": 3.0,
        "T_compute_dask_graph": 20.0,
        "T_consolidate_metadata": 2.0,
        "compute_start_unixtime": 1000.0,
        "compute_end_unixtime": 1020.0,
        "application_start_unixtime": 995.0,
        "application_end_unixtime": 1022.0,
        "T_total": 27.0,
    }


def test_driver_timeline_names_every_step():
    timing = _driver_timing()
    segments, start, end, colors, detail = _driver_timeline(
        timing, 1000.0, 1020.0, 1001.0, DRIVER_PHASES
    )
    assert (start, end) == (-6.0, 21.0)
    assert [label for label, _, _ in segments] == [
        "setup",
        "compute (task stream below)",
        "consolidate metadata",
    ]
    assert detail == [("open input", 2.0), ("build graph", 3.0)]
    assert "driver (untimed)" not in colors
    assert _driver_timeline(timing, 1000.0, 1020.0, 1001.0, None) is None


def _stream_dict():
    """Four tasks on two worker slots of one host, a reduce node and the
    driver timing (t0 = 990, t1 = 1000, t2 = 1022, t3 = 1025)."""
    starts = [1000.0, 1001.0, 1010.0, 1013.0]
    durations = [10.0, 12.0, 10.0, 9.0]
    tasks = pd.DataFrame(
        {
            "task_id": range(4),
            "T_image_cube_task": durations,
            "T_make_empty_image": 0.5,
            "T_load": 1.0,
            "T_write": 2.0,
            "start_unixtime": starts,
            "hostname": "node",
            "process_pid": [1, 2, 1, 2],
            "thread_native_id": [1, 2, 1, 2],
        }
    )
    return {
        "timing_node_tasks": tasks,
        "timing_reduce_nodes": [
            {
                "hostname": "node",
                "start_unixtime": 1022.0,
                "end_unixtime": 1024.0,
                "process_pid": 1,
                "thread_native_id": 1,
            }
        ],
        "timing_distributed_application": {
            "T_open": 2.0,
            "T_graph": 3.0,
            "T_compute_dask_graph": 30.0,
            "T_consolidate_metadata": 1.0,
            "compute_start_unixtime": 995.0,
            "compute_end_unixtime": 1025.0,
            "application_start_unixtime": 990.0,
            "application_end_unixtime": 1030.0,
            "T_total": 40.0,
        },
        "compute_resources": {"n_hosts": 1, "n_workers": 2, "n_worker_slots": 2},
    }


def test_task_stream_without_timing_phases_has_no_driver_strip(logged):
    figure = plot_task_stream(_stream_dict())
    assert len(figure.axes) == 2
    assert "no timing_phases given" in logged.read()
    plt.close(figure)
    figure = plot_task_stream(_stream_dict(), timing_phases=DRIVER_PHASES)
    assert len(figure.axes) == 3
    out = logged.read()
    assert "no timing_phases given" not in out
    assert "E end-to-end" in out and "application total" in out
    plt.close(figure)


def test_task_stream_logs_the_recorded_or_packed_parallelism(logged, tmp_path):
    html = tmp_path / "stream.html"
    figure = plot_task_stream(
        _stream_dict(), compute_resources={"n_worker_slots": 4}, html_path=str(html)
    )
    out = logged.read()
    assert "4 worker slots (recorded; 1 host, 2 reconstructed lanes)" in out
    assert "E map-reduce               :     43.0 %" in out  # 43 / 25 / 4
    assert "p = 4 worker slots" in [
        text.get_text() for text in figure.axes[0].get_legend().get_texts()
    ]
    assert "p = 4 worker slots" in html.read_text()
    plt.close(figure)
    recorded = _stream_dict()
    del recorded["compute_resources"]
    figure = plot_task_stream(recorded)
    out = logged.read()
    assert "2 worker slots (lane packing" in out
    assert "ramp to 95% busy           :      1.0 s" in out
    assert "straggler tail (p95->last) :      0.3 s" in out
    plt.close(figure)


def test_task_stream_refuses_tasks_without_start(logged):
    source = _stream_dict()
    source["timing_node_tasks"].loc[1, "start_unixtime"] = np.nan
    assert plot_task_stream(source) is None
    assert "1 of 4 tasks have no start_unixtime" in logged.read()


def test_imaging_task_columns_leave_out_absent_load_columns(logged):
    tasks = _stream_dict()["timing_node_tasks"]
    mapped = _with_imaging_task_columns(
        tasks, {"load": ("T_load", "T_stream_read"), "write": None}
    )
    assert "no T_stream_read column; left out of the load segment" in (logged.read())
    np.testing.assert_allclose(mapped["T_load"], 1.0)
    np.testing.assert_allclose(mapped["T_write"], 0.0)
    # the imager's load columns, T_make_empty_image included, sum before it is zeroed
    mapped = _with_imaging_task_columns(tasks, {})
    np.testing.assert_allclose(mapped["T_load"], 1.5)
    np.testing.assert_allclose(mapped["T_make_empty_image"], 0.0)


# --- performance_measures --------------------------------------------------- #


def test_performance_measures_of_one_run():
    m = performance_measures(_stream_dict())
    expected = {
        "n_worker_slots": 2,
        "n_map_tasks": 4,
        "n_reduce_tasks": 1,
        "work_map": 41.0,
        "work_reduce": 2.0,
        "work": 43.0,
        "ideal_time": 21.5,
        "task_time_median": 10.0,
        "task_time_max": 12.0,
        "application_start_unixtime": 990.0,
        "first_task_start_unixtime": 1000.0,
        "map_end_unixtime": 1022.0,
        "compute_start_unixtime": 995.0,
        "compute_end_unixtime": 1025.0,
        "fixed_compute_shift": 0.0,
        "task_window": 22.0,
        "map_reduce_window": 25.0,
        "end_to_end_window": 35.0,
        "driver_setup_before_first_task": 10.0,
        "pre_map": 5.0,
        "post_map": 3.0,
        "T_compute_dask_graph": 30.0,
        "T_total": 40.0,
        "fixed_costs": 10.0,
        "map_busy_fraction": 41.0 / 44.0,
        "time_to_95_percent_busy": 1.0,
        "p95_task_end": 21.7,
        "idle_ramp": 0.5,
        "idle_in_window": 0.85,
        "idle_tail": 0.15,
        "speedup_map_reduce": 43.0 / 25.0,
        "speedup_end_to_end": 43.0 / 35.0,
        "parallel_efficiency_map_reduce": 43.0 / 50.0,
        "parallel_efficiency_end_to_end": 43.0 / 70.0,
    }
    assert list(m) == [
        *list(expected)[:1],
        "n_worker_slots_reconstructed",
        *list(expected)[1:10],
        "application_start_reconstructed",
        *list(expected)[10:],
    ]
    for key, value in expected.items():
        assert m[key] == pytest.approx(value, rel=1e-12), key
    assert m["n_worker_slots_reconstructed"] is False
    assert m["application_start_reconstructed"] is False
    # the map work over p and the idle time make up the task window
    assert m["work_map"] / 2 + m["idle_ramp"] + m["idle_in_window"] + m[
        "idle_tail"
    ] == pytest.approx(m["task_window"])
    packed = _stream_dict()
    packed["compute_resources"] = None
    m = performance_measures(packed)
    assert m["n_worker_slots"] == 2 and m["n_worker_slots_reconstructed"] is True


def test_performance_measures_rebuild_the_application_start():
    source = _stream_dict()
    del source["timing_distributed_application"]["application_start_unixtime"]
    m = performance_measures(source, timing_phases=DRIVER_PHASES)
    assert m["application_start_unixtime"] == 990.0  # 995 - (2 + 3)
    assert m["application_start_reconstructed"] is True
    m = performance_measures(source)
    assert np.isnan(m["end_to_end_window"]) and np.isnan(m["speedup_end_to_end"])
    assert m["parallel_efficiency_map_reduce"] == pytest.approx(0.86)


def test_performance_measures_with_a_fixed_non_io_compute():
    m = performance_measures(_stream_dict(), fixed_non_io_compute=4.0)
    assert m["work_map"] == pytest.approx(4 * (0.5 + 1.0 + 2.0) + 4 * 4.0)
    assert m["fixed_compute_shift"] == pytest.approx((41.0 - 30.0) / 2)
    assert m["map_reduce_window"] == pytest.approx(25.0 - 5.5)
    assert m["map_busy_fraction"] == pytest.approx(41.0 / 44.0)  # measured work
    m = performance_measures(
        _stream_dict(), task_columns={"load": ("T_load",)}, fixed_non_io_compute=4.0
    )
    assert m["work_map"] == pytest.approx(4 * 3.0 + 16.0)


def test_performance_measures_without_start_unixtime():
    source = _stream_dict()
    source["timing_node_tasks"] = source["timing_node_tasks"].drop(
        columns="start_unixtime"
    )
    m = performance_measures(source)
    assert m["n_worker_slots"] == 2 and m["work"] == 43.0
    for key in ("task_window", "map_reduce_window", "pre_map", "idle_tail"):
        assert np.isnan(m[key]), key
    assert m["T_total"] == 40.0
    source["compute_resources"] = {}
    m = performance_measures(source)
    assert np.isnan(m["n_worker_slots"]) and np.isnan(m["ideal_time"])
    assert m["n_worker_slots_reconstructed"] is False
    m = performance_measures(_stream_dict()["timing_node_tasks"])  # a bare frame
    assert m["n_worker_slots"] == 2 and m["pre_map"] == 0.0
    assert np.isnan(m["post_map"]) and np.isnan(m["map_reduce_window"])


def test_performance_measures_of_several_runs():
    four_slots = _stream_dict()
    four_slots["compute_resources"] = {"n_worker_slots": 4}
    more_work = _stream_dict()
    more_work["timing_reduce_nodes"] = more_work["timing_reduce_nodes"] * 2
    unusable = _stream_dict()
    del unusable["timing_distributed_application"]["compute_end_unixtime"]
    runs = [four_slots, _stream_dict(), more_work, unusable]
    strong = performance_measures(runs)
    assert isinstance(strong, pd.DataFrame) and list(strong.index) == [0, 1, 2, 3]
    # baseline: the first usable run of the lowest p (run 1)
    np.testing.assert_allclose(
        strong["speedup_map_reduce_relative"], [2.0, 2.0, 2.0, np.nan]
    )
    np.testing.assert_allclose(
        strong["parallel_efficiency_map_reduce_relative"], [0.5, 1.0, 1.0, np.nan]
    )
    np.testing.assert_allclose(strong["speedup_end_to_end_relative"][:3], 2.0)
    # weak: the first usable run, the ideal window scaled by W / W_b
    weak = performance_measures(runs, scaling="weak")
    np.testing.assert_allclose(
        weak["speedup_map_reduce_relative"], [4.0, 4.0, 4.0 * 45.0 / 43.0, np.nan]
    )
    overridden = performance_measures(
        [_stream_dict(), _stream_dict()],
        compute_resources=[None, {"n_worker_slots": 8}],
    )
    assert list(overridden["n_worker_slots"]) == [2, 8]


def test_performance_measures_refuse():
    with pytest.raises(ValueError, match="scaling"):
        performance_measures(_stream_dict(), scaling="linear")
    with pytest.raises(ValueError, match="sequence aligned with"):
        performance_measures([_stream_dict()], compute_resources={"n_worker_slots": 2})
    with pytest.raises(ValueError, match="sequence aligned with"):
        performance_measures([_stream_dict()], compute_resources=[None, None])
    source = _stream_dict()
    source["compute_resources"] = {"parallelism": 2}
    with pytest.raises(ValueError, match="unknown keys"):
        performance_measures(source)


# --- plot_scaling and plot_efficiency_loss_decomposition -------------------- #


def _scaling_runs():
    """Three runs of _stream_dict (W = 43 s, map reduce window 25 s, end to
    end window 35 s, T_total 40 s) on 1, 2 and 4 hosts of two worker slots;
    the 2-host run is the same run at half speed: every time doubles."""
    runs = []
    for n_hosts in (1, 2, 4):
        run = _stream_dict()
        run["compute_resources"] = {
            "n_hosts": n_hosts,
            "n_workers": 2 * n_hosts,
            "n_worker_slots": 2 * n_hosts,
        }
        run["timing_node_tasks"]["n_channels"] = 3
        runs.append(run)
    tasks = runs[1]["timing_node_tasks"]
    for column in ("T_image_cube_task", "T_make_empty_image", "T_load", "T_write"):
        tasks[column] *= 2
    tasks["start_unixtime"] = 1000.0 + 2 * (tasks["start_unixtime"] - 1000.0)
    for reduce_node in runs[1]["timing_reduce_nodes"]:
        for key in ("start_unixtime", "end_unixtime"):
            reduce_node[key] = 1000.0 + 2 * (reduce_node[key] - 1000.0)
    timing = runs[1]["timing_distributed_application"]
    for key, value in timing.items():
        timing[key] = 1000.0 + 2 * (value - 1000.0) if "unixtime" in key else 2 * value
    return runs


def _line_data(ax):
    """``{label: xy}`` of the lines of ``ax``."""
    return {line.get_label(): line.get_xydata() for line in ax.get_lines()}


def _bar_data(ax):
    """``{label: [(y, height), ...]}`` of the bar groups of ``ax``."""
    return {
        bars.get_label(): [(bar.get_y(), bar.get_height()) for bar in bars]
        for bars in ax.containers
    }


def test_scaling_total_time(tmp_path, logged):
    open_figures = plt.get_fignums()
    runs = _scaling_runs()
    figure = plot_scaling(
        runs,
        "total_time",
        sweep_labels=["a", "a", "b"],
        reference_points=[
            {"label": "CASA", "x": [4], "total_time": [7200.0]},
            {"label": "two points", "x": [1, 4], "total_time": [3600.0, 1800.0]},
        ],
        notes=["occupancy 90 %"],
        save_path=str(tmp_path / "x_{figure}.png"),
    )
    assert plt.get_fignums() == open_figures  # no pyplot figure
    ax = figure.axes[0]
    lines = _line_data(ax)
    np.testing.assert_allclose(lines["a"], [[1, 40 / 3600], [2, 80 / 3600]])
    np.testing.assert_allclose(lines["b"], [[4, 40 / 3600]])
    np.testing.assert_allclose(lines["CASA"], [[4, 2.0]])
    np.testing.assert_allclose(lines["two points"], [[1, 1.0], [4, 0.5]])
    styles = {line.get_label(): line for line in ax.get_lines()}
    assert (
        styles["CASA"].get_marker() == "X" and styles["CASA"].get_linestyle() == "None"
    )
    assert styles["two points"].get_linestyle() == "--"
    assert ax.get_ylabel() == "Total Time (hours)" and ax.get_title() == "Total Time"
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "1\n2",
        "2\n4",
        "4\n8",
    ]
    assert ax.get_xlabel() == "Number of Nodes (total parallel workers)"
    assert [text.get_text() for text in figure.texts] == ["occupancy 90 %"]
    assert (tmp_path / "x_total_time.png").exists()
    assert "wrote" in logged.read()
    # the harness wall clock, minutes, weak scaling from 0, value ticks
    figure = plot_scaling(
        runs[::2],
        "total_time",
        x=[3840, 15360],
        scaling="weak",
        total_time=[120.0, np.nan],
        time_unit="minutes",
        x_label="Number of Channels",
        x_tick_labels="values",
    )
    ax = figure.axes[0]
    np.testing.assert_allclose(_line_data(ax)["measured"], [[3840, 2.0]])
    assert ax.get_ylim()[0] == 0
    assert [label.get_text() for label in ax.get_xticklabels()] == ["3,840", "15,360"]
    assert ax.get_xlabel() == "Number of Channels"


def test_scaling_total_time_with_a_fixed_non_io_compute():
    runs = _scaling_runs()
    shift = performance_measures(runs, fixed_non_io_compute=4.0)["fixed_compute_shift"]
    figure = plot_scaling(
        runs, "total_time", time_unit="seconds", fixed_non_io_compute=4.0
    )
    np.testing.assert_allclose(
        _line_data(figure.axes[0])["measured"][:, 1], [40.0, 80.0, 40.0] - shift
    )


def test_scaling_speedup_and_efficiency_of_one_sweep():
    runs = _scaling_runs()
    measures = performance_measures(runs)
    figure = plot_scaling(runs, "speedup")
    ax = figure.axes[0]
    lines = ax.get_lines()
    assert [line.get_label() for line in lines] == [
        r"map reduce: $S^{mr} = W/(t_3-t_1)$",
        r"end to end: $S^{e2e} = W/(t_3-t_0)$",
        r"ideal ($S = p$)",
    ]
    np.testing.assert_allclose(lines[0].get_ydata(), measures["speedup_map_reduce"])
    np.testing.assert_allclose(lines[1].get_ydata(), measures["speedup_end_to_end"])
    np.testing.assert_allclose(lines[2].get_xydata(), [[1, 2], [2, 4], [4, 8]])
    assert [line.get_color() for line in lines] == ["C0", "C1", "black"]
    assert ax.get_ylabel() == "Speedup (worker units)"
    assert ax.child_axes[0].get_ylabel() == "Speedup (node units, 2 workers/node)"
    figure = plot_scaling(runs, "parallel_efficiency_relative")
    ax = figure.axes[0]
    lines = ax.get_lines()
    np.testing.assert_allclose(lines[0].get_ydata(), [1.0, 0.25, 0.25])
    np.testing.assert_allclose(lines[2].get_xydata(), [[0, 1], [1, 1]])
    assert lines[2].get_label() == r"ideal ($E = 1$)"
    assert ax.get_ylim() == pytest.approx((min(0.75, 0.95 * 0.25), 1.02))
    assert ax.get_title() == "Parallel efficiency: relative (baseline-normalized)"


def test_scaling_weak_and_several_sweeps():
    runs = _scaling_runs()
    figure = plot_scaling(
        runs,
        "speedup_relative",
        x=[10, 20, 30],
        scaling="weak",
        x_tick_labels="values",
    )
    ax = figure.axes[0]
    lines = ax.get_lines()
    assert lines[0].get_label().startswith(r"map reduce: $S^{mr,rel} = p_0\,(W/W_0)")
    weak = performance_measures(runs, scaling="weak")["speedup_map_reduce_relative"]
    np.testing.assert_allclose(lines[0].get_ydata(), weak)
    plotted = [*weak, *lines[1].get_ydata(), 2, 4, 8]
    assert ax.get_ylim() == pytest.approx((0.95 * min(plotted), 1.02 * max(plotted)))
    figure = plot_scaling(runs, "speedup_relative", sweep_labels=["a", "b", "a"])
    ax = figure.axes[0]
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert list(lines) == [
        r"a, map reduce ($S^{mr,rel}$)",
        r"a, end to end ($S^{e2e,rel}$)",
        r"b, map reduce ($S^{mr,rel}$)",
        r"b, end to end ($S^{e2e,rel}$)",
        r"ideal ($S = p$)",
    ]
    # relative to each sweep's own baseline
    np.testing.assert_allclose(lines[r"a, map reduce ($S^{mr,rel}$)"].get_ydata(), 2.0)
    np.testing.assert_allclose(lines[r"b, map reduce ($S^{mr,rel}$)"].get_ydata(), 4.0)
    color = {label: to_hex(line.get_color()) for label, line in lines.items()}
    assert color[r"a, map reduce ($S^{mr,rel}$)"] == to_hex("C0")
    assert color[r"a, end to end ($S^{e2e,rel}$)"] == to_hex("C0")
    assert lines[r"a, end to end ($S^{e2e,rel}$)"].get_linestyle() == "--"
    assert color[r"b, map reduce ($S^{mr,rel}$)"] == to_hex("C1")
    assert lines[r"b, end to end ($S^{e2e,rel}$)"].get_marker() == "s"


def test_scaling_skips_unusable_runs(logged):
    runs = _scaling_runs()
    del runs[2]["timing_distributed_application"]["compute_end_unixtime"]
    figure = plot_scaling(runs, "parallel_efficiency")
    lines = figure.axes[0].get_lines()
    np.testing.assert_allclose(lines[0].get_xdata(), [1, 2])
    # the ticks name every run
    assert len(figure.axes[0].get_xticklabels()) == 3
    for run in runs:
        run["timing_node_tasks"] = run["timing_node_tasks"].drop(
            columns="start_unixtime"
        )
    assert plot_scaling(runs, "speedup") is None
    assert "no run has a finite map reduce window" in logged.read()
    for run in runs:
        run["timing_distributed_application"]["T_total"] = np.nan
    assert plot_scaling(runs, "total_time") is None
    assert "no run has a finite total time" in logged.read()


def test_scaling_throughput(logged):
    open_figures = plt.get_fignums()
    runs = _scaling_runs()
    figure = plot_scaling(runs, "throughput", total_time=[100.0, 50.0, 25.0])
    assert plt.get_fignums() == open_figures
    ax_throughput, ax_cost = figure.axes
    np.testing.assert_allclose(
        ax_throughput.get_lines()[0].get_xydata(), [[1, 0.12], [2, 0.24], [4, 0.48]]
    )
    np.testing.assert_allclose(
        ax_cost.get_lines()[0].get_xydata(),
        [[1, 100 / 3600], [2, 100 / 3600], [4, 100 / 3600]],
    )
    assert ax_cost.get_lines()[0].get_color() == "tab:orange"
    assert ax_throughput.get_ylim()[0] == 0 and ax_cost.get_ylim()[0] == 0
    assert ax_throughput.get_title() == "Delivered Throughput and Allocation Cost"
    figure = plot_scaling(runs, "throughput", sweep_labels=["a", "b", "b"])
    ax_throughput, ax_cost = figure.axes
    assert list(_line_data(ax_throughput)) == ["a", "b"]
    np.testing.assert_allclose(
        _line_data(ax_cost)["b"], [[2, 160 / 3600], [4, 160 / 3600]]
    )
    for run in runs:
        run["timing_node_tasks"] = run["timing_node_tasks"].drop(columns="n_channels")
    assert plot_scaling(runs, "throughput") is None
    assert "no run has the n_channels column" in logged.read()


def test_scaling_refuses():
    runs = _scaling_runs()
    with pytest.raises(ValueError, match="quantity must be one of"):
        plot_scaling(runs, "speedup_task_sum")
    with pytest.raises(ValueError, match="time_unit"):
        plot_scaling(runs, "total_time", time_unit="days")
    with pytest.raises(ValueError, match="x_tick_labels"):
        plot_scaling(runs, "total_time", x_tick_labels="parallelism")
    with pytest.raises(ValueError, match="sequence of return dicts"):
        plot_scaling(runs[0], "total_time")
    with pytest.raises(ValueError, match="holds no run"):
        plot_scaling([], "total_time")
    with pytest.raises(ValueError, match="x must be a sequence with one entry"):
        plot_scaling(runs, "total_time", x=[1, 2])
    with pytest.raises(ValueError, match="compute_resources must be a sequence"):
        plot_scaling(runs, "total_time", compute_resources={"n_hosts": 1})
    with pytest.raises(ValueError, match="scaling"):
        plot_scaling(runs, "speedup", scaling="linear")
    no_hosts = [None, None, {"n_worker_slots": 8}]
    with pytest.raises(ValueError, match="x defaults to the n_hosts"):
        plot_scaling(runs, "speedup", compute_resources=no_hosts)
    with pytest.raises(ValueError, match="the node-hours need the n_hosts"):
        plot_scaling(runs, "throughput", x=[1, 2, 3], compute_resources=no_hosts)
    # with x given the speedup needs no host count; no node-units axis then
    figure = plot_scaling(runs, "speedup", x=[1, 2, 3], compute_resources=no_hosts)
    assert not figure.axes[0].child_axes
    np.testing.assert_allclose(figure.axes[0].get_lines()[2].get_ydata(), [2, 4, 8])


def test_efficiency_loss_decomposition_losses(tmp_path, logged):
    open_figures = plt.get_fignums()
    runs = _scaling_runs()
    del runs[2]["timing_node_tasks"]["start_unixtime"]
    figure = plot_efficiency_loss_decomposition(
        runs, notes="occupancy", save_path=str(tmp_path / "loss.png")
    )
    assert plt.get_fignums() == open_figures
    assert "skipping run 2 (4 hosts): no start_unixtime anchor" in (logged.read())
    assert (tmp_path / "loss.png").exists()
    ax_time, ax_percent = figure.axes
    measures = performance_measures(runs[:2])
    parts = {
        "useful work": measures["work_map"] / measures["n_worker_slots"],
        "ramp": measures["idle_ramp"],
        "in-window gaps": measures["idle_in_window"],
        "straggler tail": measures["idle_tail"],
        "pre-map": measures["pre_map"],
        "post-map": measures["post_map"],
        "fixed costs": measures["fixed_costs"],
    }
    bars = _bar_data(ax_time)
    assert list(bars) == list(parts)
    bottom = np.zeros(2)
    for name, seconds in parts.items():
        np.testing.assert_allclose(bars[name], np.c_[bottom / 3600, seconds / 3600])
        bottom += seconds
    np.testing.assert_allclose(bottom, measures["T_total"])  # the parts are T_total
    # useful work and idle time make up the task window
    np.testing.assert_allclose(
        sum(np.asarray(parts[name]) for name in list(parts)[:4]),
        measures["task_window"],
    )
    percent = _bar_data(ax_percent)
    assert list(percent) == list(parts)[1:]
    np.testing.assert_allclose(
        np.array(percent["fixed costs"])[:, 1], 100 * np.array([10.0, 20.0]) / [40, 80]
    )
    assert [label.get_text() for label in ax_percent.get_xticklabels()] == [
        "1\n2",
        "2\n4",
    ]
    assert ax_percent.get_xlabel() == "Number of Nodes (total parallel workers)"
    assert ax_time.get_title().startswith("Where the Wall Clock Goes")
    for run in runs[:2]:
        del run["timing_node_tasks"]["start_unixtime"]
    assert plot_efficiency_loss_decomposition(runs) is None
    assert "nothing to plot" in logged.read()


def test_efficiency_loss_decomposition_driver_steps(logged):
    runs = _scaling_runs()
    del runs[2]["timing_distributed_application"]["T_graph"]
    runs[2]["timing_distributed_application"]["T_ideal"] = 21.5
    phases = [
        (*DRIVER_PHASES[0][:2], [*DRIVER_PHASES[0][2], ("never", "T_never")]),
    ]
    figure = plot_efficiency_loss_decomposition(
        runs,
        compute_resources=[None, None, {"n_hosts": 4}],
        breakdown="driver_steps",
        timing_phases=phases,
    )
    ax = figure.axes[0]
    bars = _bar_data(ax)
    # largest mean at the bottom; no compute call, no T_ideal, no unrecorded step
    assert list(bars) == ["T_graph", "T_open", "T_consolidate_metadata"]
    np.testing.assert_allclose(bars["T_graph"], [[0, 3], [0, 6], [0, 0]])
    np.testing.assert_allclose(bars["T_open"], [[3, 2], [6, 4], [0, 2]])
    np.testing.assert_allclose(
        bars["T_consolidate_metadata"], [[5, 1], [10, 2], [2, 1]]
    )
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "1\n2",
        "2\n4",
        "4",
    ]
    assert ax.get_ylabel() == "Time (s)"
    assert ax.get_title().startswith("Distributed-Application Fixed Costs")
    for run in runs:
        run["timing_distributed_application"] = {}
    assert (
        plot_efficiency_loss_decomposition(
            runs, breakdown="driver_steps", timing_phases=phases
        )
        is None
    )
    assert "no run records a driver step" in logged.read()


def test_efficiency_loss_decomposition_refuses():
    runs = _scaling_runs()
    with pytest.raises(ValueError, match="breakdown must be"):
        plot_efficiency_loss_decomposition(runs, breakdown="phases")
    with pytest.raises(ValueError, match="needs the application's timing_phases"):
        plot_efficiency_loss_decomposition(runs, breakdown="driver_steps")
    with pytest.raises(ValueError, match="not every run records n_hosts"):
        plot_efficiency_loss_decomposition(
            runs, compute_resources=[{"n_worker_slots": 2}, None, None]
        )


# --- shared helpers ---------------------------------------------------------- #


def test_running_task_count():
    times, running = _running_task_count([0.0, 1.0, 10.0], [10.0, 13.0, 20.0])
    np.testing.assert_allclose(times, [0, 1, 10, 10, 13, 20])
    np.testing.assert_allclose(running, [1, 2, 3, 2, 1, 0])  # starts before ends


def test_non_io_task_time():
    tasks = _stream_dict()["timing_node_tasks"]
    np.testing.assert_allclose(_non_io_task_time(tasks), [6.5, 8.5, 6.5, 5.5])
    tasks.loc[0, "T_load"] = np.nan  # skipped in the load sum
    columns = {"task": "T_image_cube_task", "load": ("T_load", "T_absent")}
    np.testing.assert_allclose(_non_io_task_time(tasks, columns), [8.0, 9.0, 7.0, 6.0])


def test_task_start_bins():
    tasks = pd.DataFrame({"start_unixtime": 1.75e9 + 900.0 * np.arange(9)})
    values = np.arange(9.0)
    values[8] = np.nan  # left out
    bins = _task_start_bins(tasks, values, 4)
    assert list(bins.columns) == ["bin", "hours", "fraction", "median"]
    np.testing.assert_allclose(bins["median"], [0.5, 2.5, 4.5, 6.5])
    np.testing.assert_allclose(bins["hours"], [0.125, 0.625, 1.125, 1.625])
    np.testing.assert_allclose(bins["fraction"], [0.125, 0.375, 0.625, 0.875])
    assert _task_start_bins(tasks, np.full(9, np.nan), 4).empty


def test_per_task_totals():
    frame = _slot_frame([(1, None, 0.0, 10, 1.0, 2.0, 1.0)])
    frame = pd.concat([frame, frame.iloc[:1].assign(time_seconds=[[0.0]])])
    frame = frame.reset_index(drop=True)
    frame.at[1, "read_chars"] = None
    totals = _per_task_totals(frame)
    assert list(totals.columns) == [
        "cpu_seconds",
        "mean_cpu_percent",
        "peak_rss_gib",
        "read_gib",
        "written_gib",
    ]
    assert totals.loc[0, "cpu_seconds"] == pytest.approx(9.0)
    assert totals.loc[0, "mean_cpu_percent"] == pytest.approx(100.0)
    assert totals.loc[0, "peak_rss_gib"] == pytest.approx(2.0)
    assert totals.loc[0, "written_gib"] == pytest.approx(9.0 / 1024)
    assert totals.loc[0, "read_gib"] == pytest.approx(4.5 / 1024)
    assert np.isnan(totals.loc[1, "cpu_seconds"]) and np.isnan(
        totals.loc[1, "read_gib"]
    )
    assert totals.loc[1, "peak_rss_gib"] == pytest.approx(2.0)


def test_compute_resources_resolvers():
    recorded = {"n_hosts": 2, "n_workers": 4, "n_worker_slots": 8}
    source = {"compute_resources": recorded}
    assert _compute_resources(source, None) is recorded
    override = {"n_worker_slots": 3}
    assert _compute_resources(source, override) == override  # replaces, no merge
    assert _compute_resources(pd.DataFrame(), None) == {}
    assert _n_hosts(recorded) == 2 and _n_hosts(override) is None
    assert _n_worker_slots(recorded, None) == (8, "recorded")
    tasks = _stream_dict()["timing_node_tasks"]
    assert _n_worker_slots({}, tasks) == (2, "lane packing")
    assert _host_worker_layout(recorded, "any") == (2.0, 4.0)
    hosts = {"a": {"n_workers": 1, "n_worker_slots": 14}, "b": {"n_workers": 2}}
    assert _host_worker_layout({"hosts": hosts}, "a") == (1, 14)
    assert _host_worker_layout({"hosts": hosts}, "b") is None
    assert _host_worker_layout({"hosts": hosts}, "c") is None
    assert _host_worker_layout(override, "a") is None


def test_shared_cumulative():
    np.testing.assert_allclose(_shared_cumulative([5.0, 7.0, 11.0], 1.0), [0, 2, 6])
    np.testing.assert_allclose(
        _shared_cumulative([5.0, 7.0, 11.0], np.array([1.0, 2.0, 4.0])), [0, 1, 2]
    )


def _shared_worker_frame():
    """Three tasks of one 1 s sampled worker process on one host: two run
    together for 10 s, the third alone afterwards."""
    return _slot_frame(
        [
            (1, None, 0.0, 11, 2.0, 4.0, 2.0),
            (1, None, 0.0, 11, 2.0, 4.0, 2.0),
            (1, None, 10.0, 11, 1.0, 2.0, 1.0),
        ]
    )


def test_worker_share_divisors(logged):
    df = _shared_worker_frame()
    one_worker = {"n_hosts": 1, "n_workers": 1, "n_worker_slots": 14}
    divisors = _worker_share_divisors(df, one_worker)
    np.testing.assert_allclose(divisors[0], [2] * 10 + [3])  # 3 at the 10 s handover
    np.testing.assert_allclose(divisors[2], [3] + [1] * 10)
    by_host = {"hosts": {"node": {"n_workers": 1, "n_worker_slots": 2}}}
    np.testing.assert_allclose(_worker_share_divisors(df, by_host)[0], [2] * 11)
    assert (
        _worker_share_divisors(
            df, {"n_hosts": 1, "n_workers": 14, "n_worker_slots": 14}
        )
        is None
    )
    assert logged.read() == ""
    assert _worker_share_divisors(df, {"n_workers": 1, "n_worker_slots": 14}) is None
    assert "14 task threads share each worker process" in logged.read()
    assert _worker_share_divisors(df.drop(columns="start_unixtime"), one_worker) is None
    assert "start_unixtime" in logged.read()


def test_binned_series_count_a_shared_worker_once():
    df = _shared_worker_frame()
    divisors = _worker_share_divisors(
        df, {"n_hosts": 1, "n_workers": 1, "n_worker_slots": 2}
    )
    _, edges, raw = _binned_task_series(df)
    _, same_edges, shared = _binned_task_series(df, divisors=divisors)
    np.testing.assert_array_equal(edges, same_edges)
    # the two concurrent tasks report the same 2-core process: 4 cores raw, 2 shared
    first = edges[1:] <= 10.0
    np.testing.assert_allclose(raw["cpu"][0][first] / np.diff(edges)[first], 4.0)
    np.testing.assert_allclose(shared["cpu"][0][first] / np.diff(edges)[first], 2.0)
    np.testing.assert_allclose(shared["rss"][0][first] / np.diff(edges)[first], 4.0)
    np.testing.assert_allclose(raw["running"], shared["running"])


def test_binned_maximum():
    df = _slot_frame(
        [(1, None, 0.0, 4, 1.0, 2.0, 1.0), (2, None, 2.5, 4, 1.0, 6.0, 1.0)]
    )
    edges = np.array([0.0, 2.0, 4.0, 5.0])
    maximum = _binned_maximum(df, edges, "memory_rss_bytes", 1 / 2**30, None)
    np.testing.assert_allclose(maximum, [2.0, 6.0, 6.0])
    halved = _binned_maximum(df, edges, "memory_rss_bytes", 1 / 2**30, [2.0, 3.0])
    np.testing.assert_allclose(halved, [1.0, 2.0, 2.0])


def test_performance_and_wall_clock_figures():
    open_figures = plt.get_fignums()
    figure, axes = _performance_figure((6, 4), 1, 2, sharey=True)
    assert len(axes) == 2 and plt.get_fignums() == open_figures
    figure, main, strip = _wall_clock_figure(2, (11, 9), (4, 4, 1))
    assert len(main) == 2 and len(figure.axes) == 3
    assert strip.get_xlabel() == "Wall-clock time since first task start (s)"
    assert main[0].get_shared_x_axes().joined(main[0], strip)
    assert plt.get_fignums() == open_figures


def test_save_figures(tmp_path, logged):
    figure, _ = _performance_figure((3, 2))
    with pytest.raises(ValueError, match=r"needs the field \{figure\}"):
        _figure_save_paths(str(tmp_path / "x.png"), ["cpu", "memory"])
    assert _figure_save_paths(None, ["cpu"]) == {}
    _save_figures({"cpu": figure, "io": None}, str(tmp_path / "x_{figure}.png"))
    assert (tmp_path / "x_cpu.png").exists() and not (tmp_path / "x_io.png").exists()
    _save_figures({"cpu": figure}, str(tmp_path / "single.png"))
    assert (tmp_path / "single.png").exists()
    assert logged.read().count("wrote") == 2


def test_add_notes():
    figure, _ = _performance_figure((3, 2))
    _add_notes(figure, None)
    _add_notes(figure, ["", None])
    assert not figure.texts
    _add_notes(figure, ["occupancy 94 %", None, "Lustre 32 OSTs"])
    _add_notes(figure, "one note")
    assert [text.get_text() for text in figure.texts] == [
        "occupancy 94 %\nLustre 32 OSTs",
        "one note",
    ]


def test_axis_helpers():
    figure, ax = _performance_figure((4, 4))
    _parallelism_ticks(ax, [8, 72, 8], [112, 1008, 224], "Number of Nodes")
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        "8\n112/224",
        "72\n1008",
    ]
    assert ax.get_xlabel() == "Number of Nodes (total parallel workers)"
    figure, ax = _performance_figure((4, 4))
    _parallelism_ticks(ax, [8, 16], [None, np.nan], "Number of Nodes")
    assert ax.get_xlabel() == ""
    _value_ticks(ax, [3840, 15360], "Channels")
    assert [label.get_text() for label in ax.get_xticklabels()] == ["3,840", "15,360"]
    assert ax.get_xlabel() == "Channels"
    secondary = _node_units_axis(ax, [112, 1008], [8, 72])
    assert secondary.get_ylabel() == "Speedup (node units, 14 workers/node)"
    assert _node_units_axis(ax, [112, 1000], [8, 72]) is None


def test_band_helpers():
    values = [40.0, 42.0, 44.0, 70.0]
    assert _band_or_auto(values, (35, 55), inside=0.75) == (35, 55)
    assert _band_or_auto(values, (35, 55)) is None
    assert _band_or_auto(values, None) is None
    figure, ax = _performance_figure((4, 4))
    _band_or_minimum_span(ax, [40.0, 42.0, 44.0], (35, 55))
    assert ax.get_ylim() == (35, 55)
    _band_or_minimum_span(ax, [100.0, 101.0, 102.0], (35, 55))
    np.testing.assert_allclose(ax.get_ylim(), (78.78, 123.22))  # +-22 % of 101


def test_task_stream_without_compute_anchors(logged):
    """Without the compute anchors the time after the task window is all post
    map, the driver strip cannot be placed and no efficiency is logged."""
    source = _stream_dict()
    timing = source["timing_distributed_application"]
    for key in ("compute_start_unixtime", "compute_end_unixtime"):
        del timing[key]
    figure = plot_task_stream(source, timing_phases=DRIVER_PHASES, title="run")
    out = logged.read()
    assert len(figure.axes) == 2
    assert "no absolute compute anchors" in out
    assert "30.0 s (+8.0 s outside the task window)" in out  # 30 - 22
    assert "E map-reduce" not in out and "E end-to-end" not in out
    assert "post-map: reduce residue + gather/save\n8 s" in [
        text.get_text() for text in figure.axes[0].texts
    ]
    plt.close(figure)
