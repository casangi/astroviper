"""Unit tests of the resource figures with compute resources and of the
task-time figures of astroviper.utils.resource_plots (synthetic frames,
headless matplotlib, numeric checks of the plotted data)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from astroviper.utils.resource_plots import (
    _binned_task_series,
    _worker_share_divisors,
    plot_cluster_resource_usage,
    plot_core_resource_usage,
    plot_fault_time_correlation,
    plot_task_maps,
    plot_task_phase_times,
    plot_task_resource_usage,
    plot_task_time_distribution,
    plot_task_time_drift,
    plot_task_times_per_host,
)

MOMENTS_COLUMNS = {
    "task": "T_moments_task",
    "load": ("T_load", "T_stream_read"),
    "write": "T_write",
}


def _series_frame(tasks):
    """One row per task (host, pid, start s, samples, cores, GiB, MiB/s
    written; half that read), 1 s samples."""
    rows = []
    for host, pid, start, n, cores, gib, mib_s in tasks:
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
                "hostname": host,
                "process_pid": pid,
                "thread_native_id": 10 * pid,
                "T_image_cube_task": float(n - 1),
            }
        )
    return pd.DataFrame(rows)


def _two_host_frame():
    """Host a: two 20 s tasks at 1 core, 2 GiB, 1 MiB/s written, side by
    side; host b: one 20 s task at 2 cores, 4 GiB, 2 MiB/s."""
    return _series_frame(
        [
            ("a", 1, 0.0, 21, 1.0, 2.0, 1.0),
            ("a", 2, 0.0, 21, 1.0, 2.0, 1.0),
            ("b", 3, 0.0, 21, 2.0, 4.0, 2.0),
        ]
    )


def _shared_worker_frame():
    """Three tasks of one 1 s sampled worker process on host "node": two run
    together for 10 s at 2 cores (the whole process), the third alone
    afterwards at 1 core."""
    return _series_frame(
        [
            ("node", 1, 0.0, 11, 2.0, 4.0, 2.0),
            ("node", 1, 0.0, 11, 2.0, 4.0, 2.0),
            ("node", 1, 10.0, 11, 1.0, 2.0, 1.0),
        ]
    )


def _lines(ax):
    return {line.get_label(): line for line in ax.get_lines()}


def _axhlines(ax):
    """{label: (y, linestyle)} of the horizontal lines (x = [0, 1])."""
    out = {}
    for line in ax.get_lines():
        x = np.asarray(line.get_xdata(), dtype=float)
        if len(x) == 2 and x[0] == 0 and x[1] == 1:
            out[line.get_label()] = (float(line.get_ydata()[0]), line.get_linestyle())
    return out


def _timing_frame(n_tasks=400, hours=4.0, seed=7):
    """Scalar node task timings over ``hours``: the non-I/O time grows
    linearly with the start, minor page faults follow it (1.5 s per
    million), four hosts, task ids 0..n-1."""
    rng = np.random.default_rng(seed)
    start = np.sort(rng.uniform(0, hours * 3600, n_tasks))
    faults = 1e6 + 6e6 * start / start.max()
    non_io = 40.0 + 1.5 * faults / 1e6
    load = rng.uniform(1.0, 2.0, n_tasks)
    empty = rng.uniform(0.01, 0.02, n_tasks)
    write = rng.uniform(3.0, 4.0, n_tasks)
    return pd.DataFrame(
        {
            "task_id": np.arange(n_tasks),
            "start_unixtime": 1.75e9 + start,
            "T_load": load,
            "T_make_empty_image": empty,
            "T_write": write,
            "T_image_cube_task": load + empty + write + non_io,
            "minor_page_faults": faults,
            "hostname": [f"c{100 + i % 4}.frontera" for i in range(n_tasks)],
        }
    )


# --------------------------------------------------------------------------
# plot_task_resource_usage
# --------------------------------------------------------------------------
def test_task_resource_usage_capacity_line_and_titles():
    df = _two_host_frame()
    axes = plot_task_resource_usage(
        df, compute_resources={"n_hosts": 2, "processing_function_threads": 4}
    )
    cpu = axes["cpu"]
    assert _axhlines(cpu)["per-task capacity (4 OpenMP threads)"] == (400.0, "--")
    assert cpu.get_ylim()[1] >= 420.0
    assert (
        cpu.get_title()
        == "Node-Task CPU Usage (2 hosts; per worker process, all threads)"
    )
    assert axes["memory"].get_title() == "Node-Task Memory Usage (2 hosts)"
    assert axes["io"].get_title() == (
        "Node-Task I/O (2 hosts; syscall-level, counts network filesystems)"
    )
    # pft <= 0 means hardware concurrency: no per-task line (C4)
    for threads in (0, None):
        axes = plot_task_resource_usage(
            df,
            compute_resources={"processing_function_threads": threads},
            filesystem_label="Lustre",
        )
        assert not _axhlines(axes["cpu"])
        assert axes["memory"].get_title() == "Node-Task Memory Usage"
        assert "counts Lustre)" in axes["io"].get_title()


def test_task_resource_usage_shares_a_worker_process():
    df = _shared_worker_frame()
    one_worker = {"n_hosts": 1, "n_workers": 1, "n_worker_slots": 14}
    axes = plot_task_resource_usage(df, compute_resources=one_worker)
    assert "approx. per-task share of worker process" in axes["cpu"].get_title()
    assert "approx. per-task share" in axes["memory"].get_title()
    # 2 cores of the process shared by the two concurrent tasks: 100 % each;
    # the third task runs alone at 1 core
    cpu_mean = axes["cpu"].get_lines()[-1].get_ydata()
    np.testing.assert_allclose(cpu_mean[1:9], 100.0)
    raw = plot_task_resource_usage(df)["cpu"].get_lines()[-1].get_ydata()
    np.testing.assert_allclose(raw[1:9], (200.0 + 200.0 + 100.0) / 3)
    memory_mean = axes["memory"].get_lines()[-1].get_ydata()
    np.testing.assert_allclose(memory_mean[1:9], 2.0)


def test_task_resource_usage_saves_without_pyplot(tmp_path):
    open_figures = plt.get_fignums()
    axes = plot_task_resource_usage(
        {"timing_node_tasks": _two_host_frame()},
        save_path=str(tmp_path / "task_{figure}.png"),
    )
    assert plt.get_fignums() == open_figures
    for name in ("cpu", "memory", "io"):
        assert (tmp_path / f"task_{name}.png").exists()
    assert axes["cpu"].get_lines()[0].get_alpha() == 0.08
    with pytest.raises(ValueError, match="figure"):
        plot_task_resource_usage(_two_host_frame(), save_path=str(tmp_path / "x.png"))


# --------------------------------------------------------------------------
# plot_cluster_resource_usage
# --------------------------------------------------------------------------
def test_cluster_resource_usage_per_host_divides_sums_and_lines():
    df = _two_host_frame()
    resources = {
        "n_hosts": 2,
        "n_workers": 4,
        "n_worker_slots": 4,
        "processing_function_threads": 2,
        "n_logical_cpus": 16,
        "memory_total_bytes": 64 * 2**30,
        "worker_memory_limit_bytes": 32 * 2**30,
    }
    total = plot_cluster_resource_usage(df, compute_resources=resources)
    per_host = plot_cluster_resource_usage(
        df, compute_resources=resources, per_host=True
    )
    busy = total["cpu"].get_lines()[0].get_ydata()
    np.testing.assert_allclose(busy, 4.0)  # 1 + 1 + 2 cores
    np.testing.assert_allclose(per_host["cpu"].get_lines()[0].get_ydata(), busy / 2)
    np.testing.assert_allclose(total["memory"].get_lines()[0].get_ydata(), 8.0)
    write = _lines(total["io"])["write"].get_ydata()
    np.testing.assert_allclose(write, 4.0 / 1024)  # 1 + 1 + 2 MiB/s
    assert _axhlines(total["cpu"]) == {
        "host capacity (16 logical CPUs)": (16.0, "--"),
        "worker capacity (4w x 1t x 2 OpenMP = 8 cores)": (8.0, ":"),
    }
    assert _axhlines(per_host["cpu"]) == {
        "host capacity (8 logical CPUs)": (8.0, "--"),
        "worker capacity (2w x 1t x 2 OpenMP = 4 cores)": (4.0, ":"),
    }
    assert _axhlines(per_host["memory"]) == {
        "host memory (32 GiB)": (32.0, "--"),
        "worker memory limit (16 GiB)": (16.0, ":"),
    }
    assert per_host["cpu"].get_title() == (
        "Cluster CPU Usage per Host over the Run (2 hosts; node-task processes)"
    )
    assert per_host["cpu"].get_ylabel().startswith("Busy cores per host")
    strip = per_host["cpu"].figure.axes[1]
    assert strip.get_ylabel() == "Running tasks\nper host"
    np.testing.assert_allclose(strip.get_lines()[0].get_ydata(), 1.5)
    assert per_host["cpu"].get_ylim()[1] == pytest.approx(1.25 * 8.0)


def test_cluster_resource_usage_line_rules():
    df = _two_host_frame()
    # the worker line equal to the host line is left out; pft 0 draws none
    axes = plot_cluster_resource_usage(
        df,
        compute_resources={
            "n_worker_slots": 4,
            "processing_function_threads": 4,
            "n_logical_cpus": 16,
            "memory_total_bytes": 64 * 2**30,
            "worker_memory_limit_bytes": 64 * 2**30,
        },
    )
    assert list(_axhlines(axes["cpu"])) == ["host capacity (16 logical CPUs)"]
    assert list(_axhlines(axes["memory"])) == ["host memory (64 GiB)"]
    axes = plot_cluster_resource_usage(
        df,
        compute_resources={"n_worker_slots": 4, "processing_function_threads": 0},
    )
    assert not _axhlines(axes["cpu"]) and not _axhlines(axes["memory"])
    # no host line: the worker line alone, dotted; no layout -> slots
    axes = plot_cluster_resource_usage(
        df,
        compute_resources={
            "n_worker_slots": 3,
            "processing_function_threads": 2,
            "worker_memory_limit_bytes": 12 * 2**30,
        },
    )
    assert _axhlines(axes["cpu"]) == {
        "worker capacity (3 slots x 2 OpenMP = 6 cores)": (6.0, ":")
    }
    assert _axhlines(axes["memory"]) == {"worker memory limit (12 GiB)": (12.0, ":")}
    assert (
        axes["cpu"].get_title()
        == "Cluster CPU Usage over the Run (node-task processes)"
    )
    with pytest.raises(ValueError, match="n_hosts"):
        plot_cluster_resource_usage(df, per_host=True)


def test_cluster_resource_usage_node_samples_and_footers(capsys):
    df = _two_host_frame()
    t0 = df["start_unixtime"].min()
    samples = pd.DataFrame(
        {"unixtime": t0 + np.arange(-2.0, 30.0), "busy_cores": np.full(32, 5.0)}
    )
    axes = plot_cluster_resource_usage(
        df,
        compute_resources={"n_hosts": 1},
        node_samples=samples,
        io_notes=["Lustre health: 32 OSTs"],
    )
    node = _lines(axes["cpu"])["node total (node sampler: all processes + kernel)"]
    np.testing.assert_allclose(node.get_ydata(), 5.0)
    cpu_texts = [text.get_text() for text in axes["cpu"].figure.texts]
    assert cpu_texts == [
        "node total minus worker processes: 1.00 cores on average "
        "(driver, Dask scheduler, kernel)"
    ]
    io_text = axes["io"].figure.texts[0].get_text().split("\n")
    assert io_text[0].startswith("read: ") and io_text[1].startswith("write: ")
    assert io_text[2] == "Lustre health: 32 OSTs"
    assert "in all" in capsys.readouterr().out
    # node samples of one host are not drawn for a run on two hosts
    axes = plot_cluster_resource_usage(
        df, compute_resources={"n_hosts": 2}, node_samples=samples, per_host=True
    )
    assert len(axes["cpu"].get_lines()) == 1
    assert "not drawn" in capsys.readouterr().out


def test_cluster_resource_usage_counts_a_shared_worker_once(tmp_path):
    df = _shared_worker_frame()
    resources = {"n_hosts": 1, "n_workers": 1, "n_worker_slots": 2}
    axes = plot_cluster_resource_usage(
        df, compute_resources=resources, save_path=str(tmp_path / "c_{figure}.png")
    )
    assert "de-duplicated" in axes["cpu"].get_title()
    divisors = _worker_share_divisors(df, resources)
    binw, edges, sums = _binned_task_series(df, divisors=divisors)
    np.testing.assert_allclose(
        axes["cpu"].get_lines()[0].get_ydata(), sums["cpu"][0] / np.diff(edges)
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "c_cpu.png",
        "c_io.png",
        "c_memory.png",
    ]


# --------------------------------------------------------------------------
# plot_core_resource_usage
# --------------------------------------------------------------------------
def test_core_resource_usage_per_host():
    axes = plot_core_resource_usage(_two_host_frame(), group="host")
    assert axes["cpu"].get_title() == "CPU usage per host over the run (2 hosts)"
    cpu = _lines(axes["cpu"])
    # host a: 2 tasks x 1 core, host b: 1 task x 2 cores
    np.testing.assert_allclose(cpu["peak (max over hosts)"].get_ydata(), 2.0)
    np.testing.assert_allclose(cpu["min"].get_ydata(), 2.0)
    memory = _lines(axes["memory"])
    np.testing.assert_allclose(memory["mean"].get_ydata(), 4.0)  # 2 x 2 and 4 GiB
    write = _lines(axes["io"])
    np.testing.assert_allclose(write["mean"].get_ydata(), 2.0 / 1024)
    assert not _axhlines(axes["cpu"])
    assert axes["io"].get_ylabel().startswith("Write rate per host (GiB/s)")


def test_core_resource_usage_subsets_hours_and_several_sources(tmp_path):
    df = _two_host_frame()
    axes = plot_core_resource_usage(
        [
            df,
            df.assign(
                memory_rss_bytes=[2 * np.asarray(v) for v in df["memory_rss_bytes"]]
            ),
        ],
        labels=["run 1", "run 2"],
        group="host",
        statistics=("mean", "peak"),
        quantities=("memory", "write"),
        time_unit="hours",
        save_path=str(tmp_path / "core_{figure}.png"),
    )
    assert axes["cpu"] is None
    assert len(axes["io"].figure.axes) == 2  # one write panel and the strip
    memory = _lines(axes["memory"])
    assert list(memory) == [
        "run 1 (peak (max over hosts))",
        "run 1 (mean)",
        "run 2 (peak (max over hosts))",
        "run 2 (mean)",
    ]
    np.testing.assert_allclose(memory["run 2 (mean)"].get_ydata(), 8.0)
    assert memory["run 2 (mean)"].get_color() == "C1"
    assert not axes["memory"].collections  # no min-peak band
    x = memory["run 1 (mean)"].get_xdata()
    assert x.max() < 20 / 3600 and x.min() > 0
    assert axes["memory"].figure.axes[-1].get_xlabel().endswith("(hours)")
    assert axes["memory"].get_title() == (
        "Memory usage per host over the run (2 / 2 hosts; task-running workers)"
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "core_io.png",
        "core_memory.png",
    ]
    one = plot_core_resource_usage(
        df, quantities=("cpu",), save_path=str(tmp_path / "only_cpu.png")
    )
    assert one["memory"] is None and (tmp_path / "only_cpu.png").exists()


def test_core_resource_usage_memory_lines_from_the_hosts():
    hosts = {
        "a": {
            "n_worker_slots": 2,
            "memory_total_bytes": 32 * 2**30,
            "worker_memory_limit_bytes": 12 * 2**30,
        },
        "b": {
            "n_worker_slots": 1,
            "memory_total_bytes": 32 * 2**30,
            "worker_memory_limit_bytes": 8 * 2**30,
        },
    }
    axes = plot_core_resource_usage(
        {"timing_node_tasks": _two_host_frame(), "compute_resources": {"hosts": hosts}}
    )
    assert _axhlines(axes["memory"]) == {
        "host memory per worker slot (16 GiB)": (16.0, "--"),
        "worker memory limit per slot (6 GiB)": (6.0, ":"),
    }
    assert _axhlines(axes["cpu"]) == {"one core (100 %)": (100.0, "--")}


def test_core_resource_usage_refuses():
    df = _two_host_frame()
    for kwargs in (
        {"group": "rack"},
        {"time_unit": "days"},
        {"statistics": ("max",)},
        {"quantities": ("gpu",)},
    ):
        with pytest.raises(ValueError):
            plot_core_resource_usage(df, **kwargs)
    with pytest.raises(ValueError, match="labels"):
        plot_core_resource_usage([df, df])
    with pytest.raises(ValueError, match="hostname"):
        plot_core_resource_usage(df.drop(columns="hostname"), group="host")
    with pytest.raises(ValueError, match="compute_resources"):
        plot_core_resource_usage([df, df], labels=["a", "b"], compute_resources={})


# --------------------------------------------------------------------------
# plot_task_time_drift
# --------------------------------------------------------------------------
def test_task_time_drift_medians_and_limits(tmp_path):
    tasks = _timing_frame()
    figure = plot_task_time_drift(
        {"timing_node_tasks": tasks},
        n_bins=8,
        save_path=str(tmp_path / "drift.png"),
    )
    assert (tmp_path / "drift.png").exists()
    (ax,) = figure.axes
    line = ax.get_lines()[0]
    assert line.get_color() == "tab:red" and line.get_marker() == "o"
    non_io = (
        tasks["T_image_cube_task"]
        - tasks["T_load"]
        - tasks["T_make_empty_image"]
        - tasks["T_write"]
    )
    bins = pd.qcut(tasks["start_unixtime"], 8, labels=False)
    np.testing.assert_allclose(line.get_ydata(), non_io.groupby(bins).median())
    hours = (tasks["start_unixtime"] - tasks["start_unixtime"].min()) / 3600
    np.testing.assert_allclose(line.get_xdata(), hours.groupby(bins).mean())
    assert ax.get_ylabel() == "Median non_io_task_time (s)"
    # the band does not hold 95 % of the medians: +-22 % around their median
    low, high = ax.get_ylim()
    assert (high - low) / 2 >= 0.22 * np.median(line.get_ydata()) - 1e-9
    figure = plot_task_time_drift(tasks, n_bins=8, task_time_band=(40.0, 60.0))
    assert figure.axes[0].get_ylim() == (40.0, 60.0)
    figure = plot_task_time_drift(tasks, n_bins=8, x_axis="fraction")
    np.testing.assert_allclose(
        figure.axes[0].get_lines()[0].get_xdata(), (np.arange(8) + 0.5) / 8
    )
    assert figure.axes[0].get_xlabel().startswith("Position in the run")


def test_task_time_drift_of_several_sources():
    a, b = _timing_frame(seed=1), _timing_frame(seed=2)
    figure = plot_task_time_drift(
        [a, b, a.drop(columns="start_unixtime")],
        labels=["sweep 1", "sweep 1", "sweep 2"],
        n_bins=6,
        title="Within-run drift",
    )
    ax = figure.axes[0]
    lines = ax.get_lines()
    assert len(lines) == 2  # the source without starts is skipped
    assert lines[0].get_color() == lines[1].get_color() == "C0"
    assert [line.get_label() for line in lines] == ["sweep 1", "_child1"]
    assert lines[0].get_markersize() == 4
    assert ax.get_title() == "Within-run drift"
    assert tuple(figure.get_size_inches()) == (9.0, 6.0)
    with pytest.raises(ValueError, match="task_time"):
        plot_task_time_drift([a, b], labels=["1", "2"], panels=("page_faults",))
    with pytest.raises(ValueError, match="labels"):
        plot_task_time_drift([a, b])
    with pytest.raises(ValueError):
        plot_task_time_drift(a, panels=("cpu",))
    with pytest.raises(ValueError):
        plot_task_time_drift(a, x_axis="days")
    assert plot_task_time_drift(a.drop(columns="start_unixtime")) is None


def test_task_time_drift_page_faults_and_coverage():
    tasks = _timing_frame()
    figure = plot_task_time_drift(
        tasks,
        panels=("huge_page_coverage", "page_faults", "task_time"),
        n_bins=8,
        fault_band_millions=(0.0, 8.0),
        title="Page-Fault View",
    )
    ax_time, ax_faults, ax_coverage = figure.axes
    bins = pd.qcut(tasks["start_unixtime"], 8, labels=False)
    faults = tasks["minor_page_faults"].groupby(bins).median().to_numpy()
    np.testing.assert_allclose(ax_faults.get_lines()[0].get_ydata(), faults / 1e6)
    assert ax_faults.get_ylim() == (0.0, 8.0)
    coverage = np.clip((faults.max() - faults) / (faults.max() * (1 - 1 / 512)), 0, 1)
    np.testing.assert_allclose(ax_coverage.get_lines()[0].get_ydata(), 100 * coverage)
    assert ax_time.get_title() == (
        f"Page-Fault View (per-task churn at plateau ~"
        f"{faults.max() * 4096 / 2**30:.0f} GiB)"
    )
    assert tuple(figure.get_size_inches()) == (11.0, 10.0)
    assert (
        plot_task_time_drift(
            tasks.drop(columns="minor_page_faults"), panels=("page_faults",)
        )
        is None
    )


def test_task_time_drift_worker_state():
    df = _two_host_frame()
    df["T_load"], df["T_write"] = 1.0, 1.0
    resources = {"n_hosts": 2, "n_workers": 3, "n_worker_slots": 3}
    figure = plot_task_time_drift(
        {"timing_node_tasks": df, "compute_resources": resources},
        panels=("task_time", "worker_state"),
        n_bins=2,
        memory_band_gib=(0.0, 10.0),
    )
    ax_time, ax_state, ax_io = figure.axes
    state = _lines(ax_state)
    _, edges, sums = _binned_task_series(df)
    np.testing.assert_allclose(
        state["mean worker RSS (GiB)"].get_ydata(), sums["rss"][0] / sums["running"][0]
    )
    np.testing.assert_allclose(state["max worker RSS (GiB)"].get_ydata(), 4.0)
    np.testing.assert_allclose(
        state["mean worker RSS (GiB)"].get_xdata(),
        0.5 * (edges[:-1] + edges[1:]) / 3600,
    )
    rates = _lines(ax_io)
    np.testing.assert_allclose(
        rates["write rate (GiB/s/host)"].get_ydata(), 4.0 / 1024 / 2
    )
    assert ax_state.get_ylim() == (0.0, 10.0)
    assert len(ax_state.get_legend().get_texts()) == 4
    with pytest.raises(ValueError, match="n_hosts"):
        plot_task_time_drift(df, panels=("worker_state",))
    with pytest.raises(ValueError, match="fraction"):
        plot_task_time_drift(df, panels=("worker_state",), x_axis="fraction")
    unmonitored = df.drop(columns="memory_rss_bytes")
    figure = plot_task_time_drift(unmonitored, panels=("worker_state",))
    assert "no monitoring data" in figure.axes[0].texts[0].get_text()


# --------------------------------------------------------------------------
# plot_fault_time_correlation
# --------------------------------------------------------------------------
def test_fault_time_correlation_fit(tmp_path):
    tasks = _timing_frame()
    figure = plot_fault_time_correlation(
        tasks,
        task_time_band=(0.0, 100.0),
        fault_axis_max_millions=13.0,
        save_path=str(tmp_path / "faults.png"),
    )
    assert (tmp_path / "faults.png").exists()
    ax = figure.axes[0]
    (fit,) = ax.get_lines()
    x, y = fit.get_xdata(), fit.get_ydata()
    assert len(x) == 50 and x[0] == 0.0
    assert x[-1] == pytest.approx(tasks["minor_page_faults"].max() / 1e6)
    np.testing.assert_allclose(y, 40.0 + 1.5 * x)  # the exact line of the data
    assert fit.get_label() == (
        "fit: 40.0 s + 1.50 s/Mfault (= 1.50 µs/fault),  r = 1.000"
    )
    assert len(ax.collections[0].get_offsets()) == len(tasks)
    assert ax.get_xlim() == (0.0, 13.0)
    assert ax.get_ylim() == (0.0, 100.0)


def test_fault_time_correlation_of_several_sources(capsys):
    a, b = _timing_frame(seed=1), _timing_frame(n_tasks=200, seed=2)
    few = _timing_frame(n_tasks=5)
    figure = plot_fault_time_correlation(
        [a, b, few, a.drop(columns="minor_page_faults")],
        labels=["A", "B", "C", "D"],
        max_points=100,
    )
    ax = figure.axes[0]
    assert [len(c.get_offsets()) for c in ax.collections] == [100, 100]
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels == ["A: 1.50 s/Mfault, r=1.00", "B: 1.50 s/Mfault, r=1.00"]
    assert ax.get_lines()[1].get_color() == "C1"
    assert "fewer than 10" in capsys.readouterr().out
    assert plot_fault_time_correlation(few) is None


# --------------------------------------------------------------------------
# plot_task_time_distribution
# --------------------------------------------------------------------------
def test_task_time_distribution_cdf():
    a, b = _timing_frame(seed=1), _timing_frame(n_tasks=100, seed=2)
    a.loc[0, "T_image_cube_task"] = np.nan
    figure = plot_task_time_distribution(
        [a, b], labels=["8 nodes", "72 nodes"], notes=["occupancy 90 %"]
    )
    ax = figure.axes[0]
    first, second, quantile = ax.get_lines()
    expected = np.sort(a["T_image_cube_task"].dropna().to_numpy())
    np.testing.assert_allclose(first.get_xdata(), expected)
    np.testing.assert_allclose(
        first.get_ydata(), np.arange(1, len(expected) + 1) / len(expected)
    )
    assert second.get_ydata()[-1] == 1.0
    np.testing.assert_allclose(quantile.get_ydata(), 0.95)
    assert ax.get_xlabel() == "T_image_cube_task (s)"
    assert ax.get_title() == "Per-Task Time Distribution (CDF of T_image_cube_task)"
    assert figure.texts[0].get_text() == "occupancy 90 %"
    figure = plot_task_time_distribution(
        a, quantity="non_io_task_time", quantile_line=None
    )
    assert len(figure.axes[0].get_lines()) == 1
    # the task with a NaN task time is left out
    assert figure.axes[0].get_lines()[0].get_label() == f"{len(a) - 1} tasks"


def test_task_time_distribution_histograms():
    a, b, c = (_timing_frame(n_tasks=n, seed=n) for n in (100, 120, 140))
    figure = plot_task_time_distribution(
        [a, b, c], kind="histogram", pooled=True, bins=20, log_counts=True
    )
    ax = figure.axes[0]
    assert sum(p.get_height() for p in ax.patches) == 360
    assert ax.get_yscale() == "log"
    figure = plot_task_time_distribution(
        [a, b, c],
        labels=["run 1", "run 2", "run 1"],
        panel_labels=["8 nodes", "8 nodes", "72 nodes"],
        quantity="non_io_task_time",
        kind="histogram",
        bins=11,
    )
    top, bottom = figure.axes
    assert (top.get_title(), bottom.get_title()) == ("8 nodes", "72 nodes")
    assert [p.get_label() for p in top.patches] == ["run 1", "run 2"]
    values = np.concatenate(
        [plot_values for plot_values in (_non_io(a), _non_io(b), _non_io(c))]
    )
    edges = np.linspace(values.min(), values.max(), 11)
    step = top.patches[0].get_xy()
    assert step[:, 0].min() == pytest.approx(edges[0])
    assert step[:, 0].max() == pytest.approx(edges[-1])
    with pytest.raises(ValueError):
        plot_task_time_distribution(a, kind="violin")
    with pytest.raises(ValueError):
        plot_task_time_distribution(a, quantity="cpu")
    with pytest.raises(ValueError, match="panel_labels"):
        plot_task_time_distribution(
            [a, b], labels=["1", "2"], kind="histogram", panel_labels=["x"]
        )
    with pytest.raises(ValueError, match="panel_labels"):
        plot_task_time_distribution([a, b], labels=["1", "2"], panel_labels=["x", "y"])


def _non_io(tasks):
    return (
        tasks["T_image_cube_task"]
        - tasks["T_load"]
        - tasks["T_make_empty_image"]
        - tasks["T_write"]
    ).to_numpy()


def test_task_time_distribution_read_rate():
    df = _two_host_frame().assign(T_moments=10.0)
    figure = plot_task_time_distribution(
        df, quantity="read_rate", read_time_column="T_moments", kind="histogram", bins=4
    )
    ax = figure.axes[0]
    # GiB read over 10 s in MiB/s: 0.5 MiB/s x 20 s / 10 s = 1 MiB/s (twice), 2 MiB/s
    heights = [p.get_height() for p in ax.patches]
    assert sum(heights) == 3 and heights[0] == 2 and heights[-1] == 1
    assert ax.get_xlabel() == "per-task read rate (MiB/s, syscall-level)"
    read_tib = (2 * 10 + 20) * 2**20 / 2**40
    assert ax.get_title() == f"Read throughput per task (total {read_tib:.1f} TiB read)"
    assert (
        plot_task_time_distribution(df.drop(columns="read_chars"), quantity="read_rate")
        is None
    )


# --------------------------------------------------------------------------
# plot_task_phase_times
# --------------------------------------------------------------------------
def test_task_phase_times_default_columns_never_stack_the_task_total():
    tasks = _timing_frame()
    figure = plot_task_phase_times(tasks, statistic="mean", layout="stacked")
    ax = figure.axes[0]
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels == ["load", "non-I/O", "write"]
    total = sum(p.get_height() for p in ax.patches)
    # load + non-I/O + write = the task time, not twice it (C18)
    assert total == pytest.approx(tasks["T_image_cube_task"].mean(), rel=1e-12)
    assert ax.texts[0].get_text() == f"{total:.0f} s"


def test_task_phase_times_grouped():
    a, b = _timing_frame(seed=1), _timing_frame(seed=2)
    figure = plot_task_phase_times(
        [a, b],
        labels=["8 nodes", "72 nodes"],
        columns={"T_load": "T_load", "T_write": "T_write", "missing": "T_fft"},
        notes="note",
        save_path=None,
    )
    ax_time, ax_share = figure.axes
    heights = np.array([p.get_height() for p in ax_time.patches])
    expected = [
        a["T_load"].median(),
        a["T_write"].median(),
        b["T_load"].median(),
        b["T_write"].median(),
    ]
    np.testing.assert_allclose(heights, expected)
    shares = np.array([p.get_height() for p in ax_share.patches])
    np.testing.assert_allclose(
        shares[:2], 100 * np.array(expected[:2]) / a["T_image_cube_task"].median()
    )
    assert [t.get_text() for t in ax_share.get_xticklabels()] == ["T_load", "T_write"]
    assert ax_time.get_ylabel() == "Median time per task (s)"
    assert ax_share.get_ylabel() == "% of median T_image_cube_task"
    with pytest.raises(ValueError):
        plot_task_phase_times(a, statistic="max")
    with pytest.raises(ValueError):
        plot_task_phase_times(a, layout="pie")
    with pytest.raises(ValueError, match="per_task"):
        plot_task_phase_times([a, b], labels=["1", "2"], layout="per_task")
    assert plot_task_phase_times(a, columns={"x": "T_none"}) is None


def test_task_phase_times_per_task():
    rng = np.random.default_rng(7)
    tasks = pd.DataFrame(
        {
            "T_load": rng.uniform(0, 1, 9),
            "T_stream_read": rng.uniform(5, 9, 9),
            "T_write": rng.uniform(1, 2, 9),
        }
    )
    tasks["T_moments_task"] = tasks.sum(axis=1)
    figure = plot_task_phase_times(
        tasks, layout="per_task", task_columns=MOMENTS_COLUMNS
    )
    ax = figure.axes[0]
    order = np.argsort(tasks["T_moments_task"].to_numpy())
    # the moments load segment is T_load + T_stream_read
    load = np.array([p.get_height() for p in ax.patches[:9]])
    np.testing.assert_allclose(
        load, (tasks["T_load"] + tasks["T_stream_read"]).to_numpy()[order]
    )
    tops = np.array([p.get_y() + p.get_height() for p in ax.patches[18:]])
    np.testing.assert_allclose(tops, tasks["T_moments_task"].to_numpy()[order])
    (median,) = ax.get_lines()
    assert median.get_ydata()[0] == pytest.approx(tasks["T_moments_task"].median())
    assert [text.get_text() for text in ax.get_legend().get_texts()] == [
        f"median {tasks['T_moments_task'].median():.0f} s",
        "load",
        "non-I/O",
        "write",
    ]


# --------------------------------------------------------------------------
# plot_task_times_per_host
# --------------------------------------------------------------------------
def test_task_times_per_host_lines():
    a, b = _timing_frame(seed=1), _timing_frame(seed=2)
    figure = plot_task_times_per_host(
        [a, b],
        columns={
            "read (T_load)": "T_load",
            "load and write": ("T_load", "T_write"),
            "total task (T_image_cube_task)": "T_image_cube_task",
        },
        notes=["occupancy"],
    )
    ax = figure.axes[0]
    pooled = pd.concat([a, b])
    pooled["host"] = pooled["hostname"].str.split(".").str[0]
    means = pooled.groupby("host").mean(numeric_only=True)
    order = means["T_image_cube_task"].sort_values(ascending=False).index
    lines = _lines(ax)
    np.testing.assert_allclose(
        lines["read (T_load)"].get_ydata(), means.loc[order, "T_load"]
    )
    np.testing.assert_allclose(
        lines["load and write"].get_ydata(),
        (means["T_load"] + means["T_write"]).loc[order],
    )
    total = lines["total task (T_image_cube_task)"]
    assert total.get_color() == "gray" and total.get_marker() == "."
    assert lines["load and write"].get_marker() == "s"
    assert [t.get_text() for t in ax.get_xticklabels()] == list(order)


def test_task_times_per_host_bars_and_tasks():
    tasks = _timing_frame()
    figure = plot_task_times_per_host(
        tasks,
        columns={"median (s)": "T_image_cube_task"},
        statistic="median",
        kind="bars",
    )
    ax = figure.axes[0]
    medians = tasks.groupby(tasks["hostname"].str.split(".").str[0])[
        "T_image_cube_task"
    ].median()
    bars = sorted(ax.patches, key=lambda p: p.get_y())  # bottom to top
    np.testing.assert_allclose([p.get_width() for p in bars], np.sort(medians))
    figure = plot_task_times_per_host(
        [tasks, tasks],
        labels=["run A", "run B"],
        columns={"non_io_task_time (s)": "non_io_task_time"},
        kind="tasks",
        sort="name",
    )
    ax = figure.axes[0]
    clouds = [c for c in ax.collections if len(c.get_offsets()) > 4]
    assert [len(c.get_offsets()) for c in clouds] == [len(tasks), len(tasks)]
    median = [c for c in ax.collections if c.get_label() == "per-host median"][0]
    np.testing.assert_allclose(
        median.get_offsets()[:, 1],
        pd.Series(_non_io(tasks)).groupby(tasks["hostname"].str[:4].values).median(),
    )
    with pytest.raises(ValueError, match="one column"):
        plot_task_times_per_host(
            tasks, columns={"a": "T_load", "b": "T_write"}, kind="bars"
        )
    for kwargs in ({"statistic": "max"}, {"kind": "pie"}, {"sort": "rack"}):
        with pytest.raises(ValueError):
            plot_task_times_per_host(tasks, **kwargs)
    assert plot_task_times_per_host(tasks.drop(columns="hostname")) is None
    assert plot_task_times_per_host(tasks, columns={"x": "T_none"}) is None


# --------------------------------------------------------------------------
# plot_task_maps
# --------------------------------------------------------------------------
def test_task_maps_place_tasks_by_their_chunks(tmp_path):
    rng = np.random.default_rng(7)
    rows = []
    for task_id in range(6):
        n = 5
        t = np.arange(float(n))
        rows.append(
            {
                "task_id": task_id,
                "T_moments_task": 10.0 + task_id,
                "time_seconds": t.tolist(),
                "cpu_seconds": (0.5 * (task_id + 1) * t).tolist(),
                "memory_rss_bytes": (rng.uniform(1, 2, n) * 2**30).tolist(),
                "read_chars": (task_id * 2**30 * t / 4).tolist(),
            }
        )
    tasks = pd.DataFrame(rows)
    figure = plot_task_maps(
        {"timing_node_tasks": tasks, "n_mapping_parallelism": {"l": 3, "m": 2}},
        task_columns=MOMENTS_COLUMNS,
        save_path=str(tmp_path / "maps.png"),
    )
    assert (tmp_path / "maps.png").exists()
    time_ax, rss_ax, read_ax, cpu_ax = figure.axes[:4]
    np.testing.assert_allclose(
        time_ax.images[0].get_array(), 10.0 + np.arange(6).reshape(3, 2)
    )
    np.testing.assert_allclose(
        read_ax.images[0].get_array(), np.arange(6).reshape(3, 2)
    )
    np.testing.assert_allclose(
        cpu_ax.images[0].get_array(), 50.0 * (np.arange(6) + 1).reshape(3, 2)
    )
    assert time_ax.get_ylabel() == "l chunk index"
    assert time_ax.get_xlabel() == "m chunk index"
    assert time_ax.images[0].origin == "lower"
    figure = plot_task_maps(
        tasks.iloc[:4], n_mapping_parallelism={"l": 2, "m": 2}, quantities=("cpu",)
    )
    assert len(figure.axes) == 2  # the panel and its colour bar


def test_task_maps_refuse(capsys):
    tasks = pd.DataFrame({"task_id": range(4), "T_image_cube_task": 1.0})
    assert plot_task_maps(tasks) is None
    assert plot_task_maps({"timing_node_tasks": tasks}) is None
    assert plot_task_maps(tasks, n_mapping_parallelism={"l": None, "m": None}) is None
    assert plot_task_maps(tasks, n_mapping_parallelism={"frequency": 4}) is None
    assert "two parallel axes" in capsys.readouterr().out
    assert plot_task_maps(tasks, n_mapping_parallelism={"l": 1, "m": 2}) is None
    assert "do not fit" in capsys.readouterr().out
    assert (
        plot_task_maps(
            tasks.drop(columns="task_id"), n_mapping_parallelism={"l": 2, "m": 2}
        )
        is None
    )
    assert (
        plot_task_maps(
            tasks, n_mapping_parallelism={"l": 2, "m": 2}, quantities=("read",)
        )
        is None
    )
    with pytest.raises(ValueError):
        plot_task_maps(tasks, n_mapping_parallelism={"l": 2, "m": 2}, quantities=("x",))


def test_task_time_figures_are_pyplot_free():
    open_figures = plt.get_fignums()
    tasks = _timing_frame(n_tasks=100)
    plot_task_time_drift(tasks, n_bins=4)
    plot_fault_time_correlation(tasks)
    plot_task_time_distribution(tasks)
    plot_task_phase_times(tasks)
    plot_task_times_per_host(tasks)
    plot_task_maps(tasks, n_mapping_parallelism={"l": 10, "m": 10})
    assert plt.get_fignums() == open_figures
