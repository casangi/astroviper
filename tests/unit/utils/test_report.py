"""Unit tests for astroviper.utils.report (headless matplotlib)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from astroviper.utils.report import (
    Report,
    add_processing_section,
    make_report,
    report_files_directory,
    report_path,
)

PHASES = [("DRIVER", None, [("compute", "T_compute_dask_graph")])]


def _figure():
    figure = Figure(figsize=(3, 2))
    figure.subplots().plot([0, 1], [1, 0])
    return figure


def _timing_frame(n_tasks=3, monitored=True, task_column="T_image_cube_task"):
    """A timing_node_tasks frame of staggered tasks on one host."""
    rows = []
    for task_id in range(n_tasks):
        row = {
            "task_id": task_id,
            task_column: 2.0,
            "T_make_empty_image": 0.1,
            "T_load": 0.4,
            "T_write": 0.3,
            "start_unixtime": 1.75e9 + task_id * 0.5,
            "hostname": "node",
            "process_pid": 100 + task_id,
            "thread_native_id": 1,
        }
        if monitored:
            t = np.arange(0.0, 2.01, 0.25)
            row.update(
                time_seconds=t.tolist(),
                cpu_percent=(100.0 + t).tolist(),
                memory_rss_bytes=(1e9 + 1e8 * t).tolist(),
                sample_interval_seconds=0.25,
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _return_dict(**kwargs):
    start = 1.75e9 - 0.5
    return {
        "timing_node_tasks": _timing_frame(**kwargs),
        "timing_distributed_application": {
            "T_compute_dask_graph": 3.5,
            "T_total": 4.0,
            "compute_start_unixtime": start + 0.4,
            "compute_end_unixtime": start + 3.9,
        },
    }


@pytest.mark.parametrize(
    "store, name",
    [
        ("cube.img.zarr", "cube"),
        ("data/sim.ps.zarr/", "data/sim"),
        ("cube.zarr", "cube"),
        ("cube.img.fits", "cube"),
        ("cube", "cube"),
    ],
)
def test_report_path_and_files_directory(store, name):
    assert report_path(store, "html") == f"{name}_report.html"
    assert report_path(store, "pdf") == f"{name}_report.pdf"
    assert report_path(store, "notebook") is None
    assert report_files_directory(store) == f"{name}_report_files"


def test_unknown_report_format_is_refused(tmp_path):
    with pytest.raises(ValueError, match="report must be one of"):
        report_path("cube.img.zarr", "docx")
    with pytest.raises(ValueError, match="report must be one of"):
        Report("title", "docx", str(tmp_path / "x"))
    with pytest.raises(ValueError, match="needs a path"):
        Report("title", "html")


def test_html_report_is_one_self_contained_file(tmp_path):
    path = str(tmp_path / "sub" / "run_report.html")
    with Report("A <title>", "html", path, subtitle="subtitle") as document:
        document.section("Section")
        document.subsection("Subsection")
        document.text("x < y & z")
        document.note("not recorded")
        document.preformatted("a  b\nc  d")
        document.table(pd.DataFrame({"mean": [1.234567]}, index=["T_x"]), "caption")
        document.figure(_figure(), caption="a figure")
    text = (tmp_path / "sub" / "run_report.html").read_text()
    assert text.startswith("<!doctype html>")
    assert "<title>A &lt;title&gt;</title>" in text
    assert "<h1>A &lt;title&gt;</h1>" in text
    assert "<h2>Section</h2>" in text and "<h3>Subsection</h3>" in text
    assert "x &lt; y &amp; z" in text
    assert '<p class="note">not recorded</p>' in text
    assert "<pre>a  b\nc  d</pre>" in text
    assert "1.235" in text and "<table" in text
    assert text.count("data:image/png;base64,") == 1
    assert "<figcaption>a figure</figcaption>" in text


def test_figures_are_closed_once_drawn(tmp_path):
    figure = plt.figure()
    figure.add_subplot().plot([0, 1])
    with Report("t", "html", str(tmp_path / "r.html")) as document:
        document.figure(figure)
    assert figure.number not in plt.get_fignums()


def test_pdf_report_flows_over_pages(tmp_path):
    path = tmp_path / "run_report.pdf"
    with Report("PDF", "pdf", str(path)) as document:
        for index in range(6):
            document.section(f"Section {index}")
            document.text("word " * 300)
            document.table(pd.DataFrame({"value": np.arange(30.0)}))
            document.figure(_figure(), caption="caption")
    content = path.read_bytes()
    assert content.startswith(b"%PDF")
    assert content.count(b"/Type /Page\n") + content.count(b"/Type /Page ") > 2


def test_unfinished_pdf_is_removed(tmp_path):
    path = tmp_path / "run_report.pdf"
    with pytest.raises(RuntimeError), Report("PDF", "pdf", str(path)) as document:
        document.text("text")
        raise RuntimeError("failed")
    assert not path.exists()


def test_notebook_report_is_displayed(monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, "display", shown.append)
    with Report("Notebook", "notebook") as document:
        document.text("text")
        document.figure(_figure())
    assert document.close() is None
    assert len(shown) == 1
    assert 'class="astroviper-report"' in shown[0].data
    assert "data:image/png;base64," in shown[0].data


def test_make_report_returns_the_path_or_logs_a_failure():
    assert make_report(lambda value: value, "report.html") == "report.html"

    def failing():
        raise RuntimeError("no report")

    assert make_report(failing) is None


def test_processing_section_of_a_monitored_run(tmp_path):
    """Without I/O counters (macOS): the task stream, the two task-time figures
    and the CPU and memory figures of the cluster, the worker slots and the
    tasks."""
    path = str(tmp_path / "r.html")
    with Report("t", "html", path) as document:
        add_processing_section(document, _return_dict(), PHASES, task_name="chunk")
    text = (tmp_path / "r.html").read_text()
    assert "<h2>Processing</h2>" in text
    for heading in ("Performance", "Timing", "Task stream", "Task times", "Resources"):
        assert f"<h3>{heading}</h3>" in text, heading
    assert "parallelism p" in text and "efficiency E_mr" in text
    assert "The run recorded no compute resources" in text
    assert "Distributed application steps (seconds)" in text
    assert "over the 3 tasks, one per chunk" in text
    assert "<th>median</th>" in text
    assert "Task time distribution: the fraction of the tasks" in text
    assert "Task phase times: the median over the tasks" in text
    assert "Cluster CPU" in text and "Cluster memory" in text
    for caption in ("CPU per worker slot", "Memory per worker slot"):
        assert caption in text, caption
    assert "CPU per task" in text and "Memory per task" in text
    assert "I/O was not recorded" in text
    assert text.count("data:image/png;base64,") == 9


def test_processing_section_without_monitoring_or_anchor(tmp_path):
    return_dict = _return_dict(monitored=False)
    return_dict["timing_node_tasks"] = return_dict["timing_node_tasks"].drop(
        columns="start_unixtime"
    )
    path = str(tmp_path / "r.html")
    with Report("t", "html", path) as document:
        add_processing_section(document, return_dict, PHASES)
    text = (tmp_path / "r.html").read_text()
    assert "Performance measures: the node tasks have no start_unixtime" in text
    assert "Task stream: plot_task_stream: no start_unixtime column" in text
    assert "CPU, memory and I/O were not recorded" in text
    assert text.count("data:image/png;base64,") == 2  # the task-time figures


def test_processing_section_of_a_monitored_run_without_anchor(tmp_path):
    """The task-aligned figures need no start_unixtime; the cluster and
    per-slot views say why they are left out."""
    return_dict = _return_dict()
    return_dict["timing_node_tasks"] = return_dict["timing_node_tasks"].drop(
        columns="start_unixtime"
    )
    path = str(tmp_path / "r.html")
    with Report("t", "html", path) as document:
        add_processing_section(document, return_dict, PHASES)
    text = (tmp_path / "r.html").read_text()
    assert "CPU per task" in text and "Memory per task" in text
    assert "Cluster CPU, memory and I/O: plot_cluster_resource_usage: no " in text
    assert "CPU, memory and I/O per worker slot: plot_core_resource_usage: no " in (
        text
    )
    assert "CPU, memory and I/O were not recorded" not in text
    assert text.count("data:image/png;base64,") == 2 + 2


def test_processing_section_with_other_task_columns(tmp_path):
    return_dict = _return_dict(task_column="T_simulate_task")
    path = str(tmp_path / "r.html")
    with Report("t", "html", path) as document:
        add_processing_section(
            document,
            return_dict,
            PHASES,
            task_columns={"task": "T_simulate_task", "load": (), "write": "T_write"},
        )
    text = (tmp_path / "r.html").read_text()
    assert "Task stream: one lane per worker" in text
    assert "most a given time (T_simulate_task)" in text


class _Document:
    """The calls of a report section, keeping the tables and figures."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.calls.append((name, args, kwargs))


def _calls(document, name):
    return [args for kind, args, _ in document.calls if kind == name]


def test_processing_section_groups_the_node_task_table(tmp_path):
    """The node task table and the phase times follow the phase layout."""
    node_task_timing_phases = [
        ("NODE I/O", None, [("load", "T_load"), ("write", "T_write")]),
        ("NOT RECORDED", "T_absent", []),
    ]
    path = str(tmp_path / "r.html")
    with Report("t", "html", path) as document:
        add_processing_section(
            document,
            _return_dict(monitored=False),
            PHASES,
            node_task_timing_phases=node_task_timing_phases,
        )
    text = (tmp_path / "r.html").read_text()
    assert "grouped by phase" in text
    assert "NODE I/O" in text and "phase total" in text
    assert "(outside the phases)" in text and "T_image_cube_task" in text
    assert "NOT RECORDED" not in text

    document = _Document()
    add_processing_section(
        document,
        _return_dict(monitored=False),
        PHASES,
        node_task_timing_phases=node_task_timing_phases,
    )
    (_, phase_times) = _calls(document, "figure")[1:3]
    ax_time, ax_share = phase_times[0].axes
    assert [label.get_text() for label in ax_share.get_xticklabels()] == ["NODE I/O"]
    assert ax_time.patches[0].get_height() == pytest.approx(0.4 + 0.3)
    plt.close("all")


def test_processing_section_notes_a_phase_layout_with_nothing_recorded():
    document = _Document()
    add_processing_section(
        document,
        _return_dict(monitored=False),
        PHASES,
        node_task_timing_phases=[("NOT RECORDED", "T_absent", [])],
    )
    assert ("Task phase times: not recorded.",) in _calls(document, "note")
    plt.close("all")


def test_processing_section_shows_the_compute_resources(tmp_path):
    compute_resources = {
        "compute_backend": "dask",
        "scheduler": "distributed",
        "processing_function_threads": 1,
        "mpi_world_size": None,
        "n_hosts": 1,
        "n_workers": 3,
        "n_worker_slots": 3,
        "worker_memory_limit_bytes": None,
        "n_logical_cpus": 8,
        "memory_total_bytes": 16 * 2**30,
        "hosts": {
            "node": {
                "n_workers": 3,
                "n_worker_slots": 3,
                "worker_memory_limit_bytes": None,
                "n_logical_cpus": 8,
                "memory_total_bytes": 16 * 2**30,
            }
        },
    }
    document = _Document()
    add_processing_section(
        document, {**_return_dict(), "compute_resources": compute_resources}, PHASES
    )
    tables = [args[0] for args in _calls(document, "table")]
    resources = next(t for t in tables if t.index.name == "compute resources")
    assert list(resources.index) == [
        "compute_backend",
        "scheduler",
        "processing_function_threads",
        "n_hosts",
        "n_workers",
        "n_worker_slots",
        "n_logical_cpus",
        "memory_total_bytes",
    ]
    assert resources.loc["memory_total_bytes", "value"] == 16 * 2**30
    measures = next(t for t in tables if t.index.name == "performance measure")
    assert measures.loc["parallelism p", "value"] == 3
    assert not any(
        "no compute resources" in args[0] for args in _calls(document, "note")
    )
    plt.close("all")


def test_performance_measures_table_keeps_the_finite_rows():
    from astroviper.utils.report import _performance_measures_table
    from astroviper.utils.resource_plots import performance_measures

    measures = performance_measures(_return_dict(), timing_phases=PHASES)
    table = _performance_measures_table(measures)
    assert list(table.columns) == ["value", "unit"]
    # three staggered 2 s tasks in three lanes, t3 - t1 = 3.4 s
    assert table.loc["parallelism p", "value"] == 3
    assert table.loc["parallelism p", "unit"] == "worker slots"
    assert table.loc["work W of the map and reduce tasks", "value"] == 6.0
    assert table.loc["efficiency E_mr = S_mr / p", "value"] == pytest.approx(
        6.0 / 3.4 / 3
    )
    assert table.loc["map-reduce window t3 - t1", "unit"] == "s"
    table = _performance_measures_table(
        {"n_worker_slots": 4, "work": np.nan, "ideal_time": None, "n_map_tasks": 8}
    )
    assert list(table.index) == ["parallelism p", "map tasks"]
