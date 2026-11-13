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
    node_task_timing_table,
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


def test_node_task_timing_table():
    frame = _timing_frame(monitored=False)
    frame["hostname"] = "node"
    table = node_task_timing_table(frame)
    assert list(table.columns) == ["mean", "max", "total"]
    assert list(table.index) == [
        "T_image_cube_task",
        "T_make_empty_image",
        "T_load",
        "T_write",
    ]
    assert table.loc["T_load", "total"] == pytest.approx(1.2)


def test_processing_section_of_a_monitored_run(tmp_path):
    path = str(tmp_path / "r.html")
    with Report("t", "html", path) as document:
        add_processing_section(document, _return_dict(), PHASES, task_name="chunk")
    text = (tmp_path / "r.html").read_text()
    assert "<h2>Processing</h2>" in text
    assert "Distributed application steps (seconds)" in text
    assert "over the 3 tasks, one per chunk" in text
    assert "Task stream" in text
    assert "Cluster CPU" in text and "Cluster memory" in text
    assert "Cluster I/O was not recorded" in text
    assert text.count("data:image/png;base64,") == 3


def test_processing_section_without_monitoring_or_anchor(tmp_path):
    return_dict = _return_dict(monitored=False)
    return_dict["timing_node_tasks"] = return_dict["timing_node_tasks"].drop(
        columns="start_unixtime"
    )
    path = str(tmp_path / "r.html")
    with Report("t", "html", path) as document:
        add_processing_section(document, return_dict, PHASES)
    text = (tmp_path / "r.html").read_text()
    assert "Task stream: plot_task_stream: no start_unixtime column" in text
    assert "Cluster CPU, memory and I/O were not recorded" in text
    assert "data:image/png;base64," not in text


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
    assert "Task stream: one lane per worker" in (tmp_path / "r.html").read_text()
