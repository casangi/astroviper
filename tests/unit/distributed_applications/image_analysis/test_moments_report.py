"""The ``report`` of the moments distributed application, on small synthetic images."""

import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xradio.image import load_image, make_empty_sky_image

import astroviper.distributed_applications.image_analysis.moments_report as module
from astroviper.distributed_applications.image_analysis import moments
from tests.unit.processing_functions.image_analysis.moments_test_utils import (
    RAD_PER_ARCSEC,
    make_test_image_xds,
    write_test_image,
)


@pytest.fixture
def tiled_image(tmp_path):
    """Image of 18 x 14 pixels in on-disk chunks of 6 x 7: 3 x 2 (l, m) tiles."""
    img_xds = make_test_image_xds(n_frequency=6, n_l=18, n_m=14)
    path = str(tmp_path / "input.img.zarr")
    img_xds.to_zarr(
        path,
        mode="w",
        zarr_format=3,
        encoding={
            "SKY": {"chunks": (1, 1, 2, 6, 7)},
            "MASK": {"chunks": (1, 1, 2, 6, 7)},
        },
    )
    return path


def _moments(input_image_store, moments_image_store, **kwargs):
    arguments = {
        "input_image_store": input_image_store,
        "moments_image_store": moments_image_store,
        "moments": ["maximum", "mean"],
        "moment_axis": "frequency",
        **kwargs,
    }
    return moments(**arguments)


def _text(path):
    with open(path, encoding="utf-8") as file:
        return file.read()


def test_html_report_of_a_monitored_tiled_run(tiled_image, tmp_path):
    """The processing section follows the moments driver steps and names the
    (l, m) chunks; the results section draws every moment map."""
    pytest.importorskip("psutil")
    result = _moments(
        tiled_image,
        str(tmp_path / "tiled.img.zarr"),
        monitor_resources_seconds=0.02,
        report="html",
    )
    assert result["report"] == str(tmp_path / "tiled_report.html")
    text = _text(result["report"])
    headings = (
        "Processing",
        "Resources",
        "Task maps",
        "Results",
        "Task phase times",
        "Read rate",
        "Moment maps",
    )
    positions = [text.index(f">{heading}</h") for heading in headings]
    assert positions == sorted(positions)
    for step in ("open input image", "get compute resources", "collect node task"):
        assert step in text, step
    assert "(l, m) chunk" in text
    assert "Cluster CPU" in text
    assert "(l up the rows, m across; task_id = i_l x 2 + i_m)" in text
    assert "T_stream_read (the reads and their decompression)" in text
    for name in ("SKY_MOMENT_MAXIMUM", "SKY_MOMENT_MEAN"):
        assert f"<figcaption>The moment map {name} of every Stokes plane" in text
    # processing (9 without I/O counters), task maps, task phases, 2 moments
    assert text.count("data:image/png;base64,") >= 9 + 1 + 1 + 2


def test_no_report(tiled_image, tmp_path):
    result = _moments(tiled_image, str(tmp_path / "tiled.img.zarr"))
    assert result["report"] is None
    assert not os.path.exists(tmp_path / "tiled_report.html")


def test_notebook_report_and_a_failing_report(tiled_image, tmp_path, monkeypatch):
    import IPython.display

    shown = []
    monkeypatch.setattr(IPython.display, "display", shown.append)
    result = _moments(tiled_image, str(tmp_path / "tiled.img.zarr"), report="notebook")
    assert result["report"] is None
    assert len(shown) == 1 and "Moment maps" in shown[0].data

    def failing(*args, **kwargs):
        raise RuntimeError("no report")

    monkeypatch.setattr(module, "moments_report", failing)
    out = str(tmp_path / "again.img.zarr")
    result = _moments(tiled_image, out, report="html")
    assert result["report"] is None  # logged; the moment maps are intact
    assert result["moments_image_store"] == out
    assert "SKY_MOMENT_MAXIMUM" in load_image(out).data_vars
    assert not os.path.exists(tmp_path / "again_report.html")


def test_pdf_report(tiled_image, tmp_path):
    result = _moments(tiled_image, str(tmp_path / "tiled.img.zarr"), report="pdf")
    assert result["report"] == str(tmp_path / "tiled_report.pdf")
    with open(result["report"], "rb") as file:
        assert file.read(4) == b"%PDF"


def test_unknown_report_format_is_refused(tiled_image, tmp_path):
    with pytest.raises(AssertionError):  # the param.json schema
        _moments(tiled_image, str(tmp_path / "tiled.img.zarr"), report="docx")
    assert not (tmp_path / "tiled.img.zarr").exists()


def test_report_after_a_run_over_one_axis(tiled_image, tmp_path):
    """A moment over m, mapped over frequency: the tasks are frequency chunks
    and the moment maps are not (l, m) images."""
    result = _moments(
        tiled_image,
        str(tmp_path / "m_axis.img.zarr"),
        moment_axis="m",
        n_mapping_parallelism={"frequency": 2},
    )
    path = module.moments_report(result, report="html")
    assert path == str(tmp_path / "m_axis_report.html")
    text = _text(path)
    assert "frequency chunk" in text
    assert (
        "Task maps: plot_task_maps: task maps need two parallel axes; this run "
        "was mapped over [&#x27;frequency&#x27;]." in text
    )
    assert (
        "The moment maps are not (l, m) images (frequency 6, l 18, m 1); not drawn."
        in text
    )


def test_task_name():
    assert module._task_name({"m": 2}) == "m chunk"
    assert module._task_name({"l": 3, "m": 2}) == "(l, m) chunk"
    assert module._task_name({"frequency": 2, "l": 3, "m": 2}) == (
        "(frequency, l, m) chunk"
    )


def _task_frame(streamed, read=True):
    """Four moments node tasks; task k read k + 1 MiB."""
    frame = pd.DataFrame(
        {
            "task_id": range(4),
            "T_load": [0.1, 0.2, 0.1, 0.2],
            "T_moments": [1.0, 2.0, 4.0, 1.0],
            "T_stream_read": [0.6, 1.5, 3.0, 0.5] if streamed else [np.nan] * 4,
            "T_accumulate": [0.4, 0.5, 1.0, 0.5] if streamed else [np.nan] * 4,
            "T_write": [0.3, 0.3, 0.3, 0.3],
            "T_moments_task": [1.5, 2.6, 4.5, 1.6],
        }
    )
    if read:
        frame["time_seconds"] = [[0.0, 1.0]] * 4
        frame["read_chars"] = [[0, (k + 1) * 2**20] for k in range(4)]
    return frame


def test_task_phase_columns():
    assert module._task_phase_columns(_task_frame(True)) == (
        module.STREAMED_TASK_PHASE_COLUMNS
    )
    assert module._task_phase_columns(_task_frame(False)) == (
        module.SLAB_TASK_PHASE_COLUMNS
    )
    frame = _task_frame(True).drop(columns="T_accumulate")
    assert module._task_phase_columns(frame) == module.SLAB_TASK_PHASE_COLUMNS


class _Document:
    """The calls a report section makes, keeping the figures."""

    def __init__(self):
        self.calls = []

    def subsection(self, title):
        self.calls.append(("subsection", title))

    def note(self, text):
        self.calls.append(("note", text))

    def figure(self, figure, caption):
        self.calls.append(("figure", figure, caption))


def test_moment_maps_every_kth_pixel_at_the_first_time(tmp_path, monkeypatch):
    """A map larger than MAX_MOMENT_IMAGE_SIZE is drawn every k-th pixel along
    l and m; a store of several times is drawn at its first time."""
    rng = np.random.default_rng(7)
    img_xds = make_empty_sky_image(
        phase_center=[0.6, -0.2],
        image_size=[18, 14],
        cell_size=[15 * RAD_PER_ARCSEC, 15 * RAD_PER_ARCSEC],
        frequency_coords=np.linspace(1.4e9, 1.5e9, 3),
        pol_coords=["I", "Q"],
        time_coords=[0.0, 1.0],
    )
    img_xds["SKY"] = xr.DataArray(
        rng.normal(size=(2, 3, 2, 18, 14)).astype(np.float32),
        dims=["time", "frequency", "polarization", "l", "m"],
    )
    img_xds["SKY"].attrs["units"] = "Jy/beam"
    img_xds.attrs["data_groups"] = {"base": {"sky": "SKY"}}
    input_image_store = tmp_path / "two_times.img.zarr"
    write_test_image(img_xds, input_image_store)
    out = str(tmp_path / "maximum.img.zarr")
    result = _moments(
        str(input_image_store),
        out,
        moments=["maximum"],
        n_mapping_parallelism={"m": 1},
    )

    monkeypatch.setattr(module, "MAX_MOMENT_IMAGE_SIZE", 8)
    document = _Document()
    module._add_moment_maps(document, result["moments_image_store"])
    assert document.calls[0] == ("subsection", "Moment maps")
    (_, figure, caption) = document.calls[1]
    assert len(document.calls) == 2
    assert "one pixel in 3 along l and m" in caption
    assert "at the first of its 2 times" in caption
    expected = (
        xr.open_zarr(out)["SKY_MOMENT_MAXIMUM"]
        .isel(time=0, frequency=0, l=slice(None, None, 3), m=slice(None, None, 3))
        .values
    )
    assert expected.shape == (2, 6, 5)
    panels = [ax for ax in figure.axes if ax.get_title().startswith("Stokes")]
    assert [ax.get_title() for ax in panels] == ["Stokes I", "Stokes Q"]
    for ax, plane in zip(panels, expected, strict=True):  # drawn with l across
        np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), plane.T)
    np.testing.assert_array_equal(
        expected[0], img_xds["SKY"].values[0, :, 0, ::3, ::3].max(axis=0)
    )


@pytest.mark.parametrize("streamed", [True, False])
def test_task_phase_times_and_read_rate(streamed):
    """Every task's phases stacked, sorted by T_moments_task; the read rate
    over T_moments on the streamed path and over T_load on the slab path."""
    import matplotlib.pyplot as plt

    frame = _task_frame(streamed)
    document = _Document()
    module._add_task_phase_times(document, {"timing_node_tasks": frame})
    module._add_read_rate(document, {"timing_node_tasks": frame})
    assert [call[0] for call in document.calls] == ["subsection", "figure"] * 2
    (_, phases, phase_caption), (_, rates, rate_caption) = document.calls[1::2]

    columns = (
        module.STREAMED_TASK_PHASE_COLUMNS
        if streamed
        else module.SLAB_TASK_PHASE_COLUMNS
    )
    ax = phases.axes[0]
    order = np.argsort(frame["T_moments_task"].to_numpy())
    heights = np.array([patch.get_height() for patch in ax.patches])
    np.testing.assert_allclose(
        heights.reshape(len(columns), 4), frame[list(columns)].to_numpy()[order].T
    )
    assert ("T_accumulate" in phase_caption) == streamed

    read_time_column = "T_moments" if streamed else "T_load"
    assert f"over {read_time_column} (MiB/s)" in rate_caption
    rate = np.arange(1, 5) / frame[read_time_column].to_numpy()
    ax = rates.axes[0]
    assert sum(patch.get_height() for patch in ax.patches) == 4
    left = min(patch.get_x() for patch in ax.patches)
    right = max(patch.get_x() + patch.get_width() for patch in ax.patches)
    np.testing.assert_allclose([left, right], [rate.min(), rate.max()])
    plt.close("all")


def test_read_rate_and_task_maps_notes():
    document = _Document()
    frame = _task_frame(True, read=False)
    module._add_read_rate(document, {"timing_node_tasks": frame})
    module._add_task_maps(
        document, {"timing_node_tasks": frame, "n_mapping_parallelism": {"m": 4}}
    )
    assert document.calls == [
        ("subsection", "Read rate"),
        (
            "note",
            "The read rate was not recorded: it needs monitor_resources_seconds "
            "and the I/O counters of the monitor, which exist on Linux only.",
        ),
        ("subsection", "Task maps"),
        (
            "note",
            "Task maps: plot_task_maps: task maps need two parallel axes; this run "
            "was mapped over ['m'].",
        ),
    ]


def test_phase_times_and_read_rate_notes_without_values():
    """No phase column, and read counters with one sample per task."""
    frame = _task_frame(False)
    document = _Document()
    module._add_task_phase_times(
        document,
        {"timing_node_tasks": frame.drop(columns=list(module.SLAB_TASK_PHASE_COLUMNS))},
    )
    frame["read_chars"] = [[2**20]] * 4
    module._add_read_rate(document, {"timing_node_tasks": frame})
    notes = [call[1] for call in document.calls if call[0] == "note"]
    assert notes[0].startswith("Task phase times: plot_task_phase_times: no T_load")
    assert notes[1] == (
        "Read rate: plot_task_time_distribution: no read_rate values; skipping."
    )
