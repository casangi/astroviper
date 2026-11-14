"""The ``report`` of the moments distributed application, on small synthetic images."""

import os

import numpy as np
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
    for heading in ("Processing", "Results", "Moment maps"):
        assert f">{heading}</h" in text, heading
    for step in ("open input image", "get compute resources", "collect node task"):
        assert step in text, step
    assert "(l, m) chunk" in text
    assert "Cluster CPU" in text
    for name in ("SKY_MOMENT_MAXIMUM", "SKY_MOMENT_MEAN"):
        assert f"<figcaption>The moment map {name} of every Stokes plane" in text
    assert text.count("data:image/png;base64,") >= 1 + 2 + 2


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
        "The moment maps are not (l, m) images (frequency 6, l 18, m 1); not drawn."
        in text
    )


def test_task_name():
    assert module._task_name({"m": 2}) == "m chunk"
    assert module._task_name({"l": 3, "m": 2}) == "(l, m) chunk"
    assert module._task_name({"frequency": 2, "l": 3, "m": 2}) == (
        "(frequency, l, m) chunk"
    )


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
