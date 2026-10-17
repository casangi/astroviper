"""Unit tests for astroviper.utils.report_plots (headless matplotlib)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import xarray as xr
from matplotlib.figure import Figure

from astroviper.processing_functions.image_analysis.point_spread_function_gaussian_fit import (
    point_spread_function_gaussian_fit,
)
from astroviper.utils.report_plots import (
    _asinh_ticks,
    plot_antenna_positions,
    plot_deconvolution_summary,
    plot_point_spread_function_cross_sections,
    plot_spectra,
    plot_stokes_images,
    plot_uv_coverage,
    point_spread_function_cross_sections,
)

ARCSEC = np.pi / (180 * 3600)
CELL = np.array([-0.1, 0.15]) * ARCSEC  # different pixel sizes along l and m


def _gaussian_plane(beam, shape=(129, 129), cell=CELL):
    """A Gaussian of ``beam`` in the convention of the PSF fit: the major axis
    along ``(sin(pa), -cos(pa))`` in the frame of the pixel axes."""
    major, minor, pa = beam
    l_offset = (np.arange(shape[0]) - shape[0] // 2) * abs(cell[0])
    m_offset = (np.arange(shape[1]) - shape[1] // 2) * abs(cell[1])
    l_grid, m_grid = np.meshgrid(l_offset, m_offset, indexing="ij")
    along_major = l_grid * np.sin(pa) - m_grid * np.cos(pa)
    along_minor = l_grid * np.cos(pa) + m_grid * np.sin(pa)
    return np.exp(
        -4.0 * np.log(2.0) * ((along_major / major) ** 2 + (along_minor / minor) ** 2)
    )


def test_cross_sections_of_a_gaussian_are_the_gaussian():
    # A well sampled beam: the linear interpolation between pixels errs by
    # about cell**2 / (8 sigma**2), 0.4 percent here.
    beam = (4.0 * ARCSEC, 2.0 * ARCSEC, np.radians(30.0))
    cuts = point_spread_function_cross_sections(
        _gaussian_plane(beam, shape=(257, 257)), beam, CELL
    )
    assert set(cuts) == {"major", "minor", "diagonal"}
    for cut in cuts.values():
        valid = np.isfinite(cut["psf"])
        assert valid.mean() > 0.95
        np.testing.assert_allclose(cut["psf"][valid], cut["gaussian"][valid], atol=5e-3)
        assert cut["offset"][0] == pytest.approx(-cut["offset"][-1])
    assert cuts["major"]["fwhm"] == pytest.approx(beam[0])
    assert cuts["minor"]["fwhm"] == pytest.approx(beam[1])
    assert cuts["diagonal"]["fwhm"] == pytest.approx(
        1.0 / np.sqrt(0.5 / beam[0] ** 2 + 0.5 / beam[1] ** 2)
    )


def test_cross_sections_follow_the_fitted_beam():
    """The cuts of a PSF along its fitted beam match the fitted Gaussian: the
    cut directions and the fit share one position angle convention (a cut
    along a wrong direction of this 2:1 beam would miss by far more than the
    tolerance, which covers the linear interpolation between pixels)."""
    plane = _gaussian_plane((3.0 * ARCSEC, 1.5 * ARCSEC, np.radians(-50.0)))
    n_l, n_m = plane.shape
    img_xds = xr.Dataset(
        {
            "POINT_SPREAD_FUNCTION": (
                ("time", "frequency", "polarization", "l", "m"),
                plane[None, None, None],
            )
        },
        coords={
            "time": [0.0],
            "frequency": [1e11],
            "polarization": ["I"],
            "l": (np.arange(n_l) - n_l // 2) * CELL[0],
            "m": (np.arange(n_m) - n_m // 2) * CELL[1],
        },
        attrs={
            "data_groups": {"image": {"point_spread_function": "POINT_SPREAD_FUNCTION"}}
        },
    )
    fitted = point_spread_function_gaussian_fit(img_xds)
    beam = fitted["BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION"].values[0, 0, 0]
    cuts = point_spread_function_cross_sections(plane, beam, CELL)
    for cut in cuts.values():
        valid = np.isfinite(cut["psf"])
        np.testing.assert_allclose(
            cut["psf"][valid], cut["gaussian"][valid], atol=0.015
        )


def test_cross_sections_stay_inside_the_image():
    beam = (30.0 * ARCSEC, 10.0 * ARCSEC, 0.0)  # three majors exceed the image
    cuts = point_spread_function_cross_sections(_gaussian_plane(beam), beam, CELL)
    half_width = 64 * abs(CELL[0])
    assert cuts["major"]["offset"].max() == pytest.approx(half_width)


def _assert_pyplot_free(figure, n_axes):
    assert isinstance(figure, Figure)
    assert len(figure.axes) >= n_axes
    assert figure.canvas.manager is None  # not registered with pyplot


def test_point_spread_function_figure():
    beam = (2.0 * ARCSEC, 1.0 * ARCSEC, 0.3)
    figure = plot_point_spread_function_cross_sections(
        _gaussian_plane(beam), beam, CELL, title="beam"
    )
    _assert_pyplot_free(figure, 3)


@pytest.mark.parametrize("stretch", ["linear", "asinh"])
def test_stokes_images(stretch):
    planes = xr.DataArray(
        np.random.default_rng(1).normal(size=(2, 16, 12)),
        dims=("polarization", "l", "m"),
        coords={
            "polarization": ["I", "Q"],
            "l": -np.arange(16) * ARCSEC,
            "m": np.arange(12) * ARCSEC,
        },
    )
    figure = plot_stokes_images(planes, title="t", colorbar_label="Jy", stretch=stretch)
    _assert_pyplot_free(figure, 4)  # two images and their color bars
    image_axes = figure.axes[0]
    assert image_axes.get_xlim()[0] > image_axes.get_xlim()[1]  # east to the left


def test_asinh_ticks_avoid_the_linear_part():
    values = np.array([-0.5, 0.0, 1.0, 2.5e-3])
    ticks = _asinh_ticks(0.01, values)
    assert ticks == pytest.approx([-0.1, 0.0, 0.1, 1.0])


def test_spectra_and_deconvolution_summary():
    frequency = np.array([372.7626, 372.7632, 372.7638]) * 1e9
    figure = plot_spectra(frequency, np.ones((3, 2)), ["I", "Q"], "Jy")
    _assert_pyplot_free(figure, 2)
    assert figure.axes[0].xaxis.get_major_formatter().get_useOffset() is False

    shape = (3, 2)
    figure = plot_deconvolution_summary(
        frequency,
        ["I", "Q"],
        np.ones(shape),
        {1: [2, 2, 2], 2: [1, 1, 1]},
        np.full(shape, 300),
        [2, 2, 2],
        np.full(shape, np.nan),
        np.ones(shape),
        {1: "Reached max_iter"},
        {1: "Reached max_iter_per_cycle_updated", 2: "Reached threshold_per_cycle"},
    )
    _assert_pyplot_free(figure, 6)


def test_antenna_positions_and_uv_coverage():
    figure = plot_antenna_positions(
        ["A1", "A2", "A3"], [0.0, 10.0, -5.0], [0.0, 5.0, 8.0], [0.1, 0.2, 0.3]
    )
    _assert_pyplot_free(figure, 2)
    labels = [label.get_text() for label in figure.axes[1].get_xticklabels()]
    assert labels == ["A1", "A2", "A3"]

    u = np.random.default_rng(2).normal(size=1000) * 1e3
    figure = plot_uv_coverage(u, u[::-1], title="uv", max_points=100)
    _assert_pyplot_free(figure, 1)
    # every drawn sample and its conjugate
    assert sum(len(line.get_xdata()) for line in figure.axes[0].lines) == 200
