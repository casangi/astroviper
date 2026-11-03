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
    plot_plane_deconvolution_summary,
    plot_point_spread_function_cross_sections,
    plot_signed_images,
    plot_spectra,
    plot_stokes_images,
    plot_uv_coverage,
    plot_visibility_comparison,
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


def _record(n, closing, asp, history):
    """A convergence record of ``n`` model updates: with ``closing`` the fields
    of the residual hold the closing stop test too (cube imaging), with ``asp``
    the component counts of Asp, with ``history`` the iteration history of
    those components."""
    rng = np.random.default_rng(3)
    n_residual = n + closing
    iterations = list(range(10, 10 + n))
    record = {
        "model_update_stop_code": [2] * (n - 1) + [1],
        "iter_done": iterations + [0] * closing,
        # signed, as the imagers record them
        "start_peakres": np.geomspace(1.0, 1e-3, n) * (-1.0) ** np.arange(n),
        "peakres": np.geomspace(0.5, 5e-4, n_residual)
        * (-1.0) ** np.arange(n_residual),
        "threshold_per_cycle": np.geomspace(0.8, 8e-4, n),
        "model_flux": np.linspace(1.0, 2.0, n),
        "model_flux_negative": -np.linspace(0.0, 0.02, n),
        "n_point_components": iterations,
        "n_scale_components": [0] * n,
        "start_rms_in_mask": np.geomspace(0.1, 1e-3, n),
        "start_mad_rms_outside_mask": np.geomspace(0.05, 1e-3, n),
        "start_skew_in_mask": np.linspace(1.0, 0.0, n),
        "start_excess_kurtosis_in_mask": np.linspace(2.0, -1.0, n),
        "entropy": [np.nan, *np.linspace(5.0, 7.0, n_residual - 1)],
    }
    if asp:
        record |= {
            "n_point_components": [size - 6 for size in iterations],
            "n_scale_components": [4] * n,
            "n_residual_components": [2] * n,
            "n_significance_components": [1] * n,
            "n_hogbom_runs": [0] + [1] * (n - 1),
        }
    if history:
        empty = [np.array([])] * closing
        record |= {
            name: [rng.normal(size=size) for size in iterations] + empty
            for name in (
                "iteration_peak_residual",
                "iteration_residual_rms",
                "iteration_model_flux",
                "iteration_component_flux",
            )
        }
        # residual-shaped components first, then 4 Asp Gaussians and points
        record["iteration_component_scale"] = [
            np.r_[-1.0, -1.0, [4.0] * 4, [0.0] * (size - 6)] if asp else np.zeros(size)
            for size in iterations
        ] + empty
        for name in ("iteration_component_l", "iteration_component_m"):
            record[name] = [
                rng.integers(0, 32, size).astype(float) for size in iterations
            ] + empty
    return record


STOP_DESCRIPTIONS = {1: "iteration limit", 2: "threshold"}


def _summary(record, **kwargs):
    plane = np.random.default_rng(1).normal(size=(32, 32))
    return plot_plane_deconvolution_summary(
        record,
        np.abs(plane),
        plane,
        stop_descriptions=STOP_DESCRIPTIONS,
        **kwargs,
    )


def _legend(ax):
    return [text.get_text() for text in ax.get_legend().get_texts()]


def _threshold_per_cycle(figure):
    (lines,) = (
        collection
        for collection in figure.subfigs[1].axes[0].collections
        if collection.get_label() == "threshold_per_cycle"
    )
    return [segment[0, 1] for segment in lines.get_segments()]


@pytest.mark.parametrize(
    "closing, asp", [(True, True), (False, False)], ids=["cube_asp", "continuum"]
)
def test_plane_deconvolution_summary(closing, asp):
    rng = np.random.default_rng(1)
    plane = rng.normal(size=(32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    figures = [
        _summary(
            _record(4, closing, asp, history),
            restored=np.abs(plane),
            mask=mask if asp else None,
            truth=np.abs(plane),
            model_error=plane,
            true_flux=2.0,
            threshold=1e-3,
            sigma=1.0,
            title="plane",
        )
        for history in (True, False)
    ]
    for figure in figures:
        _assert_pyplot_free(figure, 13)
    # the iteration history adds its row of six panels
    assert len(figures[0].axes) == len(figures[1].axes) + 6


SIGMA = 1.0


def _drawn(closing):
    """A summary of 4 model updates with their history and planes neither
    square nor symmetric: the cube with a mask and its closing entropy, the
    continuum without a mask and its final entropy given. The residual has one
    pixel beyond 6 sigma, outside the mask."""
    rng = np.random.default_rng(5)
    record = _record(4, closing, asp=True, history=True)
    names = ("truth", "restored", "model", "residual", "model_error")
    planes = {name: rng.normal(size=(32, 24)) for name in names}
    planes["residual"][0, 0] = 10 * SIGMA
    mask = None
    if closing:
        mask = np.zeros((32, 24), dtype=bool)
        mask[4:20, 2:10] = True
    figure = plot_plane_deconvolution_summary(
        record,
        planes["model"],
        planes["residual"],
        restored=planes["restored"],
        mask=mask,
        truth=planes["truth"],
        model_error=planes["model_error"],
        true_flux=2.5,
        threshold=1e-3,
        sigma=SIGMA,
        stop_descriptions=STOP_DESCRIPTIONS,
        final_entropy=None if closing else 6.5,
    )
    return figure, record, planes, mask


def _reference(ax, label):
    (line,) = (line for line in ax.lines if line.get_label() == label)
    return line.get_ydata()[0]


@pytest.mark.parametrize("closing", [True, False], ids=["cube", "continuum"])
def test_plane_deconvolution_summary_draws_the_images(closing):
    """Every image as ``plane.T``: the truth, the restored image and the model
    on an asinh scale linear to 0.01 of their largest value, the residual to
    5 sigma, the model error to its largest value; and the distribution of the
    residual in units of sigma as a density over all its pixels."""
    figure, _, planes, mask = _drawn(closing)
    images = figure.subfigs[0].axes
    names = ("truth", "restored", "model", "residual", "model_error")
    for ax, name in zip(images, names, strict=False):
        np.testing.assert_array_equal(ax.images[0].get_array(), planes[name].T)
    for ax, name in zip(images, names[:3], strict=False):
        largest = np.abs(planes[name]).max()
        assert ax.images[0].norm.linear_width == pytest.approx(0.01 * largest)
    norm = images[3].images[0].norm
    assert (norm.vmin, norm.vmax) == (-5 * SIGMA, 5 * SIGMA)
    largest = np.abs(planes["model_error"]).max()
    norm = images[4].images[0].norm
    assert (norm.vmin, norm.vmax) == (-largest, largest)
    residual = planes["residual"]
    if closing:
        regions = [("in the mask", residual[mask]), ("outside", residual[~mask])]
    else:
        regions = [("all pixels", residual.ravel())]
    histogram = images[5]
    edges = np.linspace(-6.0, 6.0, 97)
    assert _legend(histogram) == [
        label + (" (1 beyond ±6)" if label != "in the mask" else "")
        for label, _ in regions
    ] + ["N(0, 1)"]
    for step, (_, values) in zip(histogram.patches, regions, strict=True):
        weights = np.full(values.size, 1.0 / (values.size * np.diff(edges)[0]))
        heights, _ = np.histogram(values / SIGMA, edges, weights=weights)
        np.testing.assert_array_equal(
            np.unique(step.get_xy()[:, 1]), np.unique(np.r_[0.0, heights])
        )


@pytest.mark.parametrize("closing", [True, False], ids=["cube", "continuum"])
def test_plane_deconvolution_summary_draws_the_record(closing):
    """Every panel of the model updates draws its field of the record at the
    model updates, and the residual written as the final star."""
    from scipy.stats import kurtosis, skew

    figure, record, planes, mask = _drawn(closing)
    n = len(record["model_update_stop_code"])
    update = np.arange(1, n + 1)
    residual = planes["residual"]
    inside = residual.ravel() if mask is None else residual[mask]
    peak, flux, iterations, components, rms, moments, entropy = figure.subfigs[1].axes
    region = "in the mask" if closing else "in the image"

    def drawn(ax, label, values):
        (line,) = (line for line in ax.lines if line.get_label() == label)
        np.testing.assert_array_equal(line.get_xdata(), update)
        np.testing.assert_array_equal(line.get_ydata(), values)

    def stars(ax):
        lines = [line for line in ax.lines if line.get_marker() == "*"]
        assert all(list(line.get_xdata()) == [n + 1] for line in lines)
        return [line.get_ydata()[0] for line in lines]

    assert peak.get_title() == f"|peak residual| {region}"
    drawn(peak, "at the start", np.abs(record["start_peakres"]))
    drawn(peak, "at the end", np.abs(record["peakres"][:n]))
    assert stars(peak) == [np.abs(inside).max()]
    assert _threshold_per_cycle(figure) == pytest.approx(record["threshold_per_cycle"])
    assert _reference(peak, "threshold") == 1e-3
    drawn(flux, "model", record["model_flux"])
    assert _reference(flux, "truth") == 2.5
    assert flux.get_title() == "model flux (negative 1.00 %)"
    centres, heights = zip(
        *sorted(
            (patch.get_x() + patch.get_width() / 2, patch.get_height())
            for container in iterations.containers
            for patch in container.patches
        ),
        strict=True,
    )
    assert centres == pytest.approx(update)
    assert list(heights) == record["iter_done"][:n]
    assert iterations.get_title() == f"iterations ({sum(record['iter_done'])})"
    counts = {
        "residual-shaped": "n_residual_components",
        "Asp Gaussian": "n_scale_components",
        "significance (a subset)": "n_significance_components",
        "point": "n_point_components",
        "Hogbom runs": "n_hogbom_runs",
    }
    assert _legend(components) == [
        f"{label} ({sum(record[name])})" for label, name in counts.items()
    ]
    for name, line in zip(counts.values(), components.lines, strict=True):
        np.testing.assert_array_equal(line.get_xdata(), update)
        values = np.asarray(record[name], dtype=float)
        np.testing.assert_array_equal(
            line.get_ydata(), np.where(values > 0, values, np.nan)
        )
    drawn(rms, f"rms {region}", record["start_rms_in_mask"])
    expected = [np.sqrt(np.mean(inside**2))]
    if closing:
        drawn(rms, "MAD rms outside", record["start_mad_rms_outside_mask"])
        outside = residual[~mask]
        expected.append(1.4826 * np.median(np.abs(outside - np.median(outside))))
    assert stars(rms) == pytest.approx(expected)
    assert _reference(rms, "sigma") == SIGMA
    drawn(moments, "skew", record["start_skew_in_mask"])
    drawn(moments, "excess kurtosis", record["start_excess_kurtosis_in_mask"])
    assert stars(moments) == pytest.approx([skew(inside), kurtosis(inside)])
    # the entropy at the start of every model update, then the final star
    np.testing.assert_array_equal(entropy.lines[0].get_xdata(), update)
    np.testing.assert_array_equal(entropy.lines[0].get_ydata(), record["entropy"][:n])
    assert stars(entropy) == [record["entropy"][n] if closing else 6.5]


@pytest.mark.parametrize("closing", [True, False], ids=["cube", "continuum"])
def test_plane_deconvolution_summary_draws_the_history(closing):
    """Every panel of the iteration history draws its field per component,
    between the boundaries of the model updates, and the components where
    they are, with the net flux and number of each kind."""
    from matplotlib.collections import LineCollection, PathCollection

    figure, record, _, mask = _drawn(closing)
    n = len(record["model_update_stop_code"])
    history = figure.subfigs[2].axes
    flat = {
        name: np.concatenate(record[f"iteration_{name}"][:n])
        for name in (
            "peak_residual",
            "residual_rms",
            "model_flux",
            "component_scale",
            "component_l",
            "component_m",
            "component_flux",
        )
    }
    component = np.arange(flat["component_scale"].size)
    starts = np.cumsum(
        [0] + [np.size(s) for s in record["iteration_component_scale"][:n]]
    )

    def points(ax):
        return sorted(
            tuple(offset)
            for c in ax.collections
            if isinstance(c, PathCollection)
            for offset in c.get_offsets()
        )

    for ax, values in zip(
        history[:4],
        (
            np.abs(flat["peak_residual"]),
            flat["residual_rms"],
            None,
            flat["component_scale"],
        ),
        strict=True,
    ):
        if values is not None:
            assert points(ax) == sorted(zip(component, values, strict=True))
        boundaries = [
            line.get_xdata()[0]
            for line in ax.lines
            if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1]
        ]
        assert boundaries == list(starts[1:-1])
    region = "in the mask" if closing else "in the image"
    assert history[1].get_title() == f"residual rms {region} after it"
    np.testing.assert_array_equal(history[2].lines[0].get_xdata(), component)
    np.testing.assert_array_equal(history[2].lines[0].get_ydata(), flat["model_flux"])
    truth = [
        line.get_ydata()[0]
        for line in history[2].lines
        if list(line.get_xdata()) == [0, 1]
    ]
    assert truth == [2.5]
    # the record and the history count the same Asp Gaussians
    gaussians = [
        text for text in _legend(figure.subfigs[1].axes[3]) if text.startswith("Asp")
    ]
    assert gaussians == [text for text in _legend(history[0]) if text.startswith("Asp")]
    assert gaussians == ["Asp Gaussian (16)"]
    # threshold_per_cycle over the components of its model update
    segments = [
        (segment[0, 0], segment[1, 0], segment[0, 1])
        for lines in history[0].collections
        if isinstance(lines, LineCollection)
        for segment in lines.get_segments()
    ]
    assert segments == list(
        zip(starts[:-1], starts[1:], record["threshold_per_cycle"], strict=True)
    )
    # every component where the history puts it, the mask along l and m
    assert points(history[4]) == sorted(
        zip(flat["component_l"], flat["component_m"], strict=True)
    )
    outlines = [c for c in history[4].collections if not isinstance(c, PathCollection)]
    if closing:
        vertices = np.concatenate([path.vertices for path in outlines[0].get_paths()])
        assert [*vertices.min(axis=0), *vertices.max(axis=0)] == [3.5, 1.5, 19.5, 9.5]
    else:
        assert not outlines
    # the net flux and the number of every kind
    scale = flat["component_scale"]
    kinds = (scale < 0, scale > 0, scale == 0)
    assert [patch.get_height() for patch in history[5].patches] == pytest.approx(
        [flat["component_flux"][chosen].sum() for chosen in kinds]
    )
    assert [text.get_text() for text in history[5].texts] == [
        str(np.count_nonzero(chosen)) for chosen in kinds
    ]


def test_plane_deconvolution_summary_of_one_model_update():
    """One model update draws its threshold_per_cycle, and a stop code keeps
    its color whatever codes occur."""
    one = _record(1, True, asp=False, history=False)
    one["model_update_stop_code"] = [2]
    figures = [_summary(record) for record in (_record(4, True, False, False), one)]

    def color(figure, code):
        (bars,) = (
            bars
            for bars in figure.subfigs[1].axes[2].containers
            if bars.get_label() == f"{code} {STOP_DESCRIPTIONS[code]}"
        )
        return bars.patches[0].get_facecolor()

    assert color(figures[0], 2) == color(figures[1], 2) != color(figures[0], 1)
    assert _threshold_per_cycle(figures[1]) == pytest.approx(one["threshold_per_cycle"])


def test_plane_deconvolution_summary_without_iterations():
    """Model updates of no iteration draw, and a run without a model update
    draws its images only: no record (continuum imaging) or one without the
    fields of the model updates (cube imaging)."""
    idle = _record(3, True, asp=False, history=False)
    idle["iter_done"] = [0, 0, 0, 0]
    idle["n_point_components"] = [0, 0, 0]
    assert _summary(idle).subfigs[1].axes[2].get_ylim()[0] == 0
    for record in (None, {"iter_done": [0], "peakres": [2.3], "entropy": [np.nan]}):
        figure = _summary(record, title="plane")
        assert len(figure.subfigs) == 1
        assert figure.get_suptitle() == "plane: no model update"


def test_plane_deconvolution_summary_of_degenerate_planes():
    """An all-NaN plane, a mask of no pixel, NaN pixels, iteration counts of
    None and an all-NaN residual draw without a warning; the final stars are
    those of the finite pixels."""
    import warnings

    plane = np.random.default_rng(1).normal(size=(32, 32))
    plane[:4] = np.nan
    plane[10, 10] = np.nan
    mask = np.zeros(plane.shape, dtype=bool)
    mask[8:24, 8:24] = True
    record = _record(2, False, asp=False, history=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        figure = plot_plane_deconvolution_summary(
            record,
            np.full_like(plane, np.nan),
            plane,
            restored=np.zeros_like(plane),
            mask=np.zeros(plane.shape, dtype=bool),
        )
        titles = [ax.get_title() for ax in figure.subfigs[0].axes]
        assert "model (no finite pixel)" in titles
        figure = plot_plane_deconvolution_summary(
            record, np.abs(plane), plane, mask=mask
        )
        (star,) = (
            line for line in figure.subfigs[1].axes[0].lines if line.get_marker() == "*"
        )
        assert star.get_ydata()[0] == np.nanmax(np.abs(plane[mask]))
        idle = record | {"iter_done": [None, None]}
        figure = plot_plane_deconvolution_summary(
            idle, np.abs(plane), np.full_like(plane, np.nan)
        )
        assert figure.subfigs[1].axes[2].get_ylim() == (0.0, 1.45)


def test_signed_images_and_visibility_comparison():
    rng = np.random.default_rng(4)
    planes = [rng.normal(size=(16, 12)) for _ in range(2)] + [np.zeros((16, 12))]
    mask = np.zeros((16, 12), dtype=bool)
    mask[2:6, 1:9] = True
    figure = plot_signed_images(planes, ["a", "b", "c"], masks=[mask, None, None])
    _assert_pyplot_free(figure, 3)
    np.testing.assert_array_equal(figure.axes[0].images[0].get_array(), planes[0].T)
    # every panel on a linear scale symmetric about 0, to its largest value
    for ax, plane in zip(figure.axes[:3], planes, strict=True):
        largest = np.abs(plane).max() or 1.0
        norm = ax.images[0].norm
        assert (norm.vmin, norm.vmax) == (-largest, largest)
    # the mask outlined on the first panel only, along l (x) and m (y)
    assert [len(ax.collections) for ax in figure.axes[:3]] == [1, 0, 0]
    vertices = np.concatenate(
        [path.vertices for path in figure.axes[0].collections[0].get_paths()]
    )
    assert [*vertices.min(axis=0), *vertices.max(axis=0)] == [1.5, 0.5, 5.5, 8.5]
    assert figure.axes[2].get_title() == "c (all zero)"
    figure = plot_signed_images([np.where(mask, np.nan, 0.0)], ["d"])
    assert figure.axes[0].get_title() == "d (all zero)"

    expected = rng.normal(size=50) + 1j * rng.normal(size=50)
    visibility = expected.copy()
    visibility[::3] += 1e-12
    uv_distance = np.linspace(1e3, 1e5, 50)
    noise = rng.normal(size=500)
    noise[0] = 7.0
    figure = plot_visibility_comparison(uv_distance, visibility, expected, noise)
    _assert_pyplot_free(figure, 3)
    # the noise as a density over all the samples, against N(0, 1)
    ax = figure.axes[2]
    assert _legend(ax) == ["noise (1 beyond ±5)", "N(0, 1)"]
    edges = np.linspace(-5.0, 5.0, 81)
    weights = np.full(noise.size, 1.0 / (noise.size * np.diff(edges)[0]))
    heights, _ = np.histogram(noise, edges, weights=weights)
    np.testing.assert_array_equal(
        np.unique(ax.patches[0].get_xy()[:, 1]), np.unique(np.r_[0.0, heights])
    )
    (gaussian,) = ax.lines
    x = np.linspace(-5.0, 5.0, 201)
    np.testing.assert_array_equal(gaussian.get_xdata(), x)
    np.testing.assert_array_equal(
        gaussian.get_ydata(), np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    )
    assert ax.get_title() == (
        f"noise: mean {noise.mean():.3f}, standard deviation {noise.std():.3f}"
    )
    for line, values in zip(figure.axes[0].lines, (expected, visibility), strict=True):
        np.testing.assert_array_equal(line.get_xdata(), uv_distance / 1e3)
        np.testing.assert_array_equal(line.get_ydata(), np.abs(values))
    unequal = visibility != expected
    (difference,) = figure.axes[1].lines
    np.testing.assert_array_equal(difference.get_xdata(), uv_distance[unequal] / 1e3)
    np.testing.assert_array_equal(
        difference.get_ydata(), np.abs(visibility - expected)[unequal]
    )
    legend = figure.axes[0].get_legend()
    assert [text.get_text() for text in legend.get_texts()] == [
        "prediction",
        "visibilities",
    ]
    assert [handle.get_marker() for handle in legend.legend_handles] == ["o", "."]
    assert figure.axes[1].get_title() == "difference (33 samples equal)"
