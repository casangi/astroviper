"""
``least_squares_params``: the last model update solves the model inside the
clean mask by conjugate residual, with the PSF of twice the image size.

On arrays: the solution against a dense solve of the normal equations, with
and without a smooth basis, the residual against a direct convolution, the stop
rule against its own record, planes without a mask, a PSF per plane, the
precision kept, the history. Through the 48 x 48 imaging cycle loop of
``test_max_cycles_auto``: the record ends with the stop test and keeps why the
cycles ended, the residual written is the one the steps predict, the solve
needs the PSF of twice the image size, the mask solved on is the mask written,
no model update means no solve, and the summary figure draws the steps.
"""

import numpy as np
import pytest
from scipy.signal import convolve2d

from astroviper.processing_functions.imaging.deconvolvers import least_squares
from astroviper.processing_functions.imaging.deconvolvers.least_squares import (
    least_squares_solve,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_MAX_CYCLES,
    IMAGING_NO_PROGRESS,
    MODEL_UPDATE_ALGORITHM,
    MODEL_UPDATE_CONTINUE,
    MODEL_UPDATE_MAX_ITER_PER_CYCLE,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    COMPONENT_COUNT_FIELDS,
)
from tests.unit.processing_functions.imaging.test_exact_psf_subtraction import _run

N = 24
BOX = (slice(8, 14), slice(9, 16))


def _psf(width=1.5):
    """A Gaussian PSF of twice the image size, peak 1 at its centre."""
    offsets = np.arange(2 * N) - N
    return np.exp(-(offsets[:, None] ** 2 + offsets[None, :] ** 2) / (2 * width**2))


def _problem(mask, dtype=np.float64, seed=0, width=1.5):
    """Cubes of one plane: residual, PSF, model and mask."""
    residual = np.random.default_rng(seed).normal(size=(N, N)).astype(dtype)
    return (
        residual[None, None, None].copy(),
        _psf(width)[None, None, None],
        np.zeros((1, 1, 1, N, N), dtype=dtype),
        mask[None, None, None],
    )


def _box_mask():
    mask = np.zeros((N, N), dtype=bool)
    mask[BOX] = True
    mask[10, 9:13] = False
    return mask


def _solve(cubes, max_steps, record=False, **params):
    residual, psf, model, mask = cubes
    return least_squares_solve(
        residual,
        psf,
        model,
        {
            "least_squares_params": {"max_steps": max_steps, **params},
            "record_iteration_history": record,
        },
        mask_cube=mask,
    )


def _dense(mask, width):
    """``M P M`` on the mask's pixels, from the PSF."""
    pixels = np.argwhere(mask)
    offsets = pixels[:, None, :] - pixels[None, :, :]
    return _psf(width)[offsets[..., 0] + N, offsets[..., 1] + N]


def test_solves_the_normal_equations():
    mask = _box_mask()
    cubes = _problem(mask, width=0.7)
    expected = np.linalg.solve(_dense(mask, 0.7), cubes[0][0, 0, 0][mask])

    _solve(cubes, int(mask.sum()))

    np.testing.assert_allclose(cubes[2][0, 0, 0][mask], expected, atol=1e-8)
    assert not cubes[2][0, 0, 0][~mask].any()


def test_a_smooth_basis_solves_the_normal_equations_in_it():
    from scipy.ndimage import gaussian_filter

    mask = _box_mask()
    cubes = _problem(mask, width=0.7)
    pixels = np.argwhere(mask)
    sigma = 1.2 / (2 * np.sqrt(2 * np.log(2)))
    smooth = np.empty((len(pixels), len(pixels)))
    for column, (row, col) in enumerate(pixels):
        unit = np.zeros((N, N))
        unit[row, col] = 1.0
        box = gaussian_filter(unit[BOX], sigma, mode="constant", truncate=3.5)
        smooth[:, column] = box[mask[BOX]]
    normal = smooth @ _dense(mask, 0.7) @ smooth
    coefficients = np.linalg.solve(normal, smooth @ cubes[0][0, 0, 0][mask])

    _solve(cubes, 400, smooth_fwhm_pixels=1.2)

    np.testing.assert_allclose(
        cubes[2][0, 0, 0][mask], smooth @ coefficients, atol=1e-7
    )
    assert not cubes[2][0, 0, 0][~mask].any()


def test_the_residual_follows_the_model_in_the_box():
    mask = _box_mask()
    cubes = _problem(mask)
    start = cubes[0][0, 0, 0].copy()

    _solve(cubes, 5)

    response = convolve2d(cubes[2][0, 0, 0], _psf()[1:, 1:], mode="same")
    np.testing.assert_allclose(
        cubes[0][0, 0, 0][BOX], (start - response)[BOX], atol=1e-9
    )
    outside = np.ones_like(mask)
    outside[BOX] = False
    assert np.array_equal(cubes[0][0, 0, 0][outside], start[outside])


def test_stops_a_window_after_the_rms_outside_the_mask_levels(monkeypatch):
    monkeypatch.setattr(least_squares, "STOP_EVERY", 2)
    monkeypatch.setattr(least_squares, "STOP_WINDOW", 6)
    mask = _box_mask()
    rng = np.random.default_rng(0)
    sky = np.where(mask, rng.uniform(0.5, 1.5, (N, N)), 0.0)
    residual = convolve2d(sky, _psf()[1:, 1:], mode="same")
    residual += 0.01 * rng.normal(size=(N, N))
    cubes = (
        residual[None, None, None].copy(),
        _psf()[None, None, None],
        np.zeros((1, 1, 1, N, N)),
        mask[None, None, None],
    )

    results = _solve(cubes, 400)

    steps = results["iterations_performed"][0, 0, 0]
    rms = results["least_squares_rms_outside_mask"][0, 0, 0]
    assert results["stop_code"][0, 0, 0] == MODEL_UPDATE_ALGORITHM
    assert len(rms) == steps // 2
    fires = [rms[i] > 0.95 * rms[i - 3] for i in range(3, len(rms))]
    assert len(fires) > 1 and fires[-1] and not any(fires[:-1])
    outside = cubes[0][0, 0, 0][BOX][~mask[BOX]]
    np.testing.assert_allclose(rms[-1], np.sqrt(np.mean(outside**2)))


def test_max_steps_ends_a_plane():
    results = _solve(_problem(_box_mask()), 4)
    assert results["iterations_performed"][0, 0, 0] == 4
    assert results["stop_code"][0, 0, 0] == MODEL_UPDATE_MAX_ITER_PER_CYCLE
    assert "iteration_peak_residual" not in results


def test_an_exact_solution_is_no_divergence():
    cubes = list(_problem(_box_mask()))
    cubes[1] = np.zeros_like(cubes[1])
    cubes[1][0, 0, 0, N, N] = 1.0
    results = _solve(cubes, 10)
    assert results["iterations_performed"][0, 0, 0] == 1
    assert results["stop_code"][0, 0, 0] == MODEL_UPDATE_ALGORITHM


def test_a_plane_without_mask_or_with_a_bad_residual_is_left_alone():
    empty = _problem(np.zeros((N, N), dtype=bool))
    start = empty[0].copy()
    results = _solve(empty, 5)
    assert results["iterations_performed"][0, 0, 0] == 0
    assert np.array_equal(empty[0], start)
    assert not empty[2].any()

    bad = _problem(_box_mask())
    bad[0][0, 0, 0, 9, 10] = np.nan
    results = _solve(bad, 5)
    assert results["iterations_performed"][0, 0, 0] == 0
    assert results["stop_code"][0, 0, 0] == MODEL_UPDATE_CONTINUE
    assert not bad[2].any()

    flagged = _problem(_box_mask())
    flagged[0][...] = 0.0
    flagged[1][...] = 0.0
    assert _solve(flagged, 5)["stop_code"][0, 0, 0] == MODEL_UPDATE_CONTINUE


def test_a_psf_without_a_peak_of_one_is_refused():
    cubes = list(_problem(_box_mask()))
    cubes[1] = -cubes[1]
    with pytest.raises(ValueError, match="peaks at"):
        _solve(cubes, 5)


def test_a_direct_call_checks_its_parameters():
    with pytest.raises(ValueError, match=">= 1"):
        _solve(_problem(_box_mask()), 0)


def test_keeps_the_precision_of_the_image_and_works_in_float64():
    cubes = _problem(_box_mask(), dtype=np.float32)
    double = [cube.astype(np.float64) for cube in cubes[:3]] + [cubes[3]]
    _solve(cubes, 5)
    _solve(double, 5)
    np.testing.assert_array_equal(cubes[0], double[0].astype(np.float32))
    np.testing.assert_array_equal(cubes[2], double[2].astype(np.float32))
    assert cubes[2].any()


def test_every_plane_takes_its_own_psf():
    planes = [_problem(_box_mask(), seed=i, width=0.6 + 0.2 * i) for i in range(8)]
    cube = [np.stack([plane[i][0, 0, 0] for plane in planes]) for i in range(4)]
    cube = [part.reshape(2, 2, 2, *part.shape[1:]) for part in cube]
    _solve(cube, 5)
    for index, plane in zip(np.ndindex(2, 2, 2), planes, strict=True):
        _solve(plane, 5)
        assert np.array_equal(cube[0][index], plane[0][0, 0, 0])
        assert np.array_equal(cube[2][index], plane[2][0, 0, 0])


def test_the_history_has_one_entry_per_step():
    mask = _box_mask()
    cubes = _problem(mask)
    start = cubes[0][0, 0, 0][mask]
    results = _solve(cubes, 6, record=True)
    peak, rms, flux, l, m, scale = (
        results[name][0][0][0]
        for name in (
            "iteration_peak_residual",
            "iteration_residual_rms",
            "iteration_component_flux",
            "iteration_component_l",
            "iteration_component_m",
            "iteration_component_scale",
        )
    )
    assert all(values.shape == (6,) for values in (peak, rms, flux, l, m, scale))
    assert peak[0] == start[np.abs(start).argmax()]
    assert rms[-1] == pytest.approx(np.sqrt(np.mean(cubes[0][0, 0, 0][mask] ** 2)))
    assert flux.sum() == pytest.approx(cubes[2].sum())
    assert (l == -1).all() and (m == -1).all() and (scale == -2).all()
    for field in COMPONENT_COUNT_FIELDS:
        assert not results[field].any()


# ---------------------------------------------------------------------------
# The imaging cycle loop
# ---------------------------------------------------------------------------


def _mask():
    """Wider than half the image, so only the PSF of twice the image size holds
    every offset of its box."""
    mask = np.zeros((48, 48), dtype=bool)
    mask[6:42, 6:42] = True
    return mask


CYCLES = {"max_cycles": 2, "record_iteration_history": True}
NO_PROGRESS = {**CYCLES, "max_cycles": -1, "threshold": 0.0, "max_iter_per_cycle": 0}


@pytest.mark.parametrize(
    "controls, stop", [(CYCLES, IMAGING_MAX_CYCLES), (NO_PROGRESS, IMAGING_NO_PROGRESS)]
)
def test_the_solve_is_the_last_model_update(controls, stop):
    _, _, record = _run(
        controls, clean_mask=_mask(), least_squares_params={"max_steps": 30}
    )
    fields = next(iter(record.data.values()))
    assert fields["iter_done"][-1] == 0
    assert 0 < fields["iter_done"][-2] <= 30
    assert fields["iter_done"][-2] == len(fields["iteration_residual_rms"][-2])
    assert len(fields["least_squares_rms_outside_mask"]) == 0
    assert fields["stop_code"].imaging == stop
    assert fields["stop_code"].model_update == fields["model_update_stop_code"][-1]


def test_the_residual_written_is_the_one_the_steps_predict():
    _, _, record = _run(
        CYCLES, clean_mask=_mask(), least_squares_params={"max_steps": 30}
    )
    fields = next(iter(record.data.values()))
    predicted = fields["iteration_residual_rms"][-2][-1]
    np.testing.assert_allclose(fields["rms_in_mask"][-1], predicted, rtol=1e-3)


def test_the_solve_lowers_the_residual_in_the_mask():
    plain, _, _ = _run(CYCLES, clean_mask=_mask())
    solved, _, _ = _run(
        CYCLES, clean_mask=_mask(), least_squares_params={"max_steps": 30}
    )
    inside = _mask()
    assert (
        solved.SKY_RESIDUAL.values[0, 0, 0][inside].std()
        < plain.SKY_RESIDUAL.values[0, 0, 0][inside].std()
    )


def test_the_mask_solved_on_is_the_mask_written(monkeypatch):
    solved_on = []
    solve = least_squares.least_squares_solve

    def spy(*args, mask_cube, **kwargs):
        solved_on.append(mask_cube.copy())
        return solve(*args, mask_cube=mask_cube, **kwargs)

    monkeypatch.setattr(least_squares, "least_squares_solve", spy)
    image, _, _ = _run(
        CYCLES,
        auto_mask_params={},
        least_squares_params={"max_steps": 5},
    )
    assert len(solved_on) == 1
    assert np.array_equal(solved_on[0], image.MASK.values)


@pytest.mark.parametrize("controls", [{"max_cycles": 0}, {"max_iter": 0}])
def test_no_model_update_means_no_solve(monkeypatch, controls):
    def no_solve(*args, **kwargs):
        raise AssertionError("solved without a model update")

    monkeypatch.setattr(least_squares, "least_squares_solve", no_solve)
    image, _, _ = _run(
        controls, clean_mask=_mask(), least_squares_params={"max_steps": 5}
    )
    assert not image.SKY_MODEL.values.any()


@pytest.mark.parametrize("deconvolver", ["asp", "hogbom"])
def test_the_summary_draws_the_steps(deconvolver):
    from astroviper.utils.report_plots import plot_plane_deconvolution_summary

    image, _, record = _run(
        CYCLES,
        deconvolver=deconvolver,
        clean_mask=_mask(),
        least_squares_params={"max_steps": 30},
    )
    fields = next(fields for key, fields in record.data.items() if key.pol == 0)
    if deconvolver == "asp":
        for name in COMPONENT_COUNT_FIELDS:
            assert len(fields[name]) == len(fields["model_update_stop_code"]), name
    figure = plot_plane_deconvolution_summary(
        fields,
        image.SKY_MODEL.values[0, 0, 0],
        image.SKY_RESIDUAL.values[0, 0, 0],
        mask=_mask(),
    )
    labels = [
        text.get_text()
        for axis in figure.axes
        if axis.get_legend() is not None
        for text in axis.get_legend().get_texts()
    ]
    assert any(label.startswith("least-squares step") for label in labels)
