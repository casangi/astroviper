"""
``least_squares_params``: the last model update solves the model inside the
clean mask by conjugate residual, with the PSF of twice the image size.

On arrays: the solution against a dense solve of the normal equations, the
residual against a direct convolution, the stop on a plateau, planes without a
mask, the precision kept. Through the 48 x 48 imaging cycle loop of
``test_max_cycles_auto``: ``None`` changes nothing, the record ends with the
stop test, the residual written is the one the steps predict, the mask written
is the mask solved on, and no model update means no solve.
"""

import importlib

import numpy as np
import pytest
from scipy.signal import convolve2d

from astroviper.processing_functions.imaging.config_check import (
    CUBE_SINGLE_FIELD,
    check_model_update,
)
from astroviper.processing_functions.imaging.config_check._predicates import (
    check_least_squares_params,
)
from astroviper.processing_functions.imaging.deconvolvers import least_squares
from astroviper.processing_functions.imaging.deconvolvers.least_squares import (
    least_squares_solve,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    MODEL_UPDATE_ALGORITHM,
    MODEL_UPDATE_MAX_ITER_PER_CYCLE,
)
from tests.unit.processing_functions.imaging.test_exact_psf_subtraction import (
    CONTROLS,
    _run,
    _same,
)

N = 24


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
    mask[8:14, 9:16] = True
    mask[10, 9:13] = False
    return mask


def _solve(cubes, max_steps, record=False):
    residual, psf, model, mask = cubes
    return least_squares_solve(
        residual,
        psf,
        model,
        {
            "least_squares_params": {"max_steps": max_steps},
            "record_iteration_history": record,
        },
        mask_cube=mask,
    )


def test_solves_the_normal_equations():
    mask = _box_mask()
    cubes = _problem(mask, width=0.7)
    start = cubes[0][0, 0, 0].copy()
    pixels = np.argwhere(mask)
    offsets = pixels[:, None, :] - pixels[None, :, :]
    matrix = _psf(0.7)[offsets[..., 0] + N, offsets[..., 1] + N]
    expected = np.linalg.solve(matrix, start[mask])

    _solve(cubes, len(pixels))

    np.testing.assert_allclose(cubes[2][0, 0, 0][mask], expected, atol=1e-8)
    assert not cubes[2][0, 0, 0][~mask].any()


def test_the_residual_follows_the_model_in_the_box():
    mask = _box_mask()
    cubes = _problem(mask)
    start = cubes[0][0, 0, 0].copy()

    _solve(cubes, 5)

    model = cubes[2][0, 0, 0]
    box = (slice(8, 14), slice(9, 16))
    response = convolve2d(model, _psf()[1:, 1:], mode="same", boundary="fill")
    np.testing.assert_allclose(
        cubes[0][0, 0, 0][box], (start - response)[box], atol=1e-9
    )
    outside = np.ones_like(mask)
    outside[box] = False
    assert np.array_equal(cubes[0][0, 0, 0][outside], start[outside])


def test_stops_when_the_rms_outside_the_mask_stops_falling(monkeypatch):
    monkeypatch.setattr(least_squares, "STOP_EVERY", 1)
    monkeypatch.setattr(least_squares, "STOP_WINDOW", 2)
    cubes = _problem(_box_mask(), width=0.3)

    results = _solve(cubes, 50)

    assert results["iterations_performed"][0, 0, 0] == 3
    assert results["stop_code"][0, 0, 0] == MODEL_UPDATE_ALGORITHM


def test_max_steps_ends_a_plane():
    results = _solve(_problem(_box_mask()), 4)
    assert results["iterations_performed"][0, 0, 0] == 4
    assert results["stop_code"][0, 0, 0] == MODEL_UPDATE_MAX_ITER_PER_CYCLE


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
    assert _solve(bad, 5)["iterations_performed"][0, 0, 0] == 0
    assert not bad[2].any()


def test_a_psf_without_a_peak_of_one_is_refused():
    cubes = list(_problem(_box_mask()))
    cubes[1] = -cubes[1]
    with pytest.raises(ValueError, match="peaks at"):
        _solve(cubes, 5)


def test_keeps_the_precision_of_the_image():
    cubes = _problem(_box_mask(), dtype=np.float32)
    _solve(cubes, 5)
    assert cubes[0].dtype == np.float32
    assert cubes[2].dtype == np.float32
    assert cubes[2].any()


def test_the_history_has_one_entry_per_step():
    results = _solve(_problem(_box_mask()), 6, record=True)
    for name in (
        "iteration_peak_residual",
        "iteration_residual_rms",
        "iteration_component_flux",
        "iteration_component_l",
        "iteration_component_scale",
    ):
        assert results[name][0][0][0].shape == (6,)
    assert (results["iteration_component_scale"][0][0][0] == -2).all()


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, error, match",
    [
        (["max_steps"], TypeError, "must be a mapping"),
        ({}, ValueError, "needs max_steps"),
        ({"steps": 5}, ValueError, "Unknown least_squares_params keys"),
        ({"max_steps": 0}, ValueError, ">= 1"),
        ({"max_steps": 1.5}, TypeError, "integer"),
    ],
)
def test_refuses_a_bad_value(value, error, match):
    with pytest.raises(error, match=match):
        check_least_squares_params(value)


def test_needs_a_mask_and_a_gridded_image():
    good = {"least_squares_params": {"max_steps": 5}, "auto_mask_params": {}}
    check_model_update(good, mode=CUBE_SINGLE_FIELD)
    with pytest.raises(ValueError, match="clean_mask or auto_mask_params"):
        check_model_update(
            {"least_squares_params": {"max_steps": 5}}, mode=CUBE_SINGLE_FIELD
        )
    with pytest.raises(ValueError, match="gridded image"):
        check_model_update(
            {**good, "gridding_kernel_params": {"type": "dft"}}, mode=CUBE_SINGLE_FIELD
        )


# ---------------------------------------------------------------------------
# The imaging cycle loop
# ---------------------------------------------------------------------------


def _mask():
    mask = np.zeros((48, 48), dtype=bool)
    mask[14:34, 14:34] = True
    return mask


CYCLES = {"max_cycles": 2, "record_iteration_history": True}


def test_none_changes_nothing():
    _same(_run(CONTROLS), _run(CONTROLS, least_squares_params=None))


def test_the_solve_is_the_last_model_update():
    image, _, record = _run(
        CYCLES, clean_mask=_mask(), least_squares_params={"max_steps": 30}
    )
    fields = next(iter(record.data.values()))
    assert len(fields["iter_done"]) == 2 + 1 + 1
    assert fields["iter_done"][-1] == 0
    assert 0 < fields["iter_done"][-2] <= 30
    assert fields["iter_done"][-2] == len(fields["iteration_residual_rms"][-2])


def test_the_residual_written_is_the_one_the_steps_predict():
    image, _, record = _run(
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


def test_the_mask_written_is_the_mask_solved_on(monkeypatch):
    module = importlib.import_module(
        "astroviper.processing_functions.image_analysis.make_auto_mask"
    )
    calls = []
    make_auto_mask = module.make_auto_mask

    def spy(*args, **kwargs):
        calls.append(kwargs.get("planes_stopped"))
        return make_auto_mask(*args, **kwargs)

    monkeypatch.setattr(module, "make_auto_mask", spy)
    _run(CYCLES, auto_mask_params={}, least_squares_params={"max_steps": 5})
    assert len(calls) == 3


@pytest.mark.parametrize("controls", [{"max_cycles": 0}, {"max_iter": 0}])
def test_no_model_update_means_no_solve(monkeypatch, controls):
    def no_solve(*args, **kwargs):
        raise AssertionError("solved without a model update")

    monkeypatch.setattr(least_squares, "least_squares_solve", no_solve)
    image, _, _ = _run(
        controls, clean_mask=_mask(), least_squares_params={"max_steps": 5}
    )
    assert not image.SKY_MODEL.values.any()
