"""Conjugate gradient and Nesterov momentum steps of the continuum imaging cycle.

The toy problems are image-domain imaging cycles on two independent
(time, polarization) planes: a circular block-Hankel Taylor Hessian ``H``,
the dirty image ``b = H sky`` and the residual update ``r(theta) = b - H theta``.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.cycle_acceleration import (
    conjugate_gradient_model_step,
    nesterov_momentum_model_step,
)

DIMS = ("time", "taylor_term", "polarization", "l", "m")
PLANE_DIMS = ("time", "polarization")
SIZE = 16
# Fractional frequency offsets (nu - nu_0) / nu_0 of the Taylor expansion.
FREQUENCY_OFFSETS = np.array([-0.4, -0.1, 0.2, 0.5])


def _dot(first, second):
    """Per-plane inner product over Taylor terms and pixels."""
    return np.einsum("atpxy,atpxy->ap", first, second)


def _planes(values):
    return values[:, None, :, None, None]


class _TaylorProblem:
    """Two (time, polarization) planes, each with its own PSF.

    ``"gaussian"`` PSFs are strictly positive definite; ``"coverage"`` PSFs
    sample random uv weights without the zero spacing and have sidelobes.
    The Taylor PSFs are ``PSF_k = mean_nu(x_nu**k) PSF``.
    """

    def __init__(self, nterms, psf="gaussian", widths=(0.6, 0.75), seed=1):
        rng = np.random.default_rng(seed)
        kl, km = np.meshgrid(np.fft.fftfreq(SIZE), np.fft.fftfreq(SIZE), indexing="ij")
        radius = np.hypot(kl, km)
        transfers = []
        for width in widths:
            if psf == "gaussian":
                transfer = np.exp(-2 * np.pi**2 * (width * radius) ** 2)
            else:
                weights = rng.uniform(0.2, 1.0, radius.shape)
                weights = 0.5 * (weights + np.roll(weights[::-1, ::-1], 1, (0, 1)))
                transfer = weights * ((radius >= 1 / SIZE) & (radius <= 5 / SIZE))
                transfer *= np.exp(-0.5 * (radius * SIZE / (4 * width)) ** 2)
            transfers.append(transfer / transfer.mean())
        moments = [np.mean(FREQUENCY_OFFSETS**order) for order in range(2 * nterms - 1)]
        self.transfer = np.asarray([[m * t for t in transfers] for m in moments])
        self.nterms = nterms
        self.sky = np.zeros((1, nterms, 2, SIZE, SIZE))
        for pol in range(2):
            pixels = rng.choice(np.arange(3, SIZE - 3), (4, 2))
            flux = rng.uniform(0.3, 1.0, 4)
            for term in range(nterms):
                self.sky[0, term, pol, pixels[:, 0], pixels[:, 1]] = (
                    flux * (-0.7) ** term
                )
        self.dirty = self.hessian(self.sky)
        hessian = self._fourier_hessian(self.transfer)
        self.largest_eigenvalue = np.linalg.eigvalsh(hessian).max()

    def _fourier_hessian(self, transfer):
        """Taylor Hessian per (polarization, l, m) Fourier mode."""
        terms = np.arange(self.nterms)
        return np.moveaxis(transfer[np.add.outer(terms, terms)], (0, 1), (-2, -1))

    def hessian(self, theta, transfer=None):
        transfer = self.transfer if transfer is None else transfer
        theta_fourier = np.fft.fft2(theta)
        result = np.zeros_like(theta_fourier)
        for s in range(self.nterms):
            for t in range(self.nterms):
                result[:, s] += transfer[s + t] * theta_fourier[:, t]
        return np.fft.ifft2(result).real.copy()

    def residual(self, theta):
        return self.dirty - self.hessian(theta)

    def inverse(self, residual):
        residual_fourier = np.moveaxis(np.fft.fft2(residual)[0], 0, -1)[..., None]
        solution = np.linalg.solve(
            self._fourier_hessian(self.transfer), residual_fourier
        )
        return np.fft.ifft2(np.moveaxis(solution[..., 0], -1, 0)[None]).real.copy()


def _richardson(problem, residual, transfer=None):
    """Fixed preconditioner ``z = r / lambda_max``; returns ``z, r - B z``."""
    increment = residual / problem.largest_eigenvalue
    return increment, residual - problem.hessian(increment, transfer)


def _exact_inverse(problem, residual):
    """Exact inverse ``z = H^-1 r``."""
    increment = problem.inverse(residual)
    return increment, residual - problem.hessian(increment)


def _hogbom(problem, residual, cycle_threshold, threshold, max_iter, gain=0.1):
    """Högbom on Taylor term zero with the circular PSF.

    Returns the increment, the deconvolver's residual (only Taylor term zero
    is updated) and the planes whose own residual reached ``threshold``.
    """
    psf = np.fft.ifft2(problem.transfer[0]).real
    increment = np.zeros_like(residual)
    after = residual.copy()
    for pol in range(residual.shape[2]):
        plane = after[0, 0, pol]
        for _ in range(max_iter):
            peak = np.unravel_index(np.argmax(np.abs(plane)), plane.shape)
            if abs(plane[peak]) <= cycle_threshold[pol]:
                break
            component = gain * plane[peak]
            increment[0, 0, pol][peak] += component
            plane -= component * np.roll(psf[pol], peak, axis=(0, 1))
    return increment, after, np.abs(after[:, 0]).max(axis=(-2, -1)) <= threshold


def _da(values):
    return xr.DataArray(values, dims=DIMS)


def _step(acceleration, residual, after, increment, state, plain=None, mu=0.5):
    """Apply one acceleration to numpy Taylor stacks."""
    plain = None if plain is None else xr.DataArray(plain, dims=PLANE_DIMS)
    if acceleration == "none":
        return increment, None
    if acceleration == "conjugate_gradient":
        step, state = conjugate_gradient_model_step(
            _da(residual), _da(after), _da(increment), plain, state
        )
    else:
        step, state = nesterov_momentum_model_step(
            _da(residual), _da(increment), plain, state, momentum_coefficient=mu
        )
    return step.values, state


def _converged(problem, residual):
    """``||r|| < 1e-6 ||b||`` on every plane."""
    return np.all(_dot(residual, residual) < 1e-12 * _dot(problem.dirty, problem.dirty))


def _cycles_to_tolerance(problem, acceleration, deconvolve, max_cycles, mu=0.5):
    """Imaging cycles until :func:`_converged`, or ``None``."""
    theta = np.zeros_like(problem.dirty)
    state = None
    for cycle in range(max_cycles + 1):
        residual = problem.residual(theta)
        if _converged(problem, residual):
            return cycle
        increment, after = deconvolve(problem, residual)
        step, state = _step(acceleration, residual, after, increment, state, mu=mu)
        theta = theta + step
    return None


def _cycles_to_threshold(problem, acceleration, max_iter, max_cycles=200):
    """Imaging cycles until the Taylor-term-zero peak reaches the threshold."""
    threshold = 1e-3 * np.abs(problem.dirty[:, 0]).max()
    theta = np.zeros_like(problem.dirty)
    state = None
    for cycle in range(max_cycles + 1):
        residual = problem.residual(theta)
        peak = np.abs(residual[0, 0]).max(axis=(-2, -1))
        if np.all(peak <= threshold):
            return cycle
        increment, after, reached = _hogbom(
            problem, residual, np.maximum(0.2 * peak, threshold), threshold, max_iter
        )
        step, state = _step(acceleration, residual, after, increment, state, reached)
        theta = theta + step
    return None


def _textbook_pcg(problem, n_cycles):
    """Models of preconditioned CG with the preconditioner ``1 / lambda_max``."""
    theta = np.zeros_like(problem.dirty)
    residual = problem.dirty
    direction = residual / problem.largest_eigenvalue
    r_z = _dot(residual, direction)
    models = []
    for _ in range(n_cycles):
        alpha = r_z / _dot(direction, problem.hessian(direction))
        theta = theta + _planes(alpha) * direction
        models.append(theta)
        residual = problem.residual(theta)
        increment = residual / problem.largest_eigenvalue
        r_z, previous_r_z = _dot(residual, increment), r_z
        direction = increment + _planes(r_z / previous_r_z) * direction
    return models


def _pixels(*planes, dtype=np.float64):
    """(time=1, polarization, l=2, m=2) from one 4-pixel list per plane."""
    values = np.asarray(planes, dtype=dtype).reshape(1, len(planes), 2, 2)
    return xr.DataArray(values, dims=("time", "polarization", "l", "m"))


def _cg_state(direction, direction_convolved, previous_residual):
    return xr.Dataset(
        {
            "SKY_MODEL_DIRECTION": direction,
            "SKY_MODEL_DIRECTION_CONVOLVED": direction_convolved,
            "SKY_RESIDUAL_PREVIOUS": previous_residual,
        },
        attrs={"cycle_acceleration": "conjugate_gradient"},
    )


def _momentum_state(
    momentum, previous_norm=np.inf, cycles=10, grew=False, disabled=False
):
    """Momentum state over the planes of :func:`_pixels`. With the default
    ``cycles`` the next coefficient is the cap ``mu`` for any ``mu <= 10 / 13``."""
    shape = momentum.shape[:2]
    return xr.Dataset(
        {
            "SKY_MODEL_MOMENTUM": momentum,
            "SKY_RESIDUAL_PREVIOUS_NORM": (
                PLANE_DIMS,
                np.array(np.broadcast_to(previous_norm, shape), dtype=np.float64),
            ),
            "MOMENTUM_CYCLES": (
                PLANE_DIMS,
                np.array(np.broadcast_to(cycles, shape), dtype=np.int64),
            ),
            "RESIDUAL_NORM_GREW": (
                PLANE_DIMS,
                np.array(np.broadcast_to(grew, shape), dtype=bool),
            ),
            "MOMENTUM_DISABLED": (
                PLANE_DIMS,
                np.array(np.broadcast_to(disabled, shape), dtype=bool),
            ),
        },
        attrs={"cycle_acceleration": "nesterov_momentum"},
    )


def _with_stack_dtype(state, dtype):
    """Cast the Taylor stacks of a state, not its per-plane values."""
    state = state.copy()
    for name in list(state.data_vars):
        if "taylor_term" in state[name].dims:
            state[name] = state[name].astype(dtype)
    return state


def _two_cycles(nterms=2):
    """Cycle-1 inputs and cycle-0 states under the fixed preconditioner."""
    problem = _TaylorProblem(nterms)
    residual = problem.dirty
    increment, after = _richardson(problem, residual)
    inputs = {}
    for acceleration in ("conjugate_gradient", "nesterov_momentum"):
        step, state = _step(acceleration, residual, after, increment, None)
        residual_1 = problem.residual(step)
        increment_1, after_1 = _richardson(problem, residual_1)
        inputs[acceleration] = (residual_1, after_1, increment_1, state)
    return inputs


@pytest.mark.parametrize("nterms", [1, 2])
def test_conjugate_gradient_matches_textbook_pcg(nterms):
    """With B = H and a fixed preconditioner the step is preconditioned CG."""
    problem = _TaylorProblem(nterms)
    theta = np.zeros_like(problem.dirty)
    state = None
    for expected in _textbook_pcg(problem, 8):
        residual = problem.residual(theta)
        increment, after = _richardson(problem, residual)
        step, state = _step("conjugate_gradient", residual, after, increment, state)
        theta = theta + step
        assert np.linalg.norm(theta - expected) <= 1e-10 * np.linalg.norm(expected)


@pytest.mark.parametrize("nterms", [1, 2])
def test_conjugate_gradient_needs_the_textbook_pcg_cycle_count(nterms):
    """Measured 49 (one Taylor term) and 119 (two) imaging cycles for both;
    rounding moves the crossing of the tolerance by one cycle."""
    problem = _TaylorProblem(nterms)
    expected = 1 + next(
        cycle
        for cycle, theta in enumerate(_textbook_pcg(problem, 400))
        if _converged(problem, problem.residual(theta))
    )
    cycles = _cycles_to_tolerance(problem, "conjugate_gradient", _richardson, 400)
    assert abs(cycles - expected) <= 1
    assert _cycles_to_tolerance(problem, "none", _richardson, 5 * expected) is None


def test_accelerations_reduce_shallow_hogbom_cycles():
    """10 iterations per model update; measured 58 (none), 15 (conjugate
    gradient) and 23 (momentum) imaging cycles, asserted with a factor 2."""
    problem = _TaylorProblem(2, psf="coverage", widths=(1.0, 1.3))
    plain = _cycles_to_threshold(problem, "none", max_iter=10)
    conjugate_gradient = _cycles_to_threshold(problem, "conjugate_gradient", 10)
    momentum = _cycles_to_threshold(problem, "nesterov_momentum", 10)
    assert None not in (plain, conjugate_gradient, momentum)
    assert 2 * conjugate_gradient <= plain
    assert 2 * momentum <= plain


@pytest.mark.parametrize(
    "acceleration", ["none", "conjugate_gradient", "nesterov_momentum"]
)
def test_deep_hogbom_cycles_reach_the_threshold(acceleration):
    problem = _TaylorProblem(2, psf="coverage", widths=(1.0, 1.3))
    assert _cycles_to_threshold(problem, acceleration, max_iter=10**6) is not None


@pytest.mark.parametrize("nterms", [1, 2])
def test_conjugate_directions_use_the_true_hessian_under_psf_mismatch(nterms):
    problem = _TaylorProblem(nterms)
    wider = _TaylorProblem(nterms, widths=(0.66, 0.825)).transfer
    theta = np.zeros_like(problem.dirty)
    state = None
    conjugate = 0
    for _ in range(8):
        residual = problem.residual(theta)
        increment, after = _richardson(problem, residual, wider)
        previous = None if state is None else state.SKY_MODEL_DIRECTION.values
        step, state = _step("conjugate_gradient", residual, after, increment, state)
        theta = theta + step
        if previous is None:
            continue
        direction = state.SKY_MODEL_DIRECTION.values
        previous_convolved = problem.hessian(previous)
        for pol in range(2):
            if np.array_equal(direction[:, :, pol], increment[:, :, pol]):
                continue
            conjugate += 1
            first, second = direction[:, :, pol], previous_convolved[:, :, pol]
            assert abs(np.vdot(first, second)) <= 1e-12 * np.linalg.norm(
                first
            ) * np.linalg.norm(second)
    assert conjugate > 0


def test_first_cycle_steps():
    problem = _TaylorProblem(2)
    residual = problem.dirty
    wider = _TaylorProblem(2, widths=(0.66, 0.825)).transfer
    increment, after = _richardson(problem, residual, wider)
    step, state = _step("conjugate_gradient", residual, after, increment, None)
    alpha = _dot(residual, increment) / _dot(increment, residual - after)
    np.testing.assert_allclose(step, _planes(alpha) * increment, rtol=1e-14)
    np.testing.assert_array_equal(state.SKY_MODEL_DIRECTION, increment)
    step, state = _step("nesterov_momentum", residual, after, increment, None, mu=0.3)
    np.testing.assert_array_equal(step, increment)
    np.testing.assert_array_equal(state.SKY_MODEL_MOMENTUM, increment)
    np.testing.assert_array_equal(state.MOMENTUM_CYCLES, 1)


# (r, r_previous, B p_previous) with z = e1, B z = e1 / 2 and p_previous = e2.
_RESTARTS = {
    "negative_numerator_and_denominator": ([1, 1, 0, 0], [2, 0, 0, 0], [0, 1, 0, 0]),
    "negative_denominator_with_descent": ([1, -1, 0, 0], [0, -2, 0, 0], [0, 1, 0, 0]),
    "negative_hestenes_stiefel": ([1, 0, 0, 0], [2, 1, 0, 0], [0, 1, 0, 0]),
    "nonpositive_descent": ([1, -10, 0, 0], [-9, 0, 0, 0], [0, 1, 0, 0]),
    "nonpositive_curvature": ([1, 0, 0, 0], [0, 1, 0, 0], [0, -10, 0, 0]),
}


@pytest.mark.parametrize("case", _RESTARTS)
def test_conjugate_gradient_restarts_along_the_increment(case):
    residual, previous_residual, direction_convolved = _RESTARTS[case]
    increment = _pixels([1, 0, 0, 0])
    after = _pixels(residual) - _pixels([0.5, 0, 0, 0])
    state = _cg_state(
        _pixels([0, 1, 0, 0]), _pixels(direction_convolved), _pixels(previous_residual)
    )
    step, new_state = conjugate_gradient_model_step(
        _pixels(residual), after, increment, cycle_acceleration_state_xds=state
    )
    np.testing.assert_array_equal(step, 2 * increment)
    np.testing.assert_array_equal(new_state.SKY_MODEL_DIRECTION, increment)
    np.testing.assert_array_equal(
        new_state.SKY_MODEL_DIRECTION_CONVOLVED, _pixels([0.5, 0, 0, 0])
    )


def test_conjugate_gradient_uses_a_valid_conjugate_direction():
    state = _cg_state(
        _pixels([0, 1, 0, 0]), _pixels([0, 1, 0, 0]), _pixels([0, 1, 0, 0])
    )
    step, new_state = conjugate_gradient_model_step(
        _pixels([1, 0, 0, 0]),
        _pixels([0.5, 0, 0, 0]),
        _pixels([1, 0, 0, 0]),
        cycle_acceleration_state_xds=state,
    )
    np.testing.assert_allclose(step, _pixels([2 / 3, 2 / 3, 0, 0]), rtol=1e-15)
    np.testing.assert_array_equal(new_state.SKY_MODEL_DIRECTION, _pixels([1, 1, 0, 0]))
    np.testing.assert_array_equal(
        new_state.SKY_MODEL_DIRECTION_CONVOLVED, _pixels([0.5, 1, 0, 0])
    )


@pytest.mark.parametrize(
    "residual,increment,after",
    [
        ([1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]),
        ([-1, 0, 0, 0], [1, 0, 0, 0], [-1.5, 0, 0, 0]),
        ([1, 0, 0, 0], [1, 0, 0, 0], [1.5, 0, 0, 0]),
        ([np.nan, 0, 0, 0], [1, 0, 0, 0], [np.nan, 0, 0, 0]),
    ],
    ids=["zero_increment", "nonpositive_descent", "nonpositive_curvature", "nan"],
)
def test_conjugate_gradient_falls_back_to_the_plain_step(residual, increment, after):
    residual = _pixels([1, 0, 0, 0], residual)
    increment = _pixels([1, 0, 0, 0], increment)
    after = _pixels([0.5, 0, 0, 0], after)
    step, state = conjugate_gradient_model_step(residual, after, increment)
    np.testing.assert_array_equal(step[:, 0], 2 * increment[:, 0])
    np.testing.assert_array_equal(step[:, 1], increment[:, 1])
    np.testing.assert_array_equal(state.SKY_MODEL_DIRECTION, increment)
    np.testing.assert_array_equal(state.SKY_MODEL_DIRECTION_CONVOLVED, residual - after)


def test_momentum_restarts_and_freezes_planes():
    state = _momentum_state(_pixels([-10, 0, 0, 0], [1, 0, 0, 0]))
    increment = _pixels([1, 0, 0, 0], [0, 0, 0, 0])
    step, new_state = nesterov_momentum_model_step(
        _pixels([1, 0, 0, 0], [1, 0, 0, 0]), increment, None, state
    )
    np.testing.assert_array_equal(step, increment)
    np.testing.assert_array_equal(new_state.SKY_MODEL_MOMENTUM, 0.0)
    np.testing.assert_array_equal(new_state.MOMENTUM_CYCLES, 0)


@pytest.mark.parametrize(
    "mu,coefficients",
    [
        # (plane 0, plane 1) per call; plane 0 is a plain-step plane in call 4.
        (
            0.5,
            [(0, 0), (1 / 4, 1 / 4), (2 / 5, 2 / 5), (None, 1 / 2)]
            + [(0, 1 / 2), (1 / 4, 1 / 2), (2 / 5, 1 / 2)],
        ),
        (
            0.3,
            [(0, 0), (1 / 4, 1 / 4), (0.3, 0.3), (None, 0.3)]
            + [(0, 0.3), (1 / 4, 0.3), (0.3, 0.3)],
        ),
    ],
    ids=["mu_0.5", "mu_0.3"],
)
def test_momentum_coefficient_ramps_up_from_every_start_to_the_cap(mu, coefficients):
    """``mu_j = min(mu, (j - 1) / (j + 2))`` with ``j`` counted per plane
    since its last start or restart, so the first step after either is the
    plain ``z``. ``r = z = e1`` in every call, so no restart test fires."""
    residual = _pixels([1, 0, 0, 0], [1, 0, 0, 0])
    state = None
    momentum = [0.0, 0.0]
    cycles = [0, 0]
    for call, call_coefficients in enumerate(coefficients):
        plain = xr.DataArray([[call == 3, False]], dims=PLANE_DIMS)
        step, state = nesterov_momentum_model_step(
            residual, residual, plain, state, momentum_coefficient=mu
        )
        for plane, coefficient in enumerate(call_coefficients):
            if coefficient is None:
                momentum[plane], cycles[plane], expected_step = 0.0, 0, 1.0
            else:
                momentum[plane] = coefficient * momentum[plane] + 1.0
                cycles[plane] += 1
                expected_step = 1.0 + coefficient * momentum[plane]
            if coefficient in (None, 0):
                np.testing.assert_array_equal(step[0, plane], residual[0, plane])
            np.testing.assert_allclose(
                step[0, plane], [[expected_step, 0], [0, 0]], rtol=1e-15
            )
            np.testing.assert_allclose(
                state.SKY_MODEL_MOMENTUM[0, plane],
                [[momentum[plane], 0], [0, 0]],
                rtol=1e-15,
            )
        np.testing.assert_array_equal(state.MOMENTUM_CYCLES, [cycles])


def test_momentum_is_updated_with_the_current_coefficient():
    """``v = mu_j v_previous + z``, so the first increment ``e1`` enters the
    second step with the weight ``mu_2**2 = 1/16``: the step is
    ``1.25 z_2 + z_1 / 16``, where the look-ahead form
    ``v = mu_(j-1) v_previous + z`` would take ``1.25 z_2``."""
    first, second = _pixels([1, 0, 0, 0]), _pixels([0, 1, 0, 0])
    step, state = nesterov_momentum_model_step(first, first)
    np.testing.assert_array_equal(step, first)
    step, state = nesterov_momentum_model_step(second, second, None, state)
    np.testing.assert_array_equal(step, _pixels([1 / 16, 1.25, 0, 0]))
    np.testing.assert_array_equal(state.SKY_MODEL_MOMENTUM, _pixels([1 / 4, 1, 0, 0]))
    np.testing.assert_array_equal(state.MOMENTUM_CYCLES, [[2]])


@pytest.mark.parametrize(
    "mu,residual,increment,momentum,expected_step,expected_momentum",
    [
        (
            0.5,
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, -4, 0, 0],
            [1.5, 0.5, 0, 0],
            [1, -1, 0, 0],
        ),
        (
            0.5,
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [-1.5, 1, 0, 0],
            [1.125, 1.75, 0, 0],
            [0.25, 1.5, 0, 0],
        ),
        (0.5, [1, 0, 0, 0], [1, 0, 0, 0], [-2, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]),
        # <r, z> = 1 and <r, v> = -2: 1 - 0.3 * 2 > 0, but 1 - 0.7 * 2 < 0.
        (
            0.3,
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [-2, 0, 0, 0],
            [1.12, 1.3, 0, 0],
            [0.4, 1, 0, 0],
        ),
        # <r, z> = -1 and <r, v> = 2: -1 + 0.3 * 2 < 0, but -1 + 0.7 * 2 > 0.
        (0.3, [1, 0, 0, 0], [-1, 1, 0, 0], [2, 0, 0, 0], [-1, 1, 0, 0], [0, 0, 0, 0]),
    ],
    ids=[
        "residual_not_increment",
        "coefficient_weighted",
        "zero_descent",
        "mu_0.3_descent",
        "mu_0.3_restart",
    ],
)
def test_momentum_restart_tests_the_residual_against_the_weighted_step(
    mu, residual, increment, momentum, expected_step, expected_momentum
):
    """The restart test is ``<r, mu v_previous + z> <= 0``; at ``mu = 0.3``
    the weight ``1 - mu`` would reverse both outcomes."""
    step, new_state = nesterov_momentum_model_step(
        _pixels(residual),
        _pixels(increment),
        None,
        _momentum_state(_pixels(momentum)),
        momentum_coefficient=mu,
    )
    np.testing.assert_allclose(step, _pixels(expected_step), rtol=1e-15, atol=0)
    np.testing.assert_allclose(
        new_state.SKY_MODEL_MOMENTUM, _pixels(expected_momentum), rtol=1e-15, atol=0
    )


@pytest.mark.parametrize(
    "increment,momentum,expected_step",
    [
        # <r, z> = 1 and <r, v> = -3: 1 - 3 / 4 > 0, but 1 - 3 mu < 0.
        ([1, 1, 0, 0], [-3, 0, 0, 0], [1.0625, 1.25, 0, 0]),
        # <r, z> = -1 and <r, v> = 5: -1 + 5 / 4 > 0, but -1 + 5 mu_1 < 0.
        ([-1, 1, 0, 0], [5, 0, 0, 0], [-0.9375, 1.25, 0, 0]),
    ],
    ids=["not_the_cap", "not_the_previous_coefficient"],
)
def test_momentum_restart_weights_v_with_the_ramp_coefficient(
    increment, momentum, expected_step
):
    """In the second imaging cycle after a start, ``mu_j = 1/4`` lies below
    the cap ``mu = 0.5`` and above the first step's ``mu_(j-1) = 0``. Both
    cases keep the momentum, ``v = [1/4, 1]``; the restart test weighted
    with the cap would restart the first, with ``mu_(j-1)`` the second."""
    step, new_state = nesterov_momentum_model_step(
        _pixels([1, 0, 0, 0]),
        _pixels(increment),
        None,
        _momentum_state(_pixels(momentum), cycles=1),
    )
    np.testing.assert_array_equal(step, _pixels(expected_step))
    np.testing.assert_array_equal(
        new_state.SKY_MODEL_MOMENTUM, _pixels([0.25, 1, 0, 0])
    )
    np.testing.assert_array_equal(new_state.MOMENTUM_CYCLES, [[2]])


def test_momentum_restarts_on_the_first_residual_norm_growth_and_stops_on_the_second():
    """``r = z = e1`` in every call, so ``||r|| = 1``; before each call the
    stored norm of each plane is set to 0.5 (``||r||`` grows) or 2 (falls).
    Plane 0 starts with momentum at the cap, plane 1 without momentum. Growth
    on a plane without momentum is not counted, and a restart starts the
    ramp again with the plain step."""
    residual = _pixels([1, 0, 0, 0], [1, 0, 0, 0])
    state = _momentum_state(_pixels([1, 0, 0, 0], [0, 0, 0, 0]), cycles=[[10, 0]])
    expected = [
        # previous norm, step, momentum (e1 components of planes 0 and 1),
        # MOMENTUM_CYCLES, RESIDUAL_NORM_GREW, MOMENTUM_DISABLED
        ([0.5, 0.5], [1, 1], [0, 1], [0, 1], [True, False], [False, False]),
        ([0.5, 2.0], [1, 1.3125], [1, 1.25], [1, 2], [True, False], [False, False]),
        ([2.0, 0.5], [1.3125, 1], [1.25, 0], [2, 0], [True, True], [False, False]),
        ([0.5, 0.5], [1, 1], [0, 1], [0, 1], [True, True], [True, False]),
        ([2.0, 2.0], [1, 1.3125], [0, 1.25], [0, 2], [True, True], [True, False]),
    ]
    for previous_norm, step_e1, momentum_e1, cycles, grew, disabled in expected:
        state["SKY_RESIDUAL_PREVIOUS_NORM"][:] = [previous_norm]
        step, state = nesterov_momentum_model_step(residual, residual, None, state)
        np.testing.assert_array_equal(
            step, _pixels([step_e1[0], 0, 0, 0], [step_e1[1], 0, 0, 0])
        )
        np.testing.assert_array_equal(
            state.SKY_MODEL_MOMENTUM,
            _pixels([momentum_e1[0], 0, 0, 0], [momentum_e1[1], 0, 0, 0]),
        )
        np.testing.assert_array_equal(state.SKY_RESIDUAL_PREVIOUS_NORM, [[1.0, 1.0]])
        np.testing.assert_array_equal(state.MOMENTUM_CYCLES, [cycles])
        np.testing.assert_array_equal(state.RESIDUAL_NORM_GREW, [grew])
        np.testing.assert_array_equal(state.MOMENTUM_DISABLED, [disabled])


def test_momentum_reads_the_incoming_flags_of_each_plane():
    """All three planes carry momentum at the cap and ``r = z = e1``. Plane 0
    has grown before and grows again, so it is switched off; plane 1 was
    switched off and stays off although its norm falls; plane 2 grows for
    the first time and only restarts."""
    residual = _pixels([1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0])
    state = _momentum_state(
        residual,
        previous_norm=[[0.5, 2.0, 0.5]],
        grew=[[True, False, False]],
        disabled=[[False, True, False]],
    )
    step, new_state = nesterov_momentum_model_step(residual, residual, None, state)
    np.testing.assert_array_equal(step, residual)
    np.testing.assert_array_equal(new_state.SKY_MODEL_MOMENTUM, 0.0)
    np.testing.assert_array_equal(new_state.MOMENTUM_CYCLES, 0)
    np.testing.assert_array_equal(new_state.RESIDUAL_NORM_GREW, [[True, False, True]])
    np.testing.assert_array_equal(new_state.MOMENTUM_DISABLED, [[True, True, False]])


@pytest.mark.parametrize(
    "mu,cycles,counted",
    [(0.5, 1, False), (0.5, 2, True), (0.0, 10, False)],
    ids=["after_the_plain_first_step", "after_a_step_with_momentum", "zero_mu"],
)
def test_momentum_counts_a_residual_norm_growth_only_after_a_step_with_momentum(
    mu, cycles, counted
):
    """``r = z = e1`` and the stored norm 0.5 is below ``||r|| = 1``; the
    previous step had ``mu_j = min(mu, (cycles - 1) / (cycles + 2))``."""
    residual = _pixels([1, 0, 0, 0])
    state = _momentum_state(residual, previous_norm=0.5, cycles=cycles)
    step, new_state = nesterov_momentum_model_step(
        residual, residual, None, state, momentum_coefficient=mu
    )
    np.testing.assert_array_equal(new_state.RESIDUAL_NORM_GREW, [[counted]])
    np.testing.assert_array_equal(new_state.MOMENTUM_DISABLED, [[False]])
    if counted:
        np.testing.assert_array_equal(step, residual)
        np.testing.assert_array_equal(new_state.MOMENTUM_CYCLES, [[0]])
    else:
        coefficient = min(mu, cycles / (cycles + 3))
        np.testing.assert_allclose(
            step, (1 + coefficient * (1 + coefficient)) * residual, rtol=1e-15
        )
        np.testing.assert_array_equal(new_state.MOMENTUM_CYCLES, [[cycles + 1]])


def test_zero_momentum_coefficient_never_switches_the_momentum_off():
    """A plain imaging cycle whose residual norm grows twice: every step is
    ``z`` and neither flag is set, so the driver warns of nothing."""
    state = None
    for peak in [1.0, 1.2, 0.9, 1.1, 0.8]:
        residual = _pixels([peak, 0, 0, 0])
        increment = 0.5 * residual
        step, state = nesterov_momentum_model_step(
            residual, increment, None, state, momentum_coefficient=0.0
        )
        np.testing.assert_array_equal(step, increment)
        np.testing.assert_array_equal(state.RESIDUAL_NORM_GREW, [[False]])
        np.testing.assert_array_equal(state.MOMENTUM_DISABLED, [[False]])


def test_momentum_converges_when_the_model_update_over_corrects():
    """With ``H = 2 I`` the model update removes 0.9 times the Taylor-term-0
    error and 1.7 times the Taylor-term-1 error. Measured: none 39, momentum
    44 imaging cycles; without the residual-norm test the momentum raised
    ``||r||`` from 14 to 1.4e5 in 39 imaging cycles."""
    rng = np.random.default_rng(1)
    dirty = 2.0 * rng.standard_normal((1, 2, 2, 4, 4))
    problem = SimpleNamespace(dirty=dirty, residual=lambda theta: dirty - 2.0 * theta)
    gains = np.array([0.9, 1.7])[None, :, None, None, None]

    def over_correcting(problem, residual):
        increment = gains * residual / 2.0
        return increment, residual - 2.0 * increment

    plain = _cycles_to_tolerance(problem, "none", over_correcting, 200)
    momentum = _cycles_to_tolerance(problem, "nesterov_momentum", over_correcting, 200)
    assert plain is not None
    assert momentum is not None
    assert momentum <= plain + 10


def test_momentum_converges_when_the_weak_taylor_term_is_over_corrected():
    """``H = diag(2, 0.02)`` over the Taylor terms. The model update solves
    Taylor term 0 exactly, so any extrapolated step overshoots it, and
    removes 1.7 times the error of the weakly constrained Taylor term 1,
    which dominates ``||r||`` only once Taylor term 0 has converged. Measured:
    none 27, momentum 31 imaging cycles (35 with a constant coefficient,
    whose first step ``1.5 z`` overshoots Taylor term 0 by half), with the
    first residual-norm growth in the third; without the residual-norm test
    the momentum diverges."""
    rng = np.random.default_rng(1)
    hessian = np.array([2.0, 0.02])[None, :, None, None, None]
    dirty = hessian * rng.standard_normal((1, 2, 2, 4, 4))
    problem = SimpleNamespace(
        dirty=dirty, residual=lambda theta: dirty - hessian * theta
    )
    gains = np.array([1.0, 1.7])[None, :, None, None, None]

    def over_correcting(problem, residual):
        increment = gains * residual / hessian
        return increment, residual - hessian * increment

    plain = _cycles_to_tolerance(problem, "none", over_correcting, 200)
    momentum = _cycles_to_tolerance(problem, "nesterov_momentum", over_correcting, 200)
    assert plain is not None
    assert momentum is not None
    assert momentum <= plain + 8


@pytest.mark.parametrize(
    "name",
    [
        "SKY_MODEL_MOMENTUM",
        "SKY_RESIDUAL_PREVIOUS_NORM",
        "MOMENTUM_CYCLES",
        "RESIDUAL_NORM_GREW",
        "MOMENTUM_DISABLED",
    ],
)
def test_momentum_rejects_an_incomplete_state(name):
    increment = _pixels([1, 0, 0, 0])
    state = _momentum_state(increment).drop_vars(name)
    with pytest.raises(ValueError, match=name):
        nesterov_momentum_model_step(increment, increment, None, state)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zero_momentum_coefficient_gives_the_plain_step(dtype):
    rng = np.random.default_rng(3)
    residual, increment, momentum = (
        _pixels(*rng.standard_normal((2, 4)), dtype=dtype) for _ in range(3)
    )
    state = _momentum_state(momentum)
    step, _ = nesterov_momentum_model_step(
        residual, increment, None, state, momentum_coefficient=0.0
    )
    assert step.dtype == dtype
    np.testing.assert_array_equal(step, increment)


def test_momentum_reduces_cycles_with_a_shallow_preconditioner():
    """``z = 0.1 r / lambda_max``; measured 1032 (none) and 510 (momentum)
    imaging cycles, asserted with a factor 1.5."""
    problem = _TaylorProblem(1, widths=(0.4, 0.5))

    def shallow(problem, residual):
        increment = 0.1 * residual / problem.largest_eigenvalue
        return increment, residual - problem.hessian(increment)

    plain = _cycles_to_tolerance(problem, "none", shallow, 3000)
    momentum = _cycles_to_tolerance(problem, "nesterov_momentum", shallow, 3000)
    assert None not in (plain, momentum)
    assert 1.5 * momentum <= plain


@pytest.mark.parametrize("nterms", [1, 2])
def test_exact_inverse_deconvolver_converges_in_one_cycle(nterms):
    problem = _TaylorProblem(nterms)
    increment, after = _exact_inverse(problem, problem.dirty)
    step, _ = _step("conjugate_gradient", problem.dirty, after, increment, None)
    np.testing.assert_allclose(step, increment, rtol=1e-12, atol=1e-12)
    assert _cycles_to_tolerance(problem, "conjugate_gradient", _exact_inverse, 5) == 1


@pytest.mark.parametrize("acceleration", ["conjugate_gradient", "nesterov_momentum"])
@pytest.mark.parametrize(
    "variant", ["regular", "zero_increment", "nan_residual", "nonfinite_state"]
)
def test_planes_are_independent(acceleration, variant):
    residual, after, increment, state = _two_cycles()[acceleration]
    baseline = _step(acceleration, residual, after, increment, state)
    # Copies in the same memory order keep the summation order of plane 0.
    residual, after, increment = (
        np.copy(value, order="K") for value in (residual, after, increment)
    )
    state = state.copy(
        data={name: np.copy(state[name].values, order="K") for name in state}
    )
    if variant == "regular":
        residual[:, :, 1] *= 3.0
        increment[:, :, 1] *= -2.0
    elif variant == "zero_increment":
        increment[:, :, 1] = 0.0
    elif variant == "nan_residual":
        residual[:, :, 1] = np.nan
    else:
        for variable in state.data_vars.values():
            if "taylor_term" in variable.dims:
                variable.values[:, :, 1, :2] = np.nan
                variable.values[:, :, 1, 2:] = np.inf
            elif variable.dtype == np.float64:
                variable.values[:, 1] = np.nan
    step, new_state = _step(acceleration, residual, after, increment, state)
    np.testing.assert_array_equal(step[:, :, 0], baseline[0][:, :, 0])
    for name, variable in new_state.data_vars.items():
        np.testing.assert_array_equal(
            variable.isel(polarization=0), baseline[1][name].isel(polarization=0)
        )
    if acceleration == "conjugate_gradient":
        assert not np.array_equal(
            baseline[1].SKY_MODEL_DIRECTION[:, :, 0], increment[:, :, 0]
        )
    else:
        assert not np.array_equal(baseline[0][:, :, 0], increment[:, :, 0])

    # The faulted plane restarts along its own increment or takes it.
    if variant == "regular":
        return
    if acceleration == "conjugate_gradient" and variant == "nonfinite_state":
        convolved = residual - after
        alpha = _dot(residual, increment) / _dot(increment, convolved)
        np.testing.assert_allclose(
            step[:, :, 1], (_planes(alpha) * increment)[:, :, 1], rtol=1e-14
        )
        np.testing.assert_array_equal(
            new_state.SKY_MODEL_DIRECTION[:, :, 1], increment[:, :, 1]
        )
        np.testing.assert_array_equal(
            new_state.SKY_MODEL_DIRECTION_CONVOLVED[:, :, 1], convolved[:, :, 1]
        )
        assert np.isfinite(step).all()
    else:
        np.testing.assert_array_equal(step[:, :, 1], increment[:, :, 1])
    if acceleration == "nesterov_momentum":
        np.testing.assert_array_equal(new_state.SKY_MODEL_MOMENTUM[:, :, 1], 0.0)


@pytest.mark.parametrize("acceleration", ["conjugate_gradient", "nesterov_momentum"])
def test_inputs_and_state_are_not_modified(acceleration):
    residual, after, increment, state = _two_cycles()[acceleration]
    arguments = [_da(residual), _da(after), _da(increment)]
    if acceleration == "nesterov_momentum":
        del arguments[1]
    copies = [argument.copy(deep=True) for argument in arguments]
    state_copy = state.copy(deep=True)
    step_function = (
        conjugate_gradient_model_step
        if acceleration == "conjugate_gradient"
        else nesterov_momentum_model_step
    )
    step_function(*arguments, cycle_acceleration_state_xds=state)
    for argument, copy in zip(arguments, copies, strict=True):
        xr.testing.assert_identical(argument, copy)
    xr.testing.assert_identical(state, state_copy)


@pytest.mark.parametrize("acceleration", ["conjugate_gradient", "nesterov_momentum"])
def test_single_precision_and_dimension_order_are_preserved(acceleration):
    residual, after, increment, state = _two_cycles()[acceleration]
    expected, _ = _step(acceleration, residual, after, increment, state)

    def arguments(dtype, order):
        values = [residual, after, increment]
        if acceleration == "nesterov_momentum":
            del values[1]
        return [_da(value.astype(dtype)).transpose(*order) for value in values]

    step_function = (
        conjugate_gradient_model_step
        if acceleration == "conjugate_gradient"
        else nesterov_momentum_model_step
    )
    single = step_function(
        *arguments(np.float32, DIMS),
        cycle_acceleration_state_xds=_with_stack_dtype(state, np.float32),
    )
    assert single[0].dtype == np.float32
    assert all(
        variable.dtype == np.float32
        for variable in single[1].data_vars.values()
        if "taylor_term" in variable.dims
    )
    np.testing.assert_allclose(single[0], expected, rtol=1e-4, atol=1e-6)

    order = ("m", "polarization", "l", "time", "taylor_term")
    permuted = step_function(
        *arguments(np.float64, order),
        cycle_acceleration_state_xds=state.transpose("l", *PLANE_DIMS, "m", ...),
    )
    assert permuted[0].dims == order
    np.testing.assert_allclose(permuted[0].transpose(*DIMS), expected, rtol=1e-12)


@pytest.mark.parametrize("acceleration", ["conjugate_gradient", "nesterov_momentum"])
def test_cube_layout_planes_match_single_term_planes(acceleration):
    """Without ``taylor_term`` every (time, frequency, polarization) is a plane."""
    channels = [_two_cycles(1)[acceleration], _two_cycles(1)[acceleration]]
    channels[1] = tuple(2.0 * value for value in channels[1][:3]) + (
        channels[1][3].copy(deep=True),
    )
    cube_dims = ("time", "frequency", "polarization", "l", "m")

    def cube(index):
        return xr.DataArray(
            np.concatenate([channel[index] for channel in channels], axis=1),
            dims=cube_dims,
        )

    state = xr.concat(
        [
            channel[3].isel(taylor_term=0).expand_dims("frequency", axis=1)
            for channel in channels
        ],
        dim="frequency",
    )
    arguments = [cube(0), cube(1), cube(2)]
    if acceleration == "nesterov_momentum":
        del arguments[1]
    step_function = (
        conjugate_gradient_model_step
        if acceleration == "conjugate_gradient"
        else nesterov_momentum_model_step
    )
    step, _ = step_function(*arguments, cycle_acceleration_state_xds=state)
    for frequency, channel in enumerate(channels):
        expected, _ = _step(acceleration, *channel)
        np.testing.assert_allclose(
            step.isel(frequency=frequency), expected[:, 0], rtol=1e-12
        )


def _invalid_inputs(fault):
    residual = _pixels([1, 0, 0, 0], [1, 0, 0, 0])
    after = residual * 0.5
    increment = residual * 0.1
    plain = None
    state = None
    if fault == "dimensions":
        after = after.rename(l="x")
    elif fault == "sizes":
        after = after.isel(l=[0])
    elif fault == "spatial":
        residual, after, increment = (
            value.rename(l="x", m="y") for value in (residual, after, increment)
        )
    elif fault == "complex":
        residual = residual.astype(np.complex128)
    elif fault == "integer":
        increment = increment.astype(np.int64)
    elif fault == "dtype":
        after = after.astype(np.float32)
    elif fault == "plain_dims":
        plain = xr.DataArray([True, False], dims="polarization")
    elif fault == "plain_dtype":
        plain = xr.DataArray([[1, 0]], dims=PLANE_DIMS)
    elif fault == "foreign_state":
        state = xr.Dataset(
            {"SKY_MODEL_MOMENTUM": increment},
            attrs={"cycle_acceleration": "nesterov_momentum"},
        )
    elif fault == "state_shape":
        state = _cg_state(*(3 * [increment.isel(m=[0])]))
    return residual, after, increment, plain, state


@pytest.mark.parametrize(
    "fault,error,message",
    [
        ("dimensions", ValueError, "sizes"),
        ("sizes", ValueError, "sizes"),
        ("spatial", ValueError, "'l' and 'm'"),
        ("complex", TypeError, "real floating point"),
        ("integer", TypeError, "real floating point"),
        ("dtype", TypeError, "dtype"),
        ("plain_dims", ValueError, "plain_step_planes"),
        ("plain_dtype", ValueError, "plain_step_planes"),
        ("foreign_state", ValueError, "belongs to 'nesterov_momentum'"),
        ("state_shape", ValueError, "does not match"),
    ],
)
def test_conjugate_gradient_rejects_invalid_inputs(fault, error, message):
    residual, after, increment, plain, state = _invalid_inputs(fault)
    with pytest.raises(error, match=message):
        conjugate_gradient_model_step(residual, after, increment, plain, state)


def test_momentum_rejects_a_conjugate_gradient_state():
    increment = _pixels([1, 0, 0, 0])
    with pytest.raises(ValueError, match="belongs to 'conjugate_gradient'"):
        nesterov_momentum_model_step(
            increment, increment, None, _cg_state(increment, increment, increment)
        )


def test_conjugate_gradient_plain_step_planes_restart_only_their_plane():
    problem = _TaylorProblem(2)

    def over_relaxed(problem, residual):
        increment = 1.9 * residual / problem.largest_eigenvalue
        return increment, residual - problem.hessian(increment)

    increment, after = over_relaxed(problem, problem.dirty)
    theta, state = _step("conjugate_gradient", problem.dirty, after, increment, None)
    residual = problem.residual(theta)
    increment, after = over_relaxed(problem, residual)
    free, _ = _step("conjugate_gradient", residual, after, increment, state)
    plain = np.array([[False, True]])
    step, state = _step("conjugate_gradient", residual, after, increment, state, plain)
    np.testing.assert_array_equal(step[:, :, 0], free[:, :, 0])
    np.testing.assert_array_equal(step[:, :, 1], increment[:, :, 1])
    np.testing.assert_array_equal(
        state.SKY_MODEL_DIRECTION[:, :, 1], increment[:, :, 1]
    )
    np.testing.assert_array_equal(
        state.SKY_MODEL_DIRECTION_CONVOLVED[:, :, 1], (residual - after)[:, :, 1]
    )
    np.testing.assert_array_equal(state.SKY_RESIDUAL_PREVIOUS, residual)

    # The next direction is H-conjugate to the plain step z and the step is
    # the exact line search along it.
    theta = theta + step
    residual_next = problem.residual(theta)
    increment_next, after_next = over_relaxed(problem, residual_next)
    step_next, _ = _step(
        "conjugate_gradient", residual_next, after_next, increment_next, state
    )
    convolved = problem.hessian(increment)
    beta = -_dot(increment_next, convolved) / _dot(increment, convolved)
    assert beta[0, 1] > 0
    direction = increment_next + _planes(beta) * increment
    alpha = _dot(residual_next, direction) / _dot(direction, problem.hessian(direction))
    np.testing.assert_allclose(
        step_next[:, :, 1], (_planes(alpha) * direction)[:, :, 1], rtol=1e-10
    )


def test_momentum_plain_step_planes_reset_only_their_plane():
    residual, after, increment, state = _two_cycles()["nesterov_momentum"]
    free, _ = _step("nesterov_momentum", residual, after, increment, state)
    plain = np.array([[False, True]])
    step, new_state = _step(
        "nesterov_momentum", residual, after, increment, state, plain
    )
    np.testing.assert_array_equal(step[:, :, 0], free[:, :, 0])
    np.testing.assert_array_equal(step[:, :, 1], increment[:, :, 1])
    np.testing.assert_array_equal(new_state.SKY_MODEL_MOMENTUM[:, :, 1], 0.0)
