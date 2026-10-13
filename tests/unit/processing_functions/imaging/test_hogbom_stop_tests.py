"""
Stop tests of the Hogbom kernels: threshold, iteration budget and divergence.

Every plane is tested at every iteration inside the C++ kernel, identically
in ``clean_cube``, ``clean_cube_many_threads`` and the single-plane ``clean``:

- the plane stops when its peak is AT or below its threshold, so an all-zero
  plane does no iterations;
- the divergence test stops a plane when the RMS of its residual (over the
  clean box inside the mask) has been above ``divergence_rms_factor`` times
  the lowest RMS it reached for ``max_iter_divergence`` consecutive
  iterations (soft limit), when its peak exceeds ``divergence_peak_factor``
  times its starting peak (hard limit), or when its peak is not finite.

The diverging cases use a "chain" PSF: a unit main lobe with one sidelobe of
``-s`` one pixel to the right. Cleaning a single point source with loop gain
``g`` then moves the peak one pixel to the right at every iteration and
multiplies it by exactly ``r = g * s``; every pixel the peak has left keeps
``(1 - g)`` times the value it had. After ``k`` iterations the peak is
``r**k`` and the sum of the squares of the residual is::

    (1 - g)**2 * (1 + r**2 + ... + r**(2 * (k - 1))) + r**(2 * k)

so the iteration at which each limit fires is known in closed form. A plain
NumPy clean with the same rule (``_reference_clean``) gives the expected stop
for inputs without a closed form.
"""

import math

import numpy as np
import pytest

try:
    from astroviper.processing_functions.imaging.deconvolvers import hogbom

    HOGBOM_AVAILABLE = True
except ImportError:  # pragma: no cover
    HOGBOM_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
)

NY, NX = 8, 64
X0 = 2  # column of the point source; the chain runs to the right of it
STOP_NONE, STOP_MAX_ITER, STOP_THRESHOLD, STOP_DIVERGED = 0, 1, 2, 4

CUBE_KERNELS = ["clean_cube", "clean_cube_many_threads"]


def _chain_psf(sidelobe, dtype=np.float64):
    psf = np.zeros((NY, NX), dtype=dtype)
    psf[NY // 2, NX // 2] = 1.0
    psf[NY // 2, NX // 2 + 1] = -sidelobe
    return psf


def _point_source(dtype=np.float64):
    residual = np.zeros((NY, NX), dtype=dtype)
    residual[NY // 2, X0] = 1.0
    return residual


def _cube(plane):
    return np.ascontiguousarray(plane[None, None, None])


def _run_cube(
    kernel,
    residual,
    psf,
    max_iter,
    gain,
    threads=1,
    threshold=0.0,
    mask=None,
    **divergence,
):
    residual = residual.copy()
    model = np.zeros_like(residual)
    if mask is not None:
        divergence["mask_cube"] = np.ascontiguousarray(mask)
    result = getattr(hogbom, kernel)(
        residual_cube=residual,
        psf_cube=psf,
        model_cube=model,
        max_iter_remaining=np.full(residual.shape[:3], max_iter, dtype=np.int32),
        gain=gain,
        threshold=np.full(residual.shape[:3], threshold, dtype=residual.dtype),
        processing_function_threads=threads,
        **divergence,
    )
    return result, residual, model


def _chain_sum_squares(ratio, gain, k):
    """Sum of the squares of the chain residual after ``k`` iterations."""
    return (1 - gain) ** 2 * sum(ratio ** (2 * j) for j in range(k)) + ratio ** (2 * k)


def _expected_chain_stop(ratio, gain, rms_factor, peak_factor, n, max_iter):
    """Iterations performed by the chain clean under the divergence rule."""
    lowest = _chain_sum_squares(ratio, gain, 0)
    above = 0
    for k in range(max_iter):
        peak = ratio**k
        sum_squares = _chain_sum_squares(ratio, gain, k)
        if n >= 0:
            if peak > peak_factor:  # the starting peak is 1
                return k, STOP_DIVERGED
            if sum_squares > rms_factor**2 * lowest:
                above += 1
                if above >= n:
                    return k, STOP_DIVERGED
            else:
                above = 0
        lowest = min(lowest, sum_squares)
    return max_iter, STOP_MAX_ITER


def _reference_clean(
    residual, psf, gain, max_iter, n, rms_factor, peak_factor, mask=None, threshold=0.0
):
    """Hogbom clean of one plane in NumPy with the stop tests of the kernel.

    Returns the iterations performed, the stop code and the residual.
    """
    residual = residual.astype(np.float64).copy()
    ny, nx = residual.shape
    searched = np.ones_like(residual, dtype=bool) if mask is None else mask
    lowest = start = None
    above = 0
    for k in range(max_iter):
        magnitude = np.where(searched, np.abs(residual), 0.0)
        py, px = np.unravel_index(np.argmax(magnitude), magnitude.shape)
        peak = magnitude[py, px]
        sum_squares = float(np.sum(residual[searched] ** 2))
        if peak <= threshold:
            return k, STOP_THRESHOLD, residual
        if lowest is None:
            lowest, start = sum_squares, peak
        if n >= 0:
            if peak > peak_factor * start:
                return k, STOP_DIVERGED, residual
            if sum_squares > rms_factor**2 * lowest:
                above += 1
                if above >= n:
                    return k, STOP_DIVERGED, residual
            else:
                above = 0
        lowest = min(lowest, sum_squares)
        value = gain * residual[py, px]
        y1, y2 = max(0, py - ny // 2), min(ny - 1, py + ny // 2 - 1)
        x1, x2 = max(0, px - nx // 2), min(nx - 1, px + nx // 2 - 1)
        residual[y1 : y2 + 1, x1 : x2 + 1] -= (
            value
            * psf[
                ny // 2 + y1 - py : ny // 2 + y2 - py + 1,
                nx // 2 + x1 - px : nx // 2 + x2 - px + 1,
            ]
        )
    return max_iter, STOP_MAX_ITER, residual


def _ringed_psf(n=64, sidelobe=0.25):
    """A Gaussian beam with sidelobe rings of the given level."""
    y, x = np.mgrid[:n, :n]
    radius = np.hypot(y - n // 2, x - n // 2)
    psf = np.exp(-0.5 * (radius / 2.0) ** 2) + sidelobe * np.cos(radius / 1.5) * np.exp(
        -radius / 12.0
    ) * (radius > 4)
    return psf / psf[n // 2, n // 2]


def _field_of_sources(psf, n_sources=6, seed=3):
    rng = np.random.default_rng(seed)
    n = psf.shape[0]
    residual = np.zeros((n, n))
    for _ in range(n_sources):
        iy, ix = rng.integers(n // 4, 3 * n // 4, size=2)
        amplitude = rng.uniform(0.3, 1.0)
        residual += amplitude * np.roll(psf, (iy - n // 2, ix - n // 2), axis=(0, 1))
    return residual


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("kernel", CUBE_KERNELS)
class TestDivergenceTest:
    GAIN = 0.5
    RATIO = 1.02  # gain * sidelobe of the chain PSF below
    SIDELOBE = 2.04
    RMS_FACTOR, PEAK_FACTOR = 1.05, 1.5  # 1 + gain / 10 and 1 + gain

    def _run(self, kernel, threads, max_iter=40, sidelobe=None, **divergence):
        return _run_cube(
            kernel,
            _cube(_point_source()),
            _cube(_chain_psf(self.SIDELOBE if sidelobe is None else sidelobe)),
            max_iter=max_iter,
            gain=self.GAIN,
            threads=threads,
            **divergence,
        )

    def test_soft_limit_stops_at_the_first_rise_of_the_rms(self, kernel, threads):
        # After one iteration the sum of squares is 0.25 + 1.02**2 = 1.2904, an
        # RMS of 1.136 times the lowest one: above the soft limit of 1.05.
        result, _, _ = self._run(
            kernel,
            threads,
            max_iter_divergence=1,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert _expected_chain_stop(1.02, 0.5, 1.05, 1.5, 1, 40) == (1, STOP_DIVERGED)
        assert result["iterations_performed"].item() == 1
        assert result["stop_code"].item() == STOP_DIVERGED
        assert bool(result["diverged"].item()) is True

    def test_soft_limit_waits_for_the_margin(self, kernel, threads):
        # With a factor of 1.5 the sum of squares has to exceed 2.25: 2.2339
        # after four iterations does not, 2.5742 after five does.
        assert _chain_sum_squares(1.02, 0.5, 4) == pytest.approx(2.2339080258)
        assert _chain_sum_squares(1.02, 0.5, 5) == pytest.approx(2.5741579101)
        result, _, _ = self._run(
            kernel,
            threads,
            max_iter_divergence=1,
            divergence_rms_factor=1.5,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"].item() == 5
        assert result["stop_code"].item() == STOP_DIVERGED

    def test_count_asks_for_consecutive_iterations(self, kernel, threads):
        # The RMS of the chain only rises, so a count of 5 stops it four
        # iterations after a count of 1.
        for rms_factor, expected in ((self.RMS_FACTOR, 5), (1.5, 9)):
            result, _, _ = self._run(
                kernel,
                threads,
                max_iter_divergence=5,
                divergence_rms_factor=rms_factor,
                divergence_peak_factor=self.PEAK_FACTOR,
            )
            assert _expected_chain_stop(1.02, 0.5, rms_factor, 1.5, 5, 40) == (
                expected,
                STOP_DIVERGED,
            )
            assert result["iterations_performed"].item() == expected
            assert result["stop_code"].item() == STOP_DIVERGED

    def test_hard_limit_stops_what_the_soft_limit_lets_pass(self, kernel, threads):
        # With a soft limit that cannot fire the hard limit does, as soon as
        # the peak exceeds 1.5 times its starting value: 1.02**21 = 1.516.
        result, _, _ = self._run(
            kernel,
            threads,
            max_iter_divergence=1,
            divergence_rms_factor=1e9,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"].item() == 21
        assert result["stop_code"].item() == STOP_DIVERGED

    def test_hard_limit_stops_a_runaway_at_once(self, kernel, threads):
        # r = 1.6: the second peak already exceeds 1.5 times the first, while
        # a count of 30 keeps the soft limit waiting.
        result, residual, _ = self._run(
            kernel,
            threads,
            sidelobe=3.2,
            max_iter_divergence=30,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"].item() == 1
        assert result["stop_code"].item() == STOP_DIVERGED
        assert np.abs(residual).max() == pytest.approx(1.6)

    def test_minus_one_disables_the_test(self, kernel, threads):
        result, residual, _ = self._run(
            kernel,
            threads,
            max_iter=25,
            max_iter_divergence=-1,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"].item() == 25
        assert result["stop_code"].item() == STOP_MAX_ITER
        assert bool(result["diverged"].item()) is False
        assert np.abs(residual).max() == pytest.approx(1.02**25)

    def test_default_arguments_leave_the_test_off(self, kernel, threads):
        result, _, _ = self._run(kernel, threads, max_iter=25)
        assert result["iterations_performed"].item() == 25
        assert result["stop_code"].item() == STOP_MAX_ITER

    def test_planes_are_tested_independently(self, kernel, threads):
        # Plane 0: healthy (delta PSF). Plane 1: diverging chain.
        residual = np.ascontiguousarray(
            np.stack([_point_source(), _point_source()])[None, None]
        )
        psf = np.ascontiguousarray(
            np.stack([_chain_psf(0.0), _chain_psf(2.04)])[None, None]
        )
        result, _, _ = _run_cube(
            kernel,
            residual,
            psf,
            max_iter=40,
            gain=self.GAIN,
            threads=threads,
            max_iter_divergence=5,
            divergence_rms_factor=1.5,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        np.testing.assert_array_equal(result["iterations_performed"].ravel(), [40, 9])
        np.testing.assert_array_equal(
            result["stop_code"].ravel(), [STOP_MAX_ITER, STOP_DIVERGED]
        )
        np.testing.assert_array_equal(result["diverged"].ravel(), [False, True])

    def test_not_finite_peak_stops_the_plane(self, kernel, threads):
        residual = _point_source()
        residual[1, 5] = np.inf
        result, _, model = _run_cube(
            kernel,
            _cube(residual),
            _cube(_chain_psf(0.0)),
            max_iter=10,
            gain=self.GAIN,
            threads=threads,
        )
        assert result["iterations_performed"].item() == 0
        assert result["stop_code"].item() == STOP_DIVERGED
        assert bool(result["diverged"].item()) is True
        assert not model.any()

    def test_pixel_that_is_not_a_number_does_not_stop_the_plane(self, kernel, threads):
        # The peak search skips such a pixel. The sum of squares is then not a
        # number either: the soft limit is off for the plane, which cleans on.
        residual = _point_source()
        residual[1, 5] = np.nan
        result, cleaned, _ = _run_cube(
            kernel,
            _cube(residual),
            _cube(_chain_psf(0.0)),
            max_iter=10,
            gain=self.GAIN,
            threads=threads,
            max_iter_divergence=1,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"].item() == 10
        assert result["stop_code"].item() == STOP_MAX_ITER
        assert cleaned[0, 0, 0, NY // 2, X0] == pytest.approx(0.5**10)

    def test_rms_is_taken_inside_the_mask(self, kernel, threads):
        # A strong pixel outside the mask, which the diverging chain never
        # reaches, would hide the rise of the RMS if it were counted.
        residual = _point_source()
        residual[0, NX - 1] = 1000.0
        mask = np.ones((NY, NX), dtype=bool)
        mask[0, :] = False
        result, _, _ = _run_cube(
            kernel,
            _cube(residual),
            _cube(_chain_psf(self.SIDELOBE)),
            max_iter=40,
            gain=self.GAIN,
            threads=threads,
            mask=_cube(mask),
            max_iter_divergence=1,
            divergence_rms_factor=1.5,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"].item() == 5
        assert result["stop_code"].item() == STOP_DIVERGED

    def test_healthy_clean_is_untouched(self, kernel, threads):
        # A Gaussian beam with 25 percent sidelobe rings and several sources:
        # the residual jitters but never trips the test, so switching it on
        # must not change a single pixel.
        psf = _ringed_psf()
        residual = _field_of_sources(psf)
        gain = 0.1
        off, res_off, model_off = _run_cube(
            kernel, _cube(residual), _cube(psf), 300, gain, threads
        )
        on, res_on, model_on = _run_cube(
            kernel,
            _cube(residual),
            _cube(psf),
            300,
            gain,
            threads,
            max_iter_divergence=1,
            divergence_rms_factor=1 + gain / 10,
            divergence_peak_factor=1 + gain,
        )
        assert (
            on["iterations_performed"].item()
            == off["iterations_performed"].item()
            == 300
        )
        assert bool(on["diverged"].item()) is False
        np.testing.assert_array_equal(res_on, res_off)
        np.testing.assert_array_equal(model_on, model_off)

    def test_matches_the_reference_clean(self, kernel, threads):
        # A beam with sidelobe rings of 60 percent and a large gain: the clean
        # falls at first and then diverges. The kernel stops where the NumPy
        # clean with the same rule does, at the same residual.
        psf = _ringed_psf(sidelobe=0.6)
        residual = _field_of_sources(psf, seed=11)
        gain = 0.8
        for n in (1, 4):
            expected_iter, expected_code, expected_residual = _reference_clean(
                residual, psf, gain, 400, n, 1 + gain / 10, 1 + gain
            )
            assert expected_code == STOP_DIVERGED
            assert 2 < expected_iter < 400
            result, cleaned, _ = _run_cube(
                kernel,
                _cube(residual),
                _cube(psf),
                400,
                gain,
                threads,
                max_iter_divergence=n,
                divergence_rms_factor=1 + gain / 10,
                divergence_peak_factor=1 + gain,
            )
            assert result["iterations_performed"].item() == expected_iter
            assert result["stop_code"].item() == STOP_DIVERGED
            np.testing.assert_allclose(
                cleaned[0, 0, 0], expected_residual, rtol=0, atol=1e-10
            )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kernels_and_thread_counts_decide_identically(dtype):
    """The sum of squares is added in the same order everywhere, so all cube
    kernels stop at the same iteration with the same residual, bit for bit."""
    psf = _ringed_psf(sidelobe=0.6).astype(dtype)
    residual = _field_of_sources(psf, seed=11).astype(dtype)
    residual_cube = np.ascontiguousarray(
        np.stack([residual, 0.5 * residual[::-1, :]])[None, None]
    )
    psf_cube = np.ascontiguousarray(np.stack([psf, psf])[None, None])
    gain = 0.8
    outcomes = []
    for kernel in CUBE_KERNELS:
        for threads in (1, 2, 5):
            result, cleaned, model = _run_cube(
                kernel,
                residual_cube,
                psf_cube,
                400,
                gain,
                threads,
                max_iter_divergence=1,
                divergence_rms_factor=1 + gain / 10,
                divergence_peak_factor=1 + gain,
            )
            outcomes.append((result, cleaned, model))
    first = outcomes[0]
    assert (first[0]["stop_code"] == STOP_DIVERGED).all()
    for result, cleaned, model in outcomes[1:]:
        np.testing.assert_array_equal(
            result["iterations_performed"], first[0]["iterations_performed"]
        )
        np.testing.assert_array_equal(result["stop_code"], first[0]["stop_code"])
        np.testing.assert_array_equal(cleaned, first[1])
        np.testing.assert_array_equal(model, first[2])


@pytest.mark.parametrize("kernel", CUBE_KERNELS)
def test_single_precision_stops_like_double_precision(kernel):
    # The sum of squares is accumulated in double precision for both.
    for rms_factor, expected in ((1.05, 1), (1.5, 5)):
        result, _, _ = _run_cube(
            kernel,
            _cube(_point_source(np.float32)),
            _cube(_chain_psf(2.04, np.float32)),
            max_iter=40,
            gain=0.5,
            max_iter_divergence=1,
            divergence_rms_factor=rms_factor,
            divergence_peak_factor=1.5,
        )
        assert result["iterations_performed"].item() == expected
        assert result["stop_code"].item() == STOP_DIVERGED


@pytest.mark.parametrize("kernel", CUBE_KERNELS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
class TestThresholdBoundary:
    def test_zero_plane_does_no_iterations(self, kernel, dtype):
        residual = np.zeros((1, 1, 2, NY, NX), dtype=dtype)
        psf = np.ascontiguousarray(
            np.broadcast_to(_chain_psf(0.0, dtype), (1, 1, 2, NY, NX))
        )
        result, _, model = _run_cube(
            kernel,
            residual,
            psf,
            500,
            0.1,
            threshold=0.0,
            max_iter_divergence=1,
            divergence_rms_factor=1.01,
            divergence_peak_factor=1.1,
        )
        np.testing.assert_array_equal(result["iterations_performed"].ravel(), [0, 0])
        np.testing.assert_array_equal(
            result["stop_code"].ravel(), [STOP_THRESHOLD, STOP_THRESHOLD]
        )
        assert result["converged"].all()
        assert not model.any()

    def test_peak_at_the_threshold_stops(self, kernel, dtype):
        # gain 0.5 on a delta PSF halves the peak: 1.0 -> 0.5 -> 0.25. With a
        # threshold of exactly 0.25 the plane stops after two iterations.
        result, residual, _ = _run_cube(
            kernel,
            _cube(_point_source(dtype)),
            _cube(_chain_psf(0.0, dtype)),
            100,
            0.5,
            threshold=0.25,
        )
        assert result["iterations_performed"].item() == 2
        assert result["stop_code"].item() == STOP_THRESHOLD
        assert np.abs(residual).max() == pytest.approx(0.25)

    def test_no_budget_reports_stop_none(self, kernel, dtype):
        result, _, _ = _run_cube(
            kernel,
            _cube(_point_source(dtype)),
            _cube(_chain_psf(0.0, dtype)),
            0,
            0.5,
        )
        assert result["iterations_performed"].item() == 0
        assert result["stop_code"].item() == STOP_NONE


class TestSinglePlaneClean:
    def _run(self, psf, max_iter, **divergence):
        residual = _point_source()
        model = np.zeros_like(residual)
        return hogbom.clean(
            dirty_image=residual,
            psf=psf,
            model=model,
            max_iter_remaining=max_iter,
            gain=0.5,
            threshold=0.0,
            **divergence,
        )

    def test_reports_divergence(self):
        result = self._run(
            _chain_psf(2.04),
            40,
            max_iter_divergence=5,
            divergence_rms_factor=1.5,
            divergence_peak_factor=1.5,
        )
        assert result["iterations_performed"] == 9
        assert result["stop_code"] == STOP_DIVERGED
        assert result["diverged"] is True

    def test_hard_limit(self):
        result = self._run(
            _chain_psf(2.04),
            40,
            max_iter_divergence=1,
            divergence_rms_factor=1e9,
            divergence_peak_factor=1.5,
        )
        assert result["iterations_performed"] == 21
        assert result["stop_code"] == STOP_DIVERGED

    def test_default_is_off(self):
        result = self._run(_chain_psf(2.04), 25)
        assert result["iterations_performed"] == 25
        assert result["stop_code"] == STOP_MAX_ITER
        assert result["diverged"] is False

    def test_matches_the_cube_kernel(self):
        psf = _ringed_psf(sidelobe=0.6)
        residual = _field_of_sources(psf, seed=11)
        gain = 0.8
        single_residual = residual.copy()
        single_model = np.zeros_like(residual)
        single = hogbom.clean(
            dirty_image=single_residual,
            psf=psf,
            model=single_model,
            max_iter_remaining=400,
            gain=gain,
            threshold=0.0,
            max_iter_divergence=1,
            divergence_rms_factor=1 + gain / 10,
            divergence_peak_factor=1 + gain,
        )
        cube, cube_residual, cube_model = _run_cube(
            "clean_cube",
            _cube(residual),
            _cube(psf),
            400,
            gain,
            max_iter_divergence=1,
            divergence_rms_factor=1 + gain / 10,
            divergence_peak_factor=1 + gain,
        )
        assert single["iterations_performed"] == cube["iterations_performed"].item()
        assert single["stop_code"] == cube["stop_code"].item() == STOP_DIVERGED
        np.testing.assert_array_equal(single_residual, cube_residual[0, 0, 0])
        np.testing.assert_array_equal(single_model, cube_model[0, 0, 0])


def test_chain_psf_history_is_known_in_closed_form():
    """The construction the divergence tests rely on: after k iterations the
    peak is (gain * sidelobe) ** k, one pixel further right each time, and
    the sum of the squares of the residual follows the geometric series."""
    for k in (1, 5, 12):
        _, residual, _ = _run_cube(
            "clean_cube", _cube(_point_source()), _cube(_chain_psf(2.04)), k, 0.5
        )
        assert np.abs(residual).max() == pytest.approx(1.02**k, rel=1e-12)
        assert np.abs(residual).argmax() == (NY // 2) * NX + X0 + k
        assert np.sum(residual**2) == pytest.approx(
            _chain_sum_squares(1.02, 0.5, k), rel=1e-12
        )
    assert math.isclose(0.5 * 2.04, 1.02)
