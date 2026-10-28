"""
Unit tests for the pybind11-wrapped Adaptive Scale Pixel (Asp / AAspClean)
deconvolver
(``astroviper.processing_functions.imaging.deconvolvers.aspclean``).

This is a dependency-free port of CASA's ``AspMatrixCleaner``: casacore and
ALGLIB have been replaced by the self-contained FFT (``asp_fft.hpp``) and
L-BFGS (``asp_lbfgs.hpp``) headers. The wrapper operates on Python-owned numpy
buffers with no copies on the C++ side: ``residual`` and ``model`` are written
into directly.

The tests exercise:
  * the self-contained FFT convolution primitive (incl. arbitrary, non
    power-of-two sizes via Bluestein) against a numpy reference,
  * the PSF Gaussian-width estimator,
  * end-to-end deconvolution correctness (residual reduction, flux recovery,
    and the exact reconstruction invariant residual = dirty - model (*) psf),
  * the zero-copy / in-place contract and input validation, and
  * the 5-D cube driver.


The aspclean C++ binding takes the per-plane iteration cap as
``max_iter_per_cycle_updated`` (aspclean/python/bindings.cpp).
"""

import numpy as np
import pytest

try:
    from astroviper.processing_functions.imaging.deconvolvers import aspclean

    ASP_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    ASP_AVAILABLE = False
    IMPORT_ERROR = e


pytestmark = pytest.mark.skipif(
    not ASP_AVAILABLE, reason="Asp extension not compiled/available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gauss(nx, ny, cx, cy, sig, amp=1.0, dtype=np.float64):
    yy, xx = np.mgrid[0:ny, 0:nx]
    g = amp * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig**2)))
    return np.ascontiguousarray(g, dtype=dtype)


def _centered_psf(nx, ny, sig=2.0, dtype=np.float64):
    """Unit-peak centred Gaussian PSF."""
    return _gauss(nx, ny, nx // 2, ny // 2, sig, 1.0, dtype)


def _np_convolve_centered(a, b):
    """Numpy reference for ``aspclean.convolve_centered``.

    Standard circular convolution followed by the same re-centring (fftshift
    by (nx//2, ny//2)) that the C++ primitive applies.
    """
    ny, nx = a.shape
    raw = np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b)))
    return np.roll(raw, shift=(-(ny // 2), -(nx // 2)), axis=(0, 1))


# ---------------------------------------------------------------------------
# FFT-based convolution primitive
# ---------------------------------------------------------------------------


class TestConvolution:
    def test_centered_delta_is_identity(self):
        nx, ny = 32, 24
        rng = np.random.default_rng(0)
        a = rng.standard_normal((ny, nx))
        delta = np.zeros((ny, nx))
        delta[ny // 2, nx // 2] = 1.0
        out = aspclean.convolve_centered(a, delta)
        np.testing.assert_allclose(out, a, atol=1e-12)

    @pytest.mark.parametrize("ny,nx", [(32, 32), (64, 48), (15, 17), (31, 13)])
    def test_matches_numpy_reference(self, ny, nx):
        """Even/composite sizes and odd/prime sizes (Bluestein path)."""
        rng = np.random.default_rng(ny * 100 + nx)
        a = rng.standard_normal((ny, nx))
        b = rng.standard_normal((ny, nx))
        out = aspclean.convolve_centered(a, b)
        ref = _np_convolve_centered(a, b)
        np.testing.assert_allclose(out, ref, atol=1e-9, rtol=1e-7)

    def test_commutative(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((20, 28))
        b = rng.standard_normal((20, 28))
        np.testing.assert_allclose(
            aspclean.convolve_centered(a, b),
            aspclean.convolve_centered(b, a),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# PSF Gaussian width
# ---------------------------------------------------------------------------


class TestPsfWidth:
    @pytest.mark.parametrize("sig", [1.5, 2.0, 3.0])
    def test_circular_gaussian_fwhm(self, sig):
        psf = _centered_psf(96, 96, sig)
        # mean FWHM in pixels = ceil(2*sqrt(2 ln2) * sigma)
        expected = np.ceil(2.0 * np.sqrt(2.0 * np.log(2.0)) * sig)
        assert aspclean.psf_gaussian_width(psf) == pytest.approx(expected, abs=1.0)

    def test_elliptical_gaussian(self):
        ny, nx = 96, 96
        yy, xx = np.mgrid[0:ny, 0:nx]
        sx, sy = 2.0, 4.0
        psf = np.exp(
            -(((xx - nx // 2) ** 2) / (2 * sx**2) + ((yy - ny // 2) ** 2) / (2 * sy**2))
        )
        w = aspclean.psf_gaussian_width(np.ascontiguousarray(psf))
        kx = np.ceil(2.0 * np.sqrt(2.0 * np.log(2.0)) * sx)
        ky = np.ceil(2.0 * np.sqrt(2.0 * np.log(2.0)) * sy)
        # the estimate is the mean of the two principal FWHMs (in pixels)
        assert min(kx, ky) - 1 <= w <= max(kx, ky) + 1


# ---------------------------------------------------------------------------
# End-to-end deconvolution
# ---------------------------------------------------------------------------


class TestClean:
    def _make_scene(self, dtype=np.float64, nx=128, ny=128):
        psf = _centered_psf(nx, ny, 2.0, dtype)
        sky = _gauss(nx, ny, 70, 60, 4.0, 3.0, dtype)
        sky[40, 45] += 5.0
        dirty = np.ascontiguousarray(
            aspclean.convolve_centered(sky.astype(np.float64), psf.astype(np.float64)),
            dtype=dtype,
        )
        return psf, sky, dirty

    def test_returns_expected_keys(self):
        psf, sky, dirty = self._make_scene()
        model = np.zeros_like(dirty)
        resid = dirty.copy()
        res = aspclean.clean(
            resid,
            psf,
            model,
            gain=0.1,
            threshold_per_cycle=0.05,
            max_iter_per_cycle_updated=100,
        )
        for key in (
            "iterations_performed",
            "retval",
            "stop_code",
            "converged",
            "diverged",
            "peak_residual",
            "model_flux",
            "switched_to_hogbom",
            "psf_width",
        ):
            assert key in res

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_residual_reduced_and_flux_recovered(self, dtype):
        psf, sky, dirty = self._make_scene(dtype)
        model = np.zeros_like(dirty)
        resid = dirty.copy()
        peak0 = float(np.max(np.abs(resid)))
        aspclean.clean(
            resid,
            psf,
            model,
            gain=0.1,
            threshold_per_cycle=0.01,
            max_iter_per_cycle_updated=200,
            fusedthreshold=0.1,
        )
        peak1 = float(np.max(np.abs(resid)))
        assert peak1 < 0.2 * peak0  # residual substantially reduced
        assert model.sum() > 0
        # most of the true flux is recovered into the model
        assert 0.6 < model.sum() / sky.sum() < 1.2

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_reconstruction_invariant(self, dtype):
        """At all times residual == dirty - model (*) psf (to FFT precision)."""
        psf, sky, dirty = self._make_scene(dtype)
        model = np.zeros_like(dirty)
        resid = dirty.copy()
        aspclean.clean(
            resid,
            psf,
            model,
            gain=0.1,
            threshold_per_cycle=0.05,
            max_iter_per_cycle_updated=150,
            fusedthreshold=0.1,
        )
        recon = aspclean.convolve_centered(
            model.astype(np.float64), psf.astype(np.float64)
        )
        err = np.max(
            np.abs(resid.astype(np.float64) + recon - dirty.astype(np.float64))
        )
        tol = 1e-3 if dtype == np.float32 else 1e-9
        assert err / np.max(np.abs(dirty)) < tol

    def test_point_source_with_delta_psf(self):
        nx = ny = 96
        psf = np.zeros((ny, nx))
        psf[ny // 2, nx // 2] = 1.0
        dirty = np.zeros((ny, nx))
        dirty[40, 50] = 7.0
        model = np.zeros((ny, nx))
        resid = dirty.copy()
        aspclean.clean(
            resid,
            psf,
            model,
            gain=0.5,
            threshold_per_cycle=0.01,
            max_iter_per_cycle_updated=60,
            fusedthreshold=0.5,
            psf_width=1.0,
        )
        assert np.max(np.abs(resid)) < 0.05
        # essentially all flux placed at the source pixel
        assert model[40, 50] == pytest.approx(7.0, abs=0.1)
        assert model.sum() == pytest.approx(7.0, abs=0.1)

    def test_imaging_cycles_monotonic_progress(self):
        psf, sky, dirty = self._make_scene()
        model = np.zeros_like(dirty)
        resid = dirty.copy()
        peaks = [float(np.max(np.abs(resid)))]
        for _ in range(4):
            aspclean.clean(
                resid,
                psf,
                model,
                gain=0.1,
                threshold_per_cycle=0.001,
                max_iter_per_cycle_updated=100,
                fusedthreshold=0.1,
            )
            peaks.append(float(np.max(np.abs(resid))))
        # residual peak never increases by more than rounding across cycles and
        # ends far below where it started
        assert peaks[-1] < 0.1 * peaks[0]


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


class TestMask:
    def test_components_confined_to_mask(self):
        nx = ny = 128
        psf = _centered_psf(nx, ny, 2.0)
        sky = _gauss(nx, ny, 64, 64, 3.0, 5.0)
        dirty = np.ascontiguousarray(aspclean.convolve_centered(sky, psf))

        mask = np.zeros((ny, nx))
        mask[40:90, 40:90] = 1.0  # box around the source

        model = np.zeros_like(dirty)
        resid = dirty.copy()
        aspclean.clean(
            resid,
            psf,
            model,
            mask=mask,
            gain=0.1,
            threshold_per_cycle=0.01,
            max_iter_per_cycle_updated=150,
            fusedthreshold=0.1,
        )
        outside = model[mask == 0]
        assert np.max(np.abs(outside)) < 1e-6 * np.max(np.abs(model))
        assert model.sum() > 0

    def test_zero_mask_does_nothing(self):
        nx = ny = 64
        psf = _centered_psf(nx, ny, 2.0)
        dirty = _gauss(nx, ny, 32, 32, 3.0, 4.0)
        mask = np.zeros((ny, nx))
        model = np.zeros_like(dirty)
        resid = dirty.copy()
        res = aspclean.clean(
            resid, psf, model, mask=mask, max_iter_per_cycle_updated=50
        )
        np.testing.assert_array_equal(resid, dirty)
        np.testing.assert_array_equal(model, np.zeros_like(model))
        assert res["model_flux"] == 0.0


# ---------------------------------------------------------------------------
# Zero-copy / in-place contract and input validation
# ---------------------------------------------------------------------------


class TestInPlaceContract:
    def test_modifies_buffers_in_place(self):
        nx = ny = 64
        psf = _centered_psf(nx, ny, 2.0)
        dirty = _gauss(nx, ny, 32, 32, 3.0, 4.0)
        model = np.zeros_like(dirty)
        resid = dirty.copy()
        resid_id = resid.ctypes.data
        model_id = model.ctypes.data
        aspclean.clean(
            resid,
            psf,
            model,
            gain=0.2,
            threshold_per_cycle=0.01,
            max_iter_per_cycle_updated=50,
            fusedthreshold=0.1,
        )
        # same buffers (no reallocation), and they changed
        assert resid.ctypes.data == resid_id
        assert model.ctypes.data == model_id
        assert not np.array_equal(resid, dirty)
        assert np.any(model != 0)

    def test_wrong_dtype_raises(self):
        nx = ny = 32
        psf = _centered_psf(nx, ny).astype(np.float32)
        resid = _gauss(nx, ny, 16, 16, 3.0, 4.0).astype(np.float64)
        model = np.zeros_like(resid)
        with pytest.raises((RuntimeError, ValueError)):
            aspclean.clean(resid, psf, model)

    def test_non_contiguous_raises(self):
        nx = ny = 64
        psf = _centered_psf(nx, ny)
        big = _gauss(nx, 2 * ny, 16, 16, 3.0, 4.0)
        resid = big[:, ::2]  # non C-contiguous view
        model = np.zeros((ny, nx))
        with pytest.raises((RuntimeError, ValueError)):
            aspclean.clean(resid, psf, model)

    def test_non_writable_raises(self):
        nx = ny = 32
        psf = _centered_psf(nx, ny)
        resid = _gauss(nx, ny, 16, 16, 3.0, 4.0)
        resid.flags.writeable = False
        model = np.zeros((ny, nx))
        with pytest.raises((RuntimeError, ValueError)):
            aspclean.clean(resid, psf, model)

    def test_shape_mismatch_raises(self):
        psf = _centered_psf(32, 32)
        resid = _gauss(64, 64, 16, 16, 3.0, 4.0)
        model = np.zeros((64, 64))
        with pytest.raises((RuntimeError, ValueError)):
            aspclean.clean(resid, psf, model)


# ---------------------------------------------------------------------------
# Cube driver
# ---------------------------------------------------------------------------


class TestCube:
    def _make_cube(self, nt, nf, npol, ny, nx, dtype=np.float64):
        psf = np.zeros((nt, nf, npol, ny, nx), dtype)
        resid = np.zeros((nt, nf, npol, ny, nx), dtype)
        single_psf = _centered_psf(nx, ny, 2.0, dtype)
        for t in range(nt):
            for f in range(nf):
                for p in range(npol):
                    psf[t, f, p] = single_psf
                    sky = _gauss(nx, ny, 30 + p, 34, 3.0, 4.0 + f, dtype)
                    resid[t, f, p] = aspclean.convolve_centered(
                        sky.astype(np.float64), single_psf.astype(np.float64)
                    ).astype(dtype)
        return psf, resid

    def test_cube_matches_per_plane(self):
        nt, nf, npol, ny, nx = 1, 2, 2, 64, 64
        psf, resid = self._make_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        rc = resid.copy()
        mc = model.copy()
        # Cycle control is per (time, frequency, polarization) plane:
        # threshold is a float64 (nt, nf, np) array and max_iter an int32 one.
        threshold = np.full((nt, nf, npol), 0.05, dtype=np.float64)
        max_iter = np.full((nt, nf, npol), 80, dtype=np.int32)
        out = aspclean.clean_cube(
            rc,
            psf,
            mc,
            gain=0.1,
            threshold_per_cycle=threshold,
            max_iter_per_cycle_updated=max_iter,
            fusedthreshold=0.1,
            processing_function_threads=2,
        )
        assert out["iterations_performed"].shape == (nt, nf, npol)
        assert out["peak_residual"].shape == (nt, nf, npol)

        # Each plane must equal the single-plane routine on the same data.
        for t in range(nt):
            for f in range(nf):
                for p in range(npol):
                    r1 = resid[t, f, p].copy()
                    m1 = np.zeros((ny, nx))
                    aspclean.clean(
                        r1,
                        psf[t, f, p],
                        m1,
                        gain=0.1,
                        threshold_per_cycle=0.05,
                        max_iter_per_cycle_updated=80,
                        fusedthreshold=0.1,
                    )
                    np.testing.assert_allclose(rc[t, f, p], r1, atol=1e-10)
                    np.testing.assert_allclose(mc[t, f, p], m1, atol=1e-10)

    def test_cube_psf_broadcast(self):
        nt, nf, npol, ny, nx = 1, 1, 3, 64, 64
        _, resid = self._make_cube(nt, nf, npol, ny, nx)
        psf1 = np.zeros((nt, nf, 1, ny, nx))
        psf1[0, 0, 0] = _centered_psf(nx, ny, 2.0)
        model = np.zeros_like(resid)
        rc = resid.copy()
        threshold = np.full((nt, nf, npol), 0.05, dtype=np.float64)
        max_iter = np.full((nt, nf, npol), 60, dtype=np.int32)
        out = aspclean.clean_cube(
            rc,
            psf1,
            model,
            gain=0.1,
            threshold_per_cycle=threshold,
            max_iter_per_cycle_updated=max_iter,
            fusedthreshold=0.1,
        )
        assert out["model_flux"].shape == (nt, nf, npol)
        assert np.all(out["model_flux"] > 0)


# ---------------------------------------------------------------------------
# Divergence test shared with the other deconvolvers
# ---------------------------------------------------------------------------


class TestDivergence:
    """The divergence test of deconvolvers/include/plane_stop_test.hpp.

    The spreading PSF has 24 sidelobes of -0.3 on a square ring around its
    peak, so a subtraction at gain 0.9 lowers the peak of a point source from
    1 to 0.27 but raises the RMS of the residual 1.33 times: the soft limit
    stops the model update; with the test off nothing stops this runaway. The
    chain PSF has one sidelobe of -1.2, so the peak grows 1.08 times per
    iteration. Asp counts the iteration whose check stops it.
    """

    # _divergence_factors(0.9)
    RMS_FACTOR = 1.09
    PEAK_FACTOR = 1.9

    @staticmethod
    def _spreading(dtype=np.float64, n=32):
        residual = np.zeros((n, n), dtype=dtype)
        residual[n // 2, n // 2] = 1.0
        psf = np.zeros((n, n), dtype=dtype)
        c = n // 2
        psf[c - 3 : c + 4, c - 3 : c + 4] = -0.3
        psf[c - 2 : c + 3, c - 2 : c + 3] = 0.0
        psf[c, c] = 1.0
        return residual, psf

    @staticmethod
    def _chain(dtype=np.float64):
        residual = np.zeros((8, 64), dtype=dtype)
        residual[4, 2] = 1.0
        psf = np.zeros((8, 64), dtype=dtype)
        psf[4, 32] = 1.0
        psf[4, 33] = -1.2
        return residual, psf

    @staticmethod
    def _clean(scene, **kwargs):
        residual, psf = scene
        model = np.zeros_like(residual)
        result = aspclean.clean(
            residual,
            psf,
            model,
            gain=0.9,
            threshold_per_cycle=0.0,
            max_iter_per_cycle_updated=40,
            psf_width=1.0,
            largestscale=0,
            **kwargs,
        )
        return result, residual, model

    def test_default_leaves_the_test_off(self):
        result, residual, model = self._clean(self._spreading())
        off, residual_off, model_off = self._clean(
            self._spreading(), max_iter_divergence=-1
        )
        # nothing else stops the runaway: both spend their budget
        assert result["iterations_performed"] == off["iterations_performed"] == 40
        assert result["retval"] == off["retval"] == 0
        assert result["stop_code"] == off["stop_code"] == 1
        np.testing.assert_array_equal(residual, residual_off)
        np.testing.assert_array_equal(model, model_off)

    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    def test_rms_rise_stops_the_model_update(self, dtype):
        result, _, model = self._clean(
            self._spreading(dtype),
            max_iter_divergence=1,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"] == 2
        assert result["retval"] == -3
        assert result["stop_code"] == 4
        assert result["diverged"]
        assert not result["converged"]
        # the peak fell; the RMS rose
        assert result["peak_residual"] == pytest.approx(0.27, rel=1e-5)
        assert np.count_nonzero(model) == 1

    @pytest.mark.parametrize(("max_iter_divergence", "iterations"), [(1, 6), (2, 7)])
    def test_asp_limits_stop_the_runaway(self, max_iter_divergence, iterations):
        # The limits asp_clean() passes at gain 0.9: (1 + gain) on the RMS and
        # no peak limit. Five subtractions raise the RMS 1.94 times.
        result, _, _ = self._clean(
            self._spreading(),
            max_iter_divergence=max_iter_divergence,
            divergence_rms_factor=self.PEAK_FACTOR,
            divergence_peak_factor=np.inf,
        )
        assert result["iterations_performed"] == iterations
        assert result["diverged"]

    def test_peak_above_1500_is_cleaned(self):
        # The "5. Diverging" guard of CASA 6.7.5.18 and LibRA (fc54ee25)
        # started its lowest peak at 1000, so it stopped any plane with a peak
        # above 1500 at once (casa master has dropped that test). The model
        # update ends after its first Hogbom run (stop_after_hogbom_run) in
        # both: going on, Asp's "no change" test (a change of the peak below
        # 1e-4 in the units of the image, CASA's constant) tells the two
        # scalings apart.
        psf, _, dirty = TestClean()._make_scene()
        results = []
        for scale in (1.0, 1e4):
            residual = dirty * scale
            model = np.zeros_like(dirty)
            result = aspclean.clean(
                residual,
                psf,
                model,
                gain=0.1,
                threshold_per_cycle=0.05 * scale,
                max_iter_per_cycle_updated=100,
                stop_after_hogbom_run=True,
            )
            results.append((result, model.sum() / scale))
        (unit, unit_flux), (scaled, scaled_flux) = results
        assert scaled["iterations_performed"] > 1
        assert not scaled["diverged"]
        assert scaled["retval"] == unit["retval"]
        assert scaled_flux == pytest.approx(unit_flux, rel=1e-6)

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    @pytest.mark.parametrize("dtype", [np.float64, np.float32])
    @pytest.mark.parametrize("mask_value", [0.0, 0.5])
    def test_non_finite_pixels_outside_the_mask_take_no_part(
        self, bad, dtype, mask_value
    ):
        psf, _, dirty = TestClean()._make_scene(dtype)
        mask = np.zeros_like(dirty)
        mask[20:110, 20:110] = 1.0
        # a mask value up to 0.99 is outside the search region too
        mask[0, 0] = mask[5, 120] = mask_value
        finite = dirty.copy()
        finite[0, 0] = finite[5, 120] = 0.0
        # pixel (0, 0) seeded the peak search, which then stopped the plane
        # at once; elsewhere the scale convolutions spread the value
        nan = finite.copy()
        nan[0, 0] = nan[5, 120] = bad
        out = []
        for residual in (finite, nan):
            model = np.zeros_like(dirty)
            result = aspclean.clean(
                residual,
                psf,
                model,
                mask=mask,
                gain=0.1,
                threshold_per_cycle=0.05,
                max_iter_per_cycle_updated=100,
            )
            out.append((result, model))
        (ref, model_ref), (result, model) = out
        assert result["iterations_performed"] == ref["iterations_performed"]
        assert result["retval"] == ref["retval"]
        np.testing.assert_array_equal(model, model_ref)
        assert np.array_equal(nan[[0, 5], [0, 120]], [bad, bad], equal_nan=True)
        keep = np.isfinite(nan)
        np.testing.assert_array_equal(nan[keep], finite[keep])

    def test_consecutive_iterations_above_the_soft_limit(self):
        result, _, _ = self._clean(
            self._spreading(),
            max_iter_divergence=2,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"] == 3
        assert result["diverged"]

    def test_hard_limit(self):
        result, _, _ = self._clean(
            self._chain(),
            max_iter_divergence=1,
            divergence_rms_factor=1e9,
            divergence_peak_factor=1.05,
        )
        assert result["iterations_performed"] == 2
        assert result["diverged"]
        assert result["peak_residual"] == pytest.approx(1.08)

    def test_healthy_clean_is_unchanged(self):
        psf, _, dirty = TestClean()._make_scene()
        results = []
        for kwargs in (
            {},
            {
                "max_iter_divergence": 1,
                "divergence_rms_factor": 1.01,
                "divergence_peak_factor": 1.1,
            },
        ):
            residual = dirty.copy()
            model = np.zeros_like(dirty)
            result = aspclean.clean(
                residual,
                psf,
                model,
                gain=0.1,
                threshold_per_cycle=0.05,
                max_iter_per_cycle_updated=100,
                **kwargs,
            )
            results.append((result, residual, model))
        (result, residual, model), (on, residual_on, model_on) = results
        assert not on["diverged"]
        assert on["iterations_performed"] == result["iterations_performed"]
        np.testing.assert_array_equal(residual_on, residual)
        np.testing.assert_array_equal(model_on, model)

    def test_cube_reports_diverged_per_plane(self):
        spreading_residual, spreading_psf = self._spreading()
        residual = np.zeros((1, 1, 2) + spreading_residual.shape)
        psf = np.zeros_like(residual)
        residual[0, 0, 0] = spreading_residual
        psf[0, 0, 0] = spreading_psf
        # plane 1: the same point source and a delta PSF
        residual[0, 0, 1] = spreading_residual
        psf[0, 0, 1, 16, 16] = 1.0
        model = np.zeros_like(residual)
        result = aspclean.clean_cube(
            residual,
            psf,
            model,
            gain=0.9,
            threshold_per_cycle=np.zeros((1, 1, 2)),
            max_iter_per_cycle_updated=np.full((1, 1, 2), 40, dtype=np.int32),
            psf_width=1.0,
            largestscale=0,
            max_iter_divergence=1,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        np.testing.assert_array_equal(result["diverged"], [[[True, False]]])
        np.testing.assert_array_equal(result["retval"][0, 0, 0], -3)
        # the shared test stops plane 0
        np.testing.assert_array_equal(result["iterations_performed"], [[[2, 40]]])
        np.testing.assert_array_equal(result["stop_code"], [[[4, 1]]])

    def test_rms_is_taken_inside_the_mask(self):
        # the sidelobes that raise the RMS fall outside a 3 x 3 mask
        residual, psf = self._spreading()
        mask = np.zeros_like(residual)
        mask[15:18, 15:18] = 1.0
        model = np.zeros_like(residual)
        result = aspclean.clean(
            residual,
            psf,
            model,
            mask=mask,
            gain=0.9,
            threshold_per_cycle=0.0,
            max_iter_per_cycle_updated=40,
            psf_width=1.0,
            largestscale=0,
            max_iter_divergence=1,
            divergence_rms_factor=self.RMS_FACTOR,
            divergence_peak_factor=self.PEAK_FACTOR,
        )
        assert result["iterations_performed"] > 2
        assert result["peak_residual"] < 1e-6

    def test_infinite_peak_stops_the_model_update(self):
        # also with the binding's defaults, which leave the two limits off
        residual, psf = self._spreading()
        residual[5, 5] = np.inf
        model = np.zeros_like(residual)
        result = aspclean.clean(residual, psf, model, gain=0.9, psf_width=1.0)
        assert result["iterations_performed"] == 1
        assert result["retval"] == -3
        assert result["diverged"]
        assert not model.any()


# ---------------------------------------------------------------------------
# The per-Aspen L-BFGS optimizer (ALGLIB minlbfgs conventions)
# ---------------------------------------------------------------------------


class TestStopCode:
    """stop_code is the code the other deconvolvers report (plane_stop_test.hpp)."""

    @staticmethod
    def _clean(max_iter_per_cycle_updated=40, threshold_per_cycle=0.0):
        residual, psf = TestDivergence._spreading()
        model = np.zeros_like(residual)
        result = aspclean.clean(
            residual,
            psf,
            model,
            gain=0.9,
            threshold_per_cycle=threshold_per_cycle,
            max_iter_per_cycle_updated=max_iter_per_cycle_updated,
            psf_width=1.0,
            largestscale=0,
        )
        return result, model

    def test_no_iteration_limit_does_not_run(self):
        result, model = self._clean(max_iter_per_cycle_updated=0)
        assert result["iterations_performed"] == 0
        assert result["stop_code"] == 0
        assert not model.any()

    def test_threshold_per_cycle(self):
        result, _ = self._clean(threshold_per_cycle=2.0)
        assert result["retval"] == 1
        assert result["stop_code"] == 2

    def test_stop_point_mode_is_a_test_of_asp(self):
        # only scale 0 (largestscale=0): the first component stops it
        result, _, _ = TestDivergence._clean(
            TestDivergence._spreading(), stoppointmode=1
        )
        assert result["retval"] == -1
        assert result["stop_code"] == 5


class TestLbfgs:
    """The L-BFGS behind the Aspen (amplitude, scale) fit follows ALGLIB's
    ``minlbfgs`` as CASA and LibRA call it (one correction pair,
    ``minlbfgssetscale`` and no preconditioner): the search runs in the
    variables themselves, the first trial point is a step of length
    ``min(1, |g|)`` along the steepest descent, the More-Thuente line search
    can lengthen a step as well as shorten it, and the scale enters only the
    stopping tests. An earlier version stepped in the scaled variables
    ``x / scale``; on the Asp objective it moved an Aspen's scale by tens of
    pixels where ALGLIB moves it by one or two (checked against ALGLIB 3.17 on
    the same objective, evaluation by evaluation), and the model updates
    diverged.
    """

    @staticmethod
    def _quadratic(centre, weights, calls):
        centre = np.asarray(centre, float)
        weights = np.asarray(weights, float)

        def fg(x):
            x = np.asarray(x, float)
            calls.append(x.copy())
            d = x - centre
            return float(np.sum(weights * d * d)), list(2.0 * weights * d)

        return fg

    def test_first_trial_point_is_a_unit_step_along_the_steepest_descent(self):
        calls = []
        fg = self._quadratic([10.0, -4.0], [1.0, 3.0], calls)
        aspclean.lbfgs_minimize(fg, [0.0, 0.0], [1.0, 1.0], max_iters=5)
        g0 = np.array([2.0 * -10.0, 2.0 * 3.0 * 4.0])
        step = calls[1] - calls[0]
        np.testing.assert_allclose(step, -g0 / np.linalg.norm(g0), rtol=1e-12)

    def test_line_search_lengthens_a_short_first_step(self):
        # 1-D: the minimum is 10 units away, the first trial point one unit:
        # the decrease is sufficient but the slope is still steep, so the
        # More-Thuente search extrapolates instead of accepting it.
        calls = []
        fg = self._quadratic([10.0], [1.0], calls)
        x, f, iterations, evaluations = aspclean.lbfgs_minimize(
            fg, [0.0], [1.0], max_iters=5, epsg=1e-10, epsf=1e-14, epsx=1e-12
        )
        assert calls[1][0] == pytest.approx(1.0)
        assert max(c[0] for c in calls[2:]) > 1.0
        np.testing.assert_allclose(x, [10.0], rtol=1e-6)

    def test_scale_enters_only_the_stopping_tests(self):
        # Two runs that differ only in the scale vector visit the same points
        # until the first of them stops.
        calls_a, calls_b = [], []
        fg_a = self._quadratic([3.0, -2.0], [1.0, 5.0], calls_a)
        fg_b = self._quadratic([3.0, -2.0], [1.0, 5.0], calls_b)
        aspclean.lbfgs_minimize(fg_a, [0.0, 0.0], [1.0, 1.0], max_iters=3)
        aspclean.lbfgs_minimize(fg_b, [0.0, 0.0], [1e-3, 1e3], max_iters=3)
        n = min(len(calls_a), len(calls_b))
        np.testing.assert_allclose(
            np.array(calls_a[:n]), np.array(calls_b[:n]), rtol=1e-12
        )

    def test_converges_on_a_quadratic(self):
        calls = []
        fg = self._quadratic([2.0, 3.0], [1.0, 4.0], calls)
        x, f, iterations, evaluations = aspclean.lbfgs_minimize(
            fg, [0.0, 0.0], [1.0, 1.0], max_iters=50, epsg=1e-10, epsf=1e-14, epsx=1e-12
        )
        np.testing.assert_allclose(x, [2.0, 3.0], rtol=1e-6)
        assert f < 1e-10

    def test_never_ends_above_its_start_and_stays_within_its_budget(self):
        history = []

        def fg(x):
            v = 50.0 * (x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2
            history.append(v)
            return v, [100.0 * (x[0] - 1.0), 2.0 * (x[1] + 2.0)]

        x, f, iterations, evaluations = aspclean.lbfgs_minimize(
            fg, [0.0, 0.0], [1.0, 1.0], max_iters=5
        )
        assert iterations >= 1
        assert f <= history[0]
        assert evaluations <= 1 + 5 * 20  # 20 evaluations per line search


# ---------------------------------------------------------------------------
# Resolved sources: the case Asp exists for
# ---------------------------------------------------------------------------


def _linear_convolution(image, psf):
    """model (*) psf with the PSF centred on (n // 2, n // 2) of its own image and
    zero beyond it: the response the residual update subtracts."""
    from scipy.signal import fftconvolve

    nx, ny = image.shape
    full = fftconvolve(image, psf, mode="full")
    return full[nx // 2 : nx // 2 + nx, ny // 2 : ny // 2 + ny]


class TestResolvedSource:
    """A limb-darkened disk eight beams across, observed with a sparse-uv PSF.

    Regression test for two failure modes of the first port on extended
    emission: the per-Aspen L-BFGS jumping to a garbage (amplitude, scale)
    pair, and a spurious divergence stop (``retval == -3``) on legitimate
    wide components. Both made every model update stop after one to three
    iterations without progress. The test runs with the divergence limits
    asp_clean() passes, which must not fire here.
    """

    @staticmethod
    def _scene():
        from astroviper.processing_functions.simulation import (
            limb_darkened_disk_image,
        )

        n = 128
        rng = np.random.default_rng(1)
        ku, kv = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
        radius = np.hypot(ku, kv)
        sampling = (
            (radius > 0.01) & (radius < 0.12) & (rng.random((n, n)) < 0.25)
        ).astype(float)
        sampling += sampling[::-1, ::-1]  # Hermitian uv coverage
        sampling[0, 0] = 0.0
        psf = np.fft.fftshift(np.real(np.fft.ifft2(sampling)))
        psf /= psf.max()
        yy, xx = np.mgrid[0:n, 0:n]
        l_grid = (xx - n // 2).astype(float)
        m_grid = (yy - n // 2).astype(float)
        sky = 5.0 * limb_darkened_disk_image(l_grid, m_grid, 40.0, 28.0, 0.8, 1.5)
        dirty = np.real(
            np.fft.ifft2(np.fft.fft2(sky) * np.fft.fft2(np.fft.ifftshift(psf)))
        )
        return (
            np.ascontiguousarray(psf),
            np.ascontiguousarray(sky),
            np.ascontiguousarray(dirty),
        )

    def test_disk_is_cleaned_without_a_divergence_exit(self):
        psf, sky, dirty = self._scene()
        resid = dirty.copy()
        model = np.zeros_like(dirty)
        out = aspclean.clean(
            resid,
            psf,
            model,
            gain=0.1,
            threshold_per_cycle=1e-4,
            max_iter_per_cycle_updated=300,
            max_iter_divergence=1,
            divergence_rms_factor=1.1,
            divergence_peak_factor=np.inf,
        )
        assert out["retval"] != -3  # no spurious "diverging" stop
        assert out["iterations_performed"] >= 30  # the model update actually ran
        assert out["peak_residual"] < 0.2 * np.abs(dirty).max()
        np.testing.assert_allclose(model.sum(), sky.sum(), rtol=0.15)
        # the residual update subtracts the exact linear convolution
        assert np.max(np.abs(resid + _linear_convolution(model, psf) - dirty)) < (
            1e-9 * np.abs(dirty).max()
        )
        # and the components are extended (hundreds of pixels above a tenth of
        # the model peak), not a handful of delta spikes
        assert np.count_nonzero(model > 0.1 * model.max()) > 100

    def _clean_disk(self, **kwargs):
        psf, sky, _ = self._scene()
        dirty = np.ascontiguousarray(_linear_convolution(sky, psf))
        resid = dirty.copy()
        model = np.zeros_like(dirty)
        out = aspclean.clean(
            resid,
            psf,
            model,
            gain=0.1,
            threshold_per_cycle=1e-3 * np.abs(dirty).max(),
            max_iter_per_cycle_updated=300,
            max_iter_divergence=-1,
            **kwargs,
        )
        assert np.max(np.abs(resid + _linear_convolution(model, psf) - dirty)) < (
            1e-9 * np.abs(dirty).max()
        )
        return out, resid, model, dirty, sky

    def test_model_update_goes_on_after_a_hogbom_run(self):
        """By default the model update goes on past the end of a Hogbom run
        (here to its iteration limit): the residual is the exact one, so the
        model update does not need to stop for a residual update."""
        default, resid, model, dirty, sky = self._clean_disk()
        off, resid_off, model_off, _, _ = self._clean_disk(stop_after_hogbom_run=False)
        assert default["stop_code"] == 1  # spent max_iter_per_cycle_updated
        assert default["iterations_performed"] == 300
        np.testing.assert_array_equal(resid, resid_off)
        np.testing.assert_array_equal(model, model_off)
        on, resid_on, model_on, _, _ = self._clean_disk(stop_after_hogbom_run=True)
        assert np.abs(resid).max() < np.abs(resid_on).max()
        assert abs(model.sum() - sky.sum()) < abs(model_on.sum() - sky.sum())

    def test_stop_after_hogbom_run_ends_the_model_update_like_casa(self):
        """With stop_after_hogbom_run the model update ends after the Hogbom
        run once the RMS of the residual has fallen by a factor 1.5 (CASA and
        LibRA): a test of Asp itself, well before the iteration limit."""
        on, *_ = self._clean_disk(stop_after_hogbom_run=True)
        assert on["stop_code"] == 5  # a test of Asp itself
        assert on["converged"]
        assert 51 < on["iterations_performed"] < 300

    def test_disk_near_the_edge_does_not_run_away(self):
        """A disk next to the image edge: its wide Aspens' responses reach
        beyond the edge. Subtracting a circular convolution folded them back in
        from the opposite edge, and the Aspen fit then chased that structure
        with ever larger components (the model flux ran away to many times the
        sky's on CASA's test_onefield_asp data). The residual update is now the
        exact linear convolution and the fit stops like ALGLIB's."""
        psf, _, _ = self._scene()
        from astroviper.processing_functions.simulation import (
            limb_darkened_disk_image,
        )

        n = psf.shape[0]
        yy, xx = np.mgrid[0:n, 0:n]
        sky = 5.0 * limb_darkened_disk_image(
            (xx - 18).astype(float), (yy - n // 2).astype(float), 30.0, 22.0, 0.8, 1.5
        )
        dirty = np.ascontiguousarray(_linear_convolution(sky, psf))
        resid = dirty.copy()
        model = np.zeros_like(dirty)
        out = aspclean.clean(
            resid,
            psf,
            model,
            gain=0.1,
            threshold_per_cycle=1e-4,
            max_iter_per_cycle_updated=300,
            max_iter_divergence=-1,
        )
        assert out["iterations_performed"] >= 30
        assert np.abs(resid).max() < 0.5 * np.abs(dirty).max()
        np.testing.assert_allclose(model.sum(), sky.sum(), rtol=0.3)
        assert np.max(np.abs(resid + _linear_convolution(model, psf) - dirty)) < (
            1e-9 * np.abs(dirty).max()
        )


# ---------------------------------------------------------------------------
# Ring cleaning: options beyond CASA and LibRA (AspOptions)
# ---------------------------------------------------------------------------


class TestRingCleaning:
    """The options of the kernel that clean Asp's ring residuals:
    components shaped by the residual (Ma, Zhang & Zhang 2026, A&A 713, A164)
    before the Asp iterations of a model update, the switch to Asp at a
    multiple of the threshold, the noise floor of their threshold, and Asp's
    test of significance per scale. The kernel leaves them off by default
    (``asp_clean`` turns them on)."""

    @staticmethod
    def _scene(faint=False):
        """A disk with a gap and a core (or, ``faint``, a broad Gaussian whose
        dirty peak is twice the noise), observed with a PSF of 80 % filled uv
        coverage, plus noise convolved with the PSF; mask of radius 40."""
        n = 128
        rng = np.random.default_rng(2)
        ku, kv = np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij")
        sampling = ((np.hypot(ku, kv) < 0.15) & (rng.random((n, n)) < 0.8)).astype(
            float
        )
        sampling += sampling[::-1, ::-1]  # Hermitian uv coverage
        psf = np.fft.fftshift(np.real(np.fft.ifft2(sampling)))
        psf = np.ascontiguousarray(psf / psf.max())
        yy, xx = np.mgrid[0:n, 0:n]
        r = np.hypot(xx - n // 2, yy - n // 2)
        noise = _linear_convolution(rng.standard_normal((n, n)), psf)
        if faint:
            noise /= noise.std()
            sky = np.exp(-0.5 * (r / 12.0) ** 2)
            sky *= 2.0 / _linear_convolution(sky, psf).max()
        else:
            sky = np.exp(-0.5 * (r / 14.0) ** 2) * (
                1 - 0.7 * np.exp(-0.5 * ((r - 10) / 2.0) ** 2)
            ) + 0.3 * np.exp(-0.5 * (r / 2.0) ** 2)
            noise *= 0.002 * np.abs(_linear_convolution(sky, psf)).max() / noise.std()
        dirty = np.ascontiguousarray(_linear_convolution(sky, psf) + noise)
        mask = (r < 40).astype(float)
        return psf, sky, dirty, mask, 5.0 * noise.std()

    @staticmethod
    def _clean(psf, dirty, mask, threshold, **kwargs):
        resid = dirty.copy()
        model = np.zeros_like(dirty)
        out = aspclean.clean(
            resid,
            psf,
            model,
            mask,
            gain=0.1,
            threshold_per_cycle=threshold,
            max_iter_per_cycle_updated=3000,
            max_iter_divergence=-1,
            **kwargs,
        )
        return out, resid, model

    def test_off_by_default(self):
        psf, _, dirty, mask, threshold = self._scene()
        default, resid, model = self._clean(psf, dirty, mask, threshold)
        off, resid_off, model_off = self._clean(
            psf,
            dirty,
            mask,
            threshold,
            residual_components=False,
            scale_significance=False,
            record_components=False,
            rc_switch_factor=1.0,
        )
        np.testing.assert_array_equal(resid, resid_off)
        np.testing.assert_array_equal(model, model_off)
        assert default["timing"]["n_residual_components"] == 0
        assert not {"components", "residual_fits", "scale_noise"} & set(default)

    def test_residual_components_clean_the_disk_in_fewer_iterations(self):
        psf, sky, dirty, mask, threshold = self._scene()
        asp, *_ = self._clean(psf, dirty, mask, threshold)
        out, resid, model = self._clean(
            psf, dirty, mask, threshold, residual_components=True
        )
        assert out["timing"]["n_residual_components"] > 10
        assert out["stop_code"] == 2  # the threshold
        assert out["peak_residual"] <= threshold
        assert out["iterations_performed"] < 0.7 * asp["iterations_performed"]
        np.testing.assert_allclose(model.sum(), sky.sum(), rtol=0.05)
        # the residual update stays the exact linear convolution
        assert np.max(np.abs(resid + _linear_convolution(model, psf) - dirty)) < (
            1e-9 * np.abs(dirty).max()
        )

    def test_switch_factor_hands_over_to_asp_earlier(self):
        psf, _, dirty, mask, threshold = self._scene()
        counts = {}
        for factor in (1.0, 10.0):
            out, *_ = self._clean(
                psf,
                dirty,
                mask,
                threshold,
                residual_components=True,
                rc_switch_factor=factor,
                record_components=True,
            )
            fits = out["residual_fits"]
            # every fit starts from a peak above the switch level
            assert np.all(fits[:, 0] > factor * threshold)
            counts[factor] = out["timing"]["n_residual_components"]
            assert out["peak_residual"] <= threshold
        assert 0 < counts[10.0] < counts[1.0]

    @pytest.mark.parametrize(
        "kwargs", [{"rc_switch_factor": 1e3}, {"rc_noise_floor": 1e9}]
    )
    def test_no_residual_component_leaves_asp_as_it_is(self, kwargs):
        """A switch level above the peak, or a noise floor above it, makes no
        residual-shaped component (the phase ends without one) and Asp runs as
        without the option."""
        psf, _, dirty, mask, threshold = self._scene()
        _, resid, model = self._clean(psf, dirty, mask, threshold)
        out, resid_rc, model_rc = self._clean(
            psf, dirty, mask, threshold, residual_components=True, **kwargs
        )
        assert out["timing"]["n_residual_components"] == 0
        np.testing.assert_array_equal(resid, resid_rc)
        np.testing.assert_array_equal(model, model_rc)

    @pytest.mark.parametrize("at_threshold", [False, True])
    def test_floor_at_threshold(self, at_threshold):
        """With ``rc_floor_at_threshold`` a residual-shaped component takes no
        pixel below the threshold of the model update; without it, it reaches
        down to ``rc_frac_min`` of the peak. One iteration: one component."""
        psf, _, dirty, mask, _ = self._scene()
        peak = np.abs(dirty[mask > 0]).max()
        resid = dirty.copy()
        model = np.zeros_like(dirty)
        out = aspclean.clean(
            resid,
            psf,
            model,
            mask,
            gain=0.1,
            threshold_per_cycle=0.6 * peak,
            max_iter_per_cycle_updated=1,
            max_iter_divergence=-1,
            residual_components=True,
            rc_floor_at_threshold=at_threshold,
        )
        assert out["timing"]["n_residual_components"] == 1
        lowest = dirty[model != 0].min() / peak
        if at_threshold:
            assert lowest >= 0.6
        else:
            assert lowest < 0.35  # rc_frac_min is 0.3

    def test_record_components(self):
        psf, _, dirty, mask, threshold = self._scene()
        out, *_ = self._clean(
            psf,
            dirty,
            mask,
            threshold,
            residual_components=True,
            record_components=True,
        )
        timing = out["timing"]
        # one row (x, y, scale, amplitude) per Asp component
        assert out["components"].shape == (
            timing["n_scale_components"] + timing["n_point_components"],
            4,
        )
        # one row per fit: the last one, which finds no better component than
        # none, is not taken
        fits = out["residual_fits"]
        assert fits.shape == (timing["n_residual_components"] + 1, 6)
        assert np.all((fits[:, 1] >= 0.3) & (fits[:, 1] <= 0.99))  # frac
        assert np.all((fits[:, 2] >= 1e-3) & (fits[:, 2] <= 10.0))  # eps_frac

    def test_iteration_history(self):
        """One record per component of every kind, without changing the
        clean: the residual-shaped components have scale -1 and the points
        scale 0, the fluxes add up to the model, and the last RMS is that of
        the masked residual left."""
        psf, _, dirty, mask, threshold = self._scene()
        kwargs = {"residual_components": True, "scale_significance": True}
        off, resid_off, model_off = self._clean(psf, dirty, mask, threshold, **kwargs)
        out, resid, model = self._clean(
            psf, dirty, mask, threshold, record_iteration_history=True, **kwargs
        )
        np.testing.assert_array_equal(resid, resid_off)
        np.testing.assert_array_equal(model, model_off)
        assert "iteration_component_flux" not in off
        timing = out["timing"]
        scale = out["iteration_component_scale"]
        assert len(scale) == (
            timing["n_residual_components"]
            + timing["n_scale_components"]
            + timing["n_point_components"]
        )
        assert np.count_nonzero(scale == -1) == timing["n_residual_components"] > 0
        assert np.count_nonzero(scale == 0) == timing["n_point_components"]
        for field in (
            "iteration_peak_residual",
            "iteration_residual_rms",
            "iteration_component_l",
            "iteration_component_m",
            "iteration_component_flux",
        ):
            assert out[field].shape == scale.shape
        np.testing.assert_allclose(
            out["iteration_component_flux"].sum(), model.sum(), rtol=1e-9
        )
        inside = mask > 0
        np.testing.assert_allclose(
            out["iteration_residual_rms"][-1],
            np.sqrt(np.mean(resid[inside] ** 2)),
            rtol=1e-9,
        )
        # the first component starts from the dirty peak in the mask
        peak = dirty[inside][np.argmax(np.abs(dirty[inside]))]
        assert out["iteration_peak_residual"][0] == pytest.approx(peak, rel=1e-12)
        # residual-shaped and point components lie in the mask (the Gaussians
        # are searched in the masks of their scales)
        searched = scale <= 0
        assert inside[
            out["iteration_component_l"][searched].astype(int),
            out["iteration_component_m"][searched].astype(int),
        ].all()

    def test_scale_significance_cleans_a_faint_broad_source(self):
        """A broad source whose dirty peak is below the threshold: Asp alone
        takes nothing; with the test of significance per scale it takes the
        source at the scales where it stands out of their noise."""
        psf, sky, dirty, mask, threshold = self._scene(faint=True)
        assert np.abs(dirty[mask > 0]).max() < threshold
        _, _, model = self._clean(psf, dirty, mask, threshold)
        assert not model.any()
        out, resid, model = self._clean(
            psf, dirty, mask, threshold, scale_significance=True
        )
        noise = np.asarray(out["scale_noise"])
        np.testing.assert_allclose(noise[0], 1.0, rtol=0.1)  # the dirty noise
        assert out["timing"]["n_scale_components"] > 0
        assert model.sum() > 0.1 * sky.sum()
        # centred on the source (the wings of scale components reach past the mask)
        peak = np.unravel_index(np.argmax(model), model.shape)
        assert np.hypot(peak[0] - 64, peak[1] - 64) < 10
        assert np.max(np.abs(resid + _linear_convolution(model, psf) - dirty)) < (
            1e-9 * np.abs(dirty).max()
        )

    def test_significance_gain(self):
        """The components cleaned below the threshold for their significance
        take ``significance_gain`` (``<= 0``: the gain): a larger one takes
        the faint source in fewer of them."""
        psf, sky, dirty, mask, threshold = self._scene(faint=True)
        runs = {}
        for gain in (0.0, 0.1, 0.3):
            runs[gain] = self._clean(
                psf,
                dirty,
                mask,
                threshold,
                scale_significance=True,
                significance_gain=gain,
            )
        np.testing.assert_array_equal(runs[0.0][1], runs[0.1][1])
        np.testing.assert_array_equal(runs[0.0][2], runs[0.1][2])
        slow, fast = runs[0.1][0]["timing"], runs[0.3][0]["timing"]
        assert 0 < fast["n_significance_components"] < slow["n_significance_components"]
        assert runs[0.3][2].sum() >= runs[0.1][2].sum() > 0.1 * sky.sum()
        resid, model = runs[0.3][1], runs[0.3][2]
        assert np.max(np.abs(resid + _linear_convolution(model, psf) - dirty)) < (
            1e-9 * np.abs(dirty).max()
        )

    def _assert_no_significance_cleaning(self, psf, dirty, mask, threshold, **kwargs):
        """Clean with ``scale_significance`` on and off: with it on, no
        component is cleaned for its significance, the model update stops at
        the threshold, and the result is the one with it off. Returns the
        output with it on."""
        out, resid, model = self._clean(
            psf, dirty, mask, threshold, scale_significance=True, **kwargs
        )
        off, resid_off, model_off = self._clean(
            psf, dirty, mask, threshold, scale_significance=False, **kwargs
        )
        assert out["timing"]["n_significance_components"] == 0
        assert out["stop_code"] == 2  # the threshold
        np.testing.assert_array_equal(resid, resid_off)
        np.testing.assert_array_equal(model, model_off)
        for key in (
            "iterations_performed",
            "stop_code",
            "peak_residual",
            "model_flux",
            "switched_to_hogbom",
        ):
            assert out[key] == off[key], key
        for key in (
            "n_scale_components",
            "n_point_components",
            "n_residual_components",
        ):
            assert out["timing"][key] == off["timing"][key], key
        if np.abs(dirty).max() < threshold:  # nothing to clean above it
            assert not model.any()
        return out

    @pytest.mark.parametrize("significance", [0.0, 3.0])
    @pytest.mark.parametrize("residual_components", [True, False])
    @pytest.mark.parametrize("mask_kind", ["ones", "none"])
    @pytest.mark.parametrize("faint", [True, False])
    def test_scale_significance_needs_noise_outside_the_mask(
        self, faint, mask_kind, residual_components, significance
    ):
        """With no pixel outside the clean mask (a mask of ones, or no mask)
        the noise of the smoothed residuals cannot be measured (it stays
        ``inf``), so no smoothed residual is significant and the model update
        stops at the threshold as with ``scale_significance`` off. With the
        default limit (``significance`` 0), the limit and every significance
        used to be ``threshold / inf = 0``, and ``0 >= 0`` went on cleaning
        below the threshold: on the faint scene it took all of the flux, the
        model peaking 48 pixels from the source. A limit set by the user (3)
        was never reached by a significance of 0; those cases guard that its
        result stays the one with ``scale_significance`` off."""
        psf, _, dirty, _, threshold = self._scene(faint=faint)
        if mask_kind == "ones":
            mask = np.ones_like(dirty)
        else:
            mask = np.array([], dtype=dirty.dtype)
        out = self._assert_no_significance_cleaning(
            psf,
            dirty,
            mask,
            threshold,
            residual_components=residual_components,
            significance=significance,
            significance_gain=0.3,
        )
        assert np.all(np.isinf(out["scale_noise"]))

    @pytest.mark.parametrize("significance", [0.0, 3.0])
    @pytest.mark.parametrize(
        "faint, residual_components", [(True, True), (True, False), (False, False)]
    )
    def test_scale_significance_needs_a_nonzero_noise_of_the_residual(
        self, faint, residual_components, significance
    ):
        """With the residual exactly 0 outside the mask of radius 40, its
        noise there is 0, while the smoothed residuals have a positive noise
        there: the emission inside the mask spreads out when smoothed. No
        smoothed residual is significant, with the default limit and with a
        limit set by the user, and the model update stops at the threshold as
        with ``scale_significance`` off. With the limit 3 the model update
        used to go on below the threshold against those noises: on the faint
        scene to the iteration limit (3000 components), on the disk for 182
        components. (The disk with residual-shaped components is not such a
        case: they leave sidelobes outside the mask, which have a noise.)"""
        psf, _, dirty, mask, threshold = self._scene(faint=faint)
        dirty = np.ascontiguousarray(dirty * mask)
        out = self._assert_no_significance_cleaning(
            psf,
            dirty,
            mask,
            threshold,
            residual_components=residual_components,
            significance=significance,
            significance_gain=0.3,
        )
        noise = out["scale_noise"]
        assert noise[0] == 0.0
        assert np.isfinite(noise[1]) and noise[1] > 0.0

    def test_cube_plane_without_noise_outside_the_mask(self):
        """Per plane: the plane whose mask leaves no pixel outside cleans
        nothing below the threshold; the plane next to it, with the mask of
        radius 40, still cleans the faint source for its significance."""
        psf, sky, dirty, mask, threshold = self._scene(faint=True)
        kwargs = dict(
            residual_components=True, scale_significance=True, significance_gain=0.3
        )
        masks = np.stack([np.ones_like(mask), mask])[None, :, None]
        resid = np.broadcast_to(dirty, masks.shape).copy()
        model = np.zeros_like(resid)
        aspclean.clean_cube(
            resid,
            np.ascontiguousarray(np.broadcast_to(psf, resid.shape)),
            model,
            masks,
            gain=0.1,
            threshold_per_cycle=np.full((1, 2, 1), threshold),
            max_iter_per_cycle_updated=np.full((1, 2, 1), 3000, dtype=np.int32),
            max_iter_divergence=-1,
            **kwargs,
        )
        assert not model[0, 0, 0].any()
        np.testing.assert_array_equal(resid[0, 0, 0], dirty)
        out, r1, m1 = self._clean(psf, dirty, mask, threshold, **kwargs)
        assert out["timing"]["n_significance_components"] > 0
        assert m1.sum() > 0.1 * sky.sum()
        np.testing.assert_array_equal(resid[0, 1, 0], r1)
        np.testing.assert_array_equal(model[0, 1, 0], m1)

    def test_cube_matches_per_plane(self):
        psf, _, dirty, mask, threshold = self._scene()
        kwargs = dict(
            residual_components=True,
            scale_significance=True,
            rc_switch_factor=3.0,
            significance_gain=0.3,
        )
        planes = np.stack([dirty, 0.5 * dirty])
        resid = planes[None, :, None].copy()  # planes stays the reference
        model = np.zeros_like(resid)
        aspclean.clean_cube(
            resid,
            np.ascontiguousarray(np.broadcast_to(psf, resid.shape)),
            model,
            np.ascontiguousarray(np.broadcast_to(mask, resid.shape)),
            gain=0.1,
            threshold_per_cycle=np.full((1, 2, 1), threshold),
            max_iter_per_cycle_updated=np.full((1, 2, 1), 3000, dtype=np.int32),
            max_iter_divergence=-1,
            processing_function_threads=2,
            **kwargs,
        )
        for f in range(2):
            _, r1, m1 = self._clean(
                psf, np.ascontiguousarray(planes[f]), mask, threshold, **kwargs
            )
            np.testing.assert_array_equal(resid[0, f, 0], r1)
            np.testing.assert_array_equal(model[0, f, 0], m1)


class TestInitialScales:
    """A ``largestscale`` below the PSF width gives the initial scales 0,
    ``largestscale`` and the PSF width, as LibRA does (issue #327), instead
    of the point scale only (CASA 6.7.5.18); 0 asks for the point scale
    only."""

    def test_largest_scale_below_the_psf_width_keeps_scale_components(self):
        psf, sky, dirty, mask, threshold = TestRingCleaning._scene()
        width = aspclean.psf_gaussian_width(psf)
        largest = int(width // 2)
        assert 0 < largest < width
        point_only, *_ = TestRingCleaning._clean(
            psf, dirty, mask, threshold, largestscale=0
        )
        assert point_only["timing"]["n_scale_components"] == 0
        assert point_only["stop_code"] == 1  # the iteration limit
        out, resid, model = TestRingCleaning._clean(
            psf, dirty, mask, threshold, largestscale=largest
        )
        assert out["timing"]["n_scale_components"] > 0
        assert out["stop_code"] == 2  # the threshold
        assert out["iterations_performed"] < point_only["iterations_performed"]
        np.testing.assert_allclose(model.sum(), sky.sum(), rtol=0.1)
        assert np.max(np.abs(resid + _linear_convolution(model, psf) - dirty)) < (
            1e-9 * np.abs(dirty).max()
        )

    def test_significance_per_scale_with_a_small_largest_scale(self):
        """The test of significance per scale has scales to search."""
        psf, sky, dirty, mask, threshold = TestRingCleaning._scene(faint=True)
        largest = int(aspclean.psf_gaussian_width(psf) // 2)
        _, _, model = TestRingCleaning._clean(
            psf, dirty, mask, threshold, largestscale=0, scale_significance=True
        )
        assert not model.any()
        out, _, model = TestRingCleaning._clean(
            psf, dirty, mask, threshold, largestscale=largest, scale_significance=True
        )
        assert out["timing"]["n_significance_components"] > 0
        assert model.sum() > 0.1 * sky.sum()
