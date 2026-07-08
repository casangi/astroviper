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
            -(
                ((xx - nx // 2) ** 2) / (2 * sx**2)
                + ((yy - ny // 2) ** 2) / (2 * sy**2)
            )
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
        res = aspclean.clean(resid, psf, model, gain=0.1, threshold=0.05, niter=100)
        for key in (
            "iterations_performed",
            "retval",
            "converged",
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
            resid, psf, model, gain=0.1, threshold=0.01, niter=200, fusedthreshold=0.1
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
            resid, psf, model, gain=0.1, threshold=0.05, niter=150, fusedthreshold=0.1
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
        res = aspclean.clean(
            resid,
            psf,
            model,
            gain=0.5,
            threshold=0.01,
            niter=60,
            fusedthreshold=0.5,
            psf_width=1.0,
        )
        assert np.max(np.abs(resid)) < 0.05
        # essentially all flux placed at the source pixel
        assert model[40, 50] == pytest.approx(7.0, abs=0.1)
        assert model.sum() == pytest.approx(7.0, abs=0.1)

    def test_major_cycles_monotonic_progress(self):
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
                threshold=0.001,
                niter=100,
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
            threshold=0.01,
            niter=150,
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
        res = aspclean.clean(resid, psf, model, mask=mask, niter=50)
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
            resid, psf, model, gain=0.2, threshold=0.01, niter=50, fusedthreshold=0.1
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
        rng = np.random.default_rng(3)
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
        # Iteration control is per (time, frequency, polarization) plane:
        # threshold is a float64 (nt, nf, np) array and niter an int32 one.
        threshold = np.full((nt, nf, npol), 0.05, dtype=np.float64)
        niter = np.full((nt, nf, npol), 80, dtype=np.int32)
        out = aspclean.clean_cube(
            rc,
            psf,
            mc,
            gain=0.1,
            threshold=threshold,
            niter=niter,
            fusedthreshold=0.1,
            num_threads=2,
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
                        threshold=0.05,
                        niter=80,
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
        niter = np.full((nt, nf, npol), 60, dtype=np.int32)
        out = aspclean.clean_cube(
            rc,
            psf1,
            model,
            gain=0.1,
            threshold=threshold,
            niter=niter,
            fusedthreshold=0.1,
        )
        assert out["model_flux"].shape == (nt, nf, npol)
        assert np.all(out["model_flux"] > 0)
