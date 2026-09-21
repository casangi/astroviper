"""
Unit tests for the pybind11-wrapped Multi-Term Multi-Frequency Synthesis
(MTMFS) deconvolver
(``astroviper.processing_functions.imaging.deconvolvers.mtmfs``).

This is a casacore-free port of the minor cycle behind CASA tclean's
``deconvolver='mtmfs'`` (``SDAlgorithmMSMFS`` / ``MultiTermMatrixCleaner``).
The stateful ``MultiTermCleaner`` operates on Python-owned numpy buffers with
no copies on the C++ side: the ``residual`` and ``model`` Taylor-term stacks
are written into directly.

The tests are organised as:
  * ports of the CASA C++ equivalence tests ``tStdMTHessian.cc`` (Taylor
    Hessian) and ``tStdMTClean.cc`` (full ``mtclean()`` run) on the same
    synthetic inputs, verified against independent numpy references and the
    exact reconstruction invariant
    ``residual_out = residual_in - sum_t2 psf[t1+t2] (*) delta_model[t2]``;
  * synthetic minor-cycle analogues of the ``test_task_tclean.py`` mtmfs
    cases ``test_onefield_mtmfs``, ``test_onefield_mtmfs_nterms1`` and
    ``test_onefield_mtmfs_smallscalebias`` (the tclean major cycle and its
    MeasurementSets are not available here, so the *minor-cycle* semantics
    those tests exercise are checked: iteration count, residual decrease,
    model flux, Taylor-term recovery, parameter handling);
  * the zero-copy / in-place memory contract and input validation.
"""

import numpy as np
import pytest

try:
    from astroviper.processing_functions.imaging.deconvolvers import mtmfs

    MTMFS_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    MTMFS_AVAILABLE = False
    IMPORT_ERROR = e


pytestmark = pytest.mark.skipif(
    not MTMFS_AVAILABLE, reason="MTMFS extension not compiled/available"
)


# ---------------------------------------------------------------------------
# Helpers: synthetic inputs shared with the CASA ctests (tStdMTHessian.cc /
# tStdMTClean.cc) and numpy references
# ---------------------------------------------------------------------------

_PSF_AMP = np.array([1.0, 0.4, 0.5, 0.3, 0.45], dtype=np.float32)
_DIRTY_AMP = np.array([1.0, 0.3, 0.15], dtype=np.float32)


def make_psf(order, nx, ny, dtype=np.float32):
    """PSF Taylor term `order`: centred Gaussian, order-dependent amplitude
    (tStdMTHessian.cc / tStdMTClean.cc ``makePsf``)."""
    cx, cy = nx // 2, ny // 2
    sig = nx / 12.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = _PSF_AMP[order % 5]
    p = a * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig * sig))
    return np.ascontiguousarray(p, dtype=dtype)


def make_psf_stack(nterms, nx, ny, dtype=np.float32):
    return np.ascontiguousarray(
        np.stack([make_psf(o, nx, ny, dtype) for o in range(2 * nterms - 1)])
    )


def _lcg_noise(n, seed):
    """The 64-bit LCG used by tStdMTClean.cc ``makeDirty``, in storage order."""
    mask = (1 << 64) - 1
    s = seed
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & mask
        out[i] = ((s >> 33) & 0x7FFFFF) / 4194304.0 - 1.0
    return out


def make_dirty(taylor, nx, ny, seed=5, dtype=np.float32):
    """Per-Taylor-term residual: scaled point + blob + faint noise
    (tStdMTClean.cc ``makeDirty``)."""
    cx, cy = nx // 2, ny // 2
    blob_sig = nx / 6.0
    yy, xx = np.mgrid[0:ny, 0:nx]
    a = _DIRTY_AMP[taylor % 3]
    noise = _lcg_noise(nx * ny, seed + taylor * 101).reshape(ny, nx)
    d = a * (0.5 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * blob_sig**2)))
    d = d + 0.01 * noise
    d[cy, cx] += a * 1.0
    return np.ascontiguousarray(d, dtype=dtype)


def make_dirty_stack(nterms, nx, ny, seed=5, dtype=np.float32):
    return np.ascontiguousarray(
        np.stack([make_dirty(t, nx, ny, seed, dtype) for t in range(nterms)])
    )


def np_spheroidal(nu):
    """numpy port of MatrixCleaner::spheroidal (vectorised)."""
    nu = np.asarray(nu, dtype=np.float64)
    p = np.array(
        [
            [8.203343e-2, 4.028559e-3],
            [-3.644705e-1, -3.697768e-2],
            [6.278660e-1, 1.021332e-1],
            [-5.335581e-1, -1.201436e-1],
            [2.312756e-1, 6.412774e-2],
        ]
    )
    q = np.array([[1.0, 1.0], [8.212018e-1, 9.599102e-1], [2.078043e-1, 2.918724e-1]])
    part = (nu >= 0.75).astype(int)
    nuend = np.where(part == 0, 0.75, 1.0)
    delnusq = nu**2 - nuend**2
    top = p[0][part]
    for k in range(1, 5):
        top = top + p[k][part] * delnusq**k
    bot = q[0][part]
    for k in range(1, 3):
        bot = bot + q[k][part] * delnusq**k
    out = np.where(bot != 0, top / bot, 0.0)
    out = np.where(nu <= 0, 1.0, out)
    out = np.where(nu >= 1, 0.0, out)
    return out


def np_make_scale(nx, ny, scale_size):
    """numpy port of MatrixCleaner::makeScale (Cornwell tapered parabola)."""
    img = np.zeros((ny, nx), dtype=np.float64)
    refi, refj = nx // 2, ny // 2
    if scale_size == 0:
        img[refj, refi] = 1.0
        return img
    yy, xx = np.mgrid[0:ny, 0:nx]
    rad2 = ((refj - yy) / scale_size) ** 2 + ((refi - xx) / scale_size) ** 2
    inside = rad2 < 1.0
    rad = np.sqrt(np.where(inside, rad2, 0.0))
    vals = (1.0 - rad2) * np_spheroidal(rad)
    img[inside] = vals[inside]
    return img / img.sum()


def np_convolve_origin(a, b):
    """Circular convolution with the result origin at [0, 0] (fft0, no flip)."""
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b)))


def np_convolve_centered(a, b):
    """Circular convolution re-centred like ``flip_quadrants`` (casacore flip)."""
    ny, nx = a.shape
    return np.roll(np_convolve_origin(a, b), shift=(ny // 2, nx // 2), axis=(0, 1))


def np_hessian(psf_stack, nterms, scale_img):
    """Independent reference for one scale's Taylor Hessian:
    H(t1, t2) = [psf[t1+t2] (*) scale (*) scale] at the PSF peak."""
    ny, nx = psf_stack.shape[1:]
    peak = np.unravel_index(np.argmax(np.abs(psf_stack[0])), (ny, nx))
    sc2 = np_convolve_origin(scale_img, scale_img)
    h = np.zeros((nterms, nterms))
    for t1 in range(nterms):
        for t2 in range(nterms):
            h[t1, t2] = np_convolve_origin(psf_stack[t1 + t2].astype(np.float64), sc2)[
                peak
            ]
    return h


def expected_residual(residual_in, psf_stack, delta_model):
    """residual_in[t1] - sum_t2 psf[t1+t2] (*) delta_model[t2] (centred)."""
    nterms = delta_model.shape[0]
    out = residual_in.astype(np.float64).copy()
    for t1 in range(nterms):
        for t2 in range(nterms):
            out[t1] -= np_convolve_centered(
                psf_stack[t1 + t2].astype(np.float64),
                delta_model[t2].astype(np.float64),
            )
    return out


def _cleaner(nterms, nx, ny, scales=(0.0,), bias=0.0, dtype=np.float32):
    return mtmfs.MultiTermCleaner(
        nterms=nterms,
        scales=list(scales),
        shape=(ny, nx),
        small_scale_bias=bias,
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Port of tStdMTHessian.cc: Taylor Hessian and its inverse
# ---------------------------------------------------------------------------


class TestHessian:
    @pytest.mark.parametrize(
        "name,nx,ny,nterms,scales",
        [
            ("nt2", 64, 64, 2, [0.0, 6.0]),
            ("nt2b", 96, 96, 2, [0.0, 8.0, 16.0]),
            ("nt3", 64, 64, 3, [0.0, 6.0]),
        ],
    )
    def test_hessian_matches_numpy_reference(self, name, nx, ny, nterms, scales):
        psf = make_psf_stack(nterms, nx, ny)
        c = _cleaner(nterms, nx, ny, scales)
        c.set_psf(psf)
        assert c.has_psf
        for s, size in enumerate(scales):
            h = c.hessian(s)
            ref = np_hessian(psf, nterms, np_make_scale(nx, ny, size))
            np.testing.assert_allclose(h, ref, rtol=2e-5, atol=1e-7)
            # symmetric positive definite, inverse consistent
            np.testing.assert_allclose(h, h.T, rtol=1e-12)
            assert np.all(np.linalg.eigvalsh(h) > 0)
            np.testing.assert_allclose(
                c.inverse_hessian(s) @ h, np.eye(nterms), atol=1e-10
            )
            np.testing.assert_allclose(
                c.inverse_hessian(s), np.linalg.inv(h), rtol=1e-8
            )

    def test_delta_scale_hessian_is_psf_peak_amplitudes(self):
        """For the point (scale 0) scale the Hessian is just the PSF Taylor
        term peaks: H(t1, t2) = psf[t1+t2](peak) = amp[t1+t2]."""
        nterms, nx, ny = 3, 64, 64
        c = _cleaner(nterms, nx, ny)
        c.set_psf(make_psf_stack(nterms, nx, ny))
        expected = np.array(
            [[_PSF_AMP[t1 + t2] for t2 in range(nterms)] for t1 in range(nterms)]
        )
        np.testing.assert_allclose(c.hessian(0), expected, rtol=1e-6)

    def test_float64_matches_float32_hessian(self):
        nterms, nx, ny = 2, 64, 64
        c32 = _cleaner(nterms, nx, ny, [0.0, 6.0], dtype=np.float32)
        c64 = _cleaner(nterms, nx, ny, [0.0, 6.0], dtype=np.float64)
        c32.set_psf(make_psf_stack(nterms, nx, ny, np.float32))
        c64.set_psf(make_psf_stack(nterms, nx, ny, np.float64))
        for s in range(2):
            np.testing.assert_allclose(c32.hessian(s), c64.hessian(s), rtol=1e-5)

    def test_singular_hessian_raises(self):
        """Identical PSF Taylor terms give a rank-1 Hessian (the CASA
        "Non-invertible Hessian" error)."""
        nterms, nx, ny = 2, 32, 32
        p0 = make_psf(0, nx, ny)
        psf = np.ascontiguousarray(np.stack([p0, p0, p0]))
        c = _cleaner(nterms, nx, ny)
        with pytest.raises(RuntimeError, match="Non-invertible Hessian"):
            c.set_psf(psf)
        assert not c.has_psf

    def test_accessors_require_psf(self):
        c = _cleaner(2, 32, 32)
        with pytest.raises((RuntimeError, ValueError)):
            c.hessian(0)
        with pytest.raises(IndexError):
            c2 = _cleaner(2, 32, 32)
            c2.set_psf(make_psf_stack(2, 32, 32))
            c2.hessian(5)


# ---------------------------------------------------------------------------
# Port of tStdMTClean.cc: a full mtclean() run
# ---------------------------------------------------------------------------


class TestMtclean:
    @pytest.mark.parametrize(
        "name,nx,ny,nterms,scales,niter,gain",
        [
            ("nt2", 64, 64, 2, [0.0, 6.0], 15, 0.1),
            ("nt2b", 96, 96, 2, [0.0, 8.0], 12, 0.15),
        ],
    )
    def test_mtclean_reconstruction_invariant(
        self, name, nx, ny, nterms, scales, niter, gain
    ):
        psf = make_psf_stack(nterms, nx, ny)
        residual = make_dirty_stack(nterms, nx, ny)
        residual_in = residual.copy()
        model = np.zeros_like(residual)

        c = _cleaner(nterms, nx, ny, scales)
        c.set_psf(psf)
        out = c.clean(
            residual, model, niter=niter, gain=gain, threshold=1e-5, stop_fraction=0.1
        )

        assert 0 < out["iterations_performed"] <= niter
        # every Taylor term received components
        for t in range(nterms):
            assert np.any(model[t] != 0)
        # the residual is exactly the input minus the PSF-convolved new model
        ref = expected_residual(residual_in, psf, model)
        scale = np.max(np.abs(residual_in))
        np.testing.assert_allclose(residual, ref, atol=2e-5 * scale)
        # the point source at the centre dominates term 0 and gets reduced
        cy, cx = ny // 2, nx // 2
        assert abs(residual[0, cy, cx]) < abs(residual_in[0, cy, cx])
        assert out["peak_residual"] < np.max(np.abs(residual_in[0]))
        assert out["model_flux"] == pytest.approx(float(model[0].sum()), rel=1e-5)

    def test_float64_run_close_to_float32(self):
        nterms, nx, ny = 2, 64, 64
        res32 = make_dirty_stack(nterms, nx, ny, dtype=np.float32)
        res64 = res32.astype(np.float64)
        mod32 = np.zeros_like(res32)
        mod64 = np.zeros_like(res64)
        c32 = _cleaner(nterms, nx, ny, [0.0, 6.0], dtype=np.float32)
        c64 = _cleaner(nterms, nx, ny, [0.0, 6.0], dtype=np.float64)
        c32.set_psf(make_psf_stack(nterms, nx, ny, np.float32))
        c64.set_psf(make_psf_stack(nterms, nx, ny, np.float64))
        o32 = c32.clean(
            res32, mod32, niter=10, gain=0.1, threshold=1e-5, stop_fraction=0.1
        )
        o64 = c64.clean(
            res64, mod64, niter=10, gain=0.1, threshold=1e-5, stop_fraction=0.1
        )
        assert o32["iterations_performed"] == o64["iterations_performed"]
        np.testing.assert_allclose(mod32, mod64, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(res32, res64, rtol=1e-3, atol=1e-5)

    def test_threshold_stops_early(self):
        nterms, nx, ny = 2, 64, 64
        residual = make_dirty_stack(nterms, nx, ny)
        model = np.zeros_like(residual)
        c = _cleaner(nterms, nx, ny)
        c.set_psf(make_psf_stack(nterms, nx, ny))
        # threshold above the peak: nothing to do
        out = c.clean(residual, model, niter=50, gain=0.1, threshold=100.0)
        assert out["iterations_performed"] == 0
        assert out["converged"]
        assert np.all(model == 0)

    def test_principal_solution_matches_numpy(self):
        nterms, nx, ny = 3, 48, 48
        c = _cleaner(nterms, nx, ny, [0.0, 4.0])
        c.set_psf(make_psf_stack(nterms, nx, ny))
        rng = np.random.default_rng(3)
        residual = np.ascontiguousarray(
            rng.standard_normal((nterms, ny, nx)), dtype=np.float32
        )
        expected = np.einsum(
            "ij,jyx->iyx", c.inverse_hessian(0), residual.astype(np.float64)
        )
        ptr_before = residual.__array_interface__["data"][0]
        c.compute_principal_solution(residual)
        assert residual.__array_interface__["data"][0] == ptr_before
        np.testing.assert_allclose(residual, expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Synthetic analogues of test_task_tclean.py mtmfs cases
# ---------------------------------------------------------------------------


def _tclean_like_psf(nterms, nx, ny, dtype=np.float32):
    """PSF Taylor stack shaped like a real one: unit-peak tt0, a tiny tt1 (the
    tclean reference checks psf.tt1 ~ 1e-5 at the peak) and a positive tt2."""
    cx, cy = nx // 2, ny // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    core = np.exp(-r2 / (2.0 * 2.5**2))
    sidelobe = 0.15 * np.cos(np.sqrt(r2) / 2.0) * np.exp(-r2 / (2.0 * 12.0**2))
    tt0 = core + sidelobe
    tt0 /= tt0[cy, cx]
    amps = [1.0, 1e-3, 0.3, 5e-4, 0.15]
    stack = np.stack([tt0 * amps[o] for o in range(2 * nterms - 1)])
    return np.ascontiguousarray(stack, dtype=dtype)


def _dirty_from_model(psf, true_model):
    """dirty[t1] = sum_t2 psf[t1+t2] (*) true_model[t2] (the MTMFS forward model)."""
    nterms = true_model.shape[0]
    dirty = np.zeros_like(true_model, dtype=np.float64)
    for t1 in range(nterms):
        for t2 in range(nterms):
            dirty[t1] += np_convolve_centered(
                psf[t1 + t2].astype(np.float64), true_model[t2].astype(np.float64)
            )
    return np.ascontiguousarray(dirty, dtype=true_model.dtype)


class TestTcleanAnalogues:
    def test_onefield_mtmfs(self):
        """[onefield] Test_Onefield_mtmfs analogue: nterms=2, 100x100 image,
        niter=10 minor-cycle iterations (default gain 0.1, scales=[0])."""
        nterms, nx, ny = 2, 100, 100
        psf = _tclean_like_psf(nterms, nx, ny)
        true_model = np.zeros((nterms, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0  # I = 1 Jy
        true_model[1, ny // 2, nx // 2] = -0.5  # alpha * I
        residual = _dirty_from_model(psf, true_model)
        residual_in = residual.copy()
        model = np.zeros_like(residual)

        c = _cleaner(nterms, nx, ny)
        c.set_psf(psf)
        out = c.clean(residual, model, niter=10, gain=0.1)

        assert out["iterations_performed"] == 10
        assert out["peak_residual"] < np.max(np.abs(residual_in[0]))
        assert out["model_flux"] > 0
        assert np.any(model[1] != 0)  # model.tt1 produced
        # after 10 iterations at gain 0.1 the point source has 1-(0.9)^10 of its flux
        assert model[0, ny // 2, nx // 2] == pytest.approx(1.0 - 0.9**10, rel=0.05)
        np.testing.assert_allclose(
            residual, expected_residual(residual_in, psf, model), atol=2e-5
        )

    def test_onefield_mtmfs_recovers_spectral_terms(self):
        """Iterating residual-update / model-update cycles to convergence (the
        tclean major/minor loop; set_psf once, clean per cycle) recovers both
        Taylor terms of the point source (I and alpha*I), the quantities tclean
        derives alpha from. Within one minor cycle the convolved residual is
        only updated inside the psf_support patch, as in casacore, so the
        far sidelobes are removed by the end-of-cycle residual update."""
        nterms, nx, ny = 2, 100, 100
        psf = _tclean_like_psf(nterms, nx, ny)
        true_model = np.zeros((nterms, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0
        true_model[1, ny // 2, nx // 2] = -0.5
        residual = _dirty_from_model(psf, true_model)
        model = np.zeros_like(residual)
        c = _cleaner(nterms, nx, ny)
        c.set_psf(psf)
        total = 0
        for _cycle in range(10):
            out = c.clean(residual, model, niter=50, gain=0.1, threshold=1e-4)
            total += out["iterations_performed"]
            if out["converged"]:
                break
        assert out["converged"]
        assert 50 < total < 200  # ~ log(1e-4)/log(0.9) = 87 iterations
        assert model[0, ny // 2, nx // 2] == pytest.approx(1.0, rel=2e-3)
        assert model[1, ny // 2, nx // 2] == pytest.approx(-0.5, rel=2e-3)
        assert np.max(np.abs(residual[0])) < 1e-3

    def test_onefield_mtmfs_nterms1(self):
        """[onefield] Test_Onefield_mtmfs_nterms1 analogue (CAS-11364,
        CAS-11367): nterms=1 is a valid MTMFS configuration with a single PSF
        term and behaves as a (multi-scale) CLEAN."""
        nterms, nx, ny = 1, 100, 100
        psf = _tclean_like_psf(nterms, nx, ny)
        assert psf.shape == (1, ny, nx)
        true_model = np.zeros((1, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0
        residual = _dirty_from_model(psf, true_model)
        residual_in = residual.copy()
        model = np.zeros_like(residual)

        c = _cleaner(nterms, nx, ny)
        assert c.npsf_terms == 1
        c.set_psf(psf)
        out = c.clean(residual, model, niter=10, gain=0.1)
        assert out["iterations_performed"] == 10
        assert out["peak_residual"] < residual_in[0].max()
        assert model[0, ny // 2, nx // 2] == pytest.approx(1.0 - 0.9**10, rel=1e-3)
        np.testing.assert_allclose(
            residual, expected_residual(residual_in, psf, model), atol=2e-5
        )

    def test_onefield_mtmfs_smallscalebias(self):
        """[onefield] Test_Onefield_mtmfs_smallscalebias analogue: 200x200,
        nterms=1, scales=[0, 20, 40, 100], smallscalebias=0.9, niter=10 on an
        extended source."""
        nterms, nx, ny = 1, 200, 200
        psf = _tclean_like_psf(nterms, nx, ny)
        yy, xx = np.mgrid[0:ny, 0:nx]
        blob = 3.0 * np.exp(
            -((xx - nx // 2) ** 2 + (yy - ny // 2) ** 2) / (2.0 * 15.0**2)
        )
        true_model = np.ascontiguousarray(blob[None], dtype=np.float32)
        residual = _dirty_from_model(psf, true_model)
        residual_in = residual.copy()
        model = np.zeros_like(residual)

        c = _cleaner(nterms, nx, ny, scales=[0, 20, 40, 100], bias=0.9)
        assert c.scales == [0.0, 20.0, 40.0, 100.0]
        assert c.small_scale_bias == pytest.approx(0.9)
        c.set_psf(psf)
        out = c.clean(residual, model, niter=10, gain=0.1)
        assert out["iterations_performed"] == 10
        assert out["peak_residual"] < residual_in[0].max()
        assert out["model_flux"] > 0
        # an extended source cleaned with large scales spreads flux beyond a pixel
        assert np.count_nonzero(model[0]) > 100
        np.testing.assert_allclose(
            residual,
            expected_residual(residual_in, psf, model),
            atol=5e-5 * residual_in.max(),
        )

    @pytest.mark.parametrize("bias,expected", [(1.5, 1.0), (-3.0, -1.0), (0.4, 0.4)])
    def test_smallscalebias_clamped(self, bias, expected):
        """SDAlgorithmMSMFS: acceptable smallscalebias values are [-1, 1];
        out-of-range values are changed to the nearest bound."""
        c = _cleaner(1, 32, 32, scales=[0, 4], bias=bias)
        assert c.small_scale_bias == pytest.approx(expected)

    def test_default_scales_is_point_source(self):
        """SDAlgorithmMSMFS: an empty scale list becomes [0.0]."""
        c = mtmfs.MultiTermCleaner(nterms=2, shape=(32, 32))
        assert c.scales == [0.0]
        assert c.nscales == 1

    def test_too_large_scales_are_dropped(self):
        """MultiTermMatrixCleaner::verifyScaleSizes: scales larger than half
        the image are ignored."""
        c = _cleaner(1, 64, 64, scales=[0, 10, 60])
        assert c.scales == [0.0, 10.0]
        with pytest.raises(ValueError):
            _cleaner(1, 64, 64, scales=[60])

    def test_invalid_constructor_arguments(self):
        with pytest.raises(ValueError):
            _cleaner(0, 32, 32)
        with pytest.raises(ValueError):
            _cleaner(1, 32, 32, scales=[-1.0])
        with pytest.raises(RuntimeError):
            mtmfs.MultiTermCleaner(nterms=1, shape=(32, 32), dtype=np.int32)

    def test_mask_restricts_components(self):
        """Components are only placed where the (scale-convolved) mask is set."""
        nterms, nx, ny = 1, 64, 64
        psf = _tclean_like_psf(nterms, nx, ny)
        true_model = np.zeros((1, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0  # bright, outside the mask
        true_model[0, 12, 12] = 0.3  # faint, inside the mask
        residual = _dirty_from_model(psf, true_model)
        model = np.zeros_like(residual)
        mask = np.zeros((ny, nx), dtype=np.float32)
        mask[6:19, 6:19] = 1.0

        c = _cleaner(nterms, nx, ny)
        c.set_psf(psf)
        out = c.clean(residual, model, mask=mask, niter=20, gain=0.1)
        assert out["iterations_performed"] == 20
        assert np.all(model[0][mask == 0] == 0)
        assert model[0, 12, 12] > 0
        # the reported peak is the masked one
        assert out["peak_residual"] == pytest.approx(np.max(np.abs(residual[0] * mask)))


# ---------------------------------------------------------------------------
# Memory contract: in place, zero copy, and input validation
# ---------------------------------------------------------------------------


def _setup(nterms=2, nx=48, ny=40, dtype=np.float32):
    psf = make_psf_stack(nterms, nx, ny, dtype)
    residual = make_dirty_stack(nterms, nx, ny, dtype=dtype)
    model = np.zeros_like(residual)
    c = _cleaner(nterms, nx, ny, [0.0, 4.0], dtype=dtype)
    c.set_psf(psf)
    return c, psf, residual, model


class TestInPlaceContract:
    def test_residual_and_model_modified_in_place(self):
        c, psf, residual, model = _setup()
        r_ptr = residual.__array_interface__["data"][0]
        m_ptr = model.__array_interface__["data"][0]
        r_before = residual.copy()
        c.clean(residual, model, niter=5, gain=0.1)
        assert residual.__array_interface__["data"][0] == r_ptr
        assert model.__array_interface__["data"][0] == m_ptr
        assert not np.array_equal(residual, r_before)
        assert np.any(model != 0)

    def test_psf_and_mask_not_modified(self):
        c, psf, residual, model = _setup()
        psf_before = psf.copy()
        mask = np.ones(residual.shape[1:], dtype=residual.dtype)
        mask_before = mask.copy()
        c.clean(residual, model, mask=mask, niter=5, gain=0.1)
        np.testing.assert_array_equal(psf, psf_before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_model_accumulates_from_non_zero_initial(self):
        c, psf, residual, model = _setup()
        model[:] = 0.25
        init = model.copy()
        c.clean(residual, model, niter=5, gain=0.1)
        delta = model - init
        assert np.any(delta != 0)
        # only added components: untouched pixels keep the initial value
        assert np.any(model == 0.25)

    def test_repeated_cycles_reuse_setup_and_keep_improving(self):
        """SDAlgorithmMSMFS lifecycle: set_psf once, clean() every residual
        update cycle; the model accumulates and the residual keeps dropping."""
        c, psf, residual, model = _setup()
        peak0 = np.max(np.abs(residual[0]))
        o1 = c.clean(residual, model, niter=5, gain=0.1)
        flux1 = o1["model_flux"]
        peak1 = o1["peak_residual"]
        o2 = c.clean(residual, model, niter=5, gain=0.1)
        assert o1["iterations_performed"] == 5 and o2["iterations_performed"] == 5
        assert peak1 < peak0
        assert o2["peak_residual"] < peak1
        assert o2["model_flux"] > flux1
        assert o2["model_flux"] == pytest.approx(float(model[0].sum()), rel=1e-5)

    def test_set_psf_can_be_repeated(self):
        c, psf, residual, model = _setup()
        c.set_psf(psf)  # new residual-update cycle with a refreshed PSF
        out = c.clean(residual, model, niter=3, gain=0.1)
        assert out["iterations_performed"] == 3

    def test_float64_path(self):
        c, psf, residual, model = _setup(dtype=np.float64)
        assert c.dtype == np.dtype(np.float64)
        out = c.clean(residual, model, niter=5, gain=0.1)
        assert out["iterations_performed"] == 5
        assert np.any(model != 0)

    def test_return_dict_keys(self):
        c, psf, residual, model = _setup()
        out = c.clean(residual, model, niter=3, gain=0.1)
        assert set(out) == {
            "iterations_performed",
            "peak_residual",
            "model_flux",
            "converged",
        }
        assert isinstance(out["iterations_performed"], int)
        assert out["converged"] == (out["peak_residual"] <= 0.0)

    def test_properties(self):
        c, psf, residual, model = _setup(nterms=2, nx=48, ny=40)
        assert c.nterms == 2
        assert c.npsf_terms == 3
        assert c.nscales == 2
        assert c.shape == (40, 48)
        assert c.has_psf
        assert c.psf_support > 0
        assert c.dtype == np.dtype(np.float32)


class TestArrayValidation:
    def test_clean_requires_psf(self):
        c = _cleaner(2, 48, 40)
        residual = np.zeros((2, 40, 48), dtype=np.float32)
        with pytest.raises(RuntimeError, match="set_psf"):
            c.clean(residual, residual.copy())
        with pytest.raises(RuntimeError, match="set_psf"):
            c.compute_principal_solution(residual)

    def test_psf_wrong_shape_raises(self):
        c = _cleaner(2, 48, 40)
        with pytest.raises(RuntimeError, match="shape"):
            c.set_psf(make_psf_stack(1, 48, 40))  # 1 term instead of 3
        with pytest.raises(RuntimeError, match="shape"):
            c.set_psf(make_psf_stack(2, 40, 48))  # transposed image

    def test_psf_wrong_dtype_raises(self):
        c = _cleaner(2, 48, 40)
        with pytest.raises(RuntimeError, match="dtype"):
            c.set_psf(make_psf_stack(2, 48, 40, np.float64))

    def test_psf_non_contiguous_raises(self):
        c = _cleaner(2, 48, 40)
        big = make_psf_stack(2, 96, 40)
        with pytest.raises(RuntimeError, match="C-contiguous"):
            c.set_psf(big[:, :, ::2])

    def test_psf_wrong_ndim_raises(self):
        c = _cleaner(1, 48, 40)
        with pytest.raises(RuntimeError, match="3-D"):
            c.set_psf(make_psf(0, 48, 40))

    def test_residual_dtype_mismatch_raises(self):
        c, psf, residual, model = _setup()
        with pytest.raises(RuntimeError, match="dtype"):
            c.clean(residual.astype(np.float64), model)
        with pytest.raises(RuntimeError, match="dtype"):
            c.clean(residual, model.astype(np.float64))

    def test_non_contiguous_residual_raises(self):
        c, psf, residual, model = _setup()
        wide = np.zeros((2, 40, 96), dtype=np.float32)
        with pytest.raises(RuntimeError, match="C-contiguous"):
            c.clean(wide[:, :, ::2], model)

    def test_readonly_arrays_raise(self):
        c, psf, residual, model = _setup()
        ro = residual.copy()
        ro.setflags(write=False)
        with pytest.raises(RuntimeError, match="writeable"):
            c.clean(ro, model)
        rom = model.copy()
        rom.setflags(write=False)
        with pytest.raises(RuntimeError, match="writeable"):
            c.clean(residual, rom)
        with pytest.raises(RuntimeError, match="writeable"):
            c.compute_principal_solution(ro)

    def test_shape_mismatch_raises(self):
        c, psf, residual, model = _setup()
        with pytest.raises(RuntimeError, match="shape"):
            c.clean(residual[:1], model)  # wrong number of terms
        with pytest.raises(RuntimeError, match="shape"):
            c.clean(residual, np.zeros((2, 48, 40), dtype=np.float32))
        with pytest.raises(RuntimeError, match="3-D"):
            c.clean(residual[0], model[0])

    def test_mask_validation(self):
        c, psf, residual, model = _setup()
        with pytest.raises(RuntimeError, match="dtype"):
            c.clean(residual, model, mask=np.ones(residual.shape[1:], dtype=bool))
        with pytest.raises(RuntimeError, match="shape"):
            c.clean(residual, model, mask=np.ones((40, 47), dtype=np.float32))

    def test_same_buffer_for_residual_and_model_raises(self):
        c, psf, residual, model = _setup()
        with pytest.raises(RuntimeError, match="distinct"):
            c.clean(residual, residual)

    def test_no_copy_on_valid_input(self):
        """A valid call never triggers a conversion: the arrays handed in are
        the ones written (checked through a view sharing the buffer)."""
        c, psf, residual, model = _setup()
        view_r = residual.view()
        view_m = model.view()
        c.clean(residual, model, niter=3, gain=0.1)
        np.testing.assert_array_equal(view_r, residual)
        assert np.shares_memory(view_m, model) and np.any(view_m != 0)
