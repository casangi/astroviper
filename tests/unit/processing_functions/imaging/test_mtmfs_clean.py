"""
Unit tests for the stateless MTMFS CLEAN functions
(``astroviper.processing_functions.imaging.deconvolvers.mtmfs``).

Casacore-free port of the model-update cycle behind CASA tclean's
``deconvolver='mtmfs'`` (``SDAlgorithmMSMFS`` / ``MultiTermMatrixCleaner``),
exposed as free functions (``clean``, ``hessian``, ``principal_solution``)
in the style of the hogbom module. Residual and model stacks are written
in place; nothing is retained between calls.

Organised as:
  * ports of ``tStdMTHessian.cc`` and ``tStdMTClean.cc`` on the same
    synthetic inputs, checked against NumPy references and the invariant
    ``residual_out = residual_in - sum_t2 psf[t1+t2] (*) delta_model[t2]``;
  * synthetic analogues of ``test_task_tclean.py`` mtmfs cases
    ``test_onefield_mtmfs``, ``test_onefield_mtmfs_nterms1`` and
    ``test_onefield_mtmfs_smallscalebias``;
  * statelessness, the zero-copy contract and input validation.
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


_PSF_AMP = np.array([1.0, 0.4, 0.5, 0.3, 0.45], dtype=np.float32)
_DIRTY_AMP = np.array([1.0, 0.3, 0.15], dtype=np.float32)


def make_psf(order, nx, ny, dtype=np.float32):
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
    mask = (1 << 64) - 1
    s = seed
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & mask
        out[i] = ((s >> 33) & 0x7FFFFF) / 4194304.0 - 1.0
    return out


def make_dirty(taylor, nx, ny, seed=5, dtype=np.float32):
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
    return np.real(np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b)))


def np_convolve_centered(a, b):
    ny, nx = a.shape
    return np.roll(np_convolve_origin(a, b), shift=(ny // 2, nx // 2), axis=(0, 1))


def np_hessian(psf_stack, nterms, scale_img):
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
    nterms = delta_model.shape[0]
    out = residual_in.astype(np.float64).copy()
    for t1 in range(nterms):
        for t2 in range(nterms):
            out[t1] -= np_convolve_centered(
                psf_stack[t1 + t2].astype(np.float64),
                delta_model[t2].astype(np.float64),
            )
    return out


def _tclean_like_psf(nterms, nx, ny, dtype=np.float32):
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
    nterms = true_model.shape[0]
    dirty = np.zeros_like(true_model, dtype=np.float64)
    for t1 in range(nterms):
        for t2 in range(nterms):
            dirty[t1] += np_convolve_centered(
                psf[t1 + t2].astype(np.float64), true_model[t2].astype(np.float64)
            )
    return np.ascontiguousarray(dirty, dtype=true_model.dtype)


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
        out = mtmfs.hessian(psf, scales=scales)
        assert out["nterms"] == nterms
        assert out["scales"] == scales
        for s, sc in enumerate(scales):
            ref = np_hessian(psf, nterms, np_make_scale(nx, ny, sc))
            np.testing.assert_allclose(out["hessian"][s], ref, rtol=2e-5, atol=1e-6)
            np.testing.assert_allclose(out["hessian"][s], out["hessian"][s].T)
            eig = np.linalg.eigvalsh(out["hessian"][s])
            assert np.all(eig > 0)
            ident = out["inverse_hessian"][s] @ out["hessian"][s]
            np.testing.assert_allclose(ident, np.eye(nterms), atol=2e-5)
            np.testing.assert_allclose(
                out["inverse_hessian"][s], np.linalg.inv(out["hessian"][s]), rtol=2e-5
            )

    def test_delta_scale_hessian_is_psf_peak_amplitudes(self):
        nterms, nx, ny = 2, 64, 64
        psf = make_psf_stack(nterms, nx, ny)
        H = mtmfs.hessian(psf, scales=[0.0])["hessian"][0]
        peak = np.unravel_index(np.argmax(np.abs(psf[0])), (ny, nx))
        for t1 in range(nterms):
            for t2 in range(nterms):
                assert H[t1, t2] == pytest.approx(float(psf[t1 + t2][peak]), rel=2e-5)

    def test_float64_matches_float32_hessian(self):
        nterms, nx, ny = 2, 64, 64
        h32 = mtmfs.hessian(
            make_psf_stack(nterms, nx, ny, np.float32), scales=[0.0, 6.0]
        )
        h64 = mtmfs.hessian(
            make_psf_stack(nterms, nx, ny, np.float64), scales=[0.0, 6.0]
        )
        np.testing.assert_allclose(h32["hessian"], h64["hessian"], rtol=1e-5)

    def test_singular_hessian_raises(self):
        nterms, nx, ny = 2, 32, 32
        psf = np.zeros((2 * nterms - 1, ny, nx), dtype=np.float32)
        with pytest.raises(RuntimeError, match="Non-invertible Hessian"):
            mtmfs.hessian(psf)

    def test_unsorted_scales_are_sorted(self):
        nx = ny = 128
        psf = make_psf_stack(1, nx, ny)
        a = mtmfs.hessian(psf, scales=[0.0, 10.0, 2.0], small_scale_bias=0.8)
        b = mtmfs.hessian(psf, scales=[0.0, 2.0, 10.0], small_scale_bias=0.8)
        assert a["scales"] == [0.0, 2.0, 10.0]
        assert a["psf_support"] == b["psf_support"]
        np.testing.assert_allclose(a["hessian"], b["hessian"])


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
        out = mtmfs.clean(
            residual,
            psf,
            model,
            scales=scales,
            niter=niter,
            gain=gain,
            threshold=1e-5,
            stop_fraction=0.1,
        )
        assert 0 < out["iterations_performed"] <= niter
        for t in range(nterms):
            assert np.any(model[t] != 0)
        ref = expected_residual(residual_in, psf, model)
        scale = np.max(np.abs(residual_in))
        np.testing.assert_allclose(residual, ref, atol=2e-5 * scale)
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
        o32 = mtmfs.clean(
            res32,
            make_psf_stack(nterms, nx, ny, np.float32),
            mod32,
            scales=[0.0, 6.0],
            niter=10,
            gain=0.1,
            threshold=1e-5,
            stop_fraction=0.1,
        )
        o64 = mtmfs.clean(
            res64,
            make_psf_stack(nterms, nx, ny, np.float64),
            mod64,
            scales=[0.0, 6.0],
            niter=10,
            gain=0.1,
            threshold=1e-5,
            stop_fraction=0.1,
        )
        assert o32["iterations_performed"] == o64["iterations_performed"]
        np.testing.assert_allclose(mod32, mod64, rtol=1e-3, atol=1e-5)
        np.testing.assert_allclose(res32, res64, rtol=1e-3, atol=1e-5)

    def test_threshold_stops_early(self):
        residual = make_dirty_stack(2, 64, 64)
        model = np.zeros_like(residual)
        out = mtmfs.clean(
            residual,
            make_psf_stack(2, 64, 64),
            model,
            niter=50,
            gain=0.1,
            threshold=100.0,
        )
        assert out["iterations_performed"] == 0
        assert out["converged"]
        assert out["stop_code"] == mtmfs.STOP_THRESHOLD
        assert np.all(model == 0)

    def test_converged_follows_engine_threshold_not_raw_peak(self):
        ny = nx = 48
        yy, xx = np.mgrid[0:ny, 0:nx]
        g = np.exp(-((yy - ny // 2) ** 2 + (xx - nx // 2) ** 2) / (2 * 2.5**2))
        g = g / g[ny // 2, nx // 2] * 4.0
        psf = np.ascontiguousarray(g[None], dtype=np.float32)
        residual = np.zeros((1, ny, nx), dtype=np.float32)
        residual[0, ny // 2, nx // 2] = 1.0
        model = np.zeros_like(residual)
        out = mtmfs.clean(residual, psf, model, niter=50, gain=0.1, threshold=0.5)
        assert out["iterations_performed"] == 0
        assert out["converged"]
        assert out["stop_code"] == mtmfs.STOP_THRESHOLD
        assert np.all(model == 0)

    def test_principal_solution_matches_numpy(self):
        nterms, nx, ny = 3, 48, 48
        inv = mtmfs.hessian(make_psf_stack(nterms, nx, ny), scales=[0.0, 4.0])[
            "inverse_hessian"
        ][0]
        rng = np.random.default_rng(3)
        residual = np.ascontiguousarray(
            rng.standard_normal((nterms, ny, nx)), dtype=np.float32
        )
        expected = np.einsum("ij,jyx->iyx", inv, residual.astype(np.float64))
        ptr_before = residual.__array_interface__["data"][0]
        mtmfs.principal_solution(residual, inv)
        assert residual.__array_interface__["data"][0] == ptr_before
        np.testing.assert_allclose(residual, expected, rtol=1e-4, atol=1e-5)


class TestTcleanAnalogues:
    def test_onefield_mtmfs(self):
        nterms, nx, ny = 2, 100, 100
        psf = _tclean_like_psf(nterms, nx, ny)
        true_model = np.zeros((nterms, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0
        true_model[1, ny // 2, nx // 2] = -0.5
        residual = _dirty_from_model(psf, true_model)
        residual_in = residual.copy()
        model = np.zeros_like(residual)
        out = mtmfs.clean(residual, psf, model, niter=10, gain=0.1)
        assert out["iterations_performed"] == 10
        assert out["peak_residual"] < np.max(np.abs(residual_in[0]))
        assert out["model_flux"] > 0
        assert np.any(model[1] != 0)
        assert model[0, ny // 2, nx // 2] == pytest.approx(1.0 - 0.9**10, rel=0.05)
        np.testing.assert_allclose(
            residual, expected_residual(residual_in, psf, model), atol=2e-5
        )

    def test_onefield_mtmfs_recovers_spectral_terms(self):
        nterms, nx, ny = 2, 100, 100
        psf = _tclean_like_psf(nterms, nx, ny)
        true_model = np.zeros((nterms, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0
        true_model[1, ny // 2, nx // 2] = -0.5
        residual = _dirty_from_model(psf, true_model)
        model = np.zeros_like(residual)
        total = 0
        for _cycle in range(10):
            out = mtmfs.clean(residual, psf, model, niter=50, gain=0.1, threshold=1e-4)
            total += out["iterations_performed"]
            if out["converged"]:
                break
        assert out["converged"]
        assert 50 < total < 200
        assert model[0, ny // 2, nx // 2] == pytest.approx(1.0, rel=2e-3)
        assert model[1, ny // 2, nx // 2] == pytest.approx(-0.5, rel=2e-3)
        assert np.max(np.abs(residual[0])) < 1e-3

    def test_onefield_mtmfs_nterms1(self):
        nterms, nx, ny = 1, 100, 100
        psf = _tclean_like_psf(nterms, nx, ny)
        assert psf.shape == (1, ny, nx)
        true_model = np.zeros((1, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0
        residual = _dirty_from_model(psf, true_model)
        residual_in = residual.copy()
        model = np.zeros_like(residual)
        out = mtmfs.clean(residual, psf, model, niter=10, gain=0.1)
        assert out["iterations_performed"] == 10
        assert out["peak_residual"] < residual_in[0].max()
        assert model[0, ny // 2, nx // 2] == pytest.approx(1.0 - 0.9**10, rel=1e-3)
        np.testing.assert_allclose(
            residual, expected_residual(residual_in, psf, model), atol=2e-5
        )

    def test_onefield_mtmfs_smallscalebias(self):
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
        out = mtmfs.clean(
            residual,
            psf,
            model,
            scales=[0, 20, 40, 100],
            small_scale_bias=0.9,
            niter=10,
            gain=0.1,
        )
        assert out["scales"] == [0.0, 20.0, 40.0, 100.0]
        assert out["small_scale_bias"] == pytest.approx(0.9)
        assert out["iterations_performed"] == 10
        assert out["peak_residual"] < residual_in[0].max()
        assert out["model_flux"] > 0
        assert np.count_nonzero(model[0]) > 100
        np.testing.assert_allclose(
            residual,
            expected_residual(residual_in, psf, model),
            atol=5e-5 * residual_in.max(),
        )

    @pytest.mark.parametrize("bias,expected", [(1.5, 1.0), (-3.0, -1.0), (0.4, 0.4)])
    def test_smallscalebias_clamped(self, bias, expected):
        out = mtmfs.hessian(
            make_psf_stack(1, 32, 32), scales=[0, 4], small_scale_bias=bias
        )
        assert out["small_scale_bias"] == pytest.approx(expected)

    def test_default_scales_is_point_source(self):
        out = mtmfs.hessian(make_psf_stack(2, 32, 32))
        assert out["scales"] == [0.0]

    def test_too_large_scales_are_dropped(self):
        out = mtmfs.hessian(make_psf_stack(1, 64, 64), scales=[0, 10, 60])
        assert out["scales"] == [0.0, 10.0]
        with pytest.raises(ValueError):
            mtmfs.hessian(make_psf_stack(1, 64, 64), scales=[60])

    def test_trailing_zero_is_sorted_not_divided(self):
        psf = make_psf_stack(1, 64, 64)
        residual = np.zeros((1, 64, 64), dtype=np.float32)
        residual[0, 32, 32] = 1.0
        model = np.zeros_like(residual)
        out = mtmfs.clean(
            residual,
            psf,
            model,
            scales=[5.0, 0.0],
            small_scale_bias=0.6,
            niter=5,
            gain=0.1,
        )
        assert out["scales"] == [0.0, 5.0]
        assert out["iterations_performed"] == 5
        assert np.max(np.abs(model)) > 0

    def test_mask_restricts_components(self):
        nterms, nx, ny = 1, 64, 64
        psf = _tclean_like_psf(nterms, nx, ny)
        true_model = np.zeros((1, ny, nx), dtype=np.float32)
        true_model[0, ny // 2, nx // 2] = 1.0
        true_model[0, 12, 12] = 0.3
        residual = _dirty_from_model(psf, true_model)
        model = np.zeros_like(residual)
        mask = np.zeros((ny, nx), dtype=np.float32)
        mask[6:19, 6:19] = 1.0
        out = mtmfs.clean(residual, psf, model, mask=mask, niter=20, gain=0.1)
        assert out["iterations_performed"] == 20
        assert np.all(model[0][mask == 0] == 0)
        assert model[0, 12, 12] > 0
        assert out["peak_residual"] == pytest.approx(np.max(np.abs(residual[0] * mask)))

    def test_empty_mask_places_nothing(self):
        ny = nx = 32
        psf = make_psf_stack(1, nx, ny)
        residual = np.ones((1, ny, nx), dtype=np.float32)
        model = np.zeros_like(residual)
        mask = np.zeros((ny, nx), dtype=np.float32)
        out = mtmfs.clean(residual, psf, model, mask=mask, niter=3, gain=0.5)
        assert out["iterations_performed"] == 0
        assert out["stop_code"] == mtmfs.STOP_NOTHING_TO_CLEAN
        assert out["converged"]
        assert np.all(model == 0)

    def test_mask_none_is_accepted(self):
        residual = make_dirty_stack(1, 32, 32)
        model = np.zeros_like(residual)
        out = mtmfs.clean(
            residual, make_psf_stack(1, 32, 32), model, mask=None, niter=3, gain=0.1
        )
        assert out["iterations_performed"] == 3
        assert np.any(model != 0)


class TestInPlaceContract:
    def test_residual_and_model_modified_in_place(self):
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        r_ptr = residual.__array_interface__["data"][0]
        m_ptr = model.__array_interface__["data"][0]
        r_before = residual.copy()
        mtmfs.clean(
            residual, make_psf_stack(2, 48, 40), model, scales=[0.0, 4.0], niter=5
        )
        assert residual.__array_interface__["data"][0] == r_ptr
        assert model.__array_interface__["data"][0] == m_ptr
        assert not np.array_equal(residual, r_before)
        assert np.any(model != 0)

    def test_psf_and_mask_not_modified(self):
        psf = make_psf_stack(2, 48, 40)
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        mask = np.ones(residual.shape[1:], dtype=residual.dtype)
        psf_before, mask_before = psf.copy(), mask.copy()
        mtmfs.clean(residual, psf, model, mask=mask, scales=[0.0, 4.0], niter=5)
        np.testing.assert_array_equal(psf, psf_before)
        np.testing.assert_array_equal(mask, mask_before)

    def test_model_accumulates_from_non_zero_initial(self):
        residual = make_dirty_stack(2, 48, 40)
        residual_in = residual.copy()
        model = np.full_like(residual, 0.25)
        init = model.copy()
        mtmfs.clean(
            residual, make_psf_stack(2, 48, 40), model, scales=[0.0, 4.0], niter=5
        )
        delta = model - init
        assert np.any(delta != 0)
        assert np.any(model == 0.25)
        ref = expected_residual(residual_in, make_psf_stack(2, 48, 40), delta)
        np.testing.assert_allclose(
            residual, ref, atol=2e-5 * np.max(np.abs(residual_in))
        )

    def test_repeated_calls_are_independent(self):
        psf = make_psf_stack(2, 48, 40)
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        o1 = mtmfs.clean(residual, psf, model, scales=[0.0, 4.0], niter=5, gain=0.1)
        o2 = mtmfs.clean(residual, psf, model, scales=[0.0, 4.0], niter=5, gain=0.1)
        assert o1["iterations_performed"] == 5 and o2["iterations_performed"] == 5
        assert o2["peak_residual"] < o1["peak_residual"]
        assert o2["model_flux"] > o1["model_flux"]

    def test_float64_path(self):
        residual = make_dirty_stack(2, 48, 40, dtype=np.float64)
        model = np.zeros_like(residual)
        out = mtmfs.clean(
            residual, make_psf_stack(2, 48, 40, np.float64), model, niter=5, gain=0.1
        )
        assert out["iterations_performed"] == 5
        assert np.any(model != 0)

    def test_return_dict_keys(self):
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        out = mtmfs.clean(residual, make_psf_stack(2, 48, 40), model, niter=3)
        assert {
            "iterations_performed",
            "peak_residual",
            "model_flux",
            "converged",
            "stop_code",
            "scales",
            "small_scale_bias",
            "psf_support",
            "hessian",
            "inverse_hessian",
        } <= set(out)
        assert isinstance(out["iterations_performed"], int)


class TestArrayValidation:
    def test_psf_wrong_shape_raises(self):
        residual = np.zeros((2, 40, 48), dtype=np.float32)
        model = np.zeros_like(residual)
        with pytest.raises(RuntimeError, match="shape"):
            mtmfs.clean(residual, make_psf_stack(1, 48, 40), model)
        with pytest.raises(RuntimeError, match="shape"):
            mtmfs.clean(residual, make_psf_stack(2, 40, 48), model)

    def test_psf_wrong_dtype_raises(self):
        residual = np.zeros((2, 40, 48), dtype=np.float32)
        model = np.zeros_like(residual)
        with pytest.raises(RuntimeError, match="dtype"):
            mtmfs.clean(residual, make_psf_stack(2, 48, 40, np.float64), model)

    def test_psf_non_contiguous_raises(self):
        residual = np.zeros((2, 40, 48), dtype=np.float32)
        model = np.zeros_like(residual)
        big = make_psf_stack(2, 96, 40)
        with pytest.raises(RuntimeError, match="C-contiguous"):
            mtmfs.clean(residual, big[:, :, ::2], model)

    def test_residual_dtype_mismatch_raises(self):
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        psf = make_psf_stack(2, 48, 40)
        with pytest.raises(RuntimeError, match="dtype"):
            mtmfs.clean(residual.astype(np.float64), psf, model)
        with pytest.raises(RuntimeError, match="dtype"):
            mtmfs.clean(residual, psf, model.astype(np.float64))

    def test_non_contiguous_residual_raises(self):
        residual = np.zeros((2, 40, 96), dtype=np.float32)
        model = np.zeros((2, 40, 48), dtype=np.float32)
        with pytest.raises(RuntimeError, match="C-contiguous"):
            mtmfs.clean(residual[:, :, ::2], make_psf_stack(2, 48, 40), model)

    def test_readonly_arrays_raise(self):
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        psf = make_psf_stack(2, 48, 40)
        ro = residual.copy()
        ro.setflags(write=False)
        with pytest.raises(RuntimeError, match="writeable"):
            mtmfs.clean(ro, psf, model)
        rom = model.copy()
        rom.setflags(write=False)
        with pytest.raises(RuntimeError, match="writeable"):
            mtmfs.clean(residual, psf, rom)
        inv = mtmfs.hessian(psf)["inverse_hessian"][0]
        with pytest.raises(RuntimeError, match="writeable"):
            mtmfs.principal_solution(ro, inv)

    def test_shape_mismatch_raises(self):
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        psf = make_psf_stack(2, 48, 40)
        with pytest.raises(RuntimeError, match="shape"):
            mtmfs.clean(residual[:1], psf, model)
        with pytest.raises(RuntimeError, match="shape"):
            mtmfs.clean(residual, psf, np.zeros((2, 48, 40), dtype=np.float32))
        with pytest.raises(RuntimeError, match="3-D"):
            mtmfs.clean(residual[0], psf[0], model[0])

    def test_mask_validation(self):
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        psf = make_psf_stack(2, 48, 40)
        with pytest.raises(RuntimeError, match="dtype"):
            mtmfs.clean(
                residual, psf, model, mask=np.ones(residual.shape[1:], dtype=bool)
            )
        with pytest.raises(RuntimeError, match="shape"):
            mtmfs.clean(residual, psf, model, mask=np.ones((40, 47), dtype=np.float32))

    def test_same_buffer_for_residual_and_model_raises(self):
        residual = make_dirty_stack(2, 48, 40)
        with pytest.raises(RuntimeError, match="distinct"):
            mtmfs.clean(residual, make_psf_stack(2, 48, 40), residual)

    def test_nterms_too_large_raises(self):
        residual = np.zeros((17, 8, 8), dtype=np.float32)
        model = np.zeros_like(residual)
        psf = np.zeros((33, 8, 8), dtype=np.float32)
        with pytest.raises(RuntimeError, match="too many Taylor terms"):
            mtmfs.clean(residual, psf, model)

    def test_no_copy_on_valid_input(self):
        residual = make_dirty_stack(2, 48, 40)
        model = np.zeros_like(residual)
        view_r = residual.view()
        view_m = model.view()
        mtmfs.clean(residual, make_psf_stack(2, 48, 40), model, niter=3, gain=0.1)
        np.testing.assert_array_equal(view_r, residual)
        assert np.shares_memory(view_m, model) and np.any(view_m != 0)
