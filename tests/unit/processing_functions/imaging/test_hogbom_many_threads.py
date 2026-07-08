"""Unit tests for the many-threads Hogbom deconvolver.

Verifies that ``hogbom_clean_many_threads`` (C++ ``clean_cube_many_threads``,
threaded across + within planes) produces the same result as the C++
``hogbom_clean`` (``clean_cube``) on tie-free data.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

try:
    from astroviper.processing_functions.imaging.deconvolvers import (  # noqa: F401
        hogbom,
    )

    HOGBOM_AVAILABLE = hasattr(hogbom, "clean_cube_many_threads")
except ImportError:  # pragma: no cover
    HOGBOM_AVAILABLE = False

from astroviper.processing_functions.imaging.deconvolution import (
    hogbom_clean,
    hogbom_clean_many_threads,
)

requires_both = pytest.mark.skipif(
    not HOGBOM_AVAILABLE,
    reason="needs the C++ hogbom extension with clean_cube_many_threads",
)


def _data(n=128, npol=2, seed=0):
    rng = np.random.default_rng(seed)
    resid = rng.standard_normal((1, 1, npol, n, n)).astype(np.float32)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    psf2d = np.exp(-((yy - n // 2) ** 2 + (xx - n // 2) ** 2) / (2 * (n / 16) ** 2))
    psf = np.broadcast_to(psf2d.astype(np.float32), (1, 1, npol, n, n)).copy()
    mask = np.ones((1, 1, npol, n, n), dtype=bool)
    return resid, psf, mask


def _params(npol, niter=30, thr=-1.0):
    return {
        "gain": 0.1,
        "niter": niter,
        "threshold": 0.0,
        "clean_box": (-1, -1, -1, -1),
        "minpsffraction": 0.05,
        "maxpsffraction": 0.8,
        "niter_per_plane": np.full((1, 1, npol), niter, np.int32),
        "cyclethreshold_per_plane": np.full((1, 1, npol), thr, np.float32),
    }


@requires_both
class TestHogbomManyThreads:
    @pytest.mark.parametrize("npol", [1, 2])
    def test_matches_cpp_iteration_limited(self, npol):
        resid, psf, mask = _data(npol=npol, seed=1)
        p = _params(npol, niter=30, thr=-1.0)  # threshold never triggers -> niter iters
        r1, m1 = resid.copy(), np.zeros_like(resid)
        r2, m2 = resid.copy(), np.zeros_like(resid)
        o1 = hogbom_clean(r1, psf, m1, copy.deepcopy(p), 8, mask)
        o2 = hogbom_clean_many_threads(r2, psf, m2, copy.deepcopy(p), 8, mask)
        assert np.array_equal(o1["iterations_performed"], o2["iterations_performed"])
        assert np.array_equal(m1, m2)
        assert np.array_equal(r1, r2)

    def test_matches_cpp_threshold_limited(self):
        # A real (positive) threshold -> planes converge before niter.
        resid, psf, mask = _data(npol=2, seed=2)
        p = _params(2, niter=200, thr=2.0)
        r1, m1 = resid.copy(), np.zeros_like(resid)
        r2, m2 = resid.copy(), np.zeros_like(resid)
        o1 = hogbom_clean(r1, psf, m1, copy.deepcopy(p), 8, mask)
        o2 = hogbom_clean_many_threads(r2, psf, m2, copy.deepcopy(p), 8, mask)
        assert np.array_equal(o1["iterations_performed"], o2["iterations_performed"])
        np.testing.assert_allclose(m1, m2, atol=1e-6)
        np.testing.assert_allclose(r1, r2, atol=1e-6)

    def test_double_precision(self):
        resid, psf, mask = _data(npol=2, seed=3)
        resid = resid.astype(np.float64)
        psf = psf.astype(np.float64)
        p = _params(2, niter=20, thr=-1.0)
        p["cyclethreshold_per_plane"] = p["cyclethreshold_per_plane"].astype(np.float64)
        r2, m2 = resid.copy(), np.zeros_like(resid)
        o2 = hogbom_clean_many_threads(r2, psf, m2, copy.deepcopy(p), 8, mask)
        assert m2.dtype == np.float64
        assert o2["iterations_performed"].tolist() == [[[20, 20]]]

    def test_single_pol_psf_broadcast(self):
        resid, _, mask = _data(npol=2, seed=4)
        n = resid.shape[-1]
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        psf1 = np.exp(-((yy - n // 2) ** 2 + (xx - n // 2) ** 2) / (2 * (n / 16) ** 2))
        psf = psf1.astype(np.float32)[None, None, None]  # np_psf == 1
        p = _params(2, niter=15, thr=-1.0)
        r2, m2 = resid.copy(), np.zeros_like(resid)
        o2 = hogbom_clean_many_threads(r2, psf, m2, copy.deepcopy(p), 8, mask)
        assert o2["iterations_performed"].shape == (1, 1, 2)
