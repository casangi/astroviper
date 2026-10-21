"""
Unit tests for the pybind11-wrapped Hogbom CLEAN routine
(``astroviper.processing_functions.imaging.deconvolvers.hogbom.clean``).

The wrapper operates on Python-owned numpy buffers with no copies on the
C++ side: the ``dirty_image`` and ``model`` arrays are written into
directly. These tests exercise the correctness of the algorithm on small
hand-constructed inputs as well as the zero-copy / in-place contract of
the binding.
"""

import numpy as np
import pytest

try:
    from astroviper.processing_functions.imaging.deconvolvers import hogbom

    HOGBOM_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    HOGBOM_AVAILABLE = False
    IMPORT_ERROR = e


pytestmark = pytest.mark.skipif(
    not HOGBOM_AVAILABLE, reason="Hogbom extension not compiled/available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _delta_psf(ny, nx, dtype=np.float32):
    """A PSF that is a unit delta at the centre pixel."""
    psf = np.zeros((ny, nx), dtype=dtype)
    psf[ny // 2, nx // 2] = 1.0
    return psf


def _zeros(ny, nx, dtype=np.float32):
    return np.zeros((ny, nx), dtype=dtype)


def _point_source_residual(ny, nx, sources, dtype=np.float32):
    """Build a residual image with isolated point sources.

    Parameters
    ----------
    sources : list of (iy, ix, amp)
    """
    resid = np.zeros((ny, nx), dtype=dtype)
    for iy, ix, amp in sources:
        resid[iy, ix] = amp
    return resid


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


class TestBasicCorrectness:
    def test_single_point_source_fully_cleaned(self):
        """With a delta PSF and gain=1.0, a single point source must be
        removed from the residual and placed into the model in one
        iteration."""
        ny, nx = 32, 32
        src_y, src_x, amp = 10, 20, 5.0

        dirty = _point_source_residual(ny, nx, [(src_y, src_x, amp)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=10,
        )

        assert result["iterations_performed"] >= 1
        assert np.allclose(dirty, 0.0, atol=1e-6)
        assert model[src_y, src_x] == pytest.approx(amp, abs=1e-6)
        assert np.sum(np.abs(model)) == pytest.approx(amp, abs=1e-6)
        assert result["final_peak"] == pytest.approx(0.0, abs=1e-6)
        assert result["converged"] is True

    def test_multiple_point_sources(self):
        ny, nx = 32, 32
        sources = [(8, 12, 3.0), (20, 18, -2.0), (5, 27, 1.5)]

        dirty = _point_source_residual(ny, nx, sources)
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=50,
        )

        for iy, ix, amp in sources:
            assert model[iy, ix] == pytest.approx(amp, abs=1e-5)
        assert np.allclose(dirty, 0.0, atol=1e-5)

    def test_partial_clean_with_small_gain(self):
        """With gain < 1 the residual should be reduced but not zero after
        a single iteration; ``iter_done`` should equal ``max_iter`` when
        threshold is not reached."""
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=0.1,
            threshold_per_cycle=0.0,
            max_iter_per_cycle_updated=1,
        )

        assert result["iterations_performed"] == 1
        # Residual at the source position should be reduced by gain.
        assert dirty[8, 8] == pytest.approx(0.9, abs=1e-6)
        assert model[8, 8] == pytest.approx(0.1, abs=1e-6)


# ---------------------------------------------------------------------------
# In-place / zero-copy contract
# ---------------------------------------------------------------------------


class TestInPlaceContract:
    def test_residual_buffer_is_modified_in_place(self):
        """The exact Python-owned buffer must be modified; no hidden copy."""
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 2.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        # Take a view on the same memory and snapshot the pointer.
        dirty_view = dirty[...]
        ptr_before = dirty.__array_interface__["data"][0]

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=5,
        )

        assert dirty.__array_interface__["data"][0] == ptr_before
        # The view onto the same memory must see the same modifications.
        assert np.array_equal(dirty, dirty_view)
        assert np.allclose(dirty, 0.0, atol=1e-6)

    def test_model_buffer_is_modified_in_place(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 2.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        model_view = model[...]
        ptr_before = model.__array_interface__["data"][0]

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=5,
        )

        assert model.__array_interface__["data"][0] == ptr_before
        assert np.array_equal(model, model_view)
        assert model[8, 8] == pytest.approx(2.0, abs=1e-6)

    def test_model_accumulates_from_non_zero_initial(self):
        """An existing model passed in must be added to, not overwritten."""
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)

        model = np.zeros((ny, nx), dtype=np.float32)
        model[3, 3] = 7.0  # prior component, unrelated to current residual

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=5,
        )

        # Prior value is preserved, new component added.
        assert model[3, 3] == pytest.approx(7.0, abs=1e-6)
        assert model[8, 8] == pytest.approx(1.0, abs=1e-6)

    def test_psf_is_not_modified(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        psf_before = psf.copy()
        model = _zeros(ny, nx)

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=5,
        )

        assert np.array_equal(psf, psf_before)


# ---------------------------------------------------------------------------
# Dtype handling
# ---------------------------------------------------------------------------


class TestDtypeHandling:
    def test_float64(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 3.0)], dtype=np.float64)
        psf = _delta_psf(ny, nx, dtype=np.float64)
        model = _zeros(ny, nx, dtype=np.float64)

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=5,
        )

        assert dirty.dtype == np.float64
        assert model.dtype == np.float64
        assert model[8, 8] == pytest.approx(3.0, abs=1e-12)

    def test_dtype_mismatch_between_arrays_raises(self):
        ny, nx = 16, 16
        dirty = _zeros(ny, nx, dtype=np.float32)
        psf = _delta_psf(ny, nx, dtype=np.float64)  # mismatched
        model = _zeros(ny, nx, dtype=np.float32)

        with pytest.raises(RuntimeError, match="psf has wrong dtype"):
            hogbom.clean(dirty_image=dirty, psf=psf, model=model)

    def test_unsupported_dtype_raises(self):
        ny, nx = 16, 16
        dirty = _zeros(ny, nx).astype(np.int32)
        psf = _delta_psf(ny, nx).astype(np.int32)
        model = _zeros(ny, nx).astype(np.int32)

        with pytest.raises(RuntimeError, match="float32 or float64"):
            hogbom.clean(dirty_image=dirty, psf=psf, model=model)


# ---------------------------------------------------------------------------
# Array layout / writability validation
# ---------------------------------------------------------------------------


class TestArrayValidation:
    def test_non_contiguous_dirty_raises(self):
        ny, nx = 16, 16
        big = _zeros(2 * ny, 2 * nx)
        dirty = big[::2, ::2]  # strided view, not C-contiguous
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        assert not dirty.flags["C_CONTIGUOUS"]
        with pytest.raises(RuntimeError, match="C-contiguous"):
            hogbom.clean(dirty_image=dirty, psf=psf, model=model)

    def test_readonly_dirty_raises(self):
        ny, nx = 16, 16
        dirty = _zeros(ny, nx)
        dirty.setflags(write=False)
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        with pytest.raises(RuntimeError, match="writeable"):
            hogbom.clean(dirty_image=dirty, psf=psf, model=model)

    def test_readonly_model_raises(self):
        ny, nx = 16, 16
        dirty = _zeros(ny, nx)
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)
        model.setflags(write=False)

        with pytest.raises(RuntimeError, match="writeable"):
            hogbom.clean(dirty_image=dirty, psf=psf, model=model)

    def test_shape_mismatch_raises(self):
        ny, nx = 16, 16
        dirty = _zeros(ny, nx)
        psf = _delta_psf(ny + 2, nx)  # wrong shape
        model = _zeros(ny, nx)

        with pytest.raises(RuntimeError, match="psf has wrong shape"):
            hogbom.clean(dirty_image=dirty, psf=psf, model=model)

    def test_wrong_ndim_raises(self):
        dirty = np.zeros(16, dtype=np.float32)  # 1D
        psf = _delta_psf(16, 16)
        model = _zeros(16, 16)

        with pytest.raises(RuntimeError, match="dirty_image must be 2D"):
            hogbom.clean(dirty_image=dirty, psf=psf, model=model)


# ---------------------------------------------------------------------------
# Iteration controls
# ---------------------------------------------------------------------------


class TestIterationControls:
    def test_max_iter_limits_iterations(self):
        """With threshold=0 the only stopping condition is ``max_iter``."""
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=0.05,
            threshold_per_cycle=0.0,
            max_iter_per_cycle_updated=7,
        )

        assert result["iterations_performed"] == 7

    def test_threshold_stops_early(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        # With gain=1.0 the source is cleared in one iteration; threshold
        # of 0.01 triggers the break on the next iteration.
        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.01,
            max_iter_per_cycle_updated=100,
        )

        assert result["iterations_performed"] <= 5
        assert result["converged"] is True

    def test_progress_callback_invoked(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        calls = []

        def cb(it, px, py, peak):
            calls.append((it, px, py, float(peak)))

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=0.1,
            threshold_per_cycle=0.0,
            max_iter_per_cycle_updated=20,
            progress_callback=cb,
        )

        assert len(calls) > 0
        # First callback should report the initial peak.
        first_peak = calls[0][3]
        assert first_peak == pytest.approx(1.0, abs=1e-6)

    def test_stop_callback_halts_early(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        call_count = {"n": 0}

        def stop():
            call_count["n"] += 1
            return call_count["n"] >= 2  # request stop on second check

        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=0.01,
            threshold_per_cycle=0.0,
            max_iter_per_cycle_updated=1000,
            stop_callback=stop,
        )

        assert result["iterations_performed"] < 1000


# ---------------------------------------------------------------------------
# Clean box and mask
# ---------------------------------------------------------------------------


class TestMaskAndCleanBox:
    def test_clean_box_restricts_peak_search(self):
        """Sources outside the clean box must be ignored."""
        ny, nx = 32, 32
        dirty = _point_source_residual(ny, nx, [(5, 5, 10.0), (20, 20, 3.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        # Only the region around (20, 20). Box is (xmin, xmax, ymin, ymax).
        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=10,
            clean_box=(15, 25, 15, 25),
        )

        # The strong source at (5, 5) is outside the box and must survive.
        assert dirty[5, 5] == pytest.approx(10.0, abs=1e-6)
        assert model[5, 5] == pytest.approx(0.0, abs=1e-6)
        # The in-box source is cleaned.
        assert dirty[20, 20] == pytest.approx(0.0, abs=1e-6)
        assert model[20, 20] == pytest.approx(3.0, abs=1e-6)
        assert result["iterations_performed"] >= 1

    def test_mask_suppresses_peak(self):
        """Pixels with mask <= 0.5 are ignored when locating the peak."""
        ny, nx = 32, 32
        dirty = _point_source_residual(ny, nx, [(5, 5, 10.0), (20, 20, 3.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        mask = np.ones((ny, nx), dtype=bool)
        mask[5, 5] = False  # hide the strong source

        hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            mask=mask,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=10,
        )

        # Masked source is untouched; the unmasked one is cleaned.
        assert dirty[5, 5] == pytest.approx(10.0, abs=1e-6)
        assert model[5, 5] == pytest.approx(0.0, abs=1e-6)
        assert dirty[20, 20] == pytest.approx(0.0, abs=1e-6)
        assert model[20, 20] == pytest.approx(3.0, abs=1e-6)

    def test_mask_dtype_mismatch_raises(self):
        ny, nx = 16, 16
        dirty = _zeros(ny, nx, dtype=np.float32)
        psf = _delta_psf(ny, nx, dtype=np.float32)
        model = _zeros(ny, nx, dtype=np.float32)
        mask = np.ones((ny, nx), dtype=np.float64)

        with pytest.raises(RuntimeError, match="mask has wrong dtype"):
            hogbom.clean(
                dirty_image=dirty,
                psf=psf,
                model=model,
                mask=mask,
            )


# ---------------------------------------------------------------------------
# Return dict
# ---------------------------------------------------------------------------


class TestImagingDict:
    def test_keys_present(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=5,
        )

        for key in (
            "iterations_performed",
            "final_peak",
            "total_flux_cleaned",
            "converged",
        ):
            assert key in result
        # The in-place arrays should not be echoed back.
        assert "model_image" not in result
        assert "residual_image" not in result

    def test_total_flux_matches_model_l1_norm(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 2.0), (3, 12, -1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=1.0,
            threshold_per_cycle=0.1,
            max_iter_per_cycle_updated=20,
        )

        assert result["total_flux_cleaned"] == pytest.approx(
            float(np.sum(np.abs(model))), rel=1e-5
        )


# ---------------------------------------------------------------------------
# Cube binding: hogbom.clean_cube
# ---------------------------------------------------------------------------


def _delta_psf_cube(nt, nf, npol, ny, nx, dtype=np.float32):
    psf = np.zeros((nt, nf, npol, ny, nx), dtype=dtype)
    psf[..., ny // 2, nx // 2] = 1.0
    return psf


def _point_sources_cube(nt, nf, npol, ny, nx, per_plane_sources, dtype=np.float32):
    """
    Build a 5-D residual cube with a list of point sources per plane.

    Parameters
    ----------
    per_plane_sources : dict
        Maps ``(tt, nn, pp)`` -> list of ``(iy, ix, amp)``.
    """
    resid = np.zeros((nt, nf, npol, ny, nx), dtype=dtype)
    for (tt, nn, pp), sources in per_plane_sources.items():
        for iy, ix, amp in sources:
            resid[tt, nn, pp, iy, ix] = amp
    return resid


def _plane_iters(residual_cube, value):
    """Per-plane ``max_iter`` array for ``clean_cube``.

    Cycle control is independent per (time, frequency, polarization)
    plane: ``max_iter`` is an int32 array of shape ``(nt, nf, np)``.
    """
    return np.full(residual_cube.shape[:3], value, dtype=np.int32)


def _plane_threshold(residual_cube, value):
    """Per-plane ``threshold`` array for ``clean_cube``.

    ``threshold`` is a ``(nt, nf, np)`` array matching the image dtype
    (float32 or float64).
    """
    return np.full(residual_cube.shape[:3], value, dtype=residual_cube.dtype)


class TestCubeBasicCorrectness:
    def test_single_plane_cube(self):
        """A 1x1x1 cube with a single point source is cleaned identically
        to the 2-D clean on the same plane."""
        ny, nx = 32, 32
        nt, nf, npol = 1, 1, 1
        resid = _point_sources_cube(nt, nf, npol, ny, nx, {(0, 0, 0): [(10, 20, 5.0)]})
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        result = hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 10),
            processing_function_threads=1,
        )

        assert result["iterations_performed"].shape == (nt, nf, npol)
        assert model[0, 0, 0, 10, 20] == pytest.approx(5.0, abs=1e-6)
        assert np.allclose(resid, 0.0, atol=1e-6)
        assert bool(result["converged"][0, 0, 0]) is True

    def test_cube_planes_are_independent(self):
        """Each plane must be cleaned independently; a bright source in
        one plane must not contaminate another plane."""
        ny, nx = 32, 32
        nt, nf, npol = 2, 2, 2
        per_plane = {}
        expected_amp = {}
        for t in range(nt):
            for f in range(nf):
                for p in range(npol):
                    amp = 1.0 + t + 2 * f + 3 * p
                    per_plane[(t, f, p)] = [(8 + t, 12 + f + p, amp)]
                    expected_amp[(t, f, p)] = amp

        resid = _point_sources_cube(nt, nf, npol, ny, nx, per_plane)
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 10),
            processing_function_threads=4,
        )

        for (t, f, p), amp in expected_amp.items():
            iy, ix = 8 + t, 12 + f + p
            assert model[t, f, p, iy, ix] == pytest.approx(amp, abs=1e-5)
            assert np.sum(np.abs(model[t, f, p])) == pytest.approx(amp, abs=1e-5)
            assert np.allclose(resid[t, f, p], 0.0, atol=1e-5)

    def test_cube_broadcast_psf(self):
        """A single-polarization PSF must be broadcast across all image
        polarizations."""
        ny, nx = 16, 16
        nt, nf, npol = 1, 1, 3
        resid = _point_sources_cube(
            nt,
            nf,
            npol,
            ny,
            nx,
            {(0, 0, p): [(8, 8, 1.0 + p)] for p in range(npol)},
        )
        psf = _delta_psf_cube(nt, nf, 1, ny, nx)  # np_psf=1
        model = np.zeros_like(resid)

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 10),
            processing_function_threads=2,
        )

        for p in range(npol):
            assert model[0, 0, p, 8, 8] == pytest.approx(1.0 + p, abs=1e-5)

    def test_cube_threading_matches_serial(self):
        """Serial (processing_function_threads=1) and parallel (processing_function_threads=N) runs must
        produce identical results for deterministic inputs."""
        ny, nx = 24, 24
        nt, nf, npol = 2, 3, 2
        rng = np.random.default_rng(1234)

        resid0 = rng.standard_normal((nt, nf, npol, ny, nx)).astype(np.float32)
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)

        resid_a = resid0.copy()
        model_a = np.zeros_like(resid_a)
        hogbom.clean_cube(
            residual_cube=resid_a,
            psf_cube=psf,
            model_cube=model_a,
            gain=0.1,
            threshold_per_cycle=_plane_threshold(resid_a, 0.05),
            max_iter_per_cycle_updated=_plane_iters(resid_a, 30),
            processing_function_threads=1,
        )

        resid_b = resid0.copy()
        model_b = np.zeros_like(resid_b)
        hogbom.clean_cube(
            residual_cube=resid_b,
            psf_cube=psf,
            model_cube=model_b,
            gain=0.1,
            threshold_per_cycle=_plane_threshold(resid_b, 0.05),
            max_iter_per_cycle_updated=_plane_iters(resid_b, 30),
            processing_function_threads=4,
        )

        assert np.array_equal(model_a, model_b)
        assert np.array_equal(resid_a, resid_b)

    def test_cube_float64(self):
        ny, nx = 16, 16
        nt, nf, npol = 1, 2, 1
        resid = _point_sources_cube(
            nt,
            nf,
            npol,
            ny,
            nx,
            {(0, nn, 0): [(8, 8, 3.0 + nn)] for nn in range(nf)},
            dtype=np.float64,
        )
        psf = _delta_psf_cube(nt, nf, npol, ny, nx, dtype=np.float64)
        model = np.zeros_like(resid)

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 10),
            processing_function_threads=2,
        )

        assert resid.dtype == np.float64
        assert model.dtype == np.float64
        for nn in range(nf):
            assert model[0, nn, 0, 8, 8] == pytest.approx(3.0 + nn, abs=1e-12)


class TestCubeInPlaceContract:
    def test_cube_buffers_modified_in_place(self):
        ny, nx = 16, 16
        nt, nf, npol = 1, 1, 2
        resid = _point_sources_cube(
            nt,
            nf,
            npol,
            ny,
            nx,
            {(0, 0, 0): [(8, 8, 2.0)], (0, 0, 1): [(4, 4, 1.0)]},
        )
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        resid_ptr = resid.__array_interface__["data"][0]
        model_ptr = model.__array_interface__["data"][0]
        resid_view = resid[...]
        model_view = model[...]

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 5),
            processing_function_threads=2,
        )

        assert resid.__array_interface__["data"][0] == resid_ptr
        assert model.__array_interface__["data"][0] == model_ptr
        assert np.array_equal(resid, resid_view)
        assert np.array_equal(model, model_view)

    def test_cube_model_accumulates(self):
        ny, nx = 16, 16
        nt, nf, npol = 1, 1, 1
        resid = _point_sources_cube(nt, nf, npol, ny, nx, {(0, 0, 0): [(8, 8, 1.0)]})
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)

        model = np.zeros_like(resid)
        model[0, 0, 0, 3, 3] = 7.0

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 5),
            processing_function_threads=1,
        )

        assert model[0, 0, 0, 3, 3] == pytest.approx(7.0, abs=1e-6)
        assert model[0, 0, 0, 8, 8] == pytest.approx(1.0, abs=1e-6)

    def test_cube_psf_not_modified(self):
        ny, nx = 16, 16
        nt, nf, npol = 1, 1, 1
        resid = _point_sources_cube(nt, nf, npol, ny, nx, {(0, 0, 0): [(8, 8, 1.0)]})
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        psf_before = psf.copy()
        model = np.zeros_like(resid)

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 5),
            processing_function_threads=1,
        )

        assert np.array_equal(psf, psf_before)


class TestCubeValidation:
    def test_cube_requires_5d(self):
        ny, nx = 16, 16
        resid = np.zeros((1, ny, nx), dtype=np.float32)
        psf = np.zeros((1, ny, nx), dtype=np.float32)
        model = np.zeros((1, ny, nx), dtype=np.float32)

        with pytest.raises(RuntimeError, match="5D"):
            hogbom.clean_cube(
                residual_cube=resid,
                psf_cube=psf,
                model_cube=model,
            )

    def test_cube_psf_pol_mismatch(self):
        ny, nx = 16, 16
        resid = np.zeros((1, 1, 3, ny, nx), dtype=np.float32)
        psf = np.zeros((1, 1, 2, ny, nx), dtype=np.float32)  # invalid
        model = np.zeros_like(resid)

        with pytest.raises(RuntimeError, match="polarization"):
            hogbom.clean_cube(
                residual_cube=resid,
                psf_cube=psf,
                model_cube=model,
            )

    def test_cube_noncontiguous_raises(self):
        ny, nx = 16, 16
        big = np.zeros((1, 1, 1, 2 * ny, 2 * nx), dtype=np.float32)
        resid = big[..., ::2, ::2]  # strided view
        psf = np.zeros((1, 1, 1, ny, nx), dtype=np.float32)
        model = np.zeros((1, 1, 1, ny, nx), dtype=np.float32)

        assert not resid.flags["C_CONTIGUOUS"]
        with pytest.raises(RuntimeError, match="C-contiguous"):
            hogbom.clean_cube(
                residual_cube=resid,
                psf_cube=psf,
                model_cube=model,
            )

    def test_cube_readonly_residual_raises(self):
        ny, nx = 16, 16
        resid = np.zeros((1, 1, 1, ny, nx), dtype=np.float32)
        resid.setflags(write=False)
        psf = np.zeros_like(resid, dtype=np.float32)
        psf.setflags(write=True)  # ensure psf itself ok, but mutable test not needed
        psf = np.zeros((1, 1, 1, ny, nx), dtype=np.float32)
        model = np.zeros((1, 1, 1, ny, nx), dtype=np.float32)

        with pytest.raises(RuntimeError, match="writeable"):
            hogbom.clean_cube(
                residual_cube=resid,
                psf_cube=psf,
                model_cube=model,
            )

    def test_cube_dtype_mismatch_raises(self):
        ny, nx = 16, 16
        resid = np.zeros((1, 1, 1, ny, nx), dtype=np.float32)
        psf = np.zeros((1, 1, 1, ny, nx), dtype=np.float64)
        model = np.zeros((1, 1, 1, ny, nx), dtype=np.float32)

        with pytest.raises(RuntimeError, match="dtype"):
            hogbom.clean_cube(
                residual_cube=resid,
                psf_cube=psf,
                model_cube=model,
            )

    def test_cube_unsupported_dtype_raises(self):
        ny, nx = 16, 16
        resid = np.zeros((1, 1, 1, ny, nx), dtype=np.int32)
        psf = np.zeros_like(resid)
        model = np.zeros_like(resid)

        with pytest.raises(RuntimeError, match="float32 or float64"):
            hogbom.clean_cube(
                residual_cube=resid,
                psf_cube=psf,
                model_cube=model,
            )


class TestCubeMaskAndBox:
    def test_cube_clean_box_restricts_search(self):
        ny, nx = 32, 32
        nt, nf, npol = 1, 1, 1
        resid = _point_sources_cube(
            nt,
            nf,
            npol,
            ny,
            nx,
            {(0, 0, 0): [(5, 5, 10.0), (20, 20, 3.0)]},
        )
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 10),
            clean_box=(15, 25, 15, 25),
            processing_function_threads=1,
        )

        assert resid[0, 0, 0, 5, 5] == pytest.approx(10.0, abs=1e-6)
        assert model[0, 0, 0, 5, 5] == pytest.approx(0.0, abs=1e-6)
        assert resid[0, 0, 0, 20, 20] == pytest.approx(0.0, abs=1e-6)
        assert model[0, 0, 0, 20, 20] == pytest.approx(3.0, abs=1e-6)

    def test_cube_mask_suppresses_peak(self):
        ny, nx = 32, 32
        nt, nf, npol = 1, 1, 1
        resid = _point_sources_cube(
            nt,
            nf,
            npol,
            ny,
            nx,
            {(0, 0, 0): [(5, 5, 10.0), (20, 20, 3.0)]},
        )
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        mask = np.ones(resid.shape, dtype=bool)
        mask[0, 0, 0, 5, 5] = False

        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            mask_cube=mask,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 10),
            processing_function_threads=1,
        )

        assert resid[0, 0, 0, 5, 5] == pytest.approx(10.0, abs=1e-6)
        assert resid[0, 0, 0, 20, 20] == pytest.approx(0.0, abs=1e-6)
        assert model[0, 0, 0, 20, 20] == pytest.approx(3.0, abs=1e-6)


class TestCubeReturnShapes:
    def test_return_shapes(self):
        ny, nx = 16, 16
        nt, nf, npol = 2, 3, 2
        resid = _point_sources_cube(
            nt,
            nf,
            npol,
            ny,
            nx,
            {
                (t, f, p): [(8, 8, 1.0)]
                for t in range(nt)
                for f in range(nf)
                for p in range(npol)
            },
        )
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        result = hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.01),
            max_iter_per_cycle_updated=_plane_iters(resid, 5),
            processing_function_threads=2,
        )

        expected_shape = (nt, nf, npol)
        assert result["iterations_performed"].shape == expected_shape
        assert result["final_peak"].shape == expected_shape
        assert result["total_flux_cleaned"].shape == expected_shape
        assert result["converged"].shape == expected_shape
        assert result["iterations_performed"].dtype.kind == "i"
        assert result["final_peak"].dtype == np.float32

    def test_processing_function_threads_clamped(self):
        """processing_function_threads larger than the number of planes must be handled."""
        ny, nx = 8, 8
        nt, nf, npol = 1, 1, 1
        resid = _point_sources_cube(nt, nf, npol, ny, nx, {(0, 0, 0): [(4, 4, 1.0)]})
        psf = _delta_psf_cube(nt, nf, npol, ny, nx)
        model = np.zeros_like(resid)

        # Should not crash even though there is only one plane.
        hogbom.clean_cube(
            residual_cube=resid,
            psf_cube=psf,
            model_cube=model,
            gain=1.0,
            threshold_per_cycle=_plane_threshold(resid, 0.1),
            max_iter_per_cycle_updated=_plane_iters(resid, 5),
            processing_function_threads=64,
        )
        assert model[0, 0, 0, 4, 4] == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# A PSF of twice the image size: every component subtracted from every pixel
# ---------------------------------------------------------------------------

CUBE_KERNELS = ["clean_cube", "clean_cube_many_threads"]


def _exact_inputs(ny, nx, np_psf, dtype, seed=0):
    """Two residual planes of noise, a PSF of twice their size with seeded
    sidelobes out to its edge (peak 1 at (ny, nx)), a disc mask and a box."""
    rng = np.random.default_rng(seed)
    residual = rng.standard_normal((1, 1, 2, ny, nx)).astype(dtype)
    psf = (0.1 * rng.standard_normal((1, 1, np_psf, 2 * ny, 2 * nx))).astype(dtype)
    psf[..., ny, nx] = 1.0
    yy, xx = np.mgrid[:ny, :nx]
    disc = (yy - ny // 2) ** 2 + (xx - nx // 2) ** 2 <= (0.45 * min(ny, nx)) ** 2
    mask = np.broadcast_to(disc, residual.shape).copy()
    return residual, psf, mask, (2, nx - 1, 1, ny - 2)


def _run_kernel(kernel, residual, psf, model, mask, box, n_iter, gain, divergence):
    return getattr(hogbom, kernel)(
        residual_cube=residual,
        psf_cube=psf,
        model_cube=model,
        mask_cube=mask,
        clean_box=box,
        max_iter_per_cycle_updated=np.full(residual.shape[:3], n_iter, np.int32),
        gain=gain,
        threshold_per_cycle=np.zeros(residual.shape[:3], residual.dtype),
        processing_function_threads=2,
        max_iter_divergence=divergence,
        divergence_rms_factor=1.01,
        divergence_peak_factor=1.1,
        record_iteration_history=True,
    )


class TestCubeExactSubtraction:
    @pytest.mark.parametrize("kernel", CUBE_KERNELS)
    @pytest.mark.parametrize("np_psf", [1, 2])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("shape", [(24, 24), (23, 23), (20, 26)])
    @pytest.mark.parametrize("divergence", [-1, 1])
    def test_equals_the_image_sized_kernel_on_an_embedded_image(
        self, kernel, np_psf, dtype, shape, divergence
    ):
        """The PSF of twice the image size gives, to the bit, what the
        image-sized kernel gives on a 2n image holding the residual at
        (ny // 2, nx // 2), searched only there, with that PSF as its own."""
        ny, nx = shape
        residual, psf, mask, box = _exact_inputs(ny, nx, np_psf, dtype)
        oy, ox = ny // 2, nx // 2
        region = (..., slice(oy, oy + ny), slice(ox, ox + nx))
        residual_2n = np.zeros(residual.shape[:3] + (2 * ny, 2 * nx), dtype)
        residual_2n[region] = residual
        mask_2n = np.zeros(residual_2n.shape, bool)
        mask_2n[region] = mask
        box_2n = (box[0] + ox, box[1] + ox, box[2] + oy, box[3] + oy)
        model, model_2n = np.zeros_like(residual), np.zeros_like(residual_2n)

        out = _run_kernel(kernel, residual, psf, model, mask, box, 200, 0.1, divergence)
        ref = _run_kernel(
            kernel, residual_2n, psf, model_2n, mask_2n, box_2n, 200, 0.1, divergence
        )

        assert np.array_equal(residual, residual_2n[region])
        assert np.array_equal(model, model_2n[region])
        model_2n[region] = 0
        assert not model_2n.any()
        for key in ("iterations_performed", "stop_code"):
            assert np.array_equal(out[key], ref[key])
        assert out["iterations_performed"].min() > 0
        for key in ("iteration_peak_residual", "iteration_residual_rms"):
            for got, want in zip(out[key][0][0], ref[key][0][0], strict=True):
                assert np.array_equal(got, want)

    @pytest.mark.parametrize("kernel", CUBE_KERNELS)
    def test_residual_is_the_dirty_image_less_the_model_times_the_psf(self, kernel):
        """With the PSF of twice the image size the residual is the dirty image
        less the model convolved with that PSF, at every pixel. The
        image-sized PSF misses a sidelobe ring at 0.75 n."""
        n = 32
        yy, xx = np.mgrid[: 2 * n, : 2 * n] - n
        r = np.hypot(yy, xx)
        psf2d = np.exp(-(r**2) / 6.0) + 0.2 * np.exp(-((r - 0.75 * n) ** 2) / 2.0)
        psf = np.broadcast_to(psf2d / psf2d[n, n], (1, 1, 1, 2 * n, 2 * n)).copy()

        def shifted(y, x):
            return psf[0, 0, 0, n - y : 2 * n - y, n - x : 2 * n - x]

        dirty = np.zeros((1, 1, 1, n, n))
        for y, x, flux in ((5, 7, 1.0), (24, 20, -0.6), (12, 28, 0.4)):
            dirty[0, 0, 0] += flux * shifted(y, x)
        crop = psf[..., n - n // 2 : n - n // 2 + n, n - n // 2 : n - n // 2 + n]
        errors = {}
        for name, used in (("2n", psf), ("n", np.ascontiguousarray(crop))):
            residual, model = dirty.copy(), np.zeros_like(dirty)
            _run_kernel(
                kernel, residual, used, model, np.array([]), (-1,) * 4, 300, 0.1, -1
            )
            expected = dirty[0, 0, 0].copy()
            for y, x in zip(*np.nonzero(model[0, 0, 0]), strict=True):
                expected -= model[0, 0, 0, y, x] * shifted(y, x)
            errors[name] = np.abs(residual[0, 0, 0] - expected).max()
        assert errors["2n"] <= 1e-12 * np.abs(dirty).max()
        assert errors["n"] > 1e-2 * np.abs(dirty).max()

    @pytest.mark.parametrize("kernel", CUBE_KERNELS)
    @pytest.mark.parametrize(
        "psf_shape", [(17, 16), (32, 16), (33, 32)], ids=["n+1", "2n_y", "2n+1_y"]
    )
    def test_other_psf_sizes_are_refused(self, kernel, psf_shape):
        resid = np.zeros((1, 1, 1, 16, 16), dtype=np.float32)
        psf = np.zeros((1, 1, 1, *psf_shape), dtype=np.float32)
        with pytest.raises(RuntimeError, match="twice"):
            getattr(hogbom, kernel)(
                residual_cube=resid, psf_cube=psf, model_cube=np.zeros_like(resid)
            )

    def test_plane_clean_takes_an_image_sized_psf_only(self):
        dirty = _point_source_residual(16, 16, [(8, 8, 1.0)])
        with pytest.raises(RuntimeError, match="wrong shape"):
            hogbom.clean(
                dirty_image=dirty, psf=_delta_psf(32, 32), model=_zeros(16, 16)
            )


# ---------------------------------------------------------------------------
# An image-sized PSF: every sample subtracted, odd sizes included
# ---------------------------------------------------------------------------


def _hogbom_every_psf_sample(residual, psf, n_iter, gain):
    """Hogbom on one plane: the PSF centre (psf_ny // 2, psf_nx // 2) on the
    component, every PSF sample that falls in the image subtracted."""
    residual, model = residual.copy(), np.zeros_like(residual)
    gain = residual.dtype.type(gain)
    for _ in range(n_iter):
        py, px = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
        pv = gain * residual[py, px]
        model[py, px] += pv
        y0, x0 = py - psf.shape[0] // 2, px - psf.shape[1] // 2
        ys = slice(max(y0, 0), min(y0 + psf.shape[0], residual.shape[0]))
        xs = slice(max(x0, 0), min(x0 + psf.shape[1], residual.shape[1]))
        residual[ys, xs] -= (
            pv * psf[ys.start - y0 : ys.stop - y0, xs.start - x0 : xs.stop - x0]
        )
    return residual, model


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(23, 23), (21, 25), (23, 24), (24, 23)])
def test_odd_sizes_subtract_every_psf_sample_in_every_kernel(dtype, shape):
    """An image-sized PSF peaked at (ny // 2, nx // 2) is subtracted over the
    offsets -(n // 2) .. (n - 1) // 2 of each axis, the row and column at
    +n // 2 of an odd axis included, by clean, clean_cube and
    clean_cube_many_threads to the bit."""
    ny, nx = shape
    rng = np.random.default_rng(7)
    residual0 = rng.standard_normal((1, 1, 2, ny, nx)).astype(dtype)
    psf = (0.1 * rng.standard_normal((1, 1, 2, ny, nx))).astype(dtype)
    psf[..., ny // 2, nx // 2] = 1.0
    n_iter = 100
    out = {}
    for kernel in CUBE_KERNELS:
        residual, model = residual0.copy(), np.zeros_like(residual0)
        res = _run_kernel(
            kernel, residual, psf, model, np.array([]), (-1,) * 4, n_iter, 0.1, -1
        )
        assert res["iterations_performed"].tolist() == [[[n_iter, n_iter]]]
        out[kernel] = residual, model, res
    (r1, m1, o1), (r2, m2, o2) = out["clean_cube"], out["clean_cube_many_threads"]
    assert np.array_equal(r1, r2)
    assert np.array_equal(m1, m2)
    assert np.array_equal(o1["stop_code"], o2["stop_code"])
    for key in ("iteration_peak_residual", "iteration_residual_rms"):
        for got, want in zip(o1[key][0][0], o2[key][0][0], strict=True):
            assert np.array_equal(got, want)

    without_edge = psf[..., : ny - ny % 2, : nx - nx % 2]
    for p in range(2):
        want_r, want_m = _hogbom_every_psf_sample(
            residual0[0, 0, p], psf[0, 0, p], n_iter, 0.1
        )
        assert np.array_equal(r1[0, 0, p], want_r)
        assert np.array_equal(m1[0, 0, p], want_m)
        edge_dropped, _ = _hogbom_every_psf_sample(
            residual0[0, 0, p], without_edge[0, 0, p], n_iter, 0.1
        )
        assert not np.array_equal(r1[0, 0, p], edge_dropped)

    residual, model = residual0[0, 0, 0].copy(), np.zeros_like(residual0[0, 0, 0])
    hogbom.clean(
        dirty_image=residual,
        psf=np.ascontiguousarray(psf[0, 0, 0]),
        model=model,
        max_iter_per_cycle_updated=n_iter,
        gain=0.1,
        threshold_per_cycle=0.0,
        max_iter_divergence=-1,
    )
    assert np.array_equal(residual, r1[0, 0, 0])
    assert np.array_equal(model, m1[0, 0, 0])


@pytest.mark.parametrize("casa_cap", [0, 1, 5])
@pytest.mark.parametrize("cycles", [1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_explicit_budget_matches_casa_capped_point_updates(casa_cap, cycles, dtype):
    """Match CASA's inclusive cap by inputs, keeping the kernel exclusive.

    An isolated unit point with gain 0.1 has residual 0.9**updates. This
    checks physical model updates independently of either engine's counter.
    Zero remains inactive; each positive capped CASA call needs cap+1 here.
    """
    dirty = _point_source_residual(16, 16, [(8, 8, 1.0)], dtype=dtype)
    model = _zeros(16, 16, dtype=dtype)
    psf = _delta_psf(16, 16, dtype=dtype)
    budget = casa_cap + 1 if casa_cap else 0
    for _ in range(cycles):
        result = hogbom.clean(
            dirty_image=dirty,
            psf=psf,
            model=model,
            gain=0.1,
            threshold_per_cycle=0.0,
            max_iter_per_cycle_updated=budget,
        )
        assert result["iterations_performed"] == budget
    expected_residual = 0.9 ** (cycles * budget)
    assert dirty[8, 8] == pytest.approx(expected_residual, abs=2e-7)
    assert model.sum() == pytest.approx(1 - expected_residual, abs=2e-7)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
