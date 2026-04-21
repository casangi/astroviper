"""
Unit tests for the pybind11-wrapped Hogbom CLEAN routine
(``astroviper.core.imaging.deconvolvers.hogbom.clean``).

The wrapper operates on Python-owned numpy buffers with no copies on the
C++ side: the ``dirty_image`` and ``model`` arrays are written into
directly. These tests exercise the correctness of the algorithm on small
hand-constructed inputs as well as the zero-copy / in-place contract of
the binding.
"""

import numpy as np
import pytest

try:
    from astroviper.core.imaging.deconvolvers import hogbom

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
            threshold=0.1,
            max_iter=10,
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
            threshold=0.1,
            max_iter=50,
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
            threshold=0.0,
            max_iter=1,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=5,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=5,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=5,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=5,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=5,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=0.05, threshold=0.0, max_iter=7,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.01, max_iter=100,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=0.1, threshold=0.0, max_iter=20,
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
            dirty_image=dirty, psf=psf, model=model,
            gain=0.01, threshold=0.0, max_iter=1000,
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
        dirty = _point_source_residual(
            ny, nx, [(5, 5, 10.0), (20, 20, 3.0)]
        )
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        # Only the region around (20, 20). Box is (xmin, xmax, ymin, ymax).
        result = hogbom.clean(
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=10,
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
        dirty = _point_source_residual(
            ny, nx, [(5, 5, 10.0), (20, 20, 3.0)]
        )
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        mask = np.ones((ny, nx), dtype=np.float32)
        mask[5, 5] = 0.0  # hide the strong source

        hogbom.clean(
            dirty_image=dirty, psf=psf, model=model, mask=mask,
            gain=1.0, threshold=0.1, max_iter=10,
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
                dirty_image=dirty, psf=psf, model=model, mask=mask,
            )


# ---------------------------------------------------------------------------
# Return dict
# ---------------------------------------------------------------------------


class TestReturnDict:
    def test_keys_present(self):
        ny, nx = 16, 16
        dirty = _point_source_residual(ny, nx, [(8, 8, 1.0)])
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        result = hogbom.clean(
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=5,
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
        dirty = _point_source_residual(
            ny, nx, [(8, 8, 2.0), (3, 12, -1.0)]
        )
        psf = _delta_psf(ny, nx)
        model = _zeros(ny, nx)

        result = hogbom.clean(
            dirty_image=dirty, psf=psf, model=model,
            gain=1.0, threshold=0.1, max_iter=20,
        )

        assert result["total_flux_cleaned"] == pytest.approx(
            float(np.sum(np.abs(model))), rel=1e-5
        )


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
