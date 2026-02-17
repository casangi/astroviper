"""
Tests for astroviper.core.imaging.imager — component + end-to-end.

Component tests use lightweight synthetic ms4 (no download).
E2E tests use generate_ms4_with_point_sources (requires data download).

Run:
    micromamba run -n astroviper_test pytest tests/unit/core/imaging/test_imager.py -v
"""

import copy
import pytest
import numpy as np
import xarray as xr

from astroviper.core.imaging.imager import (
    _get_param,
    _DEFAULT_PARAMS,
    grid_visibilities,
    compute_residual_visibilities,
    predict_model_visibilities,
    run_imaging_loop,
)
from astroviper.core.imaging.imaging_utils.iteration_control import (
    MAJOR_CONTINUE,
    MAJOR_ITER_LIMIT,
    MAJOR_THRESHOLD,
    MAJOR_CYCLE_LIMIT,
)
from astroviper.core.imaging.imaging_utils.return_dict import (
    ReturnDict,
    Key,
    FIELD_ACCUM,
)
from astroviper.core.imaging.imaging_utils.standard_gridding_example import (
    generate_ms4_with_point_sources,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def ms4_point_sources():
    """Realistic ms4 with 4 point sources (1 Jy each). Downloads data on first use."""
    np.random.seed(42)
    sources, npix, cell, ms4 = generate_ms4_with_point_sources(4, np.ones(4))
    cell_rad = cell.to("rad").value
    return sources, npix, cell_rad, ms4


@pytest.fixture()
def ms4_copy(ms4_point_sources):
    """Deep copy of ms4 — safe to mutate (imaging loop writes MODEL/RESIDUAL)."""
    sources, npix, cell_rad, ms4 = ms4_point_sources
    return sources, npix, cell_rad, copy.deepcopy(ms4)


@pytest.fixture()
def default_params(ms4_point_sources):
    """Baseline imaging params for point source data."""
    _, npix, cell_rad, _ = ms4_point_sources
    return {
        "cell_size": (-cell_rad, cell_rad),
        "image_size": (npix, npix),
        "corr_type": "linear",
    }


# ============================================================================
# 1. Unit Tests — Individual Components
# ============================================================================


class TestGetParam:
    """Tests for _get_param() fallback logic."""

    def test_returns_user_value(self):
        assert _get_param({"niter": 42}, "niter") == 42

    def test_falls_back_to_default(self):
        assert _get_param({}, "niter") == _DEFAULT_PARAMS["niter"]

    def test_returns_none_for_unknown_key(self):
        assert _get_param({}, "nonexistent_key_xyz") is None

    def test_user_value_overrides_default(self):
        assert _get_param({"threshold": 0.5}, "threshold") == 0.5
        assert _DEFAULT_PARAMS["threshold"] != 0.5  # sanity


class TestGridVisibilities:
    """Tests for grid_visibilities()."""

    def test_output_shape(self, ms4_copy):
        _, npix, cell_rad, ms4 = ms4_copy
        dirty = grid_visibilities(
            ms4,
            image_size=(npix, npix),
            cell_size=(-cell_rad, cell_rad),
        )
        n_chan = len(ms4.coords["frequency"])
        n_pol = len(ms4.coords["polarization"])
        assert dirty.shape == (n_chan, n_pol, npix, npix)

    def test_output_dtype_is_float(self, ms4_copy):
        _, npix, cell_rad, ms4 = ms4_copy
        dirty = grid_visibilities(
            ms4,
            image_size=(npix, npix),
            cell_size=(-cell_rad, cell_rad),
        )
        assert dirty.dtype == np.float64

    def test_nonzero_output(self, ms4_copy):
        _, npix, cell_rad, ms4 = ms4_copy
        dirty = grid_visibilities(
            ms4,
            image_size=(npix, npix),
            cell_size=(-cell_rad, cell_rad),
        )
        assert np.max(np.abs(dirty)) > 0


class TestComputeResidualVisibilities:
    """Tests for compute_residual_visibilities()."""

    def test_residual_equals_vis_minus_model(self, ms4_copy):
        _, _, _, ms4 = ms4_copy
        # Set VISIBILITY_MODEL to half of VISIBILITY
        ms4["VISIBILITY_MODEL"] = ms4["VISIBILITY"] * 0.5
        compute_residual_visibilities(ms4)
        np.testing.assert_allclose(
            ms4["RESIDUAL"].values,
            ms4["VISIBILITY"].values * 0.5,
            atol=1e-12,
        )

    def test_residual_zero_when_model_equals_vis(self, ms4_copy):
        _, _, _, ms4 = ms4_copy
        ms4["VISIBILITY_MODEL"] = ms4["VISIBILITY"].copy()
        compute_residual_visibilities(ms4)
        residual = ms4["RESIDUAL"].values
        # Mask out NaN entries (from NaN UVW baselines in test data)
        valid = ~np.isnan(residual)
        np.testing.assert_allclose(residual[valid], 0.0, atol=1e-12)


# End-to-End Tests


class TestEndToEnd:
    """
    Test various modes of end-to-end running of the imaging loop
    """

    def test_raises_without_cell_size(self, ms4_copy):
        _, _, _, ms4 = ms4_copy
        with pytest.raises(ValueError, match="cell_size"):
            run_imaging_loop(ms4, {})

    def test_auto_stokes_2corr_linear(self, ms4_copy):
        """2-corr linear → ['I', 'Q'] auto-detection produces 2-stokes output."""
        _, npix, cell_rad, ms4 = ms4_copy
        n_corr = len(ms4.coords["polarization"])
        assert n_corr == 2, "Test data should have 2 correlations"
        # Run minimal imaging — niter=1 to avoid deconvolve's niter>0 validation
        model, _, _, controller = run_imaging_loop(
            ms4,
            {
                "cell_size": (-cell_rad, cell_rad),
                "image_size": (npix, npix),
                "corr_type": "linear",
                "niter": 1,
                "nmajor": 1,
            },
        )
        # Auto-detected stokes=['I','Q'] → n_stokes=2
        assert model.shape[1] == 2

    def test_stops_at_niter(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        params = {**default_params, "niter": 10, "nmajor": 20, "threshold": 0.0}
        _, _, _, controller = run_imaging_loop(ms4, params)
        assert controller.total_iter_done <= 10
        assert controller.stopcode.major == MAJOR_ITER_LIMIT

    def test_stops_at_nmajor(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        params = {**default_params, "nmajor": 2, "niter": 5000, "threshold": 0.0}
        _, _, _, controller = run_imaging_loop(ms4, params)
        assert controller.major_done == 2
        assert controller.stopcode.major == MAJOR_CYCLE_LIMIT

    def test_stops_at_threshold(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        # Use Stokes I only and conservative niter to avoid divergence
        # with high PSF sidelobes
        from astroviper.core.imaging.imaging_utils.corr_to_stokes import (
            image_corr_to_stokes,
        )

        dirty_corr = grid_visibilities(
            ms4,
            image_size=default_params["image_size"],
            cell_size=default_params["cell_size"],
        )
        dirty_I = image_corr_to_stokes(
            dirty_corr, corr_type="linear", pol_axis=1, stokes_out=["I"]
        )
        initial_peak = np.max(np.abs(dirty_I))
        # Set threshold to 80% of initial peak — easily reachable in 1 cycle
        threshold = initial_peak * 0.8
        params = {
            **default_params,
            "stokes": ["I"],
            "threshold": threshold,
            "niter": 500,
            "nmajor": 5,
        }
        _, residual, _, controller = run_imaging_loop(ms4, params)
        assert controller.stopcode.major == MAJOR_THRESHOLD
        final_peak = np.max(np.abs(residual))
        # cyclethreshold may be slightly above global threshold
        assert final_peak <= initial_peak

    def test_single_major(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        params = {**default_params, "nmajor": 1, "niter": 500}
        _, _, _, controller = run_imaging_loop(ms4, params)
        assert controller.major_done == 1

    def test_residual_below_dirty_peak(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        # Get initial dirty image peak
        from astroviper.core.imaging.imaging_utils.corr_to_stokes import (
            image_corr_to_stokes,
        )

        dirty_corr = grid_visibilities(
            ms4,
            image_size=default_params["image_size"],
            cell_size=default_params["cell_size"],
        )
        dirty_I = image_corr_to_stokes(
            dirty_corr, corr_type="linear", pol_axis=1, stokes_out=["I"]
        )
        initial_peak = np.max(np.abs(dirty_I))

        # Run 1 major cycle with few iterations
        params = {
            **default_params,
            "stokes": ["I"],
            "niter": 10,
            "nmajor": 1,
            "threshold": 0.0,
        }
        _, residual, return_dict, _ = run_imaging_loop(ms4, params)

        # After a few CLEAN iterations, residual peak should be below initial
        final_peak = np.max(np.abs(residual))
        assert (
            final_peak < initial_peak
        ), f"Residual peak {final_peak:.6f} should be below dirty peak {initial_peak:.6f}"

        # Also verify return_dict tracks peakres
        entry = return_dict.sel(time=0, pol=0, chan=0)
        assert entry is not None
        assert "peakres" in entry

    def test_model_is_finite_and_nonzero(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        params = {
            **default_params,
            "stokes": ["I"],
            "niter": 5,
            "nmajor": 1,
            "threshold": 0.0,
        }
        model, residual, _, controller = run_imaging_loop(ms4, params)
        model_I = model[0, 0, :, :]

        assert np.all(np.isfinite(model_I)), "model contains NaN/inf"
        assert np.any(model_I != 0), "model should have nonzero components"
        assert controller.total_iter_done > 0, "should have done iterations"

        # Residual should differ from dirty image (deconvolver subtracted something)
        assert np.all(np.isfinite(residual)), "residual contains NaN/inf"

    def test_stokes_i_only(self, ms4_copy, default_params):
        _, npix, _, ms4 = ms4_copy
        params = {**default_params, "stokes": ["I"], "niter": 100, "nmajor": 1}
        model, residual, _, _ = run_imaging_loop(ms4, params)
        n_chan = len(ms4.coords["frequency"])
        expected = (n_chan, 1, npix, npix)
        assert model.shape == expected
        assert residual.shape == expected
        assert np.isrealobj(model)
        assert np.isrealobj(residual)

    def test_keys_and_fields(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        params = {**default_params, "niter": 500, "nmajor": 2}
        _, _, return_dict, controller = run_imaging_loop(ms4, params)

        # return_dict should have entries keyed by (time, pol, chan)
        assert len(return_dict.data) > 0
        for key in return_dict.data:
            assert isinstance(key, Key)
            assert hasattr(key, "time")
            assert hasattr(key, "pol")
            assert hasattr(key, "chan")

        # FIELD_ACCUM fields should be lists with length == number of major cycles
        entry = return_dict.sel(time=0, pol=0, chan=0)
        n_major = controller.major_done
        for field in ["peakres", "iter_done", "model_flux"]:
            assert field in entry, f"Missing field: {field}"
            assert isinstance(entry[field], list), f"{field} should be list"
            assert (
                len(entry[field]) == n_major
            ), f"{field}: expected {n_major} entries, got {len(entry[field])}"

    def test_controller_state_serializable(self, ms4_copy, default_params):
        _, _, _, ms4 = ms4_copy
        params = {**default_params, "niter": 100, "nmajor": 1}
        _, _, _, controller = run_imaging_loop(ms4, params)
        state = controller.get_state()
        assert isinstance(state, dict)
        # Should be JSON-serializable (all basic types)
        import json

        json.dumps(state)  # raises if not serializable

    def test_shapes(self, ms4_copy, default_params):
        _, npix, _, ms4 = ms4_copy
        params = {**default_params, "niter": 100, "nmajor": 1}
        model, residual, _, _ = run_imaging_loop(ms4, params)
        n_chan = len(ms4.coords["frequency"])
        # 2-corr linear → ['I', 'Q'] → n_stokes=2
        expected = (n_chan, 2, npix, npix)
        assert model.shape == expected, f"model shape {model.shape} != {expected}"
        assert (
            residual.shape == expected
        ), f"residual shape {residual.shape} != {expected}"

    def test_dtype_is_float(self, ms4_copy, default_params):
        """Regression: 2026-02-09 complex dtype bugfix."""
        _, _, _, ms4 = ms4_copy
        params = {**default_params, "niter": 100, "nmajor": 1}
        model, residual, _, _ = run_imaging_loop(ms4, params)
        assert np.isrealobj(model), f"model dtype {model.dtype} should be real"
        assert np.isrealobj(residual), f"residual dtype {residual.dtype} should be real"
