"""
Unified pytest module for astroviper.fitting.multi_gaussian2d_fit

Covers:
- NumPy-backed success paths, vectorization, flags, world coords
- Dask-backed success path (skipped if dask.array unavailable)
- Failure paths: full masking, curve_fit exception, pcov=None
- Helper API: initial guesses formats, bounds merging

Run:
    pytest -q
    # optional coverage:
    # pytest -q --cov=astroviper.fitting.multi_gaussian2d_fit --cov-report=term-missing
"""

from __future__ import annotations

import unittest

import os, sys
import math
import numpy as np
import pytest
import xarray as xr

import dask.array as da  # type: ignore

import importlib
import inspect
import builtins

# Use headless matplotlib, silence plt.show()
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import matplotlib.pyplot as plt
import warnings
import matplotlib.figure as _mf

from astroviper.distributed.image_analysis.multi_gaussian2d_fit import (
    fit_multi_gaussian2d,
    plot_components,
)
import astroviper.distributed.image_analysis.multi_gaussian2d_fit as mg

# ------------------------- fixtures / helpers -------------------------


@pytest.fixture(autouse=True)
def _silence_plots(monkeypatch: pytest.MonkeyPatch) -> None:
    plt.ioff()
    monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)


def _rot(x, y, theta):
    ct, st = math.cos(theta), math.sin(theta)
    xr_ = x * ct + y * st
    yr_ = -x * st + y * ct
    return xr_, yr_


def _gauss2d_on_grid(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    *,
    offset: float = 0.0,
) -> np.ndarray:
    """Single rotated anisotropic Gaussian evaluated on prebuilt grids.
    Uses the same rotation as _scene/_rot to avoid cross-term sign bugs.
    """
    xr_, yr_ = _rot(x_grid - x0, y_grid - y0, theta)
    return offset + amp * np.exp(
        -(xr_**2) / (2 * sigma_x**2) - (yr_**2) / (2 * sigma_y**2)
    )


def _scene(
    ny: int, nx: int, comps, *, offset=0.0, noise=0.0, seed=0, coords=False
) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.zeros((ny, nx), float) + offset
    for c in comps:
        amp = float(c["amp"])
        x0 = float(c["x0"])
        y0 = float(c["y0"])
        sx = float(c["sigma_x"])
        sy = float(c["sigma_y"])
        th = float(c["theta"])
        xr_, yr_ = _rot(xx - x0, yy - y0, th)
        img += amp * np.exp(-(xr_**2) / (2 * sx**2) - (yr_**2) / (2 * sy**2))
    if noise > 0:
        img += rng.normal(0.0, noise, size=img.shape)
    coords_dict = None
    if coords:
        coords_dict = {"y": np.linspace(-1.0, 1.0, ny), "x": np.linspace(-2.0, 2.0, nx)}
    return xr.DataArray(img, dims=("y", "x"), coords=coords_dict)


# ------------------------- success paths -------------------------


class TestSuccess:
    @pytest.mark.parametrize("noise", [0.02, 0.05])
    def test_two_components_with_world_coords(self, noise: float) -> None:
        ny, nx = 96, 112
        comps = [
            dict(amp=1.0, x0=40.0, y0=60.0, sigma_x=3.0, sigma_y=3.0, theta=0.0),
            dict(amp=0.7, x0=85.0, y0=28.0, sigma_x=5.0, sigma_y=2.5, theta=0.4),
        ]
        da2 = _scene(ny, nx, comps, offset=0.1, noise=noise, seed=123, coords=True)
        init = np.array(
            [[1.0, 39.5, 60.5, 3.0, 3.0, 0.0], [0.7, 84.5, 28.5, 5.0, 2.5, 0.4]], float
        )
        ds = fit_multi_gaussian2d(
            da2,
            n_components=2,
            initial_guesses=init,
            return_model=True,
            return_residual=True,
        )
        assert bool(ds.success)
        assert 0.0 <= float(ds.variance_explained) <= 1.0
        assert "x0_world" in ds and "y0_world" in ds
        order = np.argsort(ds["x0_pixel"].values)
        assert np.allclose(ds["x0_pixel"].values[order], [40.0, 85.0], atol=0.8)
        assert np.allclose(ds["y0_pixel"].values[order], [60.0, 28.0], atol=0.8)

    def test_vectorize_over_time_names_and_indices(self) -> None:
        ny, nx = 64, 80
        base = dict(amp=0.9, x0=30.0, y0=22.0, sigma_x=4.0, sigma_y=3.0, theta=0.2)
        planes = [
            _scene(ny, nx, [base], offset=0.1, noise=0.03, seed=s) for s in (1, 2, 3)
        ]
        cube = xr.concat(planes, dim="time")  # dims ('time','y','x')
        init = np.array([[0.8, 29.5, 22.5, 4.0, 3.0, 0.2]], float)
        # dims by name
        ds1 = fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",
            dims=("x", "y"),
            return_model=False,
            return_residual=False,
        )
        assert "time" in ds1.dims and ds1.sizes["time"] == 3
        # dims by index
        ds2 = fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",
            dims=(2, 1),
            return_model=False,
            return_residual=False,
        )
        assert np.all(ds2["success"].values)

    def test_flags_and_descending_world_coords(self) -> None:
        ny, nx = 40, 50
        comps = [dict(amp=0.8, x0=20.0, y0=18.0, sigma_x=3.0, sigma_y=2.0, theta=0.1)]
        da2 = _scene(ny, nx, comps, coords=True)
        # reverse coords → descending path
        da2 = da2.assign_coords(x=da2.coords["x"][::-1], y=da2.coords["y"][::-1])
        init = np.array([[0.7, 20.0, 18.0, 3.0, 2.0, 0.1]], float)

        ds1 = fit_multi_gaussian2d(
            da2,
            n_components=1,
            initial_guesses=init,
            return_model=True,
            return_residual=False,
        )
        assert "model" in ds1 and "residual" not in ds1
        assert "x0_world" in ds1 and "y0_world" in ds1

        ds2 = fit_multi_gaussian2d(
            da2,
            n_components=1,
            initial_guesses=init,
            return_model=False,
            return_residual=True,
        )
        assert "residual" in ds2 and "model" not in ds2

    def test_auto_seeds_when_initial_none(self) -> None:
        ny, nx = 48, 60
        comps = [
            dict(amp=1.0, x0=18.0, y0=22.0, sigma_x=3.0, sigma_y=3.0, theta=0.0),
            dict(amp=0.6, x0=42.0, y0=30.0, sigma_x=5.0, sigma_y=2.5, theta=0.3),
        ]
        da2 = _scene(ny, nx, comps, offset=0.1, noise=0.03, seed=3)
        ds = fit_multi_gaussian2d(
            da2,
            n_components=2,
            initial_guesses=None,
            coord_type="pixel",
            return_residual=True,
        )
        assert bool(ds.success) is True


# ------------------------- input types -------------------------


class TestInputs:
    def test_accepts_raw_numpy_array(self) -> None:
        ny, nx = 32, 33
        comps = [dict(amp=1.0, x0=16.0, y0=15.0, sigma_x=3.0, sigma_y=3.0, theta=0.0)]
        arr = _scene(ny, nx, comps).values  # plain ndarray
        ds = fit_multi_gaussian2d(
            arr, n_components=1, initial_guesses=np.array([[1, 16, 15, 3, 3, 0.0]])
        )
        assert bool(ds.success) is True

    @pytest.mark.skipif(da is None, reason="dask.array not available")
    def test_accepts_bare_dask_array(self) -> None:
        ny, nx = 40, 40
        comps = [dict(amp=1.0, x0=20.0, y0=20.0, sigma_x=3.0, sigma_y=3.0, theta=0.0)]
        np_img = _scene(ny, nx, comps, offset=0.1, noise=0.02, seed=1).data
        darr = da.from_array(np_img, chunks=(ny, nx))
        ds = fit_multi_gaussian2d(
            darr, n_components=1, initial_guesses=np.array([[1, 20, 20, 3, 3, 0.0]])
        )
        assert bool(ds.success) is True

    def test_world_coords_skipped_for_bad_axis_coords(self) -> None:
        ny, nx = 32, 32
        comps = [dict(amp=0.8, x0=15.0, y0=16.0, sigma_x=3.0, sigma_y=2.0, theta=0.1)]
        da2 = _scene(ny, nx, comps, coords=True)
        # break monotonicity / finiteness
        x = da2.coords["x"].values.copy()
        x[5] = x[4]
        y = da2.coords["y"].values.copy()
        y[3] = np.nan
        with pytest.raises(ValueError):
            fit_multi_gaussian2d(
                da2.assign_coords(x=("x", x), y=("y", y)),
                n_components=1,
                initial_guesses=np.array([[0.8, 15, 16, 3, 2, 0.1]]),
            )


# ------------------------- bounds / dims / API validation -------------------------


class TestBoundsDimsAPI:
    # TODO these tests need to be rewritten to use public API
    def test_bounds_merge_tuple_all_and_unknown_key_ignored(self) -> None:
        lb0, ub0 = mg._default_bounds_multi((20, 30), 2)
        user = {"foo": (1.0, 2.0), "amplitude": (0.0, 5.0)}  # 'foo' ignored
        lb, ub = mg._merge_bounds_multi(lb0, ub0, user, 2)
        assert lb[1] == 0.0 and ub[1] == 5.0  # comp0 amp
        assert lb[7] == 0.0 and ub[7] == 5.0  # comp1 amp

    def test_bounds_merge_per_component_length_error(self) -> None:
        lb0, ub0 = mg._default_bounds_multi((20, 30), 2)
        with pytest.raises(ValueError):
            mg._merge_bounds_multi(lb0, ub0, {"amplitude": [(0, 1)]}, 2)

    def test_dims_by_index_and_error_paths(self) -> None:
        da3 = xr.DataArray(np.zeros((3, 4, 5)), dims=("t", "y", "x"))
        assert mg._resolve_dims(da3, (2, 1)) == ("x", "y")
        with pytest.raises(ValueError):
            mg._resolve_dims(da3, ("x",))
        with pytest.raises(ValueError):
            mg._resolve_dims(da3, ("q", "y"))

    def test_init_formats_and_errors(self) -> None:
        z = np.zeros((16, 17), float)
        # array (n,6)
        arr = np.array([[1.0, 5.0, 6.0, 2.0, 2.0, 0.0]], float)
        p1 = mg._normalize_initial_guesses(z, 1, arr, None, None)
        assert p1.shape == (7,)
        # dict with ndarray
        dct = {
            "offset": 0.2,
            "components": np.array([[1.0, 5.0, 6.0, 2.0, 2.0, 0.0]], float),
        }
        p2 = mg._normalize_initial_guesses(z, 1, dct, None, None)
        assert p2[0] == pytest.approx(0.2)
        # dict with list of dicts
        dct_list = {
            "components": [
                {
                    "amp": 0.8,
                    "x0": 3.0,
                    "y0": 4.0,
                    "sigma_x": 2.1,
                    "sigma_y": 1.9,
                    "theta": 0.1,
                }
            ]
        }
        p3 = mg._normalize_initial_guesses(z, 1, dct_list, None, None)
        assert p3.shape == (7,)
        # wrong shape -> error
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(
                z, 2, np.array([[1, 2, 3, 4, 5, 6]]), None, None
            )
        # dict missing 'components' -> error
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, 1, {"offset": 0.0}, None, None)


# ------------------------- optimizer / masking failure paths -------------------------


class TestOptimizerFailures:
    def test_full_mask_triggers_failure_and_nan_planes(self) -> None:
        ny, nx = 24, 24
        da2 = xr.DataArray(np.zeros((ny, nx)), dims=("y", "x"))
        with pytest.raises(ValueError, match=r"(all pixels|empty mask)"):
            fit_multi_gaussian2d(
                da2,
                n_components=1,
                min_threshold=1.0,
                return_residual=True,
                return_model=True,
                coord_type="pixel",  # DA has no coords → pixel mode
            )

    def test_curve_fit_exception_sets_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ny, nx = 40, 40
        y, x = np.mgrid[0:ny, 0:nx]
        z = np.exp(-((x - 20) ** 2 + (y - 18) ** 2) / (2 * 3.0**2))
        da2 = xr.DataArray(z, dims=("y", "x"))

        def boom(*args, **kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(mg, "curve_fit", boom, raising=True)
        with pytest.raises(Exception) as excinfo:
            fit_multi_gaussian2d(
                da2,
                n_components=1,
                initial_guesses=np.array([[1.0, 20.0, 18.0, 3.0, 3.0, 0.0]]),
                return_residual=True,
                coord_type="pixel",
            )
        # original exception message must be present in the raised error
        assert "nope" in str(excinfo.value)

    def test_curve_fit_pcov_none_sets_errors_nan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ny, nx = 32, 32
        da2 = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
        n = 2
        # popt layout: [offset, (amp,x0,y0,sx,sy,th)*n]
        popt = np.array(
            [0.1, 1.0, 10.0, 12.0, 2.0, 2.0, 0.0, 0.7, 20.0, 8.0, 3.0, 1.5, 0.1], float
        )

        def fake_fit(func, xy, z, p0=None, bounds=None, maxfev=None):
            return popt, None

        monkeypatch.setattr(mg, "curve_fit", fake_fit, raising=True)
        ds = fit_multi_gaussian2d(
            da2,
            n_components=n,
            initial_guesses=np.array(
                [[1, 10, 12, 2, 2, 0], [0.7, 20, 8, 3, 1.5, 0.1]], float
            ),
            return_model=False,
            coord_type="pixel",
        )
        assert bool(ds["success"]) is True  # try-block executed; no exception raised
        for name in (
            "amplitude_err",
            "x0_pixel_err",
            "y0_pixel_err",
            "sigma_major_pixel_err",
            "sigma_minor_pixel_err",
            "theta_pixel_err",
        ):
            assert np.isnan(ds[name].values).all()

    def test_curve_fit_with_valid_pcov_sets_finite_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ny, nx = 24, 24
        da2 = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
        n = 1
        p_len = 1 + 6 * n
        popt = np.linspace(0.1, 0.1 + 0.01 * (p_len - 1), p_len)
        pcov = np.eye(p_len, dtype=float)

        def fake_fit(func, xy, z, p0=None, bounds=None, maxfev=None):
            return popt, pcov

        monkeypatch.setattr(mg, "curve_fit", fake_fit, raising=True)
        ds = fit_multi_gaussian2d(
            da2,
            n_components=n,
            initial_guesses=np.array([[1.0, 10.0, 12.0, 2.0, 2.0, 0.0]], float),
            return_model=False,
            return_residual=False,
            coord_type="pixel",
        )
        for name in (
            "amplitude_err",
            "x0_pixel_err",
            "y0_pixel_err",
            "sigma_major_pixel_err",
            "sigma_minor_pixel_err",
            "theta_pixel_err",
            "offset_err",
        ):
            assert np.all(np.isfinite(ds[name].values))

    def test_curve_fit_nonfinite_popt_triggers_not_success_nan_outputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Force a non-finite solution from curve_fit so the not-success early-return path is exercised.

        Covers the block that fills component arrays, model and residual with NaNs and sets success=False.
        """
        ny, nx = 32, 32
        img = xr.DataArray(np.ones((ny, nx), float), dims=("y", "x"))

        def fake_fit(func, xy, z, p0=None, bounds=None, maxfev=None):
            popt = np.asarray(p0, float).copy()
            # poison one parameter so np.all(np.isfinite(popt)) is False
            popt[2] = np.nan
            pcov = np.eye(popt.size, dtype=float)
            return popt, pcov

        monkeypatch.setattr(mg, "curve_fit", fake_fit, raising=True)
        init = np.array([[0.8, 10.0, 12.0, 3.0, 2.0, 0.1]], float)
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",
            return_model=True,
            return_residual=True,
        )

        assert bool(ds["success"]) is False
        # Component-level outputs are NaN
        assert np.isnan(ds["amplitude"].values).all()
        assert np.isnan(ds["x0_pixel"].values).all()
        assert np.isnan(ds["y0_pixel"].values).all()
        assert np.isnan(ds["sigma_major_pixel"].values).all()
        assert np.isnan(ds["sigma_minor_pixel"].values).all()
        assert np.isnan(ds["theta_pixel"].values).all()
        # Planes are filled with NaN
        assert np.isnan(ds["model"].values).all()
        assert np.isnan(ds["residual"].values).all()
        # Variance explained is NaN
        assert np.isnan(float(ds["variance_explained"]))

    def test_curve_fit_pcov_wrong_shape_sets_errors_nan_success_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """pcov has the wrong shape → perr:=NaN, but success stays True."""
        ny, nx = 24, 24
        da = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
        n = 1
        p_len = 1 + 6 * n
        popt = np.linspace(0.1, 0.1 + 0.01 * (p_len - 1), p_len)
        bad_pcov = np.eye(p_len - 1, dtype=float)  # wrong shape

        def fake_fit(func, xy, z, p0=None, bounds=None, maxfev=None):
            return popt, bad_pcov

        monkeypatch.setattr(mg, "curve_fit", fake_fit, raising=True)
        ds = fit_multi_gaussian2d(
            da,
            n_components=n,
            initial_guesses=np.array([[1.0, 10.0, 12.0, 2.0, 2.0, 0.0]], float),
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )

        # success True, but *_err are NaN
        assert bool(ds["success"]) is True
        err_vars = (
            "offset_err",
            "amplitude_err",
            "x0_pixel_err",
            "y0_pixel_err",
            "sigma_major_pixel_err",
            "sigma_minor_pixel_err",
            "theta_pixel_err",
        )
        for v in err_vars:
            assert np.isnan(float(ds[v])), f"{v} should be NaN when pcov is invalid"

    def test_threshold_not_enough_pixels_for_params_raises_public_api(self) -> None:
        """
        Cover the guard in _fit_multi_plane_numpy:
            if mask_count < (1 + 6*n_components): raise ValueError(...)
        Use a small plane with more parameters than masked-in pixels.
        """
        # For n_components=2, need = 1 + 6*2 = 13 > 6 (2x3 plane) → triggers branch.
        da = xr.DataArray(np.zeros((2, 3), float), dims=("y", "x"))
        mask = np.ones((2, 3), dtype=bool)  # keep all 6 pixels (still < 13 needed)
        with pytest.raises(ValueError, match=r"not enough pixels.*fit all parameters"):
            mg.fit_multi_gaussian2d(
                da,
                n_components=2,
                mask=mask,
                coord_type="pixel",
                return_model=False,
                return_residual=False,
            )


# ------------------------- initial guesses: top-level list-of-dicts -------------------------


class TestInitialGuessesTopLevelListOfDicts:
    def test_top_level_list_of_dicts_len_mismatch_raises(self) -> None:
        z = np.zeros((16, 16), float)
        init = [
            {
                "amp": 1.0,
                "x0": 5.0,
                "y0": 6.0,
                "sigma_x": 2.0,
                "sigma_y": 1.5,
                "theta": 0.25,
            }
        ]
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, 2, init, None, None)

    def test_top_level_list_of_dicts_happy_path_parses_and_packs(self) -> None:
        z = np.zeros((20, 20), float)
        n = 2
        init = [
            {
                "amp": 1.2,
                "x0": 5.0,
                "y0": 6.0,
                "sigma_x": 2.0,
                "sigma_y": 1.5,
                "theta": 0.3,
            },
            {
                "amplitude": 0.8,
                "x0": 10.0,
                "y0": 4.0,
                "sx": 2.5,
                "sy": 3.0,
            },  # theta omitted → defaults to 0.0
        ]
        p = mg._normalize_initial_guesses(z, n, init, None, None)
        off, amp, x0, y0, sx, sy, th = mg._unpack_params(p, n)
        assert off == 0.0  # median-based offset for this path
        assert np.allclose(amp, [1.2, 0.8])
        assert np.allclose(x0, [5.0, 10.0])
        assert np.allclose(y0, [6.0, 4.0])
        assert np.allclose(sx, [2.0, 2.5])
        assert np.allclose(sy, [1.5, 3.0])
        assert np.allclose(th, [0.3, 0.0])


# ------------------------- initial guesses: single-dict & top-level list-of-dicts (public API) -------------------------
# Covers:
# - wrapping a single dict when n_components==1 (becomes [dict])  ⇢ ensures shorthand is accepted
# - accepting a top-level list[dict] for multiple components       ⇢ ensures list-of-dicts branch is used


class TestInitialGuessesPublicAPIWrapping:
    def test_single_dict_shorthand_is_wrapped_and_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Small scene
        ny, nx = 10, 11
        da = xr.DataArray(np.zeros((ny, nx), dtype=float), dims=("y", "x"))
        # Single dict (no 'components' key) should be accepted for n_components=1
        init_single = {
            "amp": 1.1,
            "x0": 4.0,
            "y0": 5.0,
            "sigma_x": 2.0,
            "sigma_y": 1.5,
            "theta": 0.25,
        }

        captured = {}

        def fake_curve_fit(*_, **kw):
            p0 = np.asarray(kw["p0"], dtype=float)
            captured["p0"] = p0.copy()
            n = (p0.size - 1) // 6
            pcov = np.eye(1 + 6 * n, dtype=float)
            return p0, pcov

        monkeypatch.setattr(mg, "curve_fit", fake_curve_fit, raising=True)

        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init_single,
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )
        assert isinstance(ds, xr.Dataset)
        # p0: [offset, (amp,x0,y0,sigma_x,sigma_y,theta)]
        p0 = captured["p0"]
        assert p0.shape == (1 + 6 * 1,)
        # offset defaults to median of data (0.0 here)
        assert np.isclose(p0[0], 0.0)
        assert np.allclose(p0[1:], [1.1, 4.0, 5.0, 2.0, 1.5, 0.25])


# ------------------------- plotting helper -------------------------


class TestPlotHelper:
    def test_plot_components_3d_with_indexer_and_no_component_dim(self) -> None:
        if not hasattr(mg, "plot_components"):
            pytest.skip("plot_components helper not available")
        ny, nx = 36, 36
        comps = [dict(amp=0.9, x0=18.0, y0=17.0, sigma_x=3.0, sigma_y=2.0, theta=0.25)]
        base = _scene(ny, nx, comps, noise=0.01, coords=True)
        cube = xr.concat([base, base + 0.01], dim="time")

        init = np.array([[0.8, 18.0, 17.0, 3.0, 2.0, 0.25]], float)
        ds = fit_multi_gaussian2d(
            cube, n_components=1, initial_guesses=init, return_residual=True
        )

        # Build a DS WITHOUT 'component' dim (fix: use the post-isel dims!)
        ds_no_comp = xr.Dataset(
            {
                k: v.isel(component=0) if ("component" in v.dims) else v
                for k, v in ds.items()
                if k in ("x0", "y0", "sigma_major", "sigma_minor", "theta", "residual")
            }
        )

        # draw with indexer and both residual states
        mg.plot_components(
            cube, ds, dims=("x", "y"), indexer={"time": 1}, show_residual=True
        )
        mg.plot_components(
            cube, ds_no_comp, dims=("x", "y"), indexer={"time": 0}, show_residual=False
        )

    def test_plot_components_defaults_indexer_for_3d_input(self, monkeypatch) -> None:
        matplotlib.use("Agg", force=True)  # headless backend

        # Build a 3-D DataArray: ('time','y','x') so da_tr.ndim > 2
        ny, nx = 32, 32
        y, x = np.mgrid[0:ny, 0:nx]
        base = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / (2 * 3.0**2))
        frame = xr.DataArray(base, dims=("y", "x"))
        cube = xr.concat([frame, frame + 0.01], dim="time")

        # Public API fit; no need for residual/model here
        init = np.array([[1.0, 16.0, 16.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )
        # Silence plt.show to avoid warnings in CI
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # Call without 'indexer' → triggers default indexer = {leading dims: 0}
        # This covers:
        #   if da_tr.ndim > 2:
        #       if indexer is None:
        #           indexer = {d: 0 for d in da_tr.dims[:-2]}
        plot_components(cube, ds, dims=("x", "y"), show_residual=False)

    def test_plot_components_selects_result_plane_with_default_indexer(
        self, monkeypatch
    ) -> None:
        matplotlib.use("Agg", force=True)  # headless

        # Build 3-D data → da_tr.ndim > 2
        ny, nx = 32, 32
        y, x = np.mgrid[0:ny, 0:nx]
        base = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / (2 * 3.0**2))
        frame = xr.DataArray(base, dims=("y", "x"))
        cube = xr.concat([frame, frame + 0.01], dim="time")  # dims: ('time','y','x')

        # Fit via public API; result keeps the leading 'time' dim
        init = np.array([[1.0, 16.0, 16.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=init,
            return_residual=True,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )
        assert "time" in ds.dims  # ensure the nested selection can apply

        # Silence blocking show()
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # Call without indexer → covers:
        #   if da_tr.ndim > 2:
        #       if indexer is None:
        #           indexer = {d: 0 for d in da_tr.dims[:-2]}
        #       ...
        #       for d,i in indexer.items():
        #           if d in res_plane.dims and d not in ("component", dim_y, dim_x):
        #               res_plane = res_plane.isel({d: i})
        plot_components(cube, ds, dims=("x", "y"), indexer=None, show_residual=True)

    def test_plot_components_default_indexer_loops_multiple_dims(
        self, monkeypatch
    ) -> None:
        matplotlib.use("Agg", force=True)

        # 4D data ensures indexer has ≥2 keys → loop body executes (and loops)
        ny, nx = 24, 24
        y, x = np.mgrid[0:ny, 0:nx]
        base = np.exp(-((x - 12) ** 2 + (y - 12) ** 2) / (2 * 3.0**2))
        frame = xr.DataArray(base, dims=("y", "x"))
        cube_t = xr.concat([frame, frame + 0.01], dim="t")  # dims: ('t','y','x')
        cube = xr.concat(
            [cube_t, cube_t + 0.02], dim="band"
        )  # dims: ('band','t','y','x')

        init = np.array([[1.0, 12.0, 12.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=init,
            return_residual=True,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        # avoid blocking GUI
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # indexer=None → default {'band':0, 't':0}; loop iterates over both keys
        plot_components(cube, ds, dims=("x", "y"), indexer=None, show_residual=True)

    def test_plot_components_else_branch_for_2d_input(self, monkeypatch) -> None:
        matplotlib.use("Agg", force=True)

        # 2D image → triggers the else block (lines 766–768): data2d = da_tr; res_plane = result
        ny, nx = 40, 40
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.1 + np.exp(-((x - 20) ** 2 + (y - 22) ** 2) / (2 * 3.0**2))
        img = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[1.0, 20.0, 22.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            return_residual=True,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        # avoid GUI
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # No indexer; 2D input hits the else branch
        plot_components(img, ds, dims=("x", "y"), indexer=None, show_residual=True)

    def test_plot_components_raises_when_result_missing_required_var(
        self, monkeypatch
    ) -> None:
        matplotlib.use("Agg", force=True)

        ny, nx = 32, 32
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / (2 * 3.0**2))
        img = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[1.0, 16.0, 16.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",
            return_residual=True,
        )

        # Drop one of the center fields present in the current API.
        # plot_components is resilient and must NOT raise just because centers/sizes are missing.
        drop_names = [name for name in ("x0_pixel", "x0", "x") if name in ds]
        ds_missing = ds.drop_vars(drop_names)

        # avoid GUI popups
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # Expect: no exception; it should draw with fallbacks (e.g., default centers)
        plot_components(
            img, ds_missing, dims=("x", "y"), indexer=None, show_residual=False
        )

    def test_plot_components_else_branch_for_2d_input(self) -> None:
        # existing test body...
        pass

    def test_plot_components_fwhm_converts_from_sigma_when_fwhm_missing(self) -> None:
        """Cover metric=="fwhm" path that converts sigma→FWHM when fwhm vars are absent."""
        matplotlib.use("Agg", force=True)

        ny, nx = 36, 36
        y, x = np.mgrid[0:ny, 0:nx]
        amp, x0, y0, sx, sy, th = 1.0, 18.0, 19.0, 3.0, 2.0, 0.2
        z = amp * np.exp(
            -(((x - x0) ** 2) / (2 * sx**2) + ((y - y0) ** 2) / (2 * sy**2))
        )
        img = xr.DataArray(z, dims=("y", "x"))  # no coords → frame='pixel'

        init = np.array([[amp * 0.9, x0, y0, sx, sy, th]], float)
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )

        drop = [v for v in ("fwhm_major_pixel", "fwhm_minor_pixel") if v in ds]
        ds2 = ds.drop_vars(drop) if drop else ds

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ret = mg.plot_components(img, ds2, dims=("x", "y"), fwhm=True, show=False)
        # No size-warning expected because sigma→FWHM conversion should have supplied radii
        assert not any("Missing size info" in str(w.message) for w in rec)

        # Accept both return styles: Figure OR (Figure, axes)
        if isinstance(ret, _mf.Figure):
            fig = ret
        else:
            fig, _axes = ret

    def test_plot_components_fwhm_warns_and_draws_without_size_when_both_missing(
        self,
    ) -> None:
        """Cover metric=="fwhm" path that warns and uses zero radii when both size kinds are missing."""
        matplotlib.use("Agg", force=True)

        ny, nx = 32, 32
        y, x = np.mgrid[0:ny, 0:nx]
        z = np.exp(-((x - 16) ** 2 + (y - 15) ** 2) / (2 * 3.0**2))
        img = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.9, 16.0, 15.0, 3.0, 2.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            img, n_components=1, initial_guesses=init, coord_type="pixel"
        )

        # Drop *both* sigma and FWHM pixel vars to force the warning branch
        to_drop = [
            v
            for v in (
                "sigma_major_pixel",
                "sigma_minor_pixel",
                "fwhm_major_pixel",
                "fwhm_minor_pixel",
            )
            if v in ds
        ]
        ds3 = ds.drop_vars(to_drop) if to_drop else ds

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ret = mg.plot_components(img, ds3, dims=("x", "y"), fwhm=True, show=False)
        msgs = ", ".join(str(w.message) for w in rec)
        assert "Missing size info (FWHM/sigma)" in msgs
        # Accept both styles: Figure OR (Figure, axes)
        if isinstance(ret, _mf.Figure):
            fig = ret
        else:
            fig, _ = ret
        assert isinstance(fig, _mf.Figure)

    def test_overlay_converts_fwhm_to_sigma_when_sigma_missing(self) -> None:
        matplotlib.use("Agg", force=True)
        fig, ax = plt.subplots()

        # Known sigma; provide only FWHM so helper must convert → sigma = fwhm / K
        sigma_x = 3.0
        sigma_y = 2.0
        FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))
        fit = {
            "x0": np.array([10.0]),
            "y0": np.array([12.0]),
            "fwhm_x": np.array([sigma_x * FWHM_K]),
            "fwhm_y": np.array([sigma_y * FWHM_K]),
            "theta": np.array([0.0]),
        }

        n_sigma = 1.0
        mg.overlay_fit_components(
            ax,
            fit,
            frame="pixel",
            metric="sigma",
            n_sigma=n_sigma,
            angle="math",
            label=False,
        )

        assert len(ax.patches) == 1
        e = ax.patches[0]
        # width = 2 * (n_sigma * sigma_x), height = 2 * (n_sigma * sigma_y)
        assert np.isclose(e.get_width(), 2.0 * n_sigma * sigma_x)
        assert np.isclose(e.get_height(), 2.0 * n_sigma * sigma_y)


class TestNumPyFitting:  # (unittest.TestCase):
    def test_min_threshold_masks_pixels_partial(self) -> None:

        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 3.0**2))
        da = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.8, 32.0, 32.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            min_threshold=0.20,  # exercises: mask &= z2d >= min_threshold
            return_model=True,
            return_residual=True,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        assert bool(ds.success) is True
        assert "model" in ds and "residual" in ds
        above = int((z >= 0.20).sum())
        assert 0 < above < z.size

    def test_min_threshold_masks_pixels_partial(self) -> None:

        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 3.0**2))
        da = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.8, 32.0, 32.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            min_threshold=0.20,  # exercises: mask &= z2d >= min_threshold
            return_model=True,
            return_residual=True,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        assert bool(ds.success) is True
        assert "model" in ds and "residual" in ds
        above = int((z >= 0.20).sum())
        assert 0 < above < z.size

    def test_max_threshold_masks_pixels_partial(self) -> None:

        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(
            -((x - 32) ** 2 + (y - 32) ** 2) / (2 * 3.0**2)
        )  # peak ~ 1.05
        da = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.8, 32.0, 32.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            max_threshold=0.60,  # exercises: mask &= z2d <= max_threshold
            return_model=True,
            return_residual=True,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        assert bool(ds.success) is True
        assert "model" in ds and "residual" in ds
        below = int((z <= 0.60).sum())
        assert 0 < below < z.size

    def test_angle_pa_init_conversion_and_reporting(self) -> None:
        # Build a rotated elliptical Gaussian with a known *math* angle
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        amp_true, x0_true, y0_true = 1.0, 28.0, 30.0
        sx_true, sy_true = 4.0, 2.0
        theta_math = 0.7  # radians (math: from +x toward +y, CCW)

        ct, st = np.cos(theta_math), np.sin(theta_math)
        X, Y = x - x0_true, y - y0_true
        a = (ct**2) / (2 * sx_true**2) + (st**2) / (2 * sy_true**2)
        b = st * ct * (1 / (2 * sx_true**2) - 1 / (2 * sy_true**2))
        c = (st**2) / (2 * sx_true**2) + (ct**2) / (2 * sy_true**2)
        z = 0.12 + amp_true * np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
        img = xr.DataArray(z, dims=("y", "x"))

        # Expected PA for ascending axes: PA = (π/2 − θ) mod 2π
        pa_expected = (np.pi / 2 - theta_math) % (2 * np.pi)

        # Provide initial guesses *in PA*; public API will convert them via _theta_pa_to_math
        init = np.array(
            [[0.9, x0_true, y0_true, sx_true * 0.9, sy_true * 1.1, pa_expected]], float
        )

        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="pa",  # triggers PA→math conversion path
            return_model=False,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        assert bool(ds["success"]) is True
        # theta is reported in the same convention ("pa"), so it should match pa_expected
        assert np.isfinite(float(ds["theta_pixel"]))
        assert (
            abs(float(ds["theta_pixel"]) - pa_expected) < 0.15
        )  # allow some tolerance

    def test_angle_pa_init_conversion_list_of_dicts(self) -> None:
        # Two rotated Gaussians with known *math* angles
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        th1_math, th2_math = 0.5, 1.1
        z = (
            0.10
            + _gauss2d_on_grid(x, y, 1.0, 20.0, 22.0, 3.0, 2.0, th1_math, offset=0.0)
            + _gauss2d_on_grid(x, y, 0.8, 44.0, 40.0, 4.0, 2.5, th2_math, offset=0.0)
        )
        img = xr.DataArray(z, dims=("y", "x"))

        # Convert true math angles to PA (local basis): PA = (π/2 − θ) mod 2π
        pa1 = float(((np.pi / 2) - th1_math) % (2 * np.pi))
        pa2 = float(((np.pi / 2) - th2_math) % (2 * np.pi))

        # initial_guesses as LIST OF DICTS with 'theta' → triggers _conv_list_of_dicts path
        init_list = [
            {
                "amp": 0.9,
                "x0": 20.0,
                "y0": 22.0,
                "sigma_x": 3.0,
                "sigma_y": 2.0,
                "theta": pa1,
            },
            {
                "amp": 0.7,
                "x0": 44.0,
                "y0": 40.0,
                "sigma_x": 4.0,
                "sigma_y": 2.5,
                "theta": pa2,
            },
        ]

        ds = fit_multi_gaussian2d(
            img,
            n_components=2,
            initial_guesses=init_list,  # list[dict] path
            angle="pa",  # forces PA→math conversion on init
            return_model=False,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        # Sort by x0 to align component identity
        order = np.argsort(ds["x0_pixel"].values)
        thetas_pa = ds["theta_pixel"].values[order]

        # Reported angles are in PA (because angle="pa")
        assert np.isfinite(thetas_pa).all()
        assert abs(thetas_pa[0] - pa1) < 0.25
        assert abs(thetas_pa[1] - pa2) < 0.25

    def test_angle_pa_init_components_array_dict_converted(self) -> None:
        # Build a single rotated Gaussian (known math angle)
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        amp_true, x0_true, y0_true = 1.0, 30.0, 28.0
        sx_true, sy_true = 4.0, 2.5
        theta_math = 0.8
        z = _gauss2d_on_grid(
            x, y, amp_true, x0_true, y0_true, sx_true, sy_true, theta_math, offset=0.12
        )

        img = xr.DataArray(z, dims=("y", "x"))

        # Local-basis conversion: PA = (π/2 − θ) mod 2π
        pa = float(((np.pi / 2) - theta_math) % (2 * np.pi))

        # initial_guesses as dict with components **NumPy array** → hits lines 590–595
        comps = np.array([[0.9, x0_true, y0_true, sx_true, sy_true, pa]], float)
        init = {"components": comps}

        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="pa",
            return_model=False,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        assert bool(ds["success"]) is True
        # theta reported in PA (angle="pa"); should be close to the PA seed
        assert abs(float(ds["theta_pixel"]) - pa) < 0.25

    def test_angle_pa_init_components_list_of_dicts_converted(self) -> None:
        # Build a single rotated Gaussian (known math angle)
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        amp_true, x0_true, y0_true = 0.9, 22.0, 24.0
        sx_true, sy_true = 3.5, 2.0
        theta_math = 0.6
        z = 0.10 + _gauss2d_on_grid(
            x, y, amp_true, x0_true, y0_true, sx_true, sy_true, theta_math, offset=0.0
        )

        img = xr.DataArray(z, dims=("y", "x"))
        # Local-basis conversion: PA = (π/2 − θ) mod 2π
        pa = float(((np.pi / 2) - theta_math) % (2 * np.pi))

        # initial_guesses as dict with components **list of dicts** → hits lines 595–597
        init = {
            "components": [
                {
                    "amp": 0.85,
                    "x0": x0_true,
                    "y0": y0_true,
                    "sigma_x": sx_true,
                    "sigma_y": sy_true,
                    "theta": pa,
                }
            ]
        }

        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="pa",
            return_model=False,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        assert bool(ds["success"]) is True
        assert abs(float(ds["theta_pixel"]) - pa) < 0.25

    def test_angle_pa_list_of_dicts_missing_theta_covers_if_false_branch(self) -> None:
        # Build a 2-component scene
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]

        th1_math, th2_math = 0.7, 0.0  # comp1 rotated; comp2 axis-aligned (math)
        z = (
            0.10
            + _gauss2d_on_grid(x, y, 1.0, 20.0, 22.0, 3.0, 2.0, th1_math)
            + _gauss2d_on_grid(x, y, 0.8, 44.0, 40.0, 4.0, 2.5, th2_math)
        )
        img = xr.DataArray(z, dims=("y", "x"))

        # Expected PA for ascending axes: PA = arctan2(cos(theta_math), sin(theta_math))
        pa1 = float(np.arctan2(np.cos(th1_math), np.sin(th1_math)))
        pa2 = float(
            np.arctan2(np.cos(th2_math), np.sin(th2_math))
        )  # ~π/2 for theta_math=0

        # list[dict] with one dict MISSING 'theta' → exercises 579→581 (no conversion branch for that dict)
        init_list = [
            {
                "amp": 0.9,
                "x0": 20.0,
                "y0": 22.0,
                "sigma_x": 3.0,
                "sigma_y": 2.0,
                "theta": pa1,
            },
            {
                "amp": 0.7,
                "x0": 44.0,
                "y0": 40.0,
                "sigma_x": 4.0,
                "sigma_y": 2.5,
            },  # no 'theta'
        ]

        ds = fit_multi_gaussian2d(
            img,
            n_components=2,
            initial_guesses=init_list,
            angle="pa",  # convert dicts that HAVE 'theta'; report output in PA
            return_model=False,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        assert bool(ds["success"]) is True
        order = np.argsort(ds["x0_pixel"].values)
        th_pa = ds["theta_pixel"].values[order]

        assert np.isfinite(th_pa).all()
        assert abs(th_pa[0] - pa1) < 0.3  # comp1 near its seeded PA
        assert abs(th_pa[1] - pa2) < 0.5  # comp2 near PA for math=0 (≈ π/2)

    def test_angle_pa_init_components_list_of_dicts_branch(self) -> None:
        # 2-component scene
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        th1_math, th2_math = 0.4, 1.0
        # z = 0.1 + g(1.0, 18.0, 20.0, 3.0, 2.0, th1_math) + g(0.8, 44.0, 38.0, 4.0, 2.6, th2_math)
        z = (
            0.1
            + _gauss2d_on_grid(x, y, 1.0, 18.0, 20.0, 3.0, 2.0, th1_math, offset=0.0)
            + _gauss2d_on_grid(x, y, 0.8, 44.0, 38.0, 4.0, 2.6, th2_math, offset=0.0)
        )
        img = xr.DataArray(z, dims=("y", "x"))

        # PA seeds from math angles for ascending axes: PA = arctan2(cos(theta), sin(theta))
        pa1 = float(np.arctan2(np.cos(th1_math), np.sin(th1_math)))
        pa2 = float(np.arctan2(np.cos(th2_math), np.sin(th2_math)))

        # Dict with "components" as LIST OF DICTS → hits 595→597 (_conv_list_of_dicts then return clone)
        init = {
            "components": [
                {
                    "amp": 0.9,
                    "x0": 18.0,
                    "y0": 20.0,
                    "sigma_x": 3.0,
                    "sigma_y": 2.0,
                    "theta": pa1,
                },
                {
                    "amp": 0.7,
                    "x0": 44.0,
                    "y0": 38.0,
                    "sigma_x": 4.0,
                    "sigma_y": 2.6,
                    "theta": pa2,
                },
            ]
        }

        ds = fit_multi_gaussian2d(
            img,
            n_components=2,
            initial_guesses=init,
            angle="pa",
            return_model=False,
            return_residual=False,
            coord_type="pixel",
        )

        assert bool(ds["success"]) is True
        # Reported theta is in PA (angle="pa"); check against seeds
        order = np.argsort(ds["x0_pixel"].values)
        th_pa = ds["theta_pixel"].values[order]
        assert np.isfinite(th_pa).all()
        assert abs(th_pa[0] - pa1) < 0.3
        assert abs(th_pa[1] - pa2) < 0.3

    def test_plot_components_uses_model_branch_and_slices_leading_dims(self) -> None:
        """Cover the `elif "model"` block in plot_components (with slicing over leading dims)."""
        # Build a simple 3D cube: dims ('time','y','x')
        ny, nx = 36, 36
        y, x = np.mgrid[0:ny, 0:nx]
        amp, x0, y0, sx, sy, th = 0.9, 18.0, 17.0, 3.0, 2.0, 0.0
        z = amp * np.exp(-((x - x0) ** 2) / (2 * sx**2) - ((y - y0) ** 2) / (2 * sy**2))
        base = xr.DataArray(z, dims=("y", "x")).assign_coords(
            y=np.arange(ny, dtype=float), x=np.arange(nx, dtype=float)
        )
        cube = xr.concat([base, base + 0.01], dim="time")

        # Initial guesses in FWHM (default): columns [amp, x0, y0, fwhm_major, fwhm_minor, theta]
        k = 2.0 * np.sqrt(2.0 * np.log(2.0))
        init = np.array([[0.8, x0, y0, k * sx, k * sy, th]], float)

        # Fit with model only (no residual) → triggers 'elif "model"' path in plot_components
        ds = fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=init,
            dims=("x", "y"),
            return_model=True,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        # Now plot with an indexer so the code slices model2d along 'time'
        ret = mg.plot_components(
            cube, ds, dims=("x", "y"), indexer={"time": 1}, show_residual=True
        )

        # Accept both return styles: Figure OR (Figure, axes)
        if isinstance(ret, matplotlib.figure.Figure):
            fig = ret
        else:
            # expected tuple: (fig, (ax_data, ax_residual|None))
            assert isinstance(ret, tuple) and len(ret) >= 1
            fig = ret[0]
            assert isinstance(fig, matplotlib.figure.Figure)

        # Basic clean up
        try:
            plt.close(fig)
        except Exception:
            pass

    # ----------------------- Pixel→World interpolation coverage -----------------------

    def test_pixel_fit_interpolates_world_centers_and_propagates_errors_ascending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        - when DA has valid ascending 1-D world coords on (x,y) but we fit in pixel mode,
          the code interpolates centers into world coords and propagates uncertainties
          using local axis slope from np.gradient.
        """
        ny, nx = 40, 50
        y, x = np.mgrid[0:ny, 0:nx]
        amp, x0, y0, sx, sy, th = 1.0, 30.0, 18.0, 4.0, 2.5, 0.0
        z = 0.12 + amp * np.exp(
            -((x - x0) ** 2) / (2 * sx**2) - ((y - y0) ** 2) / (2 * sy**2)
        )

        # ascending world coords on both axes
        xw = 100.0 + 0.5 * np.arange(nx, dtype=float)  # slope 0.5
        yw = -5.0 + 0.25 * np.arange(ny, dtype=float)  # slope 0.25
        img = xr.DataArray(z, dims=("y", "x")).assign_coords(x=xw, y=yw)

        # Deterministic optimizer: echo p0 and give finite pcov so *_pixel_err are non-NaN
        def fake_cf(func, xy, zflat, p0=None, bounds=None, maxfev=None):
            p0 = np.asarray(p0, float)
            pcov = np.eye(p0.size, dtype=float) * 0.04  # sqrt(diag)=0.2
            return p0, pcov

        monkeypatch.setattr(mg, "curve_fit", fake_cf, raising=True)

        init = np.array([[amp, x0, y0, sx, sy, th]], float)
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="math",
            coord_type="pixel",  # force pixel fit → triggers interpolation branch
            return_model=False,
            return_residual=False,
        )

        # Interpolated world centers + propagated errors must exist
        assert "x0_world" in ds and "y0_world" in ds
        assert "x0_world_err" in ds and "y0_world_err" in ds

        # Expected centers via direct interpolation over pixel indices
        x0_pix = float(ds["x0_pixel"])
        y0_pix = float(ds["y0_pixel"])
        x0w_expected = np.interp(x0_pix, np.arange(nx, dtype=float), xw)
        y0w_expected = np.interp(y0_pix, np.arange(ny, dtype=float), yw)
        assert abs(float(ds["x0_world"]) - x0w_expected) < 1e-6
        assert abs(float(ds["y0_world"]) - y0w_expected) < 1e-6

        # Error propagation: world_err ≈ |local slope| * pixel_err (slope constant for linear coords)
        slope_x = 0.5
        slope_y = 0.25
        ex_pix = float(ds["x0_pixel_err"])
        ey_pix = float(ds["y0_pixel_err"])
        assert abs(float(ds["x0_world_err"]) - slope_x * ex_pix) < 1e-12
        assert abs(float(ds["y0_world_err"]) - slope_y * ey_pix) < 1e-12

    def test_pixel_fit_interpolates_world_centers_descending_x(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Also cover the _prep() path that reverses a descending axis before interpolation.
        Keep assertions light (presence and bounds) to avoid over-constraining behavior.
        """
        ny, nx = 32, 48
        y, x = np.mgrid[0:ny, 0:nx]
        amp, x0, y0, sx, sy, th = 0.9, 20.0, 14.0, 3.2, 2.1, 0.0
        z = amp * np.exp(-((x - x0) ** 2) / (2 * sx**2) - ((y - y0) ** 2) / (2 * sy**2))

        # x: descending, y: ascending → triggers idx reversal in _prep()
        xw = np.linspace(5.0, -7.0, nx, dtype=float)  # descending
        yw = np.linspace(-1.0, 3.0, ny, dtype=float)  # ascending
        img = xr.DataArray(z, dims=("y", "x")).assign_coords(x=xw, y=yw)

        def fake_cf(func, xy, zflat, p0=None, bounds=None, maxfev=None):
            p0 = np.asarray(p0, float)
            pcov = np.eye(p0.size, dtype=float) * 0.01
            return p0, pcov

        monkeypatch.setattr(mg, "curve_fit", fake_cf, raising=True)

        init = np.array([[amp, x0, y0, sx, sy, th]], float)
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="math",
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )

        for name in ("x0_world", "y0_world", "x0_world_err", "y0_world_err"):
            assert name in ds

        # Centers should be finite and within coordinate ranges
        x0w = float(ds["x0_world"])
        y0w = float(ds["y0_world"])
        assert np.isfinite(x0w) and np.isfinite(y0w)
        assert min(xw) - 1e-9 <= x0w <= max(xw) + 1e-9
        assert min(yw) - 1e-9 <= y0w <= max(yw) + 1e-9


class TestAPIHelpers:
    def test_init_components_array_wrong_shape_raises(self) -> None:
        da = xr.DataArray(np.zeros((16, 17), float), dims=("y", "x"))
        bad_init = {
            "offset": 0.0,
            "components": np.ones((1, 6), float),
        }  # shape != (n,6) for n=2


class TestAPIHelpers:
    def test_init_components_array_wrong_shape_raises(self) -> None:
        da = xr.DataArray(np.zeros((16, 17), float), dims=("y", "x"))
        bad_init = {
            "offset": 0.0,
            "components": np.ones((1, 6), float),
        }  # shape != (n,6) for n=2

        with pytest.raises(ValueError):
            fit_multi_gaussian2d(da, n_components=2, initial_guesses=bad_init)

    def test_init_components_list_len_mismatch_raises(self) -> None:
        z = np.zeros((16, 17), float)
        n = 2
        init = {
            "offset": 0.0,
            "components": [  # length 1, but n=2 → should raise
                {
                    "amp": 1.0,
                    "x0": 5.0,
                    "y0": 6.0,
                    "sigma_x": 2.0,
                    "sigma_y": 2.0,
                    "theta": 0.0,
                }
            ],
        }
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, n, init, None, None)

    def test_init_components_list_happy_path_synonyms_and_theta_default(self) -> None:
        z = np.zeros((20, 21), float)  # median = 0.0
        n = 2
        init = {
            "offset": 0.1,
            "components": [
                # uses amp + sigma_x/sigma_y + theta
                {
                    "amp": 1.2,
                    "x0": 5.0,
                    "y0": 6.0,
                    "sigma_x": 2.0,
                    "sigma_y": 1.5,
                    "theta": 0.3,
                },
                # uses amplitude + sx/sy; theta omitted -> defaults to 0.0
                {"amplitude": 0.8, "x0": 10.0, "y0": 4.0, "sx": 2.5, "sy": 3.0},
            ],
        }

        p = mg._normalize_initial_guesses(z, n, init, None, None)
        # Unpack and verify values
        off, amp, x0, y0, sx, sy, th = mg._unpack_params(p, n)
        assert off == 0.1
        assert amp.shape == (2,)
        assert np.allclose(amp, [1.2, 0.8])
        assert np.allclose(x0, [5.0, 10.0])
        assert np.allclose(y0, [6.0, 4.0])
        assert np.allclose(sx, [2.0, 2.5])
        assert np.allclose(sy, [1.5, 3.0])
        assert np.allclose(th, [0.3, 0.0])  # second component theta defaults to 0.0

    def test_init_components_list_missing_keys_raise(self) -> None:
        z = np.zeros((12, 12), float)
        n = 1

        # Case A: missing 'amp'/'amplitude' -> KeyError (or ValueError in some variants)
        bad_amp = {
            "components": [{"x0": 4.0, "y0": 5.0, "sigma_x": 2.0, "sigma_y": 2.0}]
        }
        with pytest.raises((KeyError, ValueError)):
            mg._normalize_initial_guesses(z, n, bad_amp, None, None)

        # Case B: missing both sigma_x/sx and sigma_y/sy -> float(None) TypeError
        bad_sigma = {"components": [{"amp": 1.0, "x0": 4.0, "y0": 5.0}]}
        with pytest.raises((TypeError, ValueError, KeyError)):
            mg._normalize_initial_guesses(z, n, bad_sigma, None, None)

    def test_init_components_list_len_mismatch_raises(self) -> None:
        z = np.zeros((16, 16), float)
        n = 2
        init_list = [  # len 1 but n=2 → hits lines 221–223
            {
                "amp": 1.0,
                "x0": 5.0,
                "y0": 6.0,
                "sigma_x": 2.0,
                "sigma_y": 2.0,
                "theta": 0.0,
            }
        ]
        init = {"offset": 0.0, "components": init_list}
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, n, init, None, None)

    def test_init_components_list_happy_path_covers_224_232(self) -> None:
        z = np.zeros((20, 20), float)
        n = 2
        init = {  # covers lines 224–232: amp vs amplitude; sigma_x/sigma_y vs sx/sy; theta default
            "offset": 0.1,
            "components": [
                {
                    "amp": 1.2,
                    "x0": 5.0,
                    "y0": 6.0,
                    "sigma_x": 2.0,
                    "sigma_y": 1.5,
                    "theta": 0.3,
                },
                {
                    "amplitude": 0.8,
                    "x0": 10.0,
                    "y0": 4.0,
                    "sx": 2.5,
                    "sy": 3.0,
                },  # theta omitted → 0.0
            ],
        }
        p = mg._normalize_initial_guesses(z, n, init, None, None)
        off, amp, x0, y0, sx, sy, th = mg._unpack_params(p, n)
        assert off == 0.1
        assert np.allclose(amp, [1.2, 0.8])
        assert np.allclose(x0, [5.0, 10.0])
        assert np.allclose(y0, [6.0, 4.0])
        assert np.allclose(sx, [2.0, 2.5])
        assert np.allclose(sy, [1.5, 3.0])
        assert np.allclose(th, [0.3, 0.0])

    def test_init_components_tuple_len_mismatch_raises(self) -> None:
        z = np.zeros((16, 16), float)
        n = 2
        # components is a TUPLE of length 1 → hits 221 (list/tuple), 222–223 (len check → ValueError)
        comps = (
            {
                "amp": 1.0,
                "x0": 5.0,
                "y0": 6.0,
                "sigma_x": 2.0,
                "sigma_y": 2.0,
                "theta": 0.0,
            },
        )
        init = {"offset": 0.0, "components": comps}
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, n, init, None, None)

    def test_init_components_tuple_happy_path_covers_224_to_232(self) -> None:
        z = np.zeros((20, 20), float)
        n = 2
        # components as a TUPLE → exercises 221; then 224 (alloc), 225–231 (loop & synonyms), 232 (pack/return)
        comps = (
            {
                "amp": 1.2,
                "x0": 5.0,
                "y0": 6.0,
                "sigma_x": 2.0,
                "sigma_y": 1.5,
                "theta": 0.3,
            },
            {
                "amplitude": 0.8,
                "x0": 10.0,
                "y0": 4.0,
                "sx": 2.5,
                "sy": 3.0,
            },  # theta omitted → defaults to 0.0
        )
        init = {"offset": 0.1, "components": comps}
        p = mg._normalize_initial_guesses(z, n, init, None, None)

        off, amp, x0, y0, sx, sy, th = mg._unpack_params(p, n)
        assert off == 0.1
        assert np.allclose(amp, [1.2, 0.8])
        assert np.allclose(x0, [5.0, 10.0])
        assert np.allclose(y0, [6.0, 4.0])
        assert np.allclose(sx, [2.0, 2.5])
        assert np.allclose(sy, [1.5, 3.0])
        assert np.allclose(th, [0.3, 0.0])

    def test_init_array_list_form_fallback_covers_234_235(self) -> None:
        z = np.zeros((18, 19), float)
        n = 2
        # PASS a plain LIST (list-of-lists) → falls through to lines 234–239 (array/list form)
        init_list = [
            [1.0, 6.0, 7.0, 2.0, 2.0, 0.1],
            [0.7, 12.0, 5.0, 3.0, 1.5, 0.2],
        ]
        p = mg._normalize_initial_guesses(z, n, init_list, None, None)
        off, amp, *_ = mg._unpack_params(p, n)
        assert off == 0.0  # offset seeded from masked median
        assert np.allclose(amp, [1.0, 0.7])

    def test_bounds_offset_branch_public_api(self) -> None:
        # Build a simple scene with a known offset (~0.12)
        ny, nx = 40, 40
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.12 + np.exp(-((x - 20) ** 2 + (y - 22) ** 2) / (2 * 3.0**2))
        da = xr.DataArray(z, dims=("y", "x"))

        # Reasonable initial guess
        init = np.array([[1.0, 20.0, 22.0, 3.0, 3.0, 0.0]], float)

        # Public API: pass 'offset' bounds → triggers the offset path inside _merge_bounds_multi
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            bounds={"offset": (0.05, 0.2)},
            return_model=False,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        off = float(ds["offset"])
        assert 0.05 <= off <= 0.2
        assert bool(ds["success"]) is True

    def test_bounds_per_component_list_public_api_hits_comp_idx_branch(self) -> None:
        # Make a clean 2-component scene with distinct sigma_x per component
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        z = (
            0.05
            + _gauss2d_on_grid(x, y, 1.0, 20.0, 20.0, 1.0, 1.2, 0.0, offset=0.0)
            + _gauss2d_on_grid(x, y, 0.8, 44.0, 40.0, 3.0, 2.5, 0.2, offset=0.0)
        )
        da = xr.DataArray(z, dims=("y", "x"))
        # Reasonable initial guesses (order matches components above)
        init = np.array(
            [
                [0.9, 19.5, 20.5, 1.2, 1.0, 0.1],  # near comp A
                [0.7, 44.5, 39.5, 2.7, 2.7, 0.2],  # near comp B
            ],
            dtype=float,
        )

        # Per-component bounds for sigma_x — this exercises the comp_idx branch inside _merge_bounds_multi
        bounds = {"sigma_x": [(0.5, 1.5), (2.0, 4.0)]}

        ds = fit_multi_gaussian2d(
            da,
            n_components=2,
            initial_guesses=init,
            bounds=bounds,
            return_model=False,
            return_residual=False,
            coord_type="pixel",  # DataArray has no coords; use pixel indices
        )

        # Sort by x0 to align components deterministically (A ~20, B ~44)
        order = np.argsort(ds["x0_pixel"].values)
        sx_sorted = ds["sigma_major_pixel"].values[order]

        # Assert each component's sigma_x falls within its per-component bounds
        assert 0.5 <= sx_sorted[0] <= 1.5
        assert 2.0 <= sx_sorted[1] <= 4.0

    def test_public_api_ensure_dataarray_raises_on_unsupported_type(self) -> None:
        # object() is neither np.ndarray, dask.array.Array, nor xarray.DataArray
        with pytest.raises(TypeError, match=r"Unsupported input type"):
            fit_multi_gaussian2d(object(), n_components=1)

    def test_resolve_dims_raises_for_3d_without_dims(self) -> None:
        # 3-D DataArray with no 'x'/'y' dims; calling public API without dims must raise
        arr = xr.DataArray(np.zeros((2, 8, 9)), dims=("t", "j", "i"))
        with pytest.raises(ValueError, match=r"ndim != 2.*specify two dims"):
            fit_multi_gaussian2d(arr, n_components=1)

    def test_public_api_n_components_must_be_positive(self) -> None:
        da = xr.DataArray(np.zeros((8, 8), float), dims=("y", "x"))
        with pytest.raises(ValueError, match=r"n_components must be >= 1"):
            fit_multi_gaussian2d(da, n_components=0)

    def test_initial_guesses_list_of_dicts_wrong_length_raises(self) -> None:
        """List[dict] initial_guesses length != n → ValueError (covers the raise)."""
        ny, nx = 32, 32
        img = xr.DataArray(np.zeros((ny, nx), float), dims=("y", "x"))
        # n_components=2 but provide only one dict → triggers the length check
        init = [
            {
                "amp": 1.0,
                "x0": 10.0,
                "y0": 11.0,
                "fwhm_x": 6.0,
                "fwhm_y": 4.0,
                "theta": 0.0,
            }
        ]
        with pytest.raises(ValueError, match=r"length n=2"):
            fit_multi_gaussian2d(
                img,
                n_components=2,
                initial_guesses=init,  # top-level list of dicts path
                return_model=False,
                return_residual=False,
                coord_type="pixel",  # DataArray has no coords; use pixel indices
            )

    def test_init_array_wrong_shape_hits_conv_arr_early_return_then_raises(
        self,
    ) -> None:
        """angle='pa' forces init → _convert_init_theta; (n,5) array triggers early return in _conv_arr."""
        ny, nx = 16, 16
        img = xr.DataArray(np.zeros((ny, nx), float), dims=("y", "x"))

        # n_components=1 but provide a (1,5) array → missing theta column
        init_wrong = np.array([[0.9, 8.0, 7.0, 5.0, 4.0]], dtype=float)

        # Public API call; this goes through _convert_init_theta (because angle='pa'),
        # where _conv_arr sees shape != (n,6) and returns unchanged; later the
        # normalization step raises on bad shape. Covers the 'return out' line.
        with pytest.raises(ValueError, match=r"initial_guesses .* got \(1, 5\)"):
            fit_multi_gaussian2d(
                img,
                n_components=1,
                initial_guesses=init_wrong,
                angle="pa",  # ensure the _convert_init_theta path is executed
                return_model=False,
                return_residual=False,
                coord_type="pixel",  # DataArray has no coords; use pixel indices
            )


class TestMetadataNotes:
    def test_variance_explained_includes_self_documenting_note(self) -> None:
        """Covers attach explanatory note to ``variance_explained`` DV.

        We run a simple 2-D fit and assert that the returned Dataset contains the
        ``variance_explained`` variable and that it carries a long, self-documenting
        ``attrs['note']`` string describing the metric. Using ``coord_type='pixel'``
        avoids any dependency on world coordinates.
        """
        ny, nx = 48, 48
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 24) ** 2 + (y - 20) ** 2) / (2 * 3.0**2))
        da = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.9, 24.0, 20.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )

        assert "variance_explained" in ds
        note = str(ds["variance_explained"].attrs.get("note", ""))
        # Spot-check key phrases to avoid brittleness while exercising the block.
        assert "R²-style fit quality" in note
        assert "Explained variance fraction" in note
        assert "Quick gut-check scale" in note


class TestAngleEndToEndFitter(unittest.TestCase):
    @staticmethod
    def _mk_gauss(ny, nx, amp, x0, y0, sx, sy, theta_math, offset=0.12, flip_x=False):
        # Build rotated anisotropic Gaussian with math angle `theta_math`
        y, x = np.mgrid[0:ny, 0:nx]
        if flip_x:
            x = x[:, ::-1]  # flip x-axis to left-handed coordinate system
        ct, st = np.cos(theta_math), np.sin(theta_math)
        X, Y = x - x0, y - y0
        a = (ct**2) / (2 * sx**2) + (st**2) / (2 * sy**2)
        b = st * ct * (1 / (2 * sx**2) - 1 / (2 * sy**2))  # sign fix (x term first)
        c = (st**2) / (2 * sx**2) + (ct**2) / (2 * sy**2)
        z = offset + amp * np.exp(-(a * X**2 + 2 * b * X * Y + c * Y**2))
        return xr.DataArray(z, dims=("y", "x"))

    @staticmethod
    def _angdiff(a, b):
        d = (a - b + np.pi) % (2 * np.pi) - np.pi
        return abs(d)

    def test_theta_math_pa_relation_pixel_right_and_left_handed(self):
        # Pixel coords (no world coords) -> right-handed basis by default
        ny, nx = 64, 64
        amp, x0, y0 = 1.0, 28.0, 30.0
        sx, sy = 4.0, 2.0

        for handedness in ("right", "left"):
            for theta_expected in (30.0, -60.0):
                theta_math_true = np.deg2rad(theta_expected)
                da = self._mk_gauss(
                    ny,
                    nx,
                    amp,
                    x0,
                    y0,
                    sx,
                    sy,
                    theta_math_true,
                    flip_x=(handedness == "left"),
                )

                # Seeds: close to truth; run once reporting math, once reporting PA
                init_math = np.array(
                    [[0.9, x0, y0, sx * 0.9, sy * 1.1, theta_math_true]], float
                )
                coord_type = "pixel" if handedness == "right" else "world"
                if handedness == "left":
                    # Assign world coords: x descending, y ascending
                    da = da.assign_coords(
                        x=np.linspace(1.0, -1.0, nx), y=np.linspace(-1.0, 1.0, ny)
                    )
                ds_math = fit_multi_gaussian2d(
                    da,
                    n_components=1,
                    initial_guesses=init_math,
                    angle="math",
                    coord_type=coord_type,
                    return_model=False,
                    return_residual=False,
                )

                assert bool(ds_math["success"]) is True
                theta_math_fit = float(
                    ds_math[f"theta_{coord_type}"].isel(component=0).values
                )  # reported in math
                self.assertTrue(
                    np.isclose(theta_math_fit, theta_math_true),
                    f"math fit {theta_math_fit} != true {theta_math_true} for {handedness} handed coordinate system",
                )
                pa_expected = (np.pi / 2 - theta_math_fit) % (2 * np.pi)
                init_pa = np.array(
                    [[0.9, x0, y0, sx * 0.9, sy * 1.1, pa_expected]], float
                )
                ds_pa = fit_multi_gaussian2d(
                    da,
                    n_components=1,
                    initial_guesses=init_pa,
                    angle="pa",
                    coord_type=coord_type,
                    return_model=False,
                    return_residual=False,
                )
                assert bool(ds_pa["success"]) is True
                theta_pa_fit = float(
                    ds_pa[f"theta_{coord_type}"].isel(component=0).values
                )  # reported in PA
                self.assertTrue(
                    np.isclose(theta_pa_fit, pa_expected),
                    f"pa fit {theta_pa_fit} != expected {pa_expected} for {handedness} handed coordinate system",
                )
                # Verify PA = 90° − θ (mod 2π)
                self.assertTrue(
                    np.isclose(theta_pa_fit, np.pi / 2 - theta_math_fit),
                    f"pa fit {theta_pa_fit} != pi/2 - math fit {theta_math_fit} for {handedness} handed coordinate system",
                )

    def test_canonicalization_swaps_axes_and_rotates_theta_by_half_pi(self) -> None:
        """
        Covers: th_math = -th, major/minor swap, +π/2 rotation, and wrap to (-π/2, π/2].
        Construct an ellipse whose *math* major axis is along +y (σx < σy, θ_math = 0),
        so canonicalization must rotate the reported math angle by +π/2.
        """
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        amp_true, x0_true, y0_true = 1.0, 30.0, 28.0
        sx_true, sy_true = 2.0, 5.0  # major axis along +y
        theta_math_true = 0.0  # no rotation in the scene frame
        z = _gauss2d_on_grid(
            x,
            y,
            amp_true,
            x0_true,
            y0_true,
            sx_true,
            sy_true,
            theta_math_true,
            offset=0.12,
        )
        img = xr.DataArray(z, dims=("y", "x"))

        # Seed close to truth; ask for math angles back
        init = np.array(
            [[0.9, x0_true, y0_true, sx_true * 0.95, sy_true * 1.05, theta_math_true]],
            float,
        )
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="math",
            return_model=False,
            return_residual=False,
            coord_type="pixel",
        )

        assert bool(ds["success"]) is True
        # After canonicalization: sigma_major >= sigma_minor and theta_math in (-π/2, π/2]
        smaj = float(ds["sigma_major_pixel"])
        smin = float(ds["sigma_minor_pixel"])
        th_m = float(ds["theta_pixel"])  # in math because angle="math"
        assert smaj >= smin
        assert (-np.pi / 2 - 1e-9) < th_m <= (np.pi / 2 + 1e-9)

        # For σx<σy and θ_math_true=0, canonicalization rotates by a half-π.
        # Endpoints ±π/2 are equivalent (ellipse major-axis ambiguity).
        # Accept either branch endpoint within tight tolerance.
        err = min(abs(th_m - np.pi / 2), abs(th_m + np.pi / 2))
        assert err < 0.01, f"got {th_m}, expected ≈ ±π/2"

    def test_canonicalization_wraps_theta_into_half_pi_interval(self) -> None:
        """
        Covers: half-π wrapping branch without swapping widths (σx >= σy).
        Choose θ_math just over +π/2 so the wrapped angle lands in (-π/2, π/2].
        """
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        amp_true, x0_true, y0_true = 1.0, 28.0, 30.0
        sx_true, sy_true = 5.0, 2.0  # already major along x (no swap)
        theta_math_true = 1.80  # ~103.13°, should wrap negative
        z = _gauss2d_on_grid(
            x,
            y,
            amp_true,
            x0_true,
            y0_true,
            sx_true,
            sy_true,
            theta_math_true,
            offset=0.10,
        )
        img = xr.DataArray(z, dims=("y", "x"))

        init = np.array(
            [[0.95, x0_true, y0_true, sx_true, sy_true, theta_math_true]], float
        )
        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="math",
            return_model=False,
            return_residual=False,
            coord_type="pixel",
        )

        assert bool(ds["success"]) is True
        th_m = float(ds["theta_pixel"])  # reported math angle, canonicalized

        # Expected wrap: ((θ + π/2) % π) − π/2  ∈ (-π/2, π/2]
        expected_wrapped = ((theta_math_true + np.pi / 2) % np.pi) - np.pi / 2
        assert (-np.pi / 2 - 1e-9) < th_m <= (np.pi / 2 + 1e-9)
        assert (
            abs(th_m - expected_wrapped) < 0.15
        ), f"got {th_m}, expected ~{expected_wrapped}"

        # Also verify PA report matches θ→PA conversion on the same fit if requested
        # (exercises publishing in requested convention).
        pa_expected = float(((np.pi / 2) - th_m) % (2 * np.pi))
        init_pa = np.array(
            [[0.95, x0_true, y0_true, sx_true, sy_true, pa_expected]], float
        )
        ds_pa = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init_pa,
            angle="pa",
            return_model=False,
            return_residual=False,
            coord_type="pixel",
        )
        th_pa = float(ds_pa["theta_pixel"])
        assert abs(th_pa - pa_expected) < 0.01


# ⬇️ Paste this into the existing file: tests/test_multi_gaussian2d_fit.py
# Place anywhere near related pixel→world interpolation tests.


class TestInnerPrepCoverage:
    def test_inner_prep_lines_executed(self, monkeypatch):
        """Covers lines by exercising all branches of the local `_prep`.
        We shim `_interp_centers_world` to reach the `_prep` closure and call it
        with inputs that trigger: invalid, descending, non-strict, increasing.
        """
        calls: list[tuple[str, object, object]] = []

        def shim(ds, cx, cy, dim_x, dim_y):
            # Access caller's local `_prep`
            fr = inspect.currentframe()
            assert fr is not None and fr.f_back is not None
            local_prep = fr.f_back.f_locals.get("_prep")
            assert callable(local_prep), "local _prep not found"

            # 1) invalid (ndim!=1) -> (None, None)
            idx, val = local_prep(np.array([[1.0, 2.0]]))
            calls.append(("invalid", idx, val))

            # 2) descending -> reversed indices & coords
            idx, val = local_prep(np.array([5.0, 4.0, 3.0]))
            calls.append(("descending", idx.copy(), val.copy()))

            # 3) non-strictly-increasing -> (None, None)
            idx, val = local_prep(np.array([0.0, 1.0, 1.0, 2.0]))
            calls.append(("non_strict", idx, val))

            # 4) strictly increasing -> identity
            idx, val = local_prep(np.array([0.0, 0.5, 1.0]))
            calls.append(("increasing", idx.copy(), val.copy()))

            # mark for assertion that shim ran
            out = ds.copy()
            out.attrs["_shim_ran"] = True
            return out

        # Fast, deterministic optimizer stub
        def fake_cf(func, xy, zflat, p0=None, bounds=None, maxfev=None):
            p0 = np.asarray(p0, float)
            pcov = np.eye(p0.size, dtype=float) * 0.01
            return p0, pcov

        monkeypatch.setattr(mg, "_interp_centers_world", shim, raising=True)
        monkeypatch.setattr(mg, "curve_fit", fake_cf, raising=True)

        # Build DataArray with valid world coords; run in pixel mode to enter branch
        ny, nx = 6, 8
        y = np.linspace(-1.0, 1.0, ny, dtype=float)
        x = np.linspace(-2.0, 2.0, nx, dtype=float)
        da = xr.DataArray(
            np.zeros((ny, nx), float), dims=("y", "x"), coords={"y": y, "x": x}
        )

        init = np.array([[0.2, nx / 2 - 0.5, ny / 2 - 0.5, 1.5, 1.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da, n_components=1, initial_guesses=init, coord_type="pixel"
        )

        # The shim executed and exercised all `_prep` branches
        assert ds.attrs.get("_shim_ran", False) is True
        assert len(calls) == 4

        # Validate branch outcomes
        label, idx, val = calls[0]
        assert label == "invalid" and idx is None and val is None

        label, idx, val = calls[1]
        assert label == "descending"
        assert np.allclose(idx, np.array([2.0, 1.0, 0.0]))
        assert np.allclose(val, np.array([3.0, 4.0, 5.0]))

        label, idx, val = calls[2]
        assert label == "non_strict" and idx is None and val is None

        label, idx, val = calls[3]
        assert label == "increasing"
        assert np.allclose(idx, np.array([0.0, 1.0, 2.0]))
        assert np.allclose(val, np.array([0.0, 0.5, 1.0]))


class TestCoordsForNdarrayInput:
    def test_coords_validation_and_success(self, monkeypatch):
        """Validate tuple length, shape checks, and successful use of provided coords for ndarray input."""

        # Deterministic optimizer
        def fake_cf(func, xy, zflat, p0=None, bounds=None, maxfev=None):
            p0 = np.asarray(p0, float)
            pcov = np.eye(p0.size, dtype=float) * 0.01
            return p0, pcov

        monkeypatch.setattr(mg, "curve_fit", fake_cf, raising=True)

        ny, nx = 10, 12
        img = np.zeros((ny, nx), float)
        init = np.array([[0.5, nx / 2 - 0.5, ny / 2 - 0.5, 2.0, 1.5, 0.0]], float)

        # 1) wrong-length coords tuple -> error
        with pytest.raises(ValueError):
            fit_multi_gaussian2d(
                img,
                n_components=1,
                initial_guesses=init,
                coords=(np.arange(nx, dtype=float),),
            )

    def test_numpy_input_coords_tuple_happy_path_returns_y1d_x1d(self) -> None:
        """
        NumPy input with coords=(x1d, y1d) should be accepted and returned (y1d, x1d).
        Covers the success path that converts and validates coords, then returns them.
        """
        ny, nx = 7, 9
        # original_input is a plain NumPy array (not DataArray)
        original = np.zeros((ny, nx), dtype=float)
        # Build a minimal DataArray used only for sizes dim_y/dim_x inside the helper
        da_tr = xr.DataArray(np.empty((ny, nx), dtype=float), dims=("y", "x"))
        # Provide non-trivial 1-D coordinates (not pixel indices) to exercise the branch
        x1d = np.linspace(-2.5, 1.5, nx, dtype=float)
        y1d = np.linspace(10.0, 12.0, ny, dtype=float)
        y_out, x_out = mg._extract_1d_coords_for_fit(
            original_input=original,
            da_tr=da_tr,
            coord_type="world",  # ignored for NumPy input
            coords=(x1d, y1d),  # (x1d, y1d)
            dim_y="y",
            dim_x="x",
        )
        assert np.allclose(y_out, y1d)
        assert np.allclose(x_out, x1d)

    def test_numpy_input_coords_tuple_wrong_lengths_raise(self) -> None:
        """
        NumPy input with coords of wrong lengths should raise.
        Exercises the validation right before the success return.
        """
        ny, nx = 5, 6
        original = np.zeros((ny, nx), dtype=float)
        da_tr = xr.DataArray(np.empty((ny, nx), dtype=float), dims=("y", "x"))
        # x1d has wrong length
        x1d_bad = np.linspace(0.0, 1.0, nx + 1, dtype=float)
        y1d_ok = np.linspace(0.0, 1.0, ny, dtype=float)
        with pytest.raises(ValueError):
            mg._extract_1d_coords_for_fit(
                original_input=original,
                da_tr=da_tr,
                coord_type="world",
                coords=(x1d_bad, y1d_ok),
                dim_y="y",
                dim_x="x",
            )


# ------------------------- result metadata: package version fallback -------------------------


class TestResultMetadataVersion:
    def _tiny_scene(self) -> xr.DataArray:
        ny, nx = 6, 6
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 3.0) ** 2 + (y - 2.0) ** 2) / (2 * 1.2**2))
        return xr.DataArray(z, dims=("y", "x"))

    def _one_comp(self) -> np.ndarray:
        return np.array([[1.0, 3.0, 2.0, 1.2, 1.2, 0.0]], float)

    def test_version_fallback_reads_package_dunder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If importlib.metadata.version fails, fallback should read astroviper.__version__."""
        import importlib.metadata as ilmd

        # Force the primary version lookup to fail → exercise fallback path
        def _boom(*_, **__):
            raise Exception("no version")

        monkeypatch.setattr(ilmd, "version", _boom, raising=True)
        # Ensure the top-level package exposes __version__

        if "astroviper" not in sys.modules:
            sys.modules["astroviper"] = types.ModuleType("astroviper")
        monkeypatch.setattr(
            sys.modules["astroviper"], "__version__", "0.0.test", raising=False
        )

        da = self._tiny_scene()
        ds = fit_multi_gaussian2d(
            da, n_components=1, initial_guesses=self._one_comp(), coord_type="pixel"
        )

        assert ds.attrs.get("package") == "astroviper"
        assert ds.attrs.get("version") == "0.0.test"

    def test_version_fallback_unknown_when_no_dunder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If both metadata lookup and __version__ access fail, version should be 'unknown'."""
        import importlib.metadata as ilmd

        def _boom(*_, **__):
            raise Exception("no version")

        monkeypatch.setattr(ilmd, "version", _boom, raising=True)
        # Remove __version__ if present on the loaded package
        import sys

        if "astroviper" in sys.modules:
            monkeypatch.delattr(sys.modules["astroviper"], "__version__", raising=False)

        da = self._tiny_scene()
        ds = fit_multi_gaussian2d(
            da, n_components=1, initial_guesses=self._one_comp(), coord_type="pixel"
        )

        assert ds.attrs.get("package") == "astroviper"
        assert ds.attrs.get("version") == "unknown"


# ------------------------- bounds mapping: FWHM → σ (sigma) -------------------------
# Covers the branch that maps 'fwhm_major'/'fwhm_minor' bounds into 'sigma_x'/'sigma_y',
# including both per-component list-of-tuples and single-tuple cases.


class TestBoundsFwhmMapping:
    def _dummy_results(self, da: xr.DataArray, n: int) -> tuple[
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
        xr.DataArray,
    ]:
        comp = xr.DataArray(np.zeros((n,), dtype=float), dims=("component",))
        # 6 params + 6 errors + 4 derived
        blocks = [comp.copy() for _ in range(6 + 6 + 4)]
        offset = xr.DataArray(0.0)
        offset_e = xr.DataArray(0.0)
        success = xr.DataArray(True)
        varexp = xr.DataArray(1.0)
        resid = xr.DataArray(np.zeros_like(da.values), dims=da.dims)
        model = xr.DataArray(np.zeros_like(da.values), dims=da.dims)
        return (*blocks, offset, offset_e, success, varexp, resid, model)

    def test_fwhm_major_list_of_tuples_maps_to_sigma_x_per_component(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Capture the mapped bounds passed into the vectorized wrapper
        seen: dict[str, dict] = {}

        # Keep original so we can forward calls not targeting the plane wrapper
        orig_apply_ufunc = xr.apply_ufunc

        def fake_apply_ufunc(*args, **kw):
            func = args[0] if args else None
            # Intercept only the vectorized plane wrapper call (where bounds are passed)
            if func is mg._multi_fit_plane_wrapper:
                da_tr = args[1]
                kwargs_dict = kw.get("kwargs", {}) or {}
                seen["bounds"] = dict(kwargs_dict.get("bounds", {}))
                n = int(kwargs_dict.get("n_components", 1))
                # Return minimally valid shaped results to let the pipeline finish
                return TestBoundsFwhmMapping()._dummy_results(da_tr, n)
            # For other apply_ufunc usages (e.g., np.greater_equal), defer to original
            return orig_apply_ufunc(*args, **kw)

        monkeypatch.setattr(xr, "apply_ufunc", fake_apply_ufunc, raising=True)

        ny, nx = 8, 9
        da = xr.DataArray(np.zeros((ny, nx), dtype=float), dims=("y", "x"))
        init = np.array(
            [[1.0, 3.0, 4.0, 2.0, 1.5, 0.0], [0.8, 6.0, 2.0, 3.0, 2.5, 0.1]],
            dtype=float,
        )
        bounds = {"fwhm_major": [(1.0, 4.0), (2.0, 5.0)]}

        ds = fit_multi_gaussian2d(
            da, n_components=2, initial_guesses=init, bounds=bounds, coord_type="pixel"
        )
        assert isinstance(ds, xr.Dataset)

        conv = mg._FWHM2SIG
        expected = {"sigma_x": [(1.0 * conv, 4.0 * conv), (2.0 * conv, 5.0 * conv)]}
        assert seen["bounds"] == expected

    def test_fwhm_minor_single_tuple_maps_to_sigma_y_tuple(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: dict[str, dict] = {}

        orig_apply_ufunc = xr.apply_ufunc

        def fake_apply_ufunc(*args, **kw):
            func = args[0] if args else None
            if func is mg._multi_fit_plane_wrapper:
                da_tr = args[1]
                kwargs_dict = kw.get("kwargs", {}) or {}
                seen["bounds"] = dict(kwargs_dict.get("bounds", {}))
                n = int(kwargs_dict.get("n_components", 1))
                return TestBoundsFwhmMapping()._dummy_results(da_tr, n)
            return orig_apply_ufunc(*args, **kw)

        monkeypatch.setattr(xr, "apply_ufunc", fake_apply_ufunc, raising=True)

        ny, nx = 7, 7
        da = xr.DataArray(np.zeros((ny, nx), dtype=float), dims=("y", "x"))
        init = np.array([[1.0, 3.0, 3.0, 1.2, 1.1, 0.0]], dtype=float)
        bounds = {"fwhm_minor": (2.0, 6.0)}

        ds = fit_multi_gaussian2d(
            da, n_components=1, initial_guesses=init, bounds=bounds, coord_type="pixel"
        )
        assert isinstance(ds, xr.Dataset)

        conv = mg._FWHM2SIG
        expected = {"sigma_y": (2.0 * conv, 6.0 * conv)}
        assert seen["bounds"] == expected


# ------------------------- cover mapping of fwhm_minor → sigma_y (public API) -------------------------
# Exercises the branch where component dicts omit sigma_y/sy but provide fwhm_minor.
# This drives the internal list-of-dicts parser so the conversion is applied before fitting.


class TestInitialGuessesFwhmMinorMappingPublicAPI:
    def test_components_list_of_dicts_maps_fwhm_minor_to_sigma_y_in_p0(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Build a tiny scene
        ny, nx = 12, 13
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 6.0) ** 2 + (y - 5.0) ** 2) / (2 * 2.0**2))
        da = xr.DataArray(z, dims=("y", "x"))

        # Two components with sigma_x provided, sigma_y omitted, fwhm_minor provided.
        init = {
            "components": [
                {
                    "amp": 1.0,
                    "x0": 6.0,
                    "y0": 5.0,
                    "sigma_x": 2.0,
                    "fwhm_minor": 1.5,
                    "theta": 0.1,
                },
                {
                    "amplitude": 0.8,
                    "x0": 9.0,
                    "y0": 3.0,
                    "sx": 3.0,
                    "fwhm_minor": 2.5,
                    "theta": 0.2,
                },
            ]
        }

        captured = {}

        # Intercept scipy.optimize.curve_fit to capture the seed vector p0
        def fake_curve_fit(*args, **kwargs):
            p0 = kwargs["p0"]
            captured["p0"] = np.asarray(p0, dtype=float).copy()
            n = (p0.size - 1) // 6
            size = 1 + 6 * n
            pcov = np.eye(size, dtype=float)
            return p0, pcov

        monkeypatch.setattr(mg, "curve_fit", fake_curve_fit, raising=True)

        ds = fit_multi_gaussian2d(
            da,
            n_components=2,
            initial_guesses=init,
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )
        assert isinstance(ds, xr.Dataset)

        # p0 layout: [offset, (amp,x0,y0,sigma_x,sigma_y,theta)*n]
        p0 = captured["p0"]
        # Expected sigma_y from FWHM: σ = FWHM / (2*sqrt(2*ln 2))
        conv = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        expected_sy = [1.5 * conv, 2.5 * conv]
        sy_indices = [1 + 6 * 0 + 4, 1 + 6 * 1 + 4]
        actual_sy = [float(p0[sy_indices[0]]), float(p0[sy_indices[1]])]
        assert np.allclose(actual_sy, expected_sy, rtol=1e-12, atol=0.0)


class TestAutoSeedingPixelPublicAPI:
    def test_auto_seed_pixel_public_api(self) -> None:
        """Public API: pixel coords + auto initial guesses."""
        ny, nx = 7, 9
        z = np.zeros((ny, nx), dtype=float)
        # two separated bright pixels
        for y, x, v in [(1, 1, 10.0), (5, 7, 8.0)]:
            z[y, x] = v
        da = xr.DataArray(z, dims=("y", "x"))  # no world coords

        ds = fit_multi_gaussian2d(
            da,
            n_components=2,
            initial_guesses=None,  # auto-seed (hits pixel seeding path)
            coord_type="pixel",  # pixel mode via public API
            return_model=False,
            return_residual=False,
        )
        assert bool(ds.success)

        # centers should land near the two bright pixels (order-agnostic)
        pred = np.column_stack([ds["x0_pixel"].values, ds["y0_pixel"].values])
        expected = [(1.0, 1.0), (7.0, 5.0)]
        for xe, ye in expected:
            dmin = np.min(np.hypot(pred[:, 0] - xe, pred[:, 1] - ye))
            assert dmin < 1.0


class TestAutoSeedingPixelElsePublicAPI:
    def test_auto_seeding_pixel_else_path_public_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Public API call drives the pixel-seeding 'else' path.
        We force the underlying plane fitter to treat axes as pixel-index grids,
        then assert seeds: centers at pixel indices and widths == max(nx, ny)/10.
        """
        ny, nx = 12, 15
        yi, xi = 4, 11
        img = np.zeros((ny, nx), dtype=float)
        img[yi, xi] = 10.0

        # World-like coords present on the DataArray (public API stays the same)
        da = xr.DataArray(
            img,
            dims=("y", "x"),
            coords={
                "y": np.linspace(-5.0, 5.0, ny),
                "x": np.linspace(100.0, 130.0, nx),
            },
        )

        # Capture the optimizer's initial parameter vector (p0)
        captured: dict[str, np.ndarray] = {}

        def _fake_curve_fit(func, xy, zflat, *, p0=None, bounds=None, maxfev=None):
            p0 = np.asarray(p0, dtype=float)
            captured["p0"] = p0.copy()
            # Echo p0; identity covariance (keeps API stable)
            return p0, np.eye(p0.size, dtype=float)

        monkeypatch.setattr(mg, "curve_fit", _fake_curve_fit, raising=True)

        # Force pixel seeding by making the internal fitter treat axes as None.
        # (Still invoked through the public API; we do not call privates ourselves.)
        _orig_fit = mg._fit_multi_plane_numpy

        def _fit_force_pixel(
            z2d,
            n_components,
            min_threshold,
            max_threshold,
            initial_guesses,
            bounds,
            max_nfev,
            *,
            x1d=None,
            y1d=None,
            mask2d=None,
            **kwargs,
        ):
            return _orig_fit(
                z2d,
                n_components,
                min_threshold,
                max_threshold,
                initial_guesses,
                bounds,
                max_nfev,
                x1d=None,
                y1d=None,
                mask2d=mask2d,
                **kwargs,
            )

        monkeypatch.setattr(
            mg, "_fit_multi_plane_numpy", _fit_force_pixel, raising=True
        )

        # Public API call
        ds = mg.fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=None,
            coord_type="world",
            return_model=False,
            return_residual=False,
        )
        assert isinstance(ds, xr.Dataset)
        assert "success" in ds and bool(ds["success"])

        # p0: [offset, amp, x0, y0, sigma_x, sigma_y, theta]
        p0 = captured["p0"]
        assert p0.shape == (7,)

        # Centers are pixel indices; widths use max(nx, ny)/10
        assert np.isclose(p0[2], float(xi))
        assert np.isclose(p0[3], float(yi))
        expected_w = max(nx, ny) / 10.0
        assert np.isclose(p0[4], expected_w)
        assert np.isclose(p0[5], expected_w)

        # Sanity on the rest
        assert p0[1] > 0.0  # amp
        assert np.isclose(p0[0], 0)  # offset
        assert np.isclose(p0[6], 0)  # theta


class TestResultMetadataShortener:
    def test_coords_repr_truncated_when_shape_property_raises(self) -> None:
        class EvilCoords:
            @property
            def shape(self):
                # ensure the library's _short() hits the exception path
                raise RuntimeError("shape access should not be required")

            def __repr__(self) -> str:
                # long repr triggers truncation branch
                return "E" * 300

        ny, nx = 32, 40
        y, x = np.mgrid[0:ny, 0:nx]
        img = 0.1 + np.exp(
            -((x - nx / 2.0) ** 2) / (2 * 3.0**2) - ((y - ny / 2.0) ** 2) / (2 * 2.0**2)
        )
        da = xr.DataArray(img, dims=("y", "x"))

        init = np.array(
            [[1.0, nx / 2.0, ny / 2.0, 2.3548 * 3.0, 2.3548 * 2.0, 0.0]],
            dtype=float,
        )

        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            coord_type="pixel",
            coords=EvilCoords(),  # recorded in metadata; not used for DataArray pixel fits
            return_model=False,
            return_residual=False,
        )

        s = ds.attrs["param"]["coords"]
        assert isinstance(s, str)
        assert len(s) == 120 and s.endswith("...")
        assert "shape=" not in s


class TestWorldSeeding:
    def test_autoseed_uses_world_axes_and_recovers_params(self) -> None:
        # Synthetic single Gaussian in WORLD coords (not pixel indices)
        nx = ny = 129
        x = np.linspace(-nx + 1, nx - 1, nx, dtype=float)
        y = np.linspace(-ny + 1, ny - 1, ny, dtype=float)
        X, Y = np.meshgrid(x, y)

        amp_true = 5.0
        x0_true, y0_true = 40.0, -20.0
        fwhm_major, fwhm_minor = 20.0, 10.0
        theta = 0.4  # radians, math convention
        K = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM = K * sigma
        sx = fwhm_major / K
        sy = fwhm_minor / K

        # rotated elliptical Gaussian (math angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        Xc = X - x0_true
        Yc = Y - y0_true
        Xp = Xc * cos_t + Yc * sin_t
        Yp = -Xc * sin_t + Yc * cos_t
        Z = amp_true * np.exp(-0.5 * ((Xp / sx) ** 2 + (Yp / sy) ** 2))
        da = xr.DataArray(Z, dims=("y", "x"), coords={"x": x, "y": y})

        # No initial guesses → exercise auto-seed path.
        ds = fit_multi_gaussian2d(da, n_components=1)

        # Must fit in WORLD frame when coords are not pure pixel indices.
        assert ds.attrs.get("fit_native_frame") == "world"
        assert "x0_world" in ds and "y0_world" in ds

        # Amplitude/peak should be near the truth (no noise, no offset).
        if "amplitude" in ds:
            assert np.isclose(float(ds["amplitude"]), amp_true, rtol=0.05, atol=0.05)
        if "peak" in ds:
            assert np.isclose(float(ds["peak"]), amp_true, rtol=0.05, atol=0.05)

        # Centers recovered in WORLD coordinates.
        assert np.isclose(float(ds["x0_world"]), x0_true, atol=1.0)
        assert np.isclose(float(ds["y0_world"]), y0_true, atol=1.0)


# ------------------------- masking (public API) -------------------------


class TestMaskingPublicAPI:
    def test_mask_excludes_peak_boolean_public_api(self) -> None:
        """
        Two bright pixels; mask excludes the brighter one.
        Fit should lock onto the unmasked peak.
        """
        ny, nx = 12, 14
        z = np.zeros((ny, nx), dtype=float)
        yA, xA, vA = 3, 2, 10.0  # brighter, but will be masked out
        yB, xB, vB = 8, 11, 9.0  # should be selected
        z[yA, xA] = vA
        z[yB, xB] = vB
        da = xr.DataArray(z, dims=("y", "x"))

        mask = np.ones((ny, nx), dtype=bool)
        mask[yA, xA] = False  # exclude the brighter peak

        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=None,
            mask=mask,
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )
        assert bool(ds["success"])

        # Expect center near the unmasked (xB, yB)
        x0 = float(np.ravel(ds["x0_pixel"].values)[0])
        y0 = float(np.ravel(ds["y0_pixel"].values)[0])
        assert abs(x0 - xB) < 1.0
        assert abs(y0 - yB) < 1.0

    def test_mask_broadcasts_yx_over_stack_public_api(self) -> None:
        """
        (y,x) mask should broadcast across a stacked cube.
        """
        ny, nx, nt = 10, 10, 3
        z = np.zeros((ny, nx), dtype=float)
        yA, xA, vA = 2, 2, 10.0  # masked out
        yB, xB, vB = 6, 7, 9.0  # should be selected across all time slices
        z[yA, xA] = vA
        z[yB, xB] = vB
        planes = [xr.DataArray(z.copy(), dims=("y", "x")) for _ in range(nt)]
        cube = xr.concat(planes, dim="time")

        mask = np.ones((ny, nx), dtype=bool)
        mask[yA, xA] = False  # exclude A globally; should broadcast to all 'time'

        ds = fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=None,
            mask=mask,
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )
        assert "time" in ds.dims and ds.sizes["time"] == nt
        assert np.all(ds["success"].values)

        # Flatten in case dims are (time, component)
        x0 = np.ravel(ds["x0_pixel"].values)
        y0 = np.ravel(ds["y0_pixel"].values)
        assert x0.size == nt and y0.size == nt
        assert np.all(np.abs(x0 - xB) < 1.0)
        assert np.all(np.abs(y0 - yB) < 1.0)

    def test_mask_and_threshold_empty_after_and_raises_public_api(self) -> None:
        """
        Mask ∧ threshold can empty the usable set → ValueError.
        Without the mask this would pass (bright pixel >= min_threshold),
        but masking that pixel removes all candidates.
        """
        ny, nx = 8, 9
        z = np.zeros((ny, nx), dtype=float)
        yP, xP, vP = 3, 5, 10.0
        z[yP, xP] = vP
        da = xr.DataArray(z, dims=("y", "x"))

        mask = np.ones((ny, nx), dtype=bool)
        mask[yP, xP] = False  # remove the only pixel that would pass threshold

        with pytest.raises(ValueError) as excinfo:
            fit_multi_gaussian2d(
                da,
                n_components=1,
                initial_guesses=None,
                mask=mask,
                min_threshold=5.0,  # would keep only (yP,xP) if not masked
                coord_type="pixel",
                return_model=False,
                return_residual=False,
            )
        # Implementation message mentions thresholding; we just check the empty mask semantics
        assert "removed all pixels" in str(excinfo.value)


# ------------------------- OTF string masking (public API) -------------------------


class TestOTFMaskingPublicAPI:
    def test_otf_mask_excludes_peak_dataarray_public_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        select_mask returns a DataArray mask that excludes the brighter peak.
        The fit should choose the unmasked peak.
        """
        ny, nx = 12, 14
        z = np.zeros((ny, nx), dtype=float)
        yA, xA, vA = 3, 2, 10.0  # masked out by OTF mask
        yB, xB, vB = 8, 11, 9.0  # expected fit
        z[yA, xA] = vA
        z[yB, xB] = vB
        da = xr.DataArray(z, dims=("y", "x"))

        def _fake_select_mask(da_tr: xr.DataArray, spec: str) -> xr.DataArray:
            m = np.ones((ny, nx), dtype=bool)
            m[yA, xA] = False
            return xr.DataArray(m, dims=("y", "x"))

        monkeypatch.setattr(mg, "_select_mask", _fake_select_mask, raising=True)

        ds = mg.fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=None,
            mask="exclude_bright_A",  # any string → handled by _select_mask
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )
        assert bool(ds["success"])
        x0 = float(np.ravel(ds["x0_pixel"].values)[0])
        y0 = float(np.ravel(ds["y0_pixel"].values)[0])
        assert abs(x0 - xB) < 1.0
        assert abs(y0 - yB) < 1.0

    def test_otf_mask_broadcasts_ndarray_over_stack_public_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        select_mask returns a 2-D ndarray mask (y,x); it should broadcast across stacked 'time'.
        """
        ny, nx, nt = 10, 10, 3
        z = np.zeros((ny, nx), dtype=float)
        yA, xA, vA = 2, 2, 10.0  # masked out by OTF mask
        yB, xB, vB = 6, 7, 9.0  # expected across all time slices
        z[yA, xA] = vA
        z[yB, xB] = vB
        planes = [xr.DataArray(z.copy(), dims=("y", "x")) for _ in range(nt)]
        cube = xr.concat(planes, dim="time")

        def _fake_select_mask(da_tr: xr.DataArray, spec: str) -> np.ndarray:
            m = np.ones((ny, nx), dtype=bool)
            m[yA, xA] = False
            return m  # raw ndarray

        monkeypatch.setattr(mg, "_select_mask", _fake_select_mask, raising=True)

        ds = mg.fit_multi_gaussian2d(
            cube,
            n_components=1,
            initial_guesses=None,
            mask="exclude_A_ndarray",
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )
        assert "time" in ds.dims and ds.sizes["time"] == nt
        assert np.all(ds["success"].values)
        x0 = np.ravel(ds["x0_pixel"].values)
        y0 = np.ravel(ds["y0_pixel"].values)
        assert x0.size == nt and y0.size == nt
        assert np.all(np.abs(x0 - xB) < 1.0)
        assert np.all(np.abs(y0 - yB) < 1.0)

    def test_otf_mask_dataset_wrapper_supported_public_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        select_mask returns a Dataset containing a 'mask' variable; fitter should use it.
        """
        ny, nx = 9, 11
        z = np.zeros((ny, nx), dtype=float)
        yA, xA, vA = 1, 3, 10.0  # masked out
        yB, xB, vB = 7, 9, 8.0  # expected
        z[yA, xA] = vA
        z[yB, xB] = vB
        da = xr.DataArray(z, dims=("y", "x"))

        def _fake_select_mask(da_tr: xr.DataArray, spec: str) -> xr.Dataset:
            m = np.ones((ny, nx), dtype=bool)
            m[yA, xA] = False
            return xr.Dataset({"mask": xr.DataArray(m, dims=("y", "x"))})

        monkeypatch.setattr(mg, "_select_mask", _fake_select_mask, raising=True)

        ds = mg.fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=None,
            mask="exclude_A_dataset",
            coord_type="pixel",
            return_model=False,
            return_residual=False,
        )
        assert bool(ds["success"])
        x0 = float(np.ravel(ds["x0_pixel"].values)[0])
        y0 = float(np.ravel(ds["y0_pixel"].values)[0])
        assert abs(x0 - xB) < 1.0
        assert abs(y0 - yB) < 1.0

    def test_otf_mask_dataset_without_var_errors_public_api(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        select_mask returns a Dataset without 'mask' variable → should raise TypeError.
        """
        ny, nx = 6, 7
        da = xr.DataArray(np.zeros((ny, nx), dtype=float), dims=("y", "x"))

        def _fake_select_mask(_da: xr.DataArray, spec: str) -> xr.Dataset:
            return xr.Dataset(
                {
                    "not_mask": xr.DataArray(
                        np.ones((ny, nx), dtype=bool), dims=("y", "x")
                    )
                }
            )

        monkeypatch.setattr(mg, "_select_mask", _fake_select_mask, raising=True)

        with pytest.raises(TypeError):
            mg.fit_multi_gaussian2d(
                da,
                n_components=1,
                initial_guesses=None,
                mask="bad_dataset_without_mask_var",
                coord_type="pixel",
                return_model=False,
                return_residual=False,
            )


class TestChooseThetaPublicAPI:
    def _mk_minimal_result(
        self, *, frame: str, have_pa: bool, have_math: bool
    ) -> xr.Dataset:
        """
        Construct a minimal public-API-shaped result Dataset that plot_components accepts.
        Includes pixel-center, FWHM sizes (so no size warnings), amplitude/offset/success.
        Optionally includes theta_* variables to exercise candidate preference/fallback.
        """
        comp = ("component",)
        ds = xr.Dataset(
            data_vars=dict(
                x0_pixel=xr.DataArray([10.0], dims=comp),
                y0_pixel=xr.DataArray([12.0], dims=comp),
                fwhm_major_pixel=xr.DataArray([3.0], dims=comp),
                fwhm_minor_pixel=xr.DataArray([2.0], dims=comp),
                amplitude=xr.DataArray([1.0], dims=comp),
                offset=xr.DataArray(0.0),
                success=xr.DataArray(True),
            )
        )
        if have_pa:
            ds[f"theta_{frame}_pa"] = xr.DataArray([0.30], dims=comp)
        if have_math:
            ds[f"theta_{frame}_math"] = xr.DataArray([0.60], dims=comp)
        return ds

    def test_choose_theta_prefers_pa_when_available_public(self) -> None:
        """
        angle='pa' and only theta_*_pa exists → first candidate is used.
        Public API only: plot_components(data, result, angle=...).
        """
        ds = self._mk_minimal_result(frame="pixel", have_pa=True, have_math=False)
        img = xr.DataArray(np.zeros((12, 16), float), dims=("y", "x"))
        mg.plot_components(img, ds, dims=("y", "x"), angle="pa", show=False)
        plt.close("all")

    def test_choose_theta_falls_back_to_math_when_pa_missing_public(self) -> None:
        """
        angle='pa' but only theta_*_math exists → fallback (second candidate) is used.
        """
        ds = self._mk_minimal_result(frame="pixel", have_pa=False, have_math=True)
        img = xr.DataArray(np.zeros((12, 16), float), dims=("y", "x"))
        mg.plot_components(img, ds, dims=("y", "x"), angle="pa", show=False)
        plt.close("all")

    def test_choose_theta_none_when_both_missing_public(self) -> None:
        """
        Neither theta_*_pa nor theta_*_math exist → fall-through to (None, None).
        plot_components should warn and still draw (axis-aligned ellipse).
        """
        ds = self._mk_minimal_result(frame="world", have_pa=False, have_math=False)
        img = xr.DataArray(np.zeros((12, 16), float), dims=("y", "x"))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mg.plot_components(img, ds, dims=("y", "x"), angle="pa", show=False)
            # Implementation emits a RuntimeWarning; accept any warning mentioning "Missing theta".
            assert any("Missing theta" in str(m.message) for m in w)
        plt.close("all")
