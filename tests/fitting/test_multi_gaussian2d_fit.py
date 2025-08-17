# tests/test_multi_gaussian2d_all.py
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

import os
import math
import numpy as np
import pytest
import xarray as xr

import dask.array as da  # type: ignore

# Use headless matplotlib, silence plt.show()
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
import matplotlib.pyplot as plt

#from astroviper.fitting import multi_gaussian2d_fit as mg
from astroviper.fitting.multi_gaussian2d_fit import fit_multi_gaussian2d, plot_components
import astroviper.fitting.multi_gaussian2d_fit as mg

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


def _scene(ny: int, nx: int, comps, *, offset=0.0, noise=0.0, seed=0, coords=False) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = np.zeros((ny, nx), float) + offset
    for c in comps:
        amp = float(c["amp"])
        x0 = float(c["x0"]); y0 = float(c["y0"])
        sx = float(c["sigma_x"]); sy = float(c["sigma_y"]); th = float(c["theta"])
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
        init = np.array([[1.0, 39.5, 60.5, 3.0, 3.0, 0.0],
                         [0.7, 84.5, 28.5, 5.0, 2.5, 0.4]], float)
        ds = fit_multi_gaussian2d(da2, n_components=2, initial_guesses=init,
                                  return_model=True, return_residual=True)
        assert bool(ds.success)
        assert 0.0 <= float(ds.variance_explained) <= 1.0
        assert "x_world" in ds and "y_world" in ds
        order = np.argsort(ds["x0"].values)
        assert np.allclose(ds["x0"].values[order], [40.0, 85.0], atol=0.8)
        assert np.allclose(ds["y0"].values[order], [60.0, 28.0], atol=0.8)

    def test_vectorize_over_time_names_and_indices(self) -> None:
        ny, nx = 64, 80
        base = dict(amp=0.9, x0=30.0, y0=22.0, sigma_x=4.0, sigma_y=3.0, theta=0.2)
        planes = [_scene(ny, nx, [base], offset=0.1, noise=0.03, seed=s) for s in (1, 2, 3)]
        cube = xr.concat(planes, dim="time")  # dims ('time','y','x')
        init = np.array([[0.8, 29.5, 22.5, 4.0, 3.0, 0.2]], float)
        # dims by name
        ds1 = fit_multi_gaussian2d(cube, n_components=1, initial_guesses=init,
                                   dims=("x", "y"), return_model=False, return_residual=False)
        assert "time" in ds1.dims and ds1.sizes["time"] == 3
        # dims by index
        ds2 = fit_multi_gaussian2d(cube, n_components=1, initial_guesses=init,
                                   dims=(2, 1), return_model=False, return_residual=False)
        assert np.all(ds2["success"].values)

    def test_flags_and_descending_world_coords(self) -> None:
        ny, nx = 40, 50
        comps = [dict(amp=0.8, x0=20.0, y0=18.0, sigma_x=3.0, sigma_y=2.0, theta=0.1)]
        da2 = _scene(ny, nx, comps, coords=True)
        # reverse coords → descending path
        da2 = da2.assign_coords(x=da2.coords["x"][::-1], y=da2.coords["y"][::-1])
        init = np.array([[0.7, 20.0, 18.0, 3.0, 2.0, 0.1]], float)

        ds1 = fit_multi_gaussian2d(da2, n_components=1, initial_guesses=init,
                                   return_model=True, return_residual=False)
        assert "model" in ds1 and "residual" not in ds1
        assert "x_world" in ds1 and "y_world" in ds1

        ds2 = fit_multi_gaussian2d(da2, n_components=1, initial_guesses=init,
                                   return_model=False, return_residual=True)
        assert "residual" in ds2 and "model" not in ds2

    def test_auto_seeds_when_initial_none(self) -> None:
        ny, nx = 48, 60
        comps = [
            dict(amp=1.0, x0=18.0, y0=22.0, sigma_x=3.0, sigma_y=3.0, theta=0.0),
            dict(amp=0.6, x0=42.0, y0=30.0, sigma_x=5.0, sigma_y=2.5, theta=0.3),
        ]
        da2 = _scene(ny, nx, comps, offset=0.1, noise=0.03, seed=3)
        ds = fit_multi_gaussian2d(da2, n_components=2, initial_guesses=None, return_residual=True)
        assert bool(ds.success) is True


# ------------------------- input types -------------------------

class TestInputs:
    def test_accepts_raw_numpy_array(self) -> None:
        ny, nx = 32, 33
        comps = [dict(amp=1.0, x0=16.0, y0=15.0, sigma_x=3.0, sigma_y=3.0, theta=0.0)]
        arr = _scene(ny, nx, comps).values  # plain ndarray
        ds = fit_multi_gaussian2d(arr, n_components=1, initial_guesses=np.array([[1,16,15,3,3,0.0]]))
        assert bool(ds.success) is True

    @pytest.mark.skipif(da is None, reason="dask.array not available")
    def test_accepts_bare_dask_array(self) -> None:
        ny, nx = 40, 40
        comps = [dict(amp=1.0, x0=20.0, y0=20.0, sigma_x=3.0, sigma_y=3.0, theta=0.0)]
        np_img = _scene(ny, nx, comps, offset=0.1, noise=0.02, seed=1).data
        darr = da.from_array(np_img, chunks=(ny, nx))
        ds = fit_multi_gaussian2d(darr, n_components=1, initial_guesses=np.array([[1,20,20,3,3,0.0]]))
        assert bool(ds.success) is True

    def test_world_coords_skipped_for_bad_axis_coords(self) -> None:
        ny, nx = 32, 32
        comps = [dict(amp=0.8, x0=15.0, y0=16.0, sigma_x=3.0, sigma_y=2.0, theta=0.1)]
        da2 = _scene(ny, nx, comps, coords=True)
        # break monotonicity / finiteness
        x = da2.coords["x"].values.copy(); x[5] = x[4]
        y = da2.coords["y"].values.copy(); y[3] = np.nan
        ds = fit_multi_gaussian2d(da2.assign_coords(x=("x", x), y=("y", y)),
                                  n_components=1, initial_guesses=np.array([[0.8,15,16,3,2,0.1]]))
        assert "x_world" not in ds or "y_world" not in ds


# ------------------------- bounds / dims / API validation -------------------------

class TestBoundsDimsAPI:
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
        dct = {"offset": 0.2, "components": np.array([[1.0, 5.0, 6.0, 2.0, 2.0, 0.0]], float)}
        p2 = mg._normalize_initial_guesses(z, 1, dct, None, None)
        assert p2[0] == pytest.approx(0.2)
        # dict with list of dicts
        dct_list = {"components": [{"amp": 0.8, "x0": 3.0, "y0": 4.0, "sigma_x": 2.1, "sigma_y": 1.9, "theta": 0.1}]}
        p3 = mg._normalize_initial_guesses(z, 1, dct_list, None, None)
        assert p3.shape == (7,)
        # wrong shape -> error
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, 2, np.array([[1, 2, 3, 4, 5, 6]]), None, None)
        # dict missing 'components' -> error
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, 1, {"offset": 0.0}, None, None)


# ------------------------- optimizer / masking failure paths -------------------------

class TestOptimizerFailures:
    def test_full_mask_triggers_failure_and_nan_planes(self) -> None:
        ny, nx = 24, 24
        da2 = xr.DataArray(np.zeros((ny, nx)), dims=("y", "x"))
        ds = fit_multi_gaussian2d(da2, n_components=1, min_threshold=1.0,
                                  return_residual=True, return_model=True)
        assert bool(ds["success"]) is False
        assert np.isnan(ds["residual"].values).all()
        assert np.isnan(ds["model"].values).all()
        assert np.isnan(float(ds["variance_explained"]))

    def test_curve_fit_exception_sets_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ny, nx = 40, 40
        y, x = np.mgrid[0:ny, 0:nx]
        z = np.exp(-((x - 20) ** 2 + (y - 18) ** 2) / (2 * 3.0**2))
        da2 = xr.DataArray(z, dims=("y", "x"))
        def boom(*args, **kwargs):
            raise RuntimeError("nope")
        monkeypatch.setattr(mg, "curve_fit", boom, raising=True)
        ds = fit_multi_gaussian2d(da2, n_components=1,
                                  initial_guesses=np.array([[1.0, 20.0, 18.0, 3.0, 3.0, 0.0]]),
                                  return_residual=True)
        assert bool(ds["success"]) is False
        assert np.isnan(ds["amplitude"].values).all()
        assert np.isnan(float(ds["variance_explained"]))

    def test_curve_fit_pcov_none_sets_errors_nan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ny, nx = 32, 32
        da2 = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
        n = 2
        # popt layout: [offset, (amp,x0,y0,sx,sy,th)*n]
        popt = np.array([0.1, 1.0, 10.0, 12.0, 2.0, 2.0, 0.0,
                         0.7, 20.0, 8.0, 3.0, 1.5, 0.1], float)
        def fake_fit(func, xy, z, p0=None, bounds=None, maxfev=None):
            return popt, None
        monkeypatch.setattr(mg, "curve_fit", fake_fit, raising=True)
        ds = fit_multi_gaussian2d(da2, n_components=n,
                                  initial_guesses=np.array([[1,10,12,2,2,0],[0.7,20,8,3,1.5,0.1]], float),
                                  return_model=False)
        assert bool(ds["success"]) is True
        for name in ("amplitude_err","x0_err","y0_err","sigma_x_err","sigma_y_err","theta_err"):
            assert np.isnan(ds[name].values).all()

    def test_curve_fit_with_valid_pcov_sets_finite_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ny, nx = 24, 24
        da2 = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
        n = 1
        p_len = 1 + 6 * n
        popt = np.linspace(0.1, 0.1 + 0.01*(p_len-1), p_len)
        pcov = np.eye(p_len, dtype=float)
        def fake_fit(func, xy, z, p0=None, bounds=None, maxfev=None):
            return popt, pcov
        monkeypatch.setattr(mg, "curve_fit", fake_fit, raising=True)
        ds = fit_multi_gaussian2d(da2, n_components=n,
                                  initial_guesses=np.array([[1.0, 10.0, 12.0, 2.0, 2.0, 0.0]], float),
                                  return_model=False, return_residual=False)
        for name in ("amplitude_err","x0_err","y0_err","sigma_x_err","sigma_y_err","theta_err","offset_err"):
            assert np.all(np.isfinite(ds[name].values))


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
        ds = fit_multi_gaussian2d(cube, n_components=1, initial_guesses=init, return_residual=True)

        # Build a DS WITHOUT 'component' dim (fix: use the post-isel dims!)
        ds_no_comp = xr.Dataset({
            k: v.isel(component=0) if ("component" in v.dims) else v
            for k, v in ds.items()
            if k in ("x0", "y0", "sigma_x", "sigma_y", "theta", "residual")
        })

        # draw with indexer and both residual states
        mg.plot_components(cube, ds, dims=("x","y"), indexer={"time": 1}, show_residual=True)
        mg.plot_components(cube, ds_no_comp, dims=("x","y"), indexer={"time": 0}, show_residual=False)

    def test_plot_components_defaults_indexer_for_3d_input(self, monkeypatch) -> None:
        matplotlib.use("Agg", force=True)  # headless backend

        # Build a 3-D DataArray: ('time','y','x') so da_tr.ndim > 2
        ny, nx = 32, 32
        y, x = np.mgrid[0:ny, 0:nx]
        base = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / (2 * 3.0 ** 2))
        frame = xr.DataArray(base, dims=("y", "x"))
        cube = xr.concat([frame, frame + 0.01], dim="time")

        # Public API fit; no need for residual/model here
        init = np.array([[1.0, 16.0, 16.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(cube, n_components=1, initial_guesses=init)

        # Silence plt.show to avoid warnings in CI
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # Call without 'indexer' → triggers default indexer = {leading dims: 0}
        # This covers:
        #   if da_tr.ndim > 2:
        #       if indexer is None:
        #           indexer = {d: 0 for d in da_tr.dims[:-2]}
        plot_components(cube, ds, dims=("x", "y"), show_residual=False)

    def test_plot_components_selects_result_plane_with_default_indexer(self, monkeypatch) -> None:
        matplotlib.use("Agg", force=True)  # headless

        # Build 3-D data → da_tr.ndim > 2
        ny, nx = 32, 32
        y, x = np.mgrid[0:ny, 0:nx]
        base = np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / (2 * 3.0 ** 2))
        frame = xr.DataArray(base, dims=("y", "x"))
        cube = xr.concat([frame, frame + 0.01], dim="time")  # dims: ('time','y','x')

        # Fit via public API; result keeps the leading 'time' dim
        init = np.array([[1.0, 16.0, 16.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(cube, n_components=1, initial_guesses=init, return_residual=True)
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

    def test_plot_components_default_indexer_loops_multiple_dims(self, monkeypatch) -> None:
        matplotlib.use("Agg", force=True)

        # 4D data ensures indexer has ≥2 keys → loop body executes (and loops)
        ny, nx = 24, 24
        y, x = np.mgrid[0:ny, 0:nx]
        base = np.exp(-((x - 12) ** 2 + (y - 12) ** 2) / (2 * 3.0 ** 2))
        frame = xr.DataArray(base, dims=("y", "x"))
        cube_t = xr.concat([frame, frame + 0.01], dim="t")              # dims: ('t','y','x')
        cube = xr.concat([cube_t, cube_t + 0.02], dim="band")           # dims: ('band','t','y','x')

        init = np.array([[1.0, 12.0, 12.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(cube, n_components=1, initial_guesses=init, return_residual=True)

        # avoid blocking GUI
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # indexer=None → default {'band':0, 't':0}; loop iterates over both keys
        plot_components(cube, ds, dims=("x", "y"), indexer=None, show_residual=True)

    def test_plot_components_else_branch_for_2d_input(self, monkeypatch) -> None:
        matplotlib.use("Agg", force=True)

        # 2D image → triggers the else block (lines 766–768): data2d = da_tr; res_plane = result
        ny, nx = 40, 40
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.1 + np.exp(-((x - 20) ** 2 + (y - 22) ** 2) / (2 * 3.0 ** 2))
        img = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[1.0, 20.0, 22.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(img, n_components=1, initial_guesses=init, return_residual=True)

        # avoid GUI
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        # No indexer; 2D input hits the else branch
        plot_components(img, ds, dims=("x", "y"), indexer=None, show_residual=True)

    def test_plot_components_raises_when_result_missing_required_var(self, monkeypatch) -> None:
        matplotlib.use("Agg", force=True)

        ny, nx = 32, 32
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 16) ** 2 + (y - 16) ** 2) / (2 * 3.0 ** 2))
        img = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[1.0, 16.0, 16.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(img, n_components=1, initial_guesses=init)

        # Remove a required variable to trigger the _get(...) KeyError path
        ds_missing = ds.drop_vars("x0")

        # avoid GUI popups
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)

        with pytest.raises(KeyError, match=r"result missing 'x0'"):
            plot_components(img, ds_missing, dims=("x", "y"), indexer=None, show_residual=False)

class TestNumPyFitting:
    def test_min_threshold_masks_pixels_partial(self) -> None:

        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 3.0 ** 2))
        da = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.8, 32.0, 32.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            min_threshold=0.20,   # exercises: mask &= z2d >= min_threshold
            return_model=True,
            return_residual=True,
        )

        assert bool(ds.success) is True
        assert "model" in ds and "residual" in ds
        above = int((z >= 0.20).sum())
        assert 0 < above < z.size

    def test_min_threshold_masks_pixels_partial(self) -> None:

        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 3.0 ** 2))
        da = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.8, 32.0, 32.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            min_threshold=0.20,  # exercises: mask &= z2d >= min_threshold
            return_model=True,
            return_residual=True,
        )

        assert bool(ds.success) is True
        assert "model" in ds and "residual" in ds
        above = int((z >= 0.20).sum())
        assert 0 < above < z.size

    def test_max_threshold_masks_pixels_partial(self) -> None:

        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.05 + np.exp(-((x - 32) ** 2 + (y - 32) ** 2) / (2 * 3.0 ** 2))  # peak ~ 1.05
        da = xr.DataArray(z, dims=("y", "x"))

        init = np.array([[0.8, 32.0, 32.0, 3.0, 3.0, 0.0]], float)
        ds = fit_multi_gaussian2d(
            da,
            n_components=1,
            initial_guesses=init,
            max_threshold=0.60,  # exercises: mask &= z2d <= max_threshold
            return_model=True,
            return_residual=True,
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
        a = (ct**2)/(2*sx_true**2) + (st**2)/(2*sy_true**2)
        b = st*ct*(1/(2*sy_true**2) - 1/(2*sx_true**2))
        c = (st**2)/(2*sx_true**2) + (ct**2)/(2*sy_true**2)
        z = 0.12 + amp_true * np.exp(-(a*X**2 + 2*b*X*Y + c*Y**2))
        img = xr.DataArray(z, dims=("y", "x"))

        # Expected PA for ascending axes (sx=+1, sy=+1): PA = arctan2(cos(theta), sin(theta))
        pa_expected = np.arctan2(np.cos(theta_math), np.sin(theta_math))

        # Provide initial guesses *in PA*; public API will convert them via _theta_pa_to_math
        init = np.array([[0.9, x0_true, y0_true, sx_true*0.9, sy_true*1.1, pa_expected]], float)

        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            angle="pa",              # triggers PA→math conversion path
            return_model=False,
            return_residual=False,
        )

        assert bool(ds["success"]) is True
        # theta is reported in the same convention ("pa"), so it should match pa_expected
        assert np.isfinite(float(ds["theta"]))
        assert abs(float(ds["theta"]) - pa_expected) < 0.15  # allow some tolerance

    def test_angle_pa_init_conversion_list_of_dicts(self) -> None:
        # Two rotated Gaussians with known *math* angles
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]

        def gauss(a, x0, y0, sx, sy, th):
            ct, st = np.cos(th), np.sin(th)
            X, Y = x - x0, y - y0
            A = (ct**2)/(2*sx**2) + (st**2)/(2*sy**2)
            B = st*ct*(1/(2*sy**2) - 1/(2*sx**2))
            C = (st**2)/(2*sx**2) + (ct**2)/(2*sy**2)
            return a * np.exp(-(A*X**2 + 2*B*X*Y + C*Y**2))

        th1_math, th2_math = 0.5, 1.1
        z = (
            0.10
            + gauss(1.0, 20.0, 22.0, 3.0, 2.0, th1_math)
            + gauss(0.8,  44.0, 40.0, 4.0, 2.5, th2_math)
        )
        img = xr.DataArray(z, dims=("y", "x"))

        # Convert the true math angles to PA for ascending axes:
        # PA = arctan2(cos(theta_math), sin(theta_math))
        pa1 = float(np.arctan2(np.cos(th1_math), np.sin(th1_math)))
        pa2 = float(np.arctan2(np.cos(th2_math), np.sin(th2_math)))

        # initial_guesses as LIST OF DICTS with 'theta' → triggers _conv_list_of_dicts path
        init_list = [
            {"amp": 0.9, "x0": 20.0, "y0": 22.0, "sigma_x": 3.0, "sigma_y": 2.0, "theta": pa1},
            {"amp": 0.7, "x0": 44.0, "y0": 40.0, "sigma_x": 4.0, "sigma_y": 2.5, "theta": pa2},
        ]

        ds = fit_multi_gaussian2d(
            img,
            n_components=2,
            initial_guesses=init_list,  # list[dict] path
            angle="pa",                  # forces PA→math conversion on init
            return_model=False,
            return_residual=False,
        )

        # Sort by x0 to align component identity
        order = np.argsort(ds["x0"].values)
        thetas_pa = ds["theta"].values[order]

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
        ct, st = np.cos(theta_math), np.sin(theta_math)
        X, Y = x - x0_true, y - y0_true
        A = (ct**2)/(2*sx_true**2) + (st**2)/(2*sy_true**2)
        B = st*ct*(1/(2*sy_true**2) - 1/(2*sx_true**2))
        C = (st**2)/(2*sx_true**2) + (ct**2)/(2*sy_true**2)
        z = 0.12 + amp_true * np.exp(-(A*X**2 + 2*B*X*Y + C*Y**2))
        img = xr.DataArray(z, dims=("y", "x"))

        # Convert math→PA for ascending axes: PA = arctan2(cos(th), sin(th))
        pa = float(np.arctan2(np.cos(theta_math), np.sin(theta_math)))

        # initial_guesses as dict with components **NumPy array** → hits lines 590–595
        comps = np.array([[0.9, x0_true, y0_true, sx_true, sy_true, pa]], float)
        init = {"components": comps}

        ds = fit_multi_gaussian2d(
            img, n_components=1, initial_guesses=init,
            angle="pa", return_model=False, return_residual=False,
        )

        assert bool(ds["success"]) is True
        # theta reported in PA (angle="pa"); should be close to the PA seed
        assert abs(float(ds["theta"]) - pa) < 0.25

    def test_angle_pa_init_components_list_of_dicts_converted(self) -> None:
        # Build a single rotated Gaussian (known math angle)
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]
        amp_true, x0_true, y0_true = 0.9, 22.0, 24.0
        sx_true, sy_true = 3.5, 2.0
        theta_math = 0.6
        ct, st = np.cos(theta_math), np.sin(theta_math)
        X, Y = x - x0_true, y - y0_true
        A = (ct**2)/(2*sx_true**2) + (st**2)/(2*sy_true**2)
        B = st*ct*(1/(2*sy_true**2) - 1/(2*sx_true**2))
        C = (st**2)/(2*sx_true**2) + (ct**2)/(2*sy_true**2)
        z = 0.10 + amp_true * np.exp(-(A*X**2 + 2*B*X*Y + C*Y**2))
        img = xr.DataArray(z, dims=("y", "x"))

        # Compute PA from math angle for ascending axes
        pa = float(np.arctan2(np.cos(theta_math), np.sin(theta_math)))

        # initial_guesses as dict with components **list of dicts** → hits lines 595–597
        init = {
            "components": [
                {"amp": 0.85, "x0": x0_true, "y0": y0_true, "sigma_x": sx_true, "sigma_y": sy_true, "theta": pa}
            ]
        }

        ds = fit_multi_gaussian2d(
            img, n_components=1, initial_guesses=init,
            angle="pa", return_model=False, return_residual=False,
        )

        assert bool(ds["success"]) is True
        assert abs(float(ds["theta"]) - pa) < 0.25

    def test_angle_pa_list_of_dicts_missing_theta_covers_if_false_branch(self) -> None:
        # Build a 2-component scene
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]

        def gauss(a, x0, y0, sx, sy, th):
            ct, st = np.cos(th), np.sin(th)
            X, Y = x - x0, y - y0
            A = (ct**2)/(2*sx**2) + (st**2)/(2*sy**2)
            B = st*ct*(1/(2*sy**2) - 1/(2*sx**2))
            C = (st**2)/(2*sx**2) + (ct**2)/(2*sy**2)
            return a * np.exp(-(A*X**2 + 2*B*X*Y + C*Y**2))

        th1_math, th2_math = 0.7, 0.0  # comp1 rotated; comp2 axis-aligned (math)
        z = (
            0.10
            + gauss(1.0, 20.0, 22.0, 3.0, 2.0, th1_math)
            + gauss(0.8,  44.0, 40.0, 4.0, 2.5, th2_math)
        )
        img = xr.DataArray(z, dims=("y", "x"))

        # Expected PA for ascending axes: PA = arctan2(cos(theta_math), sin(theta_math))
        pa1 = float(np.arctan2(np.cos(th1_math), np.sin(th1_math)))
        pa2 = float(np.arctan2(np.cos(th2_math), np.sin(th2_math)))  # ~π/2 for theta_math=0

        # list[dict] with one dict MISSING 'theta' → exercises 579→581 (no conversion branch for that dict)
        init_list = [
            {"amp": 0.9, "x0": 20.0, "y0": 22.0, "sigma_x": 3.0, "sigma_y": 2.0, "theta": pa1},
            {"amp": 0.7, "x0": 44.0, "y0": 40.0, "sigma_x": 4.0, "sigma_y": 2.5},  # no 'theta'
        ]

        ds = fit_multi_gaussian2d(
            img,
            n_components=2,
            initial_guesses=init_list,
            angle="pa",                  # convert dicts that HAVE 'theta'; report output in PA
            return_model=False,
            return_residual=False,
        )

        assert bool(ds["success"]) is True
        order = np.argsort(ds["x0"].values)
        th_pa = ds["theta"].values[order]

        assert np.isfinite(th_pa).all()
        assert abs(th_pa[0] - pa1) < 0.3     # comp1 near its seeded PA
        assert abs(th_pa[1] - pa2) < 0.5     # comp2 near PA for math=0 (≈ π/2)

    def test_angle_pa_init_components_list_of_dicts_branch(self) -> None:
        # 2-component scene
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]

        def g(a, x0, y0, sx, sy, th):
            ct, st = np.cos(th), np.sin(th)
            X, Y = x - x0, y - y0
            A = (ct**2)/(2*sx**2) + (st**2)/(2*sy**2)
            B = st*ct*(1/(2*sy**2) - 1/(2*sx**2))
            C = (st**2)/(2*sx**2) + (ct**2)/(2*sy**2)
            return a * np.exp(-(A*X**2 + 2*B*X*Y + C*Y**2))

        th1_math, th2_math = 0.4, 1.0
        z = 0.1 + g(1.0, 18.0, 20.0, 3.0, 2.0, th1_math) + g(0.8, 44.0, 38.0, 4.0, 2.6, th2_math)
        img = xr.DataArray(z, dims=("y", "x"))

        # PA seeds from math angles for ascending axes: PA = arctan2(cos(theta), sin(theta))
        pa1 = float(np.arctan2(np.cos(th1_math), np.sin(th1_math)))
        pa2 = float(np.arctan2(np.cos(th2_math), np.sin(th2_math)))

        # Dict with "components" as LIST OF DICTS → hits 595→597 (_conv_list_of_dicts then return clone)
        init = {
            "components": [
                {"amp": 0.9, "x0": 18.0, "y0": 20.0, "sigma_x": 3.0, "sigma_y": 2.0, "theta": pa1},
                {"amp": 0.7, "x0": 44.0, "y0": 38.0, "sigma_x": 4.0, "sigma_y": 2.6, "theta": pa2},
            ]
        }

        ds = fit_multi_gaussian2d(
            img, n_components=2, initial_guesses=init, angle="pa",
            return_model=False, return_residual=False,
        )

        assert bool(ds["success"]) is True
        # Reported theta is in PA (angle="pa"); check against seeds
        order = np.argsort(ds["x0"].values)
        th_pa = ds["theta"].values[order]
        assert np.isfinite(th_pa).all()
        assert abs(th_pa[0] - pa1) < 0.3
        assert abs(th_pa[1] - pa2) < 0.3


class TestAPIHelpers:
    def test_init_components_array_wrong_shape_raises(self) -> None:
        da = xr.DataArray(np.zeros((16, 17), float), dims=("y", "x"))
        bad_init = {"offset": 0.0, "components": np.ones((1, 6), float)}  # shape != (n,6) for n=2

        with pytest.raises(ValueError):
            fit_multi_gaussian2d(da, n_components=2, initial_guesses=bad_init)

    def test_init_components_list_len_mismatch_raises(self) -> None:
        z = np.zeros((16, 17), float)
        n = 2
        init = {
            "offset": 0.0,
            "components": [  # length 1, but n=2 → should raise
                {"amp": 1.0, "x0": 5.0, "y0": 6.0, "sigma_x": 2.0, "sigma_y": 2.0, "theta": 0.0}
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
                {"amp": 1.2, "x0": 5.0, "y0": 6.0, "sigma_x": 2.0, "sigma_y": 1.5, "theta": 0.3},
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
        bad_amp = {"components": [{"x0": 4.0, "y0": 5.0, "sigma_x": 2.0, "sigma_y": 2.0}]}
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
            {"amp": 1.0, "x0": 5.0, "y0": 6.0, "sigma_x": 2.0, "sigma_y": 2.0, "theta": 0.0}
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
                {"amp": 1.2, "x0": 5.0, "y0": 6.0, "sigma_x": 2.0, "sigma_y": 1.5, "theta": 0.3},
                {"amplitude": 0.8, "x0": 10.0, "y0": 4.0, "sx": 2.5, "sy": 3.0},  # theta omitted → 0.0
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
            {"amp": 1.0, "x0": 5.0, "y0": 6.0, "sigma_x": 2.0, "sigma_y": 2.0, "theta": 0.0},
        )
        init = {"offset": 0.0, "components": comps}
        with pytest.raises(ValueError):
            mg._normalize_initial_guesses(z, n, init, None, None)

    def test_init_components_tuple_happy_path_covers_224_to_232(self) -> None:
        z = np.zeros((20, 20), float)
        n = 2
        # components as a TUPLE → exercises 221; then 224 (alloc), 225–231 (loop & synonyms), 232 (pack/return)
        comps = (
            {"amp": 1.2, "x0": 5.0, "y0": 6.0, "sigma_x": 2.0, "sigma_y": 1.5, "theta": 0.3},
            {"amplitude": 0.8, "x0": 10.0, "y0": 4.0, "sx": 2.5, "sy": 3.0},  # theta omitted → defaults to 0.0
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
        import numpy as np
        import xarray as xr
        from astroviper.fitting.multi_gaussian2d_fit import fit_multi_gaussian2d

        # Build a simple scene with a known offset (~0.12)
        ny, nx = 40, 40
        y, x = np.mgrid[0:ny, 0:nx]
        z = 0.12 + np.exp(-((x - 20) ** 2 + (y - 22) ** 2) / (2 * 3.0 ** 2))
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
        )

        off = float(ds["offset"])
        assert 0.05 <= off <= 0.2
        assert bool(ds["success"]) is True

    def test_bounds_per_component_list_public_api_hits_comp_idx_branch(self) -> None:
        import numpy as np
        import xarray as xr
        from astroviper.fitting.multi_gaussian2d_fit import fit_multi_gaussian2d

        # Make a clean 2-component scene with distinct sigma_x per component
        ny, nx = 64, 64
        y, x = np.mgrid[0:ny, 0:nx]

        def gauss2d(amp, x0, y0, sx, sy, th):
            ct, st = np.cos(th), np.sin(th)
            X, Y = x - x0, y - y0
            a = (ct**2)/(2*sx**2) + (st**2)/(2*sy**2)
            b = st*ct*(1/(2*sy**2) - 1/(2*sx**2))
            c = (st**2)/(2*sx**2) + (ct**2)/(2*sy**2)
            return amp * np.exp(-(a*X**2 + 2*b*X*Y + c*Y**2))

        # Component A: tight sigma_x ~1.0   | Component B: broader sigma_x ~3.0
        z = (
            0.05
            + gauss2d(1.0, 20.0, 20.0, 1.0, 1.2, 0.0)
            + gauss2d(0.8, 44.0, 40.0, 3.0, 2.5, 0.2)
        )
        da = xr.DataArray(z, dims=("y", "x"))

        # Reasonable initial guesses (order matches components above)
        init = np.array([
            [0.9, 19.5, 20.5, 1.2, 1.0, 0.1],  # near comp A
            [0.7, 44.5, 39.5, 2.7, 2.7, 0.2],  # near comp B
        ], dtype=float)

        # Per-component bounds for sigma_x — this exercises the comp_idx branch inside _merge_bounds_multi
        bounds = {"sigma_x": [(0.5, 1.5), (2.0, 4.0)]}

        ds = fit_multi_gaussian2d(
            da,
            n_components=2,
            initial_guesses=init,
            bounds=bounds,
            return_model=False,
            return_residual=False,
        )

        # Sort by x0 to align components deterministically (A ~20, B ~44)
        order = np.argsort(ds["x0"].values)
        sx_sorted = ds["sigma_x"].values[order]

        # Assert each component's sigma_x falls within its per-component bounds
        assert 0.5 <= sx_sorted[0] <= 1.5
        assert 2.0 <= sx_sorted[1] <= 4.0

    def test_public_api_ensure_dataarray_raises_on_unsupported_type(self) -> None:
        import pytest
        from astroviper.fitting.multi_gaussian2d_fit import fit_multi_gaussian2d

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

