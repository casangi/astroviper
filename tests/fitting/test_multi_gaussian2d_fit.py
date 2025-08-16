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

import math
import numpy as np
import pytest
import xarray as xr

try:  # optional Dask
    import dask.array as da
except Exception:  # pragma: no cover
    da = None  # type: ignore


# ------------------------- Helpers (inlined from conftest) -------------------------

def has_dask() -> bool:
    """True if dask.array importable."""
    return da is not None


def _rot_coords(x: np.ndarray, y: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate coordinates by theta radians."""
    ct, st = math.cos(theta), math.sin(theta)
    xr_ = x * ct + y * st
    yr_ = -x * st + y * ct
    return xr_, yr_


def make_gaussian_scene(
    ny: int,
    nx: int,
    *,
    components: list[dict],
    offset: float = 0.0,
    noise_std: float = 0.0,
    seed: int = 0,
    with_coords: bool = False,
) -> xr.DataArray:
    """
    Sum of rotated elliptical Gaussians + constant offset (+ optional white noise).

    Parameters
    ----------
    ny, nx : int
    components : list of dicts
        Keys: amp, x0, y0, sigma_x, sigma_y, theta (radians).
    offset : float
    noise_std : float
    seed : int
    with_coords : bool
        If True, attach 1-D coords for y/x to exercise world-coordinate path.

    Returns
    -------
    xr.DataArray of shape (ny, nx) with dims ("y","x").
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:ny, 0:nx]
    img = np.zeros((ny, nx), dtype=float) + offset
    for c in components:
        amp = float(c["amp"])
        x0 = float(c["x0"])
        y0 = float(c["y0"])
        sx = float(c["sigma_x"])
        sy = float(c["sigma_y"])
        th = float(c["theta"])
        xr_, yr_ = _rot_coords(x - x0, y - y0, th)
        img += amp * np.exp(-(xr_**2) / (2 * sx**2) - (yr_**2) / (2 * sy**2))
    if noise_std > 0:
        img += rng.normal(0.0, noise_std, size=img.shape)
    coords = None
    if with_coords:
        coords = {"y": np.linspace(-1.5, 1.5, ny), "x": np.linspace(-2.0, 2.0, nx)}
    return xr.DataArray(img, dims=("y", "x"), coords=coords)


# ------------------------- Tests -------------------------

from astroviper.fitting.multi_gaussian2d_fit import fit_multi_gaussian2d
from astroviper.fitting import multi_gaussian2d_fit as mg  # module for monkeypatch targets


class TestNumPyFitting:
    """NumPy-backed success paths, vectorization, flags, and world-coords."""

    @pytest.mark.parametrize("noise_std", [0.02, 0.05])
    def test_two_components_success_and_world_coords(self, noise_std: float) -> None:
        ny, nx = 96, 112
        comps = [
            dict(amp=1.0, x0=40.0, y0=60.0, sigma_x=3.0, sigma_y=3.0, theta=0.0),
            dict(amp=0.7, x0=85.0, y0=28.0, sigma_x=5.0, sigma_y=2.5, theta=0.4),
        ]
        img = make_gaussian_scene(
            ny, nx, components=comps, offset=0.1, noise_std=noise_std, seed=123, with_coords=True
        )
        init = np.array(
            [
                [1.0, 39.5, 60.5, 3.0, 3.0, 0.0],
                [0.7, 84.5, 28.5, 5.0, 2.5, 0.4],
            ],
            float,
        )

        ds = fit_multi_gaussian2d(
            img,
            n_components=2,
            initial_guesses=init,
            return_model=True,
            return_residual=True,
        )

        assert bool(ds.success)
        assert 0.0 <= float(ds.variance_explained) <= 1.0
        assert "component" in ds["x0"].dims and ds["x0"].sizes["component"] == 2
        assert set(["model", "residual"]).issubset(ds.data_vars)

        # world coords present when axis coords exist
        assert "x_world" in ds and "y_world" in ds
        assert ds["x_world"].sizes["component"] == 2
        assert ds["y_world"].sizes["component"] == 2

        # centers near truth (allow permutation)
        order = np.argsort(ds["x0"].values)
        x0 = ds["x0"].values[order]
        y0 = ds["y0"].values[order]
        assert np.allclose(x0, [40.0, 85.0], atol=0.7)
        assert np.allclose(y0, [60.0, 28.0], atol=0.7)

    def test_vectorized_over_extra_dims_no_planes(self) -> None:
        ny, nx = 64, 80
        base = dict(amp=0.9, x0=30.0, y0=22.0, sigma_x=4.0, sigma_y=3.0, theta=0.2)
        planes = [
            make_gaussian_scene(ny, nx, components=[base], offset=0.1, noise_std=0.03, seed=s)
            for s in (1, 2, 3)
        ]
        cube = xr.concat(planes, dim="time")
        init = np.array([[0.8, 29.5, 22.5, 4.0, 3.0, 0.2]], float)

        ds = fit_multi_gaussian2d(
            cube, n_components=1, initial_guesses=init, return_model=False, return_residual=False
        )
        assert "time" in ds.dims and ds.sizes["time"] == 3
        assert "model" not in ds and "residual" not in ds
        assert np.all(ds["success"].values)

    def test_thresholds_and_bounds_respected(self) -> None:
        ny, nx = 72, 72
        params = [dict(amp=1.0, x0=35.0, y0=36.0, sigma_x=6.0, sigma_y=2.0, theta=0.3)]
        img = make_gaussian_scene(ny, nx, components=params, offset=0.1, noise_std=0.02, seed=7)
        bounds = {"sigma_y": (1.5, 2.5)}
        init = np.array([[0.9, 34.0, 37.0, 5.5, 1.8, 0.2]], float)

        ds = fit_multi_gaussian2d(
            img,
            n_components=1,
            initial_guesses=init,
            bounds=bounds,
            min_threshold=0.05,
            max_threshold=None,
            return_model=False,
            return_residual=True,
        )

        sy = float(ds["sigma_y"].values[0])
        assert 1.5 <= sy <= 2.5
        assert "residual" in ds and "model" not in ds
        assert bool(ds["success"])
        assert 0.0 <= float(ds["variance_explained"]) <= 1.0

    def test_flags_return_model_or_residual_independently(self) -> None:
        ny, nx = 32, 32
        img = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
        init = np.array([[1.0, 15.0, 16.0, 3.0, 3.0, 0.0]], float)

        ds1 = fit_multi_gaussian2d(
            img, n_components=1, initial_guesses=init, return_model=True, return_residual=False
        )
        assert "model" in ds1 and "residual" not in ds1

        ds2 = fit_multi_gaussian2d(
            img, n_components=1, initial_guesses=init, return_model=False, return_residual=True
        )
        assert "residual" in ds2 and "model" not in ds2

    def test_world_coords_absent_when_no_axis_coords(self) -> None:
        ny, nx = 40, 40
        da2 = xr.DataArray(np.zeros((ny, nx)), dims=("y", "x"))  # no coords
        ds = fit_multi_gaussian2d(
            da2, n_components=1, initial_guesses=np.array([[1, 18, 21, 3, 3, 0]], float)
        )
        assert "x_world" not in ds and "y_world" not in ds


@pytest.mark.skipif(not has_dask(), reason="dask.array not available")
class TestDaskFitting:
    """Dask-backed success path."""

    def test_dask_backed_ok(self) -> None:
        ny, nx = 96, 96
        params = [dict(amp=1.2, x0=48.0, y0=44.0, sigma_x=4.0, sigma_y=4.0, theta=0.0)]
        img = make_gaussian_scene(ny, nx, components=params, offset=0.2, noise_std=0.03, seed=9)

        darr = da.from_array(img.data, chunks=(ny, nx))  # type: ignore[name-defined]
        img_da = xr.DataArray(darr, dims=("y", "x"))
        init = np.array([[1.0, 48.0, 44.0, 4.0, 4.0, 0.0]], float)

        ds = fit_multi_gaussian2d(
            img_da,
            n_components=1,
            initial_guesses=init,
            return_model=True,
            return_residual=True,
        )
        out = ds[["x0", "y0", "amplitude", "variance_explained", "success"]].compute()
        assert bool(out["success"].item()) is True
        assert 0.0 <= float(out["variance_explained"]) <= 1.0
        assert np.allclose(out["x0"].values, [48.0], atol=0.7)
        assert np.allclose(out["y0"].values, [44.0], atol=0.7)


class TestFailureModes:
    """Failure paths: masking, exceptions, covariance edge cases."""

    def test_threshold_mask_too_small_triggers_nan_outputs(self) -> None:
        ny, nx = 24, 24
        z = np.zeros((ny, nx), float)
        da2 = xr.DataArray(z, dims=("y", "x"))
        # fully masked by threshold
        ds = mg.fit_multi_gaussian2d(
            da2,
            n_components=1,
            min_threshold=1.0,
            max_threshold=None,
            return_residual=True,
            return_model=True,
        )
        assert bool(ds["success"]) is False
        for name in ("amplitude", "x0", "y0", "sigma_x", "sigma_y", "theta"):
            assert np.isnan(ds[name].values).all()
        assert np.isnan(ds["residual"].values).all()
        assert np.isnan(ds["model"].values).all()
        assert np.isnan(float(ds["variance_explained"]))

    def test_curve_fit_exception_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ny, nx = 40, 40
        y, x = np.mgrid[0:ny, 0:nx]
        z = np.exp(-((x - 20) ** 2 + (y - 18) ** 2) / (2 * 3.0**2))
        da2 = xr.DataArray(z, dims=("y", "x"))

        def boom(*args, **kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(mg, "curve_fit", boom, raising=True)
        ds = mg.fit_multi_gaussian2d(
            da2,
            n_components=1,
            initial_guesses=np.array([[1.0, 20.0, 18.0, 3.0, 3.0, 0.0]]),
            return_residual=True,
        )
        assert bool(ds["success"]) is False
        assert np.isnan(ds["amplitude"].values).all()
        assert np.isnan(float(ds["variance_explained"]))

    def test_curve_fit_pcov_none_sets_perr_nan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ny, nx = 32, 32
        da2 = xr.DataArray(np.ones((ny, nx)), dims=("y", "x"))
        n = 2
        # popt layout: [offset, (amp,x0,y0,sx,sy,th)*n]
        popt = np.array(
            [0.1, 1.0, 10.0, 12.0, 2.0, 2.0, 0.0, 0.7, 20.0, 8.0, 3.0, 1.5, 0.1],
            float,
        )

        def fake_fit(func, xy, z, p0=None, bounds=None, maxfev=None):
            return popt, None  # pcov=None

        monkeypatch.setattr(mg, "curve_fit", fake_fit, raising=True)
        ds = mg.fit_multi_gaussian2d(
            da2,
            n_components=n,
            initial_guesses=np.array([[1, 10, 12, 2, 2, 0], [0.7, 20, 8, 3, 1.5, 0.1]], float),
            return_model=False,
        )
        assert bool(ds["success"]) is True
        for name in (
            "amplitude_err",
            "x0_err",
            "y0_err",
            "sigma_x_err",
            "sigma_y_err",
            "theta_err",
        ):
            assert np.isnan(ds[name].values).all()


class TestAPIHelpers:
    """Helper API coverage: initial guesses formats, bounds merging."""

    def test_initial_guess_formats_all_paths(self) -> None:
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

    def test_bounds_merge_all_keys_and_per_component(self) -> None:
        base_lb, base_ub = mg._default_bounds_multi((32, 48), 2)
        user = {
            "offset": (-1.0, 1.0),
            "amplitude": [(0.0, 2.0), (0.0, 3.0)],
            "x0": (0.0, 47.0),
            "y0": (0.0, 31.0),
            "sigma_x": (0.5, 10.0),
            "sigma_y": (0.5, 10.0),
            "theta": (-1.57, 1.57),
        }
        lb, ub = mg._merge_bounds_multi(base_lb, base_ub, user, 2)
        # spot-check positions in packed layout
        assert lb[0] == -1.0 and ub[0] == 1.0        # offset
        assert lb[1] == 0.0 and ub[1] == 2.0         # comp0 amp
        assert lb[7] == 0.0 and ub[7] == 3.0         # comp1 amp
        assert lb[2] == 0.0 and ub[2] == 47.0        # x0 applied to all comps
        assert lb[8] == 0.0 and ub[8] == 47.0

