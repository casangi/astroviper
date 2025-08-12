# file: src/astroviper/fitting/gaussian2d_fit.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Any
import numpy as np
import dask.array as da
import xarray as xr
from scipy.optimize import curve_fit

try:
    from astropy.wcs import WCS  # optional
except Exception:  # pragma: no cover
    WCS = None  # type: ignore

Number = Union[int, float]
ArrayOrDA = Union[np.ndarray, da.Array, xr.DataArray]


# ----------------------- Gaussian & helpers -----------------------

def _gauss2d_model(
    coords: Tuple[np.ndarray, np.ndarray],
    amp: float,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    theta: float,
    offset: float,
) -> np.ndarray:
    """Elliptical 2D Gaussian rotated by theta, plus constant offset."""
    X, Y = coords
    ct, st = np.cos(theta), np.sin(theta)
    x = X - x0
    y = Y - y0
    a = (ct**2) / (2 * sigma_x**2) + (st**2) / (2 * sigma_y**2)
    b = st * ct * (1.0 / (2 * sigma_y**2) - 1.0 / (2 * sigma_x**2))
    c = (st**2) / (2 * sigma_x**2) + (ct**2) / (2 * sigma_y**2)
    return offset + amp * np.exp(-(a * x**2 + 2 * b * x * y + c * y**2))


def _fwhm_from_sigma(sigma: float) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


def _moments_initial_guess(
    z: np.ndarray,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
) -> Tuple[float, float, float, float, float, float, float]:
    """Robust moments-based seed; falls back to center+RMS if degenerate."""
    ny, nx = z.shape
    Y, X = np.mgrid[0:ny, 0:nx]

    mask = np.ones_like(z, dtype=bool)
    if min_threshold is not None:
        mask &= z >= min_threshold
    if max_threshold is not None:
        mask &= z <= max_threshold

    z_masked = np.where(mask, z, np.nan)
    z_valid = np.where(np.isfinite(z_masked), z_masked, np.nan)

    min_val = np.nanmin(z_valid)
    z_pos = z_valid - min_val

    total = np.nansum(z_pos)
    if not np.isfinite(total) or total <= 0:
        med = float(np.nanmedian(z))
        p95 = float(np.nanpercentile(z, 95))
        amp = max(p95 - med, 1e-3)
        return (amp, nx / 2.0, ny / 2.0, max(nx, 1) / 6.0, max(ny, 1) / 6.0, 0.0, med)

    x0 = float(np.nansum(X * z_pos) / total)
    y0 = float(np.nansum(Y * z_pos) / total)
    x_var = float(np.nansum(((X - x0) ** 2) * z_pos) / total)
    y_var = float(np.nansum(((Y - y0) ** 2) * z_pos) / total)

    med = float(np.nanmedian(z_valid))
    max_val = float(np.nanmax(z_valid))
    amp = max(max_val - med, 1e-3)

    sigma_x = max(np.sqrt(max(x_var, 1e-12)), 1.0)
    sigma_y = max(np.sqrt(max(y_var, 1e-12)), 1.0)
    theta = 0.0
    offset = med
    return (amp, x0, y0, sigma_x, sigma_y, theta, offset)


def _default_bounds(shape: Tuple[int, int]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    ny, nx = shape
    lb = (0.0, -1.0, -1.0, 0.5, 0.5, -np.pi / 2, -np.inf)
    ub = (np.inf, nx + 1.0, ny + 1.0, max(nx, 2.0), max(ny, 2.0), np.pi / 2, np.inf)
    return lb, ub


# ----------------------- Single-plane fitter -----------------------

def _fit_plane_numpy(
    z2d: np.ndarray,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    user_init: Optional[Sequence[float]],
    user_bounds: Optional[Tuple[Sequence[float], Sequence[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one 2D plane. Returns (best_params[7], perr[7]); NaNs on failure."""
    if z2d.ndim != 2:
        raise ValueError("Internal: _fit_plane_numpy expects a 2D array.")

    ny, nx = z2d.shape
    Y, X = np.mgrid[0:ny, 0:nx]

    mask = np.ones_like(z2d, dtype=bool)
    if min_threshold is not None:
        mask &= z2d >= min_threshold
    if max_threshold is not None:
        mask &= z2d <= max_threshold
    if mask.sum() < 7:
        return (np.full(7, np.nan), np.full(7, np.nan))

    init = tuple(user_init) if user_init is not None else _moments_initial_guess(z2d, min_threshold, max_threshold)
    bounds = user_bounds if user_bounds is not None else _default_bounds((ny, nx))

    x_flat = X[mask].astype(float)
    y_flat = Y[mask].astype(float)
    z_flat = z2d[mask].astype(float)

    try:
        popt, pcov = curve_fit(
            lambda xy, amp, x0, y0, sx, sy, th, off: _gauss2d_model(xy, amp, x0, y0, sx, sy, th, off),
            (x_flat, y_flat),
            z_flat,
            p0=init,
            bounds=bounds,
            maxfev=20000,
        )
        perr = (
            np.sqrt(np.diag(pcov))
            if (pcov is not None and np.all(np.isfinite(pcov)))
            else np.full_like(popt, np.nan)
        )
    except Exception:
        return (np.full(7, np.nan), np.full(7, np.nan))

    return popt, perr


# ----------------------- Vectorized wrapper -----------------------

def _fit_plane_wrapper(
    z2d: np.ndarray,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    return_model: bool,
) -> tuple:
    popt, perr = _fit_plane_numpy(
        z2d,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        user_init=None,
        user_bounds=None,
    )

    amp, x0, y0, sx, sy, th, off = popt
    e_amp, e_x0, e_y0, e_sx, e_sy, e_th, e_off = perr

    fwhm_major = _fwhm_from_sigma(max(sx, sy)) if np.isfinite(sx) and np.isfinite(sy) else np.nan
    fwhm_minor = _fwhm_from_sigma(min(sx, sy)) if np.isfinite(sx) and np.isfinite(sy) else np.nan
    peak = amp + off if np.isfinite(amp) and np.isfinite(off) else np.nan

    ny, nx = z2d.shape
    Y, X = np.mgrid[0:ny, 0:nx]
    if np.any(~np.isfinite(popt)):
        residual2d = np.full_like(z2d, np.nan, dtype=float)
        model2d = np.full_like(z2d, np.nan, dtype=float)
    else:
        model2d = _gauss2d_model((X, Y), amp, x0, y0, sx, sy, th, off)
        residual2d = z2d.astype(float) - model2d  # residual = data - model

    return (
        amp, x0, y0, sx, sy, th, off,
        e_amp, e_x0, e_y0, e_sx, e_sy, e_th, e_off,
        fwhm_major, fwhm_minor, peak,
        residual2d,
        (model2d if return_model else np.full_like(z2d, np.nan, dtype=float)),
    )


def _ensure_dataarray(data: ArrayOrDA) -> xr.DataArray:
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, (np.ndarray, da.Array)):
        dims = [f"dim_{i}" for i in range(data.ndim)]
        coords = {d: np.arange(s, dtype=float) for d, s in zip(dims, data.shape)}
        return xr.DataArray(data, dims=dims, coords=coords, name="data")
    raise TypeError("Unsupported input type. Use numpy.ndarray, dask.array.Array, or xarray.DataArray.")


def _resolve_dims(da: xr.DataArray, dims: Optional[Sequence[Union[str, int]]]) -> Tuple[str, str]:
    if dims is None:
        if da.ndim == 2:
            return da.dims[-1], da.dims[-2]  # last is x, second-last is y
        raise ValueError("For arrays with ndim != 2, specify two dims (names or indices).")
    if len(dims) != 2:
        raise ValueError("dims must have length 2.")
    out: list[str] = []
    for d in dims:
        if isinstance(d, int):
            out.append(da.dims[d])
        else:
            if d not in da.dims:
                raise ValueError(f"Dimension '{d}' not found in DataArray.")
            out.append(d)
    return out[0], out[1]


# ----------------------- Public API -----------------------

def fit_gaussian2d(
    data: ArrayOrDA,
    dims: Optional[Sequence[Union[str, int]]] = None,
    min_threshold: Optional[Number] = None,
    max_threshold: Optional[Number] = None,
    return_model: bool = False,
) -> xr.Dataset:
    """
    Fit an elliptical 2D Gaussian (per plane).
    Returns an xarray.Dataset with parameters, uncertainties, FWHMs, peak,
    residual (data - model), and optionally model.
    """
    da = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da, dims)

    # Move fit dims to the end -> [..., y, x]
    da_tr = da.transpose(*(d for d in da.dims if d not in (dim_y, dim_x)), dim_y, dim_x)

    core_dims = [dim_y, dim_x]
    # params + errs + extras + residual + model
    output_dtypes = [np.float64] * (7 + 7 + 3) + [np.float64] + [np.float64]

    results = xr.apply_ufunc(
        _fit_plane_wrapper,
        da_tr,
        input_core_dims=[core_dims],
        output_core_dims=[
            [], [], [], [], [], [], [],   # params
            [], [], [], [], [], [], [],   # errors
            [], [], [],                   # extras
            [dim_y, dim_x],               # residual 2D
            [dim_y, dim_x],               # model 2D
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=output_dtypes,
        kwargs=dict(min_threshold=min_threshold, max_threshold=max_threshold, return_model=return_model),
    )

    (amp, x0, y0, sx, sy, th, off,
     e_amp, e_x0, e_y0, e_sx, e_sy, e_th, e_off,
     fwhm_maj, fwhm_min, peak,
     residual_da,
     model_da) = results

    ds = xr.Dataset(
        data_vars=dict(
            amplitude=amp,
            x0=x0, y0=y0,
            sigma_x=sx, sigma_y=sy,
            theta=th, offset=off,
            amplitude_err=e_amp,
            x0_err=e_x0, y0_err=e_y0,
            sigma_x_err=e_sx, sigma_y_err=e_sy,
            theta_err=e_th, offset_err=e_off,
            fwhm_major=fwhm_maj, fwhm_minor=fwhm_min,
            peak=peak,
            residual=residual_da,
        )
    )
    if return_model:
        ds["model"] = model_da

    # Best-effort world coords via 1-D coordinate interpolation.
    try:
        xw, yw = _world_map_bulk(ds["x0"].values, ds["y0"].values, da_tr, dim_x, dim_y)
        ds["x_world"] = (ds["x0"].dims, xw)
        ds["y_world"] = (ds["y0"].dims, yw)
    except Exception:
        pass

    return ds


def _world_map_bulk(
    x0_arr: np.ndarray,
    y0_arr: np.ndarray,
    block: xr.DataArray,
    dim_x: str,
    dim_y: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized 1-D coord interpolation; silent NaNs on failure."""
    try:
        cx = block.coords[dim_x].values
        cy = block.coords[dim_y].values
        wx = np.interp(x0_arr, np.arange(cx.size), cx)
        wy = np.interp(y0_arr, np.arange(cy.size), cy)
        return wx, wy
    except Exception:
        wx = np.full_like(x0_arr, np.nan, dtype=float)
        wy = np.full_like(y0_arr, np.nan, dtype=float)
        return wx, wy


# ----------------------- Quick visual check helper -----------------------

def quicklook_gaussian2d(
    data: ArrayOrDA,
    result: xr.Dataset,
    dims: Optional[Sequence[Union[str, int]]] = None,
    indexer: Optional[dict[str, Any]] = None,
) -> None:
    """
    Quick plot of data / model / residual for a selected plane.
    Pass `indexer` to pick non-fit dims (e.g. {'time': 0, 'chan': 5}).
    """
    import matplotlib.pyplot as plt  # local import to keep dependency optional

    da = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da, dims)
    da_tr = da.transpose(*(d for d in da.dims if d not in (dim_y, dim_x)), dim_y, dim_x)

    # Select plane
    if da_tr.ndim > 2:
        if indexer is None:
            indexer = {d: 0 for d in da_tr.dims[:-2]}
        data2d = da_tr.isel(**indexer)
        if "model" in result:
            model2d = result["model"].transpose(*da_tr.dims).isel(**indexer)
        else:
            model2d = _rebuild_model_from_params(result, indexer, data2d, dim_x, dim_y)
    else:
        data2d = da_tr
        model2d = result["model"] if "model" in result else _rebuild_model_from_params(result, None, data2d, dim_x, dim_y)

    residual = data2d - model2d

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(np.asarray(data2d), origin="lower"); axes[0].set_title("Data")
    axes[1].imshow(np.asarray(model2d), origin="lower"); axes[1].set_title("Model")
    axes[2].imshow(np.asarray(residual), origin="lower"); axes[2].set_title("Residual")
    for ax in axes:
        ax.set_xlabel(dim_x)
        ax.set_ylabel(dim_y)
    plt.show()


def _rebuild_model_from_params(
    result: xr.Dataset,
    indexer: Optional[dict[str, Any]],
    data2d: xr.DataArray,
    dim_x: str,
    dim_y: str,
) -> xr.DataArray:
    """Fallback when model wasn't requested during fit (keeps coords)."""
    sel = {d: indexer[d] for d in result.dims if indexer and d in indexer}
    amp = float(result["amplitude"].isel(**sel) if sel else result["amplitude"])
    x0 = float(result["x0"].isel(**sel) if sel else result["x0"])
    y0 = float(result["y0"].isel(**sel) if sel else result["y0"])
    sx = float(result["sigma_x"].isel(**sel) if sel else result["sigma_x"])
    sy = float(result["sigma_y"].isel(**sel) if sel else result["sigma_y"])
    th = float(result["theta"].isel(**sel) if sel else result["theta"])
    off = float(result["offset"].isel(**sel) if sel else result["offset"])
    ny, nx = data2d.sizes[dim_y], data2d.sizes[dim_x]
    Y, X = np.mgrid[0:ny, 0:nx]
    model_np = _gauss2d_model((X, Y), amp, x0, y0, sx, sy, th, off)
    return xr.DataArray(model_np, dims=(dim_y, dim_x),
                        coords={dim_y: data2d.coords[dim_y], dim_x: data2d.coords[dim_x]})

