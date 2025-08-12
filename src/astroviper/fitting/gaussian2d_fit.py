# file: gaussian2d_fit.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import dask.array as da
import xarray as xr
from scipy.optimize import curve_fit
from numpy.typing import ArrayLike

try:
    from astropy.wcs import WCS  # optional
except Exception:  # pragma: no cover
    WCS = None  # type: ignore


Number = Union[int, float]
ArrayOrDA = Union[np.ndarray, da.Array, xr.DataArray]


# ---------- Gaussian model & helpers ----------

def _gauss2d_model(coords: Tuple[np.ndarray, np.ndarray],
                   amp: float, x0: float, y0: float,
                   sigma_x: float, sigma_y: float,
                   theta: float, offset: float) -> np.ndarray:
    """
    Elliptical 2D Gaussian rotated by theta, plus constant offset.
    coords: (X, Y) meshgrid in pixel coordinates.
    """
    X, Y = coords
    ct, st = np.cos(theta), np.sin(theta)
    x = X - x0
    y = Y - y0
    a = (ct**2) / (2 * sigma_x**2) + (st**2) / (2 * sigma_y**2)
    b = (-st * ct) / (2 * sigma_x**2) + (st * ct) / (2 * sigma_y**2)
    c = (st**2) / (2 * sigma_x**2) + (ct**2) / (2 * sigma_y**2)
    return offset + amp * np.exp(-(a * x**2 + 2 * b * x * y + c * y**2))


def _fwhm_from_sigma(sigma: float) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


def _moments_initial_guess(z: np.ndarray,
                           min_threshold: Optional[Number],
                           max_threshold: Optional[Number]) -> Tuple[float, float, float, float, float, float, float]:
    """
    Moments-based initial guess. Conservative defaults if degenerate.
    """
    ny, nx = z.shape
    Y, X = np.mgrid[0:ny, 0:nx]

    mask = np.ones_like(z, dtype=bool)
    if min_threshold is not None:
        mask &= z >= min_threshold
    if max_threshold is not None:
        mask &= z <= max_threshold

    z_masked = np.where(mask, z, np.nan)
    z_valid = np.where(np.isfinite(z_masked), z_masked, np.nan)

    # Use positive weights; shift to non-negative to stabilize moments on noisy backgrounds
    z_shift = z_valid - np.nanmin(z_valid)
    z_pos = np.where(np.isfinite(z_shift), z_shift, np.nan)

    total = np.nansum(z_pos)
    if not np.isfinite(total) or total <= 0:
        # Fallback: center with small sigmas, amplitude from robust stats
        med = float(np.nanmedian(z))
        p95 = float(np.nanpercentile(z, 95))
        amp = max(p95 - med, 1e-3)
        return (amp, nx / 2.0, ny / 2.0, max(nx, 1) / 6.0, max(ny, 1) / 6.0, 0.0, med)

    x0 = float(np.nansum(X * z_pos) / total)
    y0 = float(np.nansum(Y * z_pos) / total)
    x_var = float(np.nansum(((X - x0) ** 2) * z_pos) / total)
    y_var = float(np.nansum(((Y - y0) ** 2) * z_pos) / total)

    # Robust amplitude/offset
    med = float(np.nanmedian(z_valid))
    max_val = float(np.nanmax(z_valid))
    amp = max(max_val - med, 1e-3)

    sigma_x = max(np.sqrt(max(x_var, 1e-6)), 1.0)
    sigma_y = max(np.sqrt(max(y_var, 1e-6)), 1.0)
    theta = 0.0
    offset = med
    return (amp, x0, y0, sigma_x, sigma_y, theta, offset)


def _default_bounds(shape: Tuple[int, int]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    ny, nx = shape
    # Lower/upper bounds to keep optimizer stable.
    lb = (0.0,   -1.0,   -1.0,   0.5,  0.5,  -np.pi/2, -np.inf)
    ub = (np.inf, nx+1., ny+1., max(nx, 2.), max(ny, 2.), np.pi/2, np.inf)
    return lb, ub


# ---------- Single-plane fitter (NumPy) ----------

def _fit_plane_numpy(z2d: np.ndarray,
                     min_threshold: Optional[Number],
                     max_threshold: Optional[Number],
                     user_init: Optional[Sequence[float]],
                     user_bounds: Optional[Tuple[Sequence[float], Sequence[float]]]
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit one 2D plane. Returns (params, perr). On failure, NaNs.
    """
    if z2d.ndim != 2:
        raise ValueError("Internal error: _fit_plane_numpy expects a 2D array.")

    ny, nx = z2d.shape
    Y, X = np.mgrid[0:ny, 0:nx]

    # Mask by thresholds
    mask = np.ones_like(z2d, dtype=bool)
    if min_threshold is not None:
        mask &= z2d >= min_threshold
    if max_threshold is not None:
        mask &= z2d <= max_threshold

    # Ensure we have enough points
    if mask.sum() < 7:
        return (np.full(7, np.nan), np.full(7, np.nan))

    init = tuple(user_init) if user_init is not None else _moments_initial_guess(z2d, min_threshold, max_threshold)
    bounds = user_bounds if user_bounds is not None else _default_bounds((ny, nx))

    # Flatten on valid mask
    x_flat = X[mask].astype(float)
    y_flat = Y[mask].astype(float)
    z_flat = z2d[mask].astype(float)

    # Guard: curve_fit can fail on ill-conditioned data
    try:
        popt, pcov = curve_fit(
            lambda xy, amp, x0, y0, sx, sy, th, off: _gauss2d_model(xy, amp, x0, y0, sx, sy, th, off),
            (x_flat, y_flat),
            z_flat,
            p0=init,
            bounds=bounds,
            maxfev=20000,
            # Using default 'trf' via bounds; robust losses require least_squares, which curve_fit wraps internally
        )
        perr = np.sqrt(np.diag(pcov)) if (pcov is not None and np.all(np.isfinite(pcov))) else np.full_like(popt, np.nan)
    except Exception:
        return (np.full(7, np.nan), np.full(7, np.nan))

    return popt, perr


# ---------- World coordinate helpers ----------

def _world_from_coords(
    x0: float,
    y0: float,
    da2d: xr.DataArray,
    dim_x: str,
    dim_y: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Return world coords for (x0, y0) if possible.
    - If 1D coordinates exist for dims, interpolate.
    - If an astropy WCS exists in attrs['wcs'], attempt pixel->world.
    """
    wx = wy = None

    # Prefer xarray 1D coordinates
    try:
        cx = da2d.coords[dim_x]
        cy = da2d.coords[dim_y]
        if cx.ndim == 1 and cy.ndim == 1 and np.all(np.isfinite(cx)) and np.all(np.isfinite(cy)):
            # 'why': linear interp assumes monotonic 1D coords; typical in images
            wx = float(np.interp(x0, np.arange(cx.size), np.asarray(cx)))
            wy = float(np.interp(y0, np.arange(cy.size), np.asarray(cy)))
            return wx, wy
    except Exception:
        pass

    # Fallback to WCS in attrs
    try:
        wcs = da2d.attrs.get("wcs", None)
        if WCS is not None and isinstance(wcs, WCS):
            # astropy uses (x, y) in zero-based pixel convention when origin=0
            world = wcs.all_pix2world(np.array([[x0, y0]]), 0)
            if world.shape[-1] >= 2:
                wx, wy = float(world[0, 0]), float(world[0, 1])
    except Exception:
        pass

    return wx, wy


# ---------- xarray vectorized wrapper ----------

def _fit_plane_wrapper(
    z2d: np.ndarray,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
) -> Tuple[
        float, float, float, float, float, float, float,   # params
        float, float, float, float, float, float, float,   # errors
        float, float, float                                 # fwhm_major, fwhm_minor, peak
    ]:
    popt, perr = _fit_plane_numpy(
        z2d, min_threshold=min_threshold, max_threshold=max_threshold,
        user_init=None, user_bounds=None
    )

    amp, x0, y0, sx, sy, th, off = popt
    e_amp, e_x0, e_y0, e_sx, e_sy, e_th, e_off = perr

    fwhm_major = _fwhm_from_sigma(max(sx, sy)) if np.isfinite(sx) and np.isfinite(sy) else np.nan
    fwhm_minor = _fwhm_from_sigma(min(sx, sy)) if np.isfinite(sx) and np.isfinite(sy) else np.nan
    peak = amp + off if np.isfinite(amp) and np.isfinite(off) else np.nan

    return (
        amp, x0, y0, sx, sy, th, off,
        e_amp, e_x0, e_y0, e_sx, e_sy, e_th, e_off,
        fwhm_major, fwhm_minor, peak
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
            return da.dims[-1], da.dims[-2]  # (x, y) like ordering
        raise ValueError("For arrays with ndim != 2, you must specify two dims (names or indices).")
    if len(dims) != 2:
        raise ValueError("dims must be a sequence of exactly two items.")
    dim_names = []
    for d in dims:
        if isinstance(d, int):
            dim_names.append(da.dims[d])
        else:
            if d not in da.dims:
                raise ValueError(f"Dimension '{d}' not found in DataArray.")
            dim_names.append(d)
    # Normalize to (x_dim, y_dim) order by convention: last index is x progressing fastest
    return tuple(dim_names)  # type: ignore


# ---------- Public API ----------

def fit_gaussian2d(
    data: ArrayOrDA,
    dims: Optional[Sequence[Union[str, int]]] = None,
    min_threshold: Optional[Number] = None,
    max_threshold: Optional[Number] = None,
) -> xr.Dataset:
    """
    Fit an elliptical 2D Gaussian (per plane) on an array.

    Parameters
    ----------
    data
        numpy.ndarray, dask.array.Array, or xarray.DataArray.
    dims
        Two dims identifying the plane axes. Use axis indices or names (for DataArray).
        If data is 2D, this can be omitted.
    min_threshold, max_threshold
        Ignore pixels outside [min_threshold, max_threshold] during fitting.

    Returns
    -------
    xarray.Dataset
        Variables:
          - amplitude, x0, y0, sigma_x, sigma_y, theta, offset
          - *_err: 1σ uncertainties from the covariance
          - fwhm_major, fwhm_minor, peak
          - x_world, y_world (if inferable)
        Coordinates and shape follow all non-fit dims.
    """
    da = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da, dims)

    # Move fit dims to the end for a clean core signature [..., y, x]
    da_tr = da.transpose(*(d for d in da.dims if d not in (dim_y, dim_x)), dim_y, dim_x)

    core_dims = [dim_y, dim_x]
    output_dtypes = [np.float64] * (7 + 7 + 3)  # params + errs + extras

    # Vectorized apply across all other dims, parallelizing if Dask-backed
    results = xr.apply_ufunc(
        _fit_plane_wrapper,
        da_tr,
        input_core_dims=[core_dims],
        output_core_dims=[[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []],
        vectorize=True,
        dask="parallelized",
        output_dtypes=output_dtypes,
        kwargs=dict(min_threshold=min_threshold, max_threshold=max_threshold),
    )

    (amp, x0, y0, sx, sy, th, off,
     e_amp, e_x0, e_y0, e_sx, e_sy, e_th, e_off,
     fwhm_maj, fwhm_min, peak) = results

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
        )
    )

    # World coordinates per-plane (best-effort)
    # 'why': world mapping depends on plane; compute once per element using map_blocks-like vectorization
    def _world_map_block(x0_arr: np.ndarray, y0_arr: np.ndarray, block: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
        # x0_arr, y0_arr have the same non-core dims as block without y/x
        # We can't easily pass sub-DataArray per plane here; fallback using 1D coords interpolation
        # Assumes coordinates are separable along each axis
        try:
            cx = block.coords[dim_x].values
            cy = block.coords[dim_y].values
            # Broadcast interpolation per element
            wx = np.interp(x0_arr, np.arange(cx.size), cx)
            wy = np.interp(y0_arr, np.arange(cy.size), cy)
            return wx, wy
        except Exception:
            wx = np.full_like(x0_arr, np.nan, dtype=float)
            wy = np.full_like(y0_arr, np.nan, dtype=float)
            return wx, wy

    # We only attempt simple 1D coord interpolation; WCS requires context not passed by apply_ufunc.
    try:
        wx, wy = _world_map_block(ds["x0"].values, ds["y0"].values, da_tr)
        ds["x_world"] = (ds["x0"].dims, wx)
        ds["y_world"] = (ds["y0"].dims, wy)
    except Exception:
        # If we cannot compute world coords in bulk, leave them out silently.
        pass

    return ds

