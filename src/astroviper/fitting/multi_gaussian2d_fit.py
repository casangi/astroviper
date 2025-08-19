# file: src/astroviper/fitting/multi_gaussian2d_fit.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Any, Dict, List
import numpy as np
import xarray as xr
import dask.array as da
from scipy.optimize import curve_fit

Number = Union[int, float]
ArrayOrDA = Union[np.ndarray, da.Array, xr.DataArray]

# ----------------------- Core Gaussian pieces -----------------------

def _gauss2d_component(
    X: np.ndarray,
    Y: np.ndarray,
    amp: float,
    x0: float,
    y0: float,
    sx: float,
    sy: float,
    th: float,
) -> np.ndarray:
    """
    Elliptical rotated 2D Gaussian without offset; used as a building block.
    """
    ct, st = np.cos(th), np.sin(th)
    x = X - x0
    y = Y - y0
    a = (ct**2) / (2 * sx**2) + (st**2) / (2 * sy**2)
    b = st * ct * (1.0 / (2 * sy**2) - 1.0 / (2 * sx**2))
    c = (st**2) / (2 * sx**2) + (ct**2) / (2 * sy**2)
    return amp * np.exp(-(a * x**2 + 2 * b * x * y + c * y**2))

_SIG2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
_FWHM2SIG = 1.0 / _SIG2FWHM

def _fwhm_from_sigma(sigma):
    return _SIG2FWHM * np.asarray(sigma)

def _sigma_from_fwhm(fwhm):
    return _FWHM2SIG * np.asarray(fwhm)

# ----------------------- Parameter packing helpers -----------------------

def _pack_params(
    offset: float,
    amps: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    sx: np.ndarray,
    sy: np.ndarray,
    th: np.ndarray,
) -> np.ndarray:
    """
    Pack shared offset and per-component parameters into a 1-D vector:
    [offset, amp1,x01,y01,sx1,sy1,th1, ..., ampN,x0N,y0N,sxN,syN,thN]
    """
    n = int(amps.size)
    out = [offset]
    for i in range(n):
        out.extend([amps[i], x0[i], y0[i], sx[i], sy[i], th[i]])
    return np.asarray(out, dtype=float)


def _unpack_params(params: np.ndarray, n: int) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse of _pack_params.
    """
    offset = float(params[0])
    comps = np.asarray(params[1:], dtype=float).reshape(n, 6)
    amps = comps[:, 0]
    x0 = comps[:, 1]
    y0 = comps[:, 2]
    sx = comps[:, 3]
    sy = comps[:, 4]
    th = comps[:, 5]
    return offset, amps, x0, y0, sx, sy, th


# ----------------------- Multi-Gaussian model -----------------------

def _multi_gaussian2d_sum(
    X: np.ndarray,
    Y: np.ndarray,
    params: np.ndarray,
    n: int,
) -> np.ndarray:
    """
    Sum of N elliptical Gaussians plus a shared constant offset.
    """
    offset, amps, x0, y0, sx, sy, th = _unpack_params(params, n)
    model = np.full_like(X, float(offset), dtype=float)
    for i in range(n):
        model += _gauss2d_component(X, Y, amps[i], x0[i], y0[i], sx[i], sy[i], th[i])
    return model


def _multi_model_flat(
    xy: Tuple[np.ndarray, np.ndarray],
    *params: float,
    n: int,
) -> np.ndarray:
    """
    Flattened wrapper for curve_fit. xy = (x_flat, y_flat); returns z_flat.
    """
    x_flat, y_flat = xy
    # Recreate shaped grids for evaluation; they must be same flat length
    # We evaluate directly in flat form to avoid extra allocations.
    # Build shaped arrays lazily only if needed.
    p = np.asarray(params, dtype=float)
    offset, amps, x0, y0, sx, sy, th = _unpack_params(p, n)

    # Evaluate each component directly on flats, then sum; use the same quad form
    ct = np.cos(th)
    st = np.sin(th)
    z = np.full_like(x_flat, offset, dtype=float)
    for i in range(n):
        xi = x_flat - x0[i]
        yi = y_flat - y0[i]
        a = (ct[i] ** 2) / (2 * sx[i] ** 2) + (st[i] ** 2) / (2 * sy[i] ** 2)
        b = st[i] * ct[i] * (1.0 / (2 * sy[i] ** 2) - 1.0 / (2 * sx[i] ** 2))
        c = (st[i] ** 2) / (2 * sx[i] ** 2) + (ct[i] ** 2) / (2 * sy[i] ** 2)
        z += amps[i] * np.exp(-(a * xi**2 + 2 * b * xi * yi + c * yi**2))
    return z


# ----------------------- Seeding and bounds -----------------------

def _greedy_peak_seeds(z: np.ndarray, n: int, excl_radius: int = 5) -> List[Tuple[int, int, float]]:
    """
    Very simple N-peak detector: greedily pick the top N pixels with local exclusion.

    Returns list of (y, x, value).
    """
    z = np.asarray(z, dtype=float)
    z_copy = z.copy()
    seeds: List[Tuple[int, int, float]] = []
    ny, nx = z.shape
    for _ in range(n):
        j, i = np.unravel_index(np.nanargmax(z_copy), z_copy.shape)
        val = float(z_copy[j, i])
        seeds.append((j, i, val))
        y0a = max(0, j - excl_radius)
        y1a = min(ny, j + excl_radius + 1)
        x0a = max(0, i - excl_radius)
        x1a = min(nx, i + excl_radius + 1)
        z_copy[y0a:y1a, x0a:x1a] = -np.inf
    return seeds


def _default_bounds_multi(
    shape: Tuple[int, int],
    n: int,
    x_rng: Optional[Tuple[float, float]] = None,
    y_rng: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Default bounds vectors for multi-Gaussian parameters (packed order).
    If x_rng/y_rng are provided (world coords), use them for x0/y0 bounds; otherwise
    fall back to pixel-index bounds derived from shape.
    """
    ny, nx = shape
    # offset
    lb = [ -np.inf ]
    ub = [  np.inf ]
    # components
    xlo, xhi = ((-1.0, nx + 1.0) if x_rng is None else (float(x_rng[0]), float(x_rng[1])))
    ylo, yhi = ((-1.0, ny + 1.0) if y_rng is None else (float(y_rng[0]), float(y_rng[1])))
    sx_max = max((xhi - xlo), 2.0); sy_max = max((yhi - ylo), 2.0)

    for _ in range(n):
        # amp,   x0,   y0,     sx,     sy,   theta
        lb.extend([0.0, xlo,  ylo,   1e-3,   1e-3, -np.pi/2])
        ub.extend([np.inf, xhi, yhi, sx_max, sy_max,  np.pi/2])

    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

def _extract_params_from_comp_dicts(comp_list, n):
    """Parse a list of component dicts into arrays (amp, x0, y0, sx, sy, th).
    Accepts keys: amp|amplitude, x0, y0, sigma_x|sx|fwhm_major, sigma_y|sy|fwhm_minor, theta.
    Converts FWHM inputs to σ using _FWHM2SIG.
    Raises KeyError if required keys are missing.
    """
    amps = np.empty(n); x0 = np.empty(n); y0 = np.empty(n)
    sx = np.empty(n);  sy = np.empty(n);  th = np.empty(n)
    for i, comp in enumerate(comp_list):
        amps[i] = float(comp["amp"] if "amp" in comp else comp["amplitude"])
        x0[i]   = float(comp["x0"])
        y0[i]   = float(comp["y0"])
        sx_val  = comp.get("sigma_x", comp.get("sx"))
        sy_val  = comp.get("sigma_y", comp.get("sy"))
        if sx_val is None:
            fx = comp.get("fwhm_major")
            if fx is None:
                raise KeyError("component missing sigma_x/sx (or fwhm_major)")
            sx_val = float(fx) * _FWHM2SIG
        if sy_val is None:
            fy = comp.get("fwhm_minor")
            if fy is None:
                raise KeyError("component missing sigma_y/sy (or fwhm_minor)")
            sy_val = float(fy) * _FWHM2SIG
        sx[i]   = float(sx_val)
        sy[i]   = float(sy_val)
        th[i]   = float(comp.get("theta", 0.0))
    return amps, x0, y0, sx, sy, th


def _normalize_initial_guesses(
    z2d: np.ndarray,
    n: int,
    init: Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]],
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
) -> np.ndarray:
    """
    Build an initial parameter vector for N components.

    Accepts:
      - None: auto-seed using greedy peaks.
      - array-like of shape (n, 6): columns [amp, x0, y0, sx, sy, th]; offset from median.
      - list/tuple of dicts (len n): keys amp|amplitude, x0, y0, sigma_x|sx, sigma_y|sy, theta; offset from median.
      - dict with keys:
          'offset': float (optional),
          'components': array-like (n, 6) or list/tuple of dicts as above.

    Returns a packed parameter vector as in _pack_params.
    """
    ny, nx = z2d.shape

    # Threshold mask for robust median
    mask = np.ones_like(z2d, dtype=bool)
    if min_threshold is not None:
        mask &= z2d >= float(min_threshold)
    if max_threshold is not None:
        mask &= z2d <= float(max_threshold)
    z_masked = np.where(mask, z2d, np.nan)
    med = float(np.nanmedian(z_masked))

    # None → auto seeds
    if init is None:
        seeds = _greedy_peak_seeds(z_masked, n=n, excl_radius=max(3, max(ny, nx) // 50))
        amps = np.array([max(v - med, 1e-3) for (_, _, v) in seeds], dtype=float)
        x0 = np.array([float(x) for (_, x, _) in seeds], dtype=float)
        y0 = np.array([float(y) for (y, _, _) in seeds], dtype=float)
        sx = np.full(n, max(nx, ny) / 10.0, dtype=float)
        sy = np.full(n, max(nx, ny) / 10.0, dtype=float)
        th = np.zeros(n, dtype=float)
        return _pack_params(med, amps, x0, y0, sx, sy, th)

    # Allow a single top-level dict when n == 1 (wrap to list-of-dicts)
    from collections.abc import Mapping
    if isinstance(init, Mapping) and ("components" not in init and "offset" not in init):
        if n != 1:
            raise ValueError("Single-dict initial_guesses is only valid when n_components == 1.")
        init = [init]

    # NEW: top-level list/tuple of dicts (matches docstring)
    if isinstance(init, (list, tuple)) and len(init) > 0 and isinstance(init[0], dict):
        if len(init) != n:
            raise ValueError(f"initial_guesses list must have length n={n}")
        amps, x0, y0, sx, sy, th = _extract_params_from_comp_dicts(init, n)
        return _pack_params(med, amps, x0, y0, sx, sy, th)

    # Structured dict form
    if isinstance(init, dict) and ("components" in init or "offset" in init):
        offset = float(init.get("offset", med))
        comps = init.get("components", None)
        if comps is None:
            raise ValueError("init dict must include 'components' with shape (n,6) or list of dicts.")
        init = comps  # fallthrough to parse components and pack with provided offset

        if isinstance(init, np.ndarray):
            arr = np.asarray(init, dtype=float)
            if arr.shape != (n, 6):
                raise ValueError(f"init['components'] must have shape (n,6); got {arr.shape}")
            amps, x0, y0, sx, sy, th = [arr[:, k].astype(float) for k in range(6)]
            return _pack_params(offset, amps, x0, y0, sx, sy, th)

        if isinstance(init, (list, tuple)) and len(init) > 0 and isinstance(init[0], dict):
            if len(init) != n:
                raise ValueError(f"init['components'] must have length n={n}")
            amps = np.empty(n); x0 = np.empty(n); y0 = np.empty(n)
            sx = np.empty(n);  sy = np.empty(n);  th = np.empty(n)
            for i, comp in enumerate(init):
                amps[i] = float(comp["amp"] if "amp" in comp else comp["amplitude"])
                x0[i]  = float(comp["x0"])
                y0[i]  = float(comp["y0"])
                sx[i]  = float(comp.get("sigma_x", comp.get("sx")))
                sy[i]  = float(comp.get("sigma_y", comp.get("sy")))
                th[i]  = float(comp.get("theta", 0.0))
            return _pack_params(offset, amps, x0, y0, sx, sy, th)
    # Array/list form (numpy array or list-of-lists)
    arr = np.asarray(init, dtype=float)
    if arr.shape != (n, 6):
        raise ValueError(f"initial_guesses must have shape (n,6); got {arr.shape}")
    amps, x0, y0, sx, sy, th = [arr[:, k].astype(float) for k in range(6)]
    return _pack_params(med, amps, x0, y0, sx, sy, th)

def _merge_bounds_multi(
    base_lb: np.ndarray,
    base_ub: np.ndarray,
    user_bounds: Optional[Dict[str, Union[Tuple[float,float], Sequence[Tuple[float,float]]]]],
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge user-provided bounds into default [lb, ub].
    Keys allowed: 'offset','amp','x0','y0','sigma_x','sigma_y','theta'.
    Values may be a single (low, high) tuple applied to all components, or a
    sequence of length n with per-component tuples.
    """
    if user_bounds is None:
        return base_lb, base_ub
    lb = base_lb.copy(); ub = base_ub.copy()

    def _set_range(name: str, idx_in_comp: Optional[int], rng: Tuple[float,float], comp_idx: Optional[int] = None):
        lo, hi = float(rng[0]), float(rng[1])
        if name == "offset":
            lb[0] = lo; ub[0] = hi; return
        if idx_in_comp is None:
            # this should never be reached, so can't be covered
            # but leaving it in for defensive programming
            return # pragma: no cover
        if comp_idx is None:
            for i in range(n):
                j0 = 1 + i*6 + idx_in_comp
                lb[j0] = lo; ub[j0] = hi
        else:
            j0 = 1 + comp_idx*6 + idx_in_comp
            lb[j0] = lo; ub[j0] = hi

    # Normalize dictionary values
    mapping = {
        "offset": (None, "offset"),
        "amp": (0, "amp"),
        "amplitude": (0, "amp"),
        "x0": (1, "x0"),
        "y0": (2, "y0"),
        "sigma_x": (3, "sigma_x"),
        "sigma_y": (4, "sigma_y"),
        "fwhm_major": (3, "fwhm_major"),
        "fwhm_minor": (4, "fwhm_minor"),
        "theta": (5, "theta"),
    }
    # helper: convert FWHM ranges to σ if needed (for this merge routine)
    def _to_sigma_rng(canon_name: str, rng: Tuple[float, float]) -> Tuple[float, float]:
        lo, hi = float(rng[0]), float(rng[1])
        if canon_name in ("fwhm_major", "fwhm_minor"):
            return lo * _FWHM2SIG, hi * _FWHM2SIG
        return lo, hi

    for key, val in user_bounds.items():
        if key not in mapping:
            continue
        idx_in_comp, canon = mapping[key]
        if canon == "offset":
            _set_range("offset", None, tuple(val))  # type: ignore[arg-type]
            continue
        if isinstance(val, (list, tuple)) and len(val) == 2 and not isinstance(val[0], (list, tuple)):
            _set_range(canon, idx_in_comp, _to_sigma_rng(canon, (val[0], val[1])))  # same for all comps
        else:
            if len(val) != n:
                raise ValueError(f"bounds[{key!r}] length must be n={n}")
            for i, rng in enumerate(val):  # type: ignore[assignment]
                _set_range(canon, idx_in_comp, _to_sigma_rng(canon, tuple(rng)), comp_idx=i)  # type: ignore[arg-type]
    return lb, ub


# ----------------------- Single-plane multi fit -----------------------

def _fit_multi_plane_numpy(
    z2d: np.ndarray,
    n_components: int,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    initial_guesses: Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]],
    bounds: Optional[Dict[str, Union[Tuple[float,float], Sequence[Tuple[float,float]]]]],
    max_nfev: int,
    *,
    x1d: Optional[np.ndarray] = None,
    y1d: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit N Gaussians + shared offset to a single 2-D plane.

    Returns
    -------
    popt : (1 + 6N,)
        Best-fit packed parameter vector.
    perr : (1 + 6N,)
        1-sigma uncertainties from the covariance diagonal (NaN on failure).
    mask : (ny,nx) bool
        Mask used during fitting (thresholds).
    """
    if z2d.ndim != 2:
        # cannot cover using public API, defensive coding
        raise ValueError("Internal: _fit_multi_plane_numpy expects a 2-D array.") # pragma: no cover
    ny, nx = z2d.shape
    if x1d is None or y1d is None:
        # pixel-index grid
        Y, X = np.mgrid[0:ny, 0:nx]
        x_rng = None
        y_rng = None
    else:
        if y1d.shape[0] != ny or x1d.shape[0] != nx:
            raise ValueError("Length of y1d/x1d must match z2d shape.")
        # world grid
        Y = np.broadcast_to(y1d[:, None], (ny, nx))
        X = np.broadcast_to(x1d[None, :], (ny, nx))
        x_rng = (float(np.min(x1d)), float(np.max(x1d)))
        y_rng = (float(np.min(y1d)), float(np.max(y1d)))

    # thresholds
    mask = np.ones_like(z2d, dtype=bool)
    if min_threshold is not None:
        mask &= z2d >= min_threshold
    if max_threshold is not None:
        mask &= z2d <= max_threshold
    if mask.sum() < (1 + 6*n_components):
        # Not enough points relative to params
        return np.full(1 + 6*n_components, np.nan), np.full(1 + 6*n_components, np.nan), mask

    # seeds & bounds
    p0 = _normalize_initial_guesses(z2d, n_components, initial_guesses, min_threshold, max_threshold)
    lb0, ub0 = _default_bounds_multi((ny, nx), n_components, x_rng=x_rng, y_rng=y_rng)
    lb, ub = _merge_bounds_multi(lb0, ub0, bounds, n_components)

    # Ensure seeds within bounds
    p0 = np.clip(p0, lb + 1e-12, ub - 1e-12)

    x_flat = X[mask].astype(float)
    y_flat = Y[mask].astype(float)
    z_flat = z2d[mask].astype(float)

    try:
        popt, pcov = curve_fit(
            lambda xy, *params: _multi_model_flat(xy, *params, n=n_components),
            (x_flat, y_flat),
            z_flat,
            p0=p0,
            bounds=(lb, ub),
            maxfev=int(max_nfev),
        )
        perr = np.sqrt(np.diag(pcov)) if (pcov is not None and np.all(np.isfinite(pcov))) else np.full_like(popt, np.nan)
    except Exception:
        popt = np.full_like(p0, np.nan)
        perr = np.full_like(p0, np.nan)

    return popt, perr, mask


# ----------------------- Vectorized wrapper (for xarray) -----------------------

def _multi_fit_plane_wrapper(
    z2d: np.ndarray,
    y1d: np.ndarray,
    x1d: np.ndarray,
    n_components: int,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    initial_guesses: Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]],
    bounds: Optional[Dict[str, Union[Tuple[float,float], Sequence[Tuple[float,float]]]]],
    max_nfev: int,
    return_model: bool,
    return_residual: bool,
) -> tuple:
    """
    Vectorized kernel returning component-wise parameters plus diagnostics
    for one plane. Shapes:
      - component arrays: (n_components,)
      - scalars: ()
      - planes: (ny, nx)
    """
    popt, perr, mask = _fit_multi_plane_numpy(
        z2d=z2d,
        n_components=n_components,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        initial_guesses=initial_guesses,
        bounds=bounds,
        max_nfev=max_nfev,
        x1d=x1d, y1d=y1d,
    )

    ny, nx = z2d.shape
    # Y/X constructed identically in _fit_multi_plane_numpy; keep for model/residual paths
    if y1d.shape[0] != ny or x1d.shape[0] != nx:
        raise ValueError("Length of y1d/x1d must match z2d shape for world/pixel grids.")
    Y = np.broadcast_to(y1d[:, None], (ny, nx))
    X = np.broadcast_to(x1d[None, :], (ny, nx))

    success = bool(np.all(np.isfinite(popt)))
    if not success:
        # Fill with NaNs in expected shapes
        n = int(n_components)
        zeros = np.full(n, np.nan, dtype=float)
        amp = x0 = y0 = sx = sy = th = zeros
        amp_e = x0_e = y0_e = sx_e = sy_e = th_e = zeros
        fwhm_major = fwhm_minor = peak = zeros
        offset = np.nan; offset_e = np.nan
        model2d = np.full_like(z2d, np.nan, dtype=float)
        resid2d = np.full_like(z2d, np.nan, dtype=float)
        varexp = np.nan
        return (amp, x0, y0, sx, sy, th,
                amp_e, x0_e, y0_e, sx_e, sy_e, th_e,
                fwhm_major, fwhm_minor, peak,
                offset, offset_e,
                bool(False), varexp,
                resid2d, model2d)

    # Unpack best-fit and uncertainties
    n = int(n_components)
    offset, amp, x0, y0, sx, sy, th = _unpack_params(popt, n)
    _, amp_e, x0_e, y0_e, sx_e, sy_e, th_e = _unpack_params(perr, n)

    # Derived component metrics
    fwhm_major = _fwhm_from_sigma(np.maximum(sx, sy))
    fwhm_minor = _fwhm_from_sigma(np.minimum(sx, sy))
    peak = amp + offset

    # Build model & residual
    if return_model or return_residual:
        model2d_full = _multi_gaussian2d_sum(X, Y, popt, n)
        model2d = model2d_full if return_model else np.full_like(z2d, np.nan, dtype=float)
        resid2d = (z2d.astype(float) - model2d_full) if return_residual else np.full_like(z2d, np.nan, dtype=float)
    else:
        model2d = np.full_like(z2d, np.nan, dtype=float)
        resid2d = np.full_like(z2d, np.nan, dtype=float)

    # Variance explained on masked pixels
    if mask.any():
        z_masked = z2d.astype(float)[mask]
        r_masked = (z2d.astype(float) - _multi_gaussian2d_sum(X, Y, popt, n))[mask]
        tot = np.nanvar(z_masked)
        res = np.nanvar(r_masked)
        varexp = (1.0 - res / tot) if (np.isfinite(tot) and tot > 0) else np.nan
    else:
        # not coverable from public API call, defensive coding
        varexp = np.nan # pragma: no cover

    # Offset uncertainty is the first element of perr
    offset_e = float(perr[0])

    return (amp, x0, y0, sx, sy, th,
            amp_e, x0_e, y0_e, sx_e, sy_e, th_e,
            fwhm_major, fwhm_minor, peak,
            float(offset), offset_e,
            bool(True), float(varexp),
            resid2d, model2d)


def _ensure_dataarray(data: ArrayOrDA) -> xr.DataArray:
    """Normalize input to xarray.DataArray with generated dims/coords if needed."""
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, (np.ndarray, da.Array)):
        dims = [f"dim_{i}" for i in range(data.ndim)]
        coords = {d: np.arange(s, dtype=float) for d, s in zip(dims, data.shape)}
        return xr.DataArray(data, dims=dims, coords=coords, name="data")
    raise TypeError("Unsupported input type; use numpy.ndarray, dask.array.Array, or xarray.DataArray.")

def _resolve_dims(da: xr.DataArray, dims: Optional[Sequence[Union[str, int]]]) -> Tuple[str, str]:
    """
    Resolve plane dims to (x_dim, y_dim).

    Rules:
    - If 'dims' is provided: use those (names or indices).
    - Else, if DataArray has dims named 'x' and 'y': use ('x', 'y').
    - Else, if 2-D: assume last is x, second-last is y.
    - Else: error (ambiguous for ndim != 2 without explicit dims).
    """
    if dims is None:
        # ✅ new: prefer explicitly named axes when present
        if "x" in da.dims and "y" in da.dims:
            return "x", "y"
        if da.ndim == 2:
            return da.dims[-1], da.dims[-2]
        raise ValueError("For arrays with ndim != 2, specify two dims (names or indices).")

    if len(dims) != 2:
        raise ValueError("dims must be length-2.")

    out: List[str] = []
    for d in dims:
        if isinstance(d, int):
            out.append(da.dims[d])
        else:
            if d not in da.dims:
                raise ValueError(f"Dimension {d!r} not found.")
            out.append(d)
    return out[0], out[1]

def _axis_sign(coord: Optional[np.ndarray]) -> float:
    """+1 if strictly ascending, -1 if strictly descending, else +1."""
    if coord is None or coord.ndim != 1 or coord.size < 2:
        return 1.0
    c0, c1 = float(coord[0]), float(coord[1])
    return 1.0 if np.isfinite(c0) and np.isfinite(c1) and (c1 > c0) else -1.0

def _theta_pa_to_math(pa: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Convert PA (from +y toward +x) into math angle (from +x toward +y, CCW)
    in the index coordinate system whose axis directions are set by (sx, sy).
    """
    pa = np.asarray(pa, dtype=float)
    # Unit vector along major axis in *world* basis: (+x_world, +y_world)
    vx_w = np.sin(pa)
    vy_w = np.cos(pa)
    # Map to index basis by applying axis signs
    vx_i = vx_w * sx
    vy_i = vy_w * sy
    return np.arctan2(vy_i, vx_i)

def _theta_math_to_pa(theta: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """
    Convert math angle in index basis back to PA in world-like basis
    where PA is measured from +y toward +x.
    """
    theta = np.asarray(theta, dtype=float)
    vx_i = np.cos(theta)
    vy_i = np.sin(theta)
    # Map back to world basis by undoing the axis signs
    vx_w = vx_i / sx
    vy_w = vy_i / sy
    return np.arctan2(vx_w, vy_w)

def _convert_init_theta(
    init: Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]],
    to_math: bool,
    sx: float,
    sy: float,
    n: int,
) -> Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]]:
    """
    Return a copy of initial_guesses with theta converted to math-angle if to_math=True.
    Structure and other fields are preserved.
    """
    if init is None or not to_math:
        return init

    def _conv_arr(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=float, copy=True)
        if out.shape != (n, 6):
            return out
        out[:, 5] = _theta_pa_to_math(out[:, 5], sx, sy)
        return out

    def _conv_list_of_dicts(lst: Sequence[Dict[str, Number]]) -> List[Dict[str, Number]]:
        new = []
        for d in lst:
            dd = dict(d)
            if "theta" in dd:
                dd["theta"] = float(_theta_pa_to_math(np.array([dd["theta"]], float), sx, sy)[0])
            new.append(dd)
        return new

    if isinstance(init, np.ndarray):
        return _conv_arr(init)

    if isinstance(init, (list, tuple)) and (len(init) == n) and isinstance(init[0], dict):
        return _conv_list_of_dicts(init)  # type: ignore[return-value]

    if isinstance(init, dict) and ("components" in init):
        comps = init["components"]
        clone = dict(init)
        if isinstance(comps, np.ndarray):
            clone["components"] = _conv_arr(np.asarray(comps))
        elif isinstance(comps, (list, tuple)) and comps and isinstance(comps[0], dict):
            clone["components"] = _conv_list_of_dicts(comps)  # type: ignore[arg-type]
        return clone

    # defensive coding, should not be reached
    return init # pragma: no cover

def _extract_1d_coords_for_fit(
    original_input: ArrayOrDA,
    da_tr: xr.DataArray,
    coord_type: str,
    coords: Optional[Sequence[np.ndarray]],
    dim_y: str,
    dim_x: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y1d, x1d) arrays for model evaluation.

    Behavior:
    - If input is an xarray.DataArray:
        * coord_type controls behavior:
            - "world": use the DataArray's 1-D coords on (dim_y, dim_x)
            - "pixel": use index grids 0..N-1
    - If input is a NumPy/Dask array:
        * coord_type is ignored
        * if `coords=(x1d, y1d)` is provided, use those (world); otherwise use pixel indices.
    """
    ny, nx = int(da_tr.sizes[dim_y]), int(da_tr.sizes[dim_x])

    if isinstance(original_input, xr.DataArray):
        ctype = (coord_type or "world").lower()
        if ctype not in ("world", "pixel"):
            raise ValueError("coord_type must be 'world' or 'pixel' for DataArray inputs")
        if ctype == "pixel":
            return np.arange(ny, dtype=float), np.arange(nx, dtype=float)
        # world coords from the DataArray
        if (dim_x not in da_tr.coords) or (dim_y not in da_tr.coords):
            raise ValueError(f"DataArray is missing coords for dims ({dim_y}, {dim_x}) required for world fitting.")
        x1d = np.asarray(da_tr.coords[dim_x].values)
        y1d = np.asarray(da_tr.coords[dim_y].values)
        if x1d.ndim != 1 or y1d.ndim != 1 or x1d.size != nx or y1d.size != ny:
            raise ValueError("World coords must be 1-D and match the data shape along (y, x).")
        return y1d.astype(float), x1d.astype(float)

    # NumPy/Dask input: coord_type is ignored; pick by presence of coords
    if coords is not None:
        if len(coords) != 2:
            raise ValueError("coords must be a tuple/list of (x1d, y1d).")
        x1d, y1d = coords[0], coords[1]
        x1d = np.asarray(x1d, dtype=float)
        y1d = np.asarray(y1d, dtype=float)
        if x1d.ndim != 1 or y1d.ndim != 1 or x1d.size != nx or y1d.size != ny:
            raise ValueError("coords must be 1-D arrays with lengths matching (nx, ny).")
        return y1d, x1d

    # Fallback: pixel indices
    return np.arange(ny, dtype=float), np.arange(nx, dtype=float)


# ----------------------- Public API -----------------------

def fit_multi_gaussian2d(
    data: ArrayOrDA,
    n_components: int,
    dims: Optional[Sequence[Union[str, int]]] = None,
    *,
    min_threshold: Optional[Number] = None,
    max_threshold: Optional[Number] = None,
    initial_guesses: Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]] = None,
    bounds: Optional[Dict[str, Union[Tuple[float, float], Sequence[Tuple[float, float]]]]] = None,
    initial_is_fwhm: bool = True,
    max_nfev: int = 20000,
    return_model: bool = False,
    return_residual: bool = True,
    angle: str = "math",  # NEW: {"math","pa","auto"}
    coord_type: str = "world",
    coords: Optional[Sequence[np.ndarray]] = None,
) -> xr.Dataset:
    """
    Fit a sum of N rotated 2-D Gaussians (with a shared constant offset) to each plane.

    The model for one plane is:
        model(x, y) = offset + Σ_{i=1..N} amp_i · G_i(x, y; x0_i, y0_i, sigma_x_i, sigma_y_i, theta_i)

    Supports NumPy, Dask, and Xarray inputs, and vectorizes across all non-fit dims.
    When input is Dask-backed, computation is parallelized via
    ``xarray.apply_ufunc(dask="parallelized")``.

    Parameters
    ----------
    data: numpy.ndarray | dask.array.Array | xarray.DataArray
      Input array. If not a DataArray, it is wrapped with dims ('y','x') and generated numeric coords.

    n_components: int
      Number of Gaussian components (N ≥ 1).

    dims: Sequence[str | int] | None
      Two dims (names or indices) that define the fit plane (x, y). If omitted: uses ('x','y') if present; else for 2-D uses (last, second-last). Required for ndim ≠ 2 without ('x','y').

    min_threshold: float | None
      Inclusive lower threshold; pixels with values < min_threshold are ignored during the fit.

    max_threshold: float | None
      Inclusive upper threshold; pixels with values > max_threshold are ignored during the fit.
    initial_guesses: numpy.ndarray[(N,6)] | list[dict] | dict | None
    initial_guesses: numpy.ndarray[(N,6)] | list[dict] | dict | None
      Initial guesses. **Interpreted in FWHM units** for widths by default:
        • array shape (N,6): columns **[amp, x0, y0, fwhm_major, fwhm_minor, theta]**.
        • list of N dicts: keys {"amp"/"amplitude","x0","y0","fwhm_major"|"sigma_x"|"sx","fwhm_minor"|"sigma_y"|"sy","theta"}.
        • dict: {"offset": float (optional), "components": (N,6) array OR list[dict] as above}.
      If omitted, peaks are auto-seeded and offset defaults to the median of threshold-masked data.
      Note: θ in `initial_guesses` is interpreted per `angle`.
      FWHM are converted to σ internally for the optimizer. Set ``initial_is_fwhm=False`` only if
      your array-form guesses use σ columns instead.
    bounds: dict[str, (float,float) | Sequence[(float,float)]] | None
      Bounds to constrain parameters. Keys may include {"offset","amp"/"amplitude","x0","y0","fwhm_major","fwhm_minor","theta"}.
      Each value is either a single (low, high) tuple applied to all components, or a length-N sequence of (low, high) tuples for per-component bounds. To **fix** a parameter, set low == high.
    initial_is_fwhm: bool
     Default **True**. When ``True`` and ``initial_guesses`` is an array of shape (N,6), columns 3–4
     are treated as **FWHM** and converted to σ internally. Set to ``False`` only if your array guesses
     are already in σ. (Dict/list forms can use ``fwhm_major/fwhm_minor`` keys directly.)
    max_nfev: int
      Maximum function evaluations for the optimizer. Default: 20000.

    return_model: bool
      If True, include the fitted model plane(s) as variable ``model``.

    return_residual: bool
      If True, include residual plane(s) (``data − model``) as variable ``residual``.

    angle: {"math","pa","auto"}
      Controls how ``theta`` is interpreted (inputs) and reported (outputs).
        • "math": standard math angle (from +x toward +y, CCW) in data axes.
        • "pa": position angle (from +y toward +x).
        • "auto": if axes are left-handed (sign(Δx)·sign(Δy) < 0) treat as PA; otherwise math. Outputs follow the same convention.
    coord_type: {"world","pixel"}, default "world"
      Applies only to xarray.DataArray inputs: "world" uses the DataArray's 1-D coords; "pixel" uses index grids.
      Ignored for NumPy/Dask inputs.
    coords: tuple[np.ndarray, np.ndarray] | list[np.ndarray] | None
      For NumPy/Dask inputs only: provide (x1d, y1d) to fit in world coordinates. Ignored for DataArray inputs.

    Returns
    -------
    xarray.Dataset
        Per-plane results with a new core dim ``component`` (length N).

        Per-component parameters:
            - amplitude(component), x0(component), y0(component),
              sigma_x(component), sigma_y(component), theta(component)
        Per-component 1σ uncertainties:
            - amplitude_err(component), x0_err(component), y0_err(component),
              sigma_x_err(component), sigma_y_err(component), theta_err(component)
        Per-component derived:
            - fwhm_major(component), fwhm_minor(component), peak(component) [= amplitude + offset]
        Scalars:
            - offset, offset_err, success (bool), variance_explained
        Optional planes:
            - residual (if return_residual), model (if return_model)
        Optional world coordinates (per component):
            - x_world(component), y_world(component) are added **only** if both fit axes
              have 1-D numeric coordinate variables (ascending or descending).

    Examples
    --------
    Basic two-component fit on a 2-D array:
    >>> import numpy as np, xarray as xr
    >>> ny, nx = 128, 128
    >>> y, x = np.mgrid[0:ny, 0:nx]
    >>> g1 = np.exp(-((x-40)**2+(y-60)**2)/(2*3.0**2))
    >>> th = 0.4
    >>> xr_ = (x-90)*np.cos(th) + (y-30)*np.sin(th)
    >>> yr_ = -(x-90)*np.sin(th) + (y-30)*np.cos(th)
    >>> g2 = 0.7*np.exp(-(xr_**2)/(2*5.0**2) - (yr_**2)/(2*2.5**2))
    >>> img_da = xr.DataArray(g1 + g2 + 0.1 + 0.05*np.random.randn(ny, nx), dims=("y","x"))
    >>> init = np.array([[1.0, 40.0, 60.0, 3.0, 3.0, 0.0],
    ...                  [0.7, 90.0, 30.0, 5.0, 2.5, 0.4]])
    >>> ds = fit_multi_gaussian2d(img_da, n_components=2, initial_guesses=init, return_residual=True)

    Vectorized across a stack (auto-detects ('x','y') when present):
    >>> planes = [img_da + 0.01*np.random.randn(*img_da.shape) for _ in range(3)]
    >>> cube = xr.concat(planes, dim="time")  # dims ('time','y','x')
    >>> ds3 = fit_multi_gaussian2d(cube, n_components=2, initial_guesses=init)

    3-D array with explicit plane dims:
    >>> vol = xr.DataArray(np.zeros((2, ny, nx)), dims=("z","y","x"))
    >>> ds_z0 = fit_multi_gaussian2d(vol, n_components=1, dims=("x","y"))

    Using bounds (including fixed parameters via equal bounds):
    >>> bounds = {"sigma_x": [(1.0, 1.0), (2.0, 4.0)], "offset": (0.05, 0.2)}
    >>> ds_b = fit_multi_gaussian2d(img_da, n_components=2, initial_guesses=init, bounds=bounds)

    Angle conventions:
    >>> # Report and interpret theta as position angle:
    >>> ds_pa = fit_multi_gaussian2d(img_da, n_components=2, initial_guesses=init, angle="pa")
    >>> # Choose automatically based on axis handedness (descending/ascending coords):
    >>> ds_auto = fit_multi_gaussian2d(img_da, n_components=2, initial_guesses=init, angle="auto")
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    da_in = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da_in, dims)

    # Move fit dims to the end → [..., y, x]
    da_tr = da_in.transpose(*(d for d in da_in.dims if d not in (dim_y, dim_x)), dim_y, dim_x)
    core = [dim_y, dim_x]

    # 2a) Determine axis signs from coords (or defaults)
    cx = np.asarray(da_tr.coords[dim_x].values) if dim_x in da_tr.coords else None
    cy = np.asarray(da_tr.coords[dim_y].values) if dim_y in da_tr.coords else None
    sx_sign = _axis_sign(cx)
    sy_sign = _axis_sign(cy)
    is_left_handed = (sx_sign * sy_sign) < 0.0

    # Convert initial guesses: (a) optional FWHM→σ, (b) theta PA→math if requested.
    ig = initial_guesses
    if ig is not None and initial_is_fwhm and isinstance(ig, np.ndarray) and ig.shape == (int(n_components), 6):
        # columns: [amp, x0, y0, fwhm_major, fwhm_minor, theta]  → convert to σ for the fitter
        ig = ig.copy()
        ig[:, 3] = _sigma_from_fwhm(ig[:, 3])
        ig[:, 4] = _sigma_from_fwhm(ig[:, 4])
    # dict/list forms: allow fwhm_major/fwhm_minor keys by mapping to sigma_x/sigma_y
    if isinstance(ig, dict) and "components" in ig and isinstance(ig["components"], np.ndarray):
        arr = np.asarray(ig["components"], dtype=float)
        if arr.shape == (int(n_components), 6) and initial_is_fwhm:
            arr = arr.copy()
            arr[:, 3] = _sigma_from_fwhm(arr[:, 3])
            arr[:, 4] = _sigma_from_fwhm(arr[:, 4])
            ig = dict(ig); ig["components"] = arr
    elif isinstance(ig, (list, tuple)) and ig and isinstance(ig[0], dict):
        mapped: List[Dict[str, Any]] = []
        for d in ig:
            dd = dict(d)
            if "fwhm_major" in dd: dd["sigma_x"] = float(_sigma_from_fwhm(dd.pop("fwhm_major")))
            if "fwhm_minor" in dd: dd["sigma_y"] = float(_sigma_from_fwhm(dd.pop("fwhm_minor")))
            mapped.append(dd)
        ig = mapped

    want_pa = (angle == "pa") or (angle == "auto" and is_left_handed)
    init_for_fit = _convert_init_theta(ig, to_math=want_pa, sx=sx_sign, sy=sy_sign, n=int(n_components))

    out_dtypes = (
        [np.float64] * 6      # params per component
        + [np.float64] * 6    # errors per component
        + [np.float64] * 3    # derived per component
        + [np.float64, np.float64]  # offset, offset_err
        + [np.bool_, np.float64]    # success, variance_explained
        + [np.float64, np.float64]  # residual2d, model2d
    )
    # Map FWHM bounds to σ if provided (top-level convenience; supports legacy and new names)
    bnds = bounds
    if bounds is not None:
        conv = _FWHM2SIG
        b2: Dict[str, Any] = {}
        for k, v in bounds.items():
            if k in ("fwhm_major", "fwhm_minor"):
                # map directly to the underlying σ slots (3→sigma_x, 4→sigma_y).
                # For major/minor we follow the internal slot convention used during fitting.
                target = (
                    "sigma_x" if k in ("fwhm_major") else "sigma_y"
                )
                if isinstance(v, (list, tuple)) and v and isinstance(v[0], (list, tuple)):
                    b2[target] = [(float(lo) * conv, float(hi) * conv) for (lo, hi) in v]  # per-component
                else:
                    lo, hi = v  # type: ignore[misc]
                    b2[target] = (float(lo) * conv, float(hi) * conv)
            else:
                b2[k] = v
        bnds = b2

    # -- Build 1-D coordinate arrays for model evaluation --
    y1d, x1d = _extract_1d_coords_for_fit(data, da_tr, coord_type, coords, dim_y, dim_x)
    y1d_da = xr.DataArray(y1d, dims=[dim_y])
    x1d_da = xr.DataArray(x1d, dims=[dim_x])
    # Determine whether evaluation grid is world or pixel.
    # If x/y coords are exactly 0..N-1 we treat as pixel mode; otherwise world.
    world_mode = not (
        np.allclose(x1d, np.arange(x1d.shape[0])) and
        np.allclose(y1d, np.arange(y1d.shape[0]))
    )
    out_core_dims = (
        [["component"]]*6 + [["component"]]*6 + [["component"]]*3
        + [[] , []] + [[] , []]
        + [[dim_y, dim_x], [dim_y, dim_x]]
    )

    results = xr.apply_ufunc(
        _multi_fit_plane_wrapper,
        da_tr,
        y1d_da,
        x1d_da,
        input_core_dims=[core, [dim_y], [dim_x]],
        output_core_dims=out_core_dims,
        vectorize=True,
        dask="parallelized",
        output_dtypes=out_dtypes,
        dask_gufunc_kwargs={"output_sizes": {"component": int(n_components)}},
        kwargs=dict(
            n_components=int(n_components),
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            initial_guesses=init_for_fit,  # NOTE: pre-converted for angle, widths are σ
            bounds=bnds,
            max_nfev=int(max_nfev),
            return_model=bool(return_model),
            return_residual=bool(return_residual),
        ),
    )

    (amp, x0, y0, sx, sy, th,
     amp_e, x0_e, y0_e, sx_e, sy_e, th_e,
     fwhm_maj, fwhm_min, peak,
     offset, offset_e,
     success, varexp,
     residual, model) = results

    # -- Angle conventions ----------------------------------------------------
    # Fitter's internal angle has opposite sign vs the "math" convention.
    # Convert once to math (index basis), then derive both public conventions.
    th_math = -th
    theta_math = th_math
    theta_pa = xr.DataArray(
        _theta_math_to_pa(th_math.values, sx_sign, sy_sign),
        dims=th.dims, coords=th.coords
    )
    # Preserve backward-compat "theta" using requested convention
    th_public = theta_pa if want_pa else theta_math

    # --- Pixel-parameter views (centers + ellipse in pixel coords) ---
    if world_mode:
        nx = x1d.shape[0]; ny = y1d.shape[0]
        x_idx_axis = np.arange(nx, dtype=float)
        y_idx_axis = np.arange(ny, dtype=float)
        # centers: world -> pixel via inverse of coord arrays
        x0_pixel = xr.apply_ufunc(
            np.interp, x0, xr.DataArray(x1d, dims=[dim_x]), xr.DataArray(x_idx_axis, dims=[dim_x]),
            input_core_dims=[["component"], [dim_x], [dim_x]],
            output_core_dims=[["component"]], vectorize=True, dask="parallelized", output_dtypes=[float]
        )
        y0_pixel = xr.apply_ufunc(
            np.interp, y0, xr.DataArray(y1d, dims=[dim_y]), xr.DataArray(y_idx_axis, dims=[dim_y]),
            input_core_dims=[["component"], [dim_y], [dim_y]],
            output_core_dims=[["component"]], vectorize=True, dask="parallelized", output_dtypes=[float]
        )
        # local pixel scales (world units per pixel) via gradient at nearest pixel
        gx = np.gradient(x1d.astype(float))
        gy = np.gradient(y1d.astype(float))
        x0i = xr.apply_ufunc(
            np.rint, x0_pixel,
            input_core_dims=[["component"]], output_core_dims=[["component"]],
            vectorize=True, dask="parallelized", output_dtypes=[float],
        ).clip(0, nx-1).astype(int)
        y0i = xr.apply_ufunc(
            np.rint, y0_pixel,
            input_core_dims=[["component"]], output_core_dims=[["component"]],
            vectorize=True, dask="parallelized", output_dtypes=[float],
        ).clip(0, ny-1).astype(int)
        # elementwise pick with clipping (works with vectorize=True)
        def _pick1d(arr, idx):
            idx = np.asarray(idx).astype(np.int64)
            np.clip(idx, 0, arr.shape[0]-1, out=idx)
            return arr[idx]
        # pick local scale via np.take with clipping
        dx_local = xr.apply_ufunc(
            np.take, xr.DataArray(gx, dims=[dim_x]), x0i,
            input_core_dims=[[dim_x], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True, dask="parallelized", output_dtypes=[float],
        )
        dy_local = xr.apply_ufunc(
            np.take, xr.DataArray(gy, dims=[dim_y]), y0i,
            input_core_dims=[[dim_y], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True, dask="parallelized", output_dtypes=[float],
        )
        # covariance transform Σp = S Σw S (S=diag(1/dx,1/dy)) → principal axes in pixel units
        def _world_to_pixel_cov(sigx, sigy, theta, dx, dy):
            c = np.cos(theta); s = np.sin(theta)
            # Σ_world = R diag(sigx^2, sigy^2) R^T
            a = (c*c)*sigx*sigx + (s*s)*sigy*sigy
            b = (s*c)*(sigx*sigx - sigy*sigy)
            d = (s*s)*sigx*sigx + (c*c)*sigy*sigy
            invdx2 = 1.0/(dx*dx); invdy2 = 1.0/(dy*dy)
            A = a*invdx2; B = b*(1.0/(dx*dy)); D = d*invdy2
            tr = A + D
            det = A*D - B*B
            tmp = np.sqrt(np.maximum(tr*tr/4.0 - det, 0.0))
            lam1 = tr/2.0 + tmp
            lam2 = tr/2.0 - tmp
            theta_p = 0.5*np.arctan2(2*B, A - D)
            lam_max = np.maximum(lam1, lam2); lam_min = np.minimum(lam1, lam2)
            return np.sqrt(np.maximum(lam_max, 0.0)), np.sqrt(np.maximum(lam_min, 0.0)), theta_p

        # Feed math-angle into geometry/covariance computation
        sigma_major_pixel, sigma_minor_pixel, theta_pixel_math = xr.apply_ufunc(
            _world_to_pixel_cov, sx, sy, th_math, dx_local, dy_local,
            input_core_dims=[["component"]]*5,
            output_core_dims=[["component"], ["component"], ["component"]],
            vectorize=True, dask="parallelized", output_dtypes=[float, float, float]
        )
        # Report pixel-space angle in the SAME convention as 'theta'
        theta_pixel = (np.pi/2 - theta_pixel_math) if want_pa else theta_pixel_math
        fwhm_major_pixel = sigma_major_pixel * _SIG2FWHM
        fwhm_minor_pixel = sigma_minor_pixel * _SIG2FWHM
    else:
        # already in pixel space -> alias
        x0_pixel, y0_pixel = x0, y0
        sigma_major_pixel = xr.apply_ufunc(np.maximum, sx, sy, dask="parallelized")
        sigma_minor_pixel = xr.apply_ufunc(np.minimum, sx, sy, dask="parallelized")
        theta_pixel = th_public
        fwhm_major_pixel = xr.apply_ufunc(np.maximum, sx*_SIG2FWHM, sy*_SIG2FWHM, dask="parallelized")
        fwhm_minor_pixel = xr.apply_ufunc(np.minimum, sx*_SIG2FWHM, sy*_SIG2FWHM, dask="parallelized")

    # Convert internal σ → public FWHM along principal axes; also provide major/minor & errs.
    # Use DA math to preserve dims/coords.
    _fx = sx * _SIG2FWHM
    _fy = sy * _SIG2FWHM
    _fxe = sx_e * _SIG2FWHM
    _fye = sy_e * _SIG2FWHM
    fwhm_major = xr.apply_ufunc(np.maximum, _fx, _fy, dask="parallelized")
    fwhm_minor = xr.apply_ufunc(np.minimum, _fx, _fy, dask="parallelized")
    # match errs to whichever axis is major/minor per-component
    _is_fx_major = xr.apply_ufunc(np.greater_equal, _fx, _fy, dask="parallelized")
    fwhm_major_err = xr.where(_is_fx_major, _fxe, _fye)
    fwhm_minor_err = xr.where(_is_fx_major, _fye, _fxe)

    # Also expose pixel-space parameters (mirrors of world/fit-space where applicable)
    sigma_x_pixel = sigma_major_pixel
    sigma_y_pixel = sigma_minor_pixel
    # (theta_pixel is the orientation of the major axis in pixel basis)
    # Provide FWHM in pixel units too:

    ds = xr.Dataset(
        data_vars=dict(
            amplitude=amp,
            x0=x0, y0=y0,
            fwhm_major=fwhm_major, fwhm_minor=fwhm_minor,
            sigma_x=sx, sigma_y=sy,
            theta_math=theta_math,
            theta_pa=theta_pa,
            theta=th_public,
            amplitude_err=amp_e,
            x0_err=x0_e, y0_err=y0_e,
            sigma_x_err=sx_e, sigma_y_err=sy_e,
            fwhm_major_err=fwhm_major_err, fwhm_minor_err=fwhm_minor_err,
            theta_err=th_e,
            peak=peak,
            # pixel-space mirrors
            x0_pixel=x0_pixel,
            y0_pixel=y0_pixel,
            sigma_x_pixel=sigma_x_pixel,
            sigma_y_pixel=sigma_y_pixel,
            theta_pixel=theta_pixel,
            fwhm_major_pixel=fwhm_major_pixel, fwhm_minor_pixel=fwhm_minor_pixel,
            offset=offset, offset_err=offset_e,
            success=success, variance_explained=varexp,
        )
    )
    # Principal-axis sigmas (unsorted) → derive explicit major/minor in σ (Dask-safe)
    _sx_ge_sy = ds["sigma_x"] >= ds["sigma_y"]
    sigma_major = xr.where(_sx_ge_sy, ds["sigma_x"], ds["sigma_y"])
    sigma_minor = xr.where(_sx_ge_sy, ds["sigma_y"], ds["sigma_x"])

    # --- record only axes handedness; publish both angle conventions explicitly ---
    conv = "pa" if want_pa else "math"
    ds.attrs["axes_handedness"] = "left" if is_left_handed else "right"
    if "theta" in ds:
        ds["theta"].attrs["convention"] = conv
    if "theta_err" in ds:
        ds["theta_err"].attrs["convention"] = conv
    # Add explicit angular units to all theta-related outputs
    for _name in ("theta", "theta_err", "theta_math", "theta_pa", "theta_pixel"):
        if _name in ds:
            ds[_name].attrs["units"] = "rad"

    if return_residual:
        ds["residual"] = residual
    if return_model:
        ds["model"] = model

    # world coord exposure adjusted: alias if already in world, otherwise interp from pixels.
    if (dim_x in da_tr.coords) and (dim_y in da_tr.coords):
        cx = np.asarray(da_tr.coords[dim_x].values)
        cy = np.asarray(da_tr.coords[dim_y].values)
        if world_mode:
            # Already fitted in world coords → x0/y0 are world; expose direct aliases.
            ds["x_world"] = ds["x0"]
            ds["y_world"] = ds["y0"]
        else:
            def _prep(coord: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                coord = np.asarray(coord)
                if coord.ndim != 1 or coord.size == 0 or not np.all(np.isfinite(coord)):
                    return None, None
                if coord.size >= 2 and coord[1] < coord[0]:  # descending → reverse for interp
                    idx = np.arange(coord.size - 1, -1, -1, dtype=float)
                    return idx, coord[::-1]
                if not np.all(np.diff(coord) > 0):
                    return None, None
                return np.arange(coord.size, dtype=float), coord

            idx_x, val_x = _prep(cx)
            idx_y, val_y = _prep(cy)
            if idx_x is not None and idx_y is not None:
                def _interp_x(v: np.ndarray) -> np.ndarray:
                    return np.interp(v, idx_x, val_x)

                def _interp_y(v: np.ndarray) -> np.ndarray:
                    return np.interp(v, idx_y, val_y)

                ds["x_world"] = xr.apply_ufunc(
                    _interp_x, ds["x0"],
                    input_core_dims=[["component"]],
                    output_core_dims=[["component"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                )
                ds["y_world"] = xr.apply_ufunc(
                    _interp_y, ds["y0"],
                    input_core_dims=[["component"]],
                    output_core_dims=[["component"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                )
    return ds

def plot_components(
    data,
    result: "xr.Dataset",
    dims: "Optional[Sequence[Union[str, int]]]" = None,
    *,
    indexer: "Optional[Mapping[str, int]]" = None,
    show_residual: bool = True,
    fwhm: bool = False,
    angle: "Optional[str]" = None,   # ← None means: read from result.attrs
):
    """
    Quicklook plot: data (and optional residual) with fitted Gaussian components overlaid as ellipses.

    Parameters
    ----------
    data: numpy.ndarray | dask.array.Array | xarray.DataArray
      Image or cube. If not a DataArray, it is wrapped with dims ('y','x') and numeric coords.

    result: xarray.Dataset
      Output from `fit_multi_gaussian2d`. Must contain at least
      {'x0','y0','sigma_x','sigma_y','theta'}. If present, 'residual' and 'model'
      will be used for quicklook panels.

    dims: Sequence[str | int] | None
      Two dims (names or indices) that define the image plane (x, y). If omitted:
      uses ('x','y') if present; else for 2-D uses (last, second-last).

    indexer: Mapping[str, int] | None
      For N-D inputs, which indices to select for leading dims (not including x/y/component).
      If None and data is N-D, defaults to {d: 0 for d in leading dims}.

    show_residual: bool
      If True and residuals are available in `result` (or model to compute them),
      show a side-by-side residual panel.

    fwhm: bool
      If True, draw ellipses at FWHM size (2.3548 * sigma) instead of 1σ.

    angle: {"math","pa","auto"}
      Convention **of `result["theta"]`**:
        • "math": theta is math angle (+x → +y, CCW).
        • "pa": theta is position angle (+y → +x).
        • "auto": infer from axis handedness (left-handed → PA, else math).
      Matplotlib expects a **math** angle; if the dataset theta is PA, it is converted.

    Returns
    -------
    matplotlib.figure.Figure
      The created figure.

    Notes
    -----
    • Ellipses show the fitted components:
        - center: (x0, y0)
        - orientation: theta (converted to math if needed)
        - size: 1σ radii (or FWHM if `fwhm=True`)
      Width/height passed to Matplotlib’s Ellipse are **diameters**.
    """
    # -- imports guarded to avoid hard dependency at import time --
    try:  # crucial: plotting is optional at runtime/env
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.patches import Ellipse  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Matplotlib is required for plot_components.") from exc

    # -- normalize input and resolve dims --
    da = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da, dims)
    # transpose so plane dims are last
    da_tr = da.transpose(*(d for d in da.dims if d not in (dim_y, dim_x)), dim_y, dim_x)

    # -- select a single plane for plotting --
    if da_tr.ndim > 2:
        if indexer is None:
            indexer = {d: 0 for d in da_tr.dims[:-2]}
        data2d = da_tr.isel(**indexer)
        res_plane = result
        # align result along same leading dims when present
        for d, i in indexer.items():
            if d in res_plane.dims and d not in ("component", dim_y, dim_x):
                res_plane = res_plane.isel({d: i})
    else:
        data2d = da_tr
        res_plane = result

    # -- helpers to fetch variables with informative errors --
    def _get(name: str) -> "xr.DataArray":
        if name not in res_plane:
            raise KeyError(f"result missing '{name}'")
        return res_plane[name]

    # Ensure arrays even if a single component (scalar after isel)
    x0 = np.atleast_1d(_get("x0").values)
    y0 = np.atleast_1d(_get("y0").values)
    sx = np.atleast_1d(_get("sigma_x").values)
    sy = np.atleast_1d(_get("sigma_y").values)
    th = np.atleast_1d(_get("theta").values)  # convention specified by `angle` arg

    # -- axis handedness from coords (for PA<->math conversion) --
    cx = np.asarray(data2d.coords[dim_x].values) if dim_x in data2d.coords else None
    cy = np.asarray(data2d.coords[dim_y].values) if dim_y in data2d.coords else None
    sx_sign = _axis_sign(cx)
    sy_sign = _axis_sign(cy)
    left_handed = (sx_sign * sy_sign) < 0.0

    # -- interpret dataset theta per `angle`, then convert to Matplotlib math angle if needed --
    # Decide convention of result["theta"]:
    theta_conv = (str(angle).lower()
                  if angle is not None
                  else str(result.attrs.get("theta_convention", "math")).lower())
    ds_is_pa = (theta_conv == "pa") or (theta_conv == "auto" and left_handed)

    theta_plot = _theta_pa_to_math(th, sx_sign, sy_sign) if ds_is_pa else np.asarray(th, dtype=float)
    theta_deg = np.atleast_1d(np.degrees(theta_plot))

    # -- size scale: 1σ or FWHM --
    scale = 2.3548200450309493 if fwhm else 1.0

    width = np.atleast_1d(2.0 * scale * sx)
    height = np.atleast_1d(2.0 * scale * sy)

    # -- choose whether we can show residuals --
    show_resid_panel = bool(show_residual and ("residual" in res_plane or "model" in res_plane))
    fig, axes = (plt.subplots(1, 2, figsize=(10, 4.5)) if show_resid_panel
                 else plt.subplots(1, 1, figsize=(5.5, 5.0)))
    ax_data = axes[0] if show_resid_panel else axes

    im0 = ax_data.imshow(np.asarray(data2d), origin="lower", aspect="equal")
    ax_data.set_title("Data with fitted components")
    ax_data.set_xlabel(dim_x)
    ax_data.set_ylabel(dim_y)

    # overlay ellipses
    ncomp = int(np.size(x0))
    for i in range(ncomp):
        e = Ellipse(
            (float(x0[i]), float(y0[i])),
            float(width[i]),
            float(height[i]),
            angle=float(theta_deg[i]),
            fill=False,
            linewidth=1.5,
            edgecolor="k",
        )
        ax_data.add_patch(e)
        # centroid marker
        ax_data.plot(float(x0[i]), float(y0[i]), marker="+", ms=7, mec="yellow", mfc="none", mew=1.5)

    plt.colorbar(im0, ax=ax_data, fraction=0.046, pad=0.04)

    # residual panel if available/asked
    if show_resid_panel:
        ax_res = axes[1]
        if "residual" in res_plane:
            res2d = res_plane["residual"]
            # slice residual like data if it still carries extra dims
            for d, i in (indexer or {}).items():
                if d in res2d.dims and d not in (dim_y, dim_x):
                    res2d = res2d.isel({d: i})
            resid_img = np.asarray(res2d)
        elif "model" in res_plane:
            model2d = res_plane["model"]
            for d, i in (indexer or {}).items():
                if d in model2d.dims and d not in (dim_y, dim_x):
                    model2d = model2d.isel({d: i})
            resid_img = np.asarray(data2d) - np.asarray(model2d)
        else:  # pragma: no cover (guarded by show_resid_panel condition)
            resid_img = np.zeros_like(np.asarray(data2d))

        im1 = ax_res.imshow(resid_img, origin="lower", aspect="equal")
        ax_res.set_title("Residual (data − model)")
        ax_res.set_xlabel(dim_x)
        ax_res.set_ylabel(dim_y)
        plt.colorbar(im1, ax=ax_res, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
    return fig

