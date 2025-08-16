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


def _fwhm_from_sigma(sigma: float) -> float:
    """Convert sigma to FWHM."""
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * sigma


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


def _default_bounds_multi(shape: Tuple[int, int], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Default bounds vectors for multi-Gaussian parameters (packed order).
    """
    ny, nx = shape
    # offset
    lb = [ -np.inf ]
    ub = [  np.inf ]
    # components
    for _ in range(n):
        lb.extend([0.0, -1.0, -1.0, 0.5, 0.5, -np.pi/2])          # amp, x, y, sx, sy, th
        ub.extend([np.inf, nx + 1.0, ny + 1.0, max(nx, 2.0), max(ny, 2.0),  np.pi/2])
    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


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
      - list of dicts (len n): keys amp, x0, y0, sigma_x, sigma_y, theta; offset from median
      - dict with keys:
          'offset': float (optional),
          'components': array-like (n, 6) or list of dicts as above.

    Returns a packed parameter vector as in _pack_params.
    """
    ny, nx = z2d.shape
    # threshold mask just for robust stats
    mask = np.ones_like(z2d, dtype=bool)
    if min_threshold is not None:
        mask &= z2d >= min_threshold
    if max_threshold is not None:
        mask &= z2d <= max_threshold
    z_masked = np.where(mask, z2d, np.nan)
    med = float(np.nanmedian(z_masked))

    if init is None:
        seeds = _greedy_peak_seeds(z_masked, n=n, excl_radius=max(3, max(ny, nx)//50))
        amps = np.array([max(v - med, 1e-3) for (_,_,v) in seeds], dtype=float)
        x0 = np.array([float(x) for (_,x,_) in seeds], dtype=float)
        y0 = np.array([float(y) for (y,_,_) in seeds], dtype=float)
        sx = np.full(n, max(nx, ny) / 10.0, dtype=float)
        sy = np.full(n, max(nx, ny) / 10.0, dtype=float)
        th = np.zeros(n, dtype=float)
        return _pack_params(med, amps, x0, y0, sx, sy, th)

    # Structured dict form
    if isinstance(init, dict) and ("components" in init or "offset" in init):
        offset = float(init.get("offset", med))
        comps = init.get("components", None)
        if comps is None:
            raise ValueError("init dict must include 'components' with shape (n,6) or list of dicts.")
        init = comps
        # fallthrough to parse comps and finally pack with provided offset

        if isinstance(init, np.ndarray):
            arr = np.asarray(init, dtype=float)
            if arr.shape != (n, 6):
                raise ValueError(f"init['components'] must have shape (n,6); got {arr.shape}")
            amps, x0, y0, sx, sy, th = [arr[:,k].astype(float) for k in range(6)]
            return _pack_params(offset, amps, x0, y0, sx, sy, th)

        if isinstance(init, (list, tuple)):
            if len(init) != n:
                raise ValueError(f"init['components'] must have length n={n}")
            amps = np.empty(n); x0 = np.empty(n); y0 = np.empty(n); sx = np.empty(n); sy = np.empty(n); th = np.empty(n)
            for i, comp in enumerate(init):
                amps[i] = float(comp["amp"] if "amp" in comp else comp["amplitude"])
                x0[i] = float(comp["x0"])
                y0[i] = float(comp["y0"])
                sx[i] = float(comp.get("sigma_x", comp.get("sx")))
                sy[i] = float(comp.get("sigma_y", comp.get("sy")))
                th[i] = float(comp.get("theta", 0.0))
            return _pack_params(offset, amps, x0, y0, sx, sy, th)

    # Array/list form
    arr = np.asarray(init, dtype=float)
    if arr.shape != (n, 6):
        raise ValueError(f"initial_guesses must have shape (n,6); got {arr.shape}")
    amps, x0, y0, sx, sy, th = [arr[:,k].astype(float) for k in range(6)]
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
            return
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
        "theta": (5, "theta"),
    }
    for key, val in user_bounds.items():
        if key not in mapping:
            continue
        idx_in_comp, canon = mapping[key]
        if canon == "offset":
            _set_range("offset", None, tuple(val))  # type: ignore[arg-type]
            continue
        if isinstance(val, (list, tuple)) and len(val) == 2 and not isinstance(val[0], (list, tuple)):
            _set_range(canon, idx_in_comp, (val[0], val[1]))  # same for all comps
        else:
            if len(val) != n:
                raise ValueError(f"bounds[{key!r}] length must be n={n}")
            for i, rng in enumerate(val):  # type: ignore[assignment]
                _set_range(canon, idx_in_comp, tuple(rng), comp_idx=i)  # type: ignore[arg-type]
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
        raise ValueError("Internal: _fit_multi_plane_numpy expects a 2-D array.")
    ny, nx = z2d.shape
    Y, X = np.mgrid[0:ny, 0:nx]

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
    lb0, ub0 = _default_bounds_multi((ny, nx), n_components)
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
    )

    ny, nx = z2d.shape
    Y, X = np.mgrid[0:ny, 0:nx]

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
        varexp = np.nan

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
    max_nfev: int = 20000,
    return_model: bool = False,
    return_residual: bool = True,
) -> xr.Dataset:
    """
    Fit a sum of N rotated 2-D Gaussians (with a shared constant offset) to each plane.

    The model for one plane is:
        model(x, y) = offset + Σ_{i=1..N} amp_i · G_i(x, y; x0_i, y0_i, sigma_x_i, sigma_y_i, theta_i)

    Supports NumPy, Dask, and Xarray inputs, and vectorizes across all non-fit dims.
    When input is Dask-backed, computation is parallelized via ``xarray.apply_ufunc(dask="parallelized")``.

    Parameters
    ----------
    data : numpy.ndarray | dask.array.Array | xarray.DataArray
        Input array. If not a DataArray, it is wrapped with generated dims/coords.
    n_components : int
        Number of Gaussian components (N >= 1).
    dims : sequence of two (str | int), optional
        Names or indices of the two fit axes (x, y). If omitted:
          • If the DataArray has dims named exactly ``("x","y")``, those are used.
          • Else if the input is 2-D, last → x, second-last → y.
          • Else you must specify dims explicitly.
    min_threshold, max_threshold : float or None, optional
        Inclusive thresholds to mask pixels during fitting. Use None to disable a side.
    initial_guesses : array-like | list[dict] | dict, optional
        Initial parameter guesses to help the optimizer.
        Forms accepted (offset default = median(data)):
          • array with shape (N, 6): columns [amp, x0, y0, sigma_x, sigma_y, theta]
          • list of N dicts with keys {"amp"/"amplitude","x0","y0","sigma_x","sigma_y","theta"}
          • dict: {"offset": float (optional), "components": (N,6) array OR list[dict] as above}
        The same guesses are used for all planes (per-plane seeding can be added later).
    bounds : dict, optional
        Bounds to constrain parameters. Keys may include:
          {"offset","amp"/"amplitude","x0","y0","sigma_x","sigma_y","theta"}.
        Each value is either a single (low, high) tuple applied to all components,
        or a length-N sequence of (low, high) tuples for per-component bounds.
    max_nfev : int, default 20000
        Max function evaluations for the optimizer.
    return_model : bool, default False
        If True, include variable ``model`` (the fitted plane).
    return_residual : bool, default True
        If True, include variable ``residual`` (= data - model).

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
    >>> import xarray as xr
    >>> planes = [img_da + 0.01*np.random.randn(*img_da.shape) for _ in range(3)]
    >>> cube = xr.concat(planes, dim="time")  # dims ('time','y','x')
    >>> ds3 = fit_multi_gaussian2d(cube, n_components=2, initial_guesses=init)
    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")

    da_in = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da_in, dims)  # your existing helper (now prefers ('x','y') when available)

    # Move fit dims to the end → [..., y, x]
    da_tr = da_in.transpose(*(d for d in da_in.dims if d not in (dim_y, dim_x)), dim_y, dim_x)
    core_dims = [dim_y, dim_x]

    # Output signatures for apply_ufunc (per _multi_fit_plane_wrapper tuple)
    out_dtypes = (
        [np.float64] * 6   # amp, x0, y0, sx, sy, th
        + [np.float64] * 6 # amp_e, x0_e, y0_e, sx_e, sy_e, th_e
        + [np.float64] * 3 # fwhm_major, fwhm_minor, peak
        + [np.float64, np.float64]  # offset, offset_e
        + [np.bool_, np.float64]    # success, variance_explained
        + [np.float64, np.float64]  # residual2d, model2d
    )
    out_core_dims = (
        [["component"]]*6 + [["component"]]*6 + [["component"]]*3
        + [[] , []] + [[] , []]
        + [[dim_y, dim_x], [dim_y, dim_x]]
    )

    results = xr.apply_ufunc(
        _multi_fit_plane_wrapper,
        da_tr,
        input_core_dims=[core_dims],
        output_core_dims=out_core_dims,
        vectorize=True,
        dask="parallelized",
        output_dtypes=out_dtypes,
        dask_gufunc_kwargs={"output_sizes": {"component": int(n_components)}},
        kwargs=dict(
            n_components=int(n_components),
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            initial_guesses=initial_guesses,
            bounds=bounds,
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

    ds = xr.Dataset(
        data_vars=dict(
            amplitude=amp,
            x0=x0, y0=y0,
            sigma_x=sx, sigma_y=sy,
            theta=th,
            amplitude_err=amp_e,
            x0_err=x0_e, y0_err=y0_e,
            sigma_x_err=sx_e, sigma_y_err=sy_e,
            theta_err=th_e,
            fwhm_major=fwhm_maj, fwhm_minor=fwhm_min,
            peak=peak,
            offset=offset, offset_err=offset_e,
            success=success, variance_explained=varexp,
        )
    )

    if return_residual:
        ds["residual"] = residual
    if return_model:
        ds["model"] = model

    # Optional world coords only when both axes have 1-D numeric coords (supports descending)
    if (dim_x in da_tr.coords) and (dim_y in da_tr.coords):
        cx = np.asarray(da_tr.coords[dim_x].values)
        cy = np.asarray(da_tr.coords[dim_y].values)

        def _prep(coord: np.ndarray):
            coord = np.asarray(coord)
            if coord.ndim != 1 or coord.size == 0 or not np.all(np.isfinite(coord)):
                return None, None
            if coord.size >= 2 and coord[1] < coord[0]:
                idx = np.arange(coord.size - 1, -1, -1, dtype=float)
                return idx, coord[::-1]
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
    data: ArrayOrDA,
    result: xr.Dataset,
    dims: Optional[Sequence[Union[str, int]]] = None,
    indexer: Optional[dict] = None,
    *,
    show_residual: bool = False,
) -> None:
    """
    Quick visualization: overlays fitted centroids and FWHM ellipses on the data.

    Parameters
    ----------
    data : ndarray | dask.array.Array | xarray.DataArray
        The same array passed to the fitter.
    result : xarray.Dataset
        Output from fit_multi_gaussian2d (or single-Gaussian). Expects variables
        x0, y0, sigma_x, sigma_y, theta (optionally a 'component' dimension)
        and optional 'residual'.
    dims : sequence of two (str|int), optional
        Fit-plane dims; for 2-D you can omit.
    indexer : dict, optional
        For >2-D inputs, selects a single plane (e.g. {'time': 0}).
    show_residual : bool, default False
        If True, shows a second panel with residuals.

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    da = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da, dims)
    da_tr = da.transpose(*(d for d in da.dims if d not in (dim_y, dim_x)), dim_y, dim_x)

    if da_tr.ndim > 2:
        if indexer is None:
            indexer = {d: 0 for d in da_tr.dims[:-2]}
        data2d = da_tr.isel(**indexer)
        res_plane = result
        for d, i in indexer.items():
            if d in res_plane.dims and d not in ("component", dim_y, dim_x):
                res_plane = res_plane.isel({d: i})
    else:
        data2d = da_tr
        res_plane = result

    def _get(name):
        if name not in res_plane:
            raise KeyError(f"result missing '{name}'")
        return res_plane[name]

    x0 = _get("x0"); y0 = _get("y0")
    sx = _get("sigma_x"); sy = _get("sigma_y")
    th = _get("theta")

    if "component" not in x0.dims:
        x0 = x0.expand_dims({"component": [0]})
        y0 = y0.expand_dims({"component": [0]})
        sx = sx.expand_dims({"component": [0]})
        sy = sy.expand_dims({"component": [0]})
        th = th.expand_dims({"component": [0]})

    if show_residual and "residual" in result:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        ax1 = None

    ax0.imshow(np.asarray(data2d), origin="lower")
    ax0.set_title("Data with fitted components")
    ax0.set_xlabel(dim_x); ax0.set_ylabel(dim_y)

    k = 2.0 * np.sqrt(2.0 * np.log(2.0))  # FWHM factor
    for i in range(x0.sizes["component"]):
        xi = float(x0.isel(component=i)); yi = float(y0.isel(component=i))
        sxi = float(sx.isel(component=i)); syi = float(sy.isel(component=i))
        thi = float(th.isel(component=i))
        ax0.plot([xi], [yi], marker="+", linestyle="None")
        ax0.add_patch(Ellipse((xi, yi), width=k*sxi, height=k*syi, angle=np.degrees(thi), fill=False, linewidth=1.5))

    if ax1 is not None:
        res2d = result["residual"]
        if da_tr.ndim > 2:
            res2d = res2d.transpose(*da_tr.dims).isel(**indexer)
        ax1.imshow(np.asarray(res2d), origin="lower")
        ax1.set_title("Residual")
        ax1.set_xlabel(dim_x); ax1.set_ylabel(dim_y)
    plt.show()
