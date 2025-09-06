# file: src/astroviper/fitting/multi_gaussian2d_fit.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, Any, Dict, List
import numpy as np
import xarray as xr
import dask.array as da
from scipy.optimize import curve_fit

import warnings

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


def _unpack_params(
    params: np.ndarray, n: int
) -> Tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
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


def _greedy_peak_seeds(
    z: np.ndarray, n: int, excl_radius: int = 5
) -> List[Tuple[int, int, float]]:
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
    lb = [-np.inf]
    ub = [np.inf]
    # components
    xlo, xhi = (-1.0, nx + 1.0) if x_rng is None else (float(x_rng[0]), float(x_rng[1]))
    ylo, yhi = (-1.0, ny + 1.0) if y_rng is None else (float(y_rng[0]), float(y_rng[1]))
    sx_max = max((xhi - xlo), 2.0)
    sy_max = max((yhi - ylo), 2.0)

    for _ in range(n):
        # amp,   x0,   y0,     sx,     sy,   theta
        lb.extend([0.0, xlo, ylo, 1e-3, 1e-3, -np.pi / 2])
        ub.extend([np.inf, xhi, yhi, sx_max, sy_max, np.pi / 2])

    return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)


def _extract_params_from_comp_dicts(comp_list, n):
    """Parse a list of component dicts into arrays (amp, x0, y0, sx, sy, th).
    Accepts keys: amp|amplitude, x0, y0, sigma_x|sx|fwhm_major, sigma_y|sy|fwhm_minor, theta.
    Converts FWHM inputs to σ using _FWHM2SIG.
    Raises KeyError if required keys are missing.
    """
    amps = np.empty(n)
    x0 = np.empty(n)
    y0 = np.empty(n)
    sx = np.empty(n)
    sy = np.empty(n)
    th = np.empty(n)
    for i, comp in enumerate(comp_list):
        amps[i] = float(comp["amp"] if "amp" in comp else comp["amplitude"])
        x0[i] = float(comp["x0"])
        y0[i] = float(comp["y0"])
        sx_val = comp.get("sigma_x", comp.get("sx"))
        sy_val = comp.get("sigma_y", comp.get("sy"))
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
        sx[i] = float(sx_val)
        sy[i] = float(sy_val)
        th[i] = float(comp.get("theta", 0.0))
    return amps, x0, y0, sx, sy, th


def _process_list_of_dicts(
    init: Sequence[Dict[str, Number]],
    n: int,
    offset: float,
) -> np.ndarray:
    """Helper to parse list-of-dicts initial_guesses with provided offset.
    placing in own fucntion to try to get test coverage to recoognize it.
    """
    if len(init) != n:
        raise ValueError(f"init['components'] must have length n={n}")
    amps, x0, y0, sx, sy, th = _extract_params_from_comp_dicts(init, n)
    return _pack_params(offset, amps, x0, y0, sx, sy, th)


def _is_pixel_index_axes(x1d, y1d):
    # Treat pure pixel-index axes (0..N-1) as pixel mode even if arrays are present
    # putting in function to try to get test coverage to recoognize it.
    _x = np.asarray(x1d, dtype=float)
    _y = np.asarray(y1d, dtype=float)
    is_pixel_index_axes = np.allclose(_x, np.arange(_x.size)) and np.allclose(
        _y, np.arange(_y.size)
    )
    return is_pixel_index_axes


def _normalize_initial_guesses(
    z2d: np.ndarray,
    n: int,
    init: Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]],
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    *,
    x1d: Optional[np.ndarray] = None,
    y1d: Optional[np.ndarray] = None,
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

    # None → auto seeds (in same coordinate system as the fit)
    mask = np.ones_like(z2d, dtype=bool)
    if min_threshold is not None:
        mask &= z2d >= float(min_threshold)
    if max_threshold is not None:
        mask &= z2d <= float(max_threshold)
    z_masked = np.where(mask, z2d, np.nan)
    med = float(np.nanmedian(z_masked))

    # None → auto seeds (in same coordinate system as the fit)
    if init is None:
        seeds = _greedy_peak_seeds(z_masked, n=n, excl_radius=max(3, max(ny, nx) // 50))
        amps = np.array([max(v - med, 1e-3) for (_, _, v) in seeds], dtype=float)
        xi = np.array([int(x) for (_, x, _) in seeds], dtype=int)
        yi = np.array([int(y) for (y, _, _) in seeds], dtype=int)
        # Use WORLD seeding when axes are *not* pure pixel indices
        if x1d is not None and y1d is not None and not _is_pixel_index_axes(x1d, y1d):
            # WORLD: map centers and pick widths in world units
            x1d = np.asarray(x1d, dtype=float)
            y1d = np.asarray(y1d, dtype=float)
            x0 = x1d[xi].astype(float)
            y0 = y1d[yi].astype(float)
            sx = np.full(n, (np.nanmax(x1d) - np.nanmin(x1d)) / 10.0, dtype=float)
            sy = np.full(n, (np.nanmax(y1d) - np.nanmin(y1d)) / 10.0, dtype=float)
        else:
            # PIXEL: centers and widths in pixel units
            x0 = xi.astype(float)
            y0 = yi.astype(float)
            sx = np.full(n, max(nx, ny) / 10.0, dtype=float)
            sy = np.full(n, max(nx, ny) / 10.0, dtype=float)
        th = np.zeros(n, dtype=float)
        return _pack_params(med, amps, x0, y0, sx, sy, th)

    # Allow a single top-level dict when n == 1 (wrap to list-of-dicts)
    from collections.abc import Mapping

    if isinstance(init, Mapping) and (
        "components" not in init and "offset" not in init
    ):
        if n != 1:
            raise ValueError(
                "Single-dict initial_guesses is only valid when n_components == 1."
            )
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
            raise ValueError(
                "init dict must include 'components' with shape (n,6) or list of dicts."
            )
        init = comps  # fallthrough to parse components and pack with provided offset

        if isinstance(init, np.ndarray):
            arr = np.asarray(init, dtype=float)
            if arr.shape != (n, 6):
                raise ValueError(
                    f"init['components'] must have shape (n,6); got {arr.shape}"
                )
            amps, x0, y0, sx, sy, th = [arr[:, k].astype(float) for k in range(6)]
            return _pack_params(offset, amps, x0, y0, sx, sy, th)

        if (
            isinstance(init, (list, tuple))
            and len(init) > 0
            and isinstance(init[0], dict)
        ):
            return _process_list_of_dicts(init, n, offset)
    # Array/list form (numpy array or list-of-lists)
    arr = np.asarray(init, dtype=float)
    if arr.shape != (n, 6):
        raise ValueError(f"initial_guesses must have shape (n,6); got {arr.shape}")
    amps, x0, y0, sx, sy, th = [arr[:, k].astype(float) for k in range(6)]
    return _pack_params(med, amps, x0, y0, sx, sy, th)


def _merge_bounds_multi(
    base_lb: np.ndarray,
    base_ub: np.ndarray,
    user_bounds: Optional[
        Dict[str, Union[Tuple[float, float], Sequence[Tuple[float, float]]]]
    ],
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
    lb = base_lb.copy()
    ub = base_ub.copy()

    def _set_range(
        name: str,
        idx_in_comp: Optional[int],
        rng: Tuple[float, float],
        comp_idx: Optional[int] = None,
    ):
        lo, hi = float(rng[0]), float(rng[1])
        if name == "offset":
            lb[0] = lo
            ub[0] = hi
            return
        if idx_in_comp is None:
            # this should never be reached, so can't be covered
            # but leaving it in for defensive programming
            return  # pragma: no cover
        if comp_idx is None:
            for i in range(n):
                j0 = 1 + i * 6 + idx_in_comp
                lb[j0] = lo
                ub[j0] = hi
        else:
            j0 = 1 + comp_idx * 6 + idx_in_comp
            lb[j0] = lo
            ub[j0] = hi

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
        if (
            isinstance(val, (list, tuple))
            and len(val) == 2
            and not isinstance(val[0], (list, tuple))
        ):
            _set_range(
                canon, idx_in_comp, _to_sigma_rng(canon, (val[0], val[1]))
            )  # same for all comps
        else:
            if len(val) != n:
                raise ValueError(f"bounds[{key!r}] length must be n={n}")
            for i, rng in enumerate(val):  # type: ignore[assignment]
                _set_range(canon, idx_in_comp, _to_sigma_rng(canon, tuple(rng)), comp_idx=i)  # type: ignore[arg-type]
    return lb, ub


def _count_true_at_least(mask: np.ndarray, need: int, *, chunk: int = 262_144) -> int:
    """Return the number of True values in *mask*, but stop early once *need* is reached.

    Notes
    -----
    - Early exit avoids scanning the whole array when enough pixels are available.
    - Uses vectorized per-chunk reductions in C (``sum`` on ``bool``) for speed.
    """
    need = int(need)
    if need <= 0:
        return 0
    flat = np.asarray(mask, dtype=bool).ravel(order="K")
    total = 0
    for i in range(0, flat.size, chunk):
        block = flat[i : i + chunk]
        total += int(block.sum(dtype=np.intp))  # bool sum is a fast popcount in C
        if total >= need:
            return total
    return total


# ----------------------- Single-plane multi fit -----------------------


def _fit_multi_plane_numpy(
    z2d: np.ndarray,
    n_components: int,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    initial_guesses: Optional[
        Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]
    ],
    bounds: Optional[
        Dict[str, Union[Tuple[float, float], Sequence[Tuple[float, float]]]]
    ],
    max_nfev: int,
    *,
    x1d: Optional[np.ndarray] = None,
    y1d: Optional[np.ndarray] = None,
    mask2d: Optional[np.ndarray] = None,
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
        Mask used during fitting (user mask ∧ thresholds).
    """
    if z2d.ndim != 2:
        # cannot cover using public API, defensive coding
        raise ValueError(
            "Internal: _fit_multi_plane_numpy expects a 2-D array."
        )  # pragma: no cover
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

    # user mask + thresholds
    if mask2d is None:
        mask = np.ones_like(z2d, dtype=bool)
    else:
        mask = np.asarray(mask2d, dtype=bool).copy()
        if mask.shape != z2d.shape:
            raise ValueError("mask shape must match the 2-D plane being fit.")
    if min_threshold is not None:
        mask &= z2d >= min_threshold
    if max_threshold is not None:
        mask &= z2d <= max_threshold
    # Early validation with short-circuit counting
    need = 1 + 6 * n_components  # offset + 6 per component
    mask_count = _count_true_at_least(mask, max(1, need))
    if mask_count == 0:
        raise ValueError(
            "Thresholding removed all pixels (empty mask); adjust min/max thresholds."
        )
    if mask_count < need:
        # Not enough points relative to params
        raise ValueError(
            "Thresholding resulted in not enough pixels being left to fit all parameters"
        )

    # Seed parameters — honor user/threshold mask during seeding
    # (mask already combines user mask & threshold cuts above)
    z2d_for_seed = np.where(mask, z2d, np.nan)
    p0 = _normalize_initial_guesses(
        z2d_for_seed,
        n_components,
        initial_guesses,
        min_threshold,
        max_threshold,
        x1d=x1d,
        y1d=y1d,
    )

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
        # Robust error extraction: never raise on bad/odd pcov
        if pcov is None:
            # fit successful but errs can't be determined
            perr = np.full(popt.size, np.nan, dtype=float)
        else:
            pcov_arr = np.asarray(pcov, dtype=float)
            ok_shape = (
                pcov_arr.ndim == 2
                and pcov_arr.shape[0] == pcov_arr.shape[1] == popt.size
            )
            if (not ok_shape) or (not np.isfinite(pcov_arr).all()):
                """
                not ok_shape → pcov isn’t a square matrix of shape (popt.size,
                  popt.size) (wrong dims or size mismatch).
                not np.isfinite(pcov_arr).all() → pcov contains NaN or ±inf anywhere.
                  # In this branch we do not treat the fit as failed; we just can’t
                  # trust the uncertainties, so we set perr = NaN (float) and keep
                  # success=True. Typical causes: ill-conditioned Jacobian, parameter
                  # degeneracy, or a mocked pcov in tests.
                """
                perr = np.full(popt.size, np.nan, dtype=float)
            else:
                # Some fits yield small negative variances; avoid exceptions if seterr=raise
                diag = np.diag(pcov_arr).astype(float, copy=False)
                perr = np.where(diag >= 0, np.sqrt(diag), np.nan)
    except Exception as e:
        # propagate optimizer failure with original message
        raise type(e)(f"curve_fit failed: {e}") from e
    return popt, perr, mask


# ----------------------- Vectorized wrapper (for xarray) -----------------------


def _multi_fit_plane_wrapper(
    z2d: np.ndarray,
    mask2d: np.ndarray,
    y1d: np.ndarray,
    x1d: np.ndarray,
    n_components: int,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    initial_guesses: Optional[
        Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]
    ],
    bounds: Optional[
        Dict[str, Union[Tuple[float, float], Sequence[Tuple[float, float]]]]
    ],
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
        x1d=x1d,
        y1d=y1d,
        mask2d=mask2d,
    )

    ny, nx = z2d.shape
    # Y/X constructed identically in _fit_multi_plane_numpy; keep for model/residual paths
    if y1d.shape[0] != ny or x1d.shape[0] != nx:
        raise ValueError(
            "Length of y1d/x1d must match z2d shape for world/pixel grids."
        )  # pragma: no cover
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
        peak_err = zeros
        offset = np.nan
        offset_e = np.nan
        model2d = np.full_like(z2d, np.nan, dtype=float)
        resid2d = np.full_like(z2d, np.nan, dtype=float)
        varexp = np.nan
        return (
            amp,
            x0,
            y0,
            sx,
            sy,
            th,
            amp_e,
            x0_e,
            y0_e,
            sx_e,
            sy_e,
            th_e,
            fwhm_major,
            fwhm_minor,
            peak,
            peak_err,
            offset,
            offset_e,
            bool(False),
            varexp,
            resid2d,
            model2d,
        )

    # Unpack best-fit and uncertainties
    n = int(n_components)
    offset, amp, x0, y0, sx, sy, th = _unpack_params(popt, n)
    _, amp_e, x0_e, y0_e, sx_e, sy_e, th_e = _unpack_params(perr, n)
    # offset uncertainty is scalar (first element of perr); compute it *before* using in peak_err
    offset_e = float(perr[0])
    # Derived component metrics
    fwhm_major = _fwhm_from_sigma(np.maximum(sx, sy))
    fwhm_minor = _fwhm_from_sigma(np.minimum(sx, sy))
    peak = amp + offset
    # per-component uncertainty on peak = sqrt(amp_err^2 + offset_err^2)
    peak_err = np.hypot(amp_e, offset_e)
    # Build model & residual
    if return_model or return_residual:
        model2d_full = _multi_gaussian2d_sum(X, Y, popt, n)
        model2d = (
            model2d_full if return_model else np.full_like(z2d, np.nan, dtype=float)
        )
        resid2d = (
            (z2d.astype(float) - model2d_full)
            if return_residual
            else np.full_like(z2d, np.nan, dtype=float)
        )
    else:
        model2d = np.full_like(z2d, np.nan, dtype=float)
        resid2d = np.full_like(z2d, np.nan, dtype=float)

    # Variance explained on masked pixels
    if mask.any():
        z_masked = z2d.astype(float)[mask]
        _model_full = (
            model2d_full
            if "model2d_full" in locals()
            else _multi_gaussian2d_sum(X, Y, popt, n)
        )
        r_masked = (z2d.astype(float) - _model_full)[mask]
        tot = np.nanvar(z_masked)
        res = np.nanvar(r_masked)
        varexp = (1.0 - res / tot) if (np.isfinite(tot) and tot > 0) else np.nan
    else:
        # not coverable from public API call, defensive coding
        varexp = np.nan  # pragma: no cover

    return (
        amp,
        x0,
        y0,
        sx,
        sy,
        th,
        amp_e,
        x0_e,
        y0_e,
        sx_e,
        sy_e,
        th_e,
        fwhm_major,
        fwhm_minor,
        peak,
        peak_err,
        float(offset),
        offset_e,
        bool(True),
        float(varexp),
        resid2d,
        model2d,
    )


def _ensure_dataarray(data: ArrayOrDA) -> xr.DataArray:
    """Normalize input to xarray.DataArray with generated dims/coords if needed."""
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, (np.ndarray, da.Array)):
        dims = [f"dim_{i}" for i in range(data.ndim)]
        coords = {d: np.arange(s, dtype=float) for d, s in zip(dims, data.shape)}
        return xr.DataArray(data, dims=dims, coords=coords, name="data")
    raise TypeError(
        "Unsupported input type; use numpy.ndarray, dask.array.Array, or xarray.DataArray."
    )


def _resolve_dims(
    da: xr.DataArray, dims: Optional[Sequence[Union[str, int]]]
) -> Tuple[str, str]:
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
        raise ValueError(
            "For arrays with ndim != 2, specify two dims (names or indices)."
        )

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


def _select_mask(da_tr: xr.DataArray, spec: str):
    """
    Thin wrapper so tests can monkeypatch this symbol.
    Delegates to selection.select_mask in the same package.
    """
    # local import to avoid hard module dependency at import time
    from .selection import select_mask  # type: ignore, pragma: no cover

    return select_mask(da_tr, spec)  # pragma: no cover


def _theta_pa_to_math(pa: np.ndarray) -> np.ndarray:
    """
    Convert PA (from +y toward +x) into math angle (from +x toward +y, CCW)
    in the index coordinate system whose axis directions are set by (sx, sy).
    """
    theta_math = np.pi / 2 - pa
    return theta_math % (2.0 * np.pi)


def _theta_math_to_pa(theta_math: np.ndarray) -> np.ndarray:
    """
    Convert math angle in index basis back to PA in world-like basis
    where PA is measured from +y toward +x.
    """
    pa = np.pi / 2 - theta_math
    return pa % (2.0 * np.pi)


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
        out[:, 5] = _theta_pa_to_math(out[:, 5])
        return out

    def _conv_list_of_dicts(
        lst: Sequence[Dict[str, Number]],
    ) -> List[Dict[str, Number]]:
        new = []
        for d in lst:
            dd = dict(d)
            if "theta" in dd:
                dd["theta"] = float(
                    _theta_pa_to_math(np.array([dd["theta"]], float))[0]
                )
            new.append(dd)
        return new

    if isinstance(init, np.ndarray):
        return _conv_arr(init)

    if (
        isinstance(init, (list, tuple))
        and (len(init) == n)
        and isinstance(init[0], dict)
    ):
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
    return init  # pragma: no cover


def _axis_is_valid(a: np.ndarray) -> bool:
    """
    True if a is 1-D, finite, and strictly monotonic (ascending or descending).
    """
    a = np.asarray(a)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        return False
    d = np.diff(a)
    return np.all(d > 0) or np.all(d < 0)


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
            raise ValueError(
                "coord_type must be 'world' or 'pixel' for DataArray inputs"
            )
        if ctype == "pixel":
            return np.arange(ny, dtype=float), np.arange(nx, dtype=float)
        # world coords from the DataArray
        if (dim_x not in da_tr.coords) or (dim_y not in da_tr.coords):
            raise ValueError(
                f"DataArray is missing coords for dims ({dim_y}, {dim_x}) required for world fitting."
            )
        x1d = np.asarray(da_tr.coords[dim_x].values)
        y1d = np.asarray(da_tr.coords[dim_y].values)
        if x1d.ndim != 1 or y1d.ndim != 1 or x1d.size != nx or y1d.size != ny:
            raise ValueError(
                "World coords must be 1-D and match the data shape along (y, x)."
            )  # pragma: no cover
        if (not _axis_is_valid(x1d)) or (not _axis_is_valid(y1d)):
            raise ValueError(
                "World coords must be strictly monotonic and finite along both axes."
            )
        return y1d.astype(float), x1d.astype(float)

    # NumPy/Dask input: coord_type is ignored; pick by presence of coords
    if coords is not None:
        if len(coords) != 2:
            raise ValueError("coords must be a tuple/list of (x1d, y1d).")
        x1d, y1d = coords[0], coords[1]
        x1d = np.asarray(x1d, dtype=float)
        y1d = np.asarray(y1d, dtype=float)
        if x1d.ndim != 1 or y1d.ndim != 1 or x1d.size != nx or y1d.size != ny:
            raise ValueError(
                "coords must be 1-D arrays with lengths matching (nx, ny)."
            )
        return y1d, x1d

    # Fallback: pixel indices
    return np.arange(ny, dtype=float), np.arange(nx, dtype=float)


# ---------------------------------------------------------------------------
# Pixel→World interpolation helper (extracted for reliable coverage)
# ---------------------------------------------------------------------------
def _interp_centers_world(
    ds: xr.Dataset,
    cx: np.ndarray,
    cy: np.ndarray,
    dim_x: str,
    dim_y: str,
) -> xr.Dataset:
    """
    Interpolate fitted pixel centers (and propagate their errors) into world
    coordinates when the DataArray carried 1-D coords on (x,y).
    - Handles ascending and descending coords.
    - Uses local slope from `np.gradient` for uncertainty propagation.
    """

    def _prep(coord: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        coord = np.asarray(coord, dtype=float)
        if coord.size < 2 or not np.all(np.isfinite(coord)):
            return None, None
        if coord[1] < coord[0]:  # descending → reverse for interp
            idx = np.arange(coord.size - 1, -1, -1, dtype=float)
            return idx, coord[::-1]
        if not np.all(np.diff(coord) > 0):
            return None, None
        return np.arange(coord.size, dtype=float), coord

    idx_x, val_x = _prep(cx)
    idx_y, val_y = _prep(cy)
    if idx_x is None or idx_y is None:
        return ds

    def _interp_x(v: np.ndarray) -> np.ndarray:
        return np.interp(v, idx_x, val_x)

    def _interp_y(v: np.ndarray) -> np.ndarray:
        return np.interp(v, idx_y, val_y)

    # choose variable names robustly (legacy 'x0'/'y0' back-compat)
    center_x_var = "x0_pixel" if "x0_pixel" in ds else "x0"
    center_y_var = "y0_pixel" if "y0_pixel" in ds else "y0"
    err_x_var = "x0_pixel_err" if "x0_pixel_err" in ds else "x0_err"
    err_y_var = "y0_pixel_err" if "y0_pixel_err" in ds else "y0_err"

    # centers in world coordinates
    ds["x0_world"] = xr.apply_ufunc(
        _interp_x,
        ds[center_x_var],
        input_core_dims=[["component"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    ds["y0_world"] = xr.apply_ufunc(
        _interp_y,
        ds[center_y_var],
        input_core_dims=[["component"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # propagate uncertainties using local slope of the mapping
    _gx = np.gradient(val_x)
    _gy = np.gradient(val_y)

    def _slope_x(v: np.ndarray) -> np.ndarray:
        return np.interp(v, idx_x, _gx)

    def _slope_y(v: np.ndarray) -> np.ndarray:
        return np.interp(v, idx_y, _gy)

    _sx_at = xr.apply_ufunc(
        _slope_x,
        ds[center_x_var],
        input_core_dims=[["component"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    _sy_at = xr.apply_ufunc(
        _slope_y,
        ds[center_y_var],
        input_core_dims=[["component"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    ds["x0_world_err"] = xr.apply_ufunc(
        np.multiply,
        _sx_at,
        ds[err_x_var],
        input_core_dims=[["component"], ["component"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    ds["y0_world_err"] = xr.apply_ufunc(
        np.multiply,
        _sy_at,
        ds[err_y_var],
        input_core_dims=[["component"], ["component"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # Self-documenting attrs
    ds["x0_world"].attrs[
        "description"
    ] = "Component center x in world coordinates (interpolated from pixel index)."
    ds["y0_world"].attrs[
        "description"
    ] = "Component center y in world coordinates (interpolated from pixel index)."
    ds["x0_world_err"].attrs[
        "description"
    ] = "1-sigma uncertainty of x0_world via local axis slope."
    ds["y0_world_err"].attrs[
        "description"
    ] = "1-sigma uncertainty of y0_world via local axis slope."
    return ds


def _add_variance_explained(ds: xr.Dataset) -> xr.Dataset:
    ds["variance_explained"].attrs["note"] = (
        "R²-style fit quality for each 2-D image plane (y×x). "
        "Measures how much of the pixel-to-pixel variance is explained by the fitted model.\n\n"
        "Definitions (per plane): let Z be the data, \\hat{Z} the fitted model, "
        "R = Z - \\hat{Z} the residuals, and 'offset' the robust baseline (median) used by the fitter.\n\n"
        "Explained variance fraction:\n"
        "    EVF = 1 - \\sum (Z - \\hat{Z})^2 \\/ \\sum (Z - \\text{offset})^2\n"
        "Equivalently:\n"
        "    EVF = 1 - \\mathrm{Var}(R) \\/ \\mathrm{Var}(Z - \\text{offset})\n\n"
        "Range: clipped to [0, 1]. 1.0 = perfect fit; 0.0 ≈ no improvement over a flat offset. "
        "If the denominator is near zero (nearly flat plane), the metric can be unstable.\n\n"
        "Quick gut-check scale (per plane):"
        "\n\n"
        "Quick gut-check scale (per plane):\n"
        "  ≥0.9 — excellent; model captures most structure\n"
        "  0.6–0.9 — usable but imperfect\n"
        "  0.2–0.6 — poor to fair\n"
        "  ≈0.0 — no better than a flat background\n\n"
        "For example, a value of 0.2 might indicate:\n"
        "  • Source shape not well modeled by the chosen number of 2-D Gaussians\n"
        "  • Centers/widths/angle off (bad seeds or tight bounds)\n"
        "  • Background (offset) misestimated\n"
        "  • Coordinates/scales mismatched (pixel vs world, anisotropic scaling)\n"
        "  • Additional structure present (neighbors, wings, gradients)\n\n"
        "For low values, some additional ideas:\n"
        "  • Inspect residuals — should look noise-like if the model is right\n"
        "  • Loosen/improve seeds or bounds; try adding a component\n"
        "  • Check background handling\n"
        "  • Verify frame/scale (pixel vs world, local pixel size)\n"
        "  • If noise is high, a low value can be expected — consider SNR or weighting in the fit"
    )
    return ds


def _build_call(
    _inspect,
    _param,
):
    # Build "call" with only parameters that differ from defaults (best-effort)
    # putting in own fucntion to try to get coverage to recognize it
    _call = "fit_multi_gaussian2d("
    try:
        if _inspect is not None:
            _sig = _inspect.signature(fit_multi_gaussian2d)
            _defs = {k: v.default for k, v in _sig.parameters.items()}
            _pairs = []
            for k, v in _param.items():
                dv = _defs.get(k, object())
                try:
                    same = v == dv
                except Exception:
                    same = False
                if (v is None and dv is None) or same:
                    continue
                _pairs.append(f"{k}={_short(v)}")
            _call += ", ".join(_pairs)
        _call += ")"
    except Exception:
        _call = "fit_multi_gaussian2d(...)"
    return _call


def _add_angle_attrs(ds, conv, frame):
    # putting in func to try to get coverage to recognize it
    for _name, _frame in (
        ("theta_pixel", "pixel"),
        ("theta_pixel_err", "pixel"),
        ("theta_world", "world"),
        ("theta_world_err", "world"),
    ):
        # it looks like the if may not be needed, and if its left in
        # code coverage flags uncovered lines
        # if _name in ds:
        ds[_name].attrs["convention"] = conv
        ds[_name].attrs["frame"] = _frame
        ds[_name].attrs["units"] = "rad"
    return ds


# ----------------------- Public API -----------------------


def fit_multi_gaussian2d(
    data: ArrayOrDA,
    n_components: int,
    dims: Optional[Sequence[Union[str, int]]] = None,
    *,
    mask: Optional[ArrayOrDA] = None,
    min_threshold: Optional[Number] = None,
    max_threshold: Optional[Number] = None,
    initial_guesses: Optional[
        Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]
    ] = None,
    bounds: Optional[
        Dict[str, Union[Tuple[float, float], Sequence[Tuple[float, float]]]]
    ] = None,
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

    mask: numpy.ndarray | dask.array.Array | xarray.DataArray | None
      Optional boolean mask. **True keeps** a pixel. If 2-D with dims (y, x), it is
      broadcast across leading dims; if it matches the full array shape, it is used
      elementwise. Combined with thresholds as: `final_mask = mask ∧ (>= min_threshold) ∧ (<= max_threshold)`.

    min_threshold: float | None
      Inclusive lower threshold; pixels with values < min_threshold are ignored during the fit.

    max_threshold: float | None
      Inclusive upper threshold; pixels with values > max_threshold are ignored during the fit.
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
    da_tr = da_in.transpose(
        *(d for d in da_in.dims if d not in (dim_y, dim_x)), dim_y, dim_x
    )
    core = [dim_y, dim_x]

    # Resolve mask into a boolean DataArray aligned to da_tr
    if mask is None:
        mask_da = xr.ones_like(da_tr, dtype=bool)
    else:
        if isinstance(mask, str):
            # Resolve on-the-fly (OTF) string mask via selection helper
            mda = _select_mask(da_tr, mask)
            # Allow selector to return a Dataset wrapper containing 'mask'
            if isinstance(mda, xr.Dataset):
                if "mask" in mda:
                    mda = mda["mask"]
                else:
                    raise TypeError(
                        "selection.select_mask returned a Dataset without a 'mask' variable"
                    )
            # Normalize plain ndarray → DataArray with appropriate dims
            if isinstance(mda, np.ndarray):
                mda = xr.DataArray(
                    mda, dims=[dim_y, dim_x] if mda.ndim == 2 else da_tr.dims
                )
            if not isinstance(mda, xr.DataArray):
                raise TypeError("selection.select_mask returned unsupported mask type")
        elif isinstance(mask, xr.DataArray):
            mda = mask
        else:
            # Support 2-D (y,x) or full-shape masks
            _m = mask
            if getattr(_m, "ndim", None) == 2:
                mda = xr.DataArray(_m, dims=[dim_y, dim_x])
            else:
                mda = xr.DataArray(_m, dims=da_tr.dims)
        mda = mda.astype(bool)
        # Align & broadcast to data
        mda, _ = xr.align(mda, da_tr, join="right")
        mask_da = mda

    # 2a) Determine axis signs from coords (or defaults)
    cx = np.asarray(da_tr.coords[dim_x].values) if dim_x in da_tr.coords else None
    cy = np.asarray(da_tr.coords[dim_y].values) if dim_y in da_tr.coords else None
    sx_sign = _axis_sign(cx)
    sy_sign = _axis_sign(cy)
    is_left_handed = (sx_sign * sy_sign) < 0.0

    # Convert initial guesses: (a) optional FWHM→σ, (b) theta PA→math if requested.
    ig = initial_guesses
    if (
        ig is not None
        and initial_is_fwhm
        and isinstance(ig, np.ndarray)
        and ig.shape == (int(n_components), 6)
    ):
        # columns: [amp, x0, y0, fwhm_major, fwhm_minor, theta]  → convert to σ for the fitter
        ig = ig.copy()
        ig[:, 3] = _sigma_from_fwhm(ig[:, 3])
        ig[:, 4] = _sigma_from_fwhm(ig[:, 4])
    # dict/list forms: allow fwhm_major/fwhm_minor keys by mapping to sigma_x/sigma_y
    if (
        isinstance(ig, dict)
        and "components" in ig
        and isinstance(ig["components"], np.ndarray)
    ):
        arr = np.asarray(ig["components"], dtype=float)
        if arr.shape == (int(n_components), 6) and initial_is_fwhm:
            arr = arr.copy()
            arr[:, 3] = _sigma_from_fwhm(arr[:, 3])
            arr[:, 4] = _sigma_from_fwhm(arr[:, 4])
            ig = dict(ig)
            ig["components"] = arr
    elif isinstance(ig, (list, tuple)) and ig and isinstance(ig[0], dict):
        mapped: List[Dict[str, Any]] = []
        for d in ig:
            dd = dict(d)
            if "fwhm_major" in dd:
                dd["sigma_x"] = float(_sigma_from_fwhm(dd.pop("fwhm_major")))
            if "fwhm_minor" in dd:
                dd["sigma_y"] = float(_sigma_from_fwhm(dd.pop("fwhm_minor")))
            mapped.append(dd)
        ig = mapped

    want_pa = (angle == "pa") or (angle == "auto" and is_left_handed)
    init_for_fit = _convert_init_theta(
        ig, to_math=want_pa, sx=sx_sign, sy=sy_sign, n=int(n_components)
    )

    out_dtypes = (
        [np.float64] * 6  # params per component
        + [np.float64] * 6  # errors per component
        + [np.float64] * 4  # derived per component (add peak_err)
        + [np.float64, np.float64]  # offset, offset_err
        + [np.bool_, np.float64]  # success, variance_explained
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
                target = "sigma_x" if k == "fwhm_major" else "sigma_y"
                if (
                    isinstance(v, (list, tuple))
                    and v
                    and isinstance(v[0], (list, tuple))
                ):
                    b2[target] = [
                        (float(lo) * conv, float(hi) * conv) for (lo, hi) in v
                    ]  # per-component
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
        np.allclose(x1d, np.arange(x1d.shape[0]))
        and np.allclose(y1d, np.arange(y1d.shape[0]))
    )
    out_core_dims = (
        [["component"]] * 6
        + [["component"]] * 6
        + [["component"]] * 4
        + [[], []]
        + [[], []]
        + [[dim_y, dim_x], [dim_y, dim_x]]
    )
    results = xr.apply_ufunc(
        _multi_fit_plane_wrapper,
        da_tr,
        mask_da,
        y1d_da,
        x1d_da,
        input_core_dims=[core, core, [dim_y], [dim_x]],
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

    (
        amp,
        x0,
        y0,
        sx,
        sy,
        th,
        amp_e,
        x0_e,
        y0_e,
        sx_e,
        sy_e,
        th_e,
        fwhm_maj,
        fwhm_min,
        peak,
        peak_err,
        offset,
        offset_e,
        success,
        varexp,
        residual,
        model,
    ) = results

    # -- Angle conventions & CANONICALIZATION -------------------------------
    # Internal angle sign → math: th_math = -th
    th_math = -th

    # Canonicalize ellipse so that:
    #   • sigma_x ≥ sigma_y (major/minor ordering)
    #   • theta refers to the MAJOR axis
    #   • theta wrapped to (-π/2, π/2]
    _is_major_sx = xr.apply_ufunc(np.greater_equal, sx, sy, dask="parallelized")
    # swap widths & their errors if needed
    sx, sy = xr.where(_is_major_sx, sx, sy), xr.where(_is_major_sx, sy, sx)
    sx_e, sy_e = xr.where(_is_major_sx, sx_e, sy_e), xr.where(_is_major_sx, sy_e, sx_e)
    # rotate theta by π/2 when we swapped axes
    th_math = xr.where(_is_major_sx, th_math, th_math + np.pi / 2)

    # wrap to (-π/2, π/2]
    def _wrap_halfpi(t):
        return ((t + np.pi / 2) % np.pi) - np.pi / 2

    th_math = xr.apply_ufunc(_wrap_halfpi, th_math, dask="parallelized", vectorize=True)

    # Now publish θ in requested convention
    theta_math = th_math
    theta_pa = xr.DataArray(
        _theta_math_to_pa(th_math.values), dims=th.dims, coords=th.coords
    )
    # Preserve backward-compat "theta" using requested convention
    th_public = theta_pa if want_pa else theta_math

    # --- Pixel-parameter views (centers + ellipse in pixel coords) ---
    if world_mode:
        nx = x1d.shape[0]
        ny = y1d.shape[0]
        x_idx_axis = np.arange(nx, dtype=float)
        y_idx_axis = np.arange(ny, dtype=float)
        # centers: world -> pixel via inverse of coord arrays
        x0_pixel = xr.apply_ufunc(
            np.interp,
            x0,
            xr.DataArray(x1d, dims=[dim_x]),
            xr.DataArray(x_idx_axis, dims=[dim_x]),
            input_core_dims=[["component"], [dim_x], [dim_x]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        y0_pixel = xr.apply_ufunc(
            np.interp,
            y0,
            xr.DataArray(y1d, dims=[dim_y]),
            xr.DataArray(y_idx_axis, dims=[dim_y]),
            input_core_dims=[["component"], [dim_y], [dim_y]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # local pixel scales (world units per pixel) via gradient at nearest pixel
        gx = np.gradient(x1d.astype(float))
        gy = np.gradient(y1d.astype(float))
        x0i = (
            xr.apply_ufunc(
                np.rint,
                x0_pixel,
                input_core_dims=[["component"]],
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            .clip(0, nx - 1)
            .astype(int)
        )
        y0i = (
            xr.apply_ufunc(
                np.rint,
                y0_pixel,
                input_core_dims=[["component"]],
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            .clip(0, ny - 1)
            .astype(int)
        )
        """
        # elementwise pick with clipping (works with vectorize=True)
        def _pick1d(arr, idx):
            idx = np.asarray(idx).astype(np.int64)
            np.clip(idx, 0, arr.shape[0]-1, out=idx)
            return arr[idx]
        """
        # pick local scale via np.take with clipping
        dx_local = xr.apply_ufunc(
            np.take,
            xr.DataArray(gx, dims=[dim_x]),
            x0i,
            input_core_dims=[[dim_x], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        dy_local = xr.apply_ufunc(
            np.take,
            xr.DataArray(gy, dims=[dim_y]),
            y0i,
            input_core_dims=[[dim_y], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # covariance transform Σp = S Σw S (S=diag(1/dx,1/dy)) → principal axes in pixel units
        def _world_to_pixel_cov(sigx, sigy, theta, dx, dy):
            c = np.cos(theta)
            s = np.sin(theta)
            # Σ_world = R diag(sigx^2, sigy^2) R^T
            a = (c * c) * sigx * sigx + (s * s) * sigy * sigy
            b = (s * c) * (sigx * sigx - sigy * sigy)
            d = (s * s) * sigx * sigx + (c * c) * sigy * sigy
            invdx2 = 1.0 / (dx * dx)
            invdy2 = 1.0 / (dy * dy)
            A = a * invdx2
            B = b * (1.0 / (dx * dy))
            D = d * invdy2
            tr = A + D
            det = A * D - B * B
            tmp = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
            lam1 = tr / 2.0 + tmp
            lam2 = tr / 2.0 - tmp
            theta_p = 0.5 * np.arctan2(2 * B, A - D)
            lam_max = np.maximum(lam1, lam2)
            lam_min = np.minimum(lam1, lam2)
            return (
                np.sqrt(np.maximum(lam_max, 0.0)),
                np.sqrt(np.maximum(lam_min, 0.0)),
                theta_p,
            )

        # Feed math-angle into geometry/covariance computation
        sigma_major_pixel, sigma_minor_pixel, theta_pixel_math = xr.apply_ufunc(
            _world_to_pixel_cov,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            input_core_dims=[["component"]] * 5,
            output_core_dims=[["component"], ["component"], ["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float],
        )
        # Report pixel-space angle in the SAME convention as 'theta'
        theta_pixel = (np.pi / 2 - theta_pixel_math) if want_pa else theta_pixel_math
        fwhm_major_pixel = sigma_major_pixel * _SIG2FWHM
        fwhm_minor_pixel = sigma_minor_pixel * _SIG2FWHM

        # World-basis angle and its error are native in this branch
        theta_world = th_public
        theta_world_err = th_e  # same convention as 'theta_world'

        # --- σ error propagation (world → pixel) via finite differences (w.r.t σx, σy) ---
        def _fd_sigma_world_to_pixel(sigx, sigy, theta, dx, dy, epsx, epsy):
            # central differences on major/minor wrt (sigx, sigy); returns (dmaj_dsigx, dmin_dsigx, dmaj_dsigy, dmin_dsigy)
            s1M, s1m, _ = _world_to_pixel_cov(sigx + epsx, sigy, theta, dx, dy)
            s2M, s2m, _ = _world_to_pixel_cov(sigx - epsx, sigy, theta, dx, dy)
            dM_dsx = (s1M - s2M) / (2.0 * epsx)
            dm_dsx = (s1m - s2m) / (2.0 * epsx)
            s1M, s1m, _ = _world_to_pixel_cov(sigx, sigy + epsy, theta, dx, dy)
            s2M, s2m, _ = _world_to_pixel_cov(sigx, sigy - epsy, theta, dx, dy)
            dM_dsy = (s1M - s2M) / (2.0 * epsy)
            dm_dsy = (s1m - s2m) / (2.0 * epsy)
            return dM_dsx, dm_dsx, dM_dsy, dm_dsy

        _epsx = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), sx, dask="parallelized"
        )
        _epsy = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), sy, dask="parallelized"
        )
        dM_dsx, dm_dsx, dM_dsy, dm_dsy = xr.apply_ufunc(
            _fd_sigma_world_to_pixel,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            _epsx,
            _epsy,
            input_core_dims=[["component"]] * 7,
            output_core_dims=[
                ["component"],
                ["component"],
                ["component"],
                ["component"],
            ],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float, float],
        )
        sigma_major_pixel_err = xr.apply_ufunc(
            lambda a, b, c, d: np.sqrt((a * b) ** 2 + (c * d) ** 2),
            dM_dsx,
            sx_e,
            dM_dsy,
            sy_e,
            input_core_dims=[
                ["component"],
                ["component"],
                ["component"],
                ["component"],
            ],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        sigma_minor_pixel_err = xr.apply_ufunc(
            lambda a, b, c, d: np.sqrt((a * b) ** 2 + (c * d) ** 2),
            dm_dsx,
            sx_e,
            dm_dsy,
            sy_e,
            input_core_dims=[
                ["component"],
                ["component"],
                ["component"],
                ["component"],
            ],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # --- θ error propagation (world → pixel): d(theta_pixel_math)/d(theta_world_math) ---
        def _dtheta_pix_dtheta_world(sigx, sigy, theta, dx, dy, eps):
            _, _, t1 = _world_to_pixel_cov(sigx, sigy, theta + eps, dx, dy)
            _, _, t2 = _world_to_pixel_cov(sigx, sigy, theta - eps, dx, dy)
            return (t1 - t2) / (2.0 * eps)

        _epst = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), th_math, dask="parallelized"
        )
        _dtdt = xr.apply_ufunc(
            _dtheta_pix_dtheta_world,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            _epst,
            input_core_dims=[["component"]] * 6,
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # convert-to-PA is a shift, so σ is unchanged between math↔PA
        theta_pixel_err = xr.apply_ufunc(
            np.multiply,
            xr.apply_ufunc(np.abs, _dtdt, dask="parallelized"),
            th_e,
            input_core_dims=[["component"], ["component"]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # --- pixel-center uncertainties from world centers via local scale ---
        x0_pixel_err = xr.apply_ufunc(
            np.divide,
            x0_e,
            dx_local,
            input_core_dims=[["component"], ["component"]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        y0_pixel_err = xr.apply_ufunc(
            np.divide,
            y0_e,
            dy_local,
            input_core_dims=[["component"], ["component"]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        # World-view σ and FWHM are native in this branch
        sigma_x_world, sigma_y_world = sx, sy
        sigma_x_world_err, sigma_y_world_err = sx_e, sy_e
        sigma_major_world = xr.apply_ufunc(
            np.maximum, sigma_x_world, sigma_y_world, dask="parallelized"
        )
        sigma_minor_world = xr.apply_ufunc(
            np.minimum, sigma_x_world, sigma_y_world, dask="parallelized"
        )
        _is_sx_major_w = xr.apply_ufunc(
            np.greater_equal, sigma_x_world, sigma_y_world, dask="parallelized"
        )
        sigma_major_world_err = xr.where(
            _is_sx_major_w, sigma_x_world_err, sigma_y_world_err
        )
        sigma_minor_world_err = xr.where(
            _is_sx_major_w, sigma_y_world_err, sigma_x_world_err
        )
        fwhm_major_world = sigma_major_world * _SIG2FWHM
        fwhm_minor_world = sigma_minor_world * _SIG2FWHM
        fwhm_major_world_err = sigma_major_world_err * _SIG2FWHM
        fwhm_minor_world_err = sigma_minor_world_err * _SIG2FWHM
        # Pixel FWHM errors from σ errors
        fwhm_major_pixel_err = sigma_major_pixel_err * _SIG2FWHM
        fwhm_minor_pixel_err = sigma_minor_pixel_err * _SIG2FWHM

    else:
        # already in pixel space -> alias
        x0_pixel, y0_pixel = x0, y0
        sigma_major_pixel = xr.apply_ufunc(np.maximum, sx, sy, dask="parallelized")
        sigma_minor_pixel = xr.apply_ufunc(np.minimum, sx, sy, dask="parallelized")
        theta_pixel = th_public
        fwhm_major_pixel = xr.apply_ufunc(
            np.maximum, sx * _SIG2FWHM, sy * _SIG2FWHM, dask="parallelized"
        )
        fwhm_minor_pixel = xr.apply_ufunc(
            np.minimum, sx * _SIG2FWHM, sy * _SIG2FWHM, dask="parallelized"
        )
        # Angular uncertainty is native in pixel basis
        theta_pixel_err = th_e
        # world angle/uncertainty computed below from pixel→world covariance

        # and pixel-center uncertainties are the native ones
        x0_pixel_err, y0_pixel_err = x0_e, y0_e
        # Pixel σ errors are native in this branch
        _is_sx_major_p = xr.apply_ufunc(np.greater_equal, sx, sy, dask="parallelized")
        sigma_major_pixel_err = xr.where(_is_sx_major_p, sx_e, sy_e)
        sigma_minor_pixel_err = xr.where(_is_sx_major_p, sy_e, sx_e)
        fwhm_major_pixel_err = sigma_major_pixel_err * _SIG2FWHM
        fwhm_minor_pixel_err = sigma_minor_pixel_err * _SIG2FWHM

        # Also compute world-view σ/FWHM via pixel→world covariance transform if coords available
        gx = np.gradient(x1d.astype(float))
        gy = np.gradient(y1d.astype(float))
        x0i = (
            xr.apply_ufunc(
                np.rint,
                x0_pixel,
                input_core_dims=[["component"]],
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            .clip(0, x1d.shape[0] - 1)
            .astype(int)
        )
        y0i = (
            xr.apply_ufunc(
                np.rint,
                y0_pixel,
                input_core_dims=[["component"]],
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            .clip(0, y1d.shape[0] - 1)
            .astype(int)
        )
        dx_local = xr.apply_ufunc(
            np.take,
            xr.DataArray(gx, dims=[dim_x]),
            x0i,
            input_core_dims=[[dim_x], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        dy_local = xr.apply_ufunc(
            np.take,
            xr.DataArray(gy, dims=[dim_y]),
            y0i,
            input_core_dims=[[dim_y], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        def _pixel_to_world_cov(sigx, sigy, theta, dx, dy):
            c = np.cos(theta)
            s = np.sin(theta)
            # Σ_pixel = R diag(sigx^2, sigy^2) R^T
            Sxx = (c * c) * sigx * sigx + (s * s) * sigy * sigy
            Sxy = (s * c) * (sigx * sigx - sigy * sigy)
            Syy = (s * s) * sigx * sigx + (c * c) * sigy * sigy
            # world scaling
            A = (dx * dx) * Sxx
            B = (dx * dy) * Sxy
            D = (dy * dy) * Syy
            tr = A + D
            det = A * D - B * B
            tmp = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
            lam1 = tr / 2.0 + tmp
            lam2 = tr / 2.0 - tmp
            theta_w = 0.5 * np.arctan2(2 * B, A - D)
            lam_max = np.maximum(lam1, lam2)
            lam_min = np.minimum(lam1, lam2)
            return (
                np.sqrt(np.maximum(lam_max, 0.0)),
                np.sqrt(np.maximum(lam_min, 0.0)),
                theta_w,
            )

        sigma_major_world, sigma_minor_world, _theta_world_math = xr.apply_ufunc(
            _pixel_to_world_cov,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            input_core_dims=[["component"]] * 5,
            output_core_dims=[["component"], ["component"], ["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float],
        )

        # Error propagation (pixel → world) via finite differences on (σx, σy)
        def _fd_sigma_pixel_to_world(sigx, sigy, theta, dx, dy, epsx, epsy):
            s1M, s1m, _ = _pixel_to_world_cov(sigx + epsx, sigy, theta, dx, dy)
            s2M, s2m, _ = _pixel_to_world_cov(sigx - epsx, sigy, theta, dx, dy)
            dM_dsx = (s1M - s2M) / (2.0 * epsx)
            dm_dsx = (s1m - s2m) / (2.0 * epsx)
            s1M, s1m, _ = _pixel_to_world_cov(sigx, sigy + epsy, theta, dx, dy)
            s2M, s2m, _ = _pixel_to_world_cov(sigx, sigy - epsy, theta, dx, dy)
            dM_dsy = (s1M - s2M) / (2.0 * epsy)
            dm_dsy = (s1m - s2m) / (2.0 * epsy)
            return dM_dsx, dm_dsx, dM_dsy, dm_dsy

        # World angle in requested convention
        theta_world = (np.pi / 2 - _theta_world_math) if want_pa else _theta_world_math

        # --- θ error propagation (pixel → world): d(theta_world_math)/d(theta_pixel_math) ---
        def _dtheta_world_dtheta_pixel(sigx, sigy, theta, dx, dy, eps):
            _, _, t1 = _pixel_to_world_cov(sigx, sigy, theta + eps, dx, dy)
            _, _, t2 = _pixel_to_world_cov(sigx, sigy, theta - eps, dx, dy)
            return (t1 - t2) / (2.0 * eps)

        _epst_w = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), th_math, dask="parallelized"
        )
        _dtdt_w = xr.apply_ufunc(
            _dtheta_world_dtheta_pixel,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            _epst_w,
            input_core_dims=[["component"]] * 6,
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        theta_world_err = xr.apply_ufunc(
            np.multiply,
            xr.apply_ufunc(np.abs, _dtdt_w, dask="parallelized"),
            th_e,
            input_core_dims=[["component"], ["component"]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        _epsx = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), sx, dask="parallelized"
        )
        _epsy = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), sy, dask="parallelized"
        )
        dM_dsx, dm_dsx, dM_dsy, dm_dsy = xr.apply_ufunc(
            _fd_sigma_pixel_to_world,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            _epsx,
            _epsy,
            input_core_dims=[["component"]] * 7,
            output_core_dims=[
                ["component"],
                ["component"],
                ["component"],
                ["component"],
            ],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float, float],
        )
        sigma_major_world_err = xr.apply_ufunc(
            lambda a, b, c, d: np.sqrt((a * b) ** 2 + (c * d) ** 2),
            dM_dsx,
            sx_e,
            dM_dsy,
            sy_e,
            input_core_dims=[
                ["component"],
                ["component"],
                ["component"],
                ["component"],
            ],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        sigma_minor_world_err = xr.apply_ufunc(
            lambda a, b, c, d: np.sqrt((a * b) ** 2 + (c * d) ** 2),
            dm_dsx,
            sx_e,
            dm_dsy,
            sy_e,
            input_core_dims=[
                ["component"],
                ["component"],
                ["component"],
                ["component"],
            ],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        fwhm_major_world = sigma_major_world * _SIG2FWHM
        fwhm_minor_world = sigma_minor_world * _SIG2FWHM
        fwhm_major_world_err = sigma_major_world_err * _SIG2FWHM
        fwhm_minor_world_err = sigma_minor_world_err * _SIG2FWHM
        # For completeness, also expose unsorted σ in world (principal axes order)
        # Choose naming parallel to native σ_x/σ_y as principal axes (x=major, y=minor here)
        # sigma_x_world, sigma_y_world = sigma_major_world, sigma_minor_world
        # sigma_x_world_err, sigma_y_world_err = sigma_major_world_err, sigma_minor_world_err

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
    # Also expose pixel/world principal-axis σ explicitly + their errors
    _is_sx_major = xr.apply_ufunc(np.greater_equal, sx, sy, dask="parallelized")
    sigma_major = xr.where(_is_sx_major, sx, sy)
    sigma_minor = xr.where(_is_sx_major, sy, sx)
    sigma_major_err = xr.where(_is_sx_major, sx_e, sy_e)
    sigma_minor_err = xr.where(_is_sx_major, sy_e, sx_e)

    # sigma_x_pixel = sigma_major_pixel
    # sigma_y_pixel = sigma_minor_pixel
    # sigma_x_pixel_err = sigma_major_pixel_err
    # sigma_y_pixel_err = sigma_minor_pixel_err
    # World principal-axis already prepared above:
    #   sigma_x_world, sigma_y_world, ... and fwhm_*_world(_err)
    # (theta_pixel is the orientation of the major axis in pixel basis)
    # Provide FWHM in pixel units too:

    ds = xr.Dataset(
        data_vars=dict(
            amplitude=amp,
            amplitude_err=amp_e,
            peak=peak,
            peak_err=peak_err,
            # pixel-space mirrors
            x0_pixel=x0_pixel,
            y0_pixel=y0_pixel,
            x0_pixel_err=x0_pixel_err,
            y0_pixel_err=y0_pixel_err,
            sigma_major_pixel=sigma_major_pixel,
            sigma_minor_pixel=sigma_minor_pixel,
            sigma_major_pixel_err=sigma_major_pixel_err,
            sigma_minor_pixel_err=sigma_minor_pixel_err,
            theta_pixel=theta_pixel,
            theta_pixel_err=theta_pixel_err,
            theta_world=theta_world,
            theta_world_err=theta_world_err,
            fwhm_major_pixel=fwhm_major_pixel,
            fwhm_minor_pixel=fwhm_minor_pixel,
            fwhm_major_pixel_err=fwhm_major_pixel_err,
            fwhm_minor_pixel_err=fwhm_minor_pixel_err,
            # world-space mirrors (explicit)
            sigma_major_world=sigma_major_world,
            sigma_minor_world=sigma_minor_world,
            sigma_major_world_err=sigma_major_world_err,
            sigma_minor_world_err=sigma_minor_world_err,
            fwhm_major_world=fwhm_major_world,
            fwhm_minor_world=fwhm_minor_world,
            fwhm_major_world_err=fwhm_major_world_err,
            fwhm_minor_world_err=fwhm_minor_world_err,
            offset=offset,
            offset_err=offset_e,
            success=success,
            variance_explained=varexp,
        )
    )
    # --- record only axes handedness; publish both angle conventions explicitly ---
    conv = "pa" if want_pa else "math"
    ds.attrs["axes_handedness"] = "left" if is_left_handed else "right"
    # Record the chosen theta convention ("math" or "pa") for self-documentation
    ds.attrs["theta_convention"] = conv
    # Add explicit angular units to all theta-related outputs
    # Annotate angle variables with convention + frame (no bare 'theta' var in this DS)

    ds = _add_angle_attrs(ds, conv, "pixel")

    if return_residual:
        ds["residual"] = residual
    if return_model:
        ds["model"] = model

    # world coord exposure adjusted: alias if already in world, otherwise interp from pixels.
    if (dim_x in da_tr.coords) and (dim_y in da_tr.coords):
        cx = np.asarray(da_tr.coords[dim_x].values)
        cy = np.asarray(da_tr.coords[dim_y].values)

        x_valid = _axis_is_valid(cx)
        y_valid = _axis_is_valid(cy)
        axes_valid = x_valid and y_valid

        if world_mode:
            # Already fitted in world coords → x0/y0 are world; expose direct aliases.
            ds["x0_world"] = x0
            ds["y0_world"] = y0
            # Uncertainties are the same coordinate system → direct aliases
            ds["x0_world_err"] = x0_e
            ds["y0_world_err"] = y0_e
            # Self-documenting attrs
            ds["x0_world"].attrs[
                "description"
            ] = "Component center x in world coordinates."
            ds["y0_world"].attrs[
                "description"
            ] = "Component center y in world coordinates."
            ds["x0_world_err"].attrs["description"] = "1-sigma uncertainty of x0_world."
            ds["y0_world_err"].attrs["description"] = "1-sigma uncertainty of y0_world."
        else:

            def _prep(
                coord: np.ndarray,
            ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                coord = np.asarray(coord)
                if coord.ndim != 1 or coord.size == 0 or not np.all(np.isfinite(coord)):
                    return None, None
                if (
                    coord.size >= 2 and coord[1] < coord[0]
                ):  # descending → reverse for interp
                    idx = np.arange(coord.size - 1, -1, -1, dtype=float)
                    return idx, coord[::-1]
                if not np.all(np.diff(coord) > 0):
                    return None, None
                return np.arange(coord.size, dtype=float), coord

            # Extracted for testability & reliable coverage:
            ds = _interp_centers_world(ds, cx, cy, dim_x, dim_y)

    # --- record invocation metadata on the result Dataset ---
    try:
        import inspect as _inspect
    except Exception:
        _inspect = None
    try:
        from importlib import metadata as _ilm
    except Exception:
        _ilm = None

    def _short(v, maxlen=120):
        import numpy as _np
        import xarray as _xr

        if v is None:
            return None
        if isinstance(v, (float, int, bool, str)):
            return v
        if isinstance(v, dict):
            # shallow sanitize
            out = {}
            for k, vv in list(v.items())[:50]:
                out[k] = _short(vv, maxlen // 2)
            return out
        if isinstance(v, (list, tuple)):
            return [_short(x, maxlen // 2) for x in list(v)[:50]]
        # arrays / DataArrays / other objects
        try:
            shape = tuple(getattr(v, "shape", ()))
            return f"<{type(v).__name__} shape={shape}>"
        except Exception:
            s = repr(v)
            return s if len(s) <= maxlen else (s[: maxlen - 3] + "...")

    # Full parameter set (exclude raw data)
    _param = dict(
        n_components=int(n_components),
        dims=(list(dims) if dims is not None else None),
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        initial_guesses=_short(initial_guesses),
        bounds=_short(bounds),
        initial_is_fwhm=bool(initial_is_fwhm),
        max_nfev=int(max_nfev),
        return_model=bool(return_model),
        return_residual=bool(return_residual),
        angle=str(angle),
        coord_type=str(coord_type),
        coords=_short(coords),
    )
    _call = _build_call(_inspect, _param)

    # Package metadata
    _pkg = __package__.split(".")[0] if __package__ else "astroviper"
    try:
        _ver = _ilm.version(_pkg) if _ilm is not None else "unknown"
    except Exception:
        try:
            import astroviper as _av

            _ver = getattr(_av, "__version__", "unknown")
        except Exception:
            _ver = "unknown"

    ds.attrs["call"] = _call
    ds.attrs["param"] = _param
    ds.attrs["package"] = _pkg
    ds.attrs["version"] = _ver
    ds.attrs["fit_native_frame"] = "world" if world_mode else "pixel"

    # --- Self-documenting variable metadata ----------------------------------
    _dv_docs = {
        # centers
        "x0_pixel": "Gaussian center x-coordinate in pixel indices (0-based), derived from world coords if necessary.",
        "y0_pixel": "Gaussian center y-coordinate in pixel indices (0-based), derived from world coords if necessary.",
        "x0_pixel_err": "1σ uncertainty of x0_pixel (native if pixel fit; world fit propagated via local pixel scale).",
        "y0_pixel_err": "1σ uncertainty of y0_pixel (native if pixel fit; world fit propagated via local pixel scale).",
        "x0_world": "Component center x in world coordinates (if available).",
        "y0_world": "Component center y in world coordinates (if available).",
        "x0_world_err": "1σ uncertainty of x0_world (direct if world fit; else via interpolation/pixel-scale propagation).",
        "y0_world_err": "1σ uncertainty of y0_world (direct if world fit; else via interpolation/pixel-scale propagation).",
        # scales (sigma = standard deviation of the 1-D Gaussian along ellipse axes before FWHM conversion)
        "sigma_major_pixel": "Gaussian 1σ scale along the major principal axis in pixel units (after world→pixel conversion).",
        "sigma_minor_pixel": "Gaussian 1σ scale along the minor principal axis in pixel units (after world→pixel conversion).",
        "sigma_major_pixel_err": "1σ uncertainty in sigma_major_pixel in pixel units (after world→pixel conversion).",
        "sigma_minor_pixel_err": "1σ uncertainty in sigma_minor_pixel in pixel units (after world→pixel conversion).",
        "sigma_major_world": "Principal-axis 1σ (major) expressed in world coordinates.",
        "sigma_minor_world": "Principal-axis 1σ (minor) expressed in world coordinates.",
        "sigma_major_world_err": "1σ uncertainty of sigma_major_world (native if world fit; else propagated).",
        "sigma_minor_world_err": "1σ uncertainty of sigma_minor_world (native if world fit; else propagated).",
        # FWHM (2*sqrt(2*ln 2) * sigma)
        "fwhm_major_pixel": "Full-width at half-maximum along the major principal axis in pixel coordinates.",
        "fwhm_minor_pixel": "Full-width at half-maximum along the minor principal axis in pixel coordinates.",
        "fwhm_major_pixel_err": "1σ uncertainty of pixel-frame FWHM(major) (native if pixel fit; else propagated).",
        "fwhm_minor_pixel_err": "1σ uncertainty of pixel-frame FWHM(minor) (native if pixel fit; else propagated).",
        "fwhm_major_world": "FWHM of the major principal axis in world coordinates.",
        "fwhm_minor_world": "FWHM of the minor principal axis in world coordinates.",
        "fwhm_major_world_err": "1σ uncertainty of world-frame FWHM(major) (native if world fit; else propagated).",
        "fwhm_minor_world_err": "1σ uncertainty of world-frame FWHM(minor) (native if world fit; else propagated).",
        # angles (θ always refers to the MAJOR axis; wrapped to (-π/2, π/2])
        "theta_pixel": "Orientation of the ellipse MAJOR axis in pixel coordinates (same convention as 'theta').",
        "theta_pixel_err": "1σ uncertainty of theta_pixel (major-axis orientation) in radians; propagated through the world↔pixel transform.",
        "theta_world": "Orientation of the ellipse MAJOR axis in world coordinates (same convention as 'theta').",
        "theta_world_err": "1σ uncertainty of theta_world (major-axis orientation) in radians; propagated through the pixel↔world transform.",
        # amplitudes / background
        "amplitude": "Component amplitude (peak height above offset) in data units.",
        "peak": "Model peak value at the component center (offset + amplitude) in data units.",
        "peak_err": "1σ uncertainty of the model peak (quadrature of amplitude_err and offset_err).",
        "offset": "Additive constant background in data units for this fit plane.",
        "offset_err": "1σ uncertainty on the additive background.",
        # uncertainties
        "amplitude_err": "1σ uncertainty of amplitude parameter in data units.",
        # diagnostics
        "success": "Optimizer success flag (True/False).",
        "variance_explained": "Explained variance fraction by the fitted model on this plane (0–1).",
        # images
        "model": "Best-fit model image on the (y, x) grid of the input.",
        "residual": "Residual image = data - model on the (y, x) grid of the input.",
    }
    for _name, _desc in _dv_docs.items():
        if _name in ds:
            ds[_name].attrs.setdefault("description", _desc)
        if _name.startswith("theta"):
            # Ensure angular units are tagged consistently
            ds[_name].attrs["units"] = "rad"
    # Add an explanatory note for the per-plane explained-variance metric.
    # This text is intentionally verbose to make the DV self-documenting.
    if "variance_explained" in ds:
        # put into own function to try to coerce coverage
        ds = _add_variance_explained(ds)
    return ds


# TODO: move to a plotting module

import numpy as np
from matplotlib.patches import Ellipse


# ---- plotting helpers --------------------------------------------------------


def overlay_fit_components(
    ax,
    fit,
    frame: str,
    metric: str,
    n_sigma: float,
    angle: str,
    edgecolor="k",
    zorder: float = 10.0,
    lw: float = 1.5,
    alpha: float = 0.9,
    label: bool = True,
):
    """
    Draw fitted 2D-Gaussian components as ellipses on a Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    fit : Mapping-like (e.g., xarray.Dataset or dict)
        Result object holding per-component parameters. Fields are probed
        robustly with fallbacks; typical names include:
          - centers:  x0_{pixel|world}, y0_{pixel|world}, x0, y0
          - sizes:    sigma_{pixel|world}_{x|y}, sigma_{pixel|world}_{major|minor}
                      fwhm_{pixel|world}_{x|y},  fwhm_{pixel|world}_{major|minor}
                      (generic fallbacks without frame prefix are also tried)
          - angles:   theta_{pixel|world}_{math|pa}, theta_{pixel|world}, theta
    frame : {"pixel","world"}
        Which frame to prefer when looking up centers/sizes/angles.
    metric : {"sigma","fwhm"}
        Size metric to use for ellipse sizing. If the requested metric is
        unavailable, a conversion is applied from the available one.
    n_sigma : float
        Multiplicative scale applied to σ (or to FWHM after conversion) before
        converting to ellipse half-width/half-height.
        Final ellipse *width/height* are 2 * (scaled radii).
    angle : {"math","pa","auto"}
        Preferred angle convention. "pa" means position angle (east of north).
        "math" means the usual mathematical angle (CCW from +x). "auto" tries
        math then PA. If only PA is available, it is converted to Matplotlib's
        rotation via: rotation_deg = 90° − PA_deg.
    edgecolor : any Matplotlib color
    lw : float
        Line width.
    alpha : float
    label : bool
        If True, annotate each component with its index at the center.

    Notes
    -----
    This implementation is resilient: when requested fields are missing, it
    falls back to other reasonable options and, as a last resort, draws
    axis-aligned ellipses (rotation = 0°) instead of raising KeyError.
    """
    from matplotlib.patches import Ellipse as _Ellipse

    res = fit  # local alias

    # ---------- helpers ----------
    def _present(name: str) -> bool:
        try:
            return name in res
        except Exception:
            return False

    def _arr(name: str):
        if not _present(name):
            return None
        v = res[name]
        # xarray.DataArray or numpy/scalar
        if hasattr(v, "values"):
            a = np.asarray(v.values, dtype=float)
        else:
            a = np.asarray(v, dtype=float)
        return np.atleast_1d(a)

    def _first(*names):
        """Return first present array among names, else None."""
        for n in names:
            a = _arr(n)
            if a is not None:
                return a
        return None

    def _ncomp_guess() -> int:
        """Infer number of components from any present per-component field."""
        candidates = [
            f"x0_{frame}",
            f"y0_{frame}",
            "x0",
            "y0",
            f"sigma_{frame}_x",
            f"sigma_{frame}_y",
            f"sigma_{frame}_major",
            f"sigma_{frame}_minor",
            f"fwhm_{frame}_x",
            f"fwhm_{frame}_y",
            f"fwhm_{frame}_major",
            f"fwhm_{frame}_minor",
            "amp",
            "amplitude",
        ]
        for k in candidates:
            a = _arr(k)
            if a is not None:
                return int(a.shape[0])
        return 1

    def _broadcast(a, n):
        """Broadcast 1-element array to length n."""
        if a is None:
            return None
        if a.size == 1:
            return np.full((n,), float(a[0]))
        return a

    # ---- resolve centers (cx, cy) ----
    cx = _first(f"x0_{frame}", "x0", f"x_{frame}", "x")
    cy = _first(f"y0_{frame}", "y0", f"y_{frame}", "y")

    n = max(
        _ncomp_guess(),
        (1 if cx is None else cx.shape[0]),
        (1 if cy is None else cy.shape[0]),
    )
    cx = _broadcast(cx, n) if cx is not None else np.zeros((n,), dtype=float)
    cy = _broadcast(cy, n) if cy is not None else np.zeros((n,), dtype=float)

    # ---- resolve sizes (semi-axes) with conversions & fallbacks ----
    # For drawing we want radii (half-sizes in data units). We will compute:
    #   if metric == "sigma":    rx = n_sigma * sigma_x;    width = 2*rx
    #   if metric == "fwhm":     rx = 0.5 * n_sigma * fwhm_x; width = 2*rx = n_sigma*fwhm_x
    FWHM_K = 2.0 * np.sqrt(2.0 * np.log(2.0))  # ≈ 2.35482

    # Try x/y first; if not present, fall back to major/minor (both name orders)
    sigx = _first(
        f"sigma_{frame}_x",
        "sigma_x",
        f"sigma_{frame}_major",
        f"sigma_major_{frame}",
        "sigma_major",
    )
    sigy = _first(
        f"sigma_{frame}_y",
        "sigma_y",
        f"sigma_{frame}_minor",
        f"sigma_minor_{frame}",
        "sigma_minor",
    )

    fwx = _first(
        f"fwhm_{frame}_x",
        "fwhm_x",
        f"fwhm_{frame}_major",
        f"fwhm_major_{frame}",
        "fwhm_major",
    )
    fwy = _first(
        f"fwhm_{frame}_y",
        "fwhm_y",
        f"fwhm_{frame}_minor",
        f"fwhm_minor_{frame}",
        "fwhm_minor",
    )

    # broadcast sizes to n
    sigx = _broadcast(sigx, n)
    sigy = _broadcast(sigy, n)
    fwx = _broadcast(fwx, n)
    fwy = _broadcast(fwy, n)

    if metric == "sigma":
        if (sigx is None or sigy is None) and (fwx is not None and fwy is not None):
            # convert FWHM → sigma
            sigx = (fwx / FWHM_K) if sigx is None else sigx
            sigy = (fwy / FWHM_K) if sigy is None else sigy
        if sigx is None or sigy is None:
            warnings.warn(
                "Missing size info (sigma/FWHM); drawing components without size.",
                RuntimeWarning,
            )
            sigx = np.zeros((n,), dtype=float)
            sigy = np.zeros((n,), dtype=float)
        rx = n_sigma * sigx
        ry = n_sigma * sigy
    elif metric == "fwhm":
        if (fwx is None or fwy is None) and (sigx is not None and sigy is not None):
            # convert sigma → FWHM
            fwx = (sigx * FWHM_K) if fwx is None else fwx
            fwy = (sigy * FWHM_K) if fwy is None else fwy
        if fwx is None or fwy is None:
            warnings.warn(
                "Missing size info (FWHM/sigma); drawing components without size.",
                RuntimeWarning,
            )
            fwx = np.zeros((n,), dtype=float)
            fwy = np.zeros((n,), dtype=float)
        # radii are half of (n_sigma * FWHM)
        rx = 0.5 * n_sigma * fwx
        ry = 0.5 * n_sigma * fwy
    else:
        raise ValueError(f"Unknown metric {metric!r}; expected 'sigma' or 'fwhm'.")

    # ---- resolve angles with graceful fallbacks ----
    def _choose_theta(frame_name: str, prefer: str):
        """
        Return (theta_vals, kind) with kind in {'math','pa'}, values in radians.
        """
        # Branchless, behavior-preserving ordering (easier to cover)
        first, second = ("pa", "math") if prefer == "pa" else ("math", "pa")
        cands = [f"theta_{frame_name}_{first}", f"theta_{frame_name}_{second}"]
        for nm in cands:
            a = _arr(nm)
            if a is not None:
                kind = "pa" if nm.endswith("_pa") else "math"
                return a, kind
        return None, None

    prefer_kind = angle if angle in ("pa", "math") else "auto"
    theta_vals, theta_kind = _choose_theta(frame, prefer_kind)
    if theta_vals is None:
        other = "pixel" if frame == "world" else "world"
        theta_vals, theta_kind = _choose_theta(other, prefer_kind)

    if theta_vals is None:
        # Final fallback: axis-aligned
        warnings.warn(
            f"Missing theta for frame '{frame}' (and fallback frame); drawing axis-aligned ellipses.",
            RuntimeWarning,
        )
        theta_deg = np.zeros((n,), dtype=float)
    else:
        theta_vals = _broadcast(theta_vals, n)
        if theta_kind == "pa":
            # Matplotlib Ellipse angle is CCW from +x; PA is E of N
            theta_deg = 90.0 - np.degrees(theta_vals)
        else:
            theta_deg = np.degrees(theta_vals)

    # ---- draw ----
    for i in range(n):
        cx_i = float(cx[i if i < cx.size else 0])
        cy_i = float(cy[i if i < cy.size else 0])
        rx_i = float(rx[i if i < rx.size else 0])
        ry_i = float(ry[i if i < ry.size else 0])
        th_i = float(theta_deg[i if i < theta_deg.size else 0])

        if not np.isfinite([cx_i, cy_i, rx_i, ry_i, th_i]).all():
            continue
        if rx_i <= 0 or ry_i <= 0:
            # size missing/zero → skip drawing but continue
            continue

        e = _Ellipse(
            (cx_i, cy_i),
            width=2.0 * rx_i,
            height=2.0 * ry_i,
            angle=th_i,
            facecolor="none",
            edgecolor=edgecolor,
            lw=lw,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(e)

        if label:
            # simple center label; offset a touch in x so it doesn’t sit on the edge
            ax.text(
                cx_i,
                cy_i,
                f"{i+1}",
                ha="center",
                va="center",
                fontsize=9,
                color=edgecolor,
                alpha=alpha,
                zorder=zorder,
            )


def plot_components(
    data,
    result: "xr.Dataset",
    dims: "Optional[Sequence[Union[str, int]]]" = None,
    *,
    indexer: "Optional[Mapping[str, int]]" = None,
    show_residual: bool = True,
    fwhm: bool = False,
    angle: "Optional[str]" = None,  # "math" | "pa" | None(=auto)
    show: bool = None,
):
    """
    Quicklook: data (and optional residual) with fitted components overlaid as ellipses.

    • Uses world coordinates if the DataArray has numeric, monotonic coords on the plotting dims;
      otherwise falls back to pixel coordinates.
    • Draws either 1-σ or FWHM ellipses (`fwhm=True`) and respects math/PA (`angle=`).
    """
    import numpy as _np

    try:
        import matplotlib.pyplot as _plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Matplotlib is required for plot_components()") from exc

    # Normalize input & dims
    da = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(da, dims)
    da_tr = da.transpose(*(d for d in da.dims if d not in (dim_y, dim_x)), dim_y, dim_x)

    # Select a single plane for plotting
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

    # Decide plotting coords (world if possible)
    use_world = (dim_x in data2d.coords) and (dim_y in data2d.coords)
    if use_world:
        cx = _np.asarray(data2d.coords[dim_x].values)
        cy = _np.asarray(data2d.coords[dim_y].values)
        use_world = (
            cx.ndim == 1
            and cy.ndim == 1
            and _np.all(_np.isfinite(cx))
            and _np.all(_np.isfinite(cy))
            and _np.all(_np.diff(cx) != 0)
            and _np.all(_np.diff(cy) != 0)
        )

    # Figure and panels
    if show_residual and ("residual" in res_plane):
        fig, axes = _plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        ax0, ax1 = axes
    else:
        fig, ax0 = _plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        ax1 = None

    Z = _np.asarray(data2d.values, dtype=float)
    if use_world:
        x = _np.asarray(data2d.coords[dim_x].values, dtype=float)
        y = _np.asarray(data2d.coords[dim_y].values, dtype=float)
        ax0.pcolormesh(x, y, Z, shading="auto")
        frame_for_overlay = "world"
    else:
        ax0.imshow(Z, origin="lower", aspect="equal")
        frame_for_overlay = "pixel"

    ax0.set_title("Data with fitted components")
    ax0.set_xlabel(dim_x)
    ax0.set_ylabel(dim_y)

    # Pick overlay frame to match the axes:
    # pixel-like dims → use pixel frame; otherwise assume world.
    def _dims_are_pixel(dims):
        dl = tuple(n.lower() for n in dims)
        return dl in {("x", "y"), ("i", "j"), ("row", "col"), ("pixel_x", "pixel_y")}

    frame_for_overlay = "pixel" if _dims_are_pixel(dims) else "world"

    # Use the same frame as the axes: world if we're plotting with coord arrays, else pixel
    frame_for_overlay = "world" if use_world else "pixel"

    overlay_fit_components(
        ax0,
        res_plane,
        frame=frame_for_overlay,
        metric=("fwhm" if fwhm else "sigma"),
        n_sigma=1.0,
        angle=(angle or "auto"),
        edgecolor="k",
        lw=1.5,
        alpha=0.9,
        label=True,
    )
    # Residual panel (if present)
    if show_residual and ("residual" in res_plane):
        R = _np.asarray(res_plane["residual"].values, dtype=float)
        if use_world:
            ax1.pcolormesh(x, y, R, shading="auto")
        else:
            ax1.imshow(R, origin="lower", aspect="equal")
        # Draw the same fitted-component ellipses on top for visual QA
        overlay_fit_components(
            ax1,
            res_plane,
            frame=frame_for_overlay,
            metric=("fwhm" if fwhm else "sigma"),
            n_sigma=1.0,
            angle=(angle or "auto"),
            edgecolor="w",  # higher contrast on residuals
            lw=1.5,
            alpha=0.95,
            label=False,
            zorder=10.0,
        )
        ax1.set_title("Residual (data − model)")
        ax1.set_xlabel(dim_x)
        ax1.set_ylabel(dim_y)

    # Optional showing: avoid duplicate displays in notebooks.
    # If show is None, auto-detect: show in scripts, NOT in notebooks.
    if show is None:
        in_ipy = False
        try:
            get_ipython  # type: ignore[name-defined]
            in_ipy = True
        except Exception:
            pass
        show = not in_ipy

    if show:
        _plt.show()

    return fig, (ax0, ax1)
