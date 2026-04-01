# file: src/astroviper/fitting/multi_gaussian2d_fit.py
# 2D gaussian fitter
# Dave Mehringer

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, Any, Dict, List, Mapping
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
    Evaluate one rotated elliptical Gaussian component on a 2-D grid.

    Parameters
    ----------
    X : np.ndarray
        Two-dimensional array of x-coordinate values for the evaluation grid.
    Y : np.ndarray
        Two-dimensional array of y-coordinate values for the evaluation grid.
    amp : float
        Peak amplitude of the Gaussian component above the shared offset.
    x0 : float
        Component center along the x axis, in the same coordinate system as ``X``.
    y0 : float
        Component center along the y axis, in the same coordinate system as ``Y``.
    sx : float
        Gaussian sigma along the intrinsic x-like principal axis. Must be positive.
    sy : float
        Gaussian sigma along the intrinsic y-like principal axis. Must be positive.
    th : float
        Rotation angle in radians, interpreted in the internal math convention
        from +x toward +y.

    Returns
    -------
    np.ndarray
        Array with the same shape as ``X`` and ``Y`` containing the component value
        at each grid position.

    Notes
    -----
    The function does not add a constant background term. ``X`` and ``Y`` are
    assumed to be shape-compatible 2-D grids representing an astronomy image plane
    with x treated as the horizontal/image-column axis and y as the vertical/image-row
    axis.
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
    """
    Convert Gaussian sigma values to full width at half maximum.

    Parameters
    ----------
    sigma : array-like
        Scalar or array of Gaussian sigma values.

    Returns
    -------
    np.ndarray
        Values converted to FWHM using the standard Gaussian relation.

    Notes
    -----
    No positivity check is applied here; callers are expected to pass physically
    meaningful widths.
    """
    return _SIG2FWHM * np.asarray(sigma)


def _sigma_from_fwhm(fwhm):
    """
    Convert full width at half maximum values to Gaussian sigma.

    Parameters
    ----------
    fwhm : array-like
        Scalar or array of FWHM values.

    Returns
    -------
    np.ndarray
        Values converted to sigma using the standard Gaussian relation.

    Notes
    -----
    No positivity check is applied here; callers are expected to pass physically
    meaningful widths.
    """
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
    Pack shared and per-component Gaussian parameters into the optimizer layout.

    Parameters
    ----------
    offset : float
        Shared constant background term for the full model.
    amps : np.ndarray
        Per-component amplitudes.
    x0 : np.ndarray
        Per-component x centers.
    y0 : np.ndarray
        Per-component y centers.
    sx : np.ndarray
        Per-component sigma values along the intrinsic x-like principal axis.
    sy : np.ndarray
        Per-component sigma values along the intrinsic y-like principal axis.
    th : np.ndarray
        Per-component rotation angles in radians.

    Returns
    -------
    np.ndarray
        One-dimensional parameter vector with layout
        ``[offset, amp1, x01, y01, sx1, sy1, th1, ..., ampN, x0N, y0N, sxN, syN, thN]``.

    Notes
    -----
    All per-component arrays are assumed to have the same length ``N`` and already
    use the internal x-before-y parameter convention.
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
    Unpack a packed optimizer parameter vector into component arrays.

    Parameters
    ----------
    params : np.ndarray
        One-dimensional parameter vector in the layout produced by
        :func:`_pack_params`.
    n : int
        Number of Gaussian components encoded in ``params``.

    Returns
    -------
    tuple
        Tuple ``(offset, amps, x0, y0, sx, sy, th)`` where ``offset`` is a scalar
        and the remaining entries are length-``n`` arrays.

    Notes
    -----
    The function assumes ``params`` contains exactly ``1 + 6 * n`` numeric values.
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
    Evaluate the full multi-component Gaussian model on a 2-D grid.

    Parameters
    ----------
    X : np.ndarray
        Two-dimensional x-coordinate grid.
    Y : np.ndarray
        Two-dimensional y-coordinate grid.
    params : np.ndarray
        Packed parameter vector in the format produced by :func:`_pack_params`.
    n : int
        Number of Gaussian components encoded in ``params``.

    Returns
    -------
    np.ndarray
        Model image consisting of the shared offset plus the sum of all components.

    Notes
    -----
    ``X`` and ``Y`` are assumed to be shape-compatible 2-D arrays describing the
    same image plane.
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
    Evaluate the packed multi-Gaussian model on flattened coordinate vectors.

    Parameters
    ----------
    xy : tuple[np.ndarray, np.ndarray]
        Tuple ``(x_flat, y_flat)`` containing one-dimensional coordinate arrays for
        the sampled pixels passed to ``scipy.optimize.curve_fit``.
    *params : float
        Packed model parameters in the same order as :func:`_pack_params`.
    n : int
        Number of Gaussian components represented by ``params``.

    Returns
    -------
    np.ndarray
        One-dimensional model values corresponding to ``xy``.

    Notes
    -----
    This helper avoids reconstructing a 2-D image grid during optimization. The
    coordinate order is explicitly x first, y second.
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
    Select approximate seed peaks by greedily taking the brightest remaining pixel.

    Parameters
    ----------
    z : np.ndarray
        Two-dimensional image plane used for seed detection.
    n : int
        Number of peaks to return.
    excl_radius : int, default 5
        Pixel exclusion radius applied around each selected peak before choosing the
        next one.

    Returns
    -------
    list[tuple[int, int, float]]
        List of ``(y_index, x_index, value)`` tuples, ordered by greedy selection.

    Notes
    -----
    The input plane uses NumPy storage order ``[y, x]``. Returned tuples preserve
    that index order because they refer to pixel indices, not model parameters.
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
    Build default lower and upper bounds for the packed multi-Gaussian parameters.

    Parameters
    ----------
    shape : tuple[int, int]
        Image-plane shape ``(ny, nx)``.
    n : int
        Number of Gaussian components.
    x_rng : tuple[float, float] | None, optional
        Inclusive x-axis coordinate range for center bounds. If ``None``, pixel-index
        bounds derived from ``shape`` are used.
    y_rng : tuple[float, float] | None, optional
        Inclusive y-axis coordinate range for center bounds. If ``None``, pixel-index
        bounds derived from ``shape`` are used.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower-bound and upper-bound vectors in packed parameter order.

    Notes
    -----
    Width bounds are positive-only and use a simple scale based on the coordinate
    span along each axis. Theta is limited to ``[-pi/2, pi/2]`` because the model is
    later canonicalized modulo pi.
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
    """
    Parse component dictionaries into numeric parameter arrays.

    Parameters
    ----------
    comp_list : Sequence[dict]
        Sequence of component descriptions. Supported keys are:
        ``"amp"`` or ``"amplitude"`` for amplitude, ``"x0"``, ``"y0"``,
        ``"sigma_x"`` or ``"sx"`` or ``"fwhm_major"``, ``"sigma_y"`` or ``"sy"``
        or ``"fwhm_minor"``, and optional ``"theta"``.
    n : int
        Expected number of components.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays ``(amps, x0, y0, sx, sy, th)`` each of length ``n``.

    Notes
    -----
    ``fwhm_major`` and ``fwhm_minor`` are converted to sigma internally. Missing
    ``theta`` defaults to ``0.0``. The function raises ``KeyError`` when required
    center or width keys are absent.
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
    """
    Convert a list of component dictionaries plus an offset into packed parameters.

    Parameters
    ----------
    init : Sequence[dict[str, Number]]
        Sequence of component dictionaries accepted by
        :func:`_extract_params_from_comp_dicts`.
    n : int
        Expected number of components.
    offset : float
        Shared constant offset to place at the front of the packed parameter vector.

    Returns
    -------
    np.ndarray
        Packed parameter vector in optimizer order.

    Notes
    -----
    This helper centralizes the list-of-dicts path so tests can exercise it directly.
    """
    if len(init) != n:
        raise ValueError(f"init['components'] must have length n={n}")
    amps, x0, y0, sx, sy, th = _extract_params_from_comp_dicts(init, n)
    return _pack_params(offset, amps, x0, y0, sx, sy, th)


def _is_pixel_index_axes(x1d, y1d):
    """
    Detect whether supplied axis arrays are plain zero-based pixel indices.

    Parameters
    ----------
    x1d : array-like
        Candidate x-axis coordinate array.
    y1d : array-like
        Candidate y-axis coordinate array.

    Returns
    -------
    bool
        ``True`` when both axes are numerically equal to ``np.arange(N)`` for their
        respective lengths.

    Notes
    -----
    This check is used to distinguish true world-coordinate fits from pixel-grid
    fits even when explicit coordinate arrays were supplied.
    """
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
    Normalize user or auto-generated initial guesses into packed optimizer parameters.

    Parameters
    ----------
    z2d : np.ndarray
        Two-dimensional image plane used for auto-seeding and background estimation.
    n : int
        Number of Gaussian components to seed.
    init : np.ndarray | Sequence[dict] | dict | None
        Initial-guess specification. Supported forms are:
        ``None`` for auto-seeding, an ``(n, 6)`` array with columns
        ``[amp, x0, y0, sx, sy, th]``, a length-``n`` list of component dictionaries,
        or a dictionary with optional ``"offset"`` and required ``"components"``.
    min_threshold : Number | None
        Inclusive lower threshold applied before estimating the median and selecting
        automatic seed peaks.
    max_threshold : Number | None
        Inclusive upper threshold applied before estimating the median and selecting
        automatic seed peaks.
    x1d : np.ndarray | None, optional
        One-dimensional x-axis coordinates for world-coordinate auto-seeding.
    y1d : np.ndarray | None, optional
        One-dimensional y-axis coordinates for world-coordinate auto-seeding.

    Returns
    -------
    np.ndarray
        Packed parameter vector in the format expected by the optimizer.

    Notes
    -----
    Auto-seeding estimates the offset as the threshold-masked median and places seed
    centers at greedily selected peaks. When ``x1d`` and ``y1d`` describe non-pixel
    axes, seed centers and widths are expressed in world units; otherwise they remain
    in pixel units.
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
    if arr.shape == (6,) and n == 1:
        arr = arr.reshape(1, 6)
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
    Merge user-specified parameter bounds into default packed lower/upper bounds.

    Parameters
    ----------
    base_lb : np.ndarray
        Default lower-bound vector in packed parameter order.
    base_ub : np.ndarray
        Default upper-bound vector in packed parameter order.
    user_bounds : dict | None
        Optional user bounds. Supported keys are ``"offset"``, ``"amp"``,
        ``"amplitude"``, ``"x0"``, ``"y0"``, ``"sigma_x"``, ``"sigma_y"``,
        ``"fwhm_major"``, ``"fwhm_minor"``, and ``"theta"``. Each value may be a
        single ``(low, high)`` tuple for all components or a length-``n`` sequence of
        tuples for per-component bounds.
    n : int
        Number of Gaussian components.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated ``(lb, ub)`` vectors.

    Notes
    -----
    FWHM keys are converted into sigma-space bounds because the optimizer works in
    sigma units internally. Unknown keys are ignored.
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
    """
    Count ``True`` entries in a mask, stopping once a target count is reached.

    Parameters
    ----------
    mask : np.ndarray
        Boolean-like array to count.
    need : int
        Minimum number of ``True`` values of interest.
    chunk : int, default 262144
        Number of flattened elements to process per chunk.

    Returns
    -------
    int
        Count of ``True`` values observed, or an early-return count once ``need`` has
        been met.

    Notes
    -----
    Early exit avoids scanning the entire plane when there are already enough usable
    pixels to fit the requested number of parameters.
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
    Fit a multi-component Gaussian model to one concrete 2-D image plane.

    Parameters
    ----------
    z2d : np.ndarray
        Two-dimensional data plane with NumPy storage order ``[y, x]``.
    n_components : int
        Number of Gaussian components to fit.
    min_threshold : Number | None
        Inclusive lower data threshold. Pixels below this value are excluded.
    max_threshold : Number | None
        Inclusive upper data threshold. Pixels above this value are excluded.
    initial_guesses : np.ndarray | Sequence[dict] | dict | None
        Initial-guess specification accepted by :func:`_normalize_initial_guesses`.
    bounds : dict | None
        Optional parameter bounds accepted by :func:`_merge_bounds_multi`.
    max_nfev : int
        Maximum number of function evaluations passed to ``curve_fit``.
    x1d : np.ndarray | None, optional
        One-dimensional x-axis coordinates. ``None`` selects pixel-index fitting.
    y1d : np.ndarray | None, optional
        One-dimensional y-axis coordinates. ``None`` selects pixel-index fitting.
    mask2d : np.ndarray | None, optional
        Optional boolean keep-mask aligned with ``z2d``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(popt, perr, mask)`` where ``popt`` is the best-fit packed parameter
        vector, ``perr`` contains one-sigma parameter uncertainties, and ``mask`` is
        the final fit mask after combining user masking and thresholds.

    Notes
    -----
    Coordinates are modeled in x-before-y semantic order even though the input plane
    is stored in ``[y, x]`` order. The function validates that enough unmasked pixels
    remain to constrain the requested parameters before calling the optimizer.
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
    x1d: np.ndarray,
    y1d: np.ndarray,
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
    Execute one plane fit inside the vectorized ``xarray.apply_ufunc`` wrapper.

    Parameters
    ----------
    z2d : np.ndarray
        Two-dimensional data plane in ``[y, x]`` storage order.
    mask2d : np.ndarray
        Boolean keep-mask for ``z2d``.
    x1d : np.ndarray
        One-dimensional x-axis coordinates associated with the plane.
    y1d : np.ndarray
        One-dimensional y-axis coordinates associated with the plane.
    n_components : int
        Number of Gaussian components.
    min_threshold : Number | None
        Inclusive lower threshold.
    max_threshold : Number | None
        Inclusive upper threshold.
    initial_guesses : np.ndarray | Sequence[dict] | dict | None
        Initial-guess specification forwarded to :func:`_fit_multi_plane_numpy`.
    bounds : dict | None
        Optional parameter bounds.
    max_nfev : int
        Maximum function evaluations for the optimizer.
    return_model : bool
        If ``True``, include the full model plane in the returned tuple.
    return_residual : bool
        If ``True``, include the residual plane in the returned tuple.

    Returns
    -------
    tuple
        Flat tuple of component arrays, scalar diagnostics, and optional 2-D planes
        matching the ``output_core_dims`` contract used by ``xarray.apply_ufunc``.

    Notes
    -----
    Component-wise arrays are length ``n_components``; scalar diagnostics are shape
    ``()``; image outputs are shape ``(ny, nx)``. The return order is fixed by the
    public dataset-construction code below.
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
    """
    Normalize supported input data into an ``xarray.DataArray``.

    Parameters
    ----------
    data : ArrayOrDA
        Supported input object. Choices are ``numpy.ndarray``,
        ``dask.array.Array``, or ``xarray.DataArray``.

    Returns
    -------
    xr.DataArray
        DataArray view of the input. Raw arrays receive synthetic dimension names and
        zero-based numeric coordinates.

    Notes
    -----
    For non-DataArray inputs, dimensions are named ``dim_0``, ``dim_1``, ... and
    coordinates are pixel indices as floats.
    """
    if isinstance(data, xr.DataArray):
        return data
    if isinstance(data, (np.ndarray, da.Array)):
        dims = [f"dim_{i}" for i in range(data.ndim)]
        coords = {d: np.arange(s, dtype=float) for d, s in zip(dims, data.shape)}
        # Raw NumPy/Dask arrays arrive without semantic axis names. We wrap them
        # generically here; downstream dimension resolution applies the caller's
        # unlabeled_axis_order contract (or explicit dims= override) to decide
        # how those stored axes map onto semantic x/y fit-plane coordinates.
        return xr.DataArray(data, dims=dims, coords=coords, name="data")
    raise TypeError(
        "Unsupported input type; use numpy.ndarray, dask.array.Array, or xarray.DataArray."
    )


def _resolve_dims(
    da: xr.DataArray,
    dims: Optional[Sequence[Union[str, int]]],
    *,
    unlabeled_input: bool = False,
    unlabeled_axis_order: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Resolve the two dimensions that define the fit plane.

    Parameters
    ----------
    da : xr.DataArray
        Input data array.
    dims : Sequence[str | int] | None
        Optional dimension specifier. Supported choices are explicit dimension names
        or integer dimension indices.
    unlabeled_input : bool, optional
        Whether ``da`` originated from a raw NumPy/Dask input without semantic
        dimension names.
    unlabeled_axis_order : {"yx", "xy"} | None, optional
        Semantic interpretation to apply for unlabeled 2-D arrays when ``dims`` is
        omitted. ``"yx"`` means stored rows/columns order; ``"xy"`` means the first
        axis is interpreted semantically as x and the second as y. This must be
        provided explicitly whenever an unlabeled raw 2-D array would otherwise be
        ambiguous.

    Returns
    -------
    tuple[str, str]
        ``(x_dim, y_dim)`` names for the fit plane.

    Notes
    -----
    If ``dims`` is omitted, the helper prefers explicit ``("x", "y")`` dimensions.
    Otherwise, for plain 2-D arrays it assumes the last dimension is x and the
    second-last is y.
    """
    axis_order = None if unlabeled_axis_order is None else unlabeled_axis_order.lower()
    if axis_order not in (None, "yx", "xy"):
        raise ValueError("unlabeled_axis_order must be either 'yx' or 'xy'.")

    if dims is None:
        # ✅ new: prefer explicitly named axes when present
        if "x" in da.dims and "y" in da.dims:
            return "x", "y"
        if da.ndim == 2:
            if unlabeled_input and axis_order is None:
                raise ValueError(
                    "unlabeled_axis_order must be specified for unlabeled raw 2-D "
                    "arrays when dims is omitted."
                )
            if unlabeled_input and axis_order == "xy":
                # Allow callers with unlabeled arrays that already use semantic
                # (x, y) ordering to declare that contract explicitly.
                return da.dims[-2], da.dims[-1]
            # For unlabeled/raw 2-D arrays, interpret stored image-plane order as
            # (y, x): rows first, columns second. Public semantic fit-plane order
            # remains (x, y), so x maps to the last axis and y to the second-last.
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
    """
    Determine the orientation sign of a one-dimensional coordinate axis.

    Parameters
    ----------
    coord : np.ndarray | None
        Candidate coordinate axis.

    Returns
    -------
    float
        ``+1.0`` for ascending or indeterminate axes, ``-1.0`` for descending axes.

    Notes
    -----
    The helper intentionally falls back to ``+1.0`` for missing, non-1-D, too-short,
    or non-finite inputs so that handedness logic remains conservative.
    """
    if coord is None or coord.ndim != 1 or coord.size < 2:
        return 1.0
    c0, c1 = float(coord[0]), float(coord[1])
    return 1.0 if np.isfinite(c0) and np.isfinite(c1) and (c1 > c0) else -1.0


def _select_mask(da_tr: xr.DataArray, spec: str):
    """
    Resolve a string mask specification through the local selection helper.

    Parameters
    ----------
    da_tr : xr.DataArray
        Transposed data array used as the mask-selection context.
    spec : str
        Selection expression understood by ``selection.select_mask``.

    Returns
    -------
    Any
        Whatever object ``selection.select_mask`` returns. Downstream code normalizes
        this into a boolean ``DataArray``.

    Notes
    -----
    The wrapper exists so tests can monkeypatch a stable symbol in this module
    without importing the heavier selection machinery at module import time.
    """
    # local import to avoid hard module dependency at import time
    from .selection import select_mask  # type: ignore, pragma: no cover

    return select_mask(da_tr, spec)  # pragma: no cover


def _theta_pa_to_math(pa: np.ndarray) -> np.ndarray:
    """
    Convert position-angle values into the internal math-angle convention.

    Parameters
    ----------
    pa : np.ndarray
        Angle array in radians measured from +y toward +x.

    Returns
    -------
    np.ndarray
        Equivalent angles in radians measured from +x toward +y, wrapped into
        ``[0, 2*pi)``.

    Notes
    -----
    This conversion is purely geometric; it does not apply handedness flips itself.
    """
    theta_math = np.pi / 2 - pa
    return theta_math % (2.0 * np.pi)


def _theta_math_to_pa(theta_math: np.ndarray) -> np.ndarray:
    """
    Convert internal math-angle values into position-angle values.

    Parameters
    ----------
    theta_math : np.ndarray
        Angle array in radians measured from +x toward +y.

    Returns
    -------
    np.ndarray
        Equivalent PA angles in radians measured from +y toward +x and wrapped into
        ``[0, 2*pi)``.

    Notes
    -----
    This is the inverse mapping of :func:`_theta_pa_to_math`.
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
    Convert theta values in initial guesses from PA into math convention when needed.

    Parameters
    ----------
    init : np.ndarray | Sequence[dict] | dict | None
        Initial-guess structure accepted by :func:`fit_multi_gaussian2d`.
    to_math : bool
        If ``True``, convert any supplied ``theta`` values from PA to math angle.
        If ``False``, return ``init`` unchanged.
    sx : float
        Axis-orientation sign for x. Currently unused but retained as part of the
        helper contract for future handedness-aware refinements.
    sy : float
        Axis-orientation sign for y. Currently unused but retained as part of the
        helper contract for future handedness-aware refinements.
    n : int
        Expected number of components when array-form guesses are provided.

    Returns
    -------
    np.ndarray | Sequence[dict] | dict | None
        Copy of ``init`` with converted angles when conversion is requested.

    Notes
    -----
    The returned object preserves the original structural form as closely as possible.
    Unknown shapes or unsupported structures are returned unchanged.
    """
    if init is None or not to_math:
        return init

    def _conv_arr(arr: np.ndarray) -> np.ndarray:
        out = np.array(arr, dtype=float, copy=True)
        if out.shape == (6,) and n == 1:
            out = out.reshape(1, 6)
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
    Validate that a coordinate axis is usable for one-dimensional interpolation.

    Parameters
    ----------
    a : np.ndarray
        Candidate coordinate axis.

    Returns
    -------
    bool
        ``True`` when the axis is one-dimensional, finite, non-empty, and strictly
        monotonic.

    Notes
    -----
    Both ascending and descending axes are accepted because downstream interpolation
    code explicitly handles either ordering.
    """
    a = np.asarray(a)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        return False
    d = np.diff(a)
    return np.all(d > 0) or np.all(d < 0)


def _prepare_interp_pair(
    source_axis: np.ndarray,
    target_axis: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Normalize a monotonic interpolation pair so ``np.interp`` sees ascending ``xp``.

    Parameters
    ----------
    source_axis : np.ndarray
        Coordinate values to use as the ``xp`` argument to ``np.interp``.
    target_axis : np.ndarray
        Values paired elementwise with ``source_axis`` to use as the ``fp``
        argument after any required reversal.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        ``(xp, fp)`` suitable for ``np.interp`` when the input axis is strictly
        monotonic, or ``(None, None)`` when the axis is invalid.

    Notes
    -----
    ``np.interp`` requires ``xp`` to be strictly increasing. This helper accepts
    either ascending or descending physical axes and, for descending input, reverses
    both ``source_axis`` and ``target_axis`` together so the coordinate mapping is
    preserved while satisfying the interpolation precondition.
    """
    xp = np.asarray(source_axis, dtype=float)
    fp = np.asarray(target_axis, dtype=float)
    if (
        xp.ndim != 1
        or fp.ndim != 1
        or xp.size == 0
        or xp.size != fp.size
        or not np.all(np.isfinite(xp))
        or not np.all(np.isfinite(fp))
    ):
        return None, None
    if xp.size >= 2:
        d_xp = np.diff(xp)
        if not (np.all(d_xp > 0) or np.all(d_xp < 0)):
            return None, None
        if np.all(d_xp < 0):
            xp = xp[::-1]
            fp = fp[::-1]
    return xp, fp


def _extract_1d_coords_for_fit(
    original_input: ArrayOrDA,
    da_tr: xr.DataArray,
    coord_type: str,
    coords: Optional[Sequence[np.ndarray]],
    dim_y: str,
    dim_x: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resolve the one-dimensional coordinate vectors used for model evaluation.

    Parameters
    ----------
    original_input : ArrayOrDA
        Original object passed to :func:`fit_multi_gaussian2d`.
    da_tr : xr.DataArray
        DataArray after transposing the fit plane to the trailing ``(y, x)`` axes.
    coord_type : str
        Coordinate-mode selector for DataArray inputs. Supported choices are
        ``"world"`` and ``"pixel"``.
    coords : Sequence[np.ndarray] | None
        For NumPy/Dask inputs only, optional explicit coordinate vectors supplied as
        ``(x1d, y1d)``.
    dim_y : str
        Name of the y dimension in ``da_tr``.
    dim_x : str
        Name of the x dimension in ``da_tr``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple ``(x1d, y1d)`` of one-dimensional coordinate arrays.

    Notes
    -----
    DataArray inputs can use either pixel indices or their attached 1-D coordinate
    variables. NumPy/Dask inputs ignore ``coord_type`` and use ``coords`` when
    provided; otherwise they fall back to zero-based pixel indices. DataArray
    inputs requested in ``coord_type="world"`` mode also fall back to pixel-index
    coordinates when the fit dimensions do not have explicit 1-D coordinate
    variables attached.
    """
    ny, nx = int(da_tr.sizes[dim_y]), int(da_tr.sizes[dim_x])

    if isinstance(original_input, xr.DataArray):
        ctype = (coord_type or "world").lower()
        if ctype not in ("world", "pixel"):
            raise ValueError(
                "coord_type must be 'world' or 'pixel' for DataArray inputs"
            )
        if ctype == "pixel":
            return np.arange(nx, dtype=float), np.arange(ny, dtype=float)
        # world coords from the DataArray; if they are absent, fall back to
        # pixel indices just like raw NumPy/Dask inputs.
        if (dim_x not in da_tr.coords) or (dim_y not in da_tr.coords):
            return np.arange(nx, dtype=float), np.arange(ny, dtype=float)
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
        return x1d.astype(float), y1d.astype(float)

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
        return x1d, y1d

    # Fallback: pixel indices
    return np.arange(nx, dtype=float), np.arange(ny, dtype=float)


# ---------------------------------------------------------------------------
# Pixel→World interpolation helper (extracted for reliable coverage)
# ---------------------------------------------------------------------------
def _prepare_pixel_center_interp(
    coord: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare pixel-center and world-coordinate arrays for ``np.interp``.

    Parameters
    ----------
    coord : np.ndarray
        One-dimensional world-coordinate axis associated with zero-based pixel
        centers.

    Returns
    -------
    tuple[np.ndarray | None, np.ndarray | None]
        ``(idx, values)`` suitable for ``np.interp`` where ``idx`` is the
        ascending pixel-index axis and ``values`` are the paired world
        coordinates. Returns ``(None, None)`` when the coordinate axis is not a
        usable one-dimensional interpolation target.

    Notes
    -----
    This helper always constructs an ascending synthetic pixel-index axis and
    pairs it with the provided world-coordinate values. It relies on
    :func:`_prepare_interp_pair` for the shared finiteness, shape, and source-
    axis monotonicity checks needed before those arrays are passed to
    ``np.interp``.
    """
    idx = np.arange(np.asarray(coord).size, dtype=float)
    return _prepare_interp_pair(idx, coord)


def _interp_centers_world(
    ds: xr.Dataset,
    cx: np.ndarray,
    cy: np.ndarray,
    dim_x: str,
    dim_y: str,
) -> xr.Dataset:
    """
    Add world-coordinate center estimates by interpolating fitted pixel centers.

    Parameters
    ----------
    ds : xr.Dataset
        Result dataset containing pixel-center variables such as ``x0_pixel`` and
        ``y0_pixel``.
    cx : np.ndarray
        One-dimensional x-axis world coordinate values associated with pixel columns.
    cy : np.ndarray
        One-dimensional y-axis world coordinate values associated with pixel rows.
    dim_x : str
        Name of the x dimension for the original data plane.
    dim_y : str
        Name of the y dimension for the original data plane.

    Returns
    -------
    xr.Dataset
        Dataset with interpolated ``x0_world``, ``y0_world`` and corresponding error
        variables added when the coordinate axes are suitable.

    Notes
    -----
    Ascending and descending coordinate axes are both supported. Uncertainty
    propagation uses the local slope of each 1-D coordinate axis, estimated with
    ``np.gradient``.
    """

    idx_x, val_x = _prepare_pixel_center_interp(cx)
    idx_y, val_y = _prepare_pixel_center_interp(cy)
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
    """
    Attach a descriptive metadata note to the variance-explained result variable.

    Parameters
    ----------
    ds : xr.Dataset
        Result dataset containing a ``variance_explained`` variable.

    Returns
    -------
    xr.Dataset
        The same dataset with an expanded explanatory note stored in
        ``ds["variance_explained"].attrs["note"]``.

    Notes
    -----
    This helper only annotates metadata; it does not recompute the metric.
    """
    ds["variance_explained"].attrs["note"] = (
        "R²-style fit quality for each 2-D image plane (y×x). "
        "Measures how much of the pixel-to-pixel variance is explained by the fitted model.\n\n"
        "Definitions (per plane): let Z be the data, \\hat{Z} the fitted model, "
        "and R = Z - \\hat{Z} the residuals.\n\n"
        "Explained variance fraction:\n"
        "    EVF = 1 - \\mathrm{Var}(R) \\/ \\mathrm{Var}(Z)\n"
        "Equivalently:\n"
        "    EVF = 1 - \\sum (R - \\bar{R})^2 \\/ \\sum (Z - \\bar{Z})^2\n\n"
        "Range: values near 1.0 indicate an excellent fit, values near 0.0 indicate "
        "little improvement over the raw plane variance, and negative values mean "
        "the residual variance exceeds the data variance. If the denominator is near "
        "zero (nearly flat plane), the metric is reported as NaN.\n\n"
        "Quick gut-check scale (per plane):\n"
        "  ≥0.9 — excellent; model captures most structure\n"
        "  0.6–0.9 — usable but imperfect\n"
        "  0.2–0.6 — poor to fair\n"
        "  ≈0.0 — little reduction in pixel-to-pixel variance\n"
        "  <0.0 — residuals are more variable than the input plane\n\n"
        "For example, a value of 0.2 might indicate:\n"
        "  • Source shape not well modeled by the chosen number of 2-D Gaussians\n"
        "  • Centers/widths/angle off (bad seeds or tight bounds)\n"
        "  • Coordinates/scales mismatched (pixel vs world, anisotropic scaling)\n"
        "  • Additional structure present (neighbors, wings, gradients)\n\n"
        "For low values, some additional ideas:\n"
        "  • Inspect residuals — should look noise-like if the model is right\n"
        "  • Loosen/improve seeds or bounds; try adding a component\n"
        "  • Check whether the shared offset/background is appropriate\n"
        "  • Verify frame/scale (pixel vs world, local pixel size)\n"
        "  • If noise is high, a low value can be expected — consider SNR or weighting in the fit"
    )
    return ds


def _build_call(
    _inspect,
    _param,
):
    """
    Build a compact string representation of the user call for result metadata.

    Parameters
    ----------
    _inspect : module-like object
        Object expected to provide ``signature`` for ``fit_multi_gaussian2d``.
        Typically the standard-library ``inspect`` module.
    _param : dict
        Mapping of runtime parameter names to values.

    Returns
    -------
    str
        Best-effort call string that includes only arguments differing from their
        default values.

    Notes
    -----
    If introspection fails for any reason, the helper falls back to the generic
    string ``"fit_multi_gaussian2d(...)"``.
    """
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
                _pairs.append(f"{k}={_summarize_metadata_value(v)}")
            _call += ", ".join(_pairs)
        _call += ")"
    except Exception:
        _call = "fit_multi_gaussian2d(...)"
    return _call


def _add_angle_attrs(ds, conv, frame):
    """
    Add shared angle metadata to the published angle variables in a result dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Result dataset containing the theta variables produced by the fitter.
    conv : str
        Angle convention label. Supported public choices are ``"math"`` and ``"pa"``.
    frame : str
        Frame label requested by the caller. Current result variables use ``"pixel"``
        and ``"world"`` frame tags; this argument is retained for call-site symmetry.

    Returns
    -------
    xr.Dataset
        The same dataset with angle metadata populated.

    Notes
    -----
    The helper updates the published theta aliases plus the explicit
    ``*_math``/``*_pa`` variants in place.
    """
    for _name, _frame, _conv in (
        ("theta_pixel", "pixel", conv),
        ("theta_pixel_err", "pixel", conv),
        ("theta_pixel_math", "pixel", "math"),
        ("theta_pixel_math_err", "pixel", "math"),
        ("theta_pixel_pa", "pixel", "pa"),
        ("theta_pixel_pa_err", "pixel", "pa"),
        ("theta_world", "world", conv),
        ("theta_world_err", "world", conv),
        ("theta_world_math", "world", "math"),
        ("theta_world_math_err", "world", "math"),
        ("theta_world_pa", "world", "pa"),
        ("theta_world_pa_err", "world", "pa"),
    ):
        if _name in ds:
            ds[_name].attrs["convention"] = _conv
            ds[_name].attrs["frame"] = _frame
            ds[_name].attrs["units"] = "rad"
    return ds


@dataclass
class _FitExecutionContext:
    """
    Bundle the geometric and dimensional context for one public fit invocation.

    Parameters
    ----------
    da_in : xr.DataArray
        Normalized input as a DataArray before any transposition.
    da_tr : xr.DataArray
        DataArray with the fit plane moved to trailing ``(y, x)`` order.
    dim_x : str
        Name of the x dimension used by the fit plane.
    dim_y : str
        Name of the y dimension used by the fit plane.
    core : list[str]
        Core dimensions passed to ``xarray.apply_ufunc`` for the fit plane.
    coord_x : np.ndarray | None
        Attached x-axis coordinate values from ``da_tr`` when present.
    coord_y : np.ndarray | None
        Attached y-axis coordinate values from ``da_tr`` when present.
    sx_sign : float
        Orientation sign inferred from the x axis.
    sy_sign : float
        Orientation sign inferred from the y axis.
    is_left_handed : bool
        Whether the two axes form a left-handed image frame.

    Notes
    -----
    The public fitter repeatedly needs the same dimensional bookkeeping and axis
    handedness. Grouping those values in one structure keeps the orchestration
    logic short and avoids recomputing subtle frame information.
    """

    da_in: xr.DataArray
    da_tr: xr.DataArray
    dim_x: str
    dim_y: str
    core: List[str]
    coord_x: Optional[np.ndarray]
    coord_y: Optional[np.ndarray]
    sx_sign: float
    sy_sign: float
    is_left_handed: bool


def _build_fit_execution_context(
    data: ArrayOrDA,
    dims: Optional[Sequence[Union[str, int]]],
    unlabeled_axis_order: Optional[str] = None,
) -> _FitExecutionContext:
    """
    Normalize the public input into a stable fit-plane context.

    Parameters
    ----------
    data : ArrayOrDA
        Input accepted by :func:`fit_multi_gaussian2d`.
    dims : Sequence[str | int] | None
        Optional dimension specifier for the fit plane. Supported choices are
        explicit dimension names or integer dimension indices.
    unlabeled_axis_order : {"yx", "xy"} | None, optional
        Semantic axis-order contract for unlabeled NumPy/Dask inputs when
        ``dims`` is omitted. This is ignored for labeled ``xarray`` inputs.

    Returns
    -------
    _FitExecutionContext
        Context object containing the normalized DataArray, the transposed fit
        view, and handedness metadata derived from the fit axes.

    Notes
    -----
    The fitter operates internally on trailing ``(y, x)`` axes regardless of the
    original input ordering. Axis-orientation signs are derived from any attached
    1-D coordinate vectors and default conservatively when coordinates are absent.
    """
    unlabeled_input = not isinstance(data, xr.DataArray)
    da_in = _ensure_dataarray(data)
    dim_x, dim_y = _resolve_dims(
        da_in,
        dims,
        unlabeled_input=unlabeled_input,
        unlabeled_axis_order=unlabeled_axis_order,
    )
    # Normalize to trailing (y, x) because image planes are stored in row/column
    # memory order internally even though the public fit-plane API is (x, y).
    da_tr = da_in.transpose(
        *(d for d in da_in.dims if d not in (dim_y, dim_x)), dim_y, dim_x
    )
    coord_x = np.asarray(da_tr.coords[dim_x].values) if dim_x in da_tr.coords else None
    coord_y = np.asarray(da_tr.coords[dim_y].values) if dim_y in da_tr.coords else None
    sx_sign = _axis_sign(coord_x)
    sy_sign = _axis_sign(coord_y)
    return _FitExecutionContext(
        da_in=da_in,
        da_tr=da_tr,
        dim_x=dim_x,
        dim_y=dim_y,
        # Core gufunc dimensions follow the internal plane storage order (y, x).
        core=[dim_y, dim_x],
        coord_x=coord_x,
        coord_y=coord_y,
        sx_sign=sx_sign,
        sy_sign=sy_sign,
        is_left_handed=(sx_sign * sy_sign) < 0.0,
    )


def _warn_if_suboptimal_dask_chunking(context: _FitExecutionContext) -> None:
    """
    Warn when a Dask-backed input is chunked across the fit-plane dimensions.

    Parameters
    ----------
    context : _FitExecutionContext
        Prepared fit context for the current invocation.

    Notes
    -----
    Plane-wise fitting works best when each fit plane is available as a single
    chunk along the two core fit dimensions. Chunking across those axes can add
    scheduler overhead and interfere with the intended outer-dimension
    parallelization strategy.
    """
    chunksizes = getattr(context.da_in, "chunksizes", None)
    if not chunksizes:
        return

    split_fit_dims = [
        dim
        for dim in (context.dim_x, context.dim_y)
        if dim in chunksizes and len(chunksizes[dim]) > 1
    ]
    if not split_fit_dims:
        return

    suggested_chunks = {context.dim_x: -1, context.dim_y: -1}
    warnings.warn(
        "Dask-backed input is chunked across the fit-plane dimension(s) "
        f"{split_fit_dims!r}. This fitter generally performs best when each full "
        f"plane is contained in a single chunk along {context.dim_x!r} and "
        f"{context.dim_y!r}. Consider rechunking before the fit, for example "
        f"`data = data.chunk({suggested_chunks!r})`, while choosing outer-dimension "
        "chunks separately for stack-level parallelism.",
        RuntimeWarning,
        stacklevel=3,
    )


def _resolve_fit_mask(
    mask: Optional[ArrayOrDA],
    da_tr: xr.DataArray,
    dim_y: str,
    dim_x: str,
    *,
    raw_plane_dims: Optional[Sequence[str]] = None,
) -> xr.DataArray:
    """
    Normalize the public mask argument into a boolean DataArray aligned to the fit grid.

    Parameters
    ----------
    mask : ArrayOrDA | None
        Mask-like object accepted by :func:`fit_multi_gaussian2d`. Supported
        choices are ``None``, a NumPy array, a Dask array, an xarray.DataArray,
        or a CRTF/selection string.
    da_tr : xr.DataArray
        Transposed fit view whose trailing axes are ``(y, x)``.
    dim_y : str
        Name of the y dimension for the fit plane.
    dim_x : str
        Name of the x dimension for the fit plane.
    raw_plane_dims : Sequence[str] | None, optional
        Original public plane-dimension order for an accompanying unlabeled raw
        2-D mask. When provided, a plain 2-D mask is first labeled with this
        public x/y order so xarray can apply the same alignment/transposition that
        was already used to normalize the raw data into ``da_tr``.

    Returns
    -------
    xr.DataArray
        Boolean mask broadcast and aligned to ``da_tr``.

    Notes
    -----
    String masks are resolved through the image-selection helper. Plain arrays are
    wrapped with either the fit-plane dims or the full data dims depending on
    their rank before xarray alignment handles broadcasting. The fitter always
    consumes trailing ``(y, x)`` planes internally, but raw unlabeled masks must
    first be labeled in the same public plane order as the raw data so they
    undergo the same semantic x/y-to-row/column conversion during alignment.
    """
    if mask is None:
        return xr.ones_like(da_tr, dtype=bool)

    if isinstance(mask, str):
        mda = _select_mask(da_tr, mask)
        if isinstance(mda, xr.Dataset):
            if "mask" not in mda:
                raise TypeError(
                    "selection.select_mask returned a Dataset without a 'mask' variable"
                )
            mda = mda["mask"]
        if isinstance(mda, np.ndarray):
            # String masks are resolved against da_tr, which already uses the
            # fitter's internal trailing (y, x) storage order. If the selection
            # helper hands us back a bare ndarray, its axes therefore already
            # match row/column order and can be labeled directly as (y, x).
            mda = xr.DataArray(
                mda, dims=[dim_y, dim_x] if mda.ndim == 2 else da_tr.dims
            )
        if not isinstance(mda, xr.DataArray):
            raise TypeError("selection.select_mask returned unsupported mask type")
    elif isinstance(mask, xr.DataArray):
        mda = mask
    else:
        if getattr(mask, "ndim", None) == 2:
            if raw_plane_dims is not None:
                # Plain 2-D masks supplied alongside raw input data are assumed to
                # use the same public plane order as that raw data. Label them with
                # the original public x/y dim order here and let xarray alignment
                # apply the same public -> internal (y, x) transpose that created
                # da_tr from the raw image.
                mda = xr.DataArray(mask, dims=list(raw_plane_dims))
            else:
                # Without raw-plane context we only know about the internal fit
                # view, so fall back to treating the bare mask as already stored in
                # row/column order.
                mda = xr.DataArray(mask, dims=[dim_y, dim_x])
        else:
            mda = xr.DataArray(mask, dims=da_tr.dims)

    mda = mda.astype(bool)
    mda, _ = xr.align(mda, da_tr, join="right")
    return mda


def _prepare_fit_configuration(
    initial_guesses: Optional[
        Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]
    ],
    bounds: Optional[
        Dict[str, Union[Tuple[float, float], Sequence[Tuple[float, float]]]]
    ],
    initial_is_fwhm: bool,
    angle: str,
    n_components: int,
    sx_sign: float,
    sy_sign: float,
    is_left_handed: bool,
) -> tuple[
    Optional[Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]],
    Optional[Dict[str, Any]],
    bool,
]:
    """
    Prepare user-supplied configuration into the optimizer's native representation.

    Parameters
    ----------
    initial_guesses : np.ndarray | Sequence[dict] | dict | None
        Initial-guess specification accepted by :func:`fit_multi_gaussian2d`.
    bounds : dict | None
        Bounds mapping accepted by :func:`fit_multi_gaussian2d`.
    initial_is_fwhm : bool
        Whether array-form width guesses should be interpreted as FWHM and converted
        to sigma before optimization. Single-component flat ``(6,)`` array-like
        guesses are normalized to ``(1, 6)`` before this conversion so they follow
        the same width-handling path as explicit two-dimensional component arrays.
    angle : str
        Public angle convention selector. Supported choices are ``"math"``,
        ``"pa"``, and ``"auto"``. Matching is case-insensitive.
    n_components : int
        Number of Gaussian components.
    sx_sign : float
        Orientation sign inferred from the x axis.
    sy_sign : float
        Orientation sign inferred from the y axis.
    is_left_handed : bool
        Whether the fit plane is left-handed.

    Returns
    -------
    tuple
        ``(init_for_fit, bounds_for_fit, want_pa)`` where ``init_for_fit`` and
        ``bounds_for_fit`` are ready for the low-level fitter and ``want_pa``
        records whether public angle reporting should use the PA convention.

    Notes
    -----
    Width parameters are always converted into sigma units before reaching the
    optimizer. Angle guesses are converted into the internal math convention when
    the public API is operating in PA mode. Public ``fwhm_major`` and
    ``fwhm_minor`` bounds must be supplied together and must define an ordered
    principal-axis interval per component, while public ``theta`` bounds are
    rejected because the optimizer does not parameterize the major-axis angle
    directly.
    """
    angle_normalized = str(angle).lower()
    if angle_normalized not in {"math", "pa", "auto"}:
        raise ValueError(
            f"Unsupported angle value {angle!r}. Supported values are "
            "{'math', 'pa', 'auto'}."
        )

    def _convert_array_like_widths_to_sigma(
        init_like: Union[np.ndarray, Sequence[Number]],
    ) -> Union[np.ndarray, Sequence[Number]]:
        """
        Convert array-like FWHM width guesses into sigma units for the optimizer.

        Parameters
        ----------
        init_like : np.ndarray | Sequence[Number]
            Array-like initial-guess payload. Supported forms are an explicit
            ``(n_components, 6)`` array or, when ``n_components == 1``, a flat
            length-6 array-like value describing one component.

        Returns
        -------
        np.ndarray | Sequence[Number]
            Converted ``numpy`` array when the payload matches a supported packed
            component layout, otherwise the original object unchanged.

        Notes
        -----
        This helper intentionally mirrors the flat-single-component normalization
        accepted later by :func:`_normalize_initial_guesses` so public FWHM guesses
        are converted consistently before the optimizer sees them.
        """
        arr = np.asarray(init_like, dtype=float)
        if arr.shape == (6,) and int(n_components) == 1:
            arr = arr.reshape(1, 6)
        if arr.shape != (int(n_components), 6):
            return init_like
        arr = arr.copy()
        arr[:, 3] = _sigma_from_fwhm(arr[:, 3])
        arr[:, 4] = _sigma_from_fwhm(arr[:, 4])
        return arr

    ig = initial_guesses
    if ig is not None and initial_is_fwhm:
        if isinstance(ig, np.ndarray):
            ig = _convert_array_like_widths_to_sigma(ig)
        elif (
            isinstance(ig, (list, tuple))
            and len(ig) > 0
            and not isinstance(ig[0], dict)
        ):
            ig = _convert_array_like_widths_to_sigma(ig)
    if isinstance(ig, dict) and "components" in ig:
        comps = ig["components"]
        if initial_is_fwhm and (
            isinstance(comps, np.ndarray)
            or (
                isinstance(comps, (list, tuple))
                and len(comps) > 0
                and not isinstance(comps[0], dict)
            )
        ):
            arr = _convert_array_like_widths_to_sigma(comps)
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

    want_pa = (angle_normalized == "pa") or (
        angle_normalized == "auto" and is_left_handed
    )
    init_for_fit = _convert_init_theta(
        ig,
        to_math=want_pa,
        sx=sx_sign,
        sy=sy_sign,
        n=int(n_components),
    )

    bnds: Optional[Dict[str, Any]] = bounds
    if bounds is not None:
        has_major = "fwhm_major" in bounds
        has_minor = "fwhm_minor" in bounds
        if has_major != has_minor:
            raise ValueError(
                "bounds for 'fwhm_major' and 'fwhm_minor' must be provided together."
            )
        if "theta" in bounds:
            raise ValueError(
                "public bounds for 'theta' are not supported with the current width parameterization."
            )
        if has_major and has_minor:
            major_val = bounds["fwhm_major"]
            minor_val = bounds["fwhm_minor"]

            def _expand_bound_pairs(value: Any) -> List[Tuple[float, float]]:
                if (
                    isinstance(value, (list, tuple))
                    and value
                    and isinstance(value[0], (list, tuple))
                ):
                    if len(value) != int(n_components):
                        raise ValueError(
                            "paired 'fwhm_major'/'fwhm_minor' bounds must each have length "
                            f"n={int(n_components)} when specified per component."
                        )
                    return [(float(lo), float(hi)) for (lo, hi) in value]
                lo, hi = value  # type: ignore[misc]
                return [(float(lo), float(hi))] * int(n_components)

            major_pairs = _expand_bound_pairs(major_val)
            minor_pairs = _expand_bound_pairs(minor_val)
            for i, ((major_lo, major_hi), (minor_lo, minor_hi)) in enumerate(
                zip(major_pairs, minor_pairs)
            ):
                if major_lo < minor_lo or major_hi < minor_hi:
                    raise ValueError(
                        "paired principal-axis bounds must satisfy "
                        f"fwhm_major_lo >= fwhm_minor_lo and fwhm_major_hi >= fwhm_minor_hi "
                        f"for each component; component {i} received "
                        f"major=({major_lo}, {major_hi}) and minor=({minor_lo}, {minor_hi})."
                    )
        conv = _FWHM2SIG
        converted: Dict[str, Any] = {}
        for key, value in bounds.items():
            if key in ("fwhm_major", "fwhm_minor"):
                target = "sigma_x" if key == "fwhm_major" else "sigma_y"
                if (
                    isinstance(value, (list, tuple))
                    and value
                    and isinstance(value[0], (list, tuple))
                ):
                    converted[target] = [
                        (float(lo) * conv, float(hi) * conv) for (lo, hi) in value
                    ]
                else:
                    lo, hi = value  # type: ignore[misc]
                    converted[target] = (float(lo) * conv, float(hi) * conv)
            else:
                converted[key] = value
        bnds = converted

    return init_for_fit, bnds, want_pa


def _resolve_fit_coordinate_axes(
    original_input: ArrayOrDA,
    context: _FitExecutionContext,
    coord_type: str,
    coords: Optional[Sequence[np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Resolve the coordinate vectors that define the optimizer's working frame.

    Parameters
    ----------
    original_input : ArrayOrDA
        Original public input passed to :func:`fit_multi_gaussian2d`.
    context : _FitExecutionContext
        Prepared fit context containing the transposed fit plane and dim names.
    coord_type : str
        DataArray coordinate selection mode. Supported choices are ``"world"``
        and ``"pixel"``.
    coords : Sequence[np.ndarray] | None
        Optional ``(x1d, y1d)`` coordinate vectors for NumPy/Dask inputs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, bool]
        ``(x1d, y1d, world_mode)`` where ``world_mode`` indicates whether the
        optimizer is operating on nontrivial world-coordinate axes.

    Notes
    -----
    ``world_mode`` is derived numerically from the resolved axes rather than from
    the public arguments so that explicit pixel-index coordinate arrays are treated
    consistently even when provided through the world-coordinate pathways.
    """
    x1d, y1d = _extract_1d_coords_for_fit(
        original_input,
        context.da_tr,
        coord_type,
        coords,
        context.dim_y,
        context.dim_x,
    )
    return x1d, y1d, not _is_pixel_index_axes(x1d, y1d)


def _run_fit_apply_ufunc(
    context: _FitExecutionContext,
    mask_da: xr.DataArray,
    x1d: np.ndarray,
    y1d: np.ndarray,
    n_components: int,
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    init_for_fit: Optional[
        Union[np.ndarray, Sequence[Dict[str, Number]], Dict[str, Any]]
    ],
    bounds_for_fit: Optional[Dict[str, Any]],
    max_nfev: int,
    return_model: bool,
    return_residual: bool,
):
    """
    Execute the vectorized plane-by-plane Gaussian fit on the prepared coordinate grid.

    Parameters
    ----------
    context : _FitExecutionContext
        Prepared fit context containing dims and the transposed fit plane.
    mask_da : xr.DataArray
        Boolean mask aligned to ``context.da_tr``.
    x1d : np.ndarray
        One-dimensional x-axis coordinates used by the optimizer.
    y1d : np.ndarray
        One-dimensional y-axis coordinates used by the optimizer.
    n_components : int
        Number of Gaussian components.
    min_threshold : Number | None
        Inclusive lower data threshold.
    max_threshold : Number | None
        Inclusive upper data threshold.
    init_for_fit : np.ndarray | Sequence[dict] | dict | None
        Initial guesses already converted into optimizer-native units.
    bounds_for_fit : dict | None
        Bounds already converted into optimizer-native units.
    max_nfev : int
        Maximum optimizer evaluations.
    return_model : bool
        Whether to request model planes from the low-level fitter.
    return_residual : bool
        Whether to request residual planes from the low-level fitter.

    Returns
    -------
    tuple
        Raw tuple of xarray objects returned by the low-level plane-fitting gufunc.

    Notes
    -----
    This helper isolates the ``xarray.apply_ufunc`` signature and output-layout
    bookkeeping from the public API so that the orchestration code reads as a
    sequence of conceptual steps instead of a large gufunc configuration block.
    """
    x1d_da = xr.DataArray(x1d, dims=[context.dim_x])
    y1d_da = xr.DataArray(y1d, dims=[context.dim_y])
    out_dtypes = (
        [np.float64] * 6
        + [np.float64] * 6
        + [np.float64] * 4
        + [np.float64, np.float64]
        + [np.bool_, np.float64]
        + [np.float64, np.float64]
    )
    out_core_dims = (
        [["component"]] * 6
        + [["component"]] * 6
        + [["component"]] * 4
        + [[], []]
        + [[], []]
        # Optional image-plane outputs stay in the internal stored plane order
        # (y, x) here and are transposed back to public (x, y) later.
        + [[context.dim_y, context.dim_x], [context.dim_y, context.dim_x]]
    )
    return xr.apply_ufunc(
        _multi_fit_plane_wrapper,
        context.da_tr,
        mask_da,
        x1d_da,
        y1d_da,
        input_core_dims=[context.core, context.core, [context.dim_x], [context.dim_y]],
        output_core_dims=out_core_dims,
        vectorize=True,
        dask="parallelized",
        output_dtypes=out_dtypes,
        dask_gufunc_kwargs={"output_sizes": {"component": int(n_components)}},
        kwargs=dict(
            n_components=int(n_components),
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            initial_guesses=init_for_fit,
            bounds=bounds_for_fit,
            max_nfev=int(max_nfev),
            return_model=bool(return_model),
            return_residual=bool(return_residual),
        ),
    )


def _canonicalize_fit_outputs(results, want_pa: bool) -> Dict[str, xr.DataArray]:
    """
    Canonicalize raw fitter outputs into a consistent major-axis representation.

    Parameters
    ----------
    results : tuple
        Raw tuple returned by the vectorized plane-fitting helper.
    want_pa : bool
        Whether the public-facing angle aliases should use the PA convention.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping containing canonicalized centers, widths, angles, uncertainties,
        diagnostic scalars, and optional image-plane outputs.

    Notes
    -----
    The low-level fitter returns widths in sigma units and angles in the internal
    sign convention. This helper enforces the public invariant that the reported
    angle always describes the major axis and that the major/minor widths are
    ordered consistently.
    """
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

    th_math = -th
    is_major_sx = xr.apply_ufunc(np.greater_equal, sx, sy, dask="parallelized")
    sx, sy = xr.where(is_major_sx, sx, sy), xr.where(is_major_sx, sy, sx)
    sx_e, sy_e = xr.where(is_major_sx, sx_e, sy_e), xr.where(is_major_sx, sy_e, sx_e)
    th_math = xr.where(is_major_sx, th_math, th_math + np.pi / 2)

    def _wrap_halfpi(t):
        return ((t + np.pi / 2) % np.pi) - np.pi / 2

    th_math = xr.apply_ufunc(_wrap_halfpi, th_math, dask="parallelized", vectorize=True)
    theta_math = th_math
    theta_pa = xr.apply_ufunc(
        _theta_math_to_pa,
        th_math,
        input_core_dims=[["component"]],
        output_core_dims=[["component"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    theta_math_err = th_e
    theta_pa_err = th_e
    th_public = theta_pa if want_pa else theta_math
    th_public_err = theta_pa_err if want_pa else theta_math_err

    return {
        "amp": amp,
        "x0": x0,
        "y0": y0,
        "sx": sx,
        "sy": sy,
        "amp_e": amp_e,
        "x0_e": x0_e,
        "y0_e": y0_e,
        "sx_e": sx_e,
        "sy_e": sy_e,
        "th_e": th_e,
        "fwhm_maj": fwhm_maj,
        "fwhm_min": fwhm_min,
        "peak": peak,
        "peak_err": peak_err,
        "offset": offset,
        "offset_e": offset_e,
        "success": success,
        "varexp": varexp,
        "residual": residual,
        "model": model,
        "theta_math": theta_math,
        "theta_pa": theta_pa,
        "theta_math_err": theta_math_err,
        "theta_pa_err": theta_pa_err,
        "th_public": th_public,
        "th_public_err": th_public_err,
        "th_math": th_math,
    }


def _build_published_parameter_dataset(
    fit: Dict[str, xr.DataArray],
    context: _FitExecutionContext,
    x1d: np.ndarray,
    y1d: np.ndarray,
    world_mode: bool,
    want_pa: bool,
) -> xr.Dataset:
    """
    Build the published result dataset in both pixel and world parameter views.

    Parameters
    ----------
    fit : dict[str, xr.DataArray]
        Canonicalized fit outputs produced by :func:`_canonicalize_fit_outputs`.
    context : _FitExecutionContext
        Prepared fit context containing dim names and axis metadata.
    x1d : np.ndarray
        One-dimensional x-axis coordinates used by the optimizer.
    y1d : np.ndarray
        One-dimensional y-axis coordinates used by the optimizer.
    world_mode : bool
        Whether the optimizer ran natively in world coordinates.
    want_pa : bool
        Whether the public-facing angle aliases should use the PA convention.

    Returns
    -------
    xr.Dataset
        Dataset containing mirrored pixel-space and world-space component
        parameters together with fit diagnostics.

    Notes
    -----
    This helper centralizes the most mathematically dense part of the public API:
    converting native fit outputs into the alternate coordinate frame while
    propagating center, width, and angle uncertainties through local axis scales.
    World-frame widths and angles are published only when the input supplied
    usable non-pixel world-coordinate axes, or when the optimizer ran natively
    in world coordinates from explicit raw-array coordinate vectors.
    """
    x0 = fit["x0"]
    y0 = fit["y0"]
    sx = fit["sx"]
    sy = fit["sy"]
    x0_e = fit["x0_e"]
    y0_e = fit["y0_e"]
    sx_e = fit["sx_e"]
    sy_e = fit["sy_e"]
    th_math = fit["th_math"]
    theta_math = fit["theta_math"]
    theta_pa = fit["theta_pa"]
    theta_math_err = fit["theta_math_err"]
    theta_pa_err = fit["theta_pa_err"]
    th_public = fit["th_public"]
    th_public_err = fit["th_public_err"]
    coord_x = (
        None if context.coord_x is None else np.asarray(context.coord_x, dtype=float)
    )
    coord_y = (
        None if context.coord_y is None else np.asarray(context.coord_y, dtype=float)
    )
    attached_world_axes = (
        coord_x is not None
        and coord_y is not None
        and _axis_is_valid(coord_x)
        and _axis_is_valid(coord_y)
        and not _is_pixel_index_axes(coord_x, coord_y)
    )
    publish_world = bool(world_mode or attached_world_axes)

    if world_mode:
        nx = x1d.shape[0]
        ny = y1d.shape[0]
        x_idx_axis = np.arange(nx, dtype=float)
        y_idx_axis = np.arange(ny, dtype=float)
        # Convert fitted world-coordinate centers back onto pixel-index axes. The
        # physical world axes may ascend or descend, but np.interp requires an
        # increasing source axis, so prepare both axis pairs through the shared
        # monotonic helper before vectorized interpolation.
        x_interp_src, x_interp_dst = _prepare_interp_pair(x1d, x_idx_axis)
        y_interp_src, y_interp_dst = _prepare_interp_pair(y1d, y_idx_axis)
        if x_interp_src is None or y_interp_src is None:
            raise ValueError(
                "world_mode requires strictly monotonic finite coordinate axes "
                "for world-to-pixel center interpolation."
            )
        x0_pixel = xr.apply_ufunc(
            np.interp,
            x0,
            xr.DataArray(x_interp_src, dims=[context.dim_x]),
            xr.DataArray(x_interp_dst, dims=[context.dim_x]),
            input_core_dims=[["component"], [context.dim_x], [context.dim_x]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        y0_pixel = xr.apply_ufunc(
            np.interp,
            y0,
            xr.DataArray(y_interp_src, dims=[context.dim_y]),
            xr.DataArray(y_interp_dst, dims=[context.dim_y]),
            input_core_dims=[["component"], [context.dim_y], [context.dim_y]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
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
        dx_local = xr.apply_ufunc(
            np.take,
            xr.DataArray(gx, dims=[context.dim_x]),
            x0i,
            input_core_dims=[[context.dim_x], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        dy_local = xr.apply_ufunc(
            np.take,
            xr.DataArray(gy, dims=[context.dim_y]),
            y0i,
            input_core_dims=[[context.dim_y], ["component"]],
            output_core_dims=[["component"]],
            kwargs={"axis": 0, "mode": "clip"},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        def _world_to_pixel_cov(sigx, sigy, theta, dx, dy):
            c = np.cos(theta)
            s = np.sin(theta)
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
        theta_pixel_pa = xr.apply_ufunc(
            _theta_math_to_pa,
            theta_pixel_math,
            input_core_dims=[["component"]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        theta_pixel = theta_pixel_pa if want_pa else theta_pixel_math
        fwhm_major_pixel = sigma_major_pixel * _SIG2FWHM
        fwhm_minor_pixel = sigma_minor_pixel * _SIG2FWHM
        if publish_world:
            theta_world = th_public
            theta_world_err = th_public_err
            theta_world_math = theta_math
            theta_world_pa = theta_pa
            theta_world_math_err = theta_math_err
            theta_world_pa_err = theta_pa_err

        def _fd_sigma_world_to_pixel(sigx, sigy, theta, dx, dy, epsx, epsy):
            s1M, s1m, _ = _world_to_pixel_cov(sigx + epsx, sigy, theta, dx, dy)
            s2M, s2m, _ = _world_to_pixel_cov(sigx - epsx, sigy, theta, dx, dy)
            dM_dsx = (s1M - s2M) / (2.0 * epsx)
            dm_dsx = (s1m - s2m) / (2.0 * epsx)
            s1M, s1m, _ = _world_to_pixel_cov(sigx, sigy + epsy, theta, dx, dy)
            s2M, s2m, _ = _world_to_pixel_cov(sigx, sigy - epsy, theta, dx, dy)
            dM_dsy = (s1M - s2M) / (2.0 * epsy)
            dm_dsy = (s1m - s2m) / (2.0 * epsy)
            return dM_dsx, dm_dsx, dM_dsy, dm_dsy

        epsx = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), sx, dask="parallelized"
        )
        epsy = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), sy, dask="parallelized"
        )
        dM_dsx, dm_dsx, dM_dsy, dm_dsy = xr.apply_ufunc(
            _fd_sigma_world_to_pixel,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            epsx,
            epsy,
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
            input_core_dims=[["component"]] * 4,
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
            input_core_dims=[["component"]] * 4,
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        def _dtheta_pix_dtheta_world(sigx, sigy, theta, dx, dy, eps):
            _, _, t1 = _world_to_pixel_cov(sigx, sigy, theta + eps, dx, dy)
            _, _, t2 = _world_to_pixel_cov(sigx, sigy, theta - eps, dx, dy)
            return (t1 - t2) / (2.0 * eps)

        epst = xr.apply_ufunc(
            lambda v: 1e-6 + 1e-3 * np.abs(v), th_math, dask="parallelized"
        )
        dtdt = xr.apply_ufunc(
            _dtheta_pix_dtheta_world,
            sx,
            sy,
            th_math,
            dx_local,
            dy_local,
            epst,
            input_core_dims=[["component"]] * 6,
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        theta_pixel_math_err = theta_pixel_err = xr.apply_ufunc(
            np.multiply,
            xr.apply_ufunc(np.abs, dtdt, dask="parallelized"),
            fit["th_e"],
            input_core_dims=[["component"], ["component"]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        theta_pixel_pa_err = theta_pixel_math_err
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
        if publish_world:
            sigma_major_world = xr.apply_ufunc(np.maximum, sx, sy, dask="parallelized")
            sigma_minor_world = xr.apply_ufunc(np.minimum, sx, sy, dask="parallelized")
            is_sx_major_w = xr.apply_ufunc(
                np.greater_equal, sx, sy, dask="parallelized"
            )
            sigma_major_world_err = xr.where(is_sx_major_w, sx_e, sy_e)
            sigma_minor_world_err = xr.where(is_sx_major_w, sy_e, sx_e)
            fwhm_major_world = sigma_major_world * _SIG2FWHM
            fwhm_minor_world = sigma_minor_world * _SIG2FWHM
            fwhm_major_world_err = sigma_major_world_err * _SIG2FWHM
            fwhm_minor_world_err = sigma_minor_world_err * _SIG2FWHM
        fwhm_major_pixel_err = sigma_major_pixel_err * _SIG2FWHM
        fwhm_minor_pixel_err = sigma_minor_pixel_err * _SIG2FWHM
    else:
        x0_pixel, y0_pixel = x0, y0
        sigma_major_pixel = xr.apply_ufunc(np.maximum, sx, sy, dask="parallelized")
        sigma_minor_pixel = xr.apply_ufunc(np.minimum, sx, sy, dask="parallelized")
        theta_pixel_math = th_math
        theta_pixel_pa = xr.apply_ufunc(
            _theta_math_to_pa,
            th_math,
            input_core_dims=[["component"]],
            output_core_dims=[["component"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )
        theta_pixel = theta_pixel_pa if want_pa else theta_pixel_math
        fwhm_major_pixel = xr.apply_ufunc(
            np.maximum, sx * _SIG2FWHM, sy * _SIG2FWHM, dask="parallelized"
        )
        fwhm_minor_pixel = xr.apply_ufunc(
            np.minimum, sx * _SIG2FWHM, sy * _SIG2FWHM, dask="parallelized"
        )
        theta_pixel_math_err = theta_math_err
        theta_pixel_pa_err = theta_pa_err
        theta_pixel_err = th_public_err
        x0_pixel_err, y0_pixel_err = x0_e, y0_e
        is_sx_major_p = xr.apply_ufunc(np.greater_equal, sx, sy, dask="parallelized")
        sigma_major_pixel_err = xr.where(is_sx_major_p, sx_e, sy_e)
        sigma_minor_pixel_err = xr.where(is_sx_major_p, sy_e, sx_e)
        fwhm_major_pixel_err = sigma_major_pixel_err * _SIG2FWHM
        fwhm_minor_pixel_err = sigma_minor_pixel_err * _SIG2FWHM

        def _pixel_to_world_cov(sigx, sigy, theta, dx, dy):
            c = np.cos(theta)
            s = np.sin(theta)
            Sxx = (c * c) * sigx * sigx + (s * s) * sigy * sigy
            Sxy = (s * c) * (sigx * sigx - sigy * sigy)
            Syy = (s * s) * sigx * sigx + (c * c) * sigy * sigy
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

        def _dtheta_world_dtheta_pixel(sigx, sigy, theta, dx, dy, eps):
            _, _, t1 = _pixel_to_world_cov(sigx, sigy, theta + eps, dx, dy)
            _, _, t2 = _pixel_to_world_cov(sigx, sigy, theta - eps, dx, dy)
            return (t1 - t2) / (2.0 * eps)

        if publish_world:
            gx = np.gradient(coord_x.astype(float))
            gy = np.gradient(coord_y.astype(float))
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
                .clip(0, coord_x.shape[0] - 1)
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
                .clip(0, coord_y.shape[0] - 1)
                .astype(int)
            )
            dx_local = xr.apply_ufunc(
                np.take,
                xr.DataArray(gx, dims=[context.dim_x]),
                x0i,
                input_core_dims=[[context.dim_x], ["component"]],
                output_core_dims=[["component"]],
                kwargs={"axis": 0, "mode": "clip"},
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            dy_local = xr.apply_ufunc(
                np.take,
                xr.DataArray(gy, dims=[context.dim_y]),
                y0i,
                input_core_dims=[[context.dim_y], ["component"]],
                output_core_dims=[["component"]],
                kwargs={"axis": 0, "mode": "clip"},
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            sigma_major_world, sigma_minor_world, theta_world_math = xr.apply_ufunc(
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
            theta_world_pa = xr.apply_ufunc(
                _theta_math_to_pa,
                theta_world_math,
                input_core_dims=[["component"]],
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            theta_world = theta_world_pa if want_pa else theta_world_math
            epst_w = xr.apply_ufunc(
                lambda v: 1e-6 + 1e-3 * np.abs(v), th_math, dask="parallelized"
            )
            dtdt_w = xr.apply_ufunc(
                _dtheta_world_dtheta_pixel,
                sx,
                sy,
                th_math,
                dx_local,
                dy_local,
                epst_w,
                input_core_dims=[["component"]] * 6,
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            theta_world_math_err = xr.apply_ufunc(
                np.multiply,
                xr.apply_ufunc(np.abs, dtdt_w, dask="parallelized"),
                fit["th_e"],
                input_core_dims=[["component"], ["component"]],
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            theta_world_pa_err = theta_world_math_err
            theta_world_err = theta_world_pa_err if want_pa else theta_world_math_err
            epsx = xr.apply_ufunc(
                lambda v: 1e-6 + 1e-3 * np.abs(v), sx, dask="parallelized"
            )
            epsy = xr.apply_ufunc(
                lambda v: 1e-6 + 1e-3 * np.abs(v), sy, dask="parallelized"
            )
            dM_dsx, dm_dsx, dM_dsy, dm_dsy = xr.apply_ufunc(
                _fd_sigma_pixel_to_world,
                sx,
                sy,
                th_math,
                dx_local,
                dy_local,
                epsx,
                epsy,
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
                input_core_dims=[["component"]] * 4,
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
                input_core_dims=[["component"]] * 4,
                output_core_dims=[["component"]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
            fwhm_major_world = sigma_major_world * _SIG2FWHM
            fwhm_minor_world = sigma_minor_world * _SIG2FWHM
            fwhm_major_world_err = sigma_major_world_err * _SIG2FWHM
            fwhm_minor_world_err = sigma_minor_world_err * _SIG2FWHM

    data_vars = dict(
        amplitude=fit["amp"],
        amplitude_err=fit["amp_e"],
        peak=fit["peak"],
        peak_err=fit["peak_err"],
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
        theta_pixel_math=theta_pixel_math,
        theta_pixel_math_err=theta_pixel_math_err,
        theta_pixel_pa=theta_pixel_pa,
        theta_pixel_pa_err=theta_pixel_pa_err,
        fwhm_major_pixel=fwhm_major_pixel,
        fwhm_minor_pixel=fwhm_minor_pixel,
        fwhm_major_pixel_err=fwhm_major_pixel_err,
        fwhm_minor_pixel_err=fwhm_minor_pixel_err,
        offset=fit["offset"],
        offset_err=fit["offset_e"],
        success=fit["success"],
        variance_explained=fit["varexp"],
    )
    if publish_world:
        data_vars.update(
            dict(
                theta_world=theta_world,
                theta_world_err=theta_world_err,
                theta_world_math=theta_world_math,
                theta_world_math_err=theta_world_math_err,
                theta_world_pa=theta_world_pa,
                theta_world_pa_err=theta_world_pa_err,
                sigma_major_world=sigma_major_world,
                sigma_minor_world=sigma_minor_world,
                sigma_major_world_err=sigma_major_world_err,
                sigma_minor_world_err=sigma_minor_world_err,
                fwhm_major_world=fwhm_major_world,
                fwhm_minor_world=fwhm_minor_world,
                fwhm_major_world_err=fwhm_major_world_err,
                fwhm_minor_world_err=fwhm_minor_world_err,
            )
        )
    ds = xr.Dataset(data_vars=data_vars)
    conv = "pa" if want_pa else "math"
    ds.attrs["axes_handedness"] = "left" if context.is_left_handed else "right"
    ds.attrs["theta_convention"] = conv
    return _add_angle_attrs(ds, conv, "pixel")


def _attach_optional_plane_outputs(
    ds: xr.Dataset,
    fit: Dict[str, xr.DataArray],
    context: _FitExecutionContext,
    return_residual: bool,
    return_model: bool,
) -> xr.Dataset:
    """
    Attach optional model and residual planes using the public axis ordering.

    Parameters
    ----------
    ds : xr.Dataset
        Result dataset under construction.
    fit : dict[str, xr.DataArray]
        Canonicalized fit outputs containing optional plane data.
    context : _FitExecutionContext
        Prepared fit context containing public x/y dimension names.
    return_residual : bool
        Whether residual planes were requested by the caller.
    return_model : bool
        Whether model planes were requested by the caller.

    Returns
    -------
    xr.Dataset
        Dataset with optional image-plane outputs attached.

    Notes
    -----
    The low-level fitter works on trailing ``(y, x)`` axes. Public result planes are
    transposed back to ``(..., x, y)`` so their named dimensions remain consistent
    with the rest of the dataset.
    """
    if return_residual:
        ds["residual"] = fit["residual"].transpose(
            *(
                d
                for d in fit["residual"].dims
                if d not in (context.dim_y, context.dim_x)
            ),
            context.dim_x,
            context.dim_y,
        )
    if return_model:
        ds["model"] = fit["model"].transpose(
            *(d for d in fit["model"].dims if d not in (context.dim_y, context.dim_x)),
            context.dim_x,
            context.dim_y,
        )
    return ds


def _attach_world_center_outputs(
    ds: xr.Dataset,
    fit: Dict[str, xr.DataArray],
    context: _FitExecutionContext,
    world_mode: bool,
) -> xr.Dataset:
    """
    Publish component-center coordinates in world units when axis metadata permit it.

    Parameters
    ----------
    ds : xr.Dataset
        Result dataset under construction.
    fit : dict[str, xr.DataArray]
        Canonicalized fit outputs containing native center parameters.
    context : _FitExecutionContext
        Prepared fit context with any attached coordinate axes.
    world_mode : bool
        Whether the optimizer ran natively in world coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with ``x0_world``/``y0_world`` variables attached when possible.

    Notes
    -----
    If the fit was already performed in world coordinates, the world-center outputs
    are direct aliases of the native fit parameters. Otherwise they are interpolated
    from the pixel centers using attached non-pixel world-coordinate axis vectors.
    """
    attached_world_axes = (
        context.coord_x is not None
        and context.coord_y is not None
        and _axis_is_valid(np.asarray(context.coord_x, dtype=float))
        and _axis_is_valid(np.asarray(context.coord_y, dtype=float))
        and not _is_pixel_index_axes(context.coord_x, context.coord_y)
    )
    if not world_mode and not attached_world_axes:
        return ds

    if world_mode:
        ds["x0_world"] = fit["x0"]
        ds["y0_world"] = fit["y0"]
        ds["x0_world_err"] = fit["x0_e"]
        ds["y0_world_err"] = fit["y0_e"]
        ds["x0_world"].attrs["description"] = "Component center x in world coordinates."
        ds["y0_world"].attrs["description"] = "Component center y in world coordinates."
        ds["x0_world_err"].attrs["description"] = "1-sigma uncertainty of x0_world."
        ds["y0_world_err"].attrs["description"] = "1-sigma uncertainty of y0_world."
        return ds

    return _interp_centers_world(
        ds,
        np.asarray(context.coord_x),
        np.asarray(context.coord_y),
        context.dim_x,
        context.dim_y,
    )


def _summarize_metadata_value(v, maxlen: int = 120):
    """
    Create a compact metadata-safe representation of a runtime argument value.

    Parameters
    ----------
    v : Any
        Runtime value to summarize.
    maxlen : int
        Maximum representation length for generic objects.

    Returns
    -------
    Any
        Lightweight summary suitable for storing in ``Dataset.attrs``.

    Notes
    -----
    Arrays and DataArrays are summarized by type and shape to avoid bloating
    metadata. Simple scalars are preserved directly so call metadata stays readable.
    """
    if v is None:
        return None
    if isinstance(v, (float, int, bool, str)):
        return v
    if isinstance(v, dict):
        out = {}
        for k, vv in list(v.items())[:50]:
            out[k] = _summarize_metadata_value(vv, maxlen // 2)
        return out
    if isinstance(v, (list, tuple)):
        return [_summarize_metadata_value(x, maxlen // 2) for x in list(v)[:50]]
    try:
        shape = tuple(getattr(v, "shape", ()))
        return f"<{type(v).__name__} shape={shape}>"
    except Exception:
        s = repr(v)
        return s if len(s) <= maxlen else (s[: maxlen - 3] + "...")


def _attach_fit_invocation_metadata(
    ds: xr.Dataset,
    *,
    n_components: int,
    dims: Optional[Sequence[Union[str, int]]],
    min_threshold: Optional[Number],
    max_threshold: Optional[Number],
    initial_guesses,
    bounds,
    initial_is_fwhm: bool,
    max_nfev: int,
    return_model: bool,
    return_residual: bool,
    angle: str,
    coord_type: str,
    unlabeled_axis_order: Optional[str],
    coords,
    world_mode: bool,
) -> xr.Dataset:
    """
    Attach call provenance and package metadata to the fit result dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Result dataset under construction.
    n_components : int
        Number of Gaussian components used in the fit.
    dims : Sequence[str | int] | None
        Public fit-plane specification.
    min_threshold : Number | None
        Inclusive lower data threshold.
    max_threshold : Number | None
        Inclusive upper data threshold.
    initial_guesses : Any
        Original initial-guesses argument supplied by the caller.
    bounds : Any
        Original bounds argument supplied by the caller.
    initial_is_fwhm : bool
        Whether array-form width guesses were interpreted as FWHM.
    max_nfev : int
        Maximum optimizer evaluations.
    return_model : bool
        Whether model planes were requested.
    return_residual : bool
        Whether residual planes were requested.
    angle : str
        Public angle-convention selector.
    coord_type : str
        Public coordinate-frame selector.
    unlabeled_axis_order : {"yx", "xy"} | None
        Semantic axis-order contract for unlabeled NumPy/Dask inputs, or
        ``None`` when the caller supplied labeled data or otherwise did not
        need to resolve unlabeled raw-array axis order.
    coords : Any
        Optional NumPy/Dask coordinate vectors.
    world_mode : bool
        Whether the optimizer ran natively in world coordinates.

    Returns
    -------
    xr.Dataset
        Dataset with invocation and package provenance stored in ``attrs``.

    Notes
    -----
    The stored parameter payload intentionally excludes the raw image data and
    summarizes complex objects compactly so that metadata stay readable.
    """
    try:
        import inspect as _inspect
    except Exception:
        _inspect = None
    try:
        from importlib import metadata as _ilm
    except Exception:
        _ilm = None

    param = dict(
        n_components=int(n_components),
        dims=(list(dims) if dims is not None else None),
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        initial_guesses=_summarize_metadata_value(initial_guesses),
        bounds=_summarize_metadata_value(bounds),
        initial_is_fwhm=bool(initial_is_fwhm),
        max_nfev=int(max_nfev),
        return_model=bool(return_model),
        return_residual=bool(return_residual),
        angle=str(angle),
        coord_type=str(coord_type),
        unlabeled_axis_order=(
            None if unlabeled_axis_order is None else str(unlabeled_axis_order)
        ),
        coords=_summarize_metadata_value(coords),
    )
    call = _build_call(_inspect, param)
    pkg = __package__.split(".")[0] if __package__ else "astroviper"
    try:
        ver = _ilm.version(pkg) if _ilm is not None else "unknown"
    except Exception:
        try:
            import astroviper as _av

            ver = getattr(_av, "__version__", "unknown")
        except Exception:
            ver = "unknown"

    ds.attrs["call"] = call
    ds.attrs["param"] = param
    ds.attrs["package"] = pkg
    ds.attrs["version"] = ver
    ds.attrs["fit_native_frame"] = "world" if world_mode else "pixel"
    return ds


def _apply_fit_variable_docs(ds: xr.Dataset) -> xr.Dataset:
    """
    Attach self-documenting descriptions and units to published fit variables.

    Parameters
    ----------
    ds : xr.Dataset
        Result dataset returned by :func:`fit_multi_gaussian2d`.

    Returns
    -------
    xr.Dataset
        Dataset with variable descriptions and angle units populated.

    Notes
    -----
    These descriptions are intentionally verbose because the result dataset is part
    of the user-facing API and should remain understandable when inspected without
    opening the source code.
    """
    dv_docs = {
        "x0_pixel": "Gaussian center x-coordinate in pixel indices (0-based), derived from world coords if necessary.",
        "y0_pixel": "Gaussian center y-coordinate in pixel indices (0-based), derived from world coords if necessary.",
        "x0_pixel_err": "1σ uncertainty of x0_pixel (native if pixel fit; world fit propagated via local pixel scale).",
        "y0_pixel_err": "1σ uncertainty of y0_pixel (native if pixel fit; world fit propagated via local pixel scale).",
        "x0_world": "Component center x in world coordinates (if available).",
        "y0_world": "Component center y in world coordinates (if available).",
        "x0_world_err": "1σ uncertainty of x0_world (direct if world fit; else via interpolation/pixel-scale propagation).",
        "y0_world_err": "1σ uncertainty of y0_world (direct if world fit; else via interpolation/pixel-scale propagation).",
        "sigma_major_pixel": "Gaussian 1σ scale along the major principal axis in pixel units (after world→pixel conversion).",
        "sigma_minor_pixel": "Gaussian 1σ scale along the minor principal axis in pixel units (after world→pixel conversion).",
        "sigma_major_pixel_err": "1σ uncertainty in sigma_major_pixel in pixel units (after world→pixel conversion).",
        "sigma_minor_pixel_err": "1σ uncertainty in sigma_minor_pixel in pixel units (after world→pixel conversion).",
        "sigma_major_world": "Principal-axis 1σ (major) expressed in world coordinates.",
        "sigma_minor_world": "Principal-axis 1σ (minor) expressed in world coordinates.",
        "sigma_major_world_err": "1σ uncertainty of sigma_major_world (native if world fit; else propagated).",
        "sigma_minor_world_err": "1σ uncertainty of sigma_minor_world (native if world fit; else propagated).",
        "fwhm_major_pixel": "Full-width at half-maximum along the major principal axis in pixel coordinates.",
        "fwhm_minor_pixel": "Full-width at half-maximum along the minor principal axis in pixel coordinates.",
        "fwhm_major_pixel_err": "1σ uncertainty of pixel-frame FWHM(major) (native if pixel fit; else propagated).",
        "fwhm_minor_pixel_err": "1σ uncertainty of pixel-frame FWHM(minor) (native if pixel fit; else propagated).",
        "fwhm_major_world": "FWHM of the major principal axis in world coordinates.",
        "fwhm_minor_world": "FWHM of the minor principal axis in world coordinates.",
        "fwhm_major_world_err": "1σ uncertainty of world-frame FWHM(major) (native if world fit; else propagated).",
        "fwhm_minor_world_err": "1σ uncertainty of world-frame FWHM(minor) (native if world fit; else propagated).",
        "theta_pixel": "Orientation of the ellipse MAJOR axis in pixel coordinates (same convention as 'theta').",
        "theta_pixel_err": "1σ uncertainty of theta_pixel (major-axis orientation) in radians; propagated through the world↔pixel transform.",
        "theta_pixel_math": "Orientation of the ellipse MAJOR axis in pixel coordinates using the math convention (+x toward +y).",
        "theta_pixel_math_err": "1σ uncertainty of theta_pixel_math in radians.",
        "theta_pixel_pa": "Orientation of the ellipse MAJOR axis in pixel coordinates using position angle (+y toward +x).",
        "theta_pixel_pa_err": "1σ uncertainty of theta_pixel_pa in radians.",
        "theta_world": "Orientation of the ellipse MAJOR axis in world coordinates (same convention as 'theta').",
        "theta_world_err": "1σ uncertainty of theta_world (major-axis orientation) in radians; propagated through the pixel↔world transform.",
        "theta_world_math": "Orientation of the ellipse MAJOR axis in world coordinates using the math convention (+x toward +y).",
        "theta_world_math_err": "1σ uncertainty of theta_world_math in radians.",
        "theta_world_pa": "Orientation of the ellipse MAJOR axis in world coordinates using position angle (+y toward +x).",
        "theta_world_pa_err": "1σ uncertainty of theta_world_pa in radians.",
        "amplitude": "Component amplitude (peak height above offset) in data units.",
        "peak": "Model peak value at the component center (offset + amplitude) in data units.",
        "peak_err": "1σ uncertainty of the model peak (quadrature of amplitude_err and offset_err).",
        "offset": "Additive constant background in data units for this fit plane.",
        "offset_err": "1σ uncertainty on the additive background.",
        "amplitude_err": "1σ uncertainty of amplitude parameter in data units.",
        "success": "Optimizer success flag (True/False).",
        "variance_explained": "Explained variance fraction by the fitted model on this plane; values near 1 are best and negative values are possible.",
        "model": "Best-fit model image on the published public (x, y) grid of the input.",
        "residual": "Residual image = data - model on the published public (x, y) grid of the input.",
    }
    for name, desc in dv_docs.items():
        if name in ds:
            ds[name].attrs.setdefault("description", desc)
        if name.startswith("theta") and name in ds:
            ds[name].attrs["units"] = "rad"
    if "variance_explained" in ds:
        ds = _add_variance_explained(ds)
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
    angle: str = "math",
    coord_type: str = "world",
    unlabeled_axis_order: Optional[str] = None,
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
      Input array. If not a DataArray, it is wrapped with synthetic dims
      ``("dim_0", "dim_1", ...)`` and generated numeric coords.

    n_components: int
      Number of Gaussian components (N ≥ 1).

    dims: Sequence[str | int] | None
      Two dims (names or indices) that define the fit plane (x, y). If omitted: uses ('x','y') if present; else for 2-D uses (last, second-last). Required for ndim ≠ 2 without ('x','y').

    mask: numpy.ndarray | dask.array.Array | xarray.DataArray | str | None
      Optional boolean mask. **True means the associated pixel is good.** If 2-D with dims (y, x), it is
      broadcast across leading dims; if it matches the full array shape, it is used
      elementwise. Combined with thresholds as: `final_mask = mask ∧ (>= min_threshold) ∧ (<= max_threshold)`.
      A string value is interpreted as a CRTF specification.
    min_threshold: float | None
      Inclusive lower threshold; pixels with values < min_threshold are ignored during the fit.

    max_threshold: float | None
      Inclusive upper threshold; pixels with values > max_threshold are ignored during the fit.
    initial_guesses: numpy.ndarray[(N,6)] | list[float] | list[dict] | dict | None
      Initial guesses. **Interpreted in FWHM units** for widths by default:
        • array shape (N,6): columns **[amp, x0, y0, fwhm_major, fwhm_minor, theta]**.
        • for ``n_components == 1``, a flat length-6 numeric sequence is accepted and treated as one component.
        • list of N dicts: keys {"amp"/"amplitude","x0","y0","fwhm_major"|"sigma_x"|"sx","fwhm_minor"|"sigma_y"|"sy","theta"}.
        • dict: {"offset": float (optional), "components": (N,6) array OR list[dict] as above}.
      If omitted, peaks are auto-seeded and offset defaults to the median of threshold-masked data.
      Note: θ in `initial_guesses` is interpreted per `angle`.
      FWHM are converted to σ internally for the optimizer. Set ``initial_is_fwhm=False`` only if
      your array-form guesses use σ columns instead.
    bounds: dict[str, (float,float) | Sequence[(float,float)]] | None
      Bounds to constrain parameters. Keys may include {"offset","amp"/"amplitude","x0","y0","fwhm_major","fwhm_minor"}.
      Each parameter is either a single two element (low, high) tuple applied to all components,
      or a length-N sequence of
      (low, high) tuples for per-component bounds. To **fix** a parameter, use a very small, meaningless difference
      between (low, high), e.g. (1.0, 1.000001).
      FWHM bounds must provide both ``fwhm_major`` and ``fwhm_minor`` simultaneously.
      For each component, the major interval must remain ordered above the minor interval:
      ``fwhm_major_lo >= fwhm_minor_lo`` and ``fwhm_major_hi >= fwhm_minor_hi``.
      ``theta`` bounds are not supported by the current raw-width parameterization.

    initial_is_fwhm: bool
      Default **True**. When ``True`` and ``initial_guesses`` is an array of shape (N,6), columns 3–4
      are treated as **FWHM** and converted to σ internally. Set to ``False`` only if your array guesses
      are for σ widths. (Dict/list forms can use ``fwhm_major/fwhm_minor`` keys directly.)

    max_nfev: int
      Maximum function evaluations for the optimizer. Default: 20000.

    return_model: bool
      If True, include the fitted model plane(s) as variable ``model``.

    return_residual: bool
      If True, include residual plane(s) (``data − model``) as variable ``residual``.

    angle: {"math","pa","auto"}
      Convention used to interpret any input ``theta`` values supplied in
      ``initial_guesses``.
        • "math": ``theta`` is interpreted as the mathematical angle from +x toward +y.
        • "pa": ``theta`` is interpreted as astronomical position angle from +y toward +x.
        • "auto": interpret input ``theta`` as PA on left-handed axes and math otherwise.

    coord_type: {"world","pixel"}, default "world"
      Applies only to xarray.DataArray inputs.
        • "world": the optimizer evaluates the Gaussian model on the DataArray's attached 1-D coordinate axes.
        • "pixel": the optimizer evaluates the Gaussian model on zero-based pixel-index axes.
      When coordinate metadata are available, the returned dataset may still report both pixel-space and world-space component parameters; ``coord_type`` only selects the native frame in which the fit is performed.
      Ignored for NumPy/Dask inputs.
    unlabeled_axis_order: {"yx","xy"} | None, default None
      Applies only when ``data`` is a NumPy/Dask array and ``dims`` is omitted.
      In that ambiguous raw-array case the caller must specify this explicitly:
        • "yx": interpret the stored 2-D plane as rows then columns, so the last axis is x and the second-last is y.
        • "xy": interpret the stored 2-D plane semantically as x then y, so the first axis is x and the second is y.
      Ignored for labeled ``xarray.DataArray`` inputs and for calls that already
      specify ``dims`` explicitly.
    coords: tuple[np.ndarray, np.ndarray] | list[np.ndarray] | None
      For NumPy/Dask inputs only: provide (x1d, y1d) to fit in world coordinates. Ignored for DataArray inputs.

    Returns
    -------
    xarray.Dataset
        Per-plane results with a new core dim ``component`` (length N).

        Per-component fit values include:
            - ``amplitude(component)``
            - pixel-frame centers ``x0_pixel(component)``, ``y0_pixel(component)``
            - pixel-frame widths ``sigma_major_pixel(component)``,
              ``sigma_minor_pixel(component)``, ``fwhm_major_pixel(component)``,
              ``fwhm_minor_pixel(component)``
            - pixel-frame angles ``theta_pixel(component)``,
              ``theta_pixel_math(component)``, ``theta_pixel_pa(component)``
            - ``peak(component)`` [= amplitude + offset]
        Per-component 1σ uncertainties include matching ``*_err`` variables for
        all published component quantities above.
        Scalars:
            - ``offset``, ``offset_err``, ``success`` (bool), ``variance_explained``
        Optional planes:
            - ``residual`` (if ``return_residual``), ``model`` (if ``return_model``)
        Optional world-frame component variables are added only when the input
        provides usable non-pixel 1-D coordinate variables:
            - centers ``x0_world(component)``, ``y0_world(component)``
            - world-frame widths ``sigma_major_world(component)``,
              ``sigma_minor_world(component)``, ``fwhm_major_world(component)``,
              ``fwhm_minor_world(component)``
            - world-frame angles ``theta_world(component)``,
              ``theta_world_math(component)``, ``theta_world_pa(component)``
            - matching ``*_err`` variables for each published world-frame quantity

    Notes
    -----
    The fitter models coordinates in x-before-y semantic order while operating on
    image planes stored in NumPy ``[y, x]`` memory order. Supported angle choices are
    ``"math"`` for the internal mathematical convention, ``"pa"`` for astronomical
    position angle measured from +y toward +x, and ``"auto"`` to choose based on axis
    handedness. When coordinate metadata are available, the result can include both
    pixel-space and world-space views of the fitted components, but ``coord_type``
    determines which frame was used by the optimizer itself. Width parameters are
    optimized internally in sigma units even when array-form initial guesses or
    bounds are provided in FWHM. For unlabeled NumPy/Dask inputs with omitted
    ``dims``, callers must now specify ``unlabeled_axis_order`` explicitly so the
    public x/y plane semantics are not inferred silently.

    Parallelization
    ---------------
    This function is intentionally generic and remains public for callers working
    with plain NumPy, Dask, or Xarray arrays outside any Astroviper image schema.
    Its parallel execution behavior therefore depends on the backing array type and
    on the numerical libraries active in the current Python process.

    For Dask-backed inputs, vectorization across all non-fit dims is dispatched
    through ``xarray.apply_ufunc(dask="parallelized")``. In that mode, performance
    is typically controlled by Dask chunking and scheduler configuration rather than
    by this function directly. For stacks of many independent planes, parallelizing
    across the outer stack dims is usually the right first strategy. In general, do
    not chunk along the two dimensions that define the fit plane. Each fit needs the
    full ``(y, x)`` plane available as one unit, so chunking the fit axes usually
    adds overhead or can prevent the gufunc-style execution from matching the
    intended plane-by-plane pattern. Prefer chunking only along outer stack dims
    such as ``time``, ``frequency``, ``stokes``, or similar non-fit dims. For
    example, if the fit plane is ``("x", "y")``, a layout such as
    ``data.chunk({"time": 4, "y": -1, "x": -1})`` is typically a better starting
    point than chunking ``x`` or ``y``.

    As a first Dask configuration to try for CPU-bound plane fitting on one
    machine, prefer more single-threaded workers rather than fewer heavily threaded
    workers. A practical starting point is a local distributed client such as
    ``Client(n_workers=<physical_cores>, threads_per_worker=1)`` with chunking only
    along the outer plane dims. If memory is tight, reduce ``n_workers`` first. If
    there are many more planes than worker slots, choose outer-dimension chunking so
    the total number of chunks is several times larger than the total number of
    worker task slots, which usually gives better load balancing than creating only
    one chunk per worker.

    For small stacks with only a few fit tasks, timing can vary substantially under
    threaded or distributed schedulers because scheduling overhead and task
    interleaving become a larger fraction of total runtime. For benchmarking, start
    with the synchronous scheduler on a fixed input cube, then compare distributed
    configurations only after the single-process baseline is stable.

    For NumPy-backed inputs, this function does not create Dask tasks. However, the
    underlying NumPy/SciPy operations may still use multithreaded native libraries
    such as OpenBLAS, MKL, BLIS, or OpenMP-backed kernels. As a result, a NumPy
    input can still consume many CPU cores even though the public API itself has no
    explicit ``n_workers`` or ``n_threads`` parameter.

    Single-threaded native execution can be faster for workloads dominated by many
    small independent plane fits, because heavily threaded BLAS/OpenMP execution may
    add synchronization overhead and oversubscribe CPU resources. This is workload
    dependent, so benchmark representative data rather than assuming that more
    threads are always faster.

    A possible in-process, best-effort option is to limit supported native thread
    pools with ``threadpoolctl``:

    >>> from threadpoolctl import threadpool_limits
    >>> with threadpool_limits(limits=1):
    ...     ds = fit_multi_gaussian2d(cube, n_components=2, initial_guesses=init)

    This can be effective even after NumPy/SciPy have already been imported, but it
    is not guaranteed to force strict single-threaded execution for every backend or
    every library already loaded into the process. Treat it as a convenience option,
    not as a hard guarantee.

    The most reliable way to guarantee single-threaded native math is to set the
    relevant environment variables before starting Python, so that the numerical
    libraries initialize with the desired thread limits from process start. Typical
    examples include ``OMP_NUM_THREADS=1``, ``OPENBLAS_NUM_THREADS=1``,
    ``MKL_NUM_THREADS=1``, ``VECLIB_MAXIMUM_THREADS=1``, and
    ``NUMEXPR_NUM_THREADS=1``. The exact variables that matter depend on the BLAS
    and OpenMP backend linked into the user's NumPy/SciPy build. When Dask is used
    to parallelize across planes, setting those native thread counts to ``1`` is
    usually a good first choice so that Dask worker concurrency and native linear
    algebra threading do not compete with each other.

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

    """
    if n_components < 1:
        raise ValueError("n_components must be >= 1.")
    # Stage 1: normalize the public input layout and derive the fit-plane frame
    # metadata. For raw unlabeled arrays this is the point where the caller's
    # explicit public plane contract is resolved and then converted into the
    # fitter's internal trailing (y, x) row/column plane order.
    context = _build_fit_execution_context(data, dims, unlabeled_axis_order)
    _warn_if_suboptimal_dask_chunking(context)
    raw_plane_dims = None
    if not isinstance(data, xr.DataArray):
        # Preserve the original public plane-dimension order of the raw input so
        # plain 2-D masks can be labeled the same way before xarray aligns them to
        # the internal trailing (y, x) fit view.
        raw_plane_dims = tuple(
            d for d in context.da_in.dims if d in (context.dim_x, context.dim_y)
        )

    # Stage 2: align all user-facing configuration to that internal (y, x) fit
    # view. Raw unlabeled masks must honor the same public plane order as the raw
    # data first so they undergo the same semantic x/y -> row/column conversion.
    mask_da = _resolve_fit_mask(
        mask,
        context.da_tr,
        context.dim_y,
        context.dim_x,
        raw_plane_dims=raw_plane_dims,
    )
    init_for_fit, bounds_for_fit, want_pa = _prepare_fit_configuration(
        initial_guesses,
        bounds,
        initial_is_fwhm,
        angle,
        int(n_components),
        context.sx_sign,
        context.sy_sign,
        context.is_left_handed,
    )
    x1d, y1d, world_mode = _resolve_fit_coordinate_axes(
        data, context, coord_type, coords
    )

    # Stage 3: run the numerical optimizer plane-by-plane on the chosen coordinate grid.
    raw_results = _run_fit_apply_ufunc(
        context,
        mask_da,
        x1d,
        y1d,
        int(n_components),
        min_threshold,
        max_threshold,
        init_for_fit,
        bounds_for_fit,
        int(max_nfev),
        bool(return_model),
        bool(return_residual),
    )

    # Stage 4: canonicalize the raw optimizer output into the public major-axis convention.
    fit = _canonicalize_fit_outputs(raw_results, want_pa)

    # Stage 5: publish mirrored pixel/world parameter views and optional plane outputs.
    ds = _build_published_parameter_dataset(
        fit,
        context,
        x1d,
        y1d,
        world_mode,
        want_pa,
    )
    ds = _attach_optional_plane_outputs(
        ds,
        fit,
        context,
        bool(return_residual),
        bool(return_model),
    )
    ds = _attach_world_center_outputs(ds, fit, context, world_mode)

    # Stage 6: finish with call provenance and self-documenting variable metadata.
    ds = _attach_fit_invocation_metadata(
        ds,
        n_components=int(n_components),
        dims=dims,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        initial_guesses=initial_guesses,
        bounds=bounds,
        initial_is_fwhm=bool(initial_is_fwhm),
        max_nfev=int(max_nfev),
        return_model=bool(return_model),
        return_residual=bool(return_residual),
        angle=str(angle),
        coord_type=str(coord_type),
        unlabeled_axis_order=(
            None if unlabeled_axis_order is None else str(unlabeled_axis_order)
        ),
        coords=coords,
        world_mode=world_mode,
    )
    return _apply_fit_variable_docs(ds)


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

    Returns
    -------
    None
        The function draws in place on ``ax`` and does not return a value.
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
        cands = [
            (f"theta_{frame_name}_{first}", first),
            (f"theta_{frame_name}_{second}", second),
            (f"theta_{frame_name}", prefer if prefer in ("pa", "math") else "math"),
            ("theta", prefer if prefer in ("pa", "math") else "math"),
        ]
        for nm, kind_hint in cands:
            a = _arr(nm)
            if a is not None:
                if nm.endswith("_pa"):
                    kind = "pa"
                elif nm.endswith("_math"):
                    kind = "math"
                else:
                    kind = kind_hint
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
    Plot one fitted image plane and optionally its residual with component overlays.

    Parameters
    ----------
    data : ArrayOrDA
        Original image-like input accepted by :func:`fit_multi_gaussian2d`.
    result : xr.Dataset
        Dataset produced by :func:`fit_multi_gaussian2d`.
    dims : Sequence[str | int] | None, optional
        Plot-plane dimension specifier. Supported choices are explicit dimension names
        or integer dimension indices. If omitted, the same rules as
        :func:`_resolve_dims` are used.
    indexer : Mapping[str, int] | None, optional
        For inputs with leading non-plane dimensions, integer index selections used to
        choose a single plane for plotting.
    show_residual : bool, default True
        If ``True`` and the result dataset contains ``residual``, add a second panel
        showing the residual image.
    fwhm : bool, default False
        If ``True``, draw FWHM ellipses. If ``False``, draw one-sigma ellipses.
    angle : str | None, optional
        Preferred overlay angle convention. Supported choices are ``"math"``,
        ``"pa"``, or ``None`` to let the plotting helper auto-select.
    show : bool | None, optional
        If ``True``, call ``matplotlib.pyplot.show()``. If ``False``, suppress the
        display call. If ``None``, defer to the module's existing plotting behavior.

    Returns
    -------
    tuple
        Tuple ``(fig, axes)`` where ``fig`` is the Matplotlib figure and ``axes`` is
        either a single axes object or an array of axes, matching the created layout.

    Notes
    -----
    The helper uses world coordinates when the plotted dimensions carry finite,
    monotonic 1-D coordinates; otherwise it falls back to pixel plotting. Component
    overlays are drawn in the matching frame so fitted x/y centers are interpreted in
    the same convention as the displayed axes.
    """
    import numpy as _np

    try:
        import matplotlib.pyplot as _plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Matplotlib is required for plot_components()") from exc

    # Normalize input & dims. When raw unlabeled arrays are plotted without an
    # explicit dims= override, reuse the fit-time unlabeled-axis contract stored
    # in the result metadata so plotting interprets the same semantic x/y plane
    # that the fitter used before it normalized internally to trailing (y, x).
    da = _ensure_dataarray(data)
    unlabeled_input = not isinstance(data, xr.DataArray)
    unlabeled_axis_order = None
    if isinstance(getattr(result, "attrs", None), Mapping):
        _param = result.attrs.get("param")
        if isinstance(_param, Mapping):
            _stored = _param.get("unlabeled_axis_order")
            if _stored is not None:
                unlabeled_axis_order = str(_stored)
    dim_x, dim_y = _resolve_dims(
        da,
        dims,
        unlabeled_input=unlabeled_input,
        unlabeled_axis_order=unlabeled_axis_order,
    )
    # Plotting extracts a single stored image plane, so it follows the internal
    # row/column ordering (y, x) before labeling axes semantically.
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

    # If the caller passes a spatially subsetted data plane but an unsliced fit result,
    # align the result to the displayed x/y coordinates before plotting residual/model.
    if (
        isinstance(res_plane, xr.Dataset)
        and dim_x in res_plane.dims
        and dim_y in res_plane.dims
    ):
        try:
            res_plane, _ = xr.align(res_plane, data2d, join="right")
        except Exception:
            pass

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

    # Use the same frame as the axes: world if we're plotting with coordinate
    # arrays, otherwise pixel. This avoids reinterpreting the caller's raw
    # ``dims`` argument and keeps the overlay frame tied to the already-resolved
    # plotted axes.
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
        residual_da = res_plane["residual"]
        if (dim_x in residual_da.dims) and (dim_y in residual_da.dims):
            # Convert the published residual plane back to stored (y, x) order for
            # plotting routines that consume row/column image arrays.
            residual_da = residual_da.transpose(
                *(d for d in residual_da.dims if d not in (dim_y, dim_x)), dim_y, dim_x
            )
        R = _np.asarray(residual_da.values, dtype=float)
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
