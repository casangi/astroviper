"""Generic selection layer for applications + CASA region (pixel) support.

Supported ``select`` forms (basic):
- ``None`` → keep everything.
- A boolean array-like (``xarray.DataArray`` or ``numpy.ndarray``), already aligned
  or broadcastable to the target ``data``.
- A string expression over named masks using ``&``, ``|``, ``^`` and ``~``.
  Names are resolved from ``mask_source`` (a mapping or an ``xarray.Dataset``).
  If ``mask_source`` is omitted and ``data`` is a Dataset, the dataset is used.

This module exposes two public helpers:
- ``select_mask(data, select=None, mask_source=None) -> Union[xr.DataArray, np.ndarray]``
- ``apply_select(data, select=None, mask_source=None) -> same type as data``

The first returns a boolean mask aligned to ``data``; the second applies it.

CRTF (CASA Region Text Format) pixel support (new)
--------------------------------------------------
* ``select`` may be a CRTF pixel string. Supported shapes: ``box``, ``centerbox``,
  ``rotbox``, ``circle``, ``ellipse``, ``poly``, ``annulus``.
* Autodetect CRTF when the string starts with ``#CRTF`` **or** begins with a shape
  token followed by ``[[`` (e.g., ``box[[...]]``).
* **Pixel coordinates are 0-based (NumPy/xarray index space)** and all pixel
  quantities must be suffixed with ``pix`` (e.g., ``[0pix, 127pix]``).

* You can also pass a CRTF file path either as a backticked string (`` `path/file.crtf` ``)
  or a ``pathlib.Path``; the file contents are read and parsed.

Notes
-----
* Masks treat ``NaN`` as ``False``.
* Only bitwise operators are supported in expressions; ``and``/``or`` are rejected.
* Unknown names raise ``KeyError`` listing available mask names.
"""

from __future__ import annotations

from typing import Mapping, Any, Union, Literal, Optional, Tuple, List
from pathlib import Path
import dask.array as da

import math
import ast
import re

import numpy as np
import xarray as xr

ArrayLike = Union[np.ndarray, xr.DataArray]
__all__ = ["select_mask", "apply_select", "combine_with_creation"]


def apply_select(
    data: ArrayLike, select: Any | None = None, mask_source: Any | None = None
) -> ArrayLike:
    """
    Apply a selection mask to ``data`` and return a masked array/DataArray.

    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Image-like array to mask. May be NumPy-backed or Dask-backed when DataArray.
    select : None | bool array-like | str | pathlib.Path
        Selection to apply. Supported forms:
          - ``None``: keep everything (no-op).
          - Boolean array-like (NumPy/xarray/Dask): broadcast/aligned to ``data``.
          - CRTF text (pixel units) such as ``"#CRTF\\nbox[[x1pix,y1pix],[x2pix,y2pix]]"``.
          - Named-mask expression using bitwise ops ``& | ^ ~`` over names from ``mask_source``.
          - Backticked path or ``pathlib.Path`` to a CRTF file (e.g., ``"`regions.crtf`"`` or ``Path("regions.crtf")``).
    mask_source : Mapping[str, array-like] | xarray.Dataset | None
        Source of named masks referenced by expressions. Only boolean-ish arrays are used.

    Returns
    -------
    numpy.ndarray or xarray.DataArray
        Same container type as ``data``:
          - If ``data`` is an ``xarray.DataArray``, returns a DataArray with values
            outside the selection set to NaN via ``data.where(mask)``. Dims/coords
            are preserved. If the input is Dask-backed, the result remains lazy.
          - If ``data`` is a NumPy ``ndarray``, returns an array where values outside
            the selection are NaN (via ``np.where``). If the input dtype cannot
            represent NaN (e.g., integer), the result is upcast to a floating dtype.

    Notes
    -----
    - For xarray inputs, masks are aligned by dimension names; for NumPy inputs,
      masks must be broadcastable by shape.
    - In mask construction, NaNs in numeric arrays are treated as False.
    - Expressions support only ``~``, ``&``, ``|``, ``^`` and parentheses; ``and``/``or`` are rejected.
    """
    mask = select_mask(data, select=select, mask_source=mask_source)
    if isinstance(data, xr.DataArray):
        return data.where(mask)
    return np.where(np.asarray(mask, dtype=bool), data, np.nan)


ReturnKind = Literal["numpy", "dask", "dataarray-numpy", "dataarray-dask"]


def select_mask(
    data: ArrayLike,
    select: Any | None = None,
    mask_source: Any | None = None,
    *,
    return_kind: ReturnKind = "dataarray-dask",
    dask_chunks: Optional[Tuple[int, ...]] = None,
    creation_hint: Optional[str] = None,
    auto_merge_creation: bool = False,
) -> ArrayLike:
    """Build a boolean mask aligned to ``data`` from ``select``.

    Parameters
    ----------
    data:
        Template array that determines the mask's shape/dims.
    select:
        ``None`` | boolean array-like | string expression over named masks.
    mask_source:
        Mapping or ``xr.Dataset`` that provides named masks for expressions.
        If ``None`` and ``data`` is an ``xr.Dataset``, that dataset is used.

    Returns
    -------
    ``xr.DataArray`` if ``data`` is a ``DataArray``; otherwise ``np.ndarray``.
    """
    # For xr.DataArray results created from strings/paths, we record a
    # human-readable hint on how to recreate the mask.
    creation_str: Optional[str] = None

    if select is None:
        return _all_true_mask_like(data)
    # Boolean/array-like (NumPy, xarray, or Dask) → align then coerce
    if isinstance(select, (np.ndarray, xr.DataArray, da.Array)):
        # Optional provenance auto-merge for DataArray inputs created by composition.
        creation_str = creation_hint
        if (
            creation_str is None
            and auto_merge_creation
            and isinstance(select, xr.DataArray)
        ):
            c1 = select.attrs.get("creation_a")
            c2 = select.attrs.get("creation_b")
            op = select.attrs.get("creation_op")
            if c1 and c2 and op:
                creation_str = f"({c1}) {op} ({c2})"
            elif "creation" in select.attrs:
                creation_str = select.attrs.get("creation")
        aligned = _align_bool_mask_to_data(select, data)
        return _coerce_return_kind(
            aligned, data, return_kind, dask_chunks, creation=creation_str
        )
    # String/Path: backticked file or Path → load as CRTF; else treat as text.
    if isinstance(select, (str, Path)):
        s_file = _maybe_read_crtf_from_path(select)
        if s_file is not None:
            # If the user provided a file, it's CRTF by definition; parse directly.
            m = _crtf_pixel_mask(data, s_file, lazy=_want_dask(return_kind))
            aligned = _align_bool_mask_to_data(m, data)
            # Record the *file contents* (not the filename) for reproducible provenance
            creation_str = creation_hint if creation_hint is not None else s_file
            return _coerce_return_kind(
                aligned, data, return_kind, dask_chunks, creation=creation_str
            )

        # Otherwise, handle plain strings (CRTF text or named-mask expression)
        if isinstance(select, Path):
            raise FileNotFoundError(f"CRTF file not found: {select}")
        s = select.strip()
    if isinstance(select, str):
        s = select.strip()
        if _looks_like_crtf_pixel(s):
            m = _crtf_pixel_mask(data, s, lazy=_want_dask(return_kind))
            aligned = _align_bool_mask_to_data(m, data)
            creation_str = creation_hint if creation_hint is not None else select
            return _coerce_return_kind(
                aligned, data, return_kind, dask_chunks, creation=creation_str
            )
        env = _build_mask_env(
            mask_source or (data if isinstance(data, xr.Dataset) else {})
        )
        expr_mask = _eval_mask_expr(s, env)
        aligned = _align_bool_mask_to_data(expr_mask, data)
        creation_str = (
            creation_hint
            if creation_hint is not None
            else _build_creation_for_expression(select, env)
        )
        return _coerce_return_kind(
            aligned, data, return_kind, dask_chunks, creation=creation_str
        )
    raise TypeError(
        "Unsupported select type. Expected None, boolean array-like, expression/CRTF text, "
        "or a backticked CRTF file string / pathlib.Path."
    )


def _maybe_read_crtf_from_path(sel: Any) -> str | None:
    """Return file contents if `sel` is a backticked string or a Path; else None."""
    if isinstance(sel, Path):
        return sel.read_text(encoding="utf-8") if sel.is_file() else None
    if isinstance(sel, str):
        s = sel.strip()
        m = re.fullmatch(r"`([^`]+)`", s)
        if not m:
            return None
        p = Path(m.group(1))
        if not p.is_file():
            raise FileNotFoundError(f"CRTF file not found: {p}")
        return p.read_text(encoding="utf-8")
    return None  # pragma: no cover


# ---------------------------- internal helpers -----------------------------


def _all_true_mask_like(data: ArrayLike) -> ArrayLike:
    if isinstance(data, xr.DataArray):
        # xarray requires a dtype=bool explicitly for ones_like masks
        return xr.ones_like(data, dtype=bool)
    return np.ones(np.shape(data), dtype=bool)


def _align_bool_mask_to_data(mask: ArrayLike, data: ArrayLike) -> ArrayLike:
    """Coerce ``mask`` to bool and align/broadcast to ``data``.

    NaNs become False.
    """
    if isinstance(data, xr.DataArray):
        # Preserve dims to avoid accidental outer-product alignment when wrapping raw arrays
        if isinstance(mask, xr.DataArray):
            m = mask
        else:
            try:
                m = xr.DataArray(mask, dims=data.dims[: np.ndim(mask)])
            except Exception:  # defensive fallback
                m = xr.DataArray(mask)
        # why: NaNs must become False before bool-cast
        if np.issubdtype(m.dtype, np.floating):
            m = m.fillna(False)
        m = m.astype(bool)
        try:
            # align by named dimensions, broadcasting as needed
            m = m.broadcast_like(data)
            return m
        except Exception:
            # Fallback: NumPy-shape broadcast then wrap back to DataArray.
            m_np = np.asarray(m.data, dtype=bool)
            try:
                b = np.broadcast_to(m_np, data.shape)
            except ValueError as exc:
                raise ValueError("Mask is not broadcastable to data shape") from exc
            return xr.DataArray(b, dims=data.dims, coords=data.coords)
    # numpy path
    m_np = np.asarray(mask)
    # why: NaNs must become False before bool-cast
    if np.issubdtype(m_np.dtype, np.floating):
        m_np = np.nan_to_num(m_np, nan=0.0)
    m_np = m_np.astype(bool)
    try:
        # broadcast_to yields a readonly view; we do not mutate it afterwards
        m_np = np.broadcast_to(m_np, np.shape(data))
    except ValueError as exc:
        raise ValueError("Mask is not broadcastable to data shape") from exc
    return m_np


def _build_mask_env(mask_source: Any) -> Mapping[str, ArrayLike]:
    if isinstance(mask_source, xr.Dataset):
        items = {k: v for k, v in mask_source.data_vars.items() if _is_boolish(v)}
    elif isinstance(mask_source, Mapping):
        items = {str(k): v for k, v in mask_source.items() if _is_boolish(v)}
    else:
        raise TypeError("mask_source must be a Mapping or xarray.Dataset")

    if not items:
        raise ValueError("mask_source does not provide any boolean masks")

    # Normalize each value to bool arrays/DataArrays; keep xarray metadata when present
    norm: dict[str, ArrayLike] = {}
    for name, val in items.items():
        if isinstance(val, xr.DataArray):
            v = val
            if np.issubdtype(v.dtype, np.floating):
                v = v.fillna(False)
            norm[name] = v.astype(bool)
        else:
            arr = np.asarray(val)
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.nan_to_num(arr, nan=0.0)
            norm[name] = arr.astype(bool)
    return norm


def _is_boolish(obj: Any) -> bool:
    if isinstance(obj, xr.DataArray):
        return obj.dtype == bool or np.issubdtype(obj.dtype, np.number)
    arr = np.asarray(obj)
    return arr.dtype == bool or np.issubdtype(arr.dtype, np.number)


# --------------------------- expression evaluator --------------------------

_ALLOWED_BIN_OPS = (ast.BitAnd, ast.BitOr, ast.BitXor)
_ALLOWED_UNARY_OPS = (ast.Invert,)
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Constant,  # allow literal True/False
    ast.BoolOp,  # reject at runtime if it's 'and'/'or'
    # Py3.12 walks operator/context nodes too:
    ast.operator,  # BitAnd/BitOr/BitXor
    ast.unaryop,  # Invert
    ast.Load,  # Name context
)


def _eval_mask_expr(expr: str, env: Mapping[str, ArrayLike]) -> ArrayLike:
    """Safely evaluate a boolean mask expression using bitwise operators.

    Only ``~``, ``&``, ``|``, ``^`` and parentheses are accepted. Names map to
    arrays provided by ``env``. ``and``/``or`` are not supported and will error.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid selection expression") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Expression contains an unsupported construct: {type(node).__name__}"
            )
        if isinstance(node, ast.BoolOp):
            # Explicitly forbid 'and'/'or'
            raise ValueError(
                "Use '&' and '|' instead of 'and'/'or' in selection expressions"
            )

    def _eval(node: ast.AST) -> ArrayLike:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Name):
            try:
                return env[node.id]
            except KeyError as exc:
                available = ", ".join(sorted(env.keys()))
                raise KeyError(
                    f"Unknown mask name: {node.id}. Available: {available}"
                ) from exc
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
            return ~_to_bool(_eval(node.operand))
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN_OPS):
            left = _to_bool(_eval(node.left))
            right = _to_bool(_eval(node.right))
            if isinstance(node.op, ast.BitAnd):
                return left & right
            if isinstance(node.op, ast.BitOr):
                return left | right
            return left ^ right
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return np.array(node.value, dtype=bool)
        raise ValueError("Unsupported token in selection expression")

    return _eval(tree)


def _to_bool(arr: ArrayLike) -> ArrayLike:
    if isinstance(arr, xr.DataArray):
        out = arr
        # why: NaNs must become False before bool-cast
        if np.issubdtype(out.dtype, np.floating):
            out = out.fillna(False)
        return out.astype(bool)
    arr_np = np.asarray(arr)
    if np.issubdtype(arr_np.dtype, np.floating):
        arr_np = np.nan_to_num(arr_np, nan=0.0)
    return arr_np.astype(bool)


# ---------------------------------------------------------------------------
# CASA Region Text Format (CRTF) — pixel-only parser & rasterizer (new)
# ---------------------------------------------------------------------------

_SHAPES = {"box", "centerbox", "rotbox", "poly", "circle", "annulus", "ellipse"}


def _looks_like_crtf_pixel(s: str) -> bool:
    # Strip BOM + whitespace so files with UTF-8 BOM are handled.
    s = s.lstrip("\ufeff \t\r\n")
    if s.startswith("#CRTF"):
        return True
    m = re.match(r"^([+-])?\s*([A-Za-z]+)\s*\[\[", s, flags=re.IGNORECASE)
    return bool(m and m.group(2).lower() in _SHAPES)


def _crtf_pixel_mask(data: ArrayLike, text: str, *, lazy: bool = False) -> ArrayLike:
    """Parse a CRTF pixel string (single or multi-line) into a boolean mask.

    Combination semantics per line: leading '+' (OR, default) or '-' (NOT/subtract).
    """
    ny, nx = data.shape if not isinstance(data, xr.DataArray) else data.shape
    # Build index grids in either NumPy (eager) or Dask (lazy) for efficient masking
    if lazy:
        cx = da.arange(nx, dtype=float)
        ry = da.arange(ny, dtype=float)
        X, Y = da.meshgrid(cx, ry, indexing="xy")
    else:
        c = np.arange(nx, dtype=float)
        r = np.arange(ny, dtype=float)
        X, Y = np.meshgrid(c, r)

    acc = da.zeros((ny, nx), dtype=bool) if lazy else np.zeros((ny, nx), dtype=bool)
    lines = [
        ln.strip()
        for ln in re.split(r"[\n;]+", text)
        if ln.strip() and not ln.strip().startswith("#")
    ]
    for line in lines:
        if line.lower().startswith("global"):
            continue
        flag = "+"
        if line[0] in "+-":
            flag, line = line[0], line[1:].lstrip()
        shape, payload = _split_shape_payload(line)
        mask = _rasterize_shape(shape, payload, X, Y)
        if flag == "+":
            acc = acc | mask
        else:
            acc = acc & (~mask)
    return (
        xr.DataArray(acc, dims=getattr(data, "dims", ("y", "x")))
        if isinstance(data, xr.DataArray)
        else acc
    )


def _split_shape_payload(line: str) -> tuple[str, str]:
    m = re.match(r"^([A-Za-z]+)\s*(\[\[.*)$", line)
    if not m:
        raise ValueError(f"Invalid CRTF line: {line!r}")
    return m.group(1).lower(), m.group(2)


_NUM = r"[-+]?\d+(?:\.\d+)?"
# pixel coordinates/lengths MUST be suffixed with 'pix' to avoid ambiguity
_PIX_NUM = rf"{_NUM}\s*pix"
_PAIR_PIX = rf"\[\s*({_PIX_NUM})\s*,\s*({_PIX_NUM})\s*\]"


def _parse_units_val(tok: str) -> Tuple[float, str | None]:
    m = re.match(rf"^\s*({_NUM})\s*(pix|deg|rad)?\s*$", tok)
    if not m:
        raise ValueError(f"Invalid numeric token: {tok!r}")
    val = float(m.group(1))
    unit = m.group(2)
    return val, unit


def _parse_pix_val(tok: str) -> float:
    m = re.match(rf"^\s*({_NUM})\s*pix\s*$", tok)
    if not m:
        raise ValueError(f"Expected '<value>pix' for pixel quantity, got {tok!r}")
    return float(m.group(1))


def _format_pix_pair_error(src: str) -> str:
    """Build the exact error message (with suggestion) for missing 'pix' units."""
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", src)
    suggestion = f" should be '[{nums[0]}pix, {nums[1]}pix]' if these values represent pixel coordinates."
    return f"Invalid pixel pair token (require 'pix' units): {src!r}{suggestion}"


def _strip_brackets(s: str) -> str:
    if not (s.startswith("[[") and s.endswith("]")):
        raise ValueError("CRTF payload must start with '[[ ... ]]'")
    return s[1:-1]


def _rasterize_shape(
    shape: str,
    payload: str,
    X: np.ndarray | da.Array,
    Y: np.ndarray | da.Array,
) -> np.ndarray | da.Array:
    inner = _strip_brackets(payload).strip()
    parts = _smart_split_pairs(inner)
    if shape == "box":
        p1x, p1y = _parse_pair_pix(parts[0])
        p2x, p2y = _parse_pair_pix(parts[1])
        x1, x2 = sorted([p1x, p2x])
        y1, y2 = sorted([p1y, p2y])
        return (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    if shape == "centerbox":
        cx, cy = _parse_pair_pix(parts[0])
        w, h = _parse_two_pix_vals(parts[1])
        hx, hy = w / 2.0, h / 2.0
        return (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)
    if shape == "rotbox":
        cx, cy = _parse_pair_pix(parts[0])
        w, h = _parse_two_pix_vals(parts[1])
        # Require explicit rotation keyword assignment: pa=<angle> or theta_m=<angle>
        if len(parts) != 3:
            raise ValueError(
                "rotbox requires angle specified as 'pa=<angle>' or 'theta_m=<angle>', "
                "e.g., rotbox[[cx,cy],[w,h], pa=30deg]"
            )
        ang = _parse_angle_kv(parts[2])
        hx, hy = w / 2.0, h / 2.0
        xrp, yrp = _rotate_about(X, Y, cx, cy, -ang)
        return (np.abs(xrp - cx) <= hx) & (np.abs(yrp - cy) <= hy)
    if shape == "circle":
        cx, cy = _parse_pair_pix(parts[0])
        r = _parse_pix_val(parts[1])
        return ((X - cx) ** 2 + (Y - cy) ** 2) <= (r**2 + 1e-9)
    if shape == "annulus":
        cx, cy = _parse_pair_pix(parts[0])
        r1, r2 = _parse_two_pix_vals(parts[1])
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        return (d2 >= r1**2) & (d2 <= r2**2)
    if shape == "ellipse":
        cx, cy = _parse_pair_pix(parts[0])
        a, b = _parse_two_pix_vals(parts[1])  # semi-axes in pix
        # Require explicit rotation keyword assignment: pa=<angle> or theta_m=<angle>
        if len(parts) != 3:
            raise ValueError(
                "ellipse requires angle specified as 'pa=<angle>' or 'theta_m=<angle>', "
                "e.g., ellipse[[cx,cy],[a,b], theta_m=30deg]"
            )
        ang = _parse_angle_kv(parts[2])
        xp, yp = _rotate_about(X, Y, cx, cy, -ang)
        return ((xp - cx) / a) ** 2 + ((yp - cy) / b) ** 2 <= 1.0 + 1e-9
    if shape == "poly":
        pts = [_parse_pair_pix(p) for p in parts]
        return _point_in_poly(X, Y, pts)
    raise ValueError(f"Unsupported CRTF shape: {shape}")


def _parse_angle_kv(token: str) -> float:
    """
    Parse a keyword angle assignment.
    Accepted forms (case-insensitive):
      - 'pa=<angle>'       : position angle measured from +y toward +x (handedness-agnostic).
                             Converted to math angle (from +x toward +y) via (π/2 - α).
      - 'theta_m=<angle>'  : math angle measured from +x toward +y (handedness-agnostic).
                             No conversion.
    <angle> may have units 'deg' or 'rad' (default is interpreted as degrees).
    """
    m = re.match(r"^\s*(pa|theta_m)\s*=\s*(.+?)\s*$", token, flags=re.IGNORECASE)
    if not m:
        raise ValueError(
            "Rotation must be provided as 'pa=<angle>' or 'theta_m=<angle>' (e.g., pa=30deg)."
        )
    mode = m.group(1).lower()
    ang_token = m.group(2)
    ang = _parse_angle(ang_token)  # radians
    if mode == "pa":
        # Convert PA (from +y→+x) to math angle (from +x→+y) irrespective of handedness
        return (math.pi / 2.0) - ang
    return ang


def _smart_split_pairs(inner: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in inner:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        # split only on top-level commas that separate arguments
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _parse_pair_pix(pair_token: str) -> Tuple[float, float]:
    m = re.match(rf"^\s*{_PAIR_PIX}\s*$", pair_token)
    if not m:
        raise ValueError(_format_pix_pair_error(pair_token))
    x = _parse_pix_val(m.group(1))
    y = _parse_pix_val(m.group(2))
    return x, y


def _parse_two_pix_vals(token: str) -> Tuple[float, float]:
    m = re.match(rf"^\s*\[\s*({_PIX_NUM})\s*,\s*({_PIX_NUM})\s*\]\s*$", token)
    if not m:
        raise ValueError(_format_pix_pair_error(token))
    a = _parse_pix_val(m.group(1))
    b = _parse_pix_val(m.group(2))
    return a, b


def _parse_angle(token: str) -> float:
    v, unit = _parse_units_val(token)
    return float(v) if unit == "rad" else math.radians(float(v))


def _rotate_about(
    X: np.ndarray, Y: np.ndarray, cx: float, cy: float, angle_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    x = X - cx
    y = Y - cy
    xr = x * ca - y * sa
    yr = x * sa + y * ca
    return xr + cx, yr + cy


def _point_in_poly(
    X: np.ndarray | da.Array,
    Y: np.ndarray | da.Array,
    pts: list[tuple[float, float]],
) -> np.ndarray | da.Array:
    # Ray casting algorithm, vectorized
    x = X.ravel()
    y = Y.ravel()
    inside = (
        da.zeros_like(x, dtype=bool)
        if hasattr(x, "chunks")
        else np.zeros_like(x, dtype=bool)
    )
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    n = len(pts)
    j = n - 1
    for i in range(n):
        xi, yi = xs[i], ys[i]
        xj, yj = xs[j], ys[j]
        cond = (yi > y) != (yj > y)
        xints = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        inside = inside ^ (cond & (x < xints))
        j = i
    return inside.reshape(X.shape)


def _want_dask(return_kind: ReturnKind) -> bool:
    return return_kind in ("dask", "dataarray-dask")


def _coerce_return_kind(
    mask: ArrayLike,
    data: ArrayLike,
    return_kind: ReturnKind,
    dask_chunks: Optional[Tuple[int, ...]],
    creation: Optional[str] = None,
) -> ArrayLike:
    """Convert aligned mask to the requested return kind efficiently."""
    # numpy ndarray of bool
    if return_kind == "numpy":
        if isinstance(mask, xr.DataArray):
            arr = mask.data
            if getattr(arr, "__module__", "").startswith("dask"):
                return arr.astype(bool).compute()
            return np.asarray(mask.values, dtype=bool)
        if getattr(mask, "__module__", "").startswith("dask"):
            return mask.astype(bool).compute()  # type: ignore[return-value]
        return np.asarray(mask, dtype=bool)

    # dask array of bool
    if return_kind == "dask":
        if isinstance(mask, xr.DataArray):
            arr = mask.data
            if getattr(arr, "__module__", "").startswith("dask"):
                return arr.astype(bool)
            chunks = _infer_chunks_like(data, arr.shape, dask_chunks)
            return da.from_array(np.asarray(arr, dtype=bool), chunks=chunks)
        chunks = _infer_chunks_like(data, np.shape(mask), dask_chunks)
        return da.from_array(np.asarray(mask, dtype=bool), chunks=chunks)

    # xr.DataArray backed by NumPy
    if return_kind == "dataarray-numpy":
        if isinstance(mask, xr.DataArray):
            arr = mask.data
            if getattr(arr, "__module__", "").startswith("dask"):
                arr = arr.astype(bool).compute()
            da_out = xr.DataArray(
                np.asarray(arr, dtype=bool),
                dims=getattr(mask, "dims", getattr(data, "dims", ("y", "x"))),
                coords=getattr(mask, "coords", getattr(data, "coords", None)),
            )
            if creation is not None:
                da_out = da_out.assign_attrs(
                    {**getattr(mask, "attrs", {}), "creation": creation}
                )
            return da_out
        if getattr(mask, "__module__", "").startswith("dask"):
            arr = mask.astype(bool).compute()
        else:
            arr = np.asarray(mask, dtype=bool)
        dims = (
            getattr(data, "dims", ("y", "x"))
            if isinstance(data, xr.DataArray)
            else ("y", "x")
        )
        coords = (
            getattr(data, "coords", None) if isinstance(data, xr.DataArray) else None
        )
        da_out = xr.DataArray(arr, dims=dims, coords=coords)
        if creation is not None:
            da_out = da_out.assign_attrs({"creation": creation})
        return da_out

    # xr.DataArray backed by Dask (default)
    if isinstance(mask, xr.DataArray):
        arr = mask.data
        if getattr(arr, "__module__", "").startswith("dask"):
            da_out = mask.astype(bool)
            if creation is not None:
                da_out = da_out.assign_attrs(
                    {**getattr(mask, "attrs", {}), "creation": creation}
                )
            return da_out
        # numpy-backed xarray → wrap into dask with inferred chunks
        chunks = _infer_chunks_like(data, mask.shape, dask_chunks)
        darr = da.from_array(np.asarray(arr, dtype=bool), chunks=chunks)
        da_out = xr.DataArray(darr, dims=mask.dims, coords=mask.coords)
        if creation is not None:
            da_out = da_out.assign_attrs(
                {**getattr(mask, "attrs", {}), "creation": creation}
            )
        return da_out

    # ndarray/dask → dask-backed DataArray
    if getattr(mask, "__module__", "").startswith("dask"):
        darr = mask.astype(bool)  # type: ignore[assignment]
    else:
        chunks = _infer_chunks_like(data, np.shape(mask), dask_chunks)
        darr = da.from_array(np.asarray(mask, dtype=bool), chunks=chunks)
    dims = (
        getattr(data, "dims", ("y", "x"))
        if isinstance(data, xr.DataArray)
        else ("y", "x")
    )
    coords = getattr(data, "coords", None) if isinstance(data, xr.DataArray) else None
    da_out = xr.DataArray(darr, dims=dims, coords=coords)
    if creation is not None:
        da_out = da_out.assign_attrs({"creation": creation})
    return da_out


def _infer_chunks_like(
    data: Any, shape: Tuple[int, ...], dask_chunks: Optional[Tuple[int, ...]]
) -> Tuple[int, ...]:
    if dask_chunks is not None:
        return dask_chunks
    # try to mirror data's chunking
    if isinstance(data, xr.DataArray) and hasattr(data.data, "chunks"):
        try:
            ch = tuple(
                int(c)
                for c in (
                    sum(
                        tuple(
                            (t if isinstance(t, tuple) else (t,))
                            for t in data.data.chunks
                        ),
                        (),
                    )
                )
            )
            if len(ch) == len(shape):
                return ch
        except Exception:
            pass
    if hasattr(data, "chunks"):
        try:
            ch = tuple(
                int(c)
                for c in (
                    sum(
                        tuple(
                            (t if isinstance(t, tuple) else (t,)) for t in data.chunks
                        ),
                        (),
                    )
                )
            )
            if len(ch) == len(shape):
                return ch
        except Exception:
            pass
    # fallback heuristic: ~256k elements per chunk (2D only); else full shape
    if len(shape) == 2:
        ny, nx = shape
        tgt = max(1, int(256_000 // max(1, nx)))
        return (min(ny, tgt), nx)
    return shape


def _build_creation_for_expression(expr: str, env: Mapping[str, ArrayLike]) -> str:
    """
    Return an expanded textual expression where each Name is replaced by the
    underlying mask's 'creation' attribute (if present), parenthesized.
    Falls back to the original identifier when no creation is available.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:  # pragma: no cover — unreachable via public API (parse
        # keep original text if it can't be parsed
        return expr

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return expr
        if isinstance(node, ast.BoolOp):
            # 'and'/'or' are not supported; keep original text
            return expr

    def _emit(node: ast.AST) -> str:
        if isinstance(node, ast.Expression):
            return _emit(node.body)
        if isinstance(node, ast.Name):
            value = env.get(node.id, None)
            # Prefer DataArray with a 'creation' attribute
            if isinstance(value, xr.DataArray):
                c = value.attrs.get("creation")
                if isinstance(c, str) and c:
                    return f"({c})"
            # Fallback to the identifier
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY_OPS):
            return f"~({_emit(node.operand)})"
        if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN_OPS):
            op = (
                "&"
                if isinstance(node.op, ast.BitAnd)
                else "|" if isinstance(node.op, ast.BitOr) else "^"
            )
            left = _emit(node.left)
            right = _emit(node.right)
            return f"({left}) {op} ({right})"
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return "True" if node.value else "False"
        # Unknown node → keep original textual expr
        return expr

    return _emit(tree)


# ---------------------------------------------------------------------------
# Public helper: combine two masks and preserve provenance
# ---------------------------------------------------------------------------
def combine_with_creation(
    a: xr.DataArray,
    op: str,
    b: xr.DataArray,
    *,
    template: ArrayLike | None = None,
    return_kind: ReturnKind = "dataarray-dask",
    dask_chunks: Optional[Tuple[int, ...]] = None,
    creation_hint: Optional[str] = None,
) -> xr.DataArray:
    """
    Combine two boolean DataArrays with a bitwise op ('|', '&', '^') and attach a
    human-readable creation string derived from the inputs' provenance.
    """
    if op not in {"|", "&", "^"}:
        raise ValueError("op must be one of '|', '&', '^'")
    L = a.astype(bool)
    R = b.astype(bool)
    combined = (L | R) if op == "|" else (L & R) if op == "&" else (L ^ R)
    c1 = a.attrs.get("creation") or (
        a.name if isinstance(a.name, str) and a.name else "mask_a"
    )
    c2 = b.attrs.get("creation") or (
        b.name if isinstance(b.name, str) and b.name else "mask_b"
    )
    creation = creation_hint if creation_hint is not None else f"({c1}) {op} ({c2})"
    # carry hints for optional auto-merge paths if users pass `combined` directly later
    combined = combined.assign_attrs(
        {
            "creation_a": c1,
            "creation_b": c2,
            "creation_op": op,
        }
    )
    tmpl = (
        template if template is not None else (a if isinstance(a, xr.DataArray) else b)
    )
    # If template has different dim names but identical shape, rename to template dims
    # to avoid dim union (e.g., ('row','col','y','x')). On rename failure, force-wrap
    # using template dims/coords as a defensive fallback.
    if isinstance(tmpl, xr.DataArray) and isinstance(combined, xr.DataArray):
        if combined.shape == tmpl.shape and combined.dims != tmpl.dims:
            try:
                combined = combined.rename(
                    {old: new for old, new in zip(combined.dims, tmpl.dims)}
                )
            except Exception:
                try:
                    arr = combined.data if hasattr(combined, "data") else combined
                    combined = xr.DataArray(arr, dims=tmpl.dims, coords=tmpl.coords)
                except Exception:
                    pass
    return select_mask(
        tmpl,
        select=combined,
        return_kind=return_kind,
        dask_chunks=dask_chunks,
        creation_hint=creation,
    )
