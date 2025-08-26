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
  token followed by ``[[`` (e.g., ``box[[...]]``). Pixel axes follow CASA convention
  (1-based, bottom-left).
* You can also pass a CRTF file path either as a backticked string (`` `path/file.crtf` ``)
  or a ``pathlib.Path``; the file contents are read and parsed.

Notes
-----
* Masks treat ``NaN`` as ``False``.
* Only bitwise operators are supported in expressions; ``and``/``or`` are rejected.
* Unknown names raise ``KeyError`` listing available mask names.
"""
from __future__ import annotations

from typing import Mapping, Any, Union
from pathlib import Path

import math
import ast
import re

import numpy as np
import xarray as xr

ArrayLike = Union[np.ndarray, xr.DataArray]
__all__ = ["select_mask", "apply_select"]

def apply_select(data: ArrayLike, select: Any | None = None, mask_source: Any | None = None) -> ArrayLike:
    """Return ``data`` with ``select`` applied.

    ``data`` must be a ``numpy.ndarray`` or ``xarray.DataArray``.
    For ``xarray.DataArray``, the mask is broadcast/aligned by dimension names.
    For ``numpy.ndarray``, the mask must be broadcastable by shape.
    """
    mask = select_mask(data, select=select, mask_source=mask_source)
    if isinstance(data, xr.DataArray):
        return data.where(mask)
    return np.where(np.asarray(mask, dtype=bool), data, np.nan)


def select_mask(data: ArrayLike, select: Any | None = None, mask_source: Any | None = None) -> ArrayLike:
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
    if select is None:
        return _all_true_mask_like(data)

    # Boolean array-like
    if isinstance(select, (np.ndarray, xr.DataArray)):
        return _align_bool_mask_to_data(select, data)

    # String/Path: backticked file or Path → load as CRTF; else treat as text.
    if isinstance(select, (str, Path)):
        s_file = _maybe_read_crtf_from_path(select)
        if s_file is not None:
            # If the user provided a file, it's CRTF by definition; parse directly.
            m = _crtf_pixel_mask(data, s_file)
            return _align_bool_mask_to_data(m, data)
        # Otherwise, handle plain strings (CRTF text or named-mask expression)
        if isinstance(select, Path):
            raise FileNotFoundError(f"CRTF file not found: {select}")
        s = select.strip()
        if _looks_like_crtf_pixel(s):
            m = _crtf_pixel_mask(data, s)
            return _align_bool_mask_to_data(m, data)
        env = _build_mask_env(mask_source or (data if isinstance(data, xr.Dataset) else {}))
        expr_mask = _eval_mask_expr(s, env)
        return _align_bool_mask_to_data(expr_mask, data)
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
    return None

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
        m = mask if isinstance(mask, xr.DataArray) else xr.DataArray(mask)
        # why: NaNs must become False before bool-cast
        if np.issubdtype(m.dtype, np.floating):
            m = m.fillna(False)
        m = m.astype(bool)
        try:
            # align by named dimensions, broadcasting as needed
            m = m.broadcast_like(data)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Mask is not broadcastable to data dimensions") from exc
        return m

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
    ast.Constant,   # allow literal True/False
    ast.BoolOp,     # reject at runtime if it's 'and'/'or'
    # Py3.12 walks operator/context nodes too:
    ast.operator,   # BitAnd/BitOr/BitXor
    ast.unaryop,    # Invert
    ast.Load,       # Name context
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
            raise ValueError(f"Expression contains an unsupported construct: {type(node).__name__}")
        if isinstance(node, ast.BoolOp):
            # Explicitly forbid 'and'/'or'
            raise ValueError("Use '&' and '|' instead of 'and'/'or' in selection expressions")

    def _eval(node: ast.AST) -> ArrayLike:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Name):
            try:
                return env[node.id]
            except KeyError as exc:
                available = ", ".join(sorted(env.keys()))
                raise KeyError(f"Unknown mask name: {node.id}. Available: {available}") from exc
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

def _crtf_pixel_mask(data: ArrayLike, text: str) -> ArrayLike:
    """Parse a CRTF pixel string (single or multi-line) into a boolean mask.

    Combination semantics per line: leading '+' (OR, default) or '-' (NOT/subtract).
    """

    ny, nx = (data.shape if not isinstance(data, xr.DataArray) else data.shape)
    # CRTF 'pix' uses pixel *indices*. Always evaluate in pixel-index space (1..N),
    # independent of any world-coordinate arrays attached to `data`.
    c = np.arange(nx, dtype=float) + 1.0  # x pixel centers: 1..nx
    r = np.arange(ny, dtype=float) + 1.0  # y pixel centers: 1..ny
    X, Y = np.meshgrid(c, r)

    acc = np.zeros((ny, nx), dtype=bool)
    lines = [ln.strip() for ln in re.split(r"[\n;]+", text) if ln.strip() and not ln.strip().startswith("#")]
    for line in lines:
        if line.lower().startswith("global"):
            continue
        flag = "+"
        if line[0] in "+-":
            flag, line = line[0], line[1:].lstrip()
        shape, payload = _split_shape_payload(line)
        mask = _rasterize_shape(shape, payload, X, Y)
        if flag == "+":
            acc |= mask
        else:
            acc &= ~mask

    return xr.DataArray(acc, dims=getattr(data, "dims", ("y", "x"))) if isinstance(data, xr.DataArray) else acc

def _split_shape_payload(line: str) -> tuple[str, str]:
    m = re.match(r"^([A-Za-z]+)\s*(\[\[.*)$", line)
    if not m:
        raise ValueError(f"Invalid CRTF line: {line!r}")
    return m.group(1).lower(), m.group(2)

_NUM = r"[-+]?\d+(?:\.\d+)?"
_UNIT_NUM = rf"{_NUM}(?:\s*(?:pix|deg|rad))?"
_PAIR = rf"\[\s*({_UNIT_NUM})\s*,\s*({_UNIT_NUM})\s*\]"

def _parse_units_val(tok: str) -> tuple[float, str | None]:
    m = re.match(rf"^\s*({_NUM})\s*(pix|deg|rad)?\s*$", tok)
    if not m:
        raise ValueError(f"Invalid numeric token: {tok!r}")
    val = float(m.group(1)); unit = m.group(2)
    return val, unit

def _strip_brackets(s: str) -> str:
    if not (s.startswith("[[") and s.endswith("]")):
        raise ValueError("CRTF payload must start with '[[ ... ]]'")
    return s[1:-1]

def _rasterize_shape(shape: str, payload: str, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    inner = _strip_brackets(payload).strip()
    parts = _smart_split_pairs(inner)
    if shape == "box":
        p1x, p1y = _parse_pair(parts[0]); p2x, p2y = _parse_pair(parts[1])
        x1, x2 = sorted([p1x, p2x]); y1, y2 = sorted([p1y, p2y])
        return (X >= x1) & (X <= x2) & (Y >= y1) & (Y <= y2)
    if shape == "centerbox":
        cx, cy = _parse_pair(parts[0]); w, h = _parse_pair(parts[1])
        hx, hy = w / 2.0, h / 2.0
        return (np.abs(X - cx) <= hx) & (np.abs(Y - cy) <= hy)
    if shape == "rotbox":
        cx, cy = _parse_pair(parts[0]); w, h = _parse_pair(parts[1]); ang = _parse_angle(parts[2])
        hx, hy = w / 2.0, h / 2.0
        xrp, yrp = _rotate_about(X, Y, cx, cy, -ang)
        return (np.abs(xrp - cx) <= hx) & (np.abs(yrp - cy) <= hy)
    if shape == "circle":
        cx, cy = _parse_pair(parts[0]); r, _ = _parse_units_val(parts[1])
        return ((X - cx) ** 2 + (Y - cy) ** 2) <= (r ** 2 + 1e-9)
    if shape == "annulus":
        cx, cy = _parse_pair(parts[0])
        r1, _ = _parse_units_val(parts[1].strip()[1:-1].split(",")[0])
        r2, _ = _parse_units_val(parts[1].strip()[1:-1].split(",")[1])
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        return (d2 >= r1 ** 2) & (d2 <= r2 ** 2)
    if shape == "ellipse":
        cx, cy = _parse_pair(parts[0])
        a, _ = _parse_units_val(parts[1].strip()[1:-1].split(",")[0])
        b, _ = _parse_units_val(parts[1].strip()[1:-1].split(",")[1])
        pa = _parse_angle(parts[2])
        xp, yp = _rotate_about(X, Y, cx, cy, -pa)
        return ((xp - cx) / a) ** 2 + ((yp - cy) / b) ** 2 <= 1.0 + 1e-9
    if shape == "poly":
        pts = [_parse_pair(p) for p in parts]
        return _point_in_poly(X, Y, pts)
    raise ValueError(f"Unsupported CRTF shape: {shape}")

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

def _parse_pair(pair_token: str) -> tuple[float, float]:
    m = re.match(rf"^\s*\[\s*({_UNIT_NUM})\s*,\s*({_UNIT_NUM})\s*\]\s*$", pair_token)
    if not m: raise ValueError(f"Invalid pair token: {pair_token!r}")
    x, _ux = _parse_units_val(m.group(1)); y, _uy = _parse_units_val(m.group(2))
    return float(x), float(y)

def _parse_angle(token: str) -> float:
    v, unit = _parse_units_val(token)
    return float(v) if unit == "rad" else math.radians(float(v))

def _rotate_about(X: np.ndarray, Y: np.ndarray, cx: float, cy: float, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    x = X - cx; y = Y - cy
    xr = x * ca - y * sa; yr = x * sa + y * ca
    return xr + cx, yr + cy

def _point_in_poly(X: np.ndarray, Y: np.ndarray, pts: list[tuple[float, float]]) -> np.ndarray:
    # Ray casting algorithm, vectorized
    x = X.ravel(); y = Y.ravel()
    inside = np.zeros_like(x, dtype=bool)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    n = len(pts); j = n - 1
    for i in range(n):
        xi, yi = xs[i], ys[i]; xj, yj = xs[j], ys[j]
        cond = ((yi > y) != (yj > y))
        xints = (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        inside ^= cond & (x < xints)
        j = i
    return inside.reshape(X.shape)
