"""Generic selection layer for applications.

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

Notes
-----
* Masks treat ``NaN`` as ``False``.
* Only bitwise operators are supported in expressions; ``and``/``or`` are rejected.
* Unknown names raise ``KeyError`` listing available mask names.
"""
from __future__ import annotations

from typing import Mapping, Any, Union
import ast

import numpy as np
import xarray as xr

ArrayLike = Union[np.ndarray, xr.DataArray]


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

    if isinstance(select, (np.ndarray, xr.DataArray)):
        return _align_bool_mask_to_data(select, data)

    if isinstance(select, str):
        env = _build_mask_env(mask_source or (data if isinstance(data, xr.Dataset) else {}))
        expr_mask = _eval_mask_expr(select, env)
        return _align_bool_mask_to_data(expr_mask, data)

    raise TypeError(
        "Unsupported select type. Expected None, boolean array-like, or expression string."
    )


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
    ast.BoolOp,    # reject at runtime if it's 'and'/'or'
    # Py3.12 walks operator/context nodes too:
    ast.operator,  # BitAnd/BitOr/BitXor
    ast.unaryop,   # Invert
    ast.Load,      # Name context
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
