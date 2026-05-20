"""Generic selection layer for applications + CASA region (pixel) support.

Supported ``select`` forms (basic):
- ``None`` → keep everything.
- A boolean array-like (``xarray.DataArray`` or ``numpy.ndarray``), already aligned
  or broadcastable to the target ``data``.
- A plain string name that resolves exactly from ``mask_source`` before any CRTF
  file or text parsing is attempted.
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

* You can also pass a CRTF file path either as a backticked string (`` `path/file.crtf` ``),
  a plain string path such as ``"path/file.crtf"``, or a ``pathlib.Path``; the file
  contents are read and parsed.

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
import warnings

import numpy as np
import xarray as xr

ArrayLike = Union[np.ndarray, xr.DataArray]
__all__ = ["select_mask", "apply_select", "combine_with_creation"]


def _reject_dataset_input(data: Any) -> None:
    """Raise ``TypeError`` when *data* is an ``xr.Dataset``.

    Parameters
    ----------
    data : any
        The ``data`` argument passed to ``select_mask`` or ``apply_select``.

    Raises
    ------
    TypeError
        If *data* is an ``xr.Dataset``, with a message pointing users to pass
        a specific data variable (e.g. ``xds.SKY``) instead.
    """
    if isinstance(data, xr.Dataset):
        raise TypeError(
            "CRTF selection expects a DataArray (e.g. xds.SKY), not a Dataset. "
            "Pass the specific data variable you want to mask."
        )


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
    _reject_dataset_input(data)
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
        ``None`` | boolean array-like | exact mask name | string expression over named masks.
    mask_source:
        Mapping or ``xr.Dataset`` that provides named masks for expressions.
        If ``None`` and ``data`` is an ``xr.Dataset``, that dataset is used.

    Returns
    -------
    ``xr.DataArray`` if ``data`` is a ``DataArray``; otherwise ``np.ndarray``.
    """
    _reject_dataset_input(data)
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
    # String exact-name precedence: resolve a matching named mask before any file
    # or CRTF-text parsing.
    if isinstance(select, str):
        exact_name = _maybe_resolve_named_mask_name(select, mask_source)
        if exact_name is not None:
            aligned = _align_bool_mask_to_data(exact_name, data)
            creation_str = creation_hint if creation_hint is not None else select
            return _coerce_return_kind(
                aligned, data, return_kind, dask_chunks, creation=creation_str
            )

    # String/Path: file → load as CRTF; else treat as text.
    if isinstance(select, (str, Path)):
        s_file = _maybe_read_crtf_from_path(select)
        if s_file is not None:
            # If the user provided a file, it's CRTF by definition; parse directly.
            m = _crtf_mask(data, s_file, lazy=_want_dask(return_kind))
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
            m = _crtf_mask(data, s, lazy=_want_dask(return_kind))
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
    """Return CRTF file contents for supported path inputs.

    Parameters
    ----------
    sel : Any
        Candidate CRTF source. Supported path forms are ``pathlib.Path``
        instances, backticked path strings, and plain strings that point to an
        existing file.

    Returns
    -------
    str or None
        File contents for supported existing CRTF paths, otherwise ``None`` for
        inputs that should continue through inline CRTF/expression parsing.

    Raises
    ------
    FileNotFoundError
        If *sel* is a missing ``Path``, a missing backticked path, or a plain
        string that looks like a CRTF file path but does not exist.
    """
    if isinstance(sel, Path):
        if not sel.is_file():
            raise FileNotFoundError(f"CRTF file not found: {sel}")
        return sel.read_text(encoding="utf-8")
    if isinstance(sel, str):
        s = sel.strip()
        m = re.fullmatch(r"`([^`]+)`", s)
        if m:
            p = Path(m.group(1))
            if not p.is_file():
                raise FileNotFoundError(f"CRTF file not found: {p}")
            return p.read_text(encoding="utf-8")

        p = Path(s)
        if p.is_file():
            return p.read_text(encoding="utf-8")
        if _looks_like_plain_crtf_path(s):
            raise FileNotFoundError(f"CRTF file not found: {p}")
        return None
    return None  # pragma: no cover


def _looks_like_plain_crtf_path(text: str) -> bool:
    """Return whether a plain string should be interpreted as a CRTF path."""
    stripped = text.strip()
    if (
        not stripped
        or "\n" in stripped
        or "\r" in stripped
        or stripped.startswith("#CRTF")
    ):
        return False
    if _looks_like_crtf_pixel(stripped):
        return False
    return (
        Path(stripped).suffix.lower() == ".crtf" or "/" in stripped or "\\" in stripped
    )


def _maybe_resolve_named_mask_name(
    select: str, mask_source: Any | None
) -> ArrayLike | None:
    """Return an exact-name mask from ``mask_source`` when available."""
    if mask_source is None:
        return None

    if isinstance(mask_source, xr.Dataset):
        value = mask_source.data_vars.get(select)
        if value is None or not _is_boolish(value):
            return None
        return _to_bool(value)

    if isinstance(mask_source, Mapping):
        for key, value in mask_source.items():
            if str(key) == select and _is_boolish(value):
                return _to_bool(value)
        return None

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


def _parse_keyword_assignments(text: str) -> dict[str, str]:
    """Parse a comma-separated list of CRTF key=value assignments.

    Parameters
    ----------
    text : str
        Raw text such as ``"corr=[I,Q], range=[1GHz, 2GHz]"`` or the empty
        string.

    Returns
    -------
    dict[str, str]
        Mapping of key (stripped) to raw value string (stripped). Empty text
        returns an empty dict. Values may themselves contain brackets.

    Notes
    -----
    Commas that appear inside brackets are not treated as assignment separators,
    so values like ``"[1GHz, 2GHz]"`` are kept intact.
    """
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    result: dict[str, str] = {}
    for part in parts:
        if not part:
            continue
        eq = part.find("=")
        if eq < 0:
            raise ValueError(
                f"Expected 'key=value' in CRTF keyword assignments, got {part!r}"
            )
        key = part[:eq].strip()
        val = part[eq + 1 :].strip()
        result[key] = val
    return result


def _parse_crtf_globals(line: str) -> dict[str, str]:
    """Parse a CRTF ``global ...`` line into a dict of keyword assignments.

    Parameters
    ----------
    line : str
        Full global line, e.g. ``"global corr=[I,Q], coordsys=world"``.

    Returns
    -------
    dict[str, str]
        Keyword assignments extracted from the line; empty dict if none are
        present.
    """
    rest = re.sub(r"(?i)^\s*global\s*", "", line).strip()
    if not rest:
        return {}
    return _parse_keyword_assignments(rest)


def _extract_bracket_group(s: str) -> tuple[str, str]:
    """Extract the leading ``[[...]]`` group from a CRTF shape argument string.

    Parameters
    ----------
    s : str
        String starting with ``"[["`` containing shape arguments and optionally
        followed by trailing key=value pairs.

    Returns
    -------
    tuple[str, str]
        ``(group, remainder)`` where *group* is the matched ``[[...]]`` string
        (depth returns to zero at the last ``]``) and *remainder* is everything
        after the closing bracket.

    Raises
    ------
    ValueError
        If *s* does not start with ``"[["`` or contains unmatched brackets.
    """
    if not s.startswith("[["):
        raise ValueError(f"Expected '[[' at start of CRTF shape payload, got {s!r}")
    depth = 0
    for i, ch in enumerate(s):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[: i + 1], s[i + 1 :]
    raise ValueError(f"Unmatched brackets in CRTF shape payload: {s!r}")


def _parse_crtf_line(line: str) -> tuple[str, str, str, dict[str, str]]:
    """Parse a single CRTF region line into its structural components.

    Parameters
    ----------
    line : str
        A single non-comment, non-global CRTF line, optionally prefixed with
        ``'+'`` or ``'-'`` and optionally followed by trailing key=value pairs
        after the shape's ``[[...]]`` group.
        Example: ``"+box[[0pix,0pix],[10pix,10pix]], corr=[I,Q]"``.

    Returns
    -------
    tuple[str, str, str, dict[str, str]]
        ``(flag, shape, payload, kwargs)`` where *flag* is ``'+'`` or ``'-'``,
        *shape* is the lowercase shape name (e.g. ``'box'``), *payload* is the
        ``[[...]]`` bracket group string, and *kwargs* is a dict of trailing
        key=value assignments (empty dict if none are present).

    Raises
    ------
    ValueError
        If the line does not match the expected CRTF region syntax.
    """
    flag = "+"
    rest = line
    if rest and rest[0] in "+-":
        flag, rest = rest[0], rest[1:].lstrip()
    m = re.match(r"^([A-Za-z]+)\s*(\[\[.*)$", rest)
    if not m:
        raise ValueError(f"Invalid CRTF line: {line!r}")
    shape = m.group(1).lower()
    after_shape = m.group(2)  # starts with '[['
    payload, remainder = _extract_bracket_group(after_shape)
    kwargs = _parse_keyword_assignments(remainder)
    return flag, shape, payload, kwargs


# CRTF keywords that imply frame / velocity-convention conversion.  These are
# not supported in v1 and raise NotImplementedError when encountered so users
# know their coordinate intent was not silently dropped.
_REJECTED_CRTF_KEYWORDS = frozenset({"coord", "frame", "veltype", "restfreq"})

# Visualization-only CRTF keywords that carry no mask semantics.  These are
# silently ignored so that real-world CRTF files with display annotations can
# be passed in without error.
_VIZ_CRTF_KEYWORDS = frozenset(
    {
        "color",
        "linewidth",
        "linestyle",
        "symsize",
        "symthick",
        "font",
        "fontsize",
        "fontstyle",
        "usetex",
        "labelpos",
        "labelcolor",
        "labeloff",
        "label",
    }
)


def _reject_frame_keywords(kwargs: dict[str, str], context: str = "CRTF line") -> None:
    """Raise ``NotImplementedError`` for any frame-conversion keyword in *kwargs*.

    Parameters
    ----------
    kwargs : dict[str, str]
        Parsed keyword assignments from a CRTF line or global block.
    context : str
        Human-readable label used in the error message (e.g. ``"CRTF global"``).

    Raises
    ------
    NotImplementedError
        If any of ``coord=``, ``frame=``, ``veltype=``, ``restfreq=`` appear.
        Frame and velocity-convention conversions are not supported in v1;
        world coordinates must be specified in the image's native frame and
        convention.
    """
    for key in _REJECTED_CRTF_KEYWORDS:
        if key in kwargs:
            raise NotImplementedError(
                f"CRTF keyword '{key}=' is not supported (found in {context}). "
                "Frame and velocity-convention conversions are not implemented in v1. "
                "Specify coordinates in the image's native frame and convention."
            )


# ---------------------------------------------------------------------------
# range= / corr= / time= token parsers and mask builders
# ---------------------------------------------------------------------------

_FREQ_SCALE: dict[str, float] = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}

# Canonical Stokes/polarization names accepted by corr=
_VALID_STOKES: frozenset[str] = frozenset(
    {
        "I",
        "Q",
        "U",
        "V",
        "RR",
        "RL",
        "LR",
        "LL",
        "XX",
        "XY",
        "YX",
        "YY",
        "RX",
        "RY",
        "LX",
        "LY",
        "XR",
        "XL",
        "YR",
        "YL",
        "PP",
        "PQ",
        "QP",
        "QQ",
        "RCircular",
        "LCircular",
        "Linear",
        "Ptotal",
        "Plinear",
        "PFtotal",
        "PFlinear",
        "Pangle",
    }
)
_VALID_STOKES_LOWER: dict[str, str] = {s.lower(): s for s in _VALID_STOKES}

_NUM_RE = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _detect_range_family(token: str) -> str:
    """Detect the ``range=`` token family from a single token string.

    Parameters
    ----------
    token : str
        A single value token from ``range=[a, b]``, e.g. ``'1.4GHz'``,
        ``'100km/s'``, ``'5chan'``.

    Returns
    -------
    str
        One of ``'frequency'``, ``'velocity'``, or ``'channel'``.

    Raises
    ------
    ValueError
        If the token unit cannot be recognized.
    """
    t = token.strip()
    if re.search(r"(?i)(ghz|mhz|khz|hz)$", t):
        return "frequency"
    if re.search(r"(?i)km/s|m/s", t):
        return "velocity"
    if re.search(r"(?i)chan(nel)?$", t):
        return "channel"
    raise ValueError(
        f"Cannot detect range= family from token {token!r}. "
        "Expected a frequency (Hz/kHz/MHz/GHz), velocity (m/s or km/s), "
        "or channel (chan/channel) token."
    )


def _parse_freq_token(token: str) -> float:
    """Parse a frequency token and return the value in Hz.

    Parameters
    ----------
    token : str
        Token such as ``'1.4GHz'``, ``'1400MHz'``, ``'1.4e9Hz'``.

    Returns
    -------
    float
        Frequency in Hz.
    """
    m = re.match(rf"^\s*({_NUM_RE})\s*(Hz|kHz|MHz|GHz)\s*$", token, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot parse frequency token {token!r}. "
            "Expected a number followed by Hz, kHz, MHz, or GHz."
        )
    return float(m.group(1)) * _FREQ_SCALE[m.group(2).lower()]


def _parse_velocity_token(token: str) -> float:
    """Parse a velocity token and return the value in m/s.

    Parameters
    ----------
    token : str
        Token such as ``'100km/s'``, ``'-50m/s'``.

    Returns
    -------
    float
        Velocity in m/s.
    """
    m = re.match(rf"^\s*({_NUM_RE})\s*(km/s|m/s)\s*$", token, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot parse velocity token {token!r}. "
            "Expected a number followed by m/s or km/s."
        )
    val = float(m.group(1))
    return val * 1000.0 if m.group(2).lower() == "km/s" else val


def _parse_channel_token(token: str) -> int:
    """Parse a channel token and return the integer channel index.

    Parameters
    ----------
    token : str
        Token such as ``'5chan'``, ``'5channel'``, ``'5'``.

    Returns
    -------
    int
        Zero-based channel index.
    """
    m = re.match(r"^\s*(\d+)\s*(?:chan(?:nel)?)?\s*$", token, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot parse channel token {token!r}. "
            "Expected an integer optionally followed by 'chan' or 'channel'."
        )
    return int(m.group(1))


def _detect_time_family(token: str) -> tuple[str, str]:
    """Detect the ``time=`` token family and return the cleaned value string.

    Parameters
    ----------
    token : str
        Single time bound token from ``time=[a, b]``.

    Returns
    -------
    tuple[str, str]
        ``(family, value_str)`` where *family* is one of ``'mjd'``, ``'jd'``,
        ``'iso'`` and *value_str* is the numeric or string value stripped of
        suffix and surrounding quotes.

    Raises
    ------
    ValueError
        If the token cannot be classified.
    """
    t = token.strip()
    # ISO: single- or double-quoted string
    if (t.startswith("'") and t.endswith("'")) or (
        t.startswith('"') and t.endswith('"')
    ):
        return "iso", t[1:-1]
    # MJD suffix
    if re.search(r"(?i)mjd$", t):
        return "mjd", re.sub(r"(?i)mjd$", "", t).strip()
    # JD suffix
    if re.search(r"(?i)jd$", t):
        return "jd", re.sub(r"(?i)jd$", "", t).strip()
    # 'd' suffix (e.g. 60000.0d)
    if re.search(r"(?i)d$", t) and re.match(r"^\s*[-+]?\d", t):
        return "mjd", re.sub(r"(?i)d$", "", t).strip()
    # Bare number — interpret as MJD
    if re.match(rf"^\s*{_NUM_RE}\s*$", t):
        return "mjd", t.strip()
    raise ValueError(
        f"Cannot detect time= family from token {token!r}. "
        "Expected MJD (bare number or <n>d/<n>mjd), JD (<n>jd), "
        "or ISO ('YYYY-MM-DDTHH:MM:SS')."
    )


def _build_range_mask(
    data: xr.DataArray, kwargs: dict[str, str]
) -> "xr.DataArray | None":
    """Build a 1-D boolean mask on the ``frequency`` dim from a ``range=`` kwarg.

    Parameters
    ----------
    data : xr.DataArray
        The data array; must carry the required coord (``frequency`` or
        ``velocity``) — gating ensures this before this function is called.
    kwargs : dict[str, str]
        Parsed CRTF keyword assignments for the current line.

    Returns
    -------
    xr.DataArray or None
        Boolean mask with dim ``frequency``, or ``None`` if ``range=`` is
        absent from *kwargs*.

    Raises
    ------
    ValueError
        If the two range tokens belong to different families, or if the raw
        value is malformed.
    """
    raw = kwargs.get("range")
    if raw is None:
        return None
    raw = raw.strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError(f"range= value must be bracketed, got {raw!r}")
    parts = _smart_split_pairs(raw[1:-1])
    if len(parts) != 2:
        raise ValueError(f"range= requires exactly two values, got {raw!r}")
    lo_tok, hi_tok = parts[0].strip(), parts[1].strip()
    lo_fam = _detect_range_family(lo_tok)
    hi_fam = _detect_range_family(hi_tok)
    if lo_fam != hi_fam:
        raise ValueError(
            f"range= token family mismatch: {lo_tok!r} is {lo_fam}, "
            f"{hi_tok!r} is {hi_fam}. Both tokens must use the same units."
        )
    freq_dim = data.coords["frequency"].dims[0]
    if lo_fam == "frequency":
        lo, hi = _parse_freq_token(lo_tok), _parse_freq_token(hi_tok)
        freq_vals = data.coords["frequency"].values.astype(float)
        mask = xr.DataArray((freq_vals >= lo) & (freq_vals <= hi), dims=[freq_dim])
        _warn_if_axis_selection_empty("range", raw, "frequency", mask)
        return mask
    if lo_fam == "velocity":
        lo, hi = _parse_velocity_token(lo_tok), _parse_velocity_token(hi_tok)
        vel_vals = data.coords["velocity"].values.astype(float)
        mask = xr.DataArray((vel_vals >= lo) & (vel_vals <= hi), dims=[freq_dim])
        _warn_if_axis_selection_empty("range", raw, "velocity", mask)
        return mask
    # channel
    lo_ch = _parse_channel_token(lo_tok)
    hi_ch = _parse_channel_token(hi_tok)
    lo_ch, hi_ch = sorted([lo_ch, hi_ch])
    n = len(data.coords["frequency"])
    mask_vals = np.zeros(n, dtype=bool)
    mask_vals[lo_ch : hi_ch + 1] = True
    mask = xr.DataArray(mask_vals, dims=[freq_dim])
    _warn_if_axis_selection_empty("range", raw, "channel", mask)
    return mask


def _build_corr_mask(
    data: xr.DataArray, kwargs: dict[str, str]
) -> "xr.DataArray | None":
    """Build a 1-D boolean mask on the ``polarization`` dim from a ``corr=`` kwarg.

    Parameters
    ----------
    data : xr.DataArray
        The data array; must carry the ``polarization`` coord — gating ensures
        this before this function is called.
    kwargs : dict[str, str]
        Parsed CRTF keyword assignments for the current line.

    Returns
    -------
    xr.DataArray or None
        Boolean mask with dim ``polarization``, or ``None`` if ``corr=`` is
        absent from *kwargs*.

    Raises
    ------
    ValueError
        If any polarization token is not in the supported Stokes set.
    """
    raw = kwargs.get("corr")
    if raw is None:
        return None
    raw = raw.strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError(f"corr= value must be bracketed, got {raw!r}")
    tokens = [t.strip() for t in raw[1:-1].split(",") if t.strip()]
    canonical: list[str] = []
    for tok in tokens:
        c = _VALID_STOKES_LOWER.get(tok.lower())
        if c is None:
            valid = ", ".join(sorted(_VALID_STOKES))
            raise ValueError(
                f"Unknown polarization '{tok}' in corr=. Valid names: {valid}"
            )
        canonical.append(c)
    pol_coord = data.coords["polarization"]
    pol_dim = pol_coord.dims[0]
    pol_vals = pol_coord.values
    mask_vals = np.isin(pol_vals, canonical)
    return xr.DataArray(mask_vals, dims=[pol_dim])


def _build_time_mask(
    data: xr.DataArray, kwargs: dict[str, str]
) -> "xr.DataArray | None":
    """Build a 1-D boolean mask on the ``time`` dim from a ``time=`` kwarg.

    Parameters
    ----------
    data : xr.DataArray
        The data array; must carry the ``time`` coord — gating ensures this
        before this function is called.
    kwargs : dict[str, str]
        Parsed CRTF keyword assignments for the current line.

    Returns
    -------
    xr.DataArray or None
        Boolean mask with dim ``time``, or ``None`` if ``time=`` is absent
        from *kwargs*.

    Raises
    ------
    ValueError
        If the two time tokens belong to different families, or if parsing
        fails.

    Notes
    -----
    Time values are converted to MJD days and compared against the ``time``
    coord (which is expected to carry MJD days, as per xradio convention).
    The time scale is read from ``time.attrs['scale']`` (defaulting to
    ``'utc'``).
    """
    from astropy.time import Time  # import here to avoid top-level astropy dep

    raw = kwargs.get("time")
    if raw is None:
        return None
    raw = raw.strip()
    if not (raw.startswith("[") and raw.endswith("]")):
        raise ValueError(f"time= value must be bracketed, got {raw!r}")
    parts = _smart_split_pairs(raw[1:-1])
    if len(parts) != 2:
        raise ValueError(f"time= requires exactly two values, got {raw!r}")
    lo_tok, hi_tok = parts[0].strip(), parts[1].strip()
    lo_fam, lo_val = _detect_time_family(lo_tok)
    hi_fam, hi_val = _detect_time_family(hi_tok)
    if lo_fam != hi_fam:
        raise ValueError(
            f"time= token family mismatch: {lo_tok!r} is {lo_fam}, "
            f"{hi_tok!r} is {hi_fam}. Both tokens must use the same format."
        )
    time_coord = data.coords["time"]
    time_dim = time_coord.dims[0]
    scale = time_coord.attrs.get("scale", "utc")
    fam = lo_fam
    if fam == "iso":
        try:
            lo_t = Time(lo_val, format="isot", scale=scale)
        except Exception:
            lo_t = Time(lo_val, format="iso", scale=scale)
        try:
            hi_t = Time(hi_val, format="isot", scale=scale)
        except Exception:
            hi_t = Time(hi_val, format="iso", scale=scale)
    elif fam == "jd":
        lo_t = Time(float(lo_val), format="jd", scale=scale)
        hi_t = Time(float(hi_val), format="jd", scale=scale)
    else:  # mjd
        lo_t = Time(float(lo_val), format="mjd", scale=scale)
        hi_t = Time(float(hi_val), format="mjd", scale=scale)
    lo_mjd, hi_mjd = lo_t.mjd, hi_t.mjd
    time_vals = time_coord.values.astype(float)
    mask = xr.DataArray((time_vals >= lo_mjd) & (time_vals <= hi_mjd), dims=[time_dim])
    _warn_if_axis_selection_empty("time", raw, "time", mask)
    return mask


def _warn_if_axis_selection_empty(
    keyword: str,
    raw_value: str,
    family: str,
    mask: xr.DataArray,
) -> None:
    """Warn when an axis-selection keyword produces an all-False mask.

    Parameters
    ----------
    keyword : str
        CRTF keyword that produced the 1-D mask, such as ``range`` or ``time``.
    raw_value : str
        Raw bracketed keyword payload from the CRTF line.
    family : str
        Interpreted token family used for the comparison, such as
        ``frequency``, ``velocity``, ``channel``, or ``time``.
    mask : xr.DataArray
        One-dimensional boolean mask built for the target axis.

    Returns
    -------
    None
        Emits a ``UserWarning`` only when *mask* contains no selected entries.

    Assumptions
    -----------
    The caller has already validated syntax and built the axis mask.  This
    helper preserves the existing all-False return semantics and adds only a
    user-visible warning for non-overlapping selections.
    """
    if bool(np.any(np.asarray(mask.values, dtype=bool))):
        return
    warnings.warn(
        f"{keyword}={raw_value} selects no {family} entries; returning an all-False mask.",
        UserWarning,
        stacklevel=3,
    )


def _compose_line_mask(
    spatial: "np.ndarray | da.Array",
    range_mask: "xr.DataArray | None",
    corr_mask: "xr.DataArray | None",
    time_mask: "xr.DataArray | None",
    data: ArrayLike,
) -> "np.ndarray | da.Array | xr.DataArray":
    """Combine per-axis masks into a single full-data-shape boolean mask.

    Parameters
    ----------
    spatial : numpy.ndarray or dask.array.Array
        Full-data-shape spatial mask from the shape rasterizer.
    range_mask : xr.DataArray or None
        1-D mask on the ``frequency`` dim, or ``None``.
    corr_mask : xr.DataArray or None
        1-D mask on the ``polarization`` dim, or ``None``.
    time_mask : xr.DataArray or None
        1-D mask on the ``time`` dim, or ``None``.
    data : numpy.ndarray or xr.DataArray
        Original data array used to determine dims and coordinates.

    Returns
    -------
    numpy.ndarray, dask.array.Array, or xr.DataArray
        Combined boolean mask with the same logical shape as ``data``.

    Notes
    -----
    For ``xr.DataArray`` inputs, the spatial mask is wrapped with ``data``'s
    dims and coords so that the 1-D axis masks broadcast correctly by
    dimension name when combined with ``&``.
    For ndarray inputs, no axis masks can be present (gating rejects them),
    so the spatial mask is returned as-is.
    """
    if not isinstance(data, xr.DataArray):
        return spatial
    # Wrap spatial with data's full dims and coords for aligned broadcasting.
    # If spatial is already a DataArray (e.g. from world-mode rasterizer, dims=['l','m']),
    # use it directly; xarray will broadcast missing dims when combined with axis masks
    # and when folded into the full-shape accumulator.
    if isinstance(spatial, xr.DataArray):
        line_mask: xr.DataArray = spatial
    else:
        line_mask = xr.DataArray(spatial, dims=data.dims, coords=data.coords)
    for axis_mask in (range_mask, corr_mask, time_mask):
        if axis_mask is not None:
            line_mask = line_mask & axis_mask.astype(bool)
    return line_mask


# ---------------------------------------------------------------------------
# lm shape mode: angular-coordinate token parsers, grid builder, rasterizer
# ---------------------------------------------------------------------------

_ANGULAR_SCALE: dict[str, float] = {
    "arcsec": math.pi / (180.0 * 3600.0),
    "arcmin": math.pi / (180.0 * 60.0),
    "deg": math.pi / 180.0,
    "rad": 1.0,
}


def _parse_angular_val(tok: str) -> float:
    """Parse a single angular token and return the value in radians.

    Parameters
    ----------
    tok : str
        Token such as ``'30arcsec'``, ``'1.5arcmin'``, ``'0.5deg'``, ``'0.1rad'``.

    Returns
    -------
    float
        Value in radians.

    Raises
    ------
    ValueError
        If the token cannot be parsed.
    """
    m = re.match(rf"^\s*({_NUM_RE})\s*(arcsec|arcmin|deg|rad)\s*$", tok, re.IGNORECASE)
    if not m:
        raise ValueError(
            f"Cannot parse angular token {tok!r}. "
            "Expected a number followed by arcsec, arcmin, deg, or rad."
        )
    return float(m.group(1)) * _ANGULAR_SCALE[m.group(2).lower()]


def _parse_pair_angular(pair_token: str) -> tuple[float, float]:
    """Parse a bracketed angular coordinate pair ``[a, b]`` into radians.

    Parameters
    ----------
    pair_token : str
        Bracketed pair such as ``'[0arcmin, 1arcmin]'``.

    Returns
    -------
    tuple[float, float]
        ``(a_rad, b_rad)`` in radians.
    """
    s = pair_token.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected '[a, b]' angular pair, got {pair_token!r}")
    inner = s[1:-1]
    toks = [t.strip() for t in inner.split(",")]
    if len(toks) != 2:
        raise ValueError(
            f"Expected exactly two values in angular pair, got {pair_token!r}"
        )
    return _parse_angular_val(toks[0]), _parse_angular_val(toks[1])


def _parse_two_angular_vals(token: str) -> tuple[float, float]:
    """Parse a bracketed pair of angular lengths ``[a, b]`` into radians.

    Parameters
    ----------
    token : str
        Bracketed pair such as ``'[1arcmin, 2arcmin]'``.

    Returns
    -------
    tuple[float, float]
        ``(a_rad, b_rad)`` in radians.
    """
    return _parse_pair_angular(token)


def _detect_shape_family(payload: str) -> str:
    """Detect the coordinate family from the first pair in a shape payload.

    Parameters
    ----------
    payload : str
        Full shape payload starting with ``'[['``.

    Returns
    -------
    str
        One of ``'pixel'``, ``'lm'``, ``'world'``, or ``'ambiguous'``.

        - ``'pixel'``: all-``pix`` tokens.
        - ``'lm'``: ``arcsec`` or ``arcmin`` tokens.
        - ``'ambiguous'``: ``deg`` or ``rad`` tokens without explicit
          ``coordsys=`` (could be lm offsets or absolute world coords).
        - ``'world'``: sexagesimal tokens (e.g. ``18h12m24s``, ``-23d11m00s``).

    Notes
    -----
    Only the first token of the first coordinate pair is inspected; shape
    validity is checked later by the rasterizer.
    """
    try:
        inner = _strip_brackets(payload).strip()
    except ValueError:
        return "pixel"
    parts = _smart_split_pairs(inner)
    if not parts:
        return "pixel"
    # First part is always the center pair (or first vertex for poly)
    first_pair = parts[0].strip()
    if first_pair.startswith("[") and first_pair.endswith("]"):
        first_pair = first_pair[1:-1]
    toks = [t.strip() for t in first_pair.split(",") if t.strip()]
    if not toks:
        return "pixel"
    first_tok = toks[0]
    if re.search(r"(?i)pix$", first_tok):
        return "pixel"
    if re.search(r"(?i)(arcsec|arcmin)$", first_tok):
        return "lm"
    if re.search(r"(?i)(deg|rad)$", first_tok):
        return "ambiguous"
    # Sexagesimal: e.g. 18h12m24s or -23d11m00s (digit-letter-digit patterns)
    if re.search(r"[0-9][hHmMsS]", first_tok) or re.search(r"[hHmMsS][0-9]", first_tok):
        return "world"
    return "pixel"  # fallback for bare numbers or unrecognized units


def _parse_coordsys_keyword(kwargs: dict[str, str]) -> str | None:
    """Extract and validate the ``coordsys=`` keyword from a CRTF kwargs dict.

    Parameters
    ----------
    kwargs : dict[str, str]
        Keyword assignments from a CRTF line or merged global+per-line dict.

    Returns
    -------
    str or None
        One of ``'pixel'``, ``'lm'``, ``'world'``, or ``None`` if absent.

    Raises
    ------
    ValueError
        If ``coordsys=`` is present but its value is not one of the accepted
        choices.
    """
    val = kwargs.get("coordsys")
    if val is None:
        return None
    v = val.strip().lower()
    if v not in ("pixel", "lm", "world"):
        raise ValueError(
            f"Unrecognized coordsys= value {val!r}; must be 'pixel', 'lm', or 'world'."
        )
    return v


def _resolve_shape_family(payload: str, coordsys: str | None) -> str:
    """Resolve the shape coordinate family, applying an explicit ``coordsys=`` override.

    Parameters
    ----------
    payload : str
        Full shape payload starting with ``'[['``.
    coordsys : str or None
        Explicit ``coordsys=`` value (``'pixel'``, ``'lm'``, ``'world'``), or
        ``None`` if absent.

    Returns
    -------
    str
        One of ``'pixel'``, ``'lm'``, ``'world'``.

    Raises
    ------
    ValueError
        - If tokens are ambiguous (deg/rad) and *coordsys* is ``None``.
        - If an explicit *coordsys* conflicts with the auto-detected token
          family (e.g. ``coordsys='pixel'`` with sexagesimal tokens).

    Notes
    -----
    When *coordsys* is present and the auto-detected family is ``'ambiguous'``
    (deg/rad tokens without a clear semantic), *coordsys* resolves the
    ambiguity.  When the family is already unambiguous, *coordsys* must agree
    or a ``ValueError`` is raised.
    """
    detected = _detect_shape_family(payload)
    if coordsys is not None:
        if detected == "ambiguous":
            # deg/rad tokens resolved by explicit coordsys
            return coordsys
        if detected != coordsys:
            raise ValueError(
                f"Explicit coordsys={coordsys!r} conflicts with the auto-detected "
                f"coordinate family '{detected}' implied by the payload tokens: "
                f"{payload!r}."
            )
        return detected
    if detected == "ambiguous":
        raise ValueError(
            "Ambiguous deg/rad center coordinates: add coordsys=world or "
            "coordsys=lm to the CRTF line or global block."
        )
    return detected


# ---------------------------------------------------------------------------
# World-coordinate helpers
# ---------------------------------------------------------------------------


def _parse_sexa_ra(token: str) -> float:
    """Parse a sexagesimal RA token (``HhMmSs``) to radians.

    Parameters
    ----------
    token : str
        e.g. ``'18h12m24.5s'`` or ``'18H12M24S'``.

    Returns
    -------
    float
        Angle in radians.

    Raises
    ------
    ValueError
        If the token does not match the expected sexagesimal RA pattern.
    """
    m = re.match(r"^([+-]?)(\d+)[hH](\d+)[mM]([\d.]+)[sS]$", token.strip())
    if not m:
        raise ValueError(f"Cannot parse sexagesimal RA token: {token!r}")
    sign = -1.0 if m.group(1) == "-" else 1.0
    h, mn, s = int(m.group(2)), int(m.group(3)), float(m.group(4))
    deg = sign * (h * 15.0 + mn * 15.0 / 60.0 + s * 15.0 / 3600.0)
    return deg * math.pi / 180.0


def _parse_sexa_dec(token: str) -> float:
    """Parse a sexagesimal Dec token (``DdMmSs``) to radians.

    Parameters
    ----------
    token : str
        e.g. ``'-23d11m00.5s'`` or ``'+12D30M00.5S'``.

    Returns
    -------
    float
        Angle in radians.

    Raises
    ------
    ValueError
        If the token does not match the expected sexagesimal Dec pattern.
    """
    m = re.match(r"^([+-]?)(\d+)[dD](\d+)[mM]([\d.]+)[sS]$", token.strip())
    if not m:
        raise ValueError(f"Cannot parse sexagesimal Dec token: {token!r}")
    sign = -1.0 if m.group(1) == "-" else 1.0
    d, mn, s = int(m.group(2)), int(m.group(3)), float(m.group(4))
    deg = sign * (d + mn / 60.0 + s / 3600.0)
    return deg * math.pi / 180.0


def _parse_world_coord_token(token: str) -> float:
    """Parse a single world-coordinate token to radians.

    Parameters
    ----------
    token : str
        Accepted forms: sexagesimal RA (``HhMmSs``), sexagesimal Dec
        (``DdMmSs``), decimal degrees (``NNdeg``), or radians (``NNrad``).
        Angular-offset units (``arcsec``, ``arcmin``) are also accepted for
        length/radius tokens in world-mode shapes.

    Returns
    -------
    float
        Angle in radians.

    Raises
    ------
    ValueError
        If the token cannot be parsed in any accepted form.
    """
    token = token.strip()
    # Decimal degrees
    m = re.match(r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*deg$", token, re.IGNORECASE)
    if m:
        return float(m.group(1)) * math.pi / 180.0
    # Radians
    m = re.match(r"^([+-]?[\d.]+(?:[eE][+-]?\d+)?)\s*rad$", token, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # Angular offsets (arcsec, arcmin) — accepted for radii / edge lengths
    try:
        return _parse_angular_val(token)
    except ValueError:
        pass
    # Sexagesimal RA (h/m/s)
    try:
        return _parse_sexa_ra(token)
    except ValueError:
        pass
    # Sexagesimal Dec (d/m/s)
    try:
        return _parse_sexa_dec(token)
    except ValueError:
        pass
    raise ValueError(f"Cannot parse world coordinate token: {token!r}")


def _parse_world_pair(pair_token: str) -> tuple[float, float]:
    """Parse a world-coordinate pair ``[coord1, coord2]`` to ``(rad, rad)``.

    Parameters
    ----------
    pair_token : str
        Either ``'[coord1, coord2]'`` or bare ``'coord1, coord2'``.

    Returns
    -------
    tuple[float, float]
        ``(coord1_rad, coord2_rad)`` where coord1 is the RA-like (east/lon)
        axis and coord2 is the Dec-like (north/lat) axis.

    Raises
    ------
    ValueError
        If the pair does not contain exactly two comma-separated tokens.
    """
    inner = pair_token.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    toks = [t.strip() for t in inner.split(",") if t.strip()]
    if len(toks) != 2:
        raise ValueError(
            f"Expected exactly 2 coordinates in world pair, got {len(toks)}: "
            f"{pair_token!r}"
        )
    return _parse_world_coord_token(toks[0]), _parse_world_coord_token(toks[1])


def _build_skycoord_grid(data: xr.DataArray) -> "SkyCoord":
    """Build a 2-D ``SkyCoord`` grid from ``right_ascension`` / ``declination`` coords.

    Parameters
    ----------
    data : xr.DataArray
        DataArray that must carry ``right_ascension`` and ``declination`` as 2-D
        coordinates (radians, on the ``l`` × ``m`` plane).

    Returns
    -------
    astropy.coordinates.SkyCoord
        Sky-coordinate grid with shape ``(n_l, n_m)``, in ICRS.

    Notes
    -----
    The RA/Dec arrays are materialised to NumPy here.  World-mode shapes rely
    on astropy operations that cannot remain lazy; v1 accepts this.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    ra = np.asarray(data.coords["right_ascension"].values, dtype=float)
    dec = np.asarray(data.coords["declination"].values, dtype=float)
    return SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")


def _rasterize_shape_world(
    shape: str,
    payload: str,
    skycoord_grid: "SkyCoord",
) -> xr.DataArray:
    """Rasterize a world-mode CRTF shape against a per-pixel ``SkyCoord`` grid.

    Parameters
    ----------
    shape : str
        Lowercase CRTF shape name (e.g. ``'circle'``, ``'box'``).
    payload : str
        The ``[[...]]`` shape payload with world-coordinate tokens.
    skycoord_grid : astropy.coordinates.SkyCoord
        2-D ``(n_l, n_m)`` sky-coordinate grid built from ``right_ascension``
        and ``declination`` coords.

    Returns
    -------
    xr.DataArray
        2-D boolean mask with dims ``['l', 'm']``.  The ``_compose_line_mask``
        helper broadcasts it to the full data shape via xarray dimension
        alignment.

    Notes
    -----
    ``circle`` and ``annulus`` use ``SkyCoord.separation``; all other shapes
    use a ``SkyOffsetFrame`` centered on the shape's center (or polygon
    centroid) for tangent-plane geometry.  No frame conversion is performed;
    RA/Dec values are interpreted as-is in ICRS.
    """
    from astropy.coordinates import SkyCoord, SkyOffsetFrame
    import astropy.units as u

    inner = _strip_brackets(payload).strip()
    parts = _smart_split_pairs(inner)

    def _center_skycoord(lon_rad: float, lat_rad: float) -> SkyCoord:
        return SkyCoord(ra=lon_rad * u.rad, dec=lat_rad * u.rad, frame="icrs")

    def _grid_offsets(
        center: SkyCoord,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lon, lat) offsets in radians, lon in ``[-π, π]``."""
        grid_off = skycoord_grid.transform_to(SkyOffsetFrame(origin=center))
        lon = grid_off.lon.rad
        lon = np.where(lon > math.pi, lon - 2 * math.pi, lon)
        return lon, grid_off.lat.rad

    def _wrap_lon(lon_rad: float) -> float:
        """Wrap a single longitude value to ``[-π, π]``."""
        return (lon_rad + math.pi) % (2 * math.pi) - math.pi

    if shape == "circle":
        cx, cy = _parse_world_pair(parts[0])
        r_rad = _parse_angular_val(parts[1])
        center = _center_skycoord(cx, cy)
        sep = skycoord_grid.separation(center).rad
        mask_2d = sep <= r_rad + 1e-30

    elif shape == "annulus":
        cx, cy = _parse_world_pair(parts[0])
        r1, r2 = _parse_two_angular_vals(parts[1])
        center = _center_skycoord(cx, cy)
        sep = skycoord_grid.separation(center).rad
        mask_2d = (sep >= r1) & (sep <= r2 + 1e-30)

    elif shape == "centerbox":
        cx, cy = _parse_world_pair(parts[0])
        w, h = _parse_two_angular_vals(parts[1])
        center = _center_skycoord(cx, cy)
        lon, lat = _grid_offsets(center)
        mask_2d = (np.abs(lon) <= w / 2) & (np.abs(lat) <= h / 2)

    elif shape == "box":
        # box[[BLC_ra, BLC_dec], [TRC_ra, TRC_dec]]: center at midpoint
        ra1, dec1 = _parse_world_pair(parts[0])
        ra2, dec2 = _parse_world_pair(parts[1])
        mid_ra = (ra1 + ra2) / 2.0
        mid_dec = (dec1 + dec2) / 2.0
        center = _center_skycoord(mid_ra, mid_dec)
        offset_frame = SkyOffsetFrame(origin=center)
        lon, lat = _grid_offsets(center)
        # corners in offset frame
        blc = _center_skycoord(ra1, dec1).transform_to(offset_frame)
        trc = _center_skycoord(ra2, dec2).transform_to(offset_frame)
        blc_lon = _wrap_lon(blc.lon.rad)
        trc_lon = _wrap_lon(trc.lon.rad)
        lon_min = min(blc_lon, trc_lon)
        lon_max = max(blc_lon, trc_lon)
        lat_min = min(blc.lat.rad, trc.lat.rad)
        lat_max = max(blc.lat.rad, trc.lat.rad)
        mask_2d = (
            (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
        )

    elif shape == "rotbox":
        cx, cy = _parse_world_pair(parts[0])
        w, h = _parse_two_angular_vals(parts[1])
        if len(parts) != 3:
            raise ValueError("rotbox requires angle, e.g. pa=30deg or theta_m=30deg.")
        ang = _parse_angle_kv(parts[2])
        center = _center_skycoord(cx, cy)
        lon, lat = _grid_offsets(center)
        lon_r, lat_r = _rotate_about(lon, lat, 0.0, 0.0, -ang)
        mask_2d = (np.abs(lon_r) <= w / 2) & (np.abs(lat_r) <= h / 2)

    elif shape == "ellipse":
        cx, cy = _parse_world_pair(parts[0])
        a, b = _parse_two_angular_vals(parts[1])
        if len(parts) != 3:
            raise ValueError("ellipse requires angle, e.g. pa=30deg or theta_m=30deg.")
        ang = _parse_angle_kv(parts[2])
        center = _center_skycoord(cx, cy)
        lon, lat = _grid_offsets(center)
        xp, yp = _rotate_about(lon, lat, 0.0, 0.0, -ang)
        mask_2d = (xp / a) ** 2 + (yp / b) ** 2 <= 1.0 + 1e-30

    elif shape == "poly":
        pts_world = [_parse_world_pair(p) for p in parts]
        cen_ra = sum(p[0] for p in pts_world) / len(pts_world)
        cen_dec = sum(p[1] for p in pts_world) / len(pts_world)
        center = _center_skycoord(cen_ra, cen_dec)
        offset_frame = SkyOffsetFrame(origin=center)
        lon, lat = _grid_offsets(center)
        verts_sky = SkyCoord(
            ra=np.array([p[0] for p in pts_world]) * u.rad,
            dec=np.array([p[1] for p in pts_world]) * u.rad,
            frame="icrs",
        )
        verts_off = verts_sky.transform_to(offset_frame)
        verts_lon = np.where(
            verts_off.lon.rad > math.pi,
            verts_off.lon.rad - 2 * math.pi,
            verts_off.lon.rad,
        )
        pts = list(zip(verts_lon.tolist(), verts_off.lat.rad.tolist()))
        mask_2d = _point_in_poly(lon, lat, pts)

    else:
        raise ValueError(f"Unsupported CRTF shape for world mode: {shape}")

    return xr.DataArray(mask_2d, dims=["l", "m"])


def _build_lm_coordinate_grids(
    data: xr.DataArray,
    *,
    lazy: bool,
) -> tuple[np.ndarray | da.Array, np.ndarray | da.Array]:
    """Build broadcasted L/M world-coordinate grids from ``l``/``m`` coord values.

    Parameters
    ----------
    data : xr.DataArray
        DataArray that must carry ``l`` and ``m`` coordinates (1-D, radians).
    lazy : bool
        If ``True``, build Dask arrays; otherwise build NumPy arrays.

    Returns
    -------
    tuple[numpy.ndarray | dask.array.Array, numpy.ndarray | dask.array.Array]
        ``(L, M)`` grids, each broadcast to ``data.shape``, in radians.

    Notes
    -----
    Analogous to :func:`_build_pixel_coordinate_grids` but uses the actual
    physical coordinate values instead of integer pixel indices.
    """
    data_shape = data.shape
    dims = list(data.dims)
    l_axis = dims.index("l")
    m_axis = dims.index("m")
    l_vals = data.coords["l"].values.astype(float)
    m_vals = data.coords["m"].values.astype(float)
    l_view = tuple(len(l_vals) if i == l_axis else 1 for i in range(len(dims)))
    m_view = tuple(len(m_vals) if i == m_axis else 1 for i in range(len(dims)))
    if lazy:
        L = da.broadcast_to(da.reshape(da.from_array(l_vals), l_view), data_shape)
        M = da.broadcast_to(da.reshape(da.from_array(m_vals), m_view), data_shape)
    else:
        L = np.broadcast_to(l_vals.reshape(l_view), data_shape)
        M = np.broadcast_to(m_vals.reshape(m_view), data_shape)
    return L, M


def _rasterize_shape_lm(
    shape: str,
    payload: str,
    L: np.ndarray | da.Array,
    M: np.ndarray | da.Array,
) -> np.ndarray | da.Array:
    """Rasterize a CRTF shape using angular (lm) coordinates in radians.

    Parameters
    ----------
    shape : str
        Lowercase shape name (``'box'``, ``'centerbox'``, etc.).
    payload : str
        Full shape payload starting with ``'[['``.
    L : numpy.ndarray or dask.array.Array
        Grid of ``l`` coordinate values (radians), broadcast to data shape.
    M : numpy.ndarray or dask.array.Array
        Grid of ``m`` coordinate values (radians), broadcast to data shape.

    Returns
    -------
    numpy.ndarray or dask.array.Array
        Boolean mask with the same shape as ``L`` and ``M``.

    Notes
    -----
    Geometry is identical to the pixel rasterizer (:func:`_rasterize_shape`);
    only the token parser changes (angular units instead of ``pix``).  The
    rotation angle for ``rotbox`` / ``ellipse`` is still accepted in
    ``deg`` / ``rad`` via the shared :func:`_parse_angle_kv`.
    """
    inner = _strip_brackets(payload).strip()
    parts = _smart_split_pairs(inner)
    if shape == "box":
        p1x, p1y = _parse_pair_angular(parts[0])
        p2x, p2y = _parse_pair_angular(parts[1])
        x1, x2 = sorted([p1x, p2x])
        y1, y2 = sorted([p1y, p2y])
        return (L >= x1) & (L <= x2) & (M >= y1) & (M <= y2)
    if shape == "centerbox":
        cx, cy = _parse_pair_angular(parts[0])
        w, h = _parse_two_angular_vals(parts[1])
        hx, hy = w / 2.0, h / 2.0
        return (np.abs(L - cx) <= hx) & (np.abs(M - cy) <= hy)
    if shape == "rotbox":
        cx, cy = _parse_pair_angular(parts[0])
        w, h = _parse_two_angular_vals(parts[1])
        if len(parts) != 3:
            raise ValueError(
                "rotbox requires angle specified as 'pa=<angle>' or 'theta_m=<angle>', "
                "e.g., rotbox[[cx,cy],[w,h], pa=30deg]"
            )
        ang = _parse_angle_kv(parts[2])
        hx, hy = w / 2.0, h / 2.0
        xrp, yrp = _rotate_about(L, M, cx, cy, -ang)
        return (np.abs(xrp - cx) <= hx) & (np.abs(yrp - cy) <= hy)
    if shape == "circle":
        cx, cy = _parse_pair_angular(parts[0])
        r = _parse_angular_val(parts[1])
        return ((L - cx) ** 2 + (M - cy) ** 2) <= (r**2 + 1e-30)
    if shape == "annulus":
        cx, cy = _parse_pair_angular(parts[0])
        r1, r2 = _parse_two_angular_vals(parts[1])
        d2 = (L - cx) ** 2 + (M - cy) ** 2
        return (d2 >= r1**2) & (d2 <= r2**2)
    if shape == "ellipse":
        cx, cy = _parse_pair_angular(parts[0])
        a, b = _parse_two_angular_vals(parts[1])
        if len(parts) != 3:
            raise ValueError(
                "ellipse requires angle specified as 'pa=<angle>' or 'theta_m=<angle>', "
                "e.g., ellipse[[cx,cy],[a,b], pa=30deg]"
            )
        ang = _parse_angle_kv(parts[2])
        xp, yp = _rotate_about(L, M, cx, cy, -ang)
        return ((xp - cx) / a) ** 2 + ((yp - cy) / b) ** 2 <= 1.0 + 1e-30
    if shape == "poly":
        pts = [_parse_pair_angular(p) for p in parts]
        return _point_in_poly(L, M, pts)
    raise ValueError(f"Unsupported CRTF shape: {shape}")


def _required_coords_for_line(
    shape_name: str, payload: str, kwargs: dict[str, str]
) -> frozenset[str]:
    """Return the set of DataArray coord names required by a single CRTF line.

    Parameters
    ----------
    shape_name : str
        Lowercase CRTF shape name (e.g. ``'box'``).
    payload : str
        The ``[[...]]`` shape payload string; used to detect the coordinate
        family (pixel / lm / world) for spatial coord gating.
    kwargs : dict[str, str]
        Trailing key=value pairs from the CRTF line.

    Returns
    -------
    frozenset[str]
        Coord names that must be present on ``data`` for this line to be
        processed.  An empty frozenset means any input (including bare
        ``ndarray``) is accepted for that line.

    Notes
    -----
    Pixel-mode shapes with no extra keywords require no coords.  Non-pixel
    shape modes and ``range=`` / ``corr=`` / ``time=`` keywords require
    specific coords.  ``world``-mode requirements (``right_ascension``,
    ``declination``) are added in step 7.
    """
    required: set[str] = set()
    if "range" in kwargs:
        raw = kwargs["range"].strip()
        # Peek at the first token to determine the family for accurate gating
        try:
            inner = raw.lstrip("[").rstrip("]")
            first_tok = _smart_split_pairs(inner)[0].strip()
            fam = _detect_range_family(first_tok)
        except Exception:
            fam = "frequency"  # conservative fallback
        if fam == "velocity":
            required.add("velocity")
        else:
            required.add("frequency")
    if "corr" in kwargs:
        required.add("polarization")
    if "time" in kwargs:
        required.add("time")
    # Detect spatial coordinate family and add coord requirements.
    # Use coordsys= (from merged effective kwargs) to resolve deg/rad ambiguity.
    try:
        coordsys = _parse_coordsys_keyword(kwargs)
        family = _resolve_shape_family(payload, coordsys)
    except ValueError:
        # Ambiguous, conflicting, or unrecognized coordsys — dispatch will raise
        # the proper error; skip coord gating here.
        family = "pixel"
    if family == "lm":
        required.add("l")
        required.add("m")
    elif family == "world":
        required.add("right_ascension")
        required.add("declination")
    return frozenset(required)


def _assert_data_has_coords(
    data: ArrayLike,
    required: frozenset[str],
    context: str = "CRTF line",
) -> None:
    """Raise ``ValueError`` if any required coord is absent from *data*.

    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Data array to check.  NumPy ndarrays carry no named coords; any
        non-empty *required* set will raise for ndarray inputs.
    required : frozenset[str]
        Coord names that must be present.
    context : str
        Human-readable label for the error message.

    Raises
    ------
    ValueError
        If *data* is a NumPy ndarray and *required* is non-empty, or if any
        name in *required* is absent from ``data.coords``.
    """
    if not required:
        return
    if not isinstance(data, xr.DataArray):
        missing = required
    else:
        missing = frozenset(n for n in required if n not in data.coords)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(
            f"CRTF feature in {context} requires coord(s) not present on data: "
            f"{names}. Pass xds.SKY (with the relevant coordinates attached) "
            "instead of a bare array."
        )


def _crtf_mask(data: ArrayLike, text: str, *, lazy: bool = False) -> ArrayLike:
    """Parse a CRTF string (single or multi-line) into a boolean mask.

    Combination semantics per line: leading '+' (OR, default) or '-' (NOT/subtract).
    ``ann``-prefixed lines (visualization annotations) are silently skipped.
    Visualization-only keywords are silently ignored.  Frame-conversion keywords
    (``coord=``, ``frame=``, ``veltype=``, ``restfreq=``) raise
    ``NotImplementedError``.
    """
    data_shape = data.shape if isinstance(data, xr.DataArray) else np.shape(data)
    x_axis, y_axis = _infer_xy_axes(data)
    X, Y = _build_pixel_coordinate_grids(data_shape, x_axis, y_axis, lazy=lazy)
    # Use a DataArray accumulator for DataArray inputs so axis masks
    # (range/corr/time) broadcast by dimension name rather than by shape.
    if isinstance(data, xr.DataArray):
        acc: xr.DataArray | np.ndarray | da.Array = xr.zeros_like(data, dtype=bool)
    else:
        acc = (
            da.zeros(data_shape, dtype=bool)
            if lazy
            else np.zeros(data_shape, dtype=bool)
        )
    lines = [
        ln.strip()
        for ln in re.split(r"[\n;]+", text)
        if ln.strip() and not ln.strip().startswith("#")
    ]
    # Global keyword assignments accumulate across 'global' lines and serve as
    # defaults for every subsequent region line (per-line overrides win).
    globals_kwargs: dict[str, str] = {}
    for line in lines:
        if line.lower().startswith("global"):
            parsed_globals = _parse_crtf_globals(line)
            _reject_frame_keywords(parsed_globals, context="CRTF global")
            globals_kwargs.update(parsed_globals)
            continue
        # ann-prefixed lines are visualization annotations; no mask contribution
        if re.match(r"(?i)^\s*ann\b", line):
            continue
        flag, shape_name, payload, kwargs = _parse_crtf_line(line)
        _reject_frame_keywords(kwargs)
        # Merge globals with per-line kwargs; per-line takes precedence.
        effective_kwargs = {**globals_kwargs, **kwargs}
        required = _required_coords_for_line(shape_name, payload, effective_kwargs)
        _assert_data_has_coords(data, required, context=f"'{shape_name}' line")
        # Resolve coordinate family, honoring any coordsys= keyword.
        coordsys = _parse_coordsys_keyword(effective_kwargs)
        family = _resolve_shape_family(payload, coordsys)
        if family == "lm":
            L, M = _build_lm_coordinate_grids(data, lazy=lazy)
            spatial = _rasterize_shape_lm(shape_name, payload, L, M)
        elif family == "pixel":
            spatial = _rasterize_shape(shape_name, payload, X, Y)
        else:  # world — requires right_ascension / declination coords
            # _assert_data_has_coords already verified the coords are present.
            # Materialise the sky grid (astropy operations are numpy-native).
            sky_grid = _build_skycoord_grid(data)
            spatial = _rasterize_shape_world(shape_name, payload, sky_grid)
        range_mask = (
            _build_range_mask(data, effective_kwargs)
            if isinstance(data, xr.DataArray)
            else None
        )
        corr_mask = (
            _build_corr_mask(data, effective_kwargs)
            if isinstance(data, xr.DataArray)
            else None
        )
        time_mask = (
            _build_time_mask(data, effective_kwargs)
            if isinstance(data, xr.DataArray)
            else None
        )
        line_mask = _compose_line_mask(spatial, range_mask, corr_mask, time_mask, data)
        if flag == "+":
            acc = acc | line_mask
        else:
            acc = acc & (~line_mask)
    if isinstance(data, xr.DataArray):
        return (
            acc if isinstance(acc, xr.DataArray) else xr.DataArray(acc, dims=data.dims)
        )
    return acc


def _infer_xy_axes(data: ArrayLike) -> Tuple[int, int]:
    """Infer the axis indices corresponding to x and y pixel coordinates.

    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Input image data used as the shape template for CRTF rasterization.

    Returns
    -------
    tuple[int, int]
        Two integers ``(x_axis, y_axis)`` giving the axis index of x then y.

    Notes
    -----
    - For ``xarray.DataArray`` with named dimensions containing both ``"x"`` and
      ``"y"``, those named dimensions define the axis mapping.
    - For NumPy arrays, or DataArrays without both names, axes default to
      ``x_axis=0`` and ``y_axis=1``. This implies that unnamed image-like inputs
      are interpreted as having shape ``(nx, ny, ...)`` where axis 0 (length ``nx``)
      is the x-axis and axis 1 (length ``ny``) is the y-axis, so CRTF pixel
      coordinates ``(x, y)`` map directly to indices along these axes.
    - This helper is compatible with N-D inputs: it selects which two axes represent
      x and y, while any additional axes (e.g., channels, polarization, time) are
      left unchanged. When combined with :func:`_build_pixel_coordinate_grids`, the
      x/y coordinate grids are broadcast across these extra axes so the resulting
      CRTF mask is applied identically along non-(x, y) axes.
    """
    if isinstance(data, xr.DataArray) and "x" in data.dims and "y" in data.dims:
        return data.dims.index("x"), data.dims.index("y")
    if isinstance(data, xr.DataArray) and "l" in data.dims and "m" in data.dims:
        # TODO just a place holder as a reminder. will need to be updated when
        # full 5-D xradio image cubes are used.
        return data.dims.index("l"), data.dims.index("m")
    return 0, 1


def _build_pixel_coordinate_grids(
    shape: Tuple[int, ...], x_axis: int, y_axis: int, *, lazy: bool
) -> Tuple[np.ndarray | da.Array, np.ndarray | da.Array]:
    """Build broadcasted x/y pixel-index grids aligned to ``shape``.

    Parameters
    ----------
    shape : tuple[int, ...]
        Output grid shape.
    x_axis : int
        Axis index in ``shape`` corresponding to x coordinates.
    y_axis : int
        Axis index in ``shape`` corresponding to y coordinates.
    lazy : bool
        If ``True``, build Dask arrays; otherwise build NumPy arrays.

    Returns
    -------
    tuple[numpy.ndarray | dask.array.Array, numpy.ndarray | dask.array.Array]
        ``(X, Y)`` coordinate grids, each with full ``shape``.

    Notes
    -----
    The function creates 1D index vectors and broadcasts them to full-image grids
    so CRTF expressions always interpret values as ``(x, y)`` regardless of axis order.
    """
    if lazy:
        x1d = da.arange(shape[x_axis], dtype=float)
        y1d = da.arange(shape[y_axis], dtype=float)
        x_view = da.reshape(
            x1d, tuple(shape[x_axis] if i == x_axis else 1 for i in range(len(shape)))
        )
        y_view = da.reshape(
            y1d, tuple(shape[y_axis] if i == y_axis else 1 for i in range(len(shape)))
        )
        return da.broadcast_to(x_view, shape), da.broadcast_to(y_view, shape)
    x1d = np.arange(shape[x_axis], dtype=float)
    y1d = np.arange(shape[y_axis], dtype=float)
    x_view = x1d.reshape(
        tuple(shape[x_axis] if i == x_axis else 1 for i in range(len(shape)))
    )
    y_view = y1d.reshape(
        tuple(shape[y_axis] if i == y_axis else 1 for i in range(len(shape)))
    )
    return np.broadcast_to(x_view, shape), np.broadcast_to(y_view, shape)


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
                dims=getattr(mask, "dims", getattr(data, "dims", ("x", "y"))),
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
            getattr(data, "dims", ("x", "y"))
            if isinstance(data, xr.DataArray)
            else ("x", "y")
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
        getattr(data, "dims", ("x", "y"))
        if isinstance(data, xr.DataArray)
        else ("x", "y")
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
