"""
Tests for ROI selection (CRTF + expressions) using the public API only.

Public API under test:
- astroviper.distributed.image_analysis.selection.select_mask
- astroviper.distributed.image_analysis.selection.apply_select

Conventions covered:
- 0-based pixel indices with mandatory "pix" suffix (e.g., [0pix, 127pix])
- CRTF strings with and without the optional #CRTF header
- Multi-line CRTF with leading "+" (OR) and "-" (subtract)
- Named-mask expression language: &, |, ^, ~ and mask_source mapping
- Backticked file paths and pathlib.Path for CRTF input

Run:
    pytest -q
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import numpy as np
import pytest
import xarray as xr
import dask.array as da

from astroviper.distributed.image_analysis.selection import (
    select_mask,
    apply_select,
)

# ------------------------- fixtures / helpers -------------------------


@pytest.fixture(autouse=True)
def _headless_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    # keep consistent with other test modules
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        monkeypatch.setattr(plt, "show", lambda *a, **k: None, raising=False)
    except Exception:
        pass


def make_image(ny: int = 200, nx: int = 200) -> xr.DataArray:
    # 0-based pixel coordinates
    y = np.arange(ny, dtype=float)
    x = np.arange(nx, dtype=float)
    z = (y[:, None] + x[None, :]) / (ny + nx)  # non-constant for apply_select checks
    return xr.DataArray(z, dims=("y", "x"), coords={"y": y, "x": x}, name="img")


# ------------------------- CRTF basics -------------------------


class TestCRTFBasics:
    def test_box_with_and_without_header_0based_and_area(self) -> None:
        da = make_image(64, 80)
        # inclusive bounds: width = x2-x1+1, height = y2-y1+1
        s_no = "box[[10pix, 5pix], [29pix, 25pix]]"
        s_hd = "#CRTF\nbox[[10pix, 5pix], [29pix, 25pix]]"
        m_no = select_mask(da, select=s_no)
        m_hd = select_mask(da, select=s_hd)
        assert isinstance(m_no, xr.DataArray) and m_no.dtype == bool
        assert (m_no == m_hd).all()
        width = (29 - 10 + 1)
        height = (25 - 5 + 1)
        assert int(m_no.values.sum()) == width * height

    def test_circle_centroid_near_center(self) -> None:
        da = make_image(200, 200)
        cx, cy, r = 120, 80, 40
        m = select_mask(da, select=f"circle[[{cx}pix,{cy}pix], {r}pix]")
        yy, xx = np.mgrid[0:da.sizes["y"], 0:da.sizes["x"]]
        yy = yy.astype(float); xx = xx.astype(float)
        inside = m.values
        assert inside.dtype == bool and inside.any()
        # centroid of boolean mask
        ybar = float((yy[inside]).mean())
        xbar = float((xx[inside]).mean())
        assert abs(xbar - cx) <= 0.5
        assert abs(ybar - cy) <= 0.5

    @pytest.mark.parametrize("shape, spec", [
        ("rotbox", "rotbox[[120pix,80pix],[60pix,30pix], 30]"),
        ("ellipse", "ellipse[[120pix,80pix],[50pix,25pix], 60]")
    ])
    def test_rotbox_ellipse_parse_and_nonempty(self, shape: str, spec: str) -> None:
        da = make_image(180, 220)
        m = select_mask(da, select=spec)
        assert isinstance(m, xr.DataArray)
        assert m.dtype == bool and int(m.values.sum()) > 0


# ------------------------- CRTF combination (+ / -) -------------------------


class TestCRTFCombine:
    def test_multi_line_plus_minus_matches_boolean_ops(self) -> None:
        da = make_image(160, 160)
        circle = "circle[[100pix,70pix], 45pix]"
        rot = "rotbox[[100pix,70pix],[40pix,20pix], 20]"
        extra = "box[[5pix,130pix],[55pix,150pix]]"
        crtf = f"#CRTF\n+{circle}\n-{rot}\n+{extra}"
        m_file = select_mask(da, select=crtf)

        # Compose equivalent reference via public API (boolean ops over masks)
        m_circle = select_mask(da, select=circle)
        m_rot = select_mask(da, select=rot)
        m_extra = select_mask(da, select=extra)
        m_ref = (m_circle & ~m_rot) | m_extra
        assert (m_file == m_ref).all()

# ------------------------- Expressions over named masks -------------------------

class TestExpressions:
    def test_expression_roi_and_not_bad(self) -> None:
        da = make_image(128, 128)
        roi = select_mask(da, "circle[[64pix,64pix], 40pix]")
        bad = select_mask(da, "box[[0pix,0pix],[20pix,127pix]]") | select_mask(
            da, "box[[0pix,0pix],[127pix,20pix]]"
        )
        expr = "roi & ~bad"
        m = select_mask(da, select=expr, mask_source={"roi": roi, "bad": bad})
        m_ref = roi & ~bad
        assert (m == m_ref).all()

    def test_expression_invalid_minus_not_raises(self) -> None:
        da = make_image(64, 64)
        roi = select_mask(da, "circle[[32pix,32pix], 10pix]")
        bad = select_mask(da, "box[[0pix,0pix],[10pix,63pix]]")
        expr = "roi & -bad"  # using '-' instead of '~'
        with pytest.raises(Exception):
            select_mask(da, select=expr, mask_source={"roi": roi, "bad": bad})

    def test_expression_unknown_name_keyerror_lists_available(self) -> None:
        da = make_image(32, 32)
        roi = select_mask(da, "box[[5pix,5pix],[10pix,10pix]]")
        with pytest.raises(KeyError) as ei:
            select_mask(da, select="roi & unknown", mask_source={"roi": roi})
        # KeyError string repr adds quotes; compare the underlying message.
        assert ei.value.args[0] == "Unknown mask name: unknown. Available: roi"

    def test_expression_syntax_error_raises_value_error(self) -> None:
        """
        Trigger a SyntaxError in the parser (e.g., trailing '&') so the public API
        surfaces ValueError('Invalid selection expression').
        """
        da = make_image(8, 8)
        roi = select_mask(da, "box[[1pix,1pix],[2pix,2pix]]")
        with pytest.raises(ValueError) as ei:
            select_mask(da, select="roi &", mask_source={"roi": roi})
        assert ei.value.args[0] == "Invalid selection expression"

    def test_expression_with_xarray_dataset_mask_source(self) -> None:
        """
        Cover _build_mask_env branch where mask_source is an xarray.Dataset:
            if isinstance(mask_source, xr.Dataset):
                items = {k: v for k, v in mask_source.data_vars.items() if _is_boolish(v)}
        """
        da = make_image(64, 64)
        roi = select_mask(da, "circle[[32pix,32pix], 12pix]")
        bad = select_mask(da, "box[[0pix,0pix],[10pix,63pix]]")
        # include a non-boolean variable to verify filtering does not break
        ds = xr.Dataset(
            data_vars={
                "roi": roi,
                "bad": bad,
                "not_mask": da,  # should be ignored by _is_boolish
            }
        )
        expr = "roi & ~bad"
        m = select_mask(da, select=expr, mask_source=ds)
        m_ref = roi & ~bad
        assert (m == m_ref).all()

    def test_mask_source_typeerror_when_not_mapping_or_dataset(self) -> None:
        """
        Cover: raise TypeError("mask_source must be a Mapping or xarray.Dataset")
        """
        da = make_image(8, 8)
        # any non-Mapping, non-Dataset object triggers the TypeError
        with pytest.raises(TypeError) as ei:
            select_mask(da, select="roi", mask_source=object())
        assert ei.value.args[0] == "mask_source must be a Mapping or xarray.Dataset"

    def test_mask_source_empty_raises_value_error(self) -> None:
        """
        Cover: raise ValueError("mask_source does not provide any boolean masks")
        when items filtered from mask_source are empty.
        """
        da = make_image(8, 8)
        with pytest.raises(ValueError) as ei:
            select_mask(da, select="roi", mask_source={})
        assert ei.value.args[0] == "mask_source does not provide any boolean masks"

    def test_expression_unsupported_construct_compare_raises(self) -> None:
        """
        Cover: for non-allowed AST nodes -> ValueError('...unsupported construct: <Node>')
        Use '==' (ast.Compare), which is not in _ALLOWED_NODES.
        """
        da = make_image(16, 16)
        roi = select_mask(da, "box[[2pix,2pix],[5pix,5pix]]")
        bad = select_mask(da, "box[[8pix,8pix],[10pix,10pix]]")
        with pytest.raises(ValueError) as ei:
            select_mask(da, select="roi == bad", mask_source={"roi": roi, "bad": bad})
        assert "Expression contains an unsupported construct: Compare" in ei.value.args[0]

    def test_expression_boolop_and_or_forbidden_raises(self) -> None:
        """
        Cover: BoolOp path -> ValueError(\"Use '&' and '|' instead of 'and'/'or' ...\")
        """
        da = make_image(16, 16)
        roi = select_mask(da, "box[[1pix,1pix],[4pix,4pix]]")
        bad = select_mask(da, "box[[3pix,3pix],[6pix,6pix]]")
        with pytest.raises(ValueError) as ei:
            select_mask(da, select="roi and bad", mask_source={"roi": roi, "bad": bad})
        assert ei.value.args[0] == "Use '&' and '|' instead of 'and'/'or' in selection expressions"

    def test_expression_bitwise_and_or_xor_return_paths(self) -> None:
        """
        Cover returns for BinOp branches:
          - BitAnd  -> left & right
          - BitOr   -> left | right
          - BitXor  -> left ^ right
        via public API with named masks.
        """
        data = np.zeros((3, 3), dtype=float)
        A = np.array(
            [[1, 0, 1],
             [0, 1, 0],
             [1, 1, 0]],
            dtype=int,
        )
        B = np.array(
            [[0, 1, 1],
             [1, 0, 0],
             [1, 0, 1]],
            dtype=int,
        )
        A_b = A.astype(bool); B_b = B.astype(bool)
        env = {"A": A, "B": B}
        m_and = select_mask(data, select="A & B", mask_source=env)
        m_or  = select_mask(data, select="A | B", mask_source=env)
        m_xor = select_mask(data, select="A ^ B", mask_source=env)
        np.testing.assert_array_equal(m_and, (A_b & B_b))
        np.testing.assert_array_equal(m_or,  (A_b | B_b))
        np.testing.assert_array_equal(m_xor, (A_b ^ B_b))

    def test_expression_constant_true_false_returns_bool_arrays(self) -> None:
        """
        Cover Constant(bool) branch:
          - 'True'  -> np.array(True, dtype=bool) aligned to data
          - 'False' -> np.array(False, dtype=bool) aligned to data
        Provide a dummy mask_source so expression evaluation path is taken.
        """
        data = np.zeros((4, 5), dtype=float)
        dummy = np.zeros((1, 1), dtype=bool)  # not used by the expr
        env = {"dummy": dummy}
        m_true = select_mask(data, select="True", mask_source=env, return_kind="numpy")
        m_false = select_mask(data, select="False", mask_source=env, return_kind="numpy")
        assert isinstance(m_true, np.ndarray) and isinstance(m_false, np.ndarray)
        assert m_true.shape == data.shape and m_false.shape == data.shape
        assert m_true.dtype == bool and m_false.dtype == bool
        assert m_true.all() and not m_false.any()

class TestToBoolCasting:
    def test_numeric_masks_nan_to_false_and_casting(self) -> None:
        """
        Cover numpy path in boolean casting (_to_bool on numpy arrays) and avoid
        xarray dim cross-products by using ndarray `data`:
            arr_np = np.asarray(arr)
            if np.issubdtype(arr_np.dtype, np.floating):
                arr_np = np.nan_to_num(arr_np, nan=0.0)
            return arr_np.astype(bool)
        We supply float arrays with NaNs through the public expression API.
        """
        data = np.zeros((3, 3), dtype=float)  # ndarray input → no xarray broadcasting semantics
        # Float arrays with zeros, nonzeros, and NaNs
        A = np.array(
            [
                [np.nan, 0.0, 0.1],
                [-2.0,   0.0, np.nan],
                [0.0,    1.0, 0.0],
            ],
            dtype=float,
        )
        B = np.array(
            [
                [0.0,  1.0, np.nan],
                [0.0,  0.0,  3.0],
                [np.nan, 0.0, 0.0],
            ],
            dtype=float,
        )
        # Expected booleanization: NaN -> 0.0, nonzero -> True, zero -> False
        A_bool = np.nan_to_num(A, nan=0.0).astype(bool)
        B_bool = np.nan_to_num(B, nan=0.0).astype(bool)
        exp_or = A_bool | B_bool
        exp_and_not = A_bool & ~B_bool
        # Use public API with named-mask expression to trigger _to_bool on numpy arrays
        m_or = select_mask(
            data,
            select="A | B",
            mask_source={"A": A, "B": B},
            return_kind="numpy",
        )
        m_and_not = select_mask(
            data,
            select="A & ~B",
            mask_source={"A": A, "B": B},
            return_kind="numpy",
        )
        assert isinstance(m_or, np.ndarray) and isinstance(m_and_not, np.ndarray)
        assert m_or.dtype == bool and m_and_not.dtype == bool
        assert m_or.shape == (3, 3) and m_and_not.shape == (3, 3)
        np.testing.assert_array_equal(m_or, exp_or)
        np.testing.assert_array_equal(m_and_not, exp_and_not)

# ------------------------- CRTF file input -------------------------

class TestCRTFFile:
    def test_backticked_file_and_path_object_match_inline(self, tmp_path: Path) -> None:
        da = make_image(140, 140)
        text = (
            "#CRTF\n"
            "+circle[[70pix,70pix], 40pix]\n"
            "-rotbox[[70pix,70pix],[30pix,20pix], 30]\n"
            "+box[[10pix,110pix],[40pix,130pix]]\n"
        )
        p = tmp_path / "roi.crtf"
        p.write_text(text, encoding="utf-8")

        m_bt = select_mask(da, select=f"`{p.as_posix()}`")
        m_path = select_mask(da, select=p)
        m_inline = select_mask(da, select=text)
        assert (m_bt == m_inline).all()
        assert (m_path == m_inline).all()

    def test_missing_backticked_file_raises_file_not_found(self, tmp_path: Path) -> None:
        da = make_image(40, 50)
        missing = tmp_path / "nope.crtf"
        with pytest.raises(FileNotFoundError):
            select_mask(da, select=f"`{missing.as_posix()}`")

    def test_missing_path_object_raises_file_not_found(self, tmp_path: Path) -> None:
        da = make_image(20, 20)
        missing = tmp_path / "does_not_exist.crtf"
        with pytest.raises(FileNotFoundError) as ei:
            select_mask(da, select=missing)
        assert "CRTF file not found" in str(ei.value)

    def test_plain_string_not_backticked_is_parsed_as_text_via_public_api(self) -> None:
        """
        Public API only: a non-backticked string must be treated as CRTF/expr text
        (i.e., not a file), and produce a valid mask.
        """
        da = make_image(20, 20)
        s = "box[[2pix,2pix],[5pix,5pix]]"  # plain string, not backticked
        m = select_mask(da, select=s)
        assert isinstance(m, xr.DataArray)
        assert m.dtype == bool and m.shape == da.shape
        assert bool(m.values.any())

    def test_non_string_non_path_select_raises_typeerror_public_api(self) -> None:
        """
        Public API only: passing a non-str/non-Path (e.g., bytes) never enters
        _maybe_read_crtf_from_path and must raise TypeError from select_mask.
        This is the closest public-facing behavior to the helper's final `return None`.
        """
        da = make_image(10, 10)
        with pytest.raises(TypeError) as ei:
            select_mask(da, select=b"not a string or path")  # bytes
        assert "Unsupported select type" in str(ei.value)

# NOTE: The helper `_maybe_read_crtf_from_path(sel)` ends with `return None`
# when `sel` is neither `Path` nor `str`. That exact line is unreachable via
# public APIs (which only call the helper for `str`/`Path`). To cover it,
# we'd need either a public wrapper (e.g., `read_crtf(...)`) or to mark the
# final return with `# pragma: no cover`.

# ------------------------- Error messages (pix required) -------------------------


class TestErrorMessages:
    def test_box_without_pix_suggests_pix_units(self) -> None:
        da = make_image(60, 60)
        s = "#CRTF\nbox[[ 30, 40 ], [ 40, 50 ]]"  # missing 'pix'
        with pytest.raises(ValueError) as ei:
            select_mask(da, select=s)
        msg = str(ei.value)
        # Expect helpful suggestion with exact phrasing
        assert "Invalid pixel pair token (require 'pix' units): '[ 30, 40 ]'" in msg
        assert "should be '[30pix, 40pix]'" in msg

    def test_centerbox_without_pix_in_sizes_suggests_pix_units(self) -> None:
        da = make_image(60, 60)
        s = "centerbox[[30pix,40pix],[ 10, 20 ]]"  # missing 'pix' in widths
        with pytest.raises(ValueError) as ei:
            select_mask(da, select=s)
        msg = str(ei.value)
        assert "Invalid pixel pair token (require 'pix' units): '[ 10, 20 ]'" in msg
        assert "should be '[10pix, 20pix]'" in msg

class TestUnsupportedSelectType:
    def test_unsupported_select_type_typeerror_message(self) -> None:
        da = make_image(8, 8)
        with pytest.raises(TypeError) as ei:
            # invalid type for `select` (neither None/array/str/Path)
            select_mask(da, select=123)  # type: ignore[arg-type]
        expected = (
            "Unsupported select type. Expected None, boolean array-like, expression/CRTF text, "
            "or a backticked CRTF file string / pathlib.Path."
        )
        assert str(ei.value) == expected

# ------------------------- apply_select -------------------------


class TestApplySelect:
    def test_apply_select_sets_outside_to_nan(self) -> None:
        da = make_image(40, 50)
        s = "box[[5pix, 6pix],[14pix, 16pix]]"
        m = select_mask(da, select=s)
        out = apply_select(da, select=s)
        assert isinstance(out, xr.DataArray)
        assert np.isnan(out.values[~m.values]).all()
        assert np.isfinite(out.values[m.values]).all()

    def test_full_image_box_selects_all_pixels(self) -> None:
        ny, nx = 32, 48
        da = make_image(ny, nx)
        s = f"box[[0pix,0pix],[{nx-1}pix,{ny-1}pix]]"
        m = select_mask(da, select=s)
        assert int(m.values.sum()) == ny * nx

    def test_apply_select_numpy_branch_sets_outside_nan_preserves_inside(self) -> None:
        ny, nx = 4, 5
        data = np.arange(ny * nx, dtype=float).reshape(ny, nx)
        sel = "box[[1pix,1pix],[3pix,2pix]]"  # inclusive, x:1..3, y:1..2
        out = apply_select(data, select=sel)
        assert isinstance(out, np.ndarray)
        assert out.shape == data.shape
        # Verify NaNs outside and equality inside using the public mask
        mask = select_mask(data, select=sel)
        assert np.isnan(out[~mask]).all()
        np.testing.assert_array_equal(out[mask], data[mask])

# ------------------------- Polygons (point-in-polygon) -------------------------

class TestPolygon:
    def test_convex_square_membership_and_invariance(self) -> None:
        da = make_image(64, 64)
        # Axis-aligned square
        poly = "poly[[10pix,10pix],[30pix,10pix],[30pix,30pix],[10pix,30pix]]"
        m = select_mask(da, select=poly)
        # Reversed order should match
        poly_rev = "poly[[10pix,30pix],[30pix,30pix],[30pix,10pix],[10pix,10pix]]"
        m_rev = select_mask(da, select=poly_rev)
        assert (m == m_rev).all()
        # Closed polygon (repeat first vertex) should also match
        poly_closed = (
            "poly[[10pix,10pix],[30pix,10pix],[30pix,30pix],[10pix,30pix],[10pix,10pix]]"
        )
        m_closed = select_mask(da, select=poly_closed)
        assert (m == m_closed).all()
        # Interior points (well away from edges)
        inside_pts = [(12, 12), (20, 20), (28, 28)]  # (x, y)
        for x, y in inside_pts:
            assert bool(m.values[y, x]) is True
        # Outside points
        outside_pts = [(9, 9), (31, 31), (40, 10)]
        for x, y in outside_pts:
            assert bool(m.values[y, x]) is False

    def test_concave_arrow_shape_includes_and_excludes_expected_points(self) -> None:
        da = make_image(80, 100)
        # Right-pointing arrow (concave)
        poly = (
            "poly["
            "[10pix,20pix],[50pix,20pix],[50pix,15pix],"
            "[70pix,30pix],[50pix,45pix],[50pix,40pix],[10pix,40pix]"
            "]"
        )
        m = select_mask(da, select=poly)
        # Clearly inside near the arrow head
        for (x, y) in [(65, 30), (55, 30), (52, 35)]:
            assert bool(m.values[y, x]) is True
        # Note: points exactly on the polygon edges (e.g., x=50 vertical edge)
        # Clearly outside in the concavity and far away
        # Note: points near the rectangle interior can be inside; avoid ambiguous edge/near-edge picks.
        for (x, y) in [(45, 17), (5, 5), (90, 10)]:
            assert bool(m.values[y, x]) is False

    def test_polygon_with_float_vertices_behaves_sensibly(self) -> None:
        da = make_image(60, 60)
        poly = (
            "poly[[10.5pix,10.5pix],[30.5pix,10.5pix],"
            "[30.5pix,30.5pix],[10.5pix,30.5pix]]"
        )
        m = select_mask(da, select=poly)
        # Pixels strictly inside should be True
        for (x, y) in [(12, 12), (20, 20), (29, 29)]:
            assert bool(m.values[y, x]) is True
        # Pixels well outside should be False
        for (x, y) in [(9, 9), (31, 31)]:
            assert bool(m.values[y, x]) is False

    def test_polygon_file_roundtrip_matches_inline(self, tmp_path: Path) -> None:
        da = make_image(50, 50)
        text = "poly[[5pix,5pix],[20pix,5pix],[20pix,20pix],[5pix,20pix]]"
        p = tmp_path / "poly.crtf"
        p.write_text("#CRTF\n" + text + "\n", encoding="utf-8")
        m_inline = select_mask(da, select=text)
        m_bt = select_mask(da, select=f"`{p.as_posix()}`")
        assert (m_inline == m_bt).all()

# ------------------------- NumPy mask alignment path -------------------------

class TestNumpyMaskAlignment:
    def test_numpy_float_mask_nan_to_false_and_broadcast(self) -> None:
        """
        Exercise the NumPy path in _align_bool_mask_to_data:
          - float mask with NaNs → NaN->0.0 via nan_to_num
          - bool cast
          - broadcasting to data shape
        using only the public API.
        """
        ny, nx = 4, 6
        data = np.zeros((ny, nx), dtype=float)  # ndarray input → NumPy path
        # Column vector with NaNs and floats; shape (ny, 1), broadcasts across x
        col = np.array([[np.nan], [1.0], [0.0], [np.nan]], dtype=float)
        mask = select_mask(data, select=col, return_kind="numpy")
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool and mask.shape == (ny, nx)
        # Expected: NaN->False, 1.0->True, 0.0->False, broadcast across columns
        expected_row = np.array([False] * nx, dtype=bool)
        assert np.array_equal(mask[0], expected_row)        # nan -> False
        assert np.array_equal(mask[1], ~expected_row)       # 1.0 -> True
        assert np.array_equal(mask[2], expected_row)        # 0.0 -> False
        assert np.array_equal(mask[3], expected_row)        # nan -> False

    def test_numpy_mask_broadcast_error_raises(self) -> None:
        """
        Verify ValueError('Mask is not broadcastable to data shape') is raised
        when the mask cannot be broadcast to the data shape.
        """
        ny, nx = 4, 6
        data = np.zeros((ny, nx), dtype=float)
        bad = np.array([1, 0, 1], dtype=int)  # shape (3,), not broadcastable to (4,6)
        with pytest.raises(ValueError) as ei:
            _ = select_mask(data, select=bad)
        assert "Mask is not broadcastable to data shape" in str(ei.value)

# ------------------------- All-true mask (select=None) -------------------------

class TestAllTrueMaskLike:
    def test_none_select_numpy_returns_all_true(self) -> None:
        data = np.zeros((3, 4), dtype=float)
        m = select_mask(data, select=None)
        assert isinstance(m, np.ndarray)
        assert m.dtype == bool
        assert m.shape == data.shape
        assert m.all()

    def test_none_select_xarray_returns_all_true(self) -> None:
        da = make_image(5, 7)
        m = select_mask(da, select=None)
        assert isinstance(m, xr.DataArray)
        assert m.dtype == bool
        assert m.dims == da.dims
        assert bool(m.values.all())

# ------------------------- Smart split pairs (coverage of tail append/return) -------------------------

class TestSmartSplitPairs:
    def test_annulus_equals_circle_diff_and_trims_trailing_space(self) -> None:
        """
        Exercise _smart_split_pairs' final buffer append and return by using an
        'annulus' payload where the last token is a nested bracket pair and the
        string ends without a trailing comma (plus trailing spaces).
        Compare against an equivalent construction using two circles.
        """
        ny = nx = 120
        da = make_image(ny, nx)
        cx, cy = 50, 50
        r1, r2 = 10, 20
        # Trailing whitespace ensures the final 'if buf: parts.append(...)' path is taken.
        ann = f"annulus[[{cx}pix,{cy}pix], [ {r1}pix, {r2}pix]]   "
        m_ann = select_mask(da, select=ann)
        m_outer = select_mask(da, select=f"circle[[{cx}pix,{cy}pix], {r2}pix]")
        m_inner = select_mask(da, select=f"circle[[{cx}pix,{cy}pix], {r1}pix]")
        # Annulus includes the inner boundary (>= r1), while (outer & ~inner)
        # excludes it. Instead of equality, assert subset relations:
        # 1) Annulus is a subset of the outer circle
        assert bool((m_ann & ~m_outer).values.any()) is False
        # 2) A strict ring (outer minus a slightly smaller inner) is contained in annulus
        m_inner_grow = select_mask(da, select=f"circle[[{cx}pix,{cy}pix], {r1+1}pix]")
        m_ring_subset = m_outer & ~m_inner_grow
        assert bool((m_ring_subset & ~m_ann).values.any()) is False


# ------------------------- Xarray mask alignment (NaN -> False via fillna) -------------------------

class TestXarrayMaskAlignment:
    def test_xarray_float_mask_with_nan_fillna_false(self) -> None:
        """
        Hit the nested branch in _align_bool_mask_to_data for xarray:
            if isinstance(data, xr.DataArray):
                m = ...  # DataArray
                if np.issubdtype(m.dtype, np.floating):
                    m = m.fillna(False)
        by passing a float DataArray with NaNs as the mask.
        """
        da = make_image(3, 4)  # xr.DataArray with dims ('y','x')
        vals = np.array(
            [
                [np.nan, 0.0, 1.0, np.nan],
                [0.0,    2.0, 0.0, 0.0   ],
                [np.nan, 0.0, 3.0, 0.0   ],
            ],
            dtype=float,
        )
        mask_da = xr.DataArray(vals, dims=("y", "x"), coords=da.coords)
        out = select_mask(da, select=mask_da)
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool
        # Expected: NaN -> False, nonzero -> True, zero -> False
        expected = np.nan_to_num(vals, nan=0.0).astype(bool)
        np.testing.assert_array_equal(out.values, expected)

# ------------------------- Return kinds -------------------------

class TestReturnKinds:
    def test_return_kind_numpy_from_expression(self) -> None:
        data = np.zeros((6, 8), dtype=float)
        A = np.eye(6, 8, dtype=int)
        B = np.zeros((6, 8), dtype=float); B[::2, ::2] = 1.0
        m = select_mask(data, select="A | B", mask_source={"A": A, "B": B}, return_kind="numpy")
        assert isinstance(m, np.ndarray)
        assert m.dtype == bool and m.shape == data.shape

    def test_return_kind_dask_from_crtf(self) -> None:
        data = np.zeros((50, 60), dtype=float)
        s = "circle[[30pix,25pix], 10pix]"
        m = select_mask(data, select=s, return_kind="dask", dask_chunks=(20, 20))
        # dask-backed boolean array
        assert isinstance(m, da.Array)
        assert m.dtype == bool and m.shape == (50, 60)
        assert m.chunks is not None

    def test_return_kind_dataarray_numpy(self) -> None:
        da_img = xr.DataArray(np.zeros((12, 10)), dims=("y", "x"))
        s = "box[[2pix,3pix],[7pix,8pix]]"
        m = select_mask(da_img, select=s, return_kind="dataarray-numpy")
        assert isinstance(m, xr.DataArray)
        assert m.dtype == bool and m.dims == da_img.dims and m.shape == da_img.shape
        # numpy-backed
        assert not hasattr(m.data, "chunks")

    def test_return_kind_dataarray_dask_and_apply_select(self) -> None:
        darr = da.zeros((40, 30), chunks=(16, 10))
        da_img = xr.DataArray(darr, dims=("y", "x"))
        s = "centerbox[[15pix,20pix],[20pix,10pix]]"
        # mask as dask-backed DataArray (default), explicit here for clarity
        m = select_mask(da_img, select=s, return_kind="dataarray-dask")
        assert isinstance(m, xr.DataArray)
        assert hasattr(m.data, "chunks")
        # apply_select should keep laziness on data
        out = apply_select(da_img, select=s)
        assert isinstance(out, xr.DataArray)
        assert hasattr(out.data, "chunks")

    def test_return_kind_wrap_numpy_mask_to_dask_dataarray(self) -> None:
        data = xr.DataArray(da.zeros((25, 25), chunks=(10, 10)), dims=("y", "x"))
        # Provide a small numpy mask; request dask-backed DataArray mask
        m_np = np.zeros((25, 25), dtype=int); m_np[5:10, 5:10] = 1
        m = select_mask(data, select=m_np, return_kind="dataarray-dask")
        assert isinstance(m, xr.DataArray) and hasattr(m.data, "chunks")
        # round-trip through numpy return kind
        m_np_back = select_mask(data, select=m, return_kind="numpy")
        assert isinstance(m_np_back, np.ndarray) and m_np_back.dtype == bool

    def test_dataarray_numpy_from_numpy_mask_numpy_data_hits_else_branch(self) -> None:
        """
        Cover the 'dataarray-numpy' else branch where `mask` is NOT an xr.DataArray:
          if getattr(mask, "__module__", "").startswith("dask"):  # False here
              ...
          else:
              arr = np.asarray(mask, dtype=bool)
          dims/coords derived from numpy `data` fallback.
        """
        ny, nx = 6, 7
        data = np.zeros((ny, nx), dtype=float)  # numpy data → dims fallback ("y","x")
        mask_np = np.zeros((ny, nx), dtype=int); mask_np[2:4, 3:5] = 1
        out = select_mask(data, select=mask_np, return_kind="dataarray-numpy")
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool and out.shape == (ny, nx)
        # numpy-backed (no dask chunks)
        assert not hasattr(out.data, "chunks")
        # dims fallback
        assert out.dims == ("y", "x")
        np.testing.assert_array_equal(out.values, mask_np.astype(bool))

    def test_dataarray_numpy_from_xarray_dask_mask_xarray_data_computes(self) -> None:
        """
        Public API: when the input mask is an xarray.DataArray backed by dask and
        return_kind=\"dataarray-numpy\" is requested, the result is an xr.DataArray
        computed to NumPy with identical shape/dims and boolean dtype.
        """
        ny, nx = 5, 8
        data_da = xr.DataArray(np.zeros((ny, nx)), dims=("y", "x"))
        mask_da = xr.DataArray(da.ones((ny, nx), chunks=(3, 4)), dims=("y", "x"))
        out = select_mask(data_da, select=mask_da, return_kind="dataarray-numpy")
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool and out.shape == (ny, nx)
        assert out.dims == ("y", "x")
        assert not hasattr(out.data, "chunks")
        assert out.values.all()

    # ---------------- cover _coerce_return_kind return_kind == "dask" paths ----------------

    def test_return_kind_dask_from_xarray_dask_mask_returns_dask_array(self) -> None:
        """
        If mask is xr.DataArray with dask backing, return_kind='dask' returns arr.astype(bool).
        """
        ny, nx = 12, 10
        data_da = xr.DataArray(np.zeros((ny, nx)), dims=("y", "x"))
        mask_da = xr.DataArray(da.ones((ny, nx), chunks=(6, 5)), dims=("y", "x"))  # dask-backed
        out = select_mask(data_da, select=mask_da, return_kind="dask")
        assert isinstance(out, da.Array)
        assert out.dtype == bool and out.shape == (ny, nx)
        # should preserve chunking from mask (or compatible)
        assert out.chunks is not None

    def test_return_kind_dask_wraps_xarray_numpy_mask_using_inferred_or_given_chunks(self) -> None:
        """
        If mask is xr.DataArray with NumPy backing, it is wrapped via da.from_array.
        """
        ny, nx = 9, 12
        data_da = xr.DataArray(np.zeros((ny, nx)), dims=("y", "x"))
        mask_np_da = xr.DataArray(np.zeros((ny, nx), dtype=int), dims=("y", "x"))
        mask_np_da.values[2:7, 3:9] = 1
        out = select_mask(data_da, select=mask_np_da, return_kind="dask", dask_chunks=(3, 4))
        assert isinstance(out, da.Array)
        assert out.dtype == bool and out.shape == (ny, nx)
        # chunking should follow the explicit dask_chunks when provided
        assert tuple(c[0] for c in out.chunks) == (3, 4)

    def test_return_kind_dask_wraps_numpy_mask_when_data_is_numpy(self) -> None:
        """
        If mask is a NumPy array (not xarray), it is wrapped via da.from_array.
        """
        ny, nx = 8, 11
        data = np.zeros((ny, nx), dtype=float)
        mask_np = np.zeros((ny, nx), dtype=int); mask_np[1:4, 2:9] = 1
        out = select_mask(data, select=mask_np, return_kind="dask", dask_chunks=(4, 5))
        assert isinstance(out, da.Array)
        assert out.dtype == bool and out.shape == (ny, nx)
        assert tuple(c[0] for c in out.chunks) == (4, 5)

    # -------- cover _infer_chunks_like: success and exception paths via public API --------

    def test_dask_return_kind_infers_chunks_from_data_success(self) -> None:
        """
        When `data` is an xr.DataArray with a single chunk per axis (e.g., (ny,),(nx,)),
        _infer_chunks_like flattens to (ny, nx) and returns it. The 'dask' return kind
        for a NumPy-backed mask DataArray should therefore use those exact chunks.
        """
        ny, nx = 18, 22
        data_da = xr.DataArray(da.zeros((ny, nx), chunks=(ny, nx)), dims=("y", "x"))
        # NumPy-backed mask DataArray so _coerce_return_kind hits the infer-chunks branch
        mask_np_da = xr.DataArray(np.zeros((ny, nx), dtype=int), dims=("y", "x"))
        mask_np_da.values[4:10, 5:15] = 1
        out = select_mask(data_da, select=mask_np_da, return_kind="dask")
        assert isinstance(out, da.Array)
        assert out.dtype == bool and out.shape == (ny, nx)
        # Expect a single chunk per axis matching data_da's single-chunk shape
        assert out.chunks[0] == (ny,)
        assert out.chunks[1] == (nx,)

    def test_dask_return_kind_infer_chunks_exception_path_without_monkeypatch(self) -> None:
        """
        Trigger the exception path in _infer_chunks_like by supplying an xarray.DataArray
        whose underlying .data has a bogus ``chunks`` attribute that cannot be cast to ints.
        This avoids monkeypatching dask internals and uses only the public API.
        """
        class BadChunks(np.ndarray):  # numpy subclass with a non-numeric 'chunks' attribute
            @property
            def chunks(self):
                return ("bad",)

        def with_bad_chunks(shape: tuple[int, int]) -> np.ndarray:
            base = np.zeros(shape, dtype=float)
            return base.view(BadChunks)

        ny, nx = 20, 14
        # xarray.DataArray with numpy-subclass backing that exposes a bogus .chunks
        data_da = xr.DataArray(with_bad_chunks((ny, nx)), dims=("y", "x"))
        # NumPy-backed mask DataArray so _coerce_return_kind hits the infer path
        mask_np_da = xr.DataArray(np.zeros((ny, nx), dtype=int), dims=("y", "x"))

        out = select_mask(data_da, select=mask_np_da, return_kind="dask")
        assert isinstance(out, da.Array)
        assert out.dtype == bool and out.shape == (ny, nx)
        # We don't assert exact chunking (implementation-defined fallback), only that it succeede

class TestCRTFDirectives:
    def test_global_directive_lines_are_ignored(self) -> None:
        """
        Cover the 'continue' path that skips lines starting with 'global' in CRTF:
            if line.lower().startswith("global"): continue
        Ensure masks are identical with/without a global line.
        """
        da = make_image(64, 64)
        region = "box[[10pix,12pix],[30pix,40pix]]"
        with_global = "\n".join([
            "#CRTF",
            "global coord=pixel",          # should be ignored
            region,
        ])
        without_global = "\n".join([
            "#CRTF",
            region,
        ])
        m_with = select_mask(da, select=with_global)
        m_without = select_mask(da, select=without_global)
        assert isinstance(m_with, xr.DataArray) and isinstance(m_without, xr.DataArray)
        assert m_with.shape == m_without.shape == da.shape
        np.testing.assert_array_equal(m_with.values, m_without.values)
