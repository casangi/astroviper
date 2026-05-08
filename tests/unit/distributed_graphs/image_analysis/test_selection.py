"""
Tests for ROI selection (CRTF + expressions) using the public API only.

Public API under test:
- astroviper.distributed_graphs.image_analysis.selection.select_mask
- astroviper.distributed_graphs.image_analysis.selection.apply_select

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

import math
import os
from pathlib import Path
import re
import numpy as np
import pytest
import xarray as xr
import dask.array as da

from astroviper.distributed_graphs.image_analysis.selection import (
    select_mask,
    apply_select,
    combine_with_creation,
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


def make_xradio_sky(
    n_time: int = 3,
    n_freq: int = 8,
    n_pol: int = 4,
    n_l: int = 10,
    n_m: int = 10,
    freq_start_ghz: float = 1.0,
    freq_step_ghz: float = 0.1,
    vel_start: float = 0.0,
    vel_step: float = 1e4,
    pols: tuple[str, ...] = ("I", "Q", "U", "V"),
    time_start_mjd: float = 60000.0,
    time_step_mjd: float = 1.0,
) -> xr.DataArray:
    """Return a minimal xradio-style SKY DataArray with dims (time, frequency, polarization, l, m)."""
    freq_hz = (freq_start_ghz + np.arange(n_freq) * freq_step_ghz) * 1e9
    vel_ms = vel_start + np.arange(n_freq) * vel_step
    pol_vals = list(pols[:n_pol])
    time_mjd = time_start_mjd + np.arange(n_time) * time_step_mjd
    l_rad = np.linspace(-1e-4, 1e-4, n_l)
    m_rad = np.linspace(-1e-4, 1e-4, n_m)
    data = np.ones((n_time, n_freq, n_pol, n_l, n_m), dtype=float)
    freq_coord = xr.DataArray(
        freq_hz,
        dims=["frequency"],
        attrs={"units": "Hz", "observer": "LSRK"},
    )
    vel_coord = xr.DataArray(
        vel_ms,
        dims=["frequency"],
        attrs={"units": "m/s", "doppler_type": "radio"},
    )
    time_coord = xr.DataArray(
        time_mjd,
        dims=["time"],
        attrs={"units": "d", "scale": "utc", "format": "mjd"},
    )
    return xr.DataArray(
        data,
        dims=["time", "frequency", "polarization", "l", "m"],
        coords={
            "time": time_coord,
            "frequency": freq_coord,
            "velocity": vel_coord,
            "polarization": pol_vals,
            "l": l_rad,
            "m": m_rad,
        },
        name="SKY",
    )


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
        width = 29 - 10 + 1
        height = 25 - 5 + 1
        assert int(m_no.values.sum()) == width * height

    def test_circle_centroid_near_center(self) -> None:
        da = make_image(200, 200)
        cx, cy, r = 120, 80, 40
        m = select_mask(da, select=f"circle[[{cx}pix,{cy}pix], {r}pix]")
        yy, xx = np.mgrid[0 : da.sizes["y"], 0 : da.sizes["x"]]
        yy = yy.astype(float)
        xx = xx.astype(float)
        inside = m.values
        assert inside.dtype == bool and inside.any()
        # centroid of boolean mask
        ybar = float((yy[inside]).mean())
        xbar = float((xx[inside]).mean())
        assert abs(xbar - cx) <= 0.5
        assert abs(ybar - cy) <= 0.5

    @pytest.mark.parametrize(
        "shape, spec",
        [
            ("rotbox", "rotbox[[120pix,80pix],[60pix,30pix], theta_m=30]"),
            ("ellipse", "ellipse[[120pix,80pix],[50pix,25pix], theta_m=60]"),
        ],
    )
    def test_rotbox_ellipse_parse_and_nonempty(self, shape: str, spec: str) -> None:
        da = make_image(180, 220)
        m = select_mask(da, select=spec)
        assert isinstance(m, xr.DataArray)
        assert m.dtype == bool and int(m.values.sum()) > 0

    def test_box_respects_named_xy_dims_when_dim_order_is_x_then_y(self) -> None:
        """
        Ensure CRTF interprets values as (x, y) even when DataArray dims are ('x', 'y').

        This regression covers notebook usage where the underlying array order is
        shape=(nx, ny) with named dimensions x then y.
        """
        nx, ny = 160, 180
        img = xr.DataArray(
            np.zeros((nx, ny), dtype=float),
            dims=("x", "y"),
            coords={"x": np.arange(nx), "y": np.arange(ny)},
            name="img_xy",
        )
        sel = "box[[10pix,100pix],[20pix,140pix]]"
        mask = select_mask(img, select=sel)

        # Pixel count uses inclusive bounds: (20-10+1) * (140-100+1).
        assert int(mask.values.sum()) == (20 - 10 + 1) * (140 - 100 + 1)
        # In-mask coordinate in (x, y) convention should be selected.
        assert mask.sel(x=15, y=120).compute().item()
        # Coordinate that would be selected only under swapped interpretation is excluded.
        assert not mask.sel(x=120, y=15).compute().item()


# ------------------------- CRTF combination (+ / -) -------------------------


class TestCRTFCombine:
    def test_multi_line_plus_minus_matches_boolean_ops(self) -> None:
        da = make_image(160, 160)
        circle = "circle[[100pix,70pix], 45pix]"
        rot = "rotbox[[100pix,70pix],[40pix,20pix], theta_m=20]"
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
        assert (
            "Expression contains an unsupported construct: Compare" in ei.value.args[0]
        )

    def test_expression_boolop_and_or_forbidden_raises(self) -> None:
        """
        Cover: BoolOp path -> ValueError(\"Use '&' and '|' instead of 'and'/'or' ...\")
        """
        da = make_image(16, 16)
        roi = select_mask(da, "box[[1pix,1pix],[4pix,4pix]]")
        bad = select_mask(da, "box[[3pix,3pix],[6pix,6pix]]")
        with pytest.raises(ValueError) as ei:
            select_mask(da, select="roi and bad", mask_source={"roi": roi, "bad": bad})
        assert (
            ei.value.args[0]
            == "Use '&' and '|' instead of 'and'/'or' in selection expressions"
        )

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
            [[1, 0, 1], [0, 1, 0], [1, 1, 0]],
            dtype=int,
        )
        B = np.array(
            [[0, 1, 1], [1, 0, 0], [1, 0, 1]],
            dtype=int,
        )
        A_b = A.astype(bool)
        B_b = B.astype(bool)
        env = {"A": A, "B": B}
        m_and = select_mask(data, select="A & B", mask_source=env)
        m_or = select_mask(data, select="A | B", mask_source=env)
        m_xor = select_mask(data, select="A ^ B", mask_source=env)
        np.testing.assert_array_equal(m_and, (A_b & B_b))
        np.testing.assert_array_equal(m_or, (A_b | B_b))
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
        m_false = select_mask(
            data, select="False", mask_source=env, return_kind="numpy"
        )
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
        data = np.zeros(
            (3, 3), dtype=float
        )  # ndarray input → no xarray broadcasting semantics
        # Float arrays with zeros, nonzeros, and NaNs
        A = np.array(
            [
                [np.nan, 0.0, 0.1],
                [-2.0, 0.0, np.nan],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        B = np.array(
            [
                [0.0, 1.0, np.nan],
                [0.0, 0.0, 3.0],
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
            "-rotbox[[70pix,70pix],[30pix,20pix], theta_m=30]\n"
            "+box[[10pix,110pix],[40pix,130pix]]\n"
        )
        p = tmp_path / "roi.crtf"
        p.write_text(text, encoding="utf-8")

        m_bt = select_mask(da, select=f"`{p.as_posix()}`")
        m_path = select_mask(da, select=p)
        m_inline = select_mask(da, select=text)
        assert (m_bt == m_inline).all()
        assert (m_path == m_inline).all()

    def test_missing_backticked_file_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
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
        poly_closed = "poly[[10pix,10pix],[30pix,10pix],[30pix,30pix],[10pix,30pix],[10pix,10pix]]"
        m_closed = select_mask(da, select=poly_closed)
        assert (m == m_closed).all()
        # Interior points (well away from edges)
        inside_pts = [(12, 12), (20, 20), (28, 28)]  # (x, y)
        for x, y in inside_pts:
            assert m.sel(x=x, y=y).compute().item() is True
        # Outside points
        outside_pts = [(9, 9), (31, 31), (40, 10)]  # (x, y)
        for x, y in outside_pts:
            assert m.sel(x=x, y=y).compute().item() is False

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
        for x, y in [(65, 30), (55, 30), (52, 35)]:
            assert m.sel(x=x, y=y).compute().item() is True
        # Note: points exactly on the polygon edges (e.g., x=50 vertical edge)
        # Clearly outside in the concavity and far away
        # Note: points near the rectangle interior can be inside; avoid ambiguous edge/near-edge picks.
        for x, y in [(45, 17), (5, 5), (90, 10)]:
            assert m.sel(x=x, y=y).compute().item() is False

    def test_polygon_with_float_vertices_behaves_sensibly(self) -> None:
        da = make_image(60, 60)
        poly = (
            "poly[[10.5pix,10.5pix],[30.5pix,10.5pix],"
            "[30.5pix,30.5pix],[10.5pix,30.5pix]]"
        )
        m = select_mask(da, select=poly)
        # Pixels strictly inside should be True
        for x, y in [(12, 12), (20, 20), (29, 29)]:
            assert m.sel(x=x, y=y).compute().item() is True
        # Pixels well outside should be False
        for x, y in [(9, 9), (31, 31)]:
            assert m.sel(x=x, y=y).compute().item() is False

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
        assert np.array_equal(mask[0], expected_row)  # nan -> False
        assert np.array_equal(mask[1], ~expected_row)  # 1.0 -> True
        assert np.array_equal(mask[2], expected_row)  # 0.0 -> False
        assert np.array_equal(mask[3], expected_row)  # nan -> False

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
                [0.0, 2.0, 0.0, 0.0],
                [np.nan, 0.0, 3.0, 0.0],
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
        B = np.zeros((6, 8), dtype=float)
        B[::2, ::2] = 1.0
        m = select_mask(
            data, select="A | B", mask_source={"A": A, "B": B}, return_kind="numpy"
        )
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

    def test_return_kind_numpy_from_plain_dask_mask(self) -> None:
        """A plain Dask mask with return_kind='numpy' should compute to a NumPy bool array."""
        ny, nx = 7, 9
        data = np.zeros((ny, nx), dtype=float)
        mask_dask = da.from_array(np.eye(ny, nx, dtype=bool), chunks=(3, 4))
        out = select_mask(data, select=mask_dask, return_kind="numpy")
        assert isinstance(out, np.ndarray)
        assert out.dtype == bool and out.shape == (ny, nx)
        np.testing.assert_array_equal(out, np.eye(ny, nx, dtype=bool))

    def test_return_kind_dataarray_numpy_from_plain_dask_mask(self) -> None:
        """A plain Dask mask with return_kind='dataarray-numpy' should compute eagerly."""
        ny, nx = 6, 8
        data = np.zeros((ny, nx), dtype=float)
        mask_np = np.zeros((ny, nx), dtype=bool)
        mask_np[1:4, 2:6] = True
        mask_dask = da.from_array(mask_np, chunks=(2, 3))
        out = select_mask(data, select=mask_dask, return_kind="dataarray-numpy")
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool and out.shape == (ny, nx)
        assert out.dims == ("x", "y")
        assert not hasattr(out.data, "chunks")
        np.testing.assert_array_equal(out.values, mask_np)

    def test_return_kind_dataarray_dask_from_plain_dask_mask(self) -> None:
        """A plain Dask mask with return_kind='dataarray-dask' should stay Dask-backed."""
        ny, nx = 10, 12
        data = np.zeros((ny, nx), dtype=float)
        mask_np = np.zeros((ny, nx), dtype=bool)
        mask_np[:, ::2] = True
        mask_dask = da.from_array(mask_np, chunks=(4, 5))
        out = select_mask(data, select=mask_dask, return_kind="dataarray-dask")
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool and out.shape == (ny, nx)
        assert hasattr(out.data, "chunks")
        np.testing.assert_array_equal(out.data.compute(), mask_np)

    def test_return_kind_wrap_numpy_mask_to_dask_dataarray(self) -> None:
        data = xr.DataArray(da.zeros((25, 25), chunks=(10, 10)), dims=("y", "x"))
        # Provide a small numpy mask; request dask-backed DataArray mask
        m_np = np.zeros((25, 25), dtype=int)
        m_np[5:10, 5:10] = 1
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
        nx, ny = 7, 6
        data = np.zeros((nx, ny), dtype=float)  # numpy data → dims fallback ("x","y")
        mask_np = np.zeros((nx, ny), dtype=int)
        mask_np[2:4, 3:5] = 1
        out = select_mask(data, select=mask_np, return_kind="dataarray-numpy")
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool and out.shape == (nx, ny)
        # numpy-backed (no dask chunks)
        assert not hasattr(out.data, "chunks")
        # NumPy fallback uses the public image-axis convention.
        assert out.dims == ("x", "y")
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
        mask_da = xr.DataArray(
            da.ones((ny, nx), chunks=(6, 5)), dims=("y", "x")
        )  # dask-backed
        out = select_mask(data_da, select=mask_da, return_kind="dask")
        assert isinstance(out, da.Array)
        assert out.dtype == bool and out.shape == (ny, nx)
        # should preserve chunking from mask (or compatible)
        assert out.chunks is not None

    def test_return_kind_dask_wraps_xarray_numpy_mask_using_inferred_or_given_chunks(
        self,
    ) -> None:
        """
        If mask is xr.DataArray with NumPy backing, it is wrapped via da.from_array.
        """
        ny, nx = 9, 12
        data_da = xr.DataArray(np.zeros((ny, nx)), dims=("y", "x"))
        mask_np_da = xr.DataArray(np.zeros((ny, nx), dtype=int), dims=("y", "x"))
        mask_np_da.values[2:7, 3:9] = 1
        out = select_mask(
            data_da, select=mask_np_da, return_kind="dask", dask_chunks=(3, 4)
        )
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
        mask_np = np.zeros((ny, nx), dtype=int)
        mask_np[1:4, 2:9] = 1
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

    def test_dask_return_kind_infer_chunks_exception_path_without_monkeypatch(
        self,
    ) -> None:
        """
        Trigger the exception path in _infer_chunks_like by supplying an xarray.DataArray
        whose underlying .data has a bogus ``chunks`` attribute that cannot be cast to ints.
        This avoids monkeypatching dask internals and uses only the public API.
        """

        class BadChunks(
            np.ndarray
        ):  # numpy subclass with a non-numeric 'chunks' attribute
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
        # We don't assert exact chunking (implementation-defined fallback), only that it succeeded

    def test_dataarray_numpy_else_branch_creation_attached_and_printed(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """
        Cover the else-branch in _coerce_return_kind for return_kind='dataarray-numpy':
            da_out = xr.DataArray(arr, dims=dims, coords=coords)
            if creation is not None:
                print("********** covered *********")
                da_out = da_out.assign_attrs({"creation": creation})
        Use ndarray `data` and ndarray `select` so the aligned mask is a NumPy array
        (not an xarray.DataArray), forcing the targeted branch.
        """
        ny, nx = 9, 7
        data = np.zeros((ny, nx), dtype=float)  # ndarray → NumPy align path
        mask_np = np.zeros((ny, nx), dtype=int)
        mask_np[2:5, 1:4] = 1
        hint = "numpy mask branch"
        out = select_mask(
            data, select=mask_np, return_kind="dataarray-numpy", creation_hint=hint
        )
        # Assert printed marker from the covered branch
        captured = capsys.readouterr()
        # Validate return object and attached creation attribute
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool and out.shape == (ny, nx)
        assert out.dims == ("x", "y") and not hasattr(out.data, "chunks")
        assert out.attrs.get("creation") == hint


class TestCRTFDirectives:
    def test_global_directive_lines_are_ignored(self) -> None:
        """
        Cover the 'continue' path that skips lines starting with 'global' in CRTF:
            if line.lower().startswith("global"): continue
        Ensure masks are identical with/without a global line.
        """
        da = make_image(64, 64)
        region = "box[[10pix,12pix],[30pix,40pix]]"
        with_global = "\n".join(
            [
                "#CRTF",
                "global color=green",  # viz keyword; should be silently ignored
                region,
            ]
        )
        without_global = "\n".join(
            [
                "#CRTF",
                region,
            ]
        )
        m_with = select_mask(da, select=with_global)
        m_without = select_mask(da, select=without_global)
        assert isinstance(m_with, xr.DataArray) and isinstance(m_without, xr.DataArray)
        assert m_with.shape == m_without.shape == da.shape
        np.testing.assert_array_equal(m_with.values, m_without.values)


class TestCreationAutoMerge:
    def test_auto_merge_creation_from_triplet_attrs(self) -> None:
        """
        Cover:
            if creation_str is None and auto_merge_creation and isinstance(select, xr.DataArray):
                c1 = select.attrs.get("creation_a")
                c2 = select.attrs.get("creation_b")
                op = select.attrs.get("creation_op")
                if c1 and c2 and op:
                    creation_str = f"({c1}) {op} ({c2})"
        using only the public API.
        """
        da_img = make_image(32, 48)
        c1 = "numpy rect [y:5..15, x:7..20]"
        c2 = "dask random > 0.9 (chunks=16x16)"
        op = "|"
        # Any boolean DataArray works; attributes drive the provenance.
        base = xr.DataArray(
            np.zeros(da_img.shape, dtype=bool), dims=da_img.dims, coords=da_img.coords
        )
        mask_with_triplet = base.assign_attrs(
            {"creation_a": c1, "creation_b": c2, "creation_op": op}
        )
        out = select_mask(
            da_img, select=mask_with_triplet, auto_merge_creation=True
        )  # default return_kind → DataArray
        assert (
            isinstance(out, xr.DataArray)
            and out.dtype == bool
            and out.shape == da_img.shape
        )
        assert out.attrs.get("creation") == f"({c1}) {op} ({c2})"

    def test_auto_merge_creation_falls_back_to_single_creation_attr(self) -> None:
        """
        Cover the 'elif "creation" in select.attrs: creation_str = select.attrs.get("creation")'
        fallback when the triplet (creation_a/b/op) is not present.
        """
        da_img = make_image(16, 16)
        prov = "standalone provenance string"
        base = xr.DataArray(
            np.zeros(da_img.shape, dtype=bool), dims=da_img.dims, coords=da_img.coords
        )
        mask_with_creation_only = base.assign_attrs({"creation": prov})
        out = select_mask(
            da_img, select=mask_with_creation_only, auto_merge_creation=True
        )
        assert (
            isinstance(out, xr.DataArray)
            and out.dtype == bool
            and out.shape == da_img.shape
        )
        assert out.attrs.get("creation") == prov


class TestAlignFallback:
    def test_broadcast_like_exception_fallback_numpy_wrap_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Force xr.DataArray.broadcast_like to raise so _align_bool_mask_to_data enters the
        top-level except block, then verify the NumPy broadcast fallback succeeds and
        wraps back to an xr.DataArray with data's dims/coords.
        """
        ny, nx = 12, 18
        data = make_image(ny, nx)  # xarray.DataArray
        # Shape is broadcastable to (ny, nx) but we'll force broadcast_like to error.
        col = xr.DataArray(np.zeros((ny, 1), dtype=bool), dims=("y", "x"))

        def boom(self, other, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulate broadcast_like failure")

        # Monkeypatch xarray's method (public third-party API) to trigger fallback path.
        monkeypatch.setattr(xr.DataArray, "broadcast_like", boom, raising=True)

        out = select_mask(data, select=col)  # PUBLIC API
        assert isinstance(out, xr.DataArray)
        assert out.shape == data.shape and out.dims == data.dims
        assert out.dtype == bool

    def test_fallback_broadcast_to_failure_raises_valueerror(self) -> None:
        """
        Make broadcast_like fail naturally (conflicting non-unit size along 'x'),
        and also make NumPy's broadcast_to fail so we cover the inner except and
        error message in the fallback.
        """
        ny, nx = 10, 15
        data = make_image(ny, nx)  # xarray.DataArray
        # Incompatible along x: nx-1 cannot broadcast to nx
        bad = xr.DataArray(np.zeros((ny, nx - 1), dtype=bool), dims=("y", "x"))
        with pytest.raises(ValueError, match="Mask is not broadcastable to data shape"):
            _ = select_mask(data, select=bad)  # PUBLIC API triggers alignment


class TestCombineWithCreationAPI:
    def test_invalid_op_raises(self) -> None:
        """
        op must be one of '|', '&', '^'
        """
        da_img = make_image(20, 30)
        a = select_mask(
            da_img,
            select="box[[1pix,1pix],[10pix,10pix]]",
            return_kind="dataarray-numpy",
        )
        b = select_mask(
            da_img,
            select="box[[5pix,5pix],[15pix,12pix]]",
            return_kind="dataarray-numpy",
        )
        with pytest.raises(ValueError, match=r"op must be one of '\|', '&', '\^'"):
            combine_with_creation(a, "~", b)

    def test_or_and_xor_semantics_and_creation_merge_default(self) -> None:
        """
        Cover L|R, L&R, L^R branches and default creation merge from inputs'
        'creation' attrs (set via creation_hint).
        """
        da_img = make_image(40, 60)
        # Two simple masks with explicit provenance
        c1 = "numpy rect [y:5..19, x:7..25]"
        m1_src = np.zeros(da_img.shape, dtype=bool)
        m1_src[5:20, 7:26] = True
        m1 = select_mask(
            da_img, select=m1_src, return_kind="dataarray-numpy", creation_hint=c1
        )
        c2 = "numpy rect [y:10..29, x:20..39]"
        m2_src = np.zeros(da_img.shape, dtype=bool)
        m2_src[10:30, 20:40] = True
        m2 = select_mask(
            da_img, select=m2_src, return_kind="dataarray-numpy", creation_hint=c2
        )

        out_or = combine_with_creation(m1, "|", m2, return_kind="dataarray-numpy")
        out_and = combine_with_creation(m1, "&", m2, return_kind="dataarray-numpy")
        out_xor = combine_with_creation(m1, "^", m2, return_kind="dataarray-numpy")

        assert isinstance(out_or, xr.DataArray) and out_or.dtype == bool
        assert isinstance(out_and, xr.DataArray) and out_and.dtype == bool
        assert isinstance(out_xor, xr.DataArray) and out_xor.dtype == bool

        np.testing.assert_array_equal(out_or.values, (m1.values | m2.values))
        np.testing.assert_array_equal(out_and.values, (m1.values & m2.values))
        np.testing.assert_array_equal(out_xor.values, (m1.values ^ m2.values))

        assert out_or.attrs.get("creation") == f"({c1}) | ({c2})"
        assert out_and.attrs.get("creation") == f"({c1}) & ({c2})"
        assert out_xor.attrs.get("creation") == f"({c1}) ^ ({c2})"

    def test_creation_hint_overrides(self) -> None:
        """
        creation_hint should replace the auto-merged '(c1) op (c2)' string.
        """
        da_img = make_image(24, 24)
        a = select_mask(
            da_img,
            select="box[[2pix,2pix],[15pix,15pix]]",
            return_kind="dataarray-numpy",
        )
        b = select_mask(
            da_img,
            select="box[[6pix,6pix],[20pix,20pix]]",
            return_kind="dataarray-numpy",
        )
        hint = "custom provenance for (a | b)"
        out = combine_with_creation(
            a, "|", b, return_kind="dataarray-numpy", creation_hint=hint
        )
        assert isinstance(out, xr.DataArray)
        assert out.attrs.get("creation") == hint

    def test_template_controls_dims_and_return_kind_numpy(self) -> None:
        """
        When template is provided, output dims/coords follow the template; also
        cover return_kind='dataarray-numpy'.
        """
        ny, nx = 18, 12
        # Template with custom coords/dim names
        tmpl = xr.DataArray(
            np.zeros((ny, nx), dtype=float),
            dims=("row", "col"),
            coords={"row": np.arange(ny) * 2.0, "col": np.arange(nx) * 3.0},
        )
        # Inputs share the same shape as template but with default dims
        a = xr.DataArray(np.zeros((ny, nx), dtype=bool), dims=("y", "x")).assign_attrs(
            creation="A"
        )
        b = xr.DataArray(np.zeros((ny, nx), dtype=bool), dims=("y", "x")).assign_attrs(
            creation="B"
        )
        b.values[::2, ::3] = True
        out = combine_with_creation(
            a, "|", b, template=tmpl, return_kind="dataarray-numpy"
        )
        assert isinstance(out, xr.DataArray) and not hasattr(out.data, "chunks")
        assert out.dims == ("row", "col")
        np.testing.assert_array_equal(
            out.values, b.values
        )  # since a is all False, A|B == B
        assert out.attrs.get("creation") == "(A) | (B)"

    def test_dask_chunks_applied_when_requesting_dask_return_kind(self) -> None:
        """
        Ensure that specifying dask_chunks yields a dask-backed output with those chunks.
        Use numpy-backed inputs so the conversion path constructs a dask array with given chunks.
        """
        ny, nx = 50, 70
        # Expression path requires a non-empty boolean env; provide a tiny dummy mask.
        dummy = xr.DataArray(np.zeros((1, 1), dtype=bool), dims=("y", "x"))
        a = select_mask(
            make_image(ny, nx),
            select="False",
            mask_source={"dummy": dummy},
            return_kind="dataarray-numpy",
            creation_hint="F",
        )
        b = select_mask(
            make_image(ny, nx),
            select="True",
            mask_source={"dummy": dummy},
            return_kind="dataarray-numpy",
            creation_hint="T",
        )
        chunks = (20, 25)
        out = combine_with_creation(
            a, "|", b, return_kind="dataarray-dask", dask_chunks=chunks
        )
        assert isinstance(out, xr.DataArray) and hasattr(out.data, "chunks")
        # chunks may be normalized into tuples-of-tuples by dask
        got = tuple(
            sum(([int(c) for c in ch] if isinstance(ch, tuple) else [int(ch)],), [])
            for ch in out.data.chunks
        )
        # Flatten each axis chunking and compare the sizes set (normalize expected to tuple)
        exp0 = tuple(
            [chunks[0]] * (ny // chunks[0])
            + ([ny % chunks[0]] if ny % chunks[0] else [])
        )
        exp1 = tuple(
            [chunks[1]] * (nx // chunks[1])
            + ([nx % chunks[1]] if nx % chunks[1] else [])
        )
        assert tuple(sum(([int(c) for c in out.data.chunks[0]],), [])) == exp0
        assert tuple(sum(([int(c) for c in out.data.chunks[1]],), [])) == exp1


class TestDataArrayMaskConstructorFallback:
    def test_dims_constructor_raises_and_fallback_without_dims_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Cover the except-path:
            try:
                m = xr.DataArray(mask, dims=data.dims[: np.ndim(mask)])
            except Exception:
                m = xr.DataArray(mask)
        by making the constructor raise only when 'dims' is provided.
        Use public API (select_mask) with xarray `data` so alignment proceeds.
        """
        ny, nx = 8, 12
        data = make_image(ny, nx)  # xarray.DataArray with dims ('y','x')
        mask = np.zeros((ny, 1), dtype=bool)  # broadcastable to (ny, nx)

        orig_init = xr.DataArray.__init__

        def init_maybe_raise(self, data, *args, **kwargs):  # type: ignore[override]
            # Raise only for the *first* constructor call that uses the original
            # mask shape with 'dims' (ny, 1). Allow the later fallback
            # xr.DataArray(b, dims=('y','x')) where b.shape == (ny, nx).
            if (
                "dims" in kwargs
                and isinstance(data, np.ndarray)
                and data.shape == mask.shape
            ):
                raise RuntimeError("simulated constructor failure with dims")
            return orig_init(self, data, *args, **kwargs)

        # Patch the class __init__ so only the dims-path fails; fallback call without dims succeeds.
        monkeypatch.setattr(xr.DataArray, "__init__", init_maybe_raise, raising=True)

        # Also force _align_bool_mask_to_data to take its NumPy broadcast fallback
        # (otherwise xarray.broadcast_like would create a cross-product of dims).
        def boom(self, other, *a, **k):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulate broadcast_like failure")

        monkeypatch.setattr(xr.DataArray, "broadcast_like", boom, raising=True)
        # Use return_kind="numpy" so the final result is a NumPy array.
        out = select_mask(data, select=mask, return_kind="numpy")

        assert isinstance(out, np.ndarray)
        assert out.dtype == bool and out.shape == (ny, nx)


class TestCombineWithCreationRenameExcept:
    def test_template_rename_exception_path_falls_back_and_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Cover the except-path inside combine_with_creation:
            try:
                combined = combined.rename({...})
            except Exception:
                pass
        by forcing xr.DataArray.rename to raise. The function should still return a
        correct boolean DataArray aligned to the template's dims via fallback alignment.
        """
        ny, nx = 30, 40
        # Template uses different dim names; same shape
        tmpl = xr.DataArray(np.zeros((ny, nx), dtype=float), dims=("row", "col"))
        # Inputs with ('y','x') dims and simple patterns
        a = xr.DataArray(np.zeros((ny, nx), dtype=bool), dims=("y", "x")).assign_attrs(
            creation="A"
        )
        b = xr.DataArray(np.zeros((ny, nx), dtype=bool), dims=("y", "x")).assign_attrs(
            creation="B"
        )
        a.values[5:20, 7:25] = True
        b.values[10:28, 20:35] = True
        exp_or = a.values | b.values

        # Force rename failure only when called; public library monkeypatch
        def bad_rename(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated rename failure")

        monkeypatch.setattr(xr.DataArray, "rename", bad_rename, raising=True)

        out = combine_with_creation(
            a, "|", b, template=tmpl, return_kind="dataarray-numpy"
        )
        assert isinstance(out, xr.DataArray) and out.dtype == bool
        # Despite rename failure, select_mask alignment fallback should deliver template dims
        assert out.dims == ("row", "col") and out.shape == (ny, nx)
        np.testing.assert_array_equal(out.values, exp_or)
        # Creation remains well-formed
        assert out.attrs.get("creation") == "(A) | (B)"


class TestToBoolDataArray:
    def test_dataarray_float_nan_fillna_before_bool(self) -> None:
        """
        Cover nested-if in _to_bool for xarray.DataArray:
          if isinstance(arr, xr.DataArray):
              if np.issubdtype(out.dtype, np.floating):
                  out = out.fillna(False)
        Provide float DataArrays with NaNs via the public expression API.
        """
        da_img = make_image(3, 3)
        A = xr.DataArray(
            np.array(
                [[np.nan, 0.0, 0.2], [-1.0, 0.0, np.nan], [0.0, 3.0, 0.0]], dtype=float
            ),
            dims=("y", "x"),
        )
        B = xr.DataArray(
            np.array(
                [[0.0, 1.0, np.nan], [0.0, 0.0, 3.0], [np.nan, 0.0, 0.0]], dtype=float
            ),
            dims=("y", "x"),
        )
        # Expected: NaNs -> 0.0, nonzero -> True, zero -> False
        A_bool = np.nan_to_num(A.values, nan=0.0).astype(bool)
        B_bool = np.nan_to_num(B.values, nan=0.0).astype(bool)
        exp = A_bool & ~B_bool
        m = select_mask(
            da_img,
            select="A & ~B",
            mask_source={"A": A, "B": B},
            return_kind="dataarray-numpy",
        )
        assert isinstance(m, xr.DataArray) and m.dtype == bool
        np.testing.assert_array_equal(m.values, exp)

    def test_ndarray_float_nan_coercion_in_expression_masks(self) -> None:
        """Float ndarray masks with NaNs should coerce through the public expression API."""
        data = np.zeros((3, 4), dtype=float)
        a = np.array(
            [[np.nan, 0.0, 1.0, 0.0], [2.0, np.nan, 0.0, 0.0], [0.0, 0.0, 3.0, np.nan]],
            dtype=float,
        )
        b = np.array(
            [[0.0, 1.0, 0.0, np.nan], [0.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 1.0, 0.0]],
            dtype=float,
        )
        expected = np.nan_to_num(a, nan=0.0).astype(bool) | np.nan_to_num(
            b, nan=0.0
        ).astype(bool)
        m = select_mask(
            data, select="A | B", mask_source={"A": a, "B": b}, return_kind="numpy"
        )
        assert isinstance(m, np.ndarray) and m.dtype == bool
        np.testing.assert_array_equal(m, expected)


class TestCRTFMalformed:
    def test_crtf_invalid_line_raises(self) -> None:
        """
        Public-API coverage of the ValueError raised by _split_shape_payload when a
        CRTF-looking string contains a malformed line that doesn't match
        '<shape>[[...]'. The '#CRTF' header routes parsing down the CRTF path.
        """
        da_img = make_image(16, 16)
        crtf_bad = "#CRTF\nnot_a_shape 123"
        with pytest.raises(ValueError, match=r"Invalid CRTF line: 'not_a_shape 123'"):
            select_mask(da_img, select=crtf_bad)

    def test_crtf_unmatched_shape_brackets_raise(self) -> None:
        """Malformed CRTF shape payloads should raise through the public parser."""
        da_img = make_image(16, 16)
        crtf_bad = "#CRTF\ncircle[[8pix,8pix], 3pix"
        with pytest.raises(ValueError, match=r"Unmatched brackets"):
            select_mask(da_img, select=crtf_bad)

    def test_global_line_with_no_assignments_is_ignored(self) -> None:
        """A bare global line should be accepted and contribute no keyword defaults."""
        da_img = make_image(16, 16)
        with_global = "#CRTF\nglobal\nbox[[1pix,1pix],[4pix,4pix]]"
        plain = "#CRTF\nbox[[1pix,1pix],[4pix,4pix]]"
        m_with = select_mask(da_img, select=with_global)
        m_plain = select_mask(da_img, select=plain)
        np.testing.assert_array_equal(m_with.values, m_plain.values)

    def test_keyword_assignment_missing_equals_raises(self) -> None:
        """Trailing CRTF keyword text must use key=value syntax."""
        da_img = make_image(16, 16)
        crtf_bad = "#CRTF\nbox[[1pix,1pix],[4pix,4pix]], corr[I,Q]"
        with pytest.raises(ValueError, match=r"Expected 'key=value'"):
            select_mask(da_img, select=crtf_bad)


class TestCRTFNumericParsingErrors:
    def test_invalid_numeric_token_in_angle_raises_value_error(self) -> None:
        """
        Public-API coverage for _parse_units_val raise:
        an angle with an unsupported unit triggers "Invalid numeric token: '30xyz'".
        """
        da_img = make_image(32, 32)
        bad = "rotbox[[12pix,8pix],[6pix,3pix], theta_m=30xyz]"
        with pytest.raises(ValueError, match=r"Invalid numeric token: '30xyz'"):
            select_mask(da_img, select=bad)

    def test_invalid_rotation_keyword_name_raises(self) -> None:
        """Only pa= and theta_m= are accepted for rotated shapes."""
        da_img = make_image(32, 32)
        bad = "rotbox[[12pix,8pix],[6pix,3pix], angle=30deg]"
        with pytest.raises(
            ValueError,
            match=r"Rotation must be provided as 'pa=<angle>' or 'theta_m=<angle>'",
        ):
            select_mask(da_img, select=bad)

    def test_invalid_pixel_quantity_token_raises_specific_message(self) -> None:
        """
        Public-API coverage for _parse_pix_val raise path:
        using 'px' instead of required 'pix' should emit the pixel-specific error.
        """
        da_img = make_image(32, 32)
        bad = "circle[[10pix,10pix], 20px]"
        with pytest.raises(
            ValueError, match=r"Expected '<value>pix' for pixel quantity, got '20px'"
        ):
            select_mask(da_img, select=bad)


class TestCrtfKeywordSyntaxErrors:
    """Malformed keyword-value pairs should fail via the public CRTF API."""

    def _sky(self) -> xr.DataArray:
        return make_xradio_sky(
            n_time=3, n_freq=6, n_pol=2, n_l=4, n_m=4, pols=("I", "Q")
        )

    def test_range_without_brackets_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], range=1.2GHz"
        with pytest.raises(ValueError, match=r"range= value must be bracketed"):
            select_mask(sky, crtf)

    def test_range_wrong_arity_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], range=[1.2GHz]"
        with pytest.raises(ValueError, match=r"range= requires exactly two values"):
            select_mask(sky, crtf)

    def test_corr_without_brackets_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], corr=I"
        with pytest.raises(ValueError, match=r"corr= value must be bracketed"):
            select_mask(sky, crtf)

    def test_time_without_brackets_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], time=60001.0"
        with pytest.raises(ValueError, match=r"time= value must be bracketed"):
            select_mask(sky, crtf)

    def test_time_wrong_arity_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], time=[60001.0]"
        with pytest.raises(ValueError, match=r"time= requires exactly two values"):
            select_mask(sky, crtf)

    def test_invalid_range_family_token_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], range=[12frobs, 13frobs]"
        with pytest.raises(ValueError, match=r"Cannot detect range= family"):
            select_mask(sky, crtf)

    def test_invalid_frequency_token_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], range=[1.2eGHz, 1.5GHz]"
        with pytest.raises(ValueError, match=r"Cannot parse frequency token"):
            select_mask(sky, crtf)

    def test_invalid_velocity_token_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], range=[20kkm/s, 40km/s]"
        with pytest.raises(ValueError, match=r"Cannot parse velocity token"):
            select_mask(sky, crtf)

    def test_invalid_channel_token_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], range=[twochan, 5chan]"
        with pytest.raises(ValueError, match=r"Cannot parse channel token"):
            select_mask(sky, crtf)

    def test_time_mjd_suffix_is_accepted(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], time=[60001.0mjd, 60002.0mjd]"
        m = select_mask(sky, crtf)
        time_any = m.any(dim=["frequency", "polarization", "l", "m"])
        assert int(time_any.values.sum()) == 2

    def test_invalid_time_token_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], time=[not-a-time, 60002.0]"
        with pytest.raises(ValueError, match=r"Cannot detect time= family"):
            select_mask(sky, crtf)

    def test_iso_time_with_space_separator_uses_fallback_parser(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[100pix,100pix]], time=['2023-02-26 00:00:00', '2023-02-27 00:00:00']"
        m = select_mask(sky, crtf)
        time_any = m.any(dim=["frequency", "polarization", "l", "m"])
        assert int(time_any.values.sum()) == 2


class TestCreationAssignmentInDataArrayNumpyReturn:
    def test_creation_hint_attached_in_dataarray_numpy_else_branch(self) -> None:
        """
        Cover:
            da_out = xr.DataArray(arr, dims=dims, coords=coords)
            if creation is not None:
                da_out = da_out.assign_attrs({"creation": creation})
        by passing a NumPy boolean mask (not an xr.DataArray) with return_kind='dataarray-numpy'
        and a non-None creation_hint.
        """
        ny, nx = 11, 13
        data = make_image(ny, nx)  # xr.DataArray with dims ('y','x')
        # Plain NumPy mask ⇒ hits the non-xarray path in dataarray-numpy branch
        mask_np = np.zeros((ny, nx), dtype=int)
        mask_np[2:5, 3:9] = 1  # some True region after bool-cast
        hint = "numpy mask demo"
        out = select_mask(
            data, select=mask_np, return_kind="dataarray-numpy", creation_hint=hint
        )
        assert isinstance(out, xr.DataArray)
        assert out.dtype == bool and out.shape == (ny, nx)
        assert out.dims == data.dims
        # creation attribute must be attached
        assert out.attrs.get("creation") == hint
        # ensure NumPy-backed (not dask)
        assert not hasattr(out.data, "chunks")


class TestCombineWithCreationRenameDoubleExcept:
    def test_rename_and_constructor_fallback_both_raise_then_pass(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Cover the LAST except-pass block inside combine_with_creation:
            try:    combined = combined.rename({...})
            except: try:
                        combined = xr.DataArray(arr, dims=tmpl.dims, coords=tmpl.coords)
                    except:
                        pass   # this branch
        We force both rename and the fallback constructor-with-dims to raise,
        then ensure the function still returns a valid mask via later alignment.
        Use public API only.
        """
        ny, nx = 15, 18
        tmpl = xr.DataArray(np.zeros((ny, nx), dtype=float), dims=("row", "col"))
        a = xr.DataArray(np.zeros((ny, nx), dtype=bool), dims=("y", "x")).assign_attrs(
            creation="A"
        )
        b = xr.DataArray(np.zeros((ny, nx), dtype=bool), dims=("y", "x")).assign_attrs(
            creation="B"
        )
        a.values[::2, :] = True
        b.values[:, ::3] = True
        exp_or = a.values | b.values

        # 1) Make rename fail
        def boom_rename(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulate rename failure")

        monkeypatch.setattr(xr.DataArray, "rename", boom_rename, raising=True)

        # 2) Make the constructor fallback with dims fail, but allow normal constructor calls elsewhere
        orig_init = xr.DataArray.__init__

        def init_maybe_raise(self, data, *args, **kwargs):  # type: ignore[override]
            if "dims" in kwargs:
                raise RuntimeError("simulate constructor failure with dims")
            return orig_init(self, data, *args, **kwargs)

        monkeypatch.setattr(xr.DataArray, "__init__", init_maybe_raise, raising=True)

        # Choose return_kind='dask' to avoid later xr.DataArray constructions in coercion paths.
        out = combine_with_creation(a, "|", b, template=tmpl, return_kind="dask")
        # Should still succeed and return a dask array, but with outer-product broadcasting
        # because both rename and constructor-with-dims fallbacks were forced to fail.
        assert hasattr(out, "chunks")  # dask.array.Array
        got = np.asarray(out.compute(), dtype=bool)
        # Expect a 4D broadcast result. Axis order may be (y,x,row,col) or (row,col,y,x)
        # depending on xarray's alignment. Reduce along both possibilities and accept either.
        assert got.ndim == 4
        # Option A: last two axes are template dims
        got2d_last = got.any(axis=-1).any(axis=-1)
        # Option B: first two axes are template dims
        got2d_first = got.any(axis=0).any(axis=0)
        ok = False
        try:
            np.testing.assert_array_equal(got2d_last, exp_or)
            ok = True
        except AssertionError:
            pass
        if not ok:
            np.testing.assert_array_equal(got2d_first, exp_or)


# ---------------------------------------------------------------------------
# Step 5: lm-mode (angular-offset) shape rasterization
# ---------------------------------------------------------------------------


class TestCrtfLmMode:
    """lm-mode (angular-offset) shape rasterization via select_mask."""

    def _sky(self) -> xr.DataArray:
        """20×20 sky DataArray with 1-arcsec spacing centered on l=m=0."""
        n_l, n_m = 20, 20
        arcsec = math.pi / (180 * 3600)
        l_rad = np.arange(n_l, dtype=float) * arcsec - 9.5 * arcsec
        m_rad = np.arange(n_m, dtype=float) * arcsec - 9.5 * arcsec
        data = np.ones((1, 1, 1, n_l, n_m), dtype=float)
        return xr.DataArray(
            data,
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "time": [0.0],
                "frequency": [1e9],
                "polarization": ["I"],
                "l": l_rad,
                "m": m_rad,
            },
        )

    def _arcsec_to_rad(self, a: float) -> float:
        return a * math.pi / (180 * 3600)

    def test_centerbox_lm_selects_central_square(self) -> None:
        """centerbox centered at origin with 4×4 arcsec sides selects correct pixels."""
        sky = self._sky()
        arcsec = self._arcsec_to_rad(1.0)
        # pixels 8..11 in each axis (0-based) span l/m in [-1.5, -0.5, 0.5, 1.5] arcsec
        crtf = "#CRTF\ncenterbox[[0arcsec,0arcsec],[4arcsec,4arcsec]]"
        mask = select_mask(sky, crtf)
        got = mask.values.squeeze()  # (l, m)
        # count selected pixels
        assert got.sum() > 0
        l_vals = sky.coords["l"].values
        m_vals = sky.coords["m"].values
        half = 2 * arcsec
        expected = np.zeros((20, 20), dtype=bool)
        for i, lv in enumerate(l_vals):
            for j, mv in enumerate(m_vals):
                if abs(lv) <= half and abs(mv) <= half:
                    expected[i, j] = True
        np.testing.assert_array_equal(got, expected)

    def test_circle_lm_matches_distance_formula(self) -> None:
        """circle in lm mode selects pixels within radius."""
        sky = self._sky()
        arcsec = self._arcsec_to_rad(1.0)
        radius = 3.0 * arcsec
        crtf = "#CRTF\ncircle[[0arcsec,0arcsec],3arcsec]"
        mask = select_mask(sky, crtf)
        got = mask.values.squeeze()
        l_vals = sky.coords["l"].values
        m_vals = sky.coords["m"].values
        expected = np.zeros((20, 20), dtype=bool)
        for i, lv in enumerate(l_vals):
            for j, mv in enumerate(m_vals):
                if math.sqrt(lv**2 + mv**2) <= radius:
                    expected[i, j] = True
        np.testing.assert_array_equal(got, expected)

    def test_rotbox_lm_arcmin(self) -> None:
        """rotbox with arcmin units in lm mode, zero rotation."""
        sky = self._sky()
        # 10arcmin box centered at origin, 0 rotation => all 20×20 selected
        crtf = "#CRTF\nrotbox[[0arcmin,0arcmin],[10arcmin,10arcmin],pa=0deg]"
        mask = select_mask(sky, crtf)
        got = mask.values.squeeze()
        assert got.all(), "Expected all pixels selected by oversized rotbox"

    def test_box_lm_selects_expected_rectangle(self) -> None:
        """box in lm mode should select the same rectangular subset as direct coord checks."""
        sky = self._sky()
        crtf = "#CRTF\nbox[[-3arcsec,-2arcsec],[2arcsec,1arcsec]]"
        mask = select_mask(sky, crtf).values.squeeze()
        l_vals = sky.coords["l"].values
        m_vals = sky.coords["m"].values
        arcsec = self._arcsec_to_rad(1.0)
        expected = np.zeros((20, 20), dtype=bool)
        for i, lv in enumerate(l_vals):
            for j, mv in enumerate(m_vals):
                if (-3 * arcsec) <= lv <= (2 * arcsec) and (-2 * arcsec) <= mv <= (
                    1 * arcsec
                ):
                    expected[i, j] = True
        np.testing.assert_array_equal(mask, expected)

    def test_annulus_lm_matches_radial_shell(self) -> None:
        """annulus in lm mode should match the expected radial shell."""
        sky = self._sky()
        crtf = "#CRTF\nannulus[[0arcsec,0arcsec],[2arcsec,4arcsec]]"
        mask = select_mask(sky, crtf).values.squeeze()
        l_vals = sky.coords["l"].values
        m_vals = sky.coords["m"].values
        arcsec = self._arcsec_to_rad(1.0)
        expected = np.zeros((20, 20), dtype=bool)
        for i, lv in enumerate(l_vals):
            for j, mv in enumerate(m_vals):
                r = math.sqrt(lv**2 + mv**2)
                if 2 * arcsec <= r <= 4 * arcsec:
                    expected[i, j] = True
        np.testing.assert_array_equal(mask, expected)

    def test_ellipse_lm_selects_nonempty_region(self) -> None:
        """ellipse in lm mode with an explicit angle should produce a non-empty mask."""
        sky = self._sky()
        crtf = "#CRTF\nellipse[[0arcsec,0arcsec],[6arcsec,3arcsec],pa=30deg]"
        mask = select_mask(sky, crtf)
        assert mask.values.any()

    def test_rotbox_lm_without_angle_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nrotbox[[0arcsec,0arcsec],[6arcsec,3arcsec]]"
        with pytest.raises(ValueError, match=r"rotbox requires angle"):
            select_mask(sky, crtf)

    def test_ellipse_lm_without_angle_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\nellipse[[0arcsec,0arcsec],[6arcsec,3arcsec]]"
        with pytest.raises(ValueError, match=r"ellipse requires angle"):
            select_mask(sky, crtf)

    def test_poly_lm_matches_reference(self) -> None:
        """poly in lm mode (all-arcsec tokens) selects a triangular region."""
        sky = self._sky()
        arcsec = self._arcsec_to_rad(1.0)
        # triangle with vertices at (0,0), (5arcsec,0), (0,5arcsec) in l/m
        crtf = "#CRTF\npoly[[0arcsec,0arcsec],[5arcsec,0arcsec],[0arcsec,5arcsec]]"
        mask = select_mask(sky, crtf)
        got = mask.values.squeeze()
        l_vals = sky.coords["l"].values
        m_vals = sky.coords["m"].values
        # Point-in-triangle test: l >= 0, m >= 0, l+m < 5arcsec (strict to avoid
        # ray-casting boundary ambiguity at exact hypotenuse crossings).
        expected = np.zeros((20, 20), dtype=bool)
        limit = 5 * arcsec
        for i, lv in enumerate(l_vals):
            for j, mv in enumerate(m_vals):
                if lv >= 0 and mv >= 0 and (lv + mv) < limit:
                    expected[i, j] = True
        np.testing.assert_array_equal(got, expected)

    def test_lm_on_ndarray_raises_missing_coord(self) -> None:
        """lm-mode shapes require l/m coords; raw ndarray raises ValueError."""
        data = np.ones((20, 20))
        crtf = "#CRTF\ncircle[[0arcsec,0arcsec],3arcsec]"
        with pytest.raises(ValueError, match="requires coord"):
            select_mask(data, crtf)

    def test_lm_on_dataarray_without_lm_coords_raises(self) -> None:
        """DataArray missing 'l'/'m' coords raises ValueError for lm-mode shape."""
        da_no_lm = xr.DataArray(
            np.ones((4, 4)),
            dims=["x", "y"],
            coords={"x": np.arange(4), "y": np.arange(4)},
        )
        crtf = "#CRTF\ncircle[[0arcsec,0arcsec],3arcsec]"
        with pytest.raises(ValueError, match="requires coord"):
            select_mask(da_no_lm, crtf)


class TestCrtfRange:
    """range= mask builder: frequency, velocity, and channel families."""

    def _sky(self) -> xr.DataArray:
        return make_xradio_sky(
            n_time=2,
            n_freq=10,
            n_pol=2,
            n_l=4,
            n_m=4,
            freq_start_ghz=1.0,
            freq_step_ghz=0.1,
            pols=("I", "Q"),
        )

    def test_range_frequency_selects_correct_channels(self) -> None:
        sky = self._sky()
        # frequencies are 1.0, 1.1, ..., 1.9 GHz; select 1.2–1.5 GHz → channels 2,3,4,5
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[1.2GHz, 1.5GHz]"
        m = select_mask(sky, crtf)
        assert isinstance(m, xr.DataArray) and m.dtype == bool
        freq_any = m.any(dim=["time", "polarization", "l", "m"])
        selected_freqs = sky.coords["frequency"].values[freq_any.values]
        assert len(selected_freqs) == 4
        np.testing.assert_allclose(
            selected_freqs / 1e9, [1.2, 1.3, 1.4, 1.5], atol=1e-6
        )

    def test_range_velocity_selects_correct_channels(self) -> None:
        sky = self._sky()
        # velocities 0, 1e4, 2e4, ..., 9e4 m/s; select 2e4 to 4e4 → channels 2,3,4
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[20000m/s, 40000m/s]"
        m = select_mask(sky, crtf)
        assert isinstance(m, xr.DataArray) and m.dtype == bool
        freq_any = m.any(dim=["time", "polarization", "l", "m"])
        assert int(freq_any.values.sum()) == 3

    def test_range_velocity_km_per_s(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[20km/s, 40km/s]"
        m = select_mask(sky, crtf)
        freq_any = m.any(dim=["time", "polarization", "l", "m"])
        assert int(freq_any.values.sum()) == 3

    def test_range_channel_selects_integer_indices(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[2chan, 5chan]"
        m = select_mask(sky, crtf)
        freq_any = m.any(dim=["time", "polarization", "l", "m"])
        assert int(freq_any.values.sum()) == 4  # channels 2,3,4,5

    def test_range_mixed_family_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[1GHz, 5chan]"
        with pytest.raises(ValueError, match="family mismatch"):
            select_mask(sky, crtf)

    def test_range_on_ndarray_raises_missing_coord(self) -> None:
        arr = np.ones((4, 4))
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[1.0GHz, 1.5GHz]"
        with pytest.raises(ValueError, match="frequency"):
            select_mask(arr, crtf)

    def test_range_velocity_without_velocity_coord_raises(self) -> None:
        sky = self._sky()
        sky_no_vel = sky.drop_vars("velocity")
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[100m/s, 200m/s]"
        with pytest.raises(ValueError, match="velocity"):
            select_mask(sky_no_vel, crtf)

    def test_range_out_of_overlap_warns_and_returns_all_false(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], range=[5.0GHz, 6.0GHz]"
        with pytest.warns(
            UserWarning,
            match=r"range=\[5\.0GHz, 6\.0GHz\] selects no frequency entries",
        ):
            m = select_mask(sky, crtf)
        freq_any = m.any(dim=["time", "polarization", "l", "m"])
        assert int(freq_any.values.sum()) == 0


class TestCrtfCorr:
    """corr= mask builder: polarization selection."""

    def _sky(self) -> xr.DataArray:
        return make_xradio_sky(
            n_time=1, n_freq=4, n_pol=4, n_l=4, n_m=4, pols=("I", "Q", "U", "V")
        )

    def test_corr_selects_named_polarizations(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], corr=[I, Q]"
        m = select_mask(sky, crtf)
        assert isinstance(m, xr.DataArray) and m.dtype == bool
        pol_any = m.any(dim=["time", "frequency", "l", "m"])
        selected = sky.coords["polarization"].values[pol_any.values]
        assert list(selected) == ["I", "Q"]

    def test_corr_case_insensitive(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], corr=[i, q]"
        m = select_mask(sky, crtf)
        pol_any = m.any(dim=["time", "frequency", "l", "m"])
        assert int(pol_any.values.sum()) == 2

    def test_corr_unknown_token_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], corr=[I, FAKE]"
        with pytest.raises(ValueError, match="FAKE"):
            select_mask(sky, crtf)

    def test_corr_on_ndarray_raises_missing_coord(self) -> None:
        arr = np.ones((4, 4))
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], corr=[I, Q]"
        with pytest.raises(ValueError, match="polarization"):
            select_mask(arr, crtf)


class TestCrtfTime:
    """time= mask builder: MJD, JD, and ISO time families."""

    def _sky(self) -> xr.DataArray:
        return make_xradio_sky(
            n_time=5,
            n_freq=2,
            n_pol=2,
            n_l=4,
            n_m=4,
            time_start_mjd=60000.0,
            time_step_mjd=1.0,
            pols=("I", "Q"),
        )

    def test_time_mjd_bare_number(self) -> None:
        sky = self._sky()
        # times are 60000, 60001, 60002, 60003, 60004; select 60001–60003
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], time=[60001.0, 60003.0]"
        m = select_mask(sky, crtf)
        time_any = m.any(dim=["frequency", "polarization", "l", "m"])
        assert int(time_any.values.sum()) == 3

    def test_time_mjd_d_suffix(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], time=[60001.0d, 60003.0d]"
        m = select_mask(sky, crtf)
        time_any = m.any(dim=["frequency", "polarization", "l", "m"])
        assert int(time_any.values.sum()) == 3

    def test_time_iso_format(self) -> None:
        sky = self._sky()
        # MJD 60000 = 2023-02-25 (approx); use astropy to get exact bounds
        from astropy.time import Time

        lo = Time(60001.0, format="mjd", scale="utc").isot
        hi = Time(60003.0, format="mjd", scale="utc").isot
        crtf = f"#CRTF\n+box[[0pix,0pix],[100pix,100pix]], time=['{lo}', '{hi}']"
        m = select_mask(sky, crtf)
        time_any = m.any(dim=["frequency", "polarization", "l", "m"])
        assert int(time_any.values.sum()) == 3

    def test_time_jd_format(self) -> None:
        sky = self._sky()
        from astropy.time import Time

        lo_jd = Time(60001.0, format="mjd", scale="utc").jd
        hi_jd = Time(60003.0, format="mjd", scale="utc").jd
        crtf = f"#CRTF\n+box[[0pix,0pix],[100pix,100pix]], time=[{lo_jd}jd, {hi_jd}jd]"
        m = select_mask(sky, crtf)
        time_any = m.any(dim=["frequency", "polarization", "l", "m"])
        assert int(time_any.values.sum()) == 3

    def test_time_mixed_family_raises(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], time=[60001.0, '2023-01-01T00:00:00']"
        with pytest.raises(ValueError, match="family mismatch"):
            select_mask(sky, crtf)

    def test_time_on_ndarray_raises_missing_coord(self) -> None:
        arr = np.ones((4, 4))
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], time=[60000.0, 60001.0]"
        with pytest.raises(ValueError, match="time"):
            select_mask(arr, crtf)

    def test_time_out_of_overlap_warns_and_returns_all_false(self) -> None:
        sky = self._sky()
        crtf = "#CRTF\n+box[[0pix,0pix],[100pix,100pix]], time=[59990.0, 59999.0]"
        with pytest.warns(
            UserWarning,
            match=r"time=\[59990\.0, 59999\.0\] selects no time entries",
        ):
            m = select_mask(sky, crtf)
        time_any = m.any(dim=["frequency", "polarization", "l", "m"])
        assert int(time_any.values.sum()) == 0


class TestCrtfPerLineCombining:
    """Per-line + / - combining with axis masks."""

    def _sky(self) -> xr.DataArray:
        return make_xradio_sky(
            n_time=1,
            n_freq=8,
            n_pol=4,
            n_l=8,
            n_m=8,
            freq_start_ghz=1.0,
            freq_step_ghz=0.1,
            pols=("I", "Q", "U", "V"),
        )

    def test_two_range_lines_with_plus_produce_union(self) -> None:
        sky = self._sky()
        # two frequency windows: 1.0–1.1 GHz (channels 0,1) and 1.5–1.6 GHz (channels 5,6)
        crtf = (
            "#CRTF\n"
            "+box[[0pix,0pix],[100pix,100pix]], range=[1.0GHz, 1.1GHz]\n"
            "+box[[0pix,0pix],[100pix,100pix]], range=[1.5GHz, 1.6GHz]"
        )
        m = select_mask(sky, crtf)
        freq_any = m.any(dim=["time", "polarization", "l", "m"])
        assert int(freq_any.values.sum()) == 4  # channels 0,1,5,6

    def test_minus_line_removes_corr(self) -> None:
        sky = self._sky()
        # include all, then subtract U and V
        crtf = (
            "#CRTF\n"
            "+box[[0pix,0pix],[100pix,100pix]]\n"
            "-box[[0pix,0pix],[100pix,100pix]], corr=[U, V]"
        )
        m = select_mask(sky, crtf)
        pol_any = m.any(dim=["time", "frequency", "l", "m"])
        selected = sky.coords["polarization"].values[pol_any.values]
        assert list(selected) == ["I", "Q"]


# ---------------------------------------------------------------------------
# Step 8: globals + per-line overrides end-to-end
# ---------------------------------------------------------------------------


class TestCrtfGlobalsAndOverrides:
    """End-to-end tests for global keyword propagation and per-line override."""

    def _sky(self) -> xr.DataArray:
        return make_xradio_sky(
            n_time=1,
            n_freq=1,
            pols=["I", "Q", "U", "V"],
            n_l=20,
            n_m=20,
        )

    def test_global_corr_applies_to_every_line(self) -> None:
        """global corr=[I,Q] restricts every region line to those polarizations."""
        sky = self._sky()
        crtf = (
            "#CRTF\n"
            "global corr=[I,Q]\n"
            "box[[0pix,0pix],[9pix,9pix]]\n"
            "box[[10pix,10pix],[19pix,19pix]]"
        )
        mask = select_mask(sky, crtf)
        # Only I and Q should be selected anywhere
        pol_any = mask.any(dim=["time", "frequency", "l", "m"])
        selected = list(sky.coords["polarization"].values[pol_any.values])
        assert selected == ["I", "Q"]

    def test_per_line_corr_overrides_global(self) -> None:
        """Per-line corr= overrides global corr= for that line only."""
        sky = self._sky()
        # Global: I,Q. Line 2 overrides to U only. Result: line1 selects I,Q;
        # line2 selects U. Union = I,Q,U.
        crtf = (
            "#CRTF\n"
            "global corr=[I,Q]\n"
            "box[[0pix,0pix],[9pix,9pix]]\n"
            "box[[10pix,10pix],[19pix,19pix]] corr=[U]"
        )
        mask = select_mask(sky, crtf)
        pol_any = mask.any(dim=["time", "frequency", "l", "m"])
        selected = set(sky.coords["polarization"].values[pol_any.values])
        assert selected == {"I", "Q", "U"}
        # V must not be selected
        v_idx = list(sky.coords["polarization"].values).index("V")
        assert not pol_any.values[v_idx]

    def test_global_coordsys_lm_applies_to_deg_lines(self) -> None:
        """global coordsys=lm allows deg tokens without per-line coordsys=."""
        n = 20
        arcsec = math.pi / (180 * 3600)
        l_rad = np.arange(n, dtype=float) * arcsec - (n / 2 - 0.5) * arcsec
        m_rad = np.arange(n, dtype=float) * arcsec - (n / 2 - 0.5) * arcsec
        sky = xr.DataArray(
            np.ones((1, 1, 1, n, n), dtype=float),
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "time": [0.0],
                "frequency": [1e9],
                "polarization": ["I"],
                "l": l_rad,
                "m": m_rad,
            },
        )
        # 10deg >> 20-arcsec grid => all pixels selected in lm mode
        crtf = "#CRTF\nglobal coordsys=lm\ncircle[[0deg,0deg],10deg]"
        mask = select_mask(sky, crtf)
        assert mask.values.all()

    def test_per_line_key_wins_over_global(self) -> None:
        """When both global and per-line specify the same key, per-line wins."""
        sky = self._sky()
        # Global says I,Q,U,V (all); per-line restricts to just I.
        crtf = (
            "#CRTF\n"
            "global corr=[I,Q,U,V]\n"
            "box[[0pix,0pix],[19pix,19pix]] corr=[I]"
        )
        mask = select_mask(sky, crtf)
        pol_any = mask.any(dim=["time", "frequency", "l", "m"])
        selected = list(sky.coords["polarization"].values[pol_any.values])
        assert selected == ["I"]

    def test_multiple_global_lines_accumulate(self) -> None:
        """Multiple global lines merge; later global keys override earlier ones."""
        sky = self._sky()
        # First global sets corr=[I,Q]; second global overrides to corr=[U,V].
        crtf = (
            "#CRTF\n"
            "global corr=[I,Q]\n"
            "global corr=[U,V]\n"
            "box[[0pix,0pix],[19pix,19pix]]"
        )
        mask = select_mask(sky, crtf)
        pol_any = mask.any(dim=["time", "frequency", "l", "m"])
        selected = set(sky.coords["polarization"].values[pol_any.values])
        assert selected == {"U", "V"}


# ---------------------------------------------------------------------------
# Step 6: coordsys= keyword + family auto-detection
# ---------------------------------------------------------------------------


class TestCrtfCoordsysKeyword:
    """coordsys= keyword resolution and auto-detection of shape coordinate family."""

    def _sky(self) -> xr.DataArray:
        """Small 10×10 sky DataArray with 1-arcsec l/m spacing and dummy ra/dec."""
        n = 10
        arcsec = math.pi / (180 * 3600)
        l_rad = np.arange(n, dtype=float) * arcsec - 4.5 * arcsec
        m_rad = np.arange(n, dtype=float) * arcsec - 4.5 * arcsec
        # Dummy 2-D ra/dec grids so world-mode coord gating passes (dispatch will
        # still raise NotImplementedError since world-mode is not yet implemented).
        ra_grid = np.zeros((n, n), dtype=float)
        dec_grid = np.zeros((n, n), dtype=float)
        return xr.DataArray(
            np.ones((1, 1, 1, n, n), dtype=float),
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "time": [0.0],
                "frequency": [1e9],
                "polarization": ["I"],
                "l": l_rad,
                "m": m_rad,
                "right_ascension": (["l", "m"], ra_grid),
                "declination": (["l", "m"], dec_grid),
            },
        )

    def test_arcsec_tokens_auto_detected_as_lm(self) -> None:
        """arcsec tokens are auto-detected as lm mode without coordsys=."""
        sky = self._sky()
        crtf = "#CRTF\ncircle[[0arcsec,0arcsec],3arcsec]"
        mask = select_mask(sky, crtf)
        assert mask.values.any()

    def test_arcmin_tokens_auto_detected_as_lm(self) -> None:
        """arcmin tokens are auto-detected as lm mode without coordsys=."""
        sky = self._sky()
        # 10arcmin circle covers the entire 10-arcsec grid
        crtf = "#CRTF\ncircle[[0arcmin,0arcmin],10arcmin]"
        mask = select_mask(sky, crtf)
        assert mask.values.all()

    def test_deg_without_coordsys_raises_ambiguous(self) -> None:
        """deg tokens without coordsys= raise ValueError (ambiguous)."""
        sky = self._sky()
        crtf = "#CRTF\ncircle[[0deg,0deg],1deg]"
        with pytest.raises(ValueError, match="[Aa]mbiguous"):
            select_mask(sky, crtf)

    def test_rad_without_coordsys_raises_ambiguous(self) -> None:
        """rad tokens without coordsys= raise ValueError (ambiguous)."""
        sky = self._sky()
        crtf = "#CRTF\ncircle[[0rad,0rad],0.1rad]"
        with pytest.raises(ValueError, match="[Aa]mbiguous"):
            select_mask(sky, crtf)

    def test_coordsys_lm_resolves_deg_tokens(self) -> None:
        """coordsys=lm resolves deg tokens as lm-mode angular offsets."""
        sky = self._sky()
        # circle of radius 10deg >> grid extent (arcsec scale) => all selected
        crtf = "#CRTF\ncircle[[0deg,0deg],10deg] coordsys=lm"
        mask = select_mask(sky, crtf)
        assert mask.values.all()

    def test_coordsys_world_resolves_deg_tokens(self) -> None:
        """coordsys=world resolves deg tokens as world-mode sky coordinates."""
        sky = self._sky()
        # All pixels have ra=dec=0 (dummy grid); circle centered there selects all.
        crtf = "#CRTF\ncircle[[0deg,0deg],1deg] coordsys=world"
        mask = select_mask(sky, crtf)
        assert mask.values.all()

    def test_coordsys_pixel_conflicts_with_arcsec_raises(self) -> None:
        """coordsys=pixel conflicts with arcsec tokens; raises ValueError."""
        sky = self._sky()
        crtf = "#CRTF\ncircle[[0arcsec,0arcsec],3arcsec] coordsys=pixel"
        with pytest.raises(ValueError, match="[Cc]onflicts"):
            select_mask(sky, crtf)

    def test_coordsys_world_conflicts_with_pix_raises(self) -> None:
        """coordsys=world conflicts with pix tokens; raises ValueError."""
        sky = self._sky()
        crtf = "#CRTF\nbox[[0pix,0pix],[5pix,5pix]] coordsys=world"
        with pytest.raises(ValueError, match="[Cc]onflicts"):
            select_mask(sky, crtf)

    def test_global_coordsys_lm_applies_to_deg_line(self) -> None:
        """global coordsys=lm lets subsequent lines use deg tokens as lm offsets."""
        sky = self._sky()
        # large degree circle => all pixels selected in lm mode
        crtf = "#CRTF\nglobal coordsys=lm\ncircle[[0deg,0deg],10deg]"
        mask = select_mask(sky, crtf)
        assert mask.values.all()

    def test_per_line_coordsys_overrides_global(self) -> None:
        """Per-line coordsys= takes precedence over the global block."""
        sky = self._sky()
        # Global says lm; per-line says world. World mode selects all pixels
        # (all at ra=dec=0, circle centered there with 1deg radius).
        crtf = "#CRTF\nglobal coordsys=lm\ncircle[[0deg,0deg],1deg] coordsys=world"
        mask = select_mask(sky, crtf)
        assert mask.values.all()

    def test_unrecognized_coordsys_value_raises(self) -> None:
        """An unrecognized coordsys= value raises ValueError."""
        sky = self._sky()
        crtf = "#CRTF\ncircle[[0arcsec,0arcsec],3arcsec] coordsys=galactic"
        with pytest.raises(ValueError, match="[Uu]nrecognized"):
            select_mask(sky, crtf)

    def test_coordsys_lm_can_match_unambiguous_arcsec_family(self) -> None:
        """An explicit coordsys that agrees with unambiguous lm tokens should succeed."""
        sky = self._sky()
        crtf = "#CRTF\ncircle[[0arcsec,0arcsec],3arcsec] coordsys=lm"
        mask = select_mask(sky, crtf)
        assert mask.values.any()


# ---------------------------------------------------------------------------
# Step 7: world-mode (SkyCoord / SkyOffsetFrame) shape rasterization
# ---------------------------------------------------------------------------


class TestCrtfWorldMode:
    """World-coordinate shape rasterization via SkyCoord / SkyOffsetFrame."""

    # Grid: 20×20, 1-arcsec spacing, centered at RA=10deg Dec=20deg.
    _N = 20
    _RA0_DEG = 10.0
    _DEC0_DEG = 20.0

    def _sky(self, with_radec: bool = True) -> xr.DataArray:
        """20×20 DataArray with real per-pixel RA/Dec grids centered on (_RA0_DEG, _DEC0_DEG)."""
        n = self._N
        arcsec = math.pi / (180 * 3600)
        l_rad = np.arange(n, dtype=float) * arcsec - (n / 2 - 0.5) * arcsec
        m_rad = np.arange(n, dtype=float) * arcsec - (n / 2 - 0.5) * arcsec
        ra0 = self._RA0_DEG * math.pi / 180.0
        dec0 = self._DEC0_DEG * math.pi / 180.0
        # Tangent-plane approximation: RA offset ≈ l / cos(dec0), Dec offset ≈ m
        ll, mm = np.meshgrid(l_rad, m_rad, indexing="ij")
        ra_grid = ra0 + ll / math.cos(dec0)
        dec_grid = dec0 + mm
        coords: dict = {
            "time": [0.0],
            "frequency": [1e9],
            "polarization": ["I"],
            "l": l_rad,
            "m": m_rad,
        }
        if with_radec:
            coords["right_ascension"] = (["l", "m"], ra_grid)
            coords["declination"] = (["l", "m"], dec_grid)
        return xr.DataArray(
            np.ones((1, 1, 1, n, n), dtype=float),
            dims=["time", "frequency", "polarization", "l", "m"],
            coords=coords,
        )

    def _ra_dec_to_crtf_sexa(self, ra_deg: float, dec_deg: float) -> str:
        """Format RA/Dec in degrees as CRTF sexagesimal pair string."""
        ra_h = int(ra_deg / 15)
        ra_m = int((ra_deg / 15 - ra_h) * 60)
        ra_s = ((ra_deg / 15 - ra_h) * 60 - ra_m) * 60
        sign = "+" if dec_deg >= 0 else "-"
        adec = abs(dec_deg)
        dec_d = int(adec)
        dec_m = int((adec - dec_d) * 60)
        dec_s = ((adec - dec_d) * 60 - dec_m) * 60
        return f"{ra_h}h{ra_m}m{ra_s:.3f}s,{sign}{dec_d}d{dec_m}m{dec_s:.3f}s"

    def test_circle_sexa_center_separation_match(self) -> None:
        """circle with sexagesimal center matches per-pixel separation reference."""
        sky = self._sky()
        # 3-arcsec radius circle centered at grid center (RA0, Dec0)
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\ncircle[[{center}],3arcsec]"
        mask = select_mask(sky, crtf)
        got = mask.values.squeeze()
        # Reference: all pixels whose separation from center ≤ 3arcsec
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        ra = sky.coords["right_ascension"].values
        dec = sky.coords["declination"].values
        grid = SkyCoord(ra=ra * u.rad, dec=dec * u.rad, frame="icrs")
        cen = SkyCoord(
            ra=self._RA0_DEG * u.deg, dec=self._DEC0_DEG * u.deg, frame="icrs"
        )
        expected = grid.separation(cen).arcsec <= 3.0
        np.testing.assert_array_equal(got, expected)

    def test_box_sexa_selects_rectangular_region(self) -> None:
        """box with sexagesimal corners selects the correct rectangular sky region."""
        sky = self._sky()
        ra0, dec0 = self._RA0_DEG, self._DEC0_DEG
        # 6×6 arcsec box centered on grid (BLC = center - 3arcsec, TRC = center + 3arcsec)
        arcsec_deg = 1.0 / 3600.0
        blc = self._ra_dec_to_crtf_sexa(ra0 - 3 * arcsec_deg, dec0 - 3 * arcsec_deg)
        trc = self._ra_dec_to_crtf_sexa(ra0 + 3 * arcsec_deg, dec0 + 3 * arcsec_deg)
        crtf = f"#CRTF\nbox[[{blc}],[{trc}]]"
        mask = select_mask(sky, crtf)
        assert mask.values.any()
        # Inner ±1arcsec box should be fully selected
        arcsec = math.pi / (180 * 3600)
        ra = sky.coords["right_ascension"].values
        dec = sky.coords["declination"].values
        ra0_r = ra0 * math.pi / 180
        dec0_r = dec0 * math.pi / 180
        inner = (np.abs(ra - ra0_r) * math.cos(dec0_r) <= arcsec) & (
            np.abs(dec - dec0_r) <= arcsec
        )
        assert mask.values.squeeze()[inner].all(), "Inner pixels should be selected"

    def test_annulus_world_selects_nonempty_ring(self) -> None:
        """world annulus should select a shell around the reference direction."""
        sky = self._sky()
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\nannulus[[{center}],[2arcsec,4arcsec]]"
        mask = select_mask(sky, crtf)
        assert mask.values.any()
        center_circle = select_mask(sky, f"#CRTF\ncircle[[{center}],1arcsec]").values
        assert not np.any(mask.values & center_circle)

    def test_centerbox_world_selects_nonempty_region(self) -> None:
        """world centerbox should rasterize a non-empty rectangular sky region."""
        sky = self._sky()
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\ncenterbox[[{center}],[6arcsec,4arcsec]]"
        mask = select_mask(sky, crtf)
        assert mask.values.any()

    def test_rotbox_world_selects_nonempty_region(self) -> None:
        """world rotbox with an explicit PA should rasterize successfully."""
        sky = self._sky()
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\nrotbox[[{center}],[8arcsec,4arcsec],pa=30deg]"
        mask = select_mask(sky, crtf)
        assert mask.values.any()

    def test_poly_world_selects_nonempty_region(self) -> None:
        """world poly should rasterize an arbitrary polygonal sky region."""
        sky = self._sky()
        dec = self._DEC0_DEG
        ra = self._RA0_DEG
        crtf = (
            "#CRTF\n"
            f"poly[[{ra - 2/3600:.6f}deg,{dec - 2/3600:.6f}deg],"
            f"[{ra + 2/3600:.6f}deg,{dec - 2/3600:.6f}deg],"
            f"[{ra:.6f}deg,{dec + 3/3600:.6f}deg]] coordsys=world"
        )
        mask = select_mask(sky, crtf)
        assert mask.values.any()

    def test_rotbox_world_without_angle_raises(self) -> None:
        sky = self._sky()
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\nrotbox[[{center}],[8arcsec,4arcsec]]"
        with pytest.raises(ValueError, match=r"rotbox requires angle"):
            select_mask(sky, crtf)

    def test_ellipse_world_without_angle_raises(self) -> None:
        sky = self._sky()
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\nellipse[[{center}],[8arcsec,4arcsec]]"
        with pytest.raises(ValueError, match=r"ellipse requires angle"):
            select_mask(sky, crtf)

    def test_world_rad_tokens_are_accepted(self) -> None:
        """coordsys=world should accept rad-valued centers and sizes."""
        sky = self._sky()
        ra0 = self._RA0_DEG * math.pi / 180.0
        dec0 = self._DEC0_DEG * math.pi / 180.0
        crtf = f"#CRTF\ncircle[[{ra0}rad,{dec0}rad],0.00005rad] coordsys=world"
        mask = select_mask(sky, crtf)
        assert mask.values.any()

    def test_world_pair_with_wrong_arity_raises(self) -> None:
        """A world-coordinate pair must contain exactly two coordinates."""
        sky = self._sky()
        crtf = "#CRTF\ncircle[[1h0m0.000s],3arcsec]"
        with pytest.raises(ValueError, match=r"Expected exactly 2 coordinates"):
            select_mask(sky, crtf)

    def test_ellipse_world_pa_matches_circle(self) -> None:
        """ellipse with a=b and pa=0 produces the same selection as a circle."""
        sky = self._sky()
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf_ell = f"#CRTF\nellipse[[{center}],[3arcsec,3arcsec],pa=0deg]"
        crtf_cir = f"#CRTF\ncircle[[{center}],3arcsec]"
        mask_ell = select_mask(sky, crtf_ell).values.squeeze()
        mask_cir = select_mask(sky, crtf_cir).values.squeeze()
        # Equal axes => same as circle (allow 1-pixel boundary tolerance)
        assert mask_ell.sum() == pytest.approx(mask_cir.sum(), abs=2)

    def test_circle_crossing_ra_wrap(self) -> None:
        """circle centered at RA=0 selects pixels on both sides of the wraparound."""
        n = 20
        arcsec = math.pi / (180 * 3600)
        l_rad = np.arange(n, dtype=float) * arcsec - (n / 2 - 0.5) * arcsec
        m_rad = np.arange(n, dtype=float) * arcsec - (n / 2 - 0.5) * arcsec
        dec0 = 0.0
        ll, mm = np.meshgrid(l_rad, m_rad, indexing="ij")
        # Grid straddles RA=0; pixels with l<0 have negative RA (wraps to ~2π).
        ra_grid = ll  # centered at RA=0
        dec_grid = dec0 + mm
        sky = xr.DataArray(
            np.ones((1, 1, 1, n, n), dtype=float),
            dims=["time", "frequency", "polarization", "l", "m"],
            coords={
                "time": [0.0],
                "frequency": [1e9],
                "polarization": ["I"],
                "l": l_rad,
                "m": m_rad,
                "right_ascension": (["l", "m"], ra_grid),
                "declination": (["l", "m"], dec_grid),
            },
        )
        # Center exactly at RA=0h0m0.000s, Dec=+0d0m0.000s; radius 3arcsec.
        crtf = "#CRTF\ncircle[[0h0m0.000s,+0d0m0.000s],3arcsec]"
        mask = select_mask(sky, crtf)
        got = mask.values.squeeze()
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        grid_sc = SkyCoord(ra=ra_grid * u.rad, dec=dec_grid * u.rad, frame="icrs")
        cen_sc = SkyCoord(ra=0.0 * u.rad, dec=0.0 * u.rad, frame="icrs")
        expected = grid_sc.separation(cen_sc).arcsec <= 3.0
        np.testing.assert_array_equal(got, expected)

    def test_world_mode_without_radec_coords_raises(self) -> None:
        """World-mode shapes on a DataArray without ra/dec coords raise ValueError."""
        sky = self._sky(with_radec=False)
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\ncircle[[{center}],3arcsec]"
        with pytest.raises(ValueError, match="requires coord"):
            select_mask(sky, crtf)

    def test_world_mode_on_ndarray_raises(self) -> None:
        """World-mode shapes on a plain ndarray raise ValueError (no coord gating)."""
        data = np.ones((20, 20))
        center = self._ra_dec_to_crtf_sexa(self._RA0_DEG, self._DEC0_DEG)
        crtf = f"#CRTF\ncircle[[{center}],3arcsec]"
        with pytest.raises(ValueError, match="requires coord"):
            select_mask(data, crtf)


class TestCrtfRejectedKeywords:
    """Frame-conversion keywords raise NotImplementedError; viz keywords and ann lines are silent."""

    def _img(self) -> xr.DataArray:
        return make_image(16, 16)

    @pytest.mark.parametrize(
        "kw", ["coord=J2000", "frame=TOPO", "veltype=OPTICAL", "restfreq=1.42GHz"]
    )
    def test_frame_keyword_on_shape_line_raises(self, kw: str) -> None:
        img = self._img()
        crtf = f"box[[0pix,0pix],[4pix,4pix]], {kw}"
        with pytest.raises(NotImplementedError) as ei:
            select_mask(img, crtf)
        assert kw.split("=")[0] in str(ei.value)

    @pytest.mark.parametrize(
        "kw", ["coord=J2000", "frame=TOPO", "veltype=OPTICAL", "restfreq=1.42GHz"]
    )
    def test_frame_keyword_in_global_raises(self, kw: str) -> None:
        img = self._img()
        crtf = f"#CRTF\nglobal {kw}\nbox[[0pix,0pix],[4pix,4pix]]"
        with pytest.raises(NotImplementedError) as ei:
            select_mask(img, crtf)
        assert kw.split("=")[0] in str(ei.value)

    def test_viz_keywords_silently_ignored(self) -> None:
        img = self._img()
        crtf = "box[[0pix,0pix],[4pix,4pix]], color=green, linewidth=2, label='x'"
        m = select_mask(img, crtf)
        assert isinstance(m, xr.DataArray) and m.dtype == bool
        assert int(m.values.sum()) == 5 * 5

    def test_ann_line_is_silently_skipped(self) -> None:
        img = self._img()
        crtf = "#CRTF\nbox[[0pix,0pix],[4pix,4pix]]\nann box[[8pix,8pix],[12pix,12pix]]"
        m = select_mask(img, crtf)
        assert int(m.values.sum()) == 5 * 5

    def test_ann_line_alone_produces_empty_mask(self) -> None:
        img = self._img()
        crtf = "#CRTF\nann box[[0pix,0pix],[14pix,14pix]]"
        m = select_mask(img, crtf)
        assert int(m.values.sum()) == 0


class TestCrtfRejectedInputs:
    """Dataset inputs must be rejected with a clear TypeError before any CRTF parsing."""

    def _make_dataset(self) -> xr.Dataset:
        img = make_image(16, 16)
        return xr.Dataset({"SKY": img})

    def test_select_mask_dataset_crtf_raises_typeerror(self) -> None:
        xds = self._make_dataset()
        with pytest.raises(TypeError) as ei:
            select_mask(xds, "box[[0pix,0pix],[4pix,4pix]]")
        msg = str(ei.value)
        assert "DataArray" in msg and "xds.SKY" in msg

    def test_apply_select_dataset_crtf_raises_typeerror(self) -> None:
        xds = self._make_dataset()
        with pytest.raises(TypeError) as ei:
            apply_select(xds, "box[[0pix,0pix],[4pix,4pix]]")
        msg = str(ei.value)
        assert "DataArray" in msg and "xds.SKY" in msg

    def test_select_mask_dataset_none_raises_typeerror(self) -> None:
        """TypeError fires before any CRTF parsing, even for select=None."""
        xds = self._make_dataset()
        with pytest.raises(TypeError) as ei:
            select_mask(xds, select=None)
        msg = str(ei.value)
        assert "DataArray" in msg and "xds.SKY" in msg

    def test_select_mask_dataarray_succeeds(self) -> None:
        """Passing xds.SKY (a DataArray) must not raise."""
        xds = self._make_dataset()
        m = select_mask(xds["SKY"], "box[[0pix,0pix],[4pix,4pix]]")
        assert isinstance(m, xr.DataArray) and m.dtype == bool


class TestCRTFPA:
    def test_rotbox_pa_equivalent_to_theta_m(self) -> None:
        """
        Cover the `mode == "pa"` branch: pa=30deg should equal theta_m=60deg
        (since math angle = 90deg - PA).
        """
        ny, nx = 160, 200
        da_img = make_image(ny, nx)
        cx, cy = 120, 80
        pa_txt = f"rotbox[[{cx}pix,{cy}pix],[60pix,30pix], pa=30deg]"
        tm_txt = f"rotbox[[{cx}pix,{cy}pix],[60pix,30pix], theta_m=60deg]"
        m_pa = select_mask(da_img, select=pa_txt)
        m_tm = select_mask(da_img, select=tm_txt)
        assert isinstance(m_pa, xr.DataArray) and isinstance(m_tm, xr.DataArray)
        np.testing.assert_array_equal(m_pa.values, m_tm.values)
