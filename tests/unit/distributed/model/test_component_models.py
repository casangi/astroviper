import warnings

warnings.filterwarnings(
    "ignore", message="The NumPy module was reloaded", category=UserWarning
)

import unittest

import numpy as np
import dask.array as da
import xarray as xr

# Adjust import if your package layout differs
from astroviper.distributed.model import make_disk, make_gauss2d, make_pt_sources

import astroviper.distributed.model.component_models as cm


def _base_grid(
    nx=101, ny=101, x_start=-5.0, x_stop=5.0, y_start=-5.0, y_stop=5.0, reverse_x=False
):
    y = np.linspace(y_start, y_stop, ny)
    x = np.linspace(x_start, x_stop, nx)
    if reverse_x:
        x = x[::-1]
    data = xr.DataArray(
        np.zeros((ny, nx), dtype=float),
        coords={"y": y, "x": x},
        dims=("y", "x"),
        name="base",
    )
    return data


class TestAngleModes(unittest.TestCase):
    def test_auto_uses_pa_on_left_handed_grid(self):
        base = _base_grid(reverse_x=True)  # left-handed: dx * dy < 0
        out_pa = make_gauss2d(
            base,
            a=2.355,
            b=2.355,
            theta=45,  # input in degrees
            degrees=True,
            angle="auto",  # should choose PA on left-handed grids
            x0=0.0,
            y0=0.0,
            peak=10.0,
            add=False,
            output="xarray",
        )
        out_forced_pa = make_gauss2d(
            base,
            a=2.355,
            b=2.355,
            theta=45,
            degrees=True,
            angle="pa",  # force PA explicitly
            x0=0.0,
            y0=0.0,
            peak=10.0,
            add=False,
            output="xarray",
        )
        self.assertTrue(
            np.allclose(out_pa.values, out_forced_pa.values),
            "angle='auto' should act like 'pa' on left-handed grids",
        )

    def test_auto_uses_math_on_right_handed_grid(self):
        base = _base_grid(reverse_x=False)  # right-handed: dx * dy > 0
        out_math = make_disk(
            base,
            a=3.0,
            b=1.5,
            theta=np.pi / 6,  # radians
            angle="auto",  # should choose math on right-handed grids
            x0=0.0,
            y0=0.0,
            height=2.0,
            add=False,
            output="xarray",
        )
        out_forced_math = make_disk(
            base,
            a=3.0,
            b=1.5,
            theta=np.pi / 6,
            angle="math",  # force math explicitly
            x0=0.0,
            y0=0.0,
            height=2.0,
            add=False,
            output="xarray",
        )
        self.assertTrue(
            np.allclose(out_math.values, out_forced_math.values),
            "angle='auto' should act like 'math' on right-handed grids",
        )

    def test_degrees_flag(self):
        base = _base_grid()
        out_deg = make_gauss2d(
            base,
            a=2.355,
            b=2.355,
            theta=30,  # degrees
            degrees=True,
            angle="math",
            x0=0.0,
            y0=0.0,
            peak=5.0,
            add=False,
            output="xarray",
        )
        out_rad = make_gauss2d(
            base,
            a=2.355,
            b=2.355,
            theta=np.deg2rad(30),  # radians
            degrees=False,
            angle="math",
            x0=0.0,
            y0=0.0,
            peak=5.0,
            add=False,
            output="xarray",
        )
        self.assertTrue(
            np.allclose(out_deg.values, out_rad.values),
            "degrees=True should match radians when converted",
        )


class TestMakeDisk(unittest.TestCase):
    def test_add_true_adds_only_inside(self):
        base = _base_grid()
        out = make_disk(
            base,
            a=2.0,
            b=1.0,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            height=3.0,
            add=True,
            output="xarray",
        )
        self.assertIsInstance(
            out, xr.DataArray, "make_disk(add=True) should return an xarray.DataArray"
        )
        center_val_before = float(base.sel(x=0.0, y=0.0, method="nearest").values)
        center_val_after = float(out.sel(x=0.0, y=0.0, method="nearest").values)
        self.assertEqual(
            center_val_before + 3.0,
            center_val_after,
            "With add=True, center pixel should increase by A",
        )
        far_val_after = float(out.sel(x=5.0, y=5.0, method="nearest").values)
        self.assertEqual(
            0.0, far_val_after, "Pixels outside the disk should remain unchanged"
        )

    def test_add_false_replaces_inside(self):
        base = _base_grid()
        base2 = base + 5.0
        out = make_disk(
            base2,
            a=2.0,
            b=1.0,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            height=3.0,
            add=False,
            output="xarray",
        )
        inside = float(out.sel(x=0.0, y=0.0, method="nearest").values)
        outside = float(out.sel(x=5.0, y=5.0, method="nearest").values)
        self.assertEqual(3.0, inside, "With add=False, inside pixel should be A")
        self.assertEqual(
            5.0, outside, "With add=False, pixels outside disk should be unchanged"
        )

    def test_output_kinds_numpy_and_dask(self):
        base = _base_grid()
        out_np = make_disk(
            base.values,
            a=1.0,
            b=1.0,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            height=2.0,
            coords={"y": base["y"].values, "x": base["x"].values},
            output="numpy",
        )
        self.assertIsInstance(
            out_np, np.ndarray, "output='numpy' should return a numpy.ndarray"
        )
        darr = da.zeros_like(base.values)
        out_dask = make_disk(
            darr,
            a=1.0,
            b=1.0,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            height=2.0,
            coords={"y": base["y"].values, "x": base["x"].values},
            output="dask",
        )
        self.assertIsInstance(
            out_dask, da.Array, "output='dask' should return a dask.array.Array"
        )

    def test_param_validation(self):
        base = _base_grid()
        with self.assertRaises(ValueError):
            make_disk(base, a=-1.0, b=1.0, theta=0.0, x0=0.0, y0=0.0, height=1.0)
        with self.assertRaises(ValueError):
            make_disk(base, a=1.0, b=0.0, theta=np.nan, x0=0.0, y0=0.0, height=1.0)


class TestMakeGauss2D(unittest.TestCase):
    def test_peak_at_center_replacement(self):
        base = _base_grid()
        out = make_gauss2d(
            base,
            a=2.355,
            b=2.355,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            peak=10.0,
            add=False,
            output="xarray",
        )
        peak = float(out.sel(x=0.0, y=0.0, method="nearest").values)
        self.assertAlmostEqual(
            10.0, peak, places=10, msg="With add=False, peak should equal A at center"
        )

    def test_add_true_adds_gaussian(self):
        base = _base_grid()
        out = make_gauss2d(
            base + 1.0,
            a=2.355,
            b=2.355,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            peak=3.0,
            add=True,
            output="xarray",
        )
        center = float(out.sel(x=0.0, y=0.0, method="nearest").values)
        self.assertAlmostEqual(
            4.0,
            center,
            places=10,
            msg="With add=True, center should be baseline + A",
        )

    def test_output_kinds(self):
        base = _base_grid()
        out_xr = make_gauss2d(
            base, a=2.355, b=2.355, theta=0.0, x0=0.0, y0=0.0, peak=1.0, output="xarray"
        )
        self.assertIsInstance(
            out_xr, xr.DataArray, "output='xarray' should return an xarray.DataArray"
        )
        out_np = make_gauss2d(
            base.values,
            a=2.355,
            b=2.355,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            peak=1.0,
            coords={"y": base["y"].values, "x": base["x"].values},
            output="numpy",
        )
        self.assertIsInstance(
            out_np, np.ndarray, "output='numpy' should return a numpy.ndarray"
        )
        out_dask = make_gauss2d(
            da.zeros_like(base.values),
            a=2.355,
            b=2.355,
            theta=0.0,
            x0=0.0,
            y0=0.0,
            peak=1.0,
            coords={"y": base["y"].values, "x": base["x"].values},
            output="dask",
        )
        self.assertIsInstance(
            out_dask, da.Array, "output='dask' should return a dask.array.Array"
        )

    def test_decreasing_x_coords_ok(self):
        base_rev = _base_grid(reverse_x=True)
        out = make_gauss2d(
            base_rev,
            a=2.355,
            b=2.355,
            theta=0.0,
            x0=1.0,
            y0=-2.0,
            peak=5.0,
            output="xarray",
        )
        self.assertIsInstance(
            out, xr.DataArray, "make_gauss2d should return an xarray.DataArray"
        )
        self.assertTrue(
            np.isfinite(out.values).all(),
            "Gaussian output should be finite everywhere on the grid",
        )


class TestMakePtSources(unittest.TestCase):
    def test_sum_duplicates(self):
        base = _base_grid(nx=7, ny=5)
        A = [2.0, 3.0]
        xs = [0.0, 0.0]
        ys = [0.0, 0.0]
        out = make_pt_sources(
            base, amplitudes=A, xs=xs, ys=ys, add=False, output="xarray"
        )
        v = float(out.sel(x=0.0, y=0.0, method="nearest").values)
        self.assertEqual(
            5.0, v, "Amplitudes for duplicate sources should sum at the same pixel"
        )

    def test_add_true_adds_to_existing(self):
        base = _base_grid() + 1.0
        out = make_pt_sources(
            base, amplitudes=[4.0], xs=[0.0], ys=[0.0], add=True, output="xarray"
        )
        v = float(out.sel(x=0.0, y=0.0, method="nearest").values)
        self.assertEqual(5.0, v, "With add=True, pixel should be baseline + amplitude")

    def test_midpoint_tie_chooses_right(self):
        base = _base_grid(nx=2, ny=2, x_start=0.0, x_stop=1.0, y_start=0.0, y_stop=1.0)
        out = make_pt_sources(
            base, amplitudes=[1.0], xs=[0.5], ys=[0.5], add=False, output="xarray"
        )
        val_00 = float(out.sel(x=0.0, y=0.0, method="nearest").values)
        val_10 = float(out.sel(x=1.0, y=0.0, method="nearest").values)
        val_01 = float(out.sel(x=0.0, y=1.0, method="nearest").values)
        val_11 = float(out.sel(x=1.0, y=1.0, method="nearest").values)
        self.assertEqual(0.0, val_00, "Midpoint tie should not land at (0,0)")
        self.assertEqual(0.0, val_10, "Midpoint tie should not land at (1,0)")
        self.assertEqual(0.0, val_01, "Midpoint tie should not land at (0,1)")
        self.assertEqual(1.0, val_11, "Midpoint tie should land at the right/up pixel")

    def test_output_kinds_numpy_and_dask(self):
        base = _base_grid()
        out_np = make_pt_sources(
            base.values,
            amplitudes=[3.0],
            xs=[0.0],
            ys=[0.0],
            coords={"y": base["y"].values, "x": base["x"].values},
            output="numpy",
        )
        self.assertIsInstance(
            out_np, np.ndarray, "output='numpy' should return a numpy.ndarray"
        )
        darr = da.zeros_like(base.values)
        out_dask = make_pt_sources(
            darr,
            amplitudes=[3.0],
            xs=[0.0],
            ys=[0.0],
            coords={"y": base["y"].values, "x": base["x"].values},
            output="dask",
        )
        self.assertIsInstance(
            out_dask, da.Array, "output='dask' should return a dask.array.Array"
        )

    def test_requires_equal_lengths(self):
        base = _base_grid()
        with self.assertRaises(ValueError):
            make_pt_sources(base, amplitudes=[1.0, 2.0], xs=[0.0], ys=[0.0])


class TestCoerceToXda(unittest.TestCase):
    def test_rejects_unsupported_type(self):
        with self.assertRaises(TypeError, msg="data must be DataArray/ndarray/Dask"):
            # Not DA / np.ndarray / dask
            _ = cm._coerce_to_xda(data="not-an-array", x_coord="x", y_coord="y")

    def test_rejects_non_2d_numpy(self):
        arr = np.zeros((3,), dtype=float)
        with self.assertRaises(ValueError, msg="NumPy/Dask array input must be 2-D"):
            _ = cm._coerce_to_xda(
                data=arr,
                x_coord="x",
                y_coord="y",
                coords={"x": np.arange(3), "y": np.arange(1)},
            )

    def test_requires_coords_for_raw_arrays(self):
        arr = np.zeros((2, 3), dtype=float)
        with self.assertRaises(ValueError, msg="coords must be provided"):
            _ = cm._coerce_to_xda(data=arr, x_coord="x", y_coord="y")

    def test_missing_coord_keys(self):
        arr = np.zeros((2, 3), dtype=float)
        with self.assertRaises(ValueError, msg="coords must include 1-D arrays"):
            _ = cm._coerce_to_xda(
                data=arr, x_coord="x", y_coord="y", coords={"x": np.arange(3)}
            )

    def test_length_mismatch(self):
        arr = np.zeros((2, 3), dtype=float)
        with self.assertRaises(
            ValueError, msg="Coordinate lengths must match array shape"
        ):
            _ = cm._coerce_to_xda(
                data=arr,
                x_coord="x",
                y_coord="y",
                coords={"x": np.arange(2), "y": np.arange(2)},  # x length wrong
            )


class TestInferHandedness(unittest.TestCase):
    def test_requires_len_ge_2(self):
        with self.assertRaises(ValueError, msg="length >= 2"):
            _ = cm._infer_handedness(np.array([0.0]), np.array([0.0, 1.0]))

    def test_non_monotonic_x_raises(self):
        x = np.array([0.0, 1.0, 1.0])  # not strictly monotonic
        y = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(
            ValueError, msg="x coordinates must be strictly monotonic"
        ):
            _ = cm._infer_handedness(x, y)

    def test_non_monotonic_y_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 1.0])
        with self.assertRaises(
            ValueError, msg="y coordinates must be strictly monotonic"
        ):
            _ = cm._infer_handedness(x, y)


class TestNearestOutOfRangePolicies(unittest.TestCase):
    def test_error_policy_raises(self):
        coords = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError, msg="outside the coordinate range"):
            _ = cm._nearest_indices_1d(
                coords, np.array([-1.0, 3.0]), out_of_range="error"
            )

    def test_ignore_returns_mask(self):
        coords = np.array([0.0, 1.0, 2.0])
        idx, valid = cm._nearest_indices_1d(
            coords,
            np.array([-0.1, 0.1, 1.4, 2.2]),
            out_of_range="ignore",
            return_valid_mask=True,
        )
        np.testing.assert_array_equal(
            valid,
            np.array([False, True, True, False]),
            err_msg="strict ignore mask wrong",
        )

    def test_ignore_sloppy_keeps_half_pixel(self):
        coords = np.array([0.0, 1.0, 2.0])
        half = 0.5 * (coords[1] - coords[0])
        idx, valid = cm._nearest_indices_1d(
            coords,
            np.array([-half + 1e-6, 2.0 + half - 1e-6]),
            out_of_range="ignore_sloppy",
            return_valid_mask=True,
        )
        np.testing.assert_array_equal(
            valid,
            np.array([True, True]),
            err_msg="sloppy ignore tolerance failed",
        )

    def test_clip_clamps(self):
        coords = np.array([0.0, 1.0, 2.0])
        idx = cm._nearest_indices_1d(
            coords, np.array([-10.0, 10.0]), out_of_range="clip"
        )
        np.testing.assert_array_equal(
            idx,
            np.array([0, 2]),
            err_msg="clip policy did not clamp to edges",
        )


class TestFinalizeOutput(unittest.TestCase):
    def test_invalid_output_kind_raises(self):
        base = _base_grid()
        with self.assertRaises(ValueError, msg="output must be one of"):
            _ = cm._finalize_output(base, base, output="nope")  # type: ignore[arg-type]


class TestParamValidation(unittest.TestCase):
    def test_make_disk_param_validation(self):
        base = _base_grid()
        with self.assertRaises(ValueError, msg="'a' must be positive"):
            make_disk(base, a=0.0, b=1.0, theta=0.0, x0=0.0, y0=0.0, height=1.0)
        with self.assertRaises(ValueError, msg="'theta' must be a finite number"):
            make_disk(base, a=1.0, b=1.0, theta=np.nan, x0=0.0, y0=0.0, height=1.0)

    def test_make_gauss2d_param_validation(self):
        base = _base_grid()
        with self.assertRaises(ValueError, msg="'b' must be positive"):
            make_gauss2d(base, a=1.0, b=-1.0, theta=0.0, x0=0.0, y0=0.0, peak=1.0)


class TestNearestIndicesEdgeCases(unittest.TestCase):
    def test_coord_vals_must_be_1d(self):
        coords = np.array([[0.0, 1.0]])  # 2-D
        with self.assertRaises(ValueError, msg="coord_vals must be 1-D"):
            cm._nearest_indices_1d(coords, np.array([0.5]))

    def test_coord_vals_length_at_least_one(self):
        coords = np.array([])  # empty
        with self.assertRaises(ValueError, msg="length >= 1"):
            cm._nearest_indices_1d(coords, np.array([0.0]))

    def test_coord_vals_strictly_monotonic(self):
        coords = np.array([0.0, 1.0, 1.0])  # not strictly increasing
        with self.assertRaises(ValueError, msg="strictly monotonic"):
            cm._nearest_indices_1d(coords, np.array([0.5]))

    def test_invalid_out_of_range_value_raises(self):
        coords = np.array([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError, msg="out_of_range must be one of"):
            cm._nearest_indices_1d(coords, np.array([0.5]), out_of_range="nope")  # type: ignore[arg-type]

    def test_descending_branch_maps_back(self):
        coords = np.array([2.0, 1.0, 0.0])  # strictly decreasing
        # target near 0.1 → nearest index 2 in decreasing orientation, must map-back correctly
        idx = cm._nearest_indices_1d(coords, np.array([0.1]), out_of_range="clip")
        np.testing.assert_array_equal(
            idx, np.array([2]), err_msg="descending map-back failed"
        )


class TestFinalizeOutputMatchDispatch(unittest.TestCase):
    def test_match_with_dataarray(self):
        base = _base_grid()
        out = cm._finalize_output(base, base, output="match")
        self.assertIsInstance(
            out, xr.DataArray, "match dispatch: DataArray input should return DataArray"
        )

    def test_match_with_dask(self):
        base = _base_grid()
        darr = da.from_array(base.values, chunks=base.values.shape)
        out = cm._finalize_output(base, darr, output="match")
        self.assertTrue(
            cm._is_dask_array(out),
            "match dispatch: dask input should return a dask array",
        )

    def test_match_with_numpy(self):
        base = _base_grid()
        out = cm._finalize_output(base, base.values, output="match")
        self.assertIsInstance(
            out, np.ndarray, "match dispatch: numpy input should return ndarray"
        )

    def test_dask_to_numpy_compute_line(self):
        base = _base_grid()
        darr = da.from_array(base.values + 1.0, chunks=base.values.shape)
        xda = xr.DataArray(darr, coords=base.coords, dims=base.dims)
        out = cm._finalize_output(xda, xda, output="numpy")
        self.assertIsInstance(
            out, np.ndarray, "finalize: dask->numpy should return ndarray"
        )
        self.assertTrue(
            np.allclose(out, base.values + 1.0),
            "finalize: dask->numpy should compute the lazy values",
        )

    def test_numpy_to_dask_wrap_line(self):
        base = _base_grid()
        out = cm._finalize_output(base, base, output="dask")
        self.assertTrue(
            cm._is_dask_array(out),
            "finalize: numpy->dask should wrap as dask.array.Array",
        )


class TestNearestIndicesSizeOneCompletion(unittest.TestCase):
    def test_ignore_sloppy_size_one_validity(self):
        coords = np.array([1.23])
        idx, valid = cm._nearest_indices_1d(
            coords,
            np.array([1.2300000001, 1.24]),
            out_of_range="ignore_sloppy",
            return_valid_mask=True,
        )
        np.testing.assert_array_equal(
            idx,
            np.array([0, 0]),
            err_msg="size-1: indices should map to the sole pixel (0) under ignore_sloppy",
        )
        np.testing.assert_array_equal(
            valid,
            np.array([True, False]),
            err_msg="size-1: isclose target must be valid; farther target must be invalid",
        )

    def test_clip_with_mask_size_one(self):
        coords = np.array([5.0])
        idx, valid = cm._nearest_indices_1d(
            coords,
            np.array([100.0, -100.0]),
            out_of_range="clip",
            return_valid_mask=True,
        )
        np.testing.assert_array_equal(
            idx,
            np.array([0, 0]),
            err_msg="size-1: clip must clamp both extreme targets to index 0",
        )
        np.testing.assert_array_equal(
            valid,
            np.array([True, True]),
            err_msg="size-1: clip+mask must return True for all targets",
        )

    def test_ignore_plain_return_no_mask(self):
        coords = np.array([0.0, 1.0, 2.0])
        idx = cm._nearest_indices_1d(
            coords, np.array([0.2, 2.2]), out_of_range="ignore"
        )
        np.testing.assert_array_equal(
            idx,
            np.array([0, 2]),
            err_msg="ignore (no mask): nearest indices incorrect for [0.2, 2.2]",
        )


class TestMakePtSourcesEarlyExit(unittest.TestCase):
    def test_all_out_of_range_early_exit(self):
        base = _base_grid()
        # Put both sources far outside; strict ignore drops them all -> early exit path
        out = make_pt_sources(
            base,
            amplitudes=[3.0, 7.0],
            xs=[1e9, -1e9],
            ys=[1e9, -1e9],
            out_of_range="ignore",
            add=True,
            output="xarray",
        )
        # Should equal the original zeros
        self.assertIsInstance(
            out, xr.DataArray, "early-exit: output should still be an xarray.DataArray"
        )
        self.assertTrue(
            np.array_equal(out.values, base.values),
            "early-exit: result must equal the unmodified base when all sources are OOR",
        )
