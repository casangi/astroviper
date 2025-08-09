import unittest

import numpy as np
import dask.array as da
import xarray as xr

# Adjust import if your package layout differs
from astroviper.model import make_disk, make_gauss2d, make_pt_sources


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
            A=3.0,
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
            A=3.0,
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
            A=2.0,
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
            A=2.0,
            coords={"y": base["y"].values, "x": base["x"].values},
            output="dask",
        )
        self.assertIsInstance(
            out_dask, da.Array, "output='dask' should return a dask.array.Array"
        )

    def test_param_validation(self):
        base = _base_grid()
        with self.assertRaises(ValueError):
            make_disk(base, a=-1.0, b=1.0, theta=0.0, x0=0.0, y0=0.0, A=1.0)
        with self.assertRaises(ValueError):
            make_disk(base, a=1.0, b=0.0, theta=np.nan, x0=0.0, y0=0.0, A=1.0)


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
            A=10.0,
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
            A=3.0,
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
            base, a=2.355, b=2.355, theta=0.0, x0=0.0, y0=0.0, A=1.0, output="xarray"
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
            A=1.0,
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
            A=1.0,
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
            A=5.0,
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
