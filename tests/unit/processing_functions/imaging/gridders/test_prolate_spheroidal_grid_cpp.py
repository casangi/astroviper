"""Unit tests for the C++ prolate_spheroidal_grid extension.

Coverage:
  * Numerical agreement with the Numba reference (`prolate_spheroidal_grid_jit`)
  * Serial vs multi-threaded equivalence
  * `num_threads <= 0` auto-detects hardware concurrency
  * Accumulation semantics (two calls == one combined call)
  * NaN UV / NaN weight / zero weight are skipped
  * Samples whose support falls outside the grid are skipped
  * Channel / time / pol index maps (cube and continuum collapse)
  * Dirac-delta point source recovers the convolution kernel
  * Binding input validation (wrong ndim)
"""

import unittest

import numpy as np

from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid import (
    prolate_spheroidal_grid_jit,
)
from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp import (
    prolate_spheroidal_grid,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
)

SUPPORT = 7
OVERSAMPLING = 100


def _zero_grid(m_time, m_chan, m_pol, m_u, m_v):
    return (
        np.zeros((m_time, m_chan, m_pol, m_u, m_v), dtype=np.complex128),
        np.zeros((m_time, m_chan, m_pol), dtype=np.float64),
    )


def _make_random_inputs(
    n_time=2,
    n_baseline=64,
    n_vis_chan=3,
    n_pol=2,
    m_u=96,
    m_v=96,
    m_time=None,
    m_chan=None,
    m_pol=None,
    seed=0,
    uv_extent=30.0,
):
    """Build a self-consistent batch of inputs for the gridder."""
    rng = np.random.default_rng(seed)
    m_time = n_time if m_time is None else m_time
    m_chan = n_vis_chan if m_chan is None else m_chan
    m_pol = n_pol if m_pol is None else m_pol

    vis_data = (
        rng.standard_normal((n_time, n_baseline, n_vis_chan, n_pol))
        + 1j * rng.standard_normal((n_time, n_baseline, n_vis_chan, n_pol))
    ).astype(np.complex128)
    # Confine UV coordinates so no samples wander outside the padded grid.
    uvw = rng.uniform(-uv_extent, uv_extent, size=(n_time, n_baseline, 3))
    freq = np.linspace(1.0e9, 1.5e9, n_vis_chan)
    fmap = np.arange(n_vis_chan, dtype=np.int64)
    tmap = np.arange(n_time, dtype=np.int64) % m_time
    pmap = np.arange(n_pol, dtype=np.int64) % m_pol
    weight = rng.uniform(0.1, 1.0, size=(n_time, n_baseline, n_vis_chan, n_pol))
    cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT).astype(np.float64)
    n_uv = np.array([m_u, m_v], dtype=np.int64)
    delta_lm = np.array([2.0e-4, 2.0e-4], dtype=np.float64)

    return dict(
        vis_data=vis_data,
        uvw=uvw,
        frequency_coord=freq,
        frequency_map=fmap,
        time_map=tmap,
        pol_map=pmap,
        weight=weight,
        cgk_1D=cgk,
        n_uv=n_uv,
        delta_lm=delta_lm,
        m_time=m_time,
        m_chan=m_chan,
        m_pol=m_pol,
        m_u=m_u,
        m_v=m_v,
    )


def _run_cpp(inputs, num_threads=1):
    grid, norm = _zero_grid(
        inputs["m_time"],
        inputs["m_chan"],
        inputs["m_pol"],
        inputs["m_u"],
        inputs["m_v"],
    )
    prolate_spheroidal_grid(
        grid,
        norm,
        inputs["vis_data"],
        inputs["uvw"],
        inputs["frequency_coord"],
        inputs["frequency_map"],
        inputs["time_map"],
        inputs["pol_map"],
        inputs["weight"],
        inputs["cgk_1D"],
        inputs["n_uv"],
        inputs["delta_lm"],
        support=SUPPORT,
        oversampling=OVERSAMPLING,
        num_threads=num_threads,
    )
    return grid, norm


def _run_jit(inputs):
    grid, norm = _zero_grid(
        inputs["m_time"],
        inputs["m_chan"],
        inputs["m_pol"],
        inputs["m_u"],
        inputs["m_v"],
    )
    prolate_spheroidal_grid_jit(
        grid,
        norm,
        inputs["vis_data"],
        inputs["uvw"],
        inputs["frequency_coord"],
        inputs["frequency_map"],
        inputs["time_map"],
        inputs["pol_map"],
        inputs["weight"],
        inputs["cgk_1D"],
        inputs["n_uv"],
        inputs["delta_lm"],
        support=SUPPORT,
        oversampling=OVERSAMPLING,
    )
    return grid, norm


class TestProlateSpheroidalGridCpp(unittest.TestCase):
    # ------------------------------------------------------------------
    # Reference-oracle tests
    # ------------------------------------------------------------------
    def test_matches_numba_reference(self):
        """C++ gridder matches the numba reference bit-for-bit-ish."""
        inputs = _make_random_inputs(seed=1)
        cpp_grid, cpp_norm = _run_cpp(inputs, num_threads=1)
        ref_grid, ref_norm = _run_jit(inputs)
        np.testing.assert_allclose(cpp_grid, ref_grid, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cpp_norm, ref_norm, rtol=0, atol=1e-12)
        self.assertGreater(np.count_nonzero(cpp_grid), 0)

    def test_continuum_collapse(self):
        """frequency_map collapsing all channels onto image chan 0."""
        inputs = _make_random_inputs(n_vis_chan=4, m_chan=1, seed=2)
        inputs["frequency_map"] = np.zeros(4, dtype=np.int64)
        cpp_grid, cpp_norm = _run_cpp(inputs, num_threads=1)
        ref_grid, ref_norm = _run_jit(inputs)
        np.testing.assert_allclose(cpp_grid, ref_grid, rtol=0, atol=1e-12)
        np.testing.assert_allclose(cpp_norm, ref_norm, rtol=0, atol=1e-12)
        # All energy must land in the single collapsed image channel.
        self.assertEqual(cpp_grid.shape[1], 1)
        self.assertGreater(np.count_nonzero(cpp_grid[:, 0]), 0)

    def test_time_map_collapse(self):
        """time_map collapsing multiple time steps onto image time 0."""
        inputs = _make_random_inputs(n_time=3, m_time=1, seed=3)
        inputs["time_map"] = np.zeros(3, dtype=np.int64)
        cpp_grid, _ = _run_cpp(inputs, num_threads=1)
        ref_grid, _ = _run_jit(inputs)
        np.testing.assert_allclose(cpp_grid, ref_grid, rtol=0, atol=1e-12)
        self.assertEqual(cpp_grid.shape[0], 1)

    def test_pol_map_reorder(self):
        """pol_map reorders the visibility polarizations into the image pol axis."""
        inputs = _make_random_inputs(n_pol=2, seed=4)
        inputs["pol_map"] = np.array([1, 0], dtype=np.int64)  # swap the two pols
        cpp_grid, _ = _run_cpp(inputs, num_threads=1)
        ref_grid, _ = _run_jit(inputs)
        np.testing.assert_allclose(cpp_grid, ref_grid, rtol=0, atol=1e-12)

    # ------------------------------------------------------------------
    # Threading
    # ------------------------------------------------------------------
    def test_serial_vs_threaded_equivalent(self):
        """Threaded result matches the serial result to floating-point tolerance.

        Atomic accumulation changes the summation order, so the result is not
        bit-identical — but it must be numerically close.
        """
        inputs = _make_random_inputs(n_baseline=256, seed=5)
        grid_serial, norm_serial = _run_cpp(inputs, num_threads=1)
        for nt in (2, 4, 8):
            with self.subTest(num_threads=nt):
                grid_t, norm_t = _run_cpp(inputs, num_threads=nt)
                np.testing.assert_allclose(grid_t, grid_serial, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(norm_t, norm_serial, rtol=1e-10, atol=1e-10)

    def test_num_threads_auto_detect(self):
        """num_threads <= 0 falls back to hardware_concurrency()."""
        inputs = _make_random_inputs(seed=6)
        grid_serial, norm_serial = _run_cpp(inputs, num_threads=1)
        grid_auto, norm_auto = _run_cpp(inputs, num_threads=0)
        np.testing.assert_allclose(grid_auto, grid_serial, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(norm_auto, norm_serial, rtol=1e-10, atol=1e-10)

    def test_threaded_matches_reference(self):
        """Threaded C++ output matches the numba reference within tolerance."""
        inputs = _make_random_inputs(n_baseline=200, seed=7)
        cpp_grid, cpp_norm = _run_cpp(inputs, num_threads=4)
        ref_grid, ref_norm = _run_jit(inputs)
        np.testing.assert_allclose(cpp_grid, ref_grid, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(cpp_norm, ref_norm, rtol=1e-10, atol=1e-10)

    # ------------------------------------------------------------------
    # Accumulation semantics (in-place add across calls)
    # ------------------------------------------------------------------
    def test_accumulates_in_place(self):
        """Two calls on disjoint halves equal one call on the full set."""
        inputs = _make_random_inputs(n_baseline=64, seed=8)

        full_grid, full_norm = _run_cpp(inputs, num_threads=1)

        split = inputs["uvw"].shape[1] // 2
        first = dict(inputs)
        first["vis_data"] = inputs["vis_data"][:, :split]
        first["uvw"] = inputs["uvw"][:, :split]
        first["weight"] = inputs["weight"][:, :split]
        second = dict(inputs)
        second["vis_data"] = inputs["vis_data"][:, split:]
        second["uvw"] = inputs["uvw"][:, split:]
        second["weight"] = inputs["weight"][:, split:]

        grid, norm = _zero_grid(
            inputs["m_time"],
            inputs["m_chan"],
            inputs["m_pol"],
            inputs["m_u"],
            inputs["m_v"],
        )
        for part in (first, second):
            prolate_spheroidal_grid(
                grid,
                norm,
                np.ascontiguousarray(part["vis_data"]),
                np.ascontiguousarray(part["uvw"]),
                part["frequency_coord"],
                part["frequency_map"],
                part["time_map"],
                part["pol_map"],
                np.ascontiguousarray(part["weight"]),
                part["cgk_1D"],
                part["n_uv"],
                part["delta_lm"],
                support=SUPPORT,
                oversampling=OVERSAMPLING,
                num_threads=1,
            )

        np.testing.assert_allclose(grid, full_grid, rtol=0, atol=1e-12)
        np.testing.assert_allclose(norm, full_norm, rtol=0, atol=1e-12)

    # ------------------------------------------------------------------
    # Skip / guard behaviours
    # ------------------------------------------------------------------
    def test_nan_uvw_skipped(self):
        """A NaN UVW sample must not alter grid / normalization."""
        inputs = _make_random_inputs(n_baseline=16, seed=9)

        baseline_grid, baseline_norm = _run_cpp(inputs, num_threads=1)

        inputs2 = dict(inputs)
        uvw_nan = inputs["uvw"].copy()
        # Adding a NaN sample on a new baseline must not change the output.
        extra_uvw = np.full((inputs["uvw"].shape[0], 1, 3), np.nan)
        inputs2["uvw"] = np.concatenate([uvw_nan, extra_uvw], axis=1)
        inputs2["vis_data"] = np.concatenate(
            [
                inputs["vis_data"],
                np.ones(
                    (
                        inputs["vis_data"].shape[0],
                        1,
                        inputs["vis_data"].shape[2],
                        inputs["vis_data"].shape[3],
                    ),
                    dtype=np.complex128,
                ),
            ],
            axis=1,
        )
        inputs2["weight"] = np.concatenate(
            [
                inputs["weight"],
                np.ones(
                    (
                        inputs["weight"].shape[0],
                        1,
                        inputs["weight"].shape[2],
                        inputs["weight"].shape[3],
                    ),
                    dtype=np.float64,
                ),
            ],
            axis=1,
        )

        grid_with_nan, norm_with_nan = _run_cpp(inputs2, num_threads=1)
        np.testing.assert_allclose(grid_with_nan, baseline_grid, rtol=0, atol=0)
        np.testing.assert_allclose(norm_with_nan, baseline_norm, rtol=0, atol=0)

    def test_zero_weighted_data_skipped(self):
        """A sample whose weighted data is identically zero must be skipped."""
        inputs = _make_random_inputs(n_baseline=8, seed=10)

        inputs["weight"][:, 0, :, :] = 0.0
        inputs["vis_data"][:, 0, :, :] = 0.0

        # Reference with the zero row dropped entirely.
        drop = dict(inputs)
        drop["vis_data"] = inputs["vis_data"][:, 1:]
        drop["uvw"] = inputs["uvw"][:, 1:]
        drop["weight"] = inputs["weight"][:, 1:]

        full_grid, full_norm = _run_cpp(inputs, num_threads=1)
        drop_grid, drop_norm = _run_cpp(
            {
                **drop,
                "vis_data": np.ascontiguousarray(drop["vis_data"]),
                "uvw": np.ascontiguousarray(drop["uvw"]),
                "weight": np.ascontiguousarray(drop["weight"]),
            },
            num_threads=1,
        )
        np.testing.assert_allclose(full_grid, drop_grid, rtol=0, atol=0)
        np.testing.assert_allclose(full_norm, drop_norm, rtol=0, atol=0)

    def test_out_of_bounds_sample_skipped(self):
        """A sample whose support falls outside the grid is skipped entirely."""
        inputs = _make_random_inputs(n_baseline=1, m_u=64, m_v=64, seed=11)
        # Push the single visibility far outside the grid: delta_lm*m_u/c is
        # small (~1.0e-4 * 64 / 3e8 ~ 2e-11) so uvw of 1e15 gives u ~ 2e4
        # pixels, comfortably out of bounds on either axis.
        inputs["uvw"][:] = 1.0e15
        grid, norm = _run_cpp(inputs, num_threads=1)
        self.assertEqual(np.count_nonzero(grid), 0)
        self.assertEqual(np.count_nonzero(norm), 0)

    # ------------------------------------------------------------------
    # Analytical: Dirac delta at UV origin deposits the kernel
    # ------------------------------------------------------------------
    def test_dirac_delta_deposits_kernel(self):
        """A single unit visibility at uvw=0 deposits the outer-product kernel.

        At uvw=(0,0,0) the sample lands exactly on the grid centre, so the
        gridder should add the separable kernel `cgk[|i_u|] * cgk[|i_v|]`
        into a SUPPORT x SUPPORT patch centred on (m_u/2, m_v/2). Check the
        kernel peak (center) and that the patch is symmetric.
        """
        m_u = m_v = 128
        grid = np.zeros((1, 1, 1, m_u, m_v), dtype=np.complex128)
        norm = np.zeros((1, 1, 1), dtype=np.float64)
        vis = np.ones((1, 1, 1, 1), dtype=np.complex128)
        uvw = np.zeros((1, 1, 3), dtype=np.float64)
        freq = np.array([1.0e9], dtype=np.float64)
        fmap = np.array([0], dtype=np.int64)
        tmap = np.array([0], dtype=np.int64)
        pmap = np.array([0], dtype=np.int64)
        weight = np.ones((1, 1, 1, 1), dtype=np.float64)
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT).astype(
            np.float64
        )
        n_uv = np.array([m_u, m_v], dtype=np.int64)
        delta_lm = np.array([1.0e-4, 1.0e-4], dtype=np.float64)

        prolate_spheroidal_grid(
            grid,
            norm,
            vis,
            uvw,
            freq,
            fmap,
            tmap,
            pmap,
            weight,
            cgk,
            n_uv,
            delta_lm,
            support=SUPPORT,
            oversampling=OVERSAMPLING,
            num_threads=1,
        )

        s = SUPPORT // 2
        center_u, center_v = m_u // 2, m_v // 2
        patch = grid[
            0, 0, 0, center_u - s : center_u + s + 1, center_v - s : center_v + s + 1
        ]

        # Peak sits at the centre of the patch.
        peak_idx = np.unravel_index(np.argmax(np.abs(patch)), patch.shape)
        self.assertEqual(peak_idx, (s, s))

        # Expected separable outer product: cgk[|i_u|*oversampling] * cgk[|i_v|*oversampling]
        offsets = np.arange(-s, s + 1)
        kernel_1d = cgk[np.abs(offsets) * OVERSAMPLING]
        expected = np.outer(kernel_1d, kernel_1d).astype(np.complex128)
        np.testing.assert_allclose(patch, expected, rtol=0, atol=1e-12)

        # normalization == sum(kernel) since weight == 1.
        self.assertAlmostEqual(norm[0, 0, 0], float(expected.sum().real), places=12)

    # ------------------------------------------------------------------
    # Binding input validation
    # ------------------------------------------------------------------
    def test_wrong_ndim_raises(self):
        inputs = _make_random_inputs(seed=12)
        bad_grid = np.zeros(
            (inputs["m_chan"], inputs["m_pol"], inputs["m_u"], inputs["m_v"]),
            dtype=np.complex128,
        )  # 4-D, not 5-D
        norm = np.zeros(
            (inputs["m_time"], inputs["m_chan"], inputs["m_pol"]), dtype=np.float64
        )
        with self.assertRaises(RuntimeError):
            prolate_spheroidal_grid(
                bad_grid,
                norm,
                inputs["vis_data"],
                inputs["uvw"],
                inputs["frequency_coord"],
                inputs["frequency_map"],
                inputs["time_map"],
                inputs["pol_map"],
                inputs["weight"],
                inputs["cgk_1D"],
                inputs["n_uv"],
                inputs["delta_lm"],
                support=SUPPORT,
                oversampling=OVERSAMPLING,
                num_threads=1,
            )


class TestSinglePrecisionGrid(unittest.TestCase):
    """Single-precision (complex64) grid path.

    The binding dispatches on the grid dtype: a complex64 grid is accumulated
    directly (the visibilities/weights stay double), giving a complex64 result
    that matches the complex128 grid to single-precision tolerance.
    """

    def test_complex64_grid_matches_complex128(self):
        inputs = _make_random_inputs(seed=7)

        grid128, norm128 = _run_cpp(inputs, num_threads=1)

        grid64 = np.zeros(grid128.shape, dtype=np.complex64)
        norm64 = np.zeros(norm128.shape, dtype=np.float64)
        prolate_spheroidal_grid(
            grid64,
            norm64,
            inputs["vis_data"],
            inputs["uvw"],
            inputs["frequency_coord"],
            inputs["frequency_map"],
            inputs["time_map"],
            inputs["pol_map"],
            inputs["weight"],
            inputs["cgk_1D"],
            inputs["n_uv"],
            inputs["delta_lm"],
            support=SUPPORT,
            oversampling=OVERSAMPLING,
            num_threads=1,
        )

        self.assertEqual(grid64.dtype, np.complex64)
        # Normalization stays float64 and is identical (double accumulation).
        np.testing.assert_allclose(norm64, norm128, rtol=0, atol=1e-10)
        # Grid agrees to single-precision tolerance, scaled by the cell magnitude.
        scale = max(1.0, float(np.max(np.abs(grid128))))
        np.testing.assert_allclose(
            grid64.astype(np.complex128), grid128, rtol=0, atol=1e-5 * scale
        )

    def test_complex64_grid_threaded_matches_serial(self):
        inputs = _make_random_inputs(seed=11)
        common = dict(support=SUPPORT, oversampling=OVERSAMPLING)

        def run(num_threads):
            grid = np.zeros(
                (
                    inputs["m_time"],
                    inputs["m_chan"],
                    inputs["m_pol"],
                    inputs["m_u"],
                    inputs["m_v"],
                ),
                dtype=np.complex64,
            )
            norm = np.zeros(
                (inputs["m_time"], inputs["m_chan"], inputs["m_pol"]),
                dtype=np.float64,
            )
            prolate_spheroidal_grid(
                grid,
                norm,
                inputs["vis_data"],
                inputs["uvw"],
                inputs["frequency_coord"],
                inputs["frequency_map"],
                inputs["time_map"],
                inputs["pol_map"],
                inputs["weight"],
                inputs["cgk_1D"],
                inputs["n_uv"],
                inputs["delta_lm"],
                num_threads=num_threads,
                **common,
            )
            return grid

        serial = run(1)
        threaded = run(4)
        # Atomic accumulation reorders the float32 summation across threads, so
        # the result is close but not bit-identical at single precision.
        scale = max(1.0, float(np.max(np.abs(serial))))
        np.testing.assert_allclose(threaded, serial, rtol=1e-4, atol=1e-4 * scale)


class TestSinglePrecisionDegrid(unittest.TestCase):
    """Single-precision (complex64) model grid degridded to double visibilities."""

    def test_complex64_grid_degrid_to_complex128_vis(self):
        from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp import (
            prolate_spheroidal_degrid,
        )

        inputs = _make_random_inputs(seed=3)
        grid128, _ = _run_cpp(inputs, num_threads=1)
        grid64 = grid128.astype(np.complex64)

        n_time, n_baseline, n_vis_chan, n_pol = inputs["vis_data"].shape

        def degrid(grid):
            vis = np.zeros((n_time, n_baseline, n_vis_chan, n_pol), dtype=np.complex128)
            prolate_spheroidal_degrid(
                grid,
                vis,
                inputs["uvw"],
                inputs["frequency_coord"],
                inputs["frequency_map"],
                inputs["time_map"],
                inputs["pol_map"],
                inputs["cgk_1D"],
                inputs["n_uv"],
                inputs["delta_lm"],
                support=SUPPORT,
                oversampling=OVERSAMPLING,
                num_threads=1,
            )
            return vis

        vis_from_c128 = degrid(grid128)
        vis_from_c64 = degrid(grid64)

        # Output visibilities stay double precision regardless of grid dtype.
        self.assertEqual(vis_from_c64.dtype, np.complex128)
        scale = max(1.0, float(np.max(np.abs(vis_from_c128))))
        np.testing.assert_allclose(
            vis_from_c64, vis_from_c128, rtol=0, atol=1e-5 * scale
        )


if __name__ == "__main__":
    unittest.main()
