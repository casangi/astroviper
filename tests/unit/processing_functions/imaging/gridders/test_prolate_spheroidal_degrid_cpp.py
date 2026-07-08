"""Unit tests for the C++ prolate_spheroidal_degrid extension.

Coverage:
  * Numerical agreement with the Numba reference (`prolate_spheroidal_degrid_jit`)
  * Serial vs multi-threaded equivalence (exact: no atomic collisions)
  * `num_threads <= 0` auto-detects hardware concurrency
  * NaN UV / NaN vis / out-of-bounds samples are left unchanged
  * Channel / time / pol index maps (cube and continuum collapse)
  * Dirac-delta grid returns the expected separable-kernel peak
  * Binding input validation (wrong ndim)
"""

import unittest

import numpy as np

from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_degrid import (
    prolate_spheroidal_degrid_jit,
)
from astroviper.processing_functions.imaging.gridders.prolate_spheroidal_grid_cpp import (
    prolate_spheroidal_degrid,
)
from astroviper.processing_functions.imaging.gridding_convolution_functions.gcf_prolate_spheroidal import (
    create_prolate_spheroidal_kernel_1D,
)

SUPPORT = 7
OVERSAMPLING = 100


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
    """Build a self-consistent batch of inputs for the degridder.

    The grid is filled with random complex values so the degrid output is a
    non-trivial weighted average of neighbouring cells.
    """
    rng = np.random.default_rng(seed)
    m_time = n_time if m_time is None else m_time
    m_chan = n_vis_chan if m_chan is None else m_chan
    m_pol = n_pol if m_pol is None else m_pol

    grid = (
        rng.standard_normal((m_time, m_chan, m_pol, m_u, m_v))
        + 1j * rng.standard_normal((m_time, m_chan, m_pol, m_u, m_v))
    ).astype(np.complex128)
    # Starting vis_data is zeros (non-NaN) so every sample is processed.
    vis_data = np.zeros((n_time, n_baseline, n_vis_chan, n_pol), dtype=np.complex128)
    uvw = rng.uniform(-uv_extent, uv_extent, size=(n_time, n_baseline, 3))
    freq = np.linspace(1.0e9, 1.5e9, n_vis_chan)
    fmap = np.arange(n_vis_chan, dtype=np.int64)
    tmap = np.arange(n_time, dtype=np.int64) % m_time
    pmap = np.arange(n_pol, dtype=np.int64) % m_pol
    cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT).astype(np.float64)
    n_uv = np.array([m_u, m_v], dtype=np.int64)
    delta_lm = np.array([2.0e-4, 2.0e-4], dtype=np.float64)

    return dict(
        grid=grid,
        vis_data=vis_data,
        uvw=uvw,
        frequency_coord=freq,
        frequency_map=fmap,
        time_map=tmap,
        pol_map=pmap,
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
    vis_data = inputs["vis_data"].copy()
    prolate_spheroidal_degrid(
        inputs["grid"],
        vis_data,
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
        num_threads=num_threads,
    )
    return vis_data


def _run_jit(inputs):
    vis_data = inputs["vis_data"].copy()
    prolate_spheroidal_degrid_jit(
        inputs["grid"],
        vis_data,
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
    )
    return vis_data


class TestProlateSpheroidalDegridCpp(unittest.TestCase):
    # ------------------------------------------------------------------
    # Reference-oracle tests
    # ------------------------------------------------------------------
    def test_matches_numba_reference(self):
        """Serial C++ degridder matches the numba reference bit-for-bit."""
        inputs = _make_random_inputs(seed=1)
        cpp = _run_cpp(inputs, num_threads=1)
        ref = _run_jit(inputs)
        np.testing.assert_allclose(cpp, ref, rtol=0, atol=1e-12)
        # Ensure the test exercises the interpolation path — a non-trivial
        # number of samples should have been written.
        self.assertGreater(np.count_nonzero(cpp), 0)

    def test_continuum_collapse(self):
        """frequency_map collapsing all channels onto image chan 0."""
        inputs = _make_random_inputs(n_vis_chan=4, m_chan=1, seed=2)
        inputs["frequency_map"] = np.zeros(4, dtype=np.int64)
        cpp = _run_cpp(inputs, num_threads=1)
        ref = _run_jit(inputs)
        np.testing.assert_allclose(cpp, ref, rtol=0, atol=1e-12)

    def test_time_map_collapse(self):
        """time_map collapsing multiple time steps onto image time 0."""
        inputs = _make_random_inputs(n_time=3, m_time=1, seed=3)
        inputs["time_map"] = np.zeros(3, dtype=np.int64)
        cpp = _run_cpp(inputs, num_threads=1)
        ref = _run_jit(inputs)
        np.testing.assert_allclose(cpp, ref, rtol=0, atol=1e-12)

    def test_pol_map_reorder(self):
        """pol_map reordering swaps which grid pol feeds each vis pol."""
        inputs = _make_random_inputs(n_pol=2, seed=4)
        inputs["pol_map"] = np.array([1, 0], dtype=np.int64)
        cpp = _run_cpp(inputs, num_threads=1)
        ref = _run_jit(inputs)
        np.testing.assert_allclose(cpp, ref, rtol=0, atol=1e-12)

    # ------------------------------------------------------------------
    # Threading
    # ------------------------------------------------------------------
    def test_serial_vs_threaded_equivalent(self):
        """Threaded degrid matches serial bit-for-bit.

        Degrid partitions work by (i_time, i_baseline) so each worker writes
        to a disjoint slice of vis_data — no atomic collisions, no reordering
        of floating-point sums: the result is bit-identical.
        """
        inputs = _make_random_inputs(n_baseline=256, seed=5)
        serial = _run_cpp(inputs, num_threads=1)
        for nt in (2, 4, 8):
            with self.subTest(num_threads=nt):
                threaded = _run_cpp(inputs, num_threads=nt)
                np.testing.assert_array_equal(threaded, serial)

    def test_num_threads_auto_detect(self):
        """num_threads <= 0 falls back to hardware_concurrency()."""
        inputs = _make_random_inputs(seed=6)
        serial = _run_cpp(inputs, num_threads=1)
        auto = _run_cpp(inputs, num_threads=0)
        np.testing.assert_array_equal(auto, serial)

    def test_threaded_matches_reference(self):
        """Threaded C++ output matches the numba reference within tolerance."""
        inputs = _make_random_inputs(n_baseline=200, seed=7)
        cpp = _run_cpp(inputs, num_threads=4)
        ref = _run_jit(inputs)
        np.testing.assert_allclose(cpp, ref, rtol=0, atol=1e-12)

    # ------------------------------------------------------------------
    # Skip / guard behaviours
    # ------------------------------------------------------------------
    def test_nan_uvw_unchanged(self):
        """Samples whose UVW is NaN must be left exactly as on input."""
        inputs = _make_random_inputs(n_baseline=16, seed=9)
        # Seed a distinctive sentinel value into vis_data for the NaN row.
        sentinel = 7.0 + 3.0j
        inputs["vis_data"] = np.full_like(inputs["vis_data"], sentinel)
        inputs["uvw"][:, 0, :] = np.nan  # baseline 0 is fully NaN

        cpp = _run_cpp(inputs, num_threads=1)
        ref = _run_jit(inputs)
        np.testing.assert_allclose(cpp, ref, rtol=0, atol=1e-12)
        # Baseline 0 stays untouched across every time/chan/pol.
        self.assertTrue(np.all(cpp[:, 0, :, :] == sentinel))
        # Other baselines actually moved off the sentinel.
        self.assertTrue(np.any(cpp[:, 1:, :, :] != sentinel))

    def test_nan_vis_unchanged(self):
        """A NaN sample in vis_data must be left as NaN (skipped)."""
        inputs = _make_random_inputs(n_baseline=8, seed=10)
        inputs["vis_data"][0, 0, 0, 0] = complex(np.nan, np.nan)

        cpp = _run_cpp(inputs, num_threads=1)
        ref = _run_jit(inputs)

        self.assertTrue(np.isnan(cpp[0, 0, 0, 0].real))
        self.assertTrue(np.isnan(cpp[0, 0, 0, 0].imag))
        # Everywhere else (non-NaN input) should match the reference exactly.
        mask = np.ones_like(cpp, dtype=bool)
        mask[0, 0, 0, 0] = False
        np.testing.assert_allclose(cpp[mask], ref[mask], rtol=0, atol=1e-12)

    def test_out_of_bounds_sample_unchanged(self):
        """A sample whose support falls outside the grid is left as-is."""
        inputs = _make_random_inputs(n_baseline=1, m_u=64, m_v=64, seed=11)
        sentinel = -5.0 - 2.0j
        inputs["vis_data"][:] = sentinel
        inputs["uvw"][:] = 1.0e15  # far outside the grid

        cpp = _run_cpp(inputs, num_threads=1)
        self.assertTrue(np.all(cpp == sentinel))

    # ------------------------------------------------------------------
    # Round-trip: a Dirac-delta in the grid returns the kernel peak at uvw=0
    # ------------------------------------------------------------------
    def test_dirac_grid_at_origin(self):
        """A unit impulse in the grid centre is sampled as (kernel[0]**2)/sum(kernel).

        At uvw=(0,0,0) the sample lands exactly on the grid centre. The
        separable kernel produces `conv = cgk[|i_u|*ov] * cgk[|i_v|*ov]` over
        the support, so the degrid output is
           sum(conv * grid[center]) / sum(conv)
        When only the centre cell is non-zero, only the (i_u=0, i_v=0) term
        of the numerator survives, giving cgk[0]**2 / sum(conv).
        """
        m_u = m_v = 128
        grid = np.zeros((1, 1, 1, m_u, m_v), dtype=np.complex128)
        grid[0, 0, 0, m_u // 2, m_v // 2] = 1.0 + 0.0j

        vis_data = np.zeros((1, 1, 1, 1), dtype=np.complex128)
        uvw = np.zeros((1, 1, 3), dtype=np.float64)
        freq = np.array([1.0e9], dtype=np.float64)
        fmap = np.array([0], dtype=np.int64)
        tmap = np.array([0], dtype=np.int64)
        pmap = np.array([0], dtype=np.int64)
        cgk = create_prolate_spheroidal_kernel_1D(OVERSAMPLING, SUPPORT).astype(
            np.float64
        )
        n_uv = np.array([m_u, m_v], dtype=np.int64)
        delta_lm = np.array([1.0e-4, 1.0e-4], dtype=np.float64)

        prolate_spheroidal_degrid(
            grid,
            vis_data,
            uvw,
            freq,
            fmap,
            tmap,
            pmap,
            cgk,
            n_uv,
            delta_lm,
            support=SUPPORT,
            oversampling=OVERSAMPLING,
            num_threads=1,
        )

        s = SUPPORT // 2
        offsets = np.arange(-s, s + 1)
        kernel_1d = cgk[np.abs(offsets) * OVERSAMPLING]
        conv_sum = float(np.outer(kernel_1d, kernel_1d).sum())
        expected = (kernel_1d[s] ** 2) / conv_sum

        self.assertAlmostEqual(vis_data[0, 0, 0, 0].real, expected, places=12)
        self.assertAlmostEqual(vis_data[0, 0, 0, 0].imag, 0.0, places=12)

    # ------------------------------------------------------------------
    # In-place / zero-copy behaviour (regression: complex64 vis_data was
    # silently safe-cast to complex128 by pybind11, so the Python buffer
    # stayed zero. The binding now takes an untyped py::array and writes
    # directly into the caller's buffer for both complex64 and complex128.)
    # ------------------------------------------------------------------
    def test_inplace_complex128_no_copy(self):
        """complex128 vis_data is modified in place; numpy buffer is not reallocated."""
        inputs = _make_random_inputs(seed=20)
        vis = inputs["vis_data"]  # complex128, owned by Python
        self.assertEqual(vis.dtype, np.complex128)
        addr_before = vis.ctypes.data
        prolate_spheroidal_degrid(
            inputs["grid"],
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
        # Same buffer (no reallocation, no silent copy).
        self.assertEqual(vis.ctypes.data, addr_before)
        # Something was actually written.
        self.assertGreater(np.count_nonzero(vis), 0)

    def test_inplace_complex64_no_copy(self):
        """complex64 vis_data is modified in place; numpy buffer is not reallocated.

        Regression: pybind11's ``py::array_t<complex128>`` previously safe-cast
        a complex64 input into a temporary complex128 copy, so the C++ kernel
        wrote into the copy and the Python-owned array stayed zero. The
        binding now dispatches on dtype and writes through the original
        buffer.
        """
        inputs = _make_random_inputs(seed=21)
        vis64 = np.zeros_like(inputs["vis_data"], dtype=np.complex64)
        addr_before = vis64.ctypes.data
        prolate_spheroidal_degrid(
            inputs["grid"],
            vis64,
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
        self.assertEqual(vis64.ctypes.data, addr_before)
        self.assertEqual(vis64.dtype, np.complex64)
        self.assertGreater(np.count_nonzero(vis64), 0)

    def test_complex64_matches_numba_reference(self):
        """C++ output into a complex64 buffer matches the numba reference within float32 precision."""
        inputs = _make_random_inputs(seed=22)
        vis64 = np.zeros_like(inputs["vis_data"], dtype=np.complex64)
        prolate_spheroidal_degrid(
            inputs["grid"],
            vis64,
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
        ref = _run_jit(inputs)  # complex128 numba reference
        np.testing.assert_allclose(
            vis64, ref.astype(np.complex64), rtol=1e-5, atol=1e-6
        )

    def test_bad_dtype_raises_no_silent_copy(self):
        """Unsupported vis_data dtype must raise rather than silently allocate a copy."""
        inputs = _make_random_inputs(seed=23)
        bad_vis = np.zeros_like(
            inputs["vis_data"], dtype=np.float64
        )  # real, not complex
        with self.assertRaises(RuntimeError):
            prolate_spheroidal_degrid(
                inputs["grid"],
                bad_vis,
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

    def test_jit_and_cpp_produce_same_result(self):
        """End-to-end equivalence: jit and C++ produce the same vis_data for the
        same inputs across multiple seeds, baseline counts, channel maps, and
        thread counts."""
        configs = [
            dict(seed=30, n_baseline=32, n_vis_chan=2, n_pol=2, num_threads=1),
            dict(seed=31, n_baseline=128, n_vis_chan=4, n_pol=2, num_threads=4),
            dict(seed=32, n_baseline=64, n_vis_chan=3, n_pol=1, num_threads=2),
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                num_threads = cfg.pop("num_threads")
                inputs = _make_random_inputs(**cfg)
                cpp = _run_cpp(inputs, num_threads=num_threads)
                jit = _run_jit(inputs)
                np.testing.assert_allclose(cpp, jit, rtol=0, atol=1e-12)

    # ------------------------------------------------------------------
    # Binding input validation
    # ------------------------------------------------------------------
    def test_wrong_ndim_raises(self):
        inputs = _make_random_inputs(seed=12)
        bad_grid = np.zeros(
            (inputs["m_chan"], inputs["m_pol"], inputs["m_u"], inputs["m_v"]),
            dtype=np.complex128,
        )  # 4-D, not 5-D
        with self.assertRaises(RuntimeError):
            prolate_spheroidal_degrid(
                bad_grid,
                inputs["vis_data"].copy(),
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


if __name__ == "__main__":
    unittest.main()
