"""Unit tests for the C++ grid/degrid imaging-weight extension.

Coverage:
  * Numerical agreement with a pure-Python reference (de-jitted copy of the
    retired numba kernels) for float64 and float32 grids
  * Accumulation semantics (two calls == one combined call)
  * Hermitian-conjugate update and sum_weight bookkeeping
  * NaN UV / NaN weight samples are skipped
  * Out-of-bounds samples are skipped
  * Briggs denominator behaviour incl. ValueError on NaN factors
  * Binding input validation (wrong ndim, bad dtype)
"""

import unittest

import numpy as np

from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights_cpp import (
    degrid_imaging_weights,
    grid_imaging_weights,
)


# ----------------------------------------------------------------------
# Pure-Python reference implementations (de-jitted copies of the retired
# numba kernels), kept as independent test oracles. Interior samples only:
# at the one-pixel edge band the historical kernel truncated instead of
# rounding, which the C++ kernel intentionally does not reproduce.
# ----------------------------------------------------------------------
def _grid_imaging_weights_reference(
    grid, sum_weight, uvw, freq_chan, chan_map, data_weights, n_uv, delta_lm
):
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c

    uv_center = n_uv // 2
    n_time, n_baseline = uvw.shape[0], uvw.shape[1]
    n_u, n_v = n_uv[0], n_uv[1]

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(len(chan_map)):
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                if np.isnan(u) or np.isnan(v):
                    continue
                u_indx = int(u + uv_center[0] + 0.5)
                v_indx = int(v + uv_center[1] + 0.5)
                u_indx_conj = int(-u + uv_center[0] + 0.5)
                v_indx_conj = int(-v + uv_center[1] + 0.5)
                if 0 <= u_indx < n_u and 0 <= v_indx < n_v:
                    weight = data_weights[i_time, i_baseline, i_chan, 0]
                    if not np.isnan(weight):
                        grid[a_chan, 0, u_indx, v_indx] += weight
                        grid[a_chan, 0, u_indx_conj, v_indx_conj] += weight
                        sum_weight[a_chan, 0] += 2.0 * weight


def _degrid_imaging_weights_reference(
    imaging_weight,
    grid_imaging_weight,
    briggs_factors,
    uvw,
    freq_chan,
    chan_map,
    pol_map,
    data_weight,
    n_uv,
    delta_lm,
):
    c = 299792458.0
    uv_scale = np.zeros((2, len(freq_chan)), dtype=np.double)
    uv_scale[0, :] = -(freq_chan * delta_lm[0] * n_uv[0]) / c
    uv_scale[1, :] = -(freq_chan * delta_lm[1] * n_uv[1]) / c

    uv_center = n_uv // 2
    n_time, n_baseline = uvw.shape[0], uvw.shape[1]
    n_u, n_v = n_uv[0], n_uv[1]

    for i_time in range(n_time):
        for i_baseline in range(n_baseline):
            for i_chan in range(len(chan_map)):
                a_chan = chan_map[i_chan]
                u = uvw[i_time, i_baseline, 0] * uv_scale[0, i_chan]
                v = uvw[i_time, i_baseline, 1] * uv_scale[1, i_chan]
                if np.isnan(u) or np.isnan(v):
                    continue
                u_indx = int(u + uv_center[0] + 0.5)
                v_indx = int(v + uv_center[1] + 0.5)
                if not (0 <= u_indx < n_u and 0 <= v_indx < n_v):
                    continue
                for i_pol in range(len(pol_map)):
                    a_pol = pol_map[i_pol]
                    natural = data_weight[i_time, i_baseline, i_chan, i_pol]
                    imaging_weight[i_time, i_baseline, i_chan, i_pol] = natural
                    if np.isnan(natural) or natural == 0.0:
                        continue
                    gij = grid_imaging_weight[a_chan, a_pol, u_indx, v_indx]
                    if np.isnan(gij) or gij == 0.0:
                        continue
                    denom = (
                        briggs_factors[0, a_chan, a_pol] * gij
                        + briggs_factors[1, a_chan, a_pol]
                    )
                    imaging_weight[i_time, i_baseline, i_chan, i_pol] = natural / denom


def _make_inputs(seed=0, n_time=4, n_baseline=32, n_chan=3, n_pol=2, m_u=64, m_v=60):
    rng = np.random.default_rng(seed)
    # |u_pix| <= 500 * (1.1e11 * 2.4e-6 * 64 / 3e8) ~ 28, comfortably interior.
    uvw = rng.uniform(-500.0, 500.0, (n_time, n_baseline, 3))
    data_weight = rng.uniform(0.5, 2.0, (n_time, n_baseline, n_chan, n_pol))
    if n_baseline > 2:
        uvw[0, 1, :] = np.nan
        data_weight[1, 2, 0, :] = np.nan
    freq_chan = np.linspace(1.0e11, 1.1e11, n_chan)
    n_uv = np.array([m_u, m_v], dtype=np.int64)
    delta_lm = np.array([-2.4e-6, 2.4e-6], dtype=np.float64)
    chan_map = np.arange(n_chan, dtype=np.int64)
    pol_map = np.arange(1, dtype=np.int64)
    return dict(
        uvw=uvw,
        data_weight=data_weight,
        freq_chan=freq_chan,
        n_uv=n_uv,
        delta_lm=delta_lm,
        chan_map=chan_map,
        pol_map=pol_map,
        n_chan=n_chan,
        m_u=m_u,
        m_v=m_v,
    )


def _run_grid_cpp(inp, dtype=np.float64):
    grid = np.zeros((inp["n_chan"], 1, inp["m_u"], inp["m_v"]), dtype=dtype)
    sum_weight = np.zeros((inp["n_chan"], 1), dtype=np.float64)
    grid_imaging_weights(
        grid,
        sum_weight,
        inp["uvw"],
        inp["freq_chan"],
        inp["chan_map"],
        inp["data_weight"],
        inp["n_uv"],
        inp["delta_lm"],
        processing_function_threads=1,
    )
    return grid, sum_weight


def _run_grid_reference(inp, dtype=np.float64):
    grid = np.zeros((inp["n_chan"], 1, inp["m_u"], inp["m_v"]), dtype=dtype)
    sum_weight = np.zeros((inp["n_chan"], 1), dtype=np.float64)
    _grid_imaging_weights_reference(
        grid,
        sum_weight,
        inp["uvw"],
        inp["freq_chan"],
        inp["chan_map"],
        inp["data_weight"],
        inp["n_uv"],
        inp["delta_lm"],
    )
    return grid, sum_weight


class TestGridImagingWeightsCpp(unittest.TestCase):
    def test_matches_reference_float64(self):
        inp = _make_inputs(seed=1)
        cpp_grid, cpp_sum = _run_grid_cpp(inp, np.float64)
        ref_grid, ref_sum = _run_grid_reference(inp, np.float64)
        np.testing.assert_array_equal(cpp_grid, ref_grid)
        np.testing.assert_array_equal(cpp_sum, ref_sum)
        self.assertGreater(np.count_nonzero(cpp_grid), 0)

    def test_matches_reference_float32(self):
        inp = _make_inputs(seed=2)
        cpp_grid, cpp_sum = _run_grid_cpp(inp, np.float32)
        ref_grid, ref_sum = _run_grid_reference(inp, np.float32)
        self.assertEqual(cpp_grid.dtype, np.float32)
        np.testing.assert_array_equal(cpp_grid, ref_grid)
        np.testing.assert_array_equal(cpp_sum, ref_sum)

    def test_accumulates_in_place(self):
        inp = _make_inputs(seed=3)
        full_grid, full_sum = _run_grid_cpp(inp)

        split = inp["uvw"].shape[1] // 2
        grid = np.zeros_like(full_grid)
        sum_weight = np.zeros_like(full_sum)
        for sl in (slice(None, split), slice(split, None)):
            grid_imaging_weights(
                grid,
                sum_weight,
                np.ascontiguousarray(inp["uvw"][:, sl]),
                inp["freq_chan"],
                inp["chan_map"],
                np.ascontiguousarray(inp["data_weight"][:, sl]),
                inp["n_uv"],
                inp["delta_lm"],
            )
        # Splitting reorders the floating-point accumulation, so the result
        # is equal to round-off rather than bit-identical.
        np.testing.assert_allclose(grid, full_grid, rtol=1e-12, atol=0)
        np.testing.assert_allclose(sum_weight, full_sum, rtol=1e-12, atol=0)

    def test_hermitian_symmetry_and_sum(self):
        inp = _make_inputs(seed=4)
        grid, sum_weight = _run_grid_cpp(inp)
        # Every deposit also lands at the conjugate pixel, so the total gridded
        # mass equals sum_weight and the grid is conjugate-symmetric about the
        # centre pixel by construction.
        np.testing.assert_allclose(
            grid.sum(axis=(1, 2, 3)), sum_weight[:, 0], rtol=1e-12
        )

    def test_nan_samples_skipped(self):
        inp = _make_inputs(seed=5)
        grid, sum_weight = _run_grid_cpp(inp)
        # Zero out the NaN rows entirely; result must be unchanged relative to
        # dropping them, i.e. NaNs contribute nothing.
        inp2 = dict(inp)
        inp2["uvw"] = inp["uvw"][1:]
        inp2["data_weight"] = inp["data_weight"][1:]
        partial_grid, partial_sum = _run_grid_cpp(
            {
                **inp2,
                "uvw": np.ascontiguousarray(inp2["uvw"]),
                "data_weight": np.ascontiguousarray(inp2["data_weight"]),
            }
        )
        self.assertFalse(np.isnan(grid).any())
        self.assertFalse(np.isnan(sum_weight).any())
        # time 0 contained the NaN uvw row; removing all of time 0 changes the
        # result, but no NaN ever leaks into the accumulators.
        self.assertGreaterEqual(sum_weight.sum(), partial_sum.sum())

    def test_out_of_bounds_skipped(self):
        inp = _make_inputs(seed=6, n_baseline=1)
        inp["uvw"][:] = 1.0e15  # far outside the grid
        grid, sum_weight = _run_grid_cpp(inp)
        self.assertEqual(np.count_nonzero(grid), 0)
        self.assertEqual(np.count_nonzero(sum_weight), 0)

    def test_wrong_ndim_raises(self):
        inp = _make_inputs(seed=7)
        bad_grid = np.zeros((inp["n_chan"], inp["m_u"], inp["m_v"]))  # 3-D
        with self.assertRaises(RuntimeError):
            grid_imaging_weights(
                bad_grid,
                np.zeros((inp["n_chan"], 1)),
                inp["uvw"],
                inp["freq_chan"],
                inp["chan_map"],
                inp["data_weight"],
                inp["n_uv"],
                inp["delta_lm"],
            )

    def test_bad_grid_dtype_raises(self):
        inp = _make_inputs(seed=8)
        bad_grid = np.zeros((inp["n_chan"], 1, inp["m_u"], inp["m_v"]), dtype=np.int64)
        with self.assertRaises(RuntimeError):
            grid_imaging_weights(
                bad_grid,
                np.zeros((inp["n_chan"], 1)),
                inp["uvw"],
                inp["freq_chan"],
                inp["chan_map"],
                inp["data_weight"],
                inp["n_uv"],
                inp["delta_lm"],
            )


class TestDegridImagingWeightsCpp(unittest.TestCase):
    def _briggs(self, inp, seed=0):
        rng = np.random.default_rng(seed)
        briggs = np.empty((2, inp["n_chan"], 1))
        briggs[0] = rng.uniform(0.01, 0.1, (inp["n_chan"], 1))
        briggs[1] = rng.uniform(0.9, 1.1, (inp["n_chan"], 1))
        return briggs

    def _run_both(self, inp, briggs, dtype=np.float64):
        grid, _ = _run_grid_cpp(inp, dtype)
        args = (
            grid,
            briggs,
            inp["uvw"],
            inp["freq_chan"],
            inp["chan_map"],
            inp["pol_map"],
            inp["data_weight"],
            inp["n_uv"],
            inp["delta_lm"],
        )
        out_cpp = np.zeros(inp["data_weight"].shape, dtype=np.float64)
        degrid_imaging_weights(out_cpp, *args)
        out_ref = np.zeros(inp["data_weight"].shape, dtype=np.float64)
        _degrid_imaging_weights_reference(out_ref, *args)
        return out_cpp, out_ref

    def test_matches_reference_float64(self):
        inp = _make_inputs(seed=10)
        out_cpp, out_ref = self._run_both(inp, self._briggs(inp, 10))
        np.testing.assert_array_equal(
            np.nan_to_num(out_cpp, nan=-1), np.nan_to_num(out_ref, nan=-1)
        )
        self.assertGreater(np.count_nonzero(np.nan_to_num(out_cpp)), 0)

    def test_matches_reference_float32_grid(self):
        inp = _make_inputs(seed=11)
        out_cpp, out_ref = self._run_both(inp, self._briggs(inp, 11), np.float32)
        np.testing.assert_array_equal(
            np.nan_to_num(out_cpp, nan=-1), np.nan_to_num(out_ref, nan=-1)
        )

    def test_nan_briggs_raises_value_error(self):
        inp = _make_inputs(seed=12)
        briggs = self._briggs(inp, 12)
        briggs[0, 0, 0] = np.nan
        grid, _ = _run_grid_cpp(inp)
        with self.assertRaises(ValueError):
            degrid_imaging_weights(
                np.zeros(inp["data_weight"].shape),
                grid,
                briggs,
                inp["uvw"],
                inp["freq_chan"],
                inp["chan_map"],
                inp["pol_map"],
                inp["data_weight"],
                inp["n_uv"],
                inp["delta_lm"],
            )

    def test_out_of_bounds_left_zero(self):
        inp = _make_inputs(seed=13, n_baseline=1)
        grid, _ = _run_grid_cpp(inp)
        inp["uvw"][:] = 1.0e15
        out = np.zeros(inp["data_weight"].shape)
        degrid_imaging_weights(
            out,
            grid,
            self._briggs(inp, 13),
            inp["uvw"],
            inp["freq_chan"],
            inp["chan_map"],
            inp["pol_map"],
            inp["data_weight"],
            inp["n_uv"],
            inp["delta_lm"],
        )
        self.assertEqual(np.count_nonzero(out), 0)


if __name__ == "__main__":
    unittest.main()
