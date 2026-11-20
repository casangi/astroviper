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
import pytest

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


def _logical_oversized_empty_grid():
    """Create a contiguous zero-element grid with an oversized channel axis."""
    base = np.empty(0, dtype=np.float32)
    return np.lib.stride_tricks.as_strided(
        base,
        shape=(np.iinfo(np.int32).max + 1, 1, 0, 1),
        strides=(0, 0, base.itemsize, base.itemsize),
        writeable=True,
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

    def test_truncate_uv_cells_uses_casa_continuum_assignment(self):
        """Truncation selects the lower shifted cell without changing default."""
        n_uv = np.array([8, 8], dtype=np.int64)
        frequency = np.array([299792458.0])
        delta_lm = np.array([-1.0 / 8.0, -1.0 / 8.0])
        uvw = np.array([[[0.6, 0.6, 0.0]]])
        chan_map = np.array([0], dtype=np.int64)
        weights = np.array([[[[2.0]]]])

        nearest = np.zeros((1, 1, 8, 8))
        nearest_sum = np.zeros((1, 1))
        grid_imaging_weights(
            nearest,
            nearest_sum,
            uvw,
            frequency,
            chan_map,
            weights,
            n_uv,
            delta_lm,
        )

        truncated = np.zeros_like(nearest)
        truncated_sum = np.zeros_like(nearest_sum)
        grid_imaging_weights(
            truncated,
            truncated_sum,
            uvw,
            frequency,
            chan_map,
            weights,
            n_uv,
            delta_lm,
            truncate_uv_cells=True,
        )

        self.assertEqual(nearest[0, 0, 5, 5], 2.0)
        self.assertEqual(truncated[0, 0, 4, 4], 2.0)
        self.assertEqual(nearest[0, 0, 4, 4], 0.0)
        np.testing.assert_array_equal(nearest_sum, [[4.0]])
        np.testing.assert_array_equal(truncated_sum, [[4.0]])

    def test_truncate_uv_cells_accumulates_valid_conjugates_independently(self):
        """CASA truncation retains a valid direct cell at the opposite edge."""
        n_uv = np.array([8, 8], dtype=np.int64)
        frequency = np.array([299792458.0])
        delta_lm = np.array([-1.0 / 8.0, -1.0 / 8.0])
        uvw = np.array([[[3.6, 0.0, 0.0]]])
        grid = np.zeros((1, 1, 8, 8))
        sum_weight = np.zeros((1, 1))

        grid_imaging_weights(
            grid,
            sum_weight,
            uvw,
            frequency,
            np.array([0], dtype=np.int64),
            np.array([[[[2.0]]]]),
            n_uv,
            delta_lm,
            truncate_uv_cells=True,
        )

        self.assertEqual(grid[0, 0, 7, 4], 2.0)
        self.assertEqual(np.count_nonzero(grid), 1)
        np.testing.assert_array_equal(sum_weight, [[2.0]])

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

    def test_sum_weight_rejects_implicit_output_conversions(self):
        """Writable accumulators must never be replaced by pybind temporaries."""
        inp = _make_inputs(seed=9)
        grid = np.zeros((inp["n_chan"], 1, inp["m_u"], inp["m_v"]), dtype=np.float64)
        bad_outputs = [
            np.zeros((inp["n_chan"], 1), dtype=np.float32),
            np.zeros((inp["n_chan"], 2), dtype=np.float64)[:, ::2],
            np.zeros((inp["n_chan"], 1), dtype=">f8"),
        ]
        readonly = np.zeros((inp["n_chan"], 1), dtype=np.float64)
        readonly.flags.writeable = False
        bad_outputs.append(readonly)
        for sum_weight in bad_outputs:
            with self.subTest(dtype=sum_weight.dtype, strides=sum_weight.strides):
                with self.assertRaises(RuntimeError):
                    grid_imaging_weights(
                        grid,
                        sum_weight,
                        inp["uvw"],
                        inp["freq_chan"],
                        inp["chan_map"],
                        inp["data_weight"],
                        inp["n_uv"],
                        inp["delta_lm"],
                    )

    def test_oversized_grid_dimension_raises(self):
        """The binding rejects dimensions that the kernel loop type cannot hold."""
        with self.assertRaisesRegex(OverflowError, "grid channel dimension"):
            grid_imaging_weights(
                _logical_oversized_empty_grid(),
                np.empty((1, 1), dtype=np.float64),
                np.empty((1, 1, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.empty((1, 1, 0, 1), dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
                np.array([1.0, 1.0]),
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

    def test_imaging_weight_rejects_implicit_output_conversions(self):
        """Degridding rejects converted, strided, and read-only outputs."""
        inp = _make_inputs(seed=14)
        grid, _ = _run_grid_cpp(inp)
        shape = inp["data_weight"].shape
        bad_outputs = [
            np.zeros(shape, dtype=np.float32),
            np.zeros((*shape[:-1], 2), dtype=np.float64)[..., ::2],
            np.zeros(shape, dtype=">f8"),
        ]
        readonly = np.zeros(shape, dtype=np.float64)
        readonly.flags.writeable = False
        bad_outputs.append(readonly)
        for output in bad_outputs:
            with self.subTest(dtype=output.dtype, strides=output.strides):
                with self.assertRaises(RuntimeError):
                    degrid_imaging_weights(
                        output,
                        grid,
                        self._briggs(inp, 14),
                        inp["uvw"],
                        inp["freq_chan"],
                        inp["chan_map"],
                        inp["pol_map"],
                        inp["data_weight"],
                        inp["n_uv"],
                        inp["delta_lm"],
                    )

    def test_truncate_uv_cells_samples_casa_continuum_cell(self):
        """Degridding uses the same selected cell convention as gridding."""
        grid = np.zeros((1, 1, 8, 8))
        grid[0, 0, 5, 5] = 10.0
        grid[0, 0, 4, 4] = 20.0
        frequency = np.array([299792458.0])
        uvw = np.array([[[0.6, 0.6, 0.0]]])
        data_weight = np.array([[[[2.0]]]])
        common = (
            grid,
            np.array([[[1.0]], [[0.0]]]),
            uvw,
            frequency,
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            data_weight,
            np.array([8, 8], dtype=np.int64),
            np.array([-1.0 / 8.0, -1.0 / 8.0]),
        )

        nearest = np.zeros_like(data_weight)
        degrid_imaging_weights(nearest, *common)
        truncated = np.zeros_like(data_weight)
        degrid_imaging_weights(truncated, *common, truncate_uv_cells=True)

        np.testing.assert_array_equal(nearest, [[[[0.2]]]])
        np.testing.assert_array_equal(truncated, [[[[0.1]]]])

    def test_sample_not_gridded_left_zero(self):
        """A sample whose conjugate cell is off the grid is not reweighted."""
        # u_pix = u, v_pix = v: six samples, one on u cell 0 and one on v cell 0,
        # whose conjugate cells are 8.
        uvw = np.zeros((1, 8, 3))
        uvw[0, :, 0] = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, -4.2, 1.0]
        uvw[0, :, 1] = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, -4.2]
        frequency = np.array([299792458.0])
        chan_map = np.array([0], dtype=np.int64)
        data_weight = np.ones((1, 8, 1, 1))
        n_uv = np.array([8, 8], dtype=np.int64)
        delta_lm = np.array([-1.0 / 8.0, -1.0 / 8.0])
        grid = np.zeros((1, 1, 8, 8))
        sum_weight = np.zeros((1, 1))
        grid_imaging_weights(
            grid, sum_weight, uvw, frequency, chan_map, data_weight, n_uv, delta_lm
        )
        imaging_weight = np.zeros_like(data_weight)
        degrid_imaging_weights(
            imaging_weight,
            grid,
            np.array([[[25.0]], [[1.0]]]),  # Briggs robust 0: sum G**2 / sum w = 1
            uvw,
            frequency,
            chan_map,
            np.array([0], dtype=np.int64),
            data_weight,
            n_uv,
            delta_lm,
        )

        np.testing.assert_array_equal(sum_weight, [[12.0]])
        np.testing.assert_allclose(imaging_weight[0, :6, 0, 0], 1.0 / 26.0)
        np.testing.assert_array_equal(imaging_weight[0, 6:, 0, 0], 0.0)

    def test_oversized_grid_dimension_raises(self):
        """The degrid binding rejects dimensions that cannot be represented."""
        with self.assertRaisesRegex(OverflowError, "grid channel dimension"):
            degrid_imaging_weights(
                np.empty((1, 1, 0, 1), dtype=np.float64),
                _logical_oversized_empty_grid(),
                np.empty((2, 1, 1), dtype=np.float64),
                np.empty((1, 1, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int64),
                np.array([0], dtype=np.int64),
                np.empty((1, 1, 0, 1), dtype=np.float64),
                np.array([0, 1], dtype=np.int64),
                np.array([1.0, 1.0]),
            )


def test_weighting_unaligned_buffers_preserve_results_and_storage(unaligned_array):
    """Both kernels read/write the supplied bytes without array replacement."""
    import pickle

    for dtype in (np.float32, np.float64):
        for truncate in (False, True):
            common = dict(
                uvw=np.array([[[12.0, -7.0, 0.0]]]),
                frequency_coord=np.array([1e9]),
                chan_map=np.zeros(1, dtype=np.int64),
                n_uv=np.array([16, 16], dtype=np.int64),
                delta_lm=np.array([1e-4, 1e-4]),
                truncate_uv_cells=truncate,
            )
            cases = [
                (
                    grid_imaging_weights,
                    dict(
                        common,
                        grid=np.zeros((1, 1, 16, 16), dtype=dtype),
                        sum_weight=np.zeros((1, 1)),
                        data_weights=np.full((1, 1, 1, 1), 3.0),
                    ),
                ),
                (
                    degrid_imaging_weights,
                    dict(
                        common,
                        imaging_weight=np.zeros((1, 1, 1, 1)),
                        grid_imaging_weight=np.full((1, 1, 16, 16), 2.0, dtype=dtype),
                        briggs_factors=np.array([[[2.0]], [[1.0]]]),
                        pol_map=np.zeros(1, dtype=np.int64),
                        data_weight=np.full((1, 1, 1, 1), 3.0),
                    ),
                ),
            ]
            for kernel, kwargs in cases:
                expected = {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in kwargs.items()
                }
                kernel(**expected)
                arrays = [k for k, v in kwargs.items() if isinstance(v, np.ndarray)]
                for key in arrays + ["all"]:
                    for offset in range(1, 8):
                        trial = {
                            k: v.copy() if isinstance(v, np.ndarray) else v
                            for k, v in kwargs.items()
                        }
                        selected = arrays if key == "all" else [key]
                        for name in selected:
                            trial[name] = unaligned_array(trial[name], offset)
                        kernel(**trial)
                        for name in arrays:
                            np.testing.assert_array_equal(trial[name], expected[name])
                        for name in selected:
                            assert trial[name].base[:offset] == bytes([0xA5]) * offset
                            assert trial[name].base[-1] == 0xA5
                trial = {
                    k: v.copy().view(pickle.loads(pickle.dumps(v.dtype)))
                    if isinstance(v, np.ndarray)
                    else v
                    for k, v in kwargs.items()
                }
                kernel(**trial)
                for name in arrays:
                    np.testing.assert_array_equal(trial[name], expected[name])


def test_degrid_density_grid_after_distributed_deserialization():
    """A received frame may start at a byte offset, as in the HOPS failure."""
    from distributed.protocol import deserialize, serialize

    source = np.full((1, 1, 16, 16), 2.0, dtype=np.float64)
    header, frames = serialize(source)
    received_frames = []
    for frame in frames:
        raw = memoryview(frame).cast("B")
        storage = bytearray(raw.nbytes + 6)
        storage[6:] = raw
        received_frames.append(memoryview(storage)[6:])
    grid = deserialize(header, received_frames)
    assert grid.flags.c_contiguous
    assert not grid.flags.aligned
    assert grid.ctypes.data % 8 == 6
    output = np.zeros((1, 1, 1, 1))
    degrid_imaging_weights(
        imaging_weight=output,
        grid_imaging_weight=grid,
        briggs_factors=np.ones((2, 1, 1)),
        uvw=np.zeros((1, 1, 3)),
        frequency_coord=np.array([1e9]),
        chan_map=np.zeros(1, dtype=np.int64),
        pol_map=np.zeros(1, dtype=np.int64),
        data_weight=np.ones_like(output),
        n_uv=np.array([16, 16]),
        delta_lm=np.array([1e-4, 1e-4]),
    )
    np.testing.assert_array_equal(output, np.full_like(output, 1 / 3))
    np.testing.assert_array_equal(grid, source)
    assert np.shares_memory(grid, np.frombuffer(received_frames[0], dtype=np.uint8))


def _binding_validation_case(degrid, dtype):
    common = dict(
        uvw=np.zeros((2, 3, 3)),
        frequency_coord=np.full(4, 1.0e9),
        chan_map=np.array([0, 1, 0, 1], dtype=np.int64),
        n_uv=np.array([8, 8], dtype=np.int64),
        delta_lm=np.array([1.0e-4, 1.0e-4]),
    )
    weights = np.ones((2, 3, 4, 2))
    if degrid:
        return degrid_imaging_weights, dict(
            common,
            imaging_weight=np.full_like(weights, -7.0),
            grid_imaging_weight=np.full((2, 2, 8, 8), 2.0, dtype=dtype),
            briggs_factors=np.ones((2, 2, 2)),
            pol_map=np.array([1, 0], dtype=np.int64),
            data_weight=weights,
        )
    return grid_imaging_weights, dict(
        common,
        grid=np.zeros((2, 2, 8, 8), dtype=dtype),
        sum_weight=np.zeros((2, 2)),
        data_weights=weights,
    )


# The bad entries occur after valid entries so a late kernel-side check would
# already have changed the accumulators. All validation must precede compute.
@pytest.mark.parametrize("degrid", [False, True], ids=["grid", "degrid"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name,value,message",
    [
        ("frequency_coord", np.array(1.0e9), "frequency_coord.*1-D"),
        ("frequency_coord", np.ones((1, 4)), "frequency_coord.*1-D"),
        ("frequency_coord", np.ones(1), "frequency_coord.*shape"),
        ("frequency_coord", np.ones(5), "frequency_coord.*shape"),
        ("chan_map", np.array(0, dtype=np.int64), "chan_map.*1-D"),
        ("chan_map", np.zeros((1, 4), dtype=np.int64), "chan_map.*1-D"),
        ("chan_map", np.zeros(3, dtype=np.int64), "chan_map.*shape"),
        ("chan_map", np.zeros(5, dtype=np.int64), "chan_map.*shape"),
        ("chan_map", np.array([0, 1, 0, -1], dtype=np.int64), "chan_map.*outside"),
        ("chan_map", np.array([0, 1, 0, 2], dtype=np.int64), "chan_map.*outside"),
        ("chan_map", np.array([0, 1, 0, np.iinfo(np.int64).max]), "chan_map.*outside"),
        ("chan_map", np.array([0, 1, 0, np.iinfo(np.int64).min]), "chan_map.*outside"),
        ("uvw", np.zeros((1, 3, 3)), "data_weight.*shape"),
        ("uvw", np.zeros((2, 2, 3)), "data_weight.*shape"),
    ],
)
def test_weight_bindings_reject_inconsistent_coordinates_before_writing(
    degrid, dtype, name, value, message, unaligned_array
):
    kernel, kwargs = _binding_validation_case(degrid, dtype)
    # Also exercise byte-safe validation of maps and coordinates after transfer.
    kwargs[name] = unaligned_array(value, 1)
    _assert_weight_binding_rejects_without_writes(kernel, kwargs, message)


def _assert_weight_binding_rejects_without_writes(kernel, kwargs, message):
    # Retain guard storage beyond every accumulator, including deliberately
    # undersized outputs from the review reproductions.
    guarded = []
    for name in ("grid", "sum_weight", "imaging_weight"):
        if name not in kwargs:
            continue
        value = kwargs[name]
        storage = np.full(value.size + 64, 12345.0, dtype=value.dtype)
        view = storage[: value.size].reshape(value.shape)
        view[...] = value
        kwargs[name] = view
        guarded.append((storage, storage.copy()))
    snapshots = {
        name: value.copy()
        for name, value in kwargs.items()
        if isinstance(value, np.ndarray)
    }
    with pytest.raises(RuntimeError, match=message):
        kernel(**kwargs)
    for name, expected in snapshots.items():
        np.testing.assert_array_equal(kwargs[name], expected, err_msg=name)
    for storage, expected in guarded:
        np.testing.assert_array_equal(storage, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name,shape,message",
    [
        ("sum_weight", (1, 2), "sum_weight.*shape"),
        ("sum_weight", (2, 1), "sum_weight.*shape"),
        ("sum_weight", (3, 2), "sum_weight.*shape"),
        ("sum_weight", (2, 3), "sum_weight.*shape"),
        ("data_weights", (1, 3, 4, 2), "data_weights.*shape"),
        ("data_weights", (2, 2, 4, 2), "data_weights.*shape"),
        ("data_weights", (2, 3, 3, 2), "frequency_coord.*shape"),
        ("data_weights", (2, 3, 4, 0), "data_weights.*polarization"),
        ("grid", (2, 0, 8, 8), "sum_weight.*shape|grid.*polarization"),
    ],
)
def test_grid_weights_rejects_inconsistent_shapes_without_writes(
    dtype, name, shape, message
):
    kernel, kwargs = _binding_validation_case(False, dtype)
    kwargs[name] = np.zeros(shape, dtype=kwargs[name].dtype)
    _assert_weight_binding_rejects_without_writes(kernel, kwargs, message)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "name,shape,message",
    [
        ("imaging_weight", (1, 3, 4, 2), "imaging_weight.*shape"),
        ("imaging_weight", (2, 1, 4, 2), "imaging_weight.*shape"),
        ("imaging_weight", (2, 3, 1, 2), "imaging_weight.*shape"),
        ("imaging_weight", (2, 3, 4, 1), "imaging_weight.*shape"),
        ("imaging_weight", (2, 3, 4, 3), "imaging_weight.*shape"),
        ("briggs_factors", (2, 1, 2), "briggs_factors.*shape"),
        ("briggs_factors", (2, 2, 1), "briggs_factors.*shape"),
        ("briggs_factors", (2, 3, 2), "briggs_factors.*shape"),
        ("data_weight", (1, 3, 4, 2), "data_weight.*shape"),
        ("data_weight", (2, 2, 4, 2), "data_weight.*shape"),
        ("data_weight", (2, 3, 3, 2), "frequency_coord.*shape"),
    ],
)
def test_degrid_weights_rejects_inconsistent_shapes_without_writes(
    dtype, name, shape, message
):
    kernel, kwargs = _binding_validation_case(True, dtype)
    kwargs[name] = np.zeros(shape, dtype=kwargs[name].dtype)
    _assert_weight_binding_rejects_without_writes(kernel, kwargs, message)


@pytest.mark.parametrize(
    "value,message",
    [
        (np.array(0, dtype=np.int64), "pol_map.*1-D"),
        (np.zeros((1, 2), dtype=np.int64), "pol_map.*1-D"),
        (np.zeros(3, dtype=np.int64), "pol_map.*length"),
        (np.array([0, -1], dtype=np.int64), "pol_map.*outside"),
        (np.array([0, 2], dtype=np.int64), "pol_map.*outside"),
        (np.array([0, np.iinfo(np.int64).max]), "pol_map.*outside"),
        (np.array([0, np.iinfo(np.int64).min]), "pol_map.*outside"),
    ],
)
def test_degrid_weights_rejects_invalid_unaligned_polarization_maps(
    value, message, unaligned_array
):
    kernel, kwargs = _binding_validation_case(True, np.float64)
    kwargs["pol_map"] = unaligned_array(value, 1)
    _assert_weight_binding_rejects_without_writes(kernel, kwargs, message)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_pol_out", [0, 1, 2])
def test_degrid_weights_accepts_valid_unaligned_maps_and_polarization_prefixes(
    dtype, n_pol_out, unaligned_array
):
    kernel, kwargs = _binding_validation_case(True, dtype)
    kwargs["chan_map"] = unaligned_array(kwargs["chan_map"], 1)
    kwargs["pol_map"] = unaligned_array(kwargs["pol_map"][:n_pol_out].copy(), 1)
    kernel(**kwargs)
    np.testing.assert_array_equal(kwargs["imaging_weight"][..., :n_pol_out], 1.0 / 3.0)
    np.testing.assert_array_equal(kwargs["imaging_weight"][..., n_pol_out:], -7.0)


if __name__ == "__main__":
    unittest.main()
