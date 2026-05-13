"""Unit tests for ``calculate_imaging_weights`` and ``_equalize_parallel_hand_weights``.

Coverage:
  * Parallel-hand weight equalization (CASA arithmetic mean vs. formal
    error-propagation form) across the supported polarization counts.
  * Natural weighting end-to-end on a small DataTree (output shape/values,
    flag handling, multi-MS support, data-group registration, in-place
    mutation of the input weight array, overwrite guard).
  * Dispatch:
      - ``uniform`` is rewritten to ``briggs`` with ``robust = -2.0``.
      - ``casa_weighting_implementation`` defaults to ``False`` via the
        parameter checker.
  * Briggs path with the gridder/Briggs internals mocked out so the tests
    do not depend on Numba JIT or numerical gridding behaviour.
"""

import unittest
from unittest import mock

import numpy as np
import xarray as xr

from astroviper.processing_functions.imaging.calculate_imaging_weights import (
    _equalize_parallel_hand_weights,
    calculate_imaging_weights,
)
from astroviper.processing_functions.imaging.check_imaging_parameters import (
    check_imaging_weights_params,
)


_MS_DIMS = ("time", "baseline", "frequency", "polarization")
_MOD_PATH = (
    "astroviper.processing_functions.imaging.calculate_imaging_weights"
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------
def _make_ms_ds(
    n_time=1,
    n_baseline=4,
    n_chan=2,
    n_pol=2,
    weight_per_pol=None,
    flag_fill=0,
    seed=0,
):
    """Build a minimal MS-like ``xr.Dataset`` wired with a ``base`` data group.

    Parameters
    ----------
    weight_per_pol : sequence of float, optional
        If given, weights are constant along (time, baseline, chan) and
        vary per polarization. Must have length ``n_pol``. If ``None``,
        weights are drawn uniformly in ``[0.5, 2.0)``.
    flag_fill : int
        Value to fill the FLAG array with (``0`` = unflagged).
    """
    shape = (n_time, n_baseline, n_chan, n_pol)
    rng = np.random.default_rng(seed)
    if weight_per_pol is None:
        weight = rng.uniform(0.5, 2.0, shape).astype(np.float64)
    else:
        weight = np.broadcast_to(
            np.asarray(weight_per_pol, dtype=np.float64), shape
        ).copy()
    flag = np.full(shape, flag_fill, dtype=np.int8)
    uvw = rng.uniform(-100.0, 100.0, (n_time, n_baseline, 3))
    freq = np.linspace(1.0e9, 1.1e9, n_chan)

    ds = xr.Dataset(
        data_vars={
            "WEIGHT": (_MS_DIMS, weight),
            "FLAG": (_MS_DIMS, flag),
            "UVW": (("time", "baseline", "uvw_label"), uvw),
        },
        coords={"frequency": freq},
    )
    ds.attrs["data_groups"] = {
        "base": {"weight": "WEIGHT", "flag": "FLAG", "uvw": "UVW"}
    }
    return ds


def _make_ps_xdt(n_ms=1, **kwargs):
    """Wrap one or more MS datasets in a Processing Set ``DataTree``."""
    ps_xdt = xr.DataTree()
    for i in range(n_ms):
        ps_xdt[f"ms_{i}"] = xr.DataTree(dataset=_make_ms_ds(seed=i, **kwargs))
    return ps_xdt


def _make_img_xds(n_l=8, n_m=8, n_chan=2, delta=1.0e-5):
    """Minimal image-like ``xr.Dataset`` used only for the Briggs dispatch.

    The function under test reads ``sizes['l']``, ``sizes['m']``,
    ``sizes['frequency']`` and calls ``img_xds.xr_img.get_lm_cell_size()``.
    The latter accessor is registered as a side-effect of importing
    ``xradio.image.image_xds`` inside the test that needs it.
    """
    l_coord = (np.arange(n_l) - n_l / 2) * delta
    m_coord = (np.arange(n_m) - n_m / 2) * delta
    freq = np.linspace(1.0e9, 1.1e9, n_chan)
    img_xds = xr.Dataset(
        coords={"l": l_coord, "m": m_coord, "frequency": freq},
    )
    # Required by the xradio xr_img accessor (see IMAGE_DATASET_TYPES).
    img_xds.attrs["type"] = "image_dataset"
    return img_xds


# ---------------------------------------------------------------------------
# _equalize_parallel_hand_weights
# ---------------------------------------------------------------------------
class TestEqualizeParallelHandWeights(unittest.TestCase):
    """Pure-numpy tests for the parallel-hand equalization helper."""

    def test_n_pol_2_casa_is_arithmetic_mean(self):
        """``casa=True`` averages the two parallel-hand weights."""
        w = np.broadcast_to(np.array([2.0, 6.0]), (3, 4, 1, 2)).copy()
        out = _equalize_parallel_hand_weights(w, casa_weighting_implementation=True)
        self.assertEqual(out.shape, (3, 4, 1, 1))
        np.testing.assert_allclose(out, np.full((3, 4, 1, 1), 4.0))

    def test_n_pol_2_non_casa_is_error_propagation(self):
        """``casa=False`` uses ``2*w0*w1/(w0+w1)`` (i.e. w_I/2)."""
        w = np.broadcast_to(np.array([2.0, 6.0]), (3, 4, 1, 2)).copy()
        out = _equalize_parallel_hand_weights(w, casa_weighting_implementation=False)
        # 2*2*6 / (2+6) = 24/8 = 3.0
        np.testing.assert_allclose(out, np.full((3, 4, 1, 1), 3.0))

    def test_n_pol_4_uses_pol0_and_pol3_for_parallel_hands(self):
        """For 4 pols, pol indices 0 and 3 are the parallel hands;
        the cross-hands (indices 1, 2) are ignored by the helper."""
        # pol 0 = 2, pol 1 = 999 (cross), pol 2 = 999 (cross), pol 3 = 6.
        w = np.broadcast_to(
            np.array([2.0, 999.0, 999.0, 6.0]), (2, 3, 1, 4)
        ).copy()
        out_casa = _equalize_parallel_hand_weights(
            w, casa_weighting_implementation=True
        )
        out_err = _equalize_parallel_hand_weights(
            w, casa_weighting_implementation=False
        )
        self.assertEqual(out_casa.shape, (2, 3, 1, 1))
        np.testing.assert_allclose(out_casa, np.full((2, 3, 1, 1), 4.0))
        np.testing.assert_allclose(out_err, np.full((2, 3, 1, 1), 3.0))

    def test_equal_weights_make_both_formulas_agree(self):
        """The two formulas are identical when ``w_xx == w_yy``."""
        w = np.full((2, 2, 1, 2), 5.0)
        out_casa = _equalize_parallel_hand_weights(w, True)
        out_err = _equalize_parallel_hand_weights(w, False)
        np.testing.assert_allclose(out_casa, out_err)
        np.testing.assert_allclose(out_casa, np.full((2, 2, 1, 1), 5.0))

    def test_non_casa_is_strictly_less_than_arithmetic_mean(self):
        """Per Moellenbrock 2025: harmonic-mean-style form is always
        ≤ arithmetic mean, with equality iff the weights are equal."""
        w = np.broadcast_to(np.array([1.0, 10.0]), (1, 1, 1, 2)).copy()
        out_casa = _equalize_parallel_hand_weights(w, True)
        out_err = _equalize_parallel_hand_weights(w, False)
        self.assertTrue(np.all(out_err < out_casa))

    def test_n_pol_1_passes_through_unchanged(self):
        """Single-correlation data is returned unchanged
        (the natural Stokes-I assumption requires no equalization)."""
        w = np.full((2, 2, 1, 1), 7.0)
        out = _equalize_parallel_hand_weights(w, False)
        self.assertEqual(out.shape, (2, 2, 1, 1))
        np.testing.assert_array_equal(out, w)

    def test_n_pol_3_passes_through_unchanged(self):
        """Unsupported polarization counts are returned unchanged."""
        w = np.full((1, 1, 1, 3), 1.0)
        out = _equalize_parallel_hand_weights(w, True)
        np.testing.assert_array_equal(out, w)

    def test_nan_in_parallel_hand_propagates(self):
        """NaN in either parallel-hand correlation gives NaN out."""
        w = np.array([[[[np.nan, 5.0]]]])
        out_casa = _equalize_parallel_hand_weights(w, True)
        out_err = _equalize_parallel_hand_weights(w, False)
        self.assertTrue(np.isnan(out_casa).all())
        self.assertTrue(np.isnan(out_err).all())

    def test_input_array_not_mutated(self):
        """The helper returns a new array; it does not mutate its input."""
        w = np.broadcast_to(np.array([2.0, 6.0]), (1, 1, 1, 2)).copy()
        original = w.copy()
        _ = _equalize_parallel_hand_weights(w, True)
        np.testing.assert_array_equal(w, original)


# ---------------------------------------------------------------------------
# Natural-weighting path
# ---------------------------------------------------------------------------
class TestCalculateImagingWeightsNatural(unittest.TestCase):
    """End-to-end tests of the natural-weighting branch.

    ``img_xds`` is unused on the natural path, so we pass ``None``.
    """

    def test_writes_weight_imaging_with_input_dims(self):
        """Output variable is added with the same dims/shape as WEIGHT."""
        ps_xdt = _make_ps_xdt(weight_per_pol=[2.0, 6.0])
        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        ms = ps_xdt["ms_0"]
        self.assertIn("WEIGHT_IMAGING", ms.ds.data_vars)
        self.assertEqual(ms["WEIGHT_IMAGING"].dims, ms["WEIGHT"].dims)
        self.assertEqual(ms["WEIGHT_IMAGING"].shape, ms["WEIGHT"].shape)

    def test_casa_implementation_uses_arithmetic_mean(self):
        """With ``casa_weighting_implementation=True`` the output equals the
        arithmetic mean of the two parallel-hand weights, tiled to ``n_pol``."""
        ps_xdt = _make_ps_xdt(weight_per_pol=[2.0, 6.0])
        calculate_imaging_weights(
            ps_xdt,
            None,
            imaging_weights_params={
                "weighting": "natural",
                "casa_weighting_implementation": True,
            },
        )
        out = ps_xdt["ms_0"]["WEIGHT_IMAGING"].values
        np.testing.assert_allclose(out, np.full_like(out, 4.0))

    def test_error_propagation_is_default(self):
        """No explicit casa flag → the formal error-propagation form is used."""
        ps_xdt = _make_ps_xdt(weight_per_pol=[2.0, 6.0])
        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        out = ps_xdt["ms_0"]["WEIGHT_IMAGING"].values
        # 2*2*6 / (2+6) = 3.0
        np.testing.assert_allclose(out, np.full_like(out, 3.0))

    def test_4_pol_input_tiles_to_4_pol_output(self):
        """For 4 pols the equalized weight is broadcast across all four
        polarizations on output."""
        ps_xdt = _make_ps_xdt(
            n_pol=4, weight_per_pol=[2.0, 100.0, 100.0, 6.0]
        )
        calculate_imaging_weights(
            ps_xdt,
            None,
            imaging_weights_params={
                "weighting": "natural",
                "casa_weighting_implementation": True,
            },
        )
        out = ps_xdt["ms_0"]["WEIGHT_IMAGING"].values
        self.assertEqual(out.shape[-1], 4)
        np.testing.assert_allclose(out, np.full_like(out, 4.0))

    def test_flagged_samples_propagate_nan(self):
        """A flagged correlation injects NaN into the equalized output."""
        ds = _make_ms_ds(weight_per_pol=[2.0, 6.0], flag_fill=0)
        # Flag pol 0 at (t=0, b=0, f=0).
        ds["FLAG"].values[0, 0, 0, 0] = 1
        ps_xdt = xr.DataTree()
        ps_xdt["ms_0"] = xr.DataTree(dataset=ds)

        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        out = ps_xdt["ms_0"]["WEIGHT_IMAGING"].values
        # NaN at the flagged (t,b,f) is broadcast to all output polarizations
        # because the equalization collapses the pol axis.
        self.assertTrue(np.all(np.isnan(out[0, 0, 0, :])))
        # Other samples remain finite (3.0 from the error-propagation form).
        self.assertTrue(np.all(np.isfinite(out[0, 1, :, :])))

    def test_registers_imaging_data_group(self):
        """The new ``imaging`` data group is added on every MS."""
        ps_xdt = _make_ps_xdt(n_ms=2)
        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        for _, ms in ps_xdt.items():
            groups = ms.ds.attrs["data_groups"]
            self.assertIn("imaging", groups)
            self.assertEqual(groups["imaging"]["weight_imaging"], "WEIGHT_IMAGING")
            # Audit-trail fields from modify_data_groups_xds.
            self.assertIn("date", groups["imaging"])
            self.assertIn("description", groups["imaging"])

    def test_return_weight_density_grid_ignored_on_natural(self):
        """Natural weighting always returns ``None`` even when the flag is set."""
        ps_xdt = _make_ps_xdt()
        result = calculate_imaging_weights(
            ps_xdt,
            None,
            imaging_weights_params={"weighting": "natural"},
            return_weight_density_grid=True,
        )
        self.assertIsNone(result)

    def test_processes_every_ms_in_processing_set(self):
        """A multi-MS DataTree gets WEIGHT_IMAGING populated on every child."""
        ps_xdt = _make_ps_xdt(n_ms=3, weight_per_pol=[2.0, 6.0])
        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        for _, ms in ps_xdt.items():
            self.assertIn("WEIGHT_IMAGING", ms.ds.data_vars)
            out = ms["WEIGHT_IMAGING"].values
            np.testing.assert_allclose(out, np.full_like(out, 3.0))

    def test_overwrite_false_second_call_raises(self):
        """Re-running on a dataset that already has the output variable
        raises (guard against accidental clobber)."""
        ps_xdt = _make_ps_xdt()
        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        with self.assertRaises(AssertionError):
            calculate_imaging_weights(
                ps_xdt,
                None,
                imaging_weights_params={"weighting": "natural"},
                overwrite=False,
            )

    def test_overwrite_true_second_call_succeeds(self):
        """``overwrite=True`` allows the output variable to be replaced."""
        ps_xdt = _make_ps_xdt(weight_per_pol=[2.0, 6.0])
        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        # Second call with overwrite=True must not raise.
        calculate_imaging_weights(
            ps_xdt,
            None,
            imaging_weights_params={"weighting": "natural"},
            overwrite=True,
        )

    def test_flag_mutates_input_weight_in_place(self):
        """Documented memory-constrained design: flagged samples in the
        *input* WEIGHT array are overwritten with NaN by this function."""
        ds = _make_ms_ds(weight_per_pol=[2.0, 6.0])
        ds["FLAG"].values[0, 0, 0, 0] = 1
        ps_xdt = xr.DataTree()
        ps_xdt["ms_0"] = xr.DataTree(dataset=ds)

        # Pre-condition: WEIGHT is finite everywhere.
        self.assertTrue(np.all(np.isfinite(ds["WEIGHT"].values)))
        calculate_imaging_weights(
            ps_xdt, None, imaging_weights_params={"weighting": "natural"}
        )
        # Post-condition: the flagged cell is now NaN in the input array.
        self.assertTrue(np.isnan(ds["WEIGHT"].values[0, 0, 0, 0]))


# ---------------------------------------------------------------------------
# Briggs / uniform dispatch (gridder mocked)
# ---------------------------------------------------------------------------
class TestCalculateImagingWeightsDispatch(unittest.TestCase):
    """Dispatch and Briggs-path behaviour, with the C-level gridder mocked."""

    def setUp(self):
        # Ensure the xr_img accessor is registered for img_xds.xr_img.
        import xradio.image.image_xds  # noqa: F401

        self.img_xds = _make_img_xds()
        # Patch the gridder, Briggs-factor, and degridder symbols as imported
        # into the calculate_imaging_weights module.
        self.grid_patch = mock.patch(f"{_MOD_PATH}.grid_imaging_weights")
        self.briggs_patch = mock.patch(f"{_MOD_PATH}.calculate_briggs_params")
        self.degrid_patch = mock.patch(f"{_MOD_PATH}.degrid_imaging_weights")

        self.grid_mock = self.grid_patch.start()
        self.briggs_mock = self.briggs_patch.start()
        self.degrid_mock = self.degrid_patch.start()
        self.addCleanup(self.grid_patch.stop)
        self.addCleanup(self.briggs_patch.stop)
        self.addCleanup(self.degrid_patch.stop)

        # Degrid: return the equalized data_weight unchanged so the caller's
        # tile(..., n_pol) produces an output of the correct shape.
        self.degrid_mock.side_effect = (
            lambda grid, uvw, dw, briggs, freq, n_uv, dlm: dw
        )
        self.briggs_mock.return_value = np.zeros((2, 1, 1))

    def test_uniform_dispatches_to_briggs_with_robust_neg_2(self):
        """``weighting="uniform"`` is rewritten to ``briggs`` with ``robust=-2.0``
        before the Briggs-factor calculation runs."""
        ps_xdt = _make_ps_xdt()
        calculate_imaging_weights(
            ps_xdt,
            self.img_xds,
            imaging_weights_params={"weighting": "uniform"},
        )

        self.briggs_mock.assert_called_once()
        params_arg = self.briggs_mock.call_args[0][2]
        self.assertEqual(params_arg["weighting"], "briggs")
        self.assertEqual(params_arg["robust"], -2.0)

    def test_briggs_default_casa_implementation_is_false(self):
        """The Briggs path does not crash when the caller omits
        ``casa_weighting_implementation``; the param checker fills the default."""
        ps_xdt = _make_ps_xdt()
        calculate_imaging_weights(
            ps_xdt,
            self.img_xds,
            imaging_weights_params={"weighting": "briggs", "robust": 0.5},
        )
        # The Briggs path writes WEIGHT_IMAGING on every MS.
        for _, ms in ps_xdt.items():
            self.assertIn("WEIGHT_IMAGING", ms.ds.data_vars)

    def test_briggs_path_grids_then_degrids(self):
        """The Briggs branch always grids once per MS, computes factors once,
        then degrids once per MS."""
        ps_xdt = _make_ps_xdt(n_ms=2)
        calculate_imaging_weights(
            ps_xdt,
            self.img_xds,
            imaging_weights_params={"weighting": "briggs", "robust": 0.5},
        )
        self.assertEqual(self.grid_mock.call_count, 2)
        self.assertEqual(self.briggs_mock.call_count, 1)
        self.assertEqual(self.degrid_mock.call_count, 2)

    def test_return_weight_density_grid_briggs(self):
        """With ``return_weight_density_grid=True`` on the Briggs path, the
        density grid is returned with shape ``(n_chan, 1, n_u, n_v)``."""
        ps_xdt = _make_ps_xdt()
        result = calculate_imaging_weights(
            ps_xdt,
            self.img_xds,
            imaging_weights_params={"weighting": "briggs", "robust": 0.5},
            return_weight_density_grid=True,
        )
        self.assertIsInstance(result, np.ndarray)
        n_chan = self.img_xds.sizes["frequency"]
        n_l = self.img_xds.sizes["l"]
        n_m = self.img_xds.sizes["m"]
        self.assertEqual(result.shape, (n_chan, 1, n_l, n_m))

    def test_single_precision_gridding_uses_float32_grid(self):
        """``single_precision_gridding=True`` allocates a float32 density grid."""
        ps_xdt = _make_ps_xdt()
        result = calculate_imaging_weights(
            ps_xdt,
            self.img_xds,
            imaging_weights_params={"weighting": "briggs", "robust": 0.5},
            single_precision_gridding=True,
            return_weight_density_grid=True,
        )
        self.assertEqual(result.dtype, np.float32)


# ---------------------------------------------------------------------------
# Parameter checker
# ---------------------------------------------------------------------------
class TestCheckImagingWeightsParams(unittest.TestCase):
    """Validation behaviour for ``check_imaging_weights_params``."""

    def test_casa_implementation_defaults_to_false(self):
        """When omitted, the checker writes ``casa_weighting_implementation=False``."""
        params = {"weighting": "natural"}
        self.assertTrue(check_imaging_weights_params(params))
        self.assertIn("casa_weighting_implementation", params)
        self.assertFalse(params["casa_weighting_implementation"])

    def test_casa_implementation_explicit_true_is_preserved(self):
        """An explicit ``True`` is not overwritten by the default."""
        params = {"weighting": "natural", "casa_weighting_implementation": True}
        self.assertTrue(check_imaging_weights_params(params))
        self.assertTrue(params["casa_weighting_implementation"])

    def test_briggs_requires_robust_in_range(self):
        """``robust`` outside ``[-2, 2]`` fails the check."""
        params = {"weighting": "briggs", "robust": 5.0}
        self.assertFalse(check_imaging_weights_params(params))

    def test_uniform_is_accepted(self):
        """``weighting='uniform'`` is a valid input to the parameter checker
        (dispatch to Briggs is then handled by ``calculate_imaging_weights``)."""
        params = {"weighting": "uniform"}
        self.assertTrue(check_imaging_weights_params(params))


if __name__ == "__main__":
    unittest.main()
