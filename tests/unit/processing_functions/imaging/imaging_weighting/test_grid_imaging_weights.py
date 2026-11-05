"""Unit tests for the ``grid_imaging_weights`` / ``degrid_imaging_weights``
Python wrappers' ``frequency_map`` handling.

The C++ kernels themselves are covered by ``test_grid_imaging_weights_cpp.py``;
these tests check that the wrappers (a) default to the identity channel map,
(b) honour an explicit map, accumulating each visibility channel on its image
plane, and (c) validate the map before handing raw pointers to C++.
"""

import unittest

import numpy as np

from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights import (
    degrid_imaging_weights,
    grid_imaging_weights,
)

N_UV = np.array([16, 16])
DELTA_LM = np.array([-1.0e-5, 1.0e-5])


def _visibilities(n_chan, n_baseline=4, seed=0):
    rng = np.random.default_rng(seed)
    uvw = np.zeros((1, n_baseline, 3))
    # Small |uv| so every sample lands inside the 16x16 grid.
    uvw[..., :2] = rng.uniform(-50.0, 50.0, (1, n_baseline, 2))
    data_weight = rng.uniform(0.5, 2.0, (1, n_baseline, n_chan, 1))
    freq_chan = np.linspace(1.0e9, 1.1e9, n_chan)
    return uvw, data_weight, freq_chan


class TestGridImagingWeightsFrequencyMap(unittest.TestCase):
    def test_default_is_identity_map(self):
        uvw, data_weight, freq_chan = _visibilities(n_chan=3)
        grid_default = np.zeros((3, 1, *N_UV))
        sum_default = np.zeros((3, 1))
        grid_explicit = np.zeros_like(grid_default)
        sum_explicit = np.zeros_like(sum_default)

        grid_imaging_weights(
            grid_default, sum_default, uvw, data_weight, freq_chan, N_UV, DELTA_LM
        )
        grid_imaging_weights(
            grid_explicit,
            sum_explicit,
            uvw,
            data_weight,
            freq_chan,
            N_UV,
            DELTA_LM,
            frequency_map=np.arange(3),
        )

        np.testing.assert_array_equal(grid_default, grid_explicit)
        np.testing.assert_array_equal(sum_default, sum_explicit)
        self.assertTrue(np.all(sum_default > 0.0))

    def test_explicit_map_accumulates_on_image_planes(self):
        """Two visibility channels mapped onto planes 1 and 3 of a 4-plane
        grid leave planes 0 and 2 empty."""
        uvw, data_weight, freq_chan = _visibilities(n_chan=2)
        grid = np.zeros((4, 1, *N_UV))
        sum_weight = np.zeros((4, 1))

        grid_imaging_weights(
            grid,
            sum_weight,
            uvw,
            data_weight,
            freq_chan,
            N_UV,
            DELTA_LM,
            frequency_map=np.array([1, 3]),
        )

        np.testing.assert_array_equal(grid[[0, 2]], 0.0)
        np.testing.assert_array_equal(sum_weight[[0, 2]], 0.0)
        self.assertTrue(np.all(sum_weight[[1, 3]] > 0.0))
        # Each plane holds exactly its channel's (conjugate-doubled) weight sum.
        np.testing.assert_allclose(
            sum_weight[[1, 3], 0], 2.0 * data_weight[..., 0].sum(axis=(0, 1))
        )

    def test_degrid_honours_same_map(self):
        """Degridding with the map used for gridding samples the populated
        planes (Briggs denominator > 1, so the imaging weight is below the
        data weight); degridding with the identity map instead reads the
        empty plane 0 for channel 0 and returns the data weight unchanged."""
        uvw, data_weight, freq_chan = _visibilities(n_chan=2)
        grid = np.zeros((4, 1, *N_UV))
        sum_weight = np.zeros((4, 1))
        frequency_map = np.array([1, 3])
        grid_imaging_weights(
            grid,
            sum_weight,
            uvw,
            data_weight,
            freq_chan,
            N_UV,
            DELTA_LM,
            frequency_map=frequency_map,
        )
        briggs_factors = np.ones((2, 4, 1))

        mapped = degrid_imaging_weights(
            grid,
            uvw,
            data_weight,
            briggs_factors,
            freq_chan,
            N_UV,
            DELTA_LM,
            frequency_map=frequency_map,
        )
        self.assertTrue(np.all(np.isfinite(mapped)))
        self.assertTrue(np.all(mapped > 0.0))
        self.assertTrue(np.all(mapped < data_weight))

        identity = degrid_imaging_weights(
            grid, uvw, data_weight, briggs_factors, freq_chan, N_UV, DELTA_LM
        )
        # Plane 0 is empty: denominator 0 * 1 + 1, so the weight passes through.
        np.testing.assert_allclose(identity[:, :, 0], data_weight[:, :, 0])

    def test_invalid_maps_raise_before_calling_kernel(self):
        uvw, data_weight, freq_chan = _visibilities(n_chan=2)
        grid = np.zeros((2, 1, *N_UV))
        sum_weight = np.zeros((2, 1))
        briggs_factors = np.ones((2, 2, 1))

        for bad_map, message in (
            ([0], "one grid-plane index"),
            ([0, 2], "outside the imaging-weight grid"),
            ([-1, 0], "outside the imaging-weight grid"),
        ):
            with self.subTest(frequency_map=bad_map):
                with self.assertRaisesRegex(ValueError, message):
                    grid_imaging_weights(
                        grid,
                        sum_weight,
                        uvw,
                        data_weight,
                        freq_chan,
                        N_UV,
                        DELTA_LM,
                        frequency_map=np.array(bad_map),
                    )
                with self.assertRaisesRegex(ValueError, message):
                    degrid_imaging_weights(
                        grid,
                        uvw,
                        data_weight,
                        briggs_factors,
                        freq_chan,
                        N_UV,
                        DELTA_LM,
                        frequency_map=np.array(bad_map),
                    )
        np.testing.assert_array_equal(grid, 0.0)


if __name__ == "__main__":
    unittest.main()
