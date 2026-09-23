"""Unit tests for the ``grid_imaging_weights`` / ``degrid_imaging_weights``
Python wrappers' ``frequency_map`` handling.

The C++ kernels themselves are covered by ``test_grid_imaging_weights_cpp.py``;
these tests check that the wrappers (a) default to the identity channel map,
(b) honour an explicit map, accumulating each visibility channel on its image
plane, and (c) validate the map before handing raw pointers to C++.
"""

import unittest

import numpy as np
import pytest

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


@pytest.mark.parametrize("truncate", [False, True])
def test_continuum_alias_collapses_channels_like_frequency_map(truncate):
    uvw, weights, frequencies = _visibilities(3)
    outputs = []
    for name in ("channel_map", "frequency_map"):
        grid = np.zeros((1, 1, *N_UV))
        total = np.zeros((1, 1))
        grid_imaging_weights(
            grid,
            total,
            uvw,
            weights,
            frequencies,
            N_UV,
            DELTA_LM,
            truncate_uv_cells=truncate,
            **{name: np.zeros(3, dtype=np.int64)},
        )
        np.testing.assert_allclose(total.sum(), 2 * weights.sum())
        outputs.append(grid)
    np.testing.assert_array_equal(*outputs)


@pytest.mark.parametrize("mapping", [[0, 1], [1, 0]])
def test_rejects_both_map_arguments_without_modifying_grid(mapping):
    uvw, weights, frequencies = _visibilities(2)
    grid = np.zeros((2, 1, *N_UV))
    total = np.zeros((2, 1))
    with pytest.raises(ValueError, match="only one of channel_map and frequency_map"):
        grid_imaging_weights(
            grid,
            total,
            uvw,
            weights,
            frequencies,
            N_UV,
            DELTA_LM,
            channel_map=[0, 1],
            frequency_map=mapping,
        )
    assert not grid.any() and not total.any()


@pytest.mark.parametrize("mapping", [[0], [-1, 0], [0, 2]])
def test_legacy_map_is_validated(mapping):
    uvw, weights, frequencies = _visibilities(2)
    with pytest.raises(ValueError):
        grid_imaging_weights(
            np.zeros((2, 1, *N_UV)),
            np.zeros((2, 1)),
            uvw,
            weights,
            frequencies,
            N_UV,
            DELTA_LM,
            channel_map=mapping,
        )


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("dtype", [">f4", ">f8"])
def test_mapped_degrid_accepts_strided_non_native_density(truncate, dtype):
    uvw, weights, frequencies = _visibilities(2)
    backing = np.empty((4, 1, N_UV[0], 2 * N_UV[1]), dtype=dtype)
    density = backing[..., ::2]
    density[:] = np.arange(1, 5)[:, None, None, None]
    factors = np.ones((2, 4, 1))
    result = degrid_imaging_weights(
        density,
        uvw,
        weights,
        factors,
        frequencies,
        N_UV,
        DELTA_LM,
        truncate_uv_cells=truncate,
        frequency_map=[1, 3],
    )
    np.testing.assert_allclose(result, weights / np.array([3, 5])[None, None, :, None])
    assert density.dtype.str == dtype  # caller's storage is not mutated


@pytest.mark.parametrize("truncate,cell", [(False, 9), (True, 8)])
def test_mapping_and_uv_cell_convention_are_independent(truncate, cell):
    # Place a sample at u=+0.75 pixels, v=0. With centre 8, nearest-cell
    # assignment selects 9, while continuum truncation selects 8.
    frequency = np.array([1.0e9])
    uvw = np.zeros((1, 1, 3))
    uvw[0, 0, 0] = 0.75 * 299792458.0 / (-frequency[0] * DELTA_LM[0] * N_UV[0])
    weights = np.ones((1, 1, 1, 1))
    density = np.zeros((2, 1, *N_UV))
    total = np.zeros((2, 1))
    grid_imaging_weights(
        density,
        total,
        uvw,
        weights,
        frequency,
        N_UV,
        DELTA_LM,
        truncate_uv_cells=truncate,
        frequency_map=[1],
    )
    assert density[1, 0, cell, 8] == 1
    assert not density[0].any()
    # A spatially varying grid independently checks the sampling convention.
    density[1, 0] = np.arange(N_UV[0])[:, None]
    result = degrid_imaging_weights(
        density,
        uvw,
        weights,
        np.ones((2, 2, 1)),
        frequency,
        N_UV,
        DELTA_LM,
        truncate_uv_cells=truncate,
        frequency_map=[1],
    )
    np.testing.assert_allclose(result, 1 / (cell + 1))


if __name__ == "__main__":
    unittest.main()
