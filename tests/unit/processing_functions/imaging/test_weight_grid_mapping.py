"""Weight-grid mapping and read-only density normalization regressions."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights import (
    degrid_imaging_weights,
    grid_imaging_weights,
)


@pytest.mark.parametrize("truncate", [False, True])
@pytest.mark.parametrize("density_dtype", [np.float32, np.float64, ">f8"])
def test_collapsed_frequency_grid_and_normalized_lookup(truncate, density_dtype):
    grid = np.zeros((1, 1, 8, 8))
    sums = np.zeros((1, 1))
    uvw = np.zeros((1, 1, 3))
    weights = np.array([1.0, 3.0]).reshape(1, 1, 2, 1)
    frequencies = np.array([1e9, 1.1e9])
    mapping = np.zeros(2, dtype=np.int64)
    grid_imaging_weights(
        grid,
        sums,
        uvw,
        weights,
        frequencies,
        [8, 8],
        [1e-5, 1e-5],
        frequency_map=mapping,
        truncate_uv_cells=truncate,
    )
    assert grid.sum() == 8.0
    assert sums[0, 0] == 8.0
    # Swapping equal-sized spatial axes supplies a non-contiguous read-only input.
    density = grid.astype(density_dtype).swapaxes(-1, -2)
    density.flags.writeable = False
    result = degrid_imaging_weights(
        density,
        uvw,
        weights,
        np.array([1.0, 0.0]).reshape(2, 1, 1),
        frequencies,
        [8, 8],
        [1e-5, 1e-5],
        frequency_map=mapping,
        truncate_uv_cells=truncate,
    )
    np.testing.assert_array_equal(result, weights / 8.0)


@pytest.mark.parametrize("mapping", [None, [0], [-1, 0], [0, 1]])
@pytest.mark.parametrize("operation", ["grid", "degrid"])
def test_invalid_mapping_rejected_before_kernel(operation, mapping):
    grid = np.zeros((1, 1, 8, 8))
    uvw = np.zeros((1, 1, 3))
    weights = np.ones((1, 1, 2, 1))
    frequencies = np.array([1e9, 1.1e9])
    with pytest.raises(ValueError, match="frequency_map"):
        if operation == "grid":
            grid_imaging_weights(
                grid,
                np.zeros((1, 1)),
                uvw,
                weights,
                frequencies,
                [8, 8],
                [1e-5, 1e-5],
                frequency_map=mapping,
            )
        else:
            degrid_imaging_weights(
                grid,
                uvw,
                weights,
                np.ones((2, 1, 1)),
                frequencies,
                [8, 8],
                [1e-5, 1e-5],
                frequency_map=mapping,
            )
