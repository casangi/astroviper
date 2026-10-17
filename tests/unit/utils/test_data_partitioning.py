"""Tests for astroviper.utils.data_partitioning."""

import itertools
import math

import numpy as np
import pytest
import xarray as xr
from graphviper.graph_tools.coordinate_utils import make_parallel_coord

from astroviper.utils.data_partitioning import calculate_data_chunking

# The simulated observation of tests/component/test_point_source_flux_recovery.py
FLUX_RECOVERY_DIMS_SIZES = {"time": 48, "frequency": 5}


def graph_chunking(chunking_dims_sizes, n_threads, tasks_per_thread=2):
    """Chunking with enough memory that the target is n_threads * tasks_per_thread,
    capped at the number of samples."""
    return calculate_data_chunking(
        1e-9,
        chunking_dims_sizes,
        {"n_threads": n_threads, "memory_per_thread": 1.0},
        tasks_per_thread=tasks_per_thread,
    )


def check_chunking(chunking_dims_sizes, n_chunks_dict, n_chunks_target):
    """Every axis gets 1 to its size chunks, at least n_chunks_target in all, and the
    parallel coordinates made from them assign every sample exactly once."""
    assert n_chunks_dict.keys() == chunking_dims_sizes.keys()
    assert math.prod(n_chunks_dict.values()) >= n_chunks_target
    for dim, size in chunking_dims_sizes.items():
        assert 1 <= n_chunks_dict[dim] <= size
        coord = xr.DataArray(np.arange(size, dtype=float), dims=dim)
        parallel_coord = make_parallel_coord(coord=coord, n_chunks=n_chunks_dict[dim])
        np.testing.assert_array_equal(
            np.concatenate(list(parallel_coord["data_chunks"].values())), coord.values
        )


@pytest.mark.parametrize("n_threads", [1, 2, 4, 8, 12, 16, 24, 32, 48])
def test_time_and_frequency_chunking(n_threads):
    """Issue 308: 24 threads x 2 tasks per thread = 48 chunks raised IndexError."""
    n_chunks_dict = graph_chunking(FLUX_RECOVERY_DIMS_SIZES, n_threads)
    check_chunking(FLUX_RECOVERY_DIMS_SIZES, n_chunks_dict, 2 * n_threads)
    assert math.prod(n_chunks_dict.values()) == 2 * n_threads


@pytest.mark.parametrize(
    "chunking_dims_sizes, n_threads",
    [
        ({"time": 48, "frequency": 1}, 24),  # raised IndexError
        ({"time": 1, "frequency": 1}, 1),  # gave 2 chunks per axis
        ({"time": 2, "frequency": 2}, 24),
        ({"time": 3, "frequency": 3}, 4),  # 8 chunks: looped forever
        ({"time": 4, "frequency": 3}, 24),  # 12 chunks: looped forever
    ],
)
def test_short_axes(chunking_dims_sizes, n_threads):
    n_chunks_dict = graph_chunking(chunking_dims_sizes, n_threads)
    n_chunks_target = min(2 * n_threads, math.prod(chunking_dims_sizes.values()))
    check_chunking(chunking_dims_sizes, n_chunks_dict, n_chunks_target)


def test_memory_sets_the_chunk_count():
    """A memory driven target of 37 chunks, a prime larger than either axis, is met."""
    chunking_dims_sizes = {"l": 30, "m": 30}
    n_chunks_dict = calculate_data_chunking(
        1.0,
        chunking_dims_sizes,
        {"n_threads": 2, "memory_per_thread": 900 / 36.5},
        tasks_per_thread=1,
    )
    check_chunking(chunking_dims_sizes, n_chunks_dict, 37)


def test_every_target_on_small_axes():
    for sizes in itertools.chain(
        itertools.product(range(1, 9), repeat=2),
        itertools.product(range(1, 5), repeat=3),
    ):
        chunking_dims_sizes = {f"dim_{i}": size for i, size in enumerate(sizes)}
        for n_chunks_target in range(1, math.prod(sizes) + 1):
            n_chunks_dict = graph_chunking(chunking_dims_sizes, n_chunks_target, 1)
            check_chunking(chunking_dims_sizes, n_chunks_dict, n_chunks_target)
