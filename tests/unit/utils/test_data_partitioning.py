"""Tests for astroviper.utils.data_partitioning."""

import itertools
import math
import socket
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import dask
import numpy as np
import pytest
import xarray as xr
from graphviper.graph_tools.coordinate_utils import make_parallel_coord

from astroviper.utils.data_partitioning import (
    COMPUTE_RESOURCES_HOST_KEYS,
    COMPUTE_RESOURCES_KEYS,
    calculate_data_chunking,
    calculate_frequency_chunking,
    check_compute_resources,
    get_compute_resources,
)

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


def test_calculate_frequency_chunking_accounts_for_constant_memory(monkeypatch):
    """The shared frequency helper forwards fixed and per-channel allocations."""
    call = {}

    def chunking(memory_per_frequency, sizes, thread_info, **kwargs):
        call.update(
            memory_per_frequency=memory_per_frequency,
            sizes=sizes,
            thread_info=thread_info,
            **kwargs,
        )
        return {"frequency": 3}

    monkeypatch.setattr(
        "astroviper.utils.data_partitioning.calculate_data_chunking", chunking
    )
    assert (
        calculate_frequency_chunking(
            12,
            0.25,
            1.5,
            None,
            {"n_threads": 2, "memory_per_thread": 4.0},
        )
        == 3
    )
    assert call == {
        "memory_per_frequency": 0.25,
        "sizes": {"frequency": 12},
        "thread_info": {"n_threads": 2, "memory_per_thread": 4.0},
        "constant_memory": 1.5,
        "tasks_per_thread": 4,
    }


def test_calculate_frequency_chunking_keeps_explicit_override(monkeypatch):
    """An explicit task count does not query cluster state or memory."""
    monkeypatch.setattr(
        "astroviper.utils.data_partitioning.get_thread_info",
        lambda: pytest.fail("explicit chunking must not inspect the cluster"),
    )
    assert calculate_frequency_chunking(12, 0.25, 1.5, 7) == 7


def test_compute_resources_of_the_threaded_scheduler():
    """Without a client the graphs run on the local thread pool: num_workers
    threads, else dask's CPU_COUNT, on this host."""
    from dask.system import CPU_COUNT

    with dask.config.set(scheduler="threads", num_workers=3):
        resources = get_compute_resources("dask", 2)
    assert tuple(resources) == COMPUTE_RESOURCES_KEYS
    assert resources["scheduler"] == "threads"
    assert resources["processing_function_threads"] == 2
    assert (resources["n_hosts"], resources["n_workers"]) == (1, 1)
    assert resources["n_worker_slots"] == 3
    assert resources["worker_memory_limit_bytes"] is None
    (hostname,) = resources["hosts"]
    assert hostname == socket.gethostname()
    assert tuple(resources["hosts"][hostname]) == COMPUTE_RESOURCES_HOST_KEYS
    assert resources["n_logical_cpus"] > 0 and resources["memory_total_bytes"] > 0
    with dask.config.set(scheduler="threads"):
        assert get_compute_resources("dask", 1)["n_worker_slots"] == CPU_COUNT
    check_compute_resources(resources)


def test_compute_resources_of_the_local_schedulers():
    with dask.config.set(scheduler="sync"):
        resources = get_compute_resources("dask", 1)
    assert resources["scheduler"] == "synchronous"
    assert (resources["n_workers"], resources["n_worker_slots"]) == (1, 1)
    with dask.config.set(scheduler="processes", num_workers=3):
        resources = get_compute_resources("dask", 1)
    assert resources["scheduler"] == "processes"
    assert (resources["n_workers"], resources["n_worker_slots"]) == (3, 3)


def test_compute_resources_unknown_pool_size():
    """A custom pool, or the threaded scheduler from another thread without
    num_workers, has a size the helper cannot know."""
    with ThreadPoolExecutor(2) as pool, dask.config.set(scheduler="threads", pool=pool):
        assert get_compute_resources("dask", 1)["n_worker_slots"] is None
    with dask.config.set(scheduler="processes", pool=object()):
        resources = get_compute_resources("dask", 1)
    assert resources["n_workers"] is None and resources["n_worker_slots"] is None
    found = []
    with dask.config.set(scheduler="threads"):
        thread = threading.Thread(
            target=lambda: found.append(get_compute_resources("dask", 1))
        )
        thread.start()
        thread.join()
    assert found[0]["n_worker_slots"] is None
    assert found[0]["n_hosts"] == 1


def test_compute_resources_of_an_unrecognised_scheduler():
    def scheduler(*args, **kwargs):
        raise AssertionError("not called")

    with dask.config.set(scheduler=scheduler):
        resources = get_compute_resources("dask", 1)
    assert resources["scheduler"] is None and resources["hosts"] is None
    assert resources["n_hosts"] is None and resources["n_worker_slots"] is None


def test_compute_resources_of_a_distributed_cluster():
    from distributed import Client, LocalCluster

    with (
        LocalCluster(
            n_workers=2,
            threads_per_worker=2,
            memory_limit="1GiB",
            processes=False,
            dashboard_address=None,
        ) as cluster,
        Client(cluster),
    ):
        resources = get_compute_resources("dask", 4)
    assert resources["scheduler"] == "distributed"
    assert resources["n_hosts"] == 1
    assert (resources["n_workers"], resources["n_worker_slots"]) == (2, 4)
    assert resources["worker_memory_limit_bytes"] == 2 * 2**30
    host = resources["hosts"][socket.gethostname()]
    for key in COMPUTE_RESOURCES_HOST_KEYS:
        assert host[key] == resources[key], key
    check_compute_resources(resources)


def test_compute_resources_without_worker_memory_limits():
    from distributed import Client, LocalCluster

    with (
        LocalCluster(
            n_workers=1,
            threads_per_worker=3,
            memory_limit=0,
            processes=False,
            dashboard_address=None,
        ) as cluster,
        Client(cluster),
    ):
        resources = get_compute_resources("dask", 1)
    assert resources["n_worker_slots"] == 3
    assert resources["worker_memory_limit_bytes"] is None


@pytest.mark.parametrize("world_size, n_workers", [(5, 4), (1, None)])
def test_compute_resources_of_mpi(monkeypatch, world_size, n_workers):
    """The static mpi4py.futures pool is every rank but the manager."""
    mpi = types.SimpleNamespace(
        COMM_WORLD=types.SimpleNamespace(Get_size=lambda: world_size)
    )
    monkeypatch.setitem(sys.modules, "mpi4py", types.SimpleNamespace(MPI=mpi))
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    resources = get_compute_resources("mpi", 4)
    assert resources["scheduler"] == "mpi4py.futures"
    assert resources["mpi_world_size"] == world_size
    assert resources["n_workers"] == resources["n_worker_slots"] == n_workers
    assert resources["hosts"] is None and resources["n_hosts"] is None
    check_compute_resources(resources)


def test_compute_resources_refuse_unknown_values():
    with pytest.raises(ValueError, match="Unknown compute_backend 'spark'"):
        get_compute_resources("spark", 1)
    check_compute_resources({"n_worker_slots": 112})  # absent keys are unknown
    with pytest.raises(ValueError, match="unknown keys \\['parallelism'\\]"):
        check_compute_resources({"parallelism": 4})
    with pytest.raises(ValueError, match="compute_backend"):
        check_compute_resources({"compute_backend": "spark"})
    with pytest.raises(ValueError, match="scheduler"):
        check_compute_resources({"scheduler": "slurm"})
    with pytest.raises(ValueError, match="hosts'\\]\\['node'\\] has unknown keys"):
        check_compute_resources({"hosts": {"node": {"cores": 8}}})
