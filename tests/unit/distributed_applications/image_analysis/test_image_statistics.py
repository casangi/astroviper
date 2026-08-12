"""Unit and small full-stack tests for the distributed image-statistics API."""

import importlib

import numpy as np
import pytest
import xarray as xr

statistics_module = importlib.import_module(
    "astroviper.distributed_applications.image_analysis.image_statistics"
)


@pytest.fixture
def image_store(tmp_path):
    """Create a small chunked five-axis Zarr image with a named spatial mask."""
    shape = (1, 4, 2, 3, 4)
    sky = xr.DataArray(
        np.arange(np.prod(shape), dtype=float).reshape(shape),
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={
            "time": [0],
            "frequency": [100, 101, 102, 103],
            "polarization": ["I", "Q"],
            "l": np.arange(3),
            "m": np.arange(4),
        },
        attrs={"units": "Jy/beam"},
    )
    mask = xr.DataArray(np.eye(3, 4, dtype=bool), dims=("l", "m"))
    dataset = xr.Dataset({"SKY": sky, "MASK_SKY": mask})
    path = tmp_path / "distributed-statistics.img.zarr"
    dataset.chunk({"frequency": 2, "l": 2, "m": 2}).to_zarr(path)
    return str(path), dataset


def test_automatic_partition_count_uses_selected_bytes_and_caps_axis():
    """Estimate memory-driven partitions and never create more than axis length."""
    selected = xr.DataArray(
        np.empty((4, 1024), dtype=np.float64), dims=("frequency", "x")
    )
    one = statistics_module._automatic_partition_count(
        selected,
        partition_dim="frequency",
        memory_limit_gib=1,
        working_memory_factor=2,
    )
    capped = statistics_module._automatic_partition_count(
        selected,
        partition_dim="frequency",
        memory_limit_gib=1e-9,
        working_memory_factor=2,
    )
    assert one == 1
    assert capped == 4


@pytest.mark.parametrize(
    ("memory_limit", "factor", "message"),
    [(0, 2.5, "memory_limit_gib"), (1, 0.5, "working_memory_factor")],
)
def test_automatic_partition_count_validation(memory_limit, factor, message):
    """Reject nonpositive memory targets and temporary-memory factors below one."""
    selected = xr.DataArray(np.ones((2, 2)), dims=("frequency", "x"))
    with pytest.raises(ValueError, match=message):
        statistics_module._automatic_partition_count(
            selected,
            partition_dim="frequency",
            memory_limit_gib=memory_limit,
            working_memory_factor=factor,
        )


def test_automatic_partition_count_uses_detected_worker_memory(monkeypatch):
    """Use half the detected per-thread memory when no target is supplied."""
    monkeypatch.setattr(
        "astroviper.utils.data_partitioning.get_thread_info",
        lambda: {"memory_per_thread": 2.0},
    )
    selected = xr.DataArray(np.ones((3, 2)), dims=("frequency", "x"))
    assert (
        statistics_module._automatic_partition_count(
            selected,
            partition_dim="frequency",
            memory_limit_gib=None,
            working_memory_factor=1,
        )
        == 1
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"image": xr.DataArray([1])}, TypeError, "on-disk image path"),
        ({"image": "unused", "mask": np.ones(2)}, TypeError, "must be named"),
    ],
)
def test_distributed_api_rejects_in_memory_inputs(kwargs, error, message):
    """Keep in-memory images and masks at the direct node-task access point."""
    with pytest.raises(error, match=message):
        statistics_module.image_statistics(**kwargs)


@pytest.mark.parametrize("n_partitions", [0, -1, 1.5, True])
def test_distributed_partition_count_must_be_positive_integer(
    image_store, n_partitions
):
    """Validate explicit partition counts before graph construction."""
    path, _ = image_store
    with pytest.raises(ValueError, match="positive integer"):
        statistics_module.image_statistics(
            path, data_variable="SKY", n_partitions=n_partitions
        )


def test_distributed_metadata_validation(image_store):
    """Reject unknown partition dimensions and missing named mask variables."""
    path, _ = image_store
    with pytest.raises(ValueError, match="not present"):
        statistics_module.image_statistics(
            path, data_variable="SKY", partition_dim="bad", n_partitions=1
        )
    with pytest.raises(KeyError, match="MISSING"):
        statistics_module.image_statistics(
            path, data_variable="SKY", mask="MISSING", n_partitions=1
        )


def test_reduce_adapter_passes_partition_metadata(monkeypatch):
    """Translate GraphVIPER reducer arguments into processing-layer arguments."""
    called = {}

    def merge(states, **kwargs):
        called.update(kwargs)
        return states[0]

    monkeypatch.setattr(
        "astroviper.processing_functions.image_analysis.statistics.merge_statistics_states",
        merge,
    )
    state = xr.Dataset({"value": xr.DataArray(1)})
    result = statistics_module._reduce_statistics_states(
        [state],
        {"partition_dim": "frequency", "reduction_dims": ("frequency",)},
    )
    assert result is state
    assert called == {
        "partition_dim": "frequency",
        "reduction_dims": ("frequency",),
    }


def test_distributed_and_node_on_disk_match_node_in_memory(image_store, monkeypatch):
    """Return identical statistics through all three image-statistics access paths."""
    from astroviper.node_tasks.image_analysis.image_statistics import (
        image_statistics as direct_statistics,
    )

    path, dataset = image_store

    # XRADIO's Zarr implementation ultimately performs this lazy isel; using
    # the small shim avoids importing optional CASA readers in minimal CI jobs.
    monkeypatch.setattr(
        "xradio.image.load_image",
        lambda store, block_des: xr.open_zarr(store).isel(block_des),
    )
    statistics = ("mean", "max", "maxpos", "npts")
    distributed = statistics_module.image_statistics(
        path,
        data_variable="SKY",
        axes=("time", "polarization", "l", "m"),
        chans="1~3",
        mask="MASK_SKY",
        stretch=True,
        statistics=statistics,
        partition_dim="frequency",
        n_partitions=2,
    )
    node_in_memory = direct_statistics(
        dataset,
        data_variable="SKY",
        axes=("time", "polarization", "l", "m"),
        chans="1~3",
        mask="MASK_SKY",
        stretch=True,
        statistics=statistics,
    )
    node_on_disk = direct_statistics(
        path,
        data_variable="SKY",
        axes=("time", "polarization", "l", "m"),
        chans="1~3",
        mask="MASK_SKY",
        stretch=True,
        statistics=statistics,
    )

    xr.testing.assert_allclose(node_on_disk, node_in_memory)
    xr.testing.assert_allclose(distributed, node_in_memory)
