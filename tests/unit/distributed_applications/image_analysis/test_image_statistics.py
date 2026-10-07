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
    statistics = ("mean", "max", "maxpos", "n_pixels")
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


@pytest.mark.parametrize("n_partitions", [1, 2, 4])
def test_default_statistics_match_across_layers(image_store, monkeypatch, n_partitions):
    """Keep default output membership, order, and values consistent across layers."""
    from astroviper.node_tasks.image_analysis.image_statistics import (
        image_statistics as direct_statistics,
    )
    from astroviper.processing_functions.image_analysis.statistics import (
        create_statistics_state,
        finalize_statistics_state,
    )

    path, dataset = image_store
    monkeypatch.setattr(
        "xradio.image.load_image",
        lambda store, block_des: xr.open_zarr(store).isel(block_des),
    )
    expected = {
        "min": 0.0,
        "max": 95.0,
        "peak": 95.0,
        "sum": 4560.0,
        "sumsq": 290320.0,
        "n_pixels": 96.0,
        "mean": 47.5,
        "rms": np.sqrt(290320.0 / 96),
        "std": np.std(np.arange(96.0), ddof=0),
        "minpos": [0, 0, 0, 0, 0],
        "maxpos": [0, 3, 1, 2, 3],
    }
    results = [
        statistics_module.image_statistics(
            path, data_variable="SKY", n_partitions=n_partitions
        ),
        direct_statistics(dataset, data_variable="SKY"),
        finalize_statistics_state(
            create_statistics_state(dataset["SKY"], dataset["SKY"].dims)
        ),
    ]
    for result in results:
        assert result["n_pixels"].dtype == np.dtype("float64")
        assert list(result.data_vars) == list(expected)
        for name, value in expected.items():
            np.testing.assert_allclose(result[name].values, value)
    for result in results[1:]:
        xr.testing.assert_identical(results[0], result)


@pytest.mark.parametrize("axes", [("l", "m"), ("frequency", "l", "m")])
@pytest.mark.parametrize("n_partitions", [1, 2])
def test_peak_across_layers_and_partitions(tmp_path, monkeypatch, axes, n_partitions):
    """Keep peak signs and tie ordering with retained or reduced partitions and masks."""
    from astroviper.node_tasks.image_analysis.image_statistics import (
        image_statistics as direct_statistics,
    )
    from astroviper.processing_functions.image_analysis.plane_statistics import (
        calculate_plane_statistics,
    )

    data = xr.DataArray(
        np.array([9.0, -9.0, 100.0, -9.0, 9.0, np.nan]).reshape(1, 2, 1, 1, 3),
        dims=("time", "frequency", "polarization", "l", "m"),
        coords={"time": [0], "frequency": [100, 101], "polarization": ["I"]},
        attrs={"units": "Jy/beam"},
    )
    dataset = xr.Dataset({"SKY_RESIDUAL": data, "MASK": data < 100})
    path = str(tmp_path / "peak.zarr")
    dataset.to_zarr(path)
    monkeypatch.setattr(
        "xradio.image.load_image",
        lambda store, block_des: xr.open_zarr(store).isel(block_des),
    )
    kwargs = dict(
        data_variable="SKY_RESIDUAL", axes=axes, mask="MASK", statistics=("peak",)
    )
    distributed = statistics_module.image_statistics(
        path, n_partitions=n_partitions, **kwargs
    )
    direct = direct_statistics(dataset, **kwargs)
    xr.testing.assert_identical(distributed, direct)
    if "frequency" in axes:
        assert direct["peak"].item() == 9.0
    else:
        plane = calculate_plane_statistics(dataset, mask_name="MASK")["sky_residual"]
        np.testing.assert_allclose(direct["peak"], plane["peak_masked"])
        np.testing.assert_allclose(direct["peak"].values.ravel(), [9.0, -9.0])
