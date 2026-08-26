"""Unit tests for the moments node task (chunk load + science + chunk write)."""

import numpy as np
import pytest
import xarray as xr
from numcodecs import Blosc

from astroviper.node_tasks.image_analysis.moments import moments as moments_node_task
from astroviper.processing_functions.image_analysis.moments import (
    get_moments_data_variable_definitions,
    moment_data_variable_key,
)
from astroviper.processing_functions.image_analysis.moments import (
    moments as moments_processing_function,
)
from tests.unit.processing_functions.image_analysis.moments_test_utils import (
    ALL_MOMENTS,
    make_test_image_xds,
    write_test_image,
)


@pytest.fixture
def input_image(tmp_path):
    """Synthetic image written to a Zarr store; returns (path, in-memory xds)."""
    img_xds = make_test_image_xds()
    path = tmp_path / "input.img.zarr"
    write_test_image(img_xds, path)
    return str(path), img_xds


def test_graph_mode_false_matches_processing_function(input_image):
    path, img_xds = input_image
    chunk = {"m": slice(3, 9)}
    result = moments_node_task(
        input_image_store=path,
        moments_image_store="unused",
        moments=ALL_MOMENTS,
        moment_axis="frequency",
        data_selection={"img": chunk},
        graph_mode=False,
    )
    expected = moments_processing_function(
        img_xds.isel(chunk), moments=ALL_MOMENTS, moment_axis="frequency"
    )
    for name in ALL_MOMENTS:
        variable_name = "SKY_MOMENT_" + name.upper()
        np.testing.assert_allclose(
            result[variable_name].values,
            expected[variable_name].values,
            rtol=1e-6,
            equal_nan=True,
            err_msg=variable_name,
        )


def test_selection_composes_with_data_selection(input_image):
    path, img_xds = input_image
    result = moments_node_task(
        input_image_store=path,
        moments_image_store="unused",
        moments=["mean"],
        moment_axis="polarization",
        selection={"frequency": slice(1, 5)},
        data_selection={"img": {"m": slice(0, 7)}},
        graph_mode=False,
    )
    expected = moments_processing_function(
        img_xds.isel({"frequency": slice(1, 5), "m": slice(0, 7)}),
        moments=["mean"],
        moment_axis="polarization",
    )
    np.testing.assert_allclose(
        result.SKY_MOMENT_MEAN.values,
        expected.SKY_MOMENT_MEAN.values,
        rtol=1e-6,
        equal_nan=True,
    )


def test_extra_variables_are_not_loaded(input_image, tmp_path):
    """A store with unrelated variables still works (only sky/mask are read)."""
    path, img_xds = input_image
    extra = img_xds.copy()
    extra["SOMETHING_ELSE"] = xr.DataArray(
        np.zeros(img_xds.SKY.shape, dtype=np.float64), dims=img_xds.SKY.dims
    )
    extra_path = tmp_path / "extra.img.zarr"
    write_test_image(extra, extra_path)
    result = moments_node_task(
        input_image_store=str(extra_path),
        moments_image_store="unused",
        moments=["rms"],
        moment_axis="frequency",
        data_selection={"img": {}},
        graph_mode=False,
    )
    assert "SKY_MOMENT_RMS" in result.data_vars
    assert "SOMETHING_ELSE" not in result.data_vars


def test_graph_mode_writes_chunks_into_preallocated_store(input_image, tmp_path):
    """Two manually-driven node tasks must tile the full moments store."""
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord
    from xradio.image import write_image

    from astroviper.processing_functions.image_analysis.moments import (
        collapsed_moment_axis_coords,
    )
    from astroviper.utils.io import create_empty_data_variables_on_disk

    path, img_xds = input_image
    out_path = str(tmp_path / "moments.img.zarr")
    moment_names = ["mean", "weighted_coord", "median"]
    keys = [moment_data_variable_key(name) for name in moment_names]

    # Pre-allocate the output store the way the distributed application does.
    parallel_coords = {"m": make_parallel_coord(coord=img_xds.m, n_chunks=2)}
    skeleton = xr.Dataset(
        coords=collapsed_moment_axis_coords(img_xds, "frequency").coords
    )
    write_image(skeleton, imagename=out_path, out_format="zarr", overwrite=True)
    definitions = get_moments_data_variable_definitions(
        moment_names, list(img_xds.SKY.dims), single_precision_image=True
    )
    create_empty_data_variables_on_disk(
        out_path,
        keys,
        shape_dict={**dict(img_xds.sizes), "frequency": 1},
        parallel_coords=parallel_coords,
        compressor=Blosc(cname="lz4", clevel=5),
        double_precision=False,
        data_variable_definitions=definitions,
    )

    for chunk_index, chunk_slice in parallel_coords["m"]["data_chunk_slices"].items():
        moments_node_task(
            input_image_store=path,
            moments_image_store=out_path,
            moments=moment_names,
            moment_axis="frequency",
            moments_data_variables=keys,
            task_coords={"m": {"slice": chunk_slice}},
            data_selection={"img": {"m": chunk_slice}},
            task_id=chunk_index,
            graph_mode=True,
        )

    expected = moments_processing_function(
        img_xds, moments=moment_names, moment_axis="frequency"
    )
    written = xr.open_zarr(out_path)
    for name in moment_names:
        variable_name = "SKY_MOMENT_" + name.upper()
        np.testing.assert_allclose(
            written[variable_name].values,
            expected[variable_name].values,
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
            err_msg=variable_name,
        )


def test_returns_timing_frame(input_image, tmp_path):
    import pandas as pd

    path, img_xds = input_image
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord
    from xradio.image import write_image

    from astroviper.processing_functions.image_analysis.moments import (
        collapsed_moment_axis_coords,
    )
    from astroviper.utils.io import create_empty_data_variables_on_disk

    out_path = str(tmp_path / "timing.img.zarr")
    parallel_coords = {"m": make_parallel_coord(coord=img_xds.m, n_chunks=1)}
    skeleton = xr.Dataset(
        coords=collapsed_moment_axis_coords(img_xds, "frequency").coords
    )
    write_image(skeleton, imagename=out_path, out_format="zarr", overwrite=True)
    create_empty_data_variables_on_disk(
        out_path,
        ["sky_moment_mean"],
        shape_dict={**dict(img_xds.sizes), "frequency": 1},
        parallel_coords=parallel_coords,
        compressor=None,
        double_precision=False,
        data_variable_definitions=get_moments_data_variable_definitions(
            ["mean"], list(img_xds.SKY.dims), single_precision_image=True
        ),
    )
    frame = moments_node_task(
        input_image_store=path,
        moments_image_store=out_path,
        moments=["mean"],
        moment_axis="frequency",
        task_coords={"m": {"slice": slice(0, img_xds.sizes["m"])}},
        data_selection={"img": {}},
        task_id=17,
        graph_mode=True,
    )
    # Single-dict convention so graphviper's resource monitor can attach to it.
    assert set(frame) == {"timing_node_tasks"}
    frame = frame["timing_node_tasks"]
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "task_id"] == 17
    for key in ("T_load", "T_moments", "T_write", "T_moments_task"):
        assert frame.loc[0, key] >= 0


# ---------------------------------------------------------------------------
# Streaming read path (memory strategy)
# ---------------------------------------------------------------------------
def test_read_block_covers_block_plus_decode_transient(tmp_path):
    """The block is the largest whole number of chunk/shard lengths whose
    decoded block PLUS zarr decode transient fits the budget; fewer, larger
    requests with an ample budget; never 0."""
    import importlib

    import zarr

    node = importlib.import_module("astroviper.node_tasks.image_analysis.moments")
    _open_image_lazy = node._open_image_lazy
    moment_axis_read_block = node.moment_axis_read_block

    img_xds = make_test_image_xds(n_frequency=12, n_l=16, n_m=16)
    gib = 1024**3
    block_des = {"m": slice(0, 8)}
    # Selected slice: pol(2) x l(16) x m(8) float32 -> 1024 B per plane.

    with zarr.config.set({"async.concurrency": 10}):
        # --- unsharded: chunks (1, 4, 2, 8, 8) -> 2048 B chunks, the m-slice
        # touches 2 l-chunk columns; transient = min(chunks, 10) x 2 x 2048.
        path = str(tmp_path / "chunked.img.zarr")
        img_xds.to_zarr(
            path,
            mode="w",
            zarr_format=3,
            encoding={
                "SKY": {"chunks": (1, 4, 2, 8, 8)},
                "MASK": {"chunks": (1, 4, 2, 8, 8)},
            },
        )
        lazy = _open_image_lazy(path)["SKY"]
        # block(4) = 4096 + 2 chunks in flight x 2 x 2048 = 12288 B;
        # block(8) = 8192 + 4 x 2 x 2048 = 24576 B; block(12) = 36864 B.
        assert moment_axis_read_block(lazy, block_des, "frequency", 1.0) == 12
        assert moment_axis_read_block(lazy, block_des, "frequency", 25000 / gib) == 8
        assert moment_axis_read_block(lazy, block_des, "frequency", 13000 / gib) == 4
        # Budget below one chunk length + transient: whole planes, never 0.
        assert moment_axis_read_block(lazy, block_des, "frequency", 9000 / gib) == 1
        assert moment_axis_read_block(lazy, block_des, "frequency", 0.0) == 1

        # --- sharded: chunks (1, 1, 2, 8, 8) in shards (1, 4, 2, 16, 16):
        # step = one shard (4 planes), 8 inner chunks per shard for this tile,
        # transient = min(shards, 10) x min(8, 10) x 2 x 512 B.
        sharded = str(tmp_path / "sharded.img.zarr")
        img_xds.to_zarr(
            sharded,
            mode="w",
            zarr_format=3,
            encoding={
                "SKY": {"chunks": (1, 1, 2, 8, 8), "shards": (1, 4, 2, 16, 16)},
                "MASK": {"chunks": (1, 1, 2, 8, 8), "shards": (1, 4, 2, 16, 16)},
            },
        )
        lazy = _open_image_lazy(sharded)["SKY"]
        # block(4) = 4096 + 8192 = 12288; block(8) = 8192 + 16384 = 24576;
        # block(12) = 12288 + 24576 = 36864.
        # An ample budget spans MULTIPLE shards (fewer, larger requests).
        assert moment_axis_read_block(lazy, block_des, "frequency", 1.0) == 12
        assert moment_axis_read_block(lazy, block_des, "frequency", 25000 / gib) == 8
        assert moment_axis_read_block(lazy, block_des, "frequency", 13000 / gib) == 4
        # Transient alone exceeds the budget: fall back to 1 plane.
        assert moment_axis_read_block(lazy, block_des, "frequency", 4000 / gib) == 1

    # A non-chunked axis (polarization chunk = full axis) is read in one block.
    with zarr.config.set({"async.concurrency": 10}):
        assert moment_axis_read_block(lazy, {}, "polarization", 1.0) == 2


def test_streaming_path_never_holds_the_chunk_slab(input_image, monkeypatch):
    """For streamable moments the node task must read block-by-block and
    never call the slab loader; median must use the slab loader."""
    import importlib

    node = importlib.import_module("astroviper.node_tasks.image_analysis.moments")

    path, img_xds = input_image
    calls = {"slab": 0}
    original = node._load_chunk_streaming

    def counting_loader(*args, **kwargs):
        calls["slab"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(node, "_load_chunk_streaming", counting_loader)
    common = dict(
        input_image_store=path,
        moments_image_store="unused",
        moment_axis="frequency",
        data_selection={"img": {"m": slice(2, 7)}},
        graph_mode=False,
        use_mask=True,
    )
    streamed = moments_node_task(
        moments=["maximum", "abs_mean_dev", "integrated"],
        memory_budget_gb=2 * 2 * 18 * 5 * 4 / 1024**3,  # two planes per block
        **common,
    )
    assert calls["slab"] == 0
    expected = moments_processing_function(
        img_xds.isel(m=slice(2, 7)),
        moments=["maximum", "abs_mean_dev", "integrated"],
        moment_axis="frequency",
        use_mask=True,
    )
    for name in ("maximum", "abs_mean_dev", "integrated"):
        variable = "SKY_MOMENT_" + name.upper()
        np.testing.assert_allclose(
            streamed[variable].values,
            expected[variable].values,
            rtol=1e-6,
            equal_nan=True,
            err_msg=name,
        )

    moments_node_task(moments=["median"], **common)
    assert calls["slab"] == 1


def test_timing_frame_records_strategy(input_image, tmp_path):
    path, img_xds = input_image
    store = tmp_path / "out.img.zarr"
    moment_names = ["maximum"]
    keys = [moment_data_variable_key(n) for n in moment_names]
    out = moments_processing_function(img_xds, moments=moment_names)
    for v in out.variables:
        out[v].encoding = {}
    out.to_zarr(store, mode="w")
    from graphviper.graph_tools.coordinate_utils import make_parallel_coord

    from astroviper.utils.io import create_empty_data_variables_on_disk

    parallel_coords = {"m": make_parallel_coord(coord=img_xds["m"], n_chunks=2)}
    create_empty_data_variables_on_disk(
        str(store),
        keys,
        shape_dict=dict(out.sizes),
        parallel_coords=parallel_coords,
        compressor=Blosc(cname="lz4", clevel=5),
        double_precision=False,
        data_variable_definitions=get_moments_data_variable_definitions(
            moment_names, list(img_xds["SKY"].dims), True
        ),
    )
    chunk = parallel_coords["m"]["data_chunks"][0]
    df = moments_node_task(
        input_image_store=path,
        moments_image_store=str(store),
        moments=moment_names,
        moment_axis="frequency",
        task_coords={"m": {"data": chunk, "slice": slice(0, len(chunk))}},
        data_selection={"img": {"m": slice(0, len(chunk))}},
        task_id=3,
    )["timing_node_tasks"]
    assert bool(df["streamed"].iloc[0]) is True
    assert df["n_read_block_planes"].iloc[0] >= 1
    # T_moments splits into stream-read + accumulate on the streamed path.
    assert 0.0 <= df["T_stream_read"].iloc[0] <= df["T_moments"].iloc[0]
    assert df["T_accumulate"].iloc[0] == pytest.approx(
        df["T_moments"].iloc[0] - df["T_stream_read"].iloc[0]
    )
    assert {"T_load", "T_moments", "T_write", "T_moments_task"} <= set(df.columns)


def test_dimension_flags_are_sliced_to_the_chunk(input_image):
    """Full-image flags passed to the node task must be restricted to the
    task's chunk selection before the science function sees them."""
    path, img_xds = input_image
    n_freq = img_xds.sizes["frequency"]
    flags = np.zeros(n_freq, dtype=bool)
    flags[[0, n_freq - 1]] = True  # spw-edge style flags
    chunk = {"m": slice(3, 9)}
    result = moments_node_task(
        input_image_store=path,
        moments_image_store="unused",
        moments=["maximum", "mean"],
        moment_axis="frequency",
        data_selection={"img": chunk},
        graph_mode=False,
        dimension_flags={"frequency": flags},
    )
    expected = moments_processing_function(
        img_xds.isel(chunk),
        moments=["maximum", "mean"],
        moment_axis="frequency",
        dimension_flags={"frequency": flags},
    )
    for name in ("maximum", "mean"):
        variable = "SKY_MOMENT_" + name.upper()
        np.testing.assert_allclose(
            result[variable].values,
            expected[variable].values,
            rtol=1e-6,
            equal_nan=True,
            err_msg=name,
        )
    # Flags along the chunked (parallel) dimension are sliced with it.
    result_m = moments_node_task(
        input_image_store=path,
        moments_image_store="unused",
        moments=["maximum"],
        moment_axis="frequency",
        data_selection={"img": chunk},
        graph_mode=False,
        dimension_flags={"m": [[0, 5]]},  # global m indices 0..4 -> chunk 3,4
    )
    expected_m = moments_processing_function(
        img_xds.isel(chunk),
        moments=["maximum"],
        moment_axis="frequency",
        dimension_flags={"m": [[0, 2]]},
    )
    np.testing.assert_allclose(
        result_m["SKY_MOMENT_MAXIMUM"].values,
        expected_m["SKY_MOMENT_MAXIMUM"].values,
        rtol=1e-6,
        equal_nan=True,
    )
