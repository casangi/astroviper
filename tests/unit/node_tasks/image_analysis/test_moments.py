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
    assert isinstance(frame, pd.DataFrame)
    assert frame.loc[0, "task_id"] == 17
    for key in ("T_load", "T_moments", "T_write", "T_moments_task"):
        assert frame.loc[0, key] >= 0
