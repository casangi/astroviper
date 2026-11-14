"""Unit tests for :mod:`astroviper.utils.node_task_io`."""

import numpy as np
import pytest
import xarray as xr

from astroviper.utils import node_task_io


@pytest.fixture
def image_store(tmp_path):
    from tests.unit.processing_functions.image_analysis.moments_test_utils import (
        make_test_image_xds,
        write_test_image,
    )

    path = str(tmp_path / "image.img.zarr")
    write_test_image(make_test_image_xds(n_frequency=5, n_l=12, n_m=10), path)
    return path


def test_open_image_engines_agree(image_store):
    chunk = node_task_io.open_image("chunk", image_store)
    xradio = node_task_io.open_image("xradio", image_store)
    xr.testing.assert_identical(chunk.load(), xradio.load())


def test_load_image_engines_agree(image_store):
    selection = {"frequency": slice(1, 4), "m": 3}
    chunk = node_task_io.load_image("chunk", image_store, selection)
    xradio = node_task_io.load_image("xradio", image_store, selection)
    xr.testing.assert_identical(chunk.load(), xradio.load())


def test_refuses_unknown_engine(image_store):
    with pytest.raises(ValueError, match="io_engine must be"):
        node_task_io.open_image("zarr", image_store)


def test_stored_data_group(tmp_path):
    import json
    import os

    ms = tmp_path / "ps.zarr" / "ms0"
    os.makedirs(ms)
    group = {"correlated_data": "VISIBILITY", "uvw": "UVW"}
    (ms / "zarr.json").write_text(
        json.dumps({"attributes": {"data_groups": {"base": group}}})
    )
    assert (
        node_task_io._stored_data_group(str(tmp_path / "ps.zarr"), "ms0", "base")
        == group
    )


def test_load_processing_set_engines_agree(tmp_path):
    from xradio.measurement_set import open_processing_set

    from tests.unit.distributed_applications.imaging.test_image_cube_single_field import (
        _simulate_point_source,
    )

    store = _simulate_point_source(tmp_path)
    ps = open_processing_set(store)
    ms_name = next(iter(ps))
    frequencies = ps[ms_name].frequency.values
    selection = {ms_name: {"frequency": slice(0, frequencies.size)}}
    common = dict(frequency_coords=frequencies, processing_function_threads=1)
    chunk = node_task_io.load_processing_set(
        "chunk", store, selection, "base", **common
    )
    xradio = node_task_io.load_processing_set(
        "xradio", store, selection, "base", **common
    )
    group = ps[ms_name].attrs["data_groups"]["base"]
    for role in ("correlated_data", "weight", "flag", "uvw"):
        ours, reference = chunk[ms_name][group[role]], xradio[ms_name][group[role]]
        assert ours.dtype == reference.dtype, role
        np.testing.assert_array_equal(ours.values, reference.values)
    without = node_task_io.load_processing_set(
        "xradio", store, selection, "base", load_correlated_data=False, **common
    )
    assert group["correlated_data"] not in without[ms_name]
