"""Tests of the find_continuum node tasks."""

import numpy as np
import pytest
from xradio.image import write_image

from astroviper.node_tasks.imaging.find_continuum import (
    find_continuum_cube_reductions,
    merge_find_continuum_cube_reductions,
    read_small_image,
    write_small_image,
)
from astroviper.processing_functions.imaging.find_continuum import (
    finalize_find_continuum_reductions,
)
from astroviper.processing_functions.imaging.find_continuum import (
    find_continuum_cube_reductions as reduce_chunk,
)
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    make_find_continuum_image_xds,
)

N_CHANNELS = 20
VARIABLES = {"cube": "SKY_RESIDUAL", "pb": "PRIMARY_BEAM"}


@pytest.fixture
def cube_store(tmp_path):
    line = dict(
        channel=10,
        fwhm_channels=3,
        peak_jy=0.05,
        offset_pixels=(0, 0),
        size_fwhm_pixels=3,
    )
    img_xds = make_find_continuum_image_xds(
        n_pixels=16, n_channels=N_CHANNELS, lines=(line,)
    )
    path = str(tmp_path / "cube.img.zarr")
    write_image(img_xds, imagename=path, out_format="zarr", overwrite=True)
    return path, img_xds


def test_small_image_round_trip(tmp_path):
    values = np.arange(24.0).reshape(4, 3, 1, 2)
    mask = values % 5 != 0
    metadata = {"shape": [4, 3, 1, 2], "brightness_unit": "Jy/beam", "beams": None}
    path = str(tmp_path / "small.zarr")
    write_small_image(path, values, mask, metadata)
    read_values, read_mask, read_metadata = read_small_image(path)
    np.testing.assert_array_equal(read_values, values)
    np.testing.assert_array_equal(read_mask, mask)
    assert read_metadata == metadata


def test_map_and_reduce_tasks_match_one_reduction(cube_store, tmp_path):
    store, img_xds = cube_store
    mask_path = str(tmp_path / "mask.zarr")
    mask_values = np.zeros((16, 16, 1, 1))
    mask_values[4:12, 4:12] = 1.0
    write_small_image(
        mask_path, mask_values, np.ones(mask_values.shape, dtype=bool), {}
    )
    requests = [
        {
            "kind": "moments",
            "image": "cube",
            "mask": '"mask_image">0.5 && "pb">0.2',
            "channels": list(range(5, 16)),
            "moments": [0, 8],
            "frequency": img_xds.frequency.values.tolist(),
            "rest_frequency": float(img_xds.frequency.attrs["rest_frequency"]["data"]),
        },
        {
            "kind": "profile",
            "image": "cube",
            "mask": '"mask_image">0.5',
            "channels": None,
            "function": "max",
        },
    ]
    chunks = [slice(0, 7), slice(7, 14), slice(14, N_CHANNELS)]
    outputs = [
        find_continuum_cube_reductions(
            store,
            VARIABLES,
            {"mask_image": mask_path},
            requests,
            data_selection={"img": {"frequency": chunk}},
            task_id=index,
        )
        for index, chunk in enumerate(chunks)
    ]
    merged = merge_find_continuum_cube_reductions(
        [outputs[0], merge_find_continuum_cube_reductions(outputs[1:], {})], {}
    )
    result = finalize_find_continuum_reductions(merged["reductions"], requests)

    images = {"mask_image": mask_values}
    expected = finalize_find_continuum_reductions(
        reduce_chunk(
            img_xds.sel(polarization=["I"]),
            requests,
            VARIABLES,
            images,
            np.arange(N_CHANNELS),
        ),
        requests,
    )
    for moment in (0, 8):
        np.testing.assert_allclose(
            result[0][moment], expected[0][moment], equal_nan=True
        )
    np.testing.assert_allclose(result[1], expected[1])

    timing = merged["timing_node_tasks"]
    assert list(timing["task_id"]) == [0, 1, 2]
    assert list(timing["n_channels"]) == [7, 7, 6]
    assert {
        "T_load",
        "T_reduce",
        "T_find_continuum_task",
        "hostname",
        "worker_name",
    } <= set(timing)


def test_primary_beam_limit_reads_the_primary_beam(cube_store):
    store, img_xds = cube_store
    request = {
        "kind": "statistics",
        "image": "cube",
        "mask": "",
        "channels": None,
        "per_channel": False,
    }
    output = find_continuum_cube_reductions(
        store,
        {"cube": "SKY_RESIDUAL"},
        {},
        [request],
        primary_beam="PRIMARY_BEAM",
        primary_beam_limit=0.9999,
    )
    beam = img_xds["PRIMARY_BEAM"].values
    assert output["reductions"][0]["npts"] == (beam >= 0.9999).sum()


def test_whole_store_without_selection(cube_store):
    store, _ = cube_store
    request = {
        "kind": "statistics",
        "image": "cube",
        "mask": "",
        "channels": [3],
        "per_channel": False,
    }
    output = find_continuum_cube_reductions(
        store, {"cube": "SKY_RESIDUAL"}, {}, [request]
    )
    assert output["reductions"][0]["npts"] == 16 * 16
    assert output["timing_node_tasks"]["n_channels"].item() == N_CHANNELS
