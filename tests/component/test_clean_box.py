"""Explicit image-edge CLEAN boxes match whole-image cube/MFS/MVC deconvolution."""

from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from tests.unit.distributed_applications.imaging.test_continuum import (
    _tw_hydra_image_params,
)


@pytest.fixture
def imaging_input(tmp_path, tw_hydra_source_store):
    with TemporaryDirectory(prefix="clean_box_input_", dir=tmp_path) as directory:
        store = Path(directory) / "input.ps.zarr"
        copytree(tw_hydra_source_store, store)
        source = open_processing_set(str(store))
        yield store, _tw_hydra_image_params(source)


def _run(store, output, params, workflow, box):
    """Image and restore with one fixed Högbom cycle and the supplied rectangle."""
    controls = {
        "max_iter": 2,
        "max_cycles": 1,
        "max_iter_per_cycle": 2,
        "threshold": 0.0,
        "gain": 0.1,
        "clean_box": box,
        "psf_sidelobe_factor": 1.5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    options = {
        "image_params": params,
        "imaging_weights_params": {"weighting": "natural"},
        "cycle_control_params": controls,
        "deconvolver": "hogbom",
        "restore": True,
        "processing_set_data_group_name": "base",
        "n_mapping_parallelism": {"frequency": 2},
        "single_precision_image": False,
        "processing_function_threads": 1,
    }
    if workflow == "cube":
        driver = image_cube_single_field
    else:
        driver = image_continuum_single_field
        options["continuum_mode"] = workflow
        options["weight_memory_mode"] = "in_memory"
    with dask.config.set(scheduler="synchronous"):
        driver(str(store), str(output), **options)
    with xr.open_zarr(output) as stored:
        return stored.load()


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize("image_size", [[48, 96], [96, 48]])
def test_image_edge_box_matches_whole_image(
    imaging_input, tmp_path, workflow, image_size
):
    store, params = imaging_input
    params = {**params, "image_size": image_size, "n_taylor_terms": 1}
    n_l, n_m = image_size
    expected = _run(store, tmp_path / "default.img.zarr", params, workflow, None)
    actual = _run(
        store, tmp_path / "explicit.img.zarr", params, workflow, (0, n_m, 0, n_l)
    )
    assert np.isfinite(actual.SKY_MODEL).all()
    assert np.max(np.abs(actual.SKY_MODEL.values)) > 0
    xr.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
def test_iso_time_matches_mjd_image(imaging_input, tmp_path, workflow):
    store, params = imaging_input
    params = {
        **params,
        "image_size": [48, 48],
        "n_taylor_terms": 1,
        "time_coords": [58849.5],
    }
    expected = _run(store, tmp_path / "mjd.img.zarr", params, workflow, None)
    params = {**params, "time_coords": "2020-01-01T12:00:00"}
    actual = _run(store, tmp_path / "iso.img.zarr", params, workflow, None)
    xr.testing.assert_equal(actual, expected)
    assert params["time_coords"] == "2020-01-01T12:00:00"


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize("requested_name", ["existing.zarr", "existing.img.zarr"])
def test_existing_image_is_rejected_before_input_reopens(
    imaging_input, tmp_path, monkeypatch, workflow, requested_name
):
    from xradio.image import image as image_module

    if requested_name == "existing.zarr" and not hasattr(
        image_module, "zarr_store_name"
    ):
        pytest.skip("XRADIO keeps the requested name, so these are distinct outputs")
    store, params = imaging_input
    params = {**params, "image_size": [48, 48], "n_taylor_terms": 1}
    output = tmp_path / "existing.img.zarr"
    _run(store, output, params, workflow, None)
    before = {
        str(path.relative_to(output)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }

    def forbidden_input(*args, **kwargs):
        raise AssertionError("An existing output must be rejected before input I/O")

    monkeypatch.setattr("xradio.measurement_set.open_processing_set", forbidden_input)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        _run(store, tmp_path / requested_name, params, workflow, None)
    after = {
        str(path.relative_to(output)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before
