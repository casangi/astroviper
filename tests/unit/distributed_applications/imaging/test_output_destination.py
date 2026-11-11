"""Imaging refuses existing output before opening input data or constructing images."""

from copy import deepcopy

import pytest

from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
    configuration,
)


def options(workflow):
    config = configuration()
    config["image_params"]["fft_padding"] = 1.2
    config["image_data_variables_keep"] = [
        "sky_residual",
        "point_spread_function",
        "sky_model",
    ]
    if workflow.startswith("cube"):
        config["image_params"]["n_taylor_terms"] = 1
        config["deconvolver"] = "hogbom"
        if workflow == "cube_fits":
            config.update(output_image_format="fits", skunk_works=True)
        return image_cube_single_field, config
    config["continuum_mode"] = workflow
    return image_continuum_single_field, config


@pytest.mark.parametrize("workflow", ["cube", "cube_fits", "mfs", "mvc"])
@pytest.mark.parametrize("input_checked", [False, True])
def test_existing_destination_precedes_input_io(
    tmp_path, monkeypatch, workflow, input_checked
):
    driver, config = options(workflow)
    config["input_checked"] = input_checked
    original = deepcopy(config)
    requested = tmp_path / "cube.zarr"
    from xradio.image import image as image_module

    resolver = getattr(image_module, "zarr_store_name", lambda path: path)
    destination = requested
    if workflow != "cube_fits":
        destination = requested.with_name(resolver(requested.name))
    destination.mkdir()
    marker = destination / "marker"
    marker.write_bytes(b"preserve existing output")

    def forbidden_work(*args, **kwargs):
        raise AssertionError("Existing output was checked after input I/O or preflight")

    monkeypatch.setattr("xradio.measurement_set.open_processing_set", forbidden_work)
    monkeypatch.setattr(
        "astroviper.utils.measurement_set_v2.is_measurement_set_v2", forbidden_work
    )
    monkeypatch.setattr(
        "astroviper.node_tasks.imaging.config_check.preflight_imaging_inputs",
        forbidden_work,
    )
    monkeypatch.setattr("xradio.image.make_empty_sky_image", forbidden_work)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        driver(str(tmp_path / "never-opened.ps.zarr"), str(requested), **config)
    assert list(destination.iterdir()) == [marker]
    assert marker.read_bytes() == b"preserve existing output"
    assert config == original


@pytest.mark.parametrize("workflow", ["cube", "cube_fits", "mfs", "mvc"])
def test_overwrite_allows_input_stage_without_early_deletion(
    tmp_path, monkeypatch, workflow
):
    driver, config = options(workflow)
    path = tmp_path / "cube.img.zarr"
    path.mkdir()
    marker = path / "marker"
    marker.write_bytes(b"keep until writer")

    class ReachedInput(Exception):
        pass

    def stop_at_input(*args, **kwargs):
        raise ReachedInput

    monkeypatch.setattr(
        "astroviper.utils.measurement_set_v2.is_measurement_set_v2", stop_at_input
    )
    with pytest.raises(ReachedInput):
        driver(str(tmp_path / "input.ps.zarr"), str(path), overwrite=True, **config)
    assert list(path.iterdir()) == [marker]
    assert marker.read_bytes() == b"keep until writer"
