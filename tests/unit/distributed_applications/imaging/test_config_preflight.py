"""Public imaging drivers preserve preflight diagnostics in generated images."""

import pytest
from xradio.measurement_set import open_processing_set

from tests.unit.distributed_applications.imaging import (
    test_continuum as continuum_tests,
)


@pytest.fixture(scope="module")
def generated_store(tmp_path_factory):
    return continuum_tests.wideband_store.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
@pytest.mark.parametrize(
    "weighting",
    [
        {"weighting": "natural"},
        {"weighting": "uniform", "weighting_scope": "global"},
        {"weighting": "briggs", "robust": 0.5, "weighting_scope": "local"},
    ],
)
def test_generated_continuum_image_records_coordinate_and_actual_product_checks(
    tmp_path, generated_store, continuum_mode, weighting
):
    processing_set = open_processing_set(str(generated_store))
    result, image = continuum_tests._run_tw_hydra_continuum(
        generated_store,
        tmp_path / "validated.img.zarr",
        processing_set,
        2,
        continuum_mode,
        weighting,
        deconvolver="mtmfs",
    )
    validation = image.attrs["input_validation"]
    assert validation["spectral_design"]["spectral_design_rank"] == 2
    spectral = validation["spectral_design"]
    assert spectral["weight_assumption"] == "equal"
    assert spectral["support_assumption"] == "selected_frequency_coordinates"
    assert spectral["spectral_design_rank_precision"] == "float64"
    assert "taylor_hessian_approximation" not in spectral
    assert validation["model_products"]["effective_scales"] == [0.0]
    assert validation["model_products"]["taylor_hessian_condition_number"]
    assert (
        validation["model_products"]["taylor_hessian_spatial_blocks"][0]["sampling"]
        == "all_half_fourier_modes"
    )
    assert result["input_validation"] == validation


def _run_cube(store, output, image_params, *, single_precision_image):
    """Run the cube entry point with a fixed zero-iteration configuration."""
    import dask
    import xarray as xr

    from astroviper.distributed_applications.imaging import image_cube_single_field

    controls = {
        "max_iter": 0,
        "max_cycles": 0,
        "threshold": 0.0,
        "gain": 0.1,
        "psf_sidelobe_factor": 1.0,
        "max_iter_per_cycle": -1,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
    }
    with dask.config.set(scheduler="synchronous"):
        result = image_cube_single_field(
            str(store),
            str(output),
            image_params,
            {"weighting": "natural"},
            controls,
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 2},
            single_precision_image=single_precision_image,
            processing_function_threads=1,
        )
    return result, xr.open_zarr(output)


def test_generated_cube_records_channel_preflight(tmp_path, generated_store):
    processing_set = open_processing_set(str(generated_store))
    image_params = continuum_tests._tw_hydra_image_params(processing_set)
    image_params["n_taylor_terms"] = 1
    result, image = _run_cube(
        generated_store,
        tmp_path / "cube.img.zarr",
        image_params,
        single_precision_image=True,
    )
    assert len(image.attrs["input_validation"]["cube_channel_sampling"]) == 4
    assert result["input_validation"] == image.attrs["input_validation"]
    image.close()


def test_spectral_failure_leaves_no_output_or_graph(
    monkeypatch, tmp_path, generated_store
):
    def reject(*args, **kwargs):
        raise ValueError("Spectral-design preflight: deliberately unusable selection")

    monkeypatch.setattr(
        "astroviper.node_tasks.imaging.config_check.preflight_imaging_inputs", reject
    )
    output = tmp_path / "rejected.img.zarr"
    processing_set = open_processing_set(str(generated_store))
    with pytest.raises(ValueError, match="Spectral-design preflight"):
        continuum_tests._run_tw_hydra_continuum(
            generated_store,
            output,
            processing_set,
            2,
            "mfs",
            {"weighting": "natural"},
            deconvolver="mtmfs",
        )
    assert not output.exists()


@pytest.mark.parametrize("continuum_mode", ["mfs", "mvc"])
def test_driver_never_requests_a_bulk_gaussian_preflight(
    monkeypatch, tmp_path, generated_store, continuum_mode
):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Coordinate preflight requested a Gaussian PSF approximation"
        )

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.config_check.param_check_model_update.check_taylor_hessian_approximation",
        forbidden,
    )
    output = tmp_path / "coordinate-preflight.img.zarr"
    processing_set = open_processing_set(str(generated_store))
    _, image = continuum_tests._run_tw_hydra_continuum(
        generated_store,
        output,
        processing_set,
        2,
        continuum_mode,
        {"weighting": "natural"},
        deconvolver="mtmfs",
    )
    assert output.exists()
    assert image.attrs["input_validation"]["model_products"][
        "taylor_hessian_condition_number"
    ]


def test_continuum_rejects_msv2_through_shared_storage_check_before_io(
    monkeypatch, tmp_path
):
    from astroviper.distributed_applications.imaging import image_continuum_single_field
    from astroviper.processing_functions.imaging.config_check import (
        param_check_driver_workflow,
    )
    from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
        configuration,
    )

    source = tmp_path / "input.ms"
    source.mkdir()
    (source / "table.dat").write_bytes(b"")
    output = tmp_path / "never-created.img.zarr"
    original = param_check_driver_workflow.check_input_storage
    calls = []

    def check_storage(config, **kwargs):
        calls.append(kwargs)
        return original(config, **kwargs)

    def forbid_open(*args, **kwargs):
        raise AssertionError("MSv2 rejection must precede opening the processing set")

    monkeypatch.setattr(
        param_check_driver_workflow, "check_input_storage", check_storage
    )
    monkeypatch.setattr("xradio.measurement_set.open_processing_set", forbid_open)
    with pytest.raises(ValueError, match="Measurement Set v2"):
        image_continuum_single_field(str(source), str(output), **configuration())
    assert calls == [{"input_is_msv2": True, "allow_msv2": False}]
    assert not output.exists()
    assert sorted(p.name for p in source.iterdir()) == ["table.dat"]


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize("dimension", ["dim_0", "stokes"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_bad_polarization_dimensions_fail_before_io(
    monkeypatch, tmp_path, workflow, dimension, existing_output
):
    import xarray as xr

    from astroviper.distributed_applications.imaging import (
        image_continuum_single_field,
        image_cube_single_field,
    )
    from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
        configuration,
    )

    config = configuration()
    coordinates = xr.DataArray(["I", "Q"], dims=dimension)
    original = coordinates.copy(deep=True)
    config["image_params"]["polarization_coords"] = coordinates
    if workflow == "cube":
        config["image_params"]["n_taylor_terms"] = 1
        config["deconvolver"] = "hogbom"
        driver = image_cube_single_field
    else:
        config["continuum_mode"] = workflow
        driver = image_continuum_single_field
    output = tmp_path / "rejected.img.zarr"
    if existing_output:
        output.mkdir()
        (output / "sentinel").write_bytes(b"preserve existing output")

    def forbidden_io(*args, **kwargs):
        raise AssertionError("Invalid polarization dimensions must fail before I/O")

    monkeypatch.setattr("xradio.measurement_set.open_processing_set", forbidden_io)
    monkeypatch.setattr("astroviper.utils.io.write_zarr_image_store", forbidden_io)
    monkeypatch.setattr(
        "astroviper.utils.io.create_empty_data_variables_on_disk", forbidden_io
    )
    with pytest.raises(
        ValueError, match="polarization_coords DataArray must have dimensions"
    ):
        driver(str(tmp_path / "never-opened.ps.zarr"), str(output), **config)
    if existing_output:
        assert (output / "sentinel").read_bytes() == b"preserve existing output"
        assert list(output.iterdir()) == [output / "sentinel"]
    else:
        assert not output.exists()
    xr.testing.assert_identical(coordinates, original)


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
def test_accepted_polarization_containers_produce_identical_images(
    tmp_path, generated_store, workflow
):
    import numpy as np
    import xarray as xr

    processing_set = open_processing_set(str(generated_store))
    images = []
    for coordinates in (
        ["I", "Q"],
        np.array(["I", "Q"]),
        xr.DataArray(["I", "Q"], dims="polarization"),
    ):
        original = (
            coordinates.copy(deep=True)
            if isinstance(coordinates, xr.DataArray)
            else np.array(coordinates, copy=True)
        )
        params = continuum_tests._tw_hydra_image_params(
            processing_set, {"polarization_coords": coordinates}
        )
        output = tmp_path / f"polarization-{len(images)}.img.zarr"
        if workflow == "cube":
            params["n_taylor_terms"] = 1
            _, image = _run_cube(
                generated_store, output, params, single_precision_image=False
            )
        else:
            _, image = continuum_tests._run_tw_hydra_continuum(
                generated_store,
                output,
                processing_set,
                2,
                workflow,
                {"weighting": "natural"},
                image_params=params,
            )
        images.append(image.load())
        image.close()
        if isinstance(coordinates, xr.DataArray):
            xr.testing.assert_identical(coordinates, original)
        else:
            np.testing.assert_array_equal(coordinates, original)
    xr.testing.assert_equal(images[0], images[1])
    xr.testing.assert_equal(images[0], images[2])


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize(
    "box",
    [
        (0, 97, 0, 48),
        (0, 96, 0, 49),
        (2, 2, 0, 1),
        (0, 1, 2, 2),
        (10, 9, 0, 1),
        (-1, 96, 0, 48),
    ],
)
@pytest.mark.parametrize("existing_output", [False, True])
def test_invalid_clean_boxes_fail_before_io(
    monkeypatch, tmp_path, workflow, box, existing_output
):
    from astroviper.distributed_applications.imaging import (
        image_continuum_single_field,
        image_cube_single_field,
    )
    from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
        configuration,
    )

    config = configuration()
    config["image_params"].update(image_size=[48, 96], n_taylor_terms=1)
    config["deconvolver"] = "hogbom"
    config["cycle_control_params"]["clean_box"] = box
    if workflow == "cube":
        driver = image_cube_single_field
    else:
        driver = image_continuum_single_field
        config["continuum_mode"] = workflow
    output = tmp_path / "rejected.img.zarr"
    if existing_output:
        output.mkdir()
        (output / "sentinel").write_bytes(b"existing image")

    def forbidden_io(*args, **kwargs):
        raise AssertionError("Invalid clean_box must fail before I/O")

    monkeypatch.setattr("xradio.measurement_set.open_processing_set", forbidden_io)
    monkeypatch.setattr("astroviper.utils.io.write_zarr_image_store", forbidden_io)
    monkeypatch.setattr(
        "astroviper.utils.io.create_empty_data_variables_on_disk", forbidden_io
    )
    with pytest.raises(ValueError, match="clean_box"):
        driver(str(tmp_path / "never-opened.ps.zarr"), str(output), **config)
    if existing_output:
        assert (output / "sentinel").read_bytes() == b"existing image"
        assert list(output.iterdir()) == [output / "sentinel"]
    else:
        assert not output.exists()


@pytest.mark.parametrize("workflow", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize("weighting", ["briggs", "briggs_abs"])
@pytest.mark.parametrize("existing_output", [False, True])
def test_missing_briggs_robust_fails_before_io(
    monkeypatch, tmp_path, workflow, weighting, existing_output
):
    from astroviper.distributed_applications.imaging import (
        image_continuum_single_field,
        image_cube_single_field,
    )
    from tests.unit.processing_functions.imaging.config_check.test_full_validation import (
        configuration,
    )

    config = configuration()
    config["imaging_weights_params"] = {"weighting": weighting}
    if weighting == "briggs_abs":
        config["imaging_weights_params"]["briggs_abs_noise"] = 1.0
    if workflow == "cube":
        config["image_params"]["n_taylor_terms"] = 1
        config["deconvolver"] = "hogbom"
        driver = image_cube_single_field
    else:
        config["continuum_mode"] = workflow
        driver = image_continuum_single_field
    output = tmp_path / "missing-robust.img.zarr"
    if existing_output:
        output.mkdir()
        (output / "sentinel").write_bytes(b"existing output")

    def forbidden_io(*args, **kwargs):
        raise AssertionError("Missing robust must fail before input reads or writes")

    monkeypatch.setattr("xradio.measurement_set.open_processing_set", forbidden_io)
    monkeypatch.setattr("astroviper.utils.io.write_zarr_image_store", forbidden_io)
    monkeypatch.setattr(
        "astroviper.utils.io.create_empty_data_variables_on_disk", forbidden_io
    )
    with pytest.raises(KeyError, match="robust"):
        driver(str(tmp_path / "never-opened.ps.zarr"), str(output), **config)
    assert "robust" not in config["imaging_weights_params"]
    if existing_output:
        assert (output / "sentinel").read_bytes() == b"existing output"
        assert list(output.iterdir()) == [output / "sentinel"]
    else:
        assert not output.exists()
