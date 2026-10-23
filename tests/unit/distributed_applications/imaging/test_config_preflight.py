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
def test_generated_continuum_image_records_both_preflight_stages(
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
    approximation = validation["spectral_design"]["taylor_hessian_approximation"]
    assert approximation["approximation"] == "frequency_moment_gram"
    assert approximation["order_zero_noise_amplification"] >= 1.0 - 1e-12
    assert (
        approximation["spatial_blocks"]["approximation"]
        == "circular_gaussian_channel_psfs"
    )
    assert validation["model_products"]["effective_scales"] == [0.0]
    assert validation["model_products"]["taylor_hessian_condition_number"]
    assert (
        validation["model_products"]["taylor_hessian_spatial_blocks"][0]["sampling"]
        == "all_half_fourier_modes"
    )
    assert result["input_validation"] == validation


def test_generated_cube_records_channel_preflight(tmp_path, generated_store):
    import dask
    import xarray as xr

    from astroviper.distributed_applications.imaging.image_cube_single_field import (
        image_cube_single_field,
    )

    processing_set = open_processing_set(str(generated_store))
    image_params = continuum_tests._tw_hydra_image_params(processing_set)
    image_params["n_taylor_terms"] = 1
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
    output = tmp_path / "cube.img.zarr"
    with dask.config.set(scheduler="synchronous"):
        result = image_cube_single_field(
            str(generated_store),
            str(output),
            image_params,
            {"weighting": "natural"},
            controls,
            processing_set_data_group_name="base",
            n_mapping_parallelism={"frequency": 2},
            single_precision_image=True,
        )
    image = xr.open_zarr(output)
    assert len(image.attrs["input_validation"]["cube_channel_sampling"]) == 4
    assert result["input_validation"] == image.attrs["input_validation"]


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


def test_approximate_hessian_failure_precedes_psf_and_output(
    monkeypatch, tmp_path, generated_store
):
    def reject(*args, **kwargs):
        raise ValueError(
            "Approximate Taylor-Hessian preflight: unsafe spectral moments"
        )

    monkeypatch.setattr(
        "astroviper.processing_functions.imaging.config_check.param_check_model_update.check_taylor_hessian_approximation",
        reject,
    )
    output = tmp_path / "rejected-hessian.img.zarr"
    processing_set = open_processing_set(str(generated_store))
    with pytest.raises(ValueError, match="Approximate Taylor-Hessian preflight"):
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
