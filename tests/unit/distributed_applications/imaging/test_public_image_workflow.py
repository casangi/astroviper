"""Generated off-axis data exercise the PDF workflow through the real drivers."""

import dask
import numpy as np
import pytest
from xradio.image import check_image, load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

ARCSEC = np.pi / (180 * 3600)
CENTER = np.array([5.2336866, 0.7109449])


@pytest.fixture(scope="module")
def observation(tmp_path_factory):
    store = str(tmp_path_factory.mktemp("public_workflow") / "input.ps.zarr")
    antennas = read_telescope_layout("vla.d")
    with dask.config.set(scheduler="synchronous"):
        applications.simulation.simulate_processing_set(
            ps_store=store,
            antenna_xds=antennas,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 1800.0,
                "n_samples": 6,
            },
            frequency_params={
                "freq_start": 2e9,
                "freq_delta": 5e8,
                "n_channels": 4,
                "channel_width": 5e8,
            },
            polarization=["RR", "LL"],
            sky_components=[
                {
                    "kind": "point",
                    "flux": 1.0,
                    "ra_dec": CENTER
                    + np.array([120.0, -90.0])
                    * ARCSEC
                    / np.array([np.cos(CENTER[1]), 1.0]),
                }
            ],
            phase_center_ra_dec=CENTER[None, :],
            beam_models=[airy_disk_model("vla_nvss")],
            beam_model_map=np.zeros(antennas.sizes["antenna_name"], int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    return store


def workflow_arguments(tmp_path, observation, mode, restore):
    params = {
        "image_size": [64, 64],
        "cell_size": np.array([-15.0, 15.0]) * ARCSEC,
        "phase_direction": CENTER,
        "frequency_coords": open_processing_set(observation)
        .xr_ps.get_freq_axis()
        .values,
        "polarization_coords": ["I", "V"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }
    if mode != "cube":
        params.update(n_taylor_terms=2, reference_frequency=2.75e9)
    arguments = dict(
        ps_store=observation,
        image_store=str(tmp_path / "output.img.zarr"),
        image_params=params,
        instrument_polarization_basis="circular",
        processing_set_data_group_name="base",
        imaging_weights_params={"weighting": "natural"},
        cycle_control_params={
            "max_iter": 80,
            "max_cycles": 2,
            "max_iter_per_cycle": 40,
            "threshold": 1e-3,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.0,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        image_data_variables_keep=[
            "primary_beam",
            "point_spread_function",
            "beam_fit_params_point_spread_function",
        ],
        single_precision_image=False,
        n_mapping_parallelism={"frequency": 4},
        processing_function_threads=1,
        overwrite=True,
        restore=restore,
        restoration_output="both",
    )
    return arguments


@pytest.mark.parametrize("mode", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize(
    "restore,target,output",
    [
        (False, None, "both"),
        (True, None, "apparent"),
        (True, None, "primary_beam_corrected"),
        (True, None, "both"),
        (True, [300 * ARCSEC, 300 * ARCSEC, 0.0], "both"),
    ],
)
def test_off_axis_workflow_public_outputs(
    tmp_path, observation, mode, restore, target, output, monkeypatch
):
    import importlib

    finalizer = importlib.import_module(
        "astroviper.processing_functions.imaging.finalize_image"
    )
    finalize = finalizer.finalize_image_products
    captured = []

    def record(*args, **kwargs):
        captured.append(kwargs["apparent_inputs"])
        return finalize(*args, **kwargs)

    monkeypatch.setattr(finalizer, "finalize_image_products", record)
    arguments = workflow_arguments(tmp_path, observation, mode, restore)
    arguments["restoring_beam"] = target
    arguments["restoration_output"] = output
    with dask.config.set(scheduler="synchronous"):
        if mode == "cube":
            applications.imaging.image_cube_single_field(
                deconvolver="hogbom", **arguments
            )
        else:
            applications.imaging.image_continuum_single_field(
                deconvolver="mtmfs", continuum_mode=mode, **arguments
            )
    assert captured
    assert all(
        value == {} if restore and output in ("apparent", "both") else value is None
        for value in captured
    )
    image = load_image(arguments["image_store"])
    assert not check_image(image), str(check_image(image))
    assert image.SKY_MODEL.attrs["sky_convention"] == "flat_sky"
    assert image.SKY_RESIDUAL.attrs["residual_representation"] == "coefficient_space"
    assert all(name not in image for name in ("ALPHA", "ALPHA_ERROR", "BETA"))
    for name in ("SKY_RESTORED", "SKY_PRIMARY_BEAM_CORRECTED_RESTORED"):
        requested = output == "both" or output == (
            "apparent" if name == "SKY_RESTORED" else "primary_beam_corrected"
        )
        assert (name in image) == (restore and requested)
    if mode == "cube":
        assert "frequency" in image.PRIMARY_BEAM.dims
    else:
        assert image.SKY_MODEL.sizes["frequency_polynomial"] == 2
        assert image.POINT_SPREAD_FUNCTION.sizes["frequency_polynomial_moment"] == 3
        assert image.PRIMARY_BEAM.dims == image.SKY_MODEL.dims
        assert image.frequency_polynomial.attrs["channel_width"]["data"] == 2e9
        assert image.frequency.attrs["reference_frequency"]["data"] == 2.75e9
        assert image.frequency_polynomial.attrs["reference_frequency"]["data"] == 2.75e9
        if mode == "mvc":
            assert "COMMON_PRIMARY_BEAM" in image
    if target is not None:
        assert (
            image.attrs["data_groups"]["residual"]["beam_fit_params_sky"]
            == "BEAM_FIT_PARAMS_SKY_RESIDUAL"
        )
    if restore and output in ("primary_beam_corrected", "both"):
        from astroviper.processing_functions.imaging.restore import (
            _clean_beam_kernel_ft,
            _convolve_with_clean_beam,
        )

        beam = image.BEAM_FIT_PARAMS_SKY_RESTORED.values
        for ti, fi, pi in np.ndindex(image.SKY_MODEL.shape[:3]):
            fit = beam[ti, fi if mode == "cube" else 0, pi]
            kernel = _clean_beam_kernel_ft(fit, 64, 64, 15 * ARCSEC, np.float64, 1)
            expected = (
                _convolve_with_clean_beam(image.SKY_MODEL.values[ti, fi, pi], kernel, 1)
                + image.SKY_RESIDUAL.values[ti, fi, pi]
            )
            np.testing.assert_allclose(
                image.SKY_PRIMARY_BEAM_CORRECTED_RESTORED.values[ti, fi, pi],
                expected,
                rtol=1e-11,
                atol=1e-13,
            )


def test_persisted_weights_are_written_directly_to_the_processing_set(
    tmp_path, observation
):
    import shutil

    import zarr

    store = str(tmp_path / "persistent.ps.zarr")
    shutil.copytree(observation, store)
    arguments = workflow_arguments(tmp_path, store, "mfs", False)
    with dask.config.set(scheduler="synchronous"):
        applications.imaging.image_continuum_single_field(
            deconvolver="mtmfs",
            continuum_mode="mfs",
            write_imaging_weights_to_ps=True,
            **arguments,
        )
    reopened = open_processing_set(store)
    for name, child in reopened.items():
        assert (
            child.ds.attrs["data_groups"]["base"]["weight_imaging"] == "WEIGHT_IMAGING"
        )
        assert np.isfinite(child.ds.WEIGHT_IMAGING.values).any()
        disk = zarr.open_group(store, path=name, mode="r", use_consolidated=False)
        assert not disk["WEIGHT_IMAGING"].attrs["xradio_incomplete"]
        assert (
            disk["WEIGHT_IMAGING"].attrs["provenance"]["imaging_weights_params"][
                "weighting"
            ]
            == "natural"
        )
        assert disk["WEIGHT_IMAGING"].attrs["provenance"]["data_selection"]
        assert not any(key.startswith("_WEIGHT_IMAGING_") for key in disk.array_keys())
    image = zarr.open_group(arguments["image_store"], mode="r", use_consolidated=False)
    assert "_CONTINUUM_WEIGHT_CACHE" not in image


def test_failed_imaging_graph_does_not_publish_weights(
    tmp_path, observation, monkeypatch
):
    import importlib
    import shutil

    import zarr

    driver = importlib.import_module(
        "astroviper.distributed_applications.imaging.image_continuum_single_field"
    )

    def failed_graph(**kwargs):
        raise RuntimeError("injected graph failure")

    monkeypatch.setattr(driver, "compute_continuum_graph", failed_graph)
    store = str(tmp_path / "failed.ps.zarr")
    shutil.copytree(observation, store)
    arguments = workflow_arguments(tmp_path, store, "mfs", False)
    with (
        dask.config.set(scheduler="synchronous"),
        pytest.raises(RuntimeError, match="injected"),
    ):
        applications.imaging.image_continuum_single_field(
            deconvolver="mtmfs",
            continuum_mode="mfs",
            write_imaging_weights_to_ps=True,
            **arguments,
        )
    for name, child in open_processing_set(store).items():
        assert "weight_imaging" not in child.ds.attrs["data_groups"]["base"]
        disk = zarr.open_group(store, path=name, mode="r", use_consolidated=False)
        assert "WEIGHT_IMAGING" not in disk
        assert not any(key.startswith("_WEIGHT_IMAGING_") for key in disk.array_keys())


@pytest.mark.parametrize("mode", ["cube", "mfs", "mvc"])
@pytest.mark.parametrize(
    "kept",
    [
        None,
        [],
        ["primary_beam"],
        ["point_spread_function"],
        ["primary_beam", "point_spread_function"],
    ],
)
def test_beam_and_psf_retention_is_default_but_optional(
    tmp_path, observation, mode, kept
):
    """Discard optional products only after restoration has used them."""
    arguments = workflow_arguments(tmp_path, observation, mode, restore=True)
    arguments["cycle_control_params"].update(
        max_iter=1, max_cycles=1, max_iter_per_cycle=1
    )
    if kept is None:
        arguments.pop("image_data_variables_keep")
    else:
        arguments["image_data_variables_keep"] = kept
    original = None if kept is None else list(kept)
    with dask.config.set(scheduler="synchronous"):
        if mode == "cube":
            applications.imaging.image_cube_single_field(
                deconvolver="hogbom", **arguments
            )
        else:
            applications.imaging.image_continuum_single_field(
                deconvolver="mtmfs", continuum_mode=mode, **arguments
            )
    image = load_image(arguments["image_store"])
    assert {
        "SKY_MODEL",
        "SKY_RESIDUAL",
        "SKY_RESTORED",
        "SKY_PRIMARY_BEAM_CORRECTED_RESTORED",
    } <= set(image.data_vars)
    for key, name in [
        ("primary_beam", "PRIMARY_BEAM"),
        ("point_spread_function", "POINT_SPREAD_FUNCTION"),
    ]:
        requested = kept is None or key in kept
        assert (name in image) == requested
        assert (key in image.attrs["data_groups"]["residual"]) == requested
    if mode == "mvc":
        assert ("COMMON_PRIMARY_BEAM" in image) == (
            kept is None or "primary_beam" in kept
        )
    if "POINT_SPREAD_FUNCTION" in image and mode != "cube":
        assert image.POINT_SPREAD_FUNCTION.sizes["frequency_polynomial_moment"] == 3
    assert kept == original
