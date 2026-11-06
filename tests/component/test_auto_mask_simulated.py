"""Imaging with an automatic clean mask on the simulated VLA D input of
:mod:`tests.component.test_continuum_asp_simulated` (a 2 Jy point source and a
1 Jy Gaussian, flat spectra, two channels, thermal noise).

With ``auto_mask_params={}`` the cube and the continuum applications must find
both sources in Stokes I by themselves, and nothing in Stokes V, where the mask
stays empty and the plane stops with a zero mask. The mask must hold the peak
of each source and stay below 8 percent of the image (measured: at most 5.8),
and the model of each source must hold its apparent flux within 5 percent
(measured: within 2.9 percent). Hogbom may not put a component outside the mask.
The arguments that exclude each other are refused before anything is written.
"""

import shutil

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging import (
    image_continuum_single_field,
    image_cube_single_field,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    IMAGING_ZERO_MASK,
)
from astroviper.utils.beam_models import airy_disk_model
from tests.component.test_continuum_asp_simulated import (
    CELL_SIZE,
    IMAGE_SIZE,
    simulate_input,
    source_boxes,
)

CYCLE_CONTROL = {
    "max_iter": 1000,
    "max_cycles": 20,
    "max_iter_per_cycle": 200,
    "threshold": 0.0,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.0,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}
MAX_MASK_FRACTION = 0.08
FLUX_TOLERANCE = 0.05


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """The simulated processing set and the true sky on the imaging grid."""
    return simulate_input(tmp_path_factory.mktemp("auto_mask"))


def image_params(ps_store):
    processing_set = open_processing_set(ps_store)
    fields = processing_set.xr_ps.get_combined_field_and_source_xds()
    return {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": fields.FIELD_PHASE_CENTER_DIRECTION.sel(
            field_name=fields.attrs["center_field_name"]
        ).values,
        "frequency_coords": processing_set.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "V"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }


def run_cube(simulated, directory, deconvolver, **kwargs):
    source_store, _ = simulated
    ps_store = str(directory / "input.ps.zarr")
    shutil.copytree(source_store, ps_store)
    image_store = str(directory / "cube.img.zarr")
    arguments = {
        "cycle_control_params": CYCLE_CONTROL,
        "auto_mask_params": {},
        **kwargs,
    }
    with dask.config.set(scheduler="synchronous"):
        image_cube_single_field(
            ps_store=ps_store,
            image_store=image_store,
            image_params=image_params(ps_store),
            imaging_weights_params={"weighting": "natural"},
            deconvolver=deconvolver,
            instrument_polarization_basis="circular",
            primary_beam_params={"beam_model": airy_disk_model("vla_nvss")},
            processing_set_data_group_name="base",
            image_data_variables_keep=[
                "sky_model",
                "sky_residual",
                "primary_beam",
                "mask",
            ],
            single_precision_image=False,
            n_mapping_parallelism={"frequency": 2},
            overwrite=True,
            **arguments,
        )
    with xr.open_zarr(image_store) as stored:
        return stored.load()


def run_continuum(simulated, directory, continuum_mode, deconvolver, n_taylor_terms=1):
    source_store, _ = simulated
    ps_store = str(directory / "input.ps.zarr")
    shutil.copytree(source_store, ps_store)
    image_store = str(directory / "continuum.img.zarr")
    params = image_params(ps_store)
    with dask.config.set(scheduler="synchronous"):
        result = image_continuum_single_field(
            ps_store=ps_store,
            image_store=image_store,
            image_params={
                **params,
                "cpp_gridder": True,
                "n_taylor_terms": n_taylor_terms,
                "reference_frequency": float(np.mean(params["frequency_coords"])),
            },
            imaging_weights_params={
                "weighting": "natural",
                "casa_weighting_implementation": True,
            },
            cycle_control_params={**CYCLE_CONTROL, "max_cycles": 8},
            gridder="anti_alias",
            deconvolver=deconvolver,
            auto_mask_params={},
            continuum_mode=continuum_mode,
            instrument_polarization_basis="circular",
            restore=True,
            primary_beam_limit=0.2,
            scan_intents=["OBSERVE_TARGET#ON_SOURCE"],
            processing_set_data_group_name="base",
            image_data_variables_keep=[
                "sky_model",
                "sky_residual",
                "primary_beam",
                "mask",
            ],
            single_precision_image=False,
            processing_function_threads=1,
            n_mapping_parallelism={"frequency": 2},
            memory_mode="in_memory",
            weight_memory_mode="in_memory",
            visibility_memory_mode="recompute",
            wideband_primary_beam_memory_mode="in_memory",
            overwrite=True,
            compute_backend="dask",
        )
    with xr.open_zarr(image_store) as stored:
        return result, stored.load()


def assert_sources_masked_and_cleaned(mask, model, truth, primary_beam):
    """Stokes I planes: the mask holds both sources and the model their flux."""
    assert mask.mean() < MAX_MASK_FRACTION
    for box in source_boxes().values():
        peak = np.unravel_index(
            np.argmax((truth * primary_beam)[box]), truth[box].shape
        )
        assert mask[box][peak]
        apparent = float(np.sum(truth[box] * primary_beam[box]))
        np.testing.assert_allclose(model[box].sum(), apparent, rtol=FLUX_TOLERANCE)


@pytest.mark.parametrize("deconvolver", ["hogbom", "asp"])
def test_cube_masks_the_sources_by_itself(tmp_path, simulated, deconvolver):
    image = run_cube(simulated, tmp_path, deconvolver)
    _, truth = simulated
    for channel in range(image.sizes["frequency"]):
        mask = image.MASK.values[0, channel, 0].astype(bool)
        model = image.SKY_MODEL.values[0, channel, 0]
        assert_sources_masked_and_cleaned(
            mask, model, truth, image.PRIMARY_BEAM.values[0, channel, 0]
        )
        if deconvolver == "hogbom":
            assert not model[~mask].any()
        assert not image.MASK.values[0, channel, 1].any()
        assert not image.SKY_MODEL.values[0, channel, 1].any()


@pytest.mark.parametrize(
    "continuum_mode, deconvolver, n_taylor_terms",
    [
        ("mfs", "hogbom", 1),
        ("mfs", "asp", 1),
        ("mvc", "asp", 1),
        ("mfs", "mtmfs", 2),
    ],
)
def test_continuum_masks_the_sources_by_itself(
    tmp_path, simulated, continuum_mode, deconvolver, n_taylor_terms
):
    result, image = run_continuum(
        simulated, tmp_path, continuum_mode, deconvolver, n_taylor_terms
    )
    _, truth = simulated
    mask = image.MASK.values[0, 0, 0].astype(bool)
    primary_beam = image.PRIMARY_BEAM.isel(frequency=0).values[0, 0]
    assert_sources_masked_and_cleaned(
        mask, image.SKY_MODEL.values[0, 0, 0], truth, primary_beam
    )
    for taylor_term in range(n_taylor_terms):
        np.testing.assert_array_equal(image.MASK.values[0, taylor_term, 0], mask)
    assert not image.MASK.values[0, :, 1].any()
    assert result["controller"].stop_code_imaging[0, 0, 1] == IMAGING_ZERO_MASK
    assert result["controller"].cycles_done >= 2


def test_a_dirty_image_gets_its_mask(tmp_path, simulated):
    image = run_cube(
        simulated,
        tmp_path,
        "hogbom",
        cycle_control_params={**CYCLE_CONTROL, "max_iter": 0},
    )
    assert image.MASK.values[0, :, 0].any()
    assert not image.MASK.values[0, :, 1].any()
    assert not image.SKY_MODEL.values.any()


def test_cube_refuses_a_clean_mask_with_auto_mask_params(tmp_path, simulated):
    mask_file = tmp_path / "mask.npy"
    np.save(mask_file, np.ones(IMAGE_SIZE, dtype=bool))
    with pytest.raises(ValueError, match="exclude each other"):
        run_cube(simulated, tmp_path, "hogbom", clean_mask=str(mask_file))
    assert not (tmp_path / "cube.img.zarr").exists()


def test_cube_refuses_an_unknown_auto_mask_parameter(tmp_path, simulated):
    with pytest.raises(ValueError, match="Unknown auto_mask_params keys"):
        run_cube(simulated, tmp_path, "hogbom", auto_mask_params={"noise": 5.0})
    assert not (tmp_path / "cube.img.zarr").exists()
