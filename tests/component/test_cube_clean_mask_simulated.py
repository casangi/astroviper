"""Cube imaging with an explicit ``clean_mask`` on the simulated VLA D input
of :mod:`tests.component.test_continuum_asp_simulated` (a 2 Jy point source
and a 1 Jy Gaussian, flat spectra, two channels).

The mask is a ``.npy`` file of 17 x 17 pixels around the point source. In every
channel the deconvolver must clean the point source and leave the Gaussian in
the residual: the model of the point source holds its apparent flux within
8 percent (measured: within 3.6 percent), the model of the Gaussian less than
3 percent of its own (measured: 0 for Hogbom, at most 1.5 percent for Asp,
whose components reach past the mask), the largest model pixel lies in the
mask and the Gaussian's residual peak is more than 10 times the largest
residual in the mask (measured: at least 103 times). The mask written is the
one given, on every plane.
"""

import shutil

import dask
import numpy as np
import pytest
import xarray as xr
from xradio.measurement_set import open_processing_set

from astroviper.distributed_applications.imaging import image_cube_single_field
from astroviper.utils.beam_models import airy_disk_model
from tests.component.test_continuum_asp_simulated import (
    CELL_SIZE,
    IMAGE_SIZE,
    simulate_input,
    source_boxes,
)


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """The simulated processing set and the true sky on the imaging grid."""
    return simulate_input(tmp_path_factory.mktemp("cube_clean_mask"))


@pytest.fixture(scope="module")
def point_source_mask(tmp_path_factory):
    """A 17 x 17 pixel mask around the point source and its ``.npy`` path."""
    i, j = (box.start + 4 for box in source_boxes()["point"])
    mask = np.zeros(IMAGE_SIZE, dtype=bool)
    mask[i - 8 : i + 9, j - 8 : j + 9] = True
    mask_file = tmp_path_factory.mktemp("cube_clean_mask_file") / "mask.npy"
    np.save(mask_file, mask)
    return mask, str(mask_file)


def run_cube(simulated, directory, deconvolver, clean_mask):
    """Image a copy of the simulated processing set as a cube."""
    source_store, _ = simulated
    ps_store = str(directory / "input.ps.zarr")
    shutil.copytree(source_store, ps_store)
    processing_set = open_processing_set(ps_store)
    fields = processing_set.xr_ps.get_combined_field_and_source_xds()
    image_store = str(directory / "cube.img.zarr")
    with dask.config.set(scheduler="synchronous"):
        image_cube_single_field(
            ps_store=ps_store,
            image_store=image_store,
            image_params={
                "image_size": IMAGE_SIZE,
                "cell_size": CELL_SIZE,
                "phase_direction": fields.FIELD_PHASE_CENTER_DIRECTION.sel(
                    field_name=fields.attrs["center_field_name"]
                ).values,
                "frequency_coords": processing_set.xr_ps.get_freq_axis().values,
                "polarization_coords": ["I", "V"],
                "time_coords": [0],
                "fft_padding": 1.2,
            },
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params={
                "max_iter": 1000,
                "max_cycles": 20,
                "max_iter_per_cycle": 200,
                "threshold": 0.0,
                "gain": 0.1,
                "psf_sidelobe_factor": 1.0,
                "min_psf_fraction": 0.05,
                "max_psf_fraction": 0.8,
            },
            deconvolver=deconvolver,
            clean_mask=clean_mask,
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
        )
    with xr.open_zarr(image_store) as stored:
        return stored.load()


@pytest.mark.parametrize("deconvolver", ["hogbom", "asp"])
def test_cube_cleans_only_inside_an_explicit_clean_mask(
    tmp_path, simulated, point_source_mask, deconvolver
):
    mask, mask_file = point_source_mask
    image = run_cube(simulated, tmp_path, deconvolver, mask_file)
    _, truth = simulated
    boxes = source_boxes()

    assert (image.MASK.values == mask).all()
    for frequency in range(image.sizes["frequency"]):
        model = image.SKY_MODEL.values[0, frequency, 0]  # Stokes I
        residual = image.SKY_RESIDUAL.values[0, frequency, 0]
        primary_beam = image.PRIMARY_BEAM.values[0, frequency, 0]
        apparent = {
            name: float(np.sum(truth[box] * primary_beam[box]))
            for name, box in boxes.items()
        }
        assert mask[np.unravel_index(np.argmax(model), model.shape)]
        np.testing.assert_allclose(
            model[boxes["point"]].sum(), apparent["point"], rtol=0.08
        )
        assert abs(model[boxes["gaussian"]].sum()) < 0.03 * apparent["gaussian"]
        assert residual[boxes["gaussian"]].max() > 10 * np.abs(residual[mask]).max()


def test_cube_refuses_an_invalid_clean_mask_before_writing(tmp_path, simulated):
    mask_file = tmp_path / "wrong_shape.npy"
    np.save(mask_file, np.ones((IMAGE_SIZE[0] + 1, IMAGE_SIZE[1]), dtype=bool))
    image_store = tmp_path / "cube.img.zarr"
    with pytest.raises(ValueError, match="clean_mask must contain one 2-D image plane"):
        image_cube_single_field(
            ps_store=simulated[0],
            image_store=str(image_store),
            image_params={
                "image_size": IMAGE_SIZE,
                "cell_size": CELL_SIZE,
                "phase_direction": np.zeros(2),
                "frequency_coords": np.array([3e9]),
                "polarization_coords": ["I", "V"],
                "time_coords": [0],
                "fft_padding": 1.2,
            },
            imaging_weights_params={"weighting": "natural"},
            cycle_control_params={"max_iter": 0},
            clean_mask=str(mask_file),
            instrument_polarization_basis="circular",
        )
    assert not image_store.exists()
