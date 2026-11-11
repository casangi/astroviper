"""Continuum imaging with one Taylor term (MFS) against cube imaging of the same
channel: the same model and residual, bit for bit, over several model updates.

The continuum's Hogbom and Asp clean Taylor term zero with the cube's
deconvolvers, and both imagers add every component of a model update to the
model in place. A model update that summed its components from zero and added
that sum to the model would round differently from the second model update on,
in a pixel that holds flux and takes several components in one model update;
Asp's choices then part.

A small simulated processing set: the VLA D configuration, 6 integrations of
30 min, one channel at 3 GHz, circular feeds, no noise, a 1 Jy point source at
the phase centre and a 0.5 Jy Gaussian of 40 x 25 arcsec. Its Stokes V is
zero, so the cube's conversion of its model to the correlation basis and back,
in every residual update, is exact; with Q, U or V in the model it rounds, and
the imagers part on that account.
"""

import dask
import numpy as np
import pytest
from xradio.image import load_image
from xradio.measurement_set import open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = np.array([5.2336866, 0.7109449])
ARCSEC = np.pi / (180 * 3600)
IMAGE_SIZE = [64, 64]
CELL_SIZE = np.array([-6.0, 6.0]) * ARCSEC
FREQUENCY = 3e9
BEAM_MODEL = airy_disk_model("vla_nvss")
PRIMARY_BEAM_LIMIT = 0.2
CONTROLS = {
    "max_iter": 2000,
    "max_cycles": 4,
    "max_iter_per_cycle": 100,
    "threshold": 0.0,
    "gain": 0.1,
    "psf_sidelobe_factor": 1.0,
    "min_psf_fraction": 0.05,
    "max_psf_fraction": 0.8,
}


@pytest.fixture(scope="module")
def processing_set(tmp_path_factory):
    store = str(tmp_path_factory.mktemp("continuum_cube_agreement") / "input.ps.zarr")
    antenna_xds = read_telescope_layout("vla.d")
    offset = np.array([30.0, -24.0]) * ARCSEC / np.array([np.cos(PHASE_CENTER[1]), 1])
    with dask.config.set(scheduler="synchronous"):
        distributed_applications.simulation.simulate_processing_set(
            ps_store=store,
            antenna_xds=antenna_xds,
            time_params={
                "time_start": "2019-10-03T19:00:00.000",
                "time_delta": 1800.0,
                "n_samples": 6,
            },
            frequency_params={
                "freq_start": FREQUENCY,
                "freq_delta": 1e8,
                "n_channels": 1,
                "channel_width": 1e8,
            },
            polarization=["RR", "LL"],
            sky_components=[
                {"kind": "point", "flux": 1.0, "ra_dec": PHASE_CENTER},
                {
                    "kind": "gaussian",
                    "flux": 0.5,
                    "ra_dec": PHASE_CENTER + offset,
                    "major": 40 * ARCSEC,
                    "minor": 25 * ARCSEC,
                    "pa": 0.4,
                },
            ],
            phase_center_ra_dec=PHASE_CENTER[None, :],
            beam_models=[BEAM_MODEL],
            beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
            n_time_chunks=1,
            n_frequency_chunks=1,
            overwrite=True,
        )
    return store


def _image(ps_store, image_store, imager, deconvolver, single_precision_image):
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": PHASE_CENTER,
        "frequency_coords": open_processing_set(ps_store).xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "V"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }
    arguments = {
        "ps_store": ps_store,
        "image_store": str(image_store),
        "deconvolver": deconvolver,
        "instrument_polarization_basis": "circular",
        "processing_set_data_group_name": "base",
        "image_data_variables_keep": ["sky_model", "sky_residual"],
        "single_precision_image": single_precision_image,
        "n_mapping_parallelism": {"frequency": 1},
        # one thread grids in one order
        "processing_function_threads": 1,
        "overwrite": True,
    }
    with dask.config.set(scheduler="synchronous"):
        if imager == "cube":
            return distributed_applications.imaging.image_cube_single_field(
                image_params=image_params,
                imaging_weights_params={"weighting": "natural"},
                cycle_control_params={
                    **CONTROLS,
                    "primary_beam_limit": PRIMARY_BEAM_LIMIT,
                },
                primary_beam_params={"beam_model": BEAM_MODEL},
                **arguments,
            )
        return distributed_applications.imaging.image_continuum_single_field(
            image_params={
                **image_params,
                "n_taylor_terms": 1,
                "reference_frequency": FREQUENCY,
                "primary_beam_model": "airy",
                "list_dish_diameters": [BEAM_MODEL["dish_diameter"]],
                "list_blockage_diameters": [BEAM_MODEL["blockage_diameter"]],
            },
            imaging_weights_params={
                "weighting": "natural",
                "weighting_scope": "local",
            },
            cycle_control_params=CONTROLS,
            continuum_mode="mfs",
            primary_beam_limit=PRIMARY_BEAM_LIMIT,
            **arguments,
        )


def _planes(image_store, name):
    """``name`` as (polarization, l, m): the one channel or Taylor term zero."""
    data = load_image(str(image_store))[name]
    other = {dim: 0 for dim in data.dims if dim not in ("polarization", "l", "m")}
    return data.isel(other).transpose("polarization", "l", "m").values


def _model_updates(result):
    """Per (time, polarization) plane, the iterations and model fluxes of
    every model update (the cube's record also holds the stop test that
    ended its loop)."""
    records = {}
    for key, fields in result["deconvolution"].data.items():
        n = len(fields["model_update_stop_code"])
        records[key.time, key.pol] = {
            name: list(fields[name][:n])
            for name in ("iter_done", "start_model_flux", "model_flux")
        }
    return records


@pytest.mark.parametrize(
    "single_precision_image", [True, False], ids=["single", "double"]
)
@pytest.mark.parametrize("deconvolver", ["hogbom", "asp"])
def test_one_taylor_term_continuum_is_the_cube_bit_for_bit(
    tmp_path, processing_set, deconvolver, single_precision_image
):
    """Every plane makes the same iterations in every model update in both
    imagers, Stokes I in all four, and books the same model fluxes; the models
    and residuals written are the same bit for bit."""
    records = {}
    for imager in ("cube", "continuum"):
        result = _image(
            processing_set,
            tmp_path / f"{imager}.img.zarr",
            imager,
            deconvolver,
            single_precision_image,
        )
        records[imager] = _model_updates(result)

    assert records["continuum"] == records["cube"]
    iterations = records["cube"][0, 0]["iter_done"]
    assert len(iterations) == CONTROLS["max_cycles"]
    assert min(iterations) > 0
    for name in ("SKY_MODEL", "SKY_RESIDUAL"):
        cube = _planes(tmp_path / "cube.img.zarr", name)
        continuum = _planes(tmp_path / "continuum.img.zarr", name)
        assert cube.dtype == (np.float32 if single_precision_image else np.float64)
        np.testing.assert_array_equal(continuum, cube, err_msg=name)
