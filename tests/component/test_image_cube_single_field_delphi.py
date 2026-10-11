"""image_cube_single_field_delphi on a small simulated noisy ALMA processing set.

The PSF pass derives the parameters and writes the parameters file
(``calculate_parameters_only``); ``image_cube_single_field(parameters_file=...)``
then images the cube from the file alone.
"""

import importlib
import inspect
import json

import numpy as np
from astropy.coordinates import SkyCoord
from xradio.image import load_image

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging.image_cube_single_field import (
    image_cube_single_field,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

delphi_module = importlib.import_module(
    "astroviper.distributed_applications.imaging.image_cube_single_field_delphi"
)

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
N_ANTENNA = 10
BAND3_NOISE = {"t_receiver": 34.075, "ant_efficiency": 0.8957, "random_seed": 7}


def _simulate_noisy_alma(ps_store):
    """1 Jy Stokes I point source at the phase centre, 10 ALMA antennas,
    2 channels near 100 GHz, XX and YY, Band 3 thermal noise."""
    return distributed_applications.simulation.simulate_processing_set(
        ps_store=ps_store,
        antenna_xds=read_telescope_layout(
            "alma.cycle7.1", antenna_selection=list(range(N_ANTENNA))
        ),
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 600.0,
            "n_samples": 6,
        },
        frequency_params={
            "freq_start": 100e9,
            "freq_delta": 1e9,
            "n_channels": 2,
            "channel_width": 1e7,
        },
        polarization=["XX", "YY"],
        point_source_flux=np.array([1.0, 0, 0, 1.0])[None, None, None, :],
        point_source_ra_dec=PC[None, None, :],
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(N_ANTENNA, int),
        noise_params=BAND3_NOISE,
        n_time_chunks=1,
        n_frequency_chunks=1,
        overwrite=True,
    )["ps_store"]


def test_delphi_parameters_file_then_image_from_it(tmp_path):
    ps_store = _simulate_noisy_alma(str(tmp_path / "alma.ps.zarr"))
    image_store = str(tmp_path / "alma.img.zarr")
    parameters = delphi_module.image_cube_single_field_delphi(
        ps_store,
        image_store,
        processing_set_data_group_name="base",
        calculate_parameters_only=True,
        n_mapping_parallelism={"frequency": 2},
    )
    parameters_file = f"{image_store}.parameters.json"
    with open(parameters_file) as file:
        assert json.load(file) == parameters

    # Every argument of the imager but parameters_file itself.
    arguments = parameters["image_cube_single_field"]
    assert set(arguments) == set(
        inspect.signature(image_cube_single_field).parameters
    ) - {"parameters_file"}
    assert arguments["primary_beam_params"] == {"telescope": "alma"}
    assert arguments["instrument_polarization_basis"] == "linear"
    assert arguments["imaging_weights_params"] == {
        "weighting": "briggs",
        "robust": 0.5,
    }
    image_params = arguments["image_params"]
    assert image_params["polarization_coords"] == ["I", "Q"]
    np.testing.assert_allclose(image_params["frequency_coords"], [100e9, 101e9])
    np.testing.assert_allclose(image_params["phase_direction"], PC)

    derived = parameters["derived"]
    n_frequency, n_polarization = 2, 2
    for name in (
        "beam_fit_params_point_spread_function",
        "max_sidelobe_point_spread_function",
        "sensitivity",
        "sensitivity_natural",
        "sum_of_weights",
        "effective_number_of_visibilities",
    ):
        assert np.shape(derived[name])[:2] == (n_frequency, n_polarization), name
    for name in ("uv_distance_max", "uv_distance_min", "abs_w_max", "flagged_fraction"):
        assert np.shape(derived[name]) == (n_frequency,), name
    # No projected baseline exceeds the longest antenna separation, the
    # sample the cell of the PSF pass is sized for.
    u_max = derived["longest_antenna_separation"] * 101e9 / 299792458.0
    assert max(derived["uv_distance_max"]) <= u_max
    # 5 pixels across the fitted minor axis of the highest channel (Stokes I).
    minor = derived["beam_fit_params_point_spread_function"][1][0][1]
    np.testing.assert_allclose(image_params["cell_size"], [-minor / 5, minor / 5])
    size = image_params["image_size"][0]
    assert size % 2 == 0 and size * minor / 5 >= derived["field_of_view"]
    threshold = arguments["cycle_control_params"]["threshold"]
    np.testing.assert_allclose(threshold, 2.0 * np.array(derived["sensitivity"]))

    image_cube_single_field(parameters_file=parameters_file)
    img = load_image(image_store)
    for name in ("SKY_MODEL", "SKY_RESIDUAL", "SKY_RESTORED", "PRIMARY_BEAM"):
        assert name in img.data_vars, name
    assert img.sizes["l"] == size
    stokes_i = 0
    model_flux = np.nansum(img.SKY_MODEL.values[0, :, stokes_i], axis=(1, 2))
    np.testing.assert_allclose(model_flux, [1.0, 1.0], rtol=0.1)
    # The residual is noise like at about the theoretical sensitivity.
    residual_rms = np.nanstd(img.SKY_RESIDUAL.values[0, :, stokes_i], axis=(1, 2))
    sensitivity = np.array(derived["sensitivity"])[:, stokes_i]
    assert np.all((residual_rms > 0.5 * sensitivity) & (residual_rms < 2 * sensitivity))
