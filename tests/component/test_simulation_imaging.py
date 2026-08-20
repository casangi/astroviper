"""Component test: simulate an MSv4 with the simulator and image it with AstroVIPER.

Checks that the simulated visibilities / uvw are self-consistent with AstroVIPER's
imaging convention (the point source lands on the expected pixel with the
beam-attenuated flux) and that the thermal-noise level matches theory.
"""

import numpy as np
from astropy.coordinates import SkyCoord
from xradio.image import load_image
from xradio.measurement_set import load_processing_set, open_processing_set
from xradio.schema.check import check_datatree

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging import image_cube_single_field
from astroviper.processing_functions.simulation.antenna_beams import (
    casa_airy_disk_response,
)
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.coordinate_transforms import (
    celestial_coord_to_sin_pixel,
    sin_project,
)
from astroviper.utils.telescope_layout import read_telescope_layout

PHASE_CENTER = SkyCoord(ra="19h59m28.5s", dec="+40d44m01.5s", frame="fk5")
SOURCE = SkyCoord(ra="19h59m50.51793355s", dec="+40d48m11.3694551s", frame="fk5")
PC = np.array([PHASE_CENTER.ra.rad, PHASE_CENTER.dec.rad])
SRC = np.array([SOURCE.ra.rad, SOURCE.dec.rad])
IMAGE_SIZE = [200, 200]
CELL_SIZE = np.array([-8.0, 8.0]) * np.pi / (180 * 3600)


def simulate(tmp_path, name, flux, noise_params=None):
    ant = read_telescope_layout("vla.d")
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(tmp_path / f"{name}.ps.zarr"),
        antenna_xds=ant,
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 1800.0,
            "n_samples": 6,
        },
        frequency_params={
            "freq_start": 3e9,
            "freq_delta": 0.4e9,
            "n_channels": 2,
            "channel_width": 1e7,
        },
        polarization=["RR", "LL"],
        point_source_flux=np.array([flux, 0, 0, flux])[None, None, None, :],
        point_source_ra_dec=SRC[None, None, :],
        phase_center_ra_dec=PC[None, :],
        beam_models=[airy_disk_model("vla")],
        beam_model_map=np.zeros(27, int),
        noise_params=noise_params,
        n_time_chunks=2,
        n_frequency_chunks=2,
        overwrite=True,
    )
    # the written processing set must be a valid MSv4 (XRADIO schema checker)
    issues = check_datatree(open_processing_set(result["ps_store"]))
    assert str(issues) == "No schema issues found", str(issues)
    return result


def image(tmp_path, ps_store, name):
    ps_xdt = open_processing_set(ps_store)
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    phase_direction = combined.FIELD_PHASE_CENTER_DIRECTION.sel(
        field_name=combined.attrs["center_field_name"]
    ).values
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": CELL_SIZE,
        "phase_direction": phase_direction,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I"],
        "time_coords": [0],
        "fft_padding": 1.2,
        "cpp_gridder": True,
    }
    image_store = str(tmp_path / f"{name}.img.zarr")
    image_cube_single_field(
        ps_store=ps_store,
        image_store=image_store,
        image_params=image_params,
        imaging_weights_params={
            "weighting": "natural",
            "robust": 0.5,
            "casa_weighting_implementation": True,
        },
        iteration_control_params={
            "niter": 0,
            "nmajor": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "cyclefactor": 1.5,
            "cycleniter": -1,
            "minpsffraction": 0.05,
            "maxpsffraction": 0.8,
        },  # fmt: skip
        gridder="prolate_spheroidal",
        deconvolver="hogbom_many_threads",
        scan_intents="OBSERVE_TARGET#ON_SOURCE",
        image_data_variables_keep=["sky_residual", "point_spread_function"],
        processing_set_data_group_name="base",
        single_precision_image=False,
        processing_function_threads=1,
        n_chunks=2,
        overwrite=True,
    )
    return load_image(image_store)


def test_point_source_lands_on_expected_pixel(tmp_path):
    result = simulate(tmp_path, "point", flux=2.0)
    img = image(tmp_path, result["ps_store"], "point")
    sky = img.SKY_RESIDUAL.values[0]  # [frequency, polarization, l, m]
    expected_pixel = celestial_coord_to_sin_pixel(PC, IMAGE_SIZE, CELL_SIZE, SRC)
    lm = sin_project(PC, SRC)
    frequency = img.frequency.values
    for i_chan in range(sky.shape[0]):
        plane = sky[i_chan, 0]
        i, j = np.unravel_index(np.argmax(plane), plane.shape)
        np.testing.assert_allclose([i, j], expected_pixel, atol=0.5)
        # natural-weighted dirty peak = mean visibility amplitude = flux * power beam (casa_airy VLA)
        voltage = casa_airy_disk_response(
            lm[0],
            lm[1],
            frequency[i_chan],
            24.5,
            0.0,
            airy_disk_model("vla")["max_rad_1GHz"],
        )
        # (gridding-correction / pixelisation errors of a few % at this offset)
        np.testing.assert_allclose(plane[i, j], 2.0 * voltage**2, rtol=0.05)


def test_noise_level_matches_theory(tmp_path):
    """Zero-flux sky with thermal noise: the image RMS follows 1/sqrt(sum of weights) (Stokes I)."""
    result = simulate(
        tmp_path, "noise", flux=0.0, noise_params={"t_receiver": 50.0, "random_seed": 5}
    )
    ms = load_processing_set(result["ps_store"])[result["ms_name"]].ds
    # natural weighting: image RMS = 1 / sqrt(sum of the weights that enter the image)
    weight_sum_per_pol = ms.WEIGHT.values.sum(axis=(0, 1, 2))  # [polarization]
    expected_rms = np.sqrt(
        1.0 / weight_sum_per_pol.sum()
    )  # both hands combine into Stokes I
    img = image(tmp_path, result["ps_store"], "noise")
    sky = img.SKY_RESIDUAL.values[0, :, 0]
    measured = sky.std(axis=(1, 2))
    # per-channel images use only that channel's visibilities
    n_chan = sky.shape[0]
    expected_per_chan = expected_rms * np.sqrt(n_chan)
    np.testing.assert_allclose(measured, expected_per_chan, rtol=0.35)
    assert np.abs(sky.mean()) < 3 * expected_per_chan.mean() / np.sqrt(sky[0].size) * 10
