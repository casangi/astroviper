"""Component test: point_spread_function_cube_single_field on a tiny simulated
ALMA processing set with thermal noise and no source.

Checks that the beam fit, the largest sidelobe and the point spread function
equal what image_cube_single_field records for the same parameters with
max_iter=0, that the natural sensitivity equals the closed form for equal
weights and the measured noise of the dirty image, and that Briggs weighting
never beats natural weighting.
"""

import numpy as np
import pytest
from xradio.image import load_image
from xradio.measurement_set import load_processing_set, open_processing_set

import astroviper.distributed_applications as distributed_applications
from astroviper.distributed_applications.imaging import image_cube_single_field
from astroviper.distributed_applications.imaging.point_spread_function_cube_single_field import (
    point_spread_function_cube_single_field,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import Key
from astroviper.utils.beam_models import airy_disk_model
from astroviper.utils.telescope_layout import read_telescope_layout

# RA = local sidereal time at ALMA, Dec at the zenith (as the flux recovery test)
PHASE_CENTER = np.array([4.267, np.deg2rad(-23.0)])
CELL = 0.7 * np.pi / (180 * 3600)  # about 5 pixels across the 100 GHz beam
IMAGE_SIZE = [256, 256]
NATURAL = {"weighting": "natural", "robust": 0.5}
BRIGGS = {"weighting": "briggs", "robust": 0.5}
NOISE_RTOL = 0.05  # dirty image noise against the theoretical sensitivity
EDGE_PIXELS = 32  # left out when measuring the dirty image noise


@pytest.fixture(scope="module")
def noise_processing_set(tmp_path_factory):
    """The most compact ALMA 12 m configuration, 4 x 10 min, 2 channels at
    100 GHz, two parallel hands, Band 3 thermal noise and a zero flux sky."""
    work_dir = tmp_path_factory.mktemp("psf_cube")
    antenna_xds = read_telescope_layout("alma.cycle8.1")
    result = distributed_applications.simulation.simulate_processing_set(
        ps_store=str(work_dir / "noise.ps.zarr"),
        antenna_xds=antenna_xds,
        time_params={
            "time_start": "2019-10-03T19:00:00.000",
            "time_delta": 600.0,
            "n_samples": 4,
        },
        frequency_params={
            "freq_start": 100e9,
            "freq_delta": 0.5e9,
            "n_channels": 2,
            "channel_width": 2e6,
        },
        polarization=["XX", "YY"],
        point_source_flux=np.zeros((1, 1, 1, 4)),
        point_source_ra_dec=PHASE_CENTER[None, None, :],
        phase_center_ra_dec=PHASE_CENTER[None, :],
        beam_models=[airy_disk_model("alma")],
        beam_model_map=np.zeros(antenna_xds.sizes["antenna_name"], int),
        noise_params={"t_receiver": 34.075, "ant_efficiency": 0.7121, "random_seed": 7},
        n_time_chunks=1,
        n_frequency_chunks=2,
        overwrite=True,
    )
    ps_xdt = open_processing_set(result["ps_store"])
    combined = ps_xdt.xr_ps.get_combined_field_and_source_xds()
    image_params = {
        "image_size": IMAGE_SIZE,
        "cell_size": np.array([-CELL, CELL]),
        "phase_direction": combined.FIELD_PHASE_CENTER_DIRECTION.sel(
            field_name=combined.attrs["center_field_name"]
        ).values,
        "frequency_coords": ps_xdt.xr_ps.get_freq_axis().values,
        "polarization_coords": ["I", "Q"],
        "time_coords": [0],
        "fft_padding": 1.2,
    }
    return {"work_dir": work_dir, "image_params": image_params, **result}


def _run_point_spread_function(noise_processing_set, imaging_weights_params, name):
    image_store = str(noise_processing_set["work_dir"] / f"{name}.img.zarr")
    properties = point_spread_function_cube_single_field(
        ps_store=noise_processing_set["ps_store"],
        image_params=noise_processing_set["image_params"],
        imaging_weights_params=imaging_weights_params,
        image_store=image_store,
        processing_set_data_group_name="base",
        n_mapping_parallelism={"frequency": 2},
        overwrite=True,
    )
    return properties, load_image(image_store)


@pytest.fixture(scope="module")
def natural_point_spread_function(noise_processing_set):
    return _run_point_spread_function(noise_processing_set, NATURAL, "psf_natural")


@pytest.fixture(scope="module")
def natural_dirty_image(noise_processing_set):
    """The imager with max_iter=0: the dirty image and the setup's records."""
    image_store = str(noise_processing_set["work_dir"] / "dirty.img.zarr")
    result = image_cube_single_field(
        ps_store=noise_processing_set["ps_store"],
        image_store=image_store,
        image_params=noise_processing_set["image_params"],
        imaging_weights_params=NATURAL,
        cycle_control_params={
            "max_iter": 0,
            "max_cycles": 0,
            "threshold": 0.0,
            "gain": 0.1,
            "psf_sidelobe_factor": 1.5,
            "max_iter_per_cycle": -1,
            "min_psf_fraction": 0.05,
            "max_psf_fraction": 0.8,
        },
        image_data_variables_keep=[
            "sky_residual",
            "point_spread_function",
            "beam_fit_params_point_spread_function",
        ],
        processing_set_data_group_name="base",
        n_mapping_parallelism={"frequency": 2},
        overwrite=True,
    )
    return result, load_image(image_store)


def test_returned_dataset_layout(noise_processing_set, natural_point_spread_function):
    properties, _ = natural_point_spread_function
    image_params = noise_processing_set["image_params"]
    np.testing.assert_array_equal(
        properties.frequency.values, image_params["frequency_coords"]
    )
    assert list(properties.polarization.values) == ["I", "Q"]
    assert list(properties.beam_params_label.values) == ["major", "minor", "pa"]
    assert properties.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.dims == (
        "frequency",
        "polarization",
        "beam_params_label",
    )
    for name in (
        "MAX_SIDELOBE_POINT_SPREAD_FUNCTION",
        "SENSITIVITY",
        "SENSITIVITY_NATURAL",
        "SUM_OF_WEIGHTS",
        "EFFECTIVE_NUMBER_OF_VISIBILITIES",
    ):
        assert properties[name].dims == ("frequency", "polarization")
    for name in ("UV_DISTANCE_MAX", "UV_DISTANCE_MIN", "ABS_W_MAX", "FLAGGED_FRACTION"):
        assert properties[name].dims == ("frequency",)
    assert properties.attrs["image_size"] == IMAGE_SIZE
    assert properties.attrs["fft_padding"] == 1.2
    assert properties.attrs["psf_fitting_method"] == "astroviper"
    assert (
        properties.attrs["gridding_kernel_params"]["type"] == "prolate_spheroidal_casa"
    )
    np.testing.assert_array_equal(properties.FLAGGED_FRACTION.values, 0.0)
    assert np.all(properties.UV_DISTANCE_MAX > properties.UV_DISTANCE_MIN)
    assert np.all(properties.ABS_W_MAX > 0)


def test_matches_imager_setup(natural_point_spread_function, natural_dirty_image):
    """Beam fit, largest sidelobe and point spread function as the imager
    records them with max_iter=0 (the image stores are single precision)."""
    properties, psf_image = natural_point_spread_function
    result, dirty_image = natural_dirty_image
    np.testing.assert_array_equal(
        psf_image.POINT_SPREAD_FUNCTION.values,
        dirty_image.POINT_SPREAD_FUNCTION.values,
    )
    np.testing.assert_array_equal(
        psf_image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
        dirty_image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
    )
    np.testing.assert_allclose(
        properties.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values,
        dirty_image.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.values[0],
        rtol=1e-6,
    )
    records = result["deconvolution"].data
    imager_sidelobe = np.array(
        [
            [
                records[Key(time=0, pol=pol, chan=chan)]["max_psf_sidelobe"]
                for pol in range(properties.sizes["polarization"])
            ]
            for chan in range(properties.sizes["frequency"])
        ]
    )
    np.testing.assert_allclose(
        properties.MAX_SIDELOBE_POINT_SPREAD_FUNCTION.values, imager_sidelobe, rtol=1e-6
    )


def test_natural_sensitivity(
    noise_processing_set, natural_point_spread_function, natural_dirty_image
):
    """Equal weights: sigma_I = sigma_Q = 1 / sqrt(2 sum w) with sum w over one
    correlation, and the measured robust noise of the source free dirty image
    agrees."""
    properties, _ = natural_point_spread_function
    _, dirty_image = natural_dirty_image
    ms_xds = load_processing_set(noise_processing_set["ps_store"])[
        noise_processing_set["ms_name"]
    ].ds
    weight = ms_xds.WEIGHT.values
    assert np.unique(weight).size == 1  # the simulator writes equal weights
    n_samples = weight.shape[0] * weight.shape[1]  # per channel and correlation
    expected = 1.0 / np.sqrt(2 * n_samples * weight.flat[0])
    np.testing.assert_allclose(properties.SENSITIVITY.values, expected, rtol=1e-12)
    np.testing.assert_allclose(
        properties.SENSITIVITY_NATURAL.values, expected, rtol=1e-12
    )
    np.testing.assert_allclose(
        properties.SUM_OF_WEIGHTS.values, n_samples * weight.flat[0], rtol=1e-12
    )
    np.testing.assert_allclose(
        properties.EFFECTIVE_NUMBER_OF_VISIBILITIES.values, n_samples, rtol=1e-12
    )

    sky = dirty_image.SKY_RESIDUAL.values[
        0, :, :, EDGE_PIXELS:-EDGE_PIXELS, EDGE_PIXELS:-EDGE_PIXELS
    ]
    median = np.median(sky, axis=(-2, -1), keepdims=True)
    mad_sigma = 1.4826 * np.median(np.abs(sky - median), axis=(-2, -1))
    print("dirty image robust sigma / SENSITIVITY:", mad_sigma / expected)
    np.testing.assert_allclose(
        mad_sigma, properties.SENSITIVITY.values, rtol=NOISE_RTOL
    )


def test_briggs_sensitivity_not_below_natural(
    noise_processing_set, natural_point_spread_function
):
    """Briggs weighting gives up sensitivity for resolution: never a lower
    noise than natural weighting, and fewer effective visibilities."""
    briggs, _ = _run_point_spread_function(noise_processing_set, BRIGGS, "psf_briggs")
    natural, _ = natural_point_spread_function
    np.testing.assert_allclose(
        briggs.SENSITIVITY_NATURAL.values, natural.SENSITIVITY.values, rtol=1e-12
    )
    print(
        "Briggs / natural sensitivity:",
        briggs.SENSITIVITY.values / natural.SENSITIVITY.values,
    )
    assert np.all(briggs.SENSITIVITY.values >= natural.SENSITIVITY.values)
    assert np.all(
        briggs.EFFECTIVE_NUMBER_OF_VISIBILITIES.values
        <= natural.EFFECTIVE_NUMBER_OF_VISIBILITIES.values
    )
    # higher resolution: a smaller fitted major axis
    assert np.all(
        briggs.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(beam_params_label="major")
        < natural.BEAM_FIT_PARAMS_POINT_SPREAD_FUNCTION.sel(beam_params_label="major")
    )
