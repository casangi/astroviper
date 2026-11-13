"""Tests of the find_continuum configuration and input checks."""

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.config_check import (
    check_find_continuum_config,
    check_find_continuum_inputs,
)
from astroviper.utils.data_group_tools import modify_data_groups_xds
from tests.unit.processing_functions.imaging.find_continuum_test_utils import (
    make_find_continuum_image_xds,
)

N_CHANNELS = 6


@pytest.mark.parametrize(
    "n_bin, primary_beam_limit, n_mapping_parallelism",
    [(1, None, None), (3, 0.2, {"frequency": 4}), (1, 0, None)],
)
def test_valid_config(n_bin, primary_beam_limit, n_mapping_parallelism):
    check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism)


@pytest.mark.parametrize(
    "n_bin, primary_beam_limit, n_mapping_parallelism, error",
    [
        (0, None, None, ValueError),
        (1.5, None, None, TypeError),
        (1, 1.0, None, ValueError),
        (1, -0.1, None, ValueError),
        (1, "0.2", None, TypeError),
        (1, None, {"time": 2}, ValueError),
        (1, None, {"frequency": 0}, ValueError),
        (1, None, {"frequency": True}, TypeError),
        (1, None, [("frequency", 2)], TypeError),
    ],
)
def test_invalid_config(n_bin, primary_beam_limit, n_mapping_parallelism, error):
    with pytest.raises(error):
        check_find_continuum_config(n_bin, primary_beam_limit, n_mapping_parallelism)


@pytest.fixture
def img_xds():
    return make_find_continuum_image_xds(n_pixels=8, n_channels=N_CHANNELS)


def test_valid_inputs(img_xds):
    check_find_continuum_inputs(
        img_xds, "residual", np.ones(N_CHANNELS), [20.0] * N_CHANNELS
    )


def test_data_group_must_hold_every_cube(img_xds):
    with pytest.raises(ValueError, match="no data group"):
        check_find_continuum_inputs(
            img_xds, "model", np.ones(N_CHANNELS), np.ones(N_CHANNELS)
        )
    modify_data_groups_xds(img_xds, "sky_only", {"sky": "SKY_RESIDUAL"}, "Sky only")
    with pytest.raises(ValueError, match="lacks the roles"):
        check_find_continuum_inputs(
            img_xds, "sky_only", np.ones(N_CHANNELS), np.ones(N_CHANNELS)
        )


def test_one_time_and_stokes_i(img_xds):
    two_times = xr.concat([img_xds, img_xds.assign_coords(time=[1.0])], dim="time")
    with pytest.raises(ValueError, match="one time"):
        check_find_continuum_inputs(
            two_times, "residual", np.ones(N_CHANNELS), np.ones(N_CHANNELS)
        )
    with pytest.raises(ValueError, match="Stokes I"):
        check_find_continuum_inputs(
            img_xds.assign_coords(polarization=["Q"]),
            "residual",
            np.ones(N_CHANNELS),
            np.ones(N_CHANNELS),
        )


@pytest.mark.parametrize("which", ["transmission", "sky_temperature"])
def test_one_atmospheric_value_per_channel(img_xds, which):
    spectra = {
        "transmission": np.ones(N_CHANNELS),
        "sky_temperature": np.ones(N_CHANNELS),
    }
    spectra[which] = np.ones(N_CHANNELS + 1)
    with pytest.raises(
        ValueError, match=f"atmospheric_{which} must have one value per channel"
    ):
        check_find_continuum_inputs(
            img_xds, "residual", spectra["transmission"], spectra["sky_temperature"]
        )
