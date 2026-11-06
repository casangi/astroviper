"""Image time validation uses the same parser as XRADIO image construction."""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from astropy.time import Time
from xradio.image._util.image_factory import _time_coords_to_mjd

from astroviper.processing_functions.imaging.config_check import (
    CONTINUUM_SINGLE_FIELD,
    CUBE_SINGLE_FIELD,
    check_driver_workflow,
    check_residual_update,
)


@pytest.mark.parametrize(
    "mode,continuum_mode",
    [
        (CUBE_SINGLE_FIELD, "mfs"),
        (CONTINUUM_SINGLE_FIELD, "mfs"),
        (CONTINUUM_SINGLE_FIELD, "mvc"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        58849.5,
        [58849.5],
        np.array([58849.5]),
        "58849.5",
        ["58849.5"],
        "2020-01-01T12:00:00",
        ["2020-01-01T12:00:00"],
        np.datetime64("2020-01-01T12:00:00"),
        np.array(["2020-01-01T12:00:00"], dtype="datetime64[s]"),
        datetime(2020, 1, 1, 12),
        [datetime(2020, 1, 1, 12)],
        Time("2020-01-01T12:00:00", scale="utc"),
        Time(["2020-01-01T12:00:00"], scale="utc"),
        Time("2020-01-01T12:00:37", scale="tai"),
    ],
)
def test_xradio_time_formats_are_accepted_without_mutation(value, mode, continuum_mode):
    before = deepcopy(value)
    config = {"image_params": {"time_coords": value}, "continuum_mode": continuum_mode}
    check_residual_update(config, mode=mode)
    check_driver_workflow(config, mode=mode)
    assert config["image_params"]["time_coords"] is value
    np.testing.assert_array_equal(
        _time_coords_to_mjd(value), _time_coords_to_mjd(before)
    )
    np.testing.assert_allclose(_time_coords_to_mjd(value), [58849.5])


@pytest.mark.parametrize(
    "value",
    [
        [],
        [np.nan],
        [np.inf],
        True,
        1j,
        np.timedelta64(1, "D"),
        "not-a-time",
        [[58849.5]],
    ],
)
def test_invalid_time_values_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        check_residual_update(
            {"image_params": {"time_coords": value}}, mode=CUBE_SINGLE_FIELD
        )


def test_mvc_time_plane_count_is_checked_after_parsing():
    with pytest.raises(ValueError, match="exactly one image time plane"):
        check_residual_update(
            {
                "continuum_mode": "mvc",
                "image_params": {"time_coords": Time(["2020-01-01", "2020-01-02"])},
            },
            mode=CONTINUUM_SINGLE_FIELD,
        )


@pytest.mark.parametrize(
    "time", [58849.5, "2020-01-01T12:00:00", Time("2020-01-01T12:00:00")]
)
def test_scalar_time_cannot_hide_oversized_disk_chunks(time):
    with pytest.raises(ValueError, match="exceeds the time extent"):
        check_driver_workflow(
            {"image_params": {"time_coords": time}, "image_chunking": {"time": 2}},
            mode=CUBE_SINGLE_FIELD,
        )
