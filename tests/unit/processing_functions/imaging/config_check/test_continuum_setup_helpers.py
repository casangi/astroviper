"""Validated continuum setup metadata is scalar and leaves config unchanged."""

from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from astroviper.processing_functions.imaging.image_continuum_single_field import (
    _continuum_frequency_coord,
)
from astroviper.processing_functions.imaging.imaging_setup_continuum_single_field import (
    _attach_continuum_metadata,
    _validate_continuum_parameters,
)


@pytest.mark.parametrize("value", [1, np.int64(2)])
def test_setup_returns_scalar_metadata_without_changing_configuration(value):
    params = {"n_taylor_terms": value, "reference_frequency": np.float64(1e9)}
    before = deepcopy(params)
    order, reference = _validate_continuum_parameters(params)
    assert order == value
    assert type(order) is int
    assert reference == 1e9
    assert type(reference) is float
    assert params == before


def test_continuum_metadata_round_trip_uses_the_configuration_name():
    image = xr.Dataset()
    result = _attach_continuum_metadata(
        image, n_taylor_terms=2, reference_frequency=1e9, continuum_mode="mfs"
    )
    assert result is image
    assert image.attrs["continuum_imaging"] == {
        "continuum_mode": "mfs",
        "n_taylor_terms": 2,
        "reference_frequency": 1e9,
        "n_residual_taylor_terms": 2,
        "n_psf_taylor_terms": 3,
    }
    np.testing.assert_array_equal(_continuum_frequency_coord(image), [1e9])
