"""Type, units, and scalar contracts of the shared validation helpers."""

import astropy.units as u
import pytest

from astroviper.processing_functions.imaging.config_check._common import (
    require_finite_real,
)


def test_unit_bearing_values_are_not_silently_interpreted_as_si():
    value = 1.0 * u.GHz
    with pytest.raises(TypeError, match="plain numerical SI"):
        require_finite_real("reference_frequency", value, positive=True, scalar=True)
    assert value.unit == u.GHz and value.value == 1.0
