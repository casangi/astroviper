"""The obscured Airy pattern of airy_disk_response against its aperture integral."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, jn_zeros

from astroviper.processing_functions.imaging.primary_beam.airy_disk import (
    SPEED_OF_LIGHT,
    airy_disk_response,
)

FREQUENCY = 90e9
DISH_DIAMETER = 10.7
# Derived: (2 J1(x) / x)**2 = 1 / 2.
HALF_POWER_X = 1.6163399480


def _lm(x, angle=0.3):
    """(l, m) at the argument x = pi D sqrt(l**2 + m**2) f / c."""
    rho = np.asarray(x) * SPEED_OF_LIGHT / (np.pi * DISH_DIAMETER * FREQUENCY)
    return rho * np.cos(angle), rho * np.sin(angle)


def _aperture_voltage(x, e):
    """2 / (1 - e**2) * integral_e^1 J0(x t) t dt: the voltage pattern of a
    uniformly illuminated annular aperture, central blockage e of the diameter."""
    value, _ = quad(lambda t: j0(x * t) * t, e, 1.0, epsabs=1e-14, epsrel=1e-13)
    return 2.0 * value / (1.0 - e**2)


def test_unobscured_first_null_and_half_power():
    l, m = _lm([jn_zeros(1, 1)[0], HALF_POWER_X])  # noqa: E741
    voltage = airy_disk_response(l, m, FREQUENCY, DISH_DIAMETER)
    assert abs(voltage[0]) < 1e-14
    power = airy_disk_response(l, m, FREQUENCY, DISH_DIAMETER, ipower=2)
    assert power[1] == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("blockage_diameter", [0.0, 0.75, 3.0])
def test_voltage_is_the_aperture_integral(blockage_diameter):
    x = np.array([0.25, 1.0, 2.5, 3.9, 5.5, 7.2, 9.0, 12.5])
    l, m = _lm(x)  # noqa: E741
    expected = [
        _aperture_voltage(value, blockage_diameter / DISH_DIAMETER) for value in x
    ]
    np.testing.assert_allclose(
        airy_disk_response(l, m, FREQUENCY, DISH_DIAMETER, blockage_diameter),
        expected,
        rtol=0,
        atol=1e-12,
    )
