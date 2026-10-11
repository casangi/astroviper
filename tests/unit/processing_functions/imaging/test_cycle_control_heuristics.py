"""Unit tests for the ALMA cycle control heuristics."""

import pytest

from astroviper.processing_functions.imaging.deconvolution import (
    _validate_deconvolve_params,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    validate_entropy_params,
)
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    EXTRA_CYCLES,
    MAX_PSF_FRACTION,
    MIN_PSF_FRACTION,
    NOISY_CYCLE_SAFETY,
    NOISY_MAX_ITER_PER_CYCLE,
    PSF_SIDELOBE_FACTOR,
    alma_cycle_control_params,
    predicted_imaging_cycles,
    psf_fraction,
)

CYCLE_CONTROL_KEYS = {
    "max_iter",
    "max_cycles",
    "threshold",
    "primary_beam_limit",
    "gain",
    "psf_sidelobe_factor",
    "max_iter_per_cycle",
    "min_psf_fraction",
    "max_psf_fraction",
    "max_iter_divergence",
    "entropy_stop",
}


@pytest.mark.parametrize(
    "peak, level, fraction, expected",
    [
        (0.47, 3.3e-6, 0.086, 5),  # compact ALMA at 100 GHz, 5 to 6 cycles observed
        (0.637, 2.046e-5, 0.109, 5),  # ALMA C8 at 950 GHz, 5 cycles measured
        (0.637, 2.046e-5, 0.328, 10),  # same field, 10 cycles measured
        (1.0, 2.0, 0.5, 1),  # already at the level: one residual update
    ],
)
def test_predicted_imaging_cycles(peak, level, fraction, expected):
    cycles = predicted_imaging_cycles(peak, level, fraction)
    assert cycles == expected
    assert type(cycles) is int


@pytest.mark.parametrize(
    "args", [(0.0, 1e-5, 0.1), (1.0, 0.0, 0.1), (1.0, 1e-5, 0.0), (1.0, 1e-5, 1.0)]
)
def test_predicted_imaging_cycles_refuses_invalid_input(args):
    with pytest.raises(ValueError):
        predicted_imaging_cycles(*args)


def test_psf_fraction_is_clamped():
    assert psf_fraction(0.219) == PSF_SIDELOBE_FACTOR * 0.219
    assert psf_fraction(0.01) == MIN_PSF_FRACTION
    assert psf_fraction(10.0) == MAX_PSF_FRACTION


def _check_valid(params):
    assert set(params) == CYCLE_CONTROL_KEYS
    for key in ("max_iter", "max_cycles", "max_iter_per_cycle", "max_iter_divergence"):
        assert type(params[key]) is int, key
    validate_entropy_params(params)
    _validate_deconvolve_params(dict(params))


def test_noise_free_params():
    params = alma_cycle_control_params(0.219, 0.637, 2.046e-5)
    _check_valid(params)
    assert params["threshold"] == 2.046e-5
    assert params["max_cycles"] == 5 + EXTRA_CYCLES
    assert params["max_iter_per_cycle"] == -1
    assert params["entropy_stop"] is False
    assert alma_cycle_control_params(0.219, 0.637, 2.046e-5) == params


def test_noisy_params():
    params = alma_cycle_control_params(
        0.219, 0.637, 0.0, noisy=True, thermal_noise=0.048
    )
    _check_valid(params)
    assert params["threshold"] == 0.0
    assert params["entropy_stop"] is True
    assert params["max_iter_per_cycle"] == NOISY_MAX_ITER_PER_CYCLE
    expected = predicted_imaging_cycles(0.637, 0.048, psf_fraction(0.219))
    assert params["max_cycles"] == NOISY_CYCLE_SAFETY * expected


@pytest.mark.parametrize("thermal_noise", [None, 0.0])
def test_noisy_params_need_the_thermal_noise(thermal_noise):
    with pytest.raises(ValueError, match="thermal_noise"):
        alma_cycle_control_params(
            0.219, 0.637, 0.0, noisy=True, thermal_noise=thermal_noise
        )


def test_one_threshold_per_stokes_plane():
    """A threshold per Stokes plane passes through; Stokes I sets the cycles."""
    params = alma_cycle_control_params(0.219, 0.637, [2.046e-5, 5e-6, 5e-6, 5e-6])
    _check_valid(params)
    assert params["threshold"] == [2.046e-5, 5e-6, 5e-6, 5e-6]
    assert params["max_cycles"] == 5 + EXTRA_CYCLES
