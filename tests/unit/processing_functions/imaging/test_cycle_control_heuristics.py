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


def _point_spread_function_xds(sensitivity, max_sidelobe):
    """The two variables of the PSF pass the noisy controls read."""
    import numpy as np
    import xarray as xr

    dims = ("frequency", "polarization")
    coords = {"frequency": [1.0e11, 1.1e11], "polarization": ["I", "Q"]}
    return xr.Dataset(
        {
            "SENSITIVITY": (dims, np.asarray(sensitivity, dtype=float)),
            "MAX_SIDELOBE_POINT_SPREAD_FUNCTION": (
                dims,
                np.asarray(max_sidelobe, dtype=float),
            ),
        },
        coords=coords,
    )


def test_noisy_cycle_control_params_per_channel_threshold():
    """n_sigma x sensitivity per channel and plane; cycles from the largest sidelobe."""
    import math

    import numpy as np

    from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
        noisy_cycle_control_params,
    )

    xds = _point_spread_function_xds(
        [[1e-3, 2e-3], [3e-3, 4e-3]], [[0.1, 0.2], [0.3, 0.6]]
    )
    params = noisy_cycle_control_params(
        xds, n_sigma=3.0, max_dynamic_range=1e4, primary_beam_limit=0.3
    )
    assert set(params) == CYCLE_CONTROL_KEYS
    np.testing.assert_allclose(params["threshold"], [[3e-3, 6e-3], [9e-3, 12e-3]])
    fraction = PSF_SIDELOBE_FACTOR * 0.6
    expected = math.ceil(math.log(1e4 / 3.0) / math.log(1.0 / fraction))
    assert params["max_cycles"] == expected + EXTRA_CYCLES
    assert params["max_iter_per_cycle"] == NOISY_MAX_ITER_PER_CYCLE
    assert params["entropy_stop"] is True
    assert params["max_iter"] == 200000
    assert params["gain"] == 0.1
    assert params["max_iter_divergence"] == 1
    assert params["primary_beam_limit"] == 0.3
    # Accepted by the deconvolver's validation, row by row.
    for row in params["threshold"]:
        _validate_deconvolve_params({**params, "threshold": row})


def test_noisy_cycle_control_params_transposed_dataset_and_invalid():
    """The threshold follows (frequency, polarization) whatever the dataset order."""
    import numpy as np

    from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
        noisy_cycle_control_params,
    )

    xds = _point_spread_function_xds([[1.0, 2.0], [3.0, 4.0]], [[0.1] * 2] * 2)
    params = noisy_cycle_control_params(xds.transpose("polarization", "frequency"))
    np.testing.assert_allclose(params["threshold"], [[2.0, 4.0], [6.0, 8.0]])
    with pytest.raises(ValueError):
        noisy_cycle_control_params(xds, n_sigma=0.0)


def test_noisy_cycle_prediction_is_that_of_the_controls():
    from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
        noisy_cycle_control_params,
        noisy_cycle_prediction,
    )

    xds = _point_spread_function_xds(
        [[1e-3, 2e-3], [3e-3, 4e-3]], [[0.3, float("nan")], [0.1, 0.2]]
    )
    fraction, cycles = noisy_cycle_prediction(xds, 2.0, 1e5)
    assert fraction == psf_fraction(0.3)
    assert cycles == predicted_imaging_cycles(1e5, 2.0, fraction)
    params = noisy_cycle_control_params(xds, n_sigma=2.0, max_dynamic_range=1e5)
    assert params["max_cycles"] == cycles + EXTRA_CYCLES
    with pytest.raises(ValueError):
        noisy_cycle_prediction(xds, 2.0, 0.0)
