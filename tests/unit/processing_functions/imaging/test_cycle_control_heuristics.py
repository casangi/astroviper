"""Unit tests for the cycle control heuristics."""

import pytest

from astroviper.processing_functions.imaging.deconvolution import (
    _validate_deconvolve_params,
    validate_exact_psf_subtraction,
)
from astroviper.processing_functions.imaging.utils.cycle_control import (
    CycleController,
    validate_entropy_params,
    validate_max_cycles,
    validate_threshold_tolerance,
)
from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
    EXTRA_CYCLES,
    MAX_PSF_FRACTION,
    MIN_PSF_FRACTION,
    NOISY_CYCLE_SAFETY,
    NOISY_GAIN,
    NOISY_MAX_ITER_PER_CYCLE,
    NOISY_N_SIGMA,
    PSF_SIDELOBE_FACTOR,
    THRESHOLD_TOLERANCE,
    alma_cycle_control_params,
    noisy_cycle_control_params,
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
NOISY_CYCLE_CONTROL_KEYS = CYCLE_CONTROL_KEYS | {
    "threshold_tolerance",
    "exact_psf_subtraction",
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
    """n_sigma x sensitivity per channel and plane; the other controls are fixed."""
    import numpy as np

    sensitivity = [[1e-3, 2e-3], [3e-3, 4e-3]]
    xds = _point_spread_function_xds(sensitivity, [[0.1, 0.2], [0.3, 0.6]])
    params = noisy_cycle_control_params(
        xds, n_sigma=2.0, gain=0.2, max_iter=1000, primary_beam_limit=0.3
    )
    assert set(params) == NOISY_CYCLE_CONTROL_KEYS
    np.testing.assert_allclose(params["threshold"], 2.0 * np.array(sensitivity))
    assert {key: value for key, value in params.items() if key != "threshold"} == {
        "max_iter": 1000,
        "max_cycles": 40,
        "threshold_tolerance": 0.02,
        "gain": 0.2,
        "psf_sidelobe_factor": 1.5,
        "min_psf_fraction": 0.05,
        "max_psf_fraction": 0.8,
        "max_iter_per_cycle": -1,
        "max_iter_divergence": 1,
        "primary_beam_limit": 0.3,
        "entropy_stop": False,
        "exact_psf_subtraction": False,
    }
    for key in ("max_iter", "max_cycles", "max_iter_per_cycle", "max_iter_divergence"):
        assert type(params[key]) is int, key

    defaults = noisy_cycle_control_params(xds)
    assert NOISY_N_SIGMA == 3.0 and NOISY_GAIN == 0.3
    np.testing.assert_allclose(
        defaults["threshold"], NOISY_N_SIGMA * np.array(sensitivity)
    )
    assert defaults["gain"] == NOISY_GAIN
    assert defaults["max_iter"] == 200000
    assert defaults["primary_beam_limit"] == 0.2


def test_noisy_cycle_control_params_transposed_dataset_and_invalid():
    """The threshold follows (frequency, polarization) whatever the dataset
    order; a plane without data gets threshold 0."""
    import numpy as np

    xds = _point_spread_function_xds([[1.0, 2.0], [3.0, np.inf]], [[0.1] * 2] * 2)
    params = noisy_cycle_control_params(xds.transpose("polarization", "frequency"))
    np.testing.assert_allclose(
        params["threshold"], [[NOISY_N_SIGMA, 2 * NOISY_N_SIGMA], [9.0, 0.0]]
    )
    for n_sigma in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="n_sigma must be positive"):
            noisy_cycle_control_params(xds, n_sigma=n_sigma)


def test_noisy_controls_pass_the_imager_checks():
    """Every check of the distributed application and the deconvolver accepts
    the controls, and they build the cycle controller of a channel."""
    xds = _point_spread_function_xds([[1e-3, 2e-3], [3e-3, 4e-3]], [[0.2] * 2] * 2)
    params = noisy_cycle_control_params(xds)
    assert validate_entropy_params(params)[0] is False
    assert validate_max_cycles(params["max_cycles"], params["max_psf_fraction"]) == 40
    assert validate_threshold_tolerance(params["threshold_tolerance"]) == 0.02
    validate_exact_psf_subtraction("hogbom", params["exact_psf_subtraction"])
    for row in params["threshold"]:
        _validate_deconvolve_params({**params, "threshold": row})
        controller = CycleController(
            max_iter=params["max_iter"],
            max_cycles=params["max_cycles"],
            threshold=row,
            gain=params["gain"],
            psf_sidelobe_factor=params["psf_sidelobe_factor"],
            min_psf_fraction=params["min_psf_fraction"],
            max_psf_fraction=params["max_psf_fraction"],
            max_iter_per_cycle=params["max_iter_per_cycle"],
            entropy_stop=params["entropy_stop"],
            threshold_tolerance=params["threshold_tolerance"],
        )
        state = controller.get_state()
        assert state["threshold"] == row
        assert state["threshold_tolerance"] == THRESHOLD_TOLERANCE
        for key in ("max_cycles", "gain", "psf_sidelobe_factor", "max_iter_per_cycle"):
            assert state[key] == params[key], key
