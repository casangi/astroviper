"""Cycle heuristics separate supplied policy from measured-residual checks."""

import pytest


@pytest.mark.parametrize("value", [float("inf"), float("nan"), True, "3"])
def test_noise_policy_rejected_before_reading_psf(value):
    from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
        noisy_cycle_control_params,
    )

    with pytest.raises((ValueError, TypeError), match="n_sigma"):
        noisy_cycle_control_params(None, n_sigma=value)


def test_measured_peak_stays_checked_with_checked_policy():
    from astroviper.processing_functions.imaging.utils.cycle_control_heuristics import (
        predicted_imaging_cycles,
    )

    with pytest.raises(ValueError, match="peak_residual"):
        predicted_imaging_cycles(float("inf"), 1, 0.5, input_checked=True)
