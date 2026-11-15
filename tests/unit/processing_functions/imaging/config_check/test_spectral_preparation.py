"""Shared spectral preparation preserves design values and diagnostic context."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check._spectral import (
    make_weighted_taylor_design,
    prepare_spectral_samples,
)
from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_taylor_hessian_approximation,
)
from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_spectral_sampling,
)


def test_weighted_design_matches_known_polynomial_without_mutating_inputs():
    frequencies = np.array([1e9, 2e9, 2e9, 4e9])
    weights = np.array([1e200, 0.0, 4e200, 1e200])
    original_frequencies, original_weights = frequencies.copy(), weights.copy()
    frequencies.setflags(write=False)
    weights.setflags(write=False)
    selected_frequencies, selected_weights, reference, usable = (
        prepare_spectral_samples(
            frequencies,
            weights,
            n_taylor_terms=3,
            reference_frequency=np.float64(2e9),
            precision="float64",
        )
    )
    design, normalized, roots = make_weighted_taylor_design(
        selected_frequencies[usable],
        selected_weights[usable],
        n_taylor_terms=3,
        reference_frequency=reference,
    )
    # x = [-1/2, 0, 1], sqrt(q) = [1/2, 1, 1/2].
    np.testing.assert_array_equal(
        design, [[0.5, -0.25, 0.125], [1, 0, 0], [0.5, 0.5, 0.5]]
    )
    np.testing.assert_array_equal(normalized, [0.25, 1, 0.25])
    np.testing.assert_array_equal(roots, [0.5, 1, 0.5])
    np.testing.assert_array_equal(usable, [True, False, True, True])
    assert reference == 2e9
    np.testing.assert_array_equal(frequencies, original_frequencies)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize("approximate", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "frequencies",
        "weights",
        "shape",
        "reference",
        "count",
        "precision",
        "zero_weights",
    ],
)
def test_both_diagnostics_keep_error_context_for_shared_input_failures(
    approximate, fault
):
    frequencies, weights = [0.9e9, 1e9, 1.1e9], [1, 1, 1]
    options = {"n_taylor_terms": 2, "reference_frequency": 1e9, "precision": "float64"}
    if fault == "frequencies":
        frequencies[0] = np.nan
    elif fault == "weights":
        weights[0] = -1
    elif fault == "shape":
        weights = [1, 1]
    elif fault == "reference":
        options["reference_frequency"] = 0
    elif fault == "count":
        options["n_taylor_terms"] = True
    elif fault == "precision":
        options["precision"] = "float16"
    else:
        weights = [0, 0, 0]
    options["context"] = {"field": "target", "spectral_window": "spw0"}
    if approximate:
        options.update(
            longest_projected_baselines=1e4,
            image_size=[16, 16],
            cell_size=[-1e-5, 1e-5],
        )
    check = (
        check_taylor_hessian_approximation if approximate else check_spectral_sampling
    )
    prefix = (
        "Approximate Taylor-Hessian preflight"
        if approximate
        else "Spectral-design preflight"
    )
    with pytest.raises(ValueError, match=prefix) as caught:
        check(frequencies, weights, **options)
    assert "target" in str(caught.value)
    assert "spw0" in str(caught.value)
