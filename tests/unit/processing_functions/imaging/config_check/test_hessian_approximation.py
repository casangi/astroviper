"""PSF-free Taylor-Hessian estimates use only selected spectral metadata."""

import json
import warnings

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check.param_check_model_update import (
    check_taylor_hessian_approximation,
)


def _approximation(frequencies, weights, **kwargs):
    """Supply a small image and a known projected-baseline resolution."""
    return check_taylor_hessian_approximation(
        frequencies,
        weights,
        longest_projected_baselines=1e4,
        image_size=[16, 16],
        cell_size=[-1e-5, 1e-5],
        **kwargs,
    )


@pytest.mark.parametrize("nterms", [1, 2, 3, 4])
def test_frequency_moment_estimate_matches_direct_small_system(nterms):
    frequencies = np.linspace(0.8e9, 1.2e9, 8)
    weights = np.array([1.0, 0.0, 2.0, 0.5, 4.0, 1.5, 3.0, 1.0])
    original_frequencies, original_weights = frequencies.copy(), weights.copy()
    result = _approximation(
        frequencies,
        weights,
        n_taylor_terms=nterms,
        reference_frequency=1e9,
        context={"field": ["target"]},
    )
    # A small, well-scaled example allows a direct moment matrix and solve as
    # an independent oracle. The production path equilibrates before forming H.
    x = (frequencies - 1e9) / 1e9
    moments = [np.sum(weights * x**k) for k in range(2 * nterms - 1)]
    hessian = np.array(
        [[moments[row + column] for column in range(nterms)] for row in range(nterms)]
    )
    order_zero = np.eye(nterms)[0]
    expected = np.sqrt(hessian[0, 0] * np.linalg.solve(hessian, order_zero)[0])
    assert result["order_zero_noise_amplification"] == pytest.approx(expected)
    assert result["order_zero_noise_amplification_limit"] == 3.0
    assert result["covariance_assumption"] == "proportional_to_approximate_hessian"
    assert result["usable_channel_count"] == 7
    assert result["field"] == ["target"]
    assert json.loads(json.dumps(result)) == result
    np.testing.assert_array_equal(frequencies, original_frequencies)
    np.testing.assert_array_equal(weights, original_weights)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("factor", [1e-200, 1.0, 1e200])
def test_narrow_band_and_common_weight_units_do_not_imply_dependent_moments(
    precision, factor
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _approximation(
            [1e9 - 1e3, 1e9, 1e9 + 1e3],
            np.full(3, factor),
            n_taylor_terms=3,
            reference_frequency=1e9,
            precision=precision,
        )
    assert result["order_zero_noise_amplification"] == pytest.approx(np.sqrt(3))
    assert min(result["relative_cholesky_pivots"]) > 0.3
    assert not caught


def test_approximate_order_zero_coupling_warns_without_rejecting():
    with pytest.warns(
        RuntimeWarning,
        match="Approximate Taylor-Hessian order-zero noise amplification exceeds 3.0",
    ):
        result = _approximation(
            [1e9, 1.001e9, 1.2e9],
            [1, 1, 1],
            n_taylor_terms=3,
            reference_frequency=1.1e9,
        )
    assert result["order_zero_noise_amplification"] > 3
    assert min(result["equilibrated_eigenvalues"]) > result["eigenvalue_tolerance"]


def test_approximate_native_pivot_guard_uses_working_precision():
    args = dict(n_taylor_terms=2, reference_frequency=1e9)
    with pytest.raises(
        ValueError, match="Approximate Taylor-Hessian.*unsafe relative Cholesky pivots"
    ):
        _approximation([1.1e9, 1.10001e9], [1, 1], precision="float32", **args)
    with pytest.warns(RuntimeWarning, match="Approximate Taylor-Hessian order-zero"):
        _approximation([1.1e9, 1.10001e9], [1, 1], precision="float64", **args)


@pytest.mark.parametrize("weights", [[0, 0], [1, 0]])
def test_zero_weight_frequencies_cannot_support_two_terms(weights):
    with pytest.raises(ValueError, match="Approximate Taylor-Hessian.*fewer distinct"):
        _approximation(
            [1e9, 1.2e9], weights, n_taylor_terms=2, reference_frequency=1.1e9
        )


def test_distinct_but_numerically_dependent_moments_fail_early():
    with pytest.raises(
        ValueError,
        match="Approximate Taylor-Hessian.*(positive definite|unsafe relative)",
    ):
        _approximation(
            [1.1e9, 1.1e9 + 1],
            [1, 1],
            n_taylor_terms=2,
            reference_frequency=1e9,
        )
