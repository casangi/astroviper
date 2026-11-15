"""Spectral rank is distinct from image precision and coefficient uncertainty."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check.param_check_residual_update import (
    check_spectral_sampling,
)


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize(
    "channels,n_taylor_terms", [(128, 3), (256, 3), (300, 3), (15360, 3), (4096, 4)]
)
def test_narrow_band_resolves_independent_terms_at_any_image_precision(
    channels, n_taylor_terms, precision
):
    frequencies = np.linspace(0.99e9, 1.01e9, channels)
    original = frequencies.copy()
    result = check_spectral_sampling(
        frequencies,
        np.ones(channels),
        n_taylor_terms=n_taylor_terms,
        reference_frequency=1e9,
        precision=precision,
    )
    assert result["spectral_design_rank"] == n_taylor_terms
    assert result["spectral_design_rank_precision"] == "float64"
    assert result["spectral_design_rank_scaling"] == "unit_column_norm"
    singular_values = np.array(result["spectral_design_rank_singular_values"])
    assert singular_values[0] / singular_values[-1] < 6
    assert result["spectral_design_rank_tolerance"] == pytest.approx(
        channels * np.finfo(np.float64).eps * singular_values[0], rel=1e-14, abs=0
    )
    np.testing.assert_array_equal(frequencies, original)


def test_rank_is_invariant_to_repeated_frequency_observations():
    frequencies = np.linspace(0.99e9, 1.01e9, 128)
    options = {"n_taylor_terms": 3, "reference_frequency": 1e9, "precision": "float32"}
    first = check_spectral_sampling(frequencies, np.ones(128), **options)
    repeated = check_spectral_sampling(np.tile(frequencies, 4), np.ones(512), **options)
    for key in (
        "spectral_design_rank",
        "spectral_design_rank_tolerance",
        "spectral_design_rank_singular_values",
    ):
        assert repeated[key] == first[key]
    assert repeated["usable_channel_count"] == 512
    assert repeated["distinct_frequency_count"] == 128


def test_rank_matches_independently_column_normalized_vandermonde():
    frequencies = np.array([0.8e9, 0.95e9, 1.1e9, 1.4e9, 1.4e9])
    weights = np.array([1, 2, 3, 1, 4], dtype=float)
    result = check_spectral_sampling(
        frequencies, weights, n_taylor_terms=3, reference_frequency=1e9
    )
    unique = np.unique(frequencies)
    summed_weights = np.array([weights[frequencies == value].sum() for value in unique])
    design = np.polynomial.polynomial.polyvander((unique - 1e9) / 1e9, 2)
    design *= np.sqrt(summed_weights[:, None])
    design /= np.linalg.norm(design, axis=0)
    expected = np.linalg.svd(design, compute_uv=False)
    np.testing.assert_allclose(
        result["spectral_design_rank_singular_values"], expected, rtol=1e-14
    )


def test_numerically_dependent_distinct_frequencies_are_rejected():
    # Three near-coincident points and one distant point cannot resolve four
    # independent columns at float64 precision even after column normalization.
    frequencies = [1e9, 1e9 + 1e-6, 1e9 + 2e-6, 2e9]
    with pytest.raises(ValueError, match="column-normalized.*rank deficient"):
        check_spectral_sampling(
            frequencies, np.ones(4), n_taylor_terms=4, reference_frequency=1e9
        )


def test_insufficient_distinct_frequencies_cannot_be_fixed_by_repetition():
    with pytest.raises(ValueError, match="fewer distinct"):
        check_spectral_sampling(
            np.tile([0.99e9, 1.01e9], 128),
            np.ones(256),
            n_taylor_terms=3,
            reference_frequency=1e9,
        )
