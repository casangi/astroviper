"""Finite weighted Taylor-design construction at extreme numerical scales."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.config_check._spectral import (
    make_weighted_taylor_design,
)


def test_overflowing_taylor_powers_are_rejected_before_linear_algebra():
    frequencies = np.array([1e200, 2e200])
    original = frequencies.copy()
    with pytest.raises(ValueError, match="weighted Taylor design overflows"):
        make_weighted_taylor_design(
            frequencies, np.ones(2), n_taylor_terms=3, reference_frequency=1.0
        )
    np.testing.assert_array_equal(frequencies, original)
