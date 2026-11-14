"""Tests of the spectral windows of a frequency axis."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.utils.spectral_windows import (
    linear_channel_ranges,
)


@pytest.mark.parametrize(
    "frequency, ranges",
    [
        (
            np.append(1.0e9 + 1.0e6 * np.arange(5), 1.1e9 + 1e6 * np.arange(4)),
            [[0, 5], [5, 9]],
        ),
        (np.append(1.0e9 + 1.0e6 * np.arange(5), 1.5e9), [[0, 5], [5, 6]]),
        (2.0e9 - 1.0e6 * np.arange(7), [[0, 7]]),
        (
            np.append(1.0e9 + 1.0e6 * np.arange(4), 1.0045e9 + 2e6 * np.arange(3)),
            [[0, 4], [4, 7]],
        ),
        (1.0e9 + 1.0e6 * (np.arange(6) + [0, 1e-5, -1e-5, 0, 0, 0]), [[0, 6]]),
        (np.array([1.0e9]), [[0, 1]]),
        (np.array([]), []),
    ],
    ids=[
        "gap",
        "one_channel_window",
        "decreasing",
        "other_width",
        "within_tolerance",
        "one_channel",
        "no_channel",
    ],
)
def test_linear_channel_ranges(frequency, ranges):
    assert linear_channel_ranges(frequency) == ranges
