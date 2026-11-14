"""Tests of the spectral windows of a frequency axis."""

import numpy as np
import pytest

from astroviper.processing_functions.imaging.utils.spectral_windows import (
    linear_channel_ranges,
    window_rest_frequency,
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


def test_window_rest_frequency():
    """A cube of one spectral window keeps its rest frequency, for every range
    of its channels; a window of a cube of several takes its middle channel,
    the rest frequency of a cube of its channels alone."""
    one = 230.0e9 + 1.0e6 * np.arange(8)
    assert window_rest_frequency(one, [0, 8], 229.0e9) == 229.0e9
    assert window_rest_frequency(one, [2, 6], 229.0e9) == 229.0e9
    two = np.append(one, 232.0e9 + 2.0e6 * np.arange(5))
    assert window_rest_frequency(two, [0, 8], two[4]) == two[4]
    assert window_rest_frequency(two, [0, 8], two[6]) == two[4]
    assert window_rest_frequency(two, [8, 13], two[6]) == two[10]
    assert window_rest_frequency(two, [9, 12], two[6]) == two[10]
