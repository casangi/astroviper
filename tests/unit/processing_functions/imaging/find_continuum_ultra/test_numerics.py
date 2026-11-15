"""Tests of find_continuum_ultra's numerics against the vendored findContinuum
and the simple port."""

import random
import warnings

import numpy as np
import pytest

from astroviper.processing_functions.imaging.find_continuum import (
    channel_statistics,
    image_statistics,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.numerics import (
    channels_of_ranges,
    chauvenet_statistics,
    contiguous_runs,
    line_fit,
    median_absolute_deviation,
    one_event_sigma,
    range_count,
    ranges_of_channels,
    round_half_up,
    round_to_decimals,
    round_to_significant_digits,
    scaled_median_absolute_deviation,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    findContinuum,
    random_ranges,
    random_spectrum,
)


def _same(value, expected):
    """Equal bit for bit, NaN equal to NaN."""
    return np.array_equal(np.float64(value), np.float64(expected), equal_nan=True)


def _random_values(rng, n_values):
    """Values of many magnitudes and both signs, dyadic ones (whose decimals
    end in a tie at the digit a format rounds to) and repeated ones."""
    values = rng.standard_normal(n_values) * 10.0 ** rng.uniform(-8, 8, n_values)
    dyadic = rng.integers(-(2**20), 2**20, values[::7].size)
    values[::7] = dyadic / 2.0 ** rng.integers(1, 12, dyadic.size)
    values[::11] = np.round(values[::11], 3)
    values[::13] = -values[::13]
    return values


def _selection(ranges):
    """findContinuum's selection string of ``[start, stop)`` ranges."""
    return ";".join(f"{start}~{stop - 1}" for start, stop in ranges)


def test_round_half_up_equals_findcontinuum():
    rng = np.random.default_rng(1)
    values = np.concatenate(
        [
            _random_values(rng, 20000) / 1e6,
            np.arange(-200, 200) + 0.5,
            0.19 * np.arange(8, 4000),
            np.nextafter(np.arange(-50, 50) + 0.5, np.inf),
            np.nextafter(np.arange(-50, 50) + 0.5, -np.inf),
        ]
    )
    for value in values:
        assert _same(round_half_up(value), findContinuum.round_half_up(value)), value
    assert round_half_up(2.5) == 3.0 and round(2.5) == 2
    assert round_half_up(-2.5) == -3.0


def test_levels_written_as_text():
    """findContinuum writes levels with "%f", "%.9f" and "%g" and reads them
    back: 10^5 values with ties and negatives, NumPy and Python floats."""
    values = _random_values(np.random.default_rng(2), 100000)
    for value in [*values[::2], *values[1::2].tolist()]:
        assert _same(round_to_decimals(value, 6), float("%f" % value))  # noqa: UP031
        assert _same(round_to_decimals(value, 9), float("%.9f" % value))  # noqa: UP031
        assert _same(
            round_to_significant_digits(value, 6),
            float("%g" % value),  # noqa: UP031
        )


def test_median_absolute_deviation_equals_the_ports_imstat():
    """r0 and r8 divide by imstat's medabsdevmed of the valid pixels."""
    rng = np.random.default_rng(3)
    for size in (1, 2, 3, 4, 101, 1000):
        values = rng.standard_normal((size, 7))
        values[rng.random(values.shape) < 0.2] = np.nan
        values[::3, ::2] = np.round(values[::3, ::2], 1)
        for cast in (values, values.astype(np.float32)):
            expected = channel_statistics(cast[np.isfinite(cast)])["medabsdevmed"]
            assert _same(median_absolute_deviation(cast), expected)


def test_scaled_median_absolute_deviation_equals_findcontinuum_mad():
    rng = np.random.default_rng(4)
    cases = [
        rng.standard_normal(size) * 10.0 ** rng.uniform(-5, 5) for size in range(1, 60)
    ]
    cases += [
        random_spectrum(rng, n) for n in (8, 9, 64, 75, 480, 3840) for _ in range(5)
    ]
    for values in cases[:40]:
        values[rng.random(values.size) < 0.3] = np.nan
    cases += [np.round(rng.standard_normal(40), 1), np.full(5, np.nan), np.array([])]
    cases += [rng.standard_normal(33).astype(np.float32)]
    for values in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected = findContinuum.MAD(values)
        assert _same(scaled_median_absolute_deviation(values), expected)
        assert type(scaled_median_absolute_deviation(values)) is float


def test_chauvenet_statistics_equal_the_ports_imstat():
    """imstat(algorithm="chauvenet", maxiter=5) of the port: median, medabsdevmed
    and sigma."""
    rng = np.random.default_rng(5)
    for size in (1, 2, 3, 10, 257, 2000):
        values = rng.standard_normal(size) * 10.0 ** rng.uniform(-6, 2)
        values[: max(1, size // 50)] *= 40.0
        expected = image_statistics(values, "chauvenet", 5)
        assert all(
            _same(value, expected[key])
            for value, key in zip(
                chauvenet_statistics(values),
                ("median", "medabsdevmed", "sigma"),
                strict=True,
            )
        ), size


@pytest.mark.parametrize("n_events", [1, 10])
def test_one_event_sigma_equals_findcontinuum(n_events):
    for n_values in (11, 100, 1000, 4096, 123457, 11250**2):
        assert _same(
            one_event_sigma(n_values, n_events),
            findContinuum.oneEvent(n_values, n_events),
        )


def test_contiguous_runs_equal_findcontinuum():
    rng = np.random.default_rng(6)
    cases = [
        np.array([], dtype=int),
        np.array([5]),
        np.array([3, 2, 1]),
        np.array([4, 4, 5]),
    ]
    for _ in range(200):
        channels = np.unique(rng.integers(0, 60, rng.integers(1, 40)))
        if rng.random() < 0.3:
            channels = rng.permutation(channels)
        cases.append(channels)
    for channels in cases:
        expected = findContinuum.splitListIntoContiguousLists(channels)
        assert [run.tolist() for run in contiguous_runs(channels)] == expected


def test_channels_and_ranges_equal_findcontinuum():
    rng = random.Random(7)
    for index in range(300):
        ranges = random_ranges(rng, rng.choice([8, 40, 480, 3840]))
        if index % 3 == 1:
            rng.shuffle(ranges)
        if index % 3 == 2:
            ranges.append(list(ranges[0]))
        channels = channels_of_ranges(ranges)
        expected = findContinuum.convertSelectionIntoChannelList(_selection(ranges))
        assert channels.dtype == np.int64
        np.testing.assert_array_equal(channels, expected)
        assert _selection(ranges_of_channels(channels)) == (
            findContinuum.convertChannelListIntoSelection(channels)
        )
        assert range_count(ranges) == len(_selection(ranges).split(";"))
    assert channels_of_ranges([]).dtype == np.int64 and channels_of_ranges([]).size == 0
    assert ranges_of_channels(np.array([], dtype=int)) == []
    assert ranges_of_channels([7, 8, 9, 3, 4]) == [[7, 10], [3, 5]]
    assert range_count([]) == len("".split(";")) == 1


def test_line_fit_equals_findcontinuum_linfit():
    rng = np.random.default_rng(8)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(50):
            n = int(rng.integers(3, 200))
            frequency = 230.0 + 0.001 * np.arange(n) + rng.uniform(-1, 1)
            transmission = 0.9 + 0.01 * rng.standard_normal(n)
            medians = 1e-3 * (1 + 0.1 * rng.standard_normal(n))
            spectrum = random_spectrum(rng, n)
            index = np.flatnonzero(rng.random(n) < 0.5)
            if index.size < 2:
                index = np.arange(n)
            mad = findContinuum.MAD(spectrum[index])
            for x, y, y_error in (
                (frequency, transmission, transmission * 0.001),
                (np.arange(n), medians, medians * 0.01),
                (index, spectrum[index], mad),
                (list(index), list(spectrum[index]), mad),
            ):
                expected = findContinuum.linfit(x, y, y_error)
                assert all(
                    _same(value, reference)
                    for value, reference in zip(
                        line_fit(x, y, y_error), expected, strict=True
                    )
                )
