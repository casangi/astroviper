"""Tests of find_continuum_ultra's final products against the vendored
findContinuum, the pipeline's post-processing and the simple port's runs."""

import random

import numpy as np

from astroviper.processing_functions.imaging.find_continuum_ultra.products import (
    bandwidth_warnings,
    constant_cube,
    continuum_frequency_ranges,
    is_all_continuum,
    selection_bandwidth,
    single_range_channel_fraction,
)
from tests.unit.processing_functions.imaging.find_continuum_ultra.vendored_oracle import (
    findContinuum,
    quiet,
    random_ranges,
    vendored_block,
)

N_CHANNELS = [8, 9, 40, 74, 75, 120, 128, 480, 1920, 3840]
# The last two decide the warnings differently from n_selected / n_channels at
# exactly 5 % or 33 % of the channels.
CHANNEL_WIDTHS = [
    976562.5,
    488281.25,
    244140.625,
    30517.578125,
    1953125.0,
    15.625e6,
    564.4531,
    3906250.0 / 3,
]
# The simple port's line_set, no_lines, tdm and eight_channels runs: final ranges,
# ranges before blue pruning, aggregate bandwidth (GHz), channel width (Hz),
# channels, the two warning fractions, all continuum and the frequency ranges
# of findContinuum's .dat file (fC 4210-4229); the first frequency is 230 GHz.
PORT_RUNS = [
    (
        [[6, 86], [154, 368], [433, 473]],
        [[3, 89], [153, 389], [432, 477]],
        0.326171875,
        976562.5,
        480,
        (0.6958333333333333, 0.9729166666666667),
        False,
        "230.005371094GHz~230.083496094GHz LSRK 230.149902344GHz~230.358886719GHz LSRK "
        "230.422363281GHz~230.461425781GHz LSRK ",
    ),
    (
        [[9, 465]],
        [[3, 477]],
        0.4453125,
        976562.5,
        480,
        (0.95, 0.95),
        True,
        "230.008300781GHz~230.453613281GHz LSRK ",
    ),
    (
        [[3, 107]],
        [[3, 107]],
        1.625,
        15.625e6,
        120,
        (0.8666666666666667, 0.8666666666666667),
        False,
        "230.039062500GHz~231.664062500GHz LSRK ",
    ),
    (
        [[1, 2], [6, 8]],
        [[1, 2]],
        0.0029296875,
        976562.5,
        8,
        (0.375, 0.875),
        False,
        "230.000488281GHz~230.001464844GHz LSRK 230.005371094GHz~230.007324219GHz LSRK ",
    ),
]


def _selection(ranges):
    """findContinuum's selection string of ``[start, stop)`` ranges."""
    return ";".join(f"{start}~{stop - 1}" for start, stop in ranges)


def _chan_info(n_channels, channel_increment):
    """findContinuum's spectral axis of a window starting at 230 GHz, with the
    simple port's types."""
    return (
        np.int64(n_channels),
        np.float64(230e9),
        np.float64(230e9 + (n_channels - 1) * channel_increment),
        np.float64(channel_increment),
    )


def _cases(rng, n_cases):
    """Windows and ranges: sorted, in another order (as findContinuum's
    single-narrow-group and low-spread fixes leave them) and overlapping."""
    for index in range(n_cases):
        n_channels = rng.choice(N_CHANNELS)
        ranges = random_ranges(rng, n_channels)
        if index % 3 == 1:
            rng.shuffle(ranges)
        elif index % 3 == 2:
            start = rng.randrange(n_channels)
            ranges.insert(
                rng.randrange(len(ranges) + 1),
                [start, rng.randint(start + 1, n_channels)],
            )
        yield n_channels, ranges


def _wide_range(rng, n_channels, fraction):
    """One range of about ``fraction`` of the channels."""
    width = min(n_channels, max(1, round(fraction * n_channels) + rng.randint(-2, 2)))
    start = rng.randint(0, n_channels - width)
    return [start, start + width]


def _pipeline_frequency_ranges(selection, reference_frequency, increment):
    """The pipeline's frequency ranges of a selection before the unit
    conversion, with the reference pixel at the first channel as in the
    pipeline's cubes (imaging_utils 89-104)."""
    frequencies = []
    for channel_range in selection.split(";"):
        c0, c1 = list(map(float, channel_range.split("~")))
        if c1 < c0:
            c0, c1 = c1, c0
        f0 = reference_frequency + (c0 - 0.5 - 0.0) * increment
        f1 = reference_frequency + (c1 + 0.5 - 0.0) * increment
        frequencies.append([f0, f1] if f0 < f1 else [f1, f0])
    return frequencies


def test_constant_cube():
    assert constant_cube(0.25, 0.25) is True
    assert constant_cube(np.float64(-1e-3), np.float64(2e-3)) is False
    assert constant_cube(np.nan, np.nan) is False
    rows = np.array([np.nan, 0.5, 0.5, np.nan])
    assert constant_cube(np.nanmin(rows), np.nanmax(rows)) is True


def test_selection_bandwidth_equals_findcontinuum():
    rng = random.Random(1)
    for _, ranges in _cases(rng, 3000):
        width = rng.choice(CHANNEL_WIDTHS) * rng.choice([1, -1])
        bandwidth = selection_bandwidth(ranges, width)
        assert type(bandwidth) is float
        assert bandwidth * 1e-9 == findContinuum.computeBandwidth(
            _selection(ranges), np.float64(width), 0
        )
    assert selection_bandwidth([], 976562.5) == 0.0
    assert findContinuum.computeBandwidth("", 976562.5, 0) == 0.0
    assert selection_bandwidth([[2, 5], [4, 6]], 1.0) == 5.0


def test_bandwidth_warnings_equal_findcontinuum():
    bandwidth_fraction = vendored_block(
        3750,
        "nchan, firstFreq, lastFreq, channelWidth = chanInfo",
        3752,
        "myfraction = aggBandwidthHz/float(channelWidth*nchan)",
        ["selection", "chanInfo"],
    )
    spread_fraction = vendored_block(
        3781,
        "nchan, firstFreq, lastFreq, channelWidth = chanInfo",
        3783,
        "myfraction = spread/float(nchan*channelWidth)",
        ["selection", "chanInfo"],
    )
    rng = random.Random(2)
    cases = [
        (n_channels, ranges, rng.choice(CHANNEL_WIDTHS) * rng.choice([1, 1, 1, -1]))
        for n_channels, ranges in _cases(rng, 3000)
    ]
    ties = [(480, 24), (1920, 96), (120, 6), (100, 5), (2000, 100), (100, 33)]
    ties += [(300, 99), (1000, 330), (3000, 990), (200, 66)]
    cases += [
        (n_channels, [[start, start + n_selected]], width)
        for n_channels, n_selected in ties
        for start in (0, n_channels - n_selected)
        for width in CHANNEL_WIDTHS
    ]
    n_warnings = np.zeros(2, dtype=int)
    for n_channels, ranges, increment in cases:
        selection, chan_info = _selection(ranges), _chan_info(n_channels, increment)
        warnings = bandwidth_warnings(ranges, n_channels, abs(increment), increment)
        with quiet():
            expected = findContinuum.gatherWarnings(selection, chan_info, 0.05, 0.33)
        assert [warnings.low_bandwidth, warnings.low_spread] == [
            text != "" for text in expected
        ]
        assert type(warnings.low_bandwidth) is bool
        assert (
            warnings.bandwidth_fraction
            == bandwidth_fraction(selection, chan_info)["myfraction"]
        )
        assert (
            warnings.spread_fraction
            == spread_fraction(selection, chan_info)["myfraction"]
        )
        assert type(warnings.spread_fraction) is float
        n_warnings += [warnings.low_bandwidth, warnings.low_spread]
    assert n_warnings.min() > 100


def test_bandwidth_warnings_follow_findcontinuums_order():
    width = 3906250.0 / 3
    assert not 5 / 100 < 0.05 and not 33 / 100 < 0.33
    assert bandwidth_warnings([[0, 5]], 100, width, width).low_bandwidth
    assert not bandwidth_warnings([[0, 5]], 100, 976562.5, 976562.5).low_bandwidth
    assert bandwidth_warnings([[0, 33]], 100, 564.4531, 564.4531).low_spread
    assert not bandwidth_warnings([[0, 33]], 100, 976562.5, 976562.5).low_spread
    assert bandwidth_warnings([[0, 100]], 100, 976562.5, -976562.5).low_bandwidth
    unsorted = bandwidth_warnings([[300, 310], [10, 20]], 400, 976562.5, 976562.5)
    assert unsorted.spread_fraction < 0 and unsorted.low_spread


def test_bandwidth_warnings_of_the_ports_runs():
    for ranges, _, _, width, n_channels, fractions, _, _ in PORT_RUNS:
        warnings = bandwidth_warnings(ranges, n_channels, width, width)
        assert (warnings.bandwidth_fraction, warnings.spread_fraction) == fractions
        assert not warnings.low_bandwidth and not warnings.low_spread
    assert bandwidth_warnings([[1, 2]], 8, 976562.5, 976562.5) == (
        False,
        True,
        0.125,
        0.125,
    )


def test_is_all_continuum_equals_findcontinuum():
    block = vendored_block(
        3129,
        "if selection.find(';') > 0 or aggregateBandwidth*1e9 < 0.5*np.abs(channelWidth*nchan):",
        3134,
        "allContinuum = allContinuumSelected(selectionPreBluePruning, nchan, spw=spw)",
        [
            "selection",
            "aggregateBandwidth",
            "channelWidth",
            "nchan",
            "selectionPreBluePruning",
            "spw",
        ],
    )
    rng = random.Random(3)
    n_all_continuum = 0
    for index in range(3000):
        n_channels = rng.choice(N_CHANNELS)
        if index % 4 == 0:
            ranges = random_ranges(rng, n_channels)
        else:
            ranges = [_wide_range(rng, n_channels, rng.choice([0.5, 0.9, 1.0]))]
        if rng.random() < 0.7:
            before = [_wide_range(rng, n_channels, rng.choice([0.91, 0.925]))]
        else:
            before = random_ranges(rng, n_channels)
        last = ranges if rng.random() < 0.8 else random_ranges(rng, n_channels)
        width = rng.choice(CHANNEL_WIDTHS)
        with quiet():
            expected = block(
                _selection(ranges),
                findContinuum.computeBandwidth(_selection(last), np.float64(width), 0),
                np.float64(width),
                np.int64(n_channels),
                _selection(before),
                "",
            )["allContinuum"]
        all_continuum = is_all_continuum(
            ranges, before, selection_bandwidth(last, width), width, n_channels
        )
        assert all_continuum == expected
        n_all_continuum += all_continuum
    assert 300 < n_all_continuum < 2700


def test_is_all_continuum_cases():
    for ranges, before, aggregate, width, n_channels, _, expected, _ in PORT_RUNS:
        all_continuum = is_all_continuum(
            ranges, before, aggregate * 1e9, width, n_channels
        )
        assert all_continuum is expected
    width = 976562.5
    for n_selected, n_channels, expected in [
        (444, 480, True),
        (443, 480, False),
        (68, 74, True),
        (67, 74, False),
        (69, 75, False),
    ]:
        ranges = [[0, n_selected]]
        assert (
            is_all_continuum(ranges, ranges, n_selected * width, width, n_channels)
            is expected
        )
    # Half of 128 channels of 976562.5 Hz survives the GHz round trip exactly.
    assert is_all_continuum([[0, 64]], [[0, 120]], 64 * width, width, 128)
    assert not is_all_continuum([[0, 63]], [[0, 120]], 63 * width, width, 128)
    assert not is_all_continuum([[0, 470]], [[0, 470]], 100 * width, width, 480)
    assert not is_all_continuum(
        [[0, 200], [210, 480]], [[0, 480]], 470 * width, width, 480
    )


def test_continuum_frequency_ranges_equal_the_pipelines():
    rng = random.Random(4)
    for _, ranges in _cases(rng, 3000):
        first_frequency = rng.choice([86e9, 230e9, rng.uniform(80e9, 950e9)])
        increment = rng.choice(CHANNEL_WIDTHS) * rng.choice([1, -1])
        frequency_ranges = continuum_frequency_ranges(
            ranges, first_frequency, increment
        )
        assert frequency_ranges == _pipeline_frequency_ranges(
            _selection(ranges), first_frequency, increment
        )
        assert all(
            type(frequency) is float for pair in frequency_ranges for frequency in pair
        )
    for ranges, _, _, width, _, _, _, dat in PORT_RUNS:
        frequency_ranges = continuum_frequency_ranges(ranges, np.float64(230e9), width)
        assert (
            "".join(
                f"{low * 1e-9:.9f}GHz~{high * 1e-9:.9f}GHz LSRK "
                for low, high in frequency_ranges
            )
            == dat
        )
    assert continuum_frequency_ranges([[6, 86], [0, 1]], 230e9, -976562.5) == [
        [230e9 - 85.5 * 976562.5, 230e9 - 5.5 * 976562.5],
        [230e9 - 0.5 * 976562.5, 230e9 + 0.5 * 976562.5],
    ]
    assert continuum_frequency_ranges([], 230e9, 976562.5) == []


def test_single_range_channel_fraction_equals_the_pipelines():
    rng = random.Random(5)
    for n_channels, ranges in _cases(rng, 1000):
        counts = findContinuum.countChannelsInRanges(_selection(ranges))
        expected = counts[0] / float(n_channels) if len(counts) == 1 else None
        fraction = single_range_channel_fraction(ranges, n_channels)
        assert fraction == expected and type(fraction) is type(expected)
    assert single_range_channel_fraction([[9, 465]], 480) == 0.95
    assert single_range_channel_fraction([[3, 107]], 120) == 104 / 120
    assert single_range_channel_fraction([[1, 2], [6, 8]], 8) is None
    assert single_range_channel_fraction([], 8) is None
