"""findContinuum's final products of a spectral window and what the pipeline
derives from them.

The constant-cube test that precedes findContinuum, the aggregate bandwidth and
the warnings of a selection, the all-continuum test, and the frequency ranges
and single-range fraction of the selection. Channel ranges are ``[start, stop)``
pairs relative to the spectral window, in findContinuum's list order. ``fC``
line numbers are those of the pipeline's ``findContinuum.py``, ``heur`` and
``imaging_utils`` line numbers those of its ``hif/heuristics/findcont.py`` and
``infrastructure/utils/imaging.py`` (pipeline 2026.3.0.7).
"""

from astroviper.processing_functions.imaging.find_continuum_ultra.channel_selection import (
    all_continuum_selected,
)
from astroviper.processing_functions.imaging.find_continuum_ultra.step_results import (
    BandwidthWarnings,
)

SMALL_BANDWIDTH_FRACTION = 0.05  # fC 486
SMALL_SPREAD_FRACTION = 0.33  # fC 486
HALF_BAND_FRACTION = 0.5  # fC 3129


def constant_cube(sky_minimum: float, sky_maximum: float) -> bool:
    """Whether every valid voxel of the spectral window holds the same value,
    in which case the pipeline does not run findContinuum (heur 89-94).

    Parameters
    ----------
    sky_minimum, sky_maximum : float
        Minimum and maximum of the valid voxels of the window, NaN when there
        is none.

    Returns
    -------
    bool
        ``sky_minimum == sky_maximum``: False when there is no valid voxel.
    """
    return bool(sky_minimum == sky_maximum)


def selection_bandwidth(ranges: list, channel_width: float) -> float:
    """Aggregate bandwidth of channel ranges [findContinuum
    ``computeBandwidth``] (fC 4405-4420).

    Parameters
    ----------
    ranges : list
        ``[start, stop)`` pairs; a channel in two ranges counts twice.
    channel_width : float
        Channel width in Hz; its sign is ignored.

    Returns
    -------
    float
        ``n_selected * abs(channel_width)`` in Hz. Times 1e-9 it is
        findContinuum's value in GHz, bit for bit.
    """
    n_selected = sum(stop - start for start, stop in ranges)
    return float(n_selected * abs(channel_width))


def bandwidth_warnings(
    ranges: list, n_channels: int, channel_width: float, channel_increment: float
) -> BandwidthWarnings:
    """The low-bandwidth and low-spread warnings of channel ranges
    [findContinuum ``gatherWarnings``] (fC 3387-3403, 3745-3788).

    The fractions are computed in findContinuum's order, which decides the
    warnings at exactly 5 % and 33 % of the channels for some channel widths.
    The spread runs from the first channel of the first range to the last
    channel of the last range in list order.

    Parameters
    ----------
    ranges : list
        ``[start, stop)`` pairs, at least one.
    n_channels : int
        Channels of the window.
    channel_width : float
        Channel width in Hz.
    channel_increment : float
        Signed frequency increment per channel in Hz. A negative increment
        makes the bandwidth fraction negative and so always warns
        (fC 3750-3752).

    Returns
    -------
    BandwidthWarnings
        Each warning is a fraction below its limit, 0.05 for the bandwidth and
        0.33 for the spread.
    """
    bandwidth_fraction = (
        (selection_bandwidth(ranges, channel_width) * 1e-9) * 1e9
    ) / float(channel_increment * n_channels)
    spread_fraction = ((ranges[-1][1] - ranges[0][0]) * channel_increment) / float(
        n_channels * channel_increment
    )
    return BandwidthWarnings(
        low_bandwidth=bool(bandwidth_fraction < SMALL_BANDWIDTH_FRACTION),
        low_spread=bool(spread_fraction < SMALL_SPREAD_FRACTION),
        bandwidth_fraction=float(bandwidth_fraction),
        spread_fraction=float(spread_fraction),
    )


def is_all_continuum(
    ranges: list,
    ranges_before_blue_pruning: list,
    aggregate_bandwidth: float,
    channel_width: float,
    n_channels: int,
) -> bool:
    """Whether the whole spectral window is continuum (fC 3129-3134).

    Parameters
    ----------
    ranges : list
        The final ``[start, stop)`` pairs.
    ranges_before_blue_pruning : list
        The ``[start, stop)`` pairs before blue pruning as the rounds left
        them, which the reversion and the low-spread fix do not update.
    aggregate_bandwidth : float
        The aggregate bandwidth set last, in Hz, which is not the bandwidth of
        ``ranges`` after an empty intersection of selections (fC 2233-2265,
        2554-2588).
    channel_width : float
        Channel width in Hz.
    n_channels : int
        Channels of the window.

    Returns
    -------
    bool
        False for more than one range or an aggregate bandwidth below half the
        band; otherwise whether ``ranges_before_blue_pruning`` is one range
        wide enough for all continuum [findContinuum
        ``allContinuumSelected``].
    """
    if len(ranges) > 1 or (aggregate_bandwidth * 1e-9) * 1e9 < (
        HALF_BAND_FRACTION * abs(channel_width * n_channels)
    ):
        return False
    return all_continuum_selected(ranges_before_blue_pruning, n_channels)


def continuum_frequency_ranges(
    ranges: list, first_frequency: float, channel_increment: float
) -> list[list[float]]:
    """The frequency ranges that channel ranges span between their outer
    channel edges, as the pipeline converts findContinuum's selection
    (heur 128, imaging_utils 62-108).

    Parameters
    ----------
    ranges : list
        ``[start, stop)`` pairs.
    first_frequency : float
        Frequency of the window's first channel in Hz.
    channel_increment : float
        Signed frequency increment per channel in Hz.

    Returns
    -------
    list of list of float
        Per range, in list order, the edges
        ``first_frequency + (start - 0.5) * channel_increment`` and
        ``first_frequency + (stop - 0.5) * channel_increment`` in Hz, the lower
        first.
    """
    frequency_ranges = []
    for start, stop in ranges:
        start_edge = float(first_frequency + (start - 0.5) * channel_increment)
        stop_edge = float(first_frequency + (stop - 0.5) * channel_increment)
        frequency_ranges.append(
            [start_edge, stop_edge]
            if start_edge < stop_edge
            else [stop_edge, start_edge]
        )
    return frequency_ranges


def single_range_channel_fraction(ranges: list, n_channels: int) -> float | None:
    """The fraction of the spectral window's channels in its only continuum
    range (heur 110-114).

    Parameters
    ----------
    ranges : list
        ``[start, stop)`` pairs.
    n_channels : int
        Channels of the window.

    Returns
    -------
    float or None
        ``(stop - start) / n_channels`` for exactly one range, else None
        (999.0 in the pipeline).
    """
    if len(ranges) != 1:
        return None
    start, stop = ranges[0]
    return (stop - start) / float(n_channels)
