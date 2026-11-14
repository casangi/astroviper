"""Processing functions of find_continuum_ultra: findContinuum's algorithm on
image-shaped maps and one-dimensional spectra."""

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

__all__ = [
    "round_half_up",
    "round_to_decimals",
    "round_to_significant_digits",
    "median_absolute_deviation",
    "scaled_median_absolute_deviation",
    "chauvenet_statistics",
    "one_event_sigma",
    "contiguous_runs",
    "channels_of_ranges",
    "ranges_of_channels",
    "range_count",
    "line_fit",
]
