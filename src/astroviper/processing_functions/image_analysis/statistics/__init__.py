"""Composable numerical reductions for image statistics."""

from astroviper.processing_functions.image_analysis.statistics.reductions import (
    STATISTIC_FUNCTIONS,
    create_statistics_state,
    finalize_statistics_state,
    merge_statistics_states,
    statistics_max,
    statistics_mean,
    statistics_min,
    statistics_npts,
    statistics_sum,
)

__all__ = [
    "STATISTIC_FUNCTIONS",
    "create_statistics_state",
    "finalize_statistics_state",
    "merge_statistics_states",
    "statistics_max",
    "statistics_mean",
    "statistics_min",
    "statistics_npts",
    "statistics_sum",
]
