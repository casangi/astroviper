"""Public processing-function API for mergeable image statistics.

These functions accept already selected, loaded xarray objects. They do not
parse CASA syntax, access storage, or construct GraphVIPER workflows; those
responsibilities belong to the node-task and distributed-application layers.
"""

from astroviper.processing_functions.image_analysis.statistics.reductions import (
    DEFAULT_STATISTICS,
    STATISTIC_FUNCTIONS,
    create_statistics_state,
    finalize_statistics_state,
    merge_statistics_states,
    statistics_mad_sigma,
    statistics_max,
    statistics_maxpos,
    statistics_mean,
    statistics_median,
    statistics_min,
    statistics_minpos,
    statistics_n_pixels,
    statistics_peak,
    statistics_rms,
    statistics_std,
    statistics_sum,
    statistics_sumsq,
)

__all__ = [
    "DEFAULT_STATISTICS",
    "STATISTIC_FUNCTIONS",
    "create_statistics_state",
    "finalize_statistics_state",
    "merge_statistics_states",
    "statistics_max",
    "statistics_maxpos",
    "statistics_mean",
    "statistics_mad_sigma",
    "statistics_median",
    "statistics_min",
    "statistics_minpos",
    "statistics_n_pixels",
    "statistics_peak",
    "statistics_rms",
    "statistics_std",
    "statistics_sum",
    "statistics_sumsq",
]
