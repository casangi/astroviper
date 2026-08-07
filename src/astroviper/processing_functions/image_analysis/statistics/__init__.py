"""Public processing-function API for mergeable image statistics.

These functions accept already selected, loaded xarray objects. They do not
parse CASA syntax, access storage, or construct GraphVIPER workflows; those
responsibilities belong to the node-task and distributed-application layers.
"""

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
