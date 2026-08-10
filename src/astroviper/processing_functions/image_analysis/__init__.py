"""Image-analysis processing functions."""

from astroviper.processing_functions.image_analysis.make_mask import make_mask
from astroviper.processing_functions.image_analysis.moments import moments
from astroviper.processing_functions.image_analysis.statistics import (
    STATISTIC_FUNCTIONS,
    create_statistics_state,
    finalize_statistics_state,
    merge_statistics_states,
    statistics_max,
    statistics_maxpos,
    statistics_mean,
    statistics_medabsdevmed,
    statistics_median,
    statistics_min,
    statistics_minpos,
    statistics_npts,
    statistics_rms,
    statistics_sigma,
    statistics_sum,
    statistics_sumsq,
)

__all__ = [
    "STATISTIC_FUNCTIONS",
    "create_statistics_state",
    "finalize_statistics_state",
    "merge_statistics_states",
    "statistics_max",
    "statistics_maxpos",
    "statistics_mean",
    "statistics_medabsdevmed",
    "statistics_median",
    "statistics_min",
    "statistics_minpos",
    "statistics_npts",
    "statistics_rms",
    "statistics_sigma",
    "statistics_sum",
    "statistics_sumsq",
    "make_mask",
    "moments",
]
