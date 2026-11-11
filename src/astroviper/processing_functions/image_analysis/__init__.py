"""Image-analysis processing functions."""

from astroviper.processing_functions.image_analysis.make_mask import make_mask
from astroviper.processing_functions.image_analysis.moments import moments
from astroviper.processing_functions.image_analysis.plane_statistics import (
    calculate_plane_statistics,
    concatenate_plane_statistics,
    plane_statistics_to_dataframe,
)
from astroviper.processing_functions.image_analysis.statistics import (
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
    "make_mask",
    "moments",
    "calculate_plane_statistics",
    "concatenate_plane_statistics",
    "plane_statistics_to_dataframe",
]
