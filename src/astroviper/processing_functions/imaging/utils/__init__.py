"""Common (non-science) helpers for the imaging processing functions.

Holds the iteration-control logic, the :class:`ReturnDict` convergence
container, and small timing-bookkeeping helpers that are shared across the
imaging processing functions but are not themselves imaging science kernels.
"""

from astroviper.processing_functions.imaging.utils.return_dict import (
    ReturnDict,
    Key,
    FIELD_ACCUM,
    FIELD_SINGLE_VALUE,
)
from astroviper.processing_functions.imaging.utils.iteration_control import (
    IterationController,
    StopCode,
    merge_return_dicts,
    get_calculate_cycle_controls,
    get_peak_residual_from_returndict,
    get_masksum_from_returndict,
    get_iterations_done_from_returndict,
    get_max_psf_sidelobe_from_returndict,
    get_model_flux_from_returndict,
    format_deconvolve_dict,
    print_deconvolve_dict,
)
from astroviper.processing_functions.imaging.utils.timing import (
    accumulate_timing,
    format_timing_summary,
    IMAGING_TIMING_PHASES,
    IMAGING_TIMING_TOTAL_KEY,
)
from astroviper.processing_functions.imaging.utils.fft_sizing import (
    next_fft_friendly_size,
    padded_grid_size,
)
from astroviper.processing_functions.imaging.utils.visibility import (
    drop_auto_correlations,
)

__all__ = [
    "ReturnDict",
    "Key",
    "FIELD_ACCUM",
    "FIELD_SINGLE_VALUE",
    "IterationController",
    "StopCode",
    "merge_return_dicts",
    "get_calculate_cycle_controls",
    "get_peak_residual_from_returndict",
    "get_masksum_from_returndict",
    "get_iterations_done_from_returndict",
    "get_max_psf_sidelobe_from_returndict",
    "get_model_flux_from_returndict",
    "format_deconvolve_dict",
    "print_deconvolve_dict",
    "accumulate_timing",
    "format_timing_summary",
    "IMAGING_TIMING_PHASES",
    "IMAGING_TIMING_TOTAL_KEY",
    "next_fft_friendly_size",
    "padded_grid_size",
    "drop_auto_correlations",
]
