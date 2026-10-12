"""Common (non-science) helpers for the imaging processing functions.

Holds the cycle control logic, the :class:`ImagingDict` convergence
container, and small timing-bookkeeping helpers that are shared across the
imaging processing functions but are not themselves imaging science kernels.
"""

from astroviper.processing_functions.imaging.utils.cycle_control import (
    CycleController,
    StopCode,
    build_residual_imaging_dict,
    copy_residual_entropy,
    format_imaging_dict,
    get_iterations_done_from_imaging_dict,
    get_masksum_from_imaging_dict,
    get_max_psf_sidelobe_from_imaging_dict,
    get_model_flux_from_imaging_dict,
    get_peak_residual_from_imaging_dict,
    get_per_cycle_limits,
    merge_imaging_dicts,
    print_imaging_dict,
    resolve_auto_max_cycles,
    validate_entropy_params,
    validate_max_cycles,
)
from astroviper.processing_functions.imaging.utils.fft_sizing import (
    next_fft_friendly_size,
    padded_grid_size,
)
from astroviper.processing_functions.imaging.utils.imaging_dict import (
    FIELD_ACCUM,
    FIELD_SINGLE_VALUE,
    ITERATION_HISTORY_FIELDS,
    ImagingDict,
    Key,
    imaging_dict_to_dataframe,
)
from astroviper.processing_functions.imaging.utils.timing import (
    IMAGING_TIMING_PHASES,
    IMAGING_TIMING_TOTAL_KEY,
    accumulate_timing,
    format_timing_summary,
)
from astroviper.processing_functions.imaging.utils.visibility import (
    drop_auto_correlations,
)

__all__ = [
    "ImagingDict",
    "imaging_dict_to_dataframe",
    "Key",
    "FIELD_ACCUM",
    "FIELD_SINGLE_VALUE",
    "ITERATION_HISTORY_FIELDS",
    "CycleController",
    "StopCode",
    "merge_imaging_dicts",
    "build_residual_imaging_dict",
    "copy_residual_entropy",
    "validate_entropy_params",
    "validate_max_cycles",
    "resolve_auto_max_cycles",
    "get_per_cycle_limits",
    "get_peak_residual_from_imaging_dict",
    "get_masksum_from_imaging_dict",
    "get_iterations_done_from_imaging_dict",
    "get_max_psf_sidelobe_from_imaging_dict",
    "get_model_flux_from_imaging_dict",
    "format_imaging_dict",
    "print_imaging_dict",
    "accumulate_timing",
    "format_timing_summary",
    "IMAGING_TIMING_PHASES",
    "IMAGING_TIMING_TOTAL_KEY",
    "next_fft_friendly_size",
    "padded_grid_size",
    "drop_auto_correlations",
]
