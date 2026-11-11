"""
Imaging-weight UV gridding / degridding kernels (C++ extension).
"""

try:
    from astroviper.processing_functions.imaging.imaging_weighting.grid_imaging_weights_cpp._grid_imaging_weights_ext import (
        degrid_imaging_weights,
        grid_imaging_weights,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import grid imaging weights extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "grid_imaging_weights",
    "degrid_imaging_weights",
]
