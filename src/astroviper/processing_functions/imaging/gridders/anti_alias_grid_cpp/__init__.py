"""
Anti-aliasing convolution gridder (C++ extension).
"""

try:
    from astroviper.processing_functions.imaging.gridders.anti_alias_grid_cpp._anti_alias_grid_ext import (
        anti_alias_degrid,
        anti_alias_grid,
        anti_alias_grid_uv_sampling,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import anti-aliasing grid extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "anti_alias_grid",
    "anti_alias_grid_uv_sampling",
    "anti_alias_degrid",
]
