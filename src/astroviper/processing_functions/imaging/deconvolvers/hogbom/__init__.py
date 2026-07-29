"""
Hogbom CLEAN implementation
"""

try:
    from astroviper.processing_functions.imaging.deconvolvers.hogbom._hogbom_ext import (
        clean,
        clean_cube,
        clean_cube_many_threads,
        get_dtype_name,
        is_float32,
        is_float64,
        maximg,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import Hogbom CLEAN extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "maximg",
    "clean",
    "clean_cube",
    "clean_cube_many_threads",
    "get_dtype_name",
    "is_float32",
    "is_float64",
]
