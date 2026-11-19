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
if __import__("os").environ.get("ASTROVIPER_KERNELS"):
    from astroviper._kernels import dispatch as _dispatch

    clean_cube = _dispatch("hogbom", clean_cube)
    clean_cube_many_threads = _dispatch("hogbom", clean_cube_many_threads)
