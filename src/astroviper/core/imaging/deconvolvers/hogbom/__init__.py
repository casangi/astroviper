"""
Hogbom CLEAN implementation
"""

try:
    from ._hogbom_ext import *
except ImportError as e:
    raise ImportError(
        "Failed to import Hogbom CLEAN extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "maximg",
    "clean",
    "clean_nc",
    "get_dtype_name",
    "is_float32",
    "is_float64",
]
