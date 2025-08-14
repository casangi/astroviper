"""Hogbom CLEAN deconvolution algorithm."""

# Import the compiled extension module
from .hogbom_deconvolve import *

__version__ = "0.1.0"
__all__ = ["clean", "hclean", "maximg", "get_dtype_name", "is_float32", "is_float64"]