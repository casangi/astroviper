"""
Adaptive Scale Pixel (Asp / AAspClean) deconvolution.

A dependency-free port of CASA's ``AspMatrixCleaner`` (the algorithm behind
``SDAlgorithmAAspClean``), wrapped with pybind11. The ``clean`` routine
operates in place on Python-owned numpy buffers (no copies on the C++ side):
the ``residual`` and ``model`` arrays are written into directly.
"""

try:
    from astroviper.processing_functions.imaging.deconvolvers.aspclean._aspclean_ext import (
        clean,
        clean_cube,
        convolve_centered,
        psf_gaussian_width,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import Asp CLEAN extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "clean",
    "clean_cube",
    "psf_gaussian_width",
    "convolve_centered",
]
