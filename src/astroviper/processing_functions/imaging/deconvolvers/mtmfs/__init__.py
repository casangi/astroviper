"""
Multi-Term Multi-Frequency Synthesis (MTMFS) deconvolution.

A casacore-free, stateless port of the model update behind CASA
tclean's ``deconvolver='mtmfs'`` (``SDAlgorithmMSMFS`` driving
``MultiTermMatrixCleaner``), wrapped with pybind11 as free functions in the
same style as :mod:`~astroviper.processing_functions.imaging.deconvolvers.hogbom`.

``clean`` runs one model update in place on Python-owned numpy buffers: the
``residual`` and ``model`` Taylor-term stacks are written into directly and
no copies are made on the C++ side. Every call takes all of its inputs and
keeps nothing between calls. One to four Taylor terms are supported, requiring
one, three, five, or seven PSF planes respectively.

Layout: images are row-major ``(ny, nx)``; Taylor-term stacks are
``(nterms, ny, nx)`` and the PSF stack is ``(2 * nterms - 1, ny, nx)``.
"""

# Stop codes shared by the deconvolvers (deconvolvers/include/plane_stop_test.hpp);
# nothing left to clean in the search region is STOP_THRESHOLD.
STOP_NONE = 0
STOP_MAX_ITER = 1
STOP_THRESHOLD = 2
STOP_DIVERGED = 4

try:
    from astroviper.processing_functions.imaging.deconvolvers.mtmfs._mtmfs_ext import (
        clean,
        hessian,
        principal_solution,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import MTMFS CLEAN extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "clean",
    "hessian",
    "principal_solution",
    "STOP_NONE",
    "STOP_MAX_ITER",
    "STOP_THRESHOLD",
    "STOP_DIVERGED",
]
