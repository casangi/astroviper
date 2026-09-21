"""
Multi-Term Multi-Frequency Synthesis (MTMFS) deconvolution.

A casacore-free port of the minor cycle behind CASA tclean's
``deconvolver='mtmfs'`` (``SDAlgorithmMSMFS`` driving
``MultiTermMatrixCleaner``), wrapped with pybind11 as the stateful
:class:`MultiTermCleaner`. It mirrors the CASA lifecycle: construct once per
image geometry, call ``set_psf`` once per PSF Taylor-term stack (this builds
the PSF/scale transforms and the Taylor Hessians), then call ``clean`` after
every residual update cycle. The ``residual`` and ``model`` Taylor-term stacks
are Python-owned numpy buffers that are updated in place; no copies are made
on the C++ side.

Layout: images are row-major ``(ny, nx)``; Taylor-term stacks are
``(nterms, ny, nx)`` and the PSF stack is ``(2 * nterms - 1, ny, nx)``.
"""

try:
    from astroviper.processing_functions.imaging.deconvolvers.mtmfs._mtmfs_ext import (
        MultiTermCleaner,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import MTMFS CLEAN extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = ["MultiTermCleaner"]
