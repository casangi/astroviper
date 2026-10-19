"""Hogbom CLEAN with terminal local point-component relaxation."""

try:
    from astroviper.processing_functions.imaging.deconvolvers.relaxed_hogbom._relaxed_hogbom_ext import (
        clean_cube,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import the RelaxedHogbom extension. Build AstroVIPER first."
    ) from exc

__all__ = ["clean_cube"]
