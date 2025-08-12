"""
Hogbom CLEAN algorithm implementation

This module provides a high-performance implementation of the Hogbom CLEAN
algorithm for radio astronomy image deconvolution.
"""

import hogbom_clean as _hc

# Re-export the main functions
clean = _hc.clean
clean_with_components = _hc.clean_with_components
clean_multipol = _hc.clean_multipol
find_peak = _hc.find_peak
CleanParams = _hc.CleanParams
CleanResults = _hc.CleanResults

__version__ = "0.1.0"
__all__ = [
    "clean",
    "clean_with_components",
    "clean_multipol", 
    "find_peak",
    "CleanParams",
    "CleanResults",
]