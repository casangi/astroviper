"""C++ (pybind11) point-source visibility kernel with direction-dependent beams."""

try:
    from astroviper.processing_functions.simulation.visibility_kernel_cpp._visibility_kernel_ext import (
        bessel_j1,
        calculate_visibilities,
    )
except ImportError as e:  # pragma: no cover - only when the extension was not built
    raise ImportError(
        "Failed to import the simulation visibility kernel extension module. "
        "Make sure it is compiled and available (pip install -e . --no-build-isolation)."
    ) from e

__all__ = ["calculate_visibilities", "bessel_j1"]
