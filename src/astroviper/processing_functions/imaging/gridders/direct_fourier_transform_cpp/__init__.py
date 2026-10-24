"""
Direct Fourier transform between visibilities and an image (C++ extension).
"""

try:
    from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp._direct_fourier_transform_ext import (
        cos_sin_turns,
        dft_image,
        dft_image_uv_sampling,
        dft_predict,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import the direct Fourier transform extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "dft_image",
    "dft_image_uv_sampling",
    "dft_predict",
    "cos_sin_turns",
]
