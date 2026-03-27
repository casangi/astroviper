"""
Prolate spheroidal convolution gridder (C++ extension).
"""

try:
    from ._prolate_spheroidal_grid_ext import (
        prolate_spheroidal_grid,
        prolate_spheroidal_grid_uv_sampling,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import prolate spheroidal grid extension module. "
        "Make sure it is compiled and available."
    ) from e

__all__ = [
    "prolate_spheroidal_grid",
    "prolate_spheroidal_grid_uv_sampling",
]
