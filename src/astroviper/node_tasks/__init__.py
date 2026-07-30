# src/astroviper/__init__.py
from __future__ import annotations

from importlib import metadata

from astroviper.node_tasks import image_analysis, imaging

__all__ = [
    "__version__",
    # Namespaced high-level modules (distributed API)
    "imaging",
    "image_analysis",
]

# Package version
try:
    __version__ = metadata.version("astroviper")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"
