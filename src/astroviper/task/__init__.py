# src/astroviper/__init__.py
from __future__ import annotations

from importlib import import_module, metadata

__all__ = [
    "__version__",
    # Namespaced high-level modules (distributed API)
    "imaging",
]

# Package version
try:
    __version__ = metadata.version("astroviper")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"