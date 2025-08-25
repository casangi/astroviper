# src/astroviper/__init__.py
from __future__ import annotations

from importlib import import_module, metadata
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = [
    "__version__",
    # Namespaced high-level modules (distributed API)
    "flagging",
    "calibration",
    "imaging",
    "image_analysis",
    "visibility_manipulation",
    "utils",
    # Expose low-level namespace explicitly
    "core",
    "distributed",
]

# Package version
try:
    __version__ = metadata.version("astroviper")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# --- Lazy module proxying ---
# We expose astroviper.<module> -> astroviper.distributed.<module>
# but avoid importing distributed until first access.

_lazy_modules = {
    "flagging": "astroviper.distributed.flagging",
    "calibration": "astroviper.distributed.calibration",
    "imaging": "astroviper.distributed.imaging",
    "image_analysis": "astroviper.distributed.image_analysis",
    "visibility_manipulation": "astroviper.distributed.visibility_manipulation",
    "utils": "astroviper.distributed.utils",
    # Namespaces themselves:
    "core": "astroviper.core",
    "distributed": "astroviper.distributed",
}


def __getattr__(name: str) -> ModuleType:
    target = _lazy_modules.get(name)
    if target is None:
        raise AttributeError(f"module 'astroviper' has no attribute {name!r}")
    mod = import_module(target)
    globals()[name] = mod
    return mod


def __dir__():
    return sorted(list(globals().keys()) + list(_lazy_modules.keys()))


if TYPE_CHECKING:
    # These imports are for type checkers/IDE only (no runtime cost)
    from . import core, distributed
    from .distributed import (
        flagging,
        calibration,
        imaging,
        image_analysis,
        visibility_manipulation,
        utils,
    )
