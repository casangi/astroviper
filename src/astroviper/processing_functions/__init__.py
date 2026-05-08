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
]

# Package version
try:
    __version__ = metadata.version("astroviper")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# --- Lazy module proxying ---
# We expose astroviper.<module> -> astroviper.distributed_graphs.<module>
# but avoid importing distributed until first access.

_lazy_modules = {
    "flagging": "astroviper.distributed_graphs.flagging",
    "calibration": "astroviper.distributed_graphs.calibration",
    "imaging": "astroviper.distributed_graphs.imaging",
    "image_analysis": "astroviper.distributed_graphs.image_analysis",
    "visibility_manipulation": "astroviper.distributed_graphs.visibility_manipulation",
    "utils": "astroviper.distributed_graphs.utils",
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
    from astroviper.distributed_graphs import (
        flagging,
        calibration,
        imaging,
        image_analysis,
        visibility_manipulation,
        utils,
    )
