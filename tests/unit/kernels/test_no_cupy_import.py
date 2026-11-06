"""No AstroVIPER code imports CuPy unless a ``gpu`` kernel runs.

Third-party packages may import CuPy themselves (zarr does, at import, when it
is installed), after which an ``import cupy`` is served from ``sys.modules``
without reaching any finder. So every import route is watched: a meta-path
finder for first imports, and wrappers of ``builtins.__import__`` and
``importlib.import_module`` for every import, cached or not. Each records and
refuses an import of a ``cupy*`` package whose importing frame is AstroVIPER
code. A hooked process then checks its selection, imports every shim and the
production modules that call them, and makes a hooked call. A positive
control, an AstroVIPER-path ``import cupy`` made after zarr's, must be caught.
"""

import importlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pytest

from tests.unit.kernels.conftest import ALL_REF, SHIMS, TINY_HOGBOM

#: Production modules that import the hooked shims.
CALLERS = [
    "astroviper.processing_functions.imaging.deconvolution",
    "astroviper.processing_functions.imaging.direct_fourier_transform",
]

BLOCKED = f"""
import builtins, importlib, importlib.abc, importlib.util, json, os, sys

PACKAGE = os.path.dirname(importlib.util.find_spec("astroviper").origin) + os.sep
IMPORT_MACHINERY = ("<frozen importlib", importlib.__file__)
blocked = []


def is_cupy(name, level=0):
    return level == 0 and name.partition(".")[0].startswith("cupy")


def check(name, frame):
    while frame and frame.f_code.co_filename.startswith(IMPORT_MACHINERY):
        frame = frame.f_back
    if frame and frame.f_code.co_filename.startswith(PACKAGE):
        blocked.append(f"{{name}} from {{frame.f_code.co_filename}}:{{frame.f_lineno}}")
        raise ModuleNotFoundError(f"{{name}} imported by AstroVIPER", name=name)


class BlockCupy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if is_cupy(name):
            check(name, sys._getframe(1))
        return None


original_import, original_import_module = builtins.__import__, importlib.import_module


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if is_cupy(name, level):
        check(name, sys._getframe(1))
    return original_import(name, globals, locals, fromlist, level)


def guarded_import_module(name, package=None):
    if is_cupy(name):
        check(name, sys._getframe(1))
    return original_import_module(name, package)


sys.meta_path.insert(0, BlockCupy())
builtins.__import__ = guarded_import
importlib.import_module = guarded_import_module

from astroviper._kernels.__main__ import main

assert main(["--check"]) == 0
for module in [shim for shim, _, _ in {SHIMS!r}.values()] + {CALLERS!r}:
    importlib.import_module(module)
{TINY_HOGBOM}
tiny_hogbom()
found = list(blocked)

# Positive control, made a cached import: zarr has imported CuPy by now when it
# is installed, else this (non-AstroVIPER) frame does.
if "cupy" not in sys.modules and importlib.util.find_spec("cupy") is not None:
    import cupy
cached = "cupy" in sys.modules
control = compile("import cupy", PACKAGE + "_kernels/_control.py", "exec")
try:
    exec(control, {{}})
    caught = False
except ModuleNotFoundError:
    caught = True
print("RESULT", json.dumps({{"blocked": found, "cached": cached, "caught": caught}}))
"""


def _installed(relative):
    """Whether the extension module ``astroviper/<relative>`` is installed: its
    file, found without importing it (an import would import the reference
    shims, and fix the process ISA, at collection)."""
    package = Path(importlib.util.find_spec("astroviper").origin).parent
    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    return any((package / (relative + suffix)).is_file() for suffix in suffixes)


FAST_DFT = pytest.param(
    "dft=fast",
    marks=pytest.mark.skipif(
        not _installed("_kernels/fast/dft/_dft_fast_ext"),
        reason="fast DFT module not built",
    ),
)
FAST_ASP = pytest.param(
    "asp=fast",
    marks=pytest.mark.skipif(
        not _installed("_kernels/fast/aspclean/_aspclean_fast_ext"),
        reason="fast Asp module not built",
    ),
)


@pytest.mark.parametrize("selection", ["ref", ALL_REF, FAST_DFT, FAST_ASP])
def test_hooked_process_never_imports_cupy(run_python, selection):
    out = run_python("-c", BLOCKED, ASTROVIPER_KERNELS=selection)
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result["blocked"] == []
    assert result["caught"] is True, "the blocker missed an AstroVIPER import"
    if importlib.util.find_spec("cupy") is not None:
        assert result["cached"] is True, "the control was not a cached import"
