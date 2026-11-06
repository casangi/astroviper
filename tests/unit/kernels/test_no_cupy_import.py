"""No AstroVIPER code imports CuPy unless a ``gpu`` kernel runs.

A meta-path finder blocks (and records) every import of ``cupy`` whose
importing frame is AstroVIPER code; third-party packages that probe for CuPy
themselves (zarr, dask, astropy) are let through. A hooked process then
checks its selection, imports every shim and makes a hooked call.
"""

import json

import pytest

from tests.unit.kernels.conftest import ALL_REF, SHIMS, TINY_HOGBOM

BLOCKED = f"""
import importlib, importlib.abc, importlib.util, json, os, sys

PACKAGE = os.path.dirname(importlib.util.find_spec("astroviper").origin) + os.sep
blocked = []


class BlockCupy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.partition(".")[0] != "cupy":
            return None
        frame = sys._getframe(1)
        while frame and (
            frame.f_code.co_filename.startswith("<frozen importlib")
            or frame.f_code.co_filename == importlib.__file__
        ):
            frame = frame.f_back
        if frame and frame.f_code.co_filename.startswith(PACKAGE):
            blocked.append(f"{{frame.f_code.co_filename}}:{{frame.f_lineno}}")
            raise ModuleNotFoundError(f"{{name}} imported by AstroVIPER")
        return None


sys.meta_path.insert(0, BlockCupy())
from astroviper._kernels.__main__ import main

assert main(["--check"]) == 0
for shim, _, _ in {SHIMS!r}.values():
    importlib.import_module(shim)
{TINY_HOGBOM}
tiny_hogbom()
print("RESULT", json.dumps(blocked))
"""


@pytest.mark.parametrize("selection", ["ref", ALL_REF])
def test_hooked_process_never_imports_cupy(run_python, selection):
    out = run_python("-c", BLOCKED, ASTROVIPER_KERNELS=selection)
    assert json.loads(out.rpartition("RESULT ")[2]) == []
