"""With ``ASTROVIPER_KERNELS`` unset or empty, the hooked shims are unchanged.

Each hook is one ``if`` on the environment appended to its shim; these tests
show, in fresh interpreters, that when it is false the shim's entries are the
extension's own objects, no name is bound, ``astroviper._kernels`` is never
imported, and the process ends up with the same threads, modules and shim
namespaces as when the shims are imported with their hooks stripped from the
source. A trace directory alone hooks nothing.
"""

import json

import pytest

from tests.unit.kernels.conftest import HOOK, SHIMS, TINY_HOGBOM

IDENTITY = f"""
import json, sys
from importlib import import_module

out = {{}}
for kernel, (shim, ext, entries) in {SHIMS!r}.items():
    module = import_module(shim)
    extension = import_module(f"{{shim}}.{{ext}}")
    out[kernel] = {{
        "identical": all(getattr(module, e) is getattr(extension, e) for e in entries),
        "dispatch_bound": "_dispatch" in vars(module),
    }}
out["switch_imported"] = "astroviper._kernels" in sys.modules
print(json.dumps(out))
"""

#: Imports the shims, optionally with their hooks cut from the source, and
#: reports the threads (Linux: kernel tasks) and modules of the process and
#: each shim's namespace: every attribute, as the extension attribute it is
#: (by identity), a module, or its type and value.
FOOTPRINT = f"""
import importlib.abc, importlib.machinery, json, os, sys, threading, types
from importlib import import_module

SHIMS = [shim for shim, _, _ in {SHIMS!r}.values()]
EXTENSIONS = {{shim: ext for shim, ext, _ in {SHIMS!r}.values()}}
LOADING = ("__loader__", "__spec__", "__cached__", "__builtins__")


def describe(shim):
    module = sys.modules[shim]
    extension = sys.modules[f"{{shim}}.{{EXTENSIONS[shim]}}"]
    out = {{}}
    for name, value in vars(module).items():
        if name in LOADING:
            continue
        if getattr(extension, name, None) is value:
            out[name] = f"extension.{{name}}"
        elif isinstance(value, types.ModuleType):
            out[name] = f"module {{value.__name__}}"
        else:
            out[name] = f"{{type(value).__name__}} {{value!r}}"
    return out


class HookStripped(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_source(fullname)
        return compile(source[: source.index({HOOK!r})], self.path, "exec")


class StripHooks(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in SHIMS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        spec.loader = HookStripped(name, spec.origin)
        return spec


if sys.argv[1] == "stripped":
    sys.meta_path.insert(0, StripHooks())
for shim in SHIMS:
    import_module(shim)
tasks = "/proc/self/task"
threads = len(os.listdir(tasks)) if os.path.isdir(tasks) else threading.active_count()
print(json.dumps({{
    "threads": threads,
    "modules": sorted(sys.modules),
    "namespaces": {{shim: describe(shim) for shim in SHIMS}},
}}))
"""


@pytest.mark.parametrize(
    "env", [{}, {"ASTROVIPER_KERNELS": ""}], ids=["unset", "empty"]
)
def test_shim_entries_are_the_extension_objects(run_python, env):
    out = json.loads(run_python("-c", IDENTITY, **env))
    assert out.pop("switch_imported") is False
    assert out == {k: {"identical": True, "dispatch_bound": False} for k in SHIMS}


def test_same_footprint_as_hook_stripped_shims(run_python):
    hooked = json.loads(run_python("-c", FOOTPRINT, "hooked"))
    stripped = json.loads(run_python("-c", FOOTPRINT, "stripped"))
    assert hooked["threads"] == stripped["threads"]
    assert hooked["modules"] == stripped["modules"]
    assert hooked["namespaces"] == stripped["namespaces"]
    for shim, _, entries in SHIMS.values():
        assert {hooked["namespaces"][shim][e] for e in entries} == {
            f"extension.{e}" for e in entries
        }


def test_trace_alone_hooks_nothing(run_python, tmp_path):
    """``ASTROVIPER_KERNELS_TRACE`` without ``ASTROVIPER_KERNELS`` traces nothing."""
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        TINY_HOGBOM + "\ntiny_hogbom()\n" + IDENTITY,
        ASTROVIPER_KERNELS_TRACE=str(trace),
    )
    result = json.loads(out.strip().splitlines()[-1])
    assert result.pop("switch_imported") is False
    assert result == {k: {"identical": True, "dispatch_bound": False} for k in SHIMS}
    assert not trace.exists()
