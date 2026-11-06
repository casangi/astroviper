"""
The ``ASTROVIPER_KERNELS`` switch: grammar, registry, dispatch and trace.

Only the standard library is imported here, so that a hooked shim costs no
third-party import, and no AstroVIPER code imports CuPy unless a ``gpu``
kernel runs (third-party packages, zarr among them, may import it themselves).
"""

import functools
import json
import logging
import os
import resource
import socket
import threading
import time
from importlib import import_module
from types import MappingProxyType

#: Selects the implementation of each kernel (see the package docstring).
ENV = "ASTROVIPER_KERNELS"
#: Directory that receives one JSON line per hooked call when set.
TRACE_ENV = "ASTROVIPER_KERNELS_TRACE"

#: Every implementation name the grammar knows.
IMPLEMENTATIONS = ("ref", "fast", "gpu", "ref1")
#: Kernel names that are planned but not hooked yet; selecting one is an error.
RESERVED = ("grid", "fft", "pol", "dft_image", "mtmfs", "vis", "ring", "psf_fit", "pb")

_IMAGING = "astroviper.processing_functions.imaging"

#: Hooked entries of each kernel: the names its shim rebinds.
_ENTRIES = {
    "dft": ("dft_predict",),
    "hogbom": ("clean_cube", "clean_cube_many_threads"),
    "asp": ("clean", "clean_cube"),
}

#: Valid (kernel, implementation) pairs: the module that provides the kernel's
#: entries under the same names and signatures. A static table, extended as
#: each implementation lands; availability is checked at the first call.
_REGISTRY = {
    "dft": {
        "ref": f"{_IMAGING}.gridders.direct_fourier_transform_cpp"
        "._direct_fourier_transform_ext",
        "fast": "astroviper._kernels.fast.dft",
    },
    "hogbom": {"ref": f"{_IMAGING}.deconvolvers.hogbom._hogbom_ext"},
    "asp": {
        "ref": f"{_IMAGING}.deconvolvers.aspclean._aspclean_ext",
        "fast": "astroviper._kernels.fast.aspclean",
    },
}

_logger = logging.getLogger("astroviper.kernels")
# Guards the bookkeeping below; never held across an import or a kernel call.
_lock = threading.Lock()
_resolved = {}  # (kernel, entry, impl) -> callable
_logged = set()  # (kernel, impl) announced on the logger
_traced = set()  # (kernel, entry, impl) that wrote a trace line
_trace_failed = set()  # trace directories whose write failure was logged
_reported = threading.local()  # .value = (kernel, path): see report_path()


def _names(names):
    return ", ".join(sorted(names))


@functools.lru_cache(maxsize=8)
def parse(value):
    """Selection made by an ``ASTROVIPER_KERNELS`` value.

    Parameters
    ----------
    value : str
        Comma-separated tokens, case-insensitive, whitespace ignored. A bare
        ``<impl>`` (or ``*=<impl>``) selects ``<impl>`` for every kernel that
        has it; ``<kernel>=<impl>`` selects it for one kernel. Later tokens
        win. ``""`` selects the reference everywhere.

    Returns
    -------
    MappingProxyType
        Read-only ``{kernel: impl}`` over every registered kernel; kernels no
        token names stay ``"ref"``.

    Raises
    ------
    ValueError
        Naming the valid options, for a malformed token, a ``:variant``
        suffix, an unknown or reserved kernel, an unknown implementation, or a
        pair that is not registered.
    """
    selection = dict.fromkeys(_REGISTRY, "ref")
    if not value:
        return MappingProxyType(selection)

    def error(token, reason):
        return ValueError(f"{ENV}={value!r}, token {token!r}: {reason}")

    for raw in value.split(","):
        token = "".join(raw.split()).lower()
        if ":" in token:
            raise error(token, "variants (':...') are not supported")
        kernel, equals, name = token.rpartition("=")
        if not name or "=" in kernel or (equals and not kernel):
            raise error(token, "expected '<impl>' or '<kernel>=<impl>'")
        if name not in IMPLEMENTATIONS:
            raise error(
                token,
                f"unknown implementation {name!r}; "
                f"implementations: {_names(IMPLEMENTATIONS)}",
            )
        if kernel in ("", "*"):
            kernels = [k for k, impls in _REGISTRY.items() if name in impls]
            if not kernels:
                raise error(token, f"no kernel implements {name!r} yet")
        elif kernel in RESERVED:
            raise error(token, f"kernel {kernel!r} is reserved, not implemented yet")
        elif kernel not in _REGISTRY:
            raise error(
                token, f"unknown kernel {kernel!r}; kernels: {_names(_REGISTRY)}"
            )
        elif name not in _REGISTRY[kernel]:
            raise error(
                token,
                f"{kernel!r} has no implementation {name!r}; "
                f"its implementations: {_names(_REGISTRY[kernel])}",
            )
        else:
            kernels = [kernel]
        selection.update(dict.fromkeys(kernels, name))
    return MappingProxyType(selection)


def selected():
    """Selection ``{kernel: impl}`` that ``ASTROVIPER_KERNELS`` makes now."""
    return parse(os.environ.get(ENV, ""))


def impl(kernel, entry, name):
    """Entry ``entry`` of implementation ``name`` of ``kernel``.

    The direct API for tests and benchmarks, which works whether or not the
    shims are hooked. Raises ``ValueError`` for an unregistered combination
    and ``ImportError`` when the implementation's module is not available.
    """
    key = (kernel, entry, name)
    function = _resolved.get(key)
    if function is not None:
        return function
    if entry not in _ENTRIES.get(kernel, ()):
        raise ValueError(f"{entry!r} is not a hooked entry of kernel {kernel!r}")
    module_name = _REGISTRY[kernel].get(name)
    if module_name is None:
        raise ValueError(f"{kernel!r} has no implementation {name!r}")
    # Outside _lock: the import system serialises the import of each module,
    # and an implementation module may itself call impl() while it imports.
    try:
        module = import_module(module_name)
    except ImportError as e:
        if e.name != module_name:
            raise
        raise ImportError(
            f"{kernel}={name}: module {module_name} is not available "
            "(not built, or skipped by its compile probe)",
            name=module_name,
        ) from e
    with _lock:
        function = _resolved.setdefault(key, getattr(module, entry))
        announce = (kernel, name) not in _logged
        _logged.add((kernel, name))
    if announce:
        isa = getattr(module, "ISA", None)
        _logger.info(
            "kernel %s=%s from %s%s",
            kernel,
            name,
            module_name,
            f" (ISA {isa})" if isa else "",
        )
    return function


def dispatch(kernel, function):
    """Wrap a hooked reference entry so that each call runs the selection.

    Called by the shim hooks at import, and only when ``ASTROVIPER_KERNELS``
    is set and non-empty. The value is validated now, so a typo fails at the
    first shim import. Each call re-reads the variable (the parse is cached),
    so a process can switch between calls.

    Parameters
    ----------
    kernel : str
        Registered kernel name.
    function : callable
        The reference entry the shim imported from its extension.

    Returns
    -------
    callable
        A wrapper named like the shim attribute (``__module__`` = the shim,
        ``__name__`` = ``__qualname__`` = the entry), with the entry's
        ``__doc__`` and ``__wrapped__`` = ``function``.
    """
    entry = function.__name__
    if entry not in _ENTRIES.get(kernel, ()):
        raise ValueError(f"{entry!r} is not a hooked entry of kernel {kernel!r}")
    selected()

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        name = selected()[kernel]
        target = impl(kernel, entry, name)
        trace_dir = os.environ.get(TRACE_ENV)
        if not trace_dir:
            return target(*args, **kwargs)
        return _call_traced(trace_dir, kernel, entry, name, target, args, kwargs)

    # The shim's names, so that pickling by reference (Dask) finds the wrapper
    # (a pybind11 entry's own __qualname__ is not an attribute path).
    wrapper.__module__ = _REGISTRY[kernel]["ref"].rpartition(".")[0]
    wrapper.__qualname__ = entry
    return wrapper


def report_path(kernel, path):
    """Report the path that the running call of a hooked entry of ``kernel`` takes.

    For implementations with more than one path, such as ``dft=fast`` (``"S"``
    or ``"declined"``): called once per call, once the path is known and before
    the part that may raise for the call's arguments (the reference's run of a
    declined call), so that the call's trace line has this ``path`` instead of
    the implementation's name. A call that raises before it reports is traced
    with the implementation's name and the error. The report is kept per
    thread and belongs to the innermost traced call, if that call is of
    ``kernel``: the trace clears it before each call and restores the
    enclosing call's afterwards, so a report never reaches another call, a
    call of another kernel or another thread. Outside a traced call it has no
    effect.
    """
    _reported.value = (kernel, path)


def _arrays(args, kwargs):
    """``{argument: array}`` for the array arguments (positional by index)."""
    named = {str(i): a for i, a in enumerate(args)} | kwargs
    return {
        k: a for k, a in named.items() if hasattr(a, "shape") and hasattr(a, "dtype")
    }


def _cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def _call_traced(trace_dir, kernel, entry, name, target, args, kwargs):
    """Run ``target`` and append its trace line to ``<trace_dir>``.

    The call's result or exception is returned or raised unchanged: a trace
    line that cannot be written is logged once per directory and dropped.
    ``path`` is the path the implementation reported (``report_path``), else
    its name. ``error`` is the type name of the exception the call raised, or
    ``None``. ``rusage_cpu_s`` is the process-wide CPU time of the call
    (``cpu_scope="process"``: it includes the C++ worker threads, and any
    other thread running meanwhile). ``t_end`` is ``t_start + wall_s``: one
    monotonic interval, anchored at the epoch time of the call's start.
    """
    error = None
    enclosing = getattr(_reported, "value", None)
    _reported.value = None
    cpu0, t0, p0 = _cpu_seconds(), time.time(), time.perf_counter()
    try:
        return target(*args, **kwargs)
    except BaseException as e:
        error = type(e).__name__
        raise
    finally:
        wall, cpu = time.perf_counter() - p0, _cpu_seconds() - cpu0
        reported, _reported.value = _reported.value, enclosing
        path = reported[1] if reported and reported[0] == kernel else name
        try:
            arrays = _arrays(args, kwargs)
            with _lock:
                first = (kernel, entry, name) not in _traced
                _traced.add((kernel, entry, name))
            host, pid = socket.gethostname(), os.getpid()
            record = {
                "kernel": kernel,
                "entry": entry,
                "impl": name,
                "path": path,
                "error": error,
                "t_start": t0,
                "t_end": t0 + wall,
                "wall_s": wall,
                "rusage_cpu_s": cpu,
                "cpu_scope": "process",
                "first_call": first,
                "host": host,
                "pid": pid,
                "thread": threading.get_ident(),
                "shapes": {k: list(a.shape) for k, a in arrays.items()},
                "dtypes": {k: str(a.dtype) for k, a in arrays.items()},
            }
            line = json.dumps(record) + "\n"
            os.makedirs(trace_dir, exist_ok=True)
            path = os.path.join(trace_dir, f"kernels-{host}-{pid}.jsonl")
            with _lock, open(path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:  # never let the trace change the call's outcome
            with _lock:
                report = trace_dir not in _trace_failed
                _trace_failed.add(trace_dir)
            if report:
                _logger.warning(
                    "%s=%s: cannot write trace lines (%s: %s); "
                    "the kernel calls are not traced",
                    TRACE_ENV,
                    trace_dir,
                    type(e).__name__,
                    e,
                )
