"""
The CUDA side of the GPU kernels: CuPy, the device, its memory limit and the
modules compiled from ``cuda/*.cu``.

CuPy is imported here only, by :func:`require_device` at the first call that
runs on the GPU (its cuFFT by :meth:`Device.fft_plan`, at the first plan).
Importing this module imports only the standard library.
CUDA does not survive a fork: in a process forked after its parent opened the
device, the first GPU call raises ``RuntimeError`` (start worker processes
with the ``spawn`` or ``forkserver`` method instead).

``CUPY_GPU_MEMORY_LIMIT``
    The device memory that CuPy's memory pool may hold in this process: a byte
    count (such as ``6442450944``) or a percentage of the device's memory
    (such as ``25%``), the two forms CuPy reads; ``0`` and ``0%`` set no
    limit, as in CuPy. Any other value, such as ``4GiB``, raises
    ``ValueError`` at the first call that runs on the GPU.
``CUPY_CACHE_DIR``
    Where CuPy keeps the compiled kernels (keyed by their source, the compile
    options and the NVRTC version). It defaults to the home directory; a
    warning is logged once when it is unset. (``CUDA_CACHE_PATH``, the
    driver's cache, is used only when CuPy hands the driver PTX, on a GPU
    newer than its NVRTC.)
``CUDA_PATH``
    CuPy finds NVRTC, the CUDA headers and cuFFT with ``cuda.pathfinder``, in
    the interpreter's own site-packages (where the gpu extra installs them) or
    under ``CUDA_PATH``. An environment that sees CuPy through a ``.pth`` file
    or ``PYTHONPATH`` needs ``CUDA_PATH``, set before CuPy is first imported
    (zarr imports it): :func:`check_environment` (and :func:`check_cufft`),
    and so ``--check``, fails without it and names the directory.

Device memory that a GPU kernel keeps between its calls (``fft=gpu``'s cuFFT
plans) is registered with :func:`register_cache`, so that a call of another
kernel that would not fit frees it first (:meth:`Device.reclaim`).
"""

import importlib.resources
import importlib.util
import logging
import os
import threading
import time
import traceback
from pathlib import Path

#: The memory limit of CuPy's pool, in bytes or as a percentage.
LIMIT_ENV = "CUPY_GPU_MEMORY_LIMIT"
#: CuPy's kernel cache, which should not default to the home directory.
CACHE_ENV = "CUPY_CACHE_DIR"
#: The CUDA root of the NVIDIA wheels (CUDA 13) in a site-packages directory.
_WHEEL_CUDA = ("nvidia", "cu13")
#: NVRTC options: no multiply-add contraction. CuPy adds -ftz=true, which the
#: kernels never depend on (see the top of ``cuda/hogbom_exact.cu``).
OPTIONS = ("--std=c++17", "--fmad=false")

#: cuFFT's library in the NVIDIA wheels (CUDA 13).
_CUFFT_LIBRARY = "libcufft.so.12"

_logger = logging.getLogger("astroviper.kernels")
_lock = threading.Lock()
_device = None
_FORKED = "forked"  # _device in a child forked after the parent opened the device
_caches = {}  # name -> evict(): see register_cache()


def _after_fork_in_child():
    # The parent's CUDA context is unusable here, and its locks may have been
    # held by threads that the child does not have.
    global _device, _lock
    _lock = threading.Lock()
    if _device is not None:
        _device = _FORKED


os.register_at_fork(after_in_child=_after_fork_in_child)


def parse_memory_limit(value):
    """The limit a ``CUPY_GPU_MEMORY_LIMIT`` value sets, as CuPy reads it.

    Returns ``{"size": bytes}`` or ``{"fraction": f}`` (``0 <= f <= 1``), the
    arguments of CuPy's ``MemoryPool.set_limit``, for exactly the values CuPy
    accepts (a value that ends with ``%`` is a percentage, any other is read
    by ``int``). Raises ``ValueError`` naming the two forms for anything else.
    """
    try:
        if value.endswith("%"):
            percent = float(value[:-1])
            if 0.0 <= percent <= 100.0:
                return {"fraction": percent / 100.0}
        else:
            size = int(value)
            if size >= 0:
                return {"size": size}
    except ValueError:
        pass
    raise ValueError(
        f"{LIMIT_ENV}={value!r}: expected a byte count (such as 6442450944) "
        "or a percentage of the device memory (such as 25%)"
    )


def register_cache(name, evict):
    """Register ``evict``, which drops the device memory that a GPU kernel
    keeps between its calls (such as cuFFT plans) and returns whether it held
    any. :meth:`Device.reclaim` calls it, under the device lock, for a call
    that would not fit otherwise. A later registration of ``name`` replaces
    the earlier one."""
    _caches[name] = evict


def clear_frames(error):
    """Clear the variables of the frames of ``error``'s traceback and of the
    exceptions chained to it: they hold the call's device buffers, which its
    memory pool could not free while a caller keeps the exception. The
    tracebacks still print."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        error = error.__cause__ or error.__context__


class Device:
    """The CUDA device of this process, ready for the GPU kernels.

    Attributes: ``cp`` (the CuPy module), ``id``, ``name``, ``sm_count``,
    ``pool`` (CuPy's default memory pool, with the limit applied), ``stream``
    (a non-blocking stream for the kernels' work), ``lock`` (held by a call
    while it uses the device: one GPU kernel call at a time per process),
    ``errors`` (the CUDA exception types), ``fft_errors`` (cuFFT's, once
    :meth:`fft_plan` has loaded it), ``init_s`` (seconds to import CuPy and
    create the context) and ``versions`` (CUDA driver, runtime, NVRTC).
    """

    def __init__(self, cp, init_s, nvrtc):
        self.cp = cp
        self.id = cp.cuda.runtime.getDevice()
        properties = cp.cuda.runtime.getDeviceProperties(self.id)
        self.name = properties["name"].decode(errors="replace")
        self.sm_count = int(properties["multiProcessorCount"])
        self.pool = cp.get_default_memory_pool()
        with cp.cuda.Device(self.id):
            self.stream = cp.cuda.Stream(non_blocking=True)
        self.lock = threading.Lock()
        self.errors = (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError)
        self.fft_errors = ()
        self.init_s = init_s
        self.versions = {
            "driver": cp.cuda.runtime.driverGetVersion(),
            "runtime": cp.cuda.runtime.runtimeGetVersion(),
            "nvrtc": nvrtc,
        }
        self.compile_s = {}
        self._modules = {}
        self._max_blocks = {}

    def module(self, name, name_expressions, cooperative=False):
        """The kernels ``name_expressions`` of ``cuda/<name>.cu``: a dict of
        the module's functions by name expression, compiled by NVRTC (or loaded
        from CuPy's cache) once per process. A ``cooperative`` module's kernels
        are launched cooperatively (all their blocks resident at once)."""
        key = (name, tuple(name_expressions), cooperative)
        functions = self._modules.get(key)
        if functions is not None:
            return functions
        cp = self.cp
        t0 = time.perf_counter()
        try:
            source = (
                importlib.resources.files(__package__)
                .joinpath("cuda", f"{name}.cu")
                .read_text(encoding="utf-8")
            )
            with cp.cuda.Device(self.id):
                module = cp.RawModule(
                    code=source,
                    options=OPTIONS,
                    name_expressions=list(name_expressions),
                    enable_cooperative_groups=cooperative,
                )
                functions = {e: module.get_function(e) for e in name_expressions}
        except (
            OSError,  # the source, or CuPy's cache
            cp.cuda.compiler.CompileException,
            RuntimeError,
            *self.errors,
        ) as e:
            raise RuntimeError(f"cannot compile or load cuda/{name}.cu: {e}") from e
        self.compile_s[key] = time.perf_counter() - t0
        self._modules[key] = functions
        return functions

    def available(self):
        """``(bytes, limit)``: the device memory a call can still allocate
        (the device's free memory and the pool's free blocks, within the
        pool's limit, never negative) and the pool's limit (0: none)."""
        free, _ = self.cp.cuda.runtime.memGetInfo()
        pool = self.pool
        available = free + pool.free_bytes()
        limit = pool.get_limit()
        if limit:
            available = min(available, limit - pool.used_bytes())
        return max(available, 0), limit

    def reclaim(self):
        """Drop what the GPU kernels keep on the device between their calls
        (:func:`register_cache`) and free the pool's cached blocks, for a call
        that would not fit otherwise; under :attr:`lock`. Returns whether a
        cache held anything."""
        held = False
        for evict in list(_caches.values()):
            held = bool(evict()) or held
        self.pool.free_all_blocks()
        return held

    def fft_plan(self, array):
        """A cuFFT plan of the complex 2-D transform of ``array``, a C-ordered
        CuPy array of two axes, with its work area from the memory pool. CuPy's
        cuFFT is loaded at the first plan: ``RuntimeError`` (as
        :func:`check_cufft` raises it) when cuda.pathfinder cannot find it."""
        from cupy.cuda import cufft
        from cupyx.scipy.fft import get_fft_plan

        self.fft_errors = (cufft.CuFFTError,)
        try:
            return get_fft_plan(array, axes=(0, 1))
        except RuntimeError as e:  # DynamicLibNotFoundError is one
            pathfinder = _pathfinder()
            if pathfinder and isinstance(e, pathfinder.DynamicLibNotFoundError):
                raise _no_cufft(self.cp.__file__, e) from e
            raise

    def max_blocks(self, kernel, threads):
        """The most blocks of ``threads`` threads of ``kernel`` that the device
        holds at once: the largest grid of a cooperative launch."""
        key = (id(kernel), threads)
        blocks = self._max_blocks.get(key)
        if blocks is None:
            per_sm = self.cp.cuda.driver.occupancyMaxActiveBlocksPerMultiprocessor(
                kernel.kernel.ptr, threads, 0
            )
            if per_sm < 1:
                raise RuntimeError(
                    f"a block of {threads} threads of a GPU kernel does not fit on "
                    f"a multiprocessor of {self.name}"
                )
            blocks = self._max_blocks[key] = int(per_sm) * self.sm_count
        return blocks


_NO_CUPY = (
    "CuPy is not available (install the gpu extra, "
    "pip install 'astroviper[gpu]', on Linux x86-64): "
)


def check_environment():
    """What the GPU kernels need that can be checked without importing CuPy or
    touching the device: a valid ``CUPY_GPU_MEMORY_LIMIT`` (``ValueError``),
    an installed CuPy, and NVRTC and the CUDA headers where CuPy's
    ``cuda.pathfinder`` looks for them at the first compile (``RuntimeError``;
    a CuPy without ``cuda.pathfinder`` finds them its own way)."""
    value = os.environ.get(LIMIT_ENV)
    if value is not None:
        parse_memory_limit(value)
    spec = importlib.util.find_spec("cupy")
    if spec is None:
        raise RuntimeError(_NO_CUPY + "no module named 'cupy'")
    try:
        from cuda import pathfinder
    except ImportError:
        return
    try:
        pathfinder.load_nvidia_dynamic_lib("nvrtc")
    except pathfinder.DynamicLibNotFoundError as e:
        raise _no_toolkit(spec.origin, e) from e
    if pathfinder.find_nvidia_header_directory("cudart") is None:
        raise _no_toolkit(spec.origin, "cuda_runtime.h is not found")


def check_cufft():
    """That CuPy's ``cuda.pathfinder`` finds cuFFT, which ``fft=gpu`` loads at
    its first plan, without touching the device (``RuntimeError``, naming the
    ``CUDA_PATH`` to set when the NVIDIA wheels lie beside CuPy; a CuPy
    without ``cuda.pathfinder`` finds it its own way)."""
    spec = importlib.util.find_spec("cupy")
    if spec is None:
        raise RuntimeError(_NO_CUPY + "no module named 'cupy'")
    pathfinder = _pathfinder()
    if pathfinder is None:
        return
    try:
        pathfinder.load_nvidia_dynamic_lib("cufft")
    except pathfinder.DynamicLibNotFoundError as e:
        raise _no_cufft(spec.origin, e) from e


def _pathfinder():
    """``cuda.pathfinder``, or None for a CuPy without it."""
    try:
        from cuda import pathfinder
    except ImportError:
        return None
    return pathfinder


def require_device():
    """The :class:`Device` of this process, set up at the first call.

    Validates ``CUPY_GPU_MEMORY_LIMIT`` (``ValueError``), imports CuPy, creates
    the CUDA context on the current device and applies the limit to CuPy's
    memory pool. Raises ``RuntimeError`` when CuPy is not installed, no CUDA
    device can be used or the process was forked after its parent had opened
    the device. Thread-safe; later calls return the same device.
    """
    global _device
    device = _device
    if device is None:
        with _lock:
            if _device is None:
                _device = _open()
            device = _device
    if device is _FORKED:
        raise RuntimeError(
            "the CUDA device was opened before this process was forked from its "
            "parent, and CUDA does not survive a fork: start worker processes "
            "with the 'spawn' or 'forkserver' method"
        )
    return device


def _cuda_path_hint(cupy_init, library, them):
    """``"; ... set CUDA_PATH=<root>"`` when the NVIDIA wheels beside CuPy
    (``cupy_init``, its ``__init__.py``) hold ``library`` (``them``: the
    pronoun) and ``CUDA_PATH`` is unset, else ``""``."""
    root = Path(cupy_init).resolve().parents[1].joinpath(*_WHEEL_CUDA)
    if os.environ.get("CUDA_PATH") or not (root / "lib" / library).is_file():
        return ""
    return (
        f"; the NVIDIA wheels beside this CuPy hold {them}, but cuda.pathfinder "
        f"searches only the interpreter's own site-packages: set CUDA_PATH={root}"
    )


def _no_toolkit(cupy_init, error):
    """The ``RuntimeError`` for NVRTC or the CUDA headers not found, naming the
    ``CUDA_PATH`` to set when the NVIDIA wheels lie beside CuPy
    (``cupy_init``, its ``__init__.py``) and ``CUDA_PATH`` is unset."""
    hint = _cuda_path_hint(cupy_init, "libnvrtc.so.13", "them")
    return RuntimeError(
        "CuPy cannot find NVRTC and the CUDA headers, which compile the GPU kernels "
        "at their first use (the gpu extra installs them with cupy-cuda13x[ctk]; "
        f"CuPy also finds them under CUDA_PATH){hint}: {error}"
    )


def _no_cufft(cupy_init, error):
    """The ``RuntimeError`` for cuFFT not found (see :func:`_no_toolkit`)."""
    hint = _cuda_path_hint(cupy_init, _CUFFT_LIBRARY, "it")
    return RuntimeError(
        "CuPy cannot find cuFFT, which fft=gpu loads at its first transform (the "
        "gpu extra installs it with cupy-cuda13x[ctk]; CuPy also finds it under "
        f"CUDA_PATH){hint}: {error}"
    )


def _open():
    t0 = time.perf_counter()
    value = os.environ.get(LIMIT_ENV)
    limit = None if value is None else parse_memory_limit(value)
    try:
        import cupy as cp
    except (ImportError, OSError) as e:
        raise RuntimeError(_NO_CUPY + str(e)) from e
    errors = (cp.cuda.runtime.CUDARuntimeError, cp.cuda.driver.CUDADriverError)
    try:
        count = cp.cuda.runtime.getDeviceCount()
        if count > 0:
            with cp.cuda.Device():
                cp.cuda.runtime.free(0)  # creates the context
    except errors as e:
        raise RuntimeError(f"no usable CUDA device: {e}") from e
    if count < 1:
        raise RuntimeError("no usable CUDA device: none is visible")
    try:
        nvrtc = cp.cuda.nvrtc.getVersion()
    except Exception as e:  # cuda.pathfinder's DynamicLibNotFoundError, or a CUDA error
        raise _no_toolkit(cp.__file__, e) from e
    try:
        with cp.cuda.Device():
            if limit is not None:
                cp.get_default_memory_pool().set_limit(**limit)
            device = Device(cp, time.perf_counter() - t0, nvrtc)
    except errors as e:
        raise RuntimeError(f"no usable CUDA device: {e}") from e
    if not os.environ.get(CACHE_ENV):
        _logger.warning(
            "%s unset: CuPy keeps the compiled GPU kernels in the home directory",
            CACHE_ENV,
        )
    return device
