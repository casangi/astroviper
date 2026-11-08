"""
The CUDA side of the GPU kernels: CuPy, the device, its memory limit and the
modules compiled from ``cuda/*.cu``.

CuPy is imported here only, by :func:`require_device` at the first call that
runs on the GPU. Importing this module imports only the standard library.
CUDA does not survive a fork: in a process forked after its parent opened the
device, the first GPU call raises ``RuntimeError`` (start worker processes
with the ``spawn`` or ``forkserver`` method instead).

``CUPY_GPU_MEMORY_LIMIT``
    The device memory that CuPy's memory pool may hold in this process: a byte
    count (such as ``6442450944``) or a percentage of the device's memory
    (such as ``25%``), the two forms CuPy reads. Any other value, such as
    ``4GiB``, raises ``ValueError`` at the first call that runs on the GPU.
``CUPY_CACHE_DIR``
    Where CuPy keeps the compiled kernels (keyed by their source, the compile
    options and the NVRTC version). It defaults to the home directory; a
    warning is logged once when it is unset. (``CUDA_CACHE_PATH``, the
    driver's cache, is used only when CuPy hands the driver PTX, on a GPU
    newer than its NVRTC.)
``CUDA_PATH``
    CuPy finds NVRTC and the CUDA headers with ``cuda.pathfinder``, in the
    interpreter's own site-packages (where the gpu extra installs them) or
    under ``CUDA_PATH``. An environment that sees CuPy through a ``.pth`` file
    or ``PYTHONPATH`` needs ``CUDA_PATH``: the error names the directory.
"""

import importlib.resources
import importlib.util
import logging
import os
import threading
import time
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

_logger = logging.getLogger("astroviper.kernels")
_lock = threading.Lock()
_device = None
_FORKED = "forked"  # _device in a child forked after the parent opened the device


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


class Device:
    """The CUDA device of this process, ready for the GPU kernels.

    Attributes: ``cp`` (the CuPy module), ``id``, ``name``, ``sm_count``,
    ``pool`` (CuPy's default memory pool, with the limit applied), ``stream``
    (a non-blocking stream for the kernels' work), ``lock`` (held by a call
    while it uses the device: one GPU kernel call at a time per process),
    ``errors`` (the CUDA exception types), ``init_s`` (seconds to import CuPy
    and create the context) and ``versions`` (CUDA driver, runtime, NVRTC).
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
    touching the device: a valid ``CUPY_GPU_MEMORY_LIMIT`` (``ValueError``)
    and an installed CuPy (``RuntimeError``)."""
    value = os.environ.get(LIMIT_ENV)
    if value is not None:
        parse_memory_limit(value)
    if importlib.util.find_spec("cupy") is None:
        raise RuntimeError(_NO_CUPY + "no module named 'cupy'")


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


def _wheel_cuda_path(cp):
    """The CUDA root of the NVIDIA wheels beside the imported CuPy, if it holds
    NVRTC."""
    root = Path(cp.__file__).resolve().parents[1].joinpath(*_WHEEL_CUDA)
    return root if (root / "lib" / "libnvrtc.so.13").is_file() else None


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
        root = None if os.environ.get("CUDA_PATH") else _wheel_cuda_path(cp)
        hint = (
            "; the NVIDIA wheels beside this CuPy hold it, but cuda.pathfinder "
            f"searches only the interpreter's own site-packages: set CUDA_PATH={root}"
            if root
            else ""
        )
        raise RuntimeError(
            "CuPy cannot load NVRTC, which compiles the GPU kernels at their first "
            "use (the gpu extra installs it with cupy-cuda13x[ctk]; CuPy also finds "
            f"it, and the CUDA headers, under CUDA_PATH){hint}: {e}"
        ) from e
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
