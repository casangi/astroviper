"""
The CUDA side of the GPU kernels: CuPy, the device, its memory limit and the
modules compiled from ``cuda/*.cu``.

CuPy is imported here only, by :func:`require_device` at the first call that
runs on the GPU. Importing this module imports only the standard library.

``CUPY_GPU_MEMORY_LIMIT``
    The device memory that CuPy's memory pool may hold in this process: a byte
    count (such as ``6442450944``) or a percentage of the device's memory
    (such as ``25%``), the two forms CuPy reads. Any other value, such as
    ``4GiB``, raises ``ValueError`` at the first call that runs on the GPU.
``CUPY_CACHE_DIR``, ``CUDA_CACHE_PATH``
    Where CuPy keeps the compiled kernels (keyed by their source, the compile
    options and the NVRTC version) and the driver its own cache. Both default
    to the home directory; a warning is logged once when either is unset.
"""

import importlib.resources
import logging
import os
import threading
import time

#: The memory limit of CuPy's pool, in bytes or as a percentage.
LIMIT_ENV = "CUPY_GPU_MEMORY_LIMIT"
#: Kernel caches, which should not default to the home directory.
CACHE_ENVS = ("CUPY_CACHE_DIR", "CUDA_CACHE_PATH")
#: NVRTC options: no multiply-add contraction. CuPy adds -ftz=true, which the
#: kernels never depend on (see the top of ``cuda/hogbom_exact.cu``).
OPTIONS = ("--std=c++17", "--fmad=false")

_logger = logging.getLogger("astroviper.kernels")
_lock = threading.Lock()
_device = None


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
        source = (
            importlib.resources.files(__package__)
            .joinpath("cuda", f"{name}.cu")
            .read_text(encoding="utf-8")
        )
        t0 = time.perf_counter()
        try:
            with cp.cuda.Device(self.id):
                module = cp.RawModule(
                    code=source,
                    options=OPTIONS,
                    name_expressions=list(name_expressions),
                    enable_cooperative_groups=cooperative,
                )
                functions = {e: module.get_function(e) for e in name_expressions}
        except (cp.cuda.compiler.CompileException, RuntimeError, *self.errors) as e:
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
            blocks = self._max_blocks[key] = max(1, int(per_sm)) * self.sm_count
        return blocks


def require_device():
    """The :class:`Device` of this process, set up at the first call.

    Validates ``CUPY_GPU_MEMORY_LIMIT`` (``ValueError``), imports CuPy, creates
    the CUDA context on the current device and applies the limit to CuPy's
    memory pool. Raises ``RuntimeError`` when CuPy is not installed or no
    CUDA device can be used. Thread-safe; later calls return the same device.
    """
    global _device
    device = _device
    if device is None:
        with _lock:
            if _device is None:
                _device = _open()
            device = _device
    return device


def _open():
    t0 = time.perf_counter()
    value = os.environ.get(LIMIT_ENV)
    limit = None if value is None else parse_memory_limit(value)
    try:
        import cupy as cp
    except (ImportError, OSError) as e:
        raise RuntimeError(
            "CuPy is not available (install the gpu extra, "
            f"pip install 'astroviper[gpu]', on Linux x86-64): {e}"
        ) from e
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
        raise RuntimeError(
            "CuPy cannot load NVRTC, which compiles the GPU kernels at their first "
            "use (the gpu extra installs it with cupy-cuda13x[ctk]; CuPy also finds "
            f"it under CUDA_PATH): {e}"
        ) from e
    try:
        with cp.cuda.Device():
            if limit is not None:
                cp.get_default_memory_pool().set_limit(**limit)
            device = Device(cp, time.perf_counter() - t0, nvrtc)
    except errors as e:
        raise RuntimeError(f"no usable CUDA device: {e}") from e
    unset = [name for name in CACHE_ENVS if not os.environ.get(name)]
    if unset:
        _logger.warning(
            "%s unset: the GPU kernel caches default to the home directory",
            " and ".join(unset),
        )
    return device
