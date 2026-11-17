"""
GPU uv <-> lm transforms, selected by ``ASTROVIPER_KERNELS=fft=gpu``.

``ifft_uv_to_lm`` and ``fft_lm_to_uv`` take the arguments of the reference
functions (``processing_functions/imaging/fft_normalize_anti_alias_gridder.py``)
and return what they return, transformed on the GPU by CuPy's cuFFT: per plane
of the last two axes, the inverse transform divided by the plane's pixel count
and the forward one unscaled, between the reference's shifts:
fftshift(F(ifftshift(x))) (see ``cuda/fft_shift.cu``). When both axes are even
those are the reference's checkerboard sign flips, cb_centred * F(cb * x) with
cb = (-1)^(i + j) and cb_centred = (-1)^(i - n0/2 + j - n1/2); otherwise they
are the two rolls, ifftshift before and fftshift after. The result is in the
complex dtype of the input's precision (``complex_dtype`` for
``fft_lm_to_uv``), of the input's shape and in C order, for any leading axes.
With ``overwrite_input`` the input's buffer may be destroyed: it receives the
result, which the call returns, when the input is already of the result's
dtype; without it the input is left as it is. ``processing_function_threads``
and ``fft_backend`` change nothing on the GPU; they keep their meaning for the
calls that run on the CPU.

The results are not the reference's bit for bit: cuFFT, FFTW (pyfftw) and
pocketfft (scipy.fft) order and round their butterflies differently. With eps
the machine epsilon of the dtype (2**-52 for complex128, 2**-23 for complex64),
N the pixels of a plane, x the plane and y its exact transform
(fftshift(F(ifftshift(x)))), the tests find both the GPU's result and the
reference's within the usual bound of an FFT (Higham's, about 3.3 eps log2(N)
for radix 2),
``||result - y||_2 <= ACCURACY * eps * log2(N) * ||y||_2``, so also element
by element, where ``||y||_2 = s * sqrt(N) * ||x||_2`` (s = 1 forward, 1/N
inverse), against long double transforms. It is a tested bound, not a
guarantee: cuFFT transforms sizes with large prime factors by Bluestein's
algorithm, which Higham's analysis does not cover. That holds for finite
planes whose transform does not overflow; elsewhere both give non-finite
values, not necessarily in the same places. The GPU's results are
deterministic: the same call gives the same bits. One plane is on the device
at a time; each call ends by freeing the cached blocks of CuPy's memory pool,
and the cuFFT plans of the last ``PLAN_CACHE`` plane shapes are kept with
their work areas (about one plane each), for all threads. A call of
``fft=gpu`` or ``hogbom=gpu`` that would not fit in device memory drops them
first (``_runtime.Device.reclaim``).

Each call takes one path, reported to the kernel switch for its trace line:

``"gpu"``
    The GPU ran the call. Its trace line also has ``init_s``, the seconds the
    call spent on the first use of CUDA (the context, CuPy's import, the
    kernels' compile or load) and on making the cuFFT plan of a new plane
    shape, ``device_s``, the seconds from its device buffers to its result
    (the input's conversion on the host when its dtype is not the result's,
    then per plane the upload, the shifts, the transform and the download,
    which the host waits for; not the allocation of the buffers and the
    plan), and ``versions``, those of the CUDA driver, runtime and NVRTC.
``"routed"``
    The reference ran it, as given, because the GPU would not win or cannot
    hold it: its planes have fewer than ``ASTROVIPER_KERNELS_GPU_FFT_MIN_PIXELS``
    pixels (default ``DEFAULT_MIN_PIXELS``), an axis of one pixel or more
    pixels than a cuFFT plan takes (2**31 - 1), or its device buffers and cuFFT
    plan do not fit in the device memory that is free and that
    ``CUPY_GPU_MEMORY_LIMIT`` allows, once the kept plans are dropped (a plan
    that cuFFT cannot allocate, ``CUFFT_ALLOC_FAILED``, does not fit; logged
    once). The trace line has ``reason``.
``"declined"``
    The reference ran it, as given, because the GPU entries do not take its
    arguments as given, so that its conversions and errors stay the
    reference's: arguments that do not bind to the reference's parameters (or
    a reference whose parameters are not those the GPU entries read), an
    input that is not a C-contiguous ``numpy.ndarray`` of complex64,
    complex128, float32 or float64 with two or more axes and no empty one,
    axes other than the last two in order, a ``complex_dtype`` other than
    complex64 and complex128, a thread count that is not an integer of 1 or
    more, an ``fft_backend`` that the reference's ``check_fft_backend`` refuses
    or cannot import, an ``overwrite_input`` that is not a bool, and a
    read-only input with ``overwrite_input``.

The kernels plugin (``tests/kernels_plugin.py``) sets the pixel floor to 0 for
an ``fft=gpu`` session, so that the GPU runs the tests' planes. Errors are not
retried on the CPU: without CuPy, a CUDA device or cuFFT the first call that
needs the GPU raises ``RuntimeError`` (an invalid ``CUPY_GPU_MEMORY_LIMIT``
raises ``ValueError``), and so does a failing CUDA or cuFFT call, after which
an input given with ``overwrite_input`` may hold some planes of the result
(those before the failing one, in C order).
"""

import collections
import functools
import inspect
import logging
import operator
import os
import time
from typing import NamedTuple

import numpy as np

from astroviper._kernels._switch import impl, report_path
from astroviper._kernels.gpu import _runtime

__all__ = [
    "ACCURACY",
    "DECLINED",
    "DEFAULT_MIN_PIXELS",
    "GPU",
    "MIN_PIXELS_ENV",
    "PLAN_CACHE",
    "ROUTED",
    "check_environment",
    "fft_lm_to_uv",
    "ifft_uv_to_lm",
]

#: The paths of a call (see the module docstring).
GPU, ROUTED, DECLINED = "gpu", "routed", "declined"
#: Calls whose planes have fewer pixels run on the CPU.
MIN_PIXELS_ENV = "ASTROVIPER_KERNELS_GPU_FFT_MIN_PIXELS"
DEFAULT_MIN_PIXELS = 128 * 128
#: The constant of the accuracy bound (see the module docstring).
ACCURACY = 4.0
#: The plane shapes whose cuFFT plans are kept.
PLAN_CACHE = 4

_logger = logging.getLogger("astroviper.kernels")
_THREADS = 256  # threads per block of the shift kernels
_GRID_ROWS = 65535  # the most blocks along y (rows) of a launch
_INT_MAX = 2**31 - 1
_FORWARD, _INVERSE = -1, 1  # CUFFT_FORWARD, CUFFT_INVERSE
_CUFFT_ALLOC_FAILED = 2  # the result code of cuFFT's CuFFTError
_C64, _C128 = np.dtype(np.complex64), np.dtype(np.complex128)
#: The input dtypes the GPU takes, and the result's dtype for each (NumPy's).
_COMPLEX = {
    np.dtype(np.float32): _C64,
    np.dtype(np.float64): _C128,
    _C64: _C64,
    _C128: _C128,
}
_TYPES = {_C64: "c64", _C128: "c128"}  # the complex types of cuda/fft_shift.cu
_KERNELS = tuple(f"{k}<{t}>" for t in _TYPES.values() for k in ("checkerboard", "roll"))
_INPUT = {"ifft_uv_to_lm": "grid_2d", "fft_lm_to_uv": "image"}
#: The reference's parameters, in order: the arguments that _bind reads.
_PARAMETERS = {
    "ifft_uv_to_lm": (
        "grid_2d",
        "fft_plane_dims",
        "processing_function_threads",
        "fft_backend",
        "overwrite_input",
    ),
    "fft_lm_to_uv": (
        "image",
        "fft_plane_dims",
        "processing_function_threads",
        "fft_backend",
        "complex_dtype",
        "overwrite_input",
    ),
}
#: (n0, n1, dtype) -> cuFFT plan, the latest used last (under the device lock).
_plans = collections.OrderedDict()
_warned = False  # whether a call that does not fit in device memory was logged


def ifft_uv_to_lm(*args, **kwargs):
    """``ifft_uv_to_lm`` of the reference, with the same arguments and results
    (to an FFT's rounding), on the GPU or on the CPU (see the module docstring),
    which also raises the reference's exceptions."""
    return _call("ifft_uv_to_lm", args, kwargs)


def fft_lm_to_uv(*args, **kwargs):
    """``fft_lm_to_uv`` of the reference, with the same arguments and results;
    see :func:`ifft_uv_to_lm`."""
    return _call("fft_lm_to_uv", args, kwargs)


def min_pixels():
    """``ASTROVIPER_KERNELS_GPU_FFT_MIN_PIXELS``: calls whose planes have fewer
    pixels run on the CPU (default ``DEFAULT_MIN_PIXELS``; 0 runs every call
    on the GPU that it can hold)."""
    value = os.environ.get(MIN_PIXELS_ENV)
    if value is None:
        return DEFAULT_MIN_PIXELS
    try:
        pixels = int(value)
    except ValueError:
        pixels = -1
    if pixels < 0:
        raise ValueError(f"{MIN_PIXELS_ENV}={value!r}: expected a pixel count >= 0")
    return pixels


def check_environment():
    """The checks of ``python -m astroviper._kernels --check``, which touch no
    device: the pixel floor and ``CUPY_GPU_MEMORY_LIMIT`` (``ValueError``), and
    that CuPy is installed and can find NVRTC, the CUDA headers and cuFFT
    (``RuntimeError``)."""
    min_pixels()
    _runtime.check_environment()
    _runtime.check_cufft()


class _DoesNotFit(RuntimeError):
    """The call's device buffers and cuFFT plan do not fit in device memory."""


def _evict_plans():
    """Drop the kept cuFFT plans (see ``_runtime.register_cache``); returns
    whether there were any. Under the device lock."""
    held = bool(_plans)
    _plans.clear()
    return held


_runtime.register_cache("fft", _evict_plans)


def _call(entry, args, kwargs):
    floor = min_pixels()
    reference = impl("fft", entry, "ref")
    call = _bind(entry, reference, args, kwargs)
    if call is None:
        report_path("fft", DECLINED)
        return reference(*args, **kwargs)
    reason = _routing(call, floor)
    if reason is None:
        stats = {}
        try:
            result = _run(call, stats)
            report_path(
                "fft",
                GPU,
                init_s=stats["init_s"],
                device_s=stats.get("device_s"),
                versions=stats.get("versions"),
            )
            return result
        except _DoesNotFit as e:
            reason = str(e)
            _log_once(f"fft=gpu: {reason}; the reference runs such calls")
        except BaseException:
            report_path("fft", GPU, init_s=stats.get("init_s"))
            raise
    report_path("fft", ROUTED, reason=reason)
    return reference(*args, **kwargs)


def _log_once(message):
    global _warned
    if not _warned:
        _warned = True
        _logger.warning(message)


# ---------------------------------------------------------------------------------------------- binding
class _Call(NamedTuple):
    """A call the GPU entries take: its array and how to transform it."""

    entry: str
    array: np.ndarray  # the caller's input
    dtype: np.dtype  # the result's: complex64 or complex128
    overwrite: bool


@functools.lru_cache(maxsize=8)
def _signature(entry, reference):
    """The signature of ``reference``, the entry's reference function, or
    None when its parameters are not those that _bind reads."""
    signature = inspect.signature(reference)
    return signature if tuple(signature.parameters) == _PARAMETERS[entry] else None


def _backend_available(backend):
    """Whether the reference can use ``backend``, decided as it decides it:
    by its ``check_fft_backend``, which imports the backend's FFT module."""
    from astroviper.processing_functions.imaging.config_check._common import (
        check_fft_backend,
    )

    try:
        check_fft_backend(backend)
    except Exception:  # ImportError, or whatever a broken install raises
        return False
    return True


def _int(value):
    """An integer argument, or None for another type (a bool included)."""
    if type(value) is int or isinstance(value, np.integer):
        return operator.index(value)
    return None


def _bind(entry, reference, args, kwargs):
    """The call as the GPU runs it, or None for every call it does not take as
    given (see the module docstring)."""
    signature = _signature(entry, reference)
    if signature is None:
        return None
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    get = bound.arguments
    array = get[_INPUT[entry]]
    if type(array) is not np.ndarray or array.dtype not in _COMPLEX:
        return None
    if array.ndim < 2 or array.size == 0 or not array.flags.c_contiguous:
        return None
    axes, ndim = get["fft_plane_dims"], array.ndim
    if type(axes) not in (tuple, list) or len(axes) != 2:
        return None
    axes = [_int(a) for a in axes]
    if None in axes or not all(-ndim <= a < ndim for a in axes):
        return None
    if [a % ndim for a in axes] != [ndim - 2, ndim - 1]:
        return None
    threads = _int(get["processing_function_threads"])
    if threads is None or threads < 1:
        return None
    backend = get["fft_backend"]
    if type(backend) is not str or not _backend_available(backend):
        return None
    overwrite = get["overwrite_input"]
    if type(overwrite) not in (bool, np.bool_):
        return None
    overwrite = bool(overwrite)
    if overwrite and not array.flags.writeable:
        return None
    dtype = _COMPLEX[array.dtype]
    if entry == "fft_lm_to_uv":
        try:
            dtype = np.dtype(get["complex_dtype"])
        except TypeError:
            return None
        if dtype not in _TYPES:
            return None
    return _Call(entry, array, dtype, overwrite)


def _routing(call, floor):
    """Why the reference runs a call the GPU entries take, or None."""
    n0, n1 = call.array.shape[-2:]
    pixels = n0 * n1
    if pixels < floor:
        return f"planes of {pixels} pixels, below {MIN_PIXELS_ENV}={floor}"
    if min(n0, n1) < 2:
        return f"planes of {n0} x {n1} pixels: an axis of one pixel"
    if pixels > _INT_MAX:
        return f"planes of {pixels} pixels, more than a cuFFT plan takes"
    return None


# ---------------------------------------------------------------------------------------------- the GPU
def _run(call, stats=None):
    """Transform the call's planes on the GPU and return the result.

    Raises ``_DoesNotFit`` before it writes anything when the device buffers
    or the plan do not fit. ``stats``, a dict, receives the timings."""
    stats = {} if stats is None else stats
    t0 = time.perf_counter()
    device = _runtime.require_device()
    stats.update(init_s=time.perf_counter() - t0, versions=device.versions)
    with device.lock, device.cp.cuda.Device(device.id), device.stream:
        return _Transform(device, call, stats).run()


class _Transform:
    """One call on the GPU: its device buffers and plan, and its planes in order."""

    def __init__(self, device, call, stats):
        self.device, self.cp, self.call, self.stats = device, device.cp, call, stats
        self.n0, self.n1 = call.array.shape[-2:]
        self.even = self.n0 % 2 == 0 and self.n1 % 2 == 0
        self.inverse = call.entry == "ifft_uv_to_lm"

    def run(self):
        call = self.call
        try:
            t0 = time.perf_counter()
            functions = self.device.module("fft_shift", _KERNELS)
            self.stats["module_s"] = time.perf_counter() - t0
            name = _TYPES[call.dtype]
            self.checkerboard = functions[f"checkerboard<{name}>"]
            self.roll = functions[f"roll<{name}>"]
            self.buffers, plan = self._allocate()
            t0 = time.perf_counter()
            # The values to transform, and the array that receives the result:
            # the input itself, or its conversion, when the call may write it.
            source = np.asarray(call.array, dtype=call.dtype)
            out = source
            if source is call.array and not call.overwrite:
                out = np.empty_like(source)
            planes = source.reshape(-1, self.n0, self.n1)
            results = out.reshape(planes.shape)
            for p in range(planes.shape[0]):
                self._plane(planes[p], results[p], plan)
            self.stats["device_s"] = time.perf_counter() - t0
            return out
        except BaseException as e:
            _runtime.clear_frames(e)  # a kept exception must not keep the buffers
            if isinstance(e, (*self.device.errors, *self.device.fft_errors)):
                raise RuntimeError(f"fft=gpu: CUDA error: {e}") from e
            raise
        finally:
            self.__dict__.pop("buffers", None)
            self.device.pool.free_all_blocks()
            spent = self.stats.pop("module_s", 0.0) + self.stats.pop("plan_s", 0.0)
            self.stats["init_s"] += spent

    def _allocate(self):
        """The call's device buffers (one plane, two when it rolls) and its
        cuFFT plan, from the cache, or ``_DoesNotFit``."""
        cp, call, device = self.cp, self.call, self.device
        shape = (self.n0, self.n1)
        count = 1 if self.even else 2
        needed = count * self.n0 * self.n1 * call.dtype.itemsize
        for attempt in (0, 1):
            available, limit = device.available()
            if needed <= available:
                try:
                    buffers = [cp.empty(shape, call.dtype) for _ in range(count)]
                    return buffers, self._plan(buffers[0])
                except Exception as e:
                    if not self._out_of_memory(e):
                        raise
                    buffers = None
            # Drop the kept plans (their work areas), then try again.
            if attempt == 0 and device.reclaim():
                continue
            break
        raise _DoesNotFit(
            f"a call on planes of {self.n0} x {self.n1} {call.dtype} needs {needed} "
            f"bytes of device memory and a cuFFT plan, {available} bytes are "
            f"available ({_runtime.LIMIT_ENV}: {limit or 'unset'})"
        )

    def _out_of_memory(self, error):
        """Whether ``error`` is device memory running out: CuPy's pool's, or a
        cuFFT plan that cannot be allocated (``CUFFT_ALLOC_FAILED``)."""
        if isinstance(error, self.cp.cuda.memory.OutOfMemoryError):
            return True
        return (
            isinstance(error, self.device.fft_errors)
            and getattr(error, "result", None) == _CUFFT_ALLOC_FAILED
        )

    def _plan(self, buffer):
        key = (self.n0, self.n1, self.call.dtype)
        plan = _plans.pop(key, None)
        if plan is None:
            while len(_plans) >= PLAN_CACHE:  # room first: at most PLAN_CACHE plans
                _plans.popitem(last=False)
            t0 = time.perf_counter()
            plan = self.device.fft_plan(buffer)
            self.stats["plan_s"] = time.perf_counter() - t0
        _plans[key] = plan  # the most recent last
        return plan

    def _launch(self, kernel, arrays, scale, shifts=()):
        i32 = np.int32
        grid = (-(-self.n1 // _THREADS), min(self.n0, _GRID_ROWS))
        args = (*arrays, np.int64(self.n0), i32(self.n0), i32(self.n1))
        kernel(grid, (_THREADS,), (*args, *(i32(s) for s in shifts), np.float64(scale)))

    def _plane(self, plane, result, plan):
        """One plane: uploaded, shifted, transformed, shifted (and scaled) and
        downloaded into ``result``."""
        n0, n1 = self.n0, self.n1
        scale = 1.0 / (n0 * n1) if self.inverse else 1.0
        direction = _INVERSE if self.inverse else _FORWARD
        work = self.buffers[0]
        work.set(plane)
        if self.even:  # the reference's fold: cb_centred * F(cb * x)
            self._launch(self.checkerboard, (work,), 1.0, (0,))
            plan.fft(work, work, direction)
            self._launch(self.checkerboard, (work,), scale, ((n0 // 2 + n1 // 2) % 2,))
        else:  # ifftshift, the transform, then fftshift
            rolled = self.buffers[1]
            self._launch(self.roll, (work, rolled), 1.0, (n0 // 2, n1 // 2))
            plan.fft(rolled, rolled, direction)
            self._launch(self.roll, (rolled, work), scale, (n0 - n0 // 2, n1 - n1 // 2))
        work.get(out=result)
