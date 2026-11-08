"""
GPU Hogbom minor cycle, selected by ``ASTROVIPER_KERNELS=hogbom=gpu``.

``clean_cube`` and ``clean_cube_many_threads`` take the arguments of the
reference entries (``deconvolvers/hogbom``) and give their results bit for
bit: the residual and model cubes (in place), the iterations, stop codes,
flags, ``final_peak``, ``total_flux_cleaned`` and the iteration history. NaNs
too: an operation returns its NaN operand, quieted, or makes the default NaN,
as the reference's SSE2 code does. Only an operation on two NaNs keeps the
one that the reference compiler's operand order picks, so the sign and
payload of such a NaN are not part of the contract (as for ``hogbom=fast``).

On the GPU (``cuda/hogbom_exact.cu``) the planes run one after another: each
is uploaded from the caller's arrays, cleaned by an exact CuPy raw kernel and
its residual and model are written back in place (only planes that ran an
iteration), so the call holds one plane on the device and copies no cube on
the host. The kernel scans and sums as the reference does (see its source)
and runs the iterations in cooperative launches of about ``TARGET_MS`` each
(fewer iterations at first), so that no launch keeps the GPU for long. Its
grid barrier has a watchdog: a wait of more than ``SPIN_LIMIT`` spins aborts
the launch. ``processing_function_threads`` does not change the GPU's
results; it keeps its meaning for the calls that run on the CPU. Every call
ends by freeing the cached blocks of CuPy's default memory pool (any other
CuPy code's too), so that the device memory goes back between calls.

Each call takes one path, reported to the kernel switch for its trace line:

``"gpu"``
    The GPU ran the call. Its trace line also has ``init_s``, the seconds the
    call spent on the first use of CUDA (the context, CuPy's import, the
    kernels' compile or load; about 0 for a later call), ``device_s``, the
    seconds of its work on the device (allocation, uploads, launches and
    downloads), and ``versions``, those of the CUDA driver, runtime and NVRTC
    (whose version keys CuPy's cache of the compiled kernels).
``"routed"``
    The fast CPU kernel (``hogbom=fast``, bit for bit the same) ran it: the
    calls whose planes with a budget search fewer than
    ``ASTROVIPER_KERNELS_GPU_MIN_PIXELS`` pixels each (the pixels of the clean
    box inside the mask; default ``DEFAULT_MIN_PIXELS``), the calls whose
    planes the GPU cannot clean in short launches (see ``_iteration_ms``),
    whatever the floor, and every call the GPU entries do not take as given:
    an argument of another type than the production callers pass (an array
    subclass, a scalar that is not a Python or NumPy number, a clean box that
    is not a tuple), and the calls ``hogbom=fast`` declines.
``"declined"``
    The reference ran it, because ``hogbom=fast`` declined it (every call that
    the reference binding would not run exactly as given, so that its
    conversions and errors stay the reference's: see
    ``astroviper._kernels.fast.hogbom``), or because the fast module is not
    built (logged once).

The floor counts searched pixels because the fast kernel's iterations cost
what a plane searches, the GPU's what it holds (every iteration subtracts the
PSF over its footprint and scans the rows of the search box). In the WP9
measurements the GPU won only on planes that search about DEFAULT_MIN_PIXELS
pixels or more. The kernels plugin (``tests/kernels_plugin.py``) sets the
floor to 0 for a ``hogbom=gpu`` session, so that the tests' planes run on the
GPU.

Errors are not retried on the CPU. ``RuntimeError``, raised before the call
writes anything: CuPy or a CUDA device is missing (at the first call that
needs them), the process was forked after its parent had opened the device,
or the call needs more device memory than is free or than
``CUPY_GPU_MEMORY_LIMIT`` allows (see ``_runtime``; an invalid value of it
raises ``ValueError``). ``RuntimeError`` too when a launch fails or the
watchdog aborts it: the planes before that plane are cleaned and written,
that plane and the later ones are as they were, and the device stays usable
for the next call (the kernel ends normally). Whatever a call raises once its
kernels are loaded carries ``planes_written``, the number of planes it cleaned
and wrote back, and a note that says so; the frames of its traceback are
cleared of their variables, so that a kept exception does not keep the call's
device memory.
"""

import ctypes
import ctypes.util
import functools
import logging
import operator
import os
import time
import traceback
from typing import NamedTuple

import numpy as np

from astroviper._kernels._switch import impl, report_path
from astroviper._kernels.gpu import _runtime

__all__ = [
    "DECLINED",
    "DEFAULT_MIN_PIXELS",
    "GPU",
    "MIN_PIXELS_ENV",
    "ROUTED",
    "check_environment",
    "clean_cube",
    "clean_cube_many_threads",
]

#: The paths of a call (see the module docstring).
GPU, ROUTED, DECLINED = "gpu", "routed", "declined"
#: Calls whose planes search fewer pixels run on the CPU.
MIN_PIXELS_ENV = "ASTROVIPER_KERNELS_GPU_MIN_PIXELS"
DEFAULT_MIN_PIXELS = 1536 * 1536
#: The grid barrier's watchdog: a wait of more spins aborts the launch (about
#: 0.24 us a spin, so a barrier that cannot complete ends the launch in about
#: 0.25 s; a wait lasts one phase of an iteration, under 1 ms on a 4000 x 4000
#: plane). A kernel that the GPU switches out does not spin.
SPIN_LIMIT = 1 << 20
#: The time a launch aims at, in ms (it grows to that from a few iterations).
TARGET_MS = 25.0

_logger = logging.getLogger("astroviper.kernels")
#: The two sequential chains of an iteration, in ns a pixel (measured on an RTX 3090 and rounded up): a lane scans each
#: row of a strip column by column (for each strip of 32 rows that a block takes), then one lane adds up the rows.
_NS_PER_COLUMN, _NS_PER_ROW = 120.0, 45.0
_FIRST_CHUNK = 4
_CHUNK_MAX = 4096  # iterations per launch (the history and threshold buffers)
_THREADS = 256  # threads per block of every kernel
_STRIP_TILE = 64  # columns per shared-memory tile of a strip
_SUM_TILE = 4096  # pixels per block of the summary (SUM_TILE of the source)
_GATHER_MAX = 1 << 20  # |model| values gathered per round of the summary
_INT_MAX = 2**31 - 1
_INT, _BOOL = np.dtype(np.intc), np.dtype(np.bool_)
_FLOATS = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}
_STOP_NONE, _STOP_DIVERGED = 0, 4
_ENTRIES = {"clean_cube": 0, "clean_cube_many_threads": 1}  # ENTRY of the source
#: The control words ``ctl``: the plane's state st_i in ``[0, _BARRIER)`` (S_STOP, S_COMPONENTS and S_REASON of the
#: source's enum) and the grid barrier's count, generation and abort flag from ``_BARRIER`` on.
_S_STOP, _S_COMPONENTS, _S_REASON = 3, 4, 5
_BARRIER = 16
_ABORT = _BARRIER + 2

#: The reference's parameters, in order, with their defaults (the first three
#: have none; the per-plane arrays' defaults are rejected by the reference).
_PARAMETERS = {
    "residual_cube": None,
    "psf_cube": None,
    "model_cube": None,
    "mask_cube": None,
    "clean_box": (-1, -1, -1, -1),
    "max_iter_per_cycle_updated": None,
    "gain": 0.1,
    "threshold_per_cycle": None,
    "speedup": 0.0,
    "processing_function_threads": 1,
    "max_iter_divergence": -1,
    "divergence_rms_factor": 1.0,
    "divergence_peak_factor": 1.0,
    "record_iteration_history": False,
}
_NAMES = tuple(_PARAMETERS)
_HISTORY = (
    "iteration_peak_residual",
    "iteration_residual_rms",
    "iteration_component_l",
    "iteration_component_m",
    "iteration_component_flux",
    "iteration_component_scale",
)
_CHUNK_KERNELS = tuple(
    f"hogbom_exact_chunk<{t}, {_STRIP_TILE}, {e}>"
    for t in _FLOATS.values()
    for e in _ENTRIES.values()
)
_SUMMARY_KERNELS = tuple(
    f"{k}<{t}>" for t in _FLOATS.values() for k in ("plane_summary", "gather_abs")
)


def clean_cube(*args, **kwargs):
    """``clean_cube`` of the reference, with the same arguments and results.

    Cleans the cube in place on the GPU, or on the CPU (see the module
    docstring), which also raises the reference's exceptions.
    """
    return _call("clean_cube", args, kwargs)


def clean_cube_many_threads(*args, **kwargs):
    """``clean_cube_many_threads`` of the reference, with the same arguments
    and results; see :func:`clean_cube`."""
    return _call("clean_cube_many_threads", args, kwargs)


def min_pixels():
    """``ASTROVIPER_KERNELS_GPU_MIN_PIXELS``: calls whose planes search fewer
    pixels run on the CPU (default ``DEFAULT_MIN_PIXELS``; 0 runs every call
    on the GPU that it can clean in short launches)."""
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
    device: the pixel floor and ``CUPY_GPU_MEMORY_LIMIT`` (``ValueError``) and
    that CuPy is installed (``RuntimeError``). ``python -m
    astroviper._kernels.gpu.prewarm`` also checks the device, NVRTC and the
    kernels."""
    min_pixels()
    _runtime.check_environment()


def _call(entry, args, kwargs):
    floor = min_pixels()
    call = _bind(entry, args, kwargs)
    if call is None or not _on_gpu(call, floor):
        return _on_cpu(entry, args, kwargs)
    stats = {}
    try:
        return _run(call, stats)
    finally:
        report_path(
            "hogbom",
            GPU,
            init_s=stats.get("open_s", 0.0) + stats.get("module_s", 0.0),
            device_s=stats.get("device_s"),
            versions=stats.get("versions"),
        )


def _iteration_ms(ny, nx, strips_per_block=1):
    """An iteration's time on the GPU, in ms: its two sequential chains (a
    bound for planes whose mask or clean box leaves fewer pixels to scan)."""
    return (strips_per_block * nx * _NS_PER_COLUMN + ny * _NS_PER_ROW) * 1e-6


def _searched(call, enough):
    """The most pixels a plane of the call with a budget searches (those of
    the clean box inside its mask), counted until one has ``enough``."""
    xbeg, xend, ybeg, yend = call.box
    most = 0
    for plane, index in enumerate(np.ndindex(call.residual.shape[:3])):
        if call.max_iter[plane] <= 0:
            continue
        if call.mask is None:
            return (xend - xbeg) * (yend - ybeg)
        window = call.mask[index][ybeg:yend, xbeg:xend].view(np.uint8)
        most = max(most, int(np.count_nonzero(window)))
        if most >= enough:
            break
    return most


def _on_gpu(call, floor):
    """Whether the GPU runs the call: an iteration of its planes takes at most
    TARGET_MS (``_iteration_ms``), and a plane with a budget searches
    ``floor`` pixels or more."""
    if _iteration_ms(call.ny, call.nx) > TARGET_MS:
        return False
    return floor == 0 or (
        call.ny * call.nx >= floor and _searched(call, floor) >= floor
    )


_warned = False  # whether _on_cpu logged that the fast module is missing


def _on_cpu(entry, args, kwargs):
    """The call run by ``hogbom=fast`` (which runs the reference for the calls
    it declines), or by the reference when the fast module is not built."""
    global _warned
    try:
        function = impl("hogbom", entry, "fast")
    except ImportError as e:
        if not _warned:
            _warned = True
            _logger.warning(
                "hogbom=gpu: %s; the reference runs the calls the GPU does not", e
            )
        report_path("hogbom", DECLINED)
        return impl("hogbom", entry, "ref")(*args, **kwargs)
    report_path("hogbom", ROUTED)
    return function(*args, **kwargs)


# ---------------------------------------------------------------------------------------------- binding
class _Call(NamedTuple):
    """A call's arguments, as the reference binding converts them."""

    entry: str
    dtype: np.dtype  # the image dtype (canonical: float32 or float64)
    residual: np.ndarray
    psf: np.ndarray
    model: np.ndarray
    mask: np.ndarray | None
    #: the clean box, clamped as the binding clamps it: xbeg, xend, ybeg, yend
    box: tuple
    max_iter: np.ndarray  # per plane, flat
    threshold: np.ndarray  # per plane, flat, of the image dtype
    gain: np.floating
    speedup: np.floating
    max_iter_divergence: int
    rms_factor: float
    peak_factor: np.floating
    record: bool

    @property
    def ny(self):
        return self.residual.shape[3]

    @property
    def nx(self):
        return self.residual.shape[4]


def _int(value):
    """pybind11's conversion of ``value`` to a C int, or None where the GPU
    entries do not take ``value`` (another type, or out of range)."""
    if type(value) in (int, bool) or isinstance(value, np.integer):
        value = operator.index(value)
        return value if -_INT_MAX - 1 <= value <= _INT_MAX else None
    return None


def _float(value):
    """pybind11's conversion of ``value`` to a C double, or None."""
    if type(value) in (int, bool, float) or isinstance(
        value, np.integer | np.float16 | np.float32 | np.float64
    ):
        try:
            return float(value)
        except OverflowError:  # an int beyond the doubles: the reference raises
            return None
    return None


def _bool(value):
    """pybind11's conversion of ``value`` to a C++ bool, or None."""
    if value is None:
        return False
    if type(value) in (int, bool, float) or isinstance(
        value, np.bool_ | np.integer | np.float32 | np.float64
    ):
        return bool(value)
    return None


def _cube(array, dtype, shape, writeable):
    """The reference binding's validate_inplace_cube, as a predicate."""
    return (
        array.dtype == dtype
        and array.flags.c_contiguous
        and (array.flags.writeable or not writeable)
        and array.shape == shape
    )


def _bind(entry, args, kwargs):
    """The call as the GPU runs it, or None: for every call that the reference
    binding would not run exactly as given, and for the argument forms the
    GPU entries do not convert themselves (see the module docstring)."""
    if len(args) > len(_NAMES):
        return None
    bound = dict(zip(_NAMES, args, strict=False))
    for name, value in kwargs.items():
        if name not in _PARAMETERS or name in bound:
            return None
        bound[name] = value
    if any(name not in bound for name in _NAMES[:3]):
        return None
    arrays = [bound.get(name) for name in ("residual_cube", "model_cube", "psf_cube")]
    arrays += [
        bound.get(n) for n in ("max_iter_per_cycle_updated", "threshold_per_cycle")
    ]
    if any(type(a) is not np.ndarray for a in arrays):
        return None
    residual, model, psf, max_iter, threshold = arrays
    dtype = next((d for d in _FLOATS if residual.dtype == d), None)
    if dtype is None or residual.ndim != 5 or max(residual.shape) > _INT_MAX:
        return None
    shape = residual.shape
    planes, ny, nx = shape[:3], shape[3], shape[4]
    nplanes = planes[0] * planes[1] * planes[2]
    if nplanes == 0 or nplanes > _INT_MAX or ny == 0 or nx == 0:
        return None  # nothing to clean, or planes without pixels (fast declines those)
    if not (_cube(residual, dtype, shape, True) and _cube(model, dtype, shape, True)):
        return None
    if psf.ndim != 5 or psf.dtype != dtype or max(psf.shape) > _INT_MAX:
        return None
    if (
        psf.shape[:2] != planes[:2]
        or psf.shape[2] not in (planes[2], 1)
        or psf.shape[3:] not in ((ny, nx), (2 * ny, 2 * nx))
        or not psf.flags.c_contiguous
    ):
        return None
    mask = None
    if "mask_cube" in bound:  # an empty array (the default) is no mask
        mask = bound["mask_cube"]
        if type(mask) is not np.ndarray:
            return None
        if mask.size == 0:
            mask = None
        elif not _cube(mask, _BOOL, shape, False):
            return None
        else:
            arrays.append(mask)
    for name, dtype_of in (
        ("max_iter_per_cycle_updated", _INT),
        ("threshold_per_cycle", dtype),
    ):
        array = bound[name]
        if not (
            array.dtype == dtype_of
            and array.flags.c_contiguous
            and array.shape == planes
        ):
            return None
    # Arrays the call writes may share no memory with another argument.
    for i in (0, 1):
        if any(
            j != i and np.may_share_memory(arrays[i], a) for j, a in enumerate(arrays)
        ):
            return None
    box = bound.get("clean_box", _PARAMETERS["clean_box"])
    if type(box) is not tuple:
        return None
    xbeg, xend, ybeg, yend = 0, nx, 0, ny
    if len(box) == 4:  # as the binding parses it (-1: the full extent)
        values = [_int(v) for v in box]
        if None in values:
            return None
        xbeg, xend, ybeg, yend = values
        xbeg, xend = (0 if xbeg == -1 else xbeg), (nx if xend == -1 else xend)
        ybeg, yend = (0 if ybeg == -1 else ybeg), (ny if yend == -1 else yend)
        xbeg = max(0, min(xbeg, nx - 1))
        xend = max(xbeg + 1, min(xend, nx))
        ybeg = max(0, min(ybeg, ny - 1))
        yend = max(ybeg + 1, min(yend, ny))
    get = {name: bound.get(name, default) for name, default in _PARAMETERS.items()}
    gain, speedup = _float(get["gain"]), _float(get["speedup"])
    rms_factor, peak_factor = (
        _float(get["divergence_rms_factor"]),
        _float(get["divergence_peak_factor"]),
    )
    threads, max_iter_divergence = (
        _int(get["processing_function_threads"]),
        _int(get["max_iter_divergence"]),
    )
    record = _bool(get["record_iteration_history"])
    scalars = (
        gain,
        speedup,
        rms_factor,
        peak_factor,
        threads,
        max_iter_divergence,
        record,
    )
    if any(v is None for v in scalars):
        return None
    t = dtype.type
    with np.errstate(all="ignore"):  # the binding's casts of double to float
        gain_t, speedup_t, peak_t = t(gain), t(speedup), t(peak_factor)
    return _Call(
        entry,
        dtype,
        residual,
        psf,
        model,
        mask,
        (xbeg, xend, ybeg, yend),
        max_iter.ravel().copy(),
        threshold.ravel().copy(),
        gain_t,
        speedup_t,
        max_iter_divergence,
        rms_factor,
        peak_t,
        record,
    )


# ---------------------------------------------------------------------------------------------- the GPU
def _run(call, stats=None):
    """Clean the call's cube on the GPU and return the reference's results.

    ``stats``, a dict, receives timings and every launch (for the trace, tests
    and benchmarks)."""
    stats = {} if stats is None else stats
    t0 = time.perf_counter()
    try:
        device = _runtime.require_device()
    finally:
        stats["open_s"] = time.perf_counter() - t0
    with device.lock, device.cp.cuda.Device(device.id), device.stream:
        return _Cleaner(device, call, stats).run()


def _clear_frames(error):
    """Clear the variables of the frames of ``error``'s traceback and of the
    exceptions chained to it: they hold the call's device buffers, which its
    memory pool could not free while a caller keeps the exception. The
    tracebacks still print."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        traceback.clear_frames(error.__traceback__)
        error = error.__cause__ or error.__context__


def _launch(kernel, grid, args):
    """One launch of ``kernel`` on the current stream (a module function, so
    that tests can make a launch fail)."""
    kernel((grid,), (_THREADS,), args)


def _search_box(mask, xbeg, xend, ybeg, yend):
    """The reference's mask_bounding_box (inclusive ``i0, i1, j0, j1``) of a
    plane and the pixels its search covers: the clean box when the mask has
    none in it. A mask byte that is not 0 counts (see ``hogbom=fast``)."""
    if mask is None:
        return xbeg, xend - 1, ybeg, yend - 1, (xend - xbeg) * (yend - ybeg)
    window = mask[ybeg:yend, xbeg:xend].view(np.uint8)
    rows = np.flatnonzero(window.any(axis=1))
    if rows.size == 0:
        return xbeg, xend - 1, ybeg, yend - 1, 0
    cols = np.flatnonzero(window.any(axis=0))
    i0, i1 = xbeg + int(cols[0]), xbeg + int(cols[-1])
    return (
        i0,
        i1,
        ybeg + int(rows[0]),
        ybeg + int(rows[-1]),
        int(np.count_nonzero(window)),
    )


@functools.cache
def _libm_pow(dtype):
    """libm's pow (or powf), which the reference's std::pow calls."""
    libm = ctypes.CDLL(ctypes.util.find_library("m"))
    c = ctypes.c_float if dtype == np.float32 else ctypes.c_double
    function = libm.powf if dtype == np.float32 else libm.pow
    function.restype, function.argtypes = c, (c, c)
    return function


def _threshold_factors(dtype, speedup, begin, end):
    """2^(it / speedup) for the iterations it in [begin, end), as the reference
    computes it: std::pow(T(2), T(it) / speedup) in the image dtype."""
    t = dtype.type
    with np.errstate(all="ignore"):
        exponents = np.arange(begin, end).astype(dtype) / t(speedup)
    power = _libm_pow(dtype)
    return np.array([power(2.0, float(x)) for x in exponents], dtype)


class _Cleaner:
    """One call on the GPU: its device buffers, and its planes in order."""

    def __init__(self, device, call, stats):
        self.device, self.cp, self.call = device, device.cp, call
        self.stats = {} if stats is None else stats
        self.stats.update(
            init_s=device.init_s, versions=device.versions, launches=[], planes=[]
        )
        self.dtype = call.dtype
        self.t = self.dtype.type
        self.written = 0  # planes cleaned and written back (in C order)

    def run(self):
        cp, call = self.cp, self.call
        t0 = time.perf_counter()
        functions = self.device.module("hogbom_exact", _CHUNK_KERNELS, cooperative=True)
        name = _FLOATS[self.dtype]
        self.chunk = functions[
            f"hogbom_exact_chunk<{name}, {_STRIP_TILE}, {_ENTRIES[call.entry]}>"
        ]
        functions = self.device.module("hogbom_exact", _SUMMARY_KERNELS)
        self.summary = functions[f"plane_summary<{name}>"]
        self.gather = functions[f"gather_abs<{name}>"]
        self.stats["module_s"] = time.perf_counter() - t0
        try:
            self._allocate()
            return self._planes()
        except BaseException as e:
            _clear_frames(e)  # a kept exception must not keep the buffers
            error = e
            if isinstance(e, cp.cuda.memory.OutOfMemoryError):
                error = RuntimeError(f"hogbom=gpu: out of device memory: {e}")
            elif isinstance(e, self.device.errors):
                error = RuntimeError(f"hogbom=gpu: CUDA error: {e}")
            error.planes_written = self.written
            error.add_note(
                f"hogbom=gpu: the call cleaned and wrote back {self.written} of its "
                f"{call.max_iter.size} planes (in C order); the others are unchanged"
            )
            if error is e:
                raise
            raise error from e
        finally:
            self.__dict__.pop("buffers", None)
            self.device.pool.free_all_blocks()
            self.stats["call_s"] = time.perf_counter() - t0
            self.stats["device_s"] = self.stats["call_s"] - self.stats["module_s"]

    def _allocate(self):
        """Every device buffer of the call, after checking that they fit."""
        cp, call = self.cp, self.call
        ny, nx = call.ny, call.nx
        pny, pnx = call.psf.shape[3:]
        n, item = ny * nx, self.dtype.itemsize
        tiles = -(-n // _SUM_TILE)
        history = _CHUNK_MAX if call.record else 1
        sizes = {  # name: (shape, dtype)
            "residual": ((ny, nx), self.dtype),
            "model": ((ny, nx), self.dtype),
            "psf": ((pny, pnx), self.dtype),
            "mask": ((ny, nx) if call.mask is not None else 1, _BOOL),
            "row_key": (ny, np.uint64),
            "row_ix": (ny, np.int32),
            "row_ss": (ny, np.float64),
            "ctl": (24, np.int32),  # st_i [0, 16) and the barrier [16, 20)
            "st_d": (8, np.float64),
            "h_peak": (history, np.float64),
            "h_ss": (history, np.float64),
            "h_l": (history, np.int32),
            "h_m": (history, np.int32),
            "h_flux": (history, np.float64),
            "factor": (_CHUNK_MAX if call.speedup > 0 else 1, self.dtype),
            "part_min": (tiles, self.dtype),
            "part_max": (tiles, self.dtype),
            "part_nz": (tiles, np.int32),
            "offsets": (tiles, np.int64),
            "values": (max(_SUM_TILE, min(n, _GATHER_MAX)), self.dtype),
        }
        needed = sum(
            -(-int(np.prod(s)) * np.dtype(d).itemsize // 512) * 512
            for s, d in sizes.values()
        )
        pool = self.device.pool
        free, _ = cp.cuda.runtime.memGetInfo()
        available = free + pool.free_bytes()
        limit = pool.get_limit()
        if limit:
            available = min(available, limit - pool.used_bytes())
        self.stats["device_bytes"] = needed
        if needed > available:
            raise RuntimeError(
                f"hogbom=gpu: the call needs {needed} bytes of device memory, "
                f"{available} are available ({_runtime.LIMIT_ENV}: {limit or 'unset'}; "
                f"{item * n} per image plane)"
            )
        self.buffers = {k: cp.empty(s, d) for k, (s, d) in sizes.items()}

    def _planes(self):
        call, b = self.call, self.buffers
        nt, nf, np_img = call.residual.shape[:3]
        np_psf = call.psf.shape[2]
        shape = (nt, nf, np_img)
        iterations = np.zeros(shape, _INT)
        stop_code = np.zeros(shape, _INT)
        final_peak = np.empty(shape, self.dtype)
        total_flux = np.empty(shape, self.dtype)
        history = [None] * (nt * nf * np_img)
        last_psf = None
        for plane, (t, f, p) in enumerate(np.ndindex(shape)):
            t0 = time.perf_counter()
            residual, model = call.residual[t, f, p], call.model[t, f, p]
            mask = None if call.mask is None else call.mask[t, f, p]
            b["residual"].set(residual)
            b["model"].set(model)
            psf = (t, f, 0 if np_psf == 1 else p)
            if psf != last_psf:
                b["psf"].set(call.psf[psf])
                last_psf = psf
            if mask is not None:
                b["mask"].set(mask)
            box = _search_box(mask, *call.box)
            t1 = time.perf_counter()
            done, reason, record = self._clean(
                plane, int(call.max_iter[plane]), call.threshold[plane], box
            )
            t2 = time.perf_counter()
            final_peak[t, f, p], total_flux[t, f, p] = self._summary(mask is not None)
            if call.max_iter[plane] > 0:  # else the plane is unchanged
                b["residual"].get(out=residual)
                b["model"].get(out=model)
            self.written = plane + 1
            iterations[t, f, p], stop_code[t, f, p], history[plane] = (
                done,
                reason,
                record,
            )
            self.stats["planes"].append(
                {
                    "upload_s": t1 - t0,
                    "clean_s": t2 - t1,
                    "summary_download_s": time.perf_counter() - t2,
                }
            )
        results = {
            "iterations_performed": iterations,
            "final_peak": final_peak,
            "total_flux_cleaned": total_flux,
            "converged": final_peak <= call.threshold.reshape(shape),
            "stop_code": stop_code,
            "diverged": stop_code == _STOP_DIVERGED,
        }
        if call.record:
            for k, name in enumerate(_HISTORY):
                records = iter(h[k] for h in history)
                results[name] = [
                    [[next(records) for _ in range(np_img)] for _ in range(nf)]
                    for _ in range(nt)
                ]
        return results

    def _clean(self, plane, budget, threshold, box):
        """Clean one plane on the device: (iterations, stop code, history)."""
        cp, call, b, t = self.cp, self.call, self.buffers, self.t
        i0, i1, j0, j1, searched = box
        if budget <= 0:  # the reference runs no iteration
            done = min(budget, 0) if call.entry == "clean_cube" else 0
            return done, _STOP_NONE, tuple(np.zeros(0) for _ in _HISTORY)
        stream = cp.cuda.get_current_stream()
        for name in ("ctl", "st_d"):
            cp.cuda.runtime.memsetAsync(b[name].data.ptr, 0, b[name].nbytes, stream.ptr)
        ny, nx = call.ny, call.nx
        pny, pnx = call.psf.shape[3:]
        use_factor = bool(call.speedup > 0)
        i32 = np.int32
        head = (
            b["residual"], b["psf"], b["mask"], b["model"], i32(nx), i32(ny), i32(pnx), i32(pny),
            i32(i0), i32(i1), i32(j0), i32(j1), i32(call.mask is not None), i32(call.box[0]), i32(call.box[2]),
            i32(budget), call.gain, t(threshold), b["factor"], i32(use_factor),
            i32(call.max_iter_divergence), np.float64(call.rms_factor), call.peak_factor,
            b["row_key"], b["row_ix"], b["row_ss"], b["ctl"][:_BARRIER], b["st_d"], i32(call.record),
            b["h_peak"], b["h_ss"], b["h_l"], b["h_m"], b["h_flux"],
            b["ctl"][_BARRIER : _BARRIER + 4].view(np.uint32), np.uint32(SPIN_LIMIT),
        )  # fmt: skip
        strips = (ny + 31) // 32
        grid = min(strips, self.device.max_blocks(self.chunk, _THREADS))
        # a first launch of a few iterations, fewer on planes of long iterations
        first = TARGET_MS / _iteration_ms(ny, nx, -(-strips // grid))
        start, end = cp.cuda.Event(), cp.cuda.Event(block=True)
        parts = [[] for _ in range(5)]  # peak, l, m, flux, sum of squares
        it, chunk, components = 0, max(1, min(_FIRST_CHUNK, int(first))), 0
        while True:
            it_end = min(it + chunk, budget + 1, it + _CHUNK_MAX)
            if use_factor and it < budget:
                factors = _threshold_factors(
                    self.dtype, call.speedup, it, min(it_end, budget)
                )
                b["factor"][: factors.size].set(factors)
            start.record(stream)
            _launch(self.chunk, grid, head + (i32(it), i32(it_end)))
            end.record(stream)
            end.synchronize()
            ms = cp.cuda.get_elapsed_time(start, end)
            ctl = b["ctl"].get()
            if ctl[_ABORT]:
                raise RuntimeError(
                    f"hogbom=gpu: the kernel's grid barrier waited more than {SPIN_LIMIT} spins and "
                    f"aborted plane {plane} (iterations {it}..{it_end - 1}); the planes before it are "
                    "cleaned, it and the later planes are unchanged"
                )
            stopped, done = bool(ctl[_S_STOP]), int(ctl[_S_COMPONENTS])
            searches = (done + 1 if stopped else it_end) - it
            self.stats["launches"].append((plane, it, searches, ms))
            if call.record:
                n = done - components
                names = ("h_peak", "h_l", "h_m", "h_flux")
                for part, name in zip(parts[:4], names, strict=True):
                    part.append(b[name][:n].get())
                parts[4].append(b["h_ss"][int(it == 0) : searches].get())
            components = done
            if stopped:
                break
            it = it_end
            chunk = int(min(2 * chunk, max(1.0, TARGET_MS * searches / max(ms, 1e-3))))
        if not call.record:
            return done, int(ctl[_S_REASON]), None
        peak, l, m, flux, ss = (np.concatenate(p) for p in parts)
        with np.errstate(all="ignore"):
            rms = np.sqrt(ss / float(searched))
        record = (
            peak,
            rms,
            l.astype(np.float64),
            m.astype(np.float64),
            flux,
            np.zeros(done),
        )
        return done, int(ctl[_S_REASON]), record

    def _summary(self, has_mask):
        """The binding's final_peak and total_flux_cleaned of the plane on the
        device: max(|min|, |max|) of the residual inside the mask (maximg), and
        the sum of |model| in plane order, in the image dtype (+-0 values,
        which add nothing, are skipped)."""
        call, b, t = self.call, self.buffers, self.t
        n = call.ny * call.nx
        tiles = -(-n // _SUM_TILE)
        args = (
            b["residual"],
            b["mask"],
            np.int32(has_mask),
            b["model"],
            np.int64(n),
            t(1e20),
            t(-1e20),
        )
        _launch(
            self.summary, tiles, args + (b["part_min"], b["part_max"], b["part_nz"])
        )
        low, high = np.abs(b["part_min"].get().min()), np.abs(b["part_max"].get().max())
        final_peak = high if low < high else low
        counts = b["part_nz"].get().astype(np.int64)
        total = t(0)
        first, capacity = 0, b["values"].size
        while first < tiles:  # the tiles whose values fit the buffer, in order
            last = first + max(
                1, int(np.searchsorted(np.cumsum(counts[first:]), capacity, "right"))
            )
            sizes = counts[first:last]
            count = int(sizes.sum())
            if count:
                b["offsets"][: last - first].set(np.cumsum(sizes) - sizes)
                args = (
                    b["model"],
                    np.int64(n),
                    np.int32(first),
                    b["part_nz"],
                    b["offsets"],
                    b["values"],
                )
                _launch(self.gather, last - first, args)
                values = b["values"][:count].get()
                values[0] += total  # total + |v0|, then the rest in order
                total = np.cumsum(values, dtype=self.dtype)[-1]
            first = last
        return final_peak, total
