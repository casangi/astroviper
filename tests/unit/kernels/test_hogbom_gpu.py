"""Tests of the GPU Hogbom minor cycle (``ASTROVIPER_KERNELS=hogbom=gpu``).

The implementation is called directly (``astroviper._kernels.impl``), so these
tests run in any session; those that need the GPU skip without CuPy or a CUDA
device. Planes of at most 1024 x 1024 pixels (the larger ones are checked on
the DATA side). The oracle is the reference binding, compared as in
``test_hogbom_fast`` (whose cases and comparison are reused): bitwise on
unsigned-integer views of everything a call returns or changes, every value
that is not a NaN with the reference's bits and every NaN a NaN where the
reference has one (the contract leaves its sign and payload unspecified: the
GPU's operations give the canonical NaN).

* Both entries against their own reference entries on the GPU (the pixel
  floor set to 0): WP4's random cubes and edge cases (masks, NaN and
  infinities, ties, -0.0, subnormal and huge values, float32 and float64, PSFs
  of one and of two image sizes, clamped clean boxes, edge gains, thresholds,
  speedups, budgets and divergence controls, accumulated models), and planes of
  up to 1024 x 1024.
* Multi-plane cubes: per-plane budgets, thresholds and masks, a broadcast PSF,
  Stokes I active with V idle; the bits do not depend on the thread count.
* Paths: planes below the floor run ``hogbom=fast`` (``routed``), arguments of
  forms the GPU entries do not take run on the CPU, the calls that the
  reference rejects or that share memory are the reference's (``declined``),
  misaligned arrays run on the GPU; the trace names each call's path.
* Failures: a forced watchdog abort and a failing launch raise RuntimeError,
  leave the plane they hit and the later ones unchanged, and the next call is
  right; ``CUPY_GPU_MEMORY_LIMIT`` is validated and a call that does not fit
  raises RuntimeError having written nothing; without a device the first GPU
  call raises RuntimeError and routed calls still run.
* Resources: launches stay short, and host memory within the inputs + 10 %.
"""

import json
import os
import socket

import numpy as np
import pytest

from astroviper._kernels import impl
from astroviper._kernels._switch import ENV, TRACE_ENV, dispatch
from astroviper._kernels.gpu import _runtime, hogbom
from tests.unit.kernels.test_hogbom_fast import (
    DECLINED,
    EDGE_CASES,
    MASKS,
    NAN_PAYLOADS,
    SCALARS,
    SPLIT_EDGE_CASES,
    _arguments,
    _assert_same,
    _bits,
    _case,
    _mask,
    _misaligned,
    _outcome,
    _random_case,
    _run,
)

ENTRIES = ("clean_cube", "clean_cube_many_threads")
GPU = {entry: impl("hogbom", entry, "gpu") for entry in ENTRIES}
REFERENCE = {entry: impl("hogbom", entry, "ref") for entry in ENTRIES}


@pytest.fixture(scope="module")
def device():
    try:
        return _runtime.require_device()
    except RuntimeError as e:
        pytest.skip(f"no GPU: {e}")


@pytest.fixture
def gpu(device, monkeypatch):
    """Every plane on the GPU; the list of the entries of the calls it ran."""
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, "0")
    runs, run = [], hogbom._run

    def counted(call, stats=None):
        runs.append(call.entry)
        return run(call, stats)

    monkeypatch.setattr(hogbom, "_run", counted)
    return runs


def _same_outcome(a, b):
    """The same exception, array bytes and results (keys, dtypes, bits)."""
    assert a[1:] == b[1:]
    if a[1] is None:
        assert list(a[0]) == list(b[0])
        for key in SCALARS:
            x, y = np.asarray(a[0][key]), np.asarray(b[0][key])
            assert x.dtype == y.dtype and np.array_equal(_bits(x), _bits(y)), key


def _check(gpu, entry, case, threads=(1, 7)):
    """The GPU's results at each thread count are the reference's."""
    expected = _run(REFERENCE[entry], case)
    for count in threads:
        runs = len(gpu)
        actual = _run(GPU[entry], case, count)
        assert len(gpu) == runs + 1, "the call did not run on the GPU"
        _assert_same(expected, actual, nan_bits=False)
    return expected


# ---------------------------------------------------------------- the reference's bits
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("seed", range(40))
def test_random_cubes_are_the_reference(gpu, entry, seed):
    _check(gpu, entry, _random_case(seed)[0])


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("seed", range(20))
def test_random_large_cubes_are_the_reference(gpu, entry, seed):
    _check(gpu, entry, _random_case(1000 + seed, large=True)[0])


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(EDGE_CASES))
def test_edge_cases_are_the_reference(gpu, entry, name):
    _check(gpu, entry, EDGE_CASES[name], threads=(1,))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(SPLIT_EDGE_CASES))
def test_split_edge_cases_are_the_reference(gpu, entry, name):
    """The edge cases on planes of 96 x 104 pixels: three strips of rows."""
    _check(gpu, entry, SPLIT_EDGE_CASES[name], threads=(1,))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "dtype, mask, twice",
    [
        (np.float64, "disk", False),
        (np.float32, None, False),
        (np.float64, "boxes", True),
    ],
)
def test_large_planes_are_the_reference(gpu, entry, dtype, mask, twice):
    """1024 x 1024 planes (512 x 512 with a PSF of twice their size), with
    the history, 300 iterations: launches of several iterations, many strips
    per block when the grid is smaller than the plane's strips."""
    n = 512 if twice else 1024
    case = _case(
        n, n, dtype, planes=(1, 1, 2), mask=mask, niter=300, twice=twice, seed=50
    )
    case["residual"][0, 0, 1] *= 1e-3
    case["kwargs"]["threshold_per_cycle"][0, 0, 1] = 1.0  # V stops at once
    _check(gpu, entry, case, threads=(6,))


def test_per_plane_budgets_thresholds_and_masks(gpu):
    """2 x 3 x 4 planes, each with its own mask kind, budget (0 and -1
    included) and threshold, and a broadcast PSF."""
    for dtype in (np.float64, np.float32):
        case = _case(40, 44, dtype, planes=(2, 3, 4), np_psf=1, niter=50, seed=18)
        rng = np.random.default_rng(18)
        case["kwargs"]["mask_cube"] = np.stack(
            [_mask(MASKS[i % len(MASKS)], 40, 44, rng) for i in range(24)]
        ).reshape(2, 3, 4, 40, 44)
        case["kwargs"]["max_iter_per_cycle_updated"] = np.array(
            [0, -1, 5, 100, 1, 37] * 4, np.int32
        ).reshape(2, 3, 4)
        case["kwargs"]["threshold_per_cycle"] = np.array(
            [0.0, 1e-4, -1.0, np.nan, 0.0, 2e-3] * 4, dtype
        ).reshape(2, 3, 4)
        for entry in ENTRIES:
            _check(gpu, entry, case, threads=(1, 24))


@pytest.mark.parametrize("entry", ENTRIES)
def test_stokes_i_active_and_v_idle(gpu, entry):
    case = _case(
        96, 96, np.float64, planes=(1, 1, 2), niter=400, threshold=[1e-5, 1.0], seed=31
    )
    case["residual"][0, 0, 1] *= 1e-3
    expected = _check(gpu, entry, case, threads=(0, -1, 1, 2, 6, 12, 24))
    assert expected[0]["stop_code"].tolist() == [[[1, 2]]]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("gain", [0.1, float("nan")])
def test_nan_payloads_are_nan_where_the_reference_has_nan(gpu, entry, dtype, gain):
    case = _case(
        48, 52, dtype, planes=(1, 1, 2), mask=None, niter=40, gain=gain, seed=70
    )
    for plane in range(2):
        _bits(case["residual"])[0, 0, plane, 20, [7, 20, 33]] = NAN_PAYLOADS[dtype]
        _bits(case["psf"])[0, 0, plane, 30, 21] = NAN_PAYLOADS[dtype][2]
    assert np.isnan(_check(gpu, entry, case, threads=(1,))[1]).any()


@pytest.mark.parametrize("entry", ENTRIES)
def test_mask_bytes_other_than_one_change_only_the_history_rms(gpu, entry):
    """As for hogbom=fast: the GPU searches the pixels whose byte is not 0 and
    divides the RMS by their count, the reference by the sum of the bytes."""
    rng = np.random.default_rng(72)
    raw = rng.choice(np.array([0, 1, 2, 128, 255], np.uint8), size=(1, 1, 2, 40, 44))
    case = _case(40, 44, np.float64, planes=(1, 1, 2), niter=60, seed=72)
    case["kwargs"]["mask_cube"] = raw.view(bool)
    plain = case | {"kwargs": case["kwargs"] | {"mask_cube": raw != 0}}
    _assert_same(_run(GPU[entry], plain), _run(GPU[entry], case))
    expected = _run(REFERENCE[entry], case)
    _assert_same(expected, _run(GPU[entry], case), skip=("iteration_residual_rms",))


@pytest.mark.parametrize("entry", ENTRIES)
def test_misaligned_arrays_run_on_the_gpu(gpu, entry):
    case = _case(48, 52, np.float64, planes=(1, 1, 2), niter=80, seed=61)
    expected = _run(REFERENCE[entry], case)
    residual, model = _misaligned(case["residual"]), _misaligned(case["model"])
    kwargs = case["kwargs"] | {
        "max_iter_per_cycle_updated": _misaligned(
            case["kwargs"]["max_iter_per_cycle_updated"]
        ),
        "threshold_per_cycle": _misaligned(case["kwargs"]["threshold_per_cycle"]),
    }
    result = GPU[entry](
        residual_cube=residual,
        psf_cube=_misaligned(case["psf"]),
        model_cube=model,
        **kwargs,
    )
    assert gpu == [entry]
    _assert_same(expected, (result, residual, model))


@pytest.mark.parametrize(
    "changes",
    [
        {"clean_box": (2, 10, 3)},  # not 4 items: the full extent
        {"gain": np.float32(0.3), "processing_function_threads": np.int64(3)},
        {
            "processing_function_threads": True,
            "record_iteration_history": np.bool_(True),
        },
        {
            "speedup": 9,
            "record_iteration_history": None,
        },  # None: False, as pybind11 casts
        {"mask_cube": np.zeros(0, np.int8)},  # empty: no mask
        {"clean_box": (np.int16(3), 12, True, -1), "divergence_peak_factor": 2},
    ],
)
def test_converted_arguments_run_on_the_gpu(gpu, changes):
    for entry in ENTRIES:
        runs = len(gpu)
        result = _outcome(GPU[entry], lambda: ((), _arguments(**changes)))
        assert len(gpu) == runs + 1
        expected = _outcome(REFERENCE[entry], lambda: ((), _arguments(**changes)))
        assert expected[1] is None
        _same_outcome(expected, result)


# ---------------------------------------------------------------- paths
def _trace(monkeypatch, tmp_path):
    """The hooked entries with hogbom=gpu and a trace into tmp_path, and a
    function that returns the (entry, path, error) of the traced calls."""
    monkeypatch.setenv(ENV, "hogbom=gpu")
    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    wrappers = {entry: dispatch("hogbom", REFERENCE[entry]) for entry in ENTRIES}

    def paths():
        (path,) = tmp_path.glob(f"kernels-{socket.gethostname()}-*.jsonl")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert {r["impl"] for r in records} == {"gpu"}
        return [(r["entry"], r["path"], r["error"]) for r in records]

    return wrappers, paths


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(DECLINED))
def test_declined_calls_are_the_reference(monkeypatch, entry, name):
    """Calls that the reference rejects, or that share memory: its outcome
    (result or exception, and every array's bytes), and the GPU does not run."""
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, "0")
    monkeypatch.setattr(hogbom, "_run", None)  # the GPU must not run
    make = DECLINED[name]
    _same_outcome(_outcome(REFERENCE[entry], make), _outcome(GPU[entry], make))


def test_the_trace_names_each_calls_path(device, monkeypatch, tmp_path):
    wrappers, paths = _trace(monkeypatch, tmp_path)
    small = _case(40, 44, np.float64, planes=(1, 1, 2), niter=30, seed=62)
    monkeypatch.delenv(hogbom.MIN_PIXELS_ENV, raising=False)  # 1,760 pixels: on the CPU
    expected = _run(REFERENCE["clean_cube"], small)
    _assert_same(expected, _run(wrappers["clean_cube"], small))
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, "1760")  # at the floor: on the GPU
    _assert_same(expected, _run(wrappers["clean_cube"], small), nan_bits=False)
    subclass = small | {"residual": small["residual"].view(np.memmap)}  # on the CPU
    _assert_same(expected, _run(wrappers["clean_cube"], subclass))
    gain = small | {
        "kwargs": small["kwargs"] | {"gain": np.float128(0.1)}
    }  # on the CPU
    _assert_same(
        _run(REFERENCE["clean_cube"], gain), _run(wrappers["clean_cube"], gain)
    )
    args, kwargs = DECLINED["float32 psf, float64 residual"]()
    with pytest.raises(RuntimeError, match="psf_cube has wrong dtype"):
        wrappers["clean_cube_many_threads"](*args, **kwargs)
    assert paths() == [
        ("clean_cube", "routed", None),
        ("clean_cube", "gpu", None),
        ("clean_cube", "routed", None),
        ("clean_cube", "routed", None),
        ("clean_cube_many_threads", "declined", "RuntimeError"),
    ]


def test_the_pixel_floor_is_validated(monkeypatch):
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, "4GiB")
    with pytest.raises(ValueError, match="ASTROVIPER_KERNELS_GPU_MIN_PIXELS='4GiB'"):
        GPU["clean_cube"](**_arguments())
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, " 288 ")
    assert hogbom.min_pixels() == 288


# ---------------------------------------------------------------- failures
def _plane_unchanged(before, after, plane):
    return np.array_equal(_bits(before[0, 0, plane]), _bits(after[0, 0, plane]))


def test_a_forced_abort_raises_and_the_next_call_is_right(gpu, monkeypatch):
    """The watchdog's limit set to 0 spins for the second plane (three strips:
    blocks wait at the first barrier): RuntimeError; the first plane is cleaned
    and written, the second unchanged; then the same call is right."""
    case = _case(96, 104, np.float64, planes=(1, 1, 2), niter=50, seed=40)
    expected = _run(REFERENCE["clean_cube"], case)
    summary = hogbom._Cleaner._summary

    def abort_the_next_plane(self, has_mask):
        monkeypatch.setattr(hogbom, "SPIN_LIMIT", 0)
        return summary(self, has_mask)

    monkeypatch.setattr(hogbom._Cleaner, "_summary", abort_the_next_plane)
    residual, model = case["residual"].copy(), case["model"].copy()
    with pytest.raises(RuntimeError, match="aborted plane 1"):
        GPU["clean_cube"](residual, case["psf"], model, **case["kwargs"])
    assert _plane_unchanged(expected[1], residual, 0) and _plane_unchanged(
        expected[2], model, 0
    )
    assert _plane_unchanged(case["residual"], residual, 1) and _plane_unchanged(
        case["model"], model, 1
    )
    monkeypatch.undo()
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, "0")
    _assert_same(expected, _run(GPU["clean_cube"], case))


def test_a_failing_launch_raises_and_the_next_call_is_right(gpu, device, monkeypatch):
    case = _case(64, 70, np.float32, planes=(1, 1, 2), niter=40, seed=41)
    launch = hogbom._launch

    def failing(kernel, grid, args):
        raise device.cp.cuda.driver.CUDADriverError(720)  # cooperative launch too large

    monkeypatch.setattr(hogbom, "_launch", failing)
    residual, model = case["residual"].copy(), case["model"].copy()
    with pytest.raises(RuntimeError, match="CUDA error"):
        GPU["clean_cube_many_threads"](residual, case["psf"], model, **case["kwargs"])
    assert np.array_equal(_bits(residual), _bits(case["residual"]))
    assert np.array_equal(_bits(model), _bits(case["model"]))
    monkeypatch.setattr(hogbom, "_launch", launch)
    _check(gpu, "clean_cube_many_threads", case, threads=(1,))


LIMITED = """
import json, numpy as np
from astroviper._kernels.gpu import hogbom
from tests.unit.kernels.test_hogbom_fast import _case, _run, _bits

out = {}
for n in (32, 256):
    case = _case(n, n, np.float64, planes=(1, 1, 2), niter=20, seed=63)
    reference = _run(hogbom.impl("hogbom", "clean_cube", "ref"), case)
    residual, model = case["residual"].copy(), case["model"].copy()
    try:
        hogbom.clean_cube(residual, case["psf"], model, **case["kwargs"])
        out[n] = all(np.array_equal(_bits(a), _bits(b)) for a, b in zip(reference[1:], (residual, model)))
    except (RuntimeError, ValueError) as e:
        unchanged = np.array_equal(_bits(residual), _bits(case["residual"])) and not model.any()
        out[n] = f"{type(e).__name__}: {e}" + ("" if unchanged else " (arrays written)")
print("RESULT", json.dumps(out))
"""


@pytest.mark.parametrize("limit", ["1048576", " 50%", "4GiB", "-1", "101%", "", "50% "])
def test_the_memory_limit_is_validated_and_kept(device, run_python, limit):
    """A byte count or a percentage; anything else raises ValueError at the
    first GPU call. 1 MiB holds the call of 32 x 32 planes but not that of
    256 x 256 planes, which raises RuntimeError having written nothing."""
    out = run_python(
        "-c",
        LIMITED,
        CUPY_GPU_MEMORY_LIMIT=limit,
        ASTROVIPER_KERNELS_GPU_MIN_PIXELS="0",
        PYTHONPATH=os.getcwd(),
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    assert not any("(arrays written)" in str(v) for v in result.values()), result
    if limit in ("4GiB", "-1", "101%", "", "50% "):
        message = f"ValueError: CUPY_GPU_MEMORY_LIMIT={limit!r}: expected a byte count"
        assert result["32"].startswith(message) and result["256"].startswith(message)
    elif limit == "1048576":
        assert result["32"] is True
        assert result["256"].startswith("RuntimeError: hogbom=gpu: the call needs")
    else:
        assert result == {"32": True, "256": True}


WITHOUT_DEVICE = """
import json, os, numpy as np
from astroviper._kernels.gpu import hogbom
from tests.unit.kernels.test_hogbom_fast import _case, _run, _bits

case = _case(40, 44, np.float64, planes=(1, 1, 2), niter=20, seed=64)
reference = _run(hogbom.impl("hogbom", "clean_cube", "ref"), case)
routed = _run(hogbom.clean_cube, case)  # below the floor: on the CPU
out = {"routed": all(np.array_equal(_bits(a), _bits(b)) for a, b in zip(reference[1:], routed[1:]))}
os.environ[hogbom.MIN_PIXELS_ENV] = "0"
residual = case["residual"].copy()
try:
    hogbom.clean_cube(residual, case["psf"], case["model"].copy(), **case["kwargs"])
except RuntimeError as e:
    out["error"] = str(e)
out["unchanged"] = bool(np.array_equal(_bits(residual), _bits(case["residual"])))
print("RESULT", json.dumps(out))
"""


def test_without_a_device_the_first_gpu_call_raises(device, run_python):
    out = run_python(
        "-c", WITHOUT_DEVICE, CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.getcwd()
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result["routed"] is True and result["unchanged"] is True
    assert result["error"].startswith("no usable CUDA device")


# ---------------------------------------------------------------- resources
def test_launches_are_short_and_grow_to_the_target(gpu):
    """A 1024 x 1024 plane of 2,000 iterations: a few iterations at first,
    then launches of about TARGET_MS, none of more than 100 ms."""
    case = _case(
        1024,
        1024,
        np.float64,
        mask=None,
        niter=2000,
        threshold=-1.0,
        divergence=False,
        seed=51,
    )
    call = hogbom._bind(
        "clean_cube", (case["residual"], case["psf"], case["model"]), case["kwargs"]
    )
    stats = {}
    hogbom._run(call, stats)
    launches = stats["launches"]
    assert launches[0][2] == 4 and sum(s for _, _, s, _ in launches) == 2001
    assert max(ms for *_, ms in launches) <= 100.0
    assert max(ms for *_, ms in launches) >= 0.4 * hogbom.TARGET_MS


MEMORY = """
import json, numpy as np
from astroviper._kernels.gpu import hogbom, prewarm
from tests.unit.kernels.test_hogbom_fast import _case

prewarm.prewarm()


def status(field):
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith(field))


case = _case(1024, 1024, np.float64, planes=(1, 1, 2), mask="disk", niter=200, seed=52)
arrays = [case["residual"], case["psf"], case["model"], *[a for a in case["kwargs"].values() if isinstance(a, np.ndarray)]]
call = hogbom._bind("clean_cube", (case["residual"].copy(), case["psf"], case["model"].copy()), case["kwargs"])
hogbom._run(call)  # the first call of this shape
call = hogbom._bind("clean_cube", (case["residual"], case["psf"], case["model"]), case["kwargs"])
with open("/proc/self/clear_refs", "w") as f:
    f.write("5")  # the peak resident set is now the current one
base = status("VmRSS:")
hogbom._run(call)
print("RESULT", json.dumps({"inputs": sum(a.nbytes for a in arrays), "growth": status("VmHWM:") - base}))
"""


def test_host_memory_stays_within_the_inputs(device, run_python):
    """The planes go to and from the caller's arrays without host copies: the
    peak resident set of a call grows by less than 10 % of its inputs."""
    out = run_python(
        "-c", MEMORY, ASTROVIPER_KERNELS_GPU_MIN_PIXELS="0", PYTHONPATH=os.getcwd()
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result["growth"] <= 0.1 * result["inputs"], result
