"""Tests of the GPU Hogbom minor cycle (``ASTROVIPER_KERNELS=hogbom=gpu``).

The implementation is called directly (``astroviper._kernels.impl``), so these
tests run in any session; those that need the GPU skip without CuPy or a CUDA
device. Planes of at most 1024 x 1024 pixels (the larger ones are checked on
the DATA side). The oracle is the reference binding, compared as in
``test_hogbom_fast`` (whose cases and comparison are reused): bitwise on
unsigned-integer views of everything a call returns or changes, NaN bits
included, except where an operation on two NaNs keeps the one that the
reference compiler's operand order picks and its build is not the same in
both entries and both dtypes (the NaN-payload tests).

* Both entries against their own reference entries on the GPU (the pixel
  floor set to 0): WP4's random cubes and edge cases (masks, NaN and
  infinities, ties, -0.0, subnormal and huge values, float32 and float64, PSFs
  of one and of two image sizes, clamped clean boxes, edge gains, thresholds,
  speedups, budgets and divergence controls, accumulated models), NaNs of
  both signs in the residual and the model, two NaNs of other bits in one
  operation, the sum of |model| over its gathers, planes of up to 1024 x 1024,
  and grids of fewer blocks than a plane's strips.
* Multi-plane cubes: per-plane budgets, thresholds and masks, a broadcast PSF,
  Stokes I active with V idle; the bits do not depend on the thread count.
* Paths: calls whose planes search fewer pixels than the floor run
  ``hogbom=fast`` (``routed``), and so do planes whose iterations the GPU
  cannot keep short, whatever the floor; arguments of forms the GPU entries do
  not take run on the CPU, the calls that the reference rejects or that share
  memory are the reference's (``declined``), misaligned arrays run on the GPU;
  the trace names each call's path, and a GPU call's first-use and device
  times and versions.
* Failures: a forced watchdog abort and a failing launch raise RuntimeError,
  leave the plane they hit and the later ones unchanged, say how many planes
  they wrote, hold no device memory while the exception is kept, and the next
  call is right; ``CUPY_GPU_MEMORY_LIMIT`` is validated and a call that does
  not fit raises RuntimeError having written nothing; without a device the
  first GPU call raises RuntimeError and routed calls still run; in a process
  forked during a GPU call the GPU call raises RuntimeError at once; a missing
  kernel source raises RuntimeError; ``--check`` and the prewarm validate the
  variables and find NVRTC and the CUDA headers (the subprocesses import this
  module from the repository root).
* The generated code: the PTX of the module, compiled as CuPy compiles it, has
  no fused or flushed operation, and the host's constants mirror the source's.
* Resources: launches stay short, and host memory within the inputs + 10 %.
"""

import importlib.resources
import importlib.util
import json
import os
import re
import socket
import traceback
import warnings
from pathlib import Path

import numpy as np
import pytest

from astroviper._kernels import impl
from astroviper._kernels.__main__ import main as check_main
from astroviper._kernels._switch import ENV, TRACE_ENV, dispatch
from astroviper._kernels.gpu import _runtime, hogbom, prewarm
from tests.unit.kernels.test_hogbom_fast import (
    DECLINED,
    EDGE_CASES,
    HISTORY,
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
    _psf,
    _random_case,
    _run,
)

#: The subprocesses import this module's helpers from the repository root,
#: whatever the directory the session runs in.
ROOT = str(Path(__file__).resolve().parents[3])
PYTHONPATH = os.pathsep.join(filter(None, [ROOT, os.environ.get("PYTHONPATH")]))

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
    """The same exception, array bytes and results (keys, dtypes, bits, the
    history included)."""
    assert a[1:] == b[1:]
    if a[1] is None:
        assert list(a[0]) == list(b[0])
        for key in SCALARS:
            x, y = np.asarray(a[0][key]), np.asarray(b[0][key])
            assert x.dtype == y.dtype and np.array_equal(_bits(x), _bits(y)), key
        for key in HISTORY:
            if key in a[0]:
                for t, f, p in np.ndindex(np.asarray(a[0]["stop_code"]).shape):
                    x, y = a[0][key][t][f][p], b[0][key][t][f][p]
                    assert x.dtype == y.dtype and np.array_equal(_bits(x), _bits(y))


def _check(gpu, entry, case, threads=(1, 7), nan_bits=True):
    """The GPU's results at each thread count are the reference's."""
    expected = _run(REFERENCE[entry], case)
    for count in threads:
        runs = len(gpu)
        actual = _run(GPU[entry], case, count)
        assert len(gpu) == runs + 1, "the call did not run on the GPU"
        _assert_same(expected, actual, nan_bits=nan_bits)
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
    the history, 300 iterations: launches of several iterations (one strip a
    block on a GPU whose grid exceeds the plane's 32 strips)."""
    n = 512 if twice else 1024
    case = _case(
        n, n, dtype, planes=(1, 1, 2), mask=mask, niter=300, twice=twice, seed=50
    )
    case["residual"][0, 0, 1] *= 1e-3
    case["kwargs"]["threshold_per_cycle"][0, 0, 1] = 1.0  # V stops at once
    _check(gpu, entry, case, threads=(6,))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype, mask", [(np.float64, "disk"), (np.float32, None)])
def test_grids_of_fewer_blocks_than_strips_are_the_reference(
    gpu, monkeypatch, entry, dtype, mask
):
    """Grids of 1, 2 and 7 blocks for planes of 10 strips of rows (the last
    one partial), as on a GPU of fewer multiprocessors: blocks clean several
    strips one after another, with the history."""
    case = _case(300, 200, dtype, mask=mask, niter=120, seed=94)
    for blocks in (1, 2, 7):
        monkeypatch.setattr(_runtime.Device, "max_blocks", lambda *_, b=blocks: b)
        _check(gpu, entry, case, threads=(1,))


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
            # as in the reference, every record of the planes of budget 0 and -1
            # is an array of its own
            result = _run(GPU[entry], case)[0]
            records = [result[key][0][0][p] for key in HISTORY for p in (0, 1)]
            assert len({id(r) for r in records}) == len(records)


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
    expected = _check(gpu, entry, case, threads=(1,), nan_bits=False)
    assert np.isnan(expected[1]).any()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nan_bits_are_the_references(gpu, entry, dtype):
    """NaNs of one sign a plane: +NaN in searched pixels and a blanked border
    under the footprints, -NaN in searched pixels of the other plane, and NaN
    in the model at each plane's first component. Every NaN result keeps its
    operand's bits, as in the reference (a float32 operation on the GPU makes
    0x7fffffff, which the kernel replaces by them)."""
    case = _case(64, 64, dtype, planes=(1, 1, 2), mask=None, niter=30, seed=70)
    for plane, nan in enumerate(NAN_PAYLOADS[dtype][:2]):
        _bits(case["residual"])[0, 0, plane, 20, [7, 20, 33]] = nan
    case["residual"][0, 0, 0, :, :3] = np.nan
    once = case["kwargs"] | {"max_iter_per_cycle_updated": np.ones((1, 1, 2), np.int32)}
    first = _run(REFERENCE[entry], case | {"kwargs": once})[0]
    for plane in range(2):
        py, px = (
            int(first[key][0][0][plane][0])
            for key in ("iteration_component_l", "iteration_component_m")
        )
        case["model"][0, 0, plane, py, px] = np.nan
    expected = _check(gpu, entry, case, threads=(1,))
    assert np.isnan(expected[1]).any()
    assert np.isnan(expected[2][0, 0]).any(axis=(1, 2)).all()  # each plane's model


#: Two NaNs of other bits than each other: a signalling +A and a quiet -B.
TWO_NANS = {
    np.float32: (0x7FA0000A, 0xFFC0000B),
    np.float64: (0x7FF400000000000A, 0xFFF800000000000B),
}


def _two_nan_cases(dtype):
    """(name, case, site): the NaNs A and B meet in one operation of a plane
    of 48 x 48, whose first component is at (27, 19); site(outcome) is the
    operation's result. (33, 14) is in that component's footprint, outside the
    box searched around it, and PSF pixel (30, 19) falls on it."""
    a, b = TWO_NANS[dtype]
    gain_a = np.uint64(TWO_NANS[np.float64][0]).view(np.float64)  # pv: A, quieted
    psf = _psf(48, 48, seed=3)
    box, far = np.zeros((48, 48), bool), np.zeros((48, 48), bool)
    box[24:31, 16:23], far[36:44, 36:44] = True, True  # far: no peak, (0, 0) is it

    def case(mask=box, niter=1, gain=0.1, peak=True):
        data = 2.0 * np.roll(psf, (3, -5), (0, 1)) if peak else np.zeros((48, 48))
        c = _case(48, 48, dtype, mask=mask, niter=niter, threshold=0.0 if peak else -1.0,
                  gain=gain, divergence=False, data=data, psf=psf)  # fmt: skip
        return c | {"psf": c["psf"].copy()}  # a plane's PSF cube is a read-only view

    def residual(outcome):
        return outcome[1][0, 0, 0, 33, 14]

    def model(outcome):
        return outcome[2][0, 0, 0, 0, 0]

    def rms(outcome):
        return outcome[0]["iteration_residual_rms"][0][0][0]

    c = case()
    _bits(c["residual"])[0, 0, 0, 33, 14], _bits(c["psf"])[0, 0, 0, 30, 19] = a, b
    yield "v - pv w", c, residual
    c = case(gain=gain_a)
    _bits(c["psf"])[0, 0, 0, 30, 19] = b
    yield "pv w", c, residual
    c = case(gain=gain_a)
    _bits(c["residual"])[0, 0, 0, 33, 14] = b
    yield "v - pv w, pv NaN", c, residual
    c = case(mask=far, peak=False)
    _bits(c["residual"])[0, 0, 0, 0, 0], _bits(c["model"])[0, 0, 0, 0, 0] = b, a
    yield "model + pv", c, model
    c = case(mask=far, gain=gain_a, peak=False)
    _bits(c["residual"])[0, 0, 0, 0, 0] = b
    yield "gain x peak", c, model
    for pair in ((a, b), (b, a)):
        c = case(mask=None, niter=2)
        _bits(c["residual"])[0, 0, 0, 27, [20, 21]] = pair
        yield "a row", c, rms
        c = case(mask=None, niter=2)
        _bits(c["residual"])[0, 0, 0, [28, 29], 19] = pair
        yield "rows", c, rms
        c = case()
        _bits(c["model"])[0, 0, 0, [1, 46], [1, 46]] = pair
        yield "|model|", c, lambda outcome: outcome[0]["total_flux_cleaned"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_two_nans_make_a_nan_where_the_reference_has_one(gpu, entry, dtype):
    """Two NaNs of other bits meet in the subtraction of a component, its
    product with the PSF, the model update, gain x peak, a row's sum of
    squares, the sum over rows and the sum of |model|. Every result is the
    reference's, a NaN where it has one (which of the two it keeps is its
    compiler's pick: see hogbom.py), and the bits of the NaNs that the kernel
    picks as the reference's build does are its: the model update's, the
    float32 subtraction's and product's, and the float64 sum of |model|."""
    exact = {"model + pv"} | (
        {"v - pv w", "pv w", "v - pv w, pv NaN"} if dtype == np.float32 else {"|model|"}
    )
    for name, case, site in _two_nan_cases(dtype):
        expected, actual = _run(REFERENCE[entry], case), _run(GPU[entry], case)
        assert np.isnan(site(expected)).all(), name
        _assert_same(expected, actual, nan_bits=name in exact)


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_sum_of_the_model_adds_its_rounds_in_order(gpu, monkeypatch, entry):
    """The |model| values gathered in rounds of 4096 (the least buffer) add up
    as one sum in plane order, silently as the reference does: a signalling
    NaN in the first round and a NaN at the start of the second included (the
    reference's float64 build keeps the first)."""
    monkeypatch.setattr(hogbom, "_GATHER_MAX", 1)
    case = _case(128, 128, np.float64, mask="disk", niter=20, seed=95)
    case["model"] = np.random.default_rng(95).normal(0.0, 1.0, case["model"].shape)
    _check(gpu, entry, case, threads=(1,))
    _bits(case["model"])[0, 0, 0, [0, 32], [100, 0]] = TWO_NANS[np.float64]
    expected = _run(REFERENCE[entry], case)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _assert_same(expected, _run(GPU[entry], case))
    assert gpu == [entry] * 2 and np.isnan(expected[0]["total_flux_cleaned"]).all()


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
    function that returns the records of the traced calls."""
    monkeypatch.setenv(ENV, "hogbom=gpu")
    monkeypatch.setenv(TRACE_ENV, str(tmp_path))
    wrappers = {entry: dispatch("hogbom", REFERENCE[entry]) for entry in ENTRIES}

    def records():
        (path,) = tmp_path.glob(f"kernels-{socket.gethostname()}-*.jsonl")
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert {r["impl"] for r in records} == {"gpu"}
        return records

    return wrappers, records


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
    """The paths, and the GPU call's first-use and device times and versions;
    the floor counts the pixels a plane searches, those of its mask."""
    wrappers, records = _trace(monkeypatch, tmp_path)
    small = _case(40, 44, np.float64, planes=(1, 1, 2), niter=30, seed=62)
    searched = int(np.count_nonzero(small["kwargs"]["mask_cube"][0, 0, 0]))
    assert searched < 40 * 44
    monkeypatch.delenv(hogbom.MIN_PIXELS_ENV, raising=False)  # on the CPU
    expected = _run(REFERENCE["clean_cube"], small)
    _assert_same(expected, _run(wrappers["clean_cube"], small))
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, str(searched + 1))  # on the CPU
    _assert_same(expected, _run(wrappers["clean_cube"], small))
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, str(searched))  # at the floor: on the GPU
    _assert_same(expected, _run(wrappers["clean_cube"], small))
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
    traced = records()
    assert [(r["entry"], r["path"], r["error"]) for r in traced] == [
        ("clean_cube", "routed", None),
        ("clean_cube", "routed", None),
        ("clean_cube", "gpu", None),
        ("clean_cube", "routed", None),
        ("clean_cube", "routed", None),
        ("clean_cube_many_threads", "declined", "RuntimeError"),
    ]
    record = traced[2]
    assert record["init_s"] >= 0 and record["device_s"] > 0
    assert record["versions"] == json.loads(json.dumps(device.versions))
    assert not any("device_s" in r for r in traced[:2] + traced[3:])


def test_the_pixel_floor_is_validated(monkeypatch):
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, "4GiB")
    with pytest.raises(ValueError, match="ASTROVIPER_KERNELS_GPU_MIN_PIXELS='4GiB'"):
        GPU["clean_cube"](**_arguments())
    monkeypatch.setenv(hogbom.MIN_PIXELS_ENV, " 288 ")
    assert hogbom.min_pixels() == 288


def test_the_floor_counts_what_the_planes_with_a_budget_search():
    """The pixels of the clean box inside the mask, of the planes with a
    budget: here 100 of plane 1 (and none of plane 0, whose budget is 0)."""
    case = _case(40, 44, np.float64, planes=(1, 1, 2), clean_box=(4, 30, 6, 36))
    mask = np.zeros((1, 1, 2, 40, 44), bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1, 10:20, 5:15] = True
    mask[0, 0, 1, :3, :3] = True  # outside the clean box
    kwargs = case["kwargs"] | {
        "mask_cube": mask,
        "max_iter_per_cycle_updated": np.array([[[0, 10]]], np.int32),
    }
    arrays = (case["residual"], case["psf"], case["model"])
    call = hogbom._bind("clean_cube", arrays, kwargs)
    assert hogbom._searched(call, 10**9) == 100
    on_gpu = [hogbom._on_gpu(call, floor) for floor in (0, 100, 101)]
    assert on_gpu == [True, True, False]
    call = hogbom._bind("clean_cube", arrays, kwargs | {"mask_cube": np.array([])})
    assert hogbom._searched(call, 10**9) == 26 * 30  # the clean box


@pytest.mark.parametrize(
    "shape", [(1, 1 << 20), (1 << 20, 1), (8, 1 << 17), (1 << 17, 8)]
)
def test_planes_of_long_iterations_run_in_short_launches_or_on_the_cpu(gpu, shape):
    """An iteration scans each row in one lane and adds up the rows in one
    lane: planes whose iterations would take more than TARGET_MS run on the
    CPU whatever the floor, and the others start with fewer iterations a
    launch, so that no launch takes much longer than TARGET_MS."""
    ny, nx = shape
    case = _case(ny, nx, np.float64, mask=None, niter=6, seed=53)
    expected = _run(REFERENCE["clean_cube"], case)
    short = hogbom._iteration_ms(ny, nx) <= hogbom.TARGET_MS
    runs = len(gpu)
    _assert_same(expected, _run(GPU["clean_cube"], case))
    assert len(gpu) == runs + short
    if short:
        arrays = (case["residual"].copy(), case["psf"], case["model"].copy())
        stats = {}
        hogbom._run(hogbom._bind("clean_cube", arrays, case["kwargs"]), stats)
        first = max(1, int(hogbom.TARGET_MS / hogbom._iteration_ms(ny, nx)))
        assert stats["launches"][0][2] <= first
        assert max(ms for *_, ms in stats["launches"]) <= 100.0


# ---------------------------------------------------------------- failures
def _plane_unchanged(before, after, plane):
    return np.array_equal(_bits(before[0, 0, plane]), _bits(after[0, 0, plane]))


def test_a_forced_abort_raises_and_the_next_call_is_right(gpu, device, monkeypatch):
    """The watchdog's limit set to 0 spins for the second plane (three strips:
    blocks wait at the first barrier): RuntimeError; the first plane is cleaned
    and written, the second unchanged, and the exception says so and holds no
    device memory while it is kept; then the same call is right."""
    case = _case(96, 104, np.float64, planes=(1, 1, 2), niter=50, seed=40)
    expected = _run(REFERENCE["clean_cube"], case)
    summary = hogbom._Cleaner._summary

    def abort_the_next_plane(self, has_mask):
        monkeypatch.setattr(hogbom, "SPIN_LIMIT", 0)
        return summary(self, has_mask)

    monkeypatch.setattr(hogbom._Cleaner, "_summary", abort_the_next_plane)
    residual, model = case["residual"].copy(), case["model"].copy()
    used = device.pool.used_bytes()
    with pytest.raises(RuntimeError, match="aborted plane 1") as raised:
        GPU["clean_cube"](residual, case["psf"], model, **case["kwargs"])
    assert device.pool.used_bytes() == used, "the kept exception holds device memory"
    assert raised.value.planes_written == 1
    text = "".join(traceback.format_exception(raised.value))
    assert "cleaned and wrote back 1 of its 2 planes" in text and "_clean" in text
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
        raise device.cp.cuda.driver.CUDADriverError(720)  # any error of a launch

    monkeypatch.setattr(hogbom, "_launch", failing)
    residual, model = case["residual"].copy(), case["model"].copy()
    used = device.pool.used_bytes()
    with pytest.raises(RuntimeError, match="CUDA error") as raised:
        GPU["clean_cube_many_threads"](residual, case["psf"], model, **case["kwargs"])
    assert device.pool.used_bytes() == used and raised.value.planes_written == 0
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
        PYTHONPATH=PYTHONPATH,
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
        "-c", WITHOUT_DEVICE, CUDA_VISIBLE_DEVICES="", PYTHONPATH=PYTHONPATH
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result["routed"] is True and result["unchanged"] is True
    assert result["error"].startswith("no usable CUDA device")


FORKED = """
import json, os, signal, threading, numpy as np
from astroviper._kernels.gpu import _runtime, hogbom
from tests.unit.kernels.test_hogbom_fast import _case, _run, _bits

os.environ[hogbom.MIN_PIXELS_ENV] = "0"
case = _case(40, 44, np.float64, niter=10, seed=65)
expected = _run(hogbom.impl("hogbom", "clean_cube", "ref"), case)


def right():
    run = _run(hogbom.clean_cube, case)
    return all(np.array_equal(_bits(a), _bits(b)) for a, b in zip(expected[1:], run[1:]))


first = right()
device = _runtime.require_device()
held, release = threading.Event(), threading.Event()


def gpu_call():  # in flight at the fork: holds the device lock
    with device.lock:
        held.set()
        release.wait()


thread = threading.Thread(target=gpu_call)
thread.start()
held.wait()
read, write = os.pipe()
pid = os.fork()
if pid == 0:
    signal.alarm(30)
    try:
        right()
        message = "ran"
    except RuntimeError as e:
        message = str(e)
    os.write(write, message.encode())
    os._exit(0)
os.close(write)
release.set()
thread.join()
status = os.waitpid(pid, 0)[1]
print("RESULT", json.dumps({"first": first, "status": status, "child": os.read(read, 999).decode(), "after": right()}))
"""


def test_a_process_forked_during_a_gpu_call_raises_at_its_first(device, run_python):
    """The child does not wait for the device lock that a thread of its parent
    held at the fork, nor use the parent's context: RuntimeError at once."""
    out = run_python(
        "-W", "ignore::DeprecationWarning", "-c", FORKED, PYTHONPATH=PYTHONPATH
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result["first"] is True and result["after"] is True and result["status"] == 0
    assert result["child"].startswith("the CUDA device was opened before this process")


@pytest.mark.parametrize(
    "variable, value, message",
    [
        (hogbom.MIN_PIXELS_ENV, "abc", "ASTROVIPER_KERNELS_GPU_MIN_PIXELS='abc'"),
        (_runtime.LIMIT_ENV, "4GiB", "CUPY_GPU_MEMORY_LIMIT='4GiB'"),
        (None, None, "CuPy is not available"),
    ],
)
def test_the_check_and_the_prewarm_validate_the_gpu_environment(
    monkeypatch, capsys, variable, value, message
):
    """``--check`` (which touches no device) and the prewarm fail, naming the
    cause, on an invalid floor or memory limit and without CuPy."""
    monkeypatch.setenv(ENV, "hogbom=gpu")
    if variable is None:
        find_spec = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *args: None if name == "cupy" else find_spec(name, *args),
        )
    else:
        monkeypatch.setenv(variable, value)
    assert check_main(["--check"]) == 1 and prewarm.main([]) == 1
    prefixes = ("ASTROVIPER_KERNELS check failed: ", "GPU prewarm failed: ")
    lines = capsys.readouterr().err.splitlines()
    errors = [line for line in lines if line.startswith(prefixes)]
    assert len(errors) == 2 and all(map(str.startswith, errors, prefixes))
    assert all(message in line for line in errors)


@pytest.mark.skipif(importlib.util.find_spec("cupy") is None, reason="no CuPy")
def test_the_check_and_the_prewarm_need_nvrtc_and_the_cuda_headers(monkeypatch, capsys):
    """Where cuda.pathfinder finds no NVRTC, or no CUDA headers (as in an
    environment that sees CuPy through a .pth file, without CUDA_PATH), CuPy
    could compile no kernel: ``--check`` and the prewarm fail at once."""
    pathfinder = pytest.importorskip("cuda.pathfinder")
    monkeypatch.setenv(ENV, "hogbom=gpu")

    def no_library(name):
        raise pathfinder.DynamicLibNotFoundError(f"no {name}")

    for name, patch in [
        ("load_nvidia_dynamic_lib", no_library),
        ("find_nvidia_header_directory", lambda name: None),
    ]:
        with monkeypatch.context() as patched:
            patched.setattr(pathfinder, name, patch)
            assert check_main(["--check"]) == 1 and prewarm.main([]) == 1
        lines = capsys.readouterr().err.splitlines()
        message = "CuPy cannot find NVRTC and the CUDA headers"
        assert len([line for line in lines if message in line]) == 2, lines


def test_the_dask_preload_hook_prewarms(device):
    prewarm.dask_setup(None)  # the worker it is given is not used
    assert ("hogbom_exact", hogbom._CHUNK_KERNELS, True) in device._modules


def test_a_missing_kernel_source_raises_runtime_error(device):
    message = r"cannot compile or load cuda/missing\.cu: .*such file"
    with pytest.raises(RuntimeError, match=message):
        device.module("missing", ("kernel",))


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
        "-c", MEMORY, ASTROVIPER_KERNELS_GPU_MIN_PIXELS="0", PYTHONPATH=PYTHONPATH
    )
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result["growth"] <= 0.1 * result["inputs"], result


# ---------------------------------------------------------------- the generated code
def _source():
    files = importlib.resources.files("astroviper._kernels.gpu")
    return files.joinpath("cuda", "hogbom_exact.cu").read_text(encoding="utf-8")


def _ptx(arch):
    """The PTX of the module for compute_<arch>, compiled by NVRTC with the
    options CuPy gives it (ours, its -ftz=true and execution space); no device
    is needed."""
    cp = pytest.importorskip("cupy", reason="no CuPy")
    try:
        cp.cuda.nvrtc.getVersion()
    except Exception as e:  # cuda.pathfinder's DynamicLibNotFoundError
        pytest.skip(f"no NVRTC: {e}")
    nvrtc = cp.cuda.nvrtc
    cupy_options = ("-ftz=true", "--device-as-default-execution-space")
    program = nvrtc.createProgram(_source(), "hogbom_exact.cu", [], [])
    try:
        for name in hogbom._CHUNK_KERNELS + hogbom._SUMMARY_KERNELS:
            nvrtc.addNameExpression(program, name)
        arch_option = f"-arch=compute_{arch}"
        nvrtc.compileProgram(program, (*_runtime.OPTIONS, *cupy_options, arch_option))
        ptx = nvrtc.getPTX(program)
    finally:
        nvrtc.destroyProgram(program)
    return ptx.decode() if isinstance(ptx, bytes) else ptx


#: NVRTC 13's oldest architecture, an RTX 3090's, and two of the newest
#: (whose back end turns integer tests of a number's bits into float ones).
ARCHS = (75, 86, 100, 120)
#: Float operations that are fused or approximate, or whose NaN or subnormal
#: results PTX does not fix (the source does abs and the like on the bits).
NOT_EXACT = ("fma", "mad", "div", "rcp", "sqrt", "rsqrt", "ex2", "lg2", "sin", "cos")
NOT_EXACT += ("abs", "neg", "min", "max")


@pytest.mark.parametrize("arch", ARCHS)
def test_the_kernels_compile_without_fused_or_flushed_operations(arch):
    """Exactness depends on code that CuPy's -ftz=true and contraction would
    change if the source used plain float expressions: no float operation
    flushes subnormals, none is fused, approximate or of unfixed NaN, and
    every float add, subtract and multiply rounds to nearest (.rn)."""
    ptx = _ptx(arch)
    ops = re.findall(r"^\s*(?:@!?%p\d+\s+)?([a-z][\w.]*)\s", ptx, re.M)
    floats = [op.split(".") for op in ops if re.search(r"\.f(16|32|64)\b", op)]
    assert ".ftz" not in ptx
    assert not [".".join(op) for op in floats if op[0] in NOT_EXACT]
    arithmetic = [op for op in floats if op[0] in ("add", "sub", "mul")]
    assert {op[-1] for op in arithmetic} == {"f32", "f64"}
    assert all("rn" in op for op in arithmetic), "an add, sub or mul without .rn"


def test_the_host_mirrors_the_kernel_source():
    """The block size, the summary's tile and the control words that the host
    reads by number are the source's."""
    source = _source()
    defines = dict(
        re.findall(r"^#define (SUM_THREADS|SUM_PER_THREAD) (\d+)$", source, re.M)
    )
    threads, per_thread = int(defines["SUM_THREADS"]), int(defines["SUM_PER_THREAD"])
    assert threads == hogbom._THREADS and threads * per_thread == hogbom._SUM_TILE
    chunk = re.findall(r"__launch_bounds__\((\d+)\) hogbom_exact_chunk", source)
    assert chunk == [str(hogbom._THREADS)]
    words = re.search(r"enum \{ (S_SUB = 0,[^}]*) \};", source)[1].split(", ")
    words = [w.removesuffix(" = 0") for w in words]
    assert len(words) <= hogbom._BARRIER
    assert [words.index(f"S_{w}") for w in ("STOP", "COMPONENTS", "REASON")] == [
        hogbom._S_STOP,
        hogbom._S_COMPONENTS,
        hogbom._S_REASON,
    ]
    assert "abort_flag = bar + 2;" in source and hogbom._ABORT == hogbom._BARRIER + 2
