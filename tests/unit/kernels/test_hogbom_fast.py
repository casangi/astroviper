"""Tests of the fast Hogbom minor cycle (``ASTROVIPER_KERNELS=hogbom=fast``).

The implementation is called directly (``astroviper._kernels.impl``), so these
tests run in any session. Small inputs only. The oracle is the reference
binding, and every comparison is bitwise on unsigned-integer views (-0.0 and
NaN payloads count) of everything a call returns or changes: the residual and
model cubes, the iterations, stop codes, converged and diverged flags,
final_peak, total_flux_cleaned and the six iteration-history records.

* Each fast entry against its own reference entry (they differ in corner
  cases: a zero peak that a negative threshold keeps running, negative
  budgets), on seeded random cubes and on edge cases: masks (none, all-true,
  boxes, a disk, a ring, short runs, checkerboards, stripes, empty, a single
  pixel), NaN and infinities inside and outside the searched pixels, ties,
  -0.0, subnormal and huge values, float32 and float64, PSFs of the image's
  size and of twice it, clean boxes that the binding clamps, edge gains,
  thresholds, speedups, budgets and divergence controls, accumulated models.
* Multi-plane cubes: per-plane budgets, thresholds and masks, a broadcast PSF,
  Stokes I active with V idle, many small planes at 1 to 24 threads.
* Invariance: bitwise across thread counts (0, -1 and 1 to 24) and between the
  generic and AVX2 routines (two subprocesses).
* Declined calls are the reference's, bit for bit and exception for
  exception; arrays the call writes that share memory with another argument
  decline; misaligned arrays take the fast path through aligned copies.
* No thread outlives a call, the GIL is released during it, and the trace of
  the hooked shim names each call's path.
"""

import json
import os
import resource
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from astroviper._kernels import impl

ext = pytest.importorskip(
    "astroviper._kernels.fast.hogbom._hogbom_fast_ext",
    reason="fast Hogbom module not built",
)

ENTRIES = ("clean_cube", "clean_cube_many_threads")
FAST = {entry: impl("hogbom", entry, "fast") for entry in ENTRIES}
REFERENCE = {entry: impl("hogbom", entry, "ref") for entry in ENTRIES}
HISTORY = (
    "iteration_peak_residual",
    "iteration_residual_rms",
    "iteration_component_l",
    "iteration_component_m",
    "iteration_component_flux",
    "iteration_component_scale",
)
SCALARS = (
    "iterations_performed",
    "final_peak",
    "total_flux_cleaned",
    "converged",
    "stop_code",
    "diverged",
)


# ---------------------------------------------------------------- inputs
def _psf(ny, nx, seed=0, beam=2.0, sidelobe=0.03):
    """An image-like PSF, peak 1 at (ny // 2, nx // 2): a rotated elliptical
    main lobe and rings over the whole plane."""
    rng = np.random.default_rng(seed)
    y = (np.arange(ny) - ny // 2)[:, None]
    x = (np.arange(nx) - nx // 2)[None, :]
    theta = 0.4 + 0.2 * rng.random()
    u = x * np.cos(theta) + y * np.sin(theta)
    v = -x * np.sin(theta) + y * np.cos(theta)
    r = np.hypot(u / 1.3, v)
    psf = np.exp(-0.5 * np.hypot(u / (1.3 * beam), v / beam) ** 2)
    psf += sidelobe * np.cos(r * 2.5 / beam) / (1.0 + r / (16.0 * beam))
    psf[ny // 2, nx // 2] = 1.0
    return psf


def _dirty(ny, nx, psf, seed):
    """Point sources and a ring convolved (circularly) with the PSF, plus noise."""
    rng = np.random.default_rng(seed)
    sky = np.zeros((ny, nx))
    y = (np.arange(ny) - ny // 2)[:, None]
    x = (np.arange(nx) - nx // 2)[None, :]
    radius = max(1.0, 0.3 * min(ny, nx))
    sky += 1e-3 * np.exp(-0.5 * ((np.hypot(x, y / 0.75) - radius * 0.6) / 1.5) ** 2)
    for _ in range(4):
        sky[rng.integers(0, ny), rng.integers(0, nx)] += rng.uniform(1e-3, 5e-3)
    kernel = psf
    if psf.shape != sky.shape:  # a PSF of twice the image: its central window
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        kernel = psf[cy - ny // 2 : cy - ny // 2 + ny, cx - nx // 2 : cx - nx // 2 + nx]
    kernel = np.fft.ifftshift(kernel)
    dirty = np.fft.irfft2(np.fft.rfft2(sky) * np.fft.rfft2(kernel), s=sky.shape)
    return dirty + 2e-5 * rng.standard_normal((ny, nx))


def _mask(kind, ny, nx, rng):
    yy, xx = np.mgrid[0:ny, 0:nx]
    if kind == "disk":
        r = rng.uniform(0.15, 0.55) * min(ny, nx)
        return np.hypot(yy - ny // 2, xx - nx // 2) < r
    if kind == "boxes":
        mask = np.zeros((ny, nx), bool)
        for _ in range(int(rng.integers(1, 4))):
            y0, x0 = int(rng.integers(0, ny)), int(rng.integers(0, nx))
            mask[
                y0 : y0 + int(rng.integers(1, ny + 1)),
                x0 : x0 + int(rng.integers(1, nx + 1)),
            ] = True
        return mask
    if kind.startswith("random"):
        return rng.random((ny, nx)) < float(kind[6:]) / 100.0
    if kind == "checker":
        return (yy + xx) % 2 == 0
    if kind == "stripes":
        return xx % 3 == 0
    if kind == "rowgaps":
        return yy % 2 == 1
    if kind == "empty":
        return np.zeros((ny, nx), bool)
    if kind == "all":
        return np.ones((ny, nx), bool)
    if kind == "single":
        mask = np.zeros((ny, nx), bool)
        mask[int(rng.integers(0, ny)), int(rng.integers(0, nx))] = True
        return mask
    if kind == "border":
        mask = np.zeros((ny, nx), bool)
        w = int(rng.integers(1, 4))
        mask[:w], mask[-w:], mask[:, :w], mask[:, -w:] = True, True, True, True
        return mask
    if kind == "ring":
        r = np.hypot(yy - ny / 2, xx - nx / 2)
        return (r > 0.2 * min(ny, nx)) & (r < 0.4 * min(ny, nx))
    if kind == "runs":  # runs of every length 1..17 and gaps 1..5: lane and tail edges
        mask = np.zeros((ny, nx), bool)
        for y in range(ny):
            x = int(rng.integers(0, 4))
            while x < nx:
                length = int(rng.integers(1, 18))
                mask[y, x : x + length] = True
                x += length + int(rng.integers(1, 6))
        return mask
    raise ValueError(kind)


MASKS = (
    "disk",
    "boxes",
    "random10",
    "random50",
    "random90",
    "checker",
    "stripes",
    "rowgaps",
    "empty",
    "all",
    "single",
    "border",
    "ring",
    "runs",
)


def _dyadic_psf(ny, nx):
    """A PSF of powers of two, for exact ties."""
    psf = np.zeros((ny, nx))
    cy, cx = ny // 2, nx // 2
    psf[cy, cx] = 1.0
    for dy, dx, v in (
        (0, 1, 0.5),
        (1, 0, 0.5),
        (0, -1, 0.5),
        (-1, 0, 0.5),
        (3, 4, -0.25),
    ):
        if 0 <= cy + dy < ny and 0 <= cx + dx < nx:
            psf[cy + dy, cx + dx] = v
    return psf


def _divergence(gain, on=True):
    """deconvolution._divergence_kwargs for Hogbom (or the test off)."""
    if not on:
        return {
            "max_iter_divergence": -1,
            "divergence_rms_factor": 1.0,
            "divergence_peak_factor": 1.0,
        }
    return {
        "max_iter_divergence": 1,
        "divergence_rms_factor": 1.0 + 0.1 * gain,
        "divergence_peak_factor": 1.0 + gain,
    }


def _case(
    ny=48,
    nx=52,
    dtype=np.float64,
    planes=(1, 1, 1),
    mask="disk",
    niter=60,
    threshold=0.0,
    gain=0.1,
    divergence=True,
    history=True,
    clean_box=(-1, -1, -1, -1),
    seed=0,
    twice=False,
    data=None,
    psf=None,
    model=None,
    np_psf=None,
    speedup=None,
):
    """A call's arrays and keyword arguments (without the thread count)."""
    rng = np.random.default_rng(seed)
    pny, pnx = (2 * ny, 2 * nx) if twice else (ny, nx)
    psf_plane = _psf(pny, pnx, seed=seed) if psf is None else psf
    dirty = _dirty(ny, nx, psf_plane, seed + 1) if data is None else data
    nplanes = int(np.prod(planes))
    np_psf = planes[2] if np_psf is None else np_psf
    residual = np.empty(planes + (ny, nx), dtype)
    for i, index in enumerate(np.ndindex(planes)):
        residual[index] = (dirty * (1.0 + 0.25 * i)).astype(dtype)
    psf_cube = np.ascontiguousarray(
        np.broadcast_to(psf_plane.astype(dtype), planes[:2] + (np_psf, pny, pnx))
    )
    if mask is None:
        mask_cube = np.array([], dtype=dtype)
    elif isinstance(mask, str):
        mask_cube = np.ascontiguousarray(
            np.broadcast_to(_mask(mask, ny, nx, rng), residual.shape)
        )
    else:
        mask_cube = np.ascontiguousarray(np.broadcast_to(mask, residual.shape))
    kwargs = {
        "mask_cube": mask_cube,
        "clean_box": tuple(clean_box),
        "max_iter_per_cycle_updated": np.broadcast_to(
            np.asarray(niter, np.int32), (nplanes,)
        )
        .reshape(planes)
        .copy(),
        "gain": gain,
        "threshold_per_cycle": np.broadcast_to(np.asarray(threshold, dtype), (nplanes,))
        .reshape(planes)
        .copy(),
        "record_iteration_history": history,
        **(
            _divergence(gain, divergence)
            if isinstance(divergence, bool)
            else divergence
        ),
    }
    if speedup is not None:
        kwargs["speedup"] = speedup
    return {
        "residual": residual,
        "psf": psf_cube,
        "model": np.zeros_like(residual) if model is None else model.astype(dtype),
        "kwargs": kwargs,
    }


def _random_case(seed):
    """A seeded random cube, as the prototype's verification draws them."""
    rng = np.random.default_rng(seed)
    dtype = np.float64 if rng.random() < 0.6 else np.float32
    if rng.random() < 0.1:
        ny, nx = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    else:
        ny, nx = int(rng.integers(8, 72)), int(rng.integers(8, 72))
    planes = (1, int(rng.choice([1, 2, 3])), int(rng.choice([1, 2, 2, 4])))
    nplanes = int(np.prod(planes))
    np_psf = planes[2] if (planes[2] == 1 or rng.random() < 0.5) else 1
    kind = str(rng.choice(["smooth", "smooth", "ties", "sparse", "nan", "negzero"]))
    gain = float(rng.choice([0.1, 0.3, 0.05, 1.0]))
    twice = rng.random() < 0.2
    pny, pnx = (2 * ny, 2 * nx) if twice else (ny, nx)
    psf = _dyadic_psf(pny, pnx) if kind == "ties" else _psf(pny, pnx, seed=seed)
    if kind == "ties":
        data = rng.integers(-4, 5, size=(ny, nx)).astype(np.float64)
        gain = float(rng.choice([0.5, 0.25, 1.0]))
    elif kind == "sparse":
        data = np.zeros((ny, nx))
        k = max(1, ny * nx // 50)
        data.flat[rng.integers(0, ny * nx, k)] = rng.choice([-2.0, -1.0, 1.0, 2.0], k)
    else:
        data = _dirty(ny, nx, psf, seed + 3)
        if kind == "nan":
            data[rng.random((ny, nx)) < 0.01] = np.nan
        if kind == "negzero":
            data[rng.random((ny, nx)) < 0.05] = -0.0
            data[rng.random((ny, nx)) < 0.05] = 0.0
    if rng.random() < 0.25:
        mask = None
    else:
        mask = np.stack(
            [_mask(str(rng.choice(MASKS)), ny, nx, rng) for _ in range(nplanes)]
        ).reshape(planes + (ny, nx))
    r = rng.random()
    if r < 0.45:
        clean_box = (-1, -1, -1, -1)
    elif r < 0.75:
        x0, y0 = int(rng.integers(0, nx)), int(rng.integers(0, ny))
        clean_box = (
            x0,
            int(rng.integers(x0 + 1, nx + 1)),
            y0,
            int(rng.integers(y0 + 1, ny + 1)),
        )
    else:
        clean_box = tuple(int(v) for v in rng.integers(-12, max(ny, nx) + 12, 4))
    niter = rng.choice(
        [-2, 0, 1, 3, 40, 120, 250],
        nplanes,
        p=[0.04, 0.06, 0.05, 0.1, 0.35, 0.25, 0.15],
    )
    peak = (
        float(np.nanmax(np.abs(data))) if np.isfinite(np.nanmax(np.abs(data))) else 1.0
    )
    threshold = [
        float(
            rng.choice(
                [0.0, peak * 10 ** rng.uniform(-3, -1), 1e6, -1.0, np.nan],
                p=[0.35, 0.35, 0.1, 0.15, 0.05],
            )
        )
        for _ in range(nplanes)
    ]
    draw = rng.random()
    divergence = (
        True
        if draw < 0.6
        else False
        if draw < 0.85
        else {
            "max_iter_divergence": int(rng.choice([0, 2, 5])),
            "divergence_rms_factor": float(rng.choice([1.0, 1.01, 2.0])),
            "divergence_peak_factor": float(rng.choice([1.05, 1.5, np.inf])),
        }
    )
    case = _case(
        ny,
        nx,
        dtype,
        planes,
        mask,
        niter,
        threshold,
        gain,
        divergence,
        history=bool(rng.random() < 0.5),
        clean_box=clean_box,
        seed=seed,
        twice=twice,
        data=data,
        psf=psf,
        np_psf=np_psf,
        speedup=float(rng.uniform(5, 50)) if rng.random() < 0.15 else None,
    )
    if rng.random() < 0.3:  # an accumulated model, as the cube imager passes it
        model = case["model"]
        k = max(1, model.size // 40)
        model.flat[rng.integers(0, model.size, k)] = rng.standard_normal(k).astype(
            dtype
        )
        model.flat[rng.integers(0, model.size, k)] = -0.0
    threads = [int(t) for t in rng.choice([1, 2, 3, 5, 8, 12, 24], 2, replace=False)]
    return case, threads


# ---------------------------------------------------------------- comparison
def _bits(array):
    array = np.ascontiguousarray(array)
    return array.view({8: np.uint64, 4: np.uint32, 1: np.uint8}[array.dtype.itemsize])


def _run(function, case, threads=1, **changes):
    residual, model = case["residual"].copy(), case["model"].copy()
    kwargs = case["kwargs"] | {"processing_function_threads": threads} | changes
    result = function(
        residual_cube=residual, psf_cube=case["psf"], model_cube=model, **kwargs
    )
    return result, residual, model


def _assert_same(expected, actual):
    """Bitwise: the results (keys in order, dtypes, shapes, bits) and the cubes."""
    (result, residual, model), (fast_result, fast_residual, fast_model) = (
        expected,
        actual,
    )
    assert list(fast_result) == list(result)
    for key in SCALARS:
        a, b = np.asarray(result[key]), np.asarray(fast_result[key])
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert np.array_equal(_bits(a), _bits(b)), key
    for key in HISTORY:
        if key not in result:
            continue
        for t, f, p in np.ndindex(np.asarray(result["stop_code"]).shape):
            a, b = result[key][t][f][p], fast_result[key][t][f][p]
            assert a.dtype == b.dtype and a.shape == b.shape, (key, t, f, p)
            assert np.array_equal(_bits(a), _bits(b)), (key, t, f, p)
    assert np.array_equal(_bits(fast_residual), _bits(residual)), "residual"
    assert np.array_equal(_bits(fast_model), _bits(model)), "model"


def _check(entry, case, threads=(1, 6)):
    expected = _run(REFERENCE[entry], case)
    for count in threads:
        actual = _run(FAST[entry], case, count)
        _assert_same(expected, actual)
    return expected


# ---------------------------------------------------------------- the reference's bits
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("seed", range(40))
def test_random_cubes_are_the_reference(entry, seed):
    case, threads = _random_case(seed)
    _check(entry, case, threads)


def _edge_cases():
    cases = {}
    for dtype in (np.float64, np.float32):
        t = dtype.__name__
        tiny = 1e-310 if dtype == np.float64 else 1e-41
        big = (
            1e160 if dtype == np.float64 else 1e30
        )  # squares overflow the double sum / T
        dirty = _dirty(48, 52, _psf(48, 52, seed=5), 6)
        # A zero peak that a negative threshold keeps running: clean_cube takes
        # pixel (0, 0), clean_cube_many_threads the first of the clean box,
        # either of which may be off the searched pixels.
        cases[f"empty mask, threshold -1 {t}"] = _case(
            dtype=dtype, mask="empty", niter=6, threshold=-1.0, divergence=False
        )
        cases[f"empty mask, threshold -1, clean box {t}"] = _case(
            dtype=dtype,
            mask="empty",
            niter=6,
            threshold=-1.0,
            divergence=False,
            clean_box=(7, 30, 9, 33),
        )
        outside = np.zeros((48, 52), bool)
        outside[:5] = True
        cases[f"mask only outside the clean box, threshold -1 {t}"] = _case(
            dtype=dtype,
            mask=outside,
            niter=6,
            threshold=-1.0,
            divergence=False,
            clean_box=(5, 35, 10, 40),
        )
        cases[f"all-zero residual, threshold -1, clean box {t}"] = _case(
            dtype=dtype,
            mask=None,
            niter=5,
            threshold=-1.0,
            divergence=False,
            clean_box=(3, 20, 4, 30),
            data=np.zeros((48, 52)),
        )
        cases[f"all-zero residual, disk, threshold -1 {t}"] = _case(
            dtype=dtype, niter=5, threshold=-1.0, data=np.zeros((48, 52))
        )
        # Non-finite and extreme values.
        psf = _psf(48, 52, seed=3)
        psf[30, 21] = np.nan
        clean = _dirty(
            48, 52, _psf(48, 52, seed=3), 4
        )  # the dirty image of a finite PSF
        cases[f"NaN in the PSF {t}"] = _case(
            dtype=dtype, niter=80, psf=psf, data=clean, seed=3
        )
        psf = _psf(48, 52, seed=4)
        psf[10, 40] = -np.inf
        cases[f"-inf in the PSF {t}"] = _case(
            dtype=dtype, mask=None, niter=80, psf=psf, data=clean, seed=4
        )
        data = dirty.copy()
        data[24, 27] = -np.inf
        cases[f"-inf in the residual inside S {t}"] = _case(
            dtype=dtype, niter=50, data=data
        )
        data = dirty.copy()
        data[2, 2] = np.inf
        cases[f"+inf in the residual outside S {t}"] = _case(
            dtype=dtype, niter=50, data=data
        )
        data = dirty.copy()
        data[np.random.default_rng(7).random((48, 52)) < 0.02] = np.nan
        cases[f"NaN in and out of S {t}"] = _case(dtype=dtype, niter=60, data=data)
        cases[f"NaN everywhere, threshold -1 {t}"] = _case(
            dtype=dtype,
            niter=4,
            threshold=-1.0,
            divergence=False,
            data=np.full((48, 52), np.nan),
        )
        cases[f"subnormal residual {t}"] = _case(
            dtype=dtype, niter=80, data=dirty / np.abs(dirty).max() * tiny * 50
        )
        cases[f"huge values {t}"] = _case(
            dtype=dtype, niter=60, data=dirty / np.abs(dirty).max() * big
        )
        # Tiny images.
        for ny, nx in ((1, 1), (1, 9), (9, 1), (2, 3), (5, 5)):
            for twice in (False, True):
                name = f"tiny {ny}x{nx}{' 2n PSF' if twice else ''} {t}"
                cases[name] = _case(
                    ny, nx, dtype, mask=None, niter=7, twice=twice, seed=ny * 10 + nx
                )
        # Clean boxes that the binding clamps.
        for box in (
            (-5, 1000, 3, 2),
            (10, 5, 10, 5),
            (70, 80, -1, -1),
            (0, 1, 0, 1),
            (-2, -3, -4, -5),
            (1, 2, 3),
        ):
            cases[f"clean box {box} {t}"] = _case(
                dtype=dtype, niter=30, clean_box=box, seed=11
            )
            cases[f"clean box {box}, threshold -1, single pixel {t}"] = _case(
                dtype=dtype,
                mask="single",
                niter=6,
                threshold=-1.0,
                gain=1.0,
                divergence=False,
                clean_box=box,
                seed=12,
            )
        # Gains, speedups, budgets, thresholds and divergence controls.
        for gain in (0.0, 1.0, 1.9, -0.1, float("nan")):
            cases[f"gain {gain} {t}"] = _case(dtype=dtype, niter=40, gain=gain, seed=13)
        for speedup in (7.0, -3.0):
            cases[f"speedup {speedup} {t}"] = _case(
                dtype=dtype, niter=300, threshold=1e-5, speedup=speedup, seed=15
            )
        for threshold in (np.nan, -np.inf, np.inf):
            cases[f"threshold {threshold} {t}"] = _case(
                dtype=dtype, niter=40, threshold=threshold, seed=17
            )
        for divergence in (
            {
                "max_iter_divergence": 0,
                "divergence_rms_factor": 1.0,
                "divergence_peak_factor": 1.1,
            },
            {
                "max_iter_divergence": 3,
                "divergence_rms_factor": 0.5,
                "divergence_peak_factor": np.inf,
            },
            {
                "max_iter_divergence": 2,
                "divergence_rms_factor": np.inf,
                "divergence_peak_factor": np.nan,
            },
            {
                "max_iter_divergence": -5,
                "divergence_rms_factor": np.nan,
                "divergence_peak_factor": 0.5,
            },
        ):
            cases[f"divergence {divergence} {t}"] = _case(
                dtype=dtype, niter=200, gain=0.3, divergence=divergence, seed=16
            )
        # Ties over masks of short runs: the lanes, spans and runs of the search.
        for mask in ("runs", "checker", "random50", None):
            data = (
                np.random.default_rng(24).integers(-4, 5, (40, 45)).astype(np.float64)
            )
            cases[f"integer ties, mask {mask} {t}"] = _case(
                40,
                45,
                dtype,
                mask=mask,
                niter=150,
                gain=0.5,
                data=data,
                psf=_dyadic_psf(40, 45),
                seed=24,
            )
        # PSF of twice the image, odd and non-square.
        cases[f"2n PSF 37x51 {t}"] = _case(37, 51, dtype, niter=60, twice=True, seed=20)
        cases[f"2n PSF 37x51, no mask, clean box {t}"] = _case(
            37,
            51,
            dtype,
            mask=None,
            niter=60,
            twice=True,
            clean_box=(4, 45, 3, 30),
            seed=21,
        )
        # An accumulated model with -0.0 and NaN entries.
        model = np.zeros((1, 1, 1, 48, 52))
        model[..., ::7, ::5] = 0.25
        model[..., 3, ::3] = -0.0
        cases[f"accumulated model {t}"] = _case(
            dtype=dtype, niter=60, model=model, seed=22
        )
        model = model.copy()
        model[..., 40, 40] = np.nan
        cases[f"model with a NaN {t}"] = _case(
            dtype=dtype, niter=60, model=model, seed=23
        )
    return cases


EDGE_CASES = _edge_cases()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(EDGE_CASES))
def test_edge_cases_are_the_reference(entry, name):
    _check(entry, EDGE_CASES[name], threads=(1, 5))


def test_per_plane_budgets_thresholds_and_masks():
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
            _check(entry, case, threads=(1, 4, 24))


@pytest.mark.parametrize("entry", ENTRIES)
def test_stokes_i_active_and_v_idle(entry):
    """One frequency, I and V: V's threshold stops it at its first search,
    while I runs its budget, on every thread count."""
    case = _case(
        96,
        96,
        np.float64,
        planes=(1, 1, 2),
        niter=400,
        threshold=[1e-5, 1.0],
        mask="disk",
        seed=31,
    )
    case["residual"][0, 0, 1] *= 1e-3
    expected = _check(entry, case, threads=(1, 2, 6, 12))
    assert expected[0]["stop_code"].tolist() == [[[1, 2]]]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_many_small_planes_at_every_thread_count(entry, dtype):
    """12 small planes, each with its own mask, from 1 to 24 threads."""
    case = _case(40, 44, dtype, planes=(1, 3, 4), niter=120, seed=32)
    rng = np.random.default_rng(32)
    case["kwargs"]["mask_cube"] = np.stack(
        [_mask(MASKS[(3 * i) % len(MASKS)], 40, 44, rng) for i in range(12)]
    ).reshape(1, 3, 4, 40, 44)
    _check(entry, case, threads=(1, 2, 3, 4, 5, 6, 8, 12, 16, 24))


@pytest.mark.parametrize("entry", ENTRIES)
def test_thread_counts_below_one_count_as_one(entry):
    case = _case(64, 64, np.float32, planes=(1, 2, 2), niter=80, seed=33)
    _check(entry, case, threads=(0, -1, -7))


def test_the_two_entries_differ_only_in_their_corner_cases():
    """On a plane where no corner case occurs they agree with each other; the
    zero-peak pixel and a negative budget tell them apart."""
    case = _case(niter=60, seed=34)
    a, b = (_run(FAST[entry], case, 3) for entry in ENTRIES)
    _assert_same(a, b)
    case = _case(
        mask="empty",
        niter=4,
        threshold=-1.0,
        divergence=False,
        clean_box=(7, 30, 9, 33),
    )
    case["residual"][0, 0, 0, 0, 0] = 1.0  # clean_cube's zero-peak pixel
    a, b = (_run(FAST[entry], case, 2)[0] for entry in ENTRIES)
    assert not np.array_equal(a["total_flux_cleaned"], b["total_flux_cleaned"])
    case = _case(niter=-3)
    a, b = (_run(FAST[entry], case)[0] for entry in ENTRIES)
    assert a["iterations_performed"].tolist() == [[[-3]]]
    assert b["iterations_performed"].tolist() == [[[0]]]


# ---------------------------------------------------------------- invariance
ISA_RUN = """
import sys
import numpy as np
from astroviper._kernels import impl
from astroviper._kernels.fast.hogbom import ISA
for source, target in zip(sys.argv[1::2], sys.argv[2::2]):
    arrays = dict(np.load(source))
    residual, model = arrays.pop("residual"), arrays.pop("model")
    entry = str(arrays.pop("entry"))
    arrays["clean_box"] = tuple(int(v) for v in arrays["clean_box"])
    arrays["gain"] = float(arrays["gain"])
    arrays["record_iteration_history"] = bool(arrays["record_iteration_history"])
    for key in ("max_iter_divergence",):
        arrays[key] = int(arrays[key])
    for key in ("divergence_rms_factor", "divergence_peak_factor"):
        arrays[key] = float(arrays[key])
    result = impl("hogbom", entry, "fast")(
        residual_cube=residual, model_cube=model, processing_function_threads=3, **arrays
    )
    history = [np.asarray(c) for k in sorted(result) if k.startswith("iteration_")
               for a in result[k] for b in a for c in b]
    np.savez(target, residual=residual, model=model,
             **{k: np.asarray(v) for k, v in result.items() if not k.startswith("iteration_")},
             **{f"h{i}": h for i, h in enumerate(history)})
print(ISA)
"""


def test_generic_and_avx2_routines_give_the_same_bits(run_python, tmp_path):
    cases = {
        "disk": (
            "clean_cube_many_threads",
            _case(64, 70, np.float64, planes=(1, 1, 2), niter=150, seed=40),
        ),
        "runs32": (
            "clean_cube",
            _case(61, 67, np.float32, mask="runs", niter=150, seed=41),
        ),
        "ties": (
            "clean_cube",
            _case(
                40,
                45,
                np.float64,
                mask="checker",
                niter=150,
                gain=0.5,
                psf=_dyadic_psf(40, 45),
                data=np.random.default_rng(42)
                .integers(-4, 5, (40, 45))
                .astype(np.float64),
            ),
        ),
        "twice": (
            "clean_cube_many_threads",
            _case(37, 51, np.float32, mask=None, niter=80, twice=True, seed=43),
        ),
    }
    for name, (entry, case) in cases.items():
        np.savez(
            tmp_path / f"{name}.npz",
            residual=case["residual"],
            model=case["model"],
            psf_cube=case["psf"],
            entry=entry,
            **case["kwargs"],
        )
    results = {}
    for isa, env in (
        ("default", {}),
        ("generic", {"ASTROVIPER_KERNELS_ISA": "generic"}),
    ):
        files = []
        for name in cases:
            files += [
                str(tmp_path / f"{name}.npz"),
                str(tmp_path / f"{name}-{isa}.npz"),
            ]
        out = run_python("-c", ISA_RUN, *files, **env)
        results[out.strip().splitlines()[-1]] = [
            {
                k: _bits(v).tobytes()
                for k, v in np.load(tmp_path / f"{name}-{isa}.npz").items()
            }
            for name in cases
        ]
    if "avx2" not in results:
        pytest.skip("no AVX2 routines on this machine or build")
    assert set(results) == {"avx2", "generic"}
    assert results["avx2"] == results["generic"]


def test_an_unknown_isa_override_fails_the_import(run_python):
    done = run_python(
        "-c",
        "import astroviper._kernels.fast.hogbom",
        check=False,
        ASTROVIPER_KERNELS_ISA="avx512",
    )
    assert done.returncode != 0
    assert "ASTROVIPER_KERNELS_ISA=avx512: expected 'generic'" in done.stderr


def test_concurrent_calls_give_the_bits_of_serial_calls():
    """Python threads calling at once, each call with threads of its own."""
    cases = [
        _case(48, 52, np.float64, planes=(1, 1, 2), niter=100, seed=50),
        _case(40, 44, np.float32, seed=51),
    ]
    expected = [_run(FAST["clean_cube_many_threads"], case, 3) for case in cases]

    def same(i):
        _assert_same(
            expected[i % 2], _run(FAST["clean_cube_many_threads"], cases[i % 2], 3)
        )
        return True

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert all(pool.map(same, range(24)))


# ---------------------------------------------------------------- paths
def _thread_count():
    return len(os.listdir("/proc/self/task"))


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="needs /proc")
def test_no_thread_outlives_a_call_and_the_gil_is_released():
    """A sampler thread runs during the call (the GIL is released) and sees
    the call's threads; when the call returns, they have all ended."""
    case = _case(
        256,
        256,
        np.float64,
        planes=(1, 1, 2),
        niter=3000,
        threshold=0.0,
        mask="all",
        seed=52,
    )
    before = _thread_count()
    seen, done = [], threading.Event()

    def sample():
        while not done.is_set():
            seen.append(_thread_count())
            time.sleep(0.0005)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(30.0)
    sampler = threading.Thread(target=sample)
    try:
        sampler.start()
        _run(FAST["clean_cube_many_threads"], case, 4)
        done.set()
        sampler.join()
    finally:
        sys.setswitchinterval(interval)
    assert max(seen) > before + 1, "no sample saw the call's threads"
    assert _thread_count() == before


def _cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def test_idle_threads_sleep_and_nothing_runs_after_a_call():
    """12 small planes and 12 threads: the planes that stop at once leave
    their crews idle while one plane runs 20000 iterations (no exterior
    pixels, so no final subtractions). The idle threads poll for a bounded
    time and then sleep: the call's CPU time stays near that of the one busy
    crew (a few threads), far from 12 threads polling throughout. After the
    call returns, the process burns no CPU."""
    case = _case(128, 128, np.float64, planes=(1, 3, 4), mask=None, niter=20000)
    case["kwargs"]["threshold_per_cycle"] = np.full((1, 3, 4), 1e6)
    case["kwargs"]["threshold_per_cycle"][0, 0, 0] = 0.0
    case["kwargs"] |= _divergence(0.1, on=False)
    cpu, start = _cpu_seconds(), time.perf_counter()
    result, _, _ = _run(FAST["clean_cube"], case, 12)
    wall, cpu = time.perf_counter() - start, _cpu_seconds() - cpu
    assert result["iterations_performed"][0, 0, 0] == 20000
    assert result["iterations_performed"].sum() == 20000
    assert cpu < 4.0 * wall + 0.02, (cpu, wall)
    cpu = _cpu_seconds()
    time.sleep(0.2)
    assert _cpu_seconds() - cpu < 0.02


def _outcome(function, make):
    """Every array argument's bytes after a call, and its result or exception."""
    args, kwargs = make()
    try:
        result = function(*args, **kwargs)
        error = None
    except Exception as e:  # compared with the reference's exception
        result, error = None, (type(e), str(e))
    arrays = [
        a for a in list(args) + list(kwargs.values()) if isinstance(a, np.ndarray)
    ]
    return result, error, [_bits(a.reshape(-1)).tobytes() for a in arrays]


def _arguments(dtype=np.float64, **changes):
    case = _case(16, 18, dtype, planes=(1, 1, 2), niter=10, seed=60)
    kwargs = {
        "residual_cube": case["residual"],
        "psf_cube": case["psf"],
        "model_cube": case["model"],
    }
    kwargs |= case["kwargs"] | {"processing_function_threads": 2}
    kwargs |= changes
    return kwargs


def _declined_calls():
    def keywords(**changes):
        return lambda: (
            (),
            {k: v for k, v in _arguments(**changes).items() if v is not _ABSENT},
        )

    def positional():
        kwargs = _arguments()
        return tuple(kwargs.values()), {}

    def twice_given():
        kwargs = _arguments()
        return (kwargs.pop("residual_cube"),), kwargs | {
            "residual_cube": kwargs["model_cube"]
        }

    def too_many():
        return tuple(_arguments().values()) + (1,), {}

    base = _arguments()
    shape = base["residual_cube"].shape
    return {
        "no model": keywords(model_cube=_ABSENT),
        "unknown keyword": keywords(niter=3),
        "given twice": twice_given,
        "too many positionals": too_many,
        "int32 residual": keywords(
            residual_cube=np.zeros(shape, np.int32),
            psf_cube=np.zeros(shape, np.int32),
            model_cube=np.zeros(shape, np.int32),
        ),
        "float32 psf, float64 residual": keywords(
            psf_cube=base["psf_cube"].astype(np.float32)
        ),
        "model of another shape": keywords(model_cube=np.zeros((1, 1, 2, 16, 17))),
        "4-D residual": keywords(residual_cube=base["residual_cube"][0].copy()),
        "strided residual": keywords(
            residual_cube=np.zeros((1, 1, 2, 32, 18))[..., ::2, :]
        ),
        "read-only residual": keywords(residual_cube=_read_only(base["residual_cube"])),
        "read-only model": keywords(model_cube=_read_only(base["model_cube"])),
        "PSF of another size": keywords(psf_cube=np.zeros((1, 1, 2, 17, 18))),
        "PSF with 3 polarizations": keywords(psf_cube=np.zeros((1, 1, 3, 16, 18))),
        "uint8 mask": keywords(mask_cube=np.ones(shape, np.uint8)),
        "mask of another shape": keywords(mask_cube=np.ones((1, 1, 2, 16, 17), bool)),
        "no budgets": keywords(max_iter_per_cycle_updated=_ABSENT),
        "int64 budgets": keywords(
            max_iter_per_cycle_updated=np.full((1, 1, 2), 5, np.int64)
        ),
        "budgets of another shape": keywords(
            max_iter_per_cycle_updated=np.full((1, 2, 1), 5, np.int32)
        ),
        "float32 thresholds": keywords(
            threshold_per_cycle=np.zeros((1, 1, 2), np.float32)
        ),
        "list clean box": keywords(clean_box=[-1, -1, -1, -1]),
        "float in the clean box": keywords(clean_box=(-1, 2.5, -1, -1)),
        "str gain": keywords(gain="0.1"),
        "float thread count": keywords(processing_function_threads=2.5),
        "thread count of 2**40": keywords(processing_function_threads=2**40),
        "list residual": keywords(residual_cube=base["residual_cube"].tolist()),
        "list budgets": keywords(max_iter_per_cycle_updated=[[[5, 5]]]),
        "model is the residual": lambda: ((), _sharing("model_cube")),
        "PSF is the residual": lambda: ((), _sharing("psf_cube")),
        "thresholds in the residual": lambda: ((), _sharing("threshold_per_cycle")),
    }


_ABSENT = object()


def _read_only(array):
    array = array.copy()
    array.flags.writeable = False
    return array


def _sharing(name):
    """The arguments with `name` a view of the residual's memory."""
    kwargs = _arguments()
    residual = kwargs["residual_cube"]
    if name == "model_cube":
        kwargs["model_cube"] = residual
    elif name == "psf_cube":
        kwargs["psf_cube"] = residual
    else:
        kwargs[name] = residual.reshape(-1)[:2].reshape(1, 1, 2)
    return kwargs


DECLINED = _declined_calls()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(DECLINED))
def test_declined_calls_are_the_reference(entry, name):
    make = DECLINED[name]
    args, kwargs = make()
    assert getattr(ext, entry)(*args, **kwargs) is None
    fast_result, fast_error, fast_arrays = _outcome(FAST[entry], make)
    result, error, arrays = _outcome(REFERENCE[entry], make)
    assert fast_error == error
    assert fast_arrays == arrays
    if error is None:
        assert list(fast_result) == list(result)
        for key in SCALARS:
            assert np.array_equal(
                _bits(np.asarray(fast_result[key])), _bits(np.asarray(result[key]))
            )


@pytest.mark.parametrize("entry", ENTRIES)
def test_planes_without_pixels_decline(entry):
    """The reference reads and writes such planes out of bounds (it can
    crash), so the fast entry leaves them to it, and nothing is called here."""
    empty = np.zeros((1, 1, 2, 0, 4))
    arguments = _arguments(
        residual_cube=empty, psf_cube=empty.copy(), model_cube=empty.copy()
    )
    arguments["mask_cube"] = np.array([])
    assert getattr(ext, entry)(**arguments) is None


@pytest.mark.parametrize(
    "changes",
    [
        {
            "clean_box": (2, 10, 3)
        },  # not 4 items: the full extent, as the reference reads it
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
    ],
)
def test_converted_arguments_take_the_fast_path(changes):
    kwargs = _arguments(**changes)
    for entry in ENTRIES:
        assert (
            getattr(ext, entry)(
                **{
                    k: np.copy(v) if isinstance(v, np.ndarray) else v
                    for k, v in kwargs.items()
                }
            )
            is not None
        )
        fast = _outcome(FAST[entry], lambda: ((), _arguments(**changes)))
        reference = _outcome(REFERENCE[entry], lambda: ((), _arguments(**changes)))
        assert fast[1] is None and reference[1] is None
        assert fast[2] == reference[2]


def _misaligned(array):
    buffer = np.zeros(array.nbytes + 1, np.uint8)
    copy = buffer[1:].view(array.dtype).reshape(array.shape)
    copy[...] = array
    assert not copy.flags.aligned
    return copy


@pytest.mark.parametrize("entry", ENTRIES)
def test_misaligned_arrays_take_the_fast_path_through_aligned_copies(entry):
    """Every array misaligned: the bits of the aligned call (the reference is
    not called: it reads a misaligned array in place, which is undefined
    behaviour)."""
    case = _case(48, 52, np.float64, planes=(1, 1, 2), niter=80, seed=61)
    expected = _run(FAST[entry], case, 3)
    residual, model = _misaligned(case["residual"]), _misaligned(case["model"])
    kwargs = case["kwargs"] | {
        "max_iter_per_cycle_updated": _misaligned(
            case["kwargs"]["max_iter_per_cycle_updated"]
        ),
        "threshold_per_cycle": _misaligned(case["kwargs"]["threshold_per_cycle"]),
        "processing_function_threads": 3,
    }
    result = getattr(ext, entry)(
        residual_cube=residual,
        psf_cube=_misaligned(case["psf"]),
        model_cube=model,
        **kwargs,
    )
    assert result is not None
    _assert_same(expected, (result, residual, model))


TRACED = """
import numpy as np
from astroviper.processing_functions.imaging.deconvolvers import hogbom

residual = np.zeros((1, 1, 1, 16, 16))
residual[..., 8, 8] = 1.0
arguments = dict(max_iter_per_cycle_updated=np.full((1, 1, 1), 5, np.int32), gain=0.5,
                 threshold_per_cycle=np.zeros((1, 1, 1)))
hogbom.clean_cube(residual.copy(), residual.copy(), np.zeros_like(residual), **arguments)
shared = residual.copy()
hogbom.clean_cube_many_threads(shared, residual.copy(), shared, **arguments)
try:
    hogbom.clean_cube(residual.astype(np.float32), residual, np.zeros_like(residual), **arguments)
except RuntimeError:
    pass
"""


def test_the_trace_names_each_calls_path(run_python, tmp_path):
    trace = tmp_path / "trace"
    run_python(
        "-c",
        TRACED,
        ASTROVIPER_KERNELS="hogbom=fast",
        ASTROVIPER_KERNELS_TRACE=str(trace),
    )
    (path,) = trace.glob(f"kernels-{socket.gethostname()}-*.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["entry"], r["impl"], r["path"], r["error"]) for r in records] == [
        ("clean_cube", "fast", "fast", None),
        ("clean_cube_many_threads", "fast", "declined", None),
        ("clean_cube", "fast", "declined", "RuntimeError"),
    ]
