"""Tests of the fast Hogbom minor cycle (``ASTROVIPER_KERNELS=hogbom=fast``).

The implementation is called directly (``astroviper._kernels.impl``), so these
tests run in any session. Small inputs only. The oracle is the reference
binding, and every comparison is bitwise on unsigned-integer views (-0.0
counts) of everything a call returns or changes: the residual and model
cubes, the iterations, stop codes, converged and diverged flags, final_peak,
total_flux_cleaned and the six iteration-history records; the arguments a
call only reads must be unchanged after it. The contract leaves the sign and
payload of a NaN result unspecified (in the reference they depend on the
compiler's operand order): the NaN-payload test compares NaNs as NaNs, and
every other test feeds only ``np.nan``, whose bits both kernels keep, and so
compares NaN bits too.

* Each fast entry against its own reference entry (they differ in corner
  cases: a zero peak that a negative threshold keeps running, negative
  budgets), on seeded random cubes and on edge cases: masks (none, all-true,
  boxes, a disk, a ring, short runs, checkerboards, stripes, empty, a single
  pixel, bytes other than 0 and 1), NaN and infinities inside and outside the
  searched pixels, NaNs of other payloads, ties, -0.0, subnormal and huge
  values, float32 and float64, PSFs of the image's size and of twice it,
  clean boxes that the binding clamps, edge gains, thresholds, speedups,
  budgets and divergence controls, accumulated models, rows on either side
  of the short-row limit, more iterations than a plane defers.
* Both ways of cleaning a plane, by its shape (hclean_fast.hpp): small planes
  (at most 8,192 pixels, or at most 64 columns and 65,536 pixels) by the
  direct pass, row by row or column by column, larger ones split into
  searched and deferred pixels. The random cubes and the edge cases run at
  the sizes of each; planes and boxes either side of the limits, narrow
  planes, sparse masks and rows of several chunks run too.
* Multi-plane cubes: per-plane budgets, thresholds and masks, a broadcast PSF,
  Stokes I active with V idle, many small planes at 1 to 24 threads, and the
  production paths: crews of several threads on a plane, side by side or one
  after another, rows of several chunks, zero peaks flushed by a crew.
* Invariance: bitwise across thread counts (0, -1 and 1 to 24) and between the
  generic and AVX2 routines (two subprocesses); the bitwise tests of this
  file pass with the generic routines too (a subprocess). The AVX2 unit
  defines no weak symbol at -O0 either (an empty CMAKE_BUILD_TYPE).
* Declined calls are the reference's, bit for bit and exception for
  exception; arrays the call writes that share memory with another argument
  decline; misaligned arrays take the fast path through aligned copies.
* Resources: no thread outlives a call, small planes start no more threads
  than planes with a budget (none when they stop at their first search)
  while a small box on a large plane gets the team for its final passes,
  idle threads sleep (side by side and in a team), the GIL is released
  during a call, threads that cannot be started leave a smaller team with
  the same bits, memory does not grow with the iterations, and an error
  after the first write leaves a misaligned cube as cleaned as an aligned
  one. Without the extension the package raises ModuleNotFoundError, so that
  this module skips.
* The trace of the hooked shim names each call's path.
"""

import json
import os
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def _random_case(seed, large=False):
    """A seeded random cube, as the prototype's verification draws them; with
    `large`, of planes that are split (more than 8,192 pixels and 64 columns)
    or narrow (at most 48 columns, of up to 400 rows: the direct pass)."""
    rng = np.random.default_rng(seed)
    dtype = np.float64 if rng.random() < 0.6 else np.float32
    if large:
        if rng.random() < 0.3:
            ny, nx = int(rng.integers(150, 401)), int(rng.integers(1, 49))
        else:
            ny, nx = int(rng.integers(91, 141)), int(rng.integers(91, 141))
    elif rng.random() < 0.1:
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
    """A call on copies of the residual and model; the arguments it only reads
    must be unchanged after it."""
    residual, model = case["residual"].copy(), case["model"].copy()
    kwargs = case["kwargs"] | {"processing_function_threads": threads} | changes
    read = [case["psf"]] + [v for v in kwargs.values() if isinstance(v, np.ndarray)]
    before = [_bits(a).tobytes() for a in read]
    result = function(
        residual_cube=residual, psf_cube=case["psf"], model_cube=model, **kwargs
    )
    assert [_bits(a).tobytes() for a in read] == before, "a read-only argument changed"
    return result, residual, model


def _canonical(array):
    """The array with every NaN the dtype's np.nan (its sign and payload are
    not part of the contract)."""
    array = np.asarray(array)
    if array.dtype.kind != "f":
        return array
    return np.where(np.isnan(array), np.array(np.nan, array.dtype), array)


def _assert_same(expected, actual, nan_bits=True, skip=()):
    """Bitwise: the results (keys in order, dtypes, shapes, bits) and the cubes.
    With nan_bits=False a NaN only has to be a NaN; `skip` names results not
    compared."""
    (result, residual, model), (fast_result, fast_residual, fast_model) = (
        expected,
        actual,
    )
    view = _bits if nan_bits else (lambda a: _bits(_canonical(a)))
    assert list(fast_result) == list(result)
    for key in SCALARS:
        a, b = np.asarray(result[key]), np.asarray(fast_result[key])
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert np.array_equal(view(a), view(b)), key
    for key in HISTORY:
        if key not in result or key in skip:
            continue
        for t, f, p in np.ndindex(np.asarray(result["stop_code"]).shape):
            a, b = result[key][t][f][p], fast_result[key][t][f][p]
            assert a.dtype == b.dtype and a.shape == b.shape, (key, t, f, p)
            assert np.array_equal(view(a), view(b)), (key, t, f, p)
    assert np.array_equal(view(fast_residual), view(residual)), "residual"
    assert np.array_equal(view(fast_model), view(model)), "model"


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


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("seed", range(20))
def test_random_large_cubes_are_the_reference(entry, seed):
    """Random cubes of split planes and of narrow ones."""
    case, threads = _random_case(1000 + seed, large=True)
    _check(entry, case, threads)


def _edge_cases(ny=48, nx=52):
    """The edge cases on planes of ny x nx pixels (cleaned by the direct pass, or
    split when larger than 8,192 pixels; the tiny images only at the first),
    the ties and the odd PSF of twice the image on planes of their own sizes."""
    split = ny * nx > 8192

    def sized(**kwargs):
        return _case(ny, nx, **kwargs)

    cases = {}
    for dtype in (np.float64, np.float32):
        t = dtype.__name__
        tiny = 1e-310 if dtype == np.float64 else 1e-41
        big = (
            1e160 if dtype == np.float64 else 1e30
        )  # squares overflow the double sum / T
        dirty = _dirty(ny, nx, _psf(ny, nx, seed=5), 6)
        # A zero peak that a negative threshold keeps running: clean_cube takes
        # pixel (0, 0), clean_cube_many_threads the first of the clean box,
        # either of which may be off the searched pixels.
        cases[f"empty mask, threshold -1 {t}"] = sized(
            dtype=dtype, mask="empty", niter=6, threshold=-1.0, divergence=False
        )
        cases[f"empty mask, threshold -1, clean box {t}"] = sized(
            dtype=dtype,
            mask="empty",
            niter=6,
            threshold=-1.0,
            divergence=False,
            clean_box=(7, 30, 9, 33),
        )
        outside = np.zeros((ny, nx), bool)
        outside[:5] = True
        cases[f"mask only outside the clean box, threshold -1 {t}"] = sized(
            dtype=dtype,
            mask=outside,
            niter=6,
            threshold=-1.0,
            divergence=False,
            clean_box=(5, 35, 10, 40),
        )
        cases[f"all-zero residual, threshold -1, clean box {t}"] = sized(
            dtype=dtype,
            mask=None,
            niter=5,
            threshold=-1.0,
            divergence=False,
            clean_box=(3, 20, 4, 30),
            data=np.zeros((ny, nx)),
        )
        cases[f"all-zero residual, disk, threshold -1 {t}"] = sized(
            dtype=dtype, niter=5, threshold=-1.0, data=np.zeros((ny, nx))
        )
        # Non-finite and extreme values.
        psf = _psf(ny, nx, seed=3)
        psf[30, 21] = np.nan
        clean = _dirty(
            ny, nx, _psf(ny, nx, seed=3), 4
        )  # the dirty image of a finite PSF
        cases[f"NaN in the PSF {t}"] = sized(
            dtype=dtype, niter=80, psf=psf, data=clean, seed=3
        )
        psf = _psf(ny, nx, seed=4)
        psf[10, 40] = -np.inf
        cases[f"-inf in the PSF {t}"] = sized(
            dtype=dtype, mask=None, niter=80, psf=psf, data=clean, seed=4
        )
        data = dirty.copy()
        data[24, 27] = -np.inf
        cases[f"-inf in the residual inside S {t}"] = sized(
            dtype=dtype, niter=50, data=data
        )
        data = dirty.copy()
        data[2, 2] = np.inf
        cases[f"+inf in the residual outside S {t}"] = sized(
            dtype=dtype, niter=50, data=data
        )
        data = dirty.copy()
        data[np.random.default_rng(7).random((ny, nx)) < 0.02] = np.nan
        cases[f"NaN in and out of S {t}"] = sized(dtype=dtype, niter=60, data=data)
        cases[f"NaN everywhere, threshold -1 {t}"] = sized(
            dtype=dtype,
            niter=4,
            threshold=-1.0,
            divergence=False,
            data=np.full((ny, nx), np.nan),
        )
        cases[f"subnormal residual {t}"] = sized(
            dtype=dtype, niter=80, data=dirty / np.abs(dirty).max() * tiny * 50
        )
        cases[f"huge values {t}"] = sized(
            dtype=dtype, niter=60, data=dirty / np.abs(dirty).max() * big
        )
        # Tiny images.
        for ty, tx in ((1, 1), (1, 9), (9, 1), (2, 3), (5, 5)) if not split else ():
            for twice in (False, True):
                name = f"tiny {ty}x{tx}{' 2n PSF' if twice else ''} {t}"
                cases[name] = _case(
                    ty, tx, dtype, mask=None, niter=7, twice=twice, seed=ty * 10 + tx
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
            cases[f"clean box {box} {t}"] = sized(
                dtype=dtype, niter=30, clean_box=box, seed=11
            )
            cases[f"clean box {box}, threshold -1, single pixel {t}"] = sized(
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
            cases[f"gain {gain} {t}"] = sized(dtype=dtype, niter=40, gain=gain, seed=13)
        for speedup in (7.0, -3.0):
            cases[f"speedup {speedup} {t}"] = sized(
                dtype=dtype, niter=300, threshold=1e-5, speedup=speedup, seed=15
            )
        for threshold in (np.nan, -np.inf, np.inf):
            cases[f"threshold {threshold} {t}"] = sized(
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
            cases[f"divergence {divergence} {t}"] = sized(
                dtype=dtype, niter=200, gain=0.3, divergence=divergence, seed=16
            )
        # Ties over masks of short runs: the lanes, spans and runs of the search.
        tie = (ny, nx) if split else (40, 45)
        for mask in ("runs", "checker", "random50", None):
            data = np.random.default_rng(24).integers(-4, 5, tie).astype(np.float64)
            cases[f"integer ties, mask {mask} {t}"] = _case(
                *tie,
                dtype,
                mask=mask,
                niter=150,
                gain=0.5,
                data=data,
                psf=_dyadic_psf(*tie),
                seed=24,
            )
        # PSF of twice the image, odd and non-square.
        odd = (91, 101) if split else (37, 51)
        cases[f"2n PSF odd {t}"] = _case(*odd, dtype, niter=60, twice=True, seed=20)
        cases[f"2n PSF odd, no mask, clean box {t}"] = _case(
            *odd,
            dtype,
            mask=None,
            niter=60,
            twice=True,
            clean_box=(4, 45, 3, 30),
            seed=21,
        )
        # An accumulated model with -0.0 and NaN entries.
        model = np.zeros((1, 1, 1, ny, nx))
        model[..., ::7, ::5] = 0.25
        model[..., 3, ::3] = -0.0
        cases[f"accumulated model {t}"] = sized(
            dtype=dtype, niter=60, model=model, seed=22
        )
        model = model.copy()
        model[..., 40, 40] = np.nan
        cases[f"model with a NaN {t}"] = sized(
            dtype=dtype, niter=60, model=model, seed=23
        )
    return cases


EDGE_CASES = _edge_cases()
SPLIT_EDGE_CASES = _edge_cases(96, 104)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(EDGE_CASES))
def test_edge_cases_are_the_reference(entry, name):
    _check(entry, EDGE_CASES[name], threads=(1, 5))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(SPLIT_EDGE_CASES))
def test_split_edge_cases_are_the_reference(entry, name):
    """The edge cases on planes of 96 x 104 pixels, which are split."""
    _check(entry, SPLIT_EDGE_CASES[name], threads=(1, 5))


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


#: +NaN, -NaN and NaN of another payload, as unsigned bits.
NAN_PAYLOADS = {
    np.float32: (0x7FC00000, 0xFFC00000, 0x7FC00123),
    np.float64: (0x7FF8000000000000, 0xFFF8000000000000, 0x7FF8000000000123),
}


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("gain", [0.1, float("nan")])
def test_nan_payloads_are_nan_where_the_reference_has_nan(entry, dtype, gain):
    """NaNs of three payloads in one searched row and one in the PSF, with the
    history: every result that is not a NaN has the reference's bits, and every
    NaN result is a NaN where the reference has one. Which NaN an operation on
    two NaNs keeps is the compiler's operand order (in the reference it differs
    between its scan, its last sum and its two entries), so the sign and
    payload of a NaN are not part of the contract."""
    case = _case(
        48, 52, dtype, planes=(1, 1, 2), mask=None, niter=40, gain=gain, seed=70
    )
    for plane in range(2):
        _bits(case["residual"])[0, 0, plane, 20, [7, 20, 33]] = NAN_PAYLOADS[dtype]
        _bits(case["psf"])[0, 0, plane, 30, 21] = NAN_PAYLOADS[dtype][2]
    expected = _run(REFERENCE[entry], case)
    assert np.isnan(expected[1]).any()
    for threads in (1, 3):
        _assert_same(expected, _run(FAST[entry], case, threads), nan_bits=False)


@pytest.mark.parametrize("entry", ENTRIES)
def test_mask_bytes_other_than_one_change_only_the_history_rms(entry):
    """A mask of the bytes 0, 1, 2, 128 and 255 seen as bool (NumPy never makes
    one): both kernels search the pixels whose byte is not 0, so every result
    is the reference's but the history RMS, which the reference divides by the
    sum of the raw bytes (a bool that is not 0 or 1 is undefined behaviour);
    the fast kernel's is that of the same mask in 0 and 1."""
    rng = np.random.default_rng(72)
    raw = rng.choice(np.array([0, 1, 2, 128, 255], np.uint8), size=(1, 1, 2, 40, 44))
    case = _case(40, 44, np.float64, planes=(1, 1, 2), niter=60, seed=72)
    case["kwargs"]["mask_cube"] = raw.view(bool)
    plain = case | {"kwargs": case["kwargs"] | {"mask_cube": raw != 0}}
    for threads in (1, 2):
        fast = _run(FAST[entry], case, threads)
        _assert_same(_run(FAST[entry], plain, threads), fast)
        _assert_same(
            _run(REFERENCE[entry], case), fast, skip=("iteration_residual_rms",)
        )
        quiet = {"record_iteration_history": False}
        _assert_same(
            _run(REFERENCE[entry], case, **quiet),
            _run(FAST[entry], case, threads, **quiet),
        )


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("nx", [1, 7, 8, 9, 16, 17, 33, 47, 48, 49, 50, 64, 65])
@pytest.mark.parametrize("split", [False, True])
def test_rows_either_side_of_the_short_row_limit_are_the_reference(
    entry, dtype, nx, split
):
    """Rows of up to 48 pixels are scanned pixel by pixel, longer ones with
    lanes: widths on either side of that limit and of the lane counts, masks
    that split the rows, and a clean box that clips them; on planes of 24 x nx
    (the direct pass), and as a clean box nx columns wide on a split plane of
    24 x 400."""
    width, left = (400, 100) if split else (nx, 0)
    box = (left, left + nx, -1, -1) if split else (-1, -1, -1, -1)
    for mask in (None, "runs", "disk", "checker", "random50"):
        case = _case(24, width, dtype, mask=mask, niter=60, clean_box=box, seed=73 + nx)
        _check(entry, case, threads=(1,))
    clipped = _case(
        24,
        width,
        dtype,
        mask=None,
        niter=60,
        clean_box=(left + 1, left + nx - 1, 2, 20),
        seed=74,
    )
    _check(entry, clipped, threads=(1,))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("nx", [255, 256, 257, 300])
@pytest.mark.parametrize("ny", [12, 36])
def test_exterior_runs_either_side_of_the_block_limit_are_the_reference(
    entry, dtype, nx, ny
):
    """On split planes (36 rows), the deferred subtractions go over a run of at
    least 256 pixels outside the searched ones in register blocks, and over
    shorter runs and the tails a chunk at a time: whole rows of such widths,
    the runs beside a clean box (from column 0 and from further in), a disk
    mask, under footprints that the plane clips and under footprints that cover
    it (a PSF of twice the image). On planes of 12 rows, the direct pass
    subtracts and scans the same rows in two chunks or one."""
    middle = (nx // 2 - 4, nx // 2 + 4, 4, 8)
    for clean_box, mask, twice in (
        (middle, None, False),
        (middle, None, True),
        ((2, 10, 4, 8), None, False),
        ((-1, -1, -1, -1), "disk", False),
    ):
        case = _case(
            ny,
            nx,
            dtype,
            planes=(1, 1, 2),
            mask=mask,
            niter=60,
            clean_box=clean_box,
            twice=twice,
            seed=75 + nx,
        )
        _check(entry, case, threads=(1, 2))


def _production_cases():
    """The paths of production-sized planes, at 1, 6 and 12 threads (E, from
    the sizing rules of hclean_fast.cpp): crews of 3 side by side (4 planes of
    200 x 300), one crew of 6 over two planes (320 x 320, as for HL Tau) and
    two crews of 5 or 6, rows of two 256-pixel chunks, a lone iterating plane
    whose crew has the free threads (Stokes V, or Q, U and V, stopping at
    once), planes without a searched pixel whose zero peaks flush the deferred
    subtractions before the final passes of the whole team, and rows of three
    chunks."""
    cases = {}
    for dtype in (np.float64, np.float32):
        t = dtype.__name__
        for mask in (None, "runs", "disk"):
            cases[f"4 planes 200x300, mask {mask} {t}"] = _case(
                200, 300, dtype, planes=(1, 1, 4), mask=mask, niter=40, seed=80
            )
            cases[f"2 planes 320x320, mask {mask} {t}"] = _case(
                320, 320, dtype, planes=(1, 1, 2), mask=mask, niter=40, seed=81
            )
        cases[f"2 planes 256x256, V stops at once {t}"] = _case(
            256, 256, dtype, planes=(1, 1, 2), niter=60, threshold=[1e-5, 1e6], seed=82
        )
        cases[f"4 planes 200x300, Q, U and V stop at once {t}"] = _case(
            200,
            300,
            dtype,
            planes=(1, 1, 4),
            niter=60,
            threshold=[1e-5, 1e6, 1e6, 1e6],
            seed=83,
        )
        cases[f"2 planes 200x300, zero peaks {t}"] = _case(
            200,
            300,
            dtype,
            planes=(1, 1, 2),
            mask="empty",
            niter=8,
            threshold=-1.0,
            divergence=False,
            seed=84,
        )
        cases[f"12 rows of 700 {t}"] = _case(
            12, 700, dtype, mask="runs", niter=60, seed=85
        )
    return cases


PRODUCTION = _production_cases()


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("name", list(PRODUCTION))
def test_production_paths_are_the_reference(entry, name):
    _check(entry, PRODUCTION[name], threads=(1, 6, 12))


@pytest.mark.parametrize("entry", ENTRIES)
def test_more_subtractions_than_a_plane_defers_are_the_reference(entry):
    """More iterations than a plane defers subtractions from the pixels outside
    the searched ones (16,384): it applies them, in order, whenever that many
    are pending, so that its memory does not grow with the iterations. Split
    planes side by side (16,500 iterations), and a plane cleaned by a crew of
    two (17,000 iterations on a 100 x 100 mask of a 192 x 192 plane); and the
    direct pass, which defers none, on small planes (20,000 iterations)."""
    for ny, nx, niter in ((24, 28, 20000), (92, 92, 16500)):
        small = _case(
            ny,
            nx,
            np.float64,
            planes=(1, 1, 2),
            mask="disk",
            niter=niter,
            threshold=-1.0,
            divergence=False,
            seed=86,
        )
        _check(entry, small, threads=(1, 2))
    square = np.zeros((192, 192), bool)
    square[40:140, 50:150] = True
    crew = _case(
        192,
        192,
        np.float64,
        mask=square,
        niter=17000,
        threshold=-1.0,
        divergence=False,
        history=False,
        seed=87,
    )
    _check(entry, crew, threads=(1, 6))


#: Planes (rows, columns) either side of the limits of the direct pass: at most
#: 8,192 pixels, or at most 64 columns and 65,536 pixels (hclean_fast.cpp).
LIMITS = (
    (64, 128),  # 8,192 pixels: the direct pass
    (91, 91),  # 8,281 pixels: split
    (1, 8192),  # the direct pass
    (1, 8193),  # split
    (2731, 3),  # 8,193 pixels in 3 columns: the direct pass
    (1365, 48),  # 65,520 pixels in 48 columns: the direct pass
    (1366, 48),  # 65,568 pixels: split
    (1024, 64),  # 65,536 pixels in 64 columns: the direct pass
    (1025, 64),  # 65,600 pixels: split
    (200, 65),  # 13,000 pixels in 65 columns: split
)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("shape", LIMITS, ids=lambda shape: "x".join(map(str, shape)))
def test_planes_either_side_of_the_split_limits_are_the_reference(entry, shape):
    """The direct pass and the split on planes either side of the limits
    between them, without a mask, with a random one and with a clean box."""
    ny, nx = shape
    quarter = (nx // 4, nx - nx // 4, ny // 4, ny - ny // 4 + 1)
    for mask, clean_box in ((None, None), ("random50", None), (None, quarter)):
        case = _case(
            ny,
            nx,
            np.float64,
            planes=(1, 1, 2),
            mask=mask,
            niter=40,
            clean_box=clean_box or (-1, -1, -1, -1),
            seed=88,
        )
        _check(entry, case, threads=(1, 4))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_boxes_either_side_of_the_column_limits_are_the_reference(entry, dtype):
    """The direct pass holds a plane column by column after its first 9
    passes when the box it scans has at most 64 columns, and at least 8 rows
    or a plane of at most 64 columns and as many rows, and its budget exceeds
    16; it keeps its rows otherwise: boxes and budgets either side of those
    limits, rows that do not fill the lanes, masks that narrow a wide clean
    box, and PSFs of twice the image. Every plane spends its budget."""
    narrow = np.zeros((40, 100), bool)
    narrow[3:37, 30:51] = True
    flat = np.zeros((3, 100), bool)
    flat[:, 30:51] = True
    full = (-1, -1, -1, -1)
    for ny, nx, niter, mask, twice, clean_box in (
        (8, 64, 17, None, False, full),  # columns
        (8, 65, 17, None, False, full),  # rows: a box of 65 columns
        (7, 70, 17, None, False, (3, 67, -1, -1)),  # rows: 7 rows of a wide plane
        (20, 20, 17, None, True, (0, 20, 5, 12)),  # columns: 7 rows of a narrow plane
        (7, 64, 17, None, False, full),  # rows: a plane wider than tall
        (48, 3, 30, None, True, (2, 3, 47, 48)),  # columns: one pixel, a tall plane
        (3, 48, 30, None, False, (40, 41, 0, 1)),  # rows: one pixel, a wide plane
        (13, 20, 16, None, False, full),  # rows: a budget of 16
        (13, 20, 17, "random50", False, full),  # columns
        (17, 33, 30, "runs", True, full),  # columns
        (40, 100, 30, narrow, False, full),  # columns: the mask's box
        (3, 100, 30, flat, True, full),  # rows: the mask's 3 rows of a wide plane
    ):
        case = _case(
            ny,
            nx,
            dtype,
            planes=(1, 1, 2),
            mask=mask,
            niter=niter,
            threshold=-1.0,
            divergence=False,
            twice=twice,
            clean_box=clean_box,
            seed=92,
        )
        _check(entry, case, threads=(1, 2))


#: Narrow planes (rows, columns), among them the slowest the verifiers timed.
NARROW = ((48, 3), (300, 6), (49, 2), (520, 20), (260, 12), (2000, 7))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("shape", NARROW, ids=lambda shape: "x".join(map(str, shape)))
def test_narrow_planes_are_the_reference(entry, shape):
    """Planes of a few columns, which the direct pass cleans: masks of short
    runs and single pixels, clean boxes, PSFs of twice the image, and zero
    peaks that a negative threshold keeps running, in float64 and float32,
    six planes at 1 and 12 threads."""
    ny, nx = shape
    for i, (mask, twice, clean_box, threshold) in enumerate(
        (
            (None, False, (-1, -1, -1, -1), 0.0),
            ("random50", False, (-1, -1, -1, -1), 0.0),
            ("checker", True, (-1, -1, -1, -1), 0.0),
            ("runs", False, (0, nx - 1, ny // 3, 2 * ny // 3), 0.0),
            ("single", True, (-1, -1, -1, -1), -1.0),
            (None, True, (nx // 2, nx, 1, ny - 1), 0.0),
        )
    ):
        case = _case(
            ny,
            nx,
            (np.float64, np.float32)[i % 2],
            planes=(1, 3, 2),
            mask=mask,
            niter=80,
            threshold=threshold,
            divergence=threshold >= 0,
            clean_box=clean_box,
            twice=twice,
            seed=89 + nx,
        )
        _check(entry, case, threads=(1, 12))


@pytest.mark.parametrize("entry", ENTRIES)
def test_small_planes_with_sparse_masks_are_the_reference(entry):
    """Small planes whose masks keep little of the boxes that the direct pass
    scans: masks keeping about 5, 20, 30 and 95 % of the plane, a single
    pixel, an empty mask whose zero peaks a negative threshold keeps running,
    and calls whose planes keep everything or nothing (a quarter and a half of
    them kept), at 1, 6 and 24 threads."""
    kept = np.zeros((1, 2, 2, 24, 30), bool)
    kept[0, 0, 0] = True
    for i, (shape, mask, threshold) in enumerate(
        (
            ((3, 48), "random5", 0.0),
            ((64, 64), "random20", 0.0),
            ((64, 64), "random30", 0.0),
            ((40, 100), "random95", 0.0),
            ((48, 52), "single", 0.0),
            ((100, 71), "empty", -1.0),
            ((24, 30), kept, 0.0),
            ((24, 30), kept | kept[:, :, ::-1], 0.0),
        )
    ):
        case = _case(
            *shape,
            (np.float64, np.float32)[i % 2],
            planes=(1, 2, 2),
            mask=mask,
            niter=300,
            threshold=threshold,
            divergence=threshold >= 0,
            seed=200 + i,
        )
        _check(entry, case, threads=(1, 6, 24))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("shape", [(8, 600), (3, 2000), (13, 513)])
def test_rows_of_several_chunks_in_the_direct_pass_are_the_reference(
    entry, dtype, shape
):
    """Rows longer than a chunk (256 pixels) on planes that the direct pass
    cleans: groups of 8 rows and fewer, masks, clean boxes that begin and end
    inside a chunk, and PSFs of twice the image."""
    ny, nx = shape
    for mask, clean_box, twice in (
        (None, (-1, -1, -1, -1), False),
        ("disk", (-1, -1, -1, -1), True),
        ("random90", (3, nx - 5, 0, ny), False),
        (None, (255, nx - 1, -1, -1), True),
    ):
        case = _case(
            ny,
            nx,
            dtype,
            planes=(1, 1, 2),
            mask=mask,
            niter=50,
            clean_box=clean_box,
            twice=twice,
            seed=90,
        )
        _check(entry, case, threads=(1, 2))


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
        "narrow": (
            "clean_cube",
            _case(300, 6, np.float64, mask="random50", niter=100, seed=44),
        ),
        "split disk": (
            "clean_cube_many_threads",
            _case(96, 100, np.float64, planes=(1, 1, 2), niter=150, seed=45),
        ),
        "split runs": (
            "clean_cube",
            _case(92, 97, np.float32, mask="runs", niter=150, seed=46),
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


#: The C++ compiler and nm for a fresh compile of the AVX2 unit, if any.
CXX = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
NM = shutil.which("nm")


@pytest.mark.skipif(
    platform.machine().lower() not in ("x86_64", "amd64") or CXX is None or NM is None,
    reason="needs x86-64, a C++ compiler and nm",
)
def test_the_avx2_unit_defines_no_weak_symbol_at_o0(tmp_path):
    """A build without CMAKE_BUILD_TYPE compiles the AVX2 unit without any -O
    flag. At -O0 too it defines no weak or unique symbol (nm types W, V, u):
    it calls no function of a standard header and has no exception tables,
    so no AVX2 copy of a function can replace a generic caller's.
    test_binary_isa.py checks the levels of the build types."""
    module = Path(__file__).parents[3] / "src/astroviper/_kernels/fast/hogbom"
    target = tmp_path / "hclean_fast_avx2.o"
    subprocess.run(
        [
            CXX,
            "-std=c++23",
            "-fPIC",
            "-ffp-contract=off",
            "-mavx2",
            "-fno-lto",
            "-O0",
            "-DAV_HAVE_AVX2=1",
            f"-I{module / 'include'}",
            f"-I{module.parent / 'common'}",
            "-c",
            str(module / "src/hclean_fast_avx2.cpp"),
            "-o",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    listing = subprocess.run(
        [NM, str(target)], check=True, capture_output=True, text=True
    )
    symbols = [
        line.split()[-2:] for line in listing.stdout.splitlines() if line.strip()
    ]
    assert [s for s in symbols if s[0] in ("W", "V", "u")] == []
    assert ["T", "_ZN11hclean_fast12avx2_kernelsEv"] in symbols


def test_an_unknown_isa_override_fails_the_import(run_python):
    done = run_python(
        "-c",
        "import astroviper._kernels.fast.hogbom",
        check=False,
        ASTROVIPER_KERNELS_ISA="avx512",
    )
    assert done.returncode != 0
    assert "ASTROVIPER_KERNELS_ISA=avx512: expected 'generic'" in done.stderr


def test_the_generic_routines_pass_the_bitwise_tests(run_python, tmp_path):
    """The routines that every platform without AVX2 runs (aarch64, macOS):
    this module's comparisons with the reference, in a pytest subprocess with
    ASTROVIPER_KERNELS_ISA=generic."""
    isa = "from astroviper._kernels.fast.hogbom import ISA; print(ISA)"
    assert run_python("-c", isa, ASTROVIPER_KERNELS_ISA="generic").strip() == "generic"
    environ = {
        k: v for k, v in os.environ.items() if not k.startswith("ASTROVIPER_KERNELS")
    }
    environ |= {"ASTROVIPER_KERNELS_ISA": "generic", "PYTEST_ADDOPTS": ""}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-q",
            "-p",
            "no:cacheprovider",
            "-k",
            "(are_the_reference or where_the_reference) and not declined",
            f"--basetemp={tmp_path / 'inner'}",
        ],
        cwd=Path(__file__).parents[3],
        env=environ,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]
    passed = re.search(r"(\d+) passed", done.stdout)
    assert passed and int(passed.group(1)) >= 300, done.stdout[-2000:]


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


@pytest.mark.parametrize("planes_with_budgets", [2, 12])
def test_idle_threads_sleep_and_nothing_runs_after_a_call(planes_with_budgets):
    """12 small masked planes and 12 threads requested, one plane running
    20000 iterations (an all-true mask, so no final subtractions). The planes
    run side by side, one thread per plane with a budget (2 or 12), and the
    threads of the planes that stop at once end. The call's CPU time stays
    near that of the one busy thread, far from 12 threads polling throughout,
    and after the call returns the process burns no CPU."""
    case = _case(128, 128, np.float64, planes=(1, 3, 4), mask="all", niter=20000)
    case["kwargs"]["max_iter_per_cycle_updated"].flat[planes_with_budgets:] = 0
    case["kwargs"]["threshold_per_cycle"] = np.full((1, 3, 4), 1e6)
    case["kwargs"]["threshold_per_cycle"][0, 0, 0] = 0.0
    case["kwargs"] |= _divergence(0.1, on=False)
    cpu, start = _cpu_seconds(), time.perf_counter()
    result, _, _ = _run(FAST["clean_cube"], case, 12)
    wall, cpu = time.perf_counter() - start, _cpu_seconds() - cpu
    assert result["iterations_performed"][0, 0, 0] == 20000
    assert result["iterations_performed"].sum() == 20000
    assert cpu < 3.0 * wall + 0.02, (cpu, wall)
    cpu = _cpu_seconds()
    time.sleep(0.2)
    assert _cpu_seconds() - cpu < 0.02


def test_idle_team_threads_sleep_and_nothing_runs_after_a_call():
    """3 planes of 256 x 256 with a disk mask (a team: passes of at least
    2 x 16,384 pixels) at 24 threads, plane 0 running 20000 iterations and the
    others 10: plane 0's crew iterates while the crews of the other two end
    early and the rest of the team waits for the final passes. The waiting
    threads poll for a bounded time and then sleep, so the call's CPU time
    stays near that of plane 0's crew, far from 24 threads polling
    throughout, and after the call returns the process burns no CPU."""
    yy, xx = np.mgrid[0:256, 0:256]
    disk = np.hypot(yy - 128, xx - 128) < 102
    case = _case(256, 256, np.float64, planes=(1, 1, 3), mask=disk, niter=10, seed=77)
    case["kwargs"]["max_iter_per_cycle_updated"][0, 0, 0] = 20000
    case["kwargs"]["threshold_per_cycle"][...] = -1.0
    case["kwargs"] |= _divergence(0.1, on=False) | {"record_iteration_history": False}
    cpu, start = _cpu_seconds(), time.perf_counter()
    result, _, _ = _run(FAST["clean_cube_many_threads"], case, 24)
    wall, cpu = time.perf_counter() - start, _cpu_seconds() - cpu
    assert result["iterations_performed"].tolist() == [[[20000, 10, 10]]]
    assert cpu < 6.0 * wall + 0.05, (cpu, wall)
    cpu = _cpu_seconds()
    time.sleep(0.2)
    assert _cpu_seconds() - cpu < 0.02


def _threads_seen(call):
    """The process's threads before `call()` and the most a sampler thread saw
    during it (the sampler included)."""
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
        call()
        done.set()
        sampler.join()
    finally:
        sys.setswitchinterval(interval)
    return before, max(seen)


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="needs /proc")
@pytest.mark.parametrize("n", [64, 128])
def test_small_planes_start_no_more_threads_than_planes_with_a_budget(n):
    """12 planes of 64 x 64 (the direct pass) or 128 x 128 (split) with budgets
    on 4, at 24 threads: at most 4 threads (the caller and 3 more), as many as
    the reference clean_cube starts for 4 planes; more would only wait for the
    final passes."""
    case = _case(
        n,
        n,
        np.float64,
        planes=(1, 3, 4),
        niter=6000,
        divergence=False,
        history=False,
        seed=76,
    )
    case["kwargs"]["max_iter_per_cycle_updated"].flat[4:] = 0
    before, most = _threads_seen(lambda: _run(FAST["clean_cube"], case, 24))
    assert before + 1 < most <= before + 1 + 3, (before, most)


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="needs /proc")
@pytest.mark.parametrize("entry", ENTRIES)
def test_small_planes_that_stop_at_their_first_search_start_no_thread(entry):
    """12 planes of 16 x 16 with budgets of 1000 and thresholds above their
    peaks, at 24 threads: their first searches, made before the threads are
    chosen, stop them all, so the call runs on the calling thread alone (the
    reference starts a thread per plane), with the reference's bits."""
    case = _case(
        16,
        16,
        np.float64,
        planes=(1, 3, 4),
        niter=1000,
        threshold=1e9,
        divergence=False,
        seed=78,
    )
    expected = _run(REFERENCE[entry], case)
    actual = []
    before, most = _threads_seen(lambda: actual.append(_run(FAST[entry], case, 24)))
    _assert_same(expected, actual[0])
    assert most == before + 1, (before, most)


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="needs /proc")
@pytest.mark.parametrize("entry", ENTRIES)
def test_a_small_box_on_a_large_plane_gets_a_team_for_its_final_passes(entry):
    """Two planes of 512 x 512 with a clean box of 128 x 128 pixels (too small
    for a crew of two), Stokes I iterating and V stopping at once, at 6
    threads: the final passes subtract every component from the rest of I's
    plane, and the call's other threads share them (a thread per plane with a
    budget left them all to one thread, 8.4 times as slow on 2048 x 2048
    planes, the adversarial review measured), with the reference's bits."""
    if len(os.sched_getaffinity(0)) < 4:
        pytest.skip("needs 4 CPUs for a team of 4 threads or more")
    case = _case(
        512,
        512,
        np.float64,
        planes=(1, 1, 2),
        mask=None,
        niter=300,
        threshold=[0.0, 1e6],
        clean_box=(192, 320, 192, 320),
        history=False,
        seed=91,
    )
    expected = _run(REFERENCE[entry], case)
    actual = []
    before, most = _threads_seen(lambda: actual.append(_run(FAST[entry], case, 6)))
    _assert_same(expected, actual[0])
    assert most >= before + 1 + 3, (before, most)


#: Input of the subprocess tests below: `case(planes, n, niter, mask)` makes the
#: keyword arguments of a call (a Gaussian PSF and four point sources a plane).
SUBPROCESS_CASES = """
import json, resource, sys, threading
import numpy as np
from astroviper._kernels import impl
from astroviper._kernels.fast.hogbom import _hogbom_fast_ext as ext

def case(planes, n, niter, mask=True, threshold=0.0, history=False):
    rng = np.random.default_rng(n)
    y, x = np.mgrid[0:n, 0:n] - n // 2
    psf = np.exp(-(x * x + y * y) / 8.0)
    residual = np.zeros(planes + (n, n))
    for index in np.ndindex(planes):
        for _ in range(4):
            dy, dx = rng.integers(-n // 4, n // 4, 2)
            residual[index] += rng.uniform(0.5, 2) * np.roll(psf, (dy, dx), (0, 1))
    disk = np.broadcast_to(x * x + y * y < (0.4 * n) ** 2, residual.shape)
    return dict(
        residual_cube=residual,
        psf_cube=np.ascontiguousarray(np.broadcast_to(psf, residual.shape)),
        model_cube=np.zeros_like(residual),
        mask_cube=np.ascontiguousarray(disk) if mask else np.array([]),
        max_iter_per_cycle_updated=np.full(planes, niter, np.int32),
        gain=0.1,
        threshold_per_cycle=np.full(planes, threshold),
        max_iter_divergence=-1,
        record_iteration_history=history,
    )

def copied(kwargs):
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}

def bits(kwargs, result):  # of a call without the history
    out = [kwargs["residual_cube"].tobytes(), kwargs["model_cube"].tobytes()]
    return out + [np.asarray(value).tobytes() for value in result.values()]
"""

NO_THREADS = (
    SUBPROCESS_CASES
    + """
cases = {"direct": case((1, 3, 4), 40, 200), "split side by side": case((1, 3, 4), 128, 200),
         "team": case((1, 1, 2), 256, 200, mask=False)}
expected = {}
for name, kwargs in cases.items():
    for entry in ("clean_cube", "clean_cube_many_threads"):
        k = copied(kwargs)
        expected[name, entry] = bits(k, impl("hogbom", entry, "ref")(**k, processing_function_threads=1))
resource.setrlimit(resource.RLIMIT_NPROC, (1, resource.getrlimit(resource.RLIMIT_NPROC)[1]))
try:
    threading.Thread(target=lambda: None).start()
except RuntimeError:
    pass
else:
    print("RESULT", json.dumps(None))
    sys.exit()
same = {}
for (name, entry), want in expected.items():
    k = copied(cases[name])
    result = getattr(ext, entry)(**k, processing_function_threads=12)
    same[f"{name} {entry}"] = result is not None and bits(k, result) == want
print("RESULT", json.dumps(same))
"""
)


def test_threads_that_cannot_start_leave_the_same_bits(run_python):
    """With RLIMIT_NPROC at 1 no thread can start: a call at 12 threads takes
    the fast path on the threads it has (small and split planes side by side,
    and the team of larger ones), with the reference's bits."""
    out = run_python("-c", NO_THREADS)
    same = json.loads(out.rpartition("RESULT ")[2])
    if same is None:
        pytest.skip("RLIMIT_NPROC is not enforced here (a privileged user)")
    assert same and all(same.values()), same


MEMORY = (
    SUBPROCESS_CASES
    + """
def misaligned(array):
    buffer = np.zeros(array.nbytes + 1, np.uint8)
    copy = buffer[1:].view(array.dtype).reshape(array.shape)
    copy[...] = array
    return copy

def address_space():
    with open("/proc/self/status") as status:
        line = next(s for s in status if s.startswith("VmSize:"))
    return int(line.split()[1]) * 1024

def wide(niter):  # a split plane of 8 x 1025 pixels whose clean box is its first pixel
    residual = np.zeros((1, 1, 1, 8, 1025))
    residual[..., 0, 0], residual[..., 5, 700] = 1.0, 0.5
    psf = np.zeros_like(residual)
    psf[..., 3:6, 400:600] = 0.01
    psf[..., 4, 512] = 1.0
    return dict(residual_cube=residual, psf_cube=psf, model_cube=np.zeros_like(residual),
                clean_box=(0, 1, 0, 1), max_iter_per_cycle_updated=np.full((1, 1, 1), niter, np.int32),
                gain=0.1, threshold_per_cycle=np.full((1, 1, 1), -1.0), max_iter_divergence=-1)

ext.clean_cube(**case((1, 1, 1), 16, 100), processing_function_threads=1)
ext.clean_cube(**wide(100), processing_function_threads=1)
limit = address_space() + 16 * 2**20
resource.setrlimit(resource.RLIMIT_AS, (limit, resource.getrlimit(resource.RLIMIT_AS)[1]))
# 600,000 iterations without the history: 19 MB of subtractions kept until
# the end would exceed the limit; the 16,384 a split plane defers at most do
# not, and the direct pass defers none.
split = ext.clean_cube(**wide(600_000), processing_function_threads=1)
long = ext.clean_cube(**case((1, 1, 1), 16, 600_000, threshold=-1.0), processing_function_threads=1)
# The history of 10 million iterations cannot fit: the error comes after the
# first write, and a misaligned cube is written back as cleaned as it is.
kwargs = case((1, 1, 1), 16, 10_000_000, threshold=-1.0, history=True)
for name in ("residual_cube", "model_cube", "psf_cube"):
    kwargs[name] = misaligned(kwargs[name])
original = kwargs["residual_cube"].copy()
try:
    ext.clean_cube(**kwargs, processing_function_threads=1)
    error = None
except MemoryError:
    error = "MemoryError"
print("RESULT", json.dumps({
    "split iterations": None if split is None else int(split["iterations_performed"].sum()),
    "iterations": None if long is None else int(long["iterations_performed"].sum()),
    "error": error,
    "written_back": not np.array_equal(kwargs["residual_cube"], original),
}))
"""
)


@pytest.mark.skipif(not os.path.isfile("/proc/self/status"), reason="needs /proc")
def test_memory_does_not_grow_with_the_iterations_and_errors_write_back(run_python):
    """Under an address-space limit of 16 MiB above the process's size: calls
    of 600,000 iterations complete (a split plane keeps at most 16,384 deferred
    subtractions, 512 KiB, and the direct pass none), and a call whose
    iteration history outgrows the limit raises MemoryError after its first
    write with its misaligned cube written back, as an aligned cube would be."""
    with open("/proc/self/maps") as maps:
        if "libasan" in maps.read():
            pytest.skip("AddressSanitizer reserves address space of its own")
    out = run_python("-c", MEMORY)
    result = json.loads(out.rpartition("RESULT ")[2])
    assert result == {
        "split iterations": 600_000,
        "iterations": 600_000,
        "error": "MemoryError",
        "written_back": True,
    }


BLOCKED = """
import importlib.abc, json, sys
import pytest

NAME = "astroviper._kernels.fast.hogbom._hogbom_fast_ext"

class NotBuilt(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == NAME:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, NotBuilt())
try:
    import astroviper._kernels.fast.hogbom
    raised = None
except ImportError as error:
    raised = [type(error).__name__, error.name]
try:
    pytest.importorskip(NAME, reason="fast Hogbom module not built")
    skipped = False
except pytest.skip.Exception:
    skipped = True
print("RESULT", json.dumps({"raised": raised, "skipped": skipped}))
"""


def test_without_the_extension_the_package_raises_module_not_found(run_python):
    """A build without the extension (fast kernels OFF, or a failed compile
    probe): importing the package raises ModuleNotFoundError naming it, which
    pytest.importorskip skips (it re-raises a plain ImportError), so that this
    module skips instead of interrupting the session."""
    out = run_python("-c", BLOCKED)
    result = json.loads(out.rpartition("RESULT ")[2])
    name = "astroviper._kernels.fast.hogbom._hogbom_fast_ext"
    assert result == {"raised": ["ModuleNotFoundError", name], "skipped": True}


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


@pytest.mark.parametrize("entry", ENTRIES)
def test_an_unencodable_keyword_is_the_reference(entry, run_python):
    """A keyword name that UTF-8 cannot encode (a lone surrogate) declines,
    writing nothing (names are compared in place, by no conversion that could
    raise), and the call is then the reference's. The reference binding
    aborts on it (pybind11 converting the name for its TypeError), so each
    outcome is that of a fresh interpreter, and the outcomes are compared
    whole: exit status and standard error."""
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from astroviper._kernels import impl\n"
        "from astroviper._kernels.fast.hogbom import _hogbom_fast_ext as ext\n"
        "residual = np.zeros((1, 1, 1, 8, 8))\n"
        "residual[..., 4, 4] = 1.0\n"
        "original = residual.copy()\n"
        "arguments = {'residual_cube': residual, 'psf_cube': original.copy(),\n"
        "             'model_cube': np.zeros_like(residual),\n"
        "             'max_iter_per_cycle_updated': np.full((1, 1, 1), 5, np.int32),\n"
        "             'threshold_per_cycle': np.zeros((1, 1, 1)), '\\udc80': 1}\n"
        "if sys.argv[1] == 'fast':\n"
        f"    result = ext.{entry}(**arguments)\n"
        "    print(result, np.array_equal(residual, original), flush=True)\n"
        f"impl('hogbom', '{entry}', sys.argv[1])(**arguments)\n"
    )
    quiet = {"PYTHONFAULTHANDLER": "", "PYTHONDEVMODE": ""}  # its frames differ
    fast_run, ref_run = (
        run_python("-c", code, name, check=False, **quiet) for name in ("fast", "ref")
    )
    assert fast_run.stdout == "None True\n" and ref_run.returncode != 0
    assert (fast_run.returncode, fast_run.stderr) == (
        ref_run.returncode,
        ref_run.stderr,
    )


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
