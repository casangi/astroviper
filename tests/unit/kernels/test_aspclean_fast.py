"""Tests of the fast Asp model update (``ASTROVIPER_KERNELS=asp=fast``).

The implementation is called directly (``astroviper._kernels.impl``), so these
tests run in any session. Each one runs the reference (``asp=ref``: the
installed ``deconvolvers.aspclean`` extension, at casangi/astroviper#410) and
the fast kernel on copies of the same inputs and asserts the same bytes: the
residual, the model and every value of the returned dict (the components, the
residual-shaped fits, the iteration history and every count; only the wall
times of ``timing`` are skipped), NaN payloads and signed zeros included, or
the same exception type and message. Small inputs (at most 256^2), but for a
512^2 and a 768^2 plane, whose large work buffers are mapped from the system,
and a 480 x 8192 plane, whose scale convolutions do not fit the fused pass.

* The entries are the reference's: the docstrings (so the names, types and
  defaults of the parameters, ``asp_min_mask_coverage = 0.995`` last), the
  refused control values (``ValueError``), the binding's checks, any thread
  count, pickled arrays, misaligned arrays (the PSF and mask read in place,
  the residual and model through aligned copies), aliased arrays, empty
  planes (the reference crashes on them), concurrent calls of every kind with
  the GIL released, and a call at 1 thread starts no thread; with flush-to-zero
  or directed rounding set on the calling thread only, the bits do not depend
  on the thread count.
* Calls that decline to the reference (a PSF or model that is not finite or
  of magnitude 2^64 or more, such a residual where it is searched, a model
  that overlaps the residual): the reference's outcome at the same thread
  count, traced with path ``"declined"``; NaN, infinite and huge values
  elsewhere do not decline. In a cube only such planes run the reference,
  each alone, with the share of the threads its cube driver gives a plane.
* Initial scales beyond the range of int (asp_psf_width up to 4e307, a PSF
  much wider than the image), where the reference's conversions are
  undefined: the values of its compiled code.
* The scenes of #410's tests and more (``asp_min_mask_coverage``): two mask
  boxes, a tight mask with holes, soft masks, a full mask and no mask, a mask
  too small for every scale, fits cut back (to the widest Gaussian on the
  mask, to the scale the search tested, to a point), the rare paths of the
  cut-back (a bisection from 0 that stays at 0, a negative fitted sigma cut
  back, shares below 0 at the coverages 0, -0, 1e-300 and 5e-324), an exact
  tie of the cut-back's first test, mask values exactly at the search
  threshold 0.99 and the bounding-box threshold 1e-6 (and one double away, in
  float32 too), NaN, infinite and negative mask values, masks at the image
  edges and single pixels, odd sizes and float32 inputs; with the binding's
  defaults (gain 0.1), the kernel's as #410's tests run it (gain 0.3) and the
  options ``deconvolution.asp_clean`` passes.
* Cubes of such planes with their own thresholds, iteration limits and PSFs
  (or one broadcast PSF), float64 and float32, at 1 to 12 threads; models that
  are not zero on input (signed zeros, subnormals, values); exact ties of the
  peak search.
* Through the switch: ``deconvolution.asp_clean`` with ``asp=fast`` passes
  ``asp_min_mask_coverage`` (its default and other values) to the fast
  kernel and gives the reference's bytes, and the reference's error for a
  refused value.
"""

import importlib
import json
import math
import os
import pickle
import platform
import re
import sys
import threading
import time
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from astroviper._kernels import dispatch, impl

ext = pytest.importorskip(
    "astroviper._kernels.fast.aspclean._aspclean_fast_ext",
    reason="fast Asp module not built",
)
ENTRIES = ("clean", "clean_cube")
FAST = {entry: impl("asp", entry, "fast") for entry in ENTRIES}
REFERENCE = {entry: impl("asp", entry, "ref") for entry in ENTRIES}
REFERENCE_THREADS = 4

#: The kernel's own defaults, as #410's tests run it (no residual-shaped
#: components, no significance search).
KERNEL = {"gain": 0.3}
#: The options ``deconvolution.asp_clean`` passes by default (gain 0.3).
CORAL = {
    "gain": 0.3,
    "asp_fused_threshold": 0.0,
    "asp_psf_width": -1.0,
    "asp_largest_scale": -1,
    "asp_stop_point_mode": -1,
    "asp_scale_normalization_method": 1,
    "max_iter_divergence": 1,
    "divergence_rms_factor": 1.3,
    "divergence_peak_factor": np.inf,
    "asp_stop_after_hogbom_run": False,
    "asp_scale_significance": True,
    "asp_significance": 0.0,
    "asp_residual_components": True,
    "asp_residual_component_noise_floor": 2.0,
    "asp_residual_component_min_fraction": 0.3,
    "asp_residual_component_max_epsilon_fraction": 10.0,
    "asp_residual_component_switch_factor": 1.0,
    "asp_residual_component_floor_at_threshold": True,
    "asp_residual_component_positive_only": True,
    "asp_significance_gain": 0.3,
    "asp_no_change_tolerance": 0.0,
}
#: The binding's own defaults (gain 0.1).
DEFAULTS = {}
BASES = {"defaults": DEFAULTS, "kernel": KERNEL, "coral": CORAL}
RECORD = {"record_components": True, "record_iteration_history": True}

#: The wall times of a plane's ``timing``: the only values that may differ.
WALL_TIMES = {
    "setup",
    "search",
    "fit",
    "update_scale",
    "update_point",
    "bookkeeping",
    "residual_components",
    "significance",
}
A99, B99 = np.nextafter(0.99, 1.0), np.nextafter(0.99, 0.0)


# ---------------------------------------------------------------- comparison
def _assert_same(got, expected, where):
    """The same values, byte for byte and with the same types: dicts key by
    key (but the wall times of ``timing``), sequences item by item, an
    exception's class and message."""
    if isinstance(expected, dict):
        assert isinstance(got, dict) and got.keys() == expected.keys(), where
        for key, value in expected.items():
            if not (where.endswith("['timing']") and key in WALL_TIMES):
                _assert_same(got[key], value, f"{where}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert type(got) is type(expected) and len(got) == len(expected), where
        for k, (a, b) in enumerate(zip(got, expected, strict=True)):
            _assert_same(a, b, f"{where}[{k}]")
    elif isinstance(expected, (type, str)):
        assert got == expected, where
    else:
        assert type(got) is type(expected), where
        a, b = np.asarray(got), np.asarray(expected)
        assert (a.dtype, a.shape) == (b.dtype, b.shape), where
        assert a.tobytes() == b.tobytes(), where


def _call(function, inputs, threads, options):
    """``function`` on ``inputs`` with fresh copies of its residual and model:
    ``(residual, model, result)``, or ``(exception class, message)``."""
    arguments = inputs | {k: inputs[k].copy() for k in ("residual", "model")}
    try:
        result = function(**arguments, processing_function_threads=threads, **options)
    except Exception as e:  # the reference's exception is the expected outcome
        return type(e), str(e)
    return arguments["residual"], arguments["model"], result


def _check(inputs, threads=(1, 7), entry="clean", **options):
    """The fast entry gives the reference's outcome at each thread count;
    returns the reference's."""
    expected = _call(REFERENCE[entry], inputs, REFERENCE_THREADS, options)
    for n in threads:
        got = _call(FAST[entry], inputs, n, options)
        _assert_same(got, expected, f"{entry} at {n} threads")
    return expected


# ---------------------------------------------------------------- inputs
def _gauss(ny, nx, cy, cx, sigma, amplitude=1.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return amplitude * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))


def _psf(ny, nx, sigma=2.0):
    return _gauss(ny, nx, ny // 2, nx // 2, sigma)


def _convolve(image, psf):
    """The linear convolution with the PSF centred on (ny // 2, nx // 2): the
    response the kernels subtract."""
    ny, nx = image.shape
    shape = (2 * ny, 2 * nx)
    full = np.fft.irfft2(np.fft.rfft2(image, shape) * np.fft.rfft2(psf, shape), shape)
    return full[ny // 2 : ny // 2 + ny, nx // 2 : nx // 2 + nx]


def _disk(ny, nx, cy, cx, radius):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return (np.hypot(yy - cy, xx - cx) < radius).astype(np.float64)


def _noisy(sky, psf, seed, quiet, sigma):
    """sky (*) psf plus PSF-smoothed noise of rms ``sigma`` where ``quiet``."""
    noise = _convolve(np.random.default_rng(seed).standard_normal(sky.shape), psf)
    return _convolve(sky, psf) + noise * (sigma / noise[quiet].std())


def _threshold(residual, mask, fraction=0.1):
    """``fraction`` of the peak |residual| on the search region."""
    region = np.abs(residual if mask is None else residual[mask > 0.99])
    region = region[np.isfinite(region)]
    return fraction * float(region.max()) if region.size else 0.0


def _plane(psf, dirty, mask, threshold, max_iter, dtype=np.float64, model=None):
    """The arguments of a ``clean`` call (residual and model are copied by
    ``_call``; the others are read-only); the model is zero by default."""
    inputs = {
        "residual": np.ascontiguousarray(dirty, dtype),
        "psf": np.ascontiguousarray(psf, dtype),
        "mask": np.array([], dtype)
        if mask is None
        else np.ascontiguousarray(mask, dtype),
        "threshold_per_cycle": float(threshold),
        "max_iter_per_cycle_updated": max_iter,
    }
    inputs["model"] = (
        np.zeros_like(inputs["residual"])
        if model is None
        else np.ascontiguousarray(model, dtype)
    )
    for value in inputs.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return inputs


def _float32(inputs):
    return _plane(
        inputs["psf"],
        inputs["residual"],
        inputs["mask"] if inputs["mask"].size else None,
        inputs["threshold_per_cycle"],
        inputs["max_iter_per_cycle_updated"],
        np.float32,
    )


# Scenes: name -> (psf, dirty, mask, threshold, max_iter).
def _two_boxes():
    """#410's scene: emission between two mask boxes (off the mask, inside its
    bounding box) and a compact source in one box."""
    n = 256
    psf = _psf(n, n)
    sky = _gauss(n, n, 125, 125, 15.0) + _gauss(n, n, 60, 60, 1.0, 0.5)
    mask = np.zeros((n, n))
    mask[40:80, 40:80] = mask[170:210, 170:210] = 1.0
    return psf, _convolve(sky, psf), mask, 1e-3, 300


def _tight(holes=False):
    """A Gaussian disk, points on and off a tight mask, noise (and holes in the
    mask: next to the centre, 3 x 3, 4 x 6)."""
    n = 160
    psf = _psf(n, n)
    sky = _gauss(n, n, 80, 80, 9.0, 2e-3)
    for y, x, flux in ((75, 88, 4e-3), (95, 70, 2e-3), (80, 80, 3e-3), (40, 125, 5e-3)):
        sky[y, x] += flux
    mask = _disk(n, n, 80, 80, 26.0)
    dirty = _noisy(sky, psf, 1302, mask == 0, 9e-5)
    if holes:
        mask[80, 81] = 0.0
        mask[66:69, 88:91] = 0.0
        mask[88:92, 62:68] = 0.0
    return psf, dirty, mask, _threshold(dirty, mask), 150


def _soft_box(max_iter):
    """#410's soft mask: 0.5 everywhere, 1 on an 11 x 11 box."""
    n = 128
    psf = _psf(n, n)
    mask = np.full((n, n), 0.5)
    mask[59:70, 59:70] = 1.0
    return psf, _convolve(_gauss(n, n, 64, 64, 4.7), psf), mask, 1e-3, max_iter


def _soft_sky():
    n = 128
    psf = _psf(n, n)
    rr = np.hypot(*(np.mgrid[0:n, 0:n] - 64.0))
    sky = _gauss(n, n, 64, 64, 8.0)
    sky[60, 70] += 2.0
    sky[75, 52] += 1.0
    return psf, _noisy(sky, psf, 1304, rr >= 40, 1e-3), rr


def _taper():
    """1 inside r < 20, down to 0.2 at r = 40, 0 beyond."""
    psf, dirty, rr = _soft_sky()
    mask = np.where(rr < 20, 1.0, np.where(rr < 40, 1.0 - 0.8 * (rr - 20) / 20, 0.0))
    return psf, dirty, mask, _threshold(dirty, mask), 150


def _soft_0995():
    """0.995 on a disk (above the search threshold 0.99), 0.5 elsewhere."""
    psf, dirty, rr = _soft_sky()
    mask = np.where(rr < 30, 0.995, 0.5)
    return psf, dirty, mask, _threshold(dirty, mask), 150


def _wide(mask):
    """#410's full-mask scene: a scale wider than the image."""
    n = 128
    dirty = _gauss(n, n, 64, 64, 45.0, 5.0)
    full = np.ones((n, n)) if mask else None
    return _psf(n, n, 6.0), dirty, full, 1e-3, 30


def _too_small():
    n = 128
    mask = np.zeros((n, n))
    mask[60:69, 60:69] = 1.0
    return _psf(n, n), _gauss(n, n, 64, 64, 3.0, 5.0), mask, 1e-3, 300


def _cut_back(max_iter, hole=False):
    """#410's cut-back: a source wider than its box (and a hole next to the
    centre, which cuts narrower Gaussians than the scale tested)."""
    n = 128
    psf = _psf(n, n)
    mask = np.zeros((n, n))
    mask[44:85, 44:85] = 1.0
    if hole:
        mask[64, 65] = 0.0
    return psf, _convolve(_gauss(n, n, 64, 64, 15.0), psf), mask, 1e-3, max_iter


def _point_hole(max_iter):
    """#410's last scene: no Gaussian keeps the share a point does."""
    n = 128
    psf = _psf(n, n)
    mask = np.ones((n, n))
    mask[64, 65] = 0.0
    return psf, _convolve(_gauss(n, n, 64, 64, 1.2, 5.0), psf), mask, 1e-3, max_iter


def _point_source(shape="point"):
    ny, nx = 150, 170
    psf = _psf(ny, nx)
    if shape == "point":
        sky = np.zeros((ny, nx))
        sky[75, 85] = 5e-3
    else:
        sky = _gauss(ny, nx, 75, 85, 0.8, 5e-3)
    rr = np.hypot(*(np.mgrid[0:ny, 0:nx] - np.array([75, 85])[:, None, None]))
    return psf, _noisy(sky, psf, 4150, rr > 40, 2e-5)


def _point_patch():
    """A soft patch (0.995) under a point source: at a coverage above 0.995 no
    Gaussian there keeps it, the bisection from 0 stays at 0 (a point)."""
    psf, dirty = _point_source()
    mask = _disk(150, 170, 75, 85, 40.0)
    mask[74:77, 84:87] = 0.995
    return psf, dirty, mask, 1e-4, 40


def _centre_in_1p5():
    """Mask 1.5 on a disk, 0.995 at the centre of a narrow source: bisections
    from 0 that stay at 0, and negative fitted sigmas."""
    psf, dirty = _point_source("tiny_gauss")
    mask = 1.5 * _disk(150, 170, 75, 85, 40.0)
    mask[75, 85] = 0.995
    return psf, dirty, mask, 1e-4, 40


def _negcut():
    """A point source next to a hole: a fit steps to a negative sigma whose
    |sigma| falls short of the share and is cut back."""
    psf, dirty = _point_source()
    mask = _disk(150, 170, 75, 85, 40.0)
    mask[75, 87] = 0.0
    return psf, dirty, mask, 1e-4, 40


def _negative_ring():
    """A strongly negative ring of the mask around the source: shares below
    0."""
    ny, nx = 140, 150
    rr = np.hypot(*(np.mgrid[0:ny, 0:nx] - np.array([70, 75])[:, None, None]))
    mask = _disk(ny, nx, 70, 75, 40.0)
    mask[(rr >= 9.0) & (rr < 11.0)] = -20.0
    psf = _psf(ny, nx, 1.2)
    dirty = _noisy(_gauss(ny, nx, 70, 75, 3.5, 3e-3), psf, 4200, rr > 40, 3e-5)
    return psf, dirty, mask, 1e-4, 50


SCENES = {
    "two_boxes": _two_boxes,
    "tight": _tight,
    "holes": lambda: _tight(holes=True),
    "soft_box_1": lambda: _soft_box(1),
    "soft_box_30": lambda: _soft_box(30),
    "taper": _taper,
    "soft_0995": _soft_0995,
    "full_wide": lambda: _wide(True),
    "no_mask_wide": lambda: _wide(False),
    "too_small": _too_small,
    "cut_back_1": lambda: _cut_back(1),
    "cut_back_60": lambda: _cut_back(60),
    "cut_hole_1": lambda: _cut_back(1, hole=True),
    "cut_hole_60": lambda: _cut_back(60, hole=True),
    "point_hole_1": lambda: _point_hole(1),
    "point_hole_30": lambda: _point_hole(30),
    "point_patch": _point_patch,
    "centre_in_1p5": _centre_in_1p5,
    "negcut": _negcut,
    "negative_ring": _negative_ring,
}


@cache
def _scene(name):
    return _plane(*SCENES[name]())


#: (scene, options base, coverages): #410's scenes, and scenes whose fits take
#: the rare paths of the cut-back (counted with a traced copy of the
#: reference): bisections from 0 (cut_hole_60, point_hole_*), bisections that
#: stay at 0, cuts to a point (point_patch, centre_in_1p5), a negative fitted
#: sigma cut back (negcut at 0.98 and 0.99), shares below 0 (negative_ring).
COVERAGE_CASES = [
    ("two_boxes", "kernel", (0.0, 0.5, 0.9, 0.99, 0.995, 0.999)),
    ("two_boxes", "coral", (0.0, 0.995)),
    ("tight", "kernel", (0.0, 0.9, 0.995, 0.999)),
    ("tight", "coral", (0.9, 0.995)),
    ("holes", "kernel", (0.0, 0.9, 0.995, 0.999)),
    ("holes", "coral", (0.995,)),
    ("soft_box_1", "kernel", (0.0, 0.5, 0.7, 0.995)),
    ("soft_box_30", "kernel", (0.7, 0.995)),
    ("taper", "kernel", (0.9, 0.995)),
    ("taper", "coral", (0.995,)),
    ("soft_0995", "kernel", (0.9, 0.99, 0.995)),
    ("full_wide", "kernel", (0.0, 0.99, 0.995)),
    ("no_mask_wide", "kernel", (0.0, 0.995)),
    ("too_small", "kernel", (0.995, 0.999)),
    ("cut_back_1", "kernel", (0.9, 0.995)),
    ("cut_back_60", "kernel", (0.995,)),
    ("cut_hole_1", "kernel", (0.995,)),
    ("cut_hole_60", "kernel", (0.995,)),
    ("point_hole_1", "kernel", (0.9, 0.99, 0.995)),
    ("point_hole_30", "kernel", (0.995,)),
    ("point_patch", "kernel", (0.9975, 0.999)),
    ("point_patch", "coral", (0.999,)),
    ("centre_in_1p5", "kernel", (0.9975, 0.999)),
    ("negcut", "defaults", (0.98, 0.99, 0.995)),
    ("negative_ring", "coral", (0.0, -0.0, 1e-300, 5e-324, 0.3)),
]


@pytest.mark.parametrize(
    "scene, base, coverage",
    [
        pytest.param(s, b, c, id=f"{s}-{b}-{c!r}")
        for s, b, cs in COVERAGE_CASES
        for c in cs
    ],
)
def test_coverage_scenes(scene, base, coverage):
    _check(_scene(scene), **BASES[base], **RECORD, asp_min_mask_coverage=coverage)


def test_default_coverage_is_the_references():
    """No keyword: the default of both, 0.995 (the result differs from 0.9)."""
    inputs = _scene("cut_back_60")
    default = _check(inputs, **KERNEL, **RECORD)
    _assert_same(
        default,
        _call(
            REFERENCE["clean"],
            inputs,
            1,
            KERNEL | RECORD | {"asp_min_mask_coverage": 0.995},
        ),
        "default",
    )
    other = _call(
        REFERENCE["clean"], inputs, 1, KERNEL | RECORD | {"asp_min_mask_coverage": 0.9}
    )
    assert default[1].tobytes() != other[1].tobytes()


def _coverage(mask, ci, cj, sigma):
    """The kernels' share of the flux of the Gaussian at (ci, cj) on the mask,
    in their IEEE operations and order (math.exp is the C library's exp, which
    they call; no compensated sums)."""
    ny, nx = mask.shape

    def profile(n, c):
        inv2s2 = 0.5 / (sigma * sigma)
        return [math.exp(-float(q - c) * float(q - c) * inv2s2) for q in range(n)]

    def total(values):
        s = 0.0
        for v in values:
            s += v
        return s

    rows, columns = np.nonzero(mask > 1e-6)
    rx, ry = profile(nx, ci), profile(ny, cj)
    on_mask = 0.0
    for j in range(rows.min(), rows.max() + 1):
        row = 0.0
        for i in range(columns.min(), columns.max() + 1):
            row += float(mask[j, i]) * rx[i]
        on_mask += ry[j] * row
    return on_mask / (total(rx) * total(ry))


def test_exact_tie_of_the_cut_back():
    """A fit that widens past the scale tested and keeps the share s of the
    coverage 0.5: at the coverage s exactly it is not cut (``<`` fails at the
    tie), one double above s it is."""
    inputs = _scene("cut_back_1")
    options = KERNEL | RECORD
    first = _check(inputs, **options, asp_min_mask_coverage=0.5)
    ((i, j, sigma, _),) = first[2]["components"]
    share = _coverage(np.asarray(inputs["mask"]), int(i), int(j), float(sigma))
    assert 0.5 <= share < 0.995
    tie = _check(inputs, **options, asp_min_mask_coverage=share)
    _assert_same(tie[2]["components"], first[2]["components"], "the fit at the tie")
    above = _check(
        inputs, **options, asp_min_mask_coverage=float(np.nextafter(share, 1.0))
    )
    ((i2, j2, cut, _),) = above[2]["components"]
    assert (i2, j2) == (i, j) and 0.0 < cut < sigma


def _values_masks(base, rr):
    """Masks with values that are not 0 or 1, made from the disk ``base``
    around a source (``rr``: the distance from its centre)."""
    peak = (slice(66, 71), slice(60, 65))
    halo = (rr >= 28) & (rr < 45)
    edits = {
        # NaN outside the bounding box (every convolution of the mask is NaN),
        # in the box off the disk, on the disk; infinities
        "nan_far": [((5, 110), np.nan)],
        "nan_box_corner": [((40, 34), np.nan)],
        "nan_in_disk": [((69, 63), np.nan)],
        "inf_in_disk": [((67, 61), np.inf)],
        "minus_inf_box_corner": [((42, 36), -np.inf)],
        # negative values (shares below 0 or above 1), values above 1
        "negative_hole_next_to_peak": [((slice(66, 69), slice(62, 65)), -1.0)],
        "negative_box_corners": [
            ((slice(40, 44), slice(34, 38)), -50.0),
            ((slice(92, 96), slice(86, 90)), -50.0),
        ],
        "negative_ring_outside": [((rr > 30) & (rr < 36), -1.0)],
        "spikes": [((60, 55), 1e3), ((74, 69), 37.0)],
        "denormal_halo": [((rr >= 28) & (rr < 40), 5e-324)],
        # the search threshold 0.99 exactly (out of the search region), one
        # double above (in) and below (out); the bounding-box threshold 1e-6
        # exactly (out of the box) and one double above (in)
        "patch_at_0.99": [(peak, 0.99)],
        "patch_above_0.99": [(peak, A99)],
        "patch_below_0.99": [(peak, B99)],
        "halo_at_1e-6": [(halo, 1e-6)],
        "halo_above_1e-6": [(halo, np.nextafter(1e-6, 1.0))],
    }
    masks = {"two_on_the_disk": 2.0 * base, "disk_above_0.99": A99 * base}
    for name, changes in edits.items():
        masks[name] = base.copy()
        for where, value in changes:
            masks[name][where] = value
    return masks


@cache
def _values_scene():
    ny, nx = 136, 124
    psf = _psf(ny, nx)
    sky = _gauss(ny, nx, 68, 62, 6.0, 2e-3)
    sky[68, 62] += 3e-3
    sky[78, 70] += 1.5e-3
    rr = np.hypot(*(np.mgrid[0:ny, 0:nx] - np.array([68, 62])[:, None, None]))
    base = _disk(ny, nx, 68, 62, 28.0)
    dirty = _noisy(sky, psf, 4120, base < 0.5, 5e-5)
    return psf, dirty, base, rr


VALUES = list(_values_masks(*_values_scene()[2:]))


@pytest.mark.parametrize("name", VALUES)
@pytest.mark.parametrize("base", ["kernel", "coral"])
def test_mask_values(name, base):
    psf, dirty, disk, rr = _values_scene()
    mask = _values_masks(disk, rr)[name]
    inputs = _plane(psf, dirty, mask, _threshold(dirty, disk, 0.08), 80)
    for coverage in (0.0, 0.995):
        _check(inputs, **BASES[base], **RECORD, asp_min_mask_coverage=coverage)


def test_residual_not_finite_outside_the_search_region():
    """Zero during the model update, given back afterwards (NaN payload and
    infinity included)."""
    psf, dirty, disk, _ = _values_scene()
    dirty = dirty.copy()
    nan = np.frombuffer(np.uint64(0x7FF800000000BEEF).tobytes(), np.float64)[0]
    dirty[3, 5], dirty[130, 7], dirty[30, 40] = nan, np.inf, -np.inf
    inputs = _plane(psf, dirty, disk, _threshold(dirty, disk, 0.08), 60)
    for base in BASES.values():
        _check(inputs, **base, **RECORD)


@cache
def _edge_scene():
    ny, nx = 119, 145
    psf = _psf(ny, nx, 1.8)
    sky = _gauss(ny, nx, 60, 72, 7.0, 1e-3) + _gauss(ny, nx, 3, 72, 4.0, 2e-3)
    sky += _gauss(ny, nx, 60, 142, 3.0, 2e-3) + _gauss(ny, nx, 116, 75, 5.0, 1e-3)
    sky += _gauss(ny, nx, 58, 2, 3.0, 2e-3)
    yy, xx = np.mgrid[0:ny, 0:nx]
    masks = {
        "cross": ((yy >= 52) & (yy <= 68)) | ((xx >= 64) & (xx <= 80)),
        "frame": (yy < 4) | (yy >= ny - 4) | (xx < 4) | (xx >= nx - 4),
        "band_left": xx < 24,
        "corner_disk": np.hypot(yy, xx) < 45,
        "top_right": (yy < 30) | (xx >= nx - 25),
        "x": (np.abs(yy * (nx / ny) - xx) < 7) | (np.abs(yy * (nx / ny) + xx - nx) < 7),
    }
    return psf, sky, {k: v.astype(np.float64) for k, v in masks.items()}


@pytest.mark.parametrize(
    "name", ["cross", "frame", "band_left", "corner_disk", "top_right", "x"]
)
def test_masks_at_the_edges(name):
    psf, sky, masks = _edge_scene()
    mask = masks[name]
    dirty = _noisy(sky, psf, 4131, mask < 0.5, 9e-5)
    inputs = _plane(psf, dirty, mask, _threshold(dirty, mask), 100)
    for base, coverage in (("kernel", 0.9), ("kernel", 0.999), ("coral", 0.995)):
        _check(inputs, **BASES[base], **RECORD, asp_min_mask_coverage=coverage)


@pytest.mark.parametrize(
    "pixels, coverages",
    [
        pytest.param([(80, 72)], (0.0, 1e-300, 0.003, 0.995), id="single_at_the_peak"),
        pytest.param([(83, 70)], (0.0, 0.003), id="single_off_the_peak"),
        pytest.param([(2, 50)], (0.0, 0.995), id="single_near_an_edge"),
        pytest.param([(80, 72), (80, 73)], (0.0, 0.01, 0.995), id="pair"),
        pytest.param(
            [(80, 72), (80, 73), (81, 72), (81, 73)], (0.0, 0.03, 0.995), id="block_2x2"
        ),
    ],
)
def test_masks_of_a_few_pixels(pixels, coverages):
    ny, nx = 160, 144
    psf = _psf(ny, nx)
    sky = _gauss(ny, nx, 85, 75, 3.0, 5e-4)
    sky[80, 72] += 5e-3
    sky[70, 60] += 2e-3
    rr = np.hypot(*(np.mgrid[0:ny, 0:nx] - np.array([80, 72])[:, None, None]))
    mask = np.zeros((ny, nx))
    for j, i in pixels:
        mask[j, i] = 1.0
    inputs = _plane(psf, _noisy(sky, psf, 4130, rr > 30, 3e-5), mask, 1e-4, 60)
    for base in ("kernel", "coral"):
        for coverage in coverages:
            _check(inputs, **BASES[base], **RECORD, asp_min_mask_coverage=coverage)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "shape", [(121, 175), (175, 121), (63, 191), (47, 33), (97, 131)]
)
def test_odd_sizes(shape, dtype):
    """Odd and Bluestein-sized axes, a disk mask with holes."""
    ny, nx = shape
    cy, cx = ny // 2 + 1, nx // 2 - 2
    psf = _psf(ny, nx, 2.1)
    sky = _gauss(ny, nx, cy, cx, 0.12 * min(ny, nx), 1e-3)
    sky[cy - 2, cx + 3] += 3e-3
    mask = _disk(ny, nx, cy, cx, 0.38 * min(ny, nx))
    mask[cy, cx + 1] = 0.0
    mask[cy - 4 : cy - 2, cx - 3 : cx] = 0.0
    dirty = _noisy(sky, psf, ny * nx, mask < 0.5, 8e-5)
    inputs = _plane(psf, dirty, mask, _threshold(dirty, mask, 0.08), 100, dtype)
    for base, coverage in (("kernel", 0.995), ("coral", 0.995), ("kernel", 0.95)):
        _check(inputs, **BASES[base], **RECORD, asp_min_mask_coverage=coverage)


@pytest.mark.parametrize("scene", ["tight", "holes", "soft_box_30", "point_patch"])
def test_float32(scene):
    inputs = _float32(_scene(scene))
    for coverage in (0.0, 0.9, 0.999):
        _check(inputs, **KERNEL, **RECORD, asp_min_mask_coverage=coverage)


def test_float32_mask_at_the_search_threshold():
    """0.99 in float32 is above 0.99 (in the search region), the float below
    it is not."""
    psf, dirty, disk, _ = _values_scene()
    for value in (np.float32(0.99), np.nextafter(np.float32(0.99), np.float32(0))):
        mask = disk.copy()
        mask[66:71, 60:65] = value
        inputs = _float32(_plane(psf, dirty, mask, _threshold(dirty, disk, 0.08), 80))
        _check(inputs, **CORAL, **RECORD)


def _mixed_model(shape, dtype, seed):
    """A model as an imaging run passes it from its second major cycle on,
    with the values whose updates are easy to get wrong: -0, the smallest
    subnormals of both signs, and values."""
    rng = np.random.default_rng(seed)
    tiny = np.finfo(dtype).smallest_subnormal
    pick = rng.random(shape)
    model = (1e-3 * rng.standard_normal(shape)).astype(dtype)
    model[pick < 0.3] = -0.0
    model[(pick >= 0.3) & (pick < 0.4)] = tiny
    model[(pick >= 0.4) & (pick < 0.5)] = -tiny
    return model


@pytest.mark.parametrize(
    "scene, base, dtype",
    [
        ("tight", "kernel", np.float64),
        ("cut_back_60", "kernel", np.float64),
        ("holes", "coral", np.float64),
        ("soft_box_30", "kernel", np.float32),
    ],
)
def test_models_that_are_not_zero_on_input(scene, base, dtype):
    """The model update adds every component to the model it is given, as the
    reference does (an addition of +-0 skipped would flip a -0 model pixel)."""
    psf, dirty, mask, threshold, max_iter = SCENES[scene]()
    model = _mixed_model(np.shape(dirty), dtype, len(scene))
    inputs = _plane(psf, dirty, mask, threshold, max_iter, dtype, model=model)
    _check(inputs, **BASES[base], **RECORD)


@pytest.mark.parametrize(
    "peaks",
    [
        pytest.param(
            [((60, 150), 1.0), ((150, 60), 1.0), ((150, 200), 1.0)], id="rows"
        ),
        pytest.param([((75, 75), 1.0), ((125, 125), -1.0)], id="min_max"),
        pytest.param([((0, 0), 1.0), ((100, 100), 1.0)], id="origin_tie"),
    ],
)
@pytest.mark.parametrize("masked", [True, False], ids=["ones", "no_mask"])
def test_exact_ties_of_the_peak_search(peaks, masked):
    """Equal maxima in different rows, |min| == |max|, and a tie with the
    (0, 0) start of a whole-image box: the per-row extrema, merged in scan
    order (on 2 threads at 256^2), keep the serial scan's first pixel."""
    n = 256
    dirty = np.zeros((n, n))
    for (j, i), value in peaks:
        dirty[j, i] = value
    psf = _gauss(n, n, n // 2, n // 2, 1.5)
    inputs = _plane(psf, dirty, np.ones((n, n)) if masked else None, 1e-3, 6)
    for base in ("kernel", "defaults"):
        _check(inputs, threads=(1, 2, 7), **BASES[base], **RECORD)


@pytest.mark.parametrize(
    "options",
    [
        {"asp_largest_scale": 0},
        {"asp_largest_scale": 3},
        {"asp_largest_scale": 15},
        {"asp_stop_point_mode": 2},
        {"asp_scale_normalization_method": 2},
        {"asp_fused_threshold": 2e-3},
        {"asp_fused_threshold": -1.0},
        {"asp_psf_width": 4.0},
        {"asp_stop_after_hogbom_run": True, "asp_fused_threshold": 2e-3},
        {"max_iter_divergence": 1, "divergence_rms_factor": 1.01},
        {"divergence_peak_factor": 1.0},
        {"asp_no_change_tolerance": 0.05},
        {"asp_significance": 3.0},
        {"asp_residual_component_positive_only": False},
        {"asp_residual_component_floor_at_threshold": False},
        {"asp_residual_component_switch_factor": 3.0},
        {"record_components": False, "record_iteration_history": False},
        {"verbose": True},
    ],
    ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()),
)
def test_options(options):
    inputs = _scene("tight")
    _check(inputs, **(CORAL | RECORD | options))


@pytest.mark.parametrize("threads", [0, -3, 2, 3, 12, 24])
def test_any_thread_count(threads):
    """Below 1 thread the transforms run on one, as in the reference."""
    _check(_scene("holes"), threads=(threads,), **CORAL, **RECORD)


def test_mapped_work_buffers():
    """A 768^2 plane, whose work buffers of 4 MB or more are mapped from the
    system and unmapped when they are freed (``asp_fft2.hpp``), twice in a
    row."""
    n = 768
    psf = _psf(n, n, 3.0)
    sky = _gauss(n, n, 380, 390, 24.0, 2e-3) + _gauss(n, n, 330, 430, 2.0, 5e-3)
    mask = _disk(n, n, 384, 384, 160.0)
    dirty = _noisy(sky, psf, n, mask < 0.5, 1e-4)
    inputs = _plane(psf, dirty, mask, _threshold(dirty, mask, 0.3), 8)
    for _ in range(2):
        _check(inputs, threads=(1, 6), **CORAL, **RECORD)


def test_unfused_scale_convolutions():
    """A 480 x 8192 plane whose mask box covers every row: the rows of the
    four scale convolutions of its mask (4 x 480 x 4097 complex values, 126 MB)
    and those of its search exceed the 96 MiB one fused column pass may hold,
    so each scale makes its own passes (convolveScalesRows)."""
    ny, nx = 480, 8192
    assert 4 * ny * (nx // 2 + 1) * 16 > 96 << 20
    rng = np.random.default_rng(ny)
    dirty = _gauss(ny, nx, 240, 3200, 12.0, 2e-3) + 2e-5 * rng.standard_normal((ny, nx))
    dirty += _gauss(ny, nx, 230, 3150, 2.0, 4e-3) + _gauss(ny, nx, 250, 3260, 2.0, 3e-3)
    mask = np.zeros((ny, nx))
    mask[:, 3000:3400] = 1.0
    inputs = _plane(_psf(ny, nx), dirty, mask, _threshold(dirty, mask), 12)
    for base in ("kernel", "coral"):
        _check(inputs, threads=(1, 6), **BASES[base], **RECORD)


# ---------------------------------------------------------------- huge scales
@cache
def _extended_scene(masked):
    """A 64^2 plane of extended positive emission and two points, so that the
    smoothed residual of a scale far wider than the image (the sum of the
    residual) beats the peak; a disk mask or none."""
    n = 64
    psf = _psf(n, n)
    sky = _gauss(n, n, 30, 34, 6.0, 1e-3)
    sky[20, 40] += 2e-3
    sky[44, 25] += 1.5e-3
    mask = _disk(n, n, 32, 32, 26.0) if masked else None
    quiet = _disk(n, n, 32, 32, 26.0) < 0.5
    dirty = _noisy(sky, psf, n, quiet, 2e-5)
    return _plane(psf, dirty, mask, _threshold(dirty, mask, 0.05), 12)


#: asp_psf_width values whose initial scales (1, 2, 4 and 8 widths) reach
#: beyond the range of int where the reference converts them to int (an
#: undefined conversion, then signed overflow): 1e8 the noise border of the
#: widest scale (3 x 8e8 px); 2e8 its mask border too (1.5 x 1.6e9 px), which
#: on x86 then zeroes nothing, so that scale is searched; 1e9 and 1e300 every
#: scale; 4e307 the widest scale infinite.
HUGE_WIDTHS = (1e8, 2e8, 1e9, 1e300, 4e307)


@pytest.mark.parametrize("masked", [False, True], ids=["no_mask", "disk"])
@pytest.mark.parametrize("base", ["kernel", "coral"])
@pytest.mark.parametrize("width", HUGE_WIDTHS)
def test_scale_sizes_beyond_the_range_of_int(width, base, masked):
    _check(_extended_scene(masked), **(BASES[base] | RECORD | {"asp_psf_width": width}))


def test_a_psf_wider_than_the_image():
    """A PSF flat to 2^-52 per pixel^2 (1 - 2^-52 r^2, exact in float64): its
    estimated width is about 1.1e8 px, so the noise border of the widest
    initial scale is beyond the range of int."""
    n = 9
    yy, xx = np.mgrid[0:n, 0:n]
    psf = 1.0 - 2.0**-52 * ((xx - n // 2) ** 2 + (yy - n // 2) ** 2)
    dirty = 1e-3 * np.cos(0.9 * xx + 0.4) * np.sin(0.7 * yy + 0.2)
    dirty[3, 5] += 4e-3
    inputs = _plane(psf, dirty, None, 0.0, 8)
    for base in ("kernel", "coral"):
        *_, result = _check(inputs, **BASES[base], **RECORD)
        assert 3 * 8 * result["psf_width"] >= 2**31


# ---------------------------------------------------------------- cubes
@cache
def _cube(dtype=np.float64, broadcast=False):
    """1 x 3 x 2 planes of the scenes above, each with its own threshold,
    iteration limit and PSF (or one broadcast PSF)."""
    names = [
        "full_wide",
        "too_small",
        "cut_back_1",
        "cut_hole_60",
        "point_hole_30",
        "soft_box_30",
    ]
    planes = [SCENES[name]() for name in names]
    n = 128

    def stack(k):
        return np.stack(
            [plane[k] if plane[k] is not None else np.ones((n, n)) for plane in planes]
        )

    psf = _psf(n, n)[None, None, None] if broadcast else stack(0).reshape(1, 3, 2, n, n)
    inputs = {
        "residual": stack(1).reshape(1, 3, 2, n, n).astype(dtype),
        "psf": np.ascontiguousarray(
            np.broadcast_to(psf, (1, 3, 1, n, n)) if broadcast else psf, dtype
        ),
        "mask": stack(2).reshape(1, 3, 2, n, n).astype(dtype),
        "threshold_per_cycle": np.array([1e-3, 1e-3, 1e-3, 2e-3, 1e-3, 5e-4]).reshape(
            1, 3, 2
        ),
        "max_iter_per_cycle_updated": np.array(
            [30, 60, 1, 60, 30, 30], np.int32
        ).reshape(1, 3, 2),
    }
    inputs["model"] = np.zeros_like(inputs["residual"])
    for value in inputs.values():
        value.flags.writeable = False
    return inputs


@pytest.mark.parametrize("coverage", [0.0, 0.995, 0.999])
@pytest.mark.parametrize("base", ["kernel", "coral"])
def test_cube(base, coverage):
    _check(
        _cube(),
        threads=(1, 2, 3, 6, 12),
        entry="clean_cube",
        **BASES[base],
        record_iteration_history=True,
        asp_min_mask_coverage=coverage,
    )


@pytest.mark.parametrize(
    "broadcast", [False, True], ids=["psf_per_plane", "broadcast_psf"]
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_cube_dtypes_and_psfs(dtype, broadcast):
    _check(_cube(dtype, broadcast), threads=(1, 5, 12), entry="clean_cube", **CORAL)


# ---------------------------------------------------------------- declined calls
def _nan(sign, payload):
    """The quiet NaN with this sign bit and payload."""
    bits = (0xFFF8000000000000 if sign else 0x7FF8000000000000) | payload
    return np.frombuffer(np.uint64(bits).tobytes(), np.float64)[0]


@cache
def _small_scene(ny, nx):
    psf = _psf(ny, nx, 1.5)
    sky = _gauss(ny, nx, ny // 2, nx // 2, 3.0, 2e-3)
    sky[ny // 2 - 3, nx // 2 + 4] += 3e-3
    mask = _disk(ny, nx, ny // 2, nx // 2, 0.35 * min(ny, nx))
    return psf, _noisy(sky, psf, ny * nx, mask < 0.5, 5e-5), mask


#: Just below 2^64 in float32, and so in float64.
BELOW_2_TO_THE_64 = float(np.nextafter(np.float32(2.0**64), np.float32(0.0)))
#: name -> (argument, [(pixel, value)]): the values that decline a call (the
#: review's cases: NaN and infinite PSF pixels, a PSF whose transform
#: overflows, a float32 model with NaNs; magnitudes of 2^64 and more), and
#: values that do not: the model update never computes with them, or they are
#: below 2^64.
DECLINED = {
    "psf_nan_and_minus_inf": ("psf", [((3, 5), np.nan), ((55, 50), -np.inf)]),
    "psf_negative_nan_payload": ("psf", [((40, 30), _nan(1, 0xBEEF))]),
    "psf_huge_pair": ("psf", [((3, 5), 1.5e308), ((55, 50), 1.6e308)]),
    "psf_2_to_the_64": ("psf", [((3, 5), 2.0**64)]),
    "model_nans": (
        "model",
        [((2, 3), np.nan), ((50, 4), np.inf), ((5, 52), _nan(1, 0x31))],
    ),
    "model_2_to_the_64": ("model", [((2, 3), -(2.0**64))]),
    "residual_nan_searched": ("residual", [((32, 34), np.nan)]),
    "residual_inf_searched": ("residual", [((30, 31), np.inf), ((33, 30), -np.inf)]),
    "residual_huge_outside": ("residual", [((2, 2), 1e30)]),
}
NOT_DECLINED = {
    "residual_nan_outside": ("residual", [((2, 3), _nan(1, 0xABC)), ((58, 4), np.inf)]),
    "residual_inf_at_0.99": ("residual", [((5, 6), np.inf)]),
    "mask_nan_and_inf": ("mask", [((32, 31), np.nan), ((20, 40), np.inf)]),
    "model_below_2_to_the_64": ("model", [((2, 3), -BELOW_2_TO_THE_64)]),
}


def _edited(name, shape=(64, 64), dtype=np.float64):
    """A small plane with the values of DECLINED[name] or NOT_DECLINED[name]."""
    argument, edits = (DECLINED | NOT_DECLINED)[name]
    psf, dirty, mask = _small_scene(*shape)
    arrays = {
        "psf": psf,
        "residual": dirty,
        "mask": mask.copy(),
        "model": np.zeros(shape),
    }
    arrays[argument] = np.array(arrays[argument], np.float64)
    for pixel, value in edits:
        arrays[argument][pixel] = value
    if name == "residual_inf_at_0.99":
        arrays["mask"][5, 6] = 0.99  # not above the search threshold: set aside
    with np.errstate(over="ignore"):  # 1.5e308 is infinite in float32
        return _plane(
            arrays["psf"],
            arrays["residual"],
            arrays["mask"],
            _threshold(dirty, mask, 0.05),
            12,
            dtype,
            model=arrays["model"],
        )


def _check_at_the_same_counts(inputs, threads, entry="clean", **options):
    """The fast entry's outcome is the reference's at the same thread count (a
    declined call runs the reference at its own count, whose NaN bits can
    depend on it)."""
    for n in threads:
        expected = _call(REFERENCE[entry], inputs, n, options)
        _assert_same(
            _call(FAST[entry], inputs, n, options), expected, f"{entry} at {n}"
        )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("shape", [(64, 64), (61, 67)])
@pytest.mark.parametrize("name", list(DECLINED))
def test_declined_calls_are_the_references(name, shape, dtype):
    inputs = _edited(name, shape, dtype)
    for base in ("kernel", "coral"):
        _check_at_the_same_counts(inputs, (1, 4, 7), **BASES[base], **RECORD)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("name", list(NOT_DECLINED))
def test_values_that_do_not_decline(name, dtype):
    """Values the model update never computes with: the fast kernel's bits."""
    inputs = _edited(name, (64, 64), dtype)
    for base in ("kernel", "coral"):
        _check(inputs, **BASES[base], **RECORD)


def _edited_cube(**edits):
    """``_cube()`` with ``{argument: [(index, value)]}`` set (copies)."""
    cube = dict(_cube())
    for argument, values in edits.items():
        array = np.array(cube[argument])
        for index, value in values:
            array[index] = value
        array.flags.writeable = False
        cube[argument] = array
    return cube


def _nan_plane_cube():
    """``_cube()`` with a NaN in the search region of plane (0, 1, 1) and no
    iterations there, as ``deconvolution.asp_clean`` makes such a plane."""
    return _edited_cube(
        residual=[((0, 1, 1, 64, 70), np.nan)],
        mask=[((0, 1, 1, 64, 70), 1.0)],
        max_iter_per_cycle_updated=[((0, 1, 1), 0)],
    )


def test_a_cube_with_one_declined_plane():
    """A NaN PSF pixel in plane (0, 1, 0): that plane alone runs the reference,
    with the threads the reference's driver gives it (1 here: 1, 2 and 5
    threads of the cube for 6 planes), the others the fast kernel: the
    reference's bits at 1 thread."""
    cube = _edited_cube(psf=[((0, 1, 0, 3, 5), np.nan)])
    options = KERNEL | {"record_iteration_history": True}
    expected = _call(REFERENCE["clean_cube"], cube, 1, options)
    for threads in (1, 2, 5):
        got = _call(FAST["clean_cube"], cube, threads, options)
        _assert_same(got, expected, f"clean_cube at {threads} threads")


def test_a_cube_with_a_nan_plane_as_deconvolution_makes_it():
    """Its other planes run the fast kernel; the NaN plane, cleaned by the
    reference with no iterations, has the same bits at any thread count."""
    _check(
        _nan_plane_cube(),
        threads=(1, 2, 3, 6, 12),
        entry="clean_cube",
        **CORAL,
        record_iteration_history=True,
    )


def test_only_the_declined_planes_run_the_reference(monkeypatch):
    """The reference's clean_cube gets each plane that needs it, alone: a
    1 x 1 x 1 cube with the plane's own iteration limit and the threads the
    reference's driver gives a plane (12 // 6 for 6 planes)."""
    module = importlib.import_module(
        "astroviper.processing_functions.imaging.deconvolvers.aspclean._aspclean_ext"
    )
    original, calls = module.clean_cube, []

    def clean_cube(**arguments):
        calls.append(
            (
                arguments["residual"].shape,
                arguments["max_iter_per_cycle_updated"].tolist(),
                arguments["processing_function_threads"],
            )
        )
        return original(**arguments)

    cube = _nan_plane_cube()
    expected = _call(original, cube, 4, CORAL)
    monkeypatch.setattr(module, "clean_cube", clean_cube)
    _assert_same(_call(FAST["clean_cube"], cube, 12, CORAL), expected, "clean_cube")
    assert calls == [((1, 1, 1, 128, 128), [[[0]]], 2)]


def _overlapping_cube(name):
    """The NaN-plane cube with arrays that share memory: one array as the
    residual and the model, as the residual and the PSF, or a model one row
    into the residual's buffer."""
    cube = _nan_plane_cube()
    residual = np.array(cube["residual"])
    if name == "model_is_residual":
        return cube | {"residual": residual, "model": residual}
    if name == "psf_is_residual":
        return cube | {
            "residual": residual,
            "psf": residual,
            "model": np.zeros_like(residual),
        }
    buffer = np.zeros(residual.size + 128)
    shared = buffer[: residual.size].reshape(residual.shape)
    shared[...] = residual
    return cube | {"residual": shared, "model": buffer[128:].reshape(residual.shape)}


@pytest.mark.parametrize(
    "name, threads, path",
    [
        ("model_is_residual", 3, "fast"),
        ("psf_is_residual", 3, "declined"),
        # the reference's planes race on the overlap at more than 1 thread
        ("model_overlaps_residual", 1, "declined"),
    ],
)
def test_cube_arrays_that_overlap(name, threads, path, monkeypatch, tmp_path):
    """A model that is the residual itself: the declined plane runs alone, the
    others are updated in place. A PSF that is the residual, or a model one
    row into the residual's buffer: the reference runs the whole cube."""
    options = CORAL | {"record_iteration_history": True}

    def run(function):
        cube = _overlapping_cube(name)
        result = function(**cube, processing_function_threads=threads, **options)
        return np.array(cube["residual"]), np.array(cube["model"]), result

    expected = run(REFERENCE["clean_cube"])
    _assert_same(run(FAST["clean_cube"]), expected, name)
    assert _trace_of(monkeypatch, tmp_path, "clean_cube", run) == ("fast", path, None)


def test_a_model_that_overlaps_the_residual(monkeypatch, tmp_path):
    """A model one row into the residual's buffer, at 4 threads: the
    reference's bits (it updates the two in one order at any thread count)."""
    inputs = _scene("holes")

    def run(function, threads=4):
        n = inputs["residual"].shape[1]
        buffer = np.zeros(inputs["residual"].size + n)
        residual = buffer[: inputs["residual"].size].reshape(inputs["residual"].shape)
        residual[...] = inputs["residual"]
        model = buffer[n:].reshape(residual.shape)
        arguments = inputs | {"residual": residual, "model": model}
        result = function(
            **arguments, processing_function_threads=threads, **KERNEL, **RECORD
        )
        return np.array(residual), np.array(model), result

    expected = run(REFERENCE["clean"], 1)
    _assert_same(run(REFERENCE["clean"]), expected, "the reference at 4 threads")
    _assert_same(run(FAST["clean"]), expected, "clean")
    assert _trace_of(monkeypatch, tmp_path, "clean", run) == ("fast", "declined", None)


def test_cube_without_planes_still_checks_the_controls():
    n = 16
    inputs = {
        "residual": np.zeros((0, 1, 1, n, n)),
        "psf": np.zeros((0, 1, 1, n, n)),
        "model": np.zeros((0, 1, 1, n, n)),
        "threshold_per_cycle": np.zeros((0, 1, 1)),
        "max_iter_per_cycle_updated": np.zeros((0, 1, 1), np.int32),
    }
    _check(inputs, entry="clean_cube")
    outcome = _check(inputs, entry="clean_cube", asp_min_mask_coverage=1.0)
    assert outcome[0] is ValueError


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("ny, nx", [(0, 0), (0, 5), (5, 0)])
def test_empty_planes(entry, ny, nx):
    """The reference reads past the end of an empty PSF to estimate its width
    (and the process dies); the fast kernel takes the floor width, 1, so it
    returns what the reference returns with asp_psf_width=1."""
    shape = (ny, nx) if entry == "clean" else (1, 1, 1, ny, nx)
    inputs = {k: np.zeros(shape) for k in ("residual", "psf", "model")}
    if entry == "clean_cube":
        inputs["threshold_per_cycle"] = np.zeros((1, 1, 1))
        inputs["max_iter_per_cycle_updated"] = np.full((1, 1, 1), 5, np.int32)
    expected = _call(REFERENCE[entry], inputs, 1, CORAL | {"asp_psf_width": 1.0})
    for threads in (1, 4):
        got = _call(FAST[entry], inputs, threads, CORAL)
        _assert_same(got, expected, f"{entry} at {threads} threads")


# ---------------------------------------------------------------- the entries
def test_entries_are_the_references():
    """The same names, parameters (types, defaults, order: asp_min_mask_coverage
    = 0.995 last) and documentation, as pybind11 writes them."""
    for entry in ENTRIES:
        assert FAST[entry].__name__ == REFERENCE[entry].__name__ == entry
        assert FAST[entry].__doc__ == REFERENCE[entry].__doc__
        signature = FAST[entry].__doc__.splitlines()[0]
        assert re.search(
            r", asp_min_mask_coverage: [^,]+ = 0\.995\) -> dict$", signature
        )
    assert ext.ISA == "generic"


REPO = Path(__file__).parents[3]


def _function(path, signature):
    """The text of the function of ``path`` that starts with ``signature``, to
    its closing brace (the first one at the start of a line)."""
    text = (REPO / path).read_text()
    start = text.index(signature)
    return text[start : text.index("\n}\n", start) + 3]


def test_psf_gaussian_width_is_a_verbatim_copy_of_the_reference():
    signature = "double psf_gaussian_width(const double* psf, int nx, int ny) {"
    reference = _function(
        "src/astroviper/processing_functions/imaging/deconvolvers/aspclean/src/asp_clean.cpp",
        signature,
    )
    fast = _function(
        "src/astroviper/_kernels/fast/aspclean/src/asp_clean_fast.cpp", signature
    )
    assert fast == reference
    assert reference.count("\n") > 60


def test_an_unknown_isa_override_fails_the_import(run_python):
    done = run_python(
        "-c",
        "import astroviper._kernels.fast.aspclean",
        check=False,
        ASTROVIPER_KERNELS_ISA="avx2",
    )
    assert done.returncode != 0
    assert "ASTROVIPER_KERNELS_ISA=avx2: expected 'generic'" in done.stderr


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "name, value",
    [
        ("asp_min_mask_coverage", -0.1),
        ("asp_min_mask_coverage", 1.0),
        ("asp_min_mask_coverage", np.nan),
        ("asp_min_mask_coverage", np.inf),
        ("asp_min_mask_coverage", -np.inf),
        ("asp_min_mask_coverage", 1.5),
        ("asp_min_mask_coverage", -5e-324),
        ("asp_min_mask_coverage", True),
        ("asp_min_mask_coverage", "0.9"),
        ("asp_min_mask_coverage", None),
        ("asp_psf_width", 0.0),
        ("asp_psf_width", np.nan),
        ("asp_largest_scale", -2),
        ("asp_stop_point_mode", 0),
        ("asp_scale_normalization_method", 3),
    ],
)
def test_refused_values_raise_the_references_errors(entry, name, value):
    """The checks of the reference's controls, and pybind11's conversions (a
    ``TypeError`` repeats the call's arguments, so both run at the same thread
    count)."""
    inputs = _scene("cut_back_1") if entry == "clean" else _cube()
    outcome = _check(inputs, threads=(REFERENCE_THREADS,), entry=entry, **{name: value})
    assert outcome[0] in (ValueError, TypeError)


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -0.0,
        5e-324,
        float(np.nextafter(1.0, 0.0)),
        np.float32(0.995),
        False,
        0,
        1e-310,
    ],
    ids=repr,
)
def test_coverage_boundaries(value):
    _check(
        _scene("cut_hole_60"),
        threads=(1, 3),
        **KERNEL,
        **RECORD,
        asp_min_mask_coverage=value,
    )


def _broken(inputs, **changes):
    """``inputs`` with ``changes`` (``None`` removes an argument)."""
    return {k: v for k, v in (inputs | changes).items() if v is not None}


def _change_id(change):
    ((name, value),) = change.items()
    if value is None:
        return f"{name}-missing"
    if isinstance(value, np.ndarray):
        return f"{name}-{value.dtype}{list(value.shape)}"
    return f"{name}-{type(value).__name__}"


@pytest.mark.parametrize(
    "change",
    [
        {"residual": np.zeros((128, 128), np.int64)},
        {"residual": np.zeros((128, 128, 1))},
        {"psf": np.zeros((128, 128), np.float32)},
        {"psf": np.zeros((128, 64))},
        {"model": np.zeros((128, 64))},
        {"mask": np.zeros((64, 128))},
        {"mask": np.zeros((128, 128), np.float32)},
        {"psf": None},
        {"threshold_per_cycle": "1e-3"},
        {"max_iter_per_cycle_updated": 2.5},
        {"gain": [0.1]},
    ],
    ids=_change_id,
)
def test_plane_checks_raise_the_references_errors(change):
    _check(_broken(_scene("cut_back_1"), **change), threads=(REFERENCE_THREADS,))


def _relaid(array, layout):
    if layout == "read_only":
        array = array.copy()
        array.flags.writeable = False
        return array
    if layout == "strided":
        wide = np.zeros(array.shape[:-1] + (2 * array.shape[-1],), array.dtype)
        wide[..., ::2] = array
        return wide[..., ::2]
    return np.asfortranarray(array)


@pytest.mark.parametrize(
    "name, layout",
    [("residual", "read_only"), ("model", "read_only")]
    + [
        (n, a)
        for n in ("residual", "model", "psf", "mask")
        for a in ("strided", "fortran")
    ],
)
def test_layouts_raise_the_references_errors(name, layout):
    """The buffers are used in place, never copied: a read-only residual or
    model, or any array that is not C-contiguous, raises the reference's
    error before anything is written."""
    inputs = _scene("cut_back_1")
    outcomes = []
    for function in (REFERENCE["clean"], FAST["clean"]):
        arguments = inputs | {k: inputs[k].copy() for k in ("residual", "model")}
        arguments[name] = _relaid(inputs[name], layout)
        try:
            function(**arguments)
        except RuntimeError as e:
            outcomes.append(str(e))
        assert not arguments["model"].any()
    assert len(outcomes) == 2 and outcomes[0] == outcomes[1]


@pytest.mark.parametrize(
    "change",
    [
        {"psf": np.zeros((1, 3, 3, 128, 128))},
        {"psf": np.zeros((1, 3, 2, 128, 128), np.float32)},
        {"psf": np.zeros((1, 3, 2, 128))},
        {"mask": np.zeros((1, 3, 2, 128, 64))},
        {"threshold_per_cycle": np.zeros((1, 3, 1))},
        {"threshold_per_cycle": np.zeros((1, 3, 2), np.float32)},
        {"max_iter_per_cycle_updated": np.zeros((1, 3, 2), np.int64)},
        {"max_iter_per_cycle_updated": None},
        {"residual": np.zeros((3, 2, 128, 128))},
    ],
    ids=_change_id,
)
def test_cube_checks_raise_the_references_errors(change):
    _check(
        _broken(_cube(), **change),
        threads=(REFERENCE_THREADS,),
        entry="clean_cube",
        **KERNEL,
    )


def test_pickled_arrays():
    """Arrays rebuilt by pickle (as Dask ships them) are read in place."""
    inputs = _scene("holes")
    rebuilt = pickle.loads(pickle.dumps(inputs, protocol=5))
    expected = _call(REFERENCE["clean"], inputs, 2, CORAL | RECORD)
    _assert_same(_call(FAST["clean"], rebuilt, 6, CORAL | RECORD), expected, "pickled")


def _misaligned(array):
    """A copy of ``array`` at an address that is not a multiple of its item size."""
    raw = np.empty(array.nbytes + array.itemsize, np.uint8)
    view = raw[1 : 1 + array.nbytes].view(array.dtype).reshape(array.shape)
    view[...] = array
    assert not view.flags.aligned
    return view


@pytest.mark.parametrize("entry", ENTRIES)
def test_misaligned_arrays(entry):
    """Every array at an address that is not a multiple of its item size (as
    Dask may ship them): read through aligned copies, the residual and the
    model written back in place, the reference's bits. (The reference reads
    such arrays in place, which is undefined behaviour: it gets aligned ones.)"""
    inputs = _scene("holes") if entry == "clean" else _cube()
    options = CORAL | {"record_iteration_history": True}
    expected = _call(REFERENCE[entry], inputs, 2, options)
    shifted = {
        k: _misaligned(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()
    }
    result = FAST[entry](**shifted, processing_function_threads=5, **options)
    _assert_same((shifted["residual"], shifted["model"], result), expected, entry)


MISALIGNED_PEAK = """
import sys
import numpy as np
from astroviper._kernels import impl


def status(key):
    with open("/proc/self/status") as f:
        return next(int(line.split()[1]) * 1024 for line in f if line.startswith(key))


def reset_peak():  # VmHWM := VmRSS
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")


def moved(a):
    raw = np.empty(a.nbytes + 1, np.uint8)
    view = raw[1 : 1 + a.nbytes].view(a.dtype).reshape(a.shape)
    view[...] = a
    return view


clean = impl("asp", "clean", "fast")
n = 1024
yy, xx = np.mgrid[0:n, 0:n]
psf = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 8.0)
dirty = np.exp(-((xx - 500) ** 2 + (yy - 530) ** 2) / 800.0) + 1e-3 * np.cos(0.3 * xx * yy)
mask = (np.hypot(xx - 512.0, yy - 512.0) < 300).astype(float)
arrays = {"residual": dirty, "psf": psf, "model": np.full((n, n), 0.0), "mask": mask}
if sys.argv[1] == "misaligned":
    arrays = {k: moved(v) for k, v in arrays.items()}
small = {k: np.ascontiguousarray(v[480:544, 480:544]) for k, v in arrays.items()}
clean(**small, gain=0.3, max_iter_per_cycle_updated=2, processing_function_threads=2)
start = status("VmRSS:")
reset_peak()
clean(**arrays, gain=0.3, max_iter_per_cycle_updated=8, processing_function_threads=2)
print("RESULT", status("VmHWM:") - start)
"""


@pytest.mark.skipif(not os.path.exists("/proc/self/clear_refs"), reason="Linux /proc")
@pytest.mark.skipif(
    "asan" in os.environ.get("LD_PRELOAD", ""),
    reason="AddressSanitizer's redzones and quarantine change the peaks",
)
def test_misaligned_arrays_copy_only_the_residual_and_the_model(run_python):
    """The PSF and the mask are read in place at any address: misaligned
    arrays raise a call's peak memory by the copies of the residual and the
    model (2 planes), not of all four arrays (fresh processes, 1024^2)."""
    peaks = {
        mode: int(run_python("-c", MISALIGNED_PEAK, mode).rpartition("RESULT ")[2])
        for mode in ("aligned", "misaligned")
    }
    plane = 1024 * 1024 * 8
    assert peaks["aligned"] > plane, peaks  # the measurement sees the call's buffers
    assert peaks["misaligned"] - peaks["aligned"] < 3 * plane, peaks


@pytest.mark.parametrize("shifted", [False, True], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("alias", ["model", "psf", "mask"])
def test_arguments_that_are_the_residual(alias, shifted):
    """One array passed as the residual and as the model, the PSF or the mask:
    the kernel sees the updates of it, as the reference does (given an aligned
    array: it reads a misaligned one in place, which is undefined). As a mask,
    the residual has a value that is not finite outside its search region,
    which the model update sets aside: zero, then its value again."""
    psf, dirty, threshold = _scene("holes")["psf"], _scene("holes")["residual"], 1e-3
    scale = 1000.0 if alias == "mask" else 1.0
    dirty = scale * np.array(dirty)
    if alias == "mask":
        dirty[2, 3] = np.nan
    options = KERNEL | RECORD

    def run(function, array):
        arguments = {
            "psf": psf,
            "model": np.zeros_like(array),
            "mask": np.array([]),
            "threshold_per_cycle": scale * threshold,
            "max_iter_per_cycle_updated": 40,
        }
        arguments |= {"residual": array, alias: array}
        result = function(**arguments, processing_function_threads=3, **options)
        return np.array(array), np.array(arguments["model"]), result

    expected = run(REFERENCE["clean"], dirty.copy())
    array = _misaligned(dirty) if shifted else dirty.copy()
    _assert_same(run(FAST["clean"], array), expected, alias)


@cache
def _mapped_plane():
    """A 512^2 plane: its padded PSF spectrum (4.7 MB) is mapped from the
    system."""
    n = 512
    psf = _psf(n, n, 2.5)
    sky = _gauss(n, n, 250, 260, 16.0, 2e-3) + _gauss(n, n, 220, 290, 2.0, 5e-3)
    mask = _disk(n, n, 256, 256, 110.0)
    dirty = _noisy(sky, psf, n, mask < 0.5, 1e-4)
    return _plane(psf, dirty, mask, _threshold(dirty, mask, 0.3), 10)


def _shifted_call(function, inputs, threads, options):
    """``_call`` with every array of the call (fresh copies) at a misaligned
    address."""
    arguments = {
        k: _misaligned(v) if isinstance(v, np.ndarray) else v for k, v in inputs.items()
    }
    try:
        result = function(**arguments, processing_function_threads=threads, **options)
    except Exception as e:  # the reference's exception is the expected outcome
        return type(e), str(e)
    return np.array(arguments["residual"]), np.array(arguments["model"]), result


def test_concurrent_calls():
    """Python threads in the fast kernel at once (the GIL is released): planes
    and cubes, a plane whose large buffers are mapped, misaligned arrays, a
    declined call, cubes with a plane that the reference cleans (from the
    cube's worker threads, which take the GIL for it) and calls that raise;
    each gives the outcome the reference gives at the same thread count (on
    aligned arrays: it reads misaligned ones in place, which is undefined)."""
    jobs = {  # name -> (entry, inputs, threads, options, misaligned)
        "holes": ("clean", _scene("holes"), 3, KERNEL | RECORD, False),
        "negcut": ("clean", _scene("negcut"), 1, KERNEL | RECORD, False),
        "point_patch": ("clean", _scene("point_patch"), 2, CORAL | RECORD, False),
        "cube": (
            "clean_cube",
            _cube(),
            4,
            CORAL | {"record_iteration_history": True},
            False,
        ),
        "mapped": ("clean", _mapped_plane(), 2, CORAL | RECORD, False),
        "misaligned": ("clean", _scene("cut_back_60"), 2, KERNEL | RECORD, True),
        "misaligned_cube": ("clean_cube", _cube(), 3, KERNEL, True),
        "nan_plane_cube": (
            "clean_cube",
            _nan_plane_cube(),
            3,
            CORAL | {"record_iteration_history": True},
            False,
        ),
        "misaligned_nan_plane_cube": ("clean_cube", _nan_plane_cube(), 2, KERNEL, True),
        "declined": (
            "clean",
            _edited("psf_nan_and_minus_inf"),
            2,
            CORAL | RECORD,
            False,
        ),
        "raises": (
            "clean",
            _scene("cut_back_1"),
            2,
            {"asp_min_mask_coverage": 1.0},
            False,
        ),
        "cube_raises": ("clean_cube", _cube(), 2, {"asp_largest_scale": -2}, False),
    }
    expected = {
        name: _call(REFERENCE[entry], inputs, threads, options)
        for name, (entry, inputs, threads, options, _) in jobs.items()
    }
    got, errors = [], []

    def worker(name):
        try:
            entry, inputs, threads, options, misaligned = jobs[name]
            call = _shifted_call if misaligned else _call
            got.append((name, call(FAST[entry], inputs, threads, options)))
        except BaseException as e:  # reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in list(jobs) * 2]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert sorted(name for name, _ in got) == sorted(list(jobs) * 2)
    for name, outcome in got:
        _assert_same(outcome, expected[name], name)


def _spin_rate(during):
    """Iterations per second of a Python loop in another thread while
    ``during()`` runs in this one."""
    stop, count = threading.Event(), []

    def spin():
        n = 0
        while not stop.is_set():
            n += 1
        count.append(n)

    spinner = threading.Thread(target=spin)
    spinner.start()
    start = time.perf_counter()
    during()
    elapsed = time.perf_counter() - start
    stop.set()
    spinner.join()
    return count[0] / elapsed


@pytest.mark.parametrize("entry", ENTRIES)
def test_calls_release_the_gil(entry):
    """A Python thread keeps running while a call runs on 2 threads; one that
    held the GIL would stop it."""
    inputs = _mapped_plane() if entry == "clean" else _cube()
    idle = _spin_rate(lambda: time.sleep(0.3))
    busy = _spin_rate(lambda: _call(FAST[entry], inputs, 2, CORAL))
    assert busy > 0.25 * idle


@pytest.mark.parametrize("entry", ENTRIES)
def test_calls_use_at_most_their_threads(entry):
    """CPU time per wall second of a call at 2 threads stays below 2.5: the
    calling thread works one share, the pool the others (a call that ran on
    more threads than processing_function_threads would exceed it on a
    machine with free cores)."""
    inputs = _mapped_plane() if entry == "clean" else _cube()
    cpu, wall = time.process_time(), time.perf_counter()
    _call(FAST[entry], inputs, 2, CORAL)
    assert (time.process_time() - cpu) / (time.perf_counter() - wall) < 2.5


ONE_THREAD = """
import os
import numpy as np
from astroviper._kernels import impl

n = 96
yy, xx = np.mgrid[0:n, 0:n]
psf = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 8.0)
dirty = np.exp(-((xx - 50) ** 2 + (yy - 44) ** 2) / 50.0) + 1e-3 * np.cos(xx * yy)
before = len(os.listdir("/proc/self/task"))
for entry, args in (("clean", (dirty.copy(), psf, np.zeros_like(dirty))),
                    ("clean_cube", (dirty[None, None, None].copy(), psf[None, None, None],
                                    np.zeros((1, 1, 1, n, n))))):
    kwargs = {} if entry == "clean" else {
        "threshold_per_cycle": np.zeros((1, 1, 1)),
        "max_iter_per_cycle_updated": np.full((1, 1, 1), 30, np.int32)}
    impl("asp", entry, "fast")(*args, gain=0.3, processing_function_threads=1,
                               asp_residual_components=True, asp_scale_significance=True,
                               **kwargs)
print("RESULT", before, len(os.listdir("/proc/self/task")))
"""


@pytest.mark.skipif(not os.path.isdir("/proc/self/task"), reason="Linux /proc")
def test_a_call_at_one_thread_starts_no_thread(run_python):
    before, after = run_python("-c", ONE_THREAD).rpartition("RESULT ")[2].split()
    assert after == before


FP_STATE = """
import ctypes, hashlib, json
import numpy as np
from astroviper._kernels import impl

libm = ctypes.CDLL("libm.so.6")
mode = ctypes.create_string_buffer(8)  # glibc x86-64 femode_t: x87 control word, MXCSR at 4


def mxcsr(value=None):
    libm.fegetmode(mode)
    if value is not None:
        mode[4:8] = value.to_bytes(4, "little")
        libm.fesetmode(mode)
    return int.from_bytes(mode.raw[4:8], "little")


# a source near a corner: the Gaussian profiles of its scale components, and
# their products in the model update, reach the subnormal range in the image
n = 256
yy, xx = np.mgrid[0:n, 0:n]
psf = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 2.0)
sky = np.exp(-((xx - 24) ** 2 + (yy - 20) ** 2) / 50.0)
dirty = np.real(np.fft.ifft2(np.fft.fft2(sky) * np.fft.fft2(np.fft.ifftshift(psf))))
dirty += 1e-5 * np.cos(0.7 * xx) * np.sin(0.3 * yy)
cube = {"threshold_per_cycle": np.zeros((1, 1, 2)),
        "max_iter_per_cycle_updated": np.full((1, 1, 2), 12, np.int32)}
inputs = {"clean": (dirty, psf, {"max_iter_per_cycle_updated": 12}),
          "clean_cube": (np.stack([dirty, dirty[::-1]])[None, None], psf[None, None, None], cube)}
functions = {(k, e): impl("asp", e, k) for k in ("fast", "ref") for e in inputs}


def run(kernel, entry, threads, state):
    residual, psf_, extra = inputs[entry]
    r, m = residual.copy(), np.zeros_like(residual)
    mxcsr(state)
    try:
        functions[kernel, entry](r, psf_, m, gain=0.3, processing_function_threads=threads, **extra)
    finally:
        mxcsr(default)
    return hashlib.sha256(r.tobytes() + m.tobytes()).hexdigest()


default = mxcsr()
for kernel in ("fast", "ref"):  # both pools start in the default state
    run(kernel, "clean", 4, default)
states = {"default": default, "ftz_daz": default | 0x8040, "toward_zero": default | 0x6000}
print("RESULT", json.dumps({f"{s} {k} {e} {t}": run(k, e, t, v) for s, v in states.items()
                           for k in ("fast", "ref") for e in inputs for t in (1, 4)}))
"""


@pytest.mark.skipif(
    platform.machine() not in ("x86_64", "AMD64")
    or not sys.platform.startswith("linux"),
    reason="sets MXCSR through glibc's femode_t",
)
def test_floating_point_state_of_the_calling_thread(run_python):
    """Flush-to-zero and denormals-are-zero, or rounding toward zero, set on
    the calling thread after the pools have started (whose threads keep the
    default): the pool's tasks run in the calling thread's state, so the bits
    do not depend on the thread count; with FTZ and DAZ they are the
    reference's at 1 thread and here at 4 too (its transforms, on its pool,
    see no subnormal number)."""
    r = json.loads(run_python("-c", FP_STATE).rpartition("RESULT ")[2])
    runs = (("fast", 1), ("fast", 4), ("ref", 1), ("ref", 4))
    for entry in ENTRIES:
        bits = {
            state: {r[f"{state} {k} {entry} {t}"] for k, t in runs}
            for state in ("default", "ftz_daz")
        }
        assert len(bits["default"]) == 1, entry
        assert len(bits["ftz_daz"]) == 1, entry
        assert bits["ftz_daz"] != bits["default"], f"{entry}: no subnormal number"
        assert r[f"toward_zero fast {entry} 1"] == r[f"toward_zero fast {entry} 4"]


# ---------------------------------------------------------------- the switch
def _trace_of(monkeypatch, tmp_path, entry, call):
    """``(impl, path, error)`` of the trace line of ``call(function)``, with
    ``function`` the entry hooked by the kernel switch with ``asp=fast``."""
    trace = tmp_path / "trace"
    monkeypatch.setenv("ASTROVIPER_KERNELS", "asp=fast")
    monkeypatch.setenv("ASTROVIPER_KERNELS_TRACE", str(trace))
    call(dispatch("asp", REFERENCE[entry]))
    (path,) = trace.iterdir()
    record = json.loads(path.read_text().splitlines()[-1])
    return record["impl"], record["path"], record["error"]


def _traced(monkeypatch, tmp_path, entry, inputs, threads=2, **options):
    """``(impl, path, error)`` of the trace line of one call through the
    kernel switch with ``asp=fast``."""
    return _trace_of(
        monkeypatch, tmp_path, entry, lambda f: _call(f, inputs, threads, options)
    )


@pytest.mark.parametrize("name", list(DECLINED) + list(NOT_DECLINED))
def test_traced_paths(name, monkeypatch, tmp_path):
    path = "declined" if name in DECLINED else "fast"
    traced = _traced(monkeypatch, tmp_path, "clean", _edited(name), **KERNEL)
    assert traced == ("fast", path, None)


def test_traced_paths_of_cubes_and_refused_controls(monkeypatch, tmp_path):
    """A cube with a plane that needs the reference runs that plane alone and
    keeps the path "fast"; one whose every plane needs it declines as a
    whole; a refused control raises the reference's error before anything
    declines."""
    cube = _edited_cube(psf=[((0, 2, 0, 60, 60), np.inf)])
    every = _edited_cube(
        psf=[((0, f, p, 60, 60), np.inf) for f in range(3) for p in range(2)]
    )
    assert _traced(monkeypatch, tmp_path, "clean_cube", _cube(), **KERNEL) == (
        "fast",
        "fast",
        None,
    )
    assert _traced(monkeypatch, tmp_path, "clean_cube", cube, **KERNEL) == (
        "fast",
        "fast",
        None,
    )
    assert _traced(monkeypatch, tmp_path, "clean_cube", every, **KERNEL) == (
        "fast",
        "declined",
        None,
    )
    refused = KERNEL | {"asp_min_mask_coverage": 1.0}
    for entry, inputs in (
        ("clean", _edited("psf_nan_and_minus_inf")),
        ("clean_cube", cube),
    ):
        assert _traced(monkeypatch, tmp_path, entry, inputs, **refused) == (
            "fast",
            "fast",
            "ValueError",
        )


PASS_THROUGH = """
import json, os, sys
import numpy as np
from astroviper.processing_functions.imaging.deconvolution import asp_clean

n = 96
yy, xx = np.mgrid[0:n, 0:n]
psf = np.exp(-((xx - n // 2) ** 2 + (yy - n // 2) ** 2) / 8.0)
sky = np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / (2 * 11.0**2))
sky[40, 52] += 2.0
dirty = np.real(np.fft.ifft2(np.fft.fft2(sky) * np.fft.fft2(np.fft.ifftshift(psf))))
mask = np.zeros((n, n), bool)
mask[34:63, 34:63] = True
out = {}
for coverage in (None, 0.0, 0.9, 0.999, 1.0, -0.1, float("nan")):
    for kernels in ("asp=fast", "asp=ref"):
        os.environ["ASTROVIPER_KERNELS"] = kernels
        params = {"gain": 0.3, "max_iter": 40, "threshold": 1e-3}
        if coverage is not None:
            params["asp_min_mask_coverage"] = coverage
        residual = np.stack([dirty, -0.5 * dirty])[None, None].copy()
        model = np.zeros_like(residual)
        try:
            # input_checked: the refused coverages reach the kernels, not config_check.
            result = asp_clean(residual, psf[None, None, None].copy(), model, params,
                               processing_function_threads=3,
                               mask_cube=np.stack([mask, mask])[None, None],
                               input_checked=True)
        except ValueError as e:
            out[f"{coverage}-{kernels}"] = ["ValueError", str(e)]
            continue
        out[f"{coverage}-{kernels}"] = [residual.tobytes().hex(), model.tobytes().hex(),
            {k: np.asarray(v).tobytes().hex() for k, v in sorted(result.items())}]
print("RESULT", json.dumps(out))
"""


def test_asp_clean_passes_the_coverage_to_the_fast_kernel(run_python, tmp_path):
    """``deconvolution.asp_clean`` through the hooked shim: each coverage
    (the default, 0, 0.9, 0.999) reaches the fast kernel, which gives the
    reference's bytes; 1.0, -0.1 and NaN raise the reference's error."""
    trace = tmp_path / "trace"
    out = run_python(
        "-c",
        PASS_THROUGH,
        ASTROVIPER_KERNELS="asp=fast",
        ASTROVIPER_KERNELS_TRACE=str(trace),
    )
    results = json.loads(out.rpartition("RESULT ")[2])
    for coverage in ("None", "0.0", "0.9", "0.999", "1.0", "-0.1", "nan"):
        assert results[f"{coverage}-asp=fast"] == results[f"{coverage}-asp=ref"], (
            coverage
        )
    for coverage in ("1.0", "-0.1", "nan"):
        assert results[f"{coverage}-asp=fast"] == [
            "ValueError",
            "asp_min_mask_coverage must be at least 0 and below 1",
        ], coverage
    assert (
        len(
            {
                json.dumps(results[f"{c}-asp=fast"])
                for c in ("None", "0.0", "0.9", "0.999")
            }
        )
        > 1
    )
    (path,) = trace.iterdir()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [
        (r["kernel"], r["entry"], r["impl"], r["path"], r["error"]) for r in records
    ] == [
        ("asp", "clean_cube", impl_, impl_, error)
        for error in (None,) * 4 + ("ValueError",) * 3
        for impl_ in ("fast", "ref")
    ]
