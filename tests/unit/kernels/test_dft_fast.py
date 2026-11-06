"""Tests of the fast ``dft_predict`` (``ASTROVIPER_KERNELS=dft=fast``).

The implementation is called directly (``astroviper._kernels.impl``), so these
tests run in any session; the path of a call is read from the extension's
entry, which returns it. Small inputs only.

* The phasors are the reference's: the copied source text, and the bits (for
  finite turns).
* Path S is accurate: the oracle bounds (``oracles.py``) in the output dtype,
  on random and adversarial coherent inputs, against the exact sum and
  against the sum of the roundings path S shares with the reference (at any
  phase); in aggregate it is closer to the exact sum than the reference; a
  slightly perturbed result fails the bounds.
* Path S is invariant: bitwise across thread counts, splits of the rows into
  calls, concurrent calls, and the generic and AVX2 routines (two
  subprocesses), also through the depth-blocked GEMM of wide images.
* Coverage: every dtype pair, 1 to 4 polarizations, permuted and duplicated
  pol_map, several model planes, complex models, NaN outputs, skipped rows,
  misaligned arrays (read through aligned copies), the GIL released.
* Declined calls are the reference's, bit for bit and exception for exception.
  (No test hands the reference a misaligned array: its binding reads one in
  place, which is undefined behaviour that UBSan reports.)
* Pickled arrays and keyword calls take path S, a sparse model declines, and
  the trace of the hooked shim names each call's path.
"""

import json
import pickle
import re
import socket
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from astroviper._kernels import impl
from astroviper.processing_functions.imaging.gridders.direct_fourier_transform_cpp import (
    cos_sin_turns as reference_cos_sin_turns,
)
from tests.unit.kernels import oracles

ext = pytest.importorskip(
    "astroviper._kernels.fast.dft._dft_fast_ext", reason="fast DFT module not built"
)
fast = impl("dft", "dft_predict", "fast")
reference = impl("dft", "dft_predict", "ref")

C = 299792458.0
EPS = 2.0**-53
REPO = Path(__file__).parents[3]
PARAMETERS = ("model", "vis_data", "uvw", "frequency_coord", "frequency_map")
PARAMETERS += ("time_map", "pol_map", "l", "m")
needs_oracle = pytest.mark.skipif(
    not oracles.AVAILABLE, reason="long double is double precision here"
)


# ---------------------------------------------------------------- inputs
def _axes(n_l, n_m, cell, offset=0.0):
    """l, m as xradio makes them (bitwise mirror pairs), m optionally shifted."""
    l = np.array([(i - n_l // 2) * (-1) * cell for i in range(n_l)])
    m = np.array([(j - n_m // 2 + offset) * cell for j in range(n_m)])
    return l, m


def _call(
    model,
    seed=0,
    n_time=2,
    n_baseline=24,
    frequency=(1.0e11,),
    cell=3e-5,
    uv_m=1.5e3,
    frequency_map=None,
    time_map=None,
    pol_map=None,
    m_offset=0.0,
):
    """The arguments of a w = 0 call (no vis_data) for a 5-D model."""
    rng = np.random.default_rng(seed)
    n_l, n_m = model.shape[3:]
    l, m = _axes(n_l, n_m, cell, m_offset)
    uvw = rng.uniform(-uv_m, uv_m, (n_time, n_baseline, 3))
    uvw[..., 2] = 0.0
    frequency = np.asarray(frequency, np.float64)
    maps = {
        "frequency_map": np.zeros(frequency.size)
        if frequency_map is None
        else frequency_map,
        "time_map": np.zeros(n_time) if time_map is None else time_map,
        "pol_map": np.arange(model.shape[2]) if pol_map is None else pol_map,
    }
    return {
        "model": model,
        "uvw": uvw,
        "frequency_coord": frequency,
        "l": l,
        "m": m,
    } | {k: np.asarray(v, np.int64) for k, v in maps.items()}


def _sky(seed, shape, complex_values=False):
    rng = np.random.default_rng(seed)
    sky = rng.uniform(0.0, 1e-3, shape)
    if complex_values:
        sky = sky + 1j * rng.uniform(-1e-4, 1e-4, shape)
    return sky


def _stokes_i(seed, n_l=24, n_m=24):
    """XX = YY bitwise (a Stokes I sky)."""
    plane = _sky(seed, (n_l, n_m))
    return np.stack([plane, plane])[None, None]


def _vis(call, dtype=np.complex128):
    shape = call["uvw"].shape[:2] + (len(call["frequency_coord"]), len(call["pol_map"]))
    return np.zeros(shape, dtype)


def _args(call, vis):
    return [call["model"], vis] + [call[k] for k in PARAMETERS[2:]]


def _run(function, call, dtype=np.complex128, threads=4, vis=None):
    vis = _vis(call, dtype) if vis is None else vis
    function(*_args(call, vis), processing_function_threads=threads)
    return vis


def _path(call, dtype=np.complex128, threads=4, vis=None):
    vis = _vis(call, dtype) if vis is None else vis
    return ext.dft_predict(*_args(call, vis), processing_function_threads=threads), vis


# ---------------------------------------------------------------- the oracle bounds
def _plane_terms(call):
    """``(..., 3)``: per model plane (time, channel) of the call, path S's
    terms of the bounds:

    * L, its longest summation chain: kept rows + folded columns, a kept
      column of m > 0 folded with a kept one of -m (bitwise), once per value,
      as dft_pack.cpp folds them;
    * kappa, the largest r^2 / (1 - r^2) over its non-zero pixels, the
      condition of n = sqrt(1 - l^2 - m^2) (a relative error of n, shared by
      both kernels, up to about (kappa + 1) eps);
    * its products and quotients per sample and pass (the weights M / n of the
      kept pixels, kept rows x folded columns in the GEMM, kept rows in the
      row sums), each of which rounds by up to 2^-1075 absolute when its
      result is subnormal (a sum of subnormal values is exact): with the two
      passes (cos and sin), an underflow allowance of their number x 2^-1074.
    """
    model = np.asarray(call["model"])
    referenced = sorted(set(call["pol_map"].tolist()))
    l, m = call["l"], call["m"]
    r2 = l[:, None] ** 2 + m[None, :] ** 2
    terms = np.zeros(model.shape[:2] + (3,))
    for plane in np.ndindex(model.shape[:2]):
        nonzero = (model[plane][referenced] != 0).any(axis=0)
        rows = int(nonzero.any(axis=1).sum())
        kept = m[nonzero.any(axis=0)]
        pairs = len(set(kept[kept > 0].tolist()) & set((-kept[kept < 0]).tolist()))
        columns = kept.size - pairs
        kappa = (r2[nonzero] / (1 - r2[nonzero])).max(initial=0.0)
        terms[plane] = rows + columns, kappa, nonzero.sum() + rows * columns + rows
    return terms


def _ulp(x, dtype):
    real = np.finfo(np.dtype(dtype)).dtype
    return np.spacing(np.abs(x).astype(np.float64).astype(real)).astype(oracles.LONG)


def _violations(call, v_fast, v_ref, v_ref128=None, keep=True):
    """Where the per-sample bounds fail, per real component:

    (1) |V_fast - V_or| <= (L + 2 pi T + 8 + kappa) eps sum|M/n| + U
        + 1/2 ulp_out: the bound of plan_v2 (dft_design) against the exact
        sum V_or, with L the chain length of the sample's plane and T the
        sample's largest |u l| + |v m| + |w (n - 1)|, made valid for all
        inputs by two terms that vanish elsewhere (``_plane_terms``): kappa
        for n near the horizon, and the underflow allowance U = (products and
        quotients) x 2^-1074 for subnormal values;
    (1t) |V_fast - V_rp| <= (L + 8) eps sum|W| + U + 1/2 ulp_out against
        V_rp, the sum of the roundings that path S shares with the reference
        (``oracles.predict(rounded=True)``): path S's own arithmetic, with no
        2 pi T term, so that a non-exact path S shows at any phase (its own
        error is a few eps sum|W|);
    (c64) complex64 outputs: |V_fast - V_ref| <= 1 ulp32 + the bound (1) + the
        error of the reference's complex128 result ``v_ref128``.

    The plan's bound (2), |V_fast - V_or| <= |V_ref - V_or| + 16 eps
    sum|M/n| + 1 ulp_out, does not hold per sample for an exact path S: both
    kernels round the products u l and v m bit for bit (an error of up to
    2 pi T eps sum|M/n|), and the reference's extra rounding of the phase
    sum, independent of that error, cancels it at some samples. Its errors are
    therefore about sqrt(2) times path S's in rms, with as large maxima, and
    the bound is checked in aggregate instead, by the rms test below.

    ulp_out is that of the larger of |V_oracle| and |V_fast| in the output
    dtype. Only the predicted samples count (``keep`` excludes others, such
    as NaN outputs). Returns ``{criterion: number of failing components}``.
    """
    exact, norm, turns = oracles.predict(**call)
    rounded, norm_rounded, _ = oracles.predict(**call, rounded=True)
    used = np.isfinite(norm) & keep
    terms = _plane_terms(call)[
        call["time_map"][:, None, None, None],
        call["frequency_map"][None, None, :, None],
    ]
    chain, kappa, products = (
        np.broadcast_to(terms[..., k], norm.shape)[used] for k in range(3)
    )
    underflow = products * 2.0**-1074
    phase = 2 * np.pi * turns[used]
    budget = (chain + phase + 8 + kappa) * EPS * norm[used] + underflow
    budget_rounded = (chain + 8) * EPS * norm_rounded[used] + underflow
    budget, budget_rounded = (
        budget.astype(oracles.LONG),
        budget_rounded.astype(oracles.LONG),
    )
    failed = {"bound1": 0, "bound1t": 0, "complex64": 0}
    for part in (np.real, np.imag):
        x_or, x_rp = part(exact[used]), part(rounded[used])
        x_fast = part(v_fast[used]).astype(oracles.LONG)
        x_ref = part(v_ref[used]).astype(oracles.LONG)
        dtype = v_fast.real.dtype
        ulp = _ulp(np.maximum(np.abs(x_or), np.abs(x_fast)), dtype)
        ulp_rounded = _ulp(np.maximum(np.abs(x_rp), np.abs(x_fast)), dtype)
        failed["bound1"] += int(np.sum(np.abs(x_fast - x_or) > budget + ulp / 2))
        failed["bound1t"] += int(
            np.sum(np.abs(x_fast - x_rp) > budget_rounded + ulp_rounded / 2)
        )
        if v_fast.dtype == np.complex64:
            ref128 = np.abs(part(v_ref128[used]).astype(oracles.LONG) - x_or)
            ulp32 = _ulp(np.maximum(np.abs(x_fast), np.abs(x_ref)), np.float32)
            failed["complex64"] += int(
                np.sum(np.abs(x_fast - x_ref) > ulp32 + budget + ref128)
            )
    return failed


ACCURATE = {"bound1": 0, "bound1t": 0, "complex64": 0}


def _assert_accurate(call, dtype=np.complex128, threads=4):
    """Path S is taken, and its result is within the oracle bounds."""
    path, v_fast = _path(call, dtype, threads)
    assert path == "S"
    v_ref = _run(reference, call, dtype, threads)
    v_ref128 = (
        _run(reference, call, np.complex128, threads) if dtype == np.complex64 else None
    )
    assert _violations(call, v_fast, v_ref, v_ref128) == ACCURATE


# ---------------------------------------------------------------- the phasors
def _definitions(text):
    constants = dict(re.findall(r"constexpr double (k\w+) = ([^;]+);", text))
    functions = re.findall(
        r"^inline \w+ (?:round_to_integer|cos_sin_of_turns)\(.*?^\}", text, re.M | re.S
    )
    names = [f"kSin{i}" for i in range(6)] + [f"kCos{i}" for i in range(6)]
    return {k: constants[k] for k in ["kTwoPi", "kRoundMagic", *names]}, functions


def test_phasors_are_a_verbatim_copy_of_the_reference():
    gridders = "src/astroviper/processing_functions/imaging/gridders"
    ref_text = (
        REPO
        / gridders
        / "direct_fourier_transform_cpp/src/direct_fourier_transform.cpp"
    ).read_text()
    fast_text = (
        REPO / "src/astroviper/_kernels/fast/dft/src/dft_s_kernels.inc"
    ).read_text()
    ref_constants, ref_functions = _definitions(ref_text)
    fast_constants, fast_functions = _definitions(fast_text)
    assert fast_constants == ref_constants
    assert len(ref_functions) == 2 and fast_functions == ref_functions


def test_phasors_are_bitwise_the_reference():
    """For finite turns (path S's range guard keeps |u l|, |v m| < 2^51; for a
    NaN or infinite turn both give NaN, with another sign bit or payload)."""
    rng = np.random.default_rng(1)
    edges = [
        0.0,
        -0.0,
        5e-324,
        -5e-324,
        1e-300,
        2.0**-60,
        0.125,
        0.25,
        0.375,
        0.5,
        2.0**51 - 0.5,
    ]
    edges += [np.nextafter(x, s) for x in (0.125, 0.25, 0.5, 1.0) for s in (0.0, 2.0)]
    turns = np.concatenate(
        [
            edges,
            np.negative(edges),
            np.arange(-64, 65) / 8,
            rng.uniform(-0.6, 0.6, 20000),
            rng.uniform(-1e9, 1e9, 20000),
            rng.standard_normal(2000) * 2.0 ** rng.integers(-60, 50, 2000),
        ]
    )
    for got, expected in zip(
        ext.cos_sin_turns(turns), reference_cos_sin_turns(turns), strict=True
    ):
        assert got.tobytes() == expected.tobytes()


# ---------------------------------------------------------------- accuracy
def _depth_blocked_call():
    """3 x 2048 pixels: 1025 folded columns, three depth steps of the GEMM's
    micro-kernel (kDepth = 344), the last one partial, as for HL Tau."""
    return _call(_sky(19, (1, 1, 2, 3, 2048)), n_time=1, n_baseline=20, cell=3e-6)


@needs_oracle
@pytest.mark.parametrize(
    "case",
    [
        "stokes_i",
        "uv_zero_rows",
        "constant_model",
        "strip_1xn",
        "opposite_large_phases",
        "wide_field",
        "m_without_mirrors",
        "near_mirrors_large_phases",
        "near_horizon",
        "subnormal_model",
        "depth_blocked",
    ],
)
def test_path_s_is_within_the_oracle_bounds(case):
    if case == "stokes_i":
        call = _call(_stokes_i(2), n_baseline=40)
    elif case == "uv_zero_rows":  # every phase 0: the coherent sum of all weights
        call = _call(_stokes_i(3))
        call["uvw"][:, :8] = 0.0
    elif case == "constant_model":
        call = _call(np.full((1, 1, 1, 24, 24), 0.37), uv_m=30.0)
    elif case == "strip_1xn":
        call = _call(_sky(4, (1, 1, 1, 1, 96)), uv_m=50.0)
    elif case == "opposite_large_phases":  # |u l| up to ~1e5 turns, of both signs
        model = np.zeros((1, 1, 1, 24, 24))
        model[0, 0, 0, [1, 23, 2, 22], [1, 23, 22, 2]] = [1.0, 0.8, 0.6, 0.4]
        model[0, 0, 0, 10:14, 10:14] = 1e-3
        call = _call(model, cell=1e-3, uv_m=3e4, n_baseline=60)
    elif case == "wide_field":  # |l|, |m| up to 0.3: n down to 0.9
        call = _call(_stokes_i(5, 21, 21), cell=0.03, uv_m=1e3, frequency=(1.4e9,))
    elif case == "m_without_mirrors":  # a 0.3-pixel shift: no pairs, unfolded
        call = _call(_stokes_i(6), m_offset=0.3)
    elif case == "near_mirrors_large_phases":  # m_b = -m_a + 1 ulp: not folded
        call = _call(_stokes_i(16), uv_m=8e4)  # T up to about 2e4 turns
        half = np.flatnonzero(call["m"] > 0)[::2]
        call["m"][half] = np.nextafter(call["m"][half], 1.0)
    elif case == "near_horizon":  # n down to 4.5e-6 at the corners: kappa 5e10
        call = _call(_stokes_i(17, 6, 6), uv_m=1.0)
        call["l"] = np.array([-0.6, -0.36, -0.12, 0.12, 0.36, 0.6])
        edge = 0.8 - 1.25e-11
        call["m"] = np.array([-edge, -0.48, -0.16, 0.16, 0.48, edge])
    elif case == "subnormal_model":  # every weight and sum below 2^-1022
        call = _call(_stokes_i(18) * 1e-310)
    else:
        call = _depth_blocked_call()
    _assert_accurate(call)


@needs_oracle
@pytest.mark.parametrize("uv_m", [1.5e3, 3e4], ids=["T~400", "T~8000"])
def test_path_s_is_closer_to_the_exact_sum_than_the_reference_in_rms(uv_m):
    """The plan's bound (2) in aggregate (see ``_violations``): over 4800
    components path S's rms error against the exact sum is at most the
    reference's, which adds the rounding of its phase sums to the rounding of
    u l and v m that both share."""
    call = _call(
        _stokes_i(21), n_time=3, n_baseline=200, frequency=(1e11, 1.2e11), uv_m=uv_m
    )
    path, v_fast = _path(call)
    v_ref = _run(reference, call)
    exact, norm, _ = oracles.predict(**call)
    used = np.isfinite(norm)

    def rms(v):
        error = (v[used].astype(np.clongdouble) - exact[used]) / norm[used]
        return float(np.sqrt(np.mean(np.concatenate([error.real, error.imag]) ** 2)))

    assert path == "S" and 2 * used.sum() >= 2000
    assert rms(v_fast) <= rms(v_ref)


@needs_oracle
def test_the_bounds_detect_a_perturbed_result():
    """At the sample of the largest phase (T about 6000 turns), bound (1t)
    detects an error of (L + 16) eps sum|W| beyond path S's own, which bound
    (1) cannot see there; an error of 1e-11 sum|W| fails both."""
    call = _call(_stokes_i(7), uv_m=3e4)
    path, v_fast = _path(call)
    v_ref = _run(reference, call)
    assert path == "S" and _violations(call, v_fast, v_ref) == ACCURATE
    rounded, norm, turns = oracles.predict(**call, rounded=True)
    k = np.unravel_index(np.nanargmax(turns), turns.shape)
    chain = _plane_terms(call)[0, 0, 0]
    sign = 1.0 if v_fast[k].real >= rounded[k].real else -1.0
    for error, failed in (
        ((chain + 16) * EPS * norm[k], {"bound1": 0, "bound1t": 1}),
        (1e-11 * norm[k], {"bound1": 1, "bound1t": 1}),
    ):
        perturbed = v_fast.copy()
        perturbed[k] += sign * error
        assert _violations(call, perturbed, v_ref) == ACCURATE | failed


@needs_oracle
@pytest.mark.parametrize(
    "model_dtype", [np.float32, np.float64, np.complex64, np.complex128]
)
@pytest.mark.parametrize("vis_dtype", [np.complex64, np.complex128])
def test_every_dtype_pair(model_dtype, vis_dtype):
    model = _sky(8, (1, 1, 2, 20, 16), np.dtype(model_dtype).kind == "c").astype(
        model_dtype
    )
    _assert_accurate(_call(model), vis_dtype)


@needs_oracle
@pytest.mark.parametrize(
    "m_pol, pol_map, complex_values",
    [
        (1, None, False),
        (2, None, True),
        (3, None, False),
        (4, None, True),
        (4, [3, 1, 0, 2], True),
        (2, [0, 0, 1], True),
        (4, [1, 1, 1, 1], False),
    ],
)
def test_polarizations_and_pol_maps(m_pol, pol_map, complex_values):
    _assert_accurate(
        _call(_sky(9, (1, 1, m_pol, 16, 18), complex_values), pol_map=pol_map)
    )


@needs_oracle
def test_linear_correlations_of_a_polarized_sky():
    """XX XY YX YY of I Q U V: equal-or-negated parts are summed once."""
    i, q, u, v = (_sky(10 + k, (16, 16)) - (0 if k == 0 else 5e-4) for k in range(4))
    correlations = np.stack([i + q, u + 1j * v, u - 1j * v, i - q])
    _assert_accurate(_call(correlations[None, None]))


@needs_oracle
def test_model_planes_and_maps():
    model = _sky(11, (2, 3, 2, 16, 14))
    model[:, :, :, :3] = 0.0  # zero rows
    model[:, :, :, :, 11:] = 0.0  # zero columns
    model[1, 2] = 0.0  # a plane without components
    call = _call(
        model,
        n_time=4,
        frequency=(1e11, 1.1e11, 2.3e11, 3.4e11),
        frequency_map=[2, 0, 1, 2],
        time_map=[0, 0, 1, 1],
    )
    _assert_accurate(call)


@needs_oracle
def test_unreferenced_polarizations_may_be_non_finite():
    model = _sky(12, (1, 1, 4, 16, 16))
    model[0, 0, 1] = np.nan
    model[0, 0, 3, 2, 2] = np.inf
    _assert_accurate(_call(model, pol_map=[0, 2]))


@needs_oracle
def test_nan_outputs_and_skipped_rows_are_kept():
    call = _call(_stokes_i(13), n_baseline=30)
    call["uvw"][0, 3, 0] = np.nan
    call["uvw"][1, 4, 1] = np.inf
    call["uvw"][1, 5, 2] = -np.inf
    call["uvw"][0, 6, 0], call["uvw"][0, 6, 2] = np.nan, 0.25  # w != 0 in a skipped row
    initial = _vis(call) + (3.0 - 2.0j)  # overwritten, not accumulated
    initial[0, 9, 0, 1] = np.nan
    initial[1, 11, 0, 0] = complex(1.0, np.nan)
    path, v_fast = _path(call, vis=initial.copy())
    v_ref = _run(reference, call, vis=initial.copy())
    assert path == "S"
    assert np.array_equal(np.isnan(v_fast), np.isnan(v_ref))
    skipped = ~np.isfinite(call["uvw"]).all(axis=2)
    assert v_fast[skipped].tobytes() == initial[skipped].tobytes()
    assert not any(_violations(call, v_fast, v_ref, keep=~np.isnan(v_ref)).values())


# ---------------------------------------------------------------- invariance
def _invariance_call():
    """Two model planes, several sample blocks, odd sizes, a complex part."""
    model = _sky(14, (1, 2, 3, 30, 26), complex_values=True)
    return _call(
        model, n_time=3, n_baseline=230, frequency=(1e11, 1.3e11), frequency_map=[0, 1]
    )


def test_bitwise_across_thread_counts():
    call = _invariance_call()
    path, expected = _path(call, threads=1)
    assert path == "S"
    for threads in (0, -1, 2, 6, 12, 24):
        assert _run(fast, call, threads=threads).tobytes() == expected.tobytes(), (
            threads
        )


def test_bitwise_across_splits_of_the_rows():
    call = _invariance_call()
    expected = _run(fast, call)
    for times, baselines in [
        (slice(0, 1), slice(None)),
        (slice(1, 3), slice(0, 37)),
        (slice(0, 3), slice(37, 150)),
        (slice(2, 3), slice(150, 230)),
    ]:
        part = dict(
            call,
            uvw=np.ascontiguousarray(call["uvw"][times, baselines]),
            time_map=call["time_map"][times],
        )
        assert _run(fast, part).tobytes() == expected[times, baselines].tobytes()


def test_concurrent_calls_give_the_bits_of_serial_calls():
    """Python threads calling path S at once, on shared inputs."""
    calls = [_invariance_call(), _call(_stokes_i(23), n_baseline=50)]
    expected = [_run(fast, call, threads=2).tobytes() for call in calls]

    def bits(i):
        return _run(fast, calls[i % 2], threads=2).tobytes() == expected[i % 2]

    with ThreadPoolExecutor(max_workers=6) as pool:
        assert all(pool.map(bits, range(24)))


def test_the_gil_is_released_during_path_s():
    """Another Python thread runs while a call computes: with a switch
    interval longer than the call, it could not if the call held the GIL."""
    call = _call(_stokes_i(22, 512, 512), n_time=4, n_baseline=2000)
    started, done = threading.Event(), threading.Event()

    def predict():
        started.set()
        _run(fast, call, threads=1)  # about 0.07 s
        done.set()

    interval = sys.getswitchinterval()
    sys.setswitchinterval(30.0)
    try:
        worker = threading.Thread(target=predict)
        worker.start()
        started.wait()
        during = not done.is_set()
        worker.join()
    finally:
        sys.setswitchinterval(interval)
    assert during and done.is_set()


ISA_RUN = """
import sys
import numpy as np
from astroviper._kernels import impl
from astroviper._kernels.fast.dft import ISA
names = ("model", "vis_data", "uvw", "frequency_coord", "frequency_map", "time_map", "pol_map", "l", "m")
for source, target in zip(sys.argv[1::2], sys.argv[2::2]):
    arrays = np.load(source)
    arguments = [arrays[k] for k in names]
    impl("dft", "dft_predict", "fast")(*arguments, processing_function_threads=3)
    np.save(target, arguments[1])
print(ISA)
"""


def test_generic_and_avx2_routines_give_the_same_bits(run_python, tmp_path):
    """Also through the GEMM's depth steps of a wide image."""
    calls = {"invariance": _invariance_call(), "depth_blocked": _depth_blocked_call()}
    for name, call in calls.items():
        np.savez(tmp_path / f"{name}.npz", vis_data=_vis(call), **call)
    results = {}
    for isa, env in (
        ("default", {}),
        ("generic", {"ASTROVIPER_KERNELS_ISA": "generic"}),
    ):
        files = []
        for name in calls:
            files += [
                str(tmp_path / f"{name}.npz"),
                str(tmp_path / f"{name}-{isa}.npy"),
            ]
        out = run_python("-c", ISA_RUN, *files, **env)
        results[out.strip().splitlines()[-1]] = [
            np.load(tmp_path / f"{name}-{isa}.npy").tobytes() for name in calls
        ]
    if "avx2" not in results:
        pytest.skip("no AVX2 routines on this machine or build")
    assert set(results) == {"avx2", "generic"}
    assert results["avx2"] == results["generic"]


def test_an_unknown_isa_override_fails_the_import(run_python):
    done = run_python(
        "-c",
        "import astroviper._kernels.fast.dft",
        check=False,
        ASTROVIPER_KERNELS_ISA="avx512",
    )
    assert done.returncode != 0
    assert "ASTROVIPER_KERNELS_ISA=avx512: expected 'generic'" in done.stderr


# ---------------------------------------------------------------- paths
def _outcome(function, make):
    """The bytes of every array argument after a call, and its exception."""
    args, kwargs = make()
    try:
        function(*args, **kwargs)
        error = None
    except Exception as e:  # compared with the reference's exception
        error = (type(e), str(e))
    arrays = [a for a in [*args, *kwargs.values()] if isinstance(a, np.ndarray)]
    return [a.tobytes() for a in arrays], error


def _declined(changes=None, vis=None, extra=(), drop=0, **kwargs):
    """A factory of fresh ``(args, kwargs)`` of a call that path S declines:
    ``changes`` and ``vis`` (functions of the base call) replace arguments,
    ``extra`` positional arguments are appended, the last ``drop`` dropped and
    ``kwargs`` (functions of the base call) passed by name."""

    def make():
        call = _call(_stokes_i(15, 16, 16), n_baseline=12)
        call |= changes(call) if changes else {}
        args = _args(call, vis(call) if vis else _vis(call))
        named = {k: v(call) for k, v in kwargs.items()}
        return args[: len(args) - drop] + list(extra), named

    return make


def _uvw_with(index, value):
    def changes(call):
        uvw = call["uvw"].copy()
        uvw[index] = value
        return {"uvw": uvw}

    return changes


def _late_nan(call):
    model = np.ascontiguousarray(np.repeat(call["model"], 3, axis=1))
    model[0, 2, 1, 4, 5] = np.nan
    return {
        "model": model,
        "frequency_coord": np.array([1e11, 1.1e11, 1.2e11]),
        "frequency_map": np.arange(3),
    }


def _read_only(call):
    vis = _vis(call)
    vis.setflags(write=False)
    return vis


def _shared_with_the_model(call):
    """vis_data and the model in one buffer."""
    vis = _vis(call)
    buffer = np.zeros(vis.size + call["model"].size, np.complex128)
    model = buffer[vis.size :].real.reshape(call["model"].shape)
    model[...] = call["model"]
    call["model"] = model
    return buffer[: vis.size].reshape(vis.shape)


DECLINED = {
    "w_5e-324_in_one_row": _declined(_uvw_with((1, 7, 2), 5e-324)),
    "w_term": _declined(lambda c: {"uvw": c["uvw"] + [0.0, 0.0, 30.0]}),
    "nan_model_in_a_late_plane": _declined(_late_nan),
    "infinite_scaled_u": _declined(_uvw_with((0, 2, 0), 1e306)),
    "beyond_the_horizon": _declined(lambda c: {"l": c["l"] * 3e4}),
    "nan_l": _declined(lambda c: {"l": np.where(np.arange(16) == 3, np.nan, c["l"])}),
    "2d_map": _declined(lambda c: {"time_map": c["time_map"][:, None]}),
    "pol_map_out_of_range": _declined(lambda c: {"pol_map": np.array([0, 2])}),
    "float32_uvw": _declined(lambda c: {"uvw": c["uvw"].astype(np.float32)}),
    "fortran_uvw": _declined(lambda c: {"uvw": np.asfortranarray(c["uvw"])}),
    "int32_maps": _declined(
        lambda c: {k: c[k].astype(np.int32) for k in ("time_map", "pol_map")}
    ),
    "list_maps": _declined(lambda c: {"pol_map": [0, 1], "frequency_map": [0]}),
    "float32_l": _declined(lambda c: {"l": c["l"].astype(np.float32)}),
    "big_endian_model": _declined(lambda c: {"model": c["model"].astype(">f8")}),
    "int_model": _declined(lambda c: {"model": c["model"].astype(np.int32)}),
    "fortran_model": _declined(lambda c: {"model": np.asfortranarray(c["model"])}),
    "read_only_vis": _declined(vis=_read_only),
    "float_vis": _declined(vis=lambda c: np.zeros(_vis(c).shape)),
    "vis_sharing_memory_with_the_model": _declined(vis=_shared_with_the_model),
    "bad_keyword": _declined(threads=lambda c: 2),
    "threads_not_an_int": _declined(processing_function_threads=lambda c: 2.0),
    "too_many_arguments": _declined(extra=(2, 3)),
    "missing_argument": _declined(drop=1),
    "positional_and_keyword": _declined(m=lambda c: c["m"]),
}


@pytest.mark.parametrize("name", DECLINED)
def test_declined_calls_are_the_reference(name):
    args, kwargs = DECLINED[name]()
    before = [a.tobytes() for a in args if isinstance(a, np.ndarray)]
    assert ext.dft_predict(*args, **kwargs) == "declined"
    assert [a.tobytes() for a in args if isinstance(a, np.ndarray)] == before
    assert _outcome(fast, DECLINED[name]) == _outcome(reference, DECLINED[name])


@pytest.mark.parametrize(
    "name, changes",
    [
        ("no_polarization", {"pol_map": np.zeros(0, np.int64)}),
        ("no_rows", {"uvw": np.zeros((0, 12, 3)), "time_map": np.zeros(0, np.int64)}),
        (
            "no_channels",
            {"frequency_coord": np.zeros(0), "frequency_map": np.zeros(0, np.int64)},
        ),
        ("empty_image", {"model": np.zeros((1, 1, 2, 0, 16)), "l": np.zeros(0)}),
        ("zero_model", {"model": np.zeros((1, 1, 2, 16, 16))}),
        ("minus_zero_model", {"model": np.full((1, 1, 2, 16, 16), -0.0)}),
    ],
)
def test_trivial_calls_take_path_s_bitwise_equal_to_the_reference(name, changes):
    call = _call(_stokes_i(17, 16, 16), n_baseline=12) | changes
    initial = _vis(call) + (5.0 - 1.0j)
    path, v_fast = _path(call, vis=initial.copy())
    assert path == "S"
    assert v_fast.tobytes() == _run(reference, call, vis=initial.copy()).tobytes()


def test_pickled_arrays_and_keyword_calls_take_path_s():
    call = _call(_stokes_i(18), n_baseline=20)
    expected = _run(fast, call)
    pickled = pickle.loads(pickle.dumps(_args(call, _vis(call))))
    assert ext.dft_predict(*pickled) == "S"
    assert pickled[1].tobytes() == expected.tobytes()
    arguments = dict(zip(PARAMETERS, _args(call, _vis(call)), strict=True))
    assert (
        ext.dft_predict(
            **dict(reversed(arguments.items())), processing_function_threads=3
        )
        == "S"
    )
    assert arguments["vis_data"].tobytes() == expected.tobytes()
    vis = _vis(call)
    assert (
        ext.dft_predict(
            call["model"],
            vis,
            call["uvw"],
            m=call["m"],
            l=call["l"],
            **{k: call[k] for k in PARAMETERS[3:7]},
        )
        == "S"
    )
    assert vis.tobytes() == expected.tobytes()


def _misaligned(array):
    """A copy of ``array`` one byte off the alignment that NumPy requires, as
    Dask hands its workers arrays at arbitrary offsets of the frames it
    received."""
    buffer = np.zeros(array.nbytes + 1, np.uint8)
    copy = buffer[1:].view(array.dtype).reshape(array.shape)
    copy[...] = array
    assert not copy.flags.aligned
    return copy


def test_misaligned_arrays_take_path_s_through_aligned_copies():
    """Every array of the call misaligned: path S reads aligned copies and
    writes vis_data in place, with the bits of the aligned call (NaN values
    kept); with a w != 0 it declines and writes nothing. (The reference reads
    a misaligned array in place, which is undefined behaviour: not called.)"""
    call = _invariance_call()
    initial = _vis(call) + (3.0 - 2.0j)
    initial[0, 9, 1, 2] = np.nan
    path, expected = _path(call, vis=initial.copy())
    assert path == "S"
    misaligned = {k: _misaligned(v) for k, v in call.items()}
    vis = _misaligned(initial)
    assert (
        ext.dft_predict(*_args(misaligned, vis), processing_function_threads=3) == "S"
    )
    assert vis.tobytes() == expected.tobytes()
    misaligned["uvw"][0, 3, 2] = 0.5
    vis = _misaligned(initial)
    assert ext.dft_predict(*_args(misaligned, vis)) == "declined"
    assert vis.tobytes() == initial.tobytes()


def test_w_minus_zero_counts_as_zero():
    call = _call(_stokes_i(19))
    expected = _run(fast, call)
    call["uvw"][..., 2] = -0.0
    path, vis = _path(call)
    assert path == "S" and vis.tobytes() == expected.tobytes()


def test_the_cost_rule_declines_a_sparse_model():
    rng = np.random.default_rng(20)
    sparse = np.zeros((1, 1, 2, 256, 256))
    sparse.reshape(2, -1)[:, rng.choice(256 * 256, 200, replace=False)] = 1e-3
    call = _call(sparse, n_baseline=10, cell=1e-6)
    path, vis = _path(call)
    assert path == "declined" and not vis.any()
    assert _run(fast, call).tobytes() == _run(reference, call).tobytes()
    call["model"] = np.ascontiguousarray(sparse[:, :, :, 100:140, 100:140])
    call["l"], call["m"] = call["l"][100:140], call["m"][100:140]
    assert _path(call)[0] == "S"  # the few sources in 40 x 40 pixels: S is cheaper


TRACED = """
import numpy as np
from astroviper.processing_functions.imaging.gridders import direct_fourier_transform_cpp as shim

n = 16
l = np.array([(i - n // 2) * -2e-5 for i in range(n)])
model = np.full((1, 1, 2, n, n), 1e-3)
uvw = np.random.default_rng(0).uniform(-1e3, 1e3, (2, 9, 3))
uvw[..., 2] = 0.0
rest = (np.array([1e11]), np.zeros(1, np.int64), np.zeros(2, np.int64))
vis = np.zeros((2, 9, 1, 2), np.complex128)
shim.dft_predict(model, vis, uvw, *rest, np.arange(2), l, -l)
uvw[1, 3, 2] = 0.5
shim.dft_predict(model, vis, uvw, *rest, np.arange(2), l, -l)
try:
    shim.dft_predict(model, vis, uvw, *rest, np.array([0, 5]), l, -l)
except RuntimeError:
    pass
"""


def test_the_trace_names_each_calls_path(run_python, tmp_path):
    trace = tmp_path / "trace"
    run_python(
        "-c", TRACED, ASTROVIPER_KERNELS="dft=fast", ASTROVIPER_KERNELS_TRACE=str(trace)
    )
    (path,) = trace.glob(f"kernels-{socket.gethostname()}-*.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(r["kernel"], r["impl"], r["path"], r["error"]) for r in records] == [
        ("dft", "fast", "S", None),
        ("dft", "fast", "declined", None),
        ("dft", "fast", "declined", "RuntimeError"),
    ]
