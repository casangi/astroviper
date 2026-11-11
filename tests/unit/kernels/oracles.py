"""High-precision oracle of the direct Fourier transform prediction (tests only).

``predict`` evaluates the sum that ``dft_predict`` computes,

    V_k = sum_q (M_q / n_q) exp(+2 pi i (u_k l_q + v_k m_q + w_k (n_q - 1))),

for the double-precision inputs as given (u_k = uvw * (frequency / c), rounded
as the kernels round it), far more accurately than double precision allows:

* the phases u l and v m are exact products (Dekker's two-product, which
  needs no FMA, and NumPy never contracts a * b + c into one), reduced exactly
  to their fractional turns; w (n - 1) is formed in long double;
* n, M / n, the cosines and sines and the sums are long double, the sums by
  Neumaier's compensated summation.

With ``kernel_weights=True`` it evaluates instead the exact sum of the
weights that path S shares with the reference, bit for bit: W = fl(M / fl(n)),
n formed as the kernels form it (the phases stay exact). Path S differs from
that sum only by its own arithmetic, whatever the phases (w = 0, the case of
path S).

NumPy's long double is the x87 80-bit format on x86-64 (64-bit significand)
and IEEE binary128 on aarch64 Linux, so the oracle's error is about 1e-18 of
sum |M / n| or less; on macOS arm64 it is double precision and ``AVAILABLE``
is false (tests that need the oracle skip there).
"""

import numpy as np

C = 299792458.0
LONG = np.longdouble
#: Whether long double is wider than double (the oracle is meaningful).
AVAILABLE = np.finfo(LONG).nmant >= 63
TWO_PI = 2 * LONG("3.14159265358979323846264338327950288419716939937510582097494")
#: Samples evaluated together (bounds the memory of the pixel x sample terms).
CHUNK = 128


def two_prod(a, b):
    """``(p, e)`` with ``a * b = p + e`` exactly, for float64 arrays."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    p = a * b
    split = 134217729.0  # 2^27 + 1
    t = split * a
    a_hi = t - (t - a)
    a_lo = a - a_hi
    t = split * b
    b_hi = t - (t - b)
    b_lo = b - b_hi
    e = ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return p, e


def reduced_turns(a, b):
    """The fractional turns of the exact product ``a * b``, in long double."""
    p, e = two_prod(a, b)
    return (p - np.rint(p)).astype(LONG) + e.astype(LONG)


def neumaier_sum(terms):
    """Sum along the last axis by Neumaier's compensated summation."""
    total = np.zeros(terms.shape[:-1], terms.dtype)
    compensation = np.zeros_like(total)
    for j in range(terms.shape[-1]):
        x = terms[..., j]
        t = total + x
        compensation += np.where(
            np.abs(total) >= np.abs(x), (total - t) + x, (x - t) + total
        )
        total = t
    return total + compensation


def predict(
    model,
    uvw,
    frequency_coord,
    frequency_map,
    time_map,
    pol_map,
    l,
    m,
    kernel_weights=False,
):
    """The exact visibilities of ``dft_predict``'s sum, with its error scales.

    Parameters are those of ``dft_predict`` without ``vis_data``, and
    ``kernel_weights`` (see the module docstring).

    Returns
    -------
    vis : numpy.ndarray of clongdouble
        ``(n_time, n_baseline, n_vis_chan, n_pol)``; NaN for the rows the
        kernels skip (a non-finite uvw component).
    norm : numpy.ndarray of float64
        sum_q |M_q / n_q| of each sample's model plane and polarization.
    turns : numpy.ndarray of float64
        max over the non-zero pixels of |u l| + |v m| + |w (n - 1)| per sample.
    """
    model = np.asarray(model)
    uvw = np.asarray(uvw, np.float64)
    frequency = np.asarray(frequency_coord, np.float64)
    l = np.asarray(l, np.float64)
    m = np.asarray(m, np.float64)
    n_time, n_baseline = uvw.shape[:2]
    shape = (n_time, n_baseline, frequency.size, len(pol_map))
    vis = np.full(shape, np.nan, np.clongdouble)
    norm = np.full(shape, np.nan)
    turns = np.full(shape, np.nan)

    r2 = l.astype(LONG)[:, None] ** 2 + m.astype(LONG)[None, :] ** 2
    n = np.sqrt(1 - r2)
    n_minus_one = -r2 / (1 + n)
    if kernel_weights:  # n = sqrt(1 - (l l + m m)) in float64, as the kernels do
        n = np.sqrt(1.0 - (l[:, None] * l[:, None] + m[None, :] * m[None, :]))
    scale = frequency / C
    weights = {}
    for t in range(n_time):
        rows = np.flatnonzero(np.isfinite(uvw[t]).all(axis=1))
        for c in range(frequency.size):
            u, v, w = (uvw[t, rows, i] * scale[c] for i in range(3))
            for p, a_pol in enumerate(pol_map):
                key = (int(time_map[t]), int(frequency_map[c]), int(a_pol))
                if key not in weights:
                    plane = model[key].astype(np.complex128)
                    i, j = np.nonzero(plane)
                    if kernel_weights:  # W = fl(M / n) in float64
                        weight = (plane[i, j].real / n[i, j]).astype(LONG)
                        weight_im = (plane[i, j].imag / n[i, j]).astype(LONG)
                    else:
                        weight = plane[i, j].real.astype(LONG) / n[i, j]
                        weight_im = plane[i, j].imag.astype(LONG) / n[i, j]
                    flux = float(np.sum(np.hypot(weight, weight_im)))
                    weights[key] = (i, j, weight, weight_im, flux)
                i, j, weight, weight_im, flux = weights[key]
                for k0 in range(0, rows.size, CHUNK):
                    ks = slice(k0, k0 + CHUNK)
                    uk, vk, wk = u[ks, None], v[ks, None], w[ks, None]
                    w_turns = wk.astype(LONG) * n_minus_one[i, j][None, :]
                    phase = (
                        reduced_turns(uk, l[None, i])
                        + reduced_turns(vk, m[None, j])
                        + (w_turns - np.rint(w_turns))
                    )
                    cos, sin = np.cos(TWO_PI * phase), np.sin(TWO_PI * phase)
                    re = neumaier_sum(weight * cos - weight_im * sin)
                    im = neumaier_sum(weight * sin + weight_im * cos)
                    vis[t, rows[ks], c, p] = re + 1j * im
                    span = np.abs(uk * l[None, i]) + np.abs(vk * m[None, j])
                    span = span + np.abs(w_turns.astype(np.float64))
                    turns[t, rows[ks], c, p] = span.max(axis=1) if i.size else 0.0
                norm[t, rows, c, p] = flux
    return vis, norm, turns
