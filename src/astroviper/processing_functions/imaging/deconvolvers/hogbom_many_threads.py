"""Many-threads Hogbom CLEAN.

A numba implementation of Hogbom CLEAN whose parallelism spans **both** the
``(time, frequency, polarization)`` planes **and** the rows *within* each plane.
The existing C++ deconvolver (``deconvolvers.hogbom.clean_cube``) parallelizes
only *across* planes, so for single-channel imaging (a few planes) it leaves
most threads idle. Here the parallel work domain is ``(plane x row)``, so every
thread stays busy whether there are many planes or just one or two.

Threading model
---------------
The per-iteration peak search and PSF subtraction are run as ``nogil`` numba
kernels dispatched over a ``ThreadPoolExecutor``. This deliberately avoids
numba's ``parallel=True`` OpenMP threading layer, which clashes with the other
OpenMP / Accelerate runtimes loaded by the imaging C++ extensions and BLAS (a
segfault source on macOS). ``nogil`` njit releases the GIL, so the pool's worker
threads execute the compiled kernels truly in parallel.

The per-plane algorithm is identical to the C++ one (peak = max ``|residual|``
within the mask; subtract a centred, gain-scaled PSF; stop a plane when its peak
drops below its threshold or it reaches its iteration limit), including the
tie-breaking order (lowest row, then lowest column), so the result matches the
C++ deconvolver bit-for-bit on tie-free data.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a hard dependency in practice
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def deco(f):
            return f

        return deco


@njit(nogil=True, fastmath=True, cache=True)
def _peak_range(
    residual, mask, use_mask, niter, active, it, ny, nx, r0, r1, row_max, row_ix
):
    """Per-row max ``|residual|`` for flattened ``(plane, row)`` indices ``[r0, r1)``."""
    for r in range(r0, r1):
        pl = r // ny
        iy = r % ny
        if (not active[pl]) or (it >= niter[pl]):
            row_max[pl, iy] = -1.0
            continue
        m = 0.0
        mi = 0
        for ix in range(nx):
            if (not use_mask) or mask[pl, iy, ix]:
                v = abs(residual[pl, iy, ix])
                if v > m:
                    m = v
                    mi = ix
        row_max[pl, iy] = m
        row_ix[pl, iy] = mi


@njit(nogil=True, fastmath=True, cache=True)
def _combine(
    residual,
    model,
    active,
    niter,
    threshold,
    it,
    ny,
    gain,
    row_max,
    row_ix,
    peak_y,
    peak_x,
    peak_pv,
    iters_out,
):
    """Serial per-plane reduction: pick each plane's peak, update its model."""
    nplanes = residual.shape[0]
    any_active = False
    for pl in range(nplanes):
        if (not active[pl]) or (it >= niter[pl]):
            continue
        best = -1.0
        py = 0
        px = 0
        for iy in range(ny):
            if row_max[pl, iy] > best:  # strict -> lowest row, then lowest column
                best = row_max[pl, iy]
                py = iy
                px = row_ix[pl, iy]
        if best < threshold[pl]:
            active[pl] = False  # converged: no component this iteration
            continue
        maxval = residual[pl, py, px]
        pv = gain * maxval
        model[pl, py, px] += pv
        peak_y[pl] = py
        peak_x[pl] = px
        peak_pv[pl] = pv
        iters_out[pl] += 1
        any_active = True
    return any_active


@njit(nogil=True, fastmath=True, cache=True)
def _subtract_range(
    residual, psf, psf_map, active, niter, it, ny, nx, r0, r1, peak_y, peak_x, peak_pv
):
    """Subtract ``pv * PSF`` (centred on each plane's peak) for rows ``[r0, r1)``."""
    cy = ny // 2
    cx = nx // 2
    for r in range(r0, r1):
        pl = r // ny
        iy = r % ny
        if (not active[pl]) or (it >= niter[pl]):
            continue
        py = peak_y[pl]
        px = peak_x[pl]
        pv = peak_pv[pl]
        pp = psf_map[pl]
        psf_y = cy + iy - py
        if psf_y < 0 or psf_y >= ny:
            continue
        for ix in range(nx):
            psf_x = cx + ix - px
            if psf_x < 0 or psf_x >= nx:
                continue
            residual[pl, iy, ix] -= pv * psf[pp, psf_y, psf_x]


def _row_chunks(n_rows, n_chunks):
    """Contiguous ``[(r0, r1), ...]`` partition of ``range(n_rows)`` into n_chunks."""
    step = max(1, -(-n_rows // n_chunks))  # ceil(n_rows / n_chunks)
    return [(i, min(i + step, n_rows)) for i in range(0, n_rows, step)]


def clean_cube(
    residual_cube,
    psf_cube,
    model_cube,
    mask_cube,
    max_iter,
    gain,
    threshold,
    num_threads=1,
    clean_box=(-1, -1, -1, -1),
):
    """Hogbom CLEAN a ``(nt, nf, np, ny, nx)`` cube, threaded across+within planes.

    Drop-in analogue of ``deconvolvers.hogbom.clean_cube`` (the C++ deconvolver):
    same per-plane ``max_iter`` / ``threshold`` arrays, scalar ``gain``, and the
    residual / model cubes are updated in place.

    Parameters
    ----------
    residual_cube, model_cube : numpy.ndarray
        ``(nt, nf, np, ny, nx)`` float32/float64, C-contiguous. Updated in place.
    psf_cube : numpy.ndarray
        ``(nt, nf, np_psf, ny, nx)`` with ``np_psf == np`` or ``1`` (broadcast).
    mask_cube : numpy.ndarray or None
        ``(nt, nf, np, ny, nx)`` bool, or ``None`` / empty for no mask.
    max_iter : numpy.ndarray
        ``(nt, nf, np)`` int per-plane iteration limits.
    gain : float
        CLEAN loop gain.
    threshold : numpy.ndarray
        ``(nt, nf, np)`` per-plane stopping thresholds.
    num_threads : int, optional
        Worker-thread count.  ``<= 0`` uses ``os.cpu_count()``.  Default ``1``.
    clean_box : tuple, optional
        Only the full-image box ``(-1, -1, -1, -1)`` is supported (use a mask to
        restrict the region).

    Returns
    -------
    dict
        ``{"iterations_performed": (nt, nf, np) int array}``.
    """
    if not NUMBA_AVAILABLE:  # pragma: no cover
        raise ImportError("hogbom_many_threads requires numba.")
    if tuple(clean_box) != (-1, -1, -1, -1):
        raise NotImplementedError(
            "hogbom_many_threads supports only the full-image clean_box "
            "(-1, -1, -1, -1); use a mask to restrict the region."
        )

    nt, nf, npol, ny, nx = residual_cube.shape
    nplanes = nt * nf * npol
    npol_psf = psf_cube.shape[2]

    import os

    n_threads = (
        int(num_threads) if num_threads and num_threads > 0 else (os.cpu_count() or 1)
    )

    # Flatten the plane axes (views; the cubes are C-contiguous).
    residual = residual_cube.reshape(nplanes, ny, nx)
    model = model_cube.reshape(nplanes, ny, nx)
    psf = psf_cube.reshape(nt * nf * npol_psf, ny, nx)

    # Map each image plane to its PSF plane (single-pol PSF broadcasts).
    psf_map = np.empty(nplanes, dtype=np.int64)
    for pl in range(nplanes):
        tt = pl // (nf * npol)
        rem = pl % (nf * npol)
        nn = rem // npol
        pp = rem % npol
        pidx = 0 if npol_psf == 1 else pp
        psf_map[pl] = (tt * nf + nn) * npol_psf + pidx

    use_mask = (
        mask_cube is not None and getattr(mask_cube, "size", 0) == residual_cube.size
    )
    if use_mask:
        mask = np.ascontiguousarray(mask_cube, dtype=np.bool_).reshape(nplanes, ny, nx)
    else:
        # Dummy (never indexed when use_mask is False) -- keep numba's types stable.
        mask = np.ones((nplanes, 1, 1), dtype=np.bool_)

    niter_flat = np.ascontiguousarray(max_iter, dtype=np.int64).reshape(nplanes)
    thr_flat = np.ascontiguousarray(threshold, dtype=residual_cube.dtype).reshape(
        nplanes
    )
    iters_out = np.zeros(nplanes, dtype=np.int64)

    active = np.ones(nplanes, dtype=np.bool_)
    row_max = np.full((nplanes, ny), -1.0, dtype=residual_cube.dtype)
    row_ix = np.zeros((nplanes, ny), dtype=np.int64)
    peak_y = np.zeros(nplanes, dtype=np.int64)
    peak_x = np.zeros(nplanes, dtype=np.int64)
    peak_pv = np.zeros(nplanes, dtype=residual_cube.dtype)
    gain_t = residual_cube.dtype.type(gain)

    n_rows = nplanes * ny
    max_niter = int(niter_flat.max()) if nplanes else 0
    n_workers = max(1, min(n_threads, n_rows))
    chunks = _row_chunks(n_rows, n_workers)

    def _run(it, pool):
        # Peak search across (plane, row), then serial combine, then subtract.
        if pool is None:
            _peak_range(
                residual,
                mask,
                use_mask,
                niter_flat,
                active,
                it,
                ny,
                nx,
                0,
                n_rows,
                row_max,
                row_ix,
            )
        else:
            list(
                pool.map(
                    lambda c: _peak_range(
                        residual,
                        mask,
                        use_mask,
                        niter_flat,
                        active,
                        it,
                        ny,
                        nx,
                        c[0],
                        c[1],
                        row_max,
                        row_ix,
                    ),
                    chunks,
                )
            )
        any_active = _combine(
            residual,
            model,
            active,
            niter_flat,
            thr_flat,
            it,
            ny,
            gain_t,
            row_max,
            row_ix,
            peak_y,
            peak_x,
            peak_pv,
            iters_out,
        )
        if not any_active:
            return False
        if pool is None:
            _subtract_range(
                residual,
                psf,
                psf_map,
                active,
                niter_flat,
                it,
                ny,
                nx,
                0,
                n_rows,
                peak_y,
                peak_x,
                peak_pv,
            )
        else:
            list(
                pool.map(
                    lambda c: _subtract_range(
                        residual,
                        psf,
                        psf_map,
                        active,
                        niter_flat,
                        it,
                        ny,
                        nx,
                        c[0],
                        c[1],
                        peak_y,
                        peak_x,
                        peak_pv,
                    ),
                    chunks,
                )
            )
        return True

    if n_workers == 1:
        for it in range(max_niter):
            if not _run(it, None):
                break
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for it in range(max_niter):
                if not _run(it, pool):
                    break

    return {"iterations_performed": iters_out.reshape(nt, nf, npol)}
