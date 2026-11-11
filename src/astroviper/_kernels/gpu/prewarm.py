"""
``python -m astroviper._kernels.gpu.prewarm``: get the GPU kernels ready ahead
of the first timed call.

Checks the environment as ``python -m astroviper._kernels --check`` does,
creates the CUDA context, compiles the kernels (or loads them from CuPy's
cache, ``CUPY_CACHE_DIR``) and runs each Hogbom entry on the GPU once for each
image dtype, on a small cube, then prints the timings as one JSON line. Exits
with status 1 and the reason on standard error when the GPU cannot run them
(no CuPy, no device, no NVRTC, an invalid ``CUPY_GPU_MEMORY_LIMIT`` or
``ASTROVIPER_KERNELS_GPU_MIN_PIXELS``, a compile error). :func:`prewarm` does
the same in the calling process, which then pays none of it at its first
call; as a Dask worker preload (``--preload astroviper._kernels.gpu.prewarm``,
or ``DASK_DISTRIBUTED__WORKER__PRELOAD``) it prewarms each worker, and a worker
that cannot use the GPU fails to start.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from astroviper._kernels.gpu import _runtime, hogbom


def _cube(dtype):
    """A 2-plane cube of 64 x 64 (two strips: the grid barrier runs) with a
    mask and a point source, as keyword arguments of the entries."""
    n = 64
    y = np.arange(n)[:, None] - n // 2
    x = np.arange(n)[None, :] - n // 2
    psf = np.exp(-0.5 * (x**2 + y**2) / 4.0).astype(dtype)
    residual = np.stack([np.roll(psf, (3, -5), (0, 1)), 0.5 * psf])[None, None]
    mask = np.zeros(residual.shape, bool)
    mask[..., 8:56, 8:56] = True
    return {
        "residual_cube": np.ascontiguousarray(residual),
        "psf_cube": np.ascontiguousarray(np.broadcast_to(psf, residual.shape)),
        "model_cube": np.zeros_like(residual),
        "mask_cube": mask,
        "max_iter_per_cycle_updated": np.full((1, 1, 2), 20, np.intc),
        "gain": 0.2,
        "threshold_per_cycle": np.zeros((1, 1, 2), dtype),
        "record_iteration_history": True,
    }


def prewarm():
    """Initialize CUDA, compile the kernels and run them once; returns the
    timings (seconds) and the device, as a dict."""
    t0 = time.perf_counter()
    hogbom.check_environment()
    device = _runtime.require_device()
    t1 = time.perf_counter()
    for entry in hogbom._ENTRIES:
        for dtype in (np.float32, np.float64):
            hogbom._run(hogbom._bind(entry, (), _cube(dtype)))
    t2 = time.perf_counter()
    return {
        "device": device.name,
        "versions": device.versions,
        "init_s": device.init_s,
        "require_device_s": t1 - t0,
        "compile_or_load_s": sum(device.compile_s.values()),
        "run_s": t2 - t1,
        "total_s": t2 - t0,
        "cupy_cache_dir": os.environ.get("CUPY_CACHE_DIR"),
    }


def dask_setup(worker):
    """Dask's hook of a worker preload: prewarm the worker's process."""
    prewarm()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m astroviper._kernels.gpu.prewarm")
    parser.parse_args(argv)
    try:
        timings = prewarm()
    except (RuntimeError, ValueError) as e:
        print(f"GPU prewarm failed: {e}", file=sys.stderr)
        return 1
    print(json.dumps(timings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
