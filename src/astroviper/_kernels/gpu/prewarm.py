"""
``python -m astroviper._kernels.gpu.prewarm``: get the GPU kernels ready ahead
of the first timed call.

Checks the environment as ``python -m astroviper._kernels --check`` does,
creates the CUDA context, compiles the kernels (or loads them from CuPy's
cache, ``CUPY_CACHE_DIR``) of the GPU implementations that
``ASTROVIPER_KERNELS`` selects (all of them when it selects none) and runs them
once: each Hogbom entry for each image dtype, on a small cube, and each FFT
entry for each complex dtype, on small planes of even and odd sizes (which
loads cuFFT), then prints the timings as one JSON line. Exits with status 1
and the reason on standard error when the GPU cannot run them (no CuPy, no
device, no NVRTC, no cuFFT, an invalid ``CUPY_GPU_MEMORY_LIMIT``,
``ASTROVIPER_KERNELS_GPU_MIN_PIXELS`` or
``ASTROVIPER_KERNELS_GPU_FFT_MIN_PIXELS``, a compile error, a plane that does
not fit in the device memory that is free and allowed). :func:`prewarm` does
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

from astroviper._kernels._switch import selected
from astroviper._kernels.gpu import _runtime, fft, hogbom

#: The GPU implementation of each kernel that has one.
MODULES = {"hogbom": hogbom, "fft": fft}


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


def _bound(call, kernel, entry):
    """``call``, the prewarm's call of ``entry`` as the GPU module bound it,
    or ``RuntimeError`` when the module declined it (None)."""
    if call is None:
        raise RuntimeError(f"{kernel}=gpu: the GPU declined the prewarm's {entry} call")
    return call


def prewarm():
    """Initialize CUDA, compile the selected GPU kernels (see the module
    docstring) and run them once; returns the timings (seconds), the kernels
    and the device, as a dict."""
    kernels = [k for k, name in selected().items() if name == "gpu"] or list(MODULES)
    t0 = time.perf_counter()
    for kernel in kernels:
        MODULES[kernel].check_environment()
    device = _runtime.require_device()
    t1 = time.perf_counter()
    if "hogbom" in kernels:
        for entry in hogbom._ENTRIES:
            for dtype in (np.float32, np.float64):
                call = hogbom._bind(entry, (), _cube(dtype))
                hogbom._run(_bound(call, "hogbom", entry))
    if "fft" in kernels:
        for entry in fft._INPUT:
            reference = fft.impl("fft", entry, "ref")
            for dtype in (np.complex64, np.complex128):
                for shape in ((64, 64), (63, 65)):
                    # scipy.fft is always installed, and no backend changes the GPU's work
                    args, kwargs = (np.ones(shape, dtype),), {"fft_backend": "scipy"}
                    call = fft._bind(entry, reference, args, kwargs)
                    fft._run(_bound(call, "fft", entry))  # _DoesNotFit: RuntimeError
    t2 = time.perf_counter()
    return {
        "kernels": kernels,
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
