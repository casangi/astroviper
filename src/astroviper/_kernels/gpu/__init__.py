"""
GPU kernels (CuPy raw kernels) selected by ``ASTROVIPER_KERNELS=<kernel>=gpu``.

They are alternatives to the reference extensions, which stay untouched, beside
the fast CPU kernels (``astroviper._kernels.fast``). Layout::

    gpu/_runtime.py     CuPy, the device, CUPY_GPU_MEMORY_LIMIT, the NVRTC modules
    gpu/hogbom.py       hogbom=gpu: clean_cube, clean_cube_many_threads
    gpu/prewarm.py      python -m astroviper._kernels.gpu.prewarm
    gpu/cuda/*.cu       the kernel sources: package data, compiled at first use

Importing this package imports neither CuPy nor any CUDA library: CuPy is
imported at the first call that runs on the GPU (``_runtime.require_device``),
and a call that needs it raises ``RuntimeError`` when it is missing. The
sources are compiled by NVRTC at the first call (or loaded from CuPy's cache,
``CUPY_CACHE_DIR``): no CUDA toolkit is needed to build or install the
package. Install CuPy with the ``gpu`` extra, ``pip install 'astroviper[gpu]'``
(cupy-cuda13x with its NVRTC; Linux x86-64).
"""
