"""
Fast CPU kernels (C++ extensions) selected by ``ASTROVIPER_KERNELS=fast``.

They are alternatives to the reference extensions, which stay untouched: the
switch in ``astroviper._kernels`` routes a hooked reference entry to the
module here that implements the same entry with the same signature.

Layout, one subdirectory per kernel::

    fast/<kernel>/CMakeLists.txt   astroviper_add_fast_kernel_module(...)
    fast/<kernel>/probe.cpp        compile probe of the features it needs
    fast/<kernel>/include/         headers (reference headers are only read)
    fast/<kernel>/src/             generic sources, plus *_avx2.cpp units
    fast/<kernel>/python/bindings.cpp
    fast/<kernel>/__init__.py      re-exports the entries of the extension

The modules are built only with ``ASTROVIPER_BUILD_FAST_KERNELS=ON`` (the
default on Linux x86_64 with GCC >= 14 or Clang >= 17), and each one only if
its compile probe passes (``cmake/AstroviperKernels.cmake``). Selecting a
kernel whose module was not built raises ``ImportError`` at its first call.
"""
