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
    fast/common/                   headers shared by the modules: strict_fp.hpp,
                                   which every unit and probe must include (the
                                   floating-point guard), and isa.hpp (the
                                   instruction set chosen per process)

The modules are built only with ``ASTROVIPER_BUILD_FAST_KERNELS=ON`` (the
default on Linux x86_64 with GCC >= 14 or Clang >= 17), and only by GCC, Clang
or Apple Clang with flags under which strict_fp.hpp compiles, that is flags
that round every floating-point operation as written; each one also needs its
compile probe to pass (``cmake/AstroviperKernels.cmake``). Selecting a kernel
whose module was not built raises ``ImportError`` at its first call.
"""
