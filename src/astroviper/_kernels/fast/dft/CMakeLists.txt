# The fast dft_predict (ASTROVIPER_KERNELS=dft=fast), see __init__.py.
astroviper_add_fast_kernel_module(
    NAME _dft_fast_ext
    PROBE probe.cpp
    SOURCES src/dft_pack.cpp
            src/dft_separable.cpp
            src/dft_s_generic.cpp
            python/bindings.cpp
    AVX2_SOURCES src/dft_s_avx2.cpp
)
