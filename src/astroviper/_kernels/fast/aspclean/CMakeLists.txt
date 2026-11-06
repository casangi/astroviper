# The fast Asp model update (ASTROVIPER_KERNELS=asp=fast), see __init__.py. It
# has no AVX2 unit: the baseline lanes give the reference's NaN bits, AVX2's do
# not. The reference's headers (asp_clean.hpp, asp_fft.hpp, asp_lbfgs.hpp and
# pocketfft_hdronly.h) are read from its include directory.
astroviper_add_fast_kernel_module(
    NAME _aspclean_fast_ext
    PROBE probe.cpp
    SOURCES src/asp_clean_fast.cpp
            python/bindings.cpp
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/astroviper/processing_functions/imaging/deconvolvers/aspclean/include
)
