# The fast Hogbom minor cycle (ASTROVIPER_KERNELS=hogbom=fast), see __init__.py.
# The reference's headers (IterationHistory, the stop tests) are read only.
astroviper_add_fast_kernel_module(
    NAME _hogbom_fast_ext
    PROBE probe.cpp
    SOURCES src/hclean_fast.cpp
            src/hclean_fast_generic.cpp
            python/bindings.cpp
    AVX2_SOURCES src/hclean_fast_avx2.cpp
    INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/astroviper/processing_functions/imaging/deconvolvers/hogbom/include
)
