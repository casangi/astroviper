# Fast CPU kernels selected at run time by ASTROVIPER_KERNELS (see
# cmake/AstroviperKernels.cmake). Each kernel is a subdirectory whose
# CMakeLists.txt calls astroviper_add_fast_kernel_module(); register it with one
# add_subdirectory() line below. With ASTROVIPER_BUILD_FAST_KERNELS=OFF nothing
# here is configured and the build is the reference tree alone.
include(AstroviperKernels)

if(NOT ASTROVIPER_BUILD_FAST_KERNELS)
    message(STATUS "AstroVIPER: fast kernels are not built (ASTROVIPER_BUILD_FAST_KERNELS=OFF)")
    return()
endif()
add_subdirectory(dft)
add_subdirectory(hogbom)
