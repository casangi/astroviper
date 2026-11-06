# ---------------------------------------------------------------------------
# AstroviperKernels.cmake
#
# Build rules of the fast (non-reference) kernels under
# src/astroviper/_kernels/fast/, which the ASTROVIPER_KERNELS switch selects at
# run time beside the untouched reference extensions. Included after
# AstroviperPybind.cmake, whose compiler flags every fast module inherits.
#
#   ASTROVIPER_BUILD_FAST_KERNELS  option; OFF builds exactly the reference tree.
#   astroviper_add_fast_kernel_module()
#                                  one fast module: compile probe, pybind11
#                                  module, optional AVX2 translation units.
# ---------------------------------------------------------------------------
include_guard(GLOBAL)

include(AstroviperPybind)
include(CheckCXXSourceCompiles)

# Default ON only where the fast kernels are tested: Linux x86_64 with the
# repo's GCC / Clang floors. aarch64 and macOS stay OFF until their CI has run
# the fast-kernel tests; -DASTROVIPER_BUILD_FAST_KERNELS=ON overrides.
set(_av_fast_default OFF)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux"
   AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL AV_MIN_GNU_VERSION)
       OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
           AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL AV_MIN_CLANG_VERSION))
        set(_av_fast_default ON)
    endif()
endif()
option(ASTROVIPER_BUILD_FAST_KERNELS
    "Build the fast kernels selected by ASTROVIPER_KERNELS=fast"
    ${_av_fast_default})

# AVX2 translation units are added only when the target is x86_64.
# CMAKE_SYSTEM_PROCESSOR names the host when cross-compiling for macOS, so a
# universal or arm64 CMAKE_OSX_ARCHITECTURES turns them off as well. The
# sources also wrap their bodies in #if defined(__x86_64__) || defined(_M_X64).
set(_av_x86_64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$"
   AND (NOT CMAKE_OSX_ARCHITECTURES OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
   AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set(_av_x86_64 ON)
endif()

# ---------------------------------------------------------------------------
# astroviper_add_fast_kernel_module(NAME <name> PROBE <file> SOURCES <src>...
#                                   [AVX2_SOURCES <src>...]
#                                   [INCLUDE_DIRS <dir>...])
#
#   NAME          target / module name (e.g. _hogbom_fast_ext)
#   PROBE         a small complete C++ program (with main), relative to the
#                 calling CMakeLists.txt, using the language and library
#                 features the module needs. If it does not compile and link,
#                 the module is skipped with a message, so a toolchain gap can
#                 never break the reference wheel.
#   SOURCES       generic .cpp files (baseline ISA)
#   AVX2_SOURCES  *_avx2.cpp files, compiled with -mavx2 and without LTO (so no
#                 AVX2 code can be merged into a generic inline function); the
#                 module is compiled with AV_HAVE_AVX2=1 when they are added.
#                 Ignored on non-x86_64 targets.
#   INCLUDE_DIRS  as for astroviper_add_pybind_module()
# ---------------------------------------------------------------------------
function(astroviper_add_fast_kernel_module)
    cmake_parse_arguments(FK "" "NAME;PROBE" "SOURCES;AVX2_SOURCES;INCLUDE_DIRS" ${ARGN})
    if(NOT FK_NAME OR NOT FK_PROBE OR NOT FK_SOURCES)
        message(FATAL_ERROR
            "astroviper_add_fast_kernel_module: NAME, PROBE and SOURCES are required")
    endif()

    # The probe is built with the package's C++ standard (CMAKE_CXX_STANDARD),
    # standard library and threads, like the module.
    set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_REQUIRED_FLAGS "-stdlib=libc++")
        set(CMAKE_REQUIRED_LINK_OPTIONS "-stdlib=libc++")
    endif()
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${FK_PROBE}" _probe)
    check_cxx_source_compiles("${_probe}" AV_FAST_PROBE_${FK_NAME})
    if(NOT AV_FAST_PROBE_${FK_NAME})
        message(STATUS
            "AstroVIPER: skipping fast kernel ${FK_NAME}: its compile probe "
            "failed with ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}; "
            "ASTROVIPER_KERNELS selections of it will raise ImportError")
        return()
    endif()

    set(_include_dirs)
    if(FK_INCLUDE_DIRS)
        set(_include_dirs INCLUDE_DIRS ${FK_INCLUDE_DIRS})
    endif()
    astroviper_add_pybind_module(NAME ${FK_NAME} SOURCES ${FK_SOURCES} ${_include_dirs})

    if(FK_AVX2_SOURCES AND _av_x86_64)
        target_sources(${FK_NAME} PRIVATE ${FK_AVX2_SOURCES})
        set_source_files_properties(${FK_AVX2_SOURCES}
            PROPERTIES COMPILE_OPTIONS "-mavx2;-fno-lto")
        target_compile_definitions(${FK_NAME} PRIVATE AV_HAVE_AVX2=1)
    endif()
endfunction()
