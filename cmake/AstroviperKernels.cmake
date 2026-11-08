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
# _av_fast_check(<prefix> <code> <key> [<option>...])
#
# check_cxx_source_compiles(<code>) for each configuration that is built (each
# of CMAKE_CONFIGURATION_TYPES with a multi-config generator, else
# CMAKE_BUILD_TYPE, or none), with the caller's CMAKE_REQUIRED_* settings and
# the flags that the fast sources get there, in their order: CMAKE_CXX_FLAGS,
# CMAKE_CXX_FLAGS_<CONFIG>, the directory's compile options (add_compile_options,
# also from a toolchain file, CMAKE_PROJECT_INCLUDE or a superproject) and
# <option>.... Options inside generator expressions are not seen here, so
# strict_fp.hpp's #error stops the build instead. Each verdict is cached as
# <prefix>_<hash>, the hash taken over all that decides it (<code>, <key>, the
# compiler, the CMAKE_REQUIRED_* settings and these flags), so a changed input
# re-runs its check and an incremental build directory never keeps a stale
# verdict. Sets _av_failed in the caller's scope to the cache variable of the
# first configuration that fails, with _av_failed_where and _av_failed_flags,
# or to "" if none does.
# ---------------------------------------------------------------------------
function(_av_fast_check prefix code key)
    set(_none FALSE)
    get_property(_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_multi AND CMAKE_CONFIGURATION_TYPES)
        set(_configs ${CMAKE_CONFIGURATION_TYPES})
    elseif(CMAKE_BUILD_TYPE)
        set(_configs "${CMAKE_BUILD_TYPE}")
    else()
        # No build type: the sources get no per-configuration flags, and nor
        # does the check (CMAKE_CXX_FLAGS_NONE is normally unset).
        set(_configs None)
        set(_none TRUE)
    endif()
    get_property(_options DIRECTORY PROPERTY COMPILE_OPTIONS)
    string(GENEX_STRIP "${_options};${ARGN}" _options)
    list(TRANSFORM _options REPLACE "^SHELL:" "")
    string(REPLACE ";" " " _options "${_options}")
    foreach(_config IN LISTS _configs)
        # try_compile builds CMAKE_TRY_COMPILE_CONFIGURATION with that
        # configuration's CMAKE_CXX_FLAGS_<CONFIG>, both set in this scope only.
        string(TOUPPER "${_config}" _upper)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "${_config}")
        string(STRIP "${CMAKE_CXX_FLAGS_${_upper}} ${_options}" CMAKE_CXX_FLAGS_${_upper})
        string(CONCAT _inputs "${code}|${key}|${CMAKE_REQUIRED_INCLUDES}|${CMAKE_REQUIRED_FLAGS}"
            "|${CMAKE_REQUIRED_LINK_OPTIONS}|${CMAKE_REQUIRED_LIBRARIES}|${CMAKE_CXX_COMPILER}"
            "|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CXX_STANDARD}|${CMAKE_CXX_FLAGS}|${_config}"
            "|${CMAKE_CXX_FLAGS_${_upper}}|${CMAKE_EXE_LINKER_FLAGS}")
        string(SHA256 _hash "${_inputs}")
        string(SUBSTRING "${_hash}" 0 16 _hash)
        check_cxx_source_compiles("${code}" ${prefix}_${_hash})
        if(NOT ${prefix}_${_hash})
            set(_av_failed ${prefix}_${_hash} PARENT_SCOPE)
            if(_none)
                set(_av_failed_where "an empty CMAKE_BUILD_TYPE" PARENT_SCOPE)
            else()
                set(_av_failed_where "configuration ${_config}" PARENT_SCOPE)
            endif()
            string(STRIP "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_upper}}" _flags)
            set(_av_failed_flags "${_flags}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(_av_failed "" PARENT_SCOPE)
endfunction()

# Says once per configure run, for each reason, that the fast kernels are
# skipped and why.
function(_av_skip_fast_kernels)
    string(CONCAT _reason ${ARGN})
    string(SHA256 _id "${_reason}")
    get_property(_said GLOBAL PROPERTY _AV_FAST_SKIPPED_${_id})
    if(NOT _said)
        set_property(GLOBAL PROPERTY _AV_FAST_SKIPPED_${_id} TRUE)
        message(STATUS "AstroVIPER: skipping the fast kernels: ${_reason}; "
            "ASTROVIPER_KERNELS selections of them will raise ImportError")
    endif()
endfunction()

# ---------------------------------------------------------------------------
# astroviper_add_fast_kernel_module(NAME <name> PROBE <file> SOURCES <src>...
#                                   [AVX2_SOURCES <src>...]
#                                   [INCLUDE_DIRS <dir>...])
#
#   NAME          target / module name (e.g. _hogbom_fast_ext)
#   PROBE         a small complete C++ program (with main), relative to the
#                 calling CMakeLists.txt, using the language and library
#                 features the module needs and including strict_fp.hpp, as
#                 its sources do. If it does not compile and link, the module
#                 is skipped with a message, so a toolchain gap can never
#                 break the reference wheel. Every fast module is skipped,
#                 with one message, when the compiler is not given
#                 -ffp-contract=off, or when it and the flags of a
#                 configuration that is built do not round every
#                 floating-point operation as written
#                 (fast/common/strict_fp.hpp). Both checks are compiled with
#                 the flags the sources get (_av_fast_check).
#   SOURCES       generic .cpp files (baseline ISA)
#   AVX2_SOURCES  *_avx2.cpp files, compiled with -mavx2 and without LTO (so
#                 the link-time optimiser cannot inline their AVX2 code into
#                 generic functions); the module is compiled with
#                 AV_HAVE_AVX2=1 when they are added. Ignored on non-x86_64
#                 targets. -fno-lto does not stop the linker from picking an
#                 AVX2 copy of a weak (COMDAT) symbol: an inline function or
#                 template instantiation emitted by such a unit can replace a
#                 generic caller's copy. So an AVX2 unit keeps all its code in
#                 an anonymous namespace, uses no std containers or other
#                 header templates, and must define no weak symbol (nm: no
#                 W/V/u), which test_binary_isa.py checks on the -fno-lto
#                 objects of each fast module (a WP3 acceptance item). The
#                 inline functions of the standard headers it still uses
#                 (std::abs of <cmath>) are inlined at -O1 and above but
#                 emitted as weak symbols at -O0, so in a Debug build these
#                 units are compiled with -Og.
#   INCLUDE_DIRS  further include directories (also the probe's); the
#                 module's include/ and fast/common/, the headers shared by
#                 the fast modules, are always on the module's include path,
#                 and fast/common/ on the probe's
#
# With Clang every unit of a fast module is compiled with -fno-fast-math after
# the flags it inherits (see below).
# ---------------------------------------------------------------------------
function(astroviper_add_fast_kernel_module)
    cmake_parse_arguments(FK "" "NAME;PROBE" "SOURCES;AVX2_SOURCES;INCLUDE_DIRS" ${ARGN})
    if(NOT FK_NAME OR NOT FK_PROBE OR NOT FK_SOURCES)
        message(FATAL_ERROR
            "astroviper_add_fast_kernel_module: NAME, PROBE and SOURCES are required")
    endif()

    # Contraction (a * b + c fused into one rounding) shows in no macro, so
    # only the compilers that astroviper_cpp_flags gives -ffp-contract=off
    # build the fast modules.
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        _av_skip_fast_kernels("${CMAKE_CXX_COMPILER_ID} is not given -ffp-contract=off "
            "(cmake/AstroviperPybind.cmake gives it to GNU, Clang and AppleClang only), so "
            "its floating-point operations could be contracted")
        return()
    endif()

    # Clang announces by macros only -ffast-math with every one of its parts
    # (__FAST_MATH__) and finite math (__FINITE_MATH_ONLY__), so strict_fp.hpp
    # cannot see -funsafe-math-optimizations, -fassociative-math,
    # -freciprocal-math, -fno-signed-zeros, -fapprox-func, -fno-honor-nans,
    # -ffp-model=fast or -ffast-math -fno-finite-math-only. With Clang the
    # fast modules and their checks are therefore compiled with -fno-fast-math
    # after the flags they inherit, which turns each of these off again, and
    # -ffp-contract=off (astroviper_cpp_flags) still decides contraction. GCC
    # announces each of them, which skips the fast modules.
    set(_fp_options "")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(_fp_options -fno-fast-math)
    endif()

    # The checks are built with the package's C++ standard (CMAKE_CXX_STANDARD),
    # standard library and threads, with the flags of the sources
    # (_av_fast_check), and with fast/common/ and then the module's include
    # directories. Editing strict_fp.hpp or the probe re-runs configure, and
    # their content is part of the checks' keys.
    set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(CMAKE_REQUIRED_FLAGS "-stdlib=libc++")
        set(CMAKE_REQUIRED_LINK_OPTIONS "-stdlib=libc++")
    endif()
    set(_common "${PROJECT_SOURCE_DIR}/src/astroviper/_kernels/fast/common")
    set(CMAKE_REQUIRED_INCLUDES "${_common}")
    set(_strict_file "${_common}/strict_fp.hpp")
    set(_probe_file "${CMAKE_CURRENT_SOURCE_DIR}/${FK_PROBE}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_strict_file}" "${_probe_file}")
    file(READ "${_strict_file}" _strict)
    file(READ "${_probe_file}" _probe)
    set(_log "CMakeFiles/CMakeConfigureLog.yaml (CMakeError.log before CMake 3.26)")

    # The floating-point semantics first, shared by the fast modules, so that
    # flags that break them get their own message.
    _av_fast_check(AV_FAST_STRICT_FP "#include \"strict_fp.hpp\"\nint main() { return 0; }\n"
        "${_strict}" ${_fp_options})
    if(_av_failed)
        _av_skip_fast_kernels("${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} does not "
            "round every floating-point operation as written with the flags of "
            "${_av_failed_where} (${_av_failed_flags}): fast/common/strict_fp.hpp refuses "
            "-ffast-math, -Ofast, -funsafe-math-optimizations, -fassociative-math, "
            "-freciprocal-math, -fno-signed-zeros, -ffinite-math-only, "
            "-fsingle-precision-constant and excess precision such as -mfpmath=387 (its "
            "error is check ${_av_failed} in ${_log})")
        return()
    endif()

    foreach(_dir IN LISTS FK_INCLUDE_DIRS)
        get_filename_component(_dir "${_dir}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        list(APPEND CMAKE_REQUIRED_INCLUDES "${_dir}")
    endforeach()
    _av_fast_check(AV_FAST_PROBE_${FK_NAME} "${_probe}" "${_strict}" ${_fp_options})
    if(_av_failed)
        message(STATUS
            "AstroVIPER: skipping fast kernel ${FK_NAME}: its compile probe failed with "
            "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} and the flags of "
            "${_av_failed_where} (${_av_failed_flags}; its error is check ${_av_failed} in "
            "${_log}); ASTROVIPER_KERNELS selections of it will raise ImportError")
        return()
    endif()

    astroviper_add_pybind_module(NAME ${FK_NAME} SOURCES ${FK_SOURCES}
        INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" ${FK_INCLUDE_DIRS} "${_common}")
    target_compile_options(${FK_NAME} PRIVATE ${_fp_options})

    if(FK_AVX2_SOURCES AND _av_x86_64)
        target_sources(${FK_NAME} PRIVATE ${FK_AVX2_SOURCES})
        set_property(SOURCE ${FK_AVX2_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
            -mavx2 -fno-lto $<$<CONFIG:Debug>:-Og>)
        target_compile_definitions(${FK_NAME} PRIVATE AV_HAVE_AVX2=1)
    endif()
endfunction()
