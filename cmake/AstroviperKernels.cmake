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

# Sets _configs to the configurations that are built (each of
# CMAKE_CONFIGURATION_TYPES with a multi-config generator, else
# CMAKE_BUILD_TYPE) or, with no build type, to None, and _none to whether it
# is None: the sources then get no per-configuration flags, and nor do the
# checks (CMAKE_<LANG>_FLAGS_NONE is normally unset).
macro(_av_configurations)
    set(_none FALSE)
    get_property(_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(_multi AND CMAKE_CONFIGURATION_TYPES)
        set(_configs ${CMAKE_CONFIGURATION_TYPES})
    elseif(CMAKE_BUILD_TYPE)
        set(_configs "${CMAKE_BUILD_TYPE}")
    else()
        set(_configs None)
        set(_none TRUE)
    endif()
endmacro()

# check_cxx_source_compiles(<code> <var>) built as configuration Debug with
# <flags> as its CMAKE_CXX_FLAGS_DEBUG, both set in this scope only. The test
# project of try_compile has its generator's default configurations, not
# CMAKE_CONFIGURATION_TYPES (Ninja Multi-Config: Debug, Release and
# RelWithDebInfo), so a check built as another configuration (MinSizeRel, or a
# custom one) would fail for want of its build file; Debug is a default
# configuration of every generator. A toolchain file that sets
# CMAKE_CONFIGURATION_TYPES sets them in the test project too, perhaps without
# Debug, so with a multi-config generator a failed check is built again as
# <config> with the same flags, and <var> passes if that build does.
function(_av_check_compiles code var flags config)
    set(CMAKE_TRY_COMPILE_CONFIGURATION Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "${flags}")
    check_cxx_source_compiles("${code}" ${var})
    get_property(_multi GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
    if(NOT ${var} AND _multi AND NOT config STREQUAL "Debug")
        string(TOUPPER "${config}" _upper)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "${config}")
        set(CMAKE_CXX_FLAGS_${_upper} "${flags}")
        check_cxx_source_compiles("${code}" ${var}_AS_${_upper})
        if(${var}_AS_${_upper})
            set(${var} 1 CACHE INTERNAL "Test ${var}")
        endif()
    endif()
endfunction()

# ---------------------------------------------------------------------------
# _av_fast_check(<prefix> <code> <key> [<option>...])
#
# _av_check_compiles(<code>) for each configuration that is built (each
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
    _av_configurations()
    get_property(_options DIRECTORY PROPERTY COMPILE_OPTIONS)
    string(GENEX_STRIP "${_options};${ARGN}" _options)
    list(TRANSFORM _options REPLACE "^SHELL:" "")
    string(REPLACE ";" " " _options "${_options}")
    foreach(_config IN LISTS _configs)
        string(TOUPPER "${_config}" _upper)
        string(STRIP "${CMAKE_CXX_FLAGS_${_upper}} ${_options}" _config_flags)
        string(CONCAT _inputs "${code}|${key}|${CMAKE_REQUIRED_INCLUDES}|${CMAKE_REQUIRED_FLAGS}"
            "|${CMAKE_REQUIRED_LINK_OPTIONS}|${CMAKE_REQUIRED_LIBRARIES}|${CMAKE_CXX_COMPILER}"
            "|${CMAKE_CXX_COMPILER_VERSION}|${CMAKE_CXX_STANDARD}|${CMAKE_CXX_FLAGS}"
            "|${_config_flags}|${CMAKE_EXE_LINKER_FLAGS}")
        string(SHA256 _hash "${_inputs}")
        string(SUBSTRING "${_hash}" 0 16 _hash)
        _av_check_compiles("${code}" ${prefix}_${_hash} "${_config_flags}" "${_config}")
        if(NOT ${prefix}_${_hash})
            set(_av_failed ${prefix}_${_hash} PARENT_SCOPE)
            if(_none)
                set(_av_failed_where "an empty CMAKE_BUILD_TYPE" PARENT_SCOPE)
            else()
                set(_av_failed_where "configuration ${_config}" PARENT_SCOPE)
            endif()
            string(STRIP "${CMAKE_CXX_FLAGS} ${_config_flags}" _flags)
            set(_av_failed_flags "${_flags}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(_av_failed "" PARENT_SCOPE)
endfunction()

# Sets <out> to whether Clang 17 or 18 links crtfastmath.o (when it finds one,
# as in a GCC installation) with <flags>: when -Ofast is their last -O option,
# or -ffast-math or -funsafe-math-optimizations comes after their last
# -fno-fast-math and -fno-unsafe-math-optimizations
# (ToolChain::isFastMathRuntimeAvailable).
function(_av_crtfastmath flags out)
    separate_arguments(_args NATIVE_COMMAND "${flags}")
    set(_level "")
    set(_fast_math FALSE)
    foreach(_arg IN LISTS _args)
        if(_arg MATCHES "^(-O([0-9]+|[sgz]|fast)?|--optimize(=.*)?)$")
            set(_level "${_arg}")
        elseif(_arg MATCHES "^-f(fast-math|unsafe-math-optimizations)$")
            set(_fast_math TRUE)
        elseif(_arg MATCHES "^-fno-(fast-math|unsafe-math-optimizations)$")
            set(_fast_math FALSE)
        endif()
    endforeach()
    if(_level STREQUAL "-Ofast" OR _fast_math)
        set(${out} TRUE PARENT_SCOPE)
    else()
        set(${out} FALSE PARENT_SCOPE)
    endif()
endfunction()

# ---------------------------------------------------------------------------
# _av_clang_fast_math_runtime()
#
# With the flags of _av_crtfastmath, Clang 17 and 18 link crtfastmath.o into a
# shared object as into a program (on Linux, MinGW and Solaris; Clang 19 and
# later into programs only, unless given -mdaz-ftz). Its constructor sets
# flush-to-zero and denormals-are-zero in the thread that loads the module.
# Every module's link line, reference or fast, has CMAKE_CXX_FLAGS, the
# configuration's flags, CMAKE_MODULE_LINKER_FLAGS (LDFLAGS), the
# configuration's module linker flags and the directory's link options, so
# with such flags no fast module runs as written. The same test of a compile
# line makes them compile as if crtfastmath.o were linked (on x86 Linux,
# double operations under -fdenormal-fp-math=preserve-sign), which is the
# default that -fno-fast-math restores, so the fast units' -fno-fast-math
# cannot undo -Ofast (Linux::getDefaultDenormalModeForType and
# RenderFloatingPointOptions). For each configuration that is built, tests the
# link line and the fast units' compile line (CMAKE_CXX_FLAGS, the
# configuration's flags, the directory's compile options, then
# -fno-fast-math), each after the compiler's own arguments
# (CMAKE_CXX_COMPILER_ARG1), on any platform. Sets _av_failed in the caller's
# scope to "link" or "compile" for the first configuration that fails, with
# _av_failed_where and _av_failed_flags, or to "" if none does. Options inside
# generator expressions are not seen, nor are those of Clang's configuration
# files.
# ---------------------------------------------------------------------------
function(_av_clang_fast_math_runtime)
    _av_configurations()
    foreach(_kind COMPILE LINK)
        get_property(_options DIRECTORY PROPERTY ${_kind}_OPTIONS)
        string(GENEX_STRIP "${_options}" _options)
        list(TRANSFORM _options REPLACE "^SHELL:" "")
        string(REPLACE ";" " " _${_kind}_options "${_options}")
    endforeach()
    foreach(_config IN LISTS _configs)
        string(TOUPPER "${_config}" _upper)
        set(_flags "${CMAKE_CXX_COMPILER_ARG1} ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_upper}}")
        set(_link "${_flags} ${CMAKE_MODULE_LINKER_FLAGS} ${CMAKE_MODULE_LINKER_FLAGS_${_upper}}")
        string(REGEX REPLACE "  +" " " _link "${_link} ${_LINK_options}")
        string(REGEX REPLACE "  +" " " _compile "${_flags} ${_COMPILE_options}")
        string(STRIP "${_link}" _link)
        string(STRIP "${_compile}" _compile)
        _av_crtfastmath("${_link}" _on_link)
        _av_crtfastmath("${_compile} -fno-fast-math" _on_compile)
        if(_on_link OR _on_compile)
            if(_on_link)
                set(_av_failed link PARENT_SCOPE)
                set(_av_failed_flags "${_link}" PARENT_SCOPE)
            else()
                set(_av_failed compile PARENT_SCOPE)
                set(_av_failed_flags "${_compile}" PARENT_SCOPE)
            endif()
            if(_none)
                set(_av_failed_where "an empty CMAKE_BUILD_TYPE" PARENT_SCOPE)
            else()
                set(_av_failed_where "configuration ${_config}" PARENT_SCOPE)
            endif()
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
#                 -ffp-contract=off, when it and the flags of a
#                 configuration that is built do not round every
#                 floating-point operation as written
#                 (fast/common/strict_fp.hpp), or when the compiler is Clang
#                 17 or 18 and those flags make it link crtfastmath.o
#                 (_av_clang_fast_math_runtime). Both checks are compiled
#                 with the flags the sources get (_av_fast_check).
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

    # -fno-fast-math cannot undo what Clang 17 and 18 do with some of these
    # flags: link crtfastmath.o into every module, or compile as if they did
    # (_av_clang_fast_math_runtime).
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
        _av_clang_fast_math_runtime()
        if(_av_failed)
            _av_skip_fast_kernels("Clang ${CMAKE_CXX_COMPILER_VERSION} links crtfastmath.o, which "
                "sets flush-to-zero and denormals-are-zero in the thread that loads a module, "
                "into every module, or compiles as if it did, with the ${_av_failed} flags of "
                "${_av_failed_where} (${_av_failed_flags}): Clang 17 and 18 do so when -Ofast "
                "is their last -O option, and link it when -ffast-math or "
                "-funsafe-math-optimizations comes after their last -fno-fast-math and "
                "-fno-unsafe-math-optimizations")
            return()
        endif()
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
