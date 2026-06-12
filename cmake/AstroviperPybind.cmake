# ---------------------------------------------------------------------------
# AstroviperPybind.cmake
#
# Single source of truth for how every AstroVIPER C++ / pybind11 extension is
# compiled. Edit THIS FILE to change a compiler flag for all kernels at once.
#
# Two ways this file is used:
#   * From the repo-root CMakeLists.txt (the normal scikit-build-core package
#     build): included once, then each module's CMakeLists.txt calls
#     astroviper_add_pybind_module().
#   * Standalone (cmake -S <module_dir> -B build, to iterate on a single
#     kernel): the module's CMakeLists.txt finds and includes this same file,
#     so a standalone build uses byte-for-byte the same flags as the package
#     build -- there is no second copy to keep in sync.
#
# include_guard(GLOBAL) makes a second include() from a subdirectory a no-op,
# so the flags target and helper function below are defined exactly once.
# ---------------------------------------------------------------------------
include_guard(GLOBAL)

# Let pybind11 locate Python via CMake's FindPython. Works for both the
# scikit-build-core build and a standalone configure.
set(PYBIND11_FINDPYTHON ON)

# Locate pybind11's CMake package config. scikit-build-core injects it during
# the package build (this first lookup succeeds). For a standalone
# `cmake -S <module> -B build`, fall back to asking the active Python where
# pybind11 ships its CMake config, so standalone builds work without the caller
# passing -Dpybind11_DIR=...
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Development.Module is needed for pybind11_add_module's python_add_library;
    # Interpreter is needed to query pybind11's CMake dir below.
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _av_pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_av_pybind11_cmakedir)
        list(APPEND CMAKE_PREFIX_PATH "${_av_pybind11_cmakedir}")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
endif()

# Prefer the -pthread compile/link flag form over a bare -lpthread.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# astroviper_cpp_flags: an INTERFACE target carrying every shared setting.
# Modules receive all of this with a single target_link_libraries(), done for
# you inside astroviper_add_pybind_module().
# ---------------------------------------------------------------------------
add_library(astroviper_cpp_flags INTERFACE)

# C++20 everywhere. The gridder needs it (std::atomic_ref); it is harmless for
# the deconvolvers and keeps one standard across the codebase. CXX_EXTENSIONS OFF
# pins strict -std=c++20 (not -std=gnu++20) for portable, standard-conformant
# builds. Set globally so every target created after this include inherits it.
set(CMAKE_CXX_EXTENSIONS OFF)
target_compile_features(astroviper_cpp_flags INTERFACE cxx_std_20)

# Scientific reproducibility: disable floating-point expression contraction
# (FMA fusion) so a*b+c is a rounded multiply then a rounded add -- bit-identical
# on any IEEE-754 platform. GCC defaults to -ffp-contract=fast and Apple Clang
# to =on, which fuse with a single rounding and make CLEAN/gridding results
# drift across compilers and architectures.
#   NOTE: CMake reports macOS's compiler as "AppleClang", which the GNU,Clang
#   generator expression does NOT match -- it must be listed explicitly.
target_compile_options(astroviper_cpp_flags INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=on>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-pipe>
)

# Use libc++ with any Clang (the default on macOS; needed for a consistent
# standard library when building with Clang on Linux).
target_compile_options(astroviper_cpp_flags INTERFACE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-stdlib=libc++>)
target_link_options(astroviper_cpp_flags INTERFACE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-stdlib=libc++>)

# Embed the package version (set by scikit-build-core). Falls back gracefully in
# a standalone configure where SKBUILD_PROJECT_VERSION is unset.
if(DEFINED SKBUILD_PROJECT_VERSION)
    set(_av_version "${SKBUILD_PROJECT_VERSION}")
else()
    set(_av_version "0.0.0+local")
endif()
target_compile_definitions(astroviper_cpp_flags INTERFACE
    VERSION_INFO="${_av_version}")

# Every AstroVIPER kernel uses std::thread.
target_link_libraries(astroviper_cpp_flags INTERFACE Threads::Threads)

# Optimization level (-O3/-DNDEBUG for Release, -g for Debug) and -fPIC are left
# to CMake: the package build pins build-type=Release in pyproject.toml, and
# pybind11 modules get position-independent code automatically. There is nothing
# to hand-roll here.

# ---------------------------------------------------------------------------
# astroviper_add_pybind_module(NAME <name> SOURCES <src>...
#                              [INCLUDE_DIRS <dir>...] [DESTINATION <dir>])
#
# Declare a pybind11 extension with all shared flags applied.
#   NAME         target / module name (e.g. _hogbom_ext)
#   SOURCES      .cpp files (relative to the calling CMakeLists.txt)
#   INCLUDE_DIRS optional; defaults to <module>/include
#   DESTINATION  optional; defaults to the module's path relative to src/, i.e.
#                its location inside the installed package
# ---------------------------------------------------------------------------
function(astroviper_add_pybind_module)
    cmake_parse_arguments(PB "" "NAME;DESTINATION" "SOURCES;INCLUDE_DIRS" ${ARGN})

    if(NOT PB_NAME)
        message(FATAL_ERROR "astroviper_add_pybind_module: NAME is required")
    endif()
    if(NOT PB_SOURCES)
        message(FATAL_ERROR "astroviper_add_pybind_module(${PB_NAME}): SOURCES is required")
    endif()
    if(NOT PB_INCLUDE_DIRS)
        set(PB_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    endif()
    if(NOT PB_DESTINATION)
        if(EXISTS "${CMAKE_SOURCE_DIR}/src")
            file(RELATIVE_PATH PB_DESTINATION
                 "${CMAKE_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")
        else()
            set(PB_DESTINATION ".")  # standalone build: no installed package tree
        endif()
    endif()

    pybind11_add_module(${PB_NAME} ${PB_SOURCES})
    target_include_directories(${PB_NAME} PRIVATE ${PB_INCLUDE_DIRS})
    target_link_libraries(${PB_NAME} PRIVATE astroviper_cpp_flags)
    install(TARGETS ${PB_NAME} DESTINATION ${PB_DESTINATION})
endfunction()
