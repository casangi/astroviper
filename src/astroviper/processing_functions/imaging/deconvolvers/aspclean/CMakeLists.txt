cmake_minimum_required(VERSION 3.15...3.27)

project(astroviper_aspclean
    VERSION ${CMAKE_PROJECT_VERSION}
    DESCRIPTION "Adaptive Scale Pixel (Asp / AAspClean) deconvolution"
    LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Threads REQUIRED)

# Define source files
set(SOURCES
    src/asp_clean.cpp
    python/bindings.cpp
)

# Create pybind11 module
pybind11_add_module(_aspclean_ext ${SOURCES})

# Set target properties
target_include_directories(_aspclean_ext PRIVATE
    include
)

target_link_libraries(_aspclean_ext PRIVATE Threads::Threads)

# Compiler-specific options
target_compile_features(_aspclean_ext PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_aspclean_ext PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g>
    )
else()
    # Base optimization flags for all platforms
    target_compile_options(_aspclean_ext PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -pipe -fPIC -DNDEBUG>
    )
endif()

# Disable floating-point expression contraction (FMA fusion) so the deconvolver
# arithmetic rounds identically across compilers and architectures (GCC defaults
# to -ffp-contract=fast, Apple Clang to =on). With =off, multiply-adds are a
# rounded multiply then a rounded add -- bit-identical on any IEEE-754 platform.
# AppleClang is listed explicitly; the GNU,Clang generator expressions above do
# NOT match it (CMake reports macOS as "AppleClang").
target_compile_options(_aspclean_ext PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>
)

# Platform-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(_aspclean_ext PRIVATE -stdlib=libc++)
    target_link_options(_aspclean_ext PRIVATE -stdlib=libc++)
endif()

# Version info
target_compile_definitions(_aspclean_ext PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

# Install the module in the python package directory
install(TARGETS _aspclean_ext DESTINATION astroviper/processing_functions/imaging/deconvolvers/aspclean)
