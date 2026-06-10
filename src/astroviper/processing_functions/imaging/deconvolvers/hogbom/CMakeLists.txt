cmake_minimum_required(VERSION 3.15...3.27)

project(astroviper_hogbom
    VERSION ${CMAKE_PROJECT_VERSION}
    DESCRIPTION "Hogbom CLEAN algorithm implementation"
    LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(pybind11 REQUIRED)

# Define source files
set(SOURCES
    src/hclean.cpp
    python/bindings.cpp
)

# Create pybind11 module
pybind11_add_module(_hogbom_ext ${SOURCES})

# Set target properties
target_include_directories(_hogbom_ext PRIVATE 
    include
)

# Compiler-specific options
target_compile_features(_hogbom_ext PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_hogbom_ext PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g>
    )
else()
    # Base optimization flags for all platforms
    target_compile_options(_hogbom_ext PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -pipe -fPIC -DNDEBUG>
    )
endif()

# Disable floating-point expression contraction (FMA fusion) so the float32
# CLEAN arithmetic rounds identically across compilers/architectures. GCC
# defaults to -ffp-contract=fast and Apple Clang to =on, which fuse a*b+c with
# a single rounding; with =off it is a rounded multiply then a rounded subtract
# -- bit-identical on any IEEE-754 platform. See the note on the multi-cycle
# stakeholder test: this stabilises iter_done across platforms but puts a few
# pathologically-sensitive CLEAN planes (e.g. channel 2 of the twhya test) onto
# a different bifurcation than the contraction-on reference image was blessed
# with.
# AppleClang is listed explicitly; the GNU,Clang generator expressions above do
# NOT match it (CMake reports macOS as "AppleClang").
target_compile_options(_hogbom_ext PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>
)

# Platform-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(_hogbom_ext PRIVATE -stdlib=libc++)
    target_link_options(_hogbom_ext PRIVATE -stdlib=libc++)
endif()

# Version info
target_compile_definitions(_hogbom_ext PRIVATE 
    VERSION_INFO="${PROJECT_VERSION}"
)

# Install the module in the python package directory
install(TARGETS _hogbom_ext DESTINATION astroviper/processing_functions/imaging/deconvolvers/hogbom)
