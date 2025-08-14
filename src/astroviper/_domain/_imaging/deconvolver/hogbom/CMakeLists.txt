cmake_minimum_required(VERSION 3.15...3.27)

project(hogbom_deconvolve 
    VERSION 0.1.0
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
pybind11_add_module(hogbom_deconvolve ${SOURCES})

# Set target properties
target_include_directories(hogbom_deconvolve PRIVATE 
    include
)

# Compiler-specific options
target_compile_features(hogbom_deconvolve PRIVATE cxx_std_17)

# Set optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(hogbom_deconvolve PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g>
        $<$<CXX_COMPILER_ID:MSVC>:/Od /Zi>
    )
else()
    target_compile_options(hogbom_deconvolve PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/O2>
    )
endif()

# Platform-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(hogbom_deconvolve PRIVATE -stdlib=libc++)
    target_link_options(hogbom_deconvolve PRIVATE -stdlib=libc++)
endif()

if(MSVC)
    target_compile_options(hogbom_deconvolve PRIVATE /EHsc)
endif()

# Version info
target_compile_definitions(hogbom_deconvolve PRIVATE 
    VERSION_INFO="${PROJECT_VERSION}"
)

# Install the module in the python package directory
install(TARGETS hogbom_deconvolve DESTINATION hogbom_deconvolve)
