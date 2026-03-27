cmake_minimum_required(VERSION 3.15...3.27)

project(astroviper_prolate_spheroidal_grid
    VERSION ${CMAKE_PROJECT_VERSION}
    DESCRIPTION "Prolate spheroidal convolution gridder"
    LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
find_package(pybind11 REQUIRED)

# Define source files
set(SOURCES
    src/prolate_spheroidal_grid.cpp
    python/bindings.cpp
)

# Create pybind11 module
pybind11_add_module(_prolate_spheroidal_grid_ext ${SOURCES})

# Set target properties
target_include_directories(_prolate_spheroidal_grid_ext PRIVATE
    include
)

# Compiler-specific options
target_compile_features(_prolate_spheroidal_grid_ext PRIVATE cxx_std_17)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(_prolate_spheroidal_grid_ext PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O0 -g>
    )
else()
    target_compile_options(_prolate_spheroidal_grid_ext PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -pipe -fPIC -DNDEBUG>
    )
endif()

# Platform-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(_prolate_spheroidal_grid_ext PRIVATE -stdlib=libc++)
    target_link_options(_prolate_spheroidal_grid_ext PRIVATE -stdlib=libc++)
endif()

# Version info
target_compile_definitions(_prolate_spheroidal_grid_ext PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
)

# Install the module in the python package directory
install(TARGETS _prolate_spheroidal_grid_ext
    DESTINATION astroviper/core/imaging/gridders/prolate_spheroidal_grid)
