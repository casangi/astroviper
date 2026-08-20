cmake_minimum_required(VERSION 3.15...3.27)
project(astroviper VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# All shared C++ / pybind11 build settings (C++ standard, reproducibility flags,
# threading, install rules) live in cmake/AstroviperPybind.cmake. Edit THAT file
# to change a compiler flag for every extension at once.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AstroviperPybind)

# Register each C++ extension. To add a new one: create its module directory
# (include/ src/ python/) with a CMakeLists.txt that calls
# astroviper_add_pybind_module(), then add one add_subdirectory() line here.
set(_av_imaging src/astroviper/processing_functions/imaging)
add_subdirectory(${_av_imaging}/deconvolvers/hogbom)
add_subdirectory(${_av_imaging}/deconvolvers/aspclean)
add_subdirectory(${_av_imaging}/gridders/prolate_spheroidal_grid_cpp)
add_subdirectory(${_av_imaging}/imaging_weighting/grid_imaging_weights_cpp)
set(_av_simulation src/astroviper/processing_functions/simulation)
add_subdirectory(${_av_simulation}/visibility_kernel_cpp)
