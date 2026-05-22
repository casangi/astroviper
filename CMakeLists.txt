cmake_minimum_required(VERSION 3.15...3.27)
project(astroviper VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# Add the hogbom extension as a subdirectory
add_subdirectory(src/astroviper/processing_functions/imaging/deconvolvers/hogbom)

# Add the Asp (Adaptive Scale Pixel) extension as a subdirectory
add_subdirectory(src/astroviper/processing_functions/imaging/deconvolvers/aspclean)

# Add the prolate spheroidal gridder extension as a subdirectory
add_subdirectory(src/astroviper/processing_functions/imaging/gridders/prolate_spheroidal_grid_cpp)
