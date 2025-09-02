cmake_minimum_required(VERSION 3.15...3.27)
project(astroviper VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# Add the hogbom extension as a subdirectory
add_subdirectory(src/astroviper/core/imaging/deconvolvers/hogbom)
