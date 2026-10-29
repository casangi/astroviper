cmake_minimum_required(VERSION 3.15...3.27)

# Standalone build support (cmake -S . -B build inside this directory); when
# built from the repo root these are already set and include() is a no-op.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(astroviper_direct_fourier_transform LANGUAGES CXX)
    get_filename_component(_av_repo_root
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
    list(APPEND CMAKE_MODULE_PATH "${_av_repo_root}/cmake")
endif()

include(AstroviperPybind)

astroviper_add_pybind_module(
    NAME _direct_fourier_transform_ext
    SOURCES src/direct_fourier_transform.cpp
            python/bindings.cpp
)
