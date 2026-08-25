cmake_minimum_required(VERSION 3.15...3.27)

# Standalone build support: when this directory is configured directly
# (cmake -S . -B build, to iterate on just this kernel) rather than via the
# repo-root add_subdirectory(), bootstrap a project and put the shared cmake/
# directory on the module path. When built from the repo root these are already
# set and the include_guard in AstroviperPybind.cmake makes include() a no-op.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(astroviper_grid_imaging_weights LANGUAGES CXX)
    get_filename_component(_av_repo_root
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
    list(APPEND CMAKE_MODULE_PATH "${_av_repo_root}/cmake")
endif()

include(AstroviperPybind)

astroviper_add_pybind_module(
    NAME _grid_imaging_weights_ext
    SOURCES src/grid_imaging_weights.cpp
            python/bindings.cpp
)
