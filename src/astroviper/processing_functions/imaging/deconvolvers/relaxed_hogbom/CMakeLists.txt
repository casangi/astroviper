cmake_minimum_required(VERSION 3.15...3.27)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(astroviper_relaxed_hogbom LANGUAGES CXX)
    get_filename_component(_av_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../.." ABSOLUTE)
    list(APPEND CMAKE_MODULE_PATH "${_av_repo_root}/cmake")
endif()

include(AstroviperPybind)

get_filename_component(_deconvolver_root "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
astroviper_add_pybind_module(
    NAME _relaxed_hogbom_ext
    SOURCES
        src/relaxed_hogbom.cpp
        python/bindings.cpp
        ../hogbom/src/hclean.cpp
    INCLUDE_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${_deconvolver_root}/hogbom/include"
        "${_deconvolver_root}/include"
        "${_deconvolver_root}/aspclean/include"
)
