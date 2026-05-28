cmake_minimum_required(VERSION 3.18)
project(mss_mnn_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MSS_MNN_BUILD_APPS "Build desktop command line tools" ON)

set(MNN_ROOT "" CACHE PATH "MNN install root containing include/ and lib/")
find_path(MNN_INCLUDE_DIR
    NAMES MNN/Interpreter.hpp Interpreter.hpp
    HINTS ${MNN_ROOT}/include $ENV{MNN_ROOT}/include
)
find_library(MNN_LIBRARY
    NAMES MNN libMNN
    HINTS ${MNN_ROOT}/lib $ENV{MNN_ROOT}/lib
)

if(NOT MNN_INCLUDE_DIR OR NOT MNN_LIBRARY)
    message(FATAL_ERROR "MNN headers/library not found. Set -DMNN_ROOT=/path/to/MNN install.")
endif()

add_library(mss_mnn_core
    src/audio.cpp
    src/mnn_mask_core.cpp
    src/roformer_separator.cpp
)
target_include_directories(mss_mnn_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${MNN_INCLUDE_DIR}
)
target_link_libraries(mss_mnn_core PUBLIC ${MNN_LIBRARY})

if(MSS_MNN_BUILD_APPS)
    add_executable(mss_mnn_mask_core apps/mac_cli/mnn_mask_core_cli.cpp)
    target_link_libraries(mss_mnn_mask_core PRIVATE mss_mnn_core)

    add_executable(mss_mnn_roformer_separate apps/mac_cli/roformer_mnn_separate.cpp)
    target_link_libraries(mss_mnn_roformer_separate PRIVATE mss_mnn_core)
endif()
