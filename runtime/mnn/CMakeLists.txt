cmake_minimum_required(VERSION 3.18)
project(mss_mnn_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(MSS_MNN_BUILD_APPS "Build desktop command line tools" ON)
option(MSS_MNN_BUILD_IOS_SMOKE "Build the minimal iOS smoke-test app" OFF)
set(MSS_MNN_IOS_CORE_MODEL "" CACHE FILEPATH "MNN core model copied into the iOS smoke app bundle")
set(MSS_MNN_IOS_SEGMENT_DIR "" CACHE PATH "RoFormer segment directory copied into the iOS smoke app bundle")
set(MSS_MNN_IOS_METADATA "" CACHE FILEPATH "RoFormer metadata json copied into the iOS smoke app bundle")
set(MSS_MNN_IOS_INPUT_WAV "" CACHE FILEPATH "Input wav copied into the iOS smoke app bundle")
set(MSS_MNN_IOS_BUNDLE_ID "org.pymss.mnn.smoke" CACHE STRING "iOS smoke app bundle identifier")
set(MSS_MNN_IOS_DEVELOPMENT_TEAM "" CACHE STRING "Apple development team id for device signing")

set(MNN_ROOT "" CACHE PATH "MNN install root containing include/ and lib/")
set(MSS_MNN_INCLUDE_HINTS)
set(MSS_MNN_LIBRARY_HINTS)
if(MNN_ROOT)
    list(APPEND MSS_MNN_INCLUDE_HINTS "${MNN_ROOT}/include")
    list(APPEND MSS_MNN_LIBRARY_HINTS "${MNN_ROOT}/lib")
endif()
if(DEFINED ENV{MNN_ROOT})
    list(APPEND MSS_MNN_INCLUDE_HINTS "$ENV{MNN_ROOT}/include")
    list(APPEND MSS_MNN_LIBRARY_HINTS "$ENV{MNN_ROOT}/lib")
endif()
set(MSS_MNN_FIND_ROOT_OPTION)
if(MSS_MNN_INCLUDE_HINTS OR MSS_MNN_LIBRARY_HINTS)
    set(MSS_MNN_FIND_ROOT_OPTION NO_CMAKE_FIND_ROOT_PATH)
endif()
find_path(MNN_INCLUDE_DIR
    NAMES MNN/Interpreter.hpp Interpreter.hpp
    HINTS ${MSS_MNN_INCLUDE_HINTS}
    ${MSS_MNN_FIND_ROOT_OPTION}
)
find_library(MNN_LIBRARY
    NAMES MNN libMNN
    HINTS ${MSS_MNN_LIBRARY_HINTS}
    ${MSS_MNN_FIND_ROOT_OPTION}
)
find_library(MNN_CL_LIBRARY
    NAMES MNN_CL libMNN_CL
    HINTS ${MSS_MNN_LIBRARY_HINTS}
    ${MSS_MNN_FIND_ROOT_OPTION}
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
if(MNN_CL_LIBRARY)
    target_link_libraries(mss_mnn_core PUBLIC ${MNN_CL_LIBRARY})
endif()
if(APPLE)
    target_link_libraries(mss_mnn_core PUBLIC objc)
endif()

if(MSS_MNN_BUILD_APPS)
    add_executable(mss_mnn_mask_core apps/mac_cli/mnn_mask_core_cli.cpp)
    target_link_libraries(mss_mnn_mask_core PRIVATE mss_mnn_core)

    add_executable(mss_mnn_roformer_separate apps/mac_cli/roformer_mnn_separate.cpp)
    target_link_libraries(mss_mnn_roformer_separate PRIVATE mss_mnn_core)
endif()

if(MSS_MNN_BUILD_IOS_SMOKE)
    if(NOT APPLE)
        message(FATAL_ERROR "MSS_MNN_BUILD_IOS_SMOKE requires an Apple toolchain")
    endif()
    foreach(resource_var MSS_MNN_IOS_METADATA MSS_MNN_IOS_INPUT_WAV)
        if(NOT EXISTS "${${resource_var}}")
            message(FATAL_ERROR "${resource_var} does not exist: ${${resource_var}}")
        endif()
    endforeach()
    if(MSS_MNN_IOS_CORE_MODEL AND MSS_MNN_IOS_SEGMENT_DIR)
        message(FATAL_ERROR "Set only one of MSS_MNN_IOS_CORE_MODEL or MSS_MNN_IOS_SEGMENT_DIR")
    endif()
    if(NOT MSS_MNN_IOS_CORE_MODEL AND NOT MSS_MNN_IOS_SEGMENT_DIR)
        message(FATAL_ERROR "Set one of MSS_MNN_IOS_CORE_MODEL or MSS_MNN_IOS_SEGMENT_DIR")
    endif()
    if(MSS_MNN_IOS_CORE_MODEL AND NOT EXISTS "${MSS_MNN_IOS_CORE_MODEL}")
        message(FATAL_ERROR "MSS_MNN_IOS_CORE_MODEL does not exist: ${MSS_MNN_IOS_CORE_MODEL}")
    endif()
    if(MSS_MNN_IOS_SEGMENT_DIR AND NOT IS_DIRECTORY "${MSS_MNN_IOS_SEGMENT_DIR}")
        message(FATAL_ERROR "MSS_MNN_IOS_SEGMENT_DIR is not a directory: ${MSS_MNN_IOS_SEGMENT_DIR}")
    endif()
    enable_language(OBJCXX)
    set(CMAKE_OBJCXX_STANDARD 17)
    set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

    set(MSS_MNN_IOS_RESOURCE_FILES
        "${MSS_MNN_IOS_METADATA}"
        "${MSS_MNN_IOS_INPUT_WAV}"
    )
    set(MSS_MNN_IOS_CORE_BUNDLE_NAME "")
    set(MSS_MNN_IOS_SEGMENT_BUNDLE_NAME "")
    if(MSS_MNN_IOS_CORE_MODEL)
        get_filename_component(MSS_MNN_IOS_CORE_BUNDLE_NAME "${MSS_MNN_IOS_CORE_MODEL}" NAME)
        list(APPEND MSS_MNN_IOS_RESOURCE_FILES "${MSS_MNN_IOS_CORE_MODEL}")
    endif()
    if(MSS_MNN_IOS_SEGMENT_DIR)
        get_filename_component(MSS_MNN_IOS_SEGMENT_BUNDLE_NAME "${MSS_MNN_IOS_SEGMENT_DIR}" NAME)
    endif()
    get_filename_component(MSS_MNN_IOS_METADATA_BUNDLE_NAME "${MSS_MNN_IOS_METADATA}" NAME)
    get_filename_component(MSS_MNN_IOS_INPUT_BUNDLE_NAME "${MSS_MNN_IOS_INPUT_WAV}" NAME)

    add_executable(mss_mnn_ios_smoke MACOSX_BUNDLE
        apps/ios_smoke/main.mm
        apps/ios_smoke/AppDelegate.mm
        ${MSS_MNN_IOS_RESOURCE_FILES}
    )
    set_source_files_properties(
        ${MSS_MNN_IOS_RESOURCE_FILES}
        PROPERTIES MACOSX_PACKAGE_LOCATION Resources
    )
    if(MSS_MNN_IOS_SEGMENT_DIR)
        add_custom_command(TARGET mss_mnn_ios_smoke POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${MSS_MNN_IOS_SEGMENT_DIR}"
                "$<TARGET_BUNDLE_DIR:mss_mnn_ios_smoke>/${MSS_MNN_IOS_SEGMENT_BUNDLE_NAME}"
        )
    endif()
    target_compile_definitions(mss_mnn_ios_smoke PRIVATE
        MSS_MNN_IOS_CORE_BUNDLE_NAME="${MSS_MNN_IOS_CORE_BUNDLE_NAME}"
        MSS_MNN_IOS_SEGMENT_BUNDLE_NAME="${MSS_MNN_IOS_SEGMENT_BUNDLE_NAME}"
        MSS_MNN_IOS_METADATA_BUNDLE_NAME="${MSS_MNN_IOS_METADATA_BUNDLE_NAME}"
        MSS_MNN_IOS_INPUT_BUNDLE_NAME="${MSS_MNN_IOS_INPUT_BUNDLE_NAME}"
    )
    target_link_libraries(mss_mnn_ios_smoke PRIVATE
        mss_mnn_core
        "-framework UIKit"
        "-framework Foundation"
        "-framework AVFoundation"
        "-framework UniformTypeIdentifiers"
        "-framework Metal"
        "-framework MetalKit"
        "-framework Accelerate"
    )
    set_target_properties(mss_mnn_ios_smoke PROPERTIES
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_SOURCE_DIR}/apps/ios_smoke/Info.plist.in"
        XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${MSS_MNN_IOS_BUNDLE_ID}"
        XCODE_ATTRIBUTE_CODE_SIGN_STYLE "Automatic"
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1"
        XCODE_ATTRIBUTE_INFOPLIST_KEY_UIApplicationSceneManifest_Generation "YES"
        XCODE_ATTRIBUTE_INFOPLIST_KEY_UIApplicationSupportsIndirectInputEvents "YES"
    )
    if(MSS_MNN_IOS_DEVELOPMENT_TEAM)
        set_target_properties(mss_mnn_ios_smoke PROPERTIES
            XCODE_ATTRIBUTE_DEVELOPMENT_TEAM "${MSS_MNN_IOS_DEVELOPMENT_TEAM}"
        )
    endif()
endif()
