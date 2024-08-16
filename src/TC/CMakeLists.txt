#
# Copyright 2019 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

cmake_minimum_required(VERSION 3.21)

add_subdirectory(TC_CORE)

find_package(CUDAToolkit 11.2 REQUIRED)
include(GenerateExportHeader)

include("${PROJECT_ROOT_DIR}/common.cmake")
if (NOT DEFINED FFMPEG_ROOT)
    message (FATAL "Define FFMPEG_ROOT")
endif()
find_FFMpeg(${FFMPEG_ROOT})

add_library(
    TC
    src/MemoryInterfaces.cpp
    src/TaskConvertSurface.cpp
    src/TaskNvencEncodeFrame.cpp
    src/TaskCudaUploadFrame.cpp
    src/TaskCudaDownloadSurface.cpp
    src/TaskResizeSurface.cpp
    src/TaskDecodeFrame.cpp
    src/TaskConvertFrame.cpp
    src/NppCommon.cpp
    src/NvCodecCliOptions.cpp
    src/Utils.cpp
    src/SurfacePlane.cpp
    src/CudaUtils.cpp
    src/Surfaces.cpp
    src/LibCuda.cpp
    src/LibNpp.cpp
    src/LibraryLoader.cpp
)

if (WIN32)
    target_sources(TC PRIVATE
        src/tc_dlopen_windows.cpp
        )
else()
    target_sources(TC PRIVATE
        src/tc_dlopen_unix.cpp
        )
endif()

set(TC_VERSION_MAJOR 1)
set(TC_VERSION_MINOR 0)

configure_file(inc/Version.hpp.in tc_version.h)

option(USE_NVTX "Use NVTX for profiling" FALSE)
if(USE_NVTX)
    add_definitions(-DUSE_NVTX)
    ## Replace when requiring CMAKE 3.25:
    # target_link_libraries(TC PUBLIC CUDA::nvtx3)
    target_link_libraries(TC PUBLIC ${CMAKE_DL_LIBRARIES})
endif()

generate_export_header(TC)

target_link_libraries(
    TC
    PUBLIC
    ${FFMPEG_LIBRARIES}
    TC_CORE    
)

target_compile_features(TC PRIVATE cxx_std_17)
set_property(
    TARGET TC
    PROPERTY
        # required for shared Python modules in case library is build statically on Unix
        POSITION_INDEPENDENT_CODE
        ON
)

set(TC_VIDEO_CODEC_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
    CACHE PATH "Path to Video Codec SDK interface headers"
)
target_include_directories(
    TC
    PUBLIC
        inc
        "${PROJECT_ROOT_DIR}/extern/dlpack/include/dlpack"
        ${TC_VIDEO_CODEC_INTERFACE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)
if(NOT EXISTS ${TC_VIDEO_CODEC_INTERFACE_DIR}/nvEncodeAPI.h)
    message(
        FATAL_ERROR
        "Could not find nvEncodeAPI.h! "
        "Please set TC_VIDEO_CODEC_INTERFACE_DIR=\"${TC_VIDEO_CODEC_INTERFACE_DIR}\" to a location that contains nvEncodeAPI.h!"
    )
endif()

if(UNIX)
    target_link_libraries(TC PUBLIC pthread)
endif(UNIX)

include_directories(${FFMPEG_INCLUDE_DIRS})
set_target_properties(TC PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(TC PROPERTIES INSTALL_RPATH "$ORIGIN")
