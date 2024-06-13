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

project(PyNvCodec VERSION 1.0)

set(PYNVCODEC_VERSION_MAJOR 1)
set(PYNVCODEC_VERSION_MINOR 0)

if(POLICY CMP0135)
  #  https://cmake.org/cmake/help/latest/policy/CMP0135.html
  # From docs:
  # CMake 3.24 and above prefers to set the timestamps of all extracted contents to the time of the extraction.
  # This ensures that anything that depends on the extracted contents will be rebuilt whenever the URL changes.
  cmake_policy(SET CMP0135 NEW)
endif()

configure_file("inc/Version.hpp.in" "pynvcode_version.h")

find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter Development)

option(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# Fetch pybind11
fetchcontent_declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/v2.10.0.tar.gz
    URL_HASH
        SHA256=eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec
)
fetchcontent_makeavailable(pybind11)

pybind11_add_module(_PyNvCodec MODULE 
	src/PyDecoder.cpp
	src/PyFFMpegDemuxer.cpp
	src/PyFrameUploader.cpp
	src/PyNvCodec.cpp
	src/PyNvEncoder.cpp
	src/PySurface.cpp
	src/PySurfaceConverter.cpp
	src/PySurfaceDownloader.cpp
	src/PySurfaceResizer.cpp
	src/PyFrameConverter.cpp
)
set_property(TARGET _PyNvCodec PROPERTY CXX_STANDARD 17)
target_include_directories(_PyNvCodec PRIVATE inc)

if (NOT DEFINED FFMPEG_ROOT)
    message (FATAL "Define FFMPEG_ROOT")
endif()

include("${PROJECT_ROOT_DIR}/common.cmake")
find_FFMpeg(${FFMPEG_ROOT})

target_link_libraries(
	_PyNvCodec PUBLIC 
	TC_CORE 
	TC
	CUDA::cuda_driver
	CUDA::cudart
	CUDA::npps
	CUDA::nppig
	CUDA::nppial
	CUDA::nppicc
	CUDA::nppidei
	${FFMPEG_LIBRARIES})

include_directories(
	${FFMPEG_INCLUDE_DIRS}
	"${PROJECT_ROOT_DIR}/extern/dlpack/include/dlpack"
)

include(GNUInstallDirs)

set_target_properties(_PyNvCodec PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(_PyNvCodec PROPERTIES INSTALL_RPATH "$ORIGIN")

if (WIN32)
	install(TARGETS _PyNvCodec 
		RUNTIME_DEPENDENCIES DIRECTORIES "${FFMPEG_ROOT}/bin/"
		PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "python" "nvcuda"
		POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
		RUNTIME DESTINATION PyNvCodec 
		LIBRARY DESTINATION PyNvCodec)
else ()
	install(TARGETS _PyNvCodec 
		RUNTIME_DEPENDENCIES 
		DIRECTORIES "${FFMPEG_ROOT}/lib/"
		PRE_INCLUDE_REGEXES "${FFMPEG_LIBRARIES}"
		POST_INCLUDE_REGEXES "${FFMPEG_LIBRARIES}"
		RUNTIME DESTINATION PyNvCodec 
		LIBRARY DESTINATION PyNvCodec)
endif (WIN32)


