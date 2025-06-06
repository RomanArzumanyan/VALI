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

project(python_vali VERSION 1.0)

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

find_package(Python3 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

option(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# Fetch pybind11
fetchcontent_declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
)
fetchcontent_makeavailable(pybind11)

pybind11_add_module(_python_vali MODULE 
	src/PyDecoder.cpp
	src/PyFrameUploader.cpp
	src/VALI.cpp
	src/PyNvEncoder.cpp
	src/PySurface.cpp
	src/PySurfaceConverter.cpp
	src/PySurfaceDownloader.cpp
	src/PySurfaceResizer.cpp
	src/PyFrameConverter.cpp
	src/PyNvJpegEncoder.cpp
	src/BufferedReader.cpp
	src/PySurfaceRotator.cpp
	src/PySurfaceUD.cpp
)
set_property(TARGET _python_vali PROPERTY CXX_STANDARD 17)
target_include_directories(_python_vali PRIVATE inc)

if (NOT DEFINED FFMPEG_ROOT)
    message (FATAL "Define FFMPEG_ROOT")
endif()

include("${PROJECT_ROOT_DIR}/common.cmake")
find_FFMpeg(${FFMPEG_ROOT})

target_link_libraries(
	_python_vali PRIVATE 
	TC_CORE 
	TC
	${FFMPEG_LIBRARIES}
)

include_directories(
	${FFMPEG_INCLUDE_DIRS}
	${CUDAToolkit_INCLUDE_DIRS}
	"${PROJECT_ROOT_DIR}/extern/dlpack/include/dlpack"
)

include(GNUInstallDirs)

set_target_properties(_python_vali PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
set_target_properties(_python_vali PROPERTIES INSTALL_RPATH "$ORIGIN")

if ("$ENV{PYPI_BUILD}" STREQUAL "")
	set(PYPI_BUILD "0" CACHE INTERNAL "Copy FFMpeg libraries to wheel directory")
else()
	set(PYPI_BUILD "1" CACHE INTERNAL "Rely on wheel repair")
endif()

# Only copy ffmpeg libs for local builds. PyPi builds rely on wheel repair.
if ("${PYPI_BUILD}" STREQUAL "0")
	message(STATUS "Copy FFMpeg libs")
	if (WIN32)
		install(FILES ${FFMPEG_DLLS} DESTINATION python_vali)
	else ()
		install(FILES ${FFMPEG_LIBS_ALL} DESTINATION python_vali)
	endif (WIN32)
else()
	message(STATUS "Skip FFMpeg libs copy")
endif ()

install(TARGETS _python_vali RUNTIME DESTINATION python_vali LIBRARY DESTINATION python_vali)