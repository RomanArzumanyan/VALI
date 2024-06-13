cmake_minimum_required(VERSION 3.21)

project(VALI)

option(TRACK_TOKEN_ALLOCATIONS "Debug memory allocations within VALI" FALSE )

if(TRACK_TOKEN_ALLOCATIONS)
	add_definitions(-DTRACK_TOKEN_ALLOCATIONS)
endif(TRACK_TOKEN_ALLOCATIONS)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

find_package(CUDAToolkit 11.2 REQUIRED)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/inc")

option(FETCHCONTENT_QUIET OFF)
include(FetchContent)

# Fetch BtbN FFMpeg build
if(WIN32)
	set(
		FFMPEG_URL 
		"https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip")
else()
	set(
		FFMPEG_URL
		"https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-lgpl-shared.tar.xz")
endif(WIN32)

fetchcontent_declare(ffmpeg URL ${FFMPEG_URL})
if(NOT ffmpeg_POPULATED)
	FetchContent_Populate(ffmpeg)
endif()
fetchcontent_makeavailable(ffmpeg)
set(FFMPEG_ROOT "${FETCHCONTENT_BASE_DIR}/ffmpeg-src")

add_subdirectory(PyNvCodec)
add_subdirectory(TC)

