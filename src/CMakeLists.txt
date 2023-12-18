cmake_minimum_required(VERSION 3.21)

project(Video_Processing_Framework)

option(TRACK_TOKEN_ALLOCATIONS "Debug memory allocations within VPF" FALSE )

if(TRACK_TOKEN_ALLOCATIONS)
	add_definitions(-DTRACK_TOKEN_ALLOCATIONS)
endif(TRACK_TOKEN_ALLOCATIONS)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
	add_definitions(-DNOMINMAX)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

find_package(CUDAToolkit 11.2 REQUIRED)

find_package(FFMPEG REQUIRED PATHS "../_conan")

add_subdirectory(PyNvCodec)
add_subdirectory(TC)

