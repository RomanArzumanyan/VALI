cmake_minimum_required(VERSION 3.21)

function(get_conan_ffmpeg_libs
         CONAN_FFMPEG_LIBS 
         CONAN_FFMPEG_LIB_DIR 
         CONAN_FFMPEG_BIN_DIR)
         
	# Get path to FFmpeg shared libraries. 
	# Name taxonomy contains build type name but in upper case.
	string(TOUPPER ${CMAKE_BUILD_TYPE} SUFFIX)
	
	set(VAR_NAME "ffmpeg_BIN_DIRS_${SUFFIX}")
	set(CONAN_FFMPEG_BIN_DIR ${${VAR_NAME}})
	
	set(VAR_NAME "ffmpeg_LIB_DIRS_${SUFFIX}")
	set(CONAN_FFMPEG_LIB_DIR ${${VAR_NAME}})

	if(MSVC)
		# *.dll are put into /bin instead of /lib do *.so
		set(FFMPEG_LIBS_DIRS ${CONAN_FFMPEG_BIN_DIR})
		file(GLOB CONAN_FFMPEG_LIBS "${FFMPEG_LIBS_DIRS}/*.dll")
	else()
		set(FFMPEG_LIBS_DIRS ${CONAN_FFMPEG_LIB_DIR})
		file(GLOB CONAN_FFMPEG_LIBS "${FFMPEG_LIBS_DIRS}/*.so*")
	endif()
	
    set (CONAN_FFMPEG_LIBS    ${CONAN_FFMPEG_LIBS}    PARENT_SCOPE)
	set (CONAN_FFMPEG_BIN_DIR ${CONAN_FFMPEG_BIN_DIR} PARENT_SCOPE)
	set (CONAN_FFMPEG_LIB_DIR ${CONAN_FFMPEG_LIB_DIR} PARENT_SCOPE)

    message(STATUS "CONAN_FFMPEG_LIBS: ${CONAN_FFMPEG_LIBS}")
    message(STATUS "CONAN_FFMPEG_LIB_DIR: ${CONAN_FFMPEG_LIB_DIR}")
    message(STATUS "CONAN_FFMPEG_BIN_DIR: ${CONAN_FFMPEG_BIN_DIR}")

endfunction(get_conan_ffmpeg_libs)
