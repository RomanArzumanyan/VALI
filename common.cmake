# Brief: this function finds FFMpeg on Windows
#
# Params:
# FFMPEG_ROOT           (in)    path to directory with FFmpeg build
# FFMPEG_LIBRARIES      (out)   list of FFMpeg libraries
# FFMPEG_DLLS           (out)   list of FFMpeg DLLS. On platforms other then Windows it will be left untouched
# FFMPEG_INCLUDE_DIRS   (out)   list of paths to FFMpeg headers
function(find_FFMpeg FFMPEG_ROOT)

    # Find paths to headers
    set(FFMPEG_INC_DIR ${FFMPEG_ROOT}/include)

    find_path(AVFORMAT_INCLUDE_DIRS 
        libavformat/avformat.h 
        ${FFMPEG_INC_DIR})

    find_path(AVCODEC_INCLUDE_DIRS 
        libavcodec/avcodec.h 
        ${FFMPEG_INC_DIR})
    
    find_path(AVUTIL_INCLUDE_DIRS
        libavutil/avutil.h 
        ${FFMPEG_INC_DIR})

    find_path(SWRESAMPLE_INCLUDE_DIRS
        libswresample/swresample.h 
        ${FFMPEG_INC_DIR})

    find_path(SWSCALE_INCLUDE_DIRS
        libswscale/swscale.h 
        ${FFMPEG_INC_DIR})

    set(FFMPEG_INCLUDE_DIRS "")
    list (APPEND
        FFMPEG_INCLUDE_DIRS
        ${AVFORMAT_INCLUDE_DIRS}
        ${AVCODEC_INCLUDE_DIRS}
        ${AVUTIL_INCLUDE_DIRS}
        ${SWRESAMPLE_INCLUDE_DIRS}
        ${SWSCALE_INCLUDE_DIRS})

    # Find libraries
    set(FFMPEG_LIB_DIR ${FFMPEG_ROOT}/lib)

    find_library(AVFORMAT_LIBRARIES
        avformat 
        ${FFMPEG_LIB_DIR})

    find_library(AVCODEC_LIBRARIES
        avcodec
        ${FFMPEG_LIB_DIR})

    find_library(AVUTIL_LIBRARIES
        avutil
        ${FFMPEG_LIB_DIR})

    find_library(SWRESAMPLE_LIBRARIES
        swresample
        ${FFMPEG_LIB_DIR})

    find_library(SWSCALE_LIBRARIES
        swscale
        ${FFMPEG_LIB_DIR})        

    set(FFMPEG_LIBRARIES "")
    list (APPEND
        FFMPEG_LIBRARIES
        ${AVFORMAT_LIBRARIES}
        ${AVCODEC_LIBRARIES}
        ${AVUTIL_LIBRARIES}
        ${SWRESAMPLE_LIBRARIES}
        ${SWSCALE_LIBRARIES})

    if (WIN32)
        # Find FFMpeg DLLs
        set (FFMPEG_DLLS "")
        file(GLOB FFMPEG_DLLS "${FFMPEG_ROOT}/bin/*.dll")
    endif (WIN32)

    # Promote to parent scope
    set (FFMPEG_INCLUDE_DIRS    ${FFMPEG_INCLUDE_DIRS}  PARENT_SCOPE)
    set (FFMPEG_LIBRARIES       ${FFMPEG_LIBRARIES}     PARENT_SCOPE)
    set (FFMPEG_DLLS            ${FFMPEG_DLLS}          PARENT_SCOPE)
endfunction()
