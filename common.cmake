# Brief: this function finds FFMpeg on Windows
#
# Params:
# FFMPEG_ROOT           (in)    path to directory with FFmpeg build
# FFMPEG_LIBRARIES      (out)   list of FFMpeg libraries
# FFMPEG_INCLUDE_DIRS   (out)   list of paths to FFMpeg headers
function(findFFMpeg 
    FFMPEG_ROOT
    FFMPEG_LIBRARIES
    FFMPEG_INCLUDE_DIRS)

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

    set(FFMPEG_INCLUDE_DIRS "")
    list (APPEND
        FFMPEG_INCLUDE_DIRS
        ${AVFORMAT_INCLUDE_DIRS}
        ${AVCODEC_INCLUDE_DIRS}
        ${AVUTIL_INCLUDE_DIRS}
        ${SWRESAMPLE_INCLUDE_DIRS})

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

    set(FFMPEG_LIBRARIES "")
    list (APPEND
        FFMPEG_LIBRARIES
        ${AVFORMAT_LIBRARIES}
        ${AVCODEC_LIBRARIES}
        ${AVUTIL_LIBRARIES}
        ${SWRESAMPLE_LIBRARIES})

    # Promote to parent scope
    set (FFMPEG_INCLUDE_DIRS    ${FFMPEG_INCLUDE_DIRS}  PARENT_SCOPE)
    set (FFMPEG_LIBRARIES       ${FFMPEG_LIBRARIES}     PARENT_SCOPE)
endfunction()
