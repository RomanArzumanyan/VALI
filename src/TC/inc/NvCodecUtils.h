/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2023 Roman Arzumanyan
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <chrono>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "MemoryInterfaces.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/pixfmt.h>

struct AVFrame;
}

#define X_TEXTIFY(a) TEXTIFY(a)
#define TEXTIFY(a) #a
#define MAKE_PAIR(a) std::make_pair(a, TEXTIFY(a))

// AVFormatContext timeout handler;
class TimeoutHandler {
  std::chrono::milliseconds m_timeout;
  std::chrono::time_point<std::chrono::system_clock> m_last_time;

public:
  ~TimeoutHandler() = default;

  TimeoutHandler(uint32_t timeout_ms) : m_timeout(timeout_ms) {}

  TimeoutHandler() {
    constexpr uint32_t default_timeout_ms = 3000U;
    m_timeout = std::chrono::milliseconds(default_timeout_ms);
  }

  bool IsTimeout() const {
    auto delay = std::chrono::system_clock::now() - m_last_time;
    return std::chrono::duration_cast<std::chrono::milliseconds>(delay) >
           m_timeout;
  }

  static int Check(void* self) {
    return self && static_cast<TimeoutHandler*>(self)->IsTimeout();
  }

  void Reset() { m_last_time = std::chrono::system_clock::now(); }
};

std::string AvErrorToString(int av_error_code);

AVDictionary*
GetAvOptions(const std::map<std::string, std::string>& ffmpeg_options);

AVPixelFormat toFfmpegPixelFormat(Pixel_Format fmt);

Pixel_Format fromFfmpegPixelFormat(AVPixelFormat fmt);

Pixel_Format fromCuvidSurfaceFormat(cudaVideoSurfaceFormat fmt, int bpp);

AVColorSpace toFfmpegColorSpace(ColorSpace space);

ColorSpace fromFfmpegColorSpace(AVColorSpace space);

AVColorRange toFfmpegColorRange(ColorRange range);

ColorRange fromFfmpegColorRange(AVColorRange range);

std::string GetFormatName(Pixel_Format fmt);

/* Creates memaligned AVFrame that manages it's memory.
 */
std::shared_ptr<AVFrame> makeAVFrame(int width, int height, int format);

/* Creates Buffer that manages it's memory.
 */
std::shared_ptr<Buffer> makeBuffer(int width, int height, AVPixelFormat format);

/* Creates "wrapper" AVFrame that doesn't manage it's memory but relies on
 * incoming Buffer instead. No memcpy, just convenient wrapper to pass Buffer's
 * content into FFMpeg.
 */
std::shared_ptr<AVFrame> asAVFrame(Buffer* pBuf, int width, int height,
                                   AVPixelFormat format);

/* Creates Buffer which manages it's own memory, performs memcpy between it and
 * AVFrame.
 */
std::shared_ptr<Buffer> makeBufferFromAVFrame(std::shared_ptr<AVFrame> src);

size_t getBufferSize(int width, int height, AVPixelFormat format,
                     int alignment = 1);
