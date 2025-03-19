/*
 * Copyright 2019 NVIDIA Corporation
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
#include "MemoryInterfaces.hpp"
#include <map>
#include <stdint.h>
#include <string>

/**
 * Dictionary of dictionaries used to store metadata.
 * First level key is metadata source: format context, stream, etc.
 * Second level is an ordinary dictionary with tag names and values.
 * Example:
 * {
 *    "context": {
 *        "major_brand": "isom",
 *        "creation_time": "2024-12-31T21:00:00.000000Z",
 *    },
 *    "video_stream" : {
 *        "handler_name": "Core Media Video",
 *        "vendor_id" : "[0][0][0][0]"
 *    }
 * }
 **/
using dict = std::map<std::string, std::string>;
using metadata_dict = std::map<std::string, dict>;

struct PacketData {
  int32_t key;
  int64_t pts;
  int64_t dts;
  uint64_t pos;
  uint64_t bsl;
  uint64_t duration;
};

struct VideoContext {
  int64_t width = 0;
  int64_t height = 0;
  int64_t profile = 0;
  int64_t level = 0;
  int64_t delay = 0;
  int64_t gop_size = 0;
  int64_t num_frames = 0;
  int64_t is_vfr = 0;
  int64_t num_streams = 0;
  int64_t stream_index = 0;
  int64_t host_frame_size = 0;
  int64_t bit_rate = 0;

  double frame_rate = .0;
  double avg_frame_rate = .0;
  double time_base = .0;
  double start_time = .0;
  double duration = .0;

  Pixel_Format format = UNDEFINED;

  ColorSpace color_space = UNSPEC;
  ColorRange color_range = UDEF;

  metadata_dict metadata = {};
};

struct AudioContext {
  // Reserved for future use;
};

struct MuxingParams {
  VideoContext videoContext;
  AudioContext audioContext;
};

struct SeekContext {
  /* Will be set to false for default ctor, true otherwise;
   */
  bool use_seek;

  /* Frame number we want to get. Set by user.
   */
  int64_t seek_frame;

  /* Timestamp (s) we want to get. Set by user.
   */
  double seek_tssec;

  SeekContext() : use_seek(false), seek_frame(-1), seek_tssec(-1.0) {}

  SeekContext(double frame_ts)
      : use_seek(true), seek_tssec(frame_ts), seek_frame(-1) {}

  SeekContext(int64_t frame_num)
      : use_seek(true), seek_frame(frame_num), seek_tssec(-1.0) {}

  SeekContext(const SeekContext& other)
      : use_seek(other.use_seek), seek_frame(other.seek_frame),
        seek_tssec(other.seek_tssec) {}

  SeekContext& operator=(const SeekContext& other) {
    use_seek = other.use_seek;
    seek_frame = other.seek_frame;
    seek_tssec = other.seek_tssec;

    return *this;
  }

  bool IsByNumber() const { return 0 <= seek_frame; }
  bool IsByTimestamp() const { return 0.0 <= seek_tssec; }
};