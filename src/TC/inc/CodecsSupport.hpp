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
#include <stdint.h>

struct PacketData {
  int32_t key;
  int64_t pts;
  int64_t dts;
  uint64_t pos;
  uint64_t bsl;
  uint64_t duration;
};

struct VideoContext {
  uint32_t width;
  uint32_t height;
  uint32_t gop_size;
  uint32_t num_frames;
  uint32_t is_vfr;
  double frameRate;
  double avgFrameRate;
  double timeBase;
  uint32_t streamIndex;
  uint32_t host_frame_size;
  Pixel_Format format;
  ColorSpace color_space;
  ColorRange color_range;
};

struct AudioContext {
  // Reserved for future use;
};

struct MuxingParams {
  VideoContext videoContext;
  AudioContext audioContext;
};

enum SeekMode {
  /* Seek for exact frame number.
   * Suited for standalone demuxer seek. */
  EXACT_FRAME = 0,
  /* Seek for previous key frame in past.
   * Suitable for seek & decode.  */
  PREV_KEY_FRAME = 1,

  SEEK_MODE_NUM_ELEMS
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

  /* Mode in which we seek. */
  SeekMode mode;

  /* PTS of frame found after seek. */
  int64_t out_frame_pts;

  /* Duration of frame found after seek. */
  int64_t out_frame_duration;

  /* Number of frames that were decoded during seek. */
  int64_t num_frames_decoded;

  SeekContext()
      : use_seek(false), seek_frame(-1), seek_tssec(-1.0), mode(PREV_KEY_FRAME),
        out_frame_pts(-1), out_frame_duration(-1), num_frames_decoded(-1) {}

  SeekContext(int64_t frame_id)
      : use_seek(true), seek_frame(frame_id), seek_tssec(-1.0),
        mode(PREV_KEY_FRAME), out_frame_pts(-1), out_frame_duration(-1),
        num_frames_decoded(-1) {}

  SeekContext(double frame_ts)
      : use_seek(true), seek_tssec(frame_ts), seek_frame(-1),
        mode(PREV_KEY_FRAME), out_frame_pts(-1), out_frame_duration(-1),
        num_frames_decoded(-1) {}

  SeekContext(int64_t frame_num, SeekMode seek_mode)
      : use_seek(true), seek_frame(frame_num), seek_tssec(-1.0),
        mode(seek_mode), out_frame_pts(-1), out_frame_duration(-1),
        num_frames_decoded(-1) {}

  SeekContext(double frame_ts, SeekMode seek_mode)
      : use_seek(true), seek_tssec(frame_ts), mode(seek_mode), seek_frame(-1),
        out_frame_pts(-1), out_frame_duration(-1), num_frames_decoded(-1) {}

  SeekContext(const SeekContext& other)
      : use_seek(other.use_seek), seek_frame(other.seek_frame),
        seek_tssec(other.seek_tssec), mode(other.mode),
        out_frame_pts(other.out_frame_pts),
        out_frame_duration(other.out_frame_duration),
        num_frames_decoded(other.num_frames_decoded) {}

  SeekContext& operator=(const SeekContext& other) {
    use_seek = other.use_seek;
    seek_frame = other.seek_frame;
    seek_tssec = other.seek_tssec;
    mode = other.mode;
    out_frame_pts = other.out_frame_pts;
    out_frame_duration = other.out_frame_duration;
    num_frames_decoded = other.num_frames_decoded;
    return *this;
  }

  bool IsByNumber() const { return 0 <= seek_frame; }
  bool IsByTimestamp() const { return 0.0 <= seek_tssec; }
};
