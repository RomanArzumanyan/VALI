/*
 * Copyright 2024 Vision Labs LLC
 *
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

#include "LibraryLoader.hpp"

#include <nvjpeg.h>

class LibNvJpeg {
private:
  static const char* const filename;
  static std::shared_ptr<LibraryLoader> Load();

public:
  // nvjpeg.h:
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegHandle_t*>
      nvjpegCreateSimple;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegHandle_t,
                          nvjpegEncoderState_t*, cudaStream_t>
      nvjpegEncoderStateCreate;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegHandle_t,
                          nvjpegEncoderParams_t*, cudaStream_t>
      nvjpegEncoderParamsCreate;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t,
                          nvjpegEncoderParams_t,
                          const nvjpegChromaSubsampling_t, cudaStream_t>
      nvjpegEncoderParamsSetSamplingFactors;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t,
                          nvjpegEncoderParams_t, const int, cudaStream_t>
      nvjpegEncoderParamsSetQuality;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegHandle_t,
                          nvjpegEncoderState_t, const nvjpegEncoderParams_t,
                          const nvjpegImage_t*, nvjpegInputFormat_t, int, int,
                          cudaStream_t>
      nvjpegEncodeImage;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegHandle_t,
                          nvjpegEncoderState_t, const nvjpegEncoderParams_t,
                          const nvjpegImage_t*, nvjpegChromaSubsampling_t, int,
                          int, cudaStream_t>
      nvjpegEncodeYUV;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegHandle_t,
                          nvjpegEncoderState_t, unsigned char*, size_t*,
                          cudaStream_t>
      nvjpegEncodeRetrieveBitstream;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t,
                          nvjpegEncoderParams_t>
      nvjpegEncoderParamsDestroy;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegEncoderState_t>
      nvjpegEncoderStateDestroy;
  static LoadableFunction<LibNvJpeg::Load, nvjpegStatus_t, nvjpegHandle_t>
      nvjpegDestroy;
};
