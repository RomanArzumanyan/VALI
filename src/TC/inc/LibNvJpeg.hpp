/*
 * Copyright 2024 Vision Labs LLC
 */
#pragma once

#include "LibraryLoader.hpp"

#include <nvjpeg.h>

class LibNvJpeg {
private:
  static const char* const filename;
  static std::shared_ptr<LibraryLoader> Load();

public:
  // cuda.h:
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
