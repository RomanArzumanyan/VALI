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

#include "Tasks.hpp"
#include <iostream>
#include <sstream>

using namespace VPF;

namespace VPF {
/**
 * If return status is not NVJPEG_STATUS_SUCCESS throws std::rutime_error with
 * description message.
 */
static void ThrowOnNvJpegError(nvjpegStatus_t result, const char* err_msg,
                               int line) {
  if (NVJPEG_STATUS_SUCCESS != result) {
    std::stringstream ss;
    ss << __FILE__ << ":" << line << " " << err_msg
       << ", error code: " << result << "\n";
    throw std::runtime_error(ss.str());
  }
}

/**
 * If return status is not NVJPEG_STATUS_SUCCESS outputs description message
 * to stderr.
 */
static void ScoldOnNvJpegError(nvjpegStatus_t result, const char* err_msg,
                               int line) {
  if (NVJPEG_STATUS_SUCCESS != result) {
    std::stringstream ss;
    ss << __FILE__ << ":" << line << " " << err_msg
       << ", error code: " << result << "\n";
    std::cerr << ss.str();
  }
}

struct NvJpegEncodeContext_Impl {
  unsigned m_compression = 100U;
  bool m_initialized = false;
  nvjpegEncoderState_t m_state{};
  nvjpegEncoderParams_t m_params{};
  Pixel_Format m_pix_fmt = Pixel_Format::RGB;
  nvjpegChromaSubsampling_t m_subsampling = NVJPEG_CSS_444;
  nvjpegInputFormat_t m_input_format = NVJPEG_INPUT_RGB;

  NvJpegEncodeContext_Impl(std::shared_ptr<NvJpegEncodeFrame> encoder)
      : m_enc(encoder) {}

private:
  /* Needs to be here to prevent lifetime scope issues of underlying nvJpeg
   * API structures.
   */
  std::shared_ptr<NvJpegEncodeFrame> m_enc;
};

unsigned NvJpegEncodeContext::Compression() const {
  return pImpl->m_compression;
}

nvjpegEncoderState_t NvJpegEncodeContext::State() const {
  return pImpl->m_state;
}

nvjpegEncoderParams_t NvJpegEncodeContext::Params() const {
  return pImpl->m_params;
}

Pixel_Format NvJpegEncodeContext::PixelFormat() const {
  return pImpl->m_pix_fmt;
}

nvjpegChromaSubsampling_t NvJpegEncodeContext::Subsampling() const {
  return pImpl->m_subsampling;
}

nvjpegInputFormat_t NvJpegEncodeContext::Format() const {
  return pImpl->m_input_format;
}

NvJpegEncodeContext::NvJpegEncodeContext(
    std::shared_ptr<NvJpegEncodeFrame> encoder, unsigned compression,
    Pixel_Format format) {

  pImpl = new NvJpegEncodeContext_Impl(encoder);
  pImpl->m_compression = compression;
  pImpl->m_pix_fmt = format;

  switch (format) {
  case Pixel_Format::RGB:
    // m_subsampling is not used for encoding RGB input.
    pImpl->m_input_format = NVJPEG_INPUT_RGBI;
    break;
  case Pixel_Format::BGR:
    pImpl->m_input_format = NVJPEG_INPUT_BGRI;
    break;
  case Pixel_Format::RGB_PLANAR:
    pImpl->m_input_format = NVJPEG_INPUT_RGB;
    break;
  case Pixel_Format::YUV444:
    pImpl->m_subsampling = NVJPEG_CSS_444;
    // m_input_format is not used for encoding YUV input.
    break;
  case Pixel_Format::YUV422:
    pImpl->m_subsampling = NVJPEG_CSS_422;
    break;
  case Pixel_Format::YUV420:
    pImpl->m_subsampling = NVJPEG_CSS_420;
    break;
  default:
    throw std::invalid_argument("unsupported pixel format");
  }

  pImpl->m_state = nullptr;
  pImpl->m_params = nullptr;

  ThrowOnNvJpegError(LibNvJpeg::nvjpegEncoderStateCreate(encoder->GetHandle(),
                                                         &pImpl->m_state,
                                                         encoder->GetStream()),
                     "nvjpegEncoderStateCreate error", __LINE__);

  ThrowOnNvJpegError(LibNvJpeg::nvjpegEncoderParamsCreate(encoder->GetHandle(),
                                                          &pImpl->m_params,
                                                          encoder->GetStream()),
                     "nvjpegEncoderParamsCreate error", __LINE__);

  ThrowOnNvJpegError(
      LibNvJpeg::nvjpegEncoderParamsSetSamplingFactors(
          pImpl->m_params, pImpl->m_subsampling, encoder->GetStream()),
      "nvjpegEncoderParamsSetSamplingFactors error", __LINE__);

  ThrowOnNvJpegError(
      LibNvJpeg::nvjpegEncoderParamsSetQuality(
          pImpl->m_params, pImpl->m_compression, encoder->GetStream()),
      "nvjpegEncoderParamsSetQuality error", __LINE__);
}

NvJpegEncodeContext::~NvJpegEncodeContext() {
  if (pImpl->m_params) {
    ScoldOnNvJpegError(LibNvJpeg::nvjpegEncoderParamsDestroy(pImpl->m_params),
                       "nvjpegEncoderParamsDestroy error", __LINE__);
    pImpl->m_params = nullptr;
  }

  if (pImpl->m_state) {
    ScoldOnNvJpegError(LibNvJpeg::nvjpegEncoderStateDestroy(pImpl->m_state),
                       "nvjpegEncoderStateDestroy error", __LINE__);
    pImpl->m_state = nullptr;
  }

  delete pImpl;
  pImpl = nullptr;
}

struct NvJpegEncodeFrame_Impl {
  CUstream m_stream;
  bool m_initialized = false;
  nvjpegHandle_t m_jpegHandle{};
  NvJpegEncodeContext* m_jpegContext{};
};

void NvJpegEncodeFrame::SetEncoderContext(NvJpegEncodeContext* context) {
  pImpl->m_jpegContext = context;
}

nvjpegHandle_t NvJpegEncodeFrame::GetHandle() { return pImpl->m_jpegHandle; }

CUstream NvJpegEncodeFrame::GetStream() const { return pImpl->m_stream; }

/* No sync function is registered in constructor.
 * It is done to avoid unnecessary CUDA stream sync when encoding multiple
 * images.
 *
 * CUDA stream sync is done at higher level, inside PyNvJpegEncoder.
 */
NvJpegEncodeFrame::NvJpegEncodeFrame(CUstream stream)
    : Task("NvJpegEncodeFrame", NvJpegEncodeFrame::numInputs,
           NvJpegEncodeFrame::numOutputs, nullptr, static_cast<void*>(stream)) {
  pImpl = new NvJpegEncodeFrame_Impl();

  pImpl->m_stream = stream;
  pImpl->m_jpegHandle = nullptr;

  CudaStrSync sync(pImpl->m_stream);

  ThrowOnNvJpegError(LibNvJpeg::nvjpegCreateSimple(&pImpl->m_jpegHandle),
                     "nvjpegCreateSimple error", __LINE__);
}

NvJpegEncodeFrame::~NvJpegEncodeFrame() {
  if (pImpl->m_jpegHandle) {
    ScoldOnNvJpegError(LibNvJpeg::nvjpegDestroy(pImpl->m_jpegHandle),
                       "nvjpegDestroy error", __LINE__);
    pImpl->m_jpegHandle = nullptr;
  }

  delete pImpl;
  pImpl = nullptr;
}

TaskExecDetails NvJpegEncodeFrame::Run() {
  ClearOutputs();

  if (!pImpl->m_jpegContext) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT,
                           "the encoder context is not set");
  }

  auto pInputSurface = dynamic_cast<Surface*>(GetInput());
  if (!pInputSurface) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT,
                           "input surface is null");
  }

  // Check input pixel format support.
  if (pInputSurface->PixelFormat() != pImpl->m_jpegContext->PixelFormat()) {
    return TaskExecDetails(
        TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::INVALID_INPUT,
        "input surface pixel format does not correspond to the context");
  }

  // Setup pitches and channels.
  nvjpegImage_t nv_image = {};
  for (auto i = 0; i < pInputSurface->NumComponents(); i++) {
    nv_image.pitch[i] = pInputSurface->Pitch(i);
    nv_image.channel[i] =
        reinterpret_cast<unsigned char*>(pInputSurface->PixelPtr());
  }

  const auto f = pInputSurface->PixelFormat();
  const auto is_yuv = f == Pixel_Format::YUV444 || f == Pixel_Format::YUV422 ||
                      f == Pixel_Format::YUV420;

  // Compress image.
  auto result = NVJPEG_STATUS_SUCCESS;
  if (is_yuv) {
    result = LibNvJpeg::nvjpegEncodeYUV(
        pImpl->m_jpegHandle, pImpl->m_jpegContext->State(),
        pImpl->m_jpegContext->Params(), &nv_image,
        pImpl->m_jpegContext->Subsampling(), pInputSurface->Width(),
        pInputSurface->Height(), pImpl->m_stream);
  } else {
    result = LibNvJpeg::nvjpegEncodeImage(
        pImpl->m_jpegHandle, pImpl->m_jpegContext->State(),
        pImpl->m_jpegContext->Params(), &nv_image,
        pImpl->m_jpegContext->Format(), pInputSurface->Width(),
        pInputSurface->Height(), pImpl->m_stream);
  }

  if (NVJPEG_STATUS_SUCCESS != result) {
    ScoldOnNvJpegError(result, "nvjpegEncodeImage error", __LINE__);
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT,
                           "nvjpegEncodeImage error");
  }

  // Get compressed stream size.
  size_t outputLength = 0;
  result = LibNvJpeg::nvjpegEncodeRetrieveBitstream(
      pImpl->m_jpegHandle, pImpl->m_jpegContext->State(), nullptr,
      &outputLength, pImpl->m_stream);

  if (NVJPEG_STATUS_SUCCESS != result) {
    ScoldOnNvJpegError(result, "nvjpegEncodeRetrieveBitstream error", __LINE__);
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT,
                           "nvjpegEncodeRetrieveBitstream error");
  }

  // Allocate output Buffer.
  Buffer* pOutputBuffer = Buffer::MakeOwnMem(outputLength);
  SetOutput(pOutputBuffer, 0);

  // Retrieve compressed stream.
  result = LibNvJpeg::nvjpegEncodeRetrieveBitstream(
      pImpl->m_jpegHandle, pImpl->m_jpegContext->State(),
      static_cast<unsigned char*>(pOutputBuffer->GetRawMemPtr()), &outputLength,
      pImpl->m_stream);

  if (NVJPEG_STATUS_SUCCESS != result) {
    ScoldOnNvJpegError(result, "nvjpegEncodeRetrieveBitstream error", __LINE__);
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT,
                           "nvjpegEncodeRetrieveBitstream error");
  }

  return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                         TaskExecInfo::SUCCESS);
}

} // namespace VPF
