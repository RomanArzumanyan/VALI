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

#include "CodecsSupport.hpp"
#include "NppCommon.hpp"
#include "Surfaces.hpp"
#include "Tasks.hpp"
#include "Utils.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>

using namespace VPF;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF {

static const TaskExecDetails s_invalid_src_dst(TaskExecStatus::TASK_EXEC_FAIL,
                                               TaskExecInfo::INVALID_INPUT,
                                               "invalid src / dst");

static const TaskExecDetails s_success(TaskExecStatus::TASK_EXEC_SUCCESS,
                                       TaskExecInfo::SUCCESS);

static const TaskExecDetails s_fail(TaskExecStatus::TASK_EXEC_FAIL,
                                    TaskExecInfo::FAIL);

static const TaskExecDetails
    s_unsupp_cc_ctx(TaskExecStatus::TASK_EXEC_FAIL,
                    TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS,
                    "unsupported cc_ctx params");

struct NppConvertSurface_Impl {
  NppConvertSurface_Impl() = delete;
  NppConvertSurface_Impl(const NppConvertSurface_Impl& other) = delete;
  NppConvertSurface_Impl(const NppConvertSurface_Impl&& other) = delete;
  NppConvertSurface_Impl&
  operator=(const NppConvertSurface_Impl& other) = delete;
  NppConvertSurface_Impl&
  operator=(const NppConvertSurface_Impl&& other) = delete;

  NppConvertSurface_Impl(int gpu_id, CUstream str, Pixel_Format SRC_FMT,
                         Pixel_Format DST_FMT)
      : m_gpu_id(gpu_id), m_stream(str), srcFmt(SRC_FMT), dstFmt(DST_FMT) {
    SetupNppContext(gpu_id, m_stream, nppCtx);
  }

  virtual ~NppConvertSurface_Impl() = default;

  virtual TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                                  ColorspaceConversionContext* pCtx) = 0;

  bool Validate(Surface* pSrc, Surface* pDst) {
    if (!pSrc || !pDst) {
      return false;
    }

    if ((pSrc->PixelFormat() != srcFmt) || (pDst->PixelFormat() != dstFmt)) {
      return false;
    }

    if ((pSrc->Width() != pDst->Width()) ||
        (pSrc->Height() != pDst->Height())) {
      return false;
    }

    return true;
  }

  std::string GetNvtxTickName() const {
    std::stringstream ss("GPU_");
    ss << GetFormatName(srcFmt);
    ss << "_2_";
    ss << GetFormatName(dstFmt);

    return ss.str();
  }

  int m_gpu_id;
  CUstream m_stream;
  NppStreamContext nppCtx;
  const Pixel_Format srcFmt, dstFmt;
};

struct nv12_bgr final : public NppConvertSurface_Impl {
  nv12_bgr(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, NV12, BGR) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    auto const color_space = pCtx ? pCtx->color_space : BT_709;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U)};

    auto pDst = (Npp8u*)pOutput->PixelPtr(0U);
    NppiSize oSizeRoi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      if (JPEG == color_range) {
        err = LibNpp::nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        err = LibNpp::nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      }
      break;
    case BT_601:
      if (JPEG == color_range) {
        err = LibNpp::nppiNV12ToBGR_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        err = NPP_ERROR;
      }
      break;
    default:
      err = NPP_ERROR;
      break;

      return NPP_NO_ERROR != err ? s_success : s_fail;
    }
  }
};

struct nv12_rgb final : public NppConvertSurface_Impl {
  nv12_rgb(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, NV12, RGB) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    auto const color_space = pCtx ? pCtx->color_space : BT_709;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    const Npp8u* const pSrc[] = {(const Npp8u* const)pInput->PixelPtr(0U),
                                 (const Npp8u* const)pInput->PixelPtr(1U)};

    auto pDst = (Npp8u*)pOutput->PixelPtr();
    NppiSize oSizeRoi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      if (JPEG == color_range) {
        err = LibNpp::nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        err = LibNpp::nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      }
      break;
    case BT_601:
      if (JPEG == color_range) {
        err = LibNpp::nppiNV12ToRGB_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        return s_unsupp_cc_ctx;
      }
      break;
    default:
      return s_unsupp_cc_ctx;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct nv12_yuv420 final : public NppConvertSurface_Impl {
  nv12_yuv420(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, NV12, YUV420) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U)};

    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};

    int dstStep[] = {(int)pOutput->Pitch(0U), (int)pOutput->Pitch(1U),
                     (int)pOutput->Pitch(2U)};
    NppiSize roi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    auto const color_range = pCtx ? pCtx->color_range : JPEG;
    switch (color_range) {
    case JPEG:
      err = LibNpp::nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc, pInput->Pitch(0U), pDst,
                                                  dstStep, roi, nppCtx);
      break;
    case MPEG:
      err = LibNpp::nppiYCbCr420_8u_P2P3R_Ctx(pSrc[0], pInput->Pitch(0U),
                                              pSrc[1], pInput->Pitch(1U), pDst,
                                              dstStep, roi, nppCtx);
      break;
    default:
      return s_unsupp_cc_ctx;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct nv12_y final : public NppConvertSurface_Impl {
  nv12_y(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, NV12, Y) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    try {
      CUDA_MEMCPY2D m = {0};
      m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      m.srcDevice = pInput->PixelPtr();
      m.dstDevice = pOutput->PixelPtr();
      m.srcPitch = pInput->Pitch();
      m.dstPitch = pOutput->Pitch();
      m.Height = pInput->Height();
      m.WidthInBytes = pInput->WidthInBytes();

      CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
      ThrowOnCudaError(LibCuda::cuMemcpy2DAsync(&m, m_stream), __LINE__);
      ThrowOnCudaError(LibCuda::cuStreamSynchronize(m_stream), __LINE__);
    } catch (...) {
      return s_fail;
    }

    return s_success;
  }
};

struct rbg8_y final : public NppConvertSurface_Impl {
  rbg8_y(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB, Y) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};

    auto ret = LibNpp::nppiRGBToGray_8u_C3C1R_Ctx(
        (const Npp8u*)pInput->PixelPtr(), pInput->Pitch(),
        (Npp8u*)pOutput->PixelPtr(), pOutput->Pitch(), roi, nppCtx);

    if (NPP_NO_ERROR != ret) {
      return s_fail;
    }

    return s_success;
  }
};

struct yuv420_rgb final : public NppConvertSurface_Impl {
  yuv420_rgb(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, YUV420, RGB) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    auto pInput = (SurfaceYUV420*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep[] = {(int)pInput->Pitch(0U), (int)pInput->Pitch(1U),
                     (int)pInput->Pitch(2U)};
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      return s_unsupp_cc_ctx;
    case BT_601:
      if (JPEG == color_range) {
        err = LibNpp::nppiYUV420ToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                   roi, nppCtx);
      } else {
        err = LibNpp::nppiYCbCr420ToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst,
                                                     dstStep, roi, nppCtx);
      }
      break;
    default:
      return s_unsupp_cc_ctx;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct yuv420_bgr final : public NppConvertSurface_Impl {
  yuv420_bgr(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, YUV420, BGR) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    auto pInput = (SurfaceYUV420*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep[] = {(int)pInput->Pitch(0U), (int)pInput->Pitch(1U),
                     (int)pInput->Pitch(2U)};
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      return s_unsupp_cc_ctx;
    case BT_601:
      if (JPEG == color_range) {
        err = LibNpp::nppiYUV420ToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                   roi, nppCtx);
      } else {
        err = LibNpp::nppiYCbCr420ToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst,
                                                     dstStep, roi, nppCtx);
      }
      break;
    default:
      return s_unsupp_cc_ctx;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct yuv444_bgr final : public NppConvertSurface_Impl {
  yuv444_bgr(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, YUV444, BGR) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    if (BT_601 != color_space) {
      return s_unsupp_cc_ctx;
    }

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case MPEG:
      err = LibNpp::nppiYCbCrToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                roi, nppCtx);
      break;
    case JPEG:
      err = LibNpp::nppiYUVToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct yuv444_rgb final : public NppConvertSurface_Impl {
  yuv444_rgb(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, YUV444, RGB) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    if (BT_601 != color_space) {
      return s_unsupp_cc_ctx;
    }

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case JPEG:
      err = LibNpp::nppiYUVToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct yuv444_rgb_planar final : public NppConvertSurface_Impl {
  yuv444_rgb_planar(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, YUV444, RGB_PLANAR) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    if (BT_601 != color_space) {
      return s_unsupp_cc_ctx;
    }

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0), (Npp8u*)pOutput->PixelPtr(1),
                     (Npp8u*)pOutput->PixelPtr(2)};
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case JPEG:
      err = LibNpp::nppiYUVToRGB_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                            nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct bgr_yuv444 final : public NppConvertSurface_Impl {
  bgr_yuv444(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, BGR, YUV444) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    if (BT_601 != color_space) {
      return s_unsupp_cc_ctx;
    }

    auto pInput = (SurfaceBGR*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      s_invalid_src_dst;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case MPEG:
      err = LibNpp::nppiBGRToYCbCr_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                roi, nppCtx);
      break;
    case JPEG:
      err = LibNpp::nppiBGRToYUV_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rgb_yuv444 final : public NppConvertSurface_Impl {
  rgb_yuv444(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB, YUV444) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    if (BT_601 != color_space) {
      return s_unsupp_cc_ctx;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int srcStep = pInput->Pitch();
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};
    int dstStep = pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;
    switch (color_range) {
    case JPEG:
      err = LibNpp::nppiRGBToYUV_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              nppCtx);
      break;
    case MPEG:
      err = LibNpp::nppiRGBToYCbCr_8u_C3R_Ctx(pSrc, srcStep, pDst[0], dstStep,
                                              roi, nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rgb_planar_yuv444 final : public NppConvertSurface_Impl {
  rgb_planar_yuv444(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB_PLANAR, YUV444) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGBPlanar*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    if (BT_601 != color_space) {
      return s_unsupp_cc_ctx;
    }

    const Npp8u* pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                           (const Npp8u*)pInput->PixelPtr(1U),
                           (const Npp8u*)pInput->PixelPtr(2U)};
    int srcStep = pInput->Pitch();
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};
    int dstStep = pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;
    switch (color_range) {
    case JPEG:
      err = LibNpp::nppiRGBToYUV_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                            nppCtx);
      break;
    case MPEG:
      err = LibNpp::nppiRGBToYCbCr_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct y_yuv444 final : public NppConvertSurface_Impl {
  y_yuv444(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, Y, YUV444) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceY*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    // Make gray U and V channels;
    for (int i = 1; i < pOutput->NumPlanes(); i++) {
      const Npp8u nValue = 128U;
      Npp8u* pDst = (Npp8u*)pOutput->PixelPtr(i);
      int nDstStep = pOutput->Pitch(i);
      NppiSize roi = {(int)pOutput->Width(i), (int)pOutput->Height(i)};
      auto err =
          LibNpp::nppiSet_8u_C1R_Ctx(nValue, pDst, nDstStep, roi, nppCtx);
      if (NPP_NO_ERROR != err) {
        return s_fail;
      }
    }

    // Copy Y channel;
    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int nSrcStep = pInput->Pitch();
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr(0U);
    int nDstStep = pOutput->Pitch(0U);
    NppiSize roi = {(int)pInput->Width(), (int)pInput->Height()};
    auto err = LibNpp::nppiCopy_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, roi,
                                           nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rgb_yuv420 final : public NppConvertSurface_Impl {
  rgb_yuv420(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB, YUV420) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    auto const color_space = pCtx ? pCtx->color_space : BT_601;
    auto const color_range = pCtx ? pCtx->color_range : JPEG;

    if (BT_601 != color_space) {
      return s_unsupp_cc_ctx;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int srcStep = pInput->Pitch();
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};
    int dstStep[] = {(int)pOutput->Pitch(0U), (int)pOutput->Pitch(1U),
                     (int)pOutput->Pitch(2U)};
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = NPP_NO_ERROR;
    switch (color_range) {
    case JPEG:
      err = LibNpp::nppiRGBToYUV420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                 roi, nppCtx);
      break;

    case MPEG:
      err = LibNpp::nppiRGBToYCbCr420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                   roi, nppCtx);
      break;

    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct yuv420_nv12 final : public NppConvertSurface_Impl {
  yuv420_nv12(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, YUV420, NV12) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};

    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U)};

    int srcStep[] = {(int)pInput->Pitch(0U), (int)pInput->Pitch(1U),
                     (int)pInput->Pitch(2U)};
    int dstStep[] = {(int)pOutput->Pitch(0U), (int)pOutput->Pitch(1U)};
    NppiSize roi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = LibNpp::nppiYCbCr420_8u_P3P2R_Ctx(
        pSrc, srcStep, pDst[0], dstStep[0], pDst[1], dstStep[1], roi, nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rgb8_deinterleave final : public NppConvertSurface_Impl {
  rgb8_deinterleave(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB, RGB_PLANAR) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int nSrcStep = pInput->Pitch();
    Npp8u* aDst[] = {
        (Npp8u*)pOutput->PixelPtr(),
        (Npp8u*)pOutput->PixelPtr() + pOutput->Height() * pOutput->Pitch(),
        (Npp8u*)pOutput->PixelPtr() + pOutput->Height() * pOutput->Pitch() * 2};
    int nDstStep = pOutput->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pOutput->Height();
    oSizeRoi.width = pOutput->Width();

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = LibNpp::nppiCopy_8u_C3P3R_Ctx(pSrc, nSrcStep, aDst, nDstStep,
                                             oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rgb8_interleave final : public NppConvertSurface_Impl {
  rgb8_interleave(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB_PLANAR, RGB) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGBPlanar*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* const pSrc[] = {
        (Npp8u*)pInput->PixelPtr(),
        (Npp8u*)pInput->PixelPtr() + pInput->Height() * pInput->Pitch(),
        (Npp8u*)pInput->PixelPtr() + pInput->Height() * pInput->Pitch() * 2};
    int nSrcStep = pInput->Pitch();
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int nDstStep = pOutput->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pOutput->Height();
    oSizeRoi.width = pOutput->Width();

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = LibNpp::nppiCopy_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                             oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rgb_bgr final : public NppConvertSurface_Impl {
  rgb_bgr(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB, BGR) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int nSrcStep = pInput->Pitch();
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int nDstStep = pOutput->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pOutput->Height();
    oSizeRoi.width = pOutput->Width();
    // rgb to brg
    const int aDstOrder[3] = {2, 1, 0};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = LibNpp::nppiSwapChannels_8u_C3R_Ctx(
        pSrc, nSrcStep, pDst, nDstStep, oSizeRoi, aDstOrder, nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct bgr_rgb final : public NppConvertSurface_Impl {
  bgr_rgb(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, BGR, RGB) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceBGR*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int nSrcStep = pInput->Pitch();
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int nDstStep = pOutput->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pOutput->Height();
    oSizeRoi.width = pOutput->Width();
    // brg to rgb
    const int aDstOrder[3] = {2, 1, 0};
    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = LibNpp::nppiSwapChannels_8u_C3R_Ctx(
        pSrc, nSrcStep, pDst, nDstStep, oSizeRoi, aDstOrder, nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rbg8_rgb32f final : public NppConvertSurface_Impl {
  rbg8_rgb32f(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB, RGB_32F) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();

    int nSrcStep = pInput->Pitch();
    Npp32f* pDst = (Npp32f*)pOutput->PixelPtr();
    int nDstStep = pOutput->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pOutput->Height();
    oSizeRoi.width = pOutput->Width();
    Npp32f nMin = 0.0;
    Npp32f nMax = 1.0;
    const int aDstOrder[3] = {2, 1, 0};

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));

    auto err = LibNpp::nppiScale_8u32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                               oSizeRoi, nMin, nMax, nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct rgb32f_deinterleave final : public NppConvertSurface_Impl {
  rgb32f_deinterleave(int gpu_id, CUstream stream)
      : NppConvertSurface_Impl(gpu_id, stream, RGB_32F, RGB_32F_PLANAR) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      return s_invalid_src_dst;
    }

    const Npp32f* pSrc = (const Npp32f*)pInput->PixelPtr();
    int nSrcStep = pInput->Pitch();
    Npp32f* aDst[] = {(Npp32f*)((uint8_t*)pOutput->PixelPtr()),
                      (Npp32f*)((uint8_t*)pOutput->PixelPtr() +
                                pOutput->Height() * pOutput->Pitch()),
                      (Npp32f*)((uint8_t*)pOutput->PixelPtr() +
                                pOutput->Height() * pOutput->Pitch() * 2)};
    int nDstStep = pOutput->Pitch();
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = pOutput->Height();
    oSizeRoi.width = pOutput->Width();

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));
    auto err = LibNpp::nppiCopy_32f_C3P3R_Ctx(pSrc, nSrcStep, aDst, nDstStep,
                                              oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }

    return s_success;
  }
};

struct p16_nv12 final : public NppConvertSurface_Impl {
  /* Source format is needed because P10 and P12 to NV12 conversions share
   * the same implementation. The only difference is bit depth value which
   * is calculated at a runtime.
   */
  p16_nv12(int gpu_id, CUstream stream, Pixel_Format src_fmt)
      : NppConvertSurface_Impl(gpu_id, stream, src_fmt, NV12) {}

  TaskExecDetails Execute(Token* pSrcToken, Token* pDstToken,
                          ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());

    /* No Validate() call here because same code serves 2 different input
     * pixel formats.
     */
    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;
    if (!pInput || !pOutput) {
      return s_invalid_src_dst;
    }

    auto input_plane = pInput->GetSurfacePlane();
    if (!pScratch || pScratch->Width() != input_plane.Width() ||
        pScratch->Height() != input_plane.Height()) {
      pScratch = std::make_shared<SurfacePlane>(
          input_plane.Width(), input_plane.Height(), sizeof(uint16_t), kDLUInt,
          "u", GetContextByStream(m_stream));
    }

    CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));

    auto src_plane = pInput->GetSurfacePlane();
    auto dst_plane = pOutput->GetSurfacePlane();

    try {
      // Take 8 most significant bits, save result in 16 bit scratch buffer;
      NppiSize oSizeRoi = {0};
      oSizeRoi.height = pScratch->Height();
      oSizeRoi.width = pScratch->Width();
      ThrowOnNppError(
          LibNpp::nppiDivC_16u_C1RSfs_Ctx(
              (Npp16u*)src_plane.GpuMem(), src_plane.Pitch(),
              1 << (src_plane.ElemSize() - dst_plane.ElemSize()) * 8,
              (Npp16u*)pScratch->GpuMem(), pScratch->Pitch(), oSizeRoi, 0,
              nppCtx),
          __LINE__);

      // Bit depth conversion from 16-bit scratch to output;
      oSizeRoi.height = dst_plane.Height();
      oSizeRoi.width = dst_plane.Width();
      ThrowOnNppError(LibNpp::nppiConvert_16u8u_C1R_Ctx(
                          (Npp16u*)pScratch->GpuMem(), pScratch->Pitch(),
                          (Npp8u*)dst_plane.GpuMem(), dst_plane.Pitch(),
                          oSizeRoi, nppCtx),
                      __LINE__);
    } catch (...) {
      return s_fail;
    }

    return s_success;
  }

  std::shared_ptr<SurfacePlane> pScratch = nullptr;
};

} // namespace VPF

auto const cuda_stream_sync = [](void* stream) {
  LibCuda::cuStreamSynchronize((CUstream)stream);
};

std::list<std::pair<Pixel_Format, Pixel_Format>> const&
ConvertSurface::GetSupportedConversions() {
  static const std::list<std::pair<Pixel_Format, Pixel_Format>> convs(
      {{NV12, YUV420},
       {YUV420, NV12},
       {P10, NV12},
       {P12, NV12},
       {NV12, RGB},
       {NV12, BGR},
       {RGB, RGB_PLANAR},
       {RGB_PLANAR, RGB},
       {RGB_PLANAR, YUV444},
       {Y, YUV444},
       {YUV420, RGB},
       {RGB, YUV420},
       {RGB, YUV444},
       {RGB, BGR},
       {BGR, RGB},
       {YUV420, BGR},
       {YUV444, BGR},
       {YUV444, RGB},
       {BGR, YUV444},
       {NV12, Y},
       {RGB, RGB_32F},
       {RGB, Y},
       {RGB_32F, RGB_32F_PLANAR}});

  return convs;
}

ConvertSurface::ConvertSurface(Pixel_Format src, Pixel_Format dst, int gpu_id,
                               CUstream str)
    : Task("NppConvertSurface", ConvertSurface::numInputs,
           ConvertSurface::numOutputs, nullptr, (void*)str) {
  if (NV12 == src && YUV420 == dst) {
    pImpl = new nv12_yuv420(gpu_id, str);
  } else if (YUV420 == src && NV12 == dst) {
    pImpl = new yuv420_nv12(gpu_id, str);
  } else if (P10 == src && NV12 == dst) {
    pImpl = new p16_nv12(gpu_id, str, P10);
  } else if (P12 == src && NV12 == dst) {
    pImpl = new p16_nv12(gpu_id, str, P12);
  } else if (NV12 == src && RGB == dst) {
    pImpl = new nv12_rgb(gpu_id, str);
  } else if (NV12 == src && BGR == dst) {
    pImpl = new nv12_bgr(gpu_id, str);
  } else if (RGB == src && RGB_PLANAR == dst) {
    pImpl = new rgb8_deinterleave(gpu_id, str);
  } else if (RGB_PLANAR == src && RGB == dst) {
    pImpl = new rgb8_interleave(gpu_id, str);
  } else if (RGB_PLANAR == src && YUV444 == dst) {
    pImpl = new rgb_planar_yuv444(gpu_id, str);
  } else if (Y == src && YUV444 == dst) {
    pImpl = new y_yuv444(gpu_id, str);
  } else if (YUV420 == src && RGB == dst) {
    pImpl = new yuv420_rgb(gpu_id, str);
  } else if (RGB == src && YUV420 == dst) {
    pImpl = new rgb_yuv420(gpu_id, str);
  } else if (RGB == src && YUV444 == dst) {
    pImpl = new rgb_yuv444(gpu_id, str);
  } else if (RGB == src && BGR == dst) {
    pImpl = new rgb_bgr(gpu_id, str);
  } else if (BGR == src && RGB == dst) {
    pImpl = new bgr_rgb(gpu_id, str);
  } else if (YUV420 == src && BGR == dst) {
    pImpl = new yuv420_bgr(gpu_id, str);
  } else if (YUV444 == src && BGR == dst) {
    pImpl = new yuv444_bgr(gpu_id, str);
  } else if (YUV444 == src && RGB == dst) {
    pImpl = new yuv444_rgb(gpu_id, str);
  } else if (BGR == src && YUV444 == dst) {
    pImpl = new bgr_yuv444(gpu_id, str);
  } else if (NV12 == src && Y == dst) {
    pImpl = new nv12_y(gpu_id, str);
  } else if (RGB == src && RGB_32F == dst) {
    pImpl = new rbg8_rgb32f(gpu_id, str);
  } else if (RGB == src && Y == dst) {
    pImpl = new rbg8_y(gpu_id, str);
  } else if (RGB_32F == src && RGB_32F_PLANAR == dst) {
    pImpl = new rgb32f_deinterleave(gpu_id, str);
  } else {
    std::stringstream ss;
    ss << "Unsupported pixel format conversion: " << GetFormatName(src);
    ss << " -> " << GetFormatName(dst) << std::endl;
    throw std::invalid_argument(ss.str());
  }
}

ConvertSurface::~ConvertSurface() { delete pImpl; }

TaskExecDetails ConvertSurface::Run() {
  ClearOutputs();

  ColorspaceConversionContext* pCtx = nullptr;
  auto ctx_buf = (Buffer*)GetInput(2U);
  if (ctx_buf) {
    pCtx = ctx_buf->GetDataAs<ColorspaceConversionContext>();
  }

  return pImpl->Execute(GetInput(0), GetInput(1), pCtx);
}
