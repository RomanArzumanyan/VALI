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
#include "NvCodecUtils.h"
#include "Surfaces.hpp"
#include "Tasks.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>

using namespace VPF;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF {

struct NppConvertSurface_Impl {
  NppConvertSurface_Impl() = delete;
  NppConvertSurface_Impl(const NppConvertSurface_Impl& other) = delete;
  NppConvertSurface_Impl(const NppConvertSurface_Impl&& other) = delete;
  NppConvertSurface_Impl&
  operator=(const NppConvertSurface_Impl& other) = delete;
  NppConvertSurface_Impl&
  operator=(const NppConvertSurface_Impl&& other) = delete;

  NppConvertSurface_Impl(CUcontext ctx, CUstream str, Pixel_Format SRC_FMT,
                         Pixel_Format DST_FMT)
      : cu_ctx(ctx), cu_str(str), srcFmt(SRC_FMT), dstFmt(DST_FMT) {
    SetupNppContext(cu_ctx, cu_str, nppCtx);
    pDetails.reset(Buffer::Make(sizeof(details), (void*)&details));
  }

  virtual ~NppConvertSurface_Impl() = default;

  virtual TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
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

  static std::tuple<ColorSpace, ColorRange>
  GetParams(ColorspaceConversionContext* pCtx) {
    auto ret = std::make_tuple(BT_601, MPEG);

    if (pCtx) {
      std::get<0>(ret) = pCtx->color_space;
      std::get<1>(ret) = pCtx->color_range;
    }

    return ret;
  }

  CUcontext cu_ctx;
  CUstream cu_str;
  NppStreamContext nppCtx;
  const Pixel_Format srcFmt, dstFmt;
  std::shared_ptr<Buffer> pDetails = nullptr;
  TaskExecDetails details;
};

struct nv12_bgr final : public NppConvertSurface_Impl {
  nv12_bgr(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, BGR) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U)};

    auto pDst = (Npp8u*)pOutput->PixelPtr(0U);
    NppiSize oSizeRoi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      if (JPEG == color_range) {
        err = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        err = nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      }
      break;
    case BT_601:
      if (JPEG == color_range) {
        err = nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, pInput->Pitch(), pDst,
                                         pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return TASK_EXEC_FAIL;
      }
      break;
    default:
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct nv12_rgb final : public NppConvertSurface_Impl {
  nv12_rgb(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, RGB) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    const Npp8u* const pSrc[] = {(const Npp8u* const)pInput->PixelPtr(0U),
                                 (const Npp8u* const)pInput->PixelPtr(1U)};

    auto pDst = (Npp8u*)pOutput->PixelPtr();
    NppiSize oSizeRoi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      if (JPEG == color_range) {
        err = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        err = nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
            pSrc, pInput->Pitch(), pDst, pOutput->Pitch(), oSizeRoi, nppCtx);
      }
      break;
    case BT_601:
      if (JPEG == color_range) {
        err = nppiNV12ToRGB_8u_P2C3R_Ctx(pSrc, pInput->Pitch(), pDst,
                                         pOutput->Pitch(), oSizeRoi, nppCtx);
      } else {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return TASK_EXEC_FAIL;
      }
      break;
    default:
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct nv12_yuv420 final : public NppConvertSurface_Impl {
  nv12_yuv420(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, YUV420) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U)};

    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};

    int dstStep[] = {(int)pOutput->Pitch(0U), (int)pOutput->Pitch(1U),
                     (int)pOutput->Pitch(2U)};
    NppiSize roi = {(int)pInput->Width(), (int)pInput->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    auto const color_range = pCtx ? pCtx->color_range : MPEG;
    switch (color_range) {
    case JPEG:
      err = nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc, pInput->Pitch(0U), pDst,
                                          dstStep, roi, nppCtx);
      break;
    case MPEG:
      err = nppiYCbCr420_8u_P2P3R_Ctx(pSrc[0], pInput->Pitch(0U), pSrc[1],
                                      pInput->Pitch(1U), pDst, dstStep, roi,
                                      nppCtx);
      break;
    default:
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct nv12_y final : public NppConvertSurface_Impl {
  nv12_y(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, Y) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = pInput->PixelPtr();
    m.dstDevice = pOutput->PixelPtr();
    m.srcPitch = pInput->Pitch();
    m.dstPitch = pOutput->Pitch();
    m.Height = pInput->Height();
    m.WidthInBytes = pInput->WidthInBytes();

    CudaCtxPush ctxPush(cu_ctx);
    cuMemcpy2DAsync(&m, cu_str);
    cuStreamSynchronize(cu_str);

    return TASK_EXEC_SUCCESS;
  }
};

struct rbg8_y final : public NppConvertSurface_Impl {
  rbg8_y(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, Y) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};

    auto ret = nppiRGBToGray_8u_C3C1R_Ctx(
        (const Npp8u*)pInput->PixelPtr(), pInput->Pitch(),
        (Npp8u*)pOutput->PixelPtr(), pOutput->Pitch(), roi, nppCtx);

    return TASK_EXEC_SUCCESS;
  }
};

struct yuv420_rgb final : public NppConvertSurface_Impl {
  yuv420_rgb(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV420, RGB) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    auto pInput = (SurfaceYUV420*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep[] = {(int)pInput->Pitch(0U), (int)pInput->Pitch(1U),
                     (int)pInput->Pitch(2U)};
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    case BT_601:
      if (JPEG == color_range) {
        err = nppiYUV420ToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                           nppCtx);
      } else {
        err = nppiYCbCr420ToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                             nppCtx);
      }
      break;
    default:
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct yuv420_bgr final : public NppConvertSurface_Impl {
  yuv420_bgr(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV420, BGR) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    auto pInput = (SurfaceYUV420*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep[] = {(int)pInput->Pitch(0U), (int)pInput->Pitch(1U),
                     (int)pInput->Pitch(2U)};
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_space) {
    case BT_709:
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    case BT_601:
      if (JPEG == color_range) {
        err = nppiYUV420ToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                           nppCtx);
      } else {
        err = nppiYCbCr420ToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                             nppCtx);
      }
      break;
    default:
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct yuv444_bgr final : public NppConvertSurface_Impl {
  yuv444_bgr(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV444, BGR) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    if (BT_601 != color_space) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case MPEG:
      err = nppiYCbCrToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                        nppCtx);
      break;
    case JPEG:
      err =
          nppiYUVToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct yuv444_rgb final : public NppConvertSurface_Impl {
  yuv444_rgb(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV444, RGB) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    if (BT_601 != color_space) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr();
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case JPEG:
      err =
          nppiYUVToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct yuv444_rgb_planar final : public NppConvertSurface_Impl {
  yuv444_rgb_planar(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV444, RGB_PLANAR) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    if (BT_601 != color_space) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* const pSrc[] = {(const Npp8u*)pInput->PixelPtr(0U),
                                 (const Npp8u*)pInput->PixelPtr(1U),
                                 (const Npp8u*)pInput->PixelPtr(2U)};
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0), (Npp8u*)pOutput->PixelPtr(1),
                     (Npp8u*)pOutput->PixelPtr(2)};
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case JPEG:
      err = nppiYUVToRGB_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct bgr_yuv444 final : public NppConvertSurface_Impl {
  bgr_yuv444(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, BGR, YUV444) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    if (BT_601 != color_space) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    auto pInput = (SurfaceBGR*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};
    int srcStep = (int)pInput->Pitch();
    int dstStep = (int)pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};
    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;

    switch (color_range) {
    case MPEG:
      err = nppiBGRToYCbCr_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                        nppCtx);
      break;
    case JPEG:
      err =
          nppiBGRToYUV_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rgb_yuv444 final : public NppConvertSurface_Impl {
  rgb_yuv444(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, YUV444) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    if (BT_601 != color_space) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int srcStep = pInput->Pitch();
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};
    int dstStep = pOutput->Pitch();
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;
    switch (color_range) {
    case JPEG:
      err =
          nppiRGBToYUV_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
      break;
    case MPEG:
      err = nppiRGBToYCbCr_8u_C3R_Ctx(pSrc, srcStep, pDst[0], dstStep, roi,
                                      nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rgb_planar_yuv444 final : public NppConvertSurface_Impl {
  rgb_planar_yuv444(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB_PLANAR, YUV444) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGBPlanar*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    if (BT_601 != color_space) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
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

    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;
    switch (color_range) {
    case JPEG:
      err = nppiRGBToYUV_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
      break;
    case MPEG:
      err =
          nppiRGBToYCbCr_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi, nppCtx);
      break;
    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct y_yuv444 final : public NppConvertSurface_Impl {
  y_yuv444(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, Y, YUV444) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceY*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    // Make gray U and V channels;
    for (int i = 1; i < pOutput->NumPlanes(); i++) {
      const Npp8u nValue = 128U;
      Npp8u* pDst = (Npp8u*)pOutput->PixelPtr(i);
      int nDstStep = pOutput->Pitch(i);
      NppiSize roi = {(int)pOutput->Width(i), (int)pOutput->Height(i)};
      auto err = nppiSet_8u_C1R_Ctx(nValue, pDst, nDstStep, roi, nppCtx);
      if (NPP_NO_ERROR != err) {
        details.info = TaskExecInfo::FAIL;
        return TASK_EXEC_FAIL;
      }
    }

    // Copy Y channel;
    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int nSrcStep = pInput->Pitch();
    Npp8u* pDst = (Npp8u*)pOutput->PixelPtr(0U);
    int nDstStep = pOutput->Pitch(0U);
    NppiSize roi = {(int)pInput->Width(), (int)pInput->Height()};
    auto err = nppiCopy_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, roi, nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rgb_yuv420 final : public NppConvertSurface_Impl {
  rgb_yuv420(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, YUV420) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    auto const params = GetParams(pCtx);
    auto const color_space = std::get<0>(params);
    auto const color_range = std::get<1>(params);

    if (BT_601 != color_space) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TASK_EXEC_FAIL;
    }

    const Npp8u* pSrc = (const Npp8u*)pInput->PixelPtr();
    int srcStep = pInput->Pitch();
    Npp8u* pDst[] = {(Npp8u*)pOutput->PixelPtr(0U),
                     (Npp8u*)pOutput->PixelPtr(1U),
                     (Npp8u*)pOutput->PixelPtr(2U)};
    int dstStep[] = {(int)pOutput->Pitch(0U), (int)pOutput->Pitch(1U),
                     (int)pOutput->Pitch(2U)};
    NppiSize roi = {(int)pOutput->Width(), (int)pOutput->Height()};

    CudaCtxPush ctxPush(cu_ctx);
    auto err = NPP_NO_ERROR;
    switch (color_range) {
    case JPEG:
      err = nppiRGBToYUV420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                         nppCtx);
      break;

    case MPEG:
      err = nppiRGBToYCbCr420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                           nppCtx);
      break;

    default:
      err = NPP_NO_OPERATION_WARNING;
      break;
    }

    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct yuv420_nv12 final : public NppConvertSurface_Impl {
  yuv420_nv12(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV420, NV12) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
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

    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiYCbCr420_8u_P3P2R_Ctx(pSrc, srcStep, pDst[0], dstStep[0],
                                         pDst[1], dstStep[1], roi, nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rgb8_deinterleave final : public NppConvertSurface_Impl {
  rgb8_deinterleave(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, RGB_PLANAR) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
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

    CudaCtxPush ctxPush(cu_ctx);
    auto err =
        nppiCopy_8u_C3P3R_Ctx(pSrc, nSrcStep, aDst, nDstStep, oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rgb8_interleave final : public NppConvertSurface_Impl {
  rgb8_interleave(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB_PLANAR, RGB) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGBPlanar*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
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

    CudaCtxPush ctxPush(cu_ctx);
    auto err =
        nppiCopy_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeRoi, nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rgb_bgr final : public NppConvertSurface_Impl {
  rgb_bgr(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, BGR) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
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
    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiSwapChannels_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                           oSizeRoi, aDstOrder, nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct bgr_rgb final : public NppConvertSurface_Impl {
  bgr_rgb(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, BGR, RGB) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceBGR*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
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
    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiSwapChannels_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                           oSizeRoi, aDstOrder, nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rbg8_rgb32f final : public NppConvertSurface_Impl {
  rbg8_rgb32f(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, RGB_32F) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
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

    CudaCtxPush ctxPush(cu_ctx);

    auto err = nppiScale_8u32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeRoi,
                                       nMin, nMax, nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct rgb32f_deinterleave final : public NppConvertSurface_Impl {
  rgb32f_deinterleave(CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB_32F, RGB_32F_PLANAR) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
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

    CudaCtxPush ctxPush(cu_ctx);
    auto err = nppiCopy_32f_C3P3R_Ctx(pSrc, nSrcStep, aDst, nDstStep, oSizeRoi,
                                      nppCtx);
    if (NPP_NO_ERROR != err) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }
};

struct p16_nv12 final : public NppConvertSurface_Impl {
  /* Source format is needed because P10 and P12 to NV12 conversions share
   * the same implementation. The only difference is bit depth value which
   * is calculated at a runtime.
   */
  p16_nv12(CUcontext context, CUstream stream, Pixel_Format src_fmt)
      : NppConvertSurface_Impl(context, stream, src_fmt, NV12) {}

  TaskExecStatus Execute(Token* pSrcToken, Token* pDstToken,
                         ColorspaceConversionContext* pCtx) override {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    /* No Validate() call here because same code serves 2 different input
     * pixel formats.
     */
    auto pInput = (Surface*)pSrcToken;
    auto pOutput = (Surface*)pDstToken;
    if (!pInput || !pOutput) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return TASK_EXEC_FAIL;
    }

    auto input_plane = pInput->GetSurfacePlane();
    if (!pScratch || pScratch->Width() != input_plane.Width() ||
        pScratch->Height() != input_plane.Height()) {
      pScratch = std::make_shared<SurfacePlane>(
          input_plane.Width(), input_plane.Height(), sizeof(uint16_t), kDLUInt,
          cu_ctx);
    }

    CudaCtxPush ctxPush(cu_ctx);

    auto src_plane = pInput->GetSurfacePlane();
    auto dst_plane = pOutput->GetSurfacePlane();

    try {
      // Take 8 most significant bits, save result in 16 bit scratch buffer;
      NppiSize oSizeRoi = {0};
      oSizeRoi.height = pScratch->Height();
      oSizeRoi.width = pScratch->Width();
      ThrowOnNppError(
          nppiDivC_16u_C1RSfs_Ctx(
              (Npp16u*)src_plane.GpuMem(), src_plane.Pitch(),
              1 << (src_plane.ElemSize() - dst_plane.ElemSize()) * 8,
              (Npp16u*)pScratch->GpuMem(), pScratch->Pitch(), oSizeRoi, 0,
              nppCtx),
          __LINE__);

      // Bit depth conversion from 16-bit scratch to output;
      oSizeRoi.height = dst_plane.Height();
      oSizeRoi.width = dst_plane.Width();
      ThrowOnNppError(nppiConvert_16u8u_C1R_Ctx(
                          (Npp16u*)pScratch->GpuMem(), pScratch->Pitch(),
                          (Npp8u*)dst_plane.GpuMem(), dst_plane.Pitch(),
                          oSizeRoi, nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return TASK_EXEC_FAIL;
    }

    return TASK_EXEC_SUCCESS;
  }

  std::shared_ptr<SurfacePlane> pScratch = nullptr;
};

} // namespace VPF

auto const cuda_stream_sync = [](void* stream) {
  cuStreamSynchronize((CUstream)stream);
};

ConvertSurface::ConvertSurface(Pixel_Format src, Pixel_Format dst,
                               CUcontext ctx, CUstream str)
    : Task("NppConvertSurface", ConvertSurface::numInputs,
           ConvertSurface::numOutputs, nullptr, nullptr) {
  if (NV12 == src && YUV420 == dst) {
    pImpl = new nv12_yuv420(ctx, str);
  } else if (YUV420 == src && NV12 == dst) {
    pImpl = new yuv420_nv12(ctx, str);
  } else if (P10 == src && NV12 == dst) {
    pImpl = new p16_nv12(ctx, str, P10);
  } else if (P12 == src && NV12 == dst) {
    pImpl = new p16_nv12(ctx, str, P12);
  } else if (NV12 == src && RGB == dst) {
    pImpl = new nv12_rgb(ctx, str);
  } else if (NV12 == src && BGR == dst) {
    pImpl = new nv12_bgr(ctx, str);
  } else if (RGB == src && RGB_PLANAR == dst) {
    pImpl = new rgb8_deinterleave(ctx, str);
  } else if (RGB_PLANAR == src && RGB == dst) {
    pImpl = new rgb8_interleave(ctx, str);
  } else if (RGB_PLANAR == src && YUV444 == dst) {
    pImpl = new rgb_planar_yuv444(ctx, str);
  } else if (Y == src && YUV444 == dst) {
    pImpl = new y_yuv444(ctx, str);
  } else if (YUV420 == src && RGB == dst) {
    pImpl = new yuv420_rgb(ctx, str);
  } else if (RGB == src && YUV420 == dst) {
    pImpl = new rgb_yuv420(ctx, str);
  } else if (RGB == src && YUV444 == dst) {
    pImpl = new rgb_yuv444(ctx, str);
  } else if (RGB == src && BGR == dst) {
    pImpl = new rgb_bgr(ctx, str);
  } else if (BGR == src && RGB == dst) {
    pImpl = new bgr_rgb(ctx, str);
  } else if (YUV420 == src && BGR == dst) {
    pImpl = new yuv420_bgr(ctx, str);
  } else if (YUV444 == src && BGR == dst) {
    pImpl = new yuv444_bgr(ctx, str);
  } else if (YUV444 == src && RGB == dst) {
    pImpl = new yuv444_rgb(ctx, str);
  } else if (BGR == src && YUV444 == dst) {
    pImpl = new bgr_yuv444(ctx, str);
  } else if (NV12 == src && Y == dst) {
    pImpl = new nv12_y(ctx, str);
  } else if (RGB == src && RGB_32F == dst) {
    pImpl = new rbg8_rgb32f(ctx, str);
  } else if (RGB == src && Y == dst) {
    pImpl = new rbg8_y(ctx, str);
  } else if (RGB_32F == src && RGB_32F_PLANAR == dst) {
    pImpl = new rgb32f_deinterleave(ctx, str);
  } else {
    throw std::invalid_argument("Unsupported pixel format conversion");
  }
}

ConvertSurface::~ConvertSurface() { delete pImpl; }

TaskExecStatus ConvertSurface::Run() {
  ClearOutputs();

  ColorspaceConversionContext* pCtx = nullptr;
  auto ctx_buf = (Buffer*)GetInput(2U);
  if (ctx_buf) {
    pCtx = ctx_buf->GetDataAs<ColorspaceConversionContext>();
  }

  auto ret = pImpl->Execute(GetInput(0), GetInput(1), pCtx);
  pImpl->pDetails->CopyFrom(sizeof(pImpl->details),
                            (const void*)&pImpl->details);

  SetOutput(pImpl->pDetails.get(), 0U);

  return ret;
}
