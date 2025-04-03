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

/// @brief Conversion implementation function pointer
/// @param src_token source
/// @param dst_token destination
/// @param gpu_id gpu id
/// @param stream cuda stream
/// @param npp_ctx npp stream context
/// @param scratch scratch buffer
/// @param cc_ctx color conversion context
/// @return execution status
typedef TaskExecDetails (*ConvImpl)(
    Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
    NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
    std::optional<ColorspaceConversionContext> cc_ctx);

static TaskExecDetails
nv12_bgr(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
         NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
         std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_709;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U)};

  auto pDst = (Npp8u*)dst_surf.PixelPtr(0U);
  NppiSize oSizeRoi = {(int)src_surf.Width(), (int)src_surf.Height()};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_space) {
  case BT_709:
    if (JPEG == color_range) {
      err = LibNpp::nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(
          pSrc, src_surf.Pitch(), pDst, dst_surf.Pitch(), oSizeRoi, npp_ctx);
    } else {
      err = LibNpp::nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(
          pSrc, src_surf.Pitch(), pDst, dst_surf.Pitch(), oSizeRoi, npp_ctx);
    }
    break;
  case BT_601:
    if (JPEG == color_range) {
      err = LibNpp::nppiNV12ToBGR_8u_P2C3R_Ctx(
          pSrc, src_surf.Pitch(), pDst, dst_surf.Pitch(), oSizeRoi, npp_ctx);
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

static TaskExecDetails
nv12_rgb(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
         NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
         std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_709;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  const Npp8u* const pSrc[] = {(const Npp8u* const)src_surf.PixelPtr(0U),
                               (const Npp8u* const)src_surf.PixelPtr(1U)};

  auto pDst = (Npp8u*)dst_surf.PixelPtr();
  NppiSize oSizeRoi = {(int)src_surf.Width(), (int)src_surf.Height()};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_space) {
  case BT_709:
    if (JPEG == color_range) {
      err = LibNpp::nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
          pSrc, src_surf.Pitch(), pDst, dst_surf.Pitch(), oSizeRoi, npp_ctx);
    } else {
      err = LibNpp::nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
          pSrc, src_surf.Pitch(), pDst, dst_surf.Pitch(), oSizeRoi, npp_ctx);
    }
    break;
  case BT_601:
    if (JPEG == color_range) {
      err = LibNpp::nppiNV12ToRGB_8u_P2C3R_Ctx(
          pSrc, src_surf.Pitch(), pDst, dst_surf.Pitch(), oSizeRoi, npp_ctx);
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

static TaskExecDetails
nv12_yuv420(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
            NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
            std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U)};

  Npp8u* pDst[] = {(Npp8u*)dst_surf.PixelPtr(0U), (Npp8u*)dst_surf.PixelPtr(1U),
                   (Npp8u*)dst_surf.PixelPtr(2U)};

  int dstStep[] = {(int)dst_surf.Pitch(0U), (int)dst_surf.Pitch(1U),
                   (int)dst_surf.Pitch(2U)};
  NppiSize roi = {(int)src_surf.Width(), (int)src_surf.Height()};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;
  switch (color_range) {
  case JPEG:
    err = LibNpp::nppiNV12ToYUV420_8u_P2P3R_Ctx(pSrc, src_surf.Pitch(0U), pDst,
                                                dstStep, roi, npp_ctx);
    break;
  case MPEG:
    err = LibNpp::nppiYCbCr420_8u_P2P3R_Ctx(pSrc[0], src_surf.Pitch(0U),
                                            pSrc[1], src_surf.Pitch(1U), pDst,
                                            dstStep, roi, npp_ctx);
    break;
  default:
    return s_unsupp_cc_ctx;
  }

  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
nv12_y(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
       NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
       std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  try {
    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = src_surf.PixelPtr();
    m.dstDevice = dst_surf.PixelPtr();
    m.srcPitch = src_surf.Pitch();
    m.dstPitch = dst_surf.Pitch();
    m.Height = src_surf.Height();
    m.WidthInBytes = src_surf.WidthInBytes();

    CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
    ThrowOnCudaError(LibCuda::cuMemcpy2DAsync(&m, stream), __LINE__);
    ThrowOnCudaError(LibCuda::cuStreamSynchronize(stream), __LINE__);
  } catch (...) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rbg8_y(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
       NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
       std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};

  auto ret = LibNpp::nppiRGBToGray_8u_C3C1R_Ctx(
      (const Npp8u*)src_surf.PixelPtr(), src_surf.Pitch(),
      (Npp8u*)dst_surf.PixelPtr(), dst_surf.Pitch(), roi, npp_ctx);

  if (NPP_NO_ERROR != ret) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
yuv420_rgb(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
           NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
           std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U),
                               (const Npp8u*)src_surf.PixelPtr(2U)};
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr();
  int srcStep[] = {(int)src_surf.Pitch(0U), (int)src_surf.Pitch(1U),
                   (int)src_surf.Pitch(2U)};
  int dstStep = (int)dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_space) {
  case BT_709:
    return s_unsupp_cc_ctx;
  case BT_601:
    if (JPEG == color_range) {
      err = LibNpp::nppiYUV420ToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                 roi, npp_ctx);
    } else {
      err = LibNpp::nppiYCbCr420ToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                   roi, npp_ctx);
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

static TaskExecDetails
yuv420_bgr(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
           NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
           std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U),
                               (const Npp8u*)src_surf.PixelPtr(2U)};
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr();
  int srcStep[] = {(int)src_surf.Pitch(0U), (int)src_surf.Pitch(1U),
                   (int)src_surf.Pitch(2U)};
  int dstStep = (int)dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_space) {
  case BT_709:
    return s_unsupp_cc_ctx;
  case BT_601:
    if (JPEG == color_range) {
      err = LibNpp::nppiYUV420ToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                 roi, npp_ctx);
    } else {
      err = LibNpp::nppiYCbCr420ToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                   roi, npp_ctx);
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

static TaskExecDetails
yuv444_bgr(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
           NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
           std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  if (BT_601 != color_space) {
    return s_unsupp_cc_ctx;
  }

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U),
                               (const Npp8u*)src_surf.PixelPtr(2U)};
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr();
  int srcStep = (int)src_surf.Pitch();
  int dstStep = (int)dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_range) {
  case MPEG:
    err = LibNpp::nppiYCbCrToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              npp_ctx);
    break;
  case JPEG:
    err = LibNpp::nppiYUVToBGR_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                            npp_ctx);
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

static TaskExecDetails
yuv444_rgb(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
           NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
           std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  if (BT_601 != color_space) {
    return s_unsupp_cc_ctx;
  }

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U),
                               (const Npp8u*)src_surf.PixelPtr(2U)};
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr();
  int srcStep = (int)src_surf.Pitch();
  int dstStep = (int)dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_range) {
  case JPEG:
    err = LibNpp::nppiYUVToRGB_8u_P3C3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                            npp_ctx);
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

static TaskExecDetails
yuv444_rgb_planar(Token& src_token, Token& dst_token, int gpu_id,
                  CUstream stream, NppStreamContext& npp_ctx,
                  std::optional<SurfacePlane> scratch,
                  std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  if (BT_601 != color_space) {
    return s_unsupp_cc_ctx;
  }

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U),
                               (const Npp8u*)src_surf.PixelPtr(2U)};
  Npp8u* pDst[] = {(Npp8u*)dst_surf.PixelPtr(0), (Npp8u*)dst_surf.PixelPtr(1),
                   (Npp8u*)dst_surf.PixelPtr(2)};
  int srcStep = (int)src_surf.Pitch();
  int dstStep = (int)dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_range) {
  case JPEG:
    err = LibNpp::nppiYUVToRGB_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                          npp_ctx);
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

static TaskExecDetails
bgr_yuv444(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
           NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
           std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  if (BT_601 != color_space) {
    return s_unsupp_cc_ctx;
  }

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();
  Npp8u* pDst[] = {(Npp8u*)dst_surf.PixelPtr(0U), (Npp8u*)dst_surf.PixelPtr(1U),
                   (Npp8u*)dst_surf.PixelPtr(2U)};
  int srcStep = (int)src_surf.Pitch();
  int dstStep = (int)dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;

  switch (color_range) {
  case MPEG:
    err = LibNpp::nppiBGRToYCbCr_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                              npp_ctx);
    break;
  case JPEG:
    err = LibNpp::nppiBGRToYUV_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                            npp_ctx);
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

static TaskExecDetails
rgb_yuv444(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
           NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
           std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  if (BT_601 != color_space) {
    return s_unsupp_cc_ctx;
  }

  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();
  int srcStep = src_surf.Pitch();
  Npp8u* pDst[] = {(Npp8u*)dst_surf.PixelPtr(0U), (Npp8u*)dst_surf.PixelPtr(1U),
                   (Npp8u*)dst_surf.PixelPtr(2U)};
  int dstStep = dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;
  switch (color_range) {
  case JPEG:
    err = LibNpp::nppiRGBToYUV_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                            npp_ctx);
    break;
  case MPEG:
    err = LibNpp::nppiRGBToYCbCr_8u_C3R_Ctx(pSrc, srcStep, pDst[0], dstStep,
                                            roi, npp_ctx);
    break;
  default:
    err = NPP_NO_OPERATION_WARNING;
  }

  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rgb_planar_yuv444(Token& src_token, Token& dst_token, int gpu_id,
                  CUstream stream, NppStreamContext& npp_ctx,
                  std::optional<SurfacePlane> scratch,
                  std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  if (BT_601 != color_space) {
    return s_unsupp_cc_ctx;
  }

  const Npp8u* pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                         (const Npp8u*)src_surf.PixelPtr(1U),
                         (const Npp8u*)src_surf.PixelPtr(2U)};
  int srcStep = src_surf.Pitch();
  Npp8u* pDst[] = {(Npp8u*)dst_surf.PixelPtr(0U), (Npp8u*)dst_surf.PixelPtr(1U),
                   (Npp8u*)dst_surf.PixelPtr(2U)};
  int dstStep = dst_surf.Pitch();
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;
  switch (color_range) {
  case JPEG:
    err = LibNpp::nppiRGBToYUV_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                          npp_ctx);
    break;
  case MPEG:
    err = LibNpp::nppiRGBToYCbCr_8u_P3R_Ctx(pSrc, srcStep, pDst, dstStep, roi,
                                            npp_ctx);
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

static TaskExecDetails
y_yuv444(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
         NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
         std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  // Make gray U and V channels;
  for (int i = 1; i < dst_surf.NumPlanes(); i++) {
    const Npp8u nValue = 128U;
    Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr(i);
    int nDstStep = dst_surf.Pitch(i);
    NppiSize roi = {(int)dst_surf.Width(i), (int)dst_surf.Height(i)};
    auto err = LibNpp::nppiSet_8u_C1R_Ctx(nValue, pDst, nDstStep, roi, npp_ctx);
    if (NPP_NO_ERROR != err) {
      return s_fail;
    }
  }

  // Copy Y channel;
  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();
  int nSrcStep = src_surf.Pitch();
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr(0U);
  int nDstStep = dst_surf.Pitch(0U);
  NppiSize roi = {(int)src_surf.Width(), (int)src_surf.Height()};
  auto err =
      LibNpp::nppiCopy_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, roi, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rgb_yuv420(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
           NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
           std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  auto const color_space = cc_ctx ? cc_ctx->color_space : BT_601;
  auto const color_range = cc_ctx ? cc_ctx->color_range : JPEG;

  if (BT_601 != color_space) {
    return s_unsupp_cc_ctx;
  }

  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();
  int srcStep = src_surf.Pitch();
  Npp8u* pDst[] = {(Npp8u*)dst_surf.PixelPtr(0U), (Npp8u*)dst_surf.PixelPtr(1U),
                   (Npp8u*)dst_surf.PixelPtr(2U)};
  int dstStep[] = {(int)dst_surf.Pitch(0U), (int)dst_surf.Pitch(1U),
                   (int)dst_surf.Pitch(2U)};
  NppiSize roi = {(int)dst_surf.Width(), (int)dst_surf.Height()};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = NPP_NO_ERROR;
  switch (color_range) {
  case JPEG:
    err = LibNpp::nppiRGBToYUV420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                               roi, npp_ctx);
    break;

  case MPEG:
    err = LibNpp::nppiRGBToYCbCr420_8u_C3P3R_Ctx(pSrc, srcStep, pDst, dstStep,
                                                 roi, npp_ctx);
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

static TaskExecDetails
yuv420_nv12(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
            NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
            std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {(const Npp8u*)src_surf.PixelPtr(0U),
                               (const Npp8u*)src_surf.PixelPtr(1U),
                               (const Npp8u*)src_surf.PixelPtr(2U)};

  Npp8u* pDst[] = {(Npp8u*)dst_surf.PixelPtr(0U),
                   (Npp8u*)dst_surf.PixelPtr(1U)};

  int srcStep[] = {(int)src_surf.Pitch(0U), (int)src_surf.Pitch(1U),
                   (int)src_surf.Pitch(2U)};
  int dstStep[] = {(int)dst_surf.Pitch(0U), (int)dst_surf.Pitch(1U)};
  NppiSize roi = {(int)src_surf.Width(), (int)src_surf.Height()};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = LibNpp::nppiYCbCr420_8u_P3P2R_Ctx(
      pSrc, srcStep, pDst[0], dstStep[0], pDst[1], dstStep[1], roi, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rgb8_deinterleave(Token& src_token, Token& dst_token, int gpu_id,
                  CUstream stream, NppStreamContext& npp_ctx,
                  std::optional<SurfacePlane> scratch,
                  std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();
  int nSrcStep = src_surf.Pitch();
  Npp8u* aDst[] = {
      (Npp8u*)dst_surf.PixelPtr(),
      (Npp8u*)dst_surf.PixelPtr() + dst_surf.Height() * dst_surf.Pitch(),
      (Npp8u*)dst_surf.PixelPtr() + dst_surf.Height() * dst_surf.Pitch() * 2};
  int nDstStep = dst_surf.Pitch();
  NppiSize oSizeRoi = {0};
  oSizeRoi.height = dst_surf.Height();
  oSizeRoi.width = dst_surf.Width();

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = LibNpp::nppiCopy_8u_C3P3R_Ctx(pSrc, nSrcStep, aDst, nDstStep,
                                           oSizeRoi, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rgb8_interleave(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
                NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
                std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* const pSrc[] = {
      (Npp8u*)src_surf.PixelPtr(),
      (Npp8u*)src_surf.PixelPtr() + src_surf.Height() * src_surf.Pitch(),
      (Npp8u*)src_surf.PixelPtr() + src_surf.Height() * src_surf.Pitch() * 2};
  int nSrcStep = src_surf.Pitch();
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr();
  int nDstStep = dst_surf.Pitch();
  NppiSize oSizeRoi = {0};
  oSizeRoi.height = dst_surf.Height();
  oSizeRoi.width = dst_surf.Width();

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = LibNpp::nppiCopy_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                           oSizeRoi, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rgb_bgr(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
        NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
        std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();
  int nSrcStep = src_surf.Pitch();
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr();
  int nDstStep = dst_surf.Pitch();
  NppiSize oSizeRoi = {0};
  oSizeRoi.height = dst_surf.Height();
  oSizeRoi.width = dst_surf.Width();
  // rgb to brg
  const int aDstOrder[3] = {2, 1, 0};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = LibNpp::nppiSwapChannels_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                                 oSizeRoi, aDstOrder, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
bgr_rgb(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
        NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
        std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();
  int nSrcStep = src_surf.Pitch();
  Npp8u* pDst = (Npp8u*)dst_surf.PixelPtr();
  int nDstStep = dst_surf.Pitch();
  NppiSize oSizeRoi = {0};
  oSizeRoi.height = dst_surf.Height();
  oSizeRoi.width = dst_surf.Width();
  // brg to rgb
  const int aDstOrder[3] = {2, 1, 0};
  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = LibNpp::nppiSwapChannels_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                                 oSizeRoi, aDstOrder, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rbg8_rgb32f(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
            NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
            std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp8u* pSrc = (const Npp8u*)src_surf.PixelPtr();

  int nSrcStep = src_surf.Pitch();
  Npp32f* pDst = (Npp32f*)dst_surf.PixelPtr();
  int nDstStep = dst_surf.Pitch();
  NppiSize oSizeRoi = {0};
  oSizeRoi.height = dst_surf.Height();
  oSizeRoi.width = dst_surf.Width();
  Npp32f nMin = 0.0;
  Npp32f nMax = 1.0;
  const int aDstOrder[3] = {2, 1, 0};

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));

  auto err = LibNpp::nppiScale_8u32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep,
                                             oSizeRoi, nMin, nMax, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
rgb32f_deinterleave(Token& src_token, Token& dst_token, int gpu_id,
                    CUstream stream, NppStreamContext& npp_ctx,
                    std::optional<SurfacePlane> scratch,
                    std::optional<ColorspaceConversionContext> cc_ctx) {
  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);

  const Npp32f* pSrc = (const Npp32f*)src_surf.PixelPtr();
  int nSrcStep = src_surf.Pitch();
  Npp32f* aDst[] = {(Npp32f*)((uint8_t*)dst_surf.PixelPtr()),
                    (Npp32f*)((uint8_t*)dst_surf.PixelPtr() +
                              dst_surf.Height() * dst_surf.Pitch()),
                    (Npp32f*)((uint8_t*)dst_surf.PixelPtr() +
                              dst_surf.Height() * dst_surf.Pitch() * 2)};
  int nDstStep = dst_surf.Pitch();
  NppiSize oSizeRoi = {0};
  oSizeRoi.height = dst_surf.Height();
  oSizeRoi.width = dst_surf.Width();

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));
  auto err = LibNpp::nppiCopy_32f_C3P3R_Ctx(pSrc, nSrcStep, aDst, nDstStep,
                                            oSizeRoi, npp_ctx);
  if (NPP_NO_ERROR != err) {
    return s_fail;
  }

  return s_success;
}

static TaskExecDetails
p16_nv12(Token& src_token, Token& dst_token, int gpu_id, CUstream stream,
         NppStreamContext& npp_ctx, std::optional<SurfacePlane> scratch,
         std::optional<ColorspaceConversionContext> cc_ctx) {
  if (!scratch) {
    return s_invalid_src_dst;
  }

  NvtxMark tick(__FUNCTION__);

  auto& src_surf = static_cast<Surface&>(src_token);
  auto& dst_surf = static_cast<Surface&>(dst_token);
  auto& tmp_surf = scratch.value();

  CudaCtxPush ctxPush(GetContextByStream(gpu_id, stream));

  auto src_plane = src_surf.GetSurfacePlane();
  auto dst_plane = dst_surf.GetSurfacePlane();

  try {
    // Take 8 most significant bits, save result in 16 bit scratch buffer;
    NppiSize oSizeRoi = {0};
    oSizeRoi.height = tmp_surf.Height();
    oSizeRoi.width = tmp_surf.Width();
    ThrowOnNppError(LibNpp::nppiDivC_16u_C1RSfs_Ctx(
                        (Npp16u*)src_plane.GpuMem(), src_plane.Pitch(),
                        1 << (src_plane.ElemSize() - dst_plane.ElemSize()) * 8,
                        (Npp16u*)tmp_surf.GpuMem(), tmp_surf.Pitch(), oSizeRoi,
                        0, npp_ctx),
                    __LINE__);

    // Bit depth conversion from 16-bit scratch to output;
    oSizeRoi.height = dst_plane.Height();
    oSizeRoi.width = dst_plane.Width();
    ThrowOnNppError(LibNpp::nppiConvert_16u8u_C1R_Ctx(
                        (Npp16u*)tmp_surf.GpuMem(), tmp_surf.Pitch(),
                        (Npp8u*)dst_plane.GpuMem(), dst_plane.Pitch(), oSizeRoi,
                        npp_ctx),
                    __LINE__);
  } catch (...) {
    return s_fail;
  }

  return s_success;
}

} // namespace VPF

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

ConvertSurface::ConvertSurface(int gpu_id, CUstream str)
    : m_gpu_id(gpu_id), m_stream(str) {
  SetupNppContext(m_gpu_id, m_stream, m_npp_ctx);
}

static bool Validate(Surface& src, Surface& dst) {
  if ((src.Width() != dst.Width()) || (src.Height() != dst.Height())) {
    return false;
  }

  return true;
}

TaskExecDetails
ConvertSurface::Run(Surface& src, Surface& dst,
                    std::optional<ColorspaceConversionContext> cc_ctx) {

  if (!Validate(src, dst)) {
    return s_invalid_src_dst;
  }

  // These input formats require scratch buffer
  if (P10 == src.PixelFormat() || P12 == src.PixelFormat()) {
    auto src_plane = src.GetSurfacePlane();

    if (m_scratch) {
      if (m_scratch->Width() != src_plane.Width() ||
          m_scratch->Height() != src_plane.Height()) {
        m_scratch.reset();
      }
    }

    if (!m_scratch) {
      m_scratch.reset(new SurfacePlane(src_plane.Width(), src_plane.Height(),
                                       sizeof(uint16_t), kDLUInt, "u",
                                       GetContextByStream(m_stream)));
    }
  }

  ConvImpl impl_func = nullptr;
  auto src_fmt = src.PixelFormat();
  auto dst_fmt = dst.PixelFormat();

  if (NV12 == src_fmt && YUV420 == dst_fmt) {
    impl_func = nv12_yuv420;
  } else if (YUV420 == src_fmt && NV12 == dst_fmt) {
    impl_func = yuv420_nv12;
  } else if (P10 == src_fmt && NV12 == dst_fmt) {
    impl_func = p16_nv12;
  } else if (P12 == src_fmt && NV12 == dst_fmt) {
    impl_func = p16_nv12;
  } else if (NV12 == src_fmt && RGB == dst_fmt) {
    impl_func = nv12_rgb;
  } else if (NV12 == src_fmt && BGR == dst_fmt) {
    impl_func = nv12_bgr;
  } else if (RGB == src_fmt && RGB_PLANAR == dst_fmt) {
    impl_func = rgb8_deinterleave;
  } else if (RGB_PLANAR == src_fmt && RGB == dst_fmt) {
    impl_func = rgb8_interleave;
  } else if (RGB_PLANAR == src_fmt && YUV444 == dst_fmt) {
    impl_func = rgb_planar_yuv444;
  } else if (Y == src_fmt && YUV444 == dst_fmt) {
    impl_func = y_yuv444;
  } else if (YUV420 == src_fmt && RGB == dst_fmt) {
    impl_func = yuv420_rgb;
  } else if (RGB == src_fmt && YUV420 == dst_fmt) {
    impl_func = rgb_yuv420;
  } else if (RGB == src_fmt && YUV444 == dst_fmt) {
    impl_func = rgb_yuv444;
  } else if (RGB == src_fmt && BGR == dst_fmt) {
    impl_func = rgb_bgr;
  } else if (BGR == src_fmt && RGB == dst_fmt) {
    impl_func = bgr_rgb;
  } else if (YUV420 == src_fmt && BGR == dst_fmt) {
    impl_func = yuv420_bgr;
  } else if (YUV444 == src_fmt && BGR == dst_fmt) {
    impl_func = yuv444_bgr;
  } else if (YUV444 == src_fmt && RGB == dst_fmt) {
    impl_func = yuv444_rgb;
  } else if (BGR == src_fmt && YUV444 == dst_fmt) {
    impl_func = bgr_yuv444;
  } else if (NV12 == src_fmt && Y == dst_fmt) {
    impl_func = nv12_y;
  } else if (RGB == src_fmt && RGB_32F == dst_fmt) {
    impl_func = rbg8_rgb32f;
  } else if (RGB == src_fmt && Y == dst_fmt) {
    impl_func = rbg8_y;
  } else if (RGB_32F == src_fmt && RGB_32F_PLANAR == dst_fmt) {
    impl_func = rgb32f_deinterleave;
  } else {
    std::stringstream ss;
    ss << "Unsupported pixel format conversion: " << GetFormatName(src_fmt);
    ss << " -> " << GetFormatName(dst_fmt) << std::endl;
    throw std::invalid_argument(ss.str());
  }

  return impl_func(src, dst, m_gpu_id, m_stream, m_npp_ctx,
                   m_scratch ? std::optional(*m_scratch.get()) : std::nullopt,
                   cc_ctx);
}
