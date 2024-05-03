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
#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "NvCodecUtils.h"
#include "Tasks.hpp"

#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>

using namespace VPF;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

namespace VPF {

auto const cuda_stream_sync = [](void* stream) {
  cuStreamSynchronize((CUstream)stream);
};

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

  virtual Token* Execute(Token* pSrcToken, Token* pDstToken,
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

  Surface* GetOutput(Surface* pSrc, Surface* pDst) {
    if (!pSrc) {
      return nullptr;
    }

    if (pDst) {
      return pDst;
    }

    if (!pSurface) {
      pSurface = Surface::Make(dstFmt, pSrc->Width(), pSrc->Height(), cu_ctx);
    }

    return pSurface.get();
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
  std::shared_ptr<Surface> pSurface = nullptr;
  std::shared_ptr<Buffer> pDetails = nullptr;
  TaskExecDetails details;
};

struct nv12_bgr final : public NppConvertSurface_Impl {
  nv12_bgr(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, BGR) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto func = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx;
      switch (color_space) {
      case BT_709:
        func = (JPEG == color_range) ? nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx
                                     : nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx;
        break;
      case BT_601:
        if (JPEG == color_range) {
          func = nppiNV12ToBGR_8u_P2C3R_Ctx;
        } else {
          details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
          return nullptr;
        }
        break;
      default:
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      /* NPP assumes that both Y and UV planes of NV12 Surface have same pitch.
       * VPF doesn't impose such a restriction, hence need to check;
       */
      if (src_ctx.GetPitch()[0] != src_ctx.GetPitch()[1]) {
        details.info = TaskExecInfo::UNSUPPORTED_IMG_PITCH_PARAMS;
        return nullptr;
      }

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>().data(),
                           src_ctx.GetPitch()[0], dst_ctx.GetDataAs<Npp8u>()[0],
                           dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct nv12_rgb final : public NppConvertSurface_Impl {
  nv12_rgb(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, RGB) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto func = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx;
      switch (color_space) {
      case BT_709:
        func = (JPEG == color_range) ? nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx
                                     : nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx;
        break;
      case BT_601:
        if (JPEG == color_range) {
          func = nppiNV12ToRGB_8u_P2C3R_Ctx;
        } else {
          details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
          return nullptr;
        }
        break;
      default:
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      /* NPP assumes that both Y and UV planes of NV12 Surface have same pitch.
       * VPF doesn't impose such a restriction, hence need to check;
       */
      if (src_ctx.GetPitch()[0] != src_ctx.GetPitch()[1]) {
        details.info = TaskExecInfo::UNSUPPORTED_IMG_PITCH_PARAMS;
        return nullptr;
      }

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>().data(),
                           src_ctx.GetPitch()[0], dst_ctx.GetDataAs<Npp8u>()[0],
                           dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct nv12_yuv420 final : public NppConvertSurface_Impl {
  nv12_yuv420(uint32_t width, uint32_t height, CUcontext context,
              CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, YUV420) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      /* NPP assumes that both Y and UV planes of NV12 Surface have same pitch.
       * VPF doesn't impose such a restriction, hence need to check;
       */
      if (src_ctx.GetPitch()[0] != src_ctx.GetPitch()[1]) {
        details.info = TaskExecInfo::UNSUPPORTED_IMG_PITCH_PARAMS;
        return nullptr;
      }

      CudaCtxPush ctxPush(cu_ctx);
      if (JPEG == color_range) {
        ThrowOnNppError(nppiNV12ToYUV420_8u_P2P3R_Ctx(
                            src_ctx.GetDataAs<Npp8u>().data(),
                            src_ctx.GetPitch()[0],
                            dst_ctx.GetDataAs<Npp8u>().data(),
                            dst_ctx.GetPitch(), src_ctx.GetSize(), nppCtx),
                        __LINE__);
      } else {
        ThrowOnNppError(
            nppiYCbCr420_8u_P2P3R_Ctx(
                src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                src_ctx.GetDataAs<Npp8u>()[1], src_ctx.GetPitch()[1],
                dst_ctx.GetDataAs<Npp8u>().data(), dst_ctx.GetPitch(),
                src_ctx.GetSize(), nppCtx),
            __LINE__);
      }
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct nv12_y final : public NppConvertSurface_Impl {
  nv12_y(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, NV12, Y) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      SurfacePlane src_plane, dst_plane;
      if (!pInput->GetSurfacePlane(src_plane) ||
          !pOutput->GetSurfacePlane(dst_plane)) {
        details.info = TaskExecInfo::INVALID_INPUT;
        return nullptr;
      }

      CUDA_MEMCPY2D m = {0};
      m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
      m.srcDevice = src_plane.GpuMem();
      m.dstDevice = dst_plane.GpuMem();
      m.srcPitch = src_plane.Pitch();
      m.dstPitch = dst_plane.Pitch();
      m.Height = src_plane.Height();
      m.WidthInBytes = src_plane.Width() * src_plane.ElemSize();

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnCudaError(cuMemcpy2DAsync(&m, cu_str), __LINE__);
      ThrowOnCudaError(cuStreamSynchronize(cu_str), __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rbg8_y final : public NppConvertSurface_Impl {
  rbg8_y(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, Y) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      ThrowOnNppError(nppiRGBToGray_8u_C3C1R_Ctx(
                          src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp8u>()[0], dst_ctx.GetPitch()[0],
                          src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct yuv420_rgb final : public NppConvertSurface_Impl {
  yuv420_rgb(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV420, RGB) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceYUV420*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto func = (JPEG == color_range) ? nppiYUV420ToRGB_8u_P3C3R_Ctx
                                        : nppiYCbCr420ToRGB_8u_P3C3R_Ctx;

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>().data(),
                           src_ctx.GetPitch(), dst_ctx.GetDataAs<Npp8u>()[0],
                           dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct yuv420_bgr final : public NppConvertSurface_Impl {
  yuv420_bgr(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV420, BGR) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceYUV420*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto func = (JPEG == color_range) ? nppiYUV420ToBGR_8u_P3C3R_Ctx
                                        : nppiYCbCr420ToBGR_8u_P3C3R_Ctx;

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>().data(),
                           src_ctx.GetPitch(), dst_ctx.GetDataAs<Npp8u>()[0],
                           dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct yuv444_bgr final : public NppConvertSurface_Impl {
  yuv444_bgr(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV444, BGR) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto func = (MPEG == color_range) ? nppiYCbCrToBGR_8u_P3C3R_Ctx
                                        : nppiYUVToBGR_8u_P3C3R_Ctx;
      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>().data(),
                           src_ctx.GetPitch()[0], dst_ctx.GetDataAs<Npp8u>()[0],
                           dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct yuv444_rgb final : public NppConvertSurface_Impl {
  yuv444_rgb(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV444, RGB) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space || JPEG != color_range) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiYUVToRGB_8u_P3C3R_Ctx(
                          src_ctx.GetDataAs<Npp8u>().data(),
                          src_ctx.GetPitch()[0], dst_ctx.GetDataAs<Npp8u>()[0],
                          dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct yuv444_rgb_planar final : public NppConvertSurface_Impl {
  yuv444_rgb_planar(uint32_t width, uint32_t height, CUcontext context,
                    CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV444, RGB_PLANAR) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceYUV444*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space || JPEG != color_range) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiYUVToRGB_8u_P3R_Ctx(src_ctx.GetDataAs<Npp8u>().data(),
                                              src_ctx.GetPitch()[0],
                                              dst_ctx.GetDataAs<Npp8u>().data(),
                                              dst_ctx.GetPitch()[0],
                                              src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct bgr_yuv444 final : public NppConvertSurface_Impl {
  bgr_yuv444(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream, BGR, YUV444) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceBGR*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto func = (MPEG == color_range) ? nppiBGRToYCbCr_8u_C3P3R_Ctx
                                        : nppiBGRToYUV_8u_C3P3R_Ctx;

      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                           dst_ctx.GetDataAs<Npp8u>().data(),
                           dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }
    return pOutput;
  }
};

struct rgb_yuv444 final : public NppConvertSurface_Impl {
  rgb_yuv444(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, YUV444) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiRGBToYUV_8u_C3P3R_Ctx(
                          src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp8u>().data(),
                          dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rgb_planar_yuv444 final : public NppConvertSurface_Impl {
  rgb_planar_yuv444(uint32_t width, uint32_t height, CUcontext context,
                    CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB_PLANAR, YUV444) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGBPlanar*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto func = (JPEG == color_range) ? nppiRGBToYUV_8u_P3R_Ctx
                                        : nppiRGBToYCbCr_8u_P3R_Ctx;

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>().data(),
                           src_ctx.GetPitch()[0],
                           dst_ctx.GetDataAs<Npp8u>().data(),
                           dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct y_yuv444 final : public NppConvertSurface_Impl {
  y_yuv444(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, Y, YUV444) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceY*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      // Copy Y channel;
      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiCopy_8u_C1R_Ctx(
                          src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp8u>()[0], dst_ctx.GetPitch()[0],
                          src_ctx.GetSize(), nppCtx),
                      __LINE__);

      // Set U and V channels to gray (128U);
      for (auto i : {1U, 2U}) {
        const Npp8u gray = 128U;
        ThrowOnNppError(nppiSet_8u_C1R_Ctx(gray, dst_ctx.GetDataAs<Npp8u>()[i],
                                           dst_ctx.GetPitch()[i],
                                           src_ctx.GetSize(), nppCtx),
                        __LINE__);
      }
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rgb_yuv420 final : public NppConvertSurface_Impl {
  rgb_yuv420(uint32_t width, uint32_t height, CUcontext context,
             CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, YUV420) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto const params = GetParams(pCtx);
      auto const color_space = std::get<0>(params);
      auto const color_range = std::get<1>(params);

      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      if (BT_601 != color_space) {
        details.info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
        return nullptr;
      }

      auto func = (JPEG == color_range) ? nppiRGBToYUV420_8u_C3P3R_Ctx
                                        : nppiRGBToYCbCr420_8u_C3P3R_Ctx;

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(func(src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                           dst_ctx.GetDataAs<Npp8u>().data(),
                           dst_ctx.GetPitch(), src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct yuv420_nv12 final : public NppConvertSurface_Impl {
  yuv420_nv12(uint32_t width, uint32_t height, CUcontext context,
              CUstream stream)
      : NppConvertSurface_Impl(context, stream, YUV420, NV12) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (Surface*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiYCbCr420_8u_P3P2R_Ctx(
                          src_ctx.GetDataAs<Npp8u>().data(), src_ctx.GetPitch(),
                          dst_ctx.GetDataAs<Npp8u>()[0], dst_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp8u>()[1], dst_ctx.GetPitch()[1],
                          src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rgb8_deinterleave final : public NppConvertSurface_Impl {
  rgb8_deinterleave(uint32_t width, uint32_t height, CUcontext context,
                    CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, RGB_PLANAR) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiCopy_8u_C3P3R_Ctx(
                          src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp8u>().data(),
                          dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rgb8_interleave final : public NppConvertSurface_Impl {
  rgb8_interleave(uint32_t width, uint32_t height, CUcontext context,
                  CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB_PLANAR, RGB) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGBPlanar*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiCopy_8u_P3C3R_Ctx(
                          src_ctx.GetDataAs<Npp8u>().data(),
                          src_ctx.GetPitch()[0], dst_ctx.GetDataAs<Npp8u>()[0],
                          dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rgb_bgr final : public NppConvertSurface_Impl {
  rgb_bgr(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, BGR) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      const int aDstOrder[3] = {2, 1, 0};
      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiSwapChannels_8u_C3R_Ctx(
                          src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp8u>()[0], dst_ctx.GetPitch()[0],
                          src_ctx.GetSize(), aDstOrder, nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct bgr_rgb final : public NppConvertSurface_Impl {
  bgr_rgb(uint32_t width, uint32_t height, CUcontext context, CUstream stream)
      : NppConvertSurface_Impl(context, stream, BGR, RGB) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceBGR*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      const int aDstOrder[3] = {2, 1, 0};
      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiSwapChannels_8u_C3R_Ctx(
                          src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp8u>()[0], dst_ctx.GetPitch()[0],
                          src_ctx.GetSize(), aDstOrder, nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rbg8_rgb32f final : public NppConvertSurface_Impl {
  rbg8_rgb32f(uint32_t width, uint32_t height, CUcontext context,
              CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB, RGB_32F) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      Npp32f nMin = 0.0;
      Npp32f nMax = 1.0;

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiScale_8u32f_C3R_Ctx(
                          src_ctx.GetDataAs<Npp8u>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp32f>()[0], dst_ctx.GetPitch()[0],
                          src_ctx.GetSize(), nMin, nMax, nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct rgb32f_deinterleave final : public NppConvertSurface_Impl {
  rgb32f_deinterleave(uint32_t width, uint32_t height, CUcontext context,
                      CUstream stream)
      : NppConvertSurface_Impl(context, stream, RGB_32F, RGB_32F_PLANAR) {}

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    auto pInput = (SurfaceRGB*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);

    if (!Validate(pInput, pOutput)) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();

      /* NPP assumes that all planes of SurfaceRGB32FPlanar have same
       * pitch. VPF doesn't impose such a restriction, hence need to check;
       */
      if (src_ctx.GetPitch()[0] != src_ctx.GetPitch()[1] ||
          src_ctx.GetPitch()[0] != src_ctx.GetPitch()[2]) {
        details.info = TaskExecInfo::UNSUPPORTED_IMG_PITCH_PARAMS;
        return nullptr;
      }

      CudaCtxPush ctxPush(cu_ctx);
      ThrowOnNppError(nppiCopy_32f_C3P3R_Ctx(
                          src_ctx.GetDataAs<Npp32f>()[0], src_ctx.GetPitch()[0],
                          dst_ctx.GetDataAs<Npp32f>().data(),
                          dst_ctx.GetPitch()[0], src_ctx.GetSize(), nppCtx),
                      __LINE__);
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }
};

struct p16_nv12 final : public NppConvertSurface_Impl {
  /* Source format is needed because P10 and P12 to NV12 conversions share
   * the same implementation. The only difference is bit depth value which
   * is calculated at a runtime.
   */
  p16_nv12(uint32_t width, uint32_t height, CUcontext context, CUstream stream,
           Pixel_Format src_fmt)
      : NppConvertSurface_Impl(context, stream, src_fmt, NV12) {
    pScratch = Surface::Make(GRAY16, width, height, context);
  }

  Token* Execute(Token* pSrcToken, Token* pDstToken,
                 ColorspaceConversionContext* pCtx) {
    NvtxMark tick(GetNvtxTickName().c_str());
    details.info = TaskExecInfo::SUCCESS;

    /* No Validate() call here because same code serves 2 different input
     * pixel formats.
     */
    auto pInput = (Surface*)pSrcToken;
    auto pOutput = GetOutput(pInput, (Surface*)pDstToken);
    if (!pInput || !pOutput) {
      details.info = TaskExecInfo::INVALID_INPUT;
      return nullptr;
    }

    try {
      auto src_ctx = pInput->GetNppContext();
      auto dst_ctx = pOutput->GetNppContext();
      auto tmp_ctx = pScratch->GetNppContext();

      Npp16u nConstant = 1 << (pInput->ElemSize() - pSurface->ElemSize()) * 8;
      auto const nScaleFactor = 0;

      CudaCtxPush ctxPush(cu_ctx);
      for (int i = 0; i < pInput->NumPlanes(); i++) {
        // Take 8 most significant bits, save result in 16-bit scratch buffer;
        ThrowOnNppError(
            nppiDivC_16u_C1RSfs_Ctx(
                src_ctx.GetDataAs<Npp16u>()[i], src_ctx.GetPitch()[i],
                nConstant, tmp_ctx.GetDataAs<Npp16u>()[0],
                tmp_ctx.GetPitch()[0], src_ctx.GetSize(), nScaleFactor, nppCtx),
            __LINE__);

        // Bit depth conversion from 16-bit tmp_ctx to output;
        ThrowOnNppError(nppiConvert_16u8u_C1R_Ctx(
                            tmp_ctx.GetDataAs<Npp16u>()[0],
                            tmp_ctx.GetPitch()[0],
                            src_ctx.GetDataAs<Npp8u>()[i],
                            src_ctx.GetPitch()[i], src_ctx.GetSize(), nppCtx),
                        __LINE__);
      }
    } catch (...) {
      details.info = TaskExecInfo::FAIL;
      return nullptr;
    }

    return pOutput;
  }

  std::shared_ptr<Surface> pScratch = nullptr;
};
} // namespace VPF

ConvertSurface::ConvertSurface(uint32_t width, uint32_t height,
                               Pixel_Format inFormat, Pixel_Format outFormat,
                               CUcontext ctx, CUstream str)
    : Task("NppConvertSurface", ConvertSurface::numInputs,
           ConvertSurface::numOutputs, nullptr, nullptr) {
  if (NV12 == inFormat && YUV420 == outFormat) {
    pImpl = new nv12_yuv420(width, height, ctx, str);
  } else if (YUV420 == inFormat && NV12 == outFormat) {
    pImpl = new yuv420_nv12(width, height, ctx, str);
  } else if (P10 == inFormat && NV12 == outFormat) {
    pImpl = new p16_nv12(width, height, ctx, str, P10);
  } else if (P12 == inFormat && NV12 == outFormat) {
    pImpl = new p16_nv12(width, height, ctx, str, P12);
  } else if (NV12 == inFormat && RGB == outFormat) {
    pImpl = new nv12_rgb(width, height, ctx, str);
  } else if (NV12 == inFormat && BGR == outFormat) {
    pImpl = new nv12_bgr(width, height, ctx, str);
  } else if (RGB == inFormat && RGB_PLANAR == outFormat) {
    pImpl = new rgb8_deinterleave(width, height, ctx, str);
  } else if (RGB_PLANAR == inFormat && RGB == outFormat) {
    pImpl = new rgb8_interleave(width, height, ctx, str);
  } else if (RGB_PLANAR == inFormat && YUV444 == outFormat) {
    pImpl = new rgb_planar_yuv444(width, height, ctx, str);
  } else if (Y == inFormat && YUV444 == outFormat) {
    pImpl = new y_yuv444(width, height, ctx, str);
  } else if (YUV420 == inFormat && RGB == outFormat) {
    pImpl = new yuv420_rgb(width, height, ctx, str);
  } else if (RGB == inFormat && YUV420 == outFormat) {
    pImpl = new rgb_yuv420(width, height, ctx, str);
  } else if (RGB == inFormat && YUV444 == outFormat) {
    pImpl = new rgb_yuv444(width, height, ctx, str);
  } else if (RGB == inFormat && BGR == outFormat) {
    pImpl = new rgb_bgr(width, height, ctx, str);
  } else if (BGR == inFormat && RGB == outFormat) {
    pImpl = new bgr_rgb(width, height, ctx, str);
  } else if (YUV420 == inFormat && BGR == outFormat) {
    pImpl = new yuv420_bgr(width, height, ctx, str);
  } else if (YUV444 == inFormat && BGR == outFormat) {
    pImpl = new yuv444_bgr(width, height, ctx, str);
  } else if (YUV444 == inFormat && RGB == outFormat) {
    pImpl = new yuv444_rgb(width, height, ctx, str);
  } else if (BGR == inFormat && YUV444 == outFormat) {
    pImpl = new bgr_yuv444(width, height, ctx, str);
  } else if (NV12 == inFormat && Y == outFormat) {
    pImpl = new nv12_y(width, height, ctx, str);
  } else if (RGB == inFormat && RGB_32F == outFormat) {
    pImpl = new rbg8_rgb32f(width, height, ctx, str);
  } else if (RGB == inFormat && Y == outFormat) {
    pImpl = new rbg8_y(width, height, ctx, str);
  } else if (RGB_32F == inFormat && RGB_32F_PLANAR == outFormat) {
    pImpl = new rgb32f_deinterleave(width, height, ctx, str);
  } else {
    throw std::invalid_argument("Unsupported pixel format conversion");
  }
}

ConvertSurface::~ConvertSurface() { delete pImpl; }

ConvertSurface* ConvertSurface::Make(uint32_t width, uint32_t height,
                                     Pixel_Format inFormat,
                                     Pixel_Format outFormat, CUcontext ctx,
                                     CUstream str) {
  return new ConvertSurface(width, height, inFormat, outFormat, ctx, str);
}

TaskExecStatus ConvertSurface::Run() {
  ClearOutputs();

  ColorspaceConversionContext* pCtx = nullptr;
  auto ctx_buf = (Buffer*)GetInput(2U);
  if (ctx_buf) {
    pCtx = ctx_buf->GetDataAs<ColorspaceConversionContext>();
  }

  auto pOutput = pImpl->Execute(GetInput(0), GetInput(1), pCtx);
  pImpl->pDetails->CopyFrom(sizeof(pImpl->details),
                            (const void*)&pImpl->details);

  SetOutput(pOutput, 0U);
  SetOutput(pImpl->pDetails.get(), 1U);

  return TASK_EXEC_SUCCESS;
}
