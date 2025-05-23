/*
 * Copyright 2025 Vision Labs LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or Resizeied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "ResizeUtils.hpp"
#include "Surfaces.hpp"
#include "Tasks.hpp"

using namespace VPF;

/// @brief Resizer implementation function pointer.
/// @param src input Surface
/// @param dst output Surface
/// @param ctx NPP stream context
/// @return SUCCESS in case of success, FAIL otherwise
typedef TaskExecInfo (*Resize)(SurfacePlane& src, SurfacePlane& dst,
                               NppStreamContext& ctx);

/**
 * 8 bit implememtation
 */
TaskExecInfo Resize_8U_C1(SurfacePlane& src, SurfacePlane& dst,
                          NppStreamContext& ctx) {

  const Npp8u* pSrc = (const Npp8u*)src.GpuMem();
  int nSrcStep = (int)src.Pitch();
  NppiSize oSrcSize = {src.Width(), src.Height()};
  NppiRect oSrcRectROI = {0, 0, oSrcSize.width, oSrcSize.height};

  Npp8u* pDst = (Npp8u*)dst.GpuMem();
  int nDstStep = (int)dst.Pitch();
  NppiSize oDstSize = {dst.Width(), dst.Height()};
  NppiRect oDstRectROI = {0, 0, oDstSize.width, oDstSize.height};
  int eInterpolation = NPPI_INTER_LANCZOS;

  auto ret = LibNpp::nppiResize_8u_C1R_Ctx(
      pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize,
      oDstRectROI, eInterpolation, ctx);

  if (NPP_NO_ERROR != ret)
    return TaskExecInfo::FAIL;

  return TaskExecInfo::SUCCESS;
}

/**
 * 16 bit implementation
 */
TaskExecInfo Resize_16U_C1(SurfacePlane& src, SurfacePlane& dst,
                           NppStreamContext& ctx) {

  const Npp16u* pSrc = (const Npp16u*)src.GpuMem();
  int nSrcStep = (int)src.Pitch();
  NppiSize oSrcSize = {src.Width(), src.Height()};
  NppiRect oSrcRectROI = {0, 0, oSrcSize.width, oSrcSize.height};

  Npp16u* pDst = (Npp16u*)dst.GpuMem();
  int nDstStep = (int)dst.Pitch();
  NppiSize oDstSize = {dst.Width(), dst.Height()};
  NppiRect oDstRectROI = {0, 0, oDstSize.width, oDstSize.height};
  int eInterpolation = NPPI_INTER_LANCZOS;

  auto ret = LibNpp::nppiResize_16u_C1R_Ctx(
      pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize,
      oDstRectROI, eInterpolation, ctx);

  if (NPP_NO_ERROR != ret)
    return TaskExecInfo::FAIL;

  return TaskExecInfo::SUCCESS;
}

TaskExecInfo UDPlanar(Surface& src, Surface& dst, NppStreamContext& ctx,
                      Resize Resize_func) {
  for (auto i = 0U; i < src.NumPlanes(); i++) {
    auto ret = Resize_func(src.GetSurfacePlane(i), dst.GetSurfacePlane(i), ctx);
    if (TaskExecInfo::SUCCESS != ret)
      return ret;
  }

  return TaskExecInfo::SUCCESS;
}

TaskExecInfo UDSemiPlanar(Surface& src, Surface& dst, NppStreamContext& ctx,
                          Pixel_Format fmt) {
  UD_NV12(dst.PixelPtr(0U),
          dst.NumComponents() == 3 ? dst.PixelPtr(1U) : (CUdeviceptr)0x0,
          dst.NumComponents() == 3 ? dst.PixelPtr(2U) : (CUdeviceptr)0x0,
          dst.Pitch(), dst.Width(), dst.Height(), src.PixelPtr(0U), src.Pitch(),
          src.Width(), src.Height(), ctx.hStream, fmt);

  return TaskExecInfo::SUCCESS;
}

TaskExecInfo UDSemiPlanarHBD(Surface& src, Surface& dst, NppStreamContext& ctx,
                             Pixel_Format fmt) {
  UD_NV12_HBD(dst.PixelPtr(0U),
              dst.NumComponents() == 3 ? dst.PixelPtr(1U) : (CUdeviceptr)0x0,
              dst.NumComponents() == 3 ? dst.PixelPtr(2U) : (CUdeviceptr)0x0,
              dst.Pitch(), dst.Width(), dst.Height(), src.PixelPtr(0U),
              src.Pitch(), src.Width(), src.Height(), ctx.hStream, fmt);

  return TaskExecInfo::SUCCESS;
}

const std::list<std::pair<Pixel_Format, Pixel_Format>>&
UDSurface::SupportedConversions() const {
  static const std::list<std::pair<Pixel_Format, Pixel_Format>> convs({
      {NV12, YUV444},
      {NV12, RGB},
      {NV12, RGB_32F},
      {NV12, RGB_PLANAR},
      {NV12, RGB_32F_PLANAR},
      {P10, YUV444_10bit},
      {P10, RGB_32F},
      {P10, RGB_32F_PLANAR},
      {P12, RGB_32F},
      {P12, RGB_32F_PLANAR},
      {YUV420, YUV444},
      {YUV420_10bit, YUV444_10bit},
      {YUV422, YUV444},
  });

  return convs;
}

TaskExecDetails UDSurface::Run(Surface& src, Surface& dst) {

  auto& convs = SupportedConversions();
  bool found = false;
  for (auto& conv : convs) {
    if (conv.first == src.PixelFormat() && conv.second == dst.PixelFormat()) {
      found = true;
      break;
    }
  }

  if (!found) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::NOT_SUPPORTED);
  }

  CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));

  TaskExecInfo info = TaskExecInfo::SUCCESS;
  switch (src.PixelFormat()) {
  case NV12:
    info = UDSemiPlanar(src, dst, m_ctx, dst.PixelFormat());
    break;
  case P10:
    info = UDSemiPlanarHBD(src, dst, m_ctx, dst.PixelFormat());
    break;
  case YUV420:
  case YUV422:
    info = UDPlanar(src, dst, m_ctx, Resize_8U_C1);
    break;
  case YUV420_10bit:
    info = UDPlanar(src, dst, m_ctx, Resize_16U_C1);
    break;
  default:
    info = TaskExecInfo::NOT_SUPPORTED;
    break;
  }

  return TaskExecDetails(TaskExecInfo::SUCCESS == info
                             ? TaskExecStatus::TASK_EXEC_SUCCESS
                             : TaskExecStatus::TASK_EXEC_FAIL,
                         info);
}

UDSurface::UDSurface(int gpu_id, CUstream stream)
    : m_gpu_id(gpu_id), m_stream(stream) {
  SetupNppContext(gpu_id, stream, m_ctx);
}