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

/// @brief Planar implementation
/// @param src input surface
/// @param dst output surface
/// @param ctx NPP stream context
/// @param Resize_func resize function pointer
/// @return SUCCESS in case of success, FAIL otherwise
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
                          Resize Resize_func) {

  UD_NV12((unsigned char*)dst.GetSurfacePlane(0U).GpuMem(),
          (unsigned char*)dst.GetSurfacePlane(1U).GpuMem(),
          (unsigned char*)dst.GetSurfacePlane(2U).GpuMem(), dst.Pitch(),
          dst.Width(), dst.Height(),
          (unsigned char*)src.GetSurfacePlane(0).GpuMem(), src.Pitch(),
          src.Width(), src.Height(), ctx.hStream);

  return TaskExecInfo::SUCCESS;
}

TaskExecDetails UDSurface::Run(Surface& src, Surface& dst) {

  // Can only output to YUV444 of various bit depths
  if (YUV444 != dst.PixelFormat() && YUV444_10bit != dst.PixelFormat())
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::NOT_SUPPORTED);

  // No bit depth conversion
  if (src.ElemSize() != dst.ElemSize())
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::SRC_DST_FMT_MISMATCH);

  // Can push the context just once
  CudaCtxPush ctxPush(GetContextByStream(m_gpu_id, m_stream));

  TaskExecInfo info = TaskExecInfo::SUCCESS;
  switch (src.PixelFormat()) {
  case NV12:
    // Uses CUDA kernel, not need to pass implementation function pointer
    info = UDSemiPlanar(src, dst, m_ctx, nullptr);
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