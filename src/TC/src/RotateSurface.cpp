/*
 * Copyright 2025 Vision Labs LLC
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

#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "Surfaces.hpp"
#include "Tasks.hpp"

using namespace VPF;

/// @brief 8 bit unsigned single channel
TaskExecInfo Rot_8U_C1(double angle, double shift_x, double shift_y,
                       uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                       uint32_t dst_h, SurfacePlane& src, SurfacePlane& dst,
                       NppStreamContext& ctx) {
  NppiSize oSrcSize = {src_w, src_h};
  NppiRect oSrcROI = {0, 0, src_w, src_h};
  NppiRect oDstROI = {0, 0, dst_w, dst_h};

  auto ret = LibNpp::nppiRotate_8u_C1R_Ctx(
      (const Npp8u*)src.GpuMem(), oSrcSize, src.Pitch(), oSrcROI,
      (Npp8u*)dst.GpuMem(), dst.Pitch(), oDstROI, angle, shift_x, shift_y,
      NPPI_INTER_LANCZOS, ctx);

  return (NPP_SUCCESS == ret) ? TaskExecInfo::SUCCESS : TaskExecInfo::FAIL;
}

/// @brief 16 bit unsigned single channel
TaskExecInfo Rot_16U_C1(double angle, double shift_x, double shift_y,
                        uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                        uint32_t dst_h, SurfacePlane& src, SurfacePlane& dst,
                        NppStreamContext& ctx) {
  NppiSize oSrcSize = {src_w, src_h};
  NppiRect oSrcROI = {0, 0, src_w, src_h};
  NppiRect oDstROI = {0, 0, dst_w, dst_h};

  auto ret = LibNpp::nppiRotate_16u_C1R_Ctx(
      (const Npp16u*)src.GpuMem(), oSrcSize, src.Pitch(), oSrcROI,
      (Npp16u*)dst.GpuMem(), dst.Pitch(), oDstROI, angle, shift_x, shift_y,
      NPPI_INTER_LANCZOS, ctx);

  return (NPP_SUCCESS == ret) ? TaskExecInfo::SUCCESS : TaskExecInfo::FAIL;
}

/// @brief 32 bit float single channel
TaskExecInfo Rot_32F_C1(double angle, double shift_x, double shift_y,
                        uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                        uint32_t dst_h, SurfacePlane& src, SurfacePlane& dst,
                        NppStreamContext& ctx) {
  NppiSize oSrcSize = {src_w, src_h};
  NppiRect oSrcROI = {0, 0, src_w, src_h};
  NppiRect oDstROI = {0, 0, dst_w, dst_h};

  auto ret = LibNpp::nppiRotate_32f_C1R_Ctx(
      (const Npp32f*)src.GpuMem(), oSrcSize, src.Pitch(), oSrcROI,
      (Npp32f*)dst.GpuMem(), dst.Pitch(), oDstROI, angle, shift_x, shift_y,
      NPPI_INTER_LANCZOS, ctx);

  return (NPP_SUCCESS == ret) ? TaskExecInfo::SUCCESS : TaskExecInfo::FAIL;
}

/// @brief 8 bit unsigned 3 channels
TaskExecInfo Rot_8U_C3(double angle, double shift_x, double shift_y,
                       uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                       uint32_t dst_h, SurfacePlane& src, SurfacePlane& dst,
                       NppStreamContext& ctx) {

  NppiSize oSrcSize = {src_w, src_h};
  NppiRect oSrcROI = {0, 0, src_w, src_h};
  NppiRect oDstROI = {0, 0, dst_w, dst_h};

  auto ret = LibNpp::nppiRotate_8u_C3R_Ctx(
      (const Npp8u*)src.GpuMem(), oSrcSize, src.Pitch(), oSrcROI,
      (Npp8u*)dst.GpuMem(), dst.Pitch(), oDstROI, angle, shift_x, shift_y,
      NPPI_INTER_LINEAR, ctx);

  return (NPP_SUCCESS == ret) ? TaskExecInfo::SUCCESS : TaskExecInfo::FAIL;
}

/// @brief 16 bit unsigned 3 channels
TaskExecInfo Rot_16U_C3(double angle, double shift_x, double shift_y,
                        uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                        uint32_t dst_h, SurfacePlane& src, SurfacePlane& dst,
                        NppStreamContext& ctx) {

  NppiSize oSrcSize = {src_w, src_h};
  NppiRect oSrcROI = {0, 0, src_w, src_h};
  NppiRect oDstROI = {0, 0, dst_w, dst_h};

  auto ret = LibNpp::nppiRotate_16u_C3R_Ctx(
      (const Npp16u*)src.GpuMem(), oSrcSize, src.Pitch(), oSrcROI,
      (Npp16u*)dst.GpuMem(), dst.Pitch(), oDstROI, angle, shift_x, shift_y,
      NPPI_INTER_LINEAR, ctx);

  return (NPP_SUCCESS == ret) ? TaskExecInfo::SUCCESS : TaskExecInfo::FAIL;
}

/// @brief 32 bit float 3 channels
TaskExecInfo Rot_32F_C3(double angle, double shift_x, double shift_y,
                        uint32_t src_w, uint32_t src_h, uint32_t dst_w,
                        uint32_t dst_h, SurfacePlane& src, SurfacePlane& dst,
                        NppStreamContext& ctx) {

  NppiSize oSrcSize = {src_w, src_h};
  NppiRect oSrcROI = {0, 0, src_w, src_h};
  NppiRect oDstROI = {0, 0, dst_w, dst_h};

  auto ret = LibNpp::nppiRotate_32f_C3R_Ctx(
      (const Npp32f*)src.GpuMem(), oSrcSize, src.Pitch(), oSrcROI,
      (Npp32f*)dst.GpuMem(), dst.Pitch(), oDstROI, angle, shift_x, shift_y,
      NPPI_INTER_LINEAR, ctx);

  return (NPP_SUCCESS == ret) ? TaskExecInfo::SUCCESS : TaskExecInfo::FAIL;
}

typedef TaskExecInfo (*RotateImpl)(double angle, double shift_x, double shift_y,
                                   uint32_t src_w, uint32_t src_h,
                                   uint32_t dst_w, uint32_t dst_h,
                                   SurfacePlane& src, SurfacePlane& dst,
                                   NppStreamContext& ctx);

/// @brief Rotate planar Surface implementation
TaskExecInfo RotPlanar(double angle, double shift_x, double shift_y,
                       Surface& src, Surface& dst, NppStreamContext& ctx,
                       RotateImpl impl_func) {
  if (src.NumComponents() != src.NumPlanes())
    return TaskExecInfo::INVALID_INPUT;

  for (auto i = 0U; i < src.NumPlanes(); i++) {
    auto ret = impl_func(angle, shift_x, shift_y, src.Width(i), src.Height(i),
                         dst.Width(i), dst.Height(i), src.GetSurfacePlane(i),
                         dst.GetSurfacePlane(i), ctx);
    if (TaskExecInfo::SUCCESS != ret)
      return ret;
  }

  return TaskExecInfo::SUCCESS;
}

/// @brief Rotate packed Surface implementation
TaskExecInfo RotPacked(double angle, double shift_x, double shift_y,
                       Surface& src, Surface& dst, NppStreamContext& ctx,
                       RotateImpl impl_func) {
  if (1U != src.NumPlanes())
    return TaskExecInfo::INVALID_INPUT;

  return impl_func(angle, shift_x, shift_y, src.Width(), src.Height(),
                   dst.Width(), dst.Height(), src.GetSurfacePlane(),
                   dst.GetSurfacePlane(), ctx);
}

TaskExecDetails RotateSurface::Run(double angle, double shift_x, double shift_y,
                                   Surface& src, Surface& dst) {
  if (src.PixelFormat() != dst.PixelFormat())
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::SRC_DST_FMT_MISMATCH);

  TaskExecInfo info = TaskExecInfo::SUCCESS;
  switch (src.PixelFormat()) {
  case Y:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_8U_C1);
    break;
  case GRAY12:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_16U_C1);
    break;
  case RGB:
    info = RotPacked(angle, shift_x, shift_y, src, dst, m_ctx, Rot_8U_C3);
    break;
  case BGR:
    info = RotPacked(angle, shift_x, shift_y, src, dst, m_ctx, Rot_8U_C3);
    break;
  case RGB_PLANAR:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_8U_C1);
    break;
  case YUV420:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_8U_C1);
    break;
  case YUV422:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_8U_C1);
    break;
  case YUV444:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_8U_C1);
    break;
  case RGB_32F:
    info = RotPacked(angle, shift_x, shift_y, src, dst, m_ctx, Rot_32F_C3);
    break;
  case RGB_32F_PLANAR:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_32F_C1);
    break;
  case YUV444_10bit:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_16U_C1);
    break;
  case YUV420_10bit:
    info = RotPlanar(angle, shift_x, shift_y, src, dst, m_ctx, Rot_16U_C1);
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

RotateSurface::RotateSurface(int gpu_id, CUstream stream) : m_stream(stream) {
  SetupNppContext(gpu_id, stream, m_ctx);
}