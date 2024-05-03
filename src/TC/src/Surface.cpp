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

#include "Surface.hpp"
#include <algorithm>
#include <sstream>

namespace VPF {

std::shared_ptr<Surface> Surface::Make(Pixel_Format format) {
  std::stringstream ss;

  switch (format) {
  case Y:
    return std::make_shared<SurfaceY>();
  case RGB:
    return std::make_shared<SurfaceRGB>();
  case NV12:
    return std::make_shared<SurfaceNV12>();
  case YUV420:
    return std::make_shared<SurfaceYUV420>();
  case RGB_PLANAR:
    return std::make_shared<SurfaceRGBPlanar>();
  case YUV444:
    return std::make_shared<SurfaceYUV444>();
  case YUV444_10bit:
    return std::make_shared<SurfaceYUV444_10bit>();
  case RGB_32F:
    return std::make_shared<SurfaceRGB32F>();
  case RGB_32F_PLANAR:
    return std::make_shared<SurfaceRGB32FPlanar>();
  case YUV422:
    return std::make_shared<SurfaceYUV422>();
  case P10:
    return std::make_shared<SurfaceP10>();
  case P12:
    return std::make_shared<SurfaceP12>();
  case GRAY16:
    return std::make_shared<SurfaceY16>();
  default:
    ss << __FUNCTION__ << ": unsupported pixel format";
    throw std::runtime_error(ss.str());
  }
}

std::shared_ptr<Surface> Surface::Make(Pixel_Format format, uint32_t width,
                                       uint32_t height, CUcontext context) {
  std::stringstream ss;

  switch (format) {
  case Y:
    return std::make_shared<SurfaceY>(width, height, context);
  case NV12:
    return std::make_shared<SurfaceNV12>(width, height, context);
  case YUV420:
    return std::make_shared<SurfaceYUV420>(width, height, context);
  case RGB:
    return std::make_shared<SurfaceRGB>(width, height, context);
  case BGR:
    return std::make_shared<SurfaceBGR>(width, height, context);
  case RGB_PLANAR:
    return std::make_shared<SurfaceRGBPlanar>(width, height, context);
  case YUV444:
    return std::make_shared<SurfaceYUV444>(width, height, context);
  case YUV444_10bit:
    return std::make_shared<SurfaceYUV444_10bit>(width, height, context);
  case RGB_32F:
    return std::make_shared<SurfaceRGB32F>(width, height, context);
  case RGB_32F_PLANAR:
    return std::make_shared<SurfaceRGB32FPlanar>(width, height, context);
  case YUV422:
    return std::make_shared<SurfaceYUV422>(width, height, context);
  case P10:
    return std::make_shared<SurfaceP10>(width, height, context);
  case P12:
    return std::make_shared<SurfaceP12>(width, height, context);
  case GRAY16:
    return std::make_shared<SurfaceY16>(width, height, context);
  default:
    ss << __FUNCTION__ << ": unsupported pixel format";
    throw std::runtime_error(ss.str());
  }
}

Surface::Surface() : m_planes(NumPlanes()) {}

std::shared_ptr<Surface> Surface::Clone(CUstream stream) {
  if (Empty()) {
    return Surface::Make(PixelFormat());
  }

  auto new_surface = Surface::Make(PixelFormat(), Width(), Height(), Context());

  CudaCtxPush ctx_push(Context());
  for (auto plane = 0U; plane < NumPlanes(); plane++) {
    auto src_dptr = PlanePtr(plane);
    auto dst_dptr = new_surface->PlanePtr(plane);

    if (!src_dptr || !dst_dptr) {
      break;
    }

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = src_dptr;
    m.dstDevice = dst_dptr;
    m.srcPitch = Pitch(plane);
    m.dstPitch = new_surface->Pitch(plane);
    m.Height = Height(plane);
    m.WidthInBytes = WidthInBytes(plane);

    if (stream) {
      ThrowOnCudaError(cuMemcpy2DAsync(&m, stream), __LINE__);
    } else {
      ThrowOnCudaError(cuMemcpy2D(&m), __LINE__);
    }
  }

  if (!new_surface->IsValid()) {
    throw std::runtim_error("Surface did not pass validation");
  }

  return new_surface;
}

bool Surface::Empty() const noexcept {
  try {
    for (auto const& entry : m_planes) {
      if (!entry.m_plane.GpuMem()) {
        return true;
      }
    }
    return false;
  } catch (...) {
    return true;
  }
}

bool Surface::OwnMemory() const noexcept {
  try {
    for (auto const& entry : m_planes) {
      if (!entry.m_plane.OwnMemory()) {
        return false;
      }
    }
    return true;
  } catch (...) {
    return false;
  }
}

CUcontext Surface::Context() const noexcept {
  try {
    SurfacePlane plane;
    if (!GetSurfacePlane(plane)) {
      return (CUcontext)0x0;
    }
    return plane.Context();
  } catch (...) {
    return (CUcontext)0x0;
  }
}

size_t Surface::HostMemSize() const noexcept {
  try {
    size_t host_mem_size = 0U;
    for (const auto& entry : m_planes) {
      host_mem_size += entry.m_plane.HostMemSize();
    }
  } catch (...) {
    return 0U;
  }
}

bool Surface::ValidatePlanes() {
  // Checks if all planes have same CUDA context;
  auto cuda_ctx = m_planes.begin().m_plane.Context();
  auto res = std::all_of(m_planes.begin(), m_planes.end(),
                         [](SurfacePlaneContext& ctx) {
                           return ctx.m_plane.Context() == cuda_ctx;
                         });

  if (!res) {
    return false;
  }

  // Checks if all planes have same element size;
  auto elem_size = m_planes.begin()->ElemSize();
  res = std::all_of(m_planes.begin(), m_planes.end(),
                    [](SurfacePlaneContext& ctx) {
                      return ctx.m_plane.ElemSize() == elem_size;
                    });

  if (!res) {
    return false;
  }

  // Checkis if all planes have same element size as required by pixel format;
  if (elem_size != ElemSize()) {
    return false;
  }

  return true;
}

bool Surface::Update(std::initializer_list<SurfacePlane> planes) noexcept {
  if (planes.empty()) {
    return false;
  }

  auto backup = std::move(m_planes);
  try {
    for (auto i = 0U; i < planes.size(); i++) {
      m_planes[i].m_plane = std::move(planes[i]);
    }

    if (!ValidatePlanes()) {
      m_planes = std::move(backup);
      return false;
    }
  } catch (...) {
    m_planes = std::move(backup);
    return false;
  }
}

uint32_t Surface::Width() const noexcept {
  try {
    auto& ctx = m_planes.at(0U);
    return ctx.m_plane.Width() / ctx.m_num_elems_h;
  } catch (...) {
    return 0U;
  }
}

uint32_t Surface::Height() const noexcept {
  try {
    auto& ctx = m_planes.at(0U);
    return ctx.m_plane.Height() / ctx.m_num_elems_v;
  } catch (...) {
    return 0U;
  }
}

bool Surface::GetSurfacePlane(SurfacePlane& plane,
                              uint32_t plane_number) noexcept {
  try {
    plane = m_planes.at(plane_number).m_plane;
    return true;
  } catch (...) {
    return false;
  }
}

bool Surface::IsValid() const noexcept {
  if (!ValidatePlanes()) {
    return false;
  }

  for (auto& ctx : m_planes) {
    if (ctx.m_plane.Width() != Width() * ctx.factor_x) {
      return false;
    }

    if (ctx.m_plane.Height() != Height() * ctx.factor.y) {
      return false;
    }
  }
}

NppContext Surface::GetNppContext() {
  if (!Valid()) {
    throw std::runtime_error("Invalid surface");
  }

  Surface::NppContext ctx;
  ctx.m_dptr.reserve(NumPlanes());
  ctx.m_step.reserve(NumPlanes());

  for (auto i = 0; i < NumPlanes(); i++) {
    SurfacePlane plane;
    if (!pInput->GetSurfacePlane(plane, i)) {
      throw std::runtime_error("Invalid surface");
    }
    ctx.m_dptr[i] = plane.GpuMem();
    ctx.m_step[i] = plane.Pitch();
  }

  m_size.width = Width();
  m_size.height = Height();

  return ctx;
}

SurfacePlane& Surface::GetSurfacePlane(uint32_t plane_number = 0U) {
  return m_planes.at(plane_number).m_plane;
}

void Surface::FromChunk2D(CUdeviceptr src, CUstream str, size_t src_pitch,
                          bool async) {
  res = std::all_of(m_planes.begin(), m_planes.end(),
                    [](SurfacePlaneContext& plane_ctx) {
                      return plane_ctx.m_plane.Pitch() == src_pitch;
                    });

  if (!res) {
    throw std::runtime_error("Surface have planes with different pitches.");
  }

  CudaCtxPush ctxPush(plane.Context());
  for (auto i = 0U; i < NumPlanes(); i++) {
    SurfacePlane plane;
    if (!GetSurfacePlane(plane, i)) {
      throw std::runtime_error("Failed to get SurfacePlane.");
    }

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = src;
    m.srcPitch = src_pitch;

    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = plane.GpuMem();
    m.dstPitch = plane.Pitch();

    m.Height = plane.Height();
    m.WidthInBytes = plane.Width() * plane.ElemSize();

    ThrowOnCudaError(cuMemcpy2DAsync(&m, str), __LINE__);
    dpSrcFrame += m.srcPitch * m.Height;
  }

  if (!async) {
    ThrowOnCudaError(cuStreamSynchronize(str), __LINE__);
  }
}

void Surface::ToChunk2D(CUdeviceptr dst, CUstream str, size_t dst_pitch,
                        bool async = false) const {
  res = std::all_of(m_planes.begin(), m_planes.end(),
                    [](SurfacePlaneContext& plane_ctx) {
                      return plane_ctx.m_plane.Pitch() == dst_pitch;
                    });

  if (!res) {
    throw std::runtime_error("Surface have planes with different pitches.");
  }

  CudaCtxPush ctxPush(plane.Context());
  for (auto i = 0U; i < NumPlanes(); i++) {
    SurfacePlane plane;
    if (!GetSurfacePlane(plane, i)) {
      throw std::runtime_error("Failed to get SurfacePlane.");
    }

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = plane.GpuMem();
    m.srcPitch = plane.Pitch();

    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = dst;
    m.dstPitch = dst_pitch;

    m.Height = plane.Height();
    m.WidthInBytes = plane.Width() * plane.ElemSize();

    ThrowOnCudaError(cuMemcpy2DAsync(&m, str), __LINE__);
    dpSrcFrame += m.srcPitch * m.Height;
  }

  if (!async) {
    ThrowOnCudaError(cuStreamSynchronize(str), __LINE__);
  }
}

} // namespace VPF