/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Videonetics Technology Private Limited
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

#include "Surfaces.hpp"
#include "Utils.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <new>
#include <sstream>
#include <stdexcept>

using namespace VPF;
using namespace std;

Buffer* Buffer::Make(size_t bufferSize) {
  return new Buffer(bufferSize, false);
}

Buffer* Buffer::Make(size_t bufferSize, void* pCopyFrom) {
  return new Buffer(bufferSize, pCopyFrom, false);
}

Buffer::Buffer(size_t bufferSize, bool ownMemory)
    : mem_size(bufferSize), own_memory(ownMemory) {
  if (own_memory) {
    if (!Allocate()) {
      throw bad_alloc();
    }
  }
}

Buffer::Buffer(size_t bufferSize, void* pCopyFrom, bool ownMemory)
    : mem_size(bufferSize), own_memory(ownMemory) {
  if (own_memory) {
    if (Allocate()) {
      memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
    } else {
      throw bad_alloc();
    }
  } else {
    pRawData = pCopyFrom;
  }
}

Buffer::Buffer(size_t bufferSize, const void* pCopyFrom)
    : mem_size(bufferSize), own_memory(true) {
  if (Allocate()) {
    memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
  } else {
    throw bad_alloc();
  }
}

Buffer::~Buffer() { Deallocate(); }

size_t Buffer::GetRawMemSize() const { return mem_size; }

bool Buffer::Allocate() {
  if (GetRawMemSize()) {
    pRawData = calloc(GetRawMemSize(), sizeof(uint8_t));
    return (nullptr != pRawData);
  }
  return true;
}

void Buffer::Deallocate() {
  if (own_memory) {
    free(pRawData);
  }
  pRawData = nullptr;
}

void* Buffer::GetRawMemPtr() { return pRawData; }

const void* Buffer::GetRawMemPtr() const { return pRawData; }

void Buffer::Update(size_t newSize, void* newPtr) {
  Deallocate();

  mem_size = newSize;
  if (own_memory) {
    Allocate();
    if (newPtr) {
      memcpy(GetRawMemPtr(), newPtr, newSize);
    }
  } else {
    pRawData = newPtr;
  }
}

Buffer* Buffer::MakeOwnMem(size_t bufferSize) {
  return new Buffer(bufferSize, true);
}

bool Buffer::CopyFrom(size_t size, void const* ptr) {

  if (mem_size != size) {
    return false;
  }

  if (!ptr) {
    return false;
  }

  memcpy(GetRawMemPtr(), ptr, size);
  return true;
}

Buffer* Buffer::MakeOwnMem(size_t bufferSize, const void* pCopyFrom) {
  return new Buffer(bufferSize, pCopyFrom);
}

Surface* Surface::Make(Pixel_Format format) {
  switch (format) {
  case Y:
    return new SurfaceY;
  case RGB:
    return new SurfaceRGB;
  case NV12:
    return new SurfaceNV12;
  case YUV420:
    return new SurfaceYUV420;
  case RGB_PLANAR:
    return new SurfaceRGBPlanar;
  case YUV444:
    return new SurfaceYUV444;
  case YUV444_10bit:
    return new SurfaceYUV444_10bit;
  case RGB_32F:
    return new SurfaceRGB32F;
  case RGB_32F_PLANAR:
    return new SurfaceRGB32FPlanar;
  case YUV422:
    return new SurfaceYUV422;
  case P10:
    return new SurfaceP10;
  case YUV420_10bit:
  case P12:
    return new SurfaceP12;
  default:
    av_log(nullptr, AV_LOG_ERROR, "Unsupported pixeld format: %s \n",
           GetFormatName(format).c_str());
    return nullptr;
  }
}

Surface* Surface::Make(Pixel_Format format, uint32_t newWidth,
                       uint32_t newHeight, CUcontext context) {
  switch (format) {
  case Y:
    return new SurfaceY(newWidth, newHeight, context);
  case NV12:
    return new SurfaceNV12(newWidth, newHeight, context);
  case YUV420:
    return new SurfaceYUV420(newWidth, newHeight, context);
  case RGB:
    return new SurfaceRGB(newWidth, newHeight, context);
  case BGR:
    return new SurfaceBGR(newWidth, newHeight, context);
  case RGB_PLANAR:
    return new SurfaceRGBPlanar(newWidth, newHeight, context);
  case YUV444:
    return new SurfaceYUV444(newWidth, newHeight, context);
  case YUV444_10bit:
    return new SurfaceYUV444_10bit(newWidth, newHeight, context);
  case RGB_32F:
    return new SurfaceRGB32F(newWidth, newHeight, context);
  case RGB_32F_PLANAR:
    return new SurfaceRGB32FPlanar(newWidth, newHeight, context);
  case YUV422:
    return new SurfaceYUV422(newWidth, newHeight, context);
  case P10:
    return new SurfaceP10(newWidth, newHeight, context);
  case YUV420_10bit:
    return new SurfaceYUV420_10bit(newWidth, newHeight, context);
  case P12:
    return new SurfaceP12(newWidth, newHeight, context);
  default:
    av_log(nullptr, AV_LOG_ERROR, "Unsupported pixeld format: %s \n",
           GetFormatName(format).c_str());
    return nullptr;
  }
}

Surface* Surface::Clone() {
  if (Empty()) {
    return Surface::Make(PixelFormat());
  }

  auto newSurf = Surface::Make(PixelFormat(), Width(), Height(), Context());

  for (auto i = 0U; i < NumPlanes(); i++) {
    auto src = GetSurfacePlane(i);
    auto dst = newSurf->GetSurfacePlane(i);
    CudaCtxPush ctxPush(GetContextByDptr(src.GpuMem()));

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = src.GpuMem();
    m.dstDevice = dst.GpuMem();
    m.srcPitch = src.Pitch();
    m.dstPitch = dst.Pitch();
    m.Height = src.Height();
    m.WidthInBytes = src.Width() * src.ElemSize();

    ThrowOnCudaError(LibCuda::cuMemcpy2DAsync(&m, 0), __LINE__);
  }

  CudaStrSync sync(0);
  return newSurf;
}

bool Surface::OwnMemory() {
  bool res = true;
  for (int i = 0; i < NumPlanes(); i++) {
    if (!GetSurfacePlane(i).OwnMemory()) {
      res = false;
    }
  }
  return res;
}

uint32_t Surface::HostMemSize() const {
  auto size = 0U;
  for (auto& plane : m_planes) {
    size += plane.HostMemSize();
  }

  return size;
};

bool Surface::Empty() const {
  return std::all_of(m_planes.cbegin(), m_planes.cend(),
                     [&](const SurfacePlane& plane) { return plane.Empty(); });
}

CUcontext Surface::Context() { return GetSurfacePlane().Context(); }

std::vector<size_t> Surface::Shape() {
  std::vector<size_t> shape;

  try {
    CudaArrayInterfaceDescriptor cai;
    ToCAI(cai);

    for (auto i = 0U; i < cai.m_num_elems; i++) {
      auto& dim = cai.m_shape[i];
      if (dim) {
        shape.push_back(dim);
      }
    }
  } catch (...) {
    shape.push_back(HostMemSize() / ElemSize());
  }

  return shape;
}