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

#include "MemoryInterfaces.hpp"
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <new>
#include <sstream>
#include <stdexcept>

namespace VPF {

Buffer* Buffer::Make(size_t bufferSize) {
  return new Buffer(bufferSize, false, nullptr);
}

Buffer* Buffer::Make(size_t bufferSize, void* pCopyFrom) {
  return new Buffer(bufferSize, pCopyFrom, false, nullptr);
}

Buffer::Buffer(size_t bufferSize, bool ownMemory, CUcontext ctx)
    : mem_size(bufferSize), own_memory(ownMemory), context(ctx) {
  if (own_memory) {
    if (!Allocate()) {
      throw std::bad_alloc();
    }
  }
}

Buffer::Buffer(size_t bufferSize, void* pCopyFrom, bool ownMemory,
               CUcontext ctx)
    : mem_size(bufferSize), own_memory(ownMemory), context(ctx) {
  if (own_memory) {
    if (Allocate()) {
      memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
    } else {
      throw std::bad_alloc();
    }
  } else {
    pRawData = pCopyFrom;
  }
}

Buffer::Buffer(size_t bufferSize, const void* pCopyFrom, CUcontext ctx)
    : mem_size(bufferSize), own_memory(true), context(ctx) {
  if (Allocate()) {
    memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
  } else {
    throw std::bad_alloc();
  }
}

Buffer::~Buffer() { Deallocate(); }

size_t Buffer::GetRawMemSize() const { return mem_size; }

bool Buffer::Allocate() {
  if (GetRawMemSize()) {
    if (context) {
      CudaCtxPush lock(context);
      CUresult res = cuMemAllocHost(&pRawData, GetRawMemSize());
      ThrowOnCudaError(res, __LINE__);
    } else {
      pRawData = calloc(GetRawMemSize(), sizeof(uint8_t));
    }

    return (nullptr != pRawData);
  }
  return true;
}

void Buffer::Deallocate() {
  if (own_memory) {
    if (context) {
      auto const res = cuMemFreeHost(pRawData);
      ThrowOnCudaError(res, __LINE__);
    } else {
      free(pRawData);
    }
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

Buffer* Buffer::MakeOwnMem(size_t bufferSize, CUcontext ctx) {
  return new Buffer(bufferSize, true, ctx);
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

Buffer* Buffer::MakeOwnMem(size_t bufferSize, const void* pCopyFrom,
                           CUcontext ctx) {
  return new Buffer(bufferSize, pCopyFrom, ctx);
}

CudaBuffer* CudaBuffer::Make(size_t elemSize, size_t numElems,
                             CUcontext context) {
  return new CudaBuffer(elemSize, numElems, context);
}

CudaBuffer* CudaBuffer::Make(const void* ptr, size_t elemSize, size_t numElems,
                             CUcontext context, CUstream str) {
  return new CudaBuffer(ptr, elemSize, numElems, context, str);
}

CudaBuffer* CudaBuffer::Clone() {
  auto pCopy = CudaBuffer::Make(elem_size, num_elems, ctx);

  if (CUDA_SUCCESS !=
      cuMemcpyDtoD(pCopy->GpuMem(), GpuMem(), GetRawMemSize())) {
    delete pCopy;
    return nullptr;
  }

  return pCopy;
}

CudaBuffer::~CudaBuffer() { Deallocate(); }

CudaBuffer::CudaBuffer(size_t elemSize, size_t numElems, CUcontext context) {
  elem_size = elemSize;
  num_elems = numElems;
  ctx = context;

  if (!Allocate()) {
    throw std::bad_alloc();
  }
}

CudaBuffer::CudaBuffer(const void* ptr, size_t elemSize, size_t numElems,
                       CUcontext context, CUstream str) {
  elem_size = elemSize;
  num_elems = numElems;
  ctx = context;

  if (!Allocate()) {
    throw std::bad_alloc();
  }

  CudaCtxPush lock(ctx);
  auto res = cuMemcpyHtoDAsync(gpuMem, ptr, GetRawMemSize(), str);
  ThrowOnCudaError(res, __LINE__);

  res = cuStreamSynchronize(str);
  ThrowOnCudaError(res, __LINE__);
}

bool CudaBuffer::Allocate() {
  if (GetRawMemSize()) {
    CudaCtxPush lock(ctx);
    auto res = cuMemAlloc(&gpuMem, GetRawMemSize());
    ThrowOnCudaError(res, __LINE__);

    if (0U != gpuMem) {
      return true;
    }
  }
  return false;
}

void CudaBuffer::Deallocate() {
  ThrowOnCudaError(cuMemFree(gpuMem), __LINE__);
  gpuMem = 0U;
}

SurfaceY::SurfaceY(uint32_t width, uint32_t height, CUcontext context,
                   bool pitched) {
  auto& ctx = m_planes.at(0);
  ctx.m_plane =
      SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height, CUcontext context,
                         bool pitched) {
  // Y plane;
  {
    auto& ctx = m_planes.at(0);
    ctx.m_plane =
        SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
  }

  // UV plane;
  {
    auto& ctx = m_planes.at(1);
    ctx.factor_y = .5f;
    ctx.m_plane = SurfacePlane(width, height * ctx.factor_y, ElemSize(),
                               kDLUInt, context, pitched);
  }
}

SurfaceYUV420::SurfaceYUV420(uint32_t width, uint32_t height, CUcontext context,
                             bool pitched) {
  // Y plane;
  m_planes.at(0).m_plane =
      SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);

  // U and V planes;
  for (auto i : {1U, 2U}) {
    auto& ctx = m_planes.at(i);
    ctx.factor_x = .5f;
    ctx.factor_y = .5f;
    ctx.m_plane = SurfacePlane(width * ctx.factor_x, height * ctx.factor_y,
                               ElemSize(), kDLUInt, context, pitched);
  }
}

SurfaceYUV422::SurfaceYUV422(uint32_t width, uint32_t height, CUcontext context,
                             bool pitched) {
  // Y plane;
  m_planes.at(0).m_plane =
      SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);

  // U and V planes;
  for (auto i : {1U, 2U}) {
    auto& ctx = m_planes.at(i);
    ctx.factor_x = .5f;
    ctx.m_plane = SurfacePlane(width * ctx.factor_x, height, ElemSize(),
                               kDLUInt, context, pitched);
  }
}

SurfaceRGB::SurfaceRGB(uint32_t width, uint32_t height, CUcontext context,
                       bool pitched) {
  auto& ctx = m_planes.at(0);
  ctx.factor_x = 3.f;
  ctx.m_plane =
      SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
}

SurfaceBGR::SurfaceBGR(uint32_t width, uint32_t height, CUcontext context,
                       bool pitched) {
  auto& ctx = m_planes.at(0);
  ctx.factor_x = 3.f;
  ctx.m_plane =
      SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
}

SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                   CUcontext context, bool pitched) {
  for (auto i : {0U, 1U, 2U}) {
    auto& ctx = m_planes.at(i);
    ctx.m_plane =
        SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
  }
}

SurfaceYUV444::SurfaceYUV444(uint32_t width, uint32_t height, CUcontext context,
                             bool pitched) {
  for (auto i : {0U, 1U, 2U}) {
    auto& ctx = m_planes.at(i);
    ctx.m_plane =
        SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
  }
}

SurfaceRGB32F::SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context,
                             bool pitched) {
  auto& ctx = m_planes.at(0);
  ctx.factor_x = 3.f;
  ctx.m_plane =
      SurfacePlane(width, height, ElemSize(), kDLFloat, context, pitched);
}

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar(uint32_t width, uint32_t height,
                                         CUcontext context, bool pitched) {

  for (auto i : {0U, 1U, 2U}) {
    auto& ctx = m_planes.at(i);
    ctx.m_plane =
        SurfacePlane(width, height, ElemSize(), kDLFloat, context, pitched);
  }
}

SurfaceYUV444_10bit::SurfaceYUV444_10bit(uint32_t width, uint32_t height,
                                         CUcontext context, bool pitched) {
  for (auto i : {0U, 1U, 2U}) {
    auto& ctx = m_planes.at(i);
    ctx.m_plane =
        SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
  }
}

SurfaceP10::SurfaceP10(uint32_t width, uint32_t height, CUcontext context,
                       bool pitched) {
  // Y plane;
  {
    auto& ctx = m_planes.at(0);
    ctx.m_plane =
        SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
  }

  // UV plane;
  {
    auto& ctx = m_planes.at(1);
    ctx.factor_y = .5f;
    ctx.m_plane = SurfacePlane(width, height * ctx.factor_y, ElemSize(),
                               kDLUInt, context, pitched);
  }
}

SurfaceP12::SurfaceP12(uint32_t width, uint32_t height, CUcontext context,
                       bool pitched) {
  // Y plane;
  {
    auto& ctx = m_planes.at(0);
    ctx.m_plane =
        SurfacePlane(width, height, ElemSize(), kDLUInt, context, pitched);
  }

  // UV plane;
  {
    auto& ctx = m_planes.at(1);
    ctx.factor_y = .5f;
    ctx.m_plane = SurfacePlane(width, height * ctx.factor_y, ElemSize(),
                               kDLUInt, context, pitched);
  }
}
} // namespace VPF