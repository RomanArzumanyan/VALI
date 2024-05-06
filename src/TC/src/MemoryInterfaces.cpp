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
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <new>
#include <sstream>
#include <stdexcept>

using namespace VPF;
using namespace std;

#ifdef TRACK_TOKEN_ALLOCATIONS
#include <atomic>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace VPF {

struct AllocInfo {
  uint64_t id;
  uint64_t size;

  bool operator==(const AllocInfo& other) {
    /* Buffer size may change during the lifetime so we check id only;
     */
    return id == other.id;
  }

  explicit AllocInfo(decltype(id) const& newId, decltype(size) const& newSize)
      : id(newId), size(newSize) {}
};

struct AllocRegister {
  vector<AllocInfo> instances;
  mutex guard;
  uint64_t ID = 1U;

  decltype(AllocInfo::id) AddNote(decltype(AllocInfo::size) const& size) {
    unique_lock<decltype(guard)> lock;
    auto id = ID++;
    AllocInfo info(id, size);
    instances.push_back(info);
    return id;
  }

  void DeleteNote(AllocInfo const& allocInfo) {
    unique_lock<decltype(guard)> lock;
    instances.erase(remove(instances.begin(), instances.end(), allocInfo),
                    instances.end());
  }

  /* Call this after you're done releasing mem objects in your app;
   */
  size_t GetSize() const { return instances.size(); }

  /* Call this after you're done releasing mem objects in your app;
   */
  AllocInfo const* GetNoteByIndex(uint64_t idx) {
    return idx < instances.size() ? instances.data() + idx : nullptr;
  }
};

AllocRegister BuffersRegister, HWSurfaceRegister, CudaBuffersRegister;

bool CheckAllocationCounters() {
  auto numLeakedBuffers = BuffersRegister.GetSize();
  auto numLeakedSurfaces = HWSurfaceRegister.GetSize();
  auto numLeakedCudaBuffers = CudaBuffersRegister.GetSize();

  if (numLeakedBuffers) {
    cerr << "Leaked buffers (id : size): " << endl;
    for (auto i = 0; i < numLeakedBuffers; i++) {
      auto pNote = BuffersRegister.GetNoteByIndex(i);
      cerr << "\t" << pNote->id << "\t: " << pNote->size << endl;
    }
  }

  if (numLeakedSurfaces) {
    cerr << "Leaked surfaces (id : size): " << endl;
    for (auto i = 0; i < numLeakedSurfaces; i++) {
      auto pNote = HWSurfaceRegister.GetNoteByIndex(i);
      cerr << "\t" << pNote->id << "\t: " << pNote->size << endl;
    }
  }

  if (numLeakedCudaBuffers) {
    cerr << "Leaked CUDA buffers (id : size): " << endl;
    for (auto i = 0; i < numLeakedCudaBuffers; i++) {
      auto pNote = CudaBuffersRegister.GetNoteByIndex(i);
      cerr << "\t" << pNote->id << "\t: " << pNote->size << endl;
    }
  }

  return (0U == numLeakedBuffers) && (0U == numLeakedSurfaces) &&
         (0U == numLeakedCudaBuffers);
}

} // namespace VPF
#endif

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
      throw bad_alloc();
    }
  }
#ifdef TRACK_TOKEN_ALLOCATIONS
  id = BuffersRegister.AddNote(mem_size);
#endif
}

Buffer::Buffer(size_t bufferSize, void* pCopyFrom, bool ownMemory,
               CUcontext ctx)
    : mem_size(bufferSize), own_memory(ownMemory), context(ctx) {
  if (own_memory) {
    if (Allocate()) {
      memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
    } else {
      throw bad_alloc();
    }
  } else {
    pRawData = pCopyFrom;
  }
#ifdef TRACK_TOKEN_ALLOCATIONS
  id = BuffersRegister.AddNote(mem_size);
#endif
}

Buffer::Buffer(size_t bufferSize, const void* pCopyFrom, CUcontext ctx)
    : mem_size(bufferSize), own_memory(true), context(ctx) {
  if (Allocate()) {
    memcpy(this->GetRawMemPtr(), pCopyFrom, bufferSize);
  } else {
    throw bad_alloc();
  }
#ifdef TRACK_TOKEN_ALLOCATIONS
  id = BuffersRegister.AddNote(mem_size);
#endif
}

Buffer::~Buffer() {
  Deallocate();
#ifdef TRACK_TOKEN_ALLOCATIONS
  AllocInfo info(id, mem_size);
  BuffersRegister.DeleteNote(info);
#endif
}

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
    throw bad_alloc();
  }
}

CudaBuffer::CudaBuffer(const void* ptr, size_t elemSize, size_t numElems,
                       CUcontext context, CUstream str) {
  elem_size = elemSize;
  num_elems = numElems;
  ctx = context;

  if (!Allocate()) {
    throw bad_alloc();
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
#ifdef TRACK_TOKEN_ALLOCATIONS
      id = CudaBuffersRegister.AddNote(GetRawMemSize());
#endif
      return true;
    }
  }
  return false;
}

void CudaBuffer::Deallocate() {
  ThrowOnCudaError(cuMemFree(gpuMem), __LINE__);
  gpuMem = 0U;

#ifdef TRACK_TOKEN_ALLOCATIONS
  AllocInfo info(id, GetRawMemSize());
  CudaBuffersRegister.DeleteNote(info);
#endif
}

Surface::Surface() = default;

Surface::~Surface() = default;

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
    cerr << __FUNCTION__ << "Unsupported pixeld format: " << format << endl;
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
  case P12:
    return new SurfaceP12(newWidth, newHeight, context);
  default:
    cerr << __FUNCTION__ << "Unsupported pixeld format: " << format << endl;
    return nullptr;
  }
}

Surface* Surface::Clone() {
  if (Empty()) {
    return Surface::Make(PixelFormat());
  }

  auto newSurf = Surface::Make(PixelFormat(), Width(), Height(), Context());

  CudaCtxPush ctxPush(Context());
  for (auto plane = 0U; plane < NumPlanes(); plane++) {
    auto srcPlanePtr = PlanePtr(plane);
    auto dstPlanePtr = newSurf->PlanePtr(plane);

    if (!srcPlanePtr || !dstPlanePtr) {
      break;
    }

    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = srcPlanePtr;
    m.dstDevice = dstPlanePtr;
    m.srcPitch = Pitch(plane);
    m.dstPitch = newSurf->Pitch(plane);
    m.Height = Height(plane);
    m.WidthInBytes = WidthInBytes(plane);

    ThrowOnCudaError(cuMemcpy2D(&m), __LINE__);
  }

  return newSurf;
}

bool Surface::OwnMemory() {
  bool res = true;
  for (int i = 0; i < NumPlanes() && GetSurfacePlane(i); i++) {
    if (!GetSurfacePlane(i)->OwnMemory()) {
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

CUcontext Surface::Context() { return GetSurfacePlane()->Context(); }