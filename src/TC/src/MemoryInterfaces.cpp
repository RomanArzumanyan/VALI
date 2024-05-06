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

static bool ValidatePlanes(SurfacePlane** planes, size_t planes_num,
                           size_t surface_elem_size,
                           size_t surface_num_planes) {
  if (!planes) {
    return false;
  }

  if (planes_num != surface_num_planes) {
    return false;
  }

  const auto elem_size = planes[0]->ElemSize();
  if (surface_elem_size != elem_size) {
    return false;
  }

  for (auto i = 0U; i < planes_num; i++) {
    if (!planes[i]) {
      return false;
    }

    if (planes[i]->ElemSize() != elem_size) {
      return false;
    }
  }

  return true;
}

Surface* SurfaceY::Create() { return new SurfaceY; }

SurfaceY::SurfaceY() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceY::SurfaceY(uint32_t width, uint32_t height, CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width, height, ElemSize(), DataType(), context);
}

uint32_t SurfaceY::Width(uint32_t plane) const {
  return m_planes.at(plane).Width();
}

uint32_t SurfaceY::WidthInBytes(uint32_t plane) const {
  return m_planes.at(plane).ElemSize();
}

uint32_t SurfaceY::Height(uint32_t plane) const {
  return m_planes.at(plane).Height();
}

uint32_t SurfaceY::Pitch(uint32_t plane) const {
  return m_planes.at(plane).Pitch();
}

CUdeviceptr SurfaceY::PlanePtr(uint32_t plane) {
  return m_planes.at(plane).GpuMem();
}

SurfacePlane* SurfaceY::GetSurfacePlane(uint32_t plane) {
  return &m_planes.at(plane);
}

bool SurfaceY::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1U);
}

bool SurfaceY::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!OwnMemory()) {
    m_planes.at(0U) = *pPlanes[0];
    return true;
  }

  return false;
}

SurfaceNV12::SurfaceNV12() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height, CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width, height * 3 / 2, ElemSize(), DataType(), context);
}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height, uint32_t pitch,
                         CUdeviceptr ptr) {
  auto dlmt_ptr = SurfacePlane::DLPackContext::ToDLPack(
      width, height * 3 / 2, pitch, ElemSize(), ptr, DataType());

  auto dlmt_smart_ptr =
      std::shared_ptr<DLManagedTensor>(dlmt_ptr, dlmt_ptr->deleter);

  m_planes.clear();
  m_planes.emplace_back(*dlmt_smart_ptr.get());
}

Surface* SurfaceNV12::Create() { return new SurfaceNV12; }

uint32_t SurfaceNV12::Width(uint32_t plane) const {
  switch (plane) {
  case 0:
  case 1:
    return m_planes.at(0U).Width();
  default:
    break;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceNV12::WidthInBytes(uint32_t plane) const {
  return Width(plane) * ElemSize();
}

uint32_t SurfaceNV12::Height(uint32_t plane) const {
  switch (plane) {
  case 0:
    return m_planes.at(0U).Height() * 2 / 3;
  case 1:
    return m_planes.at(0U).Height() / 3;
  default:
    break;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceNV12::Pitch(uint32_t plane) const {
  switch (plane) {
  case 0:
  case 1:
    return m_planes.at(0U).Pitch();
  default:
    break;
  }

  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceNV12::PlanePtr(uint32_t plane) {
  if (Empty()) {
    return 0x0;
  }

  if (plane < NumPlanes()) {
    return m_planes.at(0U).GpuMem() + plane * Height() * Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceNV12::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1U);
}

bool SurfaceNV12::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!OwnMemory()) {
    m_planes.at(0U) = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceNV12::GetSurfacePlane(uint32_t plane) {
  if (plane < NumPlanes()) {
    return &m_planes.at(0U);
  }

  return nullptr;
}

SurfaceYUV420::SurfaceYUV420() {
  m_planes.clear();
  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.emplace_back();
  }
}

SurfaceYUV420::SurfaceYUV420(uint32_t width, uint32_t height,
                             CUcontext context) {
  m_planes.clear();
  /* Need to reserve place, otherwise vector may reallocate and SurfacePlane
   * instances will be copied to new address loosing the memory ownership. Sic!
   */
  m_planes.reserve(NumPlanes());
  // Y plane, full size;
  m_planes.emplace_back(width, height, ElemSize(), DataType(), context);
  // U and V planes, decimated size;
  m_planes.emplace_back(width / 2, height / 2, ElemSize(), DataType(), context);
  m_planes.emplace_back(width / 2, height / 2, ElemSize(), DataType(), context);
}

Surface* SurfaceYUV420::Create() { return new SurfaceYUV420; }

uint32_t SurfaceYUV420::Width(uint32_t plane) const {
  return m_planes.at(plane).Width();
}

uint32_t SurfaceYUV420::WidthInBytes(uint32_t plane) const {
  return Width(plane) * ElemSize();
}

uint32_t SurfaceYUV420::Height(uint32_t plane) const {
  return m_planes.at(plane).Height();
}

uint32_t SurfaceYUV420::Pitch(uint32_t plane) const {
  return m_planes.at(plane).Pitch();
}

CUdeviceptr SurfaceYUV420::PlanePtr(uint32_t plane) {
  return m_planes.at(plane).GpuMem();
}

bool SurfaceYUV420::Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
                           SurfacePlane& newPlaneV) {
  SurfacePlane* planes[] = {&newPlaneY, &newPlaneU, &newPlaneV};
  return Update(planes, 3);
}

bool SurfaceYUV420::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (OwnMemory() || !ValidatePlanes(pPlanes, planesNum, ElemSize(), 3)) {
    return false;
  }

  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.at(i) = *pPlanes[i];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceYUV420::GetSurfacePlane(uint32_t plane) {
  try {
    return &m_planes.at(plane);
  } catch (...) {
    return nullptr;
  }
}

SurfaceYUV422::SurfaceYUV422() {
  m_planes.clear();
  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.emplace_back();
  }
}

SurfaceYUV422::SurfaceYUV422(uint32_t width, uint32_t height,
                             CUcontext context) {
  m_planes.clear();
  /* Need to reserve place, otherwise vector may reallocate and SurfacePlane
   * instances will be copied to new address loosing the memory ownership. Sic!
   */
  m_planes.reserve(NumPlanes());
  // Y plane, full size;
  m_planes.emplace_back(width, height, ElemSize(), DataType(), context);
  // U and V planes, decimated size;
  m_planes.emplace_back(width / 2, height, ElemSize(), DataType(), context);
  m_planes.emplace_back(width / 2, height, ElemSize(), DataType(), context);
}

Surface* SurfaceYUV422::Create() { return new SurfaceYUV422; }

uint32_t SurfaceYUV422::Width(uint32_t plane) const {
  return m_planes.at(plane).Width();
}

uint32_t SurfaceYUV422::WidthInBytes(uint32_t plane) const {
  return Width(plane) * ElemSize();
}

uint32_t SurfaceYUV422::Height(uint32_t plane) const {
  return m_planes.at(plane).Height();
}

uint32_t SurfaceYUV422::Pitch(uint32_t plane) const {
  return m_planes.at(plane).Pitch();
}

CUdeviceptr SurfaceYUV422::PlanePtr(uint32_t plane) {
  return m_planes.at(plane).GpuMem();
}

SurfacePlane* SurfaceYUV422::GetSurfacePlane(uint32_t plane) {
  try {
    return &m_planes.at(plane);
  } catch (...) {
    return nullptr;
  }
}

bool SurfaceYUV422::Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
                           SurfacePlane& newPlaneV) {
  SurfacePlane* planes[] = {&newPlaneY, &newPlaneU, &newPlaneV};
  return Update(planes, 3);
}

bool SurfaceYUV422::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (OwnMemory() || !ValidatePlanes(pPlanes, planesNum, ElemSize(), 3)) {
    return false;
  }

  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.at(i) = *pPlanes[i];
    return true;
  }

  return false;
}

SurfaceYUV444::SurfaceYUV444() {
  m_planes.clear();
  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.emplace_back();
  }
}

SurfaceYUV444::SurfaceYUV444(uint32_t width, uint32_t height,
                             CUcontext context) {
  m_planes.clear();
  /* Need to reserve place, otherwise vector may reallocate and SurfacePlane
   * instances will be copied to new address loosing the memory ownership. Sic!
   */
  m_planes.reserve(NumPlanes());
  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.emplace_back(width, height, ElemSize(), DataType(), context);
  }
}

Surface* SurfaceYUV444::Create() { return new SurfaceYUV444; }

uint32_t SurfaceYUV444::Width(uint32_t plane) const {
  return m_planes.at(plane).Width();
}

uint32_t SurfaceYUV444::WidthInBytes(uint32_t plane) const {
  return Width(plane) * ElemSize();
}

uint32_t SurfaceYUV444::Height(uint32_t plane) const {
  return m_planes.at(plane).Height();
}

uint32_t SurfaceYUV444::Pitch(uint32_t plane) const {
  return m_planes.at(plane).Pitch();
}

CUdeviceptr SurfaceYUV444::PlanePtr(uint32_t plane) {
  return m_planes.at(plane).GpuMem();
}

SurfacePlane* SurfaceYUV444::GetSurfacePlane(uint32_t plane) {
  try {
    return &m_planes.at(plane);
  } catch (...) {
    return nullptr;
  }
}

bool SurfaceYUV444::Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
                           SurfacePlane& newPlaneV) {
  SurfacePlane* planes[] = {&newPlaneY, &newPlaneU, &newPlaneV};
  return Update(planes, 3);
}

bool SurfaceYUV444::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (OwnMemory() || !ValidatePlanes(pPlanes, planesNum, ElemSize(), 3)) {
    return false;
  }

  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.at(i) = *pPlanes[i];
    return true;
  }

  return false;
}

SurfaceRGB::~SurfaceRGB() = default;

SurfaceRGB::SurfaceRGB() = default;

SurfaceRGB::SurfaceRGB(const SurfaceRGB& other) : plane(other.plane) {}

SurfaceRGB::SurfaceRGB(uint32_t width, uint32_t height, CUcontext context)
    : plane(width * 3, height, ElemSize(), DataType(), context) {}

SurfaceRGB& SurfaceRGB::operator=(const SurfaceRGB& other) {
  plane = other.plane;
  return *this;
}

Surface* SurfaceRGB::Create() { return new SurfaceRGB; }

uint32_t SurfaceRGB::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceRGB::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceRGB::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1U);
}

bool SurfaceRGB::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!plane.OwnMemory()) {
    plane = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceRGB::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceBGR::~SurfaceBGR() = default;

SurfaceBGR::SurfaceBGR() = default;

SurfaceBGR::SurfaceBGR(const SurfaceBGR& other) : plane(other.plane) {}

SurfaceBGR::SurfaceBGR(uint32_t width, uint32_t height, CUcontext context)
    : plane(width * 3, height, ElemSize(), DataType(), context) {}

SurfaceBGR& SurfaceBGR::operator=(const SurfaceBGR& other) {
  plane = other.plane;
  return *this;
}

Surface* SurfaceBGR::Create() { return new SurfaceBGR; }

uint32_t SurfaceBGR::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceBGR::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceBGR::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceBGR::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceBGR::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceBGR::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1);
}

bool SurfaceBGR::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!plane.OwnMemory()) {
    plane = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceBGR::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceRGBPlanar::~SurfaceRGBPlanar() = default;

SurfaceRGBPlanar::SurfaceRGBPlanar() = default;

SurfaceRGBPlanar::SurfaceRGBPlanar(const SurfaceRGBPlanar& other)
    : plane(other.plane) {}

SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                   CUcontext context)
    : plane(width, height * 3, ElemSize(), DataType(), context) {}

VPF::SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                        uint32_t elemSize, CUcontext context)
    : plane(width, height * 3, elemSize, DataType(), context) {}

SurfaceRGBPlanar& SurfaceRGBPlanar::operator=(const SurfaceRGBPlanar& other) {
  plane = other.plane;
  return *this;
}

Surface* SurfaceRGBPlanar::Create() { return new SurfaceRGBPlanar; }

uint32_t SurfaceRGBPlanar::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGBPlanar::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGBPlanar::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGBPlanar::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceRGBPlanar::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceRGBPlanar::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1);
}

bool SurfaceRGBPlanar::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!plane.OwnMemory()) {
    plane = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceRGBPlanar::GetSurfacePlane(uint32_t planeNumber) {
  // return planeNumber ? nullptr : &plane;
  return planeNumber < NumPlanes() ? &plane : nullptr;
}

SurfaceRGB32F::~SurfaceRGB32F() = default;

SurfaceRGB32F::SurfaceRGB32F() = default;

SurfaceRGB32F::SurfaceRGB32F(const SurfaceRGB32F& other) : plane(other.plane) {}

SurfaceRGB32F::SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context)
    : plane(width * 3, height, ElemSize(), DataType(), context) {}

SurfaceRGB32F& SurfaceRGB32F::operator=(const SurfaceRGB32F& other) {
  plane = other.plane;
  return *this;
}

Surface* SurfaceRGB32F::Create() { return new SurfaceRGB32F; }

uint32_t SurfaceRGB32F::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB32F::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB32F::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB32F::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceRGB32F::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    return plane.GpuMem();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceRGB32F::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1);
}

bool SurfaceRGB32F::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!plane.OwnMemory()) {
    plane = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceRGB32F::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceRGB32FPlanar::~SurfaceRGB32FPlanar() = default;

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar() = default;

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar(const SurfaceRGB32FPlanar& other)
    : plane(other.plane) {}

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar(uint32_t width, uint32_t height,
                                         CUcontext context)
    : plane(width, height * 3, ElemSize(), DataType(), context) {}

SurfaceRGB32FPlanar&
SurfaceRGB32FPlanar::operator=(const SurfaceRGB32FPlanar& other) {
  plane = other.plane;
  return *this;
}

Surface* SurfaceRGB32FPlanar::Create() { return new SurfaceRGB32FPlanar; }

uint32_t SurfaceRGB32FPlanar::Width(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB32FPlanar::WidthInBytes(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Width() * plane.ElemSize();
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB32FPlanar::Height(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Height() / 3;
  }

  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceRGB32FPlanar::Pitch(uint32_t planeNumber) const {
  if (planeNumber < NumPlanes()) {
    return plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceRGB32FPlanar::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    auto dptr = plane.GpuMem();
    return dptr + planeNumber * Height() * plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceRGB32FPlanar::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1);
}

bool SurfaceRGB32FPlanar::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!plane.OwnMemory()) {
    plane = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceRGB32FPlanar::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceYUV444_10bit::SurfaceYUV444_10bit() : SurfaceRGBPlanar() {}

SurfaceYUV444_10bit::SurfaceYUV444_10bit(const SurfaceYUV444_10bit& other)
    : SurfaceRGBPlanar(other) {}

SurfaceYUV444_10bit::SurfaceYUV444_10bit(uint32_t width, uint32_t height,
                                         CUcontext context)
    : SurfaceRGBPlanar(width, height, sizeof(uint16_t), context) {}

Surface* VPF::SurfaceYUV444_10bit::Create() { return new SurfaceYUV444_10bit; }

SurfaceP12::SurfaceP12(const SurfaceP12& other) : plane(other.plane) {}

SurfaceP12::SurfaceP12(uint32_t width, uint32_t height, CUcontext context)
    : plane(width, height * 3 / 2, ElemSize(), DataType(), context) {}

SurfaceP12& SurfaceP12::operator=(const SurfaceP12& other) {
  plane = other.plane;
  return *this;
}

Surface* SurfaceP12::Create() { return new SurfaceP12; }

uint32_t SurfaceP12::Width(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Width();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceP12::WidthInBytes(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Width() * plane.ElemSize();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceP12::Height(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
    return plane.Height() * 2 / 3;
  case 1:
    return plane.Height() / 3;
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceP12::Pitch(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Pitch();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceP12::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    auto dptr = plane.GpuMem();
    return dptr + planeNumber * Height() * plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceP12::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1);
}

bool SurfaceP12::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!plane.OwnMemory()) {
    plane = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceP12::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}

SurfaceP10::SurfaceP10(const SurfaceP10& other) : plane(other.plane) {}

SurfaceP10::SurfaceP10(uint32_t width, uint32_t height, CUcontext context)
    : plane(width, height * 3 / 2, ElemSize(), DataType(), context) {}

SurfaceP10& SurfaceP10::operator=(const SurfaceP10& other) {
  plane = other.plane;
  return *this;
}

Surface* SurfaceP10::Create() { return new SurfaceP10; }

uint32_t SurfaceP10::Width(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Width();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceP10::WidthInBytes(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Width() * plane.ElemSize();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceP10::Height(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
    return plane.Height() * 2 / 3;
  case 1:
    return plane.Height() / 3;
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

uint32_t SurfaceP10::Pitch(uint32_t planeNumber) const {
  switch (planeNumber) {
  case 0:
  case 1:
    return plane.Pitch();
  default:
    break;
  }
  throw invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceP10::PlanePtr(uint32_t planeNumber) {
  if (planeNumber < NumPlanes()) {
    auto dptr = plane.GpuMem();
    return dptr + planeNumber * Height() * plane.Pitch();
  }

  throw invalid_argument("Invalid plane number");
}

bool SurfaceP10::Update(SurfacePlane& newPlane) {
  SurfacePlane* planes[] = {&newPlane};
  return Update(planes, 1);
}

bool SurfaceP10::Update(SurfacePlane** pPlanes, size_t planesNum) {
  if (!ValidatePlanes(pPlanes, planesNum, ElemSize(), 1)) {
    return false;
  }

  if (!plane.OwnMemory()) {
    plane = *pPlanes[0];
    return true;
  }

  return false;
}

SurfacePlane* SurfaceP10::GetSurfacePlane(uint32_t planeNumber) {
  return planeNumber ? nullptr : &plane;
}