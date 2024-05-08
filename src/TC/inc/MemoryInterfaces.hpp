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

#pragma once

#include "SurfacePlane.hpp"
#include "nvEncodeAPI.h"
#include <vector>

#define ALIGN(x, a) __ALIGN_MASK(x, (decltype(x))(a)-1)
#define __ALIGN_MASK(x, mask) (((x) + (mask)) & ~(mask))

using namespace VPF;

namespace VPF {

enum Pixel_Format {
  UNDEFINED = 0,
  Y = 1,
  RGB = 2,
  NV12 = 3,
  YUV420 = 4,
  RGB_PLANAR = 5,
  BGR = 6,
  YUV444 = 7,
  RGB_32F = 8,
  RGB_32F_PLANAR = 9,
  YUV422 = 10,
  P10 = 11,
  P12 = 12,
  YUV444_10bit = 13,
  YUV420_10bit = 14,
  GRAY12 = 15,
};

enum ColorSpace {
  BT_601 = 0,
  BT_709 = 1,
  UNSPEC = 2,
};

enum ColorRange {
  MPEG = 0, /* Narrow range.*/
  JPEG = 1, /* Full range. */
  UDEF = 2,
};

struct ColorspaceConversionContext {
  ColorSpace color_space;
  ColorRange color_range;

  ColorspaceConversionContext() : color_space(UNSPEC), color_range(UDEF) {}

  ColorspaceConversionContext(ColorSpace cspace, ColorRange crange)
      : color_space(cspace), color_range(crange) {}
};

#ifdef TRACK_TOKEN_ALLOCATIONS
/* Returns true if allocation counters are equal to zero, false otherwise;
 * If you want to check for dangling pointers, call this function at exit;
 */
bool TC_EXPORT CheckAllocationCounters();
#endif

/* Represents CPU-side memory.
 * May own the memory or be a wrapper around existing ponter;
 */
class TC_EXPORT Buffer final : public Token {
public:
  Buffer() = delete;
  Buffer(const Buffer& other) = delete;
  Buffer& operator=(Buffer& other) = delete;

  ~Buffer() final;
  void* GetRawMemPtr();
  const void* GetRawMemPtr() const;
  size_t GetRawMemSize() const;
  void Update(size_t newSize, void* newPtr = nullptr);
  bool CopyFrom(size_t size, void const* ptr);
  template <typename T> T* GetDataAs() { return (T*)GetRawMemPtr(); }
  template <typename T> T const* GetDataAs() const {
    return (T const*)GetRawMemPtr();
  }

  static Buffer* Make(size_t bufferSize);
  static Buffer* Make(size_t bufferSize, void* pCopyFrom);

  static Buffer* MakeOwnMem(size_t bufferSize, CUcontext ctx = nullptr);
  static Buffer* MakeOwnMem(size_t bufferSize, const void* pCopyFrom,
                            CUcontext ctx = nullptr);

private:
  explicit Buffer(size_t bufferSize, bool ownMemory = true,
                  CUcontext ctx = nullptr);
  Buffer(size_t bufferSize, void* pCopyFrom, bool ownMemory,
         CUcontext ctx = nullptr);
  Buffer(size_t bufferSize, const void* pCopyFrom, CUcontext ctx = nullptr);
  bool Allocate();
  void Deallocate();

  bool own_memory = true;
  size_t mem_size = 0UL;
  void* pRawData = nullptr;
  CUcontext context = nullptr;
#ifdef TRACK_TOKEN_ALLOCATIONS
  uint32_t id;
#endif
};

class TC_EXPORT CudaBuffer final : public Token {
public:
  CudaBuffer() = delete;
  CudaBuffer(const CudaBuffer& other) = delete;
  CudaBuffer& operator=(CudaBuffer& other) = delete;

  static CudaBuffer* Make(size_t elemSize, size_t numElems, CUcontext context);
  static CudaBuffer* Make(const void* ptr, size_t elemSize, size_t numElems,
                          CUcontext context, CUstream str);
  CudaBuffer* Clone();

  size_t GetRawMemSize() const { return elem_size * num_elems; }
  size_t GetNumElems() const { return num_elems; }
  size_t GetElemSize() const { return elem_size; }
  CUdeviceptr GpuMem() { return gpuMem; }
  ~CudaBuffer();

private:
  CudaBuffer(size_t elemSize, size_t numElems, CUcontext context);
  CudaBuffer(const void* ptr, size_t elemSize, size_t numElems,
             CUcontext context, CUstream str);
  bool Allocate();
  void Deallocate();

  CUdeviceptr gpuMem = 0UL;
  CUcontext ctx = nullptr;
  size_t elem_size = 0U;
  size_t num_elems = 0U;

#ifdef TRACK_TOKEN_ALLOCATIONS
  uint64_t id = 0;
#endif
};

/* Represents GPU-side memory.
 * Pure interface class, see ancestors;
 */
class TC_EXPORT Surface : public Token {
public:
  virtual ~Surface();

  /* Returns width in pixels;
   */
  virtual uint32_t Width(uint32_t plane = 0U) const = 0;

  /* Returns width in bytes;
   */
  virtual uint32_t WidthInBytes(uint32_t plane = 0U) const = 0;

  /* Returns height in pixels;
   */
  virtual uint32_t Height(uint32_t plane = 0U) const = 0;

  /* Returns pitch in bytes;
   */
  virtual uint32_t Pitch(uint32_t plane = 0U) const = 0;

  /* Returns element size in bytes;
   */
  virtual uint32_t ElemSize() const = 0;

  /* Return number of components;
   */
  virtual uint32_t NumComponents() const = 0;

  /* Returns pixel format;
   */
  virtual Pixel_Format PixelFormat() const = 0;

  /* Returns DLPack data type code;
   */
  virtual DLDataTypeCode DataType() const = 0;

  /* Returns Surface Plane by number;
   */
  virtual SurfacePlane& GetSurfacePlane(uint32_t plane = 0U) = 0;

  /* Update from set of image planes, don't own the memory;
   */
  virtual bool Update(std::initializer_list<SurfacePlane*> planes) = 0;

  /* Get CUDA device pointer to first pixel of given component;
   */
  virtual CUdeviceptr PixelPtr(uint32_t component = 0U) = 0;

  /* Returns number of image planes;
   */
  virtual uint32_t NumPlanes() const = 0;

  /* Copy constructor which does CUDA memalloc + deep copy;
   */
  Surface* Clone();

  /* Virtual default constructor;
   */
  virtual Surface* Create() = 0;

  /* Get associated CUDA context;
   */
  CUcontext Context();

  /* Returns true if memory was allocated in constructor, false otherwise;
   */
  bool OwnMemory();

  /* Returns total amount of memory in bytes needed
   * to store all pixels of Surface in Host memory;
   */
  uint32_t HostMemSize() const;

  /* Returns true if surface is empty (no allocated data), false otherwise;
   */
  bool Empty() const;

  /* Make empty;
   */
  static Surface* Make(Pixel_Format format);

  /* Make & own memory;
   */
  static Surface* Make(Pixel_Format format, uint32_t newWidth,
                       uint32_t newHeight, CUcontext context);

protected:
  Surface();
  std::vector<SurfacePlane> m_planes;
};
} // namespace VPF