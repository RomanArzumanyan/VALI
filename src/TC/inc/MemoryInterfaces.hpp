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
  virtual uint32_t Width(uint32_t planeNumber = 0U) const = 0;

  /* Returns width in bytes;
   */
  virtual uint32_t WidthInBytes(uint32_t planeNumber = 0U) const = 0;

  /* Returns height in pixels;
   */
  virtual uint32_t Height(uint32_t planeNumber = 0U) const = 0;

  /* Returns pitch in bytes;
   */
  virtual uint32_t Pitch(uint32_t planeNumber = 0U) const = 0;

  /* Returns number of image planes;
   */
  virtual uint32_t NumPlanes() const = 0;

  /* Returns element size in bytes;
   */
  virtual uint32_t ElemSize() const = 0;

  /* Returns SurfacePlane CUDA device pointer;
   */
  virtual CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) = 0;

  /* Returns pixel format;
   */
  virtual Pixel_Format PixelFormat() const = 0;

  /* Returns DLPack data type code;
   */
  virtual DLDataTypeCode DataType() const = 0;

  /* Returns pointer to plane by number;
   */
  virtual SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U) = 0;

  /* Update from set of image planes, don't own the memory;
   */
  virtual bool Update(SurfacePlane** pPlanes, size_t planesNum) = 0;

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
  // TODO (r.arzumanyan): remove virtual
  virtual uint32_t HostMemSize() const;

  /* Returns true if surface is empty (no allocated data), false otherwise;
   */
  // TODO (r.arzumanyan): remove virtual
  virtual bool Empty() const;

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

/* 8-bit single plane image.
 */
class TC_EXPORT SurfaceY final : public Surface {
public:
  virtual ~SurfaceY() = default;
  SurfaceY(const SurfaceY& other) = delete;
  SurfaceY(SurfaceY&& other) = delete;
  SurfaceY& operator=(const SurfaceY& other) = delete;
  SurfaceY& operator=(SurfaceY&& other) = delete;

  SurfaceY();
  SurfaceY(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  uint32_t ElemSize() const { return sizeof(uint8_t); }

  CUdeviceptr PlanePtr(uint32_t plane = 0U);
  Pixel_Format PixelFormat() const { return Y; };
  uint32_t NumPlanes() const { return 1U; };
  DLDataTypeCode DataType() const { return kDLUInt; }

  bool Update(SurfacePlane& newPlane);
  bool Update(SurfacePlane** pPlanes, size_t planesNum);
  SurfacePlane* GetSurfacePlane(uint32_t plane = 0U);
};

/* 8-bit NV12 image;
 */
class TC_EXPORT SurfaceNV12 : public Surface {
public:
  virtual ~SurfaceNV12() = default;
  SurfaceNV12(const SurfaceNV12& other) = delete;
  SurfaceNV12(SurfaceNV12&& other) = delete;
  SurfaceNV12& operator=(const SurfaceNV12& other) = delete;
  SurfaceNV12& operator=(SurfaceNV12&& other) = delete;

  SurfaceNV12();
  SurfaceNV12(uint32_t width, uint32_t height, CUcontext context);
  SurfaceNV12(uint32_t width, uint32_t height, uint32_t pitch, CUdeviceptr ptr);

  virtual Surface* Create() override;
  virtual Pixel_Format PixelFormat() const override { return NV12; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  uint32_t NumPlanes() const { return 2; }
  DLDataTypeCode DataType() const { return kDLUInt; }

  CUdeviceptr PlanePtr(uint32_t plane = 0U);
  SurfacePlane* GetSurfacePlane(uint32_t plane = 0U);

  bool Update(SurfacePlane& newPlane);
  bool Update(SurfacePlane** pPlanes, size_t planesNum);

protected:
  // For high bit depth ancestors;
  SurfaceNV12(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
              CUcontext context);
};

// 10 bit NV12;
class TC_EXPORT SurfaceP10 final : public SurfaceNV12 {
public:
  virtual ~SurfaceP10() = default;
  SurfaceP10(const SurfaceP10& other) = delete;
  SurfaceP10(SurfaceP10&& other) = delete;
  SurfaceP10& operator=(const SurfaceP10& other) = delete;
  SurfaceP10& operator=(SurfaceP10&& other) = delete;

  SurfaceP10();
  SurfaceP10(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();
  Pixel_Format PixelFormat() const { return P10; }
  uint32_t ElemSize() const { return sizeof(uint16_t); }
};

// 12 bit NV12;
class TC_EXPORT SurfaceP12 final : public SurfaceNV12 {
public:
  virtual ~SurfaceP12() = default;
  SurfaceP12(const SurfaceP12& other) = delete;
  SurfaceP12(SurfaceP12&& other) = delete;
  SurfaceP12& operator=(const SurfaceP12& other) = delete;
  SurfaceP12& operator=(SurfaceP12&& other) = delete;

  SurfaceP12();
  SurfaceP12(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();
  Pixel_Format PixelFormat() const { return P12; }
  uint32_t ElemSize() const { return sizeof(uint16_t); }
};

/* 8-bit YUV420P image;
 */
class TC_EXPORT SurfaceYUV420 final: public Surface {
public:
  virtual ~SurfaceYUV420() = default;
  SurfaceYUV420(const SurfaceYUV420& other) = delete;
  SurfaceYUV420(SurfaceYUV420&& other) = delete;
  SurfaceYUV420& operator=(const SurfaceYUV420& other) = delete;
  SurfaceYUV420& operator=(SurfaceYUV420&& other) = delete;

  SurfaceYUV420();
  SurfaceYUV420(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  uint32_t ElemSize() const { return sizeof(uint8_t); }

  CUdeviceptr PlanePtr(uint32_t plane = 0U);
  Pixel_Format PixelFormat() const { return YUV420; }
  uint32_t NumPlanes() const { return 3; }
  DLDataTypeCode DataType() const { return kDLUInt; }

  bool Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
              SurfacePlane& newPlaneV);
  bool Update(SurfacePlane** pPlanes, size_t planesNum);
  SurfacePlane* GetSurfacePlane(uint32_t plane = 0U);
};

class TC_EXPORT SurfaceYUV422 final : public Surface {
public:
  virtual ~SurfaceYUV422() = default;
  SurfaceYUV422(const SurfaceYUV422& other) = delete;
  SurfaceYUV422(SurfaceYUV422&& other) = delete;
  SurfaceYUV422& operator=(const SurfaceYUV422& other) = delete;
  SurfaceYUV422& operator=(SurfaceYUV422&& other) = delete;

  SurfaceYUV422();
  SurfaceYUV422(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const;
  uint32_t Height(uint32_t planeNumber = 0U) const;
  uint32_t Pitch(uint32_t planeNumber = 0U) const;
  Pixel_Format PixelFormat() const { return YUV422; }
  uint32_t NumPlanes() const { return 3; }
  DLDataTypeCode DataType() const { return kDLUInt; }
  uint32_t ElemSize() const { return sizeof(uint8_t); }

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U);
  SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U);

  bool Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
              SurfacePlane& newPlaneV);
  bool Update(SurfacePlane** pPlanes, size_t planesNum);
};

class TC_EXPORT SurfaceYUV444 : public Surface {
public:
  virtual ~SurfaceYUV444() = default;
  SurfaceYUV444(const SurfaceYUV444& other) = delete;
  SurfaceYUV444(SurfaceYUV444&& other) = delete;
  SurfaceYUV444& operator=(const SurfaceYUV444& other) = delete;
  SurfaceYUV444& operator=(SurfaceYUV444&& other) = delete;

  SurfaceYUV444();
  SurfaceYUV444(uint32_t width, uint32_t height, CUcontext context);

  virtual Surface* Create() override;
  virtual Pixel_Format PixelFormat() const override { return YUV422; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }

  uint32_t Width(uint32_t planeNumber = 0U) const;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const;
  uint32_t Height(uint32_t planeNumber = 0U) const;
  uint32_t Pitch(uint32_t planeNumber = 0U) const;
  uint32_t NumPlanes() const { return 3; }
  DLDataTypeCode DataType() const { return kDLUInt; }

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U);
  SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U);

  bool Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
              SurfacePlane& newPlaneV);
  bool Update(SurfacePlane** pPlanes, size_t planesNum);

protected:
  // For high bit depth ancestors;
  SurfaceYUV444(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
                CUcontext context);
};

class TC_EXPORT SurfaceYUV444_10bit final: public SurfaceYUV444 {
public:
  virtual ~SurfaceYUV444_10bit() = default;
  SurfaceYUV444_10bit(const SurfaceYUV444_10bit& other) = delete;
  SurfaceYUV444_10bit(SurfaceYUV444_10bit&& other) = delete;
  SurfaceYUV444_10bit& operator=(const SurfaceYUV444_10bit& other) = delete;
  SurfaceYUV444_10bit& operator=(SurfaceYUV444_10bit&& other) = delete;

  SurfaceYUV444_10bit();
  SurfaceYUV444_10bit(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();
  Pixel_Format PixelFormat() const { return YUV444_10bit; }
  uint32_t ElemSize() const { return sizeof(uint16_t); }
};

/* 8-bit RGB image;
 */
class TC_EXPORT SurfaceRGB : public Surface {
public:
  virtual ~SurfaceRGB() = default;
  SurfaceRGB(const SurfaceRGB& other) = delete;
  SurfaceRGB(SurfaceRGB&& other) = delete;
  SurfaceRGB& operator=(const SurfaceRGB& other) = delete;
  SurfaceRGB& operator=(SurfaceRGB&& other) = delete;

  SurfaceRGB();
  SurfaceRGB(uint32_t width, uint32_t height, CUcontext context);

  virtual Surface* Create() override;
  virtual Pixel_Format PixelFormat() const override { return RGB; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }

  uint32_t Width(uint32_t planeNumber = 0U) const;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const;
  uint32_t Height(uint32_t planeNumber = 0U) const;
  uint32_t Pitch(uint32_t planeNumber = 0U) const;
  uint32_t NumPlanes() const { return 1; }
  DLDataTypeCode DataType() const { return kDLUInt; }

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U);
  SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U);

  bool Update(SurfacePlane& newPlane);
  bool Update(SurfacePlane** pPlanes, size_t planesNum);

protected:
  // For high bit depth ancestors;
  SurfaceRGB(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
                CUcontext context);
};

/* 8-bit BGR image;
 */
class TC_EXPORT SurfaceBGR : public SurfaceRGB {
public:
  ~SurfaceBGR();

  SurfaceBGR();
  SurfaceBGR(const SurfaceBGR& other);
  SurfaceBGR(uint32_t width, uint32_t height, CUcontext context);
  SurfaceBGR& operator=(const SurfaceBGR& other);

  Surface* Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return BGR; }
  uint32_t NumPlanes() const override { return 1; }
  bool Empty() const override { return 0UL == plane.GpuMem(); }
  DLDataTypeCode DataType() const override { return kDLUInt; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }

  bool Update(SurfacePlane& newPlane);
  bool Update(SurfacePlane** pPlanes, size_t planesNum) override;
  SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U) override;
  virtual uint32_t HostMemSize() const override { return plane.HostMemSize(); }

protected:
  SurfacePlane plane;
};

/* 8-bit planar RGB image;
 */
class TC_EXPORT SurfaceRGBPlanar : public Surface {
public:
  ~SurfaceRGBPlanar();

  SurfaceRGBPlanar();
  SurfaceRGBPlanar(const SurfaceRGBPlanar& other);
  SurfaceRGBPlanar(uint32_t width, uint32_t height, CUcontext context);
  SurfaceRGBPlanar(uint32_t width, uint32_t height, uint32_t elemSize,
                   CUcontext context);
  SurfaceRGBPlanar& operator=(const SurfaceRGBPlanar& other);

  virtual Surface* Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return RGB_PLANAR; }
  uint32_t NumPlanes() const override { return 1; }
  bool Empty() const override { return 0UL == plane.GpuMem(); }
  DLDataTypeCode DataType() const override { return kDLUInt; }

  bool Update(SurfacePlane& newPlane);
  bool Update(SurfacePlane** pPlanes, size_t planesNum) override;
  SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U) override;
  virtual uint32_t HostMemSize() const override { return plane.HostMemSize(); }

protected:
  SurfacePlane plane;
};

#ifdef TRACK_TOKEN_ALLOCATIONS
/* Returns true if allocation counters are equal to zero, false otherwise;
 * If you want to check for dangling pointers, call this function at exit;
 */
bool TC_EXPORT CheckAllocationCounters();
#endif

/* 32-bit float RGB image;
 */
class TC_EXPORT SurfaceRGB32F : public Surface {
public:
  ~SurfaceRGB32F();

  SurfaceRGB32F();
  SurfaceRGB32F(const SurfaceRGB32F& other);
  SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context);
  SurfaceRGB32F& operator=(const SurfaceRGB32F& other);

  Surface* Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return RGB_32F; }
  uint32_t NumPlanes() const override { return 1; }
  bool Empty() const override { return 0UL == plane.GpuMem(); }
  DLDataTypeCode DataType() const override { return kDLFloat; }
  virtual uint32_t ElemSize() const override { return sizeof(float); }

  bool Update(SurfacePlane& newPlane);
  bool Update(SurfacePlane** pPlanes, size_t planesNum) override;
  SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U) override;
  virtual uint32_t HostMemSize() const override { return plane.HostMemSize(); }

protected:
  SurfacePlane plane;
};

/* 32-bit float planar RGB image;
 */
class TC_EXPORT SurfaceRGB32FPlanar : public Surface {
public:
  ~SurfaceRGB32FPlanar();

  SurfaceRGB32FPlanar();
  SurfaceRGB32FPlanar(const SurfaceRGB32FPlanar& other);
  SurfaceRGB32FPlanar(uint32_t width, uint32_t height, CUcontext context);
  SurfaceRGB32FPlanar& operator=(const SurfaceRGB32FPlanar& other);

  virtual Surface* Create() override;

  uint32_t Width(uint32_t planeNumber = 0U) const override;
  uint32_t WidthInBytes(uint32_t planeNumber = 0U) const override;
  uint32_t Height(uint32_t planeNumber = 0U) const override;
  uint32_t Pitch(uint32_t planeNumber = 0U) const override;

  CUdeviceptr PlanePtr(uint32_t planeNumber = 0U) override;
  Pixel_Format PixelFormat() const override { return RGB_32F_PLANAR; }
  uint32_t NumPlanes() const override { return 3; }
  bool Empty() const override { return 0UL == plane.GpuMem(); }
  DLDataTypeCode DataType() const override { return kDLFloat; }
  virtual uint32_t ElemSize() const override { return sizeof(float); }

  bool Update(SurfacePlane& newPlane);
  bool Update(SurfacePlane** pPlanes, size_t planesNum) override;
  SurfacePlane* GetSurfacePlane(uint32_t planeNumber = 0U) override;
  virtual uint32_t HostMemSize() const override { return plane.HostMemSize(); }

protected:
  SurfacePlane plane;
};
} // namespace VPF