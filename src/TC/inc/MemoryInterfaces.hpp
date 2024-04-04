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

#include "Surface.hpp"
#include "nvEncodeAPI.h"

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
};

/* 8-bit single plane image.
 */
class TC_EXPORT SurfaceY final : public Surface {
public:
  ~SurfaceY() = default;
  SurfaceY() = default;

  SurfaceY(const SurfaceY& other) = delete;
  SurfaceY(const SurfaceY&& other) = delete;
  SurfaceY& operator=(const SurfaceY& other) = delete;
  SurfaceY& operator=(const SurfaceY&& other) = delete;

  SurfaceY(uint32_t width, uint32_t height, CUcontext context = nullptr,
           bool pitched = true);

  uint32_t NumPlanes() const noexcept override { return 1U; };
  Pixel_Format PixelFormat() const noexcept override { return Y; };
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

/* 8-bit NV12 image;
 */
class TC_EXPORT SurfaceNV12 final : public Surface {
public:
  ~SurfaceNV12() = default;
  SurfaceNV12() = default;

  SurfaceNV12(const SurfaceNV12& other) = delete;
  SurfaceNV12(const SurfaceNV12&& other) = delete;
  SurfaceNV12& operator=(const SurfaceNV12& other) = delete;
  SurfaceNV12& operator=(const SurfaceNV12&& other) = delete;

  SurfaceNV12(uint32_t width, uint32_t height, CUcontext context = nullptr,
              bool pitched = true);

  uint32_t NumPlanes() const noexcept override { return 2U; };
  Pixel_Format PixelFormat() const noexcept override { return NV12; };
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

/* 8-bit YUV420P image;
 */
class TC_EXPORT SurfaceYUV420 final : public Surface {
public:
  ~SurfaceYUV420() = default;
  SurfaceYUV420() = default;

  SurfaceYUV420(const SurfaceYUV420& other) = delete;
  SurfaceYUV420(const SurfaceYUV420&& other) = delete;
  SurfaceYUV420& operator=(const SurfaceYUV420& other) = delete;
  SurfaceYUV420& operator=(const SurfaceYUV420&& other) = delete;

  SurfaceYUV420(uint32_t width, uint32_t height, CUcontext context = nullptr,
                bool pitched = true);

  Pixel_Format PixelFormat() const override { return YUV420; }
  uint32_t NumPlanes() const override { return 3U; }
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

class TC_EXPORT SurfaceYUV422 final : public Surface {
  ~SurfaceYUV422() = default;
  SurfaceYUV422() = default;

  SurfaceYUV422(const SurfaceYUV422& other) = delete;
  SurfaceYUV422(const SurfaceYUV422&& other) = delete;
  SurfaceYUV422& operator=(const SurfaceYUV422& other) = delete;
  SurfaceYUV422& operator=(const SurfaceYUV422&& other) = delete;

  SurfaceYUV422(uint32_t width, uint32_t height, CUcontext context = nullptr,
                bool pitched = true);

  Pixel_Format PixelFormat() const override { return YUV422; }
  uint32_t NumPlanes() const override { return 3U; }
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

/* 8-bit RGB image;
 */
class TC_EXPORT SurfaceRGB final : public Surface {
  ~SurfaceRGB() = default;
  SurfaceRGB() = default;

  SurfaceRGB(const SurfaceRGB& other) = delete;
  SurfaceRGB(const SurfaceRGB&& other) = delete;
  SurfaceRGB& operator=(const SurfaceRGB& other) = delete;
  SurfaceRGB& operator=(const SurfaceRGB&& other) = delete;

  SurfaceRGB(uint32_t width, uint32_t height, CUcontext context = nullptr,
             bool pitched = true);

  Pixel_Format PixelFormat() const override { return RGB; }
  uint32_t NumPlanes() const override { return 3U; }
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

/* 8-bit BGR image;
 */
class TC_EXPORT SurfaceBGR final : public SurfaceBGR {
  ~SurfaceBGR() = default;
  SurfaceBGR() = default;

  SurfaceBGR(const SurfaceBGR& other) = delete;
  SurfaceBGR(const SurfaceBGR&& other) = delete;
  SurfaceBGR& operator=(const SurfaceBGR& other) = delete;
  SurfaceBGR& operator=(const SurfaceBGR&& other) = delete;

  SurfaceBGR(uint32_t width, uint32_t height, CUcontext context = nullptr,
             bool pitched = true);

  Pixel_Format PixelFormat() const override { return BGR; }
  uint32_t NumPlanes() const override { return 3U; }
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

/* 8-bit planar RGB image;
 */
class TC_EXPORT SurfaceRGBPlanar final : public Surface {
  ~SurfaceRGBPlanar() = default;
  SurfaceRGBPlanar() = default;

  SurfaceRGBPlanar(const SurfaceRGBPlanar& other) = delete;
  SurfaceRGBPlanar(const SurfaceRGBPlanar&& other) = delete;
  SurfaceRGBPlanar& operator=(const SurfaceRGBPlanar& other) = delete;
  SurfaceRGBPlanar& operator=(const SurfaceRGBPlanar&& other) = delete;

  SurfaceRGBPlanar(uint32_t width, uint32_t height, CUcontext context = nullptr,
                   bool pitched = true);

  Pixel_Format PixelFormat() const override { return RGB_PLANAR; }
  uint32_t NumPlanes() const override { return 1U; }
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

class TC_EXPORT SurfaceYUV444 final : public Surface {
  ~SurfaceYUV444() = default;
  SurfaceYUV444() = default;

  SurfaceYUV444(const SurfaceYUV444& other) = delete;
  SurfaceYUV444(const SurfaceYUV444&& other) = delete;
  SurfaceYUV444& operator=(const SurfaceYUV444& other) = delete;
  SurfaceYUV444& operator=(const SurfaceYUV444&& other) = delete;

  SurfaceYUV444(uint32_t width, uint32_t height, CUcontext context = nullptr,
                bool pitched = true);

  Pixel_Format PixelFormat() const override { return YUV444; }
  uint32_t NumPlanes() const override { return 1U; }
  size_t ElemSize() const noexcept override { return sizeof(uint8_t;) }
};

/* 32-bit float RGB image;
 */
class TC_EXPORT SurfaceRGB32F final : public Surface {
  ~SurfaceRGB32F() = default;
  SurfaceRGB32F() = default;

  SurfaceRGB32F(const SurfaceRGB32F& other) = delete;
  SurfaceRGB32F(const SurfaceRGB32F&& other) = delete;
  SurfaceRGB32F& operator=(const SurfaceRGB32F& other) = delete;
  SurfaceRGB32F& operator=(const SurfaceRGB32F&& other) = delete;

  SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context = nullptr,
                bool pitched = true);

  Pixel_Format PixelFormat() const override { return RGB_32F; }
  uint32_t NumPlanes() const override { return 3U; }
  size_t ElemSize() const noexcept override { return sizeof(float;) }
};

/* 32-bit float planar RGB image;
 */
class TC_EXPORT SurfaceRGB32FPlanar final : public Surface {
  ~SurfaceRGB32FPlanar() = default;
  SurfaceRGB32FPlanar() = default;

  SurfaceRGB32FPlanar(const SurfaceRGB32FPlanar& other) = delete;
  SurfaceRGB32FPlanar(const SurfaceRGB32FPlanar&& other) = delete;
  SurfaceRGB32FPlanar& operator=(const SurfaceRGB32FPlanar& other) = delete;
  SurfaceRGB32FPlanar& operator=(const SurfaceRGB32FPlanar&& other) = delete;

  SurfaceRGB32FPlanar(uint32_t width, uint32_t height,
                      CUcontext context = nullptr, bool pitched = true);

  Pixel_Format PixelFormat() const override { return RGB_32F_PLANAR; }
  uint32_t NumPlanes() const override { return 1U; }
  size_t ElemSize() const noexcept override { return sizeof(float;) }
};

class TC_EXPORT SurfaceYUV444_10bit final : public Surface {
  ~SurfaceYUV444_10bit() = default;
  SurfaceYUV444_10bit() = default;

  SurfaceYUV444_10bit(const SurfaceYUV444_10bit& other) = delete;
  SurfaceYUV444_10bit(const SurfaceYUV444_10bit&& other) = delete;
  SurfaceYUV444_10bit& operator=(const SurfaceYUV444_10bit& other) = delete;
  SurfaceYUV444_10bit& operator=(const SurfaceYUV444_10bit&& other) = delete;

  SurfaceYUV444_10bit(uint32_t width, uint32_t height,
                      CUcontext context = nullptr, bool pitched = true);

  Pixel_Format PixelFormat() const override { return YUV444; }
  uint32_t NumPlanes() const override { return 1U; }
  size_t ElemSize() const noexcept override { return sizeof(uint16_t;) }
};

/* 10-bit NV12 image;
 */
class TC_EXPORT SurfaceP10 final : public Surface {
public:
  ~SurfaceP10() = default;
  SurfaceP10() = default;

  SurfaceP10(const SurfaceP10& other) = delete;
  SurfaceP10(const SurfaceP10&& other) = delete;
  SurfaceP10& operator=(const SurfaceP10& other) = delete;
  SurfaceP10& operator=(const SurfaceP10&& other) = delete;

  SurfaceP10(uint32_t width, uint32_t height, CUcontext context = nullptr,
             bool pitched = true);

  uint32_t NumPlanes() const noexcept override { return 2U; };
  Pixel_Format PixelFormat() const noexcept override { return P10; };
  size_t ElemSize() const noexcept override { return sizeof(uint16_t;) }
};

/* 12-bit NV12 image;
 */
class TC_EXPORT SurfaceP12 final : public Surface {
public:
  ~SurfaceP12() = default;
  SurfaceP12() = default;

  SurfaceP12(const SurfaceP12& other) = delete;
  SurfaceP12(const SurfaceP12&& other) = delete;
  SurfaceP12& operator=(const SurfaceP12& other) = delete;
  SurfaceP12& operator=(const SurfaceP12&& other) = delete;

  SurfaceP12(uint32_t width, uint32_t height, CUcontext context = nullptr,
             bool pitched = true);

  uint32_t NumPlanes() const noexcept override { return 2U; };
  Pixel_Format PixelFormat() const noexcept override { return P10; };
  size_t ElemSize() const noexcept override { return sizeof(uint16_t;) }
};
} // namespace VPF