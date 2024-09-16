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

#pragma once
#include "MemoryInterfaces.hpp"
#include <stdexcept>

namespace VPF {
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
  uint32_t NumComponents() const { return 1U; }
  uint32_t NumPlanes() const { return 1U; }

  Pixel_Format PixelFormat() const { return Y; };
  DLDataTypeCode DataType() const { return kDLUInt; }
  std::string TypeStr() const { return "<u1"; }
  DLManagedTensor* ToDLPack();
  void ToCAI(CudaArrayInterfaceDescriptor& cai);

  bool Update(SurfacePlane& newPlane);
  bool Update(std::initializer_list<SurfacePlane*> planes);
  SurfacePlane& GetSurfacePlane(uint32_t plane = 0U);
  CUdeviceptr PixelPtr(uint32_t component = 0U);
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
  virtual uint32_t NumComponents() const override { return 2U; }
  virtual uint32_t NumPlanes() const override { return 1U; }
  virtual CUdeviceptr PixelPtr(uint32_t component = 0U) override;

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  DLDataTypeCode DataType() const { return kDLUInt; }
  std::string TypeStr() const { return "<u1"; }

  SurfacePlane& GetSurfacePlane(uint32_t plane = 0U);

  bool Update(SurfacePlane& newPlane);
  bool Update(std::initializer_list<SurfacePlane*> planes);
  DLManagedTensor* ToDLPack();
  void ToCAI(CudaArrayInterfaceDescriptor& cai);

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
  std::string TypeStr() const { return "<u2"; }
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
  std::string TypeStr() const { return "<u2"; }
};

/* 8-bit YUV420P image;
 */
class TC_EXPORT SurfaceYUV420 final : public Surface {
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
  uint32_t NumComponents() const { return 3U; }
  uint32_t NumPlanes() const { return 3U; }

  Pixel_Format PixelFormat() const { return YUV420; }
  DLDataTypeCode DataType() const { return kDLUInt; }
  std::string TypeStr() const { return "<u1"; }

  bool Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
              SurfacePlane& newPlaneV);
  bool Update(std::initializer_list<SurfacePlane*> planes);
  SurfacePlane& GetSurfacePlane(uint32_t plane = 0U);
  CUdeviceptr PixelPtr(uint32_t component = 0U);

  /* Will throw exception, because nuber of planes > 1;
   * DLPack specification allows for 1 CUDA device ptr only;
   */
  DLManagedTensor* ToDLPack() {
    throw std::runtime_error("Number of CUDA memory allocations > 1, cant "
                             "seriaize into DLPack tensor.");
  }

  void ToCAI(CudaArrayInterfaceDescriptor& cai) {
    throw std::runtime_error("Number of CUDA memory allocations > 1, cant "
                             "seriaize into CUDA Adday Interface.");
  }
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

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  uint32_t ElemSize() const { return sizeof(uint8_t); }
  uint32_t NumComponents() const { return 3U; }
  uint32_t NumPlanes() const { return 3U; }

  Pixel_Format PixelFormat() const { return YUV422; }
  DLDataTypeCode DataType() const { return kDLUInt; }
  std::string TypeStr() const { return "<u1"; }

  bool Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
              SurfacePlane& newPlaneV);
  bool Update(std::initializer_list<SurfacePlane*> planes);
  SurfacePlane& GetSurfacePlane(uint32_t plane = 0U);
  CUdeviceptr PixelPtr(uint32_t component = 0U);

  /* Will throw exception, because nuber of planes > 1;
   * DLPack specification allows for 1 CUDA device ptr only;
   */
  DLManagedTensor* ToDLPack() {
    throw std::runtime_error("Number of CUDA memory allocations > 1, cant "
                             "seriaize into DLPack tensor.");
  }

  void ToCAI(CudaArrayInterfaceDescriptor& cai) {
    throw std::runtime_error("Number of CUDA memory allocations > 1, cant "
                             "seriaize into CUDA Adday Interface.");
  }
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
  virtual Pixel_Format PixelFormat() const override { return YUV444; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }
  virtual uint32_t NumComponents() const override { return 3U; }
  virtual uint32_t NumPlanes() const override { return 3U; }
  virtual CUdeviceptr PixelPtr(uint32_t component = 0U) override;

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  DLDataTypeCode DataType() const { return kDLUInt; }
  std::string TypeStr() const { return "<u1"; }

  SurfacePlane& GetSurfacePlane(uint32_t plane = 0U);

  bool Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
              SurfacePlane& newPlaneV);
  bool Update(std::initializer_list<SurfacePlane*> planes);

  /* Will throw exception, because nuber of planes > 1;
   * DLPack specification allows for 1 CUDA device ptr only;
   */
  DLManagedTensor* ToDLPack() {
    throw std::runtime_error("Number of CUDA memory allocations > 1, cant "
                             "seriaize into DLPack tensor.");
  }

  void ToCAI(CudaArrayInterfaceDescriptor& cai) {
    throw std::runtime_error("Number of CUDA memory allocations > 1, cant "
                             "seriaize into CUDA Adday Interface.");
  }

protected:
  // For high bit depth ancestors;
  SurfaceYUV444(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
                CUcontext context);
};

class TC_EXPORT SurfaceYUV444_10bit final : public SurfaceYUV444 {
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
  std::string TypeStr() const { return "<u2"; }
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
  virtual uint32_t NumComponents() const override { return 1U; }
  virtual uint32_t NumPlanes() const override { return 1U; }
  virtual CUdeviceptr PixelPtr(uint32_t component = 0U) override;
  virtual DLManagedTensor* ToDLPack() override;
  virtual void ToCAI(CudaArrayInterfaceDescriptor& cai);

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  DLDataTypeCode DataType() const { return kDLUInt; }
  std::string TypeStr() const { return "<u1"; }

  SurfacePlane& GetSurfacePlane(uint32_t plane = 0U);

  bool Update(SurfacePlane& newPlane);
  bool Update(std::initializer_list<SurfacePlane*> planes);

protected:
  // For high bit depth ancestors;
  SurfaceRGB(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
             CUcontext context);
};

/* 8-bit BGR image;
 */
class TC_EXPORT SurfaceBGR final : public SurfaceRGB {
public:
  virtual ~SurfaceBGR() = default;
  SurfaceBGR(const SurfaceBGR& other) = delete;
  SurfaceBGR(SurfaceBGR&& other) = delete;
  SurfaceBGR& operator=(const SurfaceBGR& other) = delete;
  SurfaceBGR& operator=(SurfaceBGR&& other) = delete;

  SurfaceBGR();
  SurfaceBGR(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();
  Pixel_Format PixelFormat() const { return BGR; }
  uint32_t ElemSize() const { return sizeof(uint8_t); }
};

/* 32-bit float RGB image;
 */
class TC_EXPORT SurfaceRGB32F final : public SurfaceRGB {
public:
  virtual ~SurfaceRGB32F() = default;
  SurfaceRGB32F(const SurfaceRGB32F& other) = delete;
  SurfaceRGB32F(SurfaceRGB32F&& other) = delete;
  SurfaceRGB32F& operator=(const SurfaceRGB32F& other) = delete;
  SurfaceRGB32F& operator=(SurfaceRGB32F&& other) = delete;

  SurfaceRGB32F();
  SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();
  Pixel_Format PixelFormat() const { return RGB_32F; }
  uint32_t ElemSize() const { return sizeof(float); }
  std::string TypeStr() const { return "<f4"; }
};

/* 8-bit planar RGB image;
 */
class TC_EXPORT SurfaceRGBPlanar : public Surface {
public:
  virtual ~SurfaceRGBPlanar() = default;
  SurfaceRGBPlanar(const SurfaceRGBPlanar& other) = delete;
  SurfaceRGBPlanar(SurfaceRGBPlanar&& other) = delete;
  SurfaceRGBPlanar& operator=(const SurfaceRGBPlanar& other) = delete;
  SurfaceRGBPlanar& operator=(SurfaceRGBPlanar&& other) = delete;

  SurfaceRGBPlanar();
  SurfaceRGBPlanar(uint32_t width, uint32_t height, CUcontext context);

  virtual Surface* Create() override;
  virtual Pixel_Format PixelFormat() const override { return RGB_PLANAR; }
  virtual uint32_t ElemSize() const override { return sizeof(uint8_t); }
  virtual uint32_t NumComponents() const override { return 3U; }
  virtual uint32_t NumPlanes() const override { return 1U; }
  virtual CUdeviceptr PixelPtr(uint32_t component = 0U) override;
  virtual DLManagedTensor* ToDLPack() override;
  virtual void ToCAI(CudaArrayInterfaceDescriptor& cai) override;

  uint32_t Width(uint32_t plane = 0U) const;
  uint32_t WidthInBytes(uint32_t plane = 0U) const;
  uint32_t Height(uint32_t plane = 0U) const;
  uint32_t Pitch(uint32_t plane = 0U) const;
  DLDataTypeCode DataType() const { return kDLUInt; }
  std::string TypeStr() const { return "<u1"; }

  SurfacePlane& GetSurfacePlane(uint32_t plane = 0U);

  bool Update(SurfacePlane& newPlane);
  bool Update(std::initializer_list<SurfacePlane*> planes);

protected:
  // For high bit depth ancestors;
  SurfaceRGBPlanar(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
                   CUcontext context);
};

/* 32-bit float planar RGB image;
 */
class TC_EXPORT SurfaceRGB32FPlanar final : public SurfaceRGBPlanar {
public:
  virtual ~SurfaceRGB32FPlanar() = default;
  SurfaceRGB32FPlanar(const SurfaceRGB32FPlanar& other) = delete;
  SurfaceRGB32FPlanar(SurfaceRGB32FPlanar&& other) = delete;
  SurfaceRGB32FPlanar& operator=(const SurfaceRGB32FPlanar& other) = delete;
  SurfaceRGB32FPlanar& operator=(SurfaceRGB32FPlanar&& other) = delete;

  SurfaceRGB32FPlanar();
  SurfaceRGB32FPlanar(uint32_t width, uint32_t height, CUcontext context);

  Surface* Create();
  Pixel_Format PixelFormat() const { return RGB_32F_PLANAR; }
  uint32_t ElemSize() const { return sizeof(float); }

  DLDataTypeCode DataType() const { return kDLFloat; }
  std::string TypeStr() const { return "<f4"; }
};
} // namespace VPF