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

#include "Surfaces.hpp"
#include <algorithm>

static bool ValidatePlanes(std::initializer_list<SurfacePlane*> planes,
                           size_t elem_size, size_t num_planes) {
  if (planes.size() < num_planes) {
    return false;
  }

  const auto planes_not_empty =
      std::all_of(planes.begin(), planes.begin() + num_planes,
                  [](SurfacePlane* plane) { return nullptr != plane; });

  if (!planes_not_empty) {
    return false;
  }

  const auto same_elem_size = std::all_of(
      planes.begin(), planes.begin() + num_planes,
      [&](SurfacePlane* plane) { return plane->ElemSize() == elem_size; });

  if (!same_elem_size) {
    return false;
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
  m_planes.emplace_back(width, height, ElemSize(), DataType(), TypeStr(),
                        context);
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

CUdeviceptr SurfaceY::PixelPtr(uint32_t component) {
  return m_planes.at(component).GpuMem();
}

SurfacePlane& SurfaceY::GetSurfacePlane(uint32_t plane) {
  return m_planes.at(plane);
}

bool SurfaceY::Update(SurfacePlane& newPlane) { return Update({&newPlane}); }

bool SurfaceY::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 1)) {
    return false;
  }

  // That's ugly!
  m_planes.at(0U) = **planes.begin();
  return true;
}

DLManagedTensor* SurfaceY::ToDLPack() { return m_planes.begin()->ToDLPack(); }

void SurfaceY::ToCAI(CudaArrayInterfaceDescriptor& cai) {
  auto& plane = GetSurfacePlane();
  plane.ToCAI(cai);
}

SurfaceNV12::SurfaceNV12() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceNV12(width, height, ElemSize(), DataType(), context) {}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height,
                         uint32_t hbd_elem_size, DLDataTypeCode code,
                         CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width, height * 3 / 2, hbd_elem_size, code, TypeStr(),
                        context);
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

  throw std::invalid_argument("Invalid plane number");
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

  throw std::invalid_argument("Invalid plane number");
}

uint32_t SurfaceNV12::Pitch(uint32_t plane) const {
  switch (plane) {
  case 0:
  case 1:
    return m_planes.at(0U).Pitch();
  default:
    break;
  }

  throw std::invalid_argument("Invalid plane number");
}

CUdeviceptr SurfaceNV12::PixelPtr(uint32_t component) {
  if (component < NumComponents()) {
    return m_planes.at(0U).GpuMem() + component * Height() * Pitch();
  }

  throw std::invalid_argument("Invalid component number");
}

bool SurfaceNV12::Update(SurfacePlane& newPlane) { return Update({&newPlane}); }

bool SurfaceNV12::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 1)) {
    return false;
  }

  m_planes.at(0U) = **planes.begin();
  return true;
}

SurfacePlane& SurfaceNV12::GetSurfacePlane(uint32_t plane) {
  if (plane < NumPlanes()) {
    return m_planes.at(0U);
  }

  throw std::invalid_argument("Invalid plane number");
}

DLManagedTensor* SurfaceNV12::ToDLPack() {
  /* NV12 is semi-planar which means it either has to be packed and shipped
   * outside VALI as is or it has to be split in 2 tensors, one for Y and
   * another for UV planes. Here we decide to pack it as-is;
   */
  return m_planes.begin()->ToDLPack();
}

void SurfaceNV12::ToCAI(CudaArrayInterfaceDescriptor& cai) {
  auto& plane = GetSurfacePlane();
  plane.ToCAI(cai);
}

SurfaceP10::SurfaceP10() : SurfaceNV12() {}

SurfaceP10::SurfaceP10(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceNV12(width, height, ElemSize(), DataType(), context) {}

Surface* SurfaceP10::Create() { return new SurfaceP10; }

SurfaceP12::SurfaceP12() : SurfaceNV12() {}

SurfaceP12::SurfaceP12(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceNV12(width, height, ElemSize(), DataType(), context) {}

Surface* SurfaceP12::Create() { return new SurfaceP12; }

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
  m_planes.emplace_back(width, height, ElemSize(), DataType(), TypeStr(),
                        context);
  // U and V planes, decimated size;
  m_planes.emplace_back(width / 2, height / 2, ElemSize(), DataType(),
                        TypeStr(), context);
  m_planes.emplace_back(width / 2, height / 2, ElemSize(), DataType(),
                        TypeStr(), context);
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

CUdeviceptr SurfaceYUV420::PixelPtr(uint32_t component) {
  return m_planes.at(component).GpuMem();
}

bool SurfaceYUV420::Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
                           SurfacePlane& newPlaneV) {
  return Update({&newPlaneY, &newPlaneU, &newPlaneV});
}

bool SurfaceYUV420::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 3)) {
    return false;
  }

  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.at(i) = **(planes.begin() + i);
  }

  return true;
}

SurfacePlane& SurfaceYUV420::GetSurfacePlane(uint32_t plane) {
  return m_planes.at(plane);
}

SurfaceYUV420::SurfaceYUV420(uint32_t width, uint32_t height,
                             uint32_t hbd_elem_size, DLDataTypeCode code,
                             CUcontext context) {
  m_planes.clear();
  /* Need to reserve place, otherwise vector may reallocate and SurfacePlane
   * instances will be copied to new address loosing the memory ownership. Sic!
   */
  m_planes.reserve(NumPlanes());

  // Y plane
  m_planes.emplace_back(width, height, hbd_elem_size, code, TypeStr(), context);

  // U and V planes, decimated size
  m_planes.emplace_back(width / 2, height / 2, hbd_elem_size, code, TypeStr(),
                        context);
  m_planes.emplace_back(width / 2, height / 2, hbd_elem_size, code, TypeStr(),
                        context);
}

SurfaceYUV420_10bit::SurfaceYUV420_10bit() : SurfaceYUV420() {}

SurfaceYUV420_10bit::SurfaceYUV420_10bit(uint32_t width, uint32_t height,
                                         CUcontext context)
    : SurfaceYUV420(width, height, ElemSize(), DataType(), context) {}

Surface* SurfaceYUV420_10bit::Create() { return new SurfaceYUV420_10bit; }

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
  m_planes.emplace_back(width, height, ElemSize(), DataType(), TypeStr(),
                        context);
  // U and V planes, decimated size;
  m_planes.emplace_back(width / 2, height, ElemSize(), DataType(), TypeStr(),
                        context);
  m_planes.emplace_back(width / 2, height, ElemSize(), DataType(), TypeStr(),
                        context);
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

CUdeviceptr SurfaceYUV422::PixelPtr(uint32_t component) {
  return m_planes.at(component).GpuMem();
}

SurfacePlane& SurfaceYUV422::GetSurfacePlane(uint32_t plane) {
  return m_planes.at(plane);
}

bool SurfaceYUV422::Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
                           SurfacePlane& newPlaneV) {
  return Update({&newPlaneY, &newPlaneU, &newPlaneV});
}

bool SurfaceYUV422::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 3)) {
    return false;
  }

  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.at(i) = **(planes.begin() + i);
  }

  return true;
}

SurfaceYUV444::SurfaceYUV444() {
  m_planes.clear();
  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.emplace_back();
  }
}

SurfaceYUV444::SurfaceYUV444(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceYUV444(width, height, ElemSize(), DataType(), context) {}

SurfaceYUV444::SurfaceYUV444(uint32_t width, uint32_t height,
                             uint32_t hbd_elem_size, DLDataTypeCode code,
                             CUcontext context) {
  m_planes.clear();
  /* Need to reserve place, otherwise vector may reallocate and SurfacePlane
   * instances will be copied to new address loosing the memory ownership. Sic!
   */
  m_planes.reserve(NumPlanes());
  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.emplace_back(width, height, hbd_elem_size, code, TypeStr(),
                          context);
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

CUdeviceptr SurfaceYUV444::PixelPtr(uint32_t component) {
  return GetSurfacePlane(component).GpuMem();
}

SurfacePlane& SurfaceYUV444::GetSurfacePlane(uint32_t plane) {
  return m_planes.at(plane);
}

bool SurfaceYUV444::Update(SurfacePlane& newPlaneY, SurfacePlane& newPlaneU,
                           SurfacePlane& newPlaneV) {
  return Update({&newPlaneY, &newPlaneU, &newPlaneV});
}

bool SurfaceYUV444::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 3)) {
    return false;
  }

  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.at(i) = **(planes.begin() + i);
  }

  return true;
}

SurfaceYUV444_10bit::SurfaceYUV444_10bit() : SurfaceYUV444() {}

SurfaceYUV444_10bit::SurfaceYUV444_10bit(uint32_t width, uint32_t height,
                                         CUcontext context)
    : SurfaceYUV444(width, height, ElemSize(), DataType(), context) {}

Surface* SurfaceYUV444_10bit::Create() { return new SurfaceYUV444_10bit; }

SurfaceRGB::SurfaceRGB() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceRGB::SurfaceRGB(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceRGB(width, height, ElemSize(), DataType(), context) {}

SurfaceRGB::SurfaceRGB(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
                       DLDataTypeCode code, CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width * 3, height, hbd_elem_size, code, TypeStr(),
                        context);
}

Surface* SurfaceRGB::Create() { return new SurfaceRGB; }

uint32_t SurfaceRGB::Width(uint32_t plane) const {
  return m_planes.at(plane).Width() / 3;
}

uint32_t SurfaceRGB::WidthInBytes(uint32_t plane) const {
  return m_planes.at(plane).Width() * ElemSize();
}

uint32_t SurfaceRGB::Height(uint32_t plane) const {
  return m_planes.at(plane).Height();
}

uint32_t SurfaceRGB::Pitch(uint32_t plane) const {
  return m_planes.at(plane).Pitch();
}

CUdeviceptr SurfaceRGB::PixelPtr(uint32_t component) {
  return m_planes.at(component).GpuMem();
}

SurfacePlane& SurfaceRGB::GetSurfacePlane(uint32_t plane) {
  return m_planes.at(plane);
}

bool SurfaceRGB::Update(SurfacePlane& newPlane) { return Update({&newPlane}); }

bool SurfaceRGB::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 1)) {
    return false;
  }

  m_planes.at(0U) = **planes.begin();
  return true;
}

DLManagedTensor* SurfaceRGB::ToDLPack() {
  auto dlmt = m_planes.begin()->ToDLPack();

  /* Re-pack tensor partially because SurfacePlane doesn't store information
   * about pixel format.
   */
  try {
    dlmt->dl_tensor.ndim = 3;

    delete[] dlmt->dl_tensor.shape;
    dlmt->dl_tensor.shape = new int64_t[dlmt->dl_tensor.ndim];
    dlmt->dl_tensor.shape[0] = Height();
    dlmt->dl_tensor.shape[1] = Width();
    dlmt->dl_tensor.shape[2] = 3U;

    delete[] dlmt->dl_tensor.strides;
    dlmt->dl_tensor.strides = new int64_t[dlmt->dl_tensor.ndim];
    // Distance between rows within single picture;
    dlmt->dl_tensor.strides[0] = Pitch() / ElemSize();
    // Distance between pixels within single row;
    dlmt->dl_tensor.strides[1] = 3U;
    // Distance between components (R, G, B) within single pixel;
    dlmt->dl_tensor.strides[2] = 1U;
  } catch (...) {
    auto deleter = dlmt->deleter;
    deleter(dlmt);
    throw std::runtime_error("Failed to create DLManagedTensor");
  }

  return dlmt;
}

void SurfaceRGB::ToCAI(CudaArrayInterfaceDescriptor& cai) {
  auto& plane = GetSurfacePlane();
  plane.ToCAI(cai);

  cai.m_shape[0] = Height();
  cai.m_shape[1] = Width();
  cai.m_shape[2] = 3;

  cai.m_strides[0] = Pitch();
  cai.m_strides[1] = ElemSize() * 3;
  cai.m_strides[2] = ElemSize();
}

SurfaceBGR::SurfaceBGR() : SurfaceRGB() {}

SurfaceBGR::SurfaceBGR(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceRGB(width, height, context) {}

Surface* SurfaceBGR::Create() { return new SurfaceBGR; }

SurfaceRGB32F::SurfaceRGB32F() : SurfaceRGB() {}

SurfaceRGB32F::SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceRGB(width, height, ElemSize(), DataType(), context) {}

Surface* SurfaceRGB32F::Create() { return new SurfaceRGB32F; }

SurfaceRGBPlanar::SurfaceRGBPlanar() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                   CUcontext context)
    : SurfaceRGBPlanar(width, height, ElemSize(), DataType(), context) {}

SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                   uint32_t hbd_elem_size, DLDataTypeCode code,
                                   CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width, height * 3, hbd_elem_size, code, TypeStr(),
                        context);
}

Surface* SurfaceRGBPlanar::Create() { return new SurfaceRGBPlanar; }

uint32_t SurfaceRGBPlanar::Width(uint32_t plane) const {
  return m_planes.at(plane).Width();
}

uint32_t SurfaceRGBPlanar::WidthInBytes(uint32_t plane) const {
  return Width(plane) * ElemSize();
}

uint32_t SurfaceRGBPlanar::Height(uint32_t plane) const {
  return m_planes.at(plane).Height() / 3;
}

uint32_t SurfaceRGBPlanar::Pitch(uint32_t plane) const {
  return m_planes.at(plane).Pitch();
}

CUdeviceptr SurfaceRGBPlanar::PixelPtr(uint32_t component) {
  if (component < NumComponents()) {
    return m_planes.at(0U).GpuMem() + Height() * Pitch() * component;
  }

  return 0U;
}

SurfacePlane& SurfaceRGBPlanar::GetSurfacePlane(uint32_t plane) {
  return m_planes.at(plane);
}

bool SurfaceRGBPlanar::Update(SurfacePlane& newPlane) {
  return Update({&newPlane});
}

bool SurfaceRGBPlanar::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 1)) {
    return false;
  }

  m_planes.at(0U) = **planes.begin();
  return true;
}

DLManagedTensor* SurfaceRGBPlanar::ToDLPack() {
  auto dlmt = m_planes.begin()->ToDLPack();

  /* Re-pack tensor partially because SurfacePlane doesn't store information
   * about pixel format.
   */
  try {
    dlmt->dl_tensor.ndim = 3;

    delete[] dlmt->dl_tensor.shape;
    dlmt->dl_tensor.shape = new int64_t[dlmt->dl_tensor.ndim];
    dlmt->dl_tensor.shape[0] = 3U;
    dlmt->dl_tensor.shape[1] = Height();
    dlmt->dl_tensor.shape[2] = Width();

    delete[] dlmt->dl_tensor.strides;
    dlmt->dl_tensor.strides = new int64_t[dlmt->dl_tensor.ndim];
    // Distance between channels (R, G, B) within single picture;
    dlmt->dl_tensor.strides[0] = Pitch() * Height() / ElemSize();
    // Distance between rows within single channel;
    dlmt->dl_tensor.strides[1] = Pitch() / ElemSize();
    // Distance between elements within single row;
    dlmt->dl_tensor.strides[2] = 1U;
  } catch (...) {
    auto deleter = dlmt->deleter;
    deleter(dlmt);
    throw std::runtime_error("Failed to create DLManagedTensor");
  }

  return dlmt;
}

void SurfaceRGBPlanar::ToCAI(CudaArrayInterfaceDescriptor& cai) {
  auto& plane = GetSurfacePlane();
  plane.ToCAI(cai);

  cai.m_shape[0] = 3;
  cai.m_shape[1] = Height();
  cai.m_shape[2] = Width();

  cai.m_strides[0] = Pitch() * Height();
  cai.m_strides[1] = Pitch();
  cai.m_strides[2] = ElemSize();
}

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar() : SurfaceRGBPlanar() {}

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar(uint32_t width, uint32_t height,
                                         CUcontext context)
    : SurfaceRGBPlanar(width, height, ElemSize(), DataType(), context) {}

Surface* SurfaceRGB32FPlanar::Create() { return new SurfaceRGB32FPlanar; }