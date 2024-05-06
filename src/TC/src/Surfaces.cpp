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
#include <stdexcept>

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

bool SurfaceY::Update(SurfacePlane& newPlane) { return Update({&newPlane}); }

bool SurfaceY::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 1)) {
    return false;
  }

  // That's ugly!
  m_planes.at(0U) = **planes.begin();
  return true;
}

SurfaceNV12::SurfaceNV12() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceNV12(width, height, ElemSize(), context) {}

SurfaceNV12::SurfaceNV12(uint32_t width, uint32_t height,
                         uint32_t hbd_elem_size, CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width, height * 3 / 2, hbd_elem_size, DataType(),
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

CUdeviceptr SurfaceNV12::PlanePtr(uint32_t plane) {
  if (Empty()) {
    return 0x0;
  }

  if (plane < NumPlanes()) {
    return m_planes.at(0U).GpuMem() + plane * Height() * Pitch();
  }

  throw std::invalid_argument("Invalid plane number");
}

bool SurfaceNV12::Update(SurfacePlane& newPlane) { return Update({&newPlane}); }

bool SurfaceNV12::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 1)) {
    return false;
  }

  m_planes.at(0U) = **planes.begin();
  return true;
}

SurfacePlane* SurfaceNV12::GetSurfacePlane(uint32_t plane) {
  if (plane < NumPlanes()) {
    return &m_planes.at(0U);
  }

  return nullptr;
}

SurfaceP10::SurfaceP10() : SurfaceNV12() {}

SurfaceP10::SurfaceP10(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceNV12(width, height, ElemSize(), context) {}

Surface* VPF::SurfaceP10::Create() { return new SurfaceP10; }

SurfaceP12::SurfaceP12() : SurfaceNV12() {}

SurfaceP12::SurfaceP12(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceNV12(width, height, ElemSize(), context) {}

Surface* VPF::SurfaceP12::Create() { return new SurfaceP12; }

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
    : SurfaceYUV444(width, height, ElemSize(), context) {}

SurfaceYUV444::SurfaceYUV444(uint32_t width, uint32_t height,
                             uint32_t hbd_elem_size, CUcontext context) {
  m_planes.clear();
  /* Need to reserve place, otherwise vector may reallocate and SurfacePlane
   * instances will be copied to new address loosing the memory ownership. Sic!
   */
  m_planes.reserve(NumPlanes());
  for (auto i = 0; i < NumPlanes(); i++) {
    m_planes.emplace_back(width, height, hbd_elem_size, DataType(), context);
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
    : SurfaceYUV444(width, height, ElemSize(), context) {}

Surface* VPF::SurfaceYUV444_10bit::Create() { return new SurfaceYUV444_10bit; }

SurfaceRGB::SurfaceRGB() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceRGB::SurfaceRGB(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceRGB(width, height, ElemSize(), context) {}

SurfaceRGB::SurfaceRGB(uint32_t width, uint32_t height, uint32_t hbd_elem_size,
                       CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width * 3, height, hbd_elem_size, DataType(), context);
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

CUdeviceptr SurfaceRGB::PlanePtr(uint32_t plane) {
  return m_planes.at(plane).GpuMem();
}

SurfacePlane* SurfaceRGB::GetSurfacePlane(uint32_t plane) {
  return &m_planes.at(plane);
}

bool SurfaceRGB::Update(SurfacePlane& newPlane) { return Update({&newPlane}); }

bool SurfaceRGB::Update(std::initializer_list<SurfacePlane*> planes) {
  if (OwnMemory() || !ValidatePlanes(planes, ElemSize(), 1)) {
    return false;
  }

  m_planes.at(0U) = **planes.begin();
  return true;
}

SurfaceBGR::SurfaceBGR() : SurfaceRGB() {}

SurfaceBGR::SurfaceBGR(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceRGB(width, height, context) {}

Surface* VPF::SurfaceBGR::Create() { return new SurfaceBGR; }

SurfaceRGB32F::SurfaceRGB32F() : SurfaceRGB() {}

SurfaceRGB32F::SurfaceRGB32F(uint32_t width, uint32_t height, CUcontext context)
    : SurfaceRGB(width, height, ElemSize(), context) {}

Surface* VPF::SurfaceRGB32F::Create() { return new SurfaceRGB32F; }

SurfaceRGBPlanar::SurfaceRGBPlanar() {
  m_planes.clear();
  m_planes.emplace_back();
}

SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                   CUcontext context)
    : SurfaceRGBPlanar(width, height * 3, ElemSize(), context) {}

VPF::SurfaceRGBPlanar::SurfaceRGBPlanar(uint32_t width, uint32_t height,
                                        uint32_t hbd_elem_size,
                                        CUcontext context) {
  m_planes.clear();
  m_planes.emplace_back(width, height * 3, hbd_elem_size, DataType(), context);
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

CUdeviceptr SurfaceRGBPlanar::PlanePtr(uint32_t plane) {
  return m_planes.at(plane).GpuMem();
}

SurfacePlane* SurfaceRGBPlanar::GetSurfacePlane(uint32_t plane) {
  return &m_planes.at(plane);
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

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar() : SurfaceRGBPlanar() {}

SurfaceRGB32FPlanar::SurfaceRGB32FPlanar(uint32_t width, uint32_t height,
                                         CUcontext context)
    : SurfaceRGBPlanar(width, height, ElemSize(), context) {}

Surface* VPF::SurfaceRGB32FPlanar::Create() { return new SurfaceRGB32FPlanar; }