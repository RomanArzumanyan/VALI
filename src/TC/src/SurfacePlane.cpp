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

#include "SurfacePlane.hpp"
#include <cstring>
#include <sstream>
#include <string>

namespace VPF {

static void DLManagedTensor_Destroy(DLManagedTensor* self) {
  if (!self) {
    return;
  }

  if (self->dl_tensor.shape) {
    delete[] self->dl_tensor.shape;
  }

  if (self->dl_tensor.strides) {
    delete[] self->dl_tensor.strides;
  }

  delete self;
}

SurfacePlane& SurfacePlane::operator=(const SurfacePlane& other) noexcept {
  MakeBlank();

  m_own_mem = false;
  m_width = other.Width();
  m_height = other.Height();
  m_pitch = other.Pitch();
  m_elem_size = other.ElemSize();
  m_borrowed_gpu_mem = other.GpuMemImpl();
  m_dlpack_ctx = other.m_dlpack_ctx;

  return *this;
}

SurfacePlane::SurfacePlane(const SurfacePlane& other) noexcept
    : m_own_mem(false), m_borrowed_gpu_mem(other.GpuMemImpl()),
      m_width(other.Width()), m_height(other.Height()), m_pitch(other.Pitch()),
      m_elem_size(other.ElemSize()), m_dlpack_ctx(other.m_dlpack_ctx) {}

SurfacePlane& SurfacePlane::operator=(SurfacePlane&& other) noexcept {
  MakeBlank();

  m_own_mem = other.OwnMemory();
  if (m_own_mem) {
    m_own_gpu_mem = other.GpuMemImpl();
  } else {
    m_borrowed_gpu_mem = other.GpuMemImpl();
  }

  m_width = other.Width();
  m_height = other.Height();
  m_pitch = other.Pitch();
  m_elem_size = other.ElemSize();
  m_dlpack_ctx = other.m_dlpack_ctx;

  other.MakeBlank();
  return *this;
}

SurfacePlane::SurfacePlane(SurfacePlane&& other) noexcept
    : m_width(other.Width()), m_height(other.Height()), m_pitch(other.Pitch()),
      m_elem_size(other.ElemSize()), m_dlpack_ctx(other.m_dlpack_ctx) {
  if (m_own_mem) {
    m_own_gpu_mem = other.GpuMemImpl();
  } else {
    m_borrowed_gpu_mem = other.GpuMemImpl();
  }

  other.MakeBlank();
}

SurfacePlane::SurfacePlane(const DLManagedTensor& dlmt) {
  auto const ndim = dlmt.dl_tensor.ndim;
  if (ndim != 2) {
    throw std::runtime_error("Only 2D tensors are supported.");
  }

  auto const device_type = dlmt.dl_tensor.device.device_type;
  if (device_type != kDLCUDA) {
    throw std::runtime_error("Only kDLCUDA tensors are supported.");
  }

  if (dlmt.dl_tensor.dtype.lanes != 1) {
    throw std::runtime_error("Only 1 lane tensors are supported.");
  }

  if ((dlmt.dl_tensor.dtype.code != kDLUInt) &&
      (dlmt.dl_tensor.dtype.code != kDLFloat)) {
    throw std::runtime_error(
        "Only kDLUInt and kDLFloat tensors are supported.");
  }

  m_own_mem = false;
  m_elem_size = dlmt.dl_tensor.dtype.bits / 8;
  m_width = dlmt.dl_tensor.shape[1];
  m_height = dlmt.dl_tensor.shape[0];
  m_pitch = dlmt.dl_tensor.strides[0] * m_elem_size;
  m_dlpack_ctx.m_type_code = (DLDataTypeCode)dlmt.dl_tensor.dtype.code;
  m_dlpack_ctx.m_ptr =
      (CUdeviceptr)dlmt.dl_tensor.data + dlmt.dl_tensor.byte_offset;
}

SurfacePlane::SurfacePlane(uint32_t width, uint32_t height, uint32_t elem_size,
                           DLDataTypeCode type_code, CUcontext context,
                           bool pitched) {
  m_own_mem = true;
  m_width = width;
  m_height = height;
  m_elem_size = elem_size;
  m_dlpack_ctx.m_type_code = type_code;

  Allocate(context, pitched);
}

SurfacePlane::~SurfacePlane() { MakeBlank(); }

void SurfacePlane::Allocate(CUcontext context, bool pitched) {
  if (!OwnMemory()) {
    throw std::runtime_error("Can't allocate memory without ownership.");
  }

  CUdeviceptr gpu_mem = 0U;
  CudaCtxPush ctxPush(context);
  if (pitched) {
    ThrowOnCudaError(LibCuda::LibCuda::cuMemAllocPitch(&gpu_mem, &m_pitch,
                                                       m_width * ElemSize(),
                                                       m_height, 16),
                     __LINE__);
  } else {
    ThrowOnCudaError(
        LibCuda::cuMemAlloc(&gpu_mem, m_width * ElemSize() * m_height),
        __LINE__);
    m_pitch = m_width * ElemSize();
  }

  // Make sure we don't throw in deleter.
  m_own_gpu_mem = std::shared_ptr<void>((void*)gpu_mem, [](void* dptr) {
    try {
      CudaCtxPush ctxPush(GetContextByDptr((CUdeviceptr)dptr));
      LibCuda::cuMemFree((CUdeviceptr)dptr);
    } catch (...) {
    }
  });
}

void SurfacePlane::MakeBlank() noexcept {
  try {
    m_own_gpu_mem.reset();
    m_borrowed_gpu_mem.reset();

    m_own_mem = false;
    m_width = 0U;
    m_height = 0U;
    m_pitch = 0U;
    m_elem_size = 0U;
    m_dlpack_ctx = {};
  } catch (...) {
  }
}

bool SurfacePlane::IsValid() const noexcept {
  try {
    if (OwnMemory()) {
      return m_own_gpu_mem != nullptr;
    } else if (FromDLPack()) {
      return true;
    } else {
      return (nullptr != m_borrowed_gpu_mem.lock());
    }
  } catch (...) {
    return false;
  }
}

DLManagedTensor* SurfacePlane::DLPackContext::ToDLPack(
    uint32_t width, uint32_t height, uint32_t pitch, uint32_t elem_size,
    CUdeviceptr dptr, DLDataTypeCode type_code) {
  DLManagedTensor* dlmt = nullptr;
  try {
    dlmt = new DLManagedTensor();
    memset((void*)dlmt, 0, sizeof(*dlmt));

    dlmt->manager_ctx = nullptr;
    dlmt->deleter = DLManagedTensor_Destroy;

    dlmt->dl_tensor.device.device_type = kDLCUDA;
    dlmt->dl_tensor.device.device_id = GetDeviceIdByDptr(dptr);
    dlmt->dl_tensor.data = (void*)GetDevicePointer(dptr);
    dlmt->dl_tensor.ndim = 2;
    dlmt->dl_tensor.byte_offset = 0U;

    dlmt->dl_tensor.dtype.code = type_code;
    dlmt->dl_tensor.dtype.bits = elem_size * 8U;
    dlmt->dl_tensor.dtype.lanes = 1;

    dlmt->dl_tensor.shape = new int64_t[dlmt->dl_tensor.ndim];
    dlmt->dl_tensor.shape[0] = height;
    dlmt->dl_tensor.shape[1] = width;

    dlmt->dl_tensor.strides = new int64_t[dlmt->dl_tensor.ndim];
    dlmt->dl_tensor.strides[0] = pitch / elem_size;
    dlmt->dl_tensor.strides[1] = 1;
  } catch (std::exception& e) {
    DLManagedTensor_Destroy(dlmt);
    throw;
  } catch (...) {
    DLManagedTensor_Destroy(dlmt);
    throw std::runtime_error("Failed to create DLManagedTensor");
  }

  return dlmt;
}

std::shared_ptr<DLManagedTensor> SurfacePlane::DLPackContext::ToDLPackSmart(
    uint32_t width, uint32_t height, uint32_t pitch, uint32_t elem_size,
    CUdeviceptr dptr, DLDataTypeCode type_code) {
  auto dlmt_ptr = SurfacePlane::DLPackContext::ToDLPack(
      width, height, pitch, elem_size, dptr, type_code);

  return std::shared_ptr<DLManagedTensor>(dlmt_ptr, dlmt_ptr->deleter);
}

DLManagedTensor* SurfacePlane::ToDLPack() {
  if (FromDLPack()) {
    throw std::runtime_error("Cant put DLPack SurfacePlane to DLPack");
  }

  return SurfacePlane::DLPackContext::ToDLPack(Width(), Height(), Pitch(),
                                               ElemSize(), GpuMem(),
                                               m_dlpack_ctx.DataType());
}

std::shared_ptr<DLManagedTensor> SurfacePlane::ToDLPackSmart() {
  auto dlmt_ptr = SurfacePlane::ToDLPack();
  return std::shared_ptr<DLManagedTensor>(dlmt_ptr, dlmt_ptr->deleter);
}

CUdeviceptr SurfacePlane::GpuMem() const noexcept {
  try {
    if (FromDLPack()) {
      return m_dlpack_ctx.GpuMem();
    } else {
      auto mem_ptr = GpuMemImpl();
      if (!mem_ptr) {
        return (CUdeviceptr)0x0;
      }
      return (CUdeviceptr)mem_ptr.get();
    }
  } catch (...) {
    return (CUdeviceptr)0x0;
  }
}

CUcontext SurfacePlane::Context() const { return GetContextByDptr(GpuMem()); }

int SurfacePlane::DeviceId() const { return GetDeviceIdByDptr(GpuMem()); }

std::shared_ptr<void> SurfacePlane::GpuMemImpl() const {
  return OwnMemory() ? m_own_gpu_mem : m_borrowed_gpu_mem.lock();
}
} // namespace VPF