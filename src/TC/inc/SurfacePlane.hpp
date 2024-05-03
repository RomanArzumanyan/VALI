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

#include "CudaUtils.hpp"
#include "dlpack.h"
#include <memory>

namespace VPF {

/* 2D chunk of GPU memory located in vRAM. It doesn't have any pixel format.
 * Just a byte storage which can be easily shared via DLPack.
 */
class TC_EXPORT SurfacePlane {
  // GPU memory allocation that is owned by class instance;
  std::shared_ptr<void> m_own_gpu_mem;

  // Weak pointer to borrowed GPU memory;
  std::weak_ptr<void> m_borrowed_gpu_mem;

  bool m_own_mem = false;

  size_t m_width = 0U;
  size_t m_height = 0U;
  size_t m_pitch = 0U;
  size_t m_elem_size = 0U;

  /* Allocate memory if we own it;
   * GPU memory deallocation will follow usual weak_ptr logic;
   * May throw exception with reason in message;
   */
  void Allocate(CUcontext context = nullptr, bool pitched = true);

  /* Reset SurfacePlane to blank state.
   * GPU memory deallocation will follow usual weak_ptr logic;
   */
  void MakeBlank() noexcept;

  /* Return smart pointer to GPU memory.
   */
  std::shared_ptr<void> GpuMemImpl() const;

public:
  /* Everything DLPack about this SurfacePlane;
   */
  struct DLPackContext {
    /* Raw GPU mem pointer. If SurfacePlane has non-zero DLPack mem pointer,
     * it's GPU memory object lifetime is defined by DLPack specification, not
     * by C++ smart pointers logic;
     */
    CUdeviceptr m_ptr = 0U;

    /* DLPack data type code;
     */
    DLDataTypeCode m_type_code = kDLOpaqueHandle;

    /* Get DLPack data type code;
     */
    inline DLDataTypeCode DataType() const noexcept { return m_type_code; }

    /* Get DLPlack device type code;
     * Supports kDLCUDA only;
     */
    inline DLDeviceType DeviceType() const noexcept { return kDLCUDA; }

    /* Get raw GPU memory ptr;
     */
    inline CUdeviceptr GpuMem() const noexcept { return m_ptr; }

    /* Serialize raw data into DLPack managed tensor;
     * Returns raw pointer because DLPack has C spec. only;
     * May throw exception with reason in message;
     *
     * Caller is responsible of checking if provided data can be serialized
     * into DLPack (e. g. memory allocation is of sufficient size);
     */
    static DLManagedTensor* ToDLPack(uint32_t width, uint32_t height,
                                     uint32_t pitch, uint32_t elem_size,
                                     CUdeviceptr dptr,
                                     DLDataTypeCode type_code);

    /* Same as previous but wrapped in smart pointer. Handy to use inside C++
     * code, no need to mess with deleter;
     */
    static std::shared_ptr<DLManagedTensor>
    ToDLPackSmart(uint32_t width, uint32_t height, uint32_t pitch,
                  uint32_t elem_size, CUdeviceptr dptr,
                  DLDataTypeCode type_code);
  } m_dlpack_ctx;

  /* Blank plane, zero size. No memory ownership;
   */
  SurfacePlane() = default;

  /* Update from another, cease memory ownership;
   */
  SurfacePlane& operator=(const SurfacePlane& other) noexcept;

  /* Update from another, take memory ownership if another owns it;
   * Another object will cease memory ownership;
   */
  SurfacePlane& operator=(SurfacePlane&& other) noexcept;

  /* Construct from another, don't own memory;
   */
  SurfacePlane(const SurfacePlane& other) noexcept;

  /* Construct from another, take memory ownership is another owns it;
   * Another object will cease memory ownership;
   */
  SurfacePlane(SurfacePlane&& other) noexcept;

  /* Construct from DLPack, don't own memory.
   * May throw exception with reason in message;
   */
  SurfacePlane(const DLManagedTensor& dlmt);

  /* Construct & own memory. If null context is given, current context will be
   * used. May throw exception with reason in message;
   */
  SurfacePlane(uint32_t width, uint32_t height, uint32_t elem_size,
               DLDataTypeCode type_code, CUcontext context = nullptr,
               bool pitched = true);

  /* Destruct object. GPU memory deallocation will follow usual weak_ptr logic;
   */
  ~SurfacePlane();

  /* Returns true if memory (own or borrowed) isn't destroyed, false otherwise;
   * If created from DLPack will always return true;
   */
  bool IsValid() const noexcept;

  /* Returns true if SurfacePlane is created from DLPack;
   */
  inline bool FromDLPack() const noexcept {
    return (m_dlpack_ctx.GpuMem() != 0U);
  };

  /* Return true if instance owns the memory, false otherwise;
   */
  inline bool OwnMemory() const noexcept { return m_own_mem; };

  /* Return true if memory is allocated with pitch, false otherwise;
   */
  inline bool Pitched() const noexcept { return m_pitch != m_width; };

  /* Return CUdeviceptr of memory allocation.
   * If created from DLPack, it will be raw CUdeviceptr.
   * Will return 0x0 if no memory is accessible (empty SurfacePlane)
   */
  CUdeviceptr GpuMem() const noexcept;

  /* Get width in pixels;
   * Will return 0 if SurfacePlane is blank.
   */
  inline uint32_t Width() const noexcept { return m_width; };

  /* Get height in pixels;
   * Will return 0 if SurfacePlane is blank.
   */
  inline uint32_t Height() const noexcept { return m_height; };

  /* Get pitch (distance between 2 adjacent vertical points) in bytes;
   * Will return 0 if SurfacePlane is blank.
   */
  inline uint32_t Pitch() const noexcept { return m_pitch; };

  /* Get element size in bytes;
   * Will return 0 if SurfacePlane is blank.
   */
  inline uint32_t ElemSize() const noexcept { return m_elem_size; };

  /* Get contiguous 1D memory chunk size in bytes required to store image;
   * Will return 0 if SurfacePlane is blank.
   */
  inline uint32_t HostMemSize() const noexcept {
    return Width() * Height() * ElemSize();
  };

  /* Get associated DLPack context;
   */
  inline const DLPackContext& DLPackCtx() const noexcept {
    return m_dlpack_ctx;
  }

  inline DLPackContext& DLPackCtx() noexcept { return m_dlpack_ctx; }

  /* Get CUDA context associated with memory object (borrowed ot its own);
   * May throw exception with reason in message;
   */
  CUcontext Context() const;

  /* Get device ID associated with memory object;
   * May throw exception with reason in message;
   */
  int DeviceId() const;

  /* Serialize SurfacePlane into DLPack managed tensor;
   * Returns raw pointer because DLPack has C spec. only;
   * May throw exception with reason in message;
   */
  DLManagedTensor* ToDLPack();

  /* Same as previous but wrapped in smart pointer. Handy to use inside C++
   * code, no need to mess with deleter;
   */
  std::shared_ptr<DLManagedTensor> ToDLPackSmart();
};
} // namespace VPF