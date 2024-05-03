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
#include "NppCommon.hpp"
#include "SurfacePlane.hpp"
#include <vector>

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
  GRAY16 = 16
};

/* Represents GPU-side memory.
 * Pure interface class, see ancestors;
 */
class TC_EXPORT Surface : public Token {
public:
  virtual ~Surface() = default;

  /* Raw bytes + metadata;
   */
  struct SurfacePlaneContext {
    /* Ratio between SurfacePlane.Width() and Surface.Width();
     * E. g. will be 3. for RGB24;
     */
    float factor_x = 1.f;

    /* Ratio between SurfacePlane.Height() and Surface.Height();
     * E. g. will be .5 for U and V planes of YUV420 surface;
     */
    float factor_y = 1.f;

    /* Raw 2D memory chunk;
     */
    SurfacePlane m_plane;
  };

  /* Data often needed for NPP functions;
   */
  struct NppContext {
    /* CUDA device pointers to memory planes;
     */
    std::vector<CUdeviceptr> m_dptr;

    /* Pitch in bytes;
     */
    std::vector<int> m_step;

    /* Full frame size in pixels;*/
    NppiSize m_size;

    /* Cuda device pointers getter;
     */
    template <typename T> std::vector<T*> GetDataAs() {
      std::vector<T*> ret;
      for (auto dptr : m_dptr) {
        ret.push_back((T*)dptr);
      }

      return ret;
    }

    /* C-style pitch getter;
     */
    int* GetPitch() { return m_step.data(); }

    /* Size getter;
     */
    NppiSize GetSize() const noexcept { return m_size; }

    /* Get full size rect;
     */
    NppiRect GetRect() const noexcept {
      NppiSize size = GetSize();
      NppiRect rect = {0, 0, size.width, size.height};
      return rect;
    }
  };

  /* Iterator over planes;
   */
  class SurfacePlaneIterator {
    Surface* const m_surface;
    size_t m_idx;

  public:
    SurfacePlaneIterator(Surface* self, size_t idx)
        : m_surface(self), m_idx(idx) {}

    bool operator!=(const SurfacePlaneIterator& other) {
      return (m_surface != other.m_surface) && (m_idx != other.m_idx);
    }

    SurfacePlaneIterator operator++() {
      ++m_idx;
      return *this;
    }

    SurfacePlane& operator*() { return m_surface->GetSurfacePlane(m_idx); }
  };

  /* More iterator stuff;
   */
  SurfacePlaneIterator begin() { return SurfacePlaneIterator(this, 0U); }
  SurfacePlaneIterator end() { return SurfacePlaneIterator(this, NumPlanes()); }

  /* Returns number of image planes;
   * In case of failure return 0U;
   */
  virtual uint32_t NumPlanes() const noexcept = 0;

  /* Returns pixel format;
   */
  virtual Pixel_Format PixelFormat() const noexcept = 0;

  /* Returns element size in bytes;
   * In case of failure return 0U;
   */
  virtual size_t ElemSize() const noexcept = 0;

  /* Check if Surface is valid;
   */
  bool IsValid() const noexcept;

  /* Get surface plane by number;
   * Returns true in case of success, false otherwise;
   */
  bool GetSurfacePlane(SurfacePlane& plane,
                       uint32_t plane_number = 0U) noexcept;

  /* Returns width in pixels;
   * In case of failure return 0U;
   */
  uint32_t Width() const noexcept;

  /* Returns height in pixels;
   * In case of failure return 0U;
   */
  uint32_t Height() const noexcept;

  /* Returns true if surface is empty (no allocated data), false otherwise;
   */
  bool Empty() const noexcept;

  /* Returns true if all planes own their memory, false otherwise;
   */
  bool OwnMemory() const noexcept;

  /* Get CUDA context associated with all planes;
   * Returns 0x0 in case of failure;
   */
  CUcontext Context() const noexcept;

  /* Returns total amount of memory in bytes needed
   * to store all pixels of Surface in Host memory;
   * In case of failure return 0U;
   */
  size_t HostMemSize() const noexcept;

  /* Update from set of image planes, don't own the memory;
   * Returns true in case of success, false otherwise;
   * If update fails, Surface will be lefr intact with zero changes to it but
   * initializer list may become crippled;
   * Will return false if initializer list size doesn't match internal planes
   * number;
   * Will return false if planes in initializer list have different CUDA
   * contexts;
   */
  bool Update(std::initializer_list<SurfacePlane> planes) noexcept;

  /* Make a deep copy.
   * If non-zero CUDA stream is given, async CUDA memcpy will be run.
   * Caller shall take care of stream sync in such case.
   * Otherwise, blocking CUDA memcpy call will be issued.
   */
  std::shared_ptr<Surface> Clone(CUstream stream = 0U);

  /* Get NPP context with related data.
   * Usefull for all sorts of NPP operations;
   */
  NppContext GetNppContext();

  /* Get SurfacePlane by reference. May throw;
   */
  SurfacePlane& GetSurfacePlane(uint32_t plane_number = 0U);

  /* Copy Surface to single 2D memory chunk.
   * All Surface Planes must have same pitch equal to that of 2D chunk;
   */
  void ToChunk2D(CUdeviceptr dst, CUstream str, size_t dst_pitch,
                          bool async = false) const;

  /* Copy single 2D memory chunk to Surface.
   * All Surface Planes must have same pitch equal to that of 2D chunk;
   */
  void FromChunk2D(CUdeviceptr src, CUstream str, size_t src_pitch,
                            bool async = false);

  /* Make empty;
   */
  static std::shared_ptr<Surface> Make(Pixel_Format format);

  /* Make & own memory;
   */
  static std::shared_ptr<Surface> Make(Pixel_Format format, uint32_t width,
                                       uint32_t height, CUcontext context);

protected:
  /* Actual pixels stored here;
   */
  std::vector<Surface::SurfacePlaneContext> m_planes;

  /* Check if Surface has valid memory planes;
   */
  bool ValidatePlanes() const noexcept;

  Surface();
};
} // namespace VPF