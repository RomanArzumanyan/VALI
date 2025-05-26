/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2025 Vision Labs LLC
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

#include "ResizeUtils.hpp"
#include <cuda_runtime.h>
#include <stdint.h>

using namespace VPF;

template <typename T>
static __global__ void
RescaleConvertYUV(cudaTextureObject_t tex_y, cudaTextureObject_t tex_uv,
                  uint8_t* dst_y, uint8_t* dst_u, uint8_t* dst_v, int pitch,
                  int width, int height, float scale_x, float scale_y) {

  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  typedef decltype(T::x) channel;
  const int MAX = 1 << (sizeof(channel) * 8);

  float luma = tex2D<float>(tex_y, x / scale_x, y / scale_y);
  float2 chroma = tex2D<float2>(tex_uv, x / (scale_x * 2), y / (scale_y * 2));

  auto const pos_bytes = y * pitch + x * sizeof(channel);
  *(channel*)(dst_y + pos_bytes) = (channel)(luma * MAX);
  *(channel*)(dst_u + pos_bytes) = (channel)(chroma.x * MAX);
  *(channel*)(dst_v + pos_bytes) = (channel)(chroma.y * MAX);
}

template <typename T>
__device__ void Denormalize(float& x, float& y, float& z) {
  const int MAX = 1 << (sizeof(T) * 8);

  x *= MAX;
  y *= MAX;
  z *= MAX;
}

template <> __device__ void Denormalize<float>(float& x, float& y, float& z) {}

template <typename T>
static __global__ void
RescaleConvertRGB(cudaTextureObject_t tex_y, cudaTextureObject_t tex_uv,
                  uint8_t* dst_r, uint8_t* dst_g, uint8_t* dst_b, int pitch,
                  int width, int height, float scale_x, float scale_y) {

  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float luma = tex2D<float>(tex_y, x / scale_x, y / scale_y);
  float2 chroma = tex2D<float2>(tex_uv, x / (scale_x * 2), y / (scale_y * 2));

  float n_y = luma;
  float n_u = chroma.x - .5f;
  float n_v = chroma.y - .5f;

  float n_r = n_y + 1.140f * n_v;
  float n_g = n_y - 0.394f * n_u - 0.581f * n_v;
  float n_b = n_y + 2.032f * n_u;

  Denormalize<T>(n_r, n_g, n_b);

  if (dst_g && dst_b) {
    // Planar RGB
    auto const pos_bytes = y * pitch + x * sizeof(T);
    *(T*)(dst_r + pos_bytes) = (T)n_r;
    *(T*)(dst_g + pos_bytes) = (T)n_g;
    *(T*)(dst_b + pos_bytes) = (T)n_b;
  } else {
    // Packed RGB
    auto pos_bytes = y * pitch + x * sizeof(T) * 3;
    *(T*)(dst_r + pos_bytes) = (T)n_r;
    pos_bytes += sizeof(T);
    *(T*)(dst_r + pos_bytes) = (T)n_g;
    pos_bytes += sizeof(T);
    *(T*)(dst_r + pos_bytes) = (T)n_b;
  }
}

template <typename T>
static void Impl(CUdeviceptr dst_x, CUdeviceptr dst_y, CUdeviceptr dst_z,
                 int dst_pitch, int dst_width, int dst_height,
                 CUdeviceptr src_nv12, int src_pitch, int src_width,
                 int src_height, cudaStream_t stream, Pixel_Format fmt) {

  cudaResourceDesc res_desc = {};
  res_desc.resType = cudaResourceTypePitch2D;
  res_desc.res.pitch2D.devPtr = (uint8_t*)src_nv12;
  res_desc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(T::x)>();
  res_desc.res.pitch2D.width = src_width;
  res_desc.res.pitch2D.height = src_height;
  res_desc.res.pitch2D.pitchInBytes = src_pitch;

  cudaTextureDesc tex_desc = {};
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.readMode = cudaReadModeNormalizedFloat;

  cudaTextureObject_t tex_y = 0;
  cudaCreateTextureObject(&tex_y, &res_desc, &tex_desc, NULL);

  res_desc.res.pitch2D.desc = cudaCreateChannelDesc<T>();
  res_desc.res.pitch2D.devPtr = (uint8_t*)src_nv12 + src_pitch * src_height;
  res_desc.res.pitch2D.width = src_width / 2;
  res_desc.res.pitch2D.height = src_height / 2;

  cudaTextureObject_t tex_uv = 0;
  cudaCreateTextureObject(&tex_uv, &res_desc, &tex_desc, NULL);

  dim3 Dg = dim3((dst_width + 15) / 16, (dst_height + 15) / 16);
  dim3 Db = dim3(16, 16);

  switch (fmt) {
  case RGB:
  case RGB_PLANAR:
    RescaleConvertRGB<uint8_t><<<Dg, Db, 0, stream>>>(
        tex_y, tex_uv, (uint8_t*)dst_x, (uint8_t*)dst_y, (uint8_t*)dst_z,
        dst_pitch, dst_width, dst_height, 1.0f * dst_width / src_width,
        1.0f * dst_height / src_height);
    break;
  case RGB_32F:
  case RGB_32F_PLANAR:
    RescaleConvertRGB<float><<<Dg, Db, 0, stream>>>(
        tex_y, tex_uv, (uint8_t*)dst_x, (uint8_t*)dst_y, (uint8_t*)dst_z,
        dst_pitch, dst_width, dst_height, 1.0f * dst_width / src_width,
        1.0f * dst_height / src_height);
    break;
  case YUV444:
  case YUV444_10bit:
    RescaleConvertYUV<T><<<Dg, Db, 0, stream>>>(
        tex_y, tex_uv, (uint8_t*)dst_x, (uint8_t*)dst_y, (uint8_t*)dst_z,
        dst_pitch, dst_width, dst_height, 1.0f * dst_width / src_width,
        1.0f * dst_height / src_height);
    break;
  default:
    break;
  }

  cudaDestroyTextureObject(tex_y);
  cudaDestroyTextureObject(tex_uv);
}

void UD_NV12(CUdeviceptr dst_x, CUdeviceptr dst_y, CUdeviceptr dst_z,
             int dst_pitch, int dst_width, int dst_height, CUdeviceptr src_nv12,
             int src_pitch, int src_width, int src_height, cudaStream_t stream,
             Pixel_Format fmt) {

  return Impl<uchar2>(dst_x, dst_y, dst_z, dst_pitch, dst_width, dst_height,
                      src_nv12, src_pitch, src_width, src_height, stream, fmt);
}

void UD_NV12_HBD(CUdeviceptr dst_x, CUdeviceptr dst_y, CUdeviceptr dst_z,
                 int dst_pitch, int dst_width, int dst_height,
                 CUdeviceptr src_nv12, int src_pitch, int src_width,
                 int src_height, cudaStream_t stream, Pixel_Format fmt) {

  return Impl<ushort2>(dst_x, dst_y, dst_z, dst_pitch, dst_width, dst_height,
                       src_nv12, src_pitch, src_width, src_height, stream, fmt);
}
