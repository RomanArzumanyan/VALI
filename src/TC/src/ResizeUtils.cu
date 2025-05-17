/*
 * Copyright 2019 NVIDIA Corporation
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

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
static __global__ void Resize(cudaTextureObject_t tex_y,
                              cudaTextureObject_t tex_uv, void* dst_y,
                              void* dst_u, void* dst_v, int pitch, int width,
                              int height, float scale_x, float scale_y) {

  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  typedef decltype(T::x) channel;
  const int MAX = 1 << (sizeof(channel) * 8);

  float luma = tex2D<float>(tex_y, x / scale_x, y / scale_y);
  float2 chroma = tex2D<float2>(tex_uv, x / (scale_x * 2), y / (scale_y * 2));

  auto y_dst = (channel*)dst_y, u_dst = (channel*)dst_u,
       v_dst = (channel*)dst_v;

  y_dst[y * pitch + x * sizeof(channel)] = (channel)(luma * MAX);
  u_dst[y * pitch + x * sizeof(channel)] = (channel)(chroma.x * MAX);
  v_dst[y * pitch + x * sizeof(channel)] = (channel)(chroma.y * MAX);
}

template <typename T>
static void ResizeImpl(CUdeviceptr dst_y, CUdeviceptr dst_u, CUdeviceptr dst_v,
                       int dst_picth, int dst_width, int dst_height,
                       CUdeviceptr src_nv12, int src_pitch, int src_width,
                       int src_height, cudaStream_t stream) {

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

  Resize<T><<<Dg, Db, 0, stream>>>(tex_y, tex_uv, (void*)dst_y, (void*)dst_u,
                                   (void*)dst_v, dst_picth, dst_width,
                                   dst_height, 1.0f * dst_width / src_width,
                                   1.0f * dst_height / src_height);

  cudaDestroyTextureObject(tex_y);
  cudaDestroyTextureObject(tex_uv);
}

void UD_NV12(CUdeviceptr dst_y, CUdeviceptr dst_u, CUdeviceptr dst_v,
             int dst_picth, int dst_width, int dst_height, CUdeviceptr src_nv12,
             int src_pitch, int src_width, int src_height,
             cudaStream_t stream) {

  return ResizeImpl<uchar2>(dst_y, dst_u, dst_v, dst_picth, dst_width,
                            dst_height, src_nv12, src_pitch, src_width,
                            src_height, stream);
}

void UD_NV12_HBD(CUdeviceptr dst_y, CUdeviceptr dst_u, CUdeviceptr dst_v,
                 int dst_picth, int dst_width, int dst_height,
                 CUdeviceptr src_nv12, int src_pitch, int src_width,
                 int src_height, cudaStream_t stream) {

  return ResizeImpl<ushort2>(dst_y, dst_u, dst_v, dst_picth, dst_width,
                             dst_height, src_nv12, src_pitch, src_width,
                             src_height, stream);
}