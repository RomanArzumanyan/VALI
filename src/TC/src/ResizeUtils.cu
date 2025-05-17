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

#include <cuda_runtime.h>
#include <stdint.h>

template <typename YuvUnitx2>
static __global__ void
Resize(cudaTextureObject_t texY, cudaTextureObject_t texUv, uint8_t* pDstY,
       uint8_t* pDstU, uint8_t* pDstV, int nPitch, int nWidth, int nHeight,
       float fxScale, float fyScale) {
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nWidth || y >= nHeight)
    return;

  typedef decltype(YuvUnitx2::x) YuvUnit;
  const int MAX = 1 << (sizeof(YuvUnit) * 8);

  float luma = tex2D<float>(texY, x / fxScale, y / fyScale);
  float2 chroma = tex2D<float2>(texUv, x / (fxScale * 2), y / (fyScale * 2));

  pDstY[y * nPitch + x * sizeof(YuvUnit)] = (YuvUnit)(luma * MAX);
  pDstU[y * nPitch + x * sizeof(YuvUnit)] = (YuvUnit)(chroma.x * MAX);
  pDstV[y * nPitch + x * sizeof(YuvUnit)] = (YuvUnit)(chroma.y * MAX);
}

template <typename YuvUnitx2>
static void ResizeImpl(unsigned char* dpDstY, unsigned char* dpDstU,
                       unsigned char* dpDstV, int nDstPitch, int nDstWidth,
                       int nDstHeight, unsigned char* dpSrc, int nSrcPitch,
                       int nSrcWidth, int nSrcHeight, cudaStream_t S) {
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = dpSrc;
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<decltype(YuvUnitx2::x)>();
  resDesc.res.pitch2D.width = nSrcWidth;
  resDesc.res.pitch2D.height = nSrcHeight;
  resDesc.res.pitch2D.pitchInBytes = nSrcPitch;

  cudaTextureDesc texDesc = {};
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeNormalizedFloat;

  cudaTextureObject_t texY = 0;
  cudaCreateTextureObject(&texY, &resDesc, &texDesc, NULL);

  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<YuvUnitx2>();
  resDesc.res.pitch2D.devPtr = dpSrc + nSrcPitch * nSrcHeight;
  resDesc.res.pitch2D.width = nSrcWidth / 2;
  resDesc.res.pitch2D.height = nSrcHeight / 2;

  cudaTextureObject_t texUv = 0;
  cudaCreateTextureObject(&texUv, &resDesc, &texDesc, NULL);

  dim3 Dg = dim3((nDstWidth + 15) / 16, (nDstHeight + 15) / 16);
  dim3 Db = dim3(16, 16);
  Resize<YuvUnitx2><<<Dg, Db, 0, S>>>(
      texY, texUv, dpDstY, dpDstU, dpDstV, nDstPitch, nDstWidth, nDstHeight,
      1.0f * nDstWidth / nSrcWidth, 1.0f * nDstHeight / nSrcHeight);

  cudaDestroyTextureObject(texY);
  cudaDestroyTextureObject(texUv);
}

void UD_NV12(unsigned char* dpDstY, unsigned char* dpDstU,
             unsigned char* dpDstV, int nDstPitch, int nDstWidth,
             int nDstHeight, unsigned char* dpSrcNv12, int nSrcPitch,
             int nSrcWidth, int nSrcHeight, cudaStream_t S) {

  return ResizeImpl<uchar2>(dpDstY, dpDstU, dpDstV, nDstPitch, nDstWidth,
                            nDstHeight, dpSrcNv12, nSrcPitch, nSrcWidth,
                            nSrcHeight, S);
}