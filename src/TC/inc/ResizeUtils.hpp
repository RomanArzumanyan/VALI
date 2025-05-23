/*
 * Copyright 2025 Vision Labs LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *    http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or Resizeied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "Tasks.hpp"

void UD_NV12(CUdeviceptr dpDstY, CUdeviceptr dpDstU, CUdeviceptr dpDstV,
             int nDstPitch, int nDstWidth, int nDstHeight,
             CUdeviceptr dpSrcNv12, int nSrcPitch, int nSrcWidth,
             int nSrcHeight, cudaStream_t stream, Pixel_Format fmt);

void UD_NV12_HBD(CUdeviceptr dpDstY, CUdeviceptr dpDstU, CUdeviceptr dpDstV,
                 int nDstPitch, int nDstWidth, int nDstHeight,
                 CUdeviceptr dpSrcNv12, int nSrcPitch, int nSrcWidth,
                 int nSrcHeight, cudaStream_t stream, Pixel_Format fmt);