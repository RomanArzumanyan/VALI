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

/// @brief Upsample + Downscale NV12 Surface
/// @param dpDstX pointer to 1st image channel
/// @param dpDstY pointer to 2nd image channel, may be NULL for packed format
/// @param dpDstZ pointer to 3rd image channel, may be NULL for packed format
/// @param nDstPitch dst Surface pitch in bytes
/// @param nDstWidth dst Surface width in pixels
/// @param nDstHeight dst Surface height in pixels
/// @param dpSrcNv12 pointer to src Surface GPU memory
/// @param nSrcPitch src Surface pitch in bytes
/// @param nSrcWidth src Surface width in pixels
/// @param nSrcHeight src Surface height in pixels
/// @param stream CUDA stream
/// @param fmt output Surface format. One of YUV444 or RGB formats
void UD_NV12(CUdeviceptr dpDstX, CUdeviceptr dpDstY, CUdeviceptr dpDstZ,
             int nDstPitch, int nDstWidth, int nDstHeight,
             CUdeviceptr dpSrcNv12, int nSrcPitch, int nSrcWidth,
             int nSrcHeight, cudaStream_t stream, Pixel_Format fmt);

/// @brief Upsample + Downscale high bit depth NV12 Surface
/// @param dpDstX pointer to 1st image channel
/// @param dpDstY pointer to 2nd image channel, may be NULL for packed format
/// @param dpDstZ pointer to 3rd image channel, may be NULL for packed format
/// @param nDstPitch dst Surface pitch in bytes
/// @param nDstWidth dst Surface width in pixels
/// @param nDstHeight dst Surface height in pixels
/// @param dpSrcNv12 pointer to src Surface GPU memory
/// @param nSrcPitch src Surface pitch in bytes
/// @param nSrcWidth src Surface width in pixels
/// @param nSrcHeight src Surface height in pixels
/// @param stream CUDA stream
/// @param fmt output Surface format. One of YUV444 or RGB formats
void UD_NV12_HBD(CUdeviceptr dpDstY, CUdeviceptr dpDstU, CUdeviceptr dpDstV,
                 int nDstPitch, int nDstWidth, int nDstHeight,
                 CUdeviceptr dpSrcNv12, int nSrcPitch, int nSrcWidth,
                 int nSrcHeight, cudaStream_t stream, Pixel_Format fmt);