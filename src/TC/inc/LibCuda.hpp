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

#include "LibraryLoader.hpp"

#include <cuda.h>

class LibCuda {
private:
  static const char* const filename;
  static std::shared_ptr<LibraryLoader> LoadCuda();

public:
  // cuda.h:
  static LoadableFunction<LoadCuda, CUresult, unsigned int> cuInit;
  static LoadableFunction<LoadCuda, CUresult, CUresult, const char**>
      cuGetErrorName;
  static LoadableFunction<LoadCuda, CUresult, CUresult, const char**>
      cuGetErrorString;
  static LoadableFunction<LoadCuda, CUresult, CUdevice*, int> cuDeviceGet;
  static LoadableFunction<LoadCuda, CUresult, int*, CUdevice_attribute,
                          CUdevice>
      cuDeviceGetAttribute;
  static LoadableFunction<LoadCuda, CUresult, int*> cuDeviceGetCount;
  static LoadableFunction<LoadCuda, CUresult, CUstream*, unsigned int>
      cuStreamCreate;
  static LoadableFunction<LoadCuda, CUresult, CUstream> cuStreamSynchronize;
  static LoadableFunction<LoadCuda, CUresult, CUstream> cuStreamDestroy;
  static LoadableFunction<LoadCuda, CUresult, CUcontext*, CUdevice>
      cuDevicePrimaryCtxRetain;
  static LoadableFunction<LoadCuda, CUresult, CUdevice>
      cuDevicePrimaryCtxRelease;
  static LoadableFunction<LoadCuda, CUresult, CUdevice*> cuCtxGetDevice;
  static LoadableFunction<LoadCuda, CUresult, CUcontext> cuCtxPushCurrent;
  static LoadableFunction<LoadCuda, CUresult, CUcontext*> cuCtxPopCurrent;
  static LoadableFunction<LoadCuda, CUresult, CUdeviceptr*, size_t> cuMemAlloc;
  static LoadableFunction<LoadCuda, CUresult, CUdeviceptr*, size_t*, size_t,
                          size_t, unsigned int>
      cuMemAllocPitch;
  static LoadableFunction<LoadCuda, CUresult, void**, size_t> cuMemAllocHost;
  static LoadableFunction<LoadCuda, CUresult, CUdeviceptr, CUdeviceptr, size_t>
      cuMemcpyDtoD_v2;
  static LoadableFunction<LoadCuda, CUresult, CUdeviceptr, CUdeviceptr, size_t,
                          CUstream>
      cuMemcpyDtoDAsync_v2;
  static LoadableFunction<LoadCuda, CUresult, void*, CUdeviceptr, size_t,
                          CUstream>
      cuMemcpyDtoHAsync_v2;
  static LoadableFunction<LoadCuda, CUresult, CUdeviceptr, const void*, size_t,
                          CUstream>
      cuMemcpyHtoDAsync_v2;
  static LoadableFunction<LoadCuda, CUresult, const CUDA_MEMCPY2D*, CUstream>
      cuMemcpy2DAsync_v2;
  static LoadableFunction<LoadCuda, CUresult, void*> cuMemFreeHost;
  static LoadableFunction<LoadCuda, CUresult, CUdeviceptr> cuMemFree;
  static LoadableFunction<LoadCuda, CUresult, void*, CUpointer_attribute,
                          CUdeviceptr>
      cuPointerGetAttribute;
  static LoadableFunction<LoadCuda, CUresult, CUstream, CUhostFn, void*>
      cuLaunchHostFunc;
  static LoadableFunction<LoadCuda, CUresult, CUstream, CUcontext*>
      cuStreamGetCtx;
};
