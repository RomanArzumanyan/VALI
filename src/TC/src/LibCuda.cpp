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

#include "LibCuda.hpp"
#include <sstream>
#include <string>

static const std::string getDynLibName(const char* cudaLibName) {
  std::stringstream ss;

#ifdef _WIN64
  ss << "nv" << cudaLibName << ".dll";
#else
  ss << "lib" << cudaLibName << ".so";
#endif
  return ss.str();
}

std::shared_ptr<LibraryLoader> LibCuda::LoadCuda() {
  auto const filename = getDynLibName("cuda");
  static LibraryLoader lib(filename.c_str());
  return std::shared_ptr<LibraryLoader>(std::shared_ptr<LibraryLoader>{}, &lib);
}

#define DEFINE(x, y)                                                           \
  decltype(x::y) x::y { #y }

// Define function pointers for cuda.h:
DEFINE(LibCuda, cuInit);
DEFINE(LibCuda, cuGetErrorName);
DEFINE(LibCuda, cuGetErrorString);
DEFINE(LibCuda, cuDeviceGet);
DEFINE(LibCuda, cuDeviceGetAttribute);
DEFINE(LibCuda, cuDeviceGetCount);
DEFINE(LibCuda, cuStreamCreate);
DEFINE(LibCuda, cuStreamSynchronize);
DEFINE(LibCuda, cuStreamDestroy_v2);
DEFINE(LibCuda, cuDevicePrimaryCtxRetain);
DEFINE(LibCuda, cuDevicePrimaryCtxRelease_v2);
DEFINE(LibCuda, cuCtxGetDevice);
DEFINE(LibCuda, cuCtxPushCurrent_v2);
DEFINE(LibCuda, cuCtxPopCurrent_v2);
DEFINE(LibCuda, cuMemAlloc_v2);
DEFINE(LibCuda, cuMemAllocPitch_v2);
DEFINE(LibCuda, cuMemAllocHost_v2);
DEFINE(LibCuda, cuMemcpyDtoD_v2);
DEFINE(LibCuda, cuMemcpyDtoDAsync_v2);
DEFINE(LibCuda, cuMemcpyDtoHAsync_v2);
DEFINE(LibCuda, cuMemcpyHtoDAsync_v2);
DEFINE(LibCuda, cuMemcpy2DAsync_v2);
DEFINE(LibCuda, cuMemFreeHost);
DEFINE(LibCuda, cuMemFree_v2);
DEFINE(LibCuda, cuPointerGetAttribute);
DEFINE(LibCuda, cuLaunchHostFunc);
DEFINE(LibCuda, cuStreamGetCtx);
DEFINE(LibCuda, cuEventRecord);
DEFINE(LibCuda, cuEventSynchronize);
DEFINE(LibCuda, cuDriverGetVersion);
