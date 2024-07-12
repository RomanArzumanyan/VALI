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
#include "CudaUtils.hpp"
#include <sstream>

namespace VPF {

CudaStrSync::CudaStrSync(CUstream stream) { str = stream; }

CudaStrSync::~CudaStrSync() { cuStreamSynchronize(str); }

CudaCtxPush::CudaCtxPush(CUcontext ctx) {
  ThrowOnCudaError(cuCtxPushCurrent(ctx), __LINE__);
}

CudaCtxPush::CudaCtxPush(CUstream str) {
  ThrowOnCudaError(cuCtxPushCurrent(GetContextByStream(str)), __LINE__);
}

CudaCtxPush::~CudaCtxPush() { cuCtxPopCurrent(nullptr); }

CudaStreamEvent::CudaStreamEvent(CUstream stream) {
  ThrowOnCudaError(cuEventRecord(m_event, stream), __LINE__);
}

void CudaStreamEvent::Wait() {
  ThrowOnCudaError(cuEventSynchronize(m_event), __LINE__);
}

CudaStreamEvent::~CudaStreamEvent() {
  try {
    Wait();
  } catch (...) {
  }
}

void ThrowOnCudaError(CUresult res, int lineNum) {
  if (CUDA_SUCCESS != res) {
    std::stringstream ss("");

    if (lineNum > 0) {
      ss << __FILE__ << ":";
      ss << lineNum << std::endl;
    }

    const char* errName = nullptr;
    if (CUDA_SUCCESS != cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << std::endl;
    } else {
      ss << "CUDA error: " << errName << std::endl;
    }

    const char* errDesc = nullptr;
    cuGetErrorString(res, &errDesc);

    if (!errDesc) {
      ss << "No error string available" << std::endl;
    } else {
      ss << errDesc << std::endl;
    }

    throw std::runtime_error(ss.str());
  }
};

void ThrowOnNppError(NppStatus res, int lineNum) {
  if (NPP_NO_ERROR != res) {
    std::stringstream ss("");

    if (lineNum > 0) {
      ss << __FILE__ << ":";
      ss << lineNum << std::endl;
    }

    ss << "NPP error with code " << res << std::endl;
    throw std::runtime_error(ss.str());
  }
}

int GetDeviceIdByDptr(CUdeviceptr dptr) {
  CudaCtxPush ctxPush(GetContextByDptr(dptr));
  CUdevice device_id = 0U;
  ThrowOnCudaError(cuCtxGetDevice(&device_id), __LINE__);
  return (int)device_id;
}

int GetDeviceIdByContext(CUcontext ctx) {
  CudaCtxPush ctxPush(ctx);
  CUdevice device_id = 0U;
  ThrowOnCudaError(cuCtxGetDevice(&device_id), __LINE__);
  return (int)device_id;
}

int GetDeviceIdByStream(CUstream str) {
  auto ctx = GetContextByStream(str);
  return GetDeviceIdByContext(ctx);
}

CUcontext GetContextByDptr(CUdeviceptr dptr) {
  CUcontext cuda_ctx = NULL;
  ThrowOnCudaError(cuPointerGetAttribute((void*)&cuda_ctx,
                                         CU_POINTER_ATTRIBUTE_CONTEXT, dptr),
                   __LINE__);
  return cuda_ctx;
}

CUdeviceptr GetDevicePointer(CUdeviceptr dptr) {
  CudaCtxPush ctxPush(GetContextByDptr(dptr));
  CUdeviceptr gpu_ptr = 0U;

  ThrowOnCudaError(cuPointerGetAttribute((void*)&gpu_ptr,
                                         CU_POINTER_ATTRIBUTE_DEVICE_POINTER,
                                         dptr),
                   __LINE__);
  return gpu_ptr;
}

CUcontext GetContextByStream(CUstream str) {
  CUcontext ctx;
  ThrowOnCudaError(cuStreamGetCtx(str, &ctx), __LINE__);
  return ctx;
}

void CudaStreamSync(void* args) {
  if (!args) {
    throw std::runtime_error("Empty argument given.");
  }

  auto stream = (CUstream)args;
  CudaCtxPush lock(GetContextByStream(stream));
  ThrowOnCudaError(cuStreamSynchronize(stream), __LINE__);
};
} // namespace VPF