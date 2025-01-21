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
#include <iostream>
#include <sstream>

namespace VPF {

CudaStrSync::CudaStrSync(CUstream stream) { str = stream; }

CudaStrSync::~CudaStrSync() { LibCuda::cuStreamSynchronize(str); }

CudaCtxPush::CudaCtxPush(CUcontext ctx) {
  ThrowOnCudaError(LibCuda::cuCtxPushCurrent(ctx), __LINE__);
}

CudaCtxPush::CudaCtxPush(CUstream str) {
  ThrowOnCudaError(LibCuda::cuCtxPushCurrent(GetContextByStream(str)),
                   __LINE__);
}

CudaCtxPush::~CudaCtxPush() { LibCuda::cuCtxPopCurrent(nullptr); }

CudaStreamEvent::CudaStreamEvent(CUstream stream, int primary_ctx_gpu_id) {
  m_str = stream;
  if (m_str && primary_ctx_gpu_id < 0) {
    m_ctx = GetContextByStream(m_str);
  } else if (!m_str && primary_ctx_gpu_id >= 0) {
    m_ctx = CudaResMgr::Instance().GetCtx(primary_ctx_gpu_id);
  } else {
    std::runtime_error("Invalid arguments combination: non default CUDA stream "
                       "and non-negative primary CUDA context GPU ID");
  }
  CudaCtxPush push(m_ctx);
  ThrowOnCudaError(LibCuda::cuEventCreate(&m_event, 0U), __LINE__);
}

void CudaStreamEvent::Record() {
  CudaCtxPush push(m_ctx);
  ThrowOnCudaError(LibCuda::cuEventRecord(m_event, m_str), __LINE__);
}

void CudaStreamEvent::Wait() {
  CudaCtxPush push(m_ctx);
  ThrowOnCudaError(LibCuda::cuEventSynchronize(m_event), __LINE__);
}

CudaStreamEvent::~CudaStreamEvent() {
  try {
    CudaCtxPush push(m_ctx);
    ThrowOnCudaError(LibCuda::cuEventDestroy(m_event), __LINE__);
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
    if (CUDA_SUCCESS != LibCuda::cuGetErrorName(res, &errName)) {
      ss << "CUDA error with code " << res << std::endl;
    } else {
      ss << "CUDA error: " << errName << std::endl;
    }

    const char* errDesc = nullptr;
    LibCuda::cuGetErrorString(res, &errDesc);

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
  ThrowOnCudaError(LibCuda::cuCtxGetDevice(&device_id), __LINE__);
  return (int)device_id;
}

int GetDeviceIdByContext(CUcontext ctx) {
  CudaCtxPush ctxPush(ctx);
  CUdevice device_id = 0U;
  ThrowOnCudaError(LibCuda::cuCtxGetDevice(&device_id), __LINE__);
  return (int)device_id;
}

int GetDeviceIdByStream(CUstream str) {
  auto ctx = GetContextByStream(str);
  return GetDeviceIdByContext(ctx);
}

CUcontext GetContextByDptr(CUdeviceptr dptr) {
  CUcontext cuda_ctx = NULL;
  ThrowOnCudaError(LibCuda::cuPointerGetAttribute(
                       (void*)&cuda_ctx, CU_POINTER_ATTRIBUTE_CONTEXT, dptr),
                   __LINE__);
  return cuda_ctx;
}

CUdeviceptr GetDevicePointer(CUdeviceptr dptr) {
  CudaCtxPush ctxPush(GetContextByDptr(dptr));
  CUdeviceptr gpu_ptr = 0U;

  ThrowOnCudaError(
      LibCuda::cuPointerGetAttribute((void*)&gpu_ptr,
                                     CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dptr),
      __LINE__);
  return gpu_ptr;
}

CUcontext GetContextByStream(CUstream str) {
  CUcontext ctx;
  ThrowOnCudaError(LibCuda::cuStreamGetCtx(str, &ctx), __LINE__);
  return ctx;
}

void CudaStreamSync(void* args) {
  if (!args) {
    throw std::runtime_error("Empty argument given.");
  }

  auto stream = (CUstream)args;
  CudaCtxPush lock(GetContextByStream(stream));
  ThrowOnCudaError(LibCuda::cuStreamSynchronize(stream), __LINE__);
};

CudaResMgr::CudaResMgr() {
  std::lock_guard<std::mutex> lock_ctx(CudaResMgr::gInsMutex);

  ThrowOnCudaError(LibCuda::cuInit(0), __LINE__);

  int nGpu;
  ThrowOnCudaError(LibCuda::cuDeviceGetCount(&nGpu), __LINE__);

  for (int i = 0; i < nGpu; i++) {
    CUdevice cuDevice = 0;
    CUcontext cuContext = nullptr;
    g_Contexts.push_back(std::make_pair(cuDevice, cuContext));

    CUstream cuStream = nullptr;
    g_Streams.push_back(cuStream);
  }
  return;
}

CUcontext CudaResMgr::GetCtx(size_t idx) {
  std::lock_guard<std::mutex> lock_ctx(CudaResMgr::gCtxMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& ctx = g_Contexts[idx];
  if (!ctx.second) {
    CUdevice cuDevice = 0;
    ThrowOnCudaError(LibCuda::cuDeviceGet(&cuDevice, idx), __LINE__);
    ThrowOnCudaError(LibCuda::cuDevicePrimaryCtxRetain(&ctx.second, cuDevice),
                     __LINE__);
  }

  return g_Contexts[idx].second;
}

CUstream CudaResMgr::GetStream(size_t idx) {
  std::lock_guard<std::mutex> lock_ctx(CudaResMgr::gStrMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& str = g_Streams[idx];
  if (!str) {
    auto ctx = GetCtx(idx);
    CudaCtxPush push(ctx);
    ThrowOnCudaError(LibCuda::cuStreamCreate(&str, CU_STREAM_NON_BLOCKING),
                     __LINE__);
  }

  return g_Streams[idx];
}

CudaResMgr::~CudaResMgr() {
  std::lock_guard<std::mutex> ins_lock(CudaResMgr::gInsMutex);
  std::lock_guard<std::mutex> ctx_lock(CudaResMgr::gCtxMutex);
  std::lock_guard<std::mutex> str_lock(CudaResMgr::gStrMutex);

  try {
    {
      for (auto& cuStream : g_Streams) {
        if (cuStream) {
          LibCuda::cuStreamDestroy(
              cuStream); // Avoiding CUDA_ERROR_DEINITIALIZED while
                         // destructing.
        }
      }
      g_Streams.clear();
    }

    {
      for (int i = 0; i < g_Contexts.size(); i++) {
        if (g_Contexts[i].second) {
          LibCuda::cuDevicePrimaryCtxRelease(
              g_Contexts[i].first); // Avoiding CUDA_ERROR_DEINITIALIZED while
                                    // destructing.
        }
      }
      g_Contexts.clear();
    }
  } catch (std::runtime_error& e) {
    std::cerr << e.what() << std::endl;
  }

#ifdef TRACK_TOKEN_ALLOCATIONS
  std::cout << "Checking token allocation counters: ";
  auto res = CheckAllocationCounters();
  std::cout << (res ? "No leaks dectected" : "Leaks detected") << std::endl;
#endif
}

CudaResMgr& CudaResMgr::Instance() {
  static CudaResMgr instance;
  return instance;
}

int CudaResMgr::GetVersion() const {
  int version;
  ThrowOnCudaError(LibCuda::cuDriverGetVersion(&version), __LINE__);
  return version;
}

size_t CudaResMgr::GetNumGpus() {
  try {
    return Instance().g_Contexts.size();
  } catch (std::exception& e) {
    return 0U;
  }
}

std::mutex CudaResMgr::gInsMutex;
std::mutex CudaResMgr::gCtxMutex;
std::mutex CudaResMgr::gStrMutex;
} // namespace VPF