#include "NppCommon.hpp"
#include "CudaUtils.hpp"
#include "Utils.hpp"
#include <cstring>
#include <iostream>
#include <mutex>

using namespace std;
using namespace VPF;

static mutex gNppMutex;

void SetupNppContext(int gpu_id, CUstream stream, NppStreamContext& nppCtx) {
  memset(&nppCtx, 0, sizeof(nppCtx));

  lock_guard<mutex> lock(gNppMutex);
  CudaCtxPush push(GetContextByStream(gpu_id, stream));

  try {
    CUdevice device;
    ThrowOnCudaError(LibCuda::cuCtxGetDevice(&device), __LINE__);

    int multiProcessorCount = 0;
    ThrowOnCudaError(LibCuda::cuDeviceGetAttribute(
                         &multiProcessorCount,
                         CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device),
                     __LINE__);

    int maxThreadsPerBlock = 0;
    ThrowOnCudaError(LibCuda::cuDeviceGetAttribute(
                         &maxThreadsPerBlock,
                         CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device),
                     __LINE__);

    int sharedMemPerBlock = 0;
    ThrowOnCudaError(LibCuda::cuDeviceGetAttribute(
                         &sharedMemPerBlock,
                         CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, device),
                     __LINE__);

    int major = 0;
    ThrowOnCudaError(
        LibCuda::cuDeviceGetAttribute(
            &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        __LINE__);

    int minor = 0;
    ThrowOnCudaError(
        LibCuda::cuDeviceGetAttribute(
            &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        __LINE__);
    nppCtx.hStream = stream;
    nppCtx.nCudaDeviceId = (int)device;
    nppCtx.nMultiProcessorCount = multiProcessorCount;
    nppCtx.nMaxThreadsPerBlock = maxThreadsPerBlock;
    nppCtx.nSharedMemPerBlock = sharedMemPerBlock;
    nppCtx.nCudaDevAttrComputeCapabilityMajor = major;
    nppCtx.nCudaDevAttrComputeCapabilityMinor = minor;
  } catch (std::exception& e) {
    av_log(nullptr, AV_LOG_ERROR, "Failed to setup NPP context: %s \n",
           e.what());
  }
}