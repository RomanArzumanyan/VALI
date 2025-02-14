#include "NppCommon.hpp"
#include "CudaUtils.hpp"
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

  CUdevice device;
  auto res = LibCuda::cuCtxGetDevice(&device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  int multiProcessorCount = 0;
  res = LibCuda::cuDeviceGetAttribute(
      &multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  int maxThreadsPerBlock = 0;
  res = LibCuda::cuDeviceGetAttribute(
      &maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  int sharedMemPerBlock = 0;
  res = LibCuda::cuDeviceGetAttribute(
      &sharedMemPerBlock, CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK, device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  int major = 0;
  res = LibCuda::cuDeviceGetAttribute(
      &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  int minor = 0;
  res = LibCuda::cuDeviceGetAttribute(
      &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
  if (CUDA_SUCCESS != res) {
    cerr << "Failed to get CUDA device. Error code: " << res << endl;
  }

  nppCtx.hStream = stream;
  nppCtx.nCudaDeviceId = (int)device;
  nppCtx.nMultiProcessorCount = multiProcessorCount;
  nppCtx.nMaxThreadsPerBlock = maxThreadsPerBlock;
  nppCtx.nSharedMemPerBlock = sharedMemPerBlock;
  nppCtx.nCudaDevAttrComputeCapabilityMajor = major;
  nppCtx.nCudaDevAttrComputeCapabilityMinor = minor;
}