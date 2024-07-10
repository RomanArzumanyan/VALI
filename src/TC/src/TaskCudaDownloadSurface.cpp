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
#include "MemoryInterfaces.hpp"
#include "Tasks.hpp"

auto const cuda_stream_sync = [](void* stream) {
  cuStreamSynchronize((CUstream)stream);
};

CudaDownloadSurface::CudaDownloadSurface(CUstream stream)
    : Task("CudaDownloadSurface", CudaDownloadSurface::numInputs,
           CudaDownloadSurface::numOutputs, cuda_stream_sync, (void*)stream),
      m_stream(stream) {}

TaskExecDetails CudaDownloadSurface::Run() {
  NvtxMark tick(GetName());

  auto src_surface = (Surface*)GetInput(0U);
  if (!src_surface) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT, "empty src");
  }

  auto dst_buffer = (Buffer*)GetInput(1U);
  if (!dst_buffer) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT, "empty dst");
  }

  if (dst_buffer->GetRawMemSize() != src_surface->HostMemSize()) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::SRC_DST_SIZE_MISMATCH,
                           "src / dst size mismatch");
  }

  ClearOutputs();

  auto context = GetContextByStream(m_stream);
  auto p_dst_host = dst_buffer->GetDataAs<uint8_t>();

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstMemoryType = CU_MEMORYTYPE_HOST;

  try {
    CudaCtxPush lock(context);
    for (auto i = 0; i < src_surface->NumPlanes(); i++) {
      auto plane = src_surface->GetSurfacePlane(i);

      m.srcDevice = plane.GpuMem();
      m.srcPitch = plane.Pitch();
      m.dstHost = p_dst_host;
      m.dstPitch = plane.Width() * plane.ElemSize();
      m.WidthInBytes = m.dstPitch;
      m.Height = plane.Height();

      ThrowOnCudaError(cuMemcpy2DAsync(&m, m_stream), __LINE__);
      p_dst_host += m.WidthInBytes * m.Height;
    }
  } catch (std::exception& e) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                           e.what());
  } catch (...) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                           "unknown exception");
  }

  return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                         TaskExecInfo::SUCCESS);
}