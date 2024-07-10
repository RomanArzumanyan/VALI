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

CudaUploadFrame::CudaUploadFrame(CUstream stream)
    : Task("CudaUploadFrame", CudaUploadFrame::numInputs,
           CudaUploadFrame::numOutputs, cuda_stream_sync, (void*)stream),
      m_stream(stream) {}

TaskExecDetails CudaUploadFrame::Run() {
  NvtxMark tick(GetName());

  auto src_buffer = (Buffer*)GetInput(0U);
  if (!src_buffer) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT, "empty src");
  }

  auto dst_surface = (Surface*)GetInput(1U);
  if (!dst_surface) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::INVALID_INPUT, "empty dst");
  }

  if (src_buffer->GetRawMemSize() != dst_surface->HostMemSize()) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                           TaskExecInfo::SRC_DST_SIZE_MISMATCH,
                           "src / dst size mismatch");
  }

  ClearOutputs();

  auto context = GetContextByStream(m_stream);
  auto p_src_host = src_buffer->GetDataAs<uint8_t>();

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_HOST;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

  try {
    CudaCtxPush lock(context);
    for (auto i = 0; i < dst_surface->NumPlanes(); i++) {
      auto plane = dst_surface->GetSurfacePlane(i);

      m.srcHost = p_src_host;
      m.srcPitch = plane.Width() * plane.ElemSize();
      m.dstDevice = plane.GpuMem();
      m.dstPitch = plane.Pitch();
      m.WidthInBytes = m.srcPitch;
      m.Height = plane.Height();

      ThrowOnCudaError(cuMemcpy2DAsync(&m, m_stream), __LINE__);
      p_src_host += m.WidthInBytes * m.Height;
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