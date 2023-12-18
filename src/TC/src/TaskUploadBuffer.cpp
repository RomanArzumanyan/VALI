#include "MemoryInterfaces.hpp"
#include "Tasks.hpp"

namespace VPF {
struct UploadBuffer_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  CudaBuffer *pBuffer = nullptr;

  UploadBuffer_Impl() = delete;
  UploadBuffer_Impl(const UploadBuffer_Impl &other) = delete;
  UploadBuffer_Impl &operator=(const UploadBuffer_Impl &other) = delete;

  UploadBuffer_Impl(CUstream stream, CUcontext context, uint32_t elem_size,
                    uint32_t num_elems)
      : cuStream(stream), cuContext(context) {
    pBuffer = CudaBuffer::Make(elem_size, num_elems, context);
  }

  ~UploadBuffer_Impl() { delete pBuffer; }
};
} // namespace VPF

auto const cuda_stream_sync = [](void *stream) {
  cuStreamSynchronize((CUstream)stream);
};

UploadBuffer *UploadBuffer::Make(CUstream cuStream, CUcontext cuContext,
                                 uint32_t elem_size, uint32_t num_elems) {
  return new UploadBuffer(cuStream, cuContext, elem_size, num_elems);
}

UploadBuffer::UploadBuffer(CUstream cuStream, CUcontext cuContext,
                           uint32_t elem_size, uint32_t num_elems)
    :

      Task("UploadBuffer", UploadBuffer::numInputs, UploadBuffer::numOutputs,
           cuda_stream_sync, (void *)cuStream) {
  pImpl = new UploadBuffer_Impl(cuStream, cuContext, elem_size, num_elems);
}

UploadBuffer::~UploadBuffer() { delete pImpl; }

TaskExecStatus UploadBuffer::Run() {
  NvtxMark tick(GetName());
  if (!GetInput()) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pBuffer = pImpl->pBuffer;
  auto pSrcHost = ((Buffer *)GetInput())->GetDataAs<void>();

  CudaCtxPush lock(context);
  if (CUDA_SUCCESS != cuMemcpyHtoDAsync(pBuffer->GpuMem(),
                                        (const void *)pSrcHost,
                                        pBuffer->GetRawMemSize(), stream)) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  SetOutput(pBuffer, 0);
  return TaskExecStatus::TASK_EXEC_SUCCESS;
}