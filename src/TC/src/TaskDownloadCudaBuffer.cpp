#include "MemoryInterfaces.hpp"
#include "Tasks.hpp"
#include <memory>

namespace VPF {
struct DownloadCudaBuffer_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  std::shared_ptr<Buffer> pHostBuffer = nullptr;

  DownloadCudaBuffer_Impl() = delete;
  DownloadCudaBuffer_Impl(const DownloadCudaBuffer_Impl &other) = delete;
  DownloadCudaBuffer_Impl &
  operator=(const DownloadCudaBuffer_Impl &other) = delete;

  DownloadCudaBuffer_Impl(CUstream stream, CUcontext context,
                          uint32_t elem_size, uint32_t num_elems)
      : cuStream(stream), cuContext(context) {
    pHostBuffer = std::shared_ptr<Buffer>(
        Buffer::MakeOwnMem(elem_size * num_elems, context));
  }

  ~DownloadCudaBuffer_Impl() = default;
};
} // namespace VPF

DownloadCudaBuffer *DownloadCudaBuffer::Make(CUstream cuStream,
                                             CUcontext cuContext,
                                             uint32_t elem_size,
                                             uint32_t num_elems) {
  return new DownloadCudaBuffer(cuStream, cuContext, elem_size, num_elems);
}

auto const cuda_stream_sync = [](void *stream) {
  cuStreamSynchronize((CUstream)stream);
};

DownloadCudaBuffer::DownloadCudaBuffer(CUstream cuStream, CUcontext cuContext,
                                       uint32_t elem_size, uint32_t num_elems)
    : Task("DownloadCudaBuffer", DownloadCudaBuffer::numInputs,
           DownloadCudaBuffer::numOutputs, cuda_stream_sync, (void *)cuStream) {
  pImpl =
      new DownloadCudaBuffer_Impl(cuStream, cuContext, elem_size, num_elems);
}

DownloadCudaBuffer::~DownloadCudaBuffer() { delete pImpl; }

TaskExecStatus DownloadCudaBuffer::Run() {
  NvtxMark tick(GetName());

  if (!GetInput()) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pCudaBuffer = (CudaBuffer *)GetInput();
  auto pDstHost = pImpl->pHostBuffer->GetDataAs<void>();

  CudaCtxPush lock(context);
  if (CUDA_SUCCESS != cuMemcpyDtoHAsync(pDstHost, pCudaBuffer->GpuMem(),
                                        pCudaBuffer->GetRawMemSize(), stream)) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  SetOutput(pImpl->pHostBuffer.get(), 0);
  return TaskExecStatus::TASK_EXEC_SUCCESS;
}