#include "MemoryInterfaces.hpp"
#include "Tasks.hpp"

namespace VPF {
struct CudaUploadFrame_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  std::shared_ptr<Surface> pSurface = nullptr;
  Pixel_Format pixelFormat;

  CudaUploadFrame_Impl() = delete;
  CudaUploadFrame_Impl(const CudaUploadFrame_Impl& other) = delete;
  CudaUploadFrame_Impl& operator=(const CudaUploadFrame_Impl& other) = delete;

  CudaUploadFrame_Impl(CUstream stream, CUcontext context, uint32_t _width,
                       uint32_t _height, Pixel_Format _pix_fmt)
      : cuStream(stream), cuContext(context), pixelFormat(_pix_fmt) {
    pSurface = Surface::Make(pixelFormat, _width, _height, context);
  }

  ~CudaUploadFrame_Impl() = default;
};
} // namespace VPF

CudaUploadFrame *CudaUploadFrame::Make(CUstream cuStream, CUcontext cuContext,
                                       uint32_t width, uint32_t height,
                                       Pixel_Format pixelFormat) {
  return new CudaUploadFrame(cuStream, cuContext, width, height, pixelFormat);
}

auto const cuda_stream_sync = [](void *stream) {
  cuStreamSynchronize((CUstream)stream);
};

CudaUploadFrame::CudaUploadFrame(CUstream cuStream, CUcontext cuContext,
                                 uint32_t width, uint32_t height,
                                 Pixel_Format pix_fmt)
    :

      Task("CudaUploadFrame", CudaUploadFrame::numInputs,
           CudaUploadFrame::numOutputs, cuda_stream_sync, (void *)cuStream) {
  pImpl = new CudaUploadFrame_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaUploadFrame::~CudaUploadFrame() { delete pImpl; }

TaskExecStatus CudaUploadFrame::Run() {
  NvtxMark tick(GetName());
  if (!GetInput()) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pSurface = pImpl->pSurface;
  auto pSrcHost = ((Buffer *)GetInput())->GetDataAs<uint8_t>();

  try {
    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_HOST;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

    for (auto& plane : *pSurface.get()) {
      CudaCtxPush lock(context);

      m.srcHost = pSrcHost;
      m.srcPitch = plane.Width() * plane.ElemSize();
      m.dstDevice = plane.GpuMem();
      m.dstPitch = plane.Pitch();
      m.WidthInBytes = m.srcPitch;
      m.Height = plane.Height();

      ThrowOnCudaError(cuMemcpy2DAsync(&m, stream), __LINE__);
      pSrcHost += m.WidthInBytes * m.Height;
    }
  } catch (...){
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  SetOutput(pSurface.get(), 0);
  return TaskExecStatus::TASK_EXEC_SUCCESS;
}