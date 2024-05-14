#include "MemoryInterfaces.hpp"
#include "Tasks.hpp"

namespace VPF {
struct CudaUploadFrame_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  Surface* pSurface = nullptr;
  Pixel_Format pixelFormat;

  CudaUploadFrame_Impl() = delete;
  CudaUploadFrame_Impl(const CudaUploadFrame_Impl& other) = delete;
  CudaUploadFrame_Impl& operator=(const CudaUploadFrame_Impl& other) = delete;

  CudaUploadFrame_Impl(CUstream stream, CUcontext context, uint32_t _width,
                       uint32_t _height, Pixel_Format _pix_fmt)
      : cuStream(stream), cuContext(context), pixelFormat(_pix_fmt) {
    pSurface = Surface::Make(pixelFormat, _width, _height, context);
  }

  ~CudaUploadFrame_Impl() { delete pSurface; }
};
} // namespace VPF

CudaUploadFrame* CudaUploadFrame::Make(CUstream cuStream, CUcontext cuContext,
                                       uint32_t width, uint32_t height,
                                       Pixel_Format pixelFormat) {
  return new CudaUploadFrame(cuStream, cuContext, width, height, pixelFormat);
}

CudaUploadFrame::CudaUploadFrame(CUstream cuStream, CUcontext cuContext,
                                 uint32_t width, uint32_t height,
                                 Pixel_Format pix_fmt)
    :

      Task("CudaUploadFrame", CudaUploadFrame::numInputs,
           CudaUploadFrame::numOutputs, nullptr, nullptr) {
  pImpl = new CudaUploadFrame_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaUploadFrame::~CudaUploadFrame() { delete pImpl; }

TaskExecStatus CudaUploadFrame::Run() {
  NvtxMark tick(GetName());

  auto input_buf = (Buffer*)GetInput();
  if (!input_buf) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pSurface = pImpl->pSurface;
  auto pSrcHost = input_buf->GetDataAs<uint8_t>();

  if (input_buf->GetRawMemSize() != pSurface->HostMemSize()) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_HOST;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

  try {
    CudaCtxPush lock(context);
    for (auto i = 0; i < pSurface->NumPlanes(); i++) {
      auto plane = pSurface->GetSurfacePlane(i);

      m.srcHost = pSrcHost;
      m.srcPitch = plane.Width() * plane.ElemSize();
      m.dstDevice = plane.GpuMem();
      m.dstPitch = plane.Pitch();
      m.WidthInBytes = m.srcPitch;
      m.Height = plane.Height();

      ThrowOnCudaError(cuMemcpy2DAsync(&m, stream), __LINE__);
      pSrcHost += m.WidthInBytes * m.Height;
    }
    ThrowOnCudaError(cuStreamSynchronize(stream), __LINE__);
  } catch (...) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  SetOutput(pSurface, 0);
  return TaskExecStatus::TASK_EXEC_SUCCESS;
}