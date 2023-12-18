#include "MemoryInterfaces.hpp"
#include "Tasks.hpp"
#include <memory>

namespace VPF {
struct CudaDownloadSurface_Impl {
  CUstream cuStream;
  CUcontext cuContext;
  Pixel_Format format;
  std::shared_ptr<Buffer> pHostFrame = nullptr;

  CudaDownloadSurface_Impl() = delete;
  CudaDownloadSurface_Impl(const CudaDownloadSurface_Impl &other) = delete;
  CudaDownloadSurface_Impl &
  operator=(const CudaDownloadSurface_Impl &other) = delete;

  CudaDownloadSurface_Impl(CUstream stream, CUcontext context, uint32_t _width,
                           uint32_t _height, Pixel_Format _pix_fmt)
      : cuStream(stream), cuContext(context), format(_pix_fmt) {}

  ~CudaDownloadSurface_Impl() = default;
};
} // namespace VPF

CudaDownloadSurface *CudaDownloadSurface::Make(CUstream cuStream,
                                               CUcontext cuContext,
                                               uint32_t width, uint32_t height,
                                               Pixel_Format pixelFormat) {
  return new CudaDownloadSurface(cuStream, cuContext, width, height,
                                 pixelFormat);
}

auto const cuda_stream_sync = [](void *stream) {
  cuStreamSynchronize((CUstream)stream);
};

CudaDownloadSurface::CudaDownloadSurface(CUstream cuStream, CUcontext cuContext,
                                         uint32_t width, uint32_t height,
                                         Pixel_Format pix_fmt)
    :

      Task("CudaDownloadSurface", CudaDownloadSurface::numInputs,
           CudaDownloadSurface::numOutputs, cuda_stream_sync,
           (void *)cuStream) {
  pImpl =
      new CudaDownloadSurface_Impl(cuStream, cuContext, width, height, pix_fmt);
}

CudaDownloadSurface::~CudaDownloadSurface() { delete pImpl; }

TaskExecStatus CudaDownloadSurface::Run() {
  NvtxMark tick(GetName());

  if (!GetInput()) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  ClearOutputs();

  auto pSurface = (Surface *)GetInput();
  if (!pImpl->pHostFrame ||
      pImpl->pHostFrame->GetRawMemSize() != pSurface->HostMemSize()) {
    pImpl->pHostFrame.reset(
        Buffer::MakeOwnMem(pSurface->HostMemSize(), pImpl->cuContext));
  }

  auto stream = pImpl->cuStream;
  auto context = pImpl->cuContext;
  auto pDstHost = pImpl->pHostFrame->GetDataAs<uint8_t>();

  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstMemoryType = CU_MEMORYTYPE_HOST;

  for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
    CudaCtxPush lock(context);

    m.srcDevice = pSurface->PlanePtr(plane);
    m.srcPitch = pSurface->Pitch(plane);
    m.dstHost = pDstHost;
    m.dstPitch = pSurface->WidthInBytes(plane);
    m.WidthInBytes = pSurface->WidthInBytes(plane);
    m.Height = pSurface->Height(plane);

    auto const ret = cuMemcpy2DAsync(&m, stream);
    if (CUDA_SUCCESS != ret) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    pDstHost += m.WidthInBytes * m.Height;
  }

  SetOutput(pImpl->pHostFrame.get(), 0);
  return TaskExecStatus::TASK_EXEC_SUCCESS;
}