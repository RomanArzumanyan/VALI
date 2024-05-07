#include "MemoryInterfaces.hpp"
#include "NppCommon.hpp"
#include "Tasks.hpp"

#include <stdexcept>

namespace VPF {
struct ResizeSurface_Impl {
  Surface* pSurface = nullptr;
  CUcontext cu_ctx;
  CUstream cu_str;
  NppStreamContext nppCtx;

  ResizeSurface_Impl(uint32_t width, uint32_t height, Pixel_Format format,
                     CUcontext ctx, CUstream str)
      : cu_ctx(ctx), cu_str(str) {
    SetupNppContext(cu_ctx, cu_str, nppCtx);
  }

  virtual ~ResizeSurface_Impl() = default;

  virtual TaskExecStatus Run(Surface& source) = 0;
};

struct NppResizeSurfacePacked3C_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePacked3C_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                                CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str) {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePacked3C_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source) {
    NvtxMark tick("NppResizeSurfacePacked3C");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto srcPlane = source.GetSurfacePlane();
    auto dstPlane = pSurface->GetSurfacePlane();

    const Npp8u* pSrc = (const Npp8u*)srcPlane.GpuMem();
    int nSrcStep = (int)source.Pitch();
    NppiSize oSrcSize = {0};
    oSrcSize.width = source.Width();
    oSrcSize.height = source.Height();
    NppiRect oSrcRectROI = {0};
    oSrcRectROI.width = oSrcSize.width;
    oSrcRectROI.height = oSrcSize.height;

    Npp8u* pDst = (Npp8u*)dstPlane.GpuMem();
    int nDstStep = (int)pSurface->Pitch();
    NppiSize oDstSize = {0};
    oDstSize.width = pSurface->Width();
    oDstSize.height = pSurface->Height();
    NppiRect oDstRectROI = {0};
    oDstRectROI.width = oDstSize.width;
    oDstRectROI.height = oDstSize.height;
    int eInterpolation = NPPI_INTER_LANCZOS;

    CudaCtxPush ctxPush(cu_ctx);
    auto ret = nppiResize_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                     pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppCtx);
    if (NPP_NO_ERROR != ret) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }
};

// Resize planar 8 bit surface (YUV420, YCbCr420);
struct NppResizeSurfacePlanar_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePlanar_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                              CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str) {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePlanar_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source) {
    NvtxMark tick("NppResizeSurfacePlanar");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
      auto srcPlane = source.GetSurfacePlane(plane);
      auto dstPlane = pSurface->GetSurfacePlane(plane);

      const Npp8u* pSrc = (const Npp8u*)srcPlane.GpuMem();
      int nSrcStep = (int)srcPlane.Pitch();
      NppiSize oSrcSize = {0};
      oSrcSize.width = srcPlane.Width();
      oSrcSize.height = srcPlane.Height();
      NppiRect oSrcRectROI = {0};
      oSrcRectROI.width = oSrcSize.width;
      oSrcRectROI.height = oSrcSize.height;

      Npp8u* pDst = (Npp8u*)dstPlane.GpuMem();
      int nDstStep = (int)dstPlane.Pitch();
      NppiSize oDstSize = {0};
      oDstSize.width = dstPlane.Width();
      oDstSize.height = dstPlane.Height();
      NppiRect oDstRectROI = {0};
      oDstRectROI.width = oDstSize.width;
      oDstRectROI.height = oDstSize.height;
      int eInterpolation = NPPI_INTER_LANCZOS;

      CudaCtxPush ctxPush(cu_ctx);
      auto ret = nppiResize_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                       pDst, nDstStep, oDstSize, oDstRectROI,
                                       eInterpolation, nppCtx);
      if (NPP_NO_ERROR != ret) {
        return TaskExecStatus::TASK_EXEC_FAIL;
      }
    }

    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }
};

// Resize semiplanar 8 bit NV12 surface;
struct ResizeSurfaceSemiPlanar_Impl final : ResizeSurface_Impl {
  ResizeSurfaceSemiPlanar_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                               CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str), _ctx(ctx),
        _str(str), last_h(0U), last_w(0U) {
    pSurface = nullptr;
    nv12_yuv420 = nullptr;
    pResizeYuv = ResizeSurface::Make(width, height, YUV420, ctx, str);
    yuv420_nv12 = ConvertSurface::Make(width, height, YUV420, NV12, ctx, str);
  }

  ~ResizeSurfaceSemiPlanar_Impl() {
    pSurface = nullptr;
    delete pResizeYuv;
    delete nv12_yuv420;
    delete yuv420_nv12;
  }

  TaskExecStatus Run(Surface& source) {
    NvtxMark tick("NppResizeSurfaceSemiPlanar");

    auto const resolution_change =
        source.Width() != last_w || source.Height() != last_h;

    if (nv12_yuv420 && resolution_change) {
      delete nv12_yuv420;
      nv12_yuv420 = nullptr;
    }

    if (!nv12_yuv420) {
      nv12_yuv420 = ConvertSurface::Make(source.Width(), source.Height(), NV12,
                                         YUV420, _ctx, _str);
    }

    // Convert from NV12 to YUV420;
    nv12_yuv420->SetInput((Token*)&source, 0U);
    if (TaskExecStatus::TASK_EXEC_SUCCESS != nv12_yuv420->Execute())
      return TaskExecStatus::TASK_EXEC_FAIL;
    auto surf_yuv420 = nv12_yuv420->GetOutput(0U);

    // Resize YUV420;
    pResizeYuv->SetInput(surf_yuv420, 0U);
    if (TaskExecStatus::TASK_EXEC_SUCCESS != pResizeYuv->Execute())
      return TaskExecStatus::TASK_EXEC_FAIL;
    auto surf_res = pResizeYuv->GetOutput(0U);

    // Convert back to NV12;
    yuv420_nv12->SetInput(surf_res, 0U);
    if (TaskExecStatus::TASK_EXEC_SUCCESS != yuv420_nv12->Execute())
      return TaskExecStatus::TASK_EXEC_FAIL;
    pSurface = (Surface*)yuv420_nv12->GetOutput(0U);

    last_w = source.Width();
    last_h = source.Height();

    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }

  ResizeSurface* pResizeYuv;
  ConvertSurface* nv12_yuv420;
  ConvertSurface* yuv420_nv12;
  CUcontext _ctx;
  CUstream _str;
  uint32_t last_w, last_h;
};

struct NppResizeSurfacePacked32F3C_Impl final : ResizeSurface_Impl {
  NppResizeSurfacePacked32F3C_Impl(uint32_t width, uint32_t height,
                                   CUcontext ctx, CUstream str,
                                   Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str) {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurfacePacked32F3C_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source) {
    NvtxMark tick("NppResizeSurfacePacked32F3C");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto srcPlane = source.GetSurfacePlane();
    auto dstPlane = pSurface->GetSurfacePlane();

    const Npp32f* pSrc = (const Npp32f*)srcPlane.GpuMem();
    int nSrcStep = (int)source.Pitch();
    NppiSize oSrcSize = {0};
    oSrcSize.width = source.Width();
    oSrcSize.height = source.Height();
    NppiRect oSrcRectROI = {0};
    oSrcRectROI.width = oSrcSize.width;
    oSrcRectROI.height = oSrcSize.height;

    Npp32f* pDst = (Npp32f*)dstPlane.GpuMem();
    int nDstStep = (int)pSurface->Pitch();
    NppiSize oDstSize = {0};
    oDstSize.width = pSurface->Width();
    oDstSize.height = pSurface->Height();
    NppiRect oDstRectROI = {0};
    oDstRectROI.width = oDstSize.width;
    oDstRectROI.height = oDstSize.height;
    int eInterpolation = NPPI_INTER_LANCZOS;

    CudaCtxPush ctxPush(cu_ctx);
    auto ret = nppiResize_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                      pDst, nDstStep, oDstSize, oDstRectROI,
                                      eInterpolation, nppCtx);
    if (NPP_NO_ERROR != ret) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }
};

// Resize planar 8 bit surface (YUV420, YCbCr420);
struct NppResizeSurface32FPlanar_Impl final : ResizeSurface_Impl {
  NppResizeSurface32FPlanar_Impl(uint32_t width, uint32_t height, CUcontext ctx,
                                 CUstream str, Pixel_Format format)
      : ResizeSurface_Impl(width, height, format, ctx, str) {
    pSurface = Surface::Make(format, width, height, ctx);
  }

  ~NppResizeSurface32FPlanar_Impl() { delete pSurface; }

  TaskExecStatus Run(Surface& source) {
    NvtxMark tick("NppResizeSurface32FPlanar");

    if (pSurface->PixelFormat() != source.PixelFormat()) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    for (auto plane = 0; plane < pSurface->NumPlanes(); plane++) {
      auto srcPlane = source.GetSurfacePlane(plane);
      auto dstPlane = pSurface->GetSurfacePlane(plane);

      const Npp32f* pSrc = (const Npp32f*)srcPlane.GpuMem();
      int nSrcStep = (int)srcPlane.Pitch();
      NppiSize oSrcSize = {0};
      oSrcSize.width = srcPlane.Width();
      oSrcSize.height = srcPlane.Height();
      NppiRect oSrcRectROI = {0};
      oSrcRectROI.width = oSrcSize.width;
      oSrcRectROI.height = oSrcSize.height;

      Npp32f* pDst = (Npp32f*)dstPlane.GpuMem();
      int nDstStep = (int)dstPlane.Pitch();
      NppiSize oDstSize = {0};
      oDstSize.width = dstPlane.Width();
      oDstSize.height = dstPlane.Height();
      NppiRect oDstRectROI = {0};
      oDstRectROI.width = oDstSize.width;
      oDstRectROI.height = oDstSize.height;
      int eInterpolation = NPPI_INTER_LANCZOS;

      CudaCtxPush ctxPush(cu_ctx);
      auto ret = nppiResize_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI,
                                        pDst, nDstStep, oDstSize, oDstRectROI,
                                        eInterpolation, nppCtx);
      if (NPP_NO_ERROR != ret) {
        return TaskExecStatus::TASK_EXEC_FAIL;
      }
    }

    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }
};
} // namespace VPF

auto const cuda_stream_sync = [](void* stream) {
  cuStreamSynchronize((CUstream)stream);
};

ResizeSurface::ResizeSurface(uint32_t width, uint32_t height,
                             Pixel_Format format, CUcontext ctx, CUstream str)
    : Task("NppResizeSurface", ResizeSurface::numInputs,
           ResizeSurface::numOutputs, cuda_stream_sync, (void*)str) {
  if (RGB == format || BGR == format) {
    pImpl = new NppResizeSurfacePacked3C_Impl(width, height, ctx, str, format);
  } else if (YUV420 == format || YUV444 == format || RGB_PLANAR == format) {
    pImpl = new NppResizeSurfacePlanar_Impl(width, height, ctx, str, format);
  } else if (RGB_32F == format) {
    pImpl =
        new NppResizeSurfacePacked32F3C_Impl(width, height, ctx, str, format);
  } else if (RGB_32F_PLANAR == format) {
    pImpl = new NppResizeSurface32FPlanar_Impl(width, height, ctx, str, format);
  } else if (NV12 == format) {
    pImpl = new ResizeSurfaceSemiPlanar_Impl(width, height, ctx, str, format);
  } else {
    throw std::runtime_error("pixel format not supported");
  }
}

ResizeSurface::~ResizeSurface() { delete pImpl; }

TaskExecStatus ResizeSurface::Run() {
  NvtxMark tick(GetName());
  ClearOutputs();

  auto pInputSurface = (Surface*)GetInput();
  if (!pInputSurface) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  if (TaskExecStatus::TASK_EXEC_SUCCESS != pImpl->Run(*pInputSurface)) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  SetOutput(pImpl->pSurface, 0U);
  return TaskExecStatus::TASK_EXEC_SUCCESS;
}

ResizeSurface* ResizeSurface::Make(uint32_t width, uint32_t height,
                                   Pixel_Format format, CUcontext ctx,
                                   CUstream str) {
  return new ResizeSurface(width, height, format, ctx, str);
}