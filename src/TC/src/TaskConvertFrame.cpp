#include "NvCodecUtils.h"
#include "Tasks.hpp"
#include <memory>
#include <stdexcept>

extern "C" {
#include <libswscale/swscale.h>
}

namespace VPF {
struct ConvertFrame_Impl {
  const AVPixelFormat m_src_fmt, m_dst_fmt;
  size_t m_width, m_height;

  std::shared_ptr<SwsContext> m_ctx = nullptr;
  std::shared_ptr<Buffer> m_details = nullptr;

  ConvertFrame_Impl(uint32_t width, uint32_t height, Pixel_Format in_Format,
                    Pixel_Format out_Format)
      : m_src_fmt(toFfmpegPixelFormat(in_Format)),
        m_dst_fmt(toFfmpegPixelFormat(out_Format)), m_width(width),
        m_height(height) {
    m_details.reset(Buffer::MakeOwnMem(sizeof(TaskExecDetails)));
    m_ctx.reset(sws_getContext(m_width, m_height, m_src_fmt, width, height,
                               m_dst_fmt, SWS_BILINEAR, nullptr, nullptr,
                               nullptr),
                [](auto* p) { sws_freeContext(p); });

    if (!m_ctx) {
      throw std::runtime_error("ConvertFrame: sws_getContext failed");
    }
  }
};
}; // namespace VPF

ConvertFrame::~ConvertFrame() { delete pImpl; }

ConvertFrame::ConvertFrame(uint32_t width, uint32_t height,
                           Pixel_Format src_fmt, Pixel_Format dst_fmt)
    : Task("FfmpegConvertFrame", ConvertFrame::numInputs,
           ConvertFrame::numOutputs, nullptr, nullptr) {

  pImpl = new ConvertFrame_Impl(width, height, src_fmt, dst_fmt);
}

ConvertFrame* ConvertFrame::Make(uint32_t width, uint32_t height,
                                 Pixel_Format m_src_fmt,
                                 Pixel_Format m_dst_fmt) {
  return new ConvertFrame(width, height, m_src_fmt, m_dst_fmt);
}

TaskExecStatus ConvertFrame::Run() {
  ClearOutputs();
  auto pDetails = pImpl->m_details->GetDataAs<TaskExecDetails>();
  try {
    auto src_buf = dynamic_cast<Buffer*>(GetInput(0));
    if (!src_buf) {
      pDetails->info = TaskExecInfo::INVALID_INPUT;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    // Output check & lazy init;
    auto dst_buf = dynamic_cast<Buffer*>(GetInput(1));
    if (!dst_buf) {
      pDetails->info = TaskExecInfo::INVALID_INPUT;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto ctx_buf = dynamic_cast<Buffer*>(GetInput(2));
    if (!ctx_buf) {
      pDetails->info = TaskExecInfo::INVALID_INPUT;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    auto src_frame =
        asAVFrame(src_buf, pImpl->m_width, pImpl->m_height, pImpl->m_src_fmt);

    auto dst_frame =
        asAVFrame(dst_buf, pImpl->m_width, pImpl->m_height, pImpl->m_dst_fmt);

    auto pCtx = ctx_buf->GetDataAs<ColorspaceConversionContext>();

    auto const colorSpace = toFfmpegColorSpace(pCtx->color_space);
    auto const isJpegRange =
        (toFfmpegColorRange(pCtx->color_range) == AVCOL_RANGE_JPEG);
    auto const brightness = 0U, contrast = 1U << 16U, saturation = 1U << 16U;
    auto err = sws_setColorspaceDetails(
        pImpl->m_ctx.get(), sws_getCoefficients(colorSpace), isJpegRange,
        sws_getCoefficients(colorSpace), isJpegRange, brightness, contrast,
        saturation);
    if (err < 0) {
      pDetails->info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    err = sws_scale(pImpl->m_ctx.get(), src_frame->data, src_frame->linesize, 0,
                    pImpl->m_height, dst_frame->data, dst_frame->linesize);
    if (err < 0) {
      pDetails->info = TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS;
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    SetOutput(dst_buf, 0U);
    SetOutput(pImpl->m_details.get(), 1U);

    return TaskExecStatus::TASK_EXEC_SUCCESS;
  } catch (...) {
    pDetails->info = TaskExecInfo::FAIL;
    return TaskExecStatus::TASK_EXEC_FAIL;
  }
}