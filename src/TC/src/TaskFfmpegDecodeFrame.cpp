/*
 * Copyright 2020 NVIDIA Corporation
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
#include "FFmpegDemuxer.h"
#include "Tasks.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
}

using namespace VPF;

namespace VPF {

enum DECODE_STATUS { DEC_SUCCESS, DEC_ERROR, DEC_MORE, DEC_EOS };

#ifndef TEGRA_BUILD
static AVPixelFormat get_format(AVCodecContext* avctx,
                                const enum AVPixelFormat* pix_fmts) {
  while (*pix_fmts != AV_PIX_FMT_NONE) {
    if (*pix_fmts == AV_PIX_FMT_CUDA) {
      avctx->hw_frames_ctx = av_hwframe_ctx_alloc(avctx->hw_device_ctx);

      if (!avctx->hw_frames_ctx)
        return AV_PIX_FMT_NONE;

      auto frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;
      frames_ctx->format = AV_PIX_FMT_CUDA;
      frames_ctx->sw_format = avctx->sw_pix_fmt;
      frames_ctx->width = FFALIGN(avctx->coded_width, 32);
      frames_ctx->height = FFALIGN(avctx->coded_height, 32);

      if (av_hwframe_ctx_init(avctx->hw_frames_ctx) < 0) {
        return AV_PIX_FMT_NONE;
      }

      return AV_PIX_FMT_CUDA;
    }

    pix_fmts++;
  }

  std::cerr << "CUDA pixel format not offered in get_format()";
  return AV_PIX_FMT_NONE;
}

static const std::map<AVCodecID, std::string>
    hwaccel_codecs({std::make_pair(AV_CODEC_ID_AV1, "av1_cuvid"),
                    std::make_pair(AV_CODEC_ID_HEVC, "hevc_cuvid"),
                    std::make_pair(AV_CODEC_ID_H264, "h264_cuvid"),
                    std::make_pair(AV_CODEC_ID_MJPEG, "mjpeg_cuvid"),
                    std::make_pair(AV_CODEC_ID_MPEG1VIDEO, "mpeg1_cuvid"),
                    std::make_pair(AV_CODEC_ID_MPEG2VIDEO, "mpeg2_cuvid"),
                    std::make_pair(AV_CODEC_ID_MPEG4, "mpeg4_cuvid"),
                    std::make_pair(AV_CODEC_ID_VP8, "vp8_cuvid"),
                    std::make_pair(AV_CODEC_ID_VP9, "vp9_cuvid"),
                    std::make_pair(AV_CODEC_ID_VC1, "vc1_cuvid")});
#else
static const std::map<AVCodecID, std::string> hwaccel_codecs;
#endif

static std::string FindDecoderById(AVCodecID id) {
  const auto it = hwaccel_codecs.find(id);
  if (it == hwaccel_codecs.end())
    return "";

  return it->second;
}

struct FfmpegDecodeFrame_Impl {
  std::shared_ptr<AVFormatContext> m_fmt_ctx;
  std::shared_ptr<SwrContext> m_swr_ctx;
  std::shared_ptr<AVCodecContext> m_avc_ctx;
  std::shared_ptr<AVFrame> m_frame;
  std::shared_ptr<AVPacket> m_pkt;
  std::map<AVFrameSideDataType, Buffer*> m_side_data;
  std::shared_ptr<TimeoutHandler> m_timeout_handler;
  PacketData m_packetData;
  CUstream m_stream;

  int video_stream_idx = -1;
  bool end_decode = false;
  bool eof = false;

  FfmpegDecodeFrame_Impl(
      const char* URL, const std::map<std::string, std::string>& ffmpeg_options,
      std::optional<CUstream> stream) {

    // Allocate format context first to set timeout before opening the input.
    AVFormatContext* fmt_ctx_ = avformat_alloc_context();
    if (!fmt_ctx_) {
      throw std::runtime_error("Failed to allocate format context");
    }

    // Set up format context options.
    AVDictionary* options = GetAvOptions(ffmpeg_options);

    /* Set the timeout.
     *
     * Unfortunately, 'timeout' and 'stimeout' AVDictionary options do work for
     * some formats and don't for others.
     *
     * So explicit timeout handler is used instead. In worst case scenario, 2
     * handlers with same value will be registered within format context which
     * is of no harm. */
    auto it = ffmpeg_options.find("timeout");
    if (it != ffmpeg_options.end()) {
      m_timeout_handler.reset(new TimeoutHandler(std::stoi(it->second)));
    } else {
      m_timeout_handler.reset(new TimeoutHandler());
    }

    AVDictionary* options_copy = nullptr;
    auto res = av_dict_copy(&options_copy, options, 0);
    if (res < 0) {
      if (options_copy) {
        av_dict_free(&options_copy);
      }
      std::stringstream ss;
      ss << "Could not copy AVOptions: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    m_timeout_handler->Reset();
    res = avformat_open_input(&fmt_ctx_, URL, NULL, &options_copy);
    if (options_copy) {
      av_dict_free(&options_copy);
    }

    if (res < 0) {
      {
        // Don't remove, it's here to make linker clear of libswresample
        // dependency. Otherwise it's required by libavcodec but never put into
        // rpath by CMake.
        SwrContext* swr_ctx_ = swr_alloc();
        m_swr_ctx = std::shared_ptr<SwrContext>(
            swr_ctx_, [](void* p) { swr_free((SwrContext**)&p); });
      }

      std::stringstream ss;
      ss << "Could not open source file" << URL;
      ss << "Error description: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    m_fmt_ctx = std::shared_ptr<AVFormatContext>(
        fmt_ctx_, [](void* p) { avformat_close_input((AVFormatContext**)&p); });

    m_timeout_handler->Reset();
    res = avformat_find_stream_info(m_fmt_ctx.get(), NULL);
    if (res < 0) {
      std::stringstream ss;
      ss << "Could not find stream information";
      ss << "Error description: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    m_timeout_handler->Reset();
    video_stream_idx = av_find_best_stream(m_fmt_ctx.get(), AVMEDIA_TYPE_VIDEO,
                                           -1, -1, NULL, 0);
    if (video_stream_idx < 0) {
      std::stringstream ss;
      ss << "Could not find " << av_get_media_type_string(AVMEDIA_TYPE_VIDEO)
         << " stream in file " << URL;
      ss << "Error description: " << AvErrorToString(video_stream_idx);
      throw std::runtime_error(ss.str());
    }

    auto video_stream = m_fmt_ctx->streams[video_stream_idx];
    if (!video_stream) {
      std::cerr << "Could not find video stream in the input, aborting";
    }

    auto p_codec =
        stream ? avcodec_find_decoder_by_name(
                     FindDecoderById(video_stream->codecpar->codec_id).c_str())
               : avcodec_find_decoder(video_stream->codecpar->codec_id);

    if (!p_codec && stream) {
      throw std::runtime_error(
          "Failed to find codec by name: " +
          FindDecoderById(video_stream->codecpar->codec_id));
    } else if (!p_codec) {
      throw std::runtime_error(
          "Failed to find codec by id: " +
          std::string(avcodec_get_name(video_stream->codecpar->codec_id)));
    }

    auto avctx_ = avcodec_alloc_context3(p_codec);
    if (!avctx_) {
      std::stringstream ss;
      ss << "Failed to allocate codec context";
      throw std::runtime_error(ss.str());
    }
    m_avc_ctx = std::shared_ptr<AVCodecContext>(
        avctx_, [](void* p) { avcodec_free_context((AVCodecContext**)&p); });

    res =
        avcodec_parameters_to_context(m_avc_ctx.get(), video_stream->codecpar);
    if (res < 0) {
      std::stringstream ss;
      ss << "Failed to pass codec parameters to codec "
            "context "
         << av_get_media_type_string(AVMEDIA_TYPE_VIDEO);
      ss << "Error description: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

/* Have no idea if Tegra supports output to CUDA memory so keep it under
 * conditional compilation. Tegra has unified memory anyway, so no big
 * performance penalty shall be imposed as there's no "actual" Host <> Device IO
 * happening.
 */
#ifndef TEGRA_BUILD
    if (stream) {
      m_stream = stream.value();
      // Push CUDA context, otherwise FFMpeg will create its own.
      CudaCtxPush push_ctx(GetContextByStream(m_stream));

      /* Attach HW context to codec. Whithout that decoded frames will be
       * copied to RAM.
       */
      AVDictionary* opts = nullptr;
      auto res = av_dict_set(&opts, "current_ctx", "1", 0);
      if (res < 0) {
        std::stringstream ss;
        ss << "Failed to set AVDictionary option: " << AvErrorToString(res);
        throw std::runtime_error(ss.str());
      }

      AVBufferRef* hwdevice_ctx = nullptr;
      res = av_hwdevice_ctx_create(&hwdevice_ctx, AV_HWDEVICE_TYPE_CUDA, NULL,
                                   opts, 0);
      av_dict_free(&opts);

      if (res < 0) {
        std::stringstream ss;
        ss << "Failed to create HW device context: " << AvErrorToString(res);
        throw std::runtime_error(ss.str());
      }

      // Make FFMpeg use given stream.
      auto hw_dev_ctx = (AVHWDeviceContext*)hwdevice_ctx->data;
      auto cuda_dev_ctx = (AVCUDADeviceContext*)hw_dev_ctx->hwctx;
      cuda_dev_ctx->stream = stream.value();

      // Add hw device context to codec context.
      m_avc_ctx->hw_device_ctx = av_buffer_ref(hwdevice_ctx);
      m_avc_ctx->get_format = get_format;
    }
#endif

    options_copy = nullptr;
    res = av_dict_copy(&options_copy, options, 0);
    if (res < 0) {
      if (options_copy) {
        av_dict_free(&options_copy);
      }
      std::stringstream ss;
      ss << "Could not copy AVOptions: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    res = avcodec_open2(m_avc_ctx.get(), p_codec, &options_copy);
    if (options_copy) {
      av_dict_free(&options_copy);
    }

    if (res < 0) {
      std::stringstream ss;
      ss << "Failed to open codec "
         << av_get_media_type_string(AVMEDIA_TYPE_VIDEO);
      ss << "Error description: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    auto frame_ = av_frame_alloc();
    if (!frame_) {
      std::stringstream ss;
      ss << "Could not allocate frame";
      throw std::runtime_error(ss.str());
    }

    m_frame = std::shared_ptr<AVFrame>(
        frame_, [](void* p) { av_frame_free((AVFrame**)&p); });

    auto avpkt_ = av_packet_alloc();
    if (!avpkt_) {
      std::stringstream ss;
      ss << "Could not allocate packet";
      throw std::runtime_error(ss.str());
    }
    m_pkt = std::shared_ptr<AVPacket>(
        avpkt_, [](void* p) { av_packet_free((AVPacket**)&p); });
  }

  void SavePacketData() {
    m_packetData.dts = m_frame->pkt_dts;
    m_packetData.pts = m_frame->pts;
    m_packetData.duration = m_frame->duration;
    m_packetData.key = m_frame->flags & AV_FRAME_FLAG_KEY;
    m_packetData.pos = m_frame->pkt_pos;
  }

  bool DecodeSingleFrame(DecodeFrame* parent, Token& dst) {
    if (end_decode) {
      return false;
    }

    // Send packets to decoder until it outputs frame;
    do {
      // Read packets from stream until we find a video packet;
      do {
        if (eof) {
          break;
        }

        m_timeout_handler->Reset();
        auto ret = av_read_frame(m_fmt_ctx.get(), m_pkt.get());

        if (AVERROR_EOF == ret) {
          eof = true;
          break;
        } else if (ret < 0) {
          end_decode = true;
          parent->SetExecDetails(TaskExecDetails(TaskExecInfo::FAIL));
          return false;
        }
      } while (m_pkt->stream_index != video_stream_idx);

      auto status = DecodeSinglePacket(eof ? nullptr : m_pkt.get(), dst);

      switch (status) {
      case DEC_SUCCESS:
        return true;
      case DEC_ERROR:
        parent->SetExecDetails(TaskExecDetails(TaskExecInfo::FAIL));
        end_decode = true;
        return false;
      case DEC_EOS:
        parent->SetExecDetails(TaskExecDetails(TaskExecInfo::END_OF_STREAM));
        end_decode = true;
        return false;
      case DEC_MORE:
        continue;
      }
    } while (true);

    return true;
  }

  void SaveMotionVectors() {
    AVFrameSideData* sd =
        av_frame_get_side_data(m_frame.get(), AV_FRAME_DATA_MOTION_VECTORS);

    if (sd) {
      auto it = m_side_data.find(AV_FRAME_DATA_MOTION_VECTORS);
      if (it == m_side_data.end()) {
        // Add entry if not found (usually upon first call);
        m_side_data[AV_FRAME_DATA_MOTION_VECTORS] =
            Buffer::MakeOwnMem(sd->size);
        it = m_side_data.find(AV_FRAME_DATA_MOTION_VECTORS);
        memcpy(it->second->GetRawMemPtr(), sd->data, sd->size);
      } else if (it->second->GetRawMemSize() != sd->size) {
        // Update entry size if changed (e. g. on video resolution change);
        it->second->Update(sd->size, sd->data);
      }
    }
  }

  bool SaveSideData() {
    SaveMotionVectors();
    return true;
  }

  uint32_t GetHostFrameSize() const {
    const int alignment = 1;
    return av_image_get_buffer_size((AVPixelFormat)m_frame->format,
                                    m_frame->width, m_frame->height, alignment);
  }

  DECODE_STATUS DecodeSinglePacket(const AVPacket* pkt_Src, Token& dst) {
    auto res = avcodec_send_packet(m_avc_ctx.get(), pkt_Src);
    if (AVERROR_EOF == res) {
      // Flush decoder;
      res = 0;
    } else if (res < 0) {
      std::cerr << "Error while sending a packet to the decoder";
      std::cerr << "Error description: " << AvErrorToString(res);
      return DEC_ERROR;
    }

    while (res >= 0) {
      res = avcodec_receive_frame(m_avc_ctx.get(), m_frame.get());
      if (res == AVERROR_EOF) {
        return DEC_EOS;
      } else if (res == AVERROR(EAGAIN)) {
        return DEC_MORE;
      } else if (res < 0) {
        std::cerr << "Error while receiving a frame from the decoder";
        std::cerr << "Error description: " << AvErrorToString(res);
        return DEC_ERROR;
      }

      if (m_frame->hw_frames_ctx) {
        auto& dst_surf = (Surface&)dst;

        CUDA_MEMCPY2D m = {0};
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

        CudaCtxPush push_ctx(GetContextByStream(m_stream));
        for (auto i = 0U; i < dst_surf.NumPlanes(); i++) {
          auto& plane = dst_surf.GetSurfacePlane(i);

          m.srcDevice = (CUdeviceptr)m_frame->data[i];
          m.srcPitch = m_frame->linesize[i];
          m.dstDevice = plane.GpuMem();
          m.dstPitch = plane.Pitch();
          m.WidthInBytes = plane.Width() * plane.ElemSize();
          m.Height = plane.Height();

          ThrowOnCudaError(cuMemcpy2DAsync(&m, m_stream), __LINE__);
        }
        ThrowOnCudaError(cuStreamSynchronize(m_stream), __LINE__);
      } else {
        auto& dstBuf = dynamic_cast<Buffer&>(dst);
        const int alignment = 1;

        av_image_copy_to_buffer(dstBuf.GetDataAs<uint8_t>(), GetHostFrameSize(),
                                m_frame->data, m_frame->linesize,
                                (AVPixelFormat)m_frame->format, m_frame->width,
                                m_frame->height, alignment);
      }

      SaveSideData();
      SavePacketData();
      return DEC_SUCCESS;
    }

    return DEC_SUCCESS;
  }

  ~FfmpegDecodeFrame_Impl() {
    for (auto& output : m_side_data) {
      if (output.second) {
        delete output.second;
        output.second = nullptr;
      }
    }
  }
};
} // namespace VPF

const PacketData& DecodeFrame::GetLastPacketData() const {
  return pImpl->m_packetData;
}

TaskExecStatus DecodeFrame::Run() {
  ClearOutputs();

  auto dst = GetInput(0U);
  if (!dst) {
    return TaskExecStatus::TASK_EXEC_FAIL;
  }

  if (pImpl->DecodeSingleFrame(this, *dst)) {
    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }

  return TaskExecStatus::TASK_EXEC_FAIL;
}

uint32_t DecodeFrame::GetHostFrameSize() const {
  return pImpl->GetHostFrameSize();
}

void DecodeFrame::GetParams(MuxingParams& params) {
  memset((void*)&params, 0, sizeof(params));
  auto fmtc = pImpl->m_fmt_ctx.get();
  pImpl->m_timeout_handler->Reset();
  auto videoStream =
      av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (videoStream < 0) {
    std::stringstream ss;
    ss << __FUNCTION__ << ": can't find video stream in input file.";
    throw std::runtime_error(ss.str());
  }

  params.videoContext.width = fmtc->streams[videoStream]->codecpar->width;
  params.videoContext.height = fmtc->streams[videoStream]->codecpar->height;
  params.videoContext.frameRate =
      (double)fmtc->streams[videoStream]->r_frame_rate.num /
      (double)fmtc->streams[videoStream]->r_frame_rate.den;
  params.videoContext.avgFrameRate =
      (double)fmtc->streams[videoStream]->avg_frame_rate.num /
      (double)fmtc->streams[videoStream]->avg_frame_rate.den;
  params.videoContext.timeBase =
      (double)fmtc->streams[videoStream]->time_base.num /
      (double)fmtc->streams[videoStream]->time_base.den;
  params.videoContext.codec = FFmpeg2NvCodecId(pImpl->m_avc_ctx->codec_id);
  params.videoContext.num_frames = fmtc->streams[videoStream]->nb_frames;

  const auto format = IsAccelerated() ? pImpl->m_avc_ctx->sw_pix_fmt
                                      : pImpl->m_avc_ctx->pix_fmt;
  switch (format) {
  case AV_PIX_FMT_NV12:
    params.videoContext.format = NV12;
    break;
  case AV_PIX_FMT_YUVJ420P:
  case AV_PIX_FMT_YUV420P:
    params.videoContext.format = YUV420;
    break;
  case AV_PIX_FMT_YUV444P:
    params.videoContext.format = YUV444;
    break;
  case AV_PIX_FMT_YUV422P:
    params.videoContext.format = YUV422;
    break;
  case AV_PIX_FMT_YUV420P10:
    params.videoContext.format = P10;
    break;
  case AV_PIX_FMT_YUV420P12:
    params.videoContext.format = P12;
    break;
  case AV_PIX_FMT_GRAY12LE:
    params.videoContext.format = GRAY12;
    break;
  default:
    std::stringstream ss;
    ss << "Unsupported FFmpeg pixel format: "
       << av_get_pix_fmt_name(pImpl->m_avc_ctx->pix_fmt);
    throw std::invalid_argument(ss.str());
    params.videoContext.format = UNDEFINED;
    break;
  }

  switch (fmtc->streams[videoStream]->codecpar->color_space) {
  case AVCOL_SPC_BT709:
    params.videoContext.color_space = BT_709;
    break;
  case AVCOL_SPC_BT470BG:
  case AVCOL_SPC_SMPTE170M:
    params.videoContext.color_space = BT_601;
    break;
  default:
    params.videoContext.color_space = UNSPEC;
    break;
  }

  switch (fmtc->streams[videoStream]->codecpar->color_range) {
  case AVCOL_RANGE_MPEG:
    params.videoContext.color_range = MPEG;
    break;
  case AVCOL_RANGE_JPEG:
    params.videoContext.color_range = JPEG;
    break;
  default:
    params.videoContext.color_range = UDEF;
    break;
  }
}

TaskExecStatus DecodeFrame::GetSideData(AVFrameSideDataType data_type) {
  SetOutput(nullptr, 0U);
  auto it = pImpl->m_side_data.find(data_type);
  if (it != pImpl->m_side_data.end()) {
    SetOutput((Token*)it->second, 0U);
    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }

  return TaskExecStatus::TASK_EXEC_FAIL;
}

DecodeFrame* DecodeFrame::Make(const char* URL, NvDecoderClInterface& cli_iface,
                               std::optional<CUstream> stream) {
  return new DecodeFrame(URL, cli_iface, stream);
}

DecodeFrame::DecodeFrame(const char* URL, NvDecoderClInterface& cli_iface,
                         std::optional<CUstream> stream)
    : Task("DecodeFrame", DecodeFrame::num_inputs, DecodeFrame::num_outputs) {
  std::map<std::string, std::string> ffmpeg_options;
  cli_iface.GetOptions(ffmpeg_options);

  // Try to use HW acceleration first and fall back to SW decoding
  try {
    pImpl = new FfmpegDecodeFrame_Impl(URL, ffmpeg_options, stream);
  } catch (std::exception& e) {
    std::cerr << "Failed to create HW decoder. Reason: " << e.what()
              << ". Using SW decoder.";
    pImpl = new FfmpegDecodeFrame_Impl(URL, ffmpeg_options, std::nullopt);
  }
}

DecodeFrame::~DecodeFrame() { delete pImpl; }

bool DecodeFrame::IsAccelerated() const {
  return pImpl->m_avc_ctx->hw_frames_ctx != nullptr;
}