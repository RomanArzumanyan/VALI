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

#include "CodecsSupport.hpp"
#include "CudaUtils.hpp"
#include "Tasks.hpp"
#include "Utils.hpp"

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
  std::shared_ptr<AVDictionary> m_options;
  std::shared_ptr<TimeoutHandler> m_timeout_handler;
  PacketData m_packetData;
  CUstream m_stream;

  int m_video_str_idx = -1;
  bool end_decode = false;
  bool eof = false;

  void CloseCodec() {
    auto ret = avcodec_close(m_avc_ctx.get());
    if (ret < 0) {
      std::cerr << "Failed to close codec: " << AvErrorToString(ret);
      return;
    }

    m_avc_ctx.reset();
  }

  /* Extracted to standalone function because in some cases we need to (re)open
   * video codec.
   */
  void OpenCodec(bool is_accelerated) {
    auto video_stream = m_fmt_ctx->streams[GetVideoStrIdx()];
    if (!video_stream) {
      std::stringstream ss;
      ss << "Could not find video stream in the input, aborting";
      throw std::runtime_error(ss.str());
    }

    auto p_codec =
        is_accelerated
            ? avcodec_find_decoder_by_name(
                  FindDecoderById(video_stream->codecpar->codec_id).c_str())
            : avcodec_find_decoder(video_stream->codecpar->codec_id);

    if (!p_codec && is_accelerated) {
      throw std::runtime_error(
          "Failed to find codec by name: " +
          FindDecoderById(video_stream->codecpar->codec_id));
    } else if (!p_codec) {
      throw std::runtime_error(
          "Failed to find codec by id: " +
          std::string(avcodec_get_name(video_stream->codecpar->codec_id)));
    }

    auto avctx = avcodec_alloc_context3(p_codec);
    if (!avctx) {
      std::stringstream ss;
      ss << "Failed to allocate codec context";
      throw std::runtime_error(ss.str());
    }
    m_avc_ctx = std::shared_ptr<AVCodecContext>(
        avctx, [](void* p) { avcodec_free_context((AVCodecContext**)&p); });

    auto res =
        avcodec_parameters_to_context(m_avc_ctx.get(), video_stream->codecpar);
    if (res < 0) {
      std::stringstream ss;
      ss << "Failed to pass codec parameters to codec "
            "context "
         << av_get_media_type_string(AVMEDIA_TYPE_VIDEO);
      ss << "Error description: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    AVDictionary* options = nullptr;
    res = av_dict_copy(&options, m_options.get(), 0);
    if (res < 0) {
      if (options) {
        av_dict_free(&options);
      }
      std::stringstream ss;
      ss << "Could not copy AVOptions: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

/* Have no idea if Tegra supports output to CUDA memory so keep it under
 * conditional compilation. Tegra has unified memory anyway, so no big
 * performance penalty shall be imposed as there's no "actual" Host <> Device IO
 * happening.
 */
#ifndef TEGRA_BUILD
    if (is_accelerated) {
      /* Attach HW context to codec. Whithout that decoded frames will be
       * copied to RAM.
       */
      AVBufferRef* hwdevice_ctx = nullptr;
      auto res = av_hwdevice_ctx_create(&hwdevice_ctx, AV_HWDEVICE_TYPE_CUDA,
                                        NULL, options, 0);

      if (res < 0) {
        std::stringstream ss;
        ss << "Failed to create HW device context: " << AvErrorToString(res);
        throw std::runtime_error(ss.str());
      }

      // Add hw device context to codec context.
      m_avc_ctx->hw_device_ctx = av_buffer_ref(hwdevice_ctx);
      m_avc_ctx->get_format = get_format;
    }
#endif

    res = avcodec_open2(m_avc_ctx.get(), p_codec, &options);
    if (options) {
      av_dict_free(&options);
    }

    if (res < 0) {
      std::stringstream ss;
      ss << "Failed to open codec "
         << av_get_media_type_string(AVMEDIA_TYPE_VIDEO);
      ss << "Error description: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }
  }

  FfmpegDecodeFrame_Impl(
      const char* URL, const std::map<std::string, std::string>& ffmpeg_options,
      std::optional<CUstream> stream) {

    // Set log level to error to avoid noisy output
    av_log_set_level(AV_LOG_ERROR);

    // Allocate format context first to set timeout before opening the input.
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    if (!fmt_ctx) {
      throw std::runtime_error("Failed to allocate format context");
    }

    /* Set neccessary AVOptions for HW decoding.
     */
    auto options = GetAvOptions(ffmpeg_options);
    if (stream) {
      m_stream = stream.value();
      auto res =
          av_dict_set(&options, "hwaccel_device",
                      std::to_string(GetDeviceIdByStream(m_stream)).c_str(), 0);
      if (res < 0) {
        av_dict_free(&options);
        std::stringstream ss;
        ss << "Failed to set hwaccel_device AVOption: " << AvErrorToString(res);
        throw std::runtime_error(ss.str());
      }

      res = av_dict_set(&options, "hwaccel", "cuvid", 0);
      if (res < 0) {
        av_dict_free(&options);
        std::stringstream ss;
        ss << "Failed to set hwaccel AVOption: " << AvErrorToString(res);
        throw std::runtime_error(ss.str());
      }

      res = av_dict_set(&options, "fps_mode", "passthrough", 0);
      if (res < 0) {
        av_dict_free(&options);
        std::stringstream ss;
        ss << "Failed set fps_mode AVOption: " << AvErrorToString(res);
        throw std::runtime_error(ss.str());
      }

      res = av_dict_set(&options, "threads", "1", 0);
      if (res < 0) {
        av_dict_free(&options);
        std::stringstream ss;
        ss << "Failed set threads AVOption: " << AvErrorToString(res);
        throw std::runtime_error(ss.str());
      }
    }

    /* After we are done settings options, save them in case we need them
     * later to (re)open codec;
     */
    m_options = std::shared_ptr<AVDictionary>(
        options, [](void* p) { av_dict_free((AVDictionary**)&p); });

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

    /* Copy class member options because some avcodec API functions like to
     * free input options and replace them with list of unrecognized options.
     */
    options = nullptr;
    auto res = av_dict_copy(&options, m_options.get(), 0);
    if (res < 0) {
      if (options) {
        av_dict_free(&options);
      }
      std::stringstream ss;
      ss << "Could not copy AVOptions: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    m_timeout_handler->Reset();
    res = avformat_open_input(&fmt_ctx, URL, NULL, &options);
    if (options) {
      av_dict_free(&options);
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
        fmt_ctx, [](void* p) { avformat_close_input((AVFormatContext**)&p); });

    m_timeout_handler->Reset();
    res = avformat_find_stream_info(m_fmt_ctx.get(), NULL);
    if (res < 0) {
      std::stringstream ss;
      ss << "Could not find stream information";
      ss << "Error description: " << AvErrorToString(res);
      throw std::runtime_error(ss.str());
    }

    m_timeout_handler->Reset();
    m_video_str_idx = av_find_best_stream(m_fmt_ctx.get(), AVMEDIA_TYPE_VIDEO,
                                          -1, -1, NULL, 0);
    if (GetVideoStrIdx() < 0) {
      std::stringstream ss;
      ss << "Could not find " << av_get_media_type_string(AVMEDIA_TYPE_VIDEO)
         << " stream in file " << URL;
      ss << "Error description: " << AvErrorToString(GetVideoStrIdx());
      throw std::runtime_error(ss.str());
    }

    OpenCodec(stream.has_value());

    auto frame = av_frame_alloc();
    if (!frame) {
      std::stringstream ss;
      ss << "Could not allocate frame";
      throw std::runtime_error(ss.str());
    }

    m_frame = std::shared_ptr<AVFrame>(
        frame, [](void* p) { av_frame_free((AVFrame**)&p); });

    auto avpkt = av_packet_alloc();
    if (!avpkt) {
      std::stringstream ss;
      ss << "Could not allocate packet";
      throw std::runtime_error(ss.str());
    }
    m_pkt = std::shared_ptr<AVPacket>(
        avpkt, [](void* p) { av_packet_free((AVPacket**)&p); });
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
      } while (m_pkt->stream_index != GetVideoStrIdx());

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
    /* Query codec context because this method may be called before decoder
     * returns any decoded frames.
     */
    const auto format =
        m_avc_ctx->hw_frames_ctx ? m_avc_ctx->sw_pix_fmt : m_avc_ctx->pix_fmt;
    const auto width = m_avc_ctx->width;
    const auto height = m_avc_ctx->height;
    const auto alignment = 1;

    const auto size = av_image_get_buffer_size((AVPixelFormat)format, width,
                                               height, alignment);
    if (size < 0) {
      std::stringstream ss;
      ss << "Failed to query host frame size: " << AvErrorToString(size);
      throw std::runtime_error(ss.str());
    }

    return static_cast<uint32_t>(size);
  }

  static void CopyToSurface(AVFrame& src, Surface& dst, CUstream stream) {
    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

    CudaCtxPush push_ctx(GetContextByStream(stream));
    for (auto i = 0U; src.data[i]; i++) {
      m.srcDevice = (CUdeviceptr)src.data[i];
      m.srcPitch = src.linesize[i];
      m.dstDevice = dst.PixelPtr(i);
      m.dstPitch = dst.Pitch(i);
      m.WidthInBytes = dst.Width(i) * dst.ElemSize();
      m.Height = dst.Height(i);

      ThrowOnCudaError(cuMemcpy2DAsync(&m, stream), __LINE__);
    }
    ThrowOnCudaError(cuStreamSynchronize(stream), __LINE__);
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
        // Codec has HW acceleration and outputs to CUDA memory
        try {
          CopyToSurface(*m_frame.get(), dynamic_cast<Surface&>(dst), m_stream);
        } catch (std::exception& e) {
          std::cerr << "Error while copying a surface from FFMpeg to VALI";
          std::cerr << "Error description: " << AvErrorToString(res);
          return DEC_ERROR;
        }
      } else {
        // No HW acceleration, outputs to RAM
        auto& dstBuf = dynamic_cast<Buffer&>(dst);
        const int alignment = 1;

        res = av_image_copy_to_buffer(
            dstBuf.GetDataAs<uint8_t>(), GetHostFrameSize(), m_frame->data,
            m_frame->linesize, (AVPixelFormat)m_frame->format, m_frame->width,
            m_frame->height, alignment);

        if (res < 0) {
          std::cerr << "Error while copying a frame from FFMpeg to VALI";
          std::cerr << "Error description: " << AvErrorToString(res);
          return DEC_ERROR;
        }
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

  int GetVideoStrIdx() const { return m_video_str_idx; }

  double GetFrameRate() const {
    return (double)m_fmt_ctx->streams[GetVideoStrIdx()]->r_frame_rate.num /
           (double)m_fmt_ctx->streams[GetVideoStrIdx()]->r_frame_rate.den;
  }

  double GetAvgFrameRate() const {
    return (double)m_fmt_ctx->streams[GetVideoStrIdx()]->avg_frame_rate.num /
           (double)m_fmt_ctx->streams[GetVideoStrIdx()]->avg_frame_rate.den;
  }

  double GetTimeBase() const {
    return (double)m_fmt_ctx->streams[GetVideoStrIdx()]->time_base.num /
           (double)m_fmt_ctx->streams[GetVideoStrIdx()]->time_base.den;
  }

  int GetWidth() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->width;
  }

  int GetHeight() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->height;
  }

  AVCodecID GetCodecId() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->codec_id;
  }

  int64_t GetNumFrames() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->nb_frames;
  }

  int64_t GetStartTime() const { return m_fmt_ctx->start_time; }

  int64_t GetStreamStartTime() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->start_time;
  }

  Pixel_Format GetPixelFormat() const {
    auto const format =
        IsAccelerated() ? m_avc_ctx->sw_pix_fmt : m_avc_ctx->pix_fmt;

    switch (format) {
    case AV_PIX_FMT_NV12:
      return NV12;
    case AV_PIX_FMT_YUVJ420P:
    case AV_PIX_FMT_YUV420P:
      return YUV420;
    case AV_PIX_FMT_YUV444P:
      return YUV444;
    case AV_PIX_FMT_YUV422P:
      return YUV422;
    case AV_PIX_FMT_YUV420P10:
      return P10;
    case AV_PIX_FMT_YUV420P12:
      return P12;
    case AV_PIX_FMT_GRAY12LE:
      return GRAY12;
    default:
      return UNDEFINED;
    }
  }

  AVColorSpace GetColorSpace() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->color_space;
  }

  AVColorRange GetColorRange() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->color_range;
  }

  bool IsVFR() const { return GetFrameRate() != GetAvgFrameRate(); }

  bool IsAccelerated() const { return m_avc_ctx->hw_frames_ctx != nullptr; }

  int64_t TsFromTime(double ts_sec) {
    /* Timestasmp in stream time base units.
     * Internal timestamp representation is integer, so multiply to AV_TIME_BASE
     * and switch to fixed point precision arithmetics.
     */
    auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);

    // Rescale the timestamp to value represented in stream time base units;
    AVRational factor = {1, AV_TIME_BASE};
    return av_rescale_q(ts_tbu, factor,
                        m_fmt_ctx->streams[GetVideoStrIdx()]->time_base);
  }

  int64_t TsFromFrameNumber(int64_t frame_num) {
    auto const ts_sec = (double)frame_num / GetFrameRate();
    return TsFromTime(ts_sec);
  }

  TaskExecStatus SeekDecode(DecodeFrame* parent, Token& dst,
                            const SeekContext& ctx) {
    /* Across this function packet presentation timestamp (PTS) values are used
     * to compare given timestamp against. That's done so because ffmpeg seek
     * relies on PTS.
     */
    if (!parent) {
      std::cerr << "Empty parent task given.";
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    if (IsVFR() && ctx.IsByNumber()) {
      parent->SetExecDetails(TaskExecDetails(TaskExecInfo::NOT_SUPPORTED));
      std::cerr
          << "Can't seek by frame number in VFR sequences. Seek by timestamp "
             "instead.";
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    /* Some formats dont have a clear idea of what frame number is, so always
     * convert to timestamp. BTW that's the reason seek by frame number is not
     * supported for VFR videos.
     *
     * Also we always seek backwards to nearest previous key frame and then
     * decode in the loop starting from it until we reach a frame with
     * desired timestamp. Sic !
     */
    auto timestamp = ctx.IsByNumber() ? TsFromFrameNumber(ctx.seek_frame)
                                      : TsFromTime(ctx.seek_tssec);
    if (AV_NOPTS_VALUE != GetStreamStartTime()) {
      timestamp += GetStreamStartTime();
    }
    m_timeout_handler->Reset();
    auto ret = av_seek_frame(m_fmt_ctx.get(), GetVideoStrIdx(), timestamp,
                             AVSEEK_FLAG_BACKWARD);

    if (ret < 0) {
      parent->SetExecDetails(TaskExecDetails(TaskExecInfo::FAIL));
      std::cerr << "Error seeking for frame: " + AvErrorToString(ret);
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    /* Decode in loop until we reach desired frame.
     */
    auto const was_accelerated = IsAccelerated();
    CloseCodec();
    OpenCodec(was_accelerated);

    while (m_frame->pts < timestamp) {
      if (!DecodeSingleFrame(parent, dst)) {
        parent->SetExecDetails(TaskExecDetails(TaskExecInfo::FAIL));
        std::cerr << "Failed to decode frame during seek.";
        return TaskExecStatus::TASK_EXEC_FAIL;
      }
    }

    return TaskExecStatus::TASK_EXEC_SUCCESS;
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

  auto seek_ctx_buf = static_cast<Buffer*>(GetInput(1U));
  if (seek_ctx_buf) {
    auto seek_ctx = seek_ctx_buf->GetDataAs<SeekContext>();
    return pImpl->SeekDecode(this, *dst, *seek_ctx);
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

  params.videoContext.width = pImpl->GetWidth();
  params.videoContext.height = pImpl->GetHeight();
  params.videoContext.frameRate = pImpl->GetFrameRate();
  params.videoContext.avgFrameRate = pImpl->GetAvgFrameRate();
  params.videoContext.timeBase = pImpl->GetTimeBase();
  params.videoContext.num_frames = pImpl->GetNumFrames();
  params.videoContext.format = pImpl->GetPixelFormat();
  params.videoContext.color_range =
      fromFfmpegColorRange(pImpl->GetColorRange());

  switch (pImpl->GetColorSpace()) {
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

bool DecodeFrame::IsAccelerated() const { pImpl->IsAccelerated(); }

bool DecodeFrame::IsVFR() const { return pImpl->IsVFR(); }