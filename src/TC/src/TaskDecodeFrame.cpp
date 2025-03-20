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

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/imgutils.h>
#include <libavutil/motion_vector.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
}

using namespace VPF;

namespace VPF {

enum DECODE_STATUS {
  DEC_SUCCESS,
  DEC_ERROR,
  DEC_EOS,
  DEC_MORE,
  DEC_RES_CHANGE
};

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
      frames_ctx->width = avctx->width;
      frames_ctx->height = avctx->height;

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
  std::shared_ptr<AVBufferRef> m_hw_ctx;
  std::map<AVFrameSideDataType, Buffer*> m_side_data;
  std::shared_ptr<AVDictionary> m_options;
  std::shared_ptr<TimeoutHandler> m_timeout_handler;
  std::shared_ptr<AVIOContext> m_io_ctx;
  PacketData m_packet_data;
  CUstream m_stream;

  // GPU id
  int m_gpu_id = -1;

  // Video stream index
  int m_stream_idx = -1;

  // Last known width
  int m_last_w = -1;

  // Last known height
  int m_last_h = -1;

  // User prefferred stream width. Mostly useful for HLS ABR streams.
  int m_preferred_width = -1;

  // Flag which signals that decode is done
  bool m_end_decode = false;

  // Flag which signals that decoder needs to be flushed
  bool m_flush = false;

  // Flag which signals that current packet needs to be resent
  bool m_resend = false;

  // Flag which signals end of file
  bool m_eof = false;

  // Flag which signals resolution change
  bool m_res_change = false;

  /* These are handy counters for debug:
   *
   * Packets read.
   * Packets sent.
   * Frames received.
   *
   * Since incrementing them is negligible overhead, only output is put under
   * conditional compilation. Change #if in class destructor to output values.
   *
   * Please note that seek heavy influences the difference between these
   * counters values, that's ok.
   */
  uint32_t m_num_pkt_read = 0U;
  uint32_t m_num_pkt_sent = 0U;
  uint32_t m_num_frm_recv = 0U;

  /// @brief Find stream with desired width
  /// @return Stream id which has the desired width, -1 if not found
  int FindStreamByWidth() {
    if (m_preferred_width == -1)
      return -1;

    for (int i = 0; m_fmt_ctx->streams[i]; i++)
      if (m_fmt_ctx->streams[i]->codecpar->width == m_preferred_width)
        return i;

    return -1;
  }

  FfmpegDecodeFrame_Impl(const char* URL,
                         std::map<std::string, std::string>& ffmpeg_options,
                         int gpu_id, std::shared_ptr<AVIOContext> p_io_ctx)
      : m_io_ctx(p_io_ctx) {

    // Extract preferred width from options because it's not ffmpeg option.
    auto it = ffmpeg_options.find("preferred_width");
    if (ffmpeg_options.end() != it) {
      m_preferred_width = std::stoi(it->second);
      ffmpeg_options.erase(it);
    }

    // Allocate format context first to set timeout before opening the input.
    AVFormatContext* fmt_ctx = avformat_alloc_context();
    if (!fmt_ctx) {
      throw std::runtime_error("Failed to allocate format context");
    }

    /* Have to determine input format by hand if using custom IO context.
     * Otherwise libavformat may read couple MB of input data to do that.
     * There's no way to tell how much data libavformat will need and that
     * amount may exceed the custom AVIOFormat buffer size.
     */
    if (m_io_ctx) {
      fmt_ctx->pb = m_io_ctx.get();
      fmt_ctx->flags |= AVFMT_FLAG_CUSTOM_IO;

      if (fmt_ctx->pb->seek) {
        std::array<uint8_t, 1024U> probe;
        auto nbytes = fmt_ctx->pb->read_packet(fmt_ctx->pb->opaque,
                                               probe.data(), probe.size());
        fmt_ctx->pb->seek(fmt_ctx->pb->opaque, 0U, SEEK_SET);

        AVProbeData probe_data = {};
        probe_data.buf = probe.data();
        probe_data.buf_size = nbytes;
        probe_data.filename = "";

        fmt_ctx->iformat = av_probe_input_format(&probe_data, 1);
      }
    }

    // Set neccessary AVOptions for HW decoding.
    auto options = GetAvOptions(ffmpeg_options);
    if (gpu_id >= 0) {
      m_gpu_id = gpu_id;
      auto res = av_dict_set(&options, "hwaccel_device",
                             std::to_string(m_gpu_id).c_str(), 0);
      ThrowOnAvError(res, "Failed to set hwaccel_device AVOption", &options);

      res = av_dict_set(&options, "current_ctx", "1", 0);
      ThrowOnAvError(res, "Failed to set current_ctx AVOption", &options);
    }

    /* After we are done settings options, save them in case we need them
     * later to (re)open codec;
     */
    m_options = std::shared_ptr<AVDictionary>(
        options, [](void* p) { av_dict_free((AVDictionary**)&p); });

    // Set the timeout.
    m_timeout_handler.reset(new TimeoutHandler(&options, fmt_ctx));

    /* Copy class member options because some avcodec API functions like to
     * free input options and replace them with list of unrecognized options.
     */
    options = nullptr;
    auto res = av_dict_copy(&options, m_options.get(), 0);
    ThrowOnAvError(res, "Can't copy AVOptions", options ? &options : nullptr);

    m_timeout_handler->Reset();
    res = avformat_open_input(&fmt_ctx, m_io_ctx ? "" : URL, NULL, &options);
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

      ThrowOnAvError(res, "Can't open souce file " + std::string(URL), nullptr);
    }

    m_fmt_ctx = std::shared_ptr<AVFormatContext>(
        fmt_ctx, [](void* p) { avformat_close_input((AVFormatContext**)&p); });

    m_timeout_handler->Reset();
    res = avformat_find_stream_info(m_fmt_ctx.get(), NULL);
    ThrowOnAvError(res, "Can't find stream information", nullptr);

    m_timeout_handler->Reset();
    m_stream_idx = av_find_best_stream(m_fmt_ctx.get(), AVMEDIA_TYPE_VIDEO,
                                       FindStreamByWidth(), -1, NULL, 0);

    if (GetVideoStrIdx() < 0) {
      std::stringstream ss;
      ss << "Could not find " << av_get_media_type_string(AVMEDIA_TYPE_VIDEO)
         << " stream in file " << URL;
      ss << "Error description: " << AvErrorToString(GetVideoStrIdx());
      throw std::runtime_error(ss.str());
    }

    OpenCodec(m_gpu_id >= 0);

    m_frame = std::shared_ptr<AVFrame>(av_frame_alloc(), [](void* p) {
      av_frame_unref((AVFrame*)p);
      av_frame_free((AVFrame**)&p);
    });

    m_pkt = std::shared_ptr<AVPacket>(av_packet_alloc(), [](void* p) {
      av_packet_unref((AVPacket*)p);
      av_packet_free((AVPacket**)&p);
    });
  }

  // Saves current resolution
  void SaveCurrentRes() {
    m_last_h = GetHeight();
    m_last_w = GetWidth();
  }

  /* Closes codec context and resets the smart pointer.
   *
   * Extracted to standalone function because in some cases we need to (re)open
   * video codec.
   */
  void CloseCodec() {
    auto ret = avcodec_close(m_avc_ctx.get());
    if (ret < 0) {
      std::cerr << "Failed to close codec: " << AvErrorToString(ret);
      return;
    }

    m_avc_ctx.reset();
  }

  /* Allocates video codec contet and opens it.
   *
   * Extracted to standalone function because in some cases we need to (re)open
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
      // Push CUDA context for FFMpeg to use it
      auto ctx = CudaResMgr::Instance().GetCtx(m_gpu_id);
      CudaCtxPush push_ctx(ctx);

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

      m_hw_ctx.reset();
      m_hw_ctx = std::shared_ptr<AVBufferRef>(
          hwdevice_ctx, [](void* p) { av_buffer_unref((AVBufferRef**)&p); });

      // Get stream from libavcodec
      auto av_hw_ctx = (AVHWDeviceContext*)m_avc_ctx->hw_device_ctx->data;
      auto av_cuda_ctx = (AVCUDADeviceContext*)av_hw_ctx->hwctx;
      m_stream = av_cuda_ctx->stream;
    }
#endif

    /* Set packet time base here because later packet PTS values will be
     * discarded. Without that, libavcodec won't be able to reconstruct
     * correct PTS values.
     */
    m_avc_ctx->pkt_timebase = m_fmt_ctx->streams[GetVideoStrIdx()]->time_base;

    res = avcodec_open2(m_avc_ctx.get(), p_codec, &options);
    if (options) {
      av_dict_free(&options);
    }

    ThrowOnAvError(
        res, "Failed to open codec " +
                 std::string(av_get_media_type_string(AVMEDIA_TYPE_VIDEO)));
  }

  CUstream GetStream() { return m_stream; }

  void SavePacketData() {
    m_packet_data = {};
    m_packet_data.pts = m_frame->pts;
    m_packet_data.key = (m_frame->flags & AV_FRAME_FLAG_KEY) != 0;

    if (!IsAccelerated()) {
      // Cuvid doesn't set these fields correctly.
      m_packet_data.dts = m_frame->pkt_dts;
      m_packet_data.duration = m_frame->duration;
      m_packet_data.pos = m_frame->pkt_pos;
    }
  }

  TaskExecDetails DecodeSingleFrame(Token& dst) {
    if (m_end_decode) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                             "decode finished");
    }

    // Send packets to decoder until it outputs frame;
    do {
      // Read packets from stream until we find a video packet;
      do {
        if (m_eof || m_flush) {
          break;
        }

        if (m_resend) {
          m_resend = false;
          break;
        }

        m_timeout_handler->Reset();
        auto ret = av_read_frame(m_fmt_ctx.get(), m_pkt.get());

        if (AVERROR_EOF == ret) {
          m_eof = true;
          break;
        } else if (ret < 0) {
          m_end_decode = true;
          return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                                 TaskExecInfo::FAIL, AvErrorToString(ret));
        } else {
          m_num_pkt_read++;
        }
      } while (m_pkt->stream_index != GetVideoStrIdx());

      auto status = DecodeSinglePacket(m_eof ? nullptr : m_pkt.get(), dst);

      switch (status) {
      case DEC_SUCCESS:
        return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                               TaskExecInfo::SUCCESS);
      case DEC_ERROR:
        m_end_decode = true;
        return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                               TaskExecInfo::FAIL, "decode error, end decode");
      case DEC_EOS:
        m_end_decode = true;
        return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                               TaskExecInfo::END_OF_STREAM, "end of stream");
      case DEC_RES_CHANGE:
        return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                               TaskExecInfo::RES_CHANGE, "resolution change");
      case DEC_MORE:
        // Just continue the loop to get more data;
        break;
      default:
        return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL,
                               TaskExecInfo::FAIL, "unknown decode error");
      }
    } while (true);

    return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                           TaskExecInfo::SUCCESS);
  }

  void SaveMotionVectors() {
    AVFrameSideDataType type = AV_FRAME_DATA_MOTION_VECTORS;
    AVFrameSideData* sd = av_frame_get_side_data(m_frame.get(), type);

    if (sd) {
      auto it = m_side_data.find(type);
      if (it == m_side_data.end()) {
        m_side_data[type] = Buffer::MakeOwnMem(sd->size);
        it = m_side_data.find(type);
        memcpy(it->second->GetRawMemPtr(), sd->data, sd->size);
      } else if (it->second->GetRawMemSize() != sd->size) {
        it->second->Update(sd->size, sd->data);
      }
    }
  }

  void SaveDisplayMatrix() {
    AVFrameSideDataType type = AV_FRAME_DATA_DISPLAYMATRIX;
    AVFrameSideData* sd = av_frame_get_side_data(m_frame.get(), type);

    if (!sd)
      return;

    auto angle = av_display_rotation_get((int32_t*)sd->data);
    auto it = m_side_data.find(type);

    if (it == m_side_data.end()) {
      m_side_data[type] = Buffer::MakeOwnMem(sizeof(angle));
      it = m_side_data.find(type);
      memcpy(it->second->GetRawMemPtr(), (void*)&angle, sizeof(angle));
    } else if (it->second->GetRawMemSize() != sd->size) {
      it->second->Update(sizeof(angle), (void*)&angle);
    }
  }

  bool SaveSideData() {
    SaveMotionVectors();
    SaveDisplayMatrix();
    return true;
  }

  uint32_t GetHostFrameSize() const {
    const auto format = toFfmpegPixelFormat(GetPixelFormat());
    const auto alignment = 1;
    const auto size =
        av_image_get_buffer_size(format, GetWidth(), GetHeight(), alignment);

    ThrowOnAvError(size, "Failed to query host frame size: ");
    return static_cast<uint32_t>(size);
  }

  void CopyToSurface(AVFrame& src, Surface& dst) {
    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;

    auto av_hw_ctx = (AVHWDeviceContext*)m_avc_ctx->hw_device_ctx->data;
    auto av_cuda_ctx = (AVCUDADeviceContext*)av_hw_ctx->hwctx;

    if (av_cuda_ctx->cuda_ctx != dst.Context()) {
      throw std::runtime_error("CUDA context mismatch between FFMpeg and VALI");
    }

    if (av_cuda_ctx->stream != m_stream) {
      throw std::runtime_error("CUDA stream mismatch between FFMpeg and VALI");
    }

    CudaCtxPush push_ctx(av_cuda_ctx->cuda_ctx);

    for (auto i = 0U; src.data[i]; i++) {
      m.srcDevice = (CUdeviceptr)src.data[i];
      m.srcPitch = src.linesize[i];
      m.dstDevice = dst.PixelPtr(i);
      m.dstPitch = dst.Pitch(i);
      m.WidthInBytes = dst.Width(i) * dst.ElemSize();
      m.Height = dst.Height(i);

      ThrowOnCudaError(LibCuda::cuMemcpy2DAsync(&m, m_stream), __LINE__);
    }
  }

  bool UpdGetResChange() {
    m_res_change = (m_last_h != GetHeight()) || (m_last_w != GetWidth());
    return m_res_change;
  }

  bool FlipGetResChange() {
    m_res_change = !m_res_change;
    return !m_res_change;
  }

  /* Copy last decoded frame to output token.
   * It doesn't check if memory amount is sufficient.
   */
  DECODE_STATUS GetLastFrame(Token& dst) {
    if (m_frame->hw_frames_ctx) {
      // Codec has HW acceleration and outputs to CUDA memory
      try {
        CopyToSurface(*m_frame.get(), dynamic_cast<Surface&>(dst));
      } catch (std::exception& e) {
        std::cerr << "Error while copying a surface from FFMpeg to VALI: "
                  << e.what();
        return DEC_ERROR;
      }
    } else {
      // No HW acceleration, outputs to RAM
      auto& dstBuf = dynamic_cast<Buffer&>(dst);
      const int alignment = 1;

      auto res = av_image_copy_to_buffer(
          dstBuf.GetDataAs<uint8_t>(), GetHostFrameSize(), m_frame->data,
          m_frame->linesize, (AVPixelFormat)m_frame->format, m_frame->width,
          m_frame->height, alignment);

      if (res < 0) {
        std::cerr << "Error while copying a frame from FFMpeg to VALI: "
                  << AvErrorToString(res);
        return DEC_ERROR;
      }
    }

    return DEC_SUCCESS;
  }

  /* Decodes single video frame.
   *
   * Upon successful decoder copies decoded frame to dst token and returns
   * DEC_SUCCESS.
   *
   * Upon resolution change doesn't copy decoded frame to dst token and
   * returns DEC_RES_CHANGE.
   *
   * Upon EOF returns DEC_EOS.
   *
   * If packet cant be sent it will return DEC_NEED_FLUSH.
   *
   * Upont error returns DEC_ERROR.
   */
  DECODE_STATUS DecodeSinglePacket(AVPacket* pkt, Token& dst) {
    SaveCurrentRes();
    int res = 0;

    if (!m_flush) {
      res = avcodec_send_packet(m_avc_ctx.get(), pkt);
      if (AVERROR_EOF == res) {
        // Flush decoder;
        res = 0;
      } else if (res == AVERROR(EAGAIN)) {
        // Need to call for avcodec_receive frame and then resend packet;
        m_flush = true;
      } else if (res < 0) {
        std::cerr << "Error while sending a packet to the decoder. ";
        std::cerr << "Error description: " << AvErrorToString(res);
        return DEC_ERROR;
      } else {
        m_num_pkt_sent++;
        if (pkt) {
          av_packet_unref(pkt);
        }
      }
    }

    res = avcodec_receive_frame(m_avc_ctx.get(), m_frame.get());
    if (res == AVERROR_EOF) {
      return DEC_EOS;
    } else if (res == AVERROR(EAGAIN)) {
      if (m_flush) {
        m_flush = false;
        m_resend = true;
      }
      return DEC_MORE;
    } else if (res < 0) {
      std::cerr << "Error while receiving a frame from the decoder. ";
      std::cerr << "Error description: " << AvErrorToString(res);
      return DEC_ERROR;
    } else {
      m_num_frm_recv++;
    }

    if (UpdGetResChange()) {
      return DEC_RES_CHANGE;
    }

    SaveSideData();
    SavePacketData();
    return GetLastFrame(dst);
  }

  ~FfmpegDecodeFrame_Impl() {
// For debug purposes
#if 0
    std::cout << "m_num_pkt_read: " << m_num_pkt_read << std::endl;
    std::cout << "m_num_pkt_sent: " << m_num_pkt_sent << std::endl;
    std::cout << "m_num_frm_recv: " << m_num_frm_recv << std::endl;
#endif

    for (auto& output : m_side_data) {
      if (output.second) {
        delete output.second;
        output.second = nullptr;
      }
    }
  }

  int GetVideoStrIdx() const { return m_stream_idx; }

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
    if (m_frame && m_frame->width > 0) {
      return m_frame->width;
    }

    return m_avc_ctx->width;
  }

  int GetHeight() const {
    if (m_frame && m_frame->height > 0) {
      return m_frame->height;
    }

    return m_avc_ctx->height;
  }

  AVCodecID GetCodecId() const { return m_avc_ctx->codec_id; }

  int64_t GetNumFrames() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->nb_frames;
  }

  int64_t GetStartTime() const { return m_fmt_ctx->start_time; }

  int64_t GetStreamStartTime() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->start_time;
  }

  double GetStartTimeS() const {
    return double(GetStreamStartTime()) / double(AV_TIME_BASE);
  }

  double GetDuration() const {
    return double(m_fmt_ctx->streams[GetVideoStrIdx()]->duration) /
           double(AV_TIME_BASE);
  }

  int64_t GetBitRate() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->bit_rate;
  }

  int64_t GetProfile() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->profile;
  }

  int64_t GetLevel() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->level;
  }

  int64_t GetDelay() const { return m_avc_ctx->delay; }

  int64_t GetNumStreams() const { return m_fmt_ctx->nb_streams; }

  int64_t GetGopSize() const { return m_avc_ctx->gop_size; }

  int64_t GetStreamIndex() const { return m_stream_idx; }

  Pixel_Format GetPixelFormat() const {
    auto const format =
        IsAccelerated() ? m_avc_ctx->sw_pix_fmt : m_avc_ctx->pix_fmt;
    auto fmt_name = av_get_pix_fmt_name(format);
    std::stringstream err_msg;

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
    case AV_PIX_FMT_P010:
      return P10;
    case AV_PIX_FMT_P012:
      return P12;
    default:
      err_msg << "Unknown pixel format: ";
      if (fmt_name) {
        err_msg << fmt_name;
      }
      std::cerr << err_msg.str() << "\n";
      return UNDEFINED;
    }
  }

  AVColorSpace GetColorSpace() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->color_space;
  }

  AVColorRange GetColorRange() const {
    return m_fmt_ctx->streams[GetVideoStrIdx()]->codecpar->color_range;
  }

  metadata_dict GetMetaData() const {
    metadata_dict metadata;
    std::map<std::string, AVDictionary*> sources = {
        {"context", m_fmt_ctx->metadata},
        {"video_stream", m_fmt_ctx->streams[GetVideoStrIdx()]->metadata}};

    for (auto& source : sources) {
      auto& name = source.first;
      auto& dict = source.second;

      auto tag = av_dict_iterate(dict, nullptr);
      while (tag) {
        metadata[name][tag->key] = tag->value;
        const auto prev_tag = tag;
        tag = av_dict_iterate(dict, prev_tag);
      }
    }

    return metadata;
  }

  bool IsVFR() const { return GetFrameRate() != GetAvgFrameRate(); }

  bool IsAccelerated() const { return m_avc_ctx->hw_frames_ctx != nullptr; }

  int64_t TsFromTime(double ts_sec) {
    /* Timestasmp in stream time base units.
     * Internal timestamp representation is integer, so multiply to
     * AV_TIME_BASE and switch to fixed point precision arithmetics.
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

  TaskExecDetails SeekDecode(Token& dst, const SeekContext& ctx) {
    /* If custom AVIOContext was used, have to check the seek support.
     * May not be enabled.
     */
    if (m_fmt_ctx->flags & AVFMT_FLAG_CUSTOM_IO) {
      if (!m_fmt_ctx->pb->seek) {
        return TaskExecDetails(
            TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::NOT_SUPPORTED,
            "Seek operation is not supported by AVIOContext.");
      }
    }

    /* Across this function packet presentation timestamp (PTS) values are
     * used to compare given timestamp against. That's done so because ffmpeg
     * seek relies on PTS.
     */
    if (IsVFR() && ctx.IsByNumber()) {
      return TaskExecDetails(
          TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::NOT_SUPPORTED,
          "Seek by frame number isn't supported for VFR sequences. "
          "Seek by timestamp instead");
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
    auto min_timestamp =
        ctx.IsByNumber()
            ? TsFromFrameNumber(std::max(ctx.seek_frame - GetGopSize(),
                                         static_cast<int64_t>(0)))
            : TsFromTime(std::max(ctx.seek_tssec - 1.0, 0.0));
    auto start_time = GetStreamStartTime();
    if (AV_NOPTS_VALUE != start_time) {
      timestamp += start_time;
      min_timestamp += start_time;
    } else {
      start_time = 0;
    }

    auto const was_accelerated = IsAccelerated();
    CloseCodec();
    OpenCodec(was_accelerated);

    m_timeout_handler->Reset();
    auto ret = avformat_seek_file(m_fmt_ctx.get(), GetVideoStrIdx(), 0,
                                  timestamp, timestamp, AVSEEK_FLAG_BACKWARD);

    if (ret < 0) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                             AvErrorToString(ret));
    } else {
      avcodec_flush_buffers(m_avc_ctx.get());
    }

    /* Discard existing frame timestamp and OEF flag.
     * Otherwise, seek will only go forward and will return EOF if seek is
     * done when decoder has previously get all available packets.
     */
    m_frame->pts = AV_NOPTS_VALUE;
    m_eof = false;

    /* Decode in loop until we reach desired frame.
     */
    while (m_frame->pts + start_time < timestamp) {
      auto details = DecodeSingleFrame(dst);
      if (details.m_status != TaskExecStatus::TASK_EXEC_SUCCESS) {
        return details;
      }
    }

    return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                           TaskExecInfo::SUCCESS);
  }
}; // namespace VPF
} // namespace VPF

TaskExecDetails DecodeFrame::Run(Token& dst, PacketData& pkt_data,
                                 std::optional<SeekContext> seek_ctx) {
  AtScopeExit set_pkt_data([&]() { pkt_data = pImpl->m_packet_data; });

  if (seek_ctx.has_value())
    return pImpl->SeekDecode(dst, seek_ctx.value());

  /* In case of resolution change decoder will reconstruct a frame but will not
   * return it to user because of inplace API. Amount of given memory
   * may be insufficient.
   *
   * So decoder will signal resolution change and stash decoded frame.
   * Next decode call will return stashed frame.
   */
  if (pImpl->FlipGetResChange()) {
    if (DEC_SUCCESS == pImpl->GetLastFrame(dst)) {
      return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                             TaskExecInfo::SUCCESS);
    }
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                           "decoder error upon resolution change");
  }

  return pImpl->DecodeSingleFrame(dst);
}

uint32_t DecodeFrame::GetHostFrameSize() const {
  return pImpl->GetHostFrameSize();
}

void DecodeFrame::GetParams(MuxingParams& params) {
  params = MuxingParams();

  params.videoContext.width = pImpl->GetWidth();
  params.videoContext.height = pImpl->GetHeight();
  params.videoContext.profile = pImpl->GetProfile();
  params.videoContext.level = pImpl->GetLevel();
  params.videoContext.delay = pImpl->GetDelay();
  params.videoContext.gop_size = pImpl->GetGopSize();
  params.videoContext.num_frames = pImpl->GetNumFrames();
  params.videoContext.is_vfr = pImpl->IsVFR();
  params.videoContext.num_streams = pImpl->GetNumStreams();
  params.videoContext.duration = pImpl->GetDuration();
  params.videoContext.stream_index = pImpl->GetStreamIndex();
  params.videoContext.host_frame_size = pImpl->GetHostFrameSize();
  params.videoContext.bit_rate = pImpl->GetBitRate();

  params.videoContext.frame_rate = pImpl->GetFrameRate();
  params.videoContext.avg_frame_rate = pImpl->GetAvgFrameRate();
  params.videoContext.time_base = pImpl->GetTimeBase();
  params.videoContext.start_time = pImpl->GetStartTimeS();

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

  params.videoContext.metadata = pImpl->GetMetaData();
}

TaskExecDetails DecodeFrame::GetSideData(AVFrameSideDataType data_type,
                                         Buffer& out) {
  auto it = pImpl->m_side_data.find(data_type);
  if (it != pImpl->m_side_data.end()) {
    out.Update(it->second->GetRawMemSize(), it->second->GetRawMemPtr());
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                           TaskExecInfo::SUCCESS);
  }

  return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                         "decoder failed to get side data");
}

DecodeFrame* DecodeFrame::Make(const char* URL, NvDecoderClInterface& cli_iface,
                               int gpu_id,
                               std::shared_ptr<AVIOContext> p_io_ctx) {
  return new DecodeFrame(URL, cli_iface, gpu_id, p_io_ctx);
}

/* Don't mention any sync call in parent class constructor because GPU
 * acceleration is optional. Sync in decode method(s) instead.
 */
DecodeFrame::DecodeFrame(const char* URL, NvDecoderClInterface& cli_iface,
                         int gpu_id, std::shared_ptr<AVIOContext> p_io_ctx) {
  std::map<std::string, std::string> ffmpeg_options;
  cli_iface.GetOptions(ffmpeg_options);

  pImpl = new FfmpegDecodeFrame_Impl(URL, ffmpeg_options, gpu_id, p_io_ctx);
}

DecodeFrame::~DecodeFrame() { delete pImpl; }

bool DecodeFrame::IsAccelerated() const { return pImpl->IsAccelerated(); }

bool DecodeFrame::IsVFR() const { return pImpl->IsVFR(); }

CUstream DecodeFrame::GetStream() const { return pImpl->GetStream(); }
