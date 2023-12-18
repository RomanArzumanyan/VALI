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

#include "FFmpegDemuxer.h"
#include "Tasks.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/motion_vector.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
}

using namespace VPF;
using namespace std;

static string AvErrorToString(int av_error_code)
{
  const auto buf_size = 1024U;
  char* err_string = (char*)calloc(buf_size, sizeof(*err_string));
  if (!err_string) {
    return string();
  }

  if (0 != av_strerror(av_error_code, err_string, buf_size - 1)) {
    free(err_string);
    stringstream ss;
    ss << "Unknown error with code " << av_error_code;
    return ss.str();
  }

  string str(err_string);
  free(err_string);
  return str;
}

namespace VPF
{

enum DECODE_STATUS { DEC_SUCCESS, DEC_ERROR, DEC_MORE, DEC_EOS };

struct FfmpegDecodeFrame_Impl {
  std::shared_ptr<AVFormatContext> m_fmt_ctx;
  std::shared_ptr<SwrContext> m_swr_ctx;
  std::shared_ptr<AVCodecContext> m_avc_ctx;
  std::shared_ptr<AVFrame> m_frame;
  std::shared_ptr<AVPacket> m_pkt;
  std::shared_ptr<Buffer> m_dec_frame;
  PacketData m_packetData;
  map<AVFrameSideDataType, Buffer*> side_data;

  int video_stream_idx = -1;
  bool end_decode = false;
  bool eof = false;

  FfmpegDecodeFrame_Impl(const char* URL, AVDictionary* pOptions)
  {
    AVFormatContext* fmt_ctx_ = nullptr;
    auto res = avformat_open_input(&fmt_ctx_, URL, NULL, &pOptions);
    if (res < 0) {
      {
        // Don't remove, it's here to make linker clear of libswresample
        // dependency. Otherwise it's required by libavcodec but never put into
        // rpath by CMake.
        SwrContext* swr_ctx_ = swr_alloc();
        m_swr_ctx =
            std::shared_ptr<SwrContext>(swr_ctx_, [](auto p) { swr_free(&p); });
      }

      stringstream ss;
      ss << "Could not open source file" << URL << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }
    m_fmt_ctx = std::shared_ptr<AVFormatContext>(
        fmt_ctx_, [](auto p) { avformat_close_input(&p); });

    res = avformat_find_stream_info(m_fmt_ctx.get(), NULL);
    if (res < 0) {
      stringstream ss;
      ss << "Could not find stream information" << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }

    video_stream_idx = av_find_best_stream(m_fmt_ctx.get(), AVMEDIA_TYPE_VIDEO,
                                           -1, -1, NULL, 0);
    if (video_stream_idx < 0) {
      stringstream ss;
      ss << "Could not find " << av_get_media_type_string(AVMEDIA_TYPE_VIDEO)
         << " stream in file " << URL << endl;
      ss << "Error description: " << AvErrorToString(video_stream_idx) << endl;
      throw runtime_error(ss.str());
    }

    auto video_stream = m_fmt_ctx->streams[video_stream_idx];
    if (!video_stream) {
      cerr << "Could not find video stream in the input, aborting" << endl;
    }

    auto p_codec = avcodec_find_decoder(video_stream->codecpar->codec_id);
    if (!p_codec) {
      stringstream ss;
      ss << "Failed to find codec by id" << endl;
      throw runtime_error(ss.str());
    }

    auto avctx_ = avcodec_alloc_context3(p_codec);
    if (!avctx_) {
      stringstream ss;
      ss << "Failed to allocate codec context" << endl;
      throw runtime_error(ss.str());
    }
    m_avc_ctx = std::shared_ptr<AVCodecContext>(
        avctx_, [](auto p) { avcodec_free_context(&p); });

    res =
        avcodec_parameters_to_context(m_avc_ctx.get(), video_stream->codecpar);
    if (res < 0) {
      stringstream ss;
      ss << "Failed to pass codec parameters to codec context "
         << av_get_media_type_string(AVMEDIA_TYPE_VIDEO) << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }

    res = avcodec_open2(m_avc_ctx.get(), p_codec, &pOptions);
    if (res < 0) {
      stringstream ss;
      ss << "Failed to open codec "
         << av_get_media_type_string(AVMEDIA_TYPE_VIDEO) << endl;
      ss << "Error description: " << AvErrorToString(res) << endl;
      throw runtime_error(ss.str());
    }

    auto frame_ = av_frame_alloc();
    if (!frame_) {
      stringstream ss;
      ss << "Could not allocate frame" << endl;
      throw runtime_error(ss.str());
    }

    m_frame =
        std::shared_ptr<AVFrame>(frame_, [](auto p) { av_frame_free(&p); });

    auto avpkt_ = av_packet_alloc();
    if (!avpkt_) {
      stringstream ss;
      ss << "Could not allocate packet" << endl;
      throw runtime_error(ss.str());
    }
    m_pkt =
        std::shared_ptr<AVPacket>(avpkt_, [](auto p) { av_packet_free(&p); });
  }

  bool SaveYUV420()
  {
    size_t size = m_frame->width * m_frame->height * 3 / 2;

    if (!m_dec_frame || (m_dec_frame && size != m_dec_frame->GetRawMemSize())) {
      m_dec_frame.reset(Buffer::MakeOwnMem(size));
    }

    auto plane = 0U;
    auto* dst = m_dec_frame->GetDataAs<uint8_t>();

    for (plane = 0; plane < 3; plane++) {
      auto* src = m_frame->data[plane];
      auto width = (0 == plane) ? m_frame->width : m_frame->width / 2;
      auto height = (0 == plane) ? m_frame->height : m_frame->height / 2;

      for (int i = 0; i < height; i++) {
        memcpy(dst, src, width);
        dst += width;
        src += m_frame->linesize[plane];
      }
    }

    return true;
  }

  bool SaveYUV422()
  {
    size_t size = m_frame->width * m_frame->height * 2;

    if (!m_dec_frame || (m_dec_frame && size != m_dec_frame->GetRawMemSize())) {
      m_dec_frame.reset(Buffer::MakeOwnMem(size));
    }

    auto plane = 0U;
    auto* dst = m_dec_frame->GetDataAs<uint8_t>();

    for (plane = 0; plane < 3; plane++) {
      auto* src = m_frame->data[plane];
      auto width = (0 == plane) ? m_frame->width : m_frame->width / 2;
      auto height = m_frame->height;

      for (int i = 0; i < height; i++) {
        memcpy(dst, src, width);
        dst += width;
        src += m_frame->linesize[plane];
      }
    }

    return true;
  }

  bool SaveGRAY12LE()
  {
    size_t size = m_frame->width * m_frame->height * 2;

    if (!m_dec_frame || (m_dec_frame && size != m_dec_frame->GetRawMemSize())) {
      m_dec_frame.reset(Buffer::MakeOwnMem(size));
    }

    auto plane = 0U;
    auto* dst = m_dec_frame->GetDataAs<uint8_t>();

    auto* src = m_frame->data[plane];
    auto width = m_frame->width;
    auto height = m_frame->height;

    for (int i = 0; i < height; i++) {
      memcpy(dst, src, 2 * width);
      dst += 2 * width;
      src += m_frame->linesize[plane];
    }

    return true;
  }

  bool SaveYUV444()
  {
    size_t size = m_frame->width * m_frame->height * 3;

    if (!m_dec_frame || (m_dec_frame && size != m_dec_frame->GetRawMemSize())) {
      m_dec_frame.reset(Buffer::MakeOwnMem(size));
    }

    auto plane = 0U;
    auto* dst = m_dec_frame->GetDataAs<uint8_t>();

    for (plane = 0; plane < 3; plane++) {
      auto* src = m_frame->data[plane];
      auto width = m_frame->width;
      auto height = m_frame->height;

      for (int i = 0; i < height; i++) {
        memcpy(dst, src, width);
        dst += width;
        src += m_frame->linesize[plane];
      }
    }

    return true;
  }

  void SavePacketData()
  {
    m_packetData.dts = m_frame->pkt_dts;
    m_packetData.pts = m_frame->pts;
    m_packetData.duration = m_frame->pkt_duration;
    m_packetData.key = m_frame->key_frame;
    m_packetData.pos = m_frame->pkt_pos;
  }

  bool DecodeSingleFrame(FfmpegDecodeFrame *parent)
  {
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

      auto status = DecodeSinglePacket(eof ? nullptr : m_pkt.get());

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

  bool SaveVideoFrame()
  {
    switch (m_frame->format) {
    case AV_PIX_FMT_YUV420P:
      return SaveYUV420();
    case AV_PIX_FMT_YUV422P:
      return SaveYUV422();
    case AV_PIX_FMT_YUV444P:
      return SaveYUV444();
    case AV_PIX_FMT_GRAY12LE:
      return SaveGRAY12LE();
    default:
      cerr << __FUNCTION__ << ": unsupported pixel format: " << m_frame->format
           << endl;
      return false;
    }
  }

  void SaveMotionVectors()
  {
    AVFrameSideData* sd =
        av_frame_get_side_data(m_frame.get(), AV_FRAME_DATA_MOTION_VECTORS);

    if (sd) {
      auto it = side_data.find(AV_FRAME_DATA_MOTION_VECTORS);
      if (it == side_data.end()) {
        // Add entry if not found (usually upon first call);
        side_data[AV_FRAME_DATA_MOTION_VECTORS] = Buffer::MakeOwnMem(sd->size);
        it = side_data.find(AV_FRAME_DATA_MOTION_VECTORS);
        memcpy(it->second->GetRawMemPtr(), sd->data, sd->size);
      } else if (it->second->GetRawMemSize() != sd->size) {
        // Update entry size if changed (e. g. on video resolution change);
        it->second->Update(sd->size, sd->data);
      }
    }
  }

  bool SaveSideData()
  {
    SaveMotionVectors();
    return true;
  }

  DECODE_STATUS DecodeSinglePacket(const AVPacket* pkt_Src)
  {
    auto res = avcodec_send_packet(m_avc_ctx.get(), pkt_Src);
    if (AVERROR_EOF == res) {
      // Flush decoder;
      res = 0;
    } else if (res < 0) {
      cerr << "Error while sending a packet to the decoder" << endl;
      cerr << "Error description: " << AvErrorToString(res) << endl;
      return DEC_ERROR;
    }

    while (res >= 0) {
      res = avcodec_receive_frame(m_avc_ctx.get(), m_frame.get());
      if (res == AVERROR_EOF) {
        return DEC_EOS;
      } else if (res == AVERROR(EAGAIN)) {
        return DEC_MORE;
      } else if (res < 0) {
        cerr << "Error while receiving a frame from the decoder" << endl;
        cerr << "Error description: " << AvErrorToString(res) << endl;
        return DEC_ERROR;
      }

      SaveVideoFrame();
      SaveSideData();
      SavePacketData();
      return DEC_SUCCESS;
    }

    return DEC_SUCCESS;
  }

  ~FfmpegDecodeFrame_Impl()
  {
    for (auto& output : side_data) {
      if (output.second) {
        delete output.second;
        output.second = nullptr;
      }
    }
  }
};
} // namespace VPF

const PacketData& FfmpegDecodeFrame::GetLastPacketData() const
{
  return pImpl->m_packetData;
}

TaskExecStatus FfmpegDecodeFrame::Run()
{
  ClearOutputs();

  if (pImpl->DecodeSingleFrame(this)) {
    SetOutput((Token*)pImpl->m_dec_frame.get(), 0U);
    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }

  return TaskExecStatus::TASK_EXEC_FAIL;
}

void FfmpegDecodeFrame::GetParams(MuxingParams& params)
{
  memset((void*)&params, 0, sizeof(params));
  auto fmtc = pImpl->m_fmt_ctx.get();
  auto videoStream =
      av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (videoStream < 0) {
    stringstream ss;
    ss << __FUNCTION__ << ": can't find video stream in input file." << endl;
    throw runtime_error(ss.str());
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

  switch (pImpl->m_avc_ctx->pix_fmt) {
  case AV_PIX_FMT_YUVJ420P:
  case AV_PIX_FMT_YUV420P:
  case AV_PIX_FMT_NV12:
    params.videoContext.format = NV12;
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
    stringstream ss;
    ss << "Unsupported FFmpeg pixel format: "
       << av_get_pix_fmt_name(pImpl->m_avc_ctx->pix_fmt) << endl;
    throw invalid_argument(ss.str());
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

TaskExecStatus FfmpegDecodeFrame::GetSideData(AVFrameSideDataType data_type)
{
  SetOutput(nullptr, 1U);
  auto it = pImpl->side_data.find(data_type);
  if (it != pImpl->side_data.end()) {
    SetOutput((Token*)it->second, 1U);
    return TaskExecStatus::TASK_EXEC_SUCCESS;
  }

  return TaskExecStatus::TASK_EXEC_FAIL;
}

FfmpegDecodeFrame* FfmpegDecodeFrame::Make(const char* URL,
                                           NvDecoderClInterface& cli_iface)
{
  return new FfmpegDecodeFrame(URL, cli_iface);
}

FfmpegDecodeFrame::FfmpegDecodeFrame(const char* URL,
                                     NvDecoderClInterface& cli_iface)
    : Task("FfmpegDecodeFrame", FfmpegDecodeFrame::num_inputs,
           FfmpegDecodeFrame::num_outputs)
{
  pImpl = new FfmpegDecodeFrame_Impl(URL, cli_iface.GetOptions());
}

FfmpegDecodeFrame::~FfmpegDecodeFrame() { delete pImpl; }
