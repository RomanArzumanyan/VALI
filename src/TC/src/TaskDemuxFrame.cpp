/*
 * Copyright 2019 NVIDIA Corporation
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
#include "NvCodecUtils.h"
#include "Tasks.hpp"

extern "C" {
#include "libavutil/avstring.h"
#include "libavutil/avutil.h"
#include "libavutil/pixdesc.h"
}

#include <iostream>
#include <limits>
#include <memory>
#include <sstream>

using namespace std;

static string AvErrorToString(int av_error_code) {
  const auto buf_size = 1024U;
  char *err_string = (char *)calloc(buf_size, sizeof(*err_string));
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

int DataProvider::GetData(uint8_t *pBuf, int nBuf) {
  if (i_str.eof()) {
    return AVERROR_EOF;
  }

  if (!i_str.good()) {
    return AVERROR_UNKNOWN;
  }

  try {
    i_str.read((char *)pBuf, nBuf);
    return i_str.gcount();
  } catch (exception &e) {
    return AVERROR_UNKNOWN;
  }
}

DataProvider::DataProvider(std::istream &istr) : i_str(istr) {}

FFmpegDemuxer::FFmpegDemuxer(const char *szFilePath,
                             const map<string, string> &ffmpeg_options)
    : FFmpegDemuxer(CreateFormatContext(szFilePath, ffmpeg_options)) {}

FFmpegDemuxer::FFmpegDemuxer(DataProvider &pDataProvider,
                             const map<string, string> &ffmpeg_options)
    : FFmpegDemuxer(CreateFormatContext(pDataProvider, ffmpeg_options)) {
  avioc = fmtc->pb;
}

uint32_t FFmpegDemuxer::GetWidth() const { return width; }

uint32_t FFmpegDemuxer::GetHeight() const { return height; }

uint32_t FFmpegDemuxer::GetGopSize() const { return gop_size; }

uint32_t FFmpegDemuxer::GetNumFrames() const { return nb_frames; }

double FFmpegDemuxer::GetFramerate() const { return framerate; }

double FFmpegDemuxer::GetAvgFramerate() const { return avg_framerate; }

double FFmpegDemuxer::GetTimebase() const { return timebase; }

bool FFmpegDemuxer::IsVFR() const { return framerate != avg_framerate; }

uint32_t FFmpegDemuxer::GetVideoStreamIndex() const { return videoStream; }

AVPixelFormat FFmpegDemuxer::GetPixelFormat() const { return eChromaFormat; }

AVColorSpace FFmpegDemuxer::GetColorSpace() const { return color_space; }

AVColorRange FFmpegDemuxer::GetColorRange() const { return color_range; }

extern unsigned long GetNumDecodeSurfaces(cudaVideoCodec eCodec,
                                          unsigned int nWidth,
                                          unsigned int nHeight);

bool FFmpegDemuxer::Demux(uint8_t *&pVideo, size_t &rVideoBytes,
                          PacketData &pktData, TaskExecDetails &details,
                          uint8_t **ppSEI, size_t *pSEIBytes) {
  rVideoBytes = 0U;

  if (!fmtc) {
    throw std::runtime_error("No AVFormatContext given.");
  }

  if (pktSrc.data) {
    av_packet_unref(&pktSrc);
  }

  if (!annexbBytes.empty()) {
    annexbBytes.clear();
  }

  if (!seiBytes.empty()) {
    seiBytes.clear();
  }

  auto appendBytes = [](vector<uint8_t> &elementaryBytes, AVPacket &avPacket,
                        AVPacket &avPacketOut, AVBSFContext *pAvbsfContext,
                        int streamId, bool isFilteringNeeded) {
    if (avPacket.stream_index != streamId) {
      return;
    }

    if (isFilteringNeeded) {
      if (avPacketOut.data) {
        av_packet_unref(&avPacketOut);
      }

      av_bsf_send_packet(pAvbsfContext, &avPacket);
      av_bsf_receive_packet(pAvbsfContext, &avPacketOut);

      if (avPacketOut.data && avPacketOut.size) {
        elementaryBytes.insert(elementaryBytes.end(), avPacketOut.data,
                               avPacketOut.data + avPacketOut.size);
      }
    } else if (avPacket.data && avPacket.size) {
      elementaryBytes.insert(elementaryBytes.end(), avPacket.data,
                             avPacket.data + avPacket.size);
    }
  };

  int ret = 0;
  bool isDone = false, gotVideo = false;

  while (!isDone) {
    ret = av_read_frame(fmtc, &pktSrc);
    gotVideo = (pktSrc.stream_index == videoStream);
    isDone = (ret < 0) || gotVideo;

    if (pSEIBytes && ppSEI) {
      // Bitstream filter lazy init;
      // We don't do this in constructor as user may not be needing SEI
      // extraction at all;
      if (!bsfc_sei) {
        // SEI has NAL type 6 for H.264 and NAL type 39 & 40 for H.265;
        const string sei_filter = is_mp4H264   ? "filter_units=pass_types=6"
                                  : is_mp4HEVC ? "filter_units=pass_types=39-40"
                                               : "unknown";
        ret = av_bsf_list_parse_str(sei_filter.c_str(), &bsfc_sei);
        if (0 > ret) {
          details.info = TaskExecInfo::FAIL;
          throw runtime_error("Error initializing " + sei_filter +
                              " bitstream filter: " + AvErrorToString(ret));
        }

        ret = avcodec_parameters_copy(bsfc_sei->par_in,
                                      fmtc->streams[videoStream]->codecpar);
        if (0 != ret) {
          details.info = TaskExecInfo::FAIL;
          throw runtime_error("Error copying codec parameters: " +
                              AvErrorToString(ret));
        }

        ret = av_bsf_init(bsfc_sei);
        if (0 != ret) {
          details.info = TaskExecInfo::FAIL;
          throw runtime_error("Error initializing " + sei_filter +
                              " bitstream filter: " + AvErrorToString(ret));
        }
      }

      // Extract SEI NAL units from packet;
      auto pCopyPacket = av_packet_clone(&pktSrc);
      appendBytes(seiBytes, *pCopyPacket, pktSei, bsfc_sei, videoStream, true);
      av_packet_free(&pCopyPacket);
    }

    /* Unref non-desired packets as we don't support them yet;
     */
    if (pktSrc.stream_index != videoStream) {
      av_packet_unref(&pktSrc);
      continue;
    }
  }

  if (ret < 0) {
    if (AVERROR_EOF != ret) {
      details.info = TaskExecInfo::FAIL;
    } else {
      // No need to report EOF;
      details.info = TaskExecInfo::END_OF_STREAM;
    }
    return false;
  }

  const bool bsf_needed = is_mp4H264 || is_mp4HEVC;
  appendBytes(annexbBytes, pktSrc, pktDst, bsfc_annexb, videoStream,
              bsf_needed);

  pVideo = annexbBytes.data();
  rVideoBytes = annexbBytes.size();

  /* Save packet props to PacketData, decoder will use it later.
   * If no BSF filters were applied, copy input packet props.
   */
  if (!bsf_needed) {
    av_packet_copy_props(&pktDst, &pktSrc);
  }

  last_packet_data.key = pktDst.flags & AV_PKT_FLAG_KEY;
  last_packet_data.pts = pktDst.pts;
  last_packet_data.dts = pktDst.dts;
  last_packet_data.pos = pktDst.pos;
  last_packet_data.duration = pktDst.duration;

  pktData = last_packet_data;

  if (pSEIBytes && ppSEI && !seiBytes.empty()) {
    *ppSEI = seiBytes.data();
    *pSEIBytes = seiBytes.size();
  }

  return true;
}

void FFmpegDemuxer::Flush() {
  avio_flush(fmtc->pb);
  avformat_flush(fmtc);
}

int64_t FFmpegDemuxer::TsFromTime(double ts_sec) {
  /* Internal timestamp representation is integer, so multiply to AV_TIME_BASE
   * and switch to fixed point precision arithmetics; */
  auto const ts_tbu = llround(ts_sec * AV_TIME_BASE);

  // Rescale the timestamp to value represented in stream base units;
  AVRational factor;
  factor.num = 1;
  factor.den = AV_TIME_BASE;
  return av_rescale_q(ts_tbu, factor, fmtc->streams[videoStream]->time_base);
}

int64_t FFmpegDemuxer::TsFromFrameNumber(int64_t frame_num) {
  auto const ts_sec = (double)frame_num / GetFramerate();
  return TsFromTime(ts_sec);
}

bool FFmpegDemuxer::Seek(SeekContext &seekCtx, uint8_t *&pVideo,
                         size_t &rVideoBytes, PacketData &pktData,
                         TaskExecDetails &details, uint8_t **ppSEI,
                         size_t *pSEIBytes) {
  /* !!! IMPORTANT !!!
   * Across this function packet decode timestamp (DTS) values are used to
   * compare given timestamp against. This is done for reason. DTS values shall
   * monotonically increase during the course of decoding unlike PTS velues
   * which may be affected by frame reordering due to B frames presence.
   */

  if (!is_seekable) {
    details.info = TaskExecInfo::FAIL;
    throw std::runtime_error("Seek isn't supported for this input.");
  }

  if (IsVFR() && seekCtx.IsByNumber()) {
    details.info = TaskExecInfo::FAIL;
    throw std::runtime_error(
        "Can't seek by frame number in VFR sequences. Seek by timestamp "
        "instead.");
  }

  // Seek for single frame;
  auto seek_frame = [&](SeekContext const &seek_ctx, int flags) {
    bool seek_backward = false;
    int64_t timestamp = 0;
    int ret = 0;

    if (seek_ctx.IsByNumber()) {
      timestamp = TsFromFrameNumber(seek_ctx.seek_frame);
      seek_backward = last_packet_data.dts > timestamp;
      ret = av_seek_frame(fmtc, GetVideoStreamIndex(), timestamp,
                          seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
    } else if (seek_ctx.IsByTimestamp()) {
      timestamp = TsFromTime(seek_ctx.seek_tssec);
      seek_backward = last_packet_data.dts > timestamp;
      ret = av_seek_frame(fmtc, GetVideoStreamIndex(), timestamp,
                          seek_backward ? AVSEEK_FLAG_BACKWARD | flags : flags);
    } else {
      details.info = TaskExecInfo::FAIL;
      throw runtime_error("Invalid seek mode");
    }

    if (ret < 0) {
      details.info = TaskExecInfo::FAIL;
      throw runtime_error("Error seeking for frame: " + AvErrorToString(ret));
    }
  };

  // Check if frame satisfies seek conditions;
  auto is_seek_done = [&](PacketData &pkt_data, SeekContext const &seek_ctx) {
    int64_t target_ts = 0;

    if (seek_ctx.IsByNumber()) {
      target_ts = TsFromFrameNumber(seek_ctx.seek_frame);
    } else if (seek_ctx.IsByTimestamp()) {
      // Rely solely on FFMpeg API for seek by timestamp;
      return 1;
    } else {
      details.info = TaskExecInfo::FAIL;
      throw runtime_error("Invalid seek mode.");
    }

    if (pkt_data.dts == target_ts) {
      return 0;
    } else if (pkt_data.dts > target_ts) {
      return 1;
    } else {
      return -1;
    };
  };

  /* This will seek for exact frame number;
   * Note that decoder may not be able to decode such frame; */
  auto seek_for_exact_frame = [&](PacketData &pkt_data, SeekContext &seek_ctx) {
    // Repetititive seek until seek condition is satisfied;
    SeekContext tmp_ctx = seek_ctx;
    seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);

    int condition = 0;
    do {
      if (!Demux(pVideo, rVideoBytes, pkt_data, details, ppSEI, pSEIBytes)) {
        break;
      }
      condition = is_seek_done(pkt_data, seek_ctx);

      // We've gone too far and need to seek backwards;
      if (condition > 0) {
        if (tmp_ctx.IsByNumber()) {
          tmp_ctx.seek_frame--;
        } else if (tmp_ctx.IsByTimestamp()) {
          tmp_ctx.seek_tssec -= this->GetTimebase();
          tmp_ctx.seek_tssec = max(0.0, tmp_ctx.seek_tssec);
        } else {
          details.info = TaskExecInfo::FAIL;
          throw runtime_error("Invalid seek mode.");
        }
        seek_frame(tmp_ctx, AVSEEK_FLAG_ANY);
      }
      // Need to read more frames until we reach requested number;
      else if (condition < 0) {
        continue;
      }
    } while (0 != condition);

    seek_ctx.out_frame_pts = pkt_data.pts;
    seek_ctx.out_frame_duration = pkt_data.duration;
  };

  // Seek for closest key frame in the past;
  auto seek_for_prev_key_frame = [&](PacketData &pkt_data,
                                     SeekContext &seek_ctx) {
    seek_frame(seek_ctx, AVSEEK_FLAG_BACKWARD);

    Demux(pVideo, rVideoBytes, pkt_data, details, ppSEI, pSEIBytes);
    seek_ctx.out_frame_pts = pkt_data.pts;
    seek_ctx.out_frame_duration = pkt_data.duration;
  };

  switch (seekCtx.mode) {
  case EXACT_FRAME:
    seek_for_exact_frame(pktData, seekCtx);
    break;
  case PREV_KEY_FRAME:
    seek_for_prev_key_frame(pktData, seekCtx);
    break;
  default:
    details.info = TaskExecInfo::FAIL;
    throw runtime_error("Unsupported seek mode");
    break;
  }

  return true;
}

int FFmpegDemuxer::ReadPacket(void *opaque, uint8_t *pBuf, int nBuf) {
  return 0;
}

AVCodecID FFmpegDemuxer::GetVideoCodec() const { return eVideoCodec; }

FFmpegDemuxer::~FFmpegDemuxer() {
  if (pktSrc.data) {
    av_packet_unref(&pktSrc);
  }
  if (pktDst.data) {
    av_packet_unref(&pktDst);
  }

  if (bsfc_annexb) {
    av_bsf_free(&bsfc_annexb);
  }

  if (bsfc_annexb) {
    av_bsf_free(&bsfc_sei);
  }

  avformat_close_input(&fmtc);

  if (avioc) {
    av_freep(&avioc->buffer);
    av_freep(&avioc);
  }
}

AVFormatContext *
FFmpegDemuxer::CreateFormatContext(DataProvider &pDataProvider,
                                   const map<string, string> &ffmpeg_options) {
  AVFormatContext *ctx = avformat_alloc_context();
  if (!ctx) {
    std::cerr << "Can't allocate AVFormatContext at " << __FILE__ << " "
              << __LINE__;
    return nullptr;
  }

  uint8_t *avioc_buffer = nullptr;
  int avioc_buffer_size = 8 * 1024 * 1024;
  avioc_buffer = (uint8_t *)av_malloc(avioc_buffer_size);
  if (!avioc_buffer) {
    std::cerr << "Can't allocate avioc_buffer at " << __FILE__ << " "
              << __LINE__;
    return nullptr;
  }
  avioc = avio_alloc_context(avioc_buffer, avioc_buffer_size, 0, &pDataProvider,
                             &ReadPacket, nullptr, nullptr);

  if (!avioc) {
    std::cerr << "Can't allocate AVIOContext at " << __FILE__ << " "
              << __LINE__;
    return nullptr;
  }
  ctx->pb = avioc;

  // Set up format context options;
  AVDictionary *options = NULL;
  for (auto &pair : ffmpeg_options) {
    auto err =
        av_dict_set(&options, pair.first.c_str(), pair.second.c_str(), 0);
    if (err < 0) {
      std::cerr << "Can't set up dictionary option: " << pair.first << " "
                << pair.second << ": " << AvErrorToString(err) << "\n";
      return nullptr;
    }
  }

  auto err = avformat_open_input(&ctx, nullptr, nullptr, &options);
  if (0 != err) {
    std::cerr << "Can't open input. Error message: " << AvErrorToString(err);
    return nullptr;
  }

  return ctx;
}

AVFormatContext *
FFmpegDemuxer::CreateFormatContext(const char *szFilePath,
                                   const map<string, string> &ffmpeg_options) {
  avformat_network_init();

  // Set up format context options;
  AVDictionary *options = NULL;
  for (auto &pair : ffmpeg_options) {
    cout << pair.first << ": " << pair.second << endl;
    auto err =
        av_dict_set(&options, pair.first.c_str(), pair.second.c_str(), 0);
    if (err < 0) {
      std::cerr << "Can't set up dictionary option: " << pair.first << " "
                << pair.second << ": " << AvErrorToString(err) << "\n";
      return nullptr;
    }
  }

  AVFormatContext *ctx = nullptr;
  // av_register_all();
  auto err = avformat_open_input(&ctx, szFilePath, nullptr, &options);
  if (err < 0 || nullptr == ctx) {
    std::cerr << "Can't open " << szFilePath << ": " << AvErrorToString(err)
              << "\n";
    return nullptr;
  }

  return ctx;
}

FFmpegDemuxer::FFmpegDemuxer(AVFormatContext *fmtcx) : fmtc(fmtcx) {
  pktSrc = {};
  pktDst = {};

  memset(&last_packet_data, 0, sizeof(last_packet_data));

  if (!fmtc) {
    stringstream ss;
    ss << __FUNCTION__ << ": no AVFormatContext provided." << endl;
    throw invalid_argument(ss.str());
  }

  auto ret = avformat_find_stream_info(fmtc, nullptr);
  if (0 != ret) {
    stringstream ss;
    ss << __FUNCTION__ << ": can't find stream info;" << AvErrorToString(ret)
       << endl;
    throw runtime_error(ss.str());
  }

  videoStream =
      av_find_best_stream(fmtc, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  if (videoStream < 0) {
    stringstream ss;
    ss << __FUNCTION__ << ": can't find video stream in input file." << endl;
    throw runtime_error(ss.str());
  }

  eVideoCodec = fmtc->streams[videoStream]->codecpar->codec_id;
  width = fmtc->streams[videoStream]->codecpar->width;
  height = fmtc->streams[videoStream]->codecpar->height;
  framerate = (double)fmtc->streams[videoStream]->r_frame_rate.num /
              (double)fmtc->streams[videoStream]->r_frame_rate.den;
  avg_framerate = (double)fmtc->streams[videoStream]->avg_frame_rate.num /
                  (double)fmtc->streams[videoStream]->avg_frame_rate.den;
  timebase = (double)fmtc->streams[videoStream]->time_base.num /
             (double)fmtc->streams[videoStream]->time_base.den;
  eChromaFormat = (AVPixelFormat)fmtc->streams[videoStream]->codecpar->format;
  nb_frames = fmtc->streams[videoStream]->nb_frames;
  color_space = fmtc->streams[videoStream]->codecpar->color_space;
  color_range = fmtc->streams[videoStream]->codecpar->color_range;

  is_mp4H264 = (eVideoCodec == AV_CODEC_ID_H264);
  is_mp4HEVC = (eVideoCodec == AV_CODEC_ID_HEVC);

  av_init_packet(&pktSrc);
  pktSrc.data = nullptr;
  pktSrc.size = 0;
  av_init_packet(&pktDst);
  pktDst.data = nullptr;
  pktDst.size = 0;
  av_init_packet(&pktSei);
  pktSei.data = nullptr;
  pktSei.size = 0;

  // Initialize Annex.B BSF;
  std::string bfs_name = "unknown";
  switch (eVideoCodec) {
  case AV_CODEC_ID_H264:
    bfs_name = "h264_mp4toannexb";
    break;

  case AV_CODEC_ID_HEVC:
    bfs_name = "hevc_mp4toannexb";
    break;

  case AV_CODEC_ID_AV1:
  case AV_CODEC_ID_VP9:
  case AV_CODEC_ID_VC1:
  case AV_CODEC_ID_MJPEG:
  case AV_CODEC_ID_MPEG4:
  case AV_CODEC_ID_MPEG1VIDEO:
  case AV_CODEC_ID_MPEG2VIDEO:
    bfs_name = "";
    break;

  default:
    break;
  }

  if (!bfs_name.empty()) {
    const AVBitStreamFilter *toAnnexB = av_bsf_get_by_name(bfs_name.c_str());
    if (!toAnnexB) {
      throw runtime_error("can't get " + bfs_name + " filter by name");
    }
    ret = av_bsf_alloc(toAnnexB, &bsfc_annexb);
    if (0 != ret) {
      throw runtime_error("Error allocating " + bfs_name +
                          " filter: " + AvErrorToString(ret));
    }

    ret = avcodec_parameters_copy(bsfc_annexb->par_in,
                                  fmtc->streams[videoStream]->codecpar);
    if (0 != ret) {
      throw runtime_error("Error copying codec parameters: " +
                          AvErrorToString(ret));
    }

    ret = av_bsf_init(bsfc_annexb);
    if (0 != ret) {
      throw runtime_error("Error initializing " + bfs_name +
                          " bitstream filter: " + AvErrorToString(ret));
    }
  }

  // SEI extraction filter has lazy init as this feature is optional;
  bsfc_sei = nullptr;

  /* Some inputs doesn't allow seek functionality.
   * Check this ahead of time. */
  is_seekable = fmtc->iformat->read_seek || fmtc->iformat->read_seek2;
}

namespace VPF {
struct DemuxFrame_Impl {
  size_t videoBytes = 0U;

  std::shared_ptr<Buffer> pElementaryVideo;
  std::shared_ptr<Buffer> pMuxingParams;
  std::shared_ptr<Buffer> pSei;
  std::shared_ptr<Buffer> pPktData;

  std::unique_ptr<FFmpegDemuxer> demuxer;
  std::unique_ptr<DataProvider> d_prov;

  DemuxFrame_Impl() = delete;
  DemuxFrame_Impl(const DemuxFrame_Impl &other) = delete;
  DemuxFrame_Impl &operator=(const DemuxFrame_Impl &other) = delete;

  explicit DemuxFrame_Impl(const string &url,
                           const map<string, string> &ffmpeg_options) {
    demuxer.reset(new FFmpegDemuxer(url.c_str(), ffmpeg_options));
    pElementaryVideo.reset(Buffer::MakeOwnMem(0U));
    pMuxingParams.reset(Buffer::MakeOwnMem(sizeof(MuxingParams)));
    pSei.reset(Buffer::MakeOwnMem(0U));
    pPktData.reset(Buffer::MakeOwnMem(0U));
  }

  explicit DemuxFrame_Impl(istream &istr,
                           const map<string, string> &ffmpeg_options) {
    d_prov.reset(new DataProvider(istr));
    demuxer.reset(new FFmpegDemuxer(*d_prov.get(), ffmpeg_options));

    pElementaryVideo.reset(Buffer::MakeOwnMem(0U));
    pMuxingParams.reset(Buffer::MakeOwnMem(sizeof(MuxingParams)));
    pSei.reset(Buffer::MakeOwnMem(0U));
    pPktData.reset(Buffer::MakeOwnMem(0U));
  }

  ~DemuxFrame_Impl() = default;
};
} // namespace VPF

DemuxFrame *DemuxFrame::Make(istream &i_str, const char **ffmpeg_options,
                             uint32_t opts_size) {
  return new DemuxFrame(i_str, ffmpeg_options, opts_size);
}

DemuxFrame *DemuxFrame::Make(const char *url, const char **ffmpeg_options,
                             uint32_t opts_size) {
  return new DemuxFrame(url, ffmpeg_options, opts_size);
}

DemuxFrame::DemuxFrame(istream &i_str, const char **ffmpeg_options,
                       uint32_t opts_size)
    : Task("DemuxFrame", DemuxFrame::numInputs, DemuxFrame::numOutputs) {
  map<string, string> options;
  if (0 == opts_size % 2) {
    for (auto i = 0; i < opts_size;) {
      auto key = string(ffmpeg_options[i]);
      i++;
      auto value = string(ffmpeg_options[i]);
      i++;

      options.insert(pair<string, string>(key, value));
    }
  }
  pImpl = new DemuxFrame_Impl(i_str, options);
}

DemuxFrame::DemuxFrame(const char *url, const char **ffmpeg_options,
                       uint32_t opts_size)
    : Task("DemuxFrame", DemuxFrame::numInputs, DemuxFrame::numOutputs) {
  map<string, string> options;
  if (0 == opts_size % 2) {
    for (auto i = 0; i < opts_size;) {
      auto key = string(ffmpeg_options[i]);
      i++;
      auto value = string(ffmpeg_options[i]);
      i++;

      options.insert(pair<string, string>(key, value));
    }
  }
  pImpl = new DemuxFrame_Impl(url, options);
}

DemuxFrame::~DemuxFrame() { delete pImpl; }

void DemuxFrame::Flush() { pImpl->demuxer->Flush(); }

int64_t DemuxFrame::TsFromTime(double ts_sec) {
  return pImpl->demuxer->TsFromTime(ts_sec);
}

int64_t DemuxFrame::TsFromFrameNumber(int64_t frame_num) {
  return pImpl->demuxer->TsFromFrameNumber(frame_num);
}

TaskExecStatus DemuxFrame::Run() {
  NvtxMark tick(GetName());
  ClearOutputs();

  uint8_t *pVideo = nullptr;
  MuxingParams params = {0};
  PacketData pkt_data = {0};

  auto &videoBytes = pImpl->videoBytes;
  auto &demuxer = pImpl->demuxer;

  uint8_t *pSEI = nullptr;
  size_t seiBytes = 0U;
  bool needSEI = (nullptr != GetInput(0U));

  auto pSeekCtxBuf = (Buffer *)GetInput(1U);
  TaskExecDetails details;
  if (pSeekCtxBuf) {
    SeekContext seek_ctx = *pSeekCtxBuf->GetDataAs<SeekContext>();
    auto ret = demuxer->Seek(seek_ctx, pVideo, videoBytes, pkt_data, details,
                             needSEI ? &pSEI : nullptr, &seiBytes);

    SetExecDetails(details);

    if (!ret) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }
  } else {
    auto ret = demuxer->Demux(pVideo, videoBytes, pkt_data, details,
                              needSEI ? &pSEI : nullptr, &seiBytes);

    SetExecDetails(details);

    if (!ret) {
      return TaskExecStatus::TASK_EXEC_FAIL;
    }
  }

  if (videoBytes) {
    pImpl->pElementaryVideo->Update(videoBytes, pVideo);
    SetOutput(pImpl->pElementaryVideo.get(), 0U);

    GetParams(params);
    pImpl->pMuxingParams->Update(sizeof(MuxingParams), &params);
    SetOutput(pImpl->pMuxingParams.get(), 1U);
  }

  if (pSEI) {
    pImpl->pSei->Update(seiBytes, pSEI);
    SetOutput(pImpl->pSei.get(), 2U);
  }

  pImpl->pPktData->Update(sizeof(pkt_data), &pkt_data);
  SetOutput((Token *)pImpl->pPktData.get(), 3U);

  return TaskExecStatus::TASK_EXEC_SUCCESS;
}

void DemuxFrame::GetParams(MuxingParams &params) const {
  params.videoContext.width = pImpl->demuxer->GetWidth();
  params.videoContext.height = pImpl->demuxer->GetHeight();
  params.videoContext.num_frames = pImpl->demuxer->GetNumFrames();
  params.videoContext.frameRate = pImpl->demuxer->GetFramerate();
  params.videoContext.avgFrameRate = pImpl->demuxer->GetAvgFramerate();
  params.videoContext.is_vfr = pImpl->demuxer->IsVFR();
  params.videoContext.timeBase = pImpl->demuxer->GetTimebase();
  params.videoContext.streamIndex = pImpl->demuxer->GetVideoStreamIndex();
  params.videoContext.codec = FFmpeg2NvCodecId(pImpl->demuxer->GetVideoCodec());
  params.videoContext.gop_size = pImpl->demuxer->GetGopSize();

  switch (pImpl->demuxer->GetPixelFormat()) {
  case AV_PIX_FMT_YUVJ420P:
  case AV_PIX_FMT_YUV420P:
  case AV_PIX_FMT_NV12:
    params.videoContext.format = NV12;
    break;
  case AV_PIX_FMT_YUV444P16LE:
  case AV_PIX_FMT_YUV444P10LE:
    params.videoContext.format = YUV444_10bit;
    break;
  case AV_PIX_FMT_P016LE:
    params.videoContext.format = P12;
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
  default:
    stringstream ss;
    ss << "Unsupported FFmpeg pixel format: "
       << av_get_pix_fmt_name(pImpl->demuxer->GetPixelFormat()) << endl;
    throw invalid_argument(ss.str());
    params.videoContext.format = UNDEFINED;
    break;
  }

  switch (pImpl->demuxer->GetColorSpace()) {
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

  switch (pImpl->demuxer->GetColorRange()) {
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