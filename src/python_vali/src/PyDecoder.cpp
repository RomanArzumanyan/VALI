/*
 * Copyright 2019 NVIDIA Corporation
 * Copyright 2021 Kognia Sports Intelligence
 * Copyright 2021 Videonetics Technology Private Limited
 *
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

#include "VALI.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyDecoder::PyDecoder(const string& pathToFile,
                     const map<string, string>& ffmpeg_options, int gpuID) {
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);
  auto stream =
      gpu_id >= 0
          ? std::optional<CUstream>(CudaResMgr::Instance().GetStream(gpu_id))
          : std::nullopt;

  upDecoder.reset(DecodeFrame::Make(pathToFile.c_str(), cli_iface, stream));
}

PyDecoder::PyDecoder(py::object buffered_reader,
                     const map<string, string>& ffmpeg_options, int gpuID) {
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);
  auto stream =
      gpu_id >= 0
          ? std::optional<CUstream>(CudaResMgr::Instance().GetStream(gpu_id))
          : std::nullopt;

  upBuff.reset(new BufferedRandom(buffered_reader));
  upDecoder.reset(DecodeFrame::Make(pathToFile.c_str(), cli_iface, stream));
}

bool PyDecoder::DecodeImpl(TaskExecDetails& details, PacketData& pkt_data,
                           Token& dst, std::optional<SeekContext> seek_ctx) {
  upDecoder->ClearInputs();
  upDecoder->ClearOutputs();
  upDecoder->SetInput(&dst, 0U);

  std::shared_ptr<Buffer> seek_ctx_buf = nullptr;
  if (seek_ctx) {
    seek_ctx_buf.reset(Buffer::Make(sizeof(SeekContext),
                                    static_cast<void*>(&seek_ctx.value())));
    upDecoder->SetInput(seek_ctx_buf.get(), 1U);
  }

  details = upDecoder->Execute();
  pkt_data = upDecoder->GetLastPacketData();

  UpdateState();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

bool PyDecoder::DecodeSingleFrame(py::array& frame, TaskExecDetails& details,
                                  PacketData& pkt_data,
                                  std::optional<SeekContext> seek_ctx) {
  if (IsAccelerated()) {
    return false;
  }

  auto const frame_size = upDecoder->GetHostFrameSize();
  if (frame_size != frame.nbytes()) {
    frame.resize({frame_size}, false);
  }

  auto dst = std::shared_ptr<Buffer>(
      Buffer::Make(frame.nbytes(), frame.mutable_data()));
  return DecodeImpl(details, pkt_data, *dst.get(), seek_ctx);
}

bool PyDecoder::DecodeSingleSurface(Surface& surf, TaskExecDetails& details,
                                    PacketData& pkt_data,
                                    std::optional<SeekContext> seek_ctx) {
  if (!IsAccelerated()) {
    return false;
  }

  if (surf.Empty()) {
    std::cerr << "Empty Surface";
    return false;
  }

  if (surf.Width() != Width() || surf.Height() != Height()) {
    std::cerr << "Surface dimensions mismatch: " << surf.Width() << "x"
              << surf.Height() << " vs " << Width() << "x" << Height();
    return false;
  }

  if (surf.PixelFormat() != PixelFormat()) {
    std::cerr << "Surface format mismatch: " << surf.PixelFormat() << " vs "
              << PixelFormat();
    return false;
  }

  return DecodeImpl(details, pkt_data, surf, seek_ctx);
}

void* PyDecoder::GetSideData(AVFrameSideDataType data_type, size_t& raw_size) {
  if (TASK_EXEC_SUCCESS == upDecoder->GetSideData(data_type).m_status) {
    auto pSideData = (Buffer*)upDecoder->GetOutput(0U);
    if (pSideData) {
      raw_size = pSideData->GetRawMemSize();
      return pSideData->GetDataAs<void>();
    }
  }
  return nullptr;
}

void PyDecoder::UpdateState() {
  last_h = Height();
  last_w = Width();
}

std::vector<MotionVector> PyDecoder::GetMotionVectors() {
  size_t num_elems = 0U;
  auto ptr =
      (AVMotionVector*)GetSideData(AV_FRAME_DATA_MOTION_VECTORS, num_elems);
  num_elems /= sizeof(*ptr);

  if (ptr && num_elems) {
    try {
      auto mvc = std::vector<MotionVector>(num_elems);

      for (auto i = 0; i < num_elems; i++) {
        mvc[i].source = ptr[i].source;
        mvc[i].w = ptr[i].w;
        mvc[i].h = ptr[i].h;
        mvc[i].src_x = ptr[i].src_x;
        mvc[i].src_y = ptr[i].src_y;
        mvc[i].dst_x = ptr[i].dst_x;
        mvc[i].dst_y = ptr[i].dst_y;
        mvc[i].motion_x = ptr[i].motion_x;
        mvc[i].motion_y = ptr[i].motion_y;
        mvc[i].motion_scale = ptr[i].motion_scale;
      }

      return mvc;
    } catch (std::exception& e) {
      return std::vector<MotionVector>();
    }
  }

  return std::vector<MotionVector>();
}

uint32_t PyDecoder::Width() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.width;
};

uint32_t PyDecoder::Height() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.height;
};

uint32_t PyDecoder::Level() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.level;
};

uint32_t PyDecoder::Profile() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.profile;
};

uint32_t PyDecoder::Delay() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.delay;
};

uint32_t PyDecoder::GopSize() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.gop_size;
};

uint32_t PyDecoder::Bitrate() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.bit_rate;
};

uint32_t PyDecoder::NumFrames() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.num_frames;
};

uint32_t PyDecoder::NumStreams() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.num_streams;
};

uint32_t PyDecoder::StreamIndex() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_index;
};

uint32_t PyDecoder::HostFrameSize() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.host_frame_size;
};

double PyDecoder::Framerate() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.frame_rate;
};

ColorSpace PyDecoder::Color_Space() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.color_space;
};

ColorRange PyDecoder::Color_Range() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.color_range;
};

double PyDecoder::AvgFramerate() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.avg_frame_rate;
};

double PyDecoder::Timebase() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.time_base;
};

double PyDecoder::StartTime() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.start_time;
};

double PyDecoder::Duration() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.duration;
};

Pixel_Format PyDecoder::PixelFormat() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.format;
};

bool PyDecoder::IsAccelerated() const { return upDecoder->IsAccelerated(); }

bool PyDecoder::IsVFR() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.is_vfr;
}

std::map<std::string, std::string> PyDecoder::Metadata() {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.metadata;
}

void Init_PyDecoder(py::module& m) {
  py::class_<PyDecoder, shared_ptr<PyDecoder>>(m, "PyDecoder",
                                               "Video decoder class.")
      .def(py::init<const string&, const map<string, string>&, int>(),
           py::arg("input"), py::arg("opts"), py::arg("gpu_id") = 0,
           R"pbdoc(
        Constructor method.

        :param input: path to input file
        :param opts: AVDictionary options that will be passed to AVFormat context.
        :param gpu_id: GPU ID. Default value is 0. Pass negative value to use CPU decoder.
    )pbdoc")
      .def(py::init<py::object, const map<string, string>&, int>(),
           py::arg("buffered_reader"), py::arg("opts"), py::arg("gpu_id") = 0,
           R"pbdoc(
        Constructor method.

        :param buffered_reader: io.BufferedReader object
        :param opts: AVDictionary options that will be passed to AVFormat context.
        :param gpu_id: GPU ID. Default value is 0. Pass negative value to use CPU decoder.
    )pbdoc")
      .def(
          "DecodeSingleFrame",
          [](PyDecoder& self, py::array& frame,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;
            PacketData pkt_data;

            return std::make_tuple(
                self.DecodeSingleFrame(frame, details, pkt_data, seek_ctx),
                details.m_info);
          },
          py::arg("frame"), py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input file.
        Only call this method for decoder without HW acceleration.

        :param frame: decoded video frame
        :param pkt_data: decoded video frame packet data, may be None
        :param seek_ctx: seek context, may be None
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleFrame",
          [](PyDecoder& self, py::array& frame, PacketData& pkt_data,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;

            return std::make_tuple(
                self.DecodeSingleFrame(frame, details, pkt_data, seek_ctx),
                details.m_info);
          },
          py::arg("frame"), py::arg("pkt_data"),
          py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input file.
        Only call this method for decoder without HW acceleration.

        :param frame: decoded video frame
        :param pkt_data: decoded video frame packet data, may be None
        :param seek_ctx: seek context, may be None
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](PyDecoder& self, Surface& surf,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;
            PacketData pkt_data;

            return std::make_tuple(
                self.DecodeSingleSurface(surf, details, pkt_data, seek_ctx),
                details.m_info);
          },
          py::arg("surf"), py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video surface from input file.
        Only call this method for HW-accelerated decoder.

        :param surf: decoded video surface
        :param pkt_data: decoded video surface packet data, may be None
        :param seek_ctx: seek context, may be None
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](PyDecoder& self, Surface& surf, PacketData& pkt_data,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;

            return std::make_tuple(
                self.DecodeSingleSurface(surf, details, pkt_data, seek_ctx),
                details.m_info);
          },
          py::arg("surf"), py::arg("pkt_data"),
          py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video surface from input file.
        Only call this method for HW-accelerated decoder.

        :param surf: decoded video surface
        :param pkt_data: decoded video surface packet data, may be None
        :param seek_ctx: seek context, may be None
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def("TakeBuffer",
           [](PyDecoder& self, py::object buf) {
             BufferedRandom buffer(buf);
             std::vector<uint8_t> scrap(128);
             BufferedRandom::read((void*)&buffer, scrap.data(), scrap.size());
           })
      .def_property_readonly("Width", &PyDecoder::Width,
                             R"pbdoc(
        Return encoded video file width in pixels.
    )pbdoc")
      .def_property_readonly("Height", &PyDecoder::Height,
                             R"pbdoc(
        Return encoded video file height in pixels.
    )pbdoc")
      .def_property_readonly("Level", &PyDecoder::Level,
                             R"pbdoc(
        Return encoded video level coding parameter.
    )pbdoc")
      .def_property_readonly("Profile", &PyDecoder::Profile,
                             R"pbdoc(
        Return encoded video profile coding parameter.
    )pbdoc")
      .def_property_readonly("Delay", &PyDecoder::Delay,
                             R"pbdoc(
        Return encoded video delay.
    )pbdoc")
      .def_property_readonly("GopSize", &PyDecoder::GopSize,
                             R"pbdoc(
        Return encoded video GOP size.
    )pbdoc")
      .def_property_readonly("Bitrate", &PyDecoder::Bitrate,
                             R"pbdoc(
        Return encoded video bitrate in bits per second.
    )pbdoc")
      .def_property_readonly("NumStreams", &PyDecoder::NumStreams,
                             R"pbdoc(
        Return number of streams in video file. E. g. 2 streams: audio and video.
    )pbdoc")
      .def_property_readonly("StreamIndex", &PyDecoder::StreamIndex,
                             R"pbdoc(
        Return number of current video stream in file. E. g. video stream has
        index 0, and audio stream has index 1. This method will return 0 then.
    )pbdoc")
      .def_property_readonly("Framerate", &PyDecoder::Framerate,
                             R"pbdoc(
        Return encoded video file framerate.
    )pbdoc")
      .def_property_readonly("AvgFramerate", &PyDecoder::AvgFramerate,
                             R"pbdoc(
        Return encoded video file average framerate.
    )pbdoc")
      .def_property_readonly("Timebase", &PyDecoder::Timebase,
                             R"pbdoc(
        Return encoded video file time base.
    )pbdoc")
      .def_property_readonly("NumFrames", &PyDecoder::NumFrames,
                             R"pbdoc(
        Return number of video frames in encoded video file.
        Please note that some video containers doesn't store this infomation.
    )pbdoc")
      .def_property_readonly("ColorSpace", &PyDecoder::Color_Space,
                             R"pbdoc(
        Get color space information stored in video file.
        Please not that some video containers may not store this information.

        :return: color space information
    )pbdoc")
      .def_property_readonly("ColorRange", &PyDecoder::Color_Range,
                             R"pbdoc(
        Get color range information stored in video file.
        Please not that some video containers may not store this information.

        :return: color range information
    )pbdoc")
      .def_property_readonly("Format", &PyDecoder::PixelFormat,
                             R"pbdoc(
        Return encoded video file pixel format.
    )pbdoc")
      .def_property_readonly("HostFrameSize", &PyDecoder::HostFrameSize,
                             R"pbdoc(
        Return amount of bytes needed to store decoded frame.
    )pbdoc")
      .def_property_readonly("StartTime", &PyDecoder::StartTime,
                             R"pbdoc(
        Return video start time in seconds.
    )pbdoc")
      .def_property_readonly("Duration", &PyDecoder::Duration,
                             R"pbdoc(
        Return video duration time in seconds. May not be present.
    )pbdoc")
      .def_property_readonly("IsVFR", &PyDecoder::IsVFR,
                             R"pbdoc(
        Return true if video has variable framerate, false otherwise.
    )pbdoc")
      .def_property_readonly("IsAccelerated", &PyDecoder::IsAccelerated,
                             R"pbdoc(
        Return true if decoder has HW acceleration support, false otherwise.
    )pbdoc")
      .def_property_readonly("MotionVectors", &PyDecoder::GetMotionVectors,
                             py::call_guard<py::gil_scoped_release>(),
                             R"pbdoc(
        Return motion vectors of last decoded frame.
        If there are no movion vectors it will return empty list.

       :return: list of motion vectors
       :rtype: List[vali.MotionVector]
    )pbdoc")
      .def_property_readonly("Metadata", &PyDecoder::Metadata,
                             R"pbdoc(
        Return dictionary with video file metadata.
    )pbdoc");

  m.attr("NO_PTS") = py::int_(AV_NOPTS_VALUE);
}
