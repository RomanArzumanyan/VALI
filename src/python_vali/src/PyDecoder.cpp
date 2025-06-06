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

  upDecoder.reset(DecodeFrame::Make(pathToFile.c_str(), cli_iface, gpu_id));
  if (gpu_id >= 0) {
    /* Libavcodec will use primary CUDA context for given GPU.
     * In case it prefers default CUDA tream (0x0) we shall not query context by
     * stream.
     */
    auto stream = upDecoder->GetStream();
    if (!stream) {
      m_event.reset(new CudaStreamEvent(upDecoder->GetStream(), gpu_id));
    } else {
      m_event.reset(new CudaStreamEvent(upDecoder->GetStream()));
    }
  }
}

PyDecoder::PyDecoder(py::object buffered_reader,
                     const map<string, string>& ffmpeg_options, int gpuID) {
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);

  upBuff.reset(new BufferedReader(buffered_reader));
  upDecoder.reset(
      DecodeFrame::Make("", cli_iface, gpu_id, upBuff->GetAVIOContext()));
  if (gpu_id >= 0) {
    /* Libavcodec will use primary CUDA context for given GPU.
     * In case it prefers default CUDA tream (0x0) we shall not query context by
     * stream.
     */
    auto stream = upDecoder->GetStream();
    if (!stream) {
      m_event.reset(new CudaStreamEvent(upDecoder->GetStream(), gpu_id));
    } else {
      m_event.reset(new CudaStreamEvent(upDecoder->GetStream()));
    }
  }
}

bool PyDecoder::DecodeImpl(TaskExecDetails& details, PacketData& pkt_data,
                           Token& dst, std::optional<SeekContext> seek_ctx) {
  details = upDecoder->Run(dst, pkt_data, seek_ctx);
  UpdateState();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

bool PyDecoder::DecodeSingleFrame(py::array& frame, TaskExecDetails& details,
                                  PacketData& pkt_data,
                                  std::optional<SeekContext> seek_ctx) {
  if (IsAccelerated()) {
    details.m_info = TaskExecInfo::FAIL;
    return false;
  }

  auto const frame_size = upDecoder->GetHostFrameSize();
  if (frame_size != frame.nbytes()) {
    frame.resize({frame_size}, false);
  }

  auto dst = std::shared_ptr<Buffer>(
      Buffer::Make(frame.nbytes(), frame.mutable_data()));

  py::gil_scoped_release gil_release{};
  return DecodeImpl(details, pkt_data, *dst.get(), seek_ctx);
}

bool PyDecoder::DecodeSingleSurface(Surface& surf, TaskExecDetails& details,
                                    PacketData& pkt_data,
                                    std::optional<SeekContext> seek_ctx) {
  if (!IsAccelerated()) {
    details.m_info = TaskExecInfo::FAIL;
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

void PyDecoder::UpdateState() {
  last_h = Height();
  last_w = Width();
}

double PyDecoder::GetDisplayRotation() const {
  Buffer buf(0U, false);
  auto ret = upDecoder->GetSideData(AV_FRAME_DATA_DISPLAYMATRIX, buf);
  if (ret.m_info != TaskExecInfo::SUCCESS)
    return 361.f;

  return *(buf.GetDataAs<double>());
}

std::vector<MotionVector> PyDecoder::GetMotionVectors() {
  Buffer buf(0U, false);
  auto ret = upDecoder->GetSideData(AV_FRAME_DATA_MOTION_VECTORS, buf);
  if (ret.m_info != TaskExecInfo::SUCCESS)
    return std::vector<MotionVector>();

  size_t num_elems = buf.GetRawMemSize() / sizeof(AVMotionVector);
  auto ptr = buf.GetDataAs<AVMotionVector>();

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
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.width;
};

uint32_t PyDecoder::Height() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.height;
};

uint32_t PyDecoder::Level() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.level;
};

uint32_t PyDecoder::Profile() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.profile;
};

uint32_t PyDecoder::Delay() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.delay;
};

uint32_t PyDecoder::GopSize() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.gop_size;
};

uint32_t PyDecoder::Bitrate() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.bit_rate;
};

uint32_t PyDecoder::NumFrames() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.num_frames;
};

uint32_t PyDecoder::NumStreams() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.num_streams;
};

uint32_t PyDecoder::StreamIndex() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_index;
};

uint32_t PyDecoder::HostFrameSize() const {
  return upDecoder->GetHostFrameSize();
};

double PyDecoder::Framerate() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.fps;
};

ColorSpace PyDecoder::Color_Space() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.color_space;
};

ColorRange PyDecoder::Color_Range() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.color_range;
};

double PyDecoder::AvgFramerate() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.avg_fps;
};

double PyDecoder::Timebase() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.time_base;
};

double PyDecoder::StartTime() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.start_time_sec;
};

double PyDecoder::Duration() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.duration_sec;
};

Pixel_Format PyDecoder::PixelFormat() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.format;
};

bool PyDecoder::IsAccelerated() const { return upDecoder->IsAccelerated(); }

bool PyDecoder::IsVFR() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.fps !=
         params.videoContext.stream_params.avg_fps;
}

CUstream PyDecoder::GetStream() const { return upDecoder->GetStream(); }

metadata_dict PyDecoder::Metadata() {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.metadata;
}

void PyDecoder::SetMode(DecodeMode new_mode) { upDecoder->SetMode(new_mode); }

DecodeMode PyDecoder::GetMode() const { return upDecoder->GetMode(); }

void Init_PyDecoder(py::module& m) {
  py::class_<PyDecoder, shared_ptr<PyDecoder>>(m, "PyDecoder",
                                               "Video decoder class.")
      .def(py::init<const string&, const map<string, string>&, int>(),
           py::arg("input"), py::arg("opts"), py::arg("gpu_id") = 0,
           R"pbdoc(
         Create a new video decoder instance from a file.

         Initializes a video decoder that can decode frames from a video file.
         The decoder can operate in either CPU or GPU mode depending on the gpu_id parameter.

         :param input: Path to the input video file
         :type input: str
         :param opts: Dictionary of options to pass to libavcodec API. Can include:
             - preferred_width: Select a stream with desired width from multiple video streams
             - Other FFmpeg options as key-value pairs
         :type opts: dict[str, str]
         :param gpu_id: GPU device ID to use for hardware acceleration. Default is 0.
             Use negative value for CPU-only decoding.
         :type gpu_id: int
         :raises RuntimeError: If decoder initialization fails
     )pbdoc")
      .def(py::init<py::object, const map<string, string>&, int>(),
           py::arg("buffered_reader"), py::arg("opts"), py::arg("gpu_id") = 0,
           R"pbdoc(
         Create a new video decoder instance from a buffered reader.

         Initializes a video decoder that can decode frames from a buffered reader object.
         The decoder can operate in either CPU or GPU mode depending on the gpu_id parameter.

         :param buffered_reader: Python object with a 'read' method (e.g., io.BufferedReader)
         :type buffered_reader: object
         :param opts: Dictionary of options to pass to libavcodec API. Can include:
             - preferred_width: Select a stream with desired width from multiple video streams
             - Other FFmpeg options as key-value pairs
         :type opts: dict[str, str]
         :param gpu_id: GPU device ID to use for hardware acceleration. Default is 0.
             Use negative value for CPU-only decoding.
         :type gpu_id: int
         :raises RuntimeError: If decoder initialization fails
     )pbdoc")
      .def_property_readonly("Mode", &PyDecoder::GetMode,
                             py::call_guard<py::gil_scoped_release>(),
                             R"pbdoc(
         Get the current decoder operation mode.

         :return: Current decode mode (e.g., KEY_FRAMES, ALL_FRAMES)
         :rtype: DecodeMode
     )pbdoc")
      .def("SetMode", &PyDecoder::SetMode,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
         Set the decoder operation mode.

         Changes how the decoder processes frames and handles seeking operations.
         When in KEY_FRAMES mode, seeking will return the closest previous key frame.
         When switching modes, the internal frame queue is preserved to avoid discarding
         decoded frames that may be needed for future operations.

         :param new_mode: The new decode mode to set
         :type new_mode: DecodeMode
         :note: Mode changes affect seek behavior and frame processing strategy
     )pbdoc")
      .def(
          "DecodeSingleFrame",
          [](PyDecoder& self, py::array& frame,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;
            PacketData pkt_data;

            auto res =
                self.DecodeSingleFrame(frame, details, pkt_data, seek_ctx);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("frame"), py::arg("seek_ctx") = std::nullopt,
          R"pbdoc(
         Decode a single video frame from the input source.

         This method is for CPU-only decoding (non-accelerated decoder).
         The frame will be decoded into the provided numpy array.

         :param frame: Numpy array to store the decoded frame
         :type frame: numpy.ndarray
         :param seek_ctx: Optional seek context for frame positioning
         :type seek_ctx: Optional[SeekContext]
         :return: Tuple containing:
             - success (bool): True if decoding was successful
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If called with hardware acceleration enabled
     )pbdoc")
      .def(
          "DecodeSingleFrame",
          [](PyDecoder& self, py::array& frame, PacketData& pkt_data,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;

            auto res =
                self.DecodeSingleFrame(frame, details, pkt_data, seek_ctx);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("frame"), py::arg("pkt_data"),
          py::arg("seek_ctx") = std::nullopt,
          R"pbdoc(
         Decode a single video frame with packet data from the input source.

         This method is for CPU-only decoding (non-accelerated decoder).
         The frame will be decoded into the provided numpy array, and packet
         metadata will be stored in pkt_data.

         :param frame: Numpy array to store the decoded frame
         :type frame: numpy.ndarray
         :param pkt_data: Object to store packet metadata
         :type pkt_data: PacketData
         :param seek_ctx: Optional seek context for frame positioning
         :type seek_ctx: Optional[SeekContext]
         :return: Tuple containing:
             - success (bool): True if decoding was successful
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If called with hardware acceleration enabled
     )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](PyDecoder& self, Surface& surf,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;
            PacketData pkt_data;

            auto res =
                self.DecodeSingleSurface(surf, details, pkt_data, seek_ctx);
            if (res) {
              self.m_event->Record();
              self.m_event->Wait();
            }
            return std::make_tuple(res, details.m_info);
          },
          py::arg("surf"), py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Decode a single video frame into a CUDA surface.

         This method is for hardware-accelerated decoding.
         The frame will be decoded directly into the provided CUDA surface.
         The operation is synchronous and will wait for completion.

         :param surf: CUDA surface to store the decoded frame
         :type surf: Surface
         :param seek_ctx: Optional seek context for frame positioning
         :type seek_ctx: Optional[SeekContext]
         :return: Tuple containing:
             - success (bool): True if decoding was successful
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If called without hardware acceleration
     )pbdoc")
      .def(
          "DecodeSingleSurfaceAsync",
          [](PyDecoder& self, Surface& surf,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;
            PacketData pkt_data;

            auto res =
                self.DecodeSingleSurface(surf, details, pkt_data, seek_ctx);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("surf"), py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Decode a single video frame into a CUDA surface asynchronously.

         This method is for hardware-accelerated decoding.
         The frame will be decoded directly into the provided CUDA surface.
         The operation is asynchronous and returns immediately.

         :param surf: CUDA surface to store the decoded frame
         :type surf: Surface
         :param seek_ctx: Optional seek context for frame positioning
         :type seek_ctx: Optional[SeekContext]
         :return: Tuple containing:
             - success (bool): True if decoding was successful
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If called without hardware acceleration
     )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](PyDecoder& self, Surface& surf, PacketData& pkt_data,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;

            auto res =
                self.DecodeSingleSurface(surf, details, pkt_data, seek_ctx);
            if (res) {
              self.m_event->Record();
              self.m_event->Wait();
            }
            return std::make_tuple(res, details.m_info);
          },
          py::arg("surf"), py::arg("pkt_data"),
          py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Decode a single video frame into a CUDA surface with packet data.

         This method is for hardware-accelerated decoding.
         The frame will be decoded directly into the provided CUDA surface,
         and packet metadata will be stored in pkt_data.
         The operation is synchronous and will wait for completion.

         :param surf: CUDA surface to store the decoded frame
         :type surf: Surface
         :param pkt_data: Object to store packet metadata
         :type pkt_data: PacketData
         :param seek_ctx: Optional seek context for frame positioning
         :type seek_ctx: Optional[SeekContext]
         :return: Tuple containing:
             - success (bool): True if decoding was successful
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If called without hardware acceleration
     )pbdoc")
      .def(
          "DecodeSingleSurfaceAsync",
          [](PyDecoder& self, Surface& surf, PacketData& pkt_data,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;

            auto res =
                self.DecodeSingleSurface(surf, details, pkt_data, seek_ctx);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("surf"), py::arg("pkt_data"),
          py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Decode a single video frame into a CUDA surface with packet data asynchronously.

         This method is for hardware-accelerated decoding.
         The frame will be decoded directly into the provided CUDA surface,
         and packet metadata will be stored in pkt_data.
         The operation is asynchronous and returns immediately.

         :param surf: CUDA surface to store the decoded frame
         :type surf: Surface
         :param pkt_data: Object to store packet metadata
         :type pkt_data: PacketData
         :param seek_ctx: Optional seek context for frame positioning
         :type seek_ctx: Optional[SeekContext]
         :return: Tuple containing:
             - success (bool): True if decoding was successful
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If called without hardware acceleration
     )pbdoc")
      .def_property_readonly(
          "Stream", [](PyDecoder& self) { return (size_t)self.GetStream(); },
          R"pbdoc(
         Get the CUDA stream used by the decoder.

         :return: CUDA stream handle as an integer
         :rtype: int
     )pbdoc")
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
      .def_property_readonly("DisplayRotation", &PyDecoder::GetDisplayRotation,
                             py::call_guard<py::gil_scoped_release>(),
                             R"pbdoc(
        Return last decoded frame display rotation info.
        If there's no such data, 361.0 will be returned.

       :return: value in degrees
    )pbdoc")
      .def_property_readonly("Metadata", &PyDecoder::Metadata,
                             R"pbdoc(
        Return dictionary with video file metadata.
    )pbdoc")
      .def_static(
          "Probe",
          [](const string& input) {
            std::list<StreamParams> info;
            NvDecoderClInterface cli_iface({});
            DecodeFrame::Probe(input.c_str(), cli_iface, info);
            return info;
          },
          py::arg("input"), R"pbdoc(
        Probe input without decoding.
        Information about streams will be returned without codec initialization.

        :param input: path to input file
        :return: list of structures with stream parameters
    )pbdoc");

  m.attr("NO_PTS") = py::int_(AV_NOPTS_VALUE);
}
