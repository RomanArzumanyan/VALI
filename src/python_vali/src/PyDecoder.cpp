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

#include "Utils.hpp"
#include "VALI.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

/**
 * @brief Construct a new PyDecoder object from a file path
 *
 * Initializes a video decoder that can decode frames from a video file.
 * The decoder can operate in either CPU or GPU mode depending on the gpuID parameter.
 *
 * @param pathToFile Path to the input video file
 * @param ffmpeg_options Dictionary of options to pass to libavcodec API
 * @param gpuID GPU device ID to use for hardware acceleration. Use negative value for CPU-only decoding
 * @param pkt_queue_size Internal decoder packet queue size
 */
PyDecoder::PyDecoder(const string& pathToFile,
                     const map<string, string>& ffmpeg_options, int gpuID,
                     int pkt_queue_size) {
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);

  upDecoder.reset(
      DecodeFrame::Make(pathToFile.c_str(), cli_iface, gpu_id, pkt_queue_size));
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

/**
 * @brief Construct a new PyDecoder object from a buffered reader
 *
 * Initializes a video decoder that can decode frames from a buffered reader object.
 * The decoder can operate in either CPU or GPU mode depending on the gpuID parameter.
 *
 * @param buffered_reader Python object with a 'read' method (e.g., io.BufferedReader)
 * @param ffmpeg_options Dictionary of options to pass to libavcodec API
 * @param gpuID GPU device ID to use for hardware acceleration. Use negative value for CPU-only decoding
 * @param pkt_queue_size Internal decoder packet queue size
 */
PyDecoder::PyDecoder(py::object buffered_reader,
                     const map<string, string>& ffmpeg_options, int gpuID,
                     int pkt_queue_size) {
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);

  upBuff.reset(new BufferedReader(buffered_reader));
  upDecoder.reset(DecodeFrame::Make("", cli_iface, gpu_id, pkt_queue_size,
                                    upBuff->GetAVIOContext()));
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

/**
 * @brief Internal implementation method for decoding frames
 *
 * This method performs the actual decoding operation and updates the decoder state.
 *
 * @param details Task execution details object to store execution information
 * @param pkt_data Packet data object to store packet metadata
 * @param dst Destination token (can be a Surface or Buffer)
 * @param seek_ctx Optional seek context for frame positioning
 * @return true if decoding was successful, false otherwise
 */
bool PyDecoder::DecodeImpl(TaskExecDetails& details, PacketData& pkt_data,
                           Token& dst, std::optional<SeekContext> seek_ctx) {
  details = upDecoder->Run(dst, pkt_data, seek_ctx);
  UpdateState();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

/**
 * @brief Decode a single video frame from the input source into a CPU buffer
 *
 * This method is for CPU-only decoding (non-accelerated decoder).
 * The frame will be decoded into the provided numpy array.
 *
 * @param frame Numpy array to store the decoded frame
 * @param details Task execution details object to store execution information
 * @param pkt_data Packet data object to store packet metadata
 * @param seek_ctx Optional seek context for frame positioning
 * @return true if decoding was successful, false otherwise
 * @throws std::runtime_error if called with hardware acceleration enabled
 */
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

/**
 * @brief Decode a single video frame into a CUDA surface
 *
 * This method is for hardware-accelerated decoding.
 * The frame will be decoded directly into the provided CUDA surface.
 *
 * @param surf CUDA surface to store the decoded frame
 * @param details Task execution details object to store execution information
 * @param pkt_data Packet data object to store packet metadata
 * @param seek_ctx Optional seek context for frame positioning
 * @return true if decoding was successful, false otherwise
 * @throws std::runtime_error if called without hardware acceleration
 */
bool PyDecoder::DecodeSingleSurface(Surface& surf, TaskExecDetails& details,
                                     PacketData& pkt_data,
                                     std::optional<SeekContext> seek_ctx) {
  if (!IsAccelerated()) {
    details.m_info = TaskExecInfo::FAIL;
    return false;
  }

  if (surf.Empty()) {
    av_log(nullptr, AV_LOG_ERROR, "Empty Surface \n");
    return false;
  }

  if (surf.Width() != Width() || surf.Height() != Height()) {
    av_log(nullptr, AV_LOG_ERROR,
           "Surface dimensions mismatch: %d x %d vs %d x %d \n", surf.Width(),
           surf.Height(), Width(), Height());
    return false;
  }

  if (surf.PixelFormat() != PixelFormat()) {
    av_log(nullptr, AV_LOG_ERROR, "Pixel format mismatch: %s vs %s \n",
           GetFormatName(surf.PixelFormat()).c_str(),
           GetFormatName(PixelFormat()).c_str());
    return false;
  }

  return DecodeImpl(details, pkt_data, surf, seek_ctx);
}

/**
 * @brief Update the decoder state with current frame dimensions
 *
 * This method stores the current width and height of the decoded frame
 * for tracking changes in video dimensions.
 */
void PyDecoder::UpdateState() {
  last_h = Height();
  last_w = Width();
}

/**
 * @brief Get the display rotation of the last decoded frame
 *
 * This method retrieves the display rotation information stored in the video file.
 * If there's no such data, 361.0 will be returned.
 *
 * @return double Value in degrees representing the display rotation
 */
double PyDecoder::GetDisplayRotation() const {
  Buffer buf(0U, false);
  auto ret = upDecoder->GetSideData(AV_FRAME_DATA_DISPLAYMATRIX, buf);
  if (ret.m_info != TaskExecInfo::SUCCESS)
    return 361.f;

  return *(buf.GetDataAs<double>());
}

/**
 * @brief Get motion vectors of last decoded frame
 *
 * This method retrieves the motion vectors stored in the video file.
 * If there are no motion vectors, it will return an empty list.
 *
 * @return std::vector<MotionVector> List of motion vectors
 */
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

/**
 * @brief Get the width of the encoded video
 *
 * @return uint32_t Width in pixels
 */
uint32_t PyDecoder::Width() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.width;
};

/**
 * @brief Get the height of the encoded video
 *
 * @return uint32_t Height in pixels
 */
uint32_t PyDecoder::Height() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.height;
};

/**
 * @brief Get the level coding parameter of the encoded video
 *
 * @return uint32_t Level coding parameter
 */
uint32_t PyDecoder::Level() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.level;
};

/**
 * @brief Get the profile coding parameter of the encoded video
 *
 * @return uint32_t Profile coding parameter
 */
uint32_t PyDecoder::Profile() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.profile;
};

/**
 * @brief Get the delay of the encoded video
 *
 * @return uint32_t Delay value
 */
uint32_t PyDecoder::Delay() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.delay;
};

/**
 * @brief Get the GOP size of the encoded video
 *
 * @return uint32_t GOP size
 */
uint32_t PyDecoder::GopSize() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.gop_size;
};

/**
 * @brief Get the bitrate of the encoded video
 *
 * @return uint32_t Bitrate in bits per second
 */
uint32_t PyDecoder::Bitrate() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.bit_rate;
};

/**
 * @brief Get the number of video frames in the encoded video file
 *
 * Please note that some video containers don't store this information.
 *
 * @return uint32_t Number of video frames
 */
uint32_t PyDecoder::NumFrames() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.num_frames;
};

/**
 * @brief Get the number of streams in the video file
 *
 * E.g. 2 streams: audio and video.
 *
 * @return uint32_t Number of streams in video file
 */
uint32_t PyDecoder::NumStreams() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.num_streams;
};

/**
 * @brief Get the index of the current video stream in the file
 *
 * E.g. video stream has index 0, and audio stream has index 1.
 * This method will return 0 then.
 *
 * @return uint32_t Index of current video stream in file
 */
uint32_t PyDecoder::StreamIndex() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_index;
};

/**
 * @brief Get the amount of bytes needed to store a decoded frame
 *
 * @return uint32_t Amount of bytes needed to store decoded frame
 */
uint32_t PyDecoder::HostFrameSize() const {
  return upDecoder->GetHostFrameSize();
};

/**
 * @brief Get the framerate of the encoded video file
 *
 * @return double Framerate
 */
double PyDecoder::Framerate() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.fps;
};

/**
 * @brief Get color space information stored in video file
 *
 * Please note that some video containers may not store this information.
 *
 * @return ColorSpace Color space information
 */
ColorSpace PyDecoder::Color_Space() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.color_space;
};

/**
 * @brief Get color range information stored in video file
 *
 * Please note that some video containers may not store this information.
 *
 * @return ColorRange Color range information
 */
ColorRange PyDecoder::Color_Range() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.color_range;
};

/**
 * @brief Get the average framerate of the encoded video file
 *
 * @return double Average framerate
 */
double PyDecoder::AvgFramerate() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.avg_fps;
};

/**
 * @brief Get the time base of the encoded video file
 *
 * @return double Time base
 */
double PyDecoder::Timebase() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.time_base;
};

/**
 * @brief Get the start time of the video in seconds
 *
 * @return double Video start time in seconds
 */
double PyDecoder::StartTime() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.start_time_sec;
};

/**
 * @brief Get the duration of the video in seconds
 *
 * May not be present in some video containers.
 *
 * @return double Video duration time in seconds
 */
double PyDecoder::Duration() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.duration_sec;
};

/**
 * @brief Get the pixel format of the encoded video file
 *
 * @return Pixel_Format Pixel format
 */
Pixel_Format PyDecoder::PixelFormat() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.codec_params.format;
};

/**
 * @brief Check if decoder has HW acceleration support
 *
 * @return true if decoder has HW acceleration support, false otherwise
 */
bool PyDecoder::IsAccelerated() const { return upDecoder->IsAccelerated(); }

/**
 * @brief Check if video has variable framerate
 *
 * @return true if video has variable framerate, false otherwise
 */
bool PyDecoder::IsVFR() const {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.stream_params.fps !=
         params.videoContext.stream_params.avg_fps;
}

/**
 * @brief Get the CUDA stream used by the decoder
 *
 * @return CUstream CUDA stream handle
 */
CUstream PyDecoder::GetStream() const { return upDecoder->GetStream(); }

/**
 * @brief Get dictionary with video file metadata
 *
 * @return metadata_dict Dictionary with video file metadata
 */
metadata_dict PyDecoder::Metadata() {
  Params params;
  upDecoder->GetParams(params);
  return params.videoContext.metadata;
}

/**
 * @brief Set the decoder operation mode
 *
 * Changes how the decoder processes frames and handles seeking operations.
 * When in KEY_FRAMES mode, seeking will return the closest previous key frame.
 * When switching modes, the internal frame queue is preserved to avoid discarding
 * decoded frames that may be needed for future operations.
 *
 * @param new_mode The new decode mode to set
 */
void PyDecoder::SetMode(DecodeMode new_mode) { upDecoder->SetMode(new_mode); }

/**
 * @brief Get the current decoder operation mode
 *
 * @return DecodeMode Current decode mode (e.g., KEY_FRAMES, ALL_FRAMES)
 */
DecodeMode PyDecoder::GetMode() const { return upDecoder->GetMode(); }

/**
 * @brief Reads single compressed video packet
 *
 * @return DECODE_STATUS Decode status
 */
DECODE_STATUS PyDecoder::ReadPacket() { return upDecoder->ReadPacket(); }

/**
 * @brief Decode a single video packet into a CUDA surface
 *
 * This method is for hardware-accelerated decoding.
 * The packet will be decoded directly into the provided CUDA surface.
 *
 * @param surf CUDA surface to store the decoded frame
 * @return DECODE_STATUS Decode status
 * @throws std::runtime_error if called without hardware acceleration
 */
DECODE_STATUS PyDecoder::DecodePacketToSurface(Surface& surf) {
  if (!IsAccelerated())
    return DEC_ERROR;

  return upDecoder->DecodePacket(surf);
}

/**
 * @brief Decode a single video packet from the input source
 *
 * This method is for CPU-only decoding (non-accelerated decoder).
 * The packet will be decoded into the provided numpy array.
 *
 * @param frame Numpy array to store the decoded frame
 * @return DECODE_STATUS Decode status
 * @throws std::runtime_error if called with hardware acceleration enabled
 */
DECODE_STATUS PyDecoder::DecodePacketToFrame(py::array& frame) {
  if (IsAccelerated())
    return DEC_ERROR;

  auto const frame_size = upDecoder->GetHostFrameSize();
  if (frame_size != frame.nbytes())
    frame.resize({frame_size}, false);

  auto dst = std::shared_ptr<Buffer>(
      Buffer::Make(frame.nbytes(), frame.mutable_data()));

  py::gil_scoped_release gil_release{};
  return upDecoder->DecodePacket(*dst.get());
}

/**
 * @brief Initialize the PyDecoder Python bindings
 *
 * This function sets up the Python bindings for the PyDecoder class using pybind11.
 * It exposes all the methods and properties of the PyDecoder class to Python.
 *
 * @param m The pybind11 module to bind to
 */
void Init_PyDecoder(py::module& m) {
  py::class_<PyDecoder, shared_ptr<PyDecoder>>(m, "PyDecoder",
                                               "Video decoder class.")
      .def(py::init<const string&, const map<string, string>&, int, int>(),
                   py::arg("input"), py::arg("opts"), py::arg("gpu_id") = 0,
                   py::arg("pkt_queue_size") = 25,
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
                :param pkt_queue_size: Internal decoder packet queue size. Default is 25.
                :type pkt_queue_size: int
                :raises RuntimeError: If decoder initialization fails
            )pbdoc")
      .def(py::init<py::object, const map<string, string>&, int, int>(),
                   py::arg("buffered_reader"), py::arg("opts"), py::arg("gpu_id") = 0,
                   py::arg("pkt_queue_size") = 25,
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
                :param pkt_queue_size: Internal decoder packet queue size. Default is 25.
                :type pkt_queue_size: int
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
                  "ReadPacket", [](PyDecoder& self) { return self.ReadPacket(); },
                  py::call_guard<py::gil_scoped_release>(),
                  R"pbdoc(
                Read a single compressed video packet.
       
                :return: Decode status
                :rtype: DecodeStatus
            )pbdoc")
      .def(
          "DecodePacketToSurface",
          [](PyDecoder& self, Surface& surf) {
            auto ret = self.DecodePacketToSurface(surf);
            if (DEC_SUCCESS == ret) {
              self.m_event->Record();
              self.m_event->Wait();
            }
            return ret;
          },
          py::arg("surf"), py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Decode a single video frame into a CUDA surface.

         This method is for hardware-accelerated decoding.
         The frame will be decoded directly into the provided CUDA surface.

         :param surf: CUDA surface to store the decoded frame
         :rtype: DecodeStatus
         :raises RuntimeError: If called without hardware acceleration
     )pbdoc")
      .def(
                  "DecodePacketToSurfaceAsync",
                  [](PyDecoder& self, Surface& surf) {
                    return self.DecodePacketToSurface(surf);
                  },
                  py::arg("surf"), py::call_guard<py::gil_scoped_release>(),
                  R"pbdoc(
                Decode a single video packet into a CUDA surface asynchronously.
       
                This method is for hardware-accelerated decoding.
                The packet will be decoded directly into the provided CUDA surface.
                The operation is asynchronous and returns immediately.
       
                :param surf: CUDA surface to store the decoded frame
                :rtype: DecodeStatus
                :raises RuntimeError: If called without hardware acceleration
            )pbdoc")
      .def(
          "DecodePacketToFrame",
          [](PyDecoder& self, py::array& frame) {
            return self.DecodePacketToFrame(frame);
          },
          py::arg("frame"),
          R"pbdoc(
         Decode a single video frame from the input source.

         This method is for CPU-only decoding (non-accelerated decoder).
         The frame will be decoded into the provided numpy array.

         :param frame: Numpy array to store the decoded frame
         :rtype: DecodeStatus
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
               Please note that some video containers don't store this information.
           )pbdoc")
      .def_property_readonly("ColorSpace", &PyDecoder::Color_Space,
                                     R"pbdoc(
               Get color space information stored in video file.
               Please note that some video containers may not store this information.
       
               :return: color space information
           )pbdoc")
      .def_property_readonly("ColorRange", &PyDecoder::Color_Range,
                                     R"pbdoc(
               Get color range information stored in video file.
               Please note that some video containers may not store this information.
       
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
               If there are no motion vectors it will return empty list.
       
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
