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
#include "dlpack.h"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

auto CopyBuffer_Ctx_Str = [](shared_ptr<CudaBuffer> dst,
                             shared_ptr<CudaBuffer> src, CUstream str) {
  if (dst->GetRawMemSize() != src->GetRawMemSize()) {
    throw runtime_error("Can't copy: buffers have different size.");
  }

  CudaCtxPush ctxPush(str);
  ThrowOnCudaError(LibCuda::cuMemcpyDtoDAsync(dst->GpuMem(), src->GpuMem(),
                                              src->GetRawMemSize(), str),
                   __LINE__);
  ThrowOnCudaError(LibCuda::cuStreamSynchronize(str), __LINE__);
};

auto CopyBuffer = [](shared_ptr<CudaBuffer> dst, shared_ptr<CudaBuffer> src,
                     int gpuID) {
  auto str = CudaResMgr::Instance().GetStream(gpuID);
  return CopyBuffer_Ctx_Str(dst, src, str);
};

enum FFMpegLogLevel {
  LOG_PANIC = AV_LOG_PANIC,
  LOG_FATAL = AV_LOG_FATAL,
  LOG_ERROR = AV_LOG_ERROR,
  LOG_WARNING = AV_LOG_WARNING,
  LOG_INFO = AV_LOG_INFO,
  LOG_VERBOSE = AV_LOG_VERBOSE,
  LOG_DEBUG = AV_LOG_DEBUG
};

void Init_PyFrameUploader(py::module&);

void Init_PySurfaceConverter(py::module&);

void Init_PySurfaceDownloader(py::module&);

void Init_PySurfaceResizer(py::module&);

void Init_PyDecoder(py::module&);

void Init_PyNvEncoder(py::module&);

void Init_PySurface(py::module&);

void Init_PyFrameConverter(py::module&);

void Init_PyNvJpegEncoder(py::module& m);

void Init_PySurfaceRotator(py::module& m);

void Init_PySurfaceUD(py::module& m);

PYBIND11_MODULE(_python_vali, m) {

  py::class_<MotionVector, std::shared_ptr<MotionVector>>(
      m, "MotionVector", "This class stores information about motion vector.")
      .def(py::init<>())
      .def_readwrite(
          "source", &MotionVector::source,
          "Frame reference direction indicator. Negative value indicates past reference, "
          "positive value indicates future reference.")
      .def_readwrite("w", &MotionVector::w, 
          "Width of the macroblock in pixels.")
      .def_readwrite("h", &MotionVector::h, 
          "Height of the macroblock in pixels.")
      .def_readwrite("src_x", &MotionVector::src_x,
          "Absolute X coordinate of the source macroblock. May be outside frame boundaries.")
      .def_readwrite("src_y", &MotionVector::src_y,
          "Absolute Y coordinate of the source macroblock. May be outside frame boundaries.")
      .def_readwrite(
          "dst_x", &MotionVector::dst_x,
          "Absolute X coordinate of the destination macroblock. May be outside frame boundaries.")
      .def_readwrite(
          "dst_y", &MotionVector::dst_y,
          "Absolute Y coordinate of the destination macroblock. May be outside frame boundaries.")
      .def_readwrite("motion_x", &MotionVector::motion_x,
          "X component of the motion vector in pixels.")
      .def_readwrite("motion_y", &MotionVector::motion_y,
          "Y component of the motion vector in pixels.")
      .def_readwrite(
          "motion_scale", &MotionVector::motion_scale,
          "Motion vector precision factor. For example, 4 indicates quarter-pixel precision.")
      .def("__repr__", [](shared_ptr<MotionVector> self) {
        std::stringstream ss;
        ss << "source:        " << self->source << "\n";
        ss << "w:             " << self->w << "\n";
        ss << "h:             " << self->h << "\n";
        ss << "src_x:         " << self->src_x << "\n";
        ss << "src_y:         " << self->src_y << "\n";
        ss << "dst_x:         " << self->dst_x << "\n";
        ss << "dst_y:         " << self->dst_y << "\n";
        ss << "motion_x:      " << self->motion_x << "\n";
        ss << "motion_y:      " << self->motion_y << "\n";
        ss << "motion_scale:  " << self->motion_scale << "\n";
        return ss.str();
      });

  PYBIND11_NUMPY_DTYPE_EX(MotionVector, source, "source", w, "w", h, "h", src_x,
                          "src_x", src_y, "src_y", dst_x, "dst_x", dst_y,
                          "dst_y", motion_x, "motion_x", motion_y, "motion_y",
                          motion_scale, "motion_scale");

  py::enum_<Pixel_Format>(m, "PixelFormat")
      .value("Y", Pixel_Format::Y, "Grayscale.")
      .value("RGB", Pixel_Format::RGB, "Interleaved 8 bit RGB.")
      .value("NV12", Pixel_Format::NV12,
             "Semi planar 8 bit: full resolution Y + quarter resolution "
             "interleaved UV.")
      .value("YUV420", Pixel_Format::YUV420,
             "Planar 8 bit: full resolution Y + quarter resolution U + quarter "
             "resolution V.")
      .value("RGB_PLANAR", Pixel_Format::RGB_PLANAR, "Planar 8 bit R+G+B.")
      .value("BGR", Pixel_Format::BGR, "Planar 8 bit R+G+B.")
      .value("YUV444", Pixel_Format::YUV444, "Planar 8 bit Y+U+V.")
      .value("YUV444_10bit", Pixel_Format::YUV444_10bit, "10 bit YUV444.")
      .value("YUV420_10bit", Pixel_Format::YUV420_10bit, "10 bit YUV420")
      .value("UNDEFINED", Pixel_Format::UNDEFINED,
             "Undefined pixel format, use to signal unsupported formats")
      .value("RGB_32F", Pixel_Format::RGB_32F, "32 bit float RGB.")
      .value("RGB_32F_PLANAR", Pixel_Format::RGB_32F_PLANAR,
             "32 bit float planar RGB")
      .value("YUV422", Pixel_Format::YUV422,
             "8 bit planar: full resolution Y + half resolution U + half "
             "resolution V.")
      .value("P10", Pixel_Format::P10, "10 bit NV12.")
      .value("P12", Pixel_Format::P12, "12 bit NV12.")
      .export_values();

  py::enum_<TaskExecInfo>(m, "TaskExecInfo")
      .value("SUCCESS", TaskExecInfo::SUCCESS, "Success")
      .value("FAIL", TaskExecInfo::FAIL, "Fail")
      .value("END_OF_STREAM", TaskExecInfo::END_OF_STREAM, "End of file")
      .value("MORE_DATA_NEEDED", TaskExecInfo::MORE_DATA_NEEDED,
             "More data needed to complete")
      .value("BIT_DEPTH_NOT_SUPPORTED", TaskExecInfo::BIT_DEPTH_NOT_SUPPORTED,
             "Bit depth isn't supported")
      .value("INVALID_INPUT", TaskExecInfo::INVALID_INPUT, "Invalid input")
      .value("UNSUPPORTED_FMT_CONV_PARAMS",
             TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS,
             "Unsupported color conversion parameters")
      .value("NOT_SUPPORTED", TaskExecInfo::NOT_SUPPORTED,
             "Unsupported feature")
      .value("RES_CHANGE", TaskExecInfo::RES_CHANGE,
             "Video resolution change happened")
      .value("SRC_DST_SIZE_MISMATCH", TaskExecInfo::SRC_DST_SIZE_MISMATCH,
             "Input and output size mismatch")
      .value("SRC_DST_FMT_MISMATCH", TaskExecInfo::SRC_DST_FMT_MISMATCH,
             "Input and output pixel format mismatch")
      .export_values();

  py::enum_<ColorSpace>(m, "ColorSpace")
      .value("BT_601", ColorSpace::BT_601, "BT.601 color space.")
      .value("BT_709", ColorSpace::BT_709, "BT.709 color space.")
      .value("UNSPEC", ColorSpace::UNSPEC, "Unspecified color space.")
      .export_values();

  py::enum_<DecodeMode>(m, "DecodeMode")
      .value("KEY_FRAMES", DecodeMode::KEY_FRAMES, "Decode key frames only.")
      .value("ALL_FRAMES", DecodeMode::ALL_FRAMES, "Decode everything.")
      .export_values();

  py::enum_<ColorRange>(m, "ColorRange")
      .value("MPEG", ColorRange::MPEG,
             "Narrow or MPEG color range. Doesn't use full [0;255] range.")
      .value("JPEG", ColorRange::JPEG,
             "Full of JPEG color range. Uses full [0;255] range.")
      .value("UDEF", ColorRange::UDEF, "Undefined color range.")
      .export_values();

  py::enum_<DLDeviceType>(m, "DLDeviceType")
      .value("kDLCPU", kDLCPU, "CPU device.")
      .value("kDLCUDA", kDLCUDA, "CUDA GPU device.")
      .value("kDLCUDAHost", kDLCUDAHost,
             "Pinned CUDA CPU memory by cudaMallocHost.")
      .value("kDLCUDAManaged", kDLCUDAManaged,
             "CUDA managed/unified memory allocated by cudaMallocManaged.")
      .export_values();

  py::enum_<FFMpegLogLevel>(m, "FfmpegLogLevel")
      .value("PANIC", FFMpegLogLevel::LOG_PANIC, "AV_LOG_PANIC")
      .value("FATAL", FFMpegLogLevel::LOG_FATAL, "AV_LOG_FATAL")
      .value("ERROR", FFMpegLogLevel::LOG_ERROR, "AV_LOG_ERROR")
      .value("WARNING", FFMpegLogLevel::LOG_WARNING, "AV_LOG_WARNING")
      .value("INFO", FFMpegLogLevel::LOG_INFO, "AV_LOG_INFO")
      .value("VERBOSE", FFMpegLogLevel::LOG_VERBOSE, "AV_LOG_VERBOSE")
      .value("DEBUG", FFMpegLogLevel::LOG_DEBUG, "AV_LOG_DEBUG")
      .export_values();

  py::class_<SeekContext, shared_ptr<SeekContext>>(
      m, "SeekContext",
      "Context object for video frame seeking operations.")
      .def(py::init<int64_t>(), py::arg("seek_frame"),
           R"pbdoc(
         Create a new seek context for frame-based seeking.

         Initializes a seek context that will seek to a specific frame number
         in the video stream.

         :param seek_frame: Target frame number to seek to (0-based index)
         :type seek_frame: int
     )pbdoc")
      .def(py::init<double>(), py::arg("seek_ts"),
           R"pbdoc(
         Create a new seek context for timestamp-based seeking.

         Initializes a seek context that will seek to a specific timestamp
         in the video stream.

         :param seek_ts: Target timestamp in seconds to seek to
         :type seek_ts: float
     )pbdoc")
      .def_readwrite("seek_frame", &SeekContext::seek_frame,
                     R"pbdoc(
         Target frame number for seeking.

         The frame number to seek to when using frame-based seeking.
         This is a 0-based index into the video stream.
     )pbdoc")
      .def_readwrite("seek_tssec", &SeekContext::seek_tssec,
                     R"pbdoc(
         Target timestamp for seeking.

         The timestamp in seconds to seek to when using timestamp-based seeking.
     )pbdoc");

  py::class_<PacketData, shared_ptr<PacketData>>(m, "PacketData",
                                                 "Video frame metadata container")
      .def(py::init<>())
      .def_readwrite("key", &PacketData::key,
                     "Frame type indicator. 1 for I-frames (key frames), 0 for other frames.")
      .def_readwrite("pts", &PacketData::pts, 
                     "Presentation timestamp of the frame.")
      .def_readwrite("dts", &PacketData::dts, 
                     "Decoding timestamp of the frame.")
      .def_readwrite("pos", &PacketData::pos,
                     "Byte position of the frame's packet in the input bitstream.")
      .def_readwrite(
          "bsl", &PacketData::bsl,
          "Number of bytes consumed by the decoder for this frame's packet. "
          "Useful for seeking operations to find previous key frames.")
      .def_readwrite("duration", &PacketData::duration, 
                     "Duration of the frame's packet in stream timebase units.")
      .def("__repr__", [](shared_ptr<PacketData> self) {
        stringstream ss;
        ss << "key:      " << self->key << "\n";
        ss << "pts:      " << self->pts << "\n";
        ss << "dts:      " << self->dts << "\n";
        ss << "pos:      " << self->pos << "\n";
        ss << "bsl:      " << self->bsl << "\n";
        ss << "duration: " << self->duration << "\n";
        return ss.str();
      });

  py::class_<CudaStreamEvent, shared_ptr<CudaStreamEvent>>(m, "CudaStreamEvent",
                                                           "CUDA stream synchronization event")
      .def(py::init<size_t, int>(), py::arg("stream"), py::arg("gpu_id"),
           R"pbdoc(
         Create a new CUDA stream event.

         Initializes a CUDA event for synchronizing operations on a specific
         CUDA stream and GPU.

         :param stream: CUDA stream handle to associate with the event
         :type stream: int
         :param gpu_id: GPU device ID to use
         :type gpu_id: int
     )pbdoc")
      .def("Record", &CudaStreamEvent::Record,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
         Record the current state of the CUDA stream.

         Creates a synchronization point in the CUDA stream. Equivalent to
         cuEventRecord in the CUDA API.

         :note: This operation is asynchronous and returns immediately
     )pbdoc")
      .def("Wait", &CudaStreamEvent::Wait,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
         Wait for the recorded event to complete.

         Blocks until all operations in the CUDA stream up to the recorded
         event have completed. Equivalent to cuEventSynchronize in the CUDA API.

         :note: This operation is synchronous and will block until completion
     )pbdoc");

  py::class_<TaskExecDetails, shared_ptr<TaskExecDetails>>(m, "TaskExecDetails",
      "Container for task execution status and information")
      .def(py::init<>())
      .def_readwrite("info", &TaskExecDetails::m_info,
          "Detailed execution information (TaskExecInfo enum)")
      .def_readwrite("status", &TaskExecDetails::m_status,
          "Execution status (TaskExecStatus enum)")
      .def_readwrite("message", &TaskExecDetails::m_msg,
          "Optional error or status message");

  py::class_<ColorspaceConversionContext,
             shared_ptr<ColorspaceConversionContext>>(
      m, "ColorspaceConversionContext",
      "Context for color space and range conversion operations")
      .def(py::init<>())
      .def(py::init<ColorSpace, ColorRange>(), py::arg("color_space"),
           py::arg("color_range"),
           R"pbdoc(
         Create a new color space conversion context.

         Initializes a context with specific color space and range settings
         for accurate color conversion operations.

         :param color_space: Source color space (e.g., BT_601, BT_709)
         :type color_space: ColorSpace
         :param color_range: Source color range (e.g., MPEG, JPEG)
         :type color_range: ColorRange
     )pbdoc")
      .def_readwrite("color_space", &ColorspaceConversionContext::color_space,
          "Color space setting (e.g., BT_601, BT_709)")
      .def_readwrite("color_range", &ColorspaceConversionContext::color_range,
          "Color range setting (e.g., MPEG, JPEG)");

  py::class_<CudaBuffer, shared_ptr<CudaBuffer>>(
      m, "CudaBuffer", "GPU memory buffer for data storage and manipulation")
      .def_property_readonly("RawMemSize", &CudaBuffer::GetRawMemSize,
                             R"pbdoc(
         Get the total size of the buffer in bytes.

         :return: Total buffer size in bytes
         :rtype: int
     )pbdoc")
      .def_property_readonly("NumElems", &CudaBuffer::GetNumElems,
                             R"pbdoc(
         Get the number of elements in the buffer.

         :return: Number of elements
         :rtype: int
     )pbdoc")
      .def_property_readonly("ElemSize", &CudaBuffer::GetElemSize,
                             R"pbdoc(
         Get the size of a single element in bytes.

         :return: Size of one element in bytes
         :rtype: int
     )pbdoc")
      .def_property_readonly("GpuMem", &CudaBuffer::GpuMem,
                             R"pbdoc(
         Get the CUDA device pointer to the buffer memory.

         :return: CUDA device pointer as integer
         :rtype: int
     )pbdoc")
      .def("Clone", &CudaBuffer::Clone, py::return_value_policy::take_ownership,
           R"pbdoc(
         Create a deep copy of the buffer.

         Allocates new GPU memory and copies the contents of this buffer.
         The caller is responsible for managing the returned buffer's lifetime.

         :return: New CudaBuffer instance with copied data
         :rtype: CudaBuffer
     )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<CudaBuffer> self, shared_ptr<CudaBuffer> other,
             size_t str) { CopyBuffer_Ctx_Str(self, other, (CUstream)str); },
          py::arg("other"), py::arg("stream"),
          R"pbdoc(
         Copy data from another CudaBuffer using a specific CUDA stream.

         :param other: Source CudaBuffer to copy from
         :type other: CudaBuffer
         :param stream: CUDA stream to use for the copy operation
         :type stream: int
         :raises RuntimeError: If buffer sizes don't match
     )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<CudaBuffer> self, shared_ptr<CudaBuffer> other,
             int gpuID) { CopyBuffer(self, other, gpuID); },
          py::arg("other"), py::arg("gpu_id"),
          R"pbdoc(
         Copy data from another CudaBuffer using the default stream of a GPU.

         :param other: Source CudaBuffer to copy from
         :type other: CudaBuffer
         :param gpu_id: GPU device ID to use for the copy operation
         :type gpu_id: int
         :raises RuntimeError: If buffer sizes don't match
     )pbdoc")
      .def_static(
          "Make",
          [](uint32_t elem_size, uint32_t num_elems, int gpuID) {
            auto pNewBuf = shared_ptr<CudaBuffer>(CudaBuffer::Make(
                elem_size, num_elems, CudaResMgr::Instance().GetCtx(gpuID)));
            return pNewBuf;
          },
          py::arg("elem_size"), py::arg("num_elems"), py::arg("gpu_id"),
          py::return_value_policy::take_ownership,
          R"pbdoc(
         Create a new CudaBuffer instance.

         Allocates GPU memory for the specified number of elements.

         :param elem_size: Size of each element in bytes
         :type elem_size: int
         :param num_elems: Number of elements to allocate
         :type num_elems: int
         :param gpu_id: GPU device ID to allocate memory on
         :type gpu_id: int
         :return: New CudaBuffer instance
         :rtype: CudaBuffer
         :raises RuntimeError: If memory allocation fails
     )pbdoc");

  py::class_<StreamParams, shared_ptr<StreamParams>>(m, "StreamParams",
                                                     "Video stream parameters container")
      .def(py::init<>())
      .def_readwrite("width", &StreamParams::width,
          "Width of the video stream in pixels")
      .def_readwrite("height", &StreamParams::height,
          "Height of the video stream in pixels")
      .def_readwrite("fourcc", &StreamParams::fourcc,
          "FourCC code identifying the codec")
      .def_readwrite("num_frames", &StreamParams::num_frames,
          "Total number of frames in the stream")
      .def_readwrite("start_time", &StreamParams::start_time,
          "Stream start time in stream timebase units")
      .def_readwrite("bit_rate", &StreamParams::bit_rate,
          "Stream bitrate in bits per second")
      .def_readwrite("profile", &StreamParams::profile,
          "Codec profile identifier")
      .def_readwrite("level", &StreamParams::level,
          "Codec level identifier")
      .def_readwrite("codec_id", &StreamParams::codec_id,
          "Codec identifier")
      .def_readwrite("color_space", &StreamParams::color_space,
          "Color space of the stream")
      .def_readwrite("color_range", &StreamParams::color_range,
          "Color range of the stream")
      .def_readwrite("fps", &StreamParams::fps,
          "Nominal frame rate of the stream")
      .def_readwrite("avg_fps", &StreamParams::avg_fps,
          "Average frame rate of the stream")
      .def_readwrite("time_base", &StreamParams::time_base,
          "Time base of the stream (1/fps)")
      .def_readwrite("start_time_sec", &StreamParams::start_time_sec,
          "Stream start time in seconds")
      .def_readwrite("duration_sec", &StreamParams::duration_sec,
          "Stream duration in seconds")
      .def("__repr__", [](StreamParams& self) {
        stringstream ss;
        ss << "width:           " << self.width << "\n";
        ss << "height:          " << self.height << "\n";
        ss << "fourcc:          " << self.fourcc << "\n";
        ss << "num_frames:      " << self.num_frames << "\n";
        ss << "start_time:      " << self.start_time << "\n";
        ss << "bit_rate:        " << self.bit_rate << "\n";
        ss << "profile:         " << self.profile << "\n";
        ss << "profile:         " << self.level << "\n";
        ss << "codec_id:        " << self.codec_id << "\n";
        ss << "color_space:     " << self.color_space << "\n";
        ss << "color_range:     " << self.color_range << "\n";
        ss << "fps:             " << self.fps << "\n";
        ss << "avg_fps:         " << self.avg_fps << "\n";
        ss << "time_base:       " << self.time_base << "\n";
        ss << "start_time_sec:  " << self.start_time_sec << "\n";
        ss << "duration_sec:    " << self.duration_sec << "\n";
        return ss.str();
      });

  m.def("GetNumGpus", &CudaResMgr::GetNumGpus, R"pbdoc(
         Get the number of available CUDA-capable GPUs in the system.

         :return: Number of available GPUs
         :rtype: int
     )pbdoc");

  m.def("GetNvencParams", &GetNvencInitParams, R"pbdoc(
         Get the list of parameters that can be used to initialize PyNvEncoder.

         :return: Dictionary of available encoder parameters
         :rtype: dict
     )pbdoc");

  m.def(
      "SetFFMpegLogLevel",
      [](FFMpegLogLevel level) { av_log_set_level(int(level)); },
      py::arg("level"),
      R"pbdoc(
         Set the logging level for FFmpeg operations.

         :param level: Logging level to set (e.g., ERROR, WARNING, INFO)
         :type level: FfmpegLogLevel
     )pbdoc");

  Init_PyDecoder(m);

  Init_PyNvEncoder(m);

  Init_PyFrameUploader(m);

  Init_PySurfaceDownloader(m);

  Init_PySurfaceConverter(m);

  Init_PySurfaceResizer(m);

  Init_PySurface(m);

  Init_PyFrameConverter(m);

  Init_PyNvJpegEncoder(m);

  Init_PySurfaceRotator(m);

  Init_PySurfaceUD(m);

  av_log_set_level(AV_LOG_ERROR);

  m.doc() = R"pbdoc(
        python_vali
        ----------
        .. currentmodule:: python_vali
        .. autosummary::
           :toctree: _generate

           GetNumGpus
           GetNvencParams
           SetFFMpegLogLevel
           PySurfaceResizer
           PySurfaceDownloader
           PySurfaceConverter
           PyNvEncoder
           PyDecoder
           PyFrameUploader
           PyBufferUploader
           PyNvJpegEncoder
           SeekContext
           SurfacePlane
           Surface
           CudaStreamEvent
           PySurfaceRotator
           PySurfaceUD

    )pbdoc";
}
