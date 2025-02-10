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

DecodeContext::DecodeContext(py::array_t<uint8_t>* sei,
                             py::array_t<uint8_t>* packet,
                             PacketData* in_pkt_data, PacketData* out_pkt_data,
                             SeekContext* seek_ctx, bool is_flush) {
  if (seek_ctx && packet) {
    throw runtime_error("Can't use seek in standalone mode.");
  }

  pSurface = nullptr;
  pSei = sei;
  pPacket = packet;
  pInPktData = in_pkt_data;
  pOutPktData = out_pkt_data;
  pSeekCtx = seek_ctx;
  flush = is_flush;
}

bool DecodeContext::IsSeek() const {
  return (nullptr != pSeekCtx) && (nullptr == pPacket);
}

bool DecodeContext::IsStandalone() const { return (nullptr != pPacket); }

bool DecodeContext::IsFlush() const { return flush; }

bool DecodeContext::HasSEI() const { return nullptr != pSei; }

bool DecodeContext::HasOutPktData() const { return nullptr != pOutPktData; }

bool DecodeContext::HasInPktData() const { return nullptr != pInPktData; }

const py::array_t<uint8_t>* DecodeContext::GetPacket() const { return pPacket; }

const PacketData* DecodeContext::GetInPacketData() const { return pInPktData; }

const SeekContext* DecodeContext::GetSeekContext() const { return pSeekCtx; }

SeekContext* DecodeContext::GetSeekContextMutable() { return pSeekCtx; }

shared_ptr<Surface> DecodeContext::GetSurfaceMutable() { return pSurface; }

void DecodeContext::SetOutPacketData(PacketData* out_pkt_data) {
  if (!out_pkt_data || !pOutPktData) {
    throw runtime_error("Invalid data pointer");
  }

  memcpy(pOutPktData, out_pkt_data, sizeof(PacketData));
}

void DecodeContext::SetOutPacketData(const PacketData& out_pkt_data) {
  if (!pOutPktData) {
    throw runtime_error("Invalid data pointer");
  }

  memcpy(pOutPktData, (const void*)&out_pkt_data, sizeof(PacketData));
}

void DecodeContext::SetSei(Buffer* sei) {
  if (!pSei) {
    throw runtime_error("Invalid data pointer");
  }

  if (!sei) {
    pSei->resize({0}, false);
    return;
  }

  pSei->resize({sei->GetRawMemSize()}, false);
  memcpy(pSei->mutable_data(), sei->GetRawMemPtr(), sei->GetRawMemSize());
}

void DecodeContext::SetCloneSurface(Surface* p_surface) {
  if (!p_surface) {
    throw runtime_error("Invalid data pointer");
  }
  pSurface = shared_ptr<Surface>(p_surface->Clone());
}

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

PYBIND11_MODULE(_python_vali, m) {

  py::class_<MotionVector, std::shared_ptr<MotionVector>>(
      m, "MotionVector", "This class stores iformation about motion vector.")
      .def(py::init<>())
      .def_readwrite(
          "source", &MotionVector::source,
          "Where the current macroblock comes from; negative value when it "
          "comes from the past, positive value when it comes from the future.")
      .def_readwrite("w", &MotionVector::w, "Macroblock width.")
      .def_readwrite("h", &MotionVector::h, "Macroblock height")
      .def_readwrite("src_x", &MotionVector::src_x,
                     "Absolute source X position. Can be outside frame area.")
      .def_readwrite("src_y", &MotionVector::src_y,
                     "Absolute source Y position. Can be outside frame area.")
      .def_readwrite(
          "dst_x", &MotionVector::dst_x,
          "Absolute detination X position. Can be outside frame area.")
      .def_readwrite(
          "dst_y", &MotionVector::dst_y,
          "Absolute detination Y position. Can be outside frame area.")
      .def_readwrite("motion_x", &MotionVector::motion_x,
                     "Motion vector X component.")
      .def_readwrite("motion_y", &MotionVector::motion_y,
                     "Motion vector Y component.")
      .def_readwrite(
          "motion_scale", &MotionVector::motion_scale,
          "Motion prediction precision. E. g. 4 for quarter-pixel precision.")
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
      .export_values();

  py::enum_<ColorSpace>(m, "ColorSpace")
      .value("BT_601", ColorSpace::BT_601, "BT.601 color space.")
      .value("BT_709", ColorSpace::BT_709, "BT.709 color space.")
      .value("UNSPEC", ColorSpace::UNSPEC, "Unspecified color space.")
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
      "Incapsulates information required by decoder to seek for a particular "
      "video frame.")
      .def(py::init<int64_t>(), py::arg("seek_frame"),
           R"pbdoc(
        Constructor method.

        :param seek_frame: number of frame to seek for, starts from 0
    )pbdoc")
      .def(py::init<double>(), py::arg("seek_ts"),
           R"pbdoc(
        Constructor method.
        Will initialize context for seek by frame timestamp.

        :param seek_frame: timestamp (s) of frame to seek for.
    )pbdoc")
      .def_readwrite("seek_frame", &SeekContext::seek_frame,
                     R"pbdoc(
        Number of frame we want to seek.
    )pbdoc")
      .def_readwrite("seek_tssec", &SeekContext::seek_tssec,
                     R"pbdoc(
        Timestamp we want to seek.
    )pbdoc");

  py::class_<PacketData, shared_ptr<PacketData>>(m, "PacketData",
                                                 "Video frame attributes")
      .def(py::init<>())
      .def_readwrite("key", &PacketData::key,
                     "1 if frame is I frame, 0 otherwise.")
      .def_readwrite("pts", &PacketData::pts, "Presentation timestamp.")
      .def_readwrite("dts", &PacketData::dts, "Decode timestamp.")
      .def_readwrite("pos", &PacketData::pos,
                     "Position of compressed packet in input bitstream.")
      .def_readwrite(
          "bsl", &PacketData::bsl,
          "Amount of bytes decoder had to consume to decode corresp. packet. "
          "Useful to see when seeking for a previous key frame.")
      .def_readwrite("duration", &PacketData::duration, "Duration of a packet.")
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
                                                           "CUDA stream event")
      .def("Wait", &CudaStreamEvent::Wait,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
      Will not return until CUDA event is synchronized.
      Acts exactly like cuEventSyncronize.
      )pbdoc");

  py::class_<TaskExecDetails, shared_ptr<TaskExecDetails>>(m, "TaskExecDetails")
      .def(py::init<>())
      .def_readwrite("info", &TaskExecDetails::m_info)
      .def_readwrite("status", &TaskExecDetails::m_status)
      .def_readwrite("message", &TaskExecDetails::m_msg);

  py::class_<ColorspaceConversionContext,
             shared_ptr<ColorspaceConversionContext>>(
      m, "ColorspaceConversionContext",
      "Stores information required for accurate color conversion.")
      .def(py::init<>())
      .def(py::init<ColorSpace, ColorRange>(), py::arg("color_space"),
           py::arg("color_range"))
      .def_readwrite("color_space", &ColorspaceConversionContext::color_space)
      .def_readwrite("color_range", &ColorspaceConversionContext::color_range);

  py::class_<CudaBuffer, shared_ptr<CudaBuffer>>(
      m, "CudaBuffer", "General purpose data storage class in GPU memory.")
      .def_property_readonly("RawMemSize", &CudaBuffer::GetRawMemSize,
                             R"pbdoc(
        Get size of buffer in bytes.

        :rtype: Int
    )pbdoc")
      .def_property_readonly("NumElems", &CudaBuffer::GetNumElems,
                             R"pbdoc(
        Get number of elements in buffer.

        :rtype: Int
    )pbdoc")
      .def_property_readonly("ElemSize", &CudaBuffer::GetElemSize,
                             R"pbdoc(
        Get size of single element in bytes

        :rtype: Int
    )pbdoc")
      .def_property_readonly("GpuMem", &CudaBuffer::GpuMem,
                             R"pbdoc(
        Get CUdeviceptr of memory allocation.

        :rtype: Int
    )pbdoc")
      .def("Clone", &CudaBuffer::Clone, py::return_value_policy::take_ownership,
           R"pbdoc(
        Deep copy = CUDA mem alloc + CUDA mem copy.

        :rtype: python_vali.CudaBuffer
    )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<CudaBuffer> self, shared_ptr<CudaBuffer> other,
             size_t str) { CopyBuffer_Ctx_Str(self, other, (CUstream)str); },
          py::arg("other"), py::arg("stream"),
          R"pbdoc(
        Copy content of another CudaBuffer into this CudaBuffer

        :param other: other CudaBuffer
        :param stream: CUDA stream to use
        :rtype: None
    )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<CudaBuffer> self, shared_ptr<CudaBuffer> other,
             int gpuID) { CopyBuffer(self, other, gpuID); },
          py::arg("other"), py::arg("gpu_id"),
          R"pbdoc(
        Copy content of another CudaBuffer into this CudaBuffer

        :param other: other CudaBuffer
        :param gpu_id: GPU to use for memcopy
        :rtype: None
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
        Constructor method

        :param elem_size: single buffer element size in bytes
        :param num_elems: number of elements in buffer
        :param gpu_id: GPU to use for memcopy
        :rtype: python_vali.CudaBuffer
    )pbdoc");

  m.def("GetNumGpus", &CudaResMgr::GetNumGpus, R"pbdoc(
        Get number of available GPUs.
    )pbdoc");

  m.def("GetNvencParams", &GetNvencInitParams, R"pbdoc(
        Get list of params PyNvEncoder can be initialized with.
    )pbdoc");

  m.def(
      "SetFFMpegLogLevel",
      [](FFMpegLogLevel level) { av_log_set_level(int(level)); },
      R"pbdoc(
        Set FFMpeg log level.
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

    )pbdoc";
}
