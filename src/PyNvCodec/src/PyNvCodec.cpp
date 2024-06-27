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

#include "PyNvCodec.hpp"
#include "dlpack.h"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

CudaResMgr::CudaResMgr() {
  lock_guard<mutex> lock_ctx(CudaResMgr::gInsMutex);

  ThrowOnCudaError(cuInit(0), __LINE__);

  int nGpu;
  ThrowOnCudaError(cuDeviceGetCount(&nGpu), __LINE__);

  for (int i = 0; i < nGpu; i++) {
    CUdevice cuDevice = 0;
    CUcontext cuContext = nullptr;
    g_Contexts.push_back(make_pair(cuDevice, cuContext));

    CUstream cuStream = nullptr;
    g_Streams.push_back(cuStream);
  }
  return;
}

CUcontext CudaResMgr::GetCtx(size_t idx) {
  lock_guard<mutex> lock_ctx(CudaResMgr::gCtxMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& ctx = g_Contexts[idx];
  if (!ctx.second) {
    CUdevice cuDevice = 0;
    ThrowOnCudaError(cuDeviceGet(&cuDevice, idx), __LINE__);
    ThrowOnCudaError(cuDevicePrimaryCtxRetain(&ctx.second, cuDevice), __LINE__);
  }

  return g_Contexts[idx].second;
}

CUstream CudaResMgr::GetStream(size_t idx) {
  lock_guard<mutex> lock_ctx(CudaResMgr::gStrMutex);

  if (idx >= GetNumGpus()) {
    return nullptr;
  }

  auto& str = g_Streams[idx];
  if (!str) {
    auto ctx = GetCtx(idx);
    CudaCtxPush push(ctx);
    ThrowOnCudaError(cuStreamCreate(&str, CU_STREAM_NON_BLOCKING), __LINE__);
  }

  return g_Streams[idx];
}

CudaResMgr::~CudaResMgr() {
  lock_guard<mutex> ins_lock(CudaResMgr::gInsMutex);
  lock_guard<mutex> ctx_lock(CudaResMgr::gCtxMutex);
  lock_guard<mutex> str_lock(CudaResMgr::gStrMutex);

  stringstream ss;
  try {
    {
      for (auto& cuStream : g_Streams) {
        if (cuStream) {
          cuStreamDestroy(cuStream); // Avoiding CUDA_ERROR_DEINITIALIZED while
                                     // destructing.
        }
      }
      g_Streams.clear();
    }

    {
      for (int i = 0; i < g_Contexts.size(); i++) {
        if (g_Contexts[i].second) {
          cuDevicePrimaryCtxRelease(
              g_Contexts[i].first); // Avoiding CUDA_ERROR_DEINITIALIZED while
                                    // destructing.
        }
      }
      g_Contexts.clear();
    }
  } catch (runtime_error& e) {
    cerr << e.what() << endl;
  }

#ifdef TRACK_TOKEN_ALLOCATIONS
  cout << "Checking token allocation counters: ";
  auto res = CheckAllocationCounters();
  cout << (res ? "No leaks dectected" : "Leaks detected") << endl;
#endif
}

CudaResMgr& CudaResMgr::Instance() {
  static CudaResMgr instance;
  return instance;
}

size_t CudaResMgr::GetNumGpus() { return Instance().g_Contexts.size(); }

mutex CudaResMgr::gInsMutex;
mutex CudaResMgr::gCtxMutex;
mutex CudaResMgr::gStrMutex;

auto CopyBuffer_Ctx_Str = [](shared_ptr<CudaBuffer> dst,
                             shared_ptr<CudaBuffer> src, CUcontext cudaCtx,
                             CUstream cudaStream) {
  if (dst->GetRawMemSize() != src->GetRawMemSize()) {
    throw runtime_error("Can't copy: buffers have different size.");
  }

  CudaCtxPush ctxPush(cudaCtx);
  ThrowOnCudaError(cuMemcpyDtoDAsync(dst->GpuMem(), src->GpuMem(),
                                     src->GetRawMemSize(), cudaStream),
                   __LINE__);
  ThrowOnCudaError(cuStreamSynchronize(cudaStream), __LINE__);
};

auto CopyBuffer = [](shared_ptr<CudaBuffer> dst, shared_ptr<CudaBuffer> src,
                     int gpuID) {
  auto ctx = CudaResMgr::Instance().GetCtx(gpuID);
  auto str = CudaResMgr::Instance().GetStream(gpuID);
  return CopyBuffer_Ctx_Str(dst, src, ctx, str);
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

void Init_PyFrameUploader(py::module&);

void Init_PySurfaceConverter(py::module&);

void Init_PySurfaceDownloader(py::module&);

void Init_PySurfaceResizer(py::module&);

void Init_PyDecoder(py::module&);

void Init_PyNvEncoder(py::module&);

void Init_PySurface(py::module&);

void Init_PyFrameConverter(py::module&);

PYBIND11_MODULE(_PyNvCodec, m) {

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

  py::register_exception<HwResetException>(m, "HwResetException");

  py::register_exception<CuvidParserException>(m, "CuvidParserException");

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
      .value("FAIL", TaskExecInfo::FAIL)
      .value("SUCCESS", TaskExecInfo::SUCCESS)
      .value("END_OF_STREAM", TaskExecInfo::END_OF_STREAM)
      .value("INVALID_INPUT", TaskExecInfo::INVALID_INPUT)
      .value("MORE_DATA_NEEDED", TaskExecInfo::MORE_DATA_NEEDED)
      .value("BIT_DEPTH_NOT_SUPPORTED", TaskExecInfo::BIT_DEPTH_NOT_SUPPORTED)
      .value("UNSUPPORTED_FMT_CONV_PARAMS",
             TaskExecInfo::UNSUPPORTED_FMT_CONV_PARAMS)
      .value("RES_CHANGE", TaskExecInfo::RES_CHANGE)
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

  py::enum_<cudaVideoCodec>(m, "CudaVideoCodec")
      .value("H264", cudaVideoCodec::cudaVideoCodec_H264)
      .value("HEVC", cudaVideoCodec::cudaVideoCodec_HEVC)
      .value("VP9", cudaVideoCodec::cudaVideoCodec_VP9)
      .value("MJPEG", cudaVideoCodec::cudaVideoCodec_JPEG)
      .value("MPEG1", cudaVideoCodec::cudaVideoCodec_MPEG1)
      .value("MPEG2", cudaVideoCodec::cudaVideoCodec_MPEG2)
      .value("MPEG4", cudaVideoCodec::cudaVideoCodec_MPEG4)
      .value("VC1", cudaVideoCodec::cudaVideoCodec_VC1)
      .value("AV1", cudaVideoCodec::cudaVideoCodec_AV1)
      .export_values();

  py::enum_<SeekMode>(m, "SeekMode")
      .value("EXACT_FRAME", SeekMode::EXACT_FRAME,
             "Use this to seek for exac frame. Notice that if it's P or B "
             "frame, decoder may not be able to get it unless it reconstructs "
             "all the frames that desired frame use for reference.")
      .value("PREV_KEY_FRAME", SeekMode::PREV_KEY_FRAME,
             "Seek for closes key frame in past.")
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
      .def(py::init<int64_t, SeekMode>(), py::arg("seek_frame"),
           py::arg("mode"),
           R"pbdoc(
        Constructor method.

        :param seek_frame: number of frame to seek for, starts from 0
        :param mode: seek to exact frame number or to closest previous key frame
    )pbdoc")
      .def(py::init<double>(), py::arg("seek_ts"),
           R"pbdoc(
        Constructor method.
        Will initialize context for seek by frame timestamp.

        :param seek_frame: timestamp (s) of frame to seek for.
    )pbdoc")
      .def(py::init<double, SeekMode>(), py::arg("seek_ts"), py::arg("mode"),
           R"pbdoc(
        Constructor method.

        :param seek_frame: timestamp (s) of frame to seek for.
        :param mode: seek to exact frame number or to closest previous key frame
    )pbdoc")
      .def_readwrite("seek_frame", &SeekContext::seek_frame,
                     R"pbdoc(
        Number of frame we want to seek.
    )pbdoc")
      .def_readwrite("seek_tssec", &SeekContext::seek_tssec,
                     R"pbdoc(
        Timestamp we want to seek.
    )pbdoc")
      .def_readwrite("mode", &SeekContext::mode,
                     R"pbdoc(
        Seek mode: by frame number or timestamp
    )pbdoc")
      .def_readwrite("out_frame_pts", &SeekContext::out_frame_pts,
                     R"pbdoc(
        PTS of frame decoded after seek.
    )pbdoc")
      .def_readonly("num_frames_decoded", &SeekContext::num_frames_decoded,
                    R"pbdoc(
        Number of frames, decoded if seek was done to closest previous key frame.
    )pbdoc");

  py::class_<PacketData, shared_ptr<PacketData>>(
      m, "PacketData", "Incapsulates information about compressed video frame")
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

  py::class_<TaskExecDetails, shared_ptr<TaskExecDetails>>(m, "TaskExecDetails")
      .def(py::init<>())
      .def_readwrite("info", &TaskExecDetails::info);

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
      .def("GetRawMemSize", &CudaBuffer::GetRawMemSize,
           R"pbdoc(
        Get size of buffer in bytes.

        :rtype: Int
    )pbdoc")
      .def("GetNumElems", &CudaBuffer::GetNumElems,
           R"pbdoc(
        Get number of elements in buffer.

        :rtype: Int
    )pbdoc")
      .def("GetElemSize", &CudaBuffer::GetElemSize,
           R"pbdoc(
        Get size of single element in bytes

        :rtype: Int
    )pbdoc")
      .def("GpuMem", &CudaBuffer::GpuMem,
           R"pbdoc(
        Get CUdeviceptr of memory allocation.

        :rtype: Int
    )pbdoc")
      .def("Clone", &CudaBuffer::Clone, py::return_value_policy::take_ownership,
           R"pbdoc(
        Deep copy = CUDA mem alloc + CUDA mem copy.

        :rtype: PyNvCodec.CudaBuffer
    )pbdoc")
      .def(
          "CopyFrom",
          [](shared_ptr<CudaBuffer> self, shared_ptr<CudaBuffer> other,
             size_t ctx, size_t str) {
            CopyBuffer_Ctx_Str(self, other, (CUcontext)ctx, (CUstream)str);
          },
          py::arg("other"), py::arg("context"), py::arg("stream"),
          R"pbdoc(
        Copy content of another CudaBuffer into this CudaBuffer

        :param other: other CudaBuffer
        :param context: CUDA context to use
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
        :rtype: PyNvCodec.CudaBuffer
    )pbdoc");

  Init_PyDecoder(m);

  Init_PyNvEncoder(m);

  Init_PyFrameUploader(m);

  Init_PySurfaceDownloader(m);

  Init_PySurfaceConverter(m);

  Init_PySurfaceResizer(m);

  Init_PySurface(m);

  Init_PyFrameConverter(m);

  m.def("GetNumGpus", &CudaResMgr::GetNumGpus, R"pbdoc(
        Get number of available GPUs.
    )pbdoc");

  m.def("GetNvencParams", &GetNvencInitParams, R"pbdoc(
        Get list of params PyNvEncoder can be initialized with.
    )pbdoc");

  m.doc() = R"pbdoc(
        PyNvCodec
        ----------
        .. currentmodule:: PyNvCodec
        .. autosummary::
           :toctree: _generate

           GetNumGpus
           GetNvencParams
           PySurfaceResizer
           PySurfaceRemaper
           PySurfaceDownloader
           PySurfaceConverter
           PyNvEncoder
           PyNvDecoder
           PyFrameUploader
           PyFFmpegDemuxer
           PyDecoder
           PyCudaBufferDownloader
           PyBufferUploader
           SeekContext
           CudaBuffer
           SurfacePlane
           Surface

    )pbdoc";
}
