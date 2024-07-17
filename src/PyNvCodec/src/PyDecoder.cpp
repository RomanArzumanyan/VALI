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

bool PyDecoder::DecodeImpl(DecodeContext& ctx, TaskExecDetails& details,
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
  ctx.SetOutPacketData(upDecoder->GetLastPacketData());

  UpdateState();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

bool PyDecoder::DecodeSingleFrame(DecodeContext& ctx, py::array& frame,
                                  TaskExecDetails& details,
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
  return DecodeImpl(ctx, details, *dst.get(), seek_ctx);
}

bool PyDecoder::DecodeSingleSurface(DecodeContext& ctx, Surface& surf,
                                    TaskExecDetails& details,
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

  return DecodeImpl(ctx, details, surf, seek_ctx);
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

double PyDecoder::Framerate() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.frameRate;
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
  return params.videoContext.avgFrameRate;
};

double PyDecoder::Timebase() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.timeBase;
};

uint32_t PyDecoder::Numframes() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.num_frames;
};

uint32_t PyDecoder::HostFrameSize() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.host_frame_size;
};

Pixel_Format PyDecoder::PixelFormat() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.format;
};

bool PyDecoder::IsAccelerated() const { return upDecoder->IsAccelerated(); }

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
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyDecoder> self, py::array& frame,
             std::optional<PacketData>& pkt_data,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;
            PacketData data;

            DecodeContext ctx(nullptr, nullptr, nullptr,
                              pkt_data ? &pkt_data.value() : &data, nullptr,
                              false);

            return std::make_tuple(
                self->DecodeSingleFrame(ctx, frame, details, seek_ctx),
                details.m_info);
          },
          py::arg("frame"), py::arg("pkt_data") = std::nullopt,
          py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input file.

        :param frame: decoded video frame
        :param pkt_data: decoded video frame packet data, may be None
        :param seek_ctx: seek context, may be None
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyDecoder> self, Surface& surf,
             std::optional<PacketData>& pkt_data,
             std::optional<SeekContext>& seek_ctx) {
            TaskExecDetails details;
            PacketData data;

            DecodeContext ctx(nullptr, nullptr, nullptr,
                              pkt_data ? &pkt_data.value() : &data, nullptr,
                              false);

            return std::make_tuple(
                self->DecodeSingleSurface(ctx, surf, details, seek_ctx),
                details.m_info);
          },
          py::arg("surf"), py::arg("pkt_data") = std::nullopt,
          py::arg("seek_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video surface from input file.

        :param surf: decoded video surface
        :param pkt_data: decoded video surface packet data, may be None
        :param seek_ctx: seek context, may be None
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def("Width", &PyDecoder::Width,
           R"pbdoc(
        Return encoded video file width in pixels.
    )pbdoc")
      .def("Height", &PyDecoder::Height,
           R"pbdoc(
        Return encoded video file height in pixels.
    )pbdoc")
      .def("Framerate", &PyDecoder::Framerate,
           R"pbdoc(
        Return encoded video file framerate.
    )pbdoc")
      .def("AvgFramerate", &PyDecoder::AvgFramerate,
           R"pbdoc(
        Return encoded video file average framerate.
    )pbdoc")
      .def("Timebase", &PyDecoder::Timebase,
           R"pbdoc(
        Return encoded video file time base.
    )pbdoc")
      .def("Numframes", &PyDecoder::Numframes,
           R"pbdoc(
        Return number of video frames in encoded video file.
        Please note that some video containers doesn't store this infomation.
    )pbdoc")
      .def("ColorSpace", &PyDecoder::Color_Space,
           R"pbdoc(
        Get color space information stored in video file.
        Please not that some video containers may not store this information.

        :return: color space information
    )pbdoc")
      .def("ColorRange", &PyDecoder::Color_Range,
           R"pbdoc(
        Get color range information stored in video file.
        Please not that some video containers may not store this information.

        :return: color range information
    )pbdoc")
      .def("Format", &PyDecoder::PixelFormat,
           R"pbdoc(
        Return encoded video file pixel format.
    )pbdoc")
      .def("HostFrameSize", &PyDecoder::HostFrameSize,
           R"pbdoc(
        Return amount of bytes needed to store decoded frame.
    )pbdoc")
      .def("Accelerated", &PyDecoder::IsAccelerated,
           R"pbdoc(
        Return true if decoder has HW acceleration support, false otherwise.
    )pbdoc")
      .def("GetMotionVectors", &PyDecoder::GetMotionVectors,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Return motion vectors of last decoded frame.
        If there are no movion vectors it will return empty list.

       :return: list of motion vectors
       :rtype: List[nvc.MotionVector]
    )pbdoc");

  m.attr("NO_PTS") = py::int_(AV_NOPTS_VALUE);
}
