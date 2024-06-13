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
                     const map<string, string>& ffmpeg_options,
                     int gpuID) {
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);
  auto stream =
      gpu_id >= 0
          ? std::optional<CUstream>(CudaResMgr::Instance().GetStream(gpu_id))
          : std::nullopt;

  upDecoder.reset(DecodeFrame::Make(pathToFile.c_str(), cli_iface, stream));
}

bool PyDecoder::DecodeImpl(DecodeContext& ctx, TaskExecDetails& details,
                           Token& dst) {
  UpdateState();

  upDecoder->SetInput(&dst, 0U);
  auto ret = upDecoder->Execute();
  upDecoder->GetExecDetails(details);
  ctx.SetOutPacketData(upDecoder->GetLastPacketData());

  return (TASK_EXEC_SUCCESS == ret);
}

bool PyDecoder::DecodeSingleFrame(DecodeContext& ctx, py::array& frame,
                                  TaskExecDetails& details) {
  if (IsAccelerated()) {
    return false;
  }

  auto const frame_size = upDecoder->GetHostFrameSize();
  if (frame_size != frame.nbytes()) {
    frame.resize({frame_size}, false);
  }

  auto dst = std::shared_ptr<Buffer>(
      Buffer::Make(frame.nbytes(), frame.mutable_data()));
  return DecodeImpl(ctx, details, *dst.get());
}

bool PyDecoder::DecodeSingleSurface(DecodeContext& ctx, Surface& surf,
                                    TaskExecDetails& details) {
  if (!IsAccelerated()) {
    return false;
  }

  if (surf.Empty() || surf.Width() != Width() || surf.Height() != Height() ||
      surf.PixelFormat() != PixelFormat()) {
    return false;
  }
  return DecodeImpl(ctx, details, surf);
}

void* PyDecoder::GetSideData(AVFrameSideDataType data_type, size_t& raw_size) {
  if (TASK_EXEC_SUCCESS == upDecoder->GetSideData(data_type)) {
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

bool PyDecoder::IsResolutionChanged() {
  if (last_h != Height()) {
    return true;
  }

  if (last_w != Width()) {
    return true;
  }

  return false;
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

cudaVideoCodec PyDecoder::Codec() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.codec;
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
          [](shared_ptr<PyDecoder> self, py::array& frame) {
            TaskExecDetails details;
            PacketData pktData;
            DecodeContext ctx(nullptr, nullptr, nullptr, &pktData, nullptr,
                              false);
            auto res = self->DecodeSingleFrame(ctx, frame, details);
            return std::make_tuple(res, details.info);
          },
          py::arg("frame"), py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input file.

        :param frame: decoded video frame
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyDecoder> self, Surface& surf) {
            TaskExecDetails details;
            PacketData pktData;
            DecodeContext ctx(nullptr, nullptr, nullptr, &pktData, nullptr,
                              false);
            auto res = self->DecodeSingleSurface(ctx, surf, details);
            return std::make_tuple(res, details.info);
          },
          py::arg("surf"), py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video surface from input file.

        :param surf: decoded video surface
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyDecoder> self, py::array& frame,
             PacketData& pktData) {
            TaskExecDetails details;
            DecodeContext ctx(nullptr, nullptr, nullptr, &pktData, nullptr,
                              false);
            auto res = self->DecodeSingleFrame(ctx, frame, details);
            return std::make_tuple(res, details.info);
          },
          py::arg("frame"), py::arg("pktData"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video frame from input file.

        :param frame: decoded video frame
        :param pktData: decoded video frame packet data
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def(
          "DecodeSingleSurface",
          [](shared_ptr<PyDecoder> self, Surface& surf, PacketData& pktData) {
            TaskExecDetails details;
            DecodeContext ctx(nullptr, nullptr, nullptr, &pktData, nullptr,
                              false);
            auto res = self->DecodeSingleSurface(ctx, surf, details);
            return std::make_tuple(res, details.info);
          },
          py::arg("surf"), py::arg("pktData"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Decode single video surface from input file.

        :param surf: decoded video surface
        :param pktData: decoded video surface packet data
        :return: tuple, first element is True in case of success, False otherwise. Second elements is TaskExecInfo.
    )pbdoc")
      .def("Codec", &PyDecoder::Codec,
           R"pbdoc(
        Return video codec used in encoded video stream.
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
}
