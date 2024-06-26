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

PyFfmpegDecoder::PyFfmpegDecoder(const string& pathToFile,
                                 const map<string, string>& ffmpeg_options,
                                 uint32_t gpuID) {
  gpu_id = gpuID;
  NvDecoderClInterface cli_iface(ffmpeg_options);
  upDecoder.reset(FfmpegDecodeFrame::Make(pathToFile.c_str(), cli_iface));
}

bool PyFfmpegDecoder::DecodeImpl(DecodeContext& ctx, TaskExecDetails& details) {
  UpdateState();

  auto ret = upDecoder->Execute();
  upDecoder->GetExecDetails(details);
  ctx.SetOutPacketData(upDecoder->GetLastPacketData());

  return (TASK_EXEC_SUCCESS == ret);
}

bool PyFfmpegDecoder::DecodeSingleFrame(DecodeContext& ctx, py::array& frame,
                                        TaskExecDetails& details) {
  if (DecodeImpl(ctx, details)) {
    auto pRawFrame = (Buffer*)upDecoder->GetOutput(0U);
    if (pRawFrame) {
      auto const frame_size = pRawFrame->GetRawMemSize();
      if (frame_size != frame.nbytes()) {
        frame.resize({frame_size}, false);
      }

      memcpy(frame.mutable_data(), pRawFrame->GetRawMemPtr(), frame_size);
      return true;
    }
  }

  return false;
}

void* PyFfmpegDecoder::GetSideData(AVFrameSideDataType data_type,
                                   size_t& raw_size) {
  if (TASK_EXEC_SUCCESS == upDecoder->GetSideData(data_type)) {
    auto pSideData = (Buffer*)upDecoder->GetOutput(1U);
    if (pSideData) {
      raw_size = pSideData->GetRawMemSize();
      return pSideData->GetDataAs<void>();
    }
  }
  return nullptr;
}

void PyFfmpegDecoder::UpdateState() {
  last_h = Height();
  last_w = Width();
}

bool PyFfmpegDecoder::IsResolutionChanged() {
  if (last_h != Height()) {
    return true;
  }

  if (last_w != Width()) {
    return true;
  }

  return false;
}

std::vector<MotionVector> PyFfmpegDecoder::GetMotionVectors() {
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

uint32_t PyFfmpegDecoder::Width() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.width;
};

uint32_t PyFfmpegDecoder::Height() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.height;
};

double PyFfmpegDecoder::Framerate() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.frameRate;
};

ColorSpace PyFfmpegDecoder::Color_Space() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.color_space;
};

ColorRange PyFfmpegDecoder::Color_Range() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.color_range;
};

cudaVideoCodec PyFfmpegDecoder::Codec() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.codec;
};

double PyFfmpegDecoder::AvgFramerate() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.avgFrameRate;
};

double PyFfmpegDecoder::Timebase() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.timeBase;
};

uint32_t PyFfmpegDecoder::Numframes() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.num_frames;
};

Pixel_Format PyFfmpegDecoder::PixelFormat() const {
  MuxingParams params;
  upDecoder->GetParams(params);
  return params.videoContext.format;
};

void Init_PyFFMpegDecoder(py::module& m) {
  py::class_<PyFfmpegDecoder, shared_ptr<PyFfmpegDecoder>>(
      m, "PyFfmpegDecoder",
      "Fallback decoder implementation which relies on FFMpeg libavcodec.")
      .def(py::init<const string&, const map<string, string>&, uint32_t>(),
           py::arg("input"), py::arg("opts"), py::arg("gpu_id") = 0,
           R"pbdoc(
        Constructor method.

        :param input: path to input file
        :param opts: AVDictionary options that will be passed to AVFormat context.
    )pbdoc")
      .def(
          "DecodeSingleFrame",
          [](shared_ptr<PyFfmpegDecoder> self, py::array& frame) {
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
          "DecodeSingleFrame",
          [](shared_ptr<PyFfmpegDecoder> self, py::array& frame,
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
      .def("Codec", &PyFfmpegDecoder::Codec,
           R"pbdoc(
        Return video codec used in encoded video stream.
    )pbdoc")
      .def("Width", &PyFfmpegDecoder::Width,
           R"pbdoc(
        Return encoded video file width in pixels.
    )pbdoc")
      .def("Height", &PyFfmpegDecoder::Height,
           R"pbdoc(
        Return encoded video file height in pixels.
    )pbdoc")
      .def("Framerate", &PyFfmpegDecoder::Framerate,
           R"pbdoc(
        Return encoded video file framerate.
    )pbdoc")
      .def("AvgFramerate", &PyFfmpegDecoder::AvgFramerate,
           R"pbdoc(
        Return encoded video file average framerate.
    )pbdoc")
      .def("Timebase", &PyFfmpegDecoder::Timebase,
           R"pbdoc(
        Return encoded video file time base.
    )pbdoc")
      .def("Numframes", &PyFfmpegDecoder::Numframes,
           R"pbdoc(
        Return number of video frames in encoded video file.
        Please note that some video containers doesn't store this infomation.
    )pbdoc")
      .def("ColorSpace", &PyFfmpegDecoder::Color_Space,
           R"pbdoc(
        Get color space information stored in video file.
        Please not that some video containers may not store this information.

        :return: color space information
    )pbdoc")
      .def("ColorRange", &PyFfmpegDecoder::Color_Range,
           R"pbdoc(
        Get color range information stored in video file.
        Please not that some video containers may not store this information.

        :return: color range information
    )pbdoc")
      .def("Format", &PyFfmpegDecoder::PixelFormat,
           R"pbdoc(
        Return encoded video file pixel format.
    )pbdoc")
      .def("GetMotionVectors", &PyFfmpegDecoder::GetMotionVectors,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Return motion vectors of last decoded frame.
        If there are no movion vectors it will return empty list.

       :return: list of motion vectors
       :rtype: List[nvc.MotionVector]
    )pbdoc");
}
