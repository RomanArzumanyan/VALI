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

using namespace VPF;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceConverter::PySurfaceConverter(int gpu_id)
    : PySurfaceConverter(gpu_id, CudaResMgr::Instance().GetStream(gpu_id)) {}

PySurfaceConverter::PySurfaceConverter(int gpu_id, CUstream str) {
  m_stream = str;
  upConverter = std::make_unique<ConvertSurface>(gpu_id, m_stream);
  m_event = std::make_shared<CudaStreamEvent>(m_stream, gpu_id);
}

bool PySurfaceConverter::Run(Surface& src, Surface& dst,
                             std::optional<ColorspaceConversionContext> context,
                             TaskExecDetails& details) {
  details = upConverter->Run(src, dst, context);
  return (TASK_EXEC_SUCCESS == details.m_status);
}

std::list<std::pair<Pixel_Format, Pixel_Format>>
PySurfaceConverter::GetConversions() {
  return ConvertSurface::GetSupportedConversions();
}

void Init_PySurfaceConverter(py::module& m) {
  py::class_<PySurfaceConverter>(
      m, "PySurfaceConverter",
      "CUDA-accelerated converter between different pixel formats.")
      .def(py::init<int>(), py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param gpu_id: what GPU to run conversion on
    )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param gpu_id: what GPU to run conversion on
        :param stream: CUDA stream to use for conversion
    )pbdoc")
      .def(
          "Run",
          [](PySurfaceConverter& self, Surface& src, Surface& dst,
             std::optional<ColorspaceConversionContext> cc_ctx) {
            TaskExecDetails details;
            auto res = self.Run(src, dst, cc_ctx, details);
            self.m_event->Record();
            self.m_event->Wait();
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"), py::arg("cc_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Perform pixel format conversion.

        :param src: input Surface. Must be of same format class instance was created with.
        :param dst: output Surface. Must be of suitable format.
        :param cc_ctx: colorspace conversion context. Describes color space and color range used for conversion. Optional parameter. If not given, VALI will automatically pick supported color conversion parameters.
        :return: tuple:
          success (Bool) True in case of success, False otherwise.
          info (TaskExecInfo) task execution information.
        :rtype: tuple
    )pbdoc")
      .def(
          "RunAsync",
          [](PySurfaceConverter& self, Surface& src, Surface& dst,
             std::optional<ColorspaceConversionContext> cc_ctx) {
            TaskExecDetails details;
            auto res = self.Run(src, dst, cc_ctx, details);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"), py::arg("cc_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Perform pixel format conversion.

        :param src: input Surface. Must be of same format class instance was created with.
        :param dst: output Surface. Must be of suitable format.
        :param cc_ctx: colorspace conversion context. Describes color space and color range used for conversion. Optional parameter. If not given, VALI will automatically pick supported color conversion parameters.
        :return: tuple:
          success (Bool) True in case of success, False otherwise.
          info (TaskExecInfo) task execution information.
        :rtype: tuple
    )pbdoc")
      .def_static("Conversions", &PySurfaceConverter::GetConversions,
                  R"pbdoc(
        Return list of supported color conversions.
    )pbdoc")
      .def_property_readonly(
          "Stream",
          [](PySurfaceConverter& self) { return (size_t)self.m_stream; },
          R"pbdoc(
        Return CUDA stream.
    )pbdoc");
}