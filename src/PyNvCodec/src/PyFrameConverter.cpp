/*
 *Copyright 2024 Vision Labs LLC
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

using namespace VPF;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyFrameConverter::PyFrameConverter(uint32_t width, uint32_t height,
                                   Pixel_Format inFormat,
                                   Pixel_Format outFormat) {}

bool PyFrameConverter::Execute(
    py::array& src, py::array& dst,
    std::shared_ptr<ColorspaceConversionContext> context,
    TaskExecDetails& details) {
  return true;
}

void Init_PyFrameConverter(py::module&) {
  py::class_<PyFrameConverter, std::shared_ptr<PyFrameConverter>>(
      m, "PyFrameConverter",
      "CUDA-accelerated converter between different pixel formats.")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format>(),
           py::arg("width"), py::arg("height"), py::arg("src_format"),
           py::arg("dst_format"),
           R"pbdoc(
        Constructor method.

        :param width: target Surface width
        :param height: target Surface height
        :param src_format: input Surface pixel format
        :param dst_format: output Surface pixel format
    )pbdoc")
      .def("Format", &PyFrameConverter::GetFormat, R"pbdoc(
        Get pixel format.
    )pbdoc")
      .def(
          "Execute",
          [](std::shared_ptr<PyFrameConverter> self, py::array& src,
             py::array& dst,
             std::shared_ptr<ColorspaceConversionContext> cc_ctx) {
            TaskExecDetails details;
            return std::make_tuple(self->Execute(src, cc_ctx, details),
                                   details.info);
          },
          py::arg("src"), py::arg("dst"), py::arg("cc_ctx"),
          py::return_value_policy::take_ownership,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Perform pixel format conversion.

        :param src: input numpy ndarray.
        :param dst: output numpy ndarray.
        :param cc_ctx: colorspace conversion context. Describes color space and color range used for conversion.
        :return: True in case of 
        :rtype: PyNvCodec.Surface
    )pbdoc");
}