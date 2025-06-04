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
         Constructor for PySurfaceConverter with GPU ID.

         Creates a new instance of PySurfaceConverter that will run on the specified GPU.
         The CUDA stream will be automatically created and managed.

         :param gpu_id: The ID of the GPU to use for pixel format conversion
         :type gpu_id: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Constructor for PySurfaceConverter with GPU ID and CUDA stream.

         Creates a new instance of PySurfaceConverter that will run on the specified GPU
         using the provided CUDA stream.

         :param gpu_id: The ID of the GPU to use for pixel format conversion
         :type gpu_id: int
         :param stream: The CUDA stream to use for conversion
         :type stream: int
         :raises RuntimeError: If the specified GPU is not available
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
         Perform pixel format conversion synchronously.

         Converts the input surface to the specified output format.
         This method blocks until the conversion is complete.
         The input surface must be in a format supported by the converter.
         The output surface must be in a format that can be converted to from the input format.

         :param src: Input surface to be converted
         :type src: Surface
         :param dst: Output surface to store the converted result
         :type dst: Surface
         :param cc_ctx: Optional colorspace conversion context that describes the color space
             and color range to use for conversion. If not provided, VALI will automatically
             select supported color conversion parameters.
         :type cc_ctx: ColorspaceConversionContext, optional
         :return: Tuple containing:
             - success (bool): True if conversion was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the conversion fails or if the formats are not compatible
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
         Perform pixel format conversion asynchronously.

         Converts the input surface to the specified output format.
         This method returns immediately without waiting for the conversion to complete.
         The input surface must be in a format supported by the converter.
         The output surface must be in a format that can be converted to from the input format.

         :param src: Input surface to be converted
         :type src: Surface
         :param dst: Output surface to store the converted result
         :type dst: Surface
         :param cc_ctx: Optional colorspace conversion context that describes the color space
             and color range to use for conversion. If not provided, VALI will automatically
             select supported color conversion parameters.
         :type cc_ctx: ColorspaceConversionContext, optional
         :return: Tuple containing:
             - success (bool): True if conversion was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the conversion fails or if the formats are not compatible
     )pbdoc")
      .def_static("Conversions", &PySurfaceConverter::GetConversions,
                  R"pbdoc(
         Get list of supported pixel format conversions.

         Returns a list of tuples containing supported input and output pixel format pairs
         that can be processed by the converter.

         :return: List of tuples containing supported (input_format, output_format) pairs
         :rtype: list[tuple[Pixel_Format, Pixel_Format]]
     )pbdoc")
      .def_property_readonly(
          "Stream",
          [](PySurfaceConverter& self) { return (size_t)self.m_stream; },
          R"pbdoc(
         Get the CUDA stream associated with this instance.

         Returns the handle to the CUDA stream used for conversion processing.

         :return: CUDA stream handle
         :rtype: int
     )pbdoc");
}