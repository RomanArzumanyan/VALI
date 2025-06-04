/*
 * Copyright 2025 Vision Labs LLC
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

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceUD::PySurfaceUD(int gpu_id)
    : PySurfaceUD(gpu_id, CudaResMgr::Instance().GetStream(gpu_id)) {}

PySurfaceUD::PySurfaceUD(int gpu_id, CUstream str) {
  m_stream = str;
  m_ud = std::make_unique<UDSurface>(gpu_id, m_stream);
  m_event = std::make_shared<CudaStreamEvent>(m_stream, gpu_id);
}

std::list<std::pair<Pixel_Format, Pixel_Format>>
PySurfaceUD::SupportedFormats() {
  return std::list<std::pair<Pixel_Format, Pixel_Format>>(
      UDSurface::SupportedConversions());
}

bool PySurfaceUD::Run(Surface& src, Surface& dst, TaskExecDetails& details) {
  details = m_ud->Run(src, dst);
  return (TASK_EXEC_SUCCESS == details.m_status);
}

void Init_PySurfaceUD(py::module& m) {
  py::class_<PySurfaceUD>(m, "PySurfaceUD",
                          "CUDA-accelerated Surface Upsampler-Downscaler")
      .def(py::init<int>(), py::arg("gpu_id"),
           R"pbdoc(
         Constructor for PySurfaceUD with GPU ID.

         Creates a new instance of PySurfaceUD that will run on the specified GPU.
         The CUDA stream will be automatically created and managed.

         :param gpu_id: The ID of the GPU to use for processing
         :type gpu_id: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Constructor for PySurfaceUD with GPU ID and CUDA stream.

         Creates a new instance of PySurfaceUD that will run on the specified GPU
         using the provided CUDA stream.

         :param gpu_id: The ID of the GPU to use for processing
         :type gpu_id: int
         :param stream: The CUDA stream to use for processing
         :type stream: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def_static("SupportedFormats", &PySurfaceUD::SupportedFormats,
                  py::call_guard<py::gil_scoped_release>(), R"pbdoc(
         Get list of supported pixel format conversions.

         Returns a list of tuples containing supported input and output pixel format pairs
         that can be processed by the upsampler-downscaler.

         :return: List of tuples containing supported (input_format, output_format) pairs
         :rtype: list[tuple[Pixel_Format, Pixel_Format]]
     )pbdoc")
      .def(
          "Run",
          [](PySurfaceUD& self, Surface& src, Surface& dst) {
            TaskExecDetails details;
            auto res = self.Run(src, dst, details);
            self.m_event->Record();
            self.m_event->Wait();
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Convert input Surface synchronously.

         Processes the input surface and stores the result in the output surface.
         This method blocks until the conversion is complete.

         :param src: Input surface to be processed
         :type src: Surface
         :param dst: Output surface to store the result
         :type dst: Surface
         :return: Tuple containing:
             - success (bool): True if conversion was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the conversion fails
     )pbdoc")
      .def(
          "RunAsync",
          [](PySurfaceUD& self, Surface& src, Surface& dst) {
            TaskExecDetails details;
            auto res = self.Run(src, dst, details);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Convert input Surface asynchronously.

         Processes the input surface and stores the result in the output surface.
         This method returns immediately without waiting for the conversion to complete.

         :param src: Input surface to be processed
         :type src: Surface
         :param dst: Output surface to store the result
         :type dst: Surface
         :return: Tuple containing:
             - success (bool): True if conversion was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the conversion fails
     )pbdoc")
      .def_property_readonly(
          "Stream", [](PySurfaceUD& self) { return (size_t)self.m_stream; },
          R"pbdoc(
         Get the CUDA stream associated with this instance.

         Returns the handle to the CUDA stream used for processing.

         :return: CUDA stream handle
         :rtype: int
     )pbdoc");
}