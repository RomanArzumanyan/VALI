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

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceResizer::PySurfaceResizer(Pixel_Format format, int gpu_id)
    : PySurfaceResizer(format, gpu_id,
                       CudaResMgr::Instance().GetStream(gpu_id)) {}

PySurfaceResizer::PySurfaceResizer(Pixel_Format format, int gpu_id,
                                   CUstream str) {
  m_stream = str;
  upResizer = std::make_unique<ResizeSurface>(format, gpu_id, m_stream);
  m_event = std::make_shared<CudaStreamEvent>(m_stream, gpu_id);
}

bool PySurfaceResizer::Run(Surface& src, Surface& dst,
                           TaskExecDetails& details) {
  upResizer->SetInput(&src, 0U);
  upResizer->SetInput(&dst, 1U);

  details = upResizer->Execute();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

void Init_PySurfaceResizer(py::module& m) {
  py::class_<PySurfaceResizer>(m, "PySurfaceResizer",
                               "CUDA-accelerated Surface resizer.")
      .def(py::init<Pixel_Format, int>(), py::arg("format"), py::arg("gpu_id"),
           R"pbdoc(
         Constructor for PySurfaceResizer with format and GPU ID.

         Creates a new instance of PySurfaceResizer that will run on the specified GPU
         with the given pixel format. The CUDA stream will be automatically created and managed.

         :param format: The pixel format to use for resizing operations
         :type format: Pixel_Format
         :param gpu_id: The ID of the GPU to use for resizing
         :type gpu_id: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def(py::init<Pixel_Format, int, size_t>(), py::arg("format"),
           py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Constructor for PySurfaceResizer with format, GPU ID, and CUDA stream.

         Creates a new instance of PySurfaceResizer that will run on the specified GPU
         with the given pixel format using the provided CUDA stream.

         :param format: The pixel format to use for resizing operations
         :type format: Pixel_Format
         :param gpu_id: The ID of the GPU to use for resizing
         :type gpu_id: int
         :param stream: The CUDA stream to use for resizing
         :type stream: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def(
          "Run",
          [](PySurfaceResizer& self, Surface& src, Surface& dst) {
            TaskExecDetails details;
            auto res = self.Run(src, dst, details);
            self.m_event->Record();
            self.m_event->Wait();
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Resize input Surface synchronously.

         Processes the input surface and stores the resized result in the output surface.
         This method blocks until the resize operation is complete.
         Both input and output surfaces must use the same pixel format that was
         specified when creating the resizer instance.

         :param src: Input surface to be resized
         :type src: Surface
         :param dst: Output surface to store the resized result
         :type dst: Surface
         :return: Tuple containing:
             - success (bool): True if resize was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the resize operation fails or if surface formats don't match
     )pbdoc")
      .def(
          "RunAsync",
          [](PySurfaceResizer& self, Surface& src, Surface& dst) {
            TaskExecDetails details;
            auto res = self.Run(src, dst, details);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Resize input Surface asynchronously.

         Processes the input surface and stores the resized result in the output surface.
         This method returns immediately without waiting for the resize operation to complete.
         Both input and output surfaces must use the same pixel format that was
         specified when creating the resizer instance.

         :param src: Input surface to be resized
         :type src: Surface
         :param dst: Output surface to store the resized result
         :type dst: Surface
         :return: Tuple containing:
             - success (bool): True if resize was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the resize operation fails or if surface formats don't match
     )pbdoc")
      .def_property_readonly(
          "Stream",
          [](PySurfaceResizer& self) { return (size_t)self.m_stream; },
          R"pbdoc(
         Get the CUDA stream associated with this instance.

         Returns the handle to the CUDA stream used for resize processing.

         :return: CUDA stream handle
         :rtype: int
     )pbdoc");
}