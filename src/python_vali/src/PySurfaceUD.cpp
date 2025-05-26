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
         Constructor method.
 
         :param gpu_id: what GPU to run on.
     )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Constructor method.
 
         :param gpu_id: what GPU to run on.
         :param stream: CUDA stream to use.
     )pbdoc")
      .def_static("SupportedFormats", &PySurfaceUD::SupportedFormats,
                  py::call_guard<py::gil_scoped_release>(), R"pbdoc(
         Get list of supported pixel formats.
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
         Convert input Surface.
 
         :param src: input Surface.
         :param dst: output Surface.
         :return: tuple containing:
           success (Bool) True in case of success, False otherwise.
           info (TaskExecInfo) task execution information.
         :rtype: tuple
     )pbdoc")
      .def(
          "RunAsync",
          [](PySurfaceUD& self, Surface& src, Surface& dst, bool record_event) {
            TaskExecDetails details;
            auto res = self.Run(src, dst, details);
            if (record_event) {
              self.m_event->Record();
            }
            return std::make_tuple(res, details.m_info,
                                   record_event ? self.m_event : nullptr);
          },
          py::arg("src"), py::arg("dst"), py::arg("record_event") = true,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Convert input Surface.

         :param src: input Surface.
         :param dst: output Surface.
         :param record_event: If False, no event will be recorded. Useful for chain calls.
         :return: tuple containing:
           success (Bool) True in case of success, False otherwise.
           info (TaskExecInfo) task execution information.
           event (CudaStreamEvent) CUDA stream event.
         :rtype: tuple
     )pbdoc");
}