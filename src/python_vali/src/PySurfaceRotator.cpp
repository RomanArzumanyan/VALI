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

PySurfaceRotator::PySurfaceRotator(int gpu_id)
    : PySurfaceRotator(gpu_id, CudaResMgr::Instance().GetStream(gpu_id)) {}

PySurfaceRotator::PySurfaceRotator(int gpu_id, CUstream str) {
  m_stream = str;
  m_rotator = std::make_unique<RotateSurface>(gpu_id, m_stream);
  m_event = std::make_shared<CudaStreamEvent>(m_stream);
}

bool PySurfaceRotator::Run(double angle, double shift_x, double shift_y,
                           Surface& src, Surface& dst,
                           TaskExecDetails& details) {
  double angle_norm = angle;
  double norm_shift_x = shift_x;
  double norm_shift_y = shift_y;

  if ((std::fmod(angle, 90.0) == 0.0) && (shift_x == 0.0) && (shift_y == 0.0)) {
    // Special case used to deal with display matrix
    // Get rid of negative angles
    auto norm_angle = std::lround(angle);
    norm_angle = (norm_angle + 360) % 360;

    switch (norm_angle) {
    case 0U:
      angle_norm = 0.0;
      break;
    case 90U:
      angle_norm = 90.0;
      norm_shift_y = src.Width();
      break;
    case 180U:
      angle_norm = 180.0;
      norm_shift_x = src.Width();
      norm_shift_y = src.Height();
      break;
    case 270U:
      angle_norm = 270.0;
      norm_shift_x = src.Height();
      break;
    }
  }

  details = m_rotator->Run(angle_norm, norm_shift_x, norm_shift_y, src, dst);
  return (TASK_EXEC_SUCCESS == details.m_status);
}

void Init_PySurfaceRotator(py::module& m) {
  py::class_<PySurfaceRotator>(m, "PySurfaceRotator",
                               "CUDA-accelerated Surface rotator.")
      .def(py::init<int>(), py::arg("gpu_id"),
           R"pbdoc(
         Constructor method.
 
         :param gpu_id: what GPU to run rotation on
     )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Constructor method.
 
         :param gpu_id: what GPU to run rotation on
         :param stream: CUDA stream to use for rotation
     )pbdoc")
      .def(
          "Run",
          [](PySurfaceRotator& self, double angle, double shift_x,
             double shift_y, Surface& src, Surface& dst) {
            TaskExecDetails details;
            auto res = self.Run(angle, shift_x, shift_y, src, dst, details);
            self.m_event->Record();
            self.m_event->Wait();
            return std::make_tuple(res, details.m_info);
          },
          py::arg("angle"), py::arg("shift_x") = 0.0, py::arg("shift_y") = 0.0,
          py::arg("src"), py::arg("dst"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Rotate input Surface.
 
         :param angle: rotation angle
         :param shift_x: shift alongside X axis in pixels
         :param shift_y: shift alongside Y axis in pixels
         :param src: input Surface.
         :param dst: output Surface.
         :return: tuple containing:
           success (Bool) True in case of success, False otherwise.
           info (TaskExecInfo) task execution information.
         :rtype: tuple
     )pbdoc")
      .def(
          "RunAsync",
          [](PySurfaceRotator& self, double angle, double shift_x,
             double shift_y, Surface& src, Surface& dst, bool record_event) {
            TaskExecDetails details;
            auto res = self.Run(angle, shift_x, shift_y, src, dst, details);
            if (record_event) {
              self.m_event->Record();
            }
            return std::make_tuple(res, details.m_info,
                                   record_event ? self.m_event : nullptr);
          },
          py::arg("angle"), py::arg("shift_x") = 0.0, py::arg("shift_y") = 0.0,
          py::arg("src"), py::arg("dst"), py::arg("record_event") = true,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Rotate input Surface.
 
         :param angle: rotation angle
         :param shift_x: shift alongside X axis in pixels
         :param shift_y: shift alongside Y axis in pixels
         :param src: input Surface.
         :param dst: output Surface.
         :param record_event: If False, no event will be recorded. Useful for chain calls.
         :return: tuple containing:
           success (Bool) True in case of success, False otherwise.
           info (TaskExecInfo) task execution information.
           event (CudaStreamEvent) CUDA stream event
         :rtype: tuple
     )pbdoc");
}