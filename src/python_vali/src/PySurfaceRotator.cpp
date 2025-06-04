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
  m_event = std::make_shared<CudaStreamEvent>(m_stream, gpu_id);
}

std::list<Pixel_Format> PySurfaceRotator::SupportedFormats() {
  return std::list<Pixel_Format>({Y, GRAY12, RGB, BGR, RGB_PLANAR, YUV420,
                                  YUV422, YUV444, RGB_32F, RGB_32F_PLANAR,
                                  YUV444_10bit, YUV420_10bit});
}

bool PySurfaceRotator::Run(double angle, double shift_x, double shift_y,
                           Surface& src, Surface& dst,
                           TaskExecDetails& details) {
  double angle_norm = angle;
  double norm_shift_x = shift_x;
  double norm_shift_y = shift_y;

  if ((std::fmod(angle, 90.0) == 0.0) && (shift_x == 0.0) && (shift_y == 0.0)) {
    /* Special case used to deal mostly with display matrix rotation.
     * Get rid of negative angles and calculate shifts to change image
     * orientation from portrait to lanscape and vice versa hassle-free.
     */
    auto norm_angle = std::lround(angle);
    norm_angle = (norm_angle + 360) % 360;

    switch (norm_angle) {
    case 0U:
      angle_norm = 0.0;
      break;
    case 90U:
      angle_norm = 90.0;
      norm_shift_y = src.Width() - 1;
      break;
    case 180U:
      angle_norm = 180.0;
      norm_shift_x = src.Width() - 1;
      norm_shift_y = src.Height() - 1;
      break;
    case 270U:
      angle_norm = 270.0;
      norm_shift_x = src.Height() - 1;
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
         Constructor for PySurfaceRotator with GPU ID.

         Creates a new instance of PySurfaceRotator that will run on the specified GPU.
         The CUDA stream will be automatically created and managed.

         :param gpu_id: The ID of the GPU to use for rotation
         :type gpu_id: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Constructor for PySurfaceRotator with GPU ID and CUDA stream.

         Creates a new instance of PySurfaceRotator that will run on the specified GPU
         using the provided CUDA stream.

         :param gpu_id: The ID of the GPU to use for rotation
         :type gpu_id: int
         :param stream: The CUDA stream to use for rotation
         :type stream: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def_property_readonly("SupportedFormats",
                             &PySurfaceRotator::SupportedFormats,
                             py::call_guard<py::gil_scoped_release>(), R"pbdoc(
         Get list of supported pixel formats for rotation.

         Returns a list of pixel formats that can be processed by the rotator.
         Supported formats include: Y, GRAY12, RGB, BGR, RGB_PLANAR, YUV420,
         YUV422, YUV444, RGB_32F, RGB_32F_PLANAR, YUV444_10bit, YUV420_10bit.

         :return: List of supported pixel formats
         :rtype: list[Pixel_Format]
     )pbdoc")
      .def(
          "Run",
          [](PySurfaceRotator& self, Surface& src, Surface& dst, double angle,
             double shift_x, double shift_y) {
            TaskExecDetails details;
            auto res = self.Run(angle, shift_x, shift_y, src, dst, details);
            self.m_event->Record();
            self.m_event->Wait();
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"), py::arg("angle"),
          py::arg("shift_x") = 0.0, py::arg("shift_y") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Rotate input Surface synchronously.

         Processes the input surface and stores the rotated result in the output surface.
         This method blocks until the rotation is complete.
         For 90-degree rotations with no shift, the method optimizes the operation
         by handling display matrix rotation cases.

         :param src: Input surface to be rotated
         :type src: Surface
         :param dst: Output surface to store the rotated result
         :type dst: Surface
         :param angle: Rotation angle in degrees
         :type angle: float
         :param shift_x: Shift along X axis in pixels (default: 0.0)
         :type shift_x: float
         :param shift_y: Shift along Y axis in pixels (default: 0.0)
         :type shift_y: float
         :return: Tuple containing:
             - success (bool): True if rotation was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the rotation fails
     )pbdoc")
      .def(
          "RunAsync",
          [](PySurfaceRotator& self, Surface& src, Surface& dst, double angle,
             double shift_x, double shift_y) {
            TaskExecDetails details;
            auto res = self.Run(angle, shift_x, shift_y, src, dst, details);
            return std::make_tuple(res, details.m_info);
          },
          py::arg("src"), py::arg("dst"), py::arg("angle"),
          py::arg("shift_x") = 0.0, py::arg("shift_y") = 0.0,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
         Rotate input Surface asynchronously.

         Processes the input surface and stores the rotated result in the output surface.
         This method returns immediately without waiting for the rotation to complete.
         For 90-degree rotations with no shift, the method optimizes the operation
         by handling display matrix rotation cases.

         :param src: Input surface to be rotated
         :type src: Surface
         :param dst: Output surface to store the rotated result
         :type dst: Surface
         :param angle: Rotation angle in degrees
         :type angle: float
         :param shift_x: Shift along X axis in pixels (default: 0.0)
         :type shift_x: float
         :param shift_y: Shift along Y axis in pixels (default: 0.0)
         :type shift_y: float
         :return: Tuple containing:
             - success (bool): True if rotation was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the rotation fails
     )pbdoc")
      .def_property_readonly(
          "Stream",
          [](PySurfaceRotator& self) { return (size_t)self.m_stream; },
          R"pbdoc(
         Get the CUDA stream associated with this instance.

         Returns the handle to the CUDA stream used for rotation processing.

         :return: CUDA stream handle
         :rtype: int
     )pbdoc");
}