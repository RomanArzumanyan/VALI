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

PySurfaceResizer::PySurfaceResizer(Pixel_Format format, CUstream str) {
  upResizer =
      std::make_unique<ResizeSurface>(format, GetContextByStream(str), str);
}

PySurfaceResizer::PySurfaceResizer(Pixel_Format format, uint32_t gpuID) {
  upResizer = std::make_unique<ResizeSurface>(
      format, CudaResMgr::Instance().GetCtx(gpuID),
      CudaResMgr::Instance().GetStream(gpuID));
}

bool PySurfaceResizer::Run(Surface& src, Surface& dst) {
  upResizer->SetInput(&src, 0U);
  upResizer->SetInput(&dst, 1U);

  return (TASK_EXEC_SUCCESS == upResizer->Execute());
}

void Init_PySurfaceResizer(py::module& m) {
  py::class_<PySurfaceResizer>(m, "PySurfaceResizer",
                               "CUDA-accelerated Surface resizer.")
      .def(py::init<Pixel_Format, uint32_t>(), py::arg("format"),
           py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param format: target Surface pixel format
        :param gpu_id: what GPU to run resize on
    )pbdoc")
      .def(py::init<Pixel_Format, size_t>(), py::arg("format"),
           py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param format: target Surface pixel format
        :param stream: CUDA stream to use for resize
    )pbdoc")
      .def("Run", &PySurfaceResizer::Run, py::arg("src"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Resize input Surface.

        :param src: input Surface. Must be of same format class instance was created with.
        :param dst: output Surface. Must be of same format class instance was created with.
        :return: true in case of success, false otherwise.
    )pbdoc");
}