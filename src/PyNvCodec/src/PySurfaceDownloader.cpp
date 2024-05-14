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

#include "CudaUtils.hpp"
#include "PyNvCodec.hpp"

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, uint32_t gpu_ID)
    : m_format(format) {
  upDownloader.reset(CudaDownloadSurface::Make(
      CudaResMgr::Instance().GetStream(gpu_ID),
      CudaResMgr::Instance().GetCtx(gpu_ID), width, height, format));
}

PySurfaceDownloader::PySurfaceDownloader(uint32_t width, uint32_t height,
                                         Pixel_Format format, CUstream str)
    : m_format(format) {
  upDownloader.reset(CudaDownloadSurface::Make(str, GetContextByStream(str),
                                               width, height, format));
}

Pixel_Format PySurfaceDownloader::GetFormat() { return m_format; }

bool PySurfaceDownloader::DownloadSingleSurface(shared_ptr<Surface> surface,
                                                py::array& frame) {
  if (!surface || frame.size() != surface->HostMemSize()) {
    return false;
  }

  auto buffer =
      std::shared_ptr<Buffer>(Buffer::Make(frame.size(), frame.mutable_data()));

  upDownloader->SetInput(surface.get(), 0U);
  upDownloader->SetInput(buffer.get(), 1U);

  if (TASK_EXEC_FAIL == upDownloader->Execute()) {
    return false;
  }

  return true;
}

void Init_PySurfaceDownloader(py::module& m) {
  py::class_<PySurfaceDownloader>(m, "PySurfaceDownloader",
                                  "This class is used to copy Surface to numpy "
                                  "ndarray using CUDA DtoH memcpy.")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param width: Surface width
        :param height: Surface height
        :param format: Surface pixel format
        :param gpu_id: what GPU does Surface belong to
    )pbdoc")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param width: Surface width
        :param height: Surface height
        :param format: Surface pixel format
        :param stream: CUDA stream to use for HtoD memcopy
    )pbdoc")
      .def("Format", &PySurfaceDownloader::GetFormat,
           R"pbdoc(
        Get pixel format.
    )pbdoc")
      .def("DownloadSingleSurface", &PySurfaceDownloader::DownloadSingleSurface,
           py::arg("surface"), py::arg("frame"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Perform DtoH memcpy.

        :param surface: input Surface
        :param frame: output numpy array
        :type frame: numpy.ndarray
        :return: True in case of success False otherwise
        :rtype: Bool
    )pbdoc");
}