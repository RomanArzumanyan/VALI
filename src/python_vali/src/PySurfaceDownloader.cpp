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
#include "VALI.hpp"
#include "memory"

using namespace VPF;
namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceDownloader::PySurfaceDownloader(int gpu_id) {
  upDownloader = std::make_unique<CudaDownloadSurface>(
      gpu_id, CudaResMgr::Instance().GetStream(gpu_id));
}

PySurfaceDownloader::PySurfaceDownloader(int gpu_id, CUstream str) {
  upDownloader = std::make_unique<CudaDownloadSurface>(gpu_id, str);
}

bool PySurfaceDownloader::Run(Surface& src, py::array& dst,
                              TaskExecDetails& details) {
  auto buffer =
      std::shared_ptr<Buffer>(Buffer::Make(dst.nbytes(), dst.mutable_data()));
  py::gil_scoped_release gil_release{};

  upDownloader->SetInput(&src, 0U);
  upDownloader->SetInput(buffer.get(), 1U);

  details = upDownloader->Execute();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

void Init_PySurfaceDownloader(py::module& m) {
  py::class_<PySurfaceDownloader>(m, "PySurfaceDownloader",
                                  "This class is used to copy Surface to numpy "
                                  "ndarray using CUDA DtoH memcpy.")
      .def(py::init<int>(), py::arg("gpu_id"),
           R"pbdoc(
         Constructor for PySurfaceDownloader with GPU ID.

         Creates a new instance of PySurfaceDownloader that will run on the specified GPU.
         The CUDA stream will be automatically created and managed.

         :param gpu_id: The ID of the GPU that owns the Surface to be downloaded
         :type gpu_id: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Constructor for PySurfaceDownloader with GPU ID and CUDA stream.

         Creates a new instance of PySurfaceDownloader that will run on the specified GPU
         using the provided CUDA stream.

         :param gpu_id: The ID of the GPU that owns the Surface to be downloaded
         :type gpu_id: int
         :param stream: The CUDA stream to use for device-to-host memory copy
         :type stream: int
         :raises RuntimeError: If the specified GPU is not available
     )pbdoc")
      .def(
          "Run",
          [](PySurfaceDownloader& self, Surface& src, py::array& dst) {
            TaskExecDetails details;
            return std::make_tuple(self.Run(src, dst, details), details.m_info);
          },
          py::arg("src"), py::arg("dst"),
          R"pbdoc(
         Perform device-to-host memory copy from Surface to numpy array.

         Copies the contents of a GPU Surface to a numpy array using CUDA DtoH memcpy.
         The numpy array must be pre-allocated with sufficient size to hold the Surface data.
         The GIL is released during the copy operation to allow other Python threads to run.

         :param src: Input Surface to be downloaded from GPU
         :type src: Surface
         :param dst: Pre-allocated numpy array to store the downloaded data
         :type dst: numpy.ndarray
         :return: Tuple containing:
             - success (bool): True if download was successful, False otherwise
             - info (TaskExecInfo): Detailed execution information
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the download operation fails or if the numpy array is too small
     )pbdoc");
}