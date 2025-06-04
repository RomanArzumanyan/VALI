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
#include "memory"

using namespace VPF;
namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyFrameUploader::PyFrameUploader(int gpu_id)
    : PyFrameUploader(gpu_id, CudaResMgr::Instance().GetStream(gpu_id)) {}

PyFrameUploader::PyFrameUploader(int gpu_id, CUstream str) {
  m_uploader = std::make_unique<CudaUploadFrame>(gpu_id, str);
}

bool PyFrameUploader::Run(py::array& src, Surface& dst,
                          TaskExecDetails details) {
  auto buffer =
      std::shared_ptr<Buffer>(Buffer::Make(src.nbytes(), src.mutable_data()));
  py::gil_scoped_release gil_release{};

  m_uploader->SetInput(buffer.get(), 0U);
  m_uploader->SetInput(&dst, 1U);
  details = m_uploader->Execute();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

void Init_PyFrameUploader(py::module& m) {
  py::class_<PyFrameUploader>(m, "PyFrameUploader",
                              "This class is used to upload numpy array to "
                              "Surface using CUDA HtoD memcpy.")
      .def(py::init<int>(), py::arg("gpu_id"),
           R"pbdoc(
         Create a new frame uploader instance.

         Initializes a CUDA frame uploader that transfers data from host memory
         (numpy arrays) to device memory (Surface) using CUDA's host-to-device
         memory copy operations.

         :param gpu_id: ID of the GPU to use for memory transfers
         :type gpu_id: int
         :raises RuntimeError: If GPU initialization fails
     )pbdoc")
      .def(py::init<int, size_t>(), py::arg("gpu_id"), py::arg("stream"),
           R"pbdoc(
         Create a new frame uploader instance with a specific CUDA stream.

         Initializes a CUDA frame uploader that transfers data from host memory
         to device memory using a specific CUDA stream. This allows for better
         control over CUDA stream management and potential overlap with other
         GPU operations.

         :param gpu_id: ID of the GPU to use for memory transfers
         :type gpu_id: int
         :param stream: CUDA stream to use for memory transfers
         :type stream: int
         :raises RuntimeError: If GPU initialization fails
     )pbdoc")
      .def(
          "Run",
          [](PyFrameUploader& self, py::array& src, Surface& dst) {
            TaskExecDetails details;
            return std::make_tuple(self.Run(src, dst, details), details.m_info);
          },
          py::arg("src"), py::arg("dst"),
          R"pbdoc(
         Perform a blocking host-to-device memory copy.

         Transfers data from a numpy array in host memory to a Surface in device
         memory. The operation is performed synchronously, meaning it will block
         until the transfer is complete.

         :param src: Source numpy array containing the data to transfer
         :type src: numpy.ndarray
         :param dst: Destination Surface that will receive the data
         :type dst: Surface
         :return: Tuple containing:
             - success (bool): True if the transfer was successful, False otherwise
             - info (TaskExecInfo): Detailed information about the transfer operation
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the memory transfer fails
     )pbdoc");
}