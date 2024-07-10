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
#include "memory"

using namespace VPF;
namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PyFrameUploader::PyFrameUploader(uint32_t gpu_ID) {
  m_uploader = std::make_unique<CudaUploadFrame>(
      CudaResMgr::Instance().GetStream(gpu_ID));
}

PyFrameUploader::PyFrameUploader(CUstream str) {
  m_uploader = std::make_unique<CudaUploadFrame>(str);
}

bool PyFrameUploader::Run(py::array& src, Surface& dst,
                          TaskExecDetails details) {
  auto buffer =
      std::shared_ptr<Buffer>(Buffer::Make(src.nbytes(), src.mutable_data()));

  m_uploader->SetInput(buffer.get(), 0U);
  m_uploader->SetInput(&dst, 1U);
  details = m_uploader->Execute();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

void Init_PyFrameUploader(py::module& m) {
  py::class_<PyFrameUploader>(m, "PyFrameUploader",
                              "This class is used to upload numpy array to "
                              "Surface using CUDA HtoD memcpy.")
      .def(py::init<uint32_t>(), py::arg("gpu_id"),
           R"pbdoc(
        :param gpu_id: what GPU to use for upload.
    )pbdoc")
      .def(py::init<size_t>(), py::arg("stream"),
           R"pbdoc(
        :param stream: CUDA stream to use for upload
    )pbdoc")
      .def(
          "Run",
          [](PyFrameUploader& self, py::array& src, Surface& dst) {
            TaskExecDetails details;
            return std::make_tuple(self.Run(src, dst, details), details.m_info);
          },
          py::arg("src"), py::arg("dst"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Blocking HtoD CUDA memcpy.

        :param src: input numpy array
        :type src: numpy.ndarray
        :param dst: output surface
        :type dst: Surface
        :return: tuple containing:
          success (Bool) True in case of success, False otherwise.
          info (TaskExecInfo) task execution information.
        :rtype: tuple
    )pbdoc");
}