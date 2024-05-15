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

bool PyFrameUploader::Run(py::array& src, Surface& dst) {
  auto buffer =
      std::shared_ptr<Buffer>(Buffer::Make(src.size(), src.mutable_data()));

  return Run(*buffer.get(), dst);
}

bool PyFrameUploader::Run(Buffer& src, Surface& dst) {
  m_uploader->SetInput(&src, 0U);
  m_uploader->SetInput(&dst, 1U);
  auto res = m_uploader->Execute();

  if (TASK_EXEC_FAIL == res) {
    return false;
  }

  return true;
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
      .def("Run",
           py::overload_cast<py::array&, Surface&>(&PyFrameUploader::Run),
           py::arg("src"), py::arg("dst"),
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Blocking HtoD CUDA memcpy.

        :param src: input numpy array
        :type src: numpy.ndarray
        :param dst: output surface
        :type dst: Surface
        :return: True in case of success, False otherwise
        :rtype: bool
    )pbdoc");
}