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

PyFrameUploader::PyFrameUploader(uint32_t width, uint32_t height,
                                 Pixel_Format format, uint32_t gpu_ID)
    : m_format(format) {
  uploader.reset(CudaUploadFrame::Make(CudaResMgr::Instance().GetStream(gpu_ID),
                                       CudaResMgr::Instance().GetCtx(gpu_ID),
                                       width, height, format));
}

PyFrameUploader::PyFrameUploader(uint32_t width, uint32_t height,
                                 Pixel_Format format, CUstream str)
    : m_format(format) {
  uploader.reset(CudaUploadFrame::Make(str, GetContextByStream(str), width,
                                       height, format));
}

Pixel_Format PyFrameUploader::GetFormat() { return m_format; }

std::shared_ptr<Surface> PyFrameUploader::UploadSingleFrame(py::array& frame) {
  auto buffer =
      std::shared_ptr<Buffer>(Buffer::Make(frame.size(), frame.mutable_data()));

  return UploadSingleFrame(buffer.get());
}

std::shared_ptr<Surface> PyFrameUploader::UploadSingleFrame(Buffer* buf) {
  uploader->SetInput(buf, 0U);
  auto res = uploader->Execute();

  if (TASK_EXEC_FAIL == res) {
    throw runtime_error("Error uploading frame to GPU");
  }

  auto pSurface = (Surface*)uploader->GetOutput(0U);
  if (!pSurface) {
    throw runtime_error("Error uploading frame to GPU");
  }

  return shared_ptr<Surface>(pSurface->Clone());
}

void Init_PyFrameUploader(py::module& m) {
  py::class_<PyFrameUploader>(m, "PyFrameUploader",
                              "This class is used to upload numpy array to "
                              "Surface using CUDA HtoD memcpy.")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, uint32_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param width: target Surface width
        :param height: target Surface height
        :param format: target Surface pixel format
        :param gpu_id: what GPU to use for upload.
    )pbdoc")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, size_t>(),
           py::arg("width"), py::arg("height"), py::arg("format"),
           py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param width: target Surface width
        :param height: target Surface height
        :param format: target Surface pixel format
        :param stream: CUDA stream to use for upload
    )pbdoc")
      .def("Format", &PyFrameUploader::GetFormat,
           R"pbdoc(
        Get pixel format.
    )pbdoc")
      .def("UploadSingleFrame",
           py::overload_cast<py::array&>(&PyFrameUploader::UploadSingleFrame),
           py::arg("frame"), py::return_value_policy::take_ownership,
           py::call_guard<py::gil_scoped_release>(),
           R"pbdoc(
        Perform HtoD memcpy.

        :param frame: input numpy array
        :type frame: numpy.ndarray
        :return: Surface
        :rtype: PyNvCodec.Surface
    )pbdoc");
}