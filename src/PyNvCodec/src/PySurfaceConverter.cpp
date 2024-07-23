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

using namespace VPF;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

PySurfaceConverter::PySurfaceConverter(Pixel_Format src, Pixel_Format dst,
                                       uint32_t gpuID) {
  upConverter = std::make_unique<ConvertSurface>(
      src, dst, CudaResMgr::Instance().GetStream(gpuID));
  upCtxBuffer.reset(Buffer::MakeOwnMem(sizeof(ColorspaceConversionContext)));
}

PySurfaceConverter::PySurfaceConverter(Pixel_Format src, Pixel_Format dst,
                                       CUstream str) {
  upConverter = std::make_unique<ConvertSurface>(src, dst, str);
  upCtxBuffer.reset(Buffer::MakeOwnMem(sizeof(ColorspaceConversionContext)));
}

bool PySurfaceConverter::Run(Surface& src, Surface& dst,
                             std::optional<ColorspaceConversionContext> context,
                             TaskExecDetails& details) {
  upConverter->ClearInputs();
  upConverter->SetInput((Token*)&src, 0U);
  upConverter->SetInput((Token*)&dst, 1U);

  if (context) {
    upCtxBuffer->CopyFrom(sizeof(ColorspaceConversionContext),
                          &context.value());
    upConverter->SetInput((Token*)upCtxBuffer.get(), 2U);
  }

  details = upConverter->Execute();
  return (TASK_EXEC_SUCCESS == details.m_status);
}

void Init_PySurfaceConverter(py::module& m) {
  py::class_<PySurfaceConverter>(
      m, "PySurfaceConverter",
      "CUDA-accelerated converter between different pixel formats.")
      .def(py::init<Pixel_Format, Pixel_Format, uint32_t>(),
           py::arg("src_format"), py::arg("dst_format"), py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param src_format: input Surface pixel format
        :param dst_format: output Surface pixel format
        :param gpu_id: what GPU to run conversion on
    )pbdoc")
      .def(py::init<Pixel_Format, Pixel_Format, size_t>(),
           py::arg("src_format"), py::arg("dst_format"), py::arg("stream"),
           R"pbdoc(
        Constructor method.

        :param src_format: input Surface pixel format
        :param dst_format: output Surface pixel format
        :param stream: CUDA stream to use for conversion
    )pbdoc")
      .def(
          "Run",
          [](PySurfaceConverter& self, Surface& src, Surface& dst,
             std::optional<ColorspaceConversionContext> cc_ctx) {
            TaskExecDetails details;
            return std::make_tuple(self.Run(src, dst, cc_ctx, details),
                                   details.m_info);
          },
          py::arg("src"), py::arg("dst"), py::arg("cc_ctx") = std::nullopt,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Perform pixel format conversion.

        :param src: input Surface. Must be of same format class instance was created with.
        :param dst: output Surface. Must be of suitable format.
        :param cc_ctx: colorspace conversion context. Describes color space and color range used for conversion. Optional parameter. If not given, VALI will automatically pick supported color conversion parameters.
        :return: tuple containing:
          success (Bool) True in case of success, False otherwise.
          info (TaskExecInfo) task execution information.
        :rtype: tuple
    )pbdoc");
}