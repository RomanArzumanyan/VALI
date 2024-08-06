/*
 *Copyright 2024 Vision Labs LLC
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
#include "Utils.hpp"

using namespace VPF;
namespace py = pybind11;

PyFrameConverter::PyFrameConverter(uint32_t width, uint32_t height,
                                   Pixel_Format inFormat,
                                   Pixel_Format outFormat)
    : m_width(width), m_height(height), m_src_fmt(inFormat),
      m_dst_fmt(outFormat) {
  m_up_cvt.reset(ConvertFrame::Make(width, height, inFormat, outFormat));
  m_up_ctx_buf.reset(Buffer::MakeOwnMem(sizeof(ColorspaceConversionContext)));
}

bool PyFrameConverter::Run(py::array& src, py::array& dst,
                           std::shared_ptr<ColorspaceConversionContext> context,
                           TaskExecDetails& details) {
  auto const src_buf_size =
      getBufferSize(m_width, m_height, toFfmpegPixelFormat(m_src_fmt));
  if (src.nbytes() != src_buf_size) {
    details.m_info = TaskExecInfo::INVALID_INPUT;
    return false;
  }

  auto const dst_buf_size =
      getBufferSize(m_width, m_height, toFfmpegPixelFormat(m_dst_fmt));
  if (dst.nbytes() != dst_buf_size) {
    dst.resize({dst_buf_size}, false);
  }

  auto src_buf = std::shared_ptr<Buffer>(
      Buffer::Make(src.nbytes(), (void*)src.mutable_data()));

  auto dst_buf = std::shared_ptr<Buffer>(
      Buffer::Make(dst.nbytes(), (void*)dst.mutable_data()));

  m_up_cvt->ClearInputs();
  m_up_cvt->SetInput(src_buf.get(), 0U);
  m_up_cvt->SetInput(dst_buf.get(), 1U);

  if (context) {
    m_up_ctx_buf->CopyFrom(sizeof(ColorspaceConversionContext), context.get());
    m_up_cvt->SetInput((Token*)m_up_ctx_buf.get(), 2U);
  }

  details = m_up_cvt->Run();
  return (details.m_status == TaskExecStatus::TASK_EXEC_SUCCESS);
}

void Init_PyFrameConverter(py::module& m) {
  py::class_<PyFrameConverter>(
      m, "PyFrameConverter",
      "libswscale converter between different pixel formats.")
      .def(py::init<uint32_t, uint32_t, Pixel_Format, Pixel_Format>(),
           py::arg("width"), py::arg("height"), py::arg("src_format"),
           py::arg("dst_format"),
           R"pbdoc(
        Constructor method.

        :param width: target frame width
        :param height: target frame height
        :param src_format: input frame pixel format
        :param dst_format: output frame pixel format
    )pbdoc")
      .def("Format", &PyFrameConverter::GetFormat, R"pbdoc(
        Get pixel format.
    )pbdoc")
      .def(
          "Run",
          [](PyFrameConverter& self, py::array& src, py::array& dst,
             std::shared_ptr<ColorspaceConversionContext> cc_ctx) {
            TaskExecDetails details;
            return std::make_tuple(self.Run(src, dst, cc_ctx, details),
                                   details.m_info);
          },
          py::arg("src"), py::arg("dst"), py::arg("cc_ctx"),
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
        Perform pixel format conversion.

        :param src: input numpy ndarray, it must be of proper size for given format and resolution.
        :param dst: output numpy ndarray, it may be resized to fit the converted frame.
        :param cc_ctx: colorspace conversion context. Describes color space and color range used for conversion.
        :return: tuple containing:
          success (Bool) True in case of success, False otherwise.
          info (TaskExecInfo) task execution information.
        :rtype: tuple
    )pbdoc");
}