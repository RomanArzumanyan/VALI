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

#include "Utils.hpp"
#include "VALI.hpp"

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

  py::gil_scoped_release gil_release{};

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
         Create a new frame converter instance.

         Initializes a frame converter that uses libswscale to convert between
         different pixel formats. The converter is configured for a specific
         resolution and source/destination pixel formats.

         :param width: Width of the frames to convert in pixels
         :type width: int
         :param height: Height of the frames to convert in pixels
         :type height: int
         :param src_format: Pixel format of the input frames
         :type src_format: Pixel_Format
         :param dst_format: Pixel format for the output frames
         :type dst_format: Pixel_Format
         :raises RuntimeError: If converter initialization fails
     )pbdoc")
      .def_property_readonly("Format", &PyFrameConverter::GetFormat, R"pbdoc(
         Get the current pixel format configuration.

         Returns a tuple containing the source and destination pixel formats
         that this converter is configured to use.

         :return: Tuple of (source_format, destination_format)
         :rtype: tuple[Pixel_Format, Pixel_Format]
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
          R"pbdoc(
         Convert a frame between pixel formats.

         Performs pixel format conversion on the input frame using libswscale.
         The input array must have the correct size for the configured resolution
         and source format. The output array will be automatically resized if
         needed to accommodate the converted frame.

         :param src: Input numpy array containing the frame to convert
         :type src: numpy.ndarray
         :param dst: Output numpy array that will receive the converted frame
         :type dst: numpy.ndarray
         :param cc_ctx: Colorspace conversion context specifying color space and range
         :type cc_ctx: ColorspaceConversionContext
         :return: Tuple containing:
             - success (bool): True if conversion was successful, False otherwise
             - info (TaskExecInfo): Detailed information about the conversion operation
         :rtype: tuple[bool, TaskExecInfo]
         :raises RuntimeError: If the conversion fails
         :raises ValueError: If the input array has incorrect dimensions
     )pbdoc");
}