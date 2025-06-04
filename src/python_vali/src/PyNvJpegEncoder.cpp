/*
 * Copyright 2024 Vision Labs LLC
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
#include <list>

using namespace VPF;

namespace py = pybind11;

PyNvJpegEncoder::PyNvJpegEncoder(int gpu_id)
    : m_stream(CudaResMgr::Instance().GetStream(gpu_id)) {
  upEncoder = std::make_shared<NvJpegEncodeFrame>(m_stream);
}

std::unique_ptr<NvJpegEncodeContext>
PyNvJpegEncoder::Context(unsigned compression, Pixel_Format format) {
  return std::make_unique<NvJpegEncodeContext>(upEncoder, compression, format);
}

TaskExecInfo
PyNvJpegEncoder::CompressImpl(NvJpegEncodeContext& encoder_context,
                              std::list<std::shared_ptr<Surface>>& surfaces,
                              std::list<py::array>& buffers) {
  std::lock_guard<std::mutex> lock(m_mutex);

  TaskExecInfo info = TaskExecInfo::SUCCESS;

  CudaCtxPush ctxPush(m_stream);
  upEncoder->SetEncoderContext(&encoder_context);

  try {
    for (auto surface : surfaces) {
      if (surface->Empty()) {
        std::cerr << "Input surface is empty\n";
        info = TaskExecInfo::FAIL;
        break;
      }

      upEncoder->SetInput(surface.get(), 0U);

      info = upEncoder->Execute().m_info;
      if (TaskExecInfo::SUCCESS != info) {
        break;
      }

      auto outBuffer = static_cast<Buffer*>(upEncoder->GetOutput());
      py::array buf(py::dtype::of<uint8_t>(), {outBuffer->GetRawMemSize()},
                    outBuffer->GetRawMemPtr());
      buffers.push_back(buf);
    }
  } catch (std::exception& e) {
    std::cerr << "Exception " << e.what() << "\n";
    info = TaskExecInfo::FAIL;
  } catch (...) {
    std::cerr << "Unknown exception\n";
    info = TaskExecInfo::FAIL;
  }

  if (TaskExecInfo::FAIL == info) {
    // Either we compress all of input Surfaces or none of them.
    buffers.clear();
  }

  /* No CUDA stream sync is done within encoder for better performance
   * when compressing multiple Surfaces. Hence we need to do it here.
   */
  CudaStrSync sync(m_stream);

  return info;
}

void Init_PyNvJpegEncoder(py::module& m) {
  py::class_<NvJpegEncodeContext>(m, "NvJpegEncodeContext")
      .def("Compression", &NvJpegEncodeContext::Compression,
           R"pbdoc(
         Get the compression coefficient for JPEG encoding.

         The compression coefficient determines the quality of the JPEG encoding,
         where 100 represents maximum quality and lower values indicate higher compression.

         :return: Current compression coefficient (1-100)
         :rtype: int
     )pbdoc")
      .def("Format", &NvJpegEncodeContext::PixelFormat,
           R"pbdoc(
         Get the pixel format used for encoding.

         :return: Current pixel format for encoding
         :rtype: Pixel_Format
     )pbdoc");

  py::class_<PyNvJpegEncoder>(m, "PyNvJpegEncoder")
      .def(py::init<int>(), py::arg("gpu_id"),
           R"pbdoc(
         Create a new JPEG encoder instance.

         Initializes a hardware-accelerated JPEG encoder on the specified GPU.
         The encoder uses NVIDIA's hardware JPEG encoding capabilities for efficient
         image compression.

         :param gpu_id: ID of the GPU to use for encoding
         :type gpu_id: int
         :raises RuntimeError: If GPU initialization fails
     )pbdoc")
      .def("Context", &PyNvJpegEncoder::Context, py::arg("compression"),
           py::arg("pixel_format"),
           R"pbdoc(
         Create a new encoding context with specified parameters.

         The context contains the state and parameters for JPEG encoding, including
         compression quality and pixel format. Each context should be used by only
         one thread at a time to ensure thread safety.

         :param compression: Compression coefficient (1-100, where 100 is maximum quality)
         :type compression: int
         :param pixel_format: Pixel format for the input surfaces
         :type pixel_format: Pixel_Format
         :return: New encoding context with specified parameters
         :rtype: NvJpegEncodeContext
         :raises RuntimeError: If context creation fails
     )pbdoc")
      .def(
          "Run",
          [](PyNvJpegEncoder& self, NvJpegEncodeContext& encoder_context,
             std::list<std::shared_ptr<Surface>>& surfaces) {
            std::list<py::array> buffers;
            auto info = self.CompressImpl(encoder_context, surfaces, buffers);
            return std::make_tuple(buffers, info);
          },
          py::arg("context"), py::arg("surfaces"),
          R"pbdoc(
         Encode multiple surfaces to JPEG format.

         Compresses a list of input surfaces into JPEG format using the specified
         encoding context. The operation is performed asynchronously on the GPU
         for better performance. If any surface fails to encode, the entire operation
         is considered failed and no compressed data is returned.

         :param context: Encoding context containing compression parameters
         :type context: NvJpegEncodeContext
         :param surfaces: List of input surfaces to encode
         :type surfaces: list[Surface]
         :return: Tuple containing:
             - List of compressed JPEG data as numpy arrays (empty if encoding fails)
             - TaskExecInfo indicating success or failure
         :rtype: tuple[list[numpy.ndarray], TaskExecInfo]
         :raises RuntimeError: If encoding fails
     )pbdoc");
}
