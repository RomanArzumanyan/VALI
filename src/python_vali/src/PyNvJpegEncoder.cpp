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
        :return: compression coefficient.
    )pbdoc")
      .def("Format", &NvJpegEncodeContext::PixelFormat,
           R"pbdoc(
        :return: pixel format.
    )pbdoc");

  py::class_<PyNvJpegEncoder>(m, "PyNvJpegEncoder")
      .def(py::init<int>(), py::arg("gpu_id"),
           R"pbdoc(
        Constructor method.

        :param gpu_id: what GPU to run encode on
    )pbdoc")
      .def("Context", &PyNvJpegEncoder::Context, py::arg("compression"),
           py::arg("pixel_format"),
           R"pbdoc(
        NvJpegEncodeContext structure contains state and parameters of PyNvJpegEncoder,
        including given compression coefficient (100 = maximum quality) and PixelFormat.
        Using one context in multiple threads is prohibited.

        :return: new NvJpegEncodeContext.
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
        Encode multiple Surfaces. In case of an error it returns empty list and TaskExecInfo.FAIL.
        
        :param context: context (NvJpegEncodeContext)
        :param surfaces: list of input Surfaces
        :return: tuple, the first element is the list of buffers compressed images. The second element is TaskExecInfo.
    )pbdoc");
}
