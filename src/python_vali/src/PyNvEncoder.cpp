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

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;

constexpr auto TASK_EXEC_SUCCESS = TaskExecStatus::TASK_EXEC_SUCCESS;
constexpr auto TASK_EXEC_FAIL = TaskExecStatus::TASK_EXEC_FAIL;

int nvcvImagePitch = 0;

struct EncodeContext {
  std::shared_ptr<Surface> rawSurface;
  py::array* pPacket;
  const py::array* pMessageSEI;
  bool sync;
  bool append;

  EncodeContext(std::shared_ptr<Surface> spRawSurface, py::array* packet,
                const py::array* messageSEI, bool is_sync, bool is_append)
      : rawSurface(spRawSurface), pPacket(packet), pMessageSEI(messageSEI),
        sync(is_sync), append(is_append) {}
};

uint32_t PyNvEncoder::Width() const { return encWidth; }

uint32_t PyNvEncoder::Height() const { return encHeight; }

Pixel_Format PyNvEncoder::GetPixelFormat() const { return eFormat; }

std::map<NV_ENC_CAPS, int> PyNvEncoder::Capabilities() {
  if (!upEncoder) {
    NvEncoderClInterface cli_interface(options);

    upEncoder.reset(NvencEncodeFrame::Make(
        m_gpu_id, m_stream, cli_interface,
        NV12 == eFormat     ? NV_ENC_BUFFER_FORMAT_NV12
        : YUV444 == eFormat ? NV_ENC_BUFFER_FORMAT_YUV444
                            : NV_ENC_BUFFER_FORMAT_UNDEFINED,
        encWidth, encHeight, verbose_ctor));
  }

  std::map<NV_ENC_CAPS, int> capabilities;
  capabilities.erase(capabilities.begin(), capabilities.end());
  for (int cap = NV_ENC_CAPS_NUM_MAX_BFRAMES; cap < NV_ENC_CAPS_EXPOSED_COUNT;
       cap++) {
    auto val = upEncoder->GetCapability((NV_ENC_CAPS)cap);
    capabilities[(NV_ENC_CAPS)cap] = val;
  }

  return capabilities;
}

int PyNvEncoder::GetFrameSizeInBytes() const {
  switch (GetPixelFormat()) {

  case NV12:
    return Width() * (Height() + (Height() + 1) / 2);
  case YUV420_10bit:
  case YUV444:
    return Width() * Height() * 3;
  case YUV444_10bit:
    return 2 * Width() * Height() * 3;
  default:
    throw invalid_argument("Invalid Buffer format");
    return 0;
  }
}

bool PyNvEncoder::Reconfigure(const map<string, string>& encodeOptions,
                              bool force_idr, bool reset_enc, bool verbose) {
  if (upEncoder) {
    NvEncoderClInterface cli_interface(encodeOptions);
    auto ret =
        upEncoder->Reconfigure(cli_interface, force_idr, reset_enc, verbose);
    if (!ret) {
      return ret;
    } else {
      encWidth = upEncoder->GetWidth();
      encHeight = upEncoder->GetHeight();
    }
  }

  return true;
}

PyNvEncoder::PyNvEncoder(const map<string, string>& encodeOptions, int gpu_id,
                         Pixel_Format format, bool verbose)
    : PyNvEncoder(encodeOptions, gpu_id,
                  CudaResMgr::Instance().GetStream(gpu_id), format, verbose) {}

PyNvEncoder::PyNvEncoder(const map<string, string>& encodeOptions, int gpu_id,
                         CUstream str, Pixel_Format format, bool verbose)
    : upEncoder(nullptr), options(encodeOptions), verbose_ctor(verbose),
      eFormat(format) {

  // Parse resolution;
  auto ParseResolution = [&](const string& res_string, uint32_t& width,
                             uint32_t& height) {
    string::size_type xPos = res_string.find('x');

    if (xPos != string::npos) {
      // Parse width;
      stringstream ssWidth;
      ssWidth << res_string.substr(0, xPos);
      ssWidth >> width;

      // Parse height;
      stringstream ssHeight;
      ssHeight << res_string.substr(xPos + 1);
      ssHeight >> height;
    } else {
      throw invalid_argument("Invalid resolution.");
    }
  };

  auto it = options.find("s");
  if (it != options.end()) {
    ParseResolution(it->second, encWidth, encHeight);
  } else {
    throw invalid_argument("No resolution given");
  }

  // Parse pixel format;
  string fmt_string;
  switch (eFormat) {
  case NV12:
    fmt_string = "NV12";
    break;
  case YUV444:
    fmt_string = "YUV444";
    break;
  case YUV444_10bit:
    fmt_string = "YUV444_10bit";
    break;
  case YUV420_10bit:
    fmt_string = "YUV420_10bit";
    break;
  default:
    fmt_string = "UNDEFINED";
    break;
  }

  it = options.find("fmt");
  if (it != options.end()) {
    it->second = fmt_string;
  } else {
    options["fmt"] = fmt_string;
  }

  m_stream = str;
  m_gpu_id = gpu_id;

  /* Don't initialize encoder here, just prepare config params;
   */
  Reconfigure(options, false, false, verbose);
}

bool PyNvEncoder::EncodeSingleSurface(EncodeContext& ctx) {
  shared_ptr<Buffer> spSEI = nullptr;
  if (ctx.pMessageSEI && ctx.pMessageSEI->size()) {
    spSEI = shared_ptr<Buffer>(
        Buffer::MakeOwnMem(ctx.pMessageSEI->size(), ctx.pMessageSEI->data()));
  }

  if (!upEncoder) {
    NvEncoderClInterface cli_interface(options);

    NV_ENC_BUFFER_FORMAT encoderFormat;

    switch (eFormat) {
    case VPF::NV12:
      encoderFormat = NV_ENC_BUFFER_FORMAT_NV12;
      break;
    case VPF::YUV444:
      encoderFormat = NV_ENC_BUFFER_FORMAT_YUV444;
      break;
    case VPF::YUV420_10bit: // P12 already has memory representation similar to
                            // 10 bit yuv420, hence reusing the same class
    case VPF::P12:
      encoderFormat = NV_ENC_BUFFER_FORMAT_YUV420_10BIT;
      break;
    case VPF::YUV444_10bit:
      encoderFormat = NV_ENC_BUFFER_FORMAT_YUV444_10BIT;
      break;
    default:
      throw invalid_argument(
          "Input buffer format not supported by VPF currently.");
      break;
    }

    upEncoder.reset(NvencEncodeFrame::Make(m_gpu_id, m_stream, cli_interface,
                                           encoderFormat, encWidth, encHeight,
                                           verbose_ctor));
  }

  upEncoder->ClearInputs();

  if (ctx.rawSurface) {
    upEncoder->SetInput(ctx.rawSurface.get(), 0U);
  } else {
    /* Flush encoder this way;
     */
    upEncoder->SetInput(nullptr, 0U);
  }

  if (ctx.sync) {
    /* Set 2nd input to any non-zero value
     * to signal sync encode;
     */
    upEncoder->SetInput((Token*)0xdeadbeefull, 1U);
  }

  if (ctx.pMessageSEI && ctx.pMessageSEI->size()) {
    /* Set 3rd input in case we have SEI message;
     */
    upEncoder->SetInput(spSEI.get(), 2U);
  }

  {
    py::gil_scoped_release gil_release{};
    if (TASK_EXEC_FAIL == upEncoder->Execute().m_status) {
      throw runtime_error("Error while encoding frame");
    }
  }

  auto encodedFrame = (Buffer*)upEncoder->GetOutput(0U);
  if (encodedFrame) {
    if (ctx.append) {
      auto old_size = ctx.pPacket->size();
      ctx.pPacket->resize({old_size + encodedFrame->GetRawMemSize()}, false);
      memcpy((uint8_t*)ctx.pPacket->mutable_data() + old_size,
             encodedFrame->GetRawMemPtr(), encodedFrame->GetRawMemSize());
    } else {
      ctx.pPacket->resize({encodedFrame->GetRawMemSize()}, false);
      memcpy(ctx.pPacket->mutable_data(), encodedFrame->GetRawMemPtr(),
             encodedFrame->GetRawMemSize());
    }
    return true;
  }

  return false;
}

bool PyNvEncoder::FlushSinglePacket(py::array& packet) {
  /* Keep feeding encoder with null input until it returns zero-size
   * surface; */
  shared_ptr<Surface> spRawSurface = nullptr;
  const py::array* messageSEI = nullptr;
  auto const is_sync = true;
  auto const is_append = false;
  EncodeContext ctx(spRawSurface, &packet, messageSEI, is_sync, is_append);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::Flush(py::array& packets) {
  uint32_t num_packets = 0U;
  do {
    if (!FlushSinglePacket(packets)) {
      break;
    }
    num_packets++;
  } while (true);
  return (num_packets > 0U);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array& packet, const py::array& messageSEI,
                                bool sync, bool append) {
  EncodeContext ctx(rawSurface, &packet, &messageSEI, sync, append);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array& packet, const py::array& messageSEI,
                                bool sync) {
  EncodeContext ctx(rawSurface, &packet, &messageSEI, sync, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array& packet, bool sync) {
  EncodeContext ctx(rawSurface, &packet, nullptr, sync, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array& packet,
                                const py::array& messageSEI) {
  EncodeContext ctx(rawSurface, &packet, &messageSEI, false, false);
  return EncodeSingleSurface(ctx);
}

bool PyNvEncoder::EncodeSurface(shared_ptr<Surface> rawSurface,
                                py::array& packet) {
  EncodeContext ctx(rawSurface, &packet, nullptr, false, false);
  return EncodeSingleSurface(ctx);
}

void Init_PyNvEncoder(py::module& m) {
  py::enum_<NV_ENC_CAPS>(m, "NV_ENC_CAPS")
      .value("NUM_MAX_BFRAMES", NV_ENC_CAPS_NUM_MAX_BFRAMES)
      .value("SUPPORTED_RATECONTROL_MODES",
             NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES)
      .value("SUPPORT_FIELD_ENCODING", NV_ENC_CAPS_SUPPORT_FIELD_ENCODING)
      .value("SUPPORT_MONOCHROME", NV_ENC_CAPS_SUPPORT_MONOCHROME)
      .value("SUPPORT_FMO", NV_ENC_CAPS_SUPPORT_FMO)
      .value("SUPPORT_QPELMV", NV_ENC_CAPS_SUPPORT_QPELMV)
      .value("SUPPORT_BDIRECT_MODE", NV_ENC_CAPS_SUPPORT_BDIRECT_MODE)
      .value("SUPPORT_CABAC", NV_ENC_CAPS_SUPPORT_CABAC)
      .value("SUPPORT_ADAPTIVE_TRANSFORM",
             NV_ENC_CAPS_SUPPORT_ADAPTIVE_TRANSFORM)
      .value("SUPPORT_STEREO_MVC", NV_ENC_CAPS_SUPPORT_STEREO_MVC)
      .value("NUM_MAX_TEMPORAL_LAYERS", NV_ENC_CAPS_NUM_MAX_TEMPORAL_LAYERS)
      .value("SUPPORT_HIERARCHICAL_PFRAMES",
             NV_ENC_CAPS_SUPPORT_HIERARCHICAL_PFRAMES)
      .value("SUPPORT_HIERARCHICAL_BFRAMES",
             NV_ENC_CAPS_SUPPORT_HIERARCHICAL_BFRAMES)
      .value("LEVEL_MAX", NV_ENC_CAPS_LEVEL_MAX)
      .value("LEVEL_MIN", NV_ENC_CAPS_LEVEL_MIN)
      .value("SEPARATE_COLOUR_PLANE", NV_ENC_CAPS_SEPARATE_COLOUR_PLANE)
      .value("WIDTH_MAX", NV_ENC_CAPS_WIDTH_MAX)
      .value("HEIGHT_MAX", NV_ENC_CAPS_HEIGHT_MAX)
      .value("SUPPORT_TEMPORAL_SVC", NV_ENC_CAPS_SUPPORT_TEMPORAL_SVC)
      .value("SUPPORT_DYN_RES_CHANGE", NV_ENC_CAPS_SUPPORT_DYN_RES_CHANGE)
      .value("SUPPORT_DYN_BITRATE_CHANGE",
             NV_ENC_CAPS_SUPPORT_DYN_BITRATE_CHANGE)
      .value("SUPPORT_DYN_FORCE_CONSTQP", NV_ENC_CAPS_SUPPORT_DYN_FORCE_CONSTQP)
      .value("SUPPORT_DYN_RCMODE_CHANGE", NV_ENC_CAPS_SUPPORT_DYN_RCMODE_CHANGE)
      .value("SUPPORT_SUBFRAME_READBACK", NV_ENC_CAPS_SUPPORT_SUBFRAME_READBACK)
      .value("SUPPORT_CONSTRAINED_ENCODING",
             NV_ENC_CAPS_SUPPORT_CONSTRAINED_ENCODING)
      .value("SUPPORT_INTRA_REFRESH", NV_ENC_CAPS_SUPPORT_INTRA_REFRESH)
      .value("SUPPORT_CUSTOM_VBV_BUF_SIZE",
             NV_ENC_CAPS_SUPPORT_CUSTOM_VBV_BUF_SIZE)
      .value("SUPPORT_DYNAMIC_SLICE_MODE",
             NV_ENC_CAPS_SUPPORT_DYNAMIC_SLICE_MODE)
      .value("SUPPORT_REF_PIC_INVALIDATION",
             NV_ENC_CAPS_SUPPORT_REF_PIC_INVALIDATION)
      .value("PREPROC_SUPPORT", NV_ENC_CAPS_PREPROC_SUPPORT)
      .value("ASYNC_ENCODE_SUPPORT", NV_ENC_CAPS_ASYNC_ENCODE_SUPPORT)
      .value("MB_NUM_MAX", NV_ENC_CAPS_MB_NUM_MAX)
      .value("MB_PER_SEC_MAX", NV_ENC_CAPS_MB_PER_SEC_MAX)
      .value("SUPPORT_YUV444_ENCODE", NV_ENC_CAPS_SUPPORT_YUV444_ENCODE)
      .value("SUPPORT_LOSSLESS_ENCODE", NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE)
      .value("SUPPORT_SAO", NV_ENC_CAPS_SUPPORT_SAO)
      .value("SUPPORT_MEONLY_MODE", NV_ENC_CAPS_SUPPORT_MEONLY_MODE)
      .value("SUPPORT_LOOKAHEAD", NV_ENC_CAPS_SUPPORT_LOOKAHEAD)
      .value("SUPPORT_TEMPORAL_AQ", NV_ENC_CAPS_SUPPORT_TEMPORAL_AQ)
      .value("SUPPORT_10BIT_ENCODE", NV_ENC_CAPS_SUPPORT_10BIT_ENCODE)
      .value("NUM_MAX_LTR_FRAMES", NV_ENC_CAPS_NUM_MAX_LTR_FRAMES)
      .value("SUPPORT_WEIGHTED_PREDICTION",
             NV_ENC_CAPS_SUPPORT_WEIGHTED_PREDICTION)
      .value("DYNAMIC_QUERY_ENCODER_CAPACITY",
             NV_ENC_CAPS_DYNAMIC_QUERY_ENCODER_CAPACITY)
      .value("SUPPORT_BFRAME_REF_MODE", NV_ENC_CAPS_SUPPORT_BFRAME_REF_MODE)
      .value("SUPPORT_EMPHASIS_LEVEL_MAP",
             NV_ENC_CAPS_SUPPORT_EMPHASIS_LEVEL_MAP)
      .value("WIDTH_MIN", NV_ENC_CAPS_WIDTH_MIN)
      .value("HEIGHT_MIN", NV_ENC_CAPS_HEIGHT_MIN)
      .value("SUPPORT_MULTIPLE_REF_FRAMES",
             NV_ENC_CAPS_SUPPORT_MULTIPLE_REF_FRAMES)
      .value("SUPPORT_ALPHA_LAYER_ENCODING",
             NV_ENC_CAPS_SUPPORT_ALPHA_LAYER_ENCODING)
      .value("EXPOSED_COUNT", NV_ENC_CAPS_EXPOSED_COUNT)
      .export_values();

  py::class_<PyNvEncoder>(m, "PyNvEncoder",
                          "HW-accelerated video encoder which uses Nvenc.")
      .def(py::init<const map<string, string>&, int, Pixel_Format, bool>(),
           py::arg("settings"), py::arg("gpu_id"), py::arg("format") = NV12,
           py::arg("verbose") = false,
           R"pbdoc(
         Create a new hardware-accelerated video encoder.

         Initializes an NVIDIA hardware-accelerated video encoder with the specified
         settings and pixel format. The encoder uses NVIDIA's NVENC hardware encoder
         for efficient video compression.

         :param settings: Dictionary containing NVENC encoder settings (e.g., bitrate, codec, etc.)
         :type settings: dict[str, str]
         :param gpu_id: ID of the GPU to use for encoding
         :type gpu_id: int
         :param format: Pixel format for input frames (default: NV12)
         :type format: Pixel_Format
         :param verbose: Whether to output detailed logging information
         :type verbose: bool
         :raises RuntimeError: If encoder initialization fails
     )pbdoc")
      .def(py::init<const map<string, string>&, int, size_t, Pixel_Format,
                    bool>(),
           py::arg("settings"), py::arg("gpu_id"), py::arg("stream"),
           py::arg("format") = NV12, py::arg("verbose") = false,
           R"pbdoc(
         Create a new hardware-accelerated video encoder with a specific CUDA stream.

         Initializes an NVIDIA hardware-accelerated video encoder with the specified
         settings, pixel format, and CUDA stream. This constructor allows for more
         control over CUDA stream management.

         :param settings: Dictionary containing NVENC encoder settings
         :type settings: dict[str, str]
         :param gpu_id: ID of the GPU to use for encoding
         :type gpu_id: int
         :param stream: CUDA stream to use for encoding operations
         :type stream: int
         :param format: Pixel format for input frames (default: NV12)
         :type format: Pixel_Format
         :param verbose: Whether to output detailed logging information
         :type verbose: bool
         :raises RuntimeError: If encoder initialization fails
     )pbdoc")
      .def("Reconfigure", &PyNvEncoder::Reconfigure, py::arg("settings"),
           py::arg("force_idr") = false, py::arg("reset_encoder") = false,
           py::arg("verbose") = false,
           R"pbdoc(
         Reconfigure the encoder with new settings.

         Updates the encoder configuration with new settings. This can be used to
         change encoding parameters during runtime, such as bitrate or resolution.

         :param settings: Dictionary containing new NVENC encoder settings
         :type settings: dict[str, str]
         :param force_idr: Force the next encoded frame to be an IDR key frame
         :type force_idr: bool
         :param reset_encoder: Force a complete encoder reset
         :type reset_encoder: bool
         :param verbose: Whether to output detailed logging information
         :type verbose: bool
         :return: True if reconfiguration was successful, False otherwise
         :rtype: bool
         :raises RuntimeError: If reconfiguration fails
     )pbdoc")
      .def_property_readonly("Width", &PyNvEncoder::Width,
                             R"pbdoc(
         Get the width of the encoded video stream.

         :return: Width of the encoded video in pixels
         :rtype: int
     )pbdoc")
      .def_property_readonly("Height", &PyNvEncoder::Height,
                             R"pbdoc(
         Get the height of the encoded video stream.

         :return: Height of the encoded video in pixels
         :rtype: int
     )pbdoc")
      .def_property_readonly("Format", &PyNvEncoder::GetPixelFormat,
                             R"pbdoc(
         Get the pixel format of the encoded video stream.

         :return: Pixel format used for encoding
         :rtype: Pixel_Format
     )pbdoc")
      .def_property_readonly("FrameSizeInBytes",
                             &PyNvEncoder::GetFrameSizeInBytes,
                             R"pbdoc(
         Get the size of a single frame in bytes.

         Calculates the size of a single frame based on the current pixel format
         and resolution. This is useful for allocating memory for frame buffers.

         :return: Size of a single frame in bytes
         :rtype: int
         :raises ValueError: If the pixel format is not supported
     )pbdoc")
      .def_property_readonly("Capabilities", &PyNvEncoder::Capabilities,
                             py::return_value_policy::move,
                             R"pbdoc(
         Get the capabilities of the NVENC encoder.

         Returns a dictionary containing all supported capabilities of the NVENC
         encoder, such as maximum resolution, supported codecs, and encoding features.

         :return: Dictionary mapping capability types to their values
         :rtype: dict[NV_ENC_CAPS, int]
     )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array&, const py::array&,
                             bool, bool>(&PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"), py::arg("append"),
           R"pbdoc(
         Encode a single surface with SEI data and synchronization options.

         Encodes a single surface into a compressed video packet. The function may
         not return immediately with a compressed packet, depending on the encoder's
         internal buffering and the sync parameter.

         :param surface: Input surface containing the frame to encode
         :type surface: Surface
         :param packet: Output buffer for the compressed video packet
         :type packet: numpy.ndarray
         :param sei: Optional SEI (Supplemental Enhancement Information) data to attach
         :type sei: numpy.ndarray
         :param sync: Whether to wait for the encoded packet before returning
         :type sync: bool
         :param append: Whether to append the new packet to existing data
         :type append: bool
         :return: True if encoding was successful, False otherwise
         :rtype: bool
         :raises RuntimeError: If encoding fails
     )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array&, const py::array&,
                             bool>(&PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           py::arg("sync"),
           R"pbdoc(
         Encode a single surface with SEI data.

         Encodes a single surface into a compressed video packet with optional
         SEI data. The sync parameter determines whether to wait for the encoded
         packet before returning.

         :param surface: Input surface containing the frame to encode
         :type surface: Surface
         :param packet: Output buffer for the compressed video packet
         :type packet: numpy.ndarray
         :param sei: Optional SEI data to attach to the encoded frame
         :type sei: numpy.ndarray
         :param sync: Whether to wait for the encoded packet before returning
         :type sync: bool
         :return: True if encoding was successful, False otherwise
         :rtype: bool
         :raises RuntimeError: If encoding fails
     )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array&, bool>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sync"),
           R"pbdoc(
         Encode a single surface with synchronization option.

         Encodes a single surface into a compressed video packet. The sync parameter
         determines whether to wait for the encoded packet before returning.

         :param surface: Input surface containing the frame to encode
         :type surface: Surface
         :param packet: Output buffer for the compressed video packet
         :type packet: numpy.ndarray
         :param sync: Whether to wait for the encoded packet before returning
         :type sync: bool
         :return: True if encoding was successful, False otherwise
         :rtype: bool
         :raises RuntimeError: If encoding fails
     )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array&, const py::array&>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"), py::arg("sei"),
           R"pbdoc(
         Encode a single surface with SEI data.

         Encodes a single surface into a compressed video packet with optional
         SEI data. The function operates asynchronously by default.

         :param surface: Input surface containing the frame to encode
         :type surface: Surface
         :param packet: Output buffer for the compressed video packet
         :type packet: numpy.ndarray
         :param sei: Optional SEI data to attach to the encoded frame
         :type sei: numpy.ndarray
         :return: True if encoding was successful, False otherwise
         :rtype: bool
         :raises RuntimeError: If encoding fails
     )pbdoc")
      .def("EncodeSingleSurface",
           py::overload_cast<shared_ptr<Surface>, py::array&>(
               &PyNvEncoder::EncodeSurface),
           py::arg("surface"), py::arg("packet"),
           R"pbdoc(
         Encode a single surface.

         Encodes a single surface into a compressed video packet. The function
         operates asynchronously by default.

         :param surface: Input surface containing the frame to encode
         :type surface: Surface
         :param packet: Output buffer for the compressed video packet
         :type packet: numpy.ndarray
         :return: True if encoding was successful, False otherwise
         :rtype: bool
         :raises RuntimeError: If encoding fails
     )pbdoc")
      .def("Flush", &PyNvEncoder::Flush, py::arg("packets"),
           R"pbdoc(
         Flush the encoder's internal buffers.

         Forces the encoder to process any remaining frames in its internal
         buffers and output them as compressed packets.

         :param packets: Output buffer for the compressed packets
         :type packets: numpy.ndarray
         :return: True if any packets were flushed, False otherwise
         :rtype: bool
         :raises RuntimeError: If flushing fails
     )pbdoc");
}
