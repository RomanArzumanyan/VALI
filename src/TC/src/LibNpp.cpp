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

#include "LibNpp.hpp"
#include "CudaUtils.hpp"
#include <sstream>
#include <string>

static const std::string getDynLibName(const char* nppLibName) {
  std::stringstream ss;
  auto const cuda_version = VPF::CudaResMgr::Instance().GetVersion() / 1000;

#ifdef _WIN64
  ss << nppLibName << "64_" << cuda_version << ".dll";
#else
  ss << "lib" << nppLibName << ".so";
#endif
  return ss.str();
}

std::shared_ptr<LibraryLoader> LibNpp::LoadNppIg() {
  auto const filename = getDynLibName("nppig");
  static LibraryLoader lib(filename.c_str());
  return std::shared_ptr<LibraryLoader>(std::shared_ptr<LibraryLoader>{}, &lib);
}
std::shared_ptr<LibraryLoader> LibNpp::LoadNppIcc() {
  auto const filename = getDynLibName("nppicc");
  static LibraryLoader lib(filename.c_str());
  return std::shared_ptr<LibraryLoader>(std::shared_ptr<LibraryLoader>{}, &lib);
}
std::shared_ptr<LibraryLoader> LibNpp::LoadNppIdei() {
  auto const filename = getDynLibName("nppidei");
  static LibraryLoader lib(filename.c_str());
  return std::shared_ptr<LibraryLoader>(std::shared_ptr<LibraryLoader>{}, &lib);
}
std::shared_ptr<LibraryLoader> LibNpp::LoadNppIal() {
  auto const filename = getDynLibName("nppial");
  static LibraryLoader lib(filename.c_str());
  return std::shared_ptr<LibraryLoader>(std::shared_ptr<LibraryLoader>{}, &lib);
}

#define DEFINE(x, y)                                                           \
  decltype(x::y) x::y { #y }

// Define function pointers for nppi_geometry_transforms.h (nppig64_11.dll):
DEFINE(LibNpp, nppiResize_8u_C3R_Ctx);
DEFINE(LibNpp, nppiResize_8u_C1R_Ctx);
DEFINE(LibNpp, nppiResize_32f_C3R_Ctx);
DEFINE(LibNpp, nppiResize_32f_C1R_Ctx);
DEFINE(LibNpp, nppiRemap_8u_C3R_Ctx);

// Define function pointers for nppi_color_conversion.h (nppicc64_11.dll):
DEFINE(LibNpp, nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx);
DEFINE(LibNpp, nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx);
DEFINE(LibNpp, nppiNV12ToBGR_8u_P2C3R_Ctx);
DEFINE(LibNpp, nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx);
DEFINE(LibNpp, nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx);
DEFINE(LibNpp, nppiNV12ToRGB_8u_P2C3R_Ctx);
DEFINE(LibNpp, nppiNV12ToYUV420_8u_P2P3R_Ctx);
DEFINE(LibNpp, nppiYCbCr420_8u_P2P3R_Ctx);
DEFINE(LibNpp, nppiRGBToGray_8u_C3C1R_Ctx);
DEFINE(LibNpp, nppiYUV420ToRGB_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiYCbCr420ToRGB_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiYUV420ToBGR_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiYCbCr420ToBGR_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiYCbCrToBGR_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiYUVToBGR_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiYUVToRGB_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiYUVToRGB_8u_P3R_Ctx);
DEFINE(LibNpp, nppiBGRToYCbCr_8u_C3P3R_Ctx);
DEFINE(LibNpp, nppiBGRToYUV_8u_C3P3R_Ctx);
DEFINE(LibNpp, nppiBGRToYCbCr420_8u_C3P3R_Ctx);
DEFINE(LibNpp, nppiRGBToYUV_8u_C3P3R_Ctx);
DEFINE(LibNpp, nppiRGBToYCbCr_8u_C3R_Ctx);
DEFINE(LibNpp, nppiRGBToYUV_8u_P3R_Ctx);
DEFINE(LibNpp, nppiRGBToYCbCr_8u_P3R_Ctx);
DEFINE(LibNpp, nppiRGBToYUV420_8u_C3P3R_Ctx);
DEFINE(LibNpp, nppiRGBToYCbCr420_8u_C3P3R_Ctx);
DEFINE(LibNpp, nppiYCbCr420_8u_P3P2R_Ctx);

// Define function pointers for nppi_data_exchange_and_initialization.h
// (nppidei64_11.dll):
DEFINE(LibNpp, nppiSet_8u_C1R_Ctx);
DEFINE(LibNpp, nppiCopy_8u_C1R_Ctx);
DEFINE(LibNpp, nppiConvert_16u8u_C1R_Ctx);
DEFINE(LibNpp, nppiCopy_8u_C3P3R_Ctx);
DEFINE(LibNpp, nppiCopy_8u_P3C3R_Ctx);
DEFINE(LibNpp, nppiSwapChannels_8u_C3R_Ctx);
DEFINE(LibNpp, nppiScale_8u32f_C3R_Ctx);
DEFINE(LibNpp, nppiCopy_32f_C3P3R_Ctx);

// Define function pointers for nppi_arithmetic_and_logical_operations.h
// (nppial64_11.dll):
DEFINE(LibNpp, nppiDivC_16u_C1RSfs_Ctx);
