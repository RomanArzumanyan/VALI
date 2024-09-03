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
#include "LibNvJpeg.hpp"
#include "CudaUtils.hpp"
#include <sstream>
#include <string>

static const std::string getDynLibName(const char* nvJpegLibName) {
  std::stringstream ss;
  auto const cuda_version = VPF::CudaResMgr::Instance().GetVersion() / 1000;

#ifdef _WIN64
  ss << nvJpegLibName << "64_" << cuda_version << ".dll";
#else
  ss << "lib" << nvJpegLibName << ".so";
#endif
  return ss.str();
}

std::shared_ptr<LibraryLoader> LibNvJpeg::Load() {
  auto const filename = getDynLibName("nvjpeg");
  static LibraryLoader lib(filename.c_str());
  return std::shared_ptr<LibraryLoader>(std::shared_ptr<LibraryLoader>{}, &lib);
}

#define DEFINE(x, y)                                                           \
  decltype(x::y) x::y { #y }

// Define function pointers for nvjpeg.h:
DEFINE(LibNvJpeg, nvjpegCreateSimple);
DEFINE(LibNvJpeg, nvjpegEncoderStateCreate);
DEFINE(LibNvJpeg, nvjpegEncoderParamsCreate);
DEFINE(LibNvJpeg, nvjpegEncoderParamsSetSamplingFactors);
DEFINE(LibNvJpeg, nvjpegEncoderParamsSetQuality);
DEFINE(LibNvJpeg, nvjpegEncodeImage);
DEFINE(LibNvJpeg, nvjpegEncodeYUV);
DEFINE(LibNvJpeg, nvjpegEncodeRetrieveBitstream);
DEFINE(LibNvJpeg, nvjpegEncoderParamsDestroy);
DEFINE(LibNvJpeg, nvjpegEncoderStateDestroy);
DEFINE(LibNvJpeg, nvjpegDestroy);
