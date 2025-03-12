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

#pragma once

#include "LibraryLoader.hpp"

#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_data_exchange_and_initialization.h>
#include <nppi_geometry_transforms.h>
#include <npps_arithmetic_and_logical_operations.h>

class LibNpp {
private:
  static const char* const filenames[];
  static std::shared_ptr<LibraryLoader> LoadNppIg();
  static std::shared_ptr<LibraryLoader> LoadNppIcc();
  static std::shared_ptr<LibraryLoader> LoadNppIdei();
  static std::shared_ptr<LibraryLoader> LoadNppIal();

public:
  // nppi_geometry_transforms.h (nppig64_11.dll):
  static LoadableFunction<LoadNppIg, NppStatus, const Npp8u*, int, NppiSize,
                          NppiRect, Npp8u*, int, NppiSize, NppiRect, int,
                          NppStreamContext>
      nppiResize_8u_C3R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp8u*, int, NppiSize,
                          NppiRect, Npp8u*, int, NppiSize, NppiRect, int,
                          NppStreamContext>
      nppiResize_8u_C1R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp32f*, int, NppiSize,
                          NppiRect, Npp32f*, int, NppiSize, NppiRect, int,
                          NppStreamContext>
      nppiResize_32f_C3R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp32f*, int, NppiSize,
                          NppiRect, Npp32f*, int, NppiSize, NppiRect, int,
                          NppStreamContext>
      nppiResize_32f_C1R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp8u*, NppiSize, int,
                          NppiRect, const Npp32f*, int, const Npp32f*, int,
                          Npp8u*, int, NppiSize, int, NppStreamContext>
      nppiRemap_8u_C3R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp8u*, NppiSize, int,
                          NppiRect, Npp8u*, int, NppiRect, double, double,
                          double, int, NppStreamContext>
      nppiRotate_8u_C1R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp16u*, NppiSize, int,
                          NppiRect, Npp16u*, int, NppiRect, double, double,
                          double, int, NppStreamContext>
      nppiRotate_16u_C1R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp32f*, NppiSize, int,
                          NppiRect, Npp32f*, int, NppiRect, double, double,
                          double, int, NppStreamContext>
      nppiRotate_32f_C1R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp8u*, NppiSize, int,
                          NppiRect, Npp8u*, int, NppiRect, double, double,
                          double, int, NppStreamContext>
      nppiRotate_8u_C3R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp16u*, NppiSize, int,
                          NppiRect, Npp16u*, int, NppiRect, double, double,
                          double, int, NppStreamContext>
      nppiRotate_16u_C3R_Ctx;
  static LoadableFunction<LoadNppIg, NppStatus, const Npp32f*, NppiSize, int,
                          NppiRect, Npp32f*, int, NppiRect, double, double,
                          double, int, NppStreamContext>
      nppiRotate_32f_C3R_Ctx;

  // nppi_color_conversion.h (nppicc64_11.dll):
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[2], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[2], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[2], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiNV12ToBGR_8u_P2C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[2], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[2], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[2], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiNV12ToRGB_8u_P2C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[2], int,
                          Npp8u* [3], int[3], NppiSize, NppStreamContext>
      nppiNV12ToYUV420_8u_P2P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const, int,
                          const Npp8u*, int, Npp8u* [3], int[3], NppiSize,
                          NppStreamContext>
      nppiYCbCr420_8u_P2P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u*, int,
                          NppiSize, NppStreamContext>
      nppiRGBToGray_8u_C3C1R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int[3],
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiYUV420ToRGB_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int[3],
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiYCbCr420ToRGB_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int[3],
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiYUV420ToBGR_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int[3],
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiYCbCr420ToBGR_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiYCbCrToBGR_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiYUVToBGR_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiYUVToRGB_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int,
                          Npp8u* [3], int, NppiSize, NppStreamContext>
      nppiYUVToRGB_8u_P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u* [3],
                          int, NppiSize, NppStreamContext>
      nppiBGRToYCbCr_8u_C3P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u* [3],
                          int, NppiSize, NppStreamContext>
      nppiBGRToYUV_8u_C3P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u* [3],
                          int[3], NppiSize, NppStreamContext>
      nppiBGRToYCbCr420_8u_C3P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u* [3],
                          int, NppiSize, NppStreamContext>
      nppiRGBToYUV_8u_C3P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u*, int,
                          NppiSize, NppStreamContext>
      nppiRGBToYCbCr_8u_C3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int,
                          Npp8u* [3], int, NppiSize, NppStreamContext>
      nppiRGBToYUV_8u_P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int,
                          Npp8u* [3], int, NppiSize, NppStreamContext>
      nppiRGBToYCbCr_8u_P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u* [3],
                          int[3], NppiSize, NppStreamContext>
      nppiRGBToYUV420_8u_C3P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u*, int, Npp8u* [3],
                          int[3], NppiSize, NppStreamContext>
      nppiRGBToYCbCr420_8u_C3P3R_Ctx;
  static LoadableFunction<LoadNppIcc, NppStatus, const Npp8u* const[3], int[3],
                          Npp8u*, int, Npp8u*, int, NppiSize, NppStreamContext>
      nppiYCbCr420_8u_P3P2R_Ctx;

  // nppi_data_exchange_and_initialization.h (nppidei64_11.dll):
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp8u, Npp8u*, int,
                          NppiSize, NppStreamContext>
      nppiSet_8u_C1R_Ctx;
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp8u*, int, Npp8u*,
                          int, NppiSize, NppStreamContext>
      nppiCopy_8u_C1R_Ctx;
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp16u*, int, Npp8u*,
                          int, NppiSize, NppStreamContext>
      nppiConvert_16u8u_C1R_Ctx;
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp8u*, int,
                          Npp8u* const[3], int, NppiSize, NppStreamContext>
      nppiCopy_8u_C3P3R_Ctx;
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp8u* const[3], int,
                          Npp8u*, int, NppiSize, NppStreamContext>
      nppiCopy_8u_P3C3R_Ctx;
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp8u*, int, Npp8u*,
                          int, NppiSize, const int[3], NppStreamContext>
      nppiSwapChannels_8u_C3R_Ctx;
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp8u*, int, Npp32f*,
                          int, NppiSize, Npp32f, Npp32f, NppStreamContext>
      nppiScale_8u32f_C3R_Ctx;
  static LoadableFunction<LoadNppIdei, NppStatus, const Npp32f*, int,
                          Npp32f* const[3], int, NppiSize, NppStreamContext>
      nppiCopy_32f_C3P3R_Ctx;

  // nppi_arithmetic_and_logical_operations.h (nppial64_11.dll):
  static LoadableFunction<LoadNppIal, NppStatus, const Npp16u*, int,
                          const Npp16u, Npp16u*, int, NppiSize, int,
                          NppStreamContext>
      nppiDivC_16u_C1RSfs_Ctx;
};
