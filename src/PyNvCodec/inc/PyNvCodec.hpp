/*
 * Copyright 2020 NVIDIA Corporation
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

#pragma once

#include "MemoryInterfaces.hpp"
#include "NvCodecCLIOptions.h"
#include "TC_CORE.hpp"
#include "Tasks.hpp"

#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <pybind11/cast.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/motion_vector.h>
}

using namespace VPF;
namespace py = pybind11;

extern int nvcvImagePitch; // global variable to hold pitch value

struct MotionVector {
  int source;
  int w, h;
  int src_x, src_y;
  int dst_x, dst_y;
  int motion_x, motion_y;
  int motion_scale;
};

class CudaResMgr {
private:
  CudaResMgr();

public:
  CudaResMgr(const CudaResMgr& other) = delete;
  CudaResMgr(const CudaResMgr&& other) = delete;
  CudaResMgr& operator=(CudaResMgr& other) = delete;
  CudaResMgr& operator=(CudaResMgr&& other) = delete;

  CUcontext GetCtx(size_t idx);
  CUstream GetStream(size_t idx);
  ~CudaResMgr();
  static CudaResMgr& Instance();
  static size_t GetNumGpus();

  std::vector<std::pair<CUdevice, CUcontext>> g_Contexts;
  std::vector<CUstream> g_Streams;

  static std::mutex gInsMutex;
  static std::mutex gCtxMutex;
  static std::mutex gStrMutex;
};

class PyFrameUploader {
  std::unique_ptr<CudaUploadFrame> m_uploader = nullptr;

public:
  PyFrameUploader(uint32_t gpu_id);
  PyFrameUploader(CUstream str);
  PyFrameUploader(size_t str) : PyFrameUploader((CUstream)str) {}

  bool Run(py::array& src, Surface& dst, TaskExecDetails details);
};

class PySurfaceDownloader {
  std::unique_ptr<CudaDownloadSurface> upDownloader = nullptr;

public:
  PySurfaceDownloader(uint32_t gpu_id);
  PySurfaceDownloader(CUstream str);
  PySurfaceDownloader(size_t str) : PySurfaceDownloader((CUstream)str) {}

  bool Run(Surface& src, py::array& dst, TaskExecDetails& details);
};

class PySurfaceConverter {
  std::unique_ptr<ConvertSurface> upConverter = nullptr;
  std::unique_ptr<Buffer> upCtxBuffer = nullptr;

public:
  PySurfaceConverter(Pixel_Format src, Pixel_Format dst, uint32_t gpuID);
  PySurfaceConverter(Pixel_Format src, Pixel_Format dst, CUstream str);
  PySurfaceConverter(Pixel_Format src, Pixel_Format dst, size_t str)
      : PySurfaceConverter(src, dst, (CUstream)str) {}

  bool Run(Surface& src, Surface& dst, ColorspaceConversionContext& context,
           TaskExecDetails& details);
};

class PyFrameConverter {
  std::unique_ptr<ConvertFrame> m_up_cvt = nullptr;
  std::unique_ptr<Buffer> m_up_ctx_buf = nullptr;
  size_t m_width = 0U;
  size_t m_height = 0U;
  Pixel_Format m_src_fmt = Pixel_Format::UNDEFINED;
  Pixel_Format m_dst_fmt = Pixel_Format::UNDEFINED;

public:
  PyFrameConverter(uint32_t width, uint32_t height, Pixel_Format inFormat,
                   Pixel_Format outFormat);

  bool Run(py::array& src, py::array& dst,
           std::shared_ptr<ColorspaceConversionContext> context,
           TaskExecDetails& details);

  Pixel_Format GetFormat() const { return m_dst_fmt; }
};

class PySurfaceResizer {
  std::unique_ptr<ResizeSurface> upResizer = nullptr;

public:
  PySurfaceResizer(Pixel_Format format, uint32_t gpuID);
  PySurfaceResizer(Pixel_Format format, CUstream str);
  PySurfaceResizer(Pixel_Format format, size_t str)
      : PySurfaceResizer(format, (CUstream)str) {}

  bool Run(Surface& src, Surface& dst, TaskExecDetails& details);
};

class DecodeContext {
private:
  std::shared_ptr<Surface> pSurface;

  py::array_t<uint8_t>* pSei;
  py::array_t<uint8_t>* pPacket;

  PacketData* pInPktData;
  PacketData* pOutPktData;

  SeekContext* pSeekCtx;

  bool flush;

public:
  DecodeContext(py::array_t<uint8_t>* sei, py::array_t<uint8_t>* packet,
                PacketData* in_pkt_data, PacketData* out_pkt_data,
                SeekContext* seek_ctx, bool is_flush = false);

  bool IsSeek() const;
  bool IsStandalone() const;
  bool IsFlush() const;
  bool HasSEI() const;
  bool HasOutPktData() const;
  bool HasInPktData() const;

  const py::array_t<uint8_t>* GetPacket() const;
  const PacketData* GetInPacketData() const;
  const SeekContext* GetSeekContext() const;

  SeekContext* GetSeekContextMutable();
  std::shared_ptr<Surface> GetSurfaceMutable();

  void SetOutPacketData(PacketData* out_pkt_data);
  void SetOutPacketData(const PacketData& out_pkt_data);
  void SetSei(Buffer* sei);
  void SetCloneSurface(Surface* p_surface);
};

class PyDecoder {
  std::unique_ptr<DecodeFrame> upDecoder = nullptr;

  void* GetSideData(AVFrameSideDataType data_type, size_t& raw_size);

  uint32_t last_w;
  uint32_t last_h;
  int gpu_id;

  void UpdateState();

public:
  PyDecoder(const std::string& pathToFile,
            const std::map<std::string, std::string>& ffmpeg_options,
            int gpuID);

  bool DecodeSingleFrame(py::array& frame, TaskExecDetails& details,
                         PacketData& pkt_data,
                         std::optional<SeekContext> seek_ctx);

  bool DecodeSingleSurface(Surface& surf, TaskExecDetails& details,
                           PacketData& pkt_data,
                           std::optional<SeekContext> seek_ctx);

  std::vector<MotionVector> GetMotionVectors();

  uint32_t Width() const;
  uint32_t Height() const;
  uint32_t Level() const;
  uint32_t Profile() const;
  uint32_t Delay() const;
  uint32_t GopSize() const;
  uint32_t Bitrate() const;
  uint32_t NumFrames() const;
  uint32_t NumStreams() const;
  uint32_t StreamIndex() const;
  uint32_t HostFrameSize() const;

  double Framerate() const;
  double AvgFramerate() const;
  double Timebase() const;
  double StartTime() const;
  double Duration() const;

  ColorSpace Color_Space() const;
  ColorRange Color_Range() const;

  Pixel_Format PixelFormat() const;

  bool IsAccelerated() const;
  bool IsVFR() const;

  std::map<std::string, std::string> Metadata();

private:
  bool DecodeImpl(TaskExecDetails& details, PacketData& pkt_data, Token& dst,
                  std::optional<SeekContext> seek_ctx);
};

class PyNvEncoder {
  std::unique_ptr<NvencEncodeFrame> upEncoder;
  uint32_t encWidth, encHeight;
  Pixel_Format eFormat;
  std::map<std::string, std::string> options;
  bool verbose_ctor;
  CUstream cuda_str;

public:
  uint32_t Width() const;
  uint32_t Height() const;
  Pixel_Format GetPixelFormat() const;
  std::map<NV_ENC_CAPS, int> Capabilities();
  int GetFrameSizeInBytes() const;
  bool Reconfigure(const std::map<std::string, std::string>& encodeOptions,
                   bool force_idr = false, bool reset_enc = false,
                   bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string>& encodeOptions,
              int gpu_id, Pixel_Format format = NV12, bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string>& encodeOptions,
              CUstream str, Pixel_Format format = NV12, bool verbose = false);

  PyNvEncoder(const std::map<std::string, std::string>& encodeOptions,
              size_t str, Pixel_Format format = NV12, bool verbose = false)
      : PyNvEncoder(encodeOptions, (CUstream)str, format, verbose) {}

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t>& packet,
                     const py::array_t<uint8_t>& messageSEI, bool sync,
                     bool append);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t>& packet,
                     const py::array_t<uint8_t>& messageSEI, bool sync);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t>& packet, bool sync);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t>& packet,
                     const py::array_t<uint8_t>& messageSEI);

  bool EncodeSurface(std::shared_ptr<Surface> rawSurface,
                     py::array_t<uint8_t>& packet);

  // Flush all the encoded frames (packets)
  bool Flush(py::array_t<uint8_t>& packets);
  // Flush only one encoded frame (packet)
  bool FlushSinglePacket(py::array_t<uint8_t>& packet);

  static void CheckValidCUDABuffer(const void* ptr) {
    if (ptr == nullptr) {
      throw std::runtime_error("NULL CUDA buffer not accepted");
    }

    cudaPointerAttributes attrs = {};
    cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
    cudaGetLastError(); // reset the cuda error (if any)
    if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered) {
      throw std::runtime_error("Buffer is not CUDA-accessible");
    }
  }

private:
  bool EncodeSingleSurface(struct EncodeContext& ctx);
};