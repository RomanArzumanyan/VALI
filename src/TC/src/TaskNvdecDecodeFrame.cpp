/*
 * Copyright 2019 NVIDIA Corporation
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <sstream>
#include <vector>

#include "CodecsSupport.hpp"
#include "MemoryInterfaces.hpp"
#include "NvCodecUtils.h"
#include "NvDecoder.h"
#include "Tasks.hpp"
#include "nvcuvid.h"

using namespace std;
using namespace VPF;

static float GetChromaHeightFactor(cudaVideoChromaFormat eChromaFormat) {
  float factor = 0.5;
  switch (eChromaFormat) {
  case cudaVideoChromaFormat_Monochrome:
    factor = 0.0;
    break;
  case cudaVideoChromaFormat_420:
    factor = 0.5;
    break;
  case cudaVideoChromaFormat_422:
    factor = 1.0;
    break;
  case cudaVideoChromaFormat_444:
    factor = 1.0;
    break;
  }

  return factor;
}

static int GetChromaPlaneCount(cudaVideoChromaFormat eChromaFormat) {
  int numPlane;
  switch (eChromaFormat) {
  case cudaVideoChromaFormat_420:
    numPlane = 1;
    break;
  case cudaVideoChromaFormat_444:
    numPlane = 2;
    break;
  default:
    numPlane = 0;
    break;
  }

  return numPlane;
}

struct Rect {
  int l, t, r, b;
};

struct Dim {
  int w, h;
};

struct NvDecoderImpl {
  bool m_bReconfigExternal = false, m_bReconfigExtPPChange = false,
       eos_set = false;

  unsigned int m_nWidth = 0U, m_nLumaHeight = 0U, m_nChromaHeight = 0U,
               m_nNumChromaPlanes = 0U, m_nMaxWidth = 0U, m_nMaxHeight = 0U;

  int m_nSurfaceHeight = 0, m_nSurfaceWidth = 0, m_nBitDepthMinus8 = 0,
      m_nFrameAlloc = 0, m_nBPP = 1, m_nDecodePicCnt = 0,
      m_nPicNumInDecodeOrder[32] = {0};

  Rect m_displayRect = {}, m_cropRect = {};

  Dim m_resizeDim = {};

  size_t m_nDeviceFramePitch = 0;

  CUvideoctxlock m_ctxLock = nullptr;
  CUstream m_cuvidStream = nullptr;
  CUcontext m_cuContext = nullptr;
  CUvideoparser m_hParser = nullptr;
  CUvideodecoder m_hDecoder = nullptr;

  CUVIDEOFORMAT m_videoFormat = {};

  cudaVideoCodec m_eCodec = cudaVideoCodec_NumCodecs;
  cudaVideoChromaFormat m_eChromaFormat = cudaVideoChromaFormat_420;
  cudaVideoSurfaceFormat m_eOutputFormat = cudaVideoSurfaceFormat_NV12;

  vector<DecodedFrameContext> m_DecFramesCtxVec;
  queue<DecodedFrameContext> m_DecFramesCtxQueue;
  map<uint64_t, PacketData> in_pdata;
  map<uint64_t, PacketData> out_pdata;

  mutex m_mtxVPFrame;

  atomic<int> decode_error;
  atomic<int> parser_error;
  atomic<int> decoder_recon;
  atomic<int> m_nDecodedFrame;
  atomic<unsigned int> bit_stream_len;
};

cudaVideoCodec NvDecoder::GetCodec() const { return p_impl->m_eCodec; }

/* Return value from HandleVideoSequence() are interpreted as:
 *   0: fail
 *   1: success
 *   > 1: override dpb size of parser (set by
 * CUVIDPARSERPARAMS::ulMaxNumDecodeSurfaces while creating parser)
 */
int NvDecoder::HandleVideoSequence(CUVIDEOFORMAT* pVideoFormat) noexcept {
  try {
    p_impl->decoder_recon++;
    CudaCtxPush ctxPush(p_impl->m_cuContext);

    // Shall be enough according to NVIDIA Nvdec mem optimization blog article
    // (https://developer.nvidia.com/blog/optimizing-video-memory-usage-with-the-nvdecode-api-and-nvidia-video-codec-sdk/)
    int nDecodeSurface = pVideoFormat->min_num_decode_surfaces + 3;

    CUVIDDECODECAPS decodecaps;
    memset(&decodecaps, 0, sizeof(decodecaps));

    decodecaps.eCodecType = pVideoFormat->codec;
    decodecaps.eChromaFormat = pVideoFormat->chroma_format;
    decodecaps.nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;

    ThrowOnCudaError(m_api.cuvidGetDecoderCaps(&decodecaps), __LINE__);

    if (!decodecaps.bIsSupported) {
      throw runtime_error("Codec not supported on this GPU");
    }

    if ((pVideoFormat->coded_width > decodecaps.nMaxWidth) ||
        (pVideoFormat->coded_height > decodecaps.nMaxHeight)) {

      ostringstream errorString;
      errorString << endl
                  << "Resolution          : " << pVideoFormat->coded_width
                  << "x" << pVideoFormat->coded_height << endl
                  << "Max Supported (wxh) : " << decodecaps.nMaxWidth << "x"
                  << decodecaps.nMaxHeight << endl
                  << "Resolution not supported on this GPU";

      throw runtime_error(errorString.str());
    }

    if ((pVideoFormat->coded_width >> 4U) * (pVideoFormat->coded_height >> 4U) >
        decodecaps.nMaxMBCount) {

      ostringstream errorString;
      errorString << endl
                  << "MBCount             : "
                  << (pVideoFormat->coded_width >> 4U) *
                         (pVideoFormat->coded_height >> 4U)
                  << endl
                  << "Max Supported mbcnt : " << decodecaps.nMaxMBCount << endl
                  << "MBCount not supported on this GPU";

      throw runtime_error(errorString.str());
    }

    if (p_impl->m_nWidth && p_impl->m_nLumaHeight && p_impl->m_nChromaHeight) {

      // cuvidCreateDecoder() has been called before, and now there's possible
      // config change
      return ReconfigureDecoder(pVideoFormat);
    }

    // eCodec has been set in the constructor (for parser). Here it's set again
    // for potential correction
    p_impl->m_eCodec = pVideoFormat->codec;
    p_impl->m_eChromaFormat = pVideoFormat->chroma_format;
    p_impl->m_nBitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    p_impl->m_nBPP = p_impl->m_nBitDepthMinus8 > 0 ? 2 : 1;

    if (p_impl->m_eChromaFormat == cudaVideoChromaFormat_420)
      p_impl->m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8
                                    ? cudaVideoSurfaceFormat_P016
                                    : cudaVideoSurfaceFormat_NV12;
    else if (p_impl->m_eChromaFormat == cudaVideoChromaFormat_444)
      p_impl->m_eOutputFormat = pVideoFormat->bit_depth_luma_minus8
                                    ? cudaVideoSurfaceFormat_YUV444_16Bit
                                    : cudaVideoSurfaceFormat_YUV444;

    p_impl->m_videoFormat = *pVideoFormat;

    CUVIDDECODECREATEINFO videoDecodeCreateInfo = {0};
    videoDecodeCreateInfo.CodecType = pVideoFormat->codec;
    videoDecodeCreateInfo.ChromaFormat = pVideoFormat->chroma_format;
    videoDecodeCreateInfo.OutputFormat = p_impl->m_eOutputFormat;
    videoDecodeCreateInfo.bitDepthMinus8 = pVideoFormat->bit_depth_luma_minus8;
    videoDecodeCreateInfo.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
    videoDecodeCreateInfo.ulNumOutputSurfaces = 2;
    // With PreferCUVID, JPEG is still decoded by CUDA while video is decoded by
    // NVDEC hardware
    videoDecodeCreateInfo.ulCreationFlags = cudaVideoCreate_PreferCUVID;
    videoDecodeCreateInfo.ulNumDecodeSurfaces = nDecodeSurface;
    videoDecodeCreateInfo.vidLock = p_impl->m_ctxLock;
    videoDecodeCreateInfo.ulWidth = pVideoFormat->coded_width;
    videoDecodeCreateInfo.ulHeight = pVideoFormat->coded_height;
    if (p_impl->m_nMaxWidth < (int)pVideoFormat->coded_width)
      p_impl->m_nMaxWidth = pVideoFormat->coded_width;
    if (p_impl->m_nMaxHeight < (int)pVideoFormat->coded_height)
      p_impl->m_nMaxHeight = pVideoFormat->coded_height;
    videoDecodeCreateInfo.ulMaxWidth = p_impl->m_nMaxWidth;
    videoDecodeCreateInfo.ulMaxHeight = p_impl->m_nMaxHeight;

    if (!(p_impl->m_cropRect.r && p_impl->m_cropRect.b) &&
        !(p_impl->m_resizeDim.w && p_impl->m_resizeDim.h)) {
      p_impl->m_nWidth =
          pVideoFormat->display_area.right - pVideoFormat->display_area.left;
      p_impl->m_nLumaHeight =
          pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
      videoDecodeCreateInfo.ulTargetWidth = pVideoFormat->coded_width;
      videoDecodeCreateInfo.ulTargetHeight = pVideoFormat->coded_height;
    } else {
      if (p_impl->m_resizeDim.w && p_impl->m_resizeDim.h) {
        videoDecodeCreateInfo.display_area.left =
            pVideoFormat->display_area.left;
        videoDecodeCreateInfo.display_area.top = pVideoFormat->display_area.top;
        videoDecodeCreateInfo.display_area.right =
            pVideoFormat->display_area.right;
        videoDecodeCreateInfo.display_area.bottom =
            pVideoFormat->display_area.bottom;
        p_impl->m_nWidth = p_impl->m_resizeDim.w;
        p_impl->m_nLumaHeight = p_impl->m_resizeDim.h;
      }

      if (p_impl->m_cropRect.r && p_impl->m_cropRect.b) {
        videoDecodeCreateInfo.display_area.left = p_impl->m_cropRect.l;
        videoDecodeCreateInfo.display_area.top = p_impl->m_cropRect.t;
        videoDecodeCreateInfo.display_area.right = p_impl->m_cropRect.r;
        videoDecodeCreateInfo.display_area.bottom = p_impl->m_cropRect.b;
        p_impl->m_nWidth = p_impl->m_cropRect.r - p_impl->m_cropRect.l;
        p_impl->m_nLumaHeight = p_impl->m_cropRect.b - p_impl->m_cropRect.t;
      }
      videoDecodeCreateInfo.ulTargetWidth = p_impl->m_nWidth;
      videoDecodeCreateInfo.ulTargetHeight = p_impl->m_nLumaHeight;
    }

    p_impl->m_nChromaHeight =
        (int)(p_impl->m_nLumaHeight *
              GetChromaHeightFactor(videoDecodeCreateInfo.ChromaFormat));
    p_impl->m_nNumChromaPlanes =
        GetChromaPlaneCount(videoDecodeCreateInfo.ChromaFormat);
    p_impl->m_nSurfaceHeight = videoDecodeCreateInfo.ulTargetHeight;
    p_impl->m_nSurfaceWidth = videoDecodeCreateInfo.ulTargetWidth;
    p_impl->m_displayRect.b = videoDecodeCreateInfo.display_area.bottom;
    p_impl->m_displayRect.t = videoDecodeCreateInfo.display_area.top;
    p_impl->m_displayRect.l = videoDecodeCreateInfo.display_area.left;
    p_impl->m_displayRect.r = videoDecodeCreateInfo.display_area.right;

    ThrowOnCudaError(
        m_api.cuvidCreateDecoder(&p_impl->m_hDecoder, &videoDecodeCreateInfo),
        __LINE__);

    return nDecodeSurface;
  } catch (exception& e) {
    cerr << e.what() << endl;
    p_impl->parser_error.store(1);
  }

  return 0;
}

int NvDecoder::ReconfigureDecoder(CUVIDEOFORMAT* pVideoFormat) {
  CudaCtxPush ctxPush(p_impl->m_cuContext);

  p_impl->eos_set = false;

  if (pVideoFormat->bit_depth_luma_minus8 !=
          p_impl->m_videoFormat.bit_depth_luma_minus8 ||
      pVideoFormat->bit_depth_chroma_minus8 !=
          p_impl->m_videoFormat.bit_depth_chroma_minus8) {
    throw runtime_error("Reconfigure Not supported for bit depth change");
  }

  if (pVideoFormat->chroma_format != p_impl->m_videoFormat.chroma_format) {
    throw runtime_error("Reconfigure Not supported for chroma format change");
  }

  bool bDecodeResChange =
      !(pVideoFormat->coded_width == p_impl->m_videoFormat.coded_width &&
        pVideoFormat->coded_height == p_impl->m_videoFormat.coded_height);
  bool bDisplayRectChange = !(pVideoFormat->display_area.bottom ==
                                  p_impl->m_videoFormat.display_area.bottom &&
                              pVideoFormat->display_area.top ==
                                  p_impl->m_videoFormat.display_area.top &&
                              pVideoFormat->display_area.left ==
                                  p_impl->m_videoFormat.display_area.left &&
                              pVideoFormat->display_area.right ==
                                  p_impl->m_videoFormat.display_area.right);

  int nDecodeSurface = pVideoFormat->min_num_decode_surfaces + 3;

  if ((pVideoFormat->coded_width > p_impl->m_nMaxWidth) ||
      (pVideoFormat->coded_height > p_impl->m_nMaxHeight)) {
    // For VP9, let driver  handle the change if new width/height >
    // maxwidth/maxheight
    if ((p_impl->m_eCodec != cudaVideoCodec_VP9) ||
        p_impl->m_bReconfigExternal) {
      throw runtime_error(
          "Reconfigure Not supported when width/height > maxwidth/maxheight");
    }
    return 1;
  }

  if (!bDecodeResChange && !p_impl->m_bReconfigExtPPChange) {
    // if the coded_width/coded_height hasn't changed but display resolution has
    // changed, then need to update width/height for correct output without
    // cropping. Example : 1920x1080 vs 1920x1088
    if (bDisplayRectChange) {
      p_impl->m_nWidth =
          pVideoFormat->display_area.right - pVideoFormat->display_area.left;
      p_impl->m_nLumaHeight =
          pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
      p_impl->m_nChromaHeight =
          int(p_impl->m_nLumaHeight *
              GetChromaHeightFactor(pVideoFormat->chroma_format));
      p_impl->m_nNumChromaPlanes =
          GetChromaPlaneCount(pVideoFormat->chroma_format);
    }

    // no need for reconfigureDecoder(). Just return
    return 1;
  }

  CUVIDRECONFIGUREDECODERINFO reconfigParams = {0};

  reconfigParams.ulWidth = p_impl->m_videoFormat.coded_width =
      pVideoFormat->coded_width;
  reconfigParams.ulHeight = p_impl->m_videoFormat.coded_height =
      pVideoFormat->coded_height;

  reconfigParams.ulTargetWidth = p_impl->m_nSurfaceWidth;
  reconfigParams.ulTargetHeight = p_impl->m_nSurfaceHeight;

  if (bDecodeResChange) {
    p_impl->m_nWidth =
        pVideoFormat->display_area.right - pVideoFormat->display_area.left;
    p_impl->m_nLumaHeight =
        pVideoFormat->display_area.bottom - pVideoFormat->display_area.top;
    p_impl->m_nChromaHeight =
        int(p_impl->m_nLumaHeight *
            GetChromaHeightFactor(pVideoFormat->chroma_format));
  }

  // If external reconfigure is called along with resolution change even if post
  // processing params is not changed, do full reconfigure params update
  if ((p_impl->m_bReconfigExternal && bDecodeResChange) ||
      p_impl->m_bReconfigExtPPChange) {
    // update display rect and target resolution if requested explicitly.
    p_impl->m_bReconfigExternal = false;
    p_impl->m_bReconfigExtPPChange = false;
    p_impl->m_videoFormat = *pVideoFormat;

    p_impl->m_nChromaHeight =
        int(p_impl->m_nLumaHeight *
            GetChromaHeightFactor(pVideoFormat->chroma_format));
    p_impl->m_nNumChromaPlanes =
        GetChromaPlaneCount(pVideoFormat->chroma_format);
    p_impl->m_nSurfaceHeight = reconfigParams.ulTargetHeight;
    p_impl->m_nSurfaceWidth = reconfigParams.ulTargetWidth;
  }

  reconfigParams.ulNumDecodeSurfaces = nDecodeSurface;

  ThrowOnCudaError(
      m_api.cuvidReconfigureDecoder(p_impl->m_hDecoder, &reconfigParams),
      __LINE__);

  return nDecodeSurface;
}

/* Return value from HandlePictureDecode() are interpreted as:
 *   0: fail
 *   >=1: suceeded
 */
int NvDecoder::HandlePictureDecode(CUVIDPICPARAMS* pPicParams) noexcept {
  try {
    p_impl->bit_stream_len.fetch_add(pPicParams->nBitstreamDataLen);

    CudaCtxPush ctxPush(p_impl->m_cuContext);

    if (!p_impl->m_hDecoder) {
      throw runtime_error("Decoder not initialized.");
    }

    p_impl->m_nPicNumInDecodeOrder[pPicParams->CurrPicIdx] =
        p_impl->m_nDecodePicCnt++;
    ThrowOnCudaError(m_api.cuvidDecodePicture(p_impl->m_hDecoder, pPicParams),
                     __LINE__);

    return 1;
  } catch (exception& e) {
    cerr << e.what();
    p_impl->parser_error.store(1);
  }
  return 0;
}

/* Return value from HandlePictureDisplay() are interpreted as:
 *   0: fail
 *   >=1: suceeded
 */
int NvDecoder::HandlePictureDisplay(CUVIDPARSERDISPINFO* pDispInfo) noexcept {
  try {
    CudaCtxPush ctxPush(p_impl->m_cuContext);

    CUVIDPROCPARAMS videoProcParams = {};
    videoProcParams.progressive_frame = pDispInfo->progressive_frame;
    videoProcParams.second_field = pDispInfo->repeat_first_field + 1;
    videoProcParams.top_field_first = pDispInfo->top_field_first;
    videoProcParams.unpaired_field = pDispInfo->repeat_first_field < 0;
    videoProcParams.output_stream = p_impl->m_cuvidStream;

    CUdeviceptr dpSrcFrame = 0;
    unsigned int nSrcPitch = 0;
    ThrowOnCudaError(
        m_api.cuvidMapVideoFrame64(p_impl->m_hDecoder, pDispInfo->picture_index,
                                   &dpSrcFrame, &nSrcPitch, &videoProcParams),
        __LINE__);

    CUVIDGETDECODESTATUS DecodeStatus;
    memset(&DecodeStatus, 0, sizeof(DecodeStatus));

    auto result = m_api.cuvidGetDecodeStatus(
        p_impl->m_hDecoder, pDispInfo->picture_index, &DecodeStatus);

    bool isStatusErr =
        (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error) ||
        (DecodeStatus.decodeStatus == cuvidDecodeStatus_Error_Concealed);

    if (result == CUDA_SUCCESS && isStatusErr) {
      auto pic_num = p_impl->m_nPicNumInDecodeOrder[pDispInfo->picture_index];
      stringstream ss;
      ss << "Decode Error occurred for picture " << pic_num << endl;
      p_impl->decode_error.store(1);
      throw runtime_error(ss.str());
    }

    CUdeviceptr pDecodedFrame = 0;
    int64_t pDecodedFrameIdx = 0;
    {
      lock_guard<mutex> lock(p_impl->m_mtxVPFrame);
      p_impl->m_nDecodedFrame++;
      bool isNotEnoughFrames =
          (p_impl->m_nDecodedFrame > p_impl->m_DecFramesCtxVec.size());

      if (isNotEnoughFrames) {
        p_impl->m_nFrameAlloc++;
        CUdeviceptr pFrame = 0;

        auto const height =
            p_impl->m_nLumaHeight +
            p_impl->m_nChromaHeight * p_impl->m_nNumChromaPlanes;

        ThrowOnCudaError(cuMemAllocPitch(&pFrame, &p_impl->m_nDeviceFramePitch,
                                         p_impl->m_nWidth * p_impl->m_nBPP,
                                         height, 16),
                         __LINE__);

        p_impl->m_DecFramesCtxVec.push_back(DecodedFrameContext(
            pFrame, pDispInfo->timestamp, pDispInfo->picture_index));
      }
      pDecodedFrameIdx = p_impl->m_nDecodedFrame - 1;
      pDecodedFrame = p_impl->m_DecFramesCtxVec[pDecodedFrameIdx].mem;

      auto input_it = p_impl->in_pdata.find(pDispInfo->timestamp);
      PacketData ready_pkt_data;
      bool have_pkt_data = false;
      if (p_impl->in_pdata.end() != input_it) {
        ready_pkt_data = input_it->second;
        have_pkt_data = true;
        p_impl->in_pdata.erase(input_it);
      }

      if (have_pkt_data) {
        auto output_it = p_impl->out_pdata.find(pDispInfo->timestamp);
        if (p_impl->out_pdata.end() == output_it) {
          p_impl->out_pdata[pDispInfo->timestamp] = ready_pkt_data;
        }
      }
    }

    // Copy data from decoded frame;
    CUDA_MEMCPY2D m = {0};
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = dpSrcFrame;
    m.srcPitch = nSrcPitch;
    m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    m.dstDevice = pDecodedFrame;
    m.dstPitch = p_impl->m_nDeviceFramePitch
                     ? p_impl->m_nDeviceFramePitch
                     : p_impl->m_nWidth * p_impl->m_nBPP;
    m.WidthInBytes = p_impl->m_nWidth * p_impl->m_nBPP;
    m.Height = p_impl->m_nLumaHeight;
    ThrowOnCudaError(cuMemcpy2DAsync(&m, p_impl->m_cuvidStream), __LINE__);

    m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame +
                                m.srcPitch * p_impl->m_nSurfaceHeight);
    m.dstDevice = (CUdeviceptr)((uint8_t*)pDecodedFrame +
                                m.dstPitch * p_impl->m_nLumaHeight);
    m.Height = p_impl->m_nChromaHeight;
    ThrowOnCudaError(cuMemcpy2DAsync(&m, p_impl->m_cuvidStream), __LINE__);

    if (p_impl->m_nNumChromaPlanes == 2) {
      m.srcDevice = (CUdeviceptr)((uint8_t*)dpSrcFrame +
                                  m.srcPitch * p_impl->m_nSurfaceHeight * 2);
      m.dstDevice = (CUdeviceptr)((uint8_t*)pDecodedFrame +
                                  m.dstPitch * p_impl->m_nLumaHeight * 2);
      m.Height = p_impl->m_nChromaHeight;
      ThrowOnCudaError(cuMemcpy2DAsync(&m, p_impl->m_cuvidStream), __LINE__);
    }

    ThrowOnCudaError(m_api.cuvidUnmapVideoFrame(p_impl->m_hDecoder, dpSrcFrame),
                     __LINE__);

    // Copy timestamp and amount of bitsream consumed by decoder;
    p_impl->m_DecFramesCtxVec[pDecodedFrameIdx].pts = pDispInfo->timestamp;
    p_impl->m_DecFramesCtxVec[pDecodedFrameIdx].bsl =
        p_impl->bit_stream_len.exchange(0U);

    return 1;
  } catch (exception& e) {
    cerr << e.what();
    p_impl->parser_error.store(1);
  }
  return 0;
}

NvDecoder::NvDecoder(CUstream cuStream, CUcontext cuContext,
                     cudaVideoCodec eCodec, bool bLowLatency, int maxWidth,
                     int maxHeight) {
  const char* err = loadCuvidSymbols(&this->m_api,
#ifdef _WIN32
                                     "nvcuvid.dll");
#else
                                     "libnvcuvid.so.1");
#endif
  if (err) {
    constexpr const char* explanation =
#if defined(_WIN32)
        "Could not dynamically load nvcuvid.dll. Please ensure "
        "Nvidia Graphics drivers are correctly installed!";
#else
        "Could not dynamically load libnvcuvid.so.1. Please "
        "ensure Nvidia Graphics drivers are correctly installed!\n"
        "If using Docker please make sure that your Docker image was "
        "launched with \"video\" driver capabilty (see "
        "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        "user-guide.html#driver-capabilities)";
#endif
    auto description = tc_dlerror();
    if (description) {
      throw std::runtime_error(std::string(err) + ": " +
                               std::string(description) + "\n" + explanation);
    } else {
      throw std::runtime_error(std::string(err) + "\n" + explanation);
    }
  }
  p_impl = new NvDecoderImpl();
  p_impl->m_cuvidStream = cuStream;
  p_impl->m_cuContext = cuContext;
  p_impl->m_eCodec = eCodec;
  p_impl->m_nMaxWidth = maxWidth;
  p_impl->m_nMaxHeight = maxHeight;
  p_impl->decode_error.store(0);
  p_impl->parser_error.store(0);

  ThrowOnCudaError(m_api.cuvidCtxLockCreate(&p_impl->m_ctxLock, cuContext),
                   __LINE__);

  CUVIDPARSERPARAMS videoParserParameters = {};
  videoParserParameters.CodecType = eCodec;
  videoParserParameters.ulMaxNumDecodeSurfaces = 1;
  videoParserParameters.ulMaxDisplayDelay = bLowLatency ? 0 : 1;
  videoParserParameters.pUserData = this;
  videoParserParameters.pfnSequenceCallback = HandleVideoSequenceProc;
  videoParserParameters.pfnDecodePicture = HandlePictureDecodeProc;
  videoParserParameters.pfnDisplayPicture = HandlePictureDisplayProc;
  ThrowOnCudaError(
      m_api.cuvidCreateVideoParser(&p_impl->m_hParser, &videoParserParameters),
      __LINE__);
}

NvDecoder::~NvDecoder() {
  CudaCtxPush ctxPush(p_impl->m_cuContext);

  if (p_impl->m_hParser) {
    m_api.cuvidDestroyVideoParser(p_impl->m_hParser);
  }

  if (p_impl->m_hDecoder) {
    m_api.cuvidDestroyDecoder(p_impl->m_hDecoder);
  }

  {
    lock_guard<mutex> lock(p_impl->m_mtxVPFrame);
    // Return all surfaces to m_vpFrame;
    while (!p_impl->m_DecFramesCtxQueue.empty()) {
      auto& surface = p_impl->m_DecFramesCtxQueue.front();
      p_impl->m_DecFramesCtxQueue.pop();
      p_impl->m_DecFramesCtxVec.push_back(surface);
    }

    for (auto& dec_frame_ctx : p_impl->m_DecFramesCtxVec) {
      cuMemFree(dec_frame_ctx.mem);
    }
  }

  m_api.cuvidCtxLockDestroy(p_impl->m_ctxLock);
  delete p_impl;
  unloadCuvidSymbols(&this->m_api);
}

int NvDecoder::GetWidth() { return p_impl->m_nWidth; }

int NvDecoder::GetHeight() { return p_impl->m_nLumaHeight; }

int NvDecoder::GetChromaHeight() {
  return p_impl->m_nChromaHeight * p_impl->m_nNumChromaPlanes;
}

int NvDecoder::GetFrameSize() {
  auto const num_pixels =
      p_impl->m_nWidth * (p_impl->m_nLumaHeight +
                          p_impl->m_nChromaHeight * p_impl->m_nNumChromaPlanes);

  return num_pixels * p_impl->m_nBPP;
}

int NvDecoder::GetDeviceFramePitch() {
  return p_impl->m_nDeviceFramePitch ? (int)p_impl->m_nDeviceFramePitch
                                     : p_impl->m_nWidth * p_impl->m_nBPP;
}

int NvDecoder::GetBitDepth() { return p_impl->m_nBitDepthMinus8 + 8; }

cudaVideoChromaFormat NvDecoder::GetChromaFormat() const {
  return p_impl->m_eChromaFormat;
}

bool NvDecoder::DecodeLockSurface(Buffer const* encFrame,
                                  PacketData const& pdata,
                                  DecodedFrameContext& decCtx) {
  if (!p_impl->m_hParser) {
    throw runtime_error("Parser not initialized.");
  }

  if (1 == p_impl->decode_error.load()) {
    throw decoder_error("HW decoder faced error. Re-create instance.");
  }

  if (1 == p_impl->parser_error.load()) {
    throw cuvid_parser_error("Cuvid parser faced error.");
  }

  // Prepare CUVID packet with elementary bitstream;
  CUVIDSOURCEDATAPACKET packet = {0};
  packet.payload =
      encFrame ? encFrame->GetDataAs<const unsigned char>() : nullptr;
  packet.payload_size = encFrame ? encFrame->GetRawMemSize() : 0U;
  packet.flags = CUVID_PKT_TIMESTAMP;
  packet.timestamp = pdata.pts;
  if (!decCtx.no_eos &&
      (nullptr == packet.payload || 0 == packet.payload_size)) {
    packet.flags |= CUVID_PKT_ENDOFSTREAM;
    p_impl->eos_set = true;
  }

  // Kick off HW decoding;
  ThrowOnCudaError(m_api.cuvidParseVideoData(p_impl->m_hParser, &packet),
                   __LINE__);

  lock_guard<mutex> lock(p_impl->m_mtxVPFrame);
  /* Add incoming packet data to map;
   */
  auto it = p_impl->in_pdata.find(pdata.pts);
  if (p_impl->in_pdata.end() != it) {
#if 0
    cerr << "Incoming packet with pts " << pdata.pts
         << " already exists in the queue" << endl;
#endif
  } else {
    p_impl->in_pdata[pdata.pts] = pdata;
  }

  /* Move all decoded surfaces from decoder-owned pool to queue of frames ready
   * for display;
   */
  while (p_impl->m_nDecodedFrame > 0) {
    p_impl->m_DecFramesCtxQueue.push(p_impl->m_DecFramesCtxVec.front());
    p_impl->m_DecFramesCtxVec.erase(p_impl->m_DecFramesCtxVec.begin());
    p_impl->m_nDecodedFrame--;
  }

  /* Multiple frames may be ready for display.
   * We return either 0 or 1 frame.
   */
  auto ret = false;

  // Prepare blank packet data in case no frames are decoded yet;
  memset(&decCtx.out_pdata, 0, sizeof(decCtx.out_pdata));

  /* In case decoder was reconfigured by cuvidParseVideoData() call made above,
   * some previously decoded frames could have been pushed to decoded frames
   * queue. Need to clean them up; */
  if (p_impl->decoder_recon > 1) {
    p_impl->decoder_recon--;
    while (!p_impl->m_DecFramesCtxQueue.empty()) {
      p_impl->m_DecFramesCtxQueue.pop();
    }
  }

  if (!p_impl->m_DecFramesCtxQueue.empty()) {
    decCtx = p_impl->m_DecFramesCtxQueue.front();
    p_impl->m_DecFramesCtxQueue.pop();
    ret = true;

    auto const out_pts = decCtx.pts;
    auto const out_packet_data = p_impl->out_pdata.find(out_pts);
    if (p_impl->out_pdata.end() != out_packet_data) {
      // We have found information about this frame, give it back to user;
      decCtx.out_pdata = out_packet_data->second;
      p_impl->out_pdata.erase(out_pts);
    }

    /* Give user info about number of Annex.B bytes consumed by decoder.
     * This is useful when Annex.B is taken from external demuxer which may
     * give data in fixed size chunks;
     */
    decCtx.out_pdata.bsl = decCtx.bsl;
  }

  return ret;
}

// Adds frame back to pool of decoder-owned frames;
void NvDecoder::UnlockSurface(CUdeviceptr& lockedSurface) {
  if (lockedSurface) {
    lock_guard<mutex> lock(p_impl->m_mtxVPFrame);
    p_impl->m_DecFramesCtxVec.push_back(
        DecodedFrameContext(lockedSurface, 0U, 0U));
  }
}

namespace VPF {
struct NvdecDecodeFrame_Impl {
  NvDecoder nvDecoder;
  Surface* pLastSurface = nullptr;
  Buffer* pPacketData = nullptr;
  Buffer* PDetails = nullptr;
  CUstream stream = 0;
  CUcontext context = nullptr;
  bool didDecode = false;

  NvdecDecodeFrame_Impl() = delete;
  NvdecDecodeFrame_Impl(const NvdecDecodeFrame_Impl& other) = delete;
  NvdecDecodeFrame_Impl& operator=(const NvdecDecodeFrame_Impl& other) = delete;

  NvdecDecodeFrame_Impl(CUstream cuStream, CUcontext cuContext,
                        cudaVideoCodec videoCodec, Pixel_Format format)
      : stream(cuStream), context(cuContext),
        nvDecoder(cuStream, cuContext, videoCodec) {
    pLastSurface = Surface::Make(format);
    pPacketData = Buffer::MakeOwnMem(sizeof(PacketData));
    PDetails = Buffer::MakeOwnMem(sizeof(TaskExecDetails));
  }

  DLDataTypeCode TypeCode() const { return kDLUInt; }

  ~NvdecDecodeFrame_Impl() {
    delete pLastSurface;
    delete pPacketData;
    delete PDetails;
  }
};
} // namespace VPF

NvdecDecodeFrame* NvdecDecodeFrame::Make(CUstream cuStream, CUcontext cuContext,
                                         cudaVideoCodec videoCodec,
                                         uint32_t decodedFramesPoolSize,
                                         uint32_t coded_width,
                                         uint32_t coded_height,
                                         Pixel_Format format) {
  return new NvdecDecodeFrame(cuStream, cuContext, videoCodec,
                              decodedFramesPoolSize, coded_width, coded_height,
                              format);
}

NvdecDecodeFrame::NvdecDecodeFrame(CUstream cuStream, CUcontext cuContext,
                                   cudaVideoCodec videoCodec,
                                   uint32_t decodedFramesPoolSize,
                                   uint32_t coded_width, uint32_t coded_height,
                                   Pixel_Format format)
    :

      Task("NvdecDecodeFrame", NvdecDecodeFrame::numInputs,
           NvdecDecodeFrame::numOutputs, nullptr, nullptr) {
  pImpl = new NvdecDecodeFrame_Impl(cuStream, cuContext, videoCodec, format);
}

NvdecDecodeFrame::~NvdecDecodeFrame() {
  auto lastSurface = pImpl->pLastSurface->PlanePtr();
  pImpl->nvDecoder.UnlockSurface(lastSurface);
  delete pImpl;
}

TaskExecStatus NvdecDecodeFrame::Run() {
  NvtxMark tick(GetName());
  ClearOutputs();
  try {
    auto& decoder = pImpl->nvDecoder;
    auto pEncFrame = (Buffer*)GetInput();

    if (!pEncFrame && !pImpl->didDecode) {
      /* Empty input given + we've never did decoding means something went
       * wrong; Otherwise (no input + we did decode) means we're flushing;
       */
      SetExecDetails(TaskExecDetails(TaskExecInfo::FAIL));
      return TaskExecStatus::TASK_EXEC_FAIL;
    }

    bool isSurfaceReturned = false;
    uint64_t timestamp = 0U;
    auto pPktData = (Buffer*)GetInput(1U);
    if (pPktData) {
      auto p_pkt_data = pPktData->GetDataAs<PacketData>();
      timestamp = p_pkt_data->pts;
      pImpl->pPacketData->Update(sizeof(*p_pkt_data), p_pkt_data);
    }

    auto const no_eos = nullptr != GetInput(2);

    /* This will feed decoder with input timestamp.
     * It will also return surface + it's timestamp.
     * So timestamp is input + output parameter. */
    DecodedFrameContext dec_ctx;
    if (no_eos) {
      dec_ctx.no_eos = true;
    }

    {
      /* Do this in separate scope because we don't want to measure
       * DecodeLockSurface() function run time;
       */
      stringstream ss;
      ss << "Start decode for frame with pts " << timestamp;
      NvtxMark decode_k_off(ss.str().c_str());
    }

    PacketData in_pkt_data = {0};
    if (pPktData) {
      auto p_pkt_data = pPktData->GetDataAs<PacketData>();
      in_pkt_data = *p_pkt_data;
    }

    isSurfaceReturned =
        decoder.DecodeLockSurface(pEncFrame, in_pkt_data, dec_ctx);
    pImpl->didDecode = true;

    if (isSurfaceReturned) {
      // Unlock last surface because we will use it later;
      auto lastSurface = pImpl->pLastSurface->PlanePtr();
      decoder.UnlockSurface(lastSurface);

      // Update the reconstructed frame data;
      auto rawW = decoder.GetWidth();
      auto rawH = decoder.GetHeight() + decoder.GetChromaHeight();
      auto rawP = decoder.GetDeviceFramePitch();

      // Element size for different bit depth;
      auto elem_size = 0U;
      switch (pImpl->nvDecoder.GetBitDepth()) {
      case 8U:
        elem_size = sizeof(uint8_t);
        break;
      case 10U:
        elem_size = sizeof(uint16_t);
        break;
      case 12U:
        elem_size = sizeof(uint16_t);
        break;
      default:
        SetExecDetails(TaskExecDetails(TaskExecInfo::BIT_DEPTH_NOT_SUPPORTED));
        return TaskExecStatus::TASK_EXEC_FAIL;
      }

      auto dlmt_ptr = SurfacePlane::DLPackContext::ToDLPackSmart(
          rawW, rawH, rawP, elem_size, dec_ctx.mem, pImpl->TypeCode());
      SurfacePlane tmpPlane(*dlmt_ptr.get());
      pImpl->pLastSurface->Update({&tmpPlane});
      SetOutput(pImpl->pLastSurface, 0U);

      // Update the reconstructed frame timestamp;
      auto p_packet_data = pImpl->pPacketData->GetDataAs<PacketData>();
      memset(p_packet_data, 0, sizeof(*p_packet_data));
      *p_packet_data = dec_ctx.out_pdata;
      SetOutput(pImpl->pPacketData, 1U);

      {
        stringstream ss;
        ss << "End decode for frame with pts " << dec_ctx.pts;
        NvtxMark display_ready(ss.str().c_str());
      }

      return TaskExecStatus::TASK_EXEC_SUCCESS;
    } else if (pEncFrame) {
      /* We have input but no output.
       * That happens in the begining of decode loop. Not an error. */
      return TaskExecStatus::TASK_EXEC_SUCCESS;
    } else {
      /* No input, no output. That means decoder is flushed and we have reached
       * end of stream. */
      SetExecDetails(TaskExecDetails(TaskExecInfo::END_OF_STREAM));
      return TaskExecStatus::TASK_EXEC_FAIL;
    }
  } catch (exception& e) {
    cerr << e.what() << endl;
    SetExecDetails(TaskExecDetails(TaskExecInfo::FAIL));
    return TaskExecStatus::TASK_EXEC_FAIL;
  }
}

void NvdecDecodeFrame::GetDecodedFrameParams(uint32_t& width, uint32_t& height,
                                             uint32_t& elem_size) {
  width = pImpl->nvDecoder.GetWidth();
  height = pImpl->nvDecoder.GetHeight();
  elem_size = (pImpl->nvDecoder.GetBitDepth() + 7) / 8;
}

uint32_t NvdecDecodeFrame::GetDeviceFramePitch() {
  return uint32_t(pImpl->nvDecoder.GetDeviceFramePitch());
}

int NvdecDecodeFrame::GetCapability(NV_DEC_CAPS cap) const {
  CUVIDDECODECAPS decode_caps;
  memset((void*)&decode_caps, 0, sizeof(decode_caps));

  decode_caps.eCodecType = pImpl->nvDecoder.GetCodec();
  decode_caps.eChromaFormat = pImpl->nvDecoder.GetChromaFormat();
  decode_caps.nBitDepthMinus8 = pImpl->nvDecoder.GetBitDepth() - 8;

  auto ret = pImpl->nvDecoder._api().cuvidGetDecoderCaps(&decode_caps);
  if (CUDA_SUCCESS != ret) {
    return -1;
  }

  switch (cap) {
  case BIT_DEPTH_MINUS_8:
    return decode_caps.nBitDepthMinus8;
  case IS_CODEC_SUPPORTED:
    return decode_caps.bIsSupported;
  case OUTPUT_FORMAT_MASK:
    return decode_caps.nOutputFormatMask;
  case MAX_WIDTH:
    return decode_caps.nMaxWidth;
  case MAX_HEIGHT:
    return decode_caps.nMaxHeight;
  case MAX_MB_COUNT:
    return decode_caps.nMaxMBCount;
  case MIN_WIDTH:
    return decode_caps.nMinWidth;
  case MIN_HEIGHT:
    return decode_caps.nMinHeight;
#if CHECK_API_VERSION(11, 0)
  case IS_HIST_SUPPORTED:
    return decode_caps.bIsHistogramSupported;
  case HIST_COUNT_BIT_DEPTH:
    return decode_caps.nCounterBitDepth;
  case HIST_COUNT_BINS:
    return decode_caps.nMaxHistogramBins;
#endif
  default:
    return -1;
  }
}