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

#include "MemoryInterfaces.hpp"
#include "NvCodecCLIOptions.h"
#include "NvEncoder.h"
#include "NvEncoderCuda.h"
#include "Tasks.hpp"

using namespace std;

#ifndef _WIN32
#include <cstring>
#include <dlfcn.h>

using namespace std;

static inline bool operator==(const GUID& guid1, const GUID& guid2) {
  return !memcmp(&guid1, &guid2, sizeof(GUID));
}

static inline bool operator!=(const GUID& guid1, const GUID& guid2) {
  return !(guid1 == guid2);
}
#endif

#include <map>
#include <queue>

NvEncoderCuda::NvEncoderCuda(int gpu_id, CUstream stream, uint32_t nWidth,
                             uint32_t nHeight,
                             NV_ENC_BUFFER_FORMAT eBufferFormat,
                             uint32_t nExtraOutputDelay,
                             bool bMotionEstimationOnly,
                             bool bOutputInVideoMemory)
    : NvEncoder(NV_ENC_DEVICE_TYPE_CUDA, gpu_id, stream, nWidth, nHeight,
                eBufferFormat, nExtraOutputDelay, bMotionEstimationOnly,
                bOutputInVideoMemory),
      m_cuda_stream(stream), m_gpu_id(gpu_id)

{
  if (!m_hEncoder) {
    NVENC_THROW_ERROR("Encoder Initialization failed",
                      NV_ENC_ERR_INVALID_DEVICE);
  }
}

NvEncoderCuda::~NvEncoderCuda() { ReleaseCudaResources(); }

void NvEncoderCuda::AllocateInputBuffers(int32_t numInputBuffers) {
  if (!IsHWEncoderInitialized()) {
    NVENC_THROW_ERROR("Encoder initialization failed",
                      NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
  }

  // for MEOnly mode we need to allocate separate set of buffers for reference
  // frame
  int numCount = m_bMotionEstimationOnly ? 2 : 1;

  for (int count = 0; count < numCount; count++) {
    CudaCtxPush lock(m_cuda_stream);
    vector<void*> inputFrames;

    for (int i = 0; i < numInputBuffers; i++) {
      CUdeviceptr pDeviceFrame;
      uint32_t chromaHeight =
          GetNumChromaPlanes(GetPixelFormat()) *
          GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
      if (GetPixelFormat() == NV_ENC_BUFFER_FORMAT_YV12 ||
          GetPixelFormat() == NV_ENC_BUFFER_FORMAT_IYUV) {
        chromaHeight = GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
      }

      CHECK_CUDA_CALL(LibCuda::cuMemAllocPitch(
          &pDeviceFrame, &m_cudaPitch,
          GetWidthInBytes(GetPixelFormat(), GetMaxEncodeWidth()),
          GetMaxEncodeHeight() + chromaHeight, 16));
      inputFrames.push_back((void*)pDeviceFrame);
    }

    RegisterInputResources(inputFrames,
                           NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR,
                           GetMaxEncodeWidth(), GetMaxEncodeHeight(),
                           (int)m_cudaPitch, GetPixelFormat(), count == 1);
  }
}

void NvEncoderCuda::ReleaseInputBuffers() { ReleaseCudaResources(); }

void NvEncoderCuda::ReleaseCudaResources() {
  if (!m_hEncoder) {
    return;
  }

  UnregisterInputResources();

  CudaCtxPush lock(m_cuda_stream);

  for (auto inputFrame : m_vInputFrames) {
    if (inputFrame.inputPtr) {
      LibCuda::cuMemFree(reinterpret_cast<CUdeviceptr>(inputFrame.inputPtr));
    }
  }
  m_vInputFrames.clear();

  for (auto referenceFrame : m_vReferenceFrames) {
    if (referenceFrame.inputPtr) {
      LibCuda::cuMemFree(
          reinterpret_cast<CUdeviceptr>(referenceFrame.inputPtr));
    }
  }
  m_vReferenceFrames.clear();
}

void NvEncoderCuda::CopyToDeviceFrame(CUstream stream, void* pSrcFrame,
                                      uint32_t nSrcPitch, CUdeviceptr pDstFrame,
                                      uint32_t dstPitch, int width, int height,
                                      CUmemorytype srcMemoryType,
                                      NV_ENC_BUFFER_FORMAT pixelFormat,
                                      const uint32_t dstChromaOffsets[],
                                      uint32_t numChromaPlanes) {
  if (srcMemoryType != CU_MEMORYTYPE_HOST &&
      srcMemoryType != CU_MEMORYTYPE_DEVICE) {
    NVENC_THROW_ERROR("Invalid source memory type for copy",
                      NV_ENC_ERR_INVALID_PARAM);
  }

  CudaCtxPush lock(stream);

  uint32_t srcPitch =
      nSrcPitch ? nSrcPitch : NvEncoder::GetWidthInBytes(pixelFormat, width);
  CUDA_MEMCPY2D m = {0};
  m.srcMemoryType = srcMemoryType;
  if (srcMemoryType == CU_MEMORYTYPE_HOST) {
    m.srcHost = pSrcFrame;
  } else {
    m.srcDevice = (CUdeviceptr)pSrcFrame;
  }
  m.srcPitch = srcPitch;
  m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  m.dstDevice = pDstFrame;
  m.dstPitch = dstPitch;
  m.WidthInBytes = NvEncoder::GetWidthInBytes(pixelFormat, width);
  m.Height = height;

  CHECK_CUDA_CALL(LibCuda::cuMemcpy2DAsync(&m, stream));

  vector<uint32_t> srcChromaOffsets;
  NvEncoder::GetChromaSubPlaneOffsets(pixelFormat, srcPitch, height,
                                      srcChromaOffsets);
  uint32_t chromaHeight = NvEncoder::GetChromaHeight(pixelFormat, height);
  uint32_t destChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, dstPitch);
  uint32_t srcChromaPitch = NvEncoder::GetChromaPitch(pixelFormat, srcPitch);
  uint32_t chromaWidthInBytes =
      NvEncoder::GetChromaWidthInBytes(pixelFormat, width);

  for (uint32_t i = 0; i < numChromaPlanes; ++i) {
    if (chromaHeight) {
      if (srcMemoryType == CU_MEMORYTYPE_HOST) {
        m.srcHost = ((uint8_t*)pSrcFrame + srcChromaOffsets[i]);
      } else {
        m.srcDevice = (CUdeviceptr)((uint8_t*)pSrcFrame + srcChromaOffsets[i]);
      }
      m.srcPitch = srcChromaPitch;

      m.dstDevice = (CUdeviceptr)((uint8_t*)pDstFrame + dstChromaOffsets[i]);
      m.dstPitch = destChromaPitch;
      m.WidthInBytes = chromaWidthInBytes;
      m.Height = chromaHeight;
      CHECK_CUDA_CALL(LibCuda::cuMemcpy2DAsync(&m, stream));
    }
  }
}

NV_ENCODE_API_FUNCTION_LIST NvEncoderCuda::GetApi() const { return m_nvenc; }

void* NvEncoderCuda::GetEncoder() const { return m_hEncoder; }

int NvEncoder::GetEncodeWidth() const { return m_nWidth; }

int NvEncoder::GetEncodeHeight() const { return m_nHeight; }

bool NvEncoder::IsHWEncoderInitialized() const {
  return (m_hEncoder != nullptr) && (m_bEncoderInitialized);
}

uint32_t NvEncoder::GetMaxEncodeWidth() const { return m_nMaxEncodeWidth; }

uint32_t NvEncoder::GetMaxEncodeHeight() const { return m_nMaxEncodeHeight; }

void* NvEncoder::GetCompletionEvent(uint32_t eventIdx) {
  return (m_vpCompletionEvents.size() == m_nEncoderBufferSize)
             ? m_vpCompletionEvents[eventIdx]
             : nullptr;
}

NV_ENC_BUFFER_FORMAT
NvEncoder::GetPixelFormat() const { return m_eBufferFormat; }

NvEncoder::NvEncoder(NV_ENC_DEVICE_TYPE eDeviceType, int gpu_id,
                     CUstream cuda_stream, uint32_t nWidth, uint32_t nHeight,
                     NV_ENC_BUFFER_FORMAT eBufferFormat,
                     uint32_t nExtraOutputDelay, bool bMotionEstimationOnly,
                     bool bOutputInVideoMemory)
    :

      m_eDeviceType(eDeviceType), m_nWidth(nWidth), m_nHeight(nHeight),
      m_nMaxEncodeWidth(nWidth), m_nMaxEncodeHeight(nHeight),
      m_eBufferFormat(eBufferFormat),
      m_bMotionEstimationOnly(bMotionEstimationOnly),
      m_bOutputInVideoMemory(bOutputInVideoMemory),
      m_nExtraOutputDelay(nExtraOutputDelay), m_hEncoder(nullptr),
      m_nvenc({0}) {
  LoadNvEncApi();

  if (!m_nvenc.nvEncOpenEncodeSession) {
    m_nEncoderBufferSize = 0;
    NVENC_THROW_ERROR("EncodeAPI not found", NV_ENC_ERR_NO_ENCODE_DEVICE);
  }

  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encodeSessionExParams = {
      NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER};
  encodeSessionExParams.device = (void*)GetContextByStream(gpu_id, cuda_stream);
  encodeSessionExParams.deviceType = m_eDeviceType;
  encodeSessionExParams.apiVersion = NVENCAPI_VERSION;
  void* hEncoder = nullptr;
  NVENC_API_CALL(
      m_nvenc.nvEncOpenEncodeSessionEx(&encodeSessionExParams, &hEncoder),
      m_nvenc.nvEncGetLastErrorString(m_hEncoder));
  m_hEncoder = hEncoder;
}

void NvEncoder::LoadNvEncApi() {
#if defined(_WIN32)
#if defined(_WIN64)
  HMODULE hModule = LoadLibrary(TEXT("nvEncodeAPI64.dll"));
#else
  HMODULE hModule = LoadLibrary(TEXT("nvEncodeAPI.dll"));
#endif
#else
  void* hModule = dlopen("libnvidia-encode.so.1", RTLD_LAZY);
#endif

  if (hModule == nullptr) {
#if defined(_WIN32)
    NVENC_THROW_ERROR(
        "Could not dynamically load nvEncodeAPI.dll. Please ensure Nvidia "
        "Graphics drivers are correctly installed!",
        NV_ENC_ERR_NO_ENCODE_DEVICE);
#else
    NVENC_THROW_ERROR(
        "Could not dynamically load libnvidia-encode.so.1. Please "
        "ensure Nvidia Graphics drivers are correctly installed!\n"
        "If using Docker please make sure that your Docker image was launched "
        "with \"video\" driver capabilty (see "
        "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        "user-guide.html#driver-capabilities)",
        NV_ENC_ERR_NO_ENCODE_DEVICE);
#endif
  }

  m_hModule = hModule;

  typedef NVENCSTATUS(NVENCAPI *
                      NvEncodeAPIGetMaxSupportedVersion_Type)(uint32_t*);
#if defined(_WIN32)
  auto NvEncodeAPIGetMaxSupportedVersion =
      (NvEncodeAPIGetMaxSupportedVersion_Type)GetProcAddress(
          hModule, "NvEncodeAPIGetMaxSupportedVersion");
#else
  auto NvEncodeAPIGetMaxSupportedVersion =
      (NvEncodeAPIGetMaxSupportedVersion_Type)dlsym(
          hModule, "NvEncodeAPIGetMaxSupportedVersion");
#endif

  uint32_t version = 0;
  uint32_t currentVersion =
      (NVENCAPI_MAJOR_VERSION << 4U) | NVENCAPI_MINOR_VERSION;
  NVENC_API_CALL(NvEncodeAPIGetMaxSupportedVersion(&version),
                 m_nvenc.nvEncGetLastErrorString(m_hEncoder));
  if (currentVersion > version) {
    NVENC_THROW_ERROR(
        "Current Driver Version does not support this NvEncodeAPI version, "
        "please upgrade driver",
        NV_ENC_ERR_INVALID_VERSION);
  }

  typedef NVENCSTATUS(NVENCAPI * NvEncodeAPICreateInstance_Type)(
      NV_ENCODE_API_FUNCTION_LIST*);
#if defined(_WIN32)
  auto NvEncodeAPICreateInstance =
      (NvEncodeAPICreateInstance_Type)GetProcAddress(
          hModule, "NvEncodeAPICreateInstance");
#else
  auto NvEncodeAPICreateInstance = (NvEncodeAPICreateInstance_Type)dlsym(
      hModule, "NvEncodeAPICreateInstance");
#endif

  if (!NvEncodeAPICreateInstance) {
    NVENC_THROW_ERROR(
        "Cannot find NvEncodeAPICreateInstance() entry in NVENC library",
        NV_ENC_ERR_NO_ENCODE_DEVICE);
  }

  m_nvenc = {NV_ENCODE_API_FUNCTION_LIST_VER};
  NVENC_API_CALL(NvEncodeAPICreateInstance(&m_nvenc),
                 m_nvenc.nvEncGetLastErrorString(m_hEncoder));
}

NvEncoder::~NvEncoder() {
  DestroyHWEncoder();

  if (m_hModule) {
#if defined(_WIN32)
    FreeLibrary((HMODULE)m_hModule);
#else
    dlclose(m_hModule);
#endif
    m_hModule = nullptr;
  }
}

void NvEncoder::CreateEncoder(const NV_ENC_INITIALIZE_PARAMS* pEncoderParams) {
  if (!m_hEncoder) {
    NVENC_THROW_ERROR("Encoder Initialization failed",
                      NV_ENC_ERR_NO_ENCODE_DEVICE);
  }

  if (!pEncoderParams) {
    NVENC_THROW_ERROR("Invalid NV_ENC_INITIALIZE_PARAMS ptr",
                      NV_ENC_ERR_INVALID_PTR);
  }

  if (pEncoderParams->encodeWidth == 0 || pEncoderParams->encodeHeight == 0) {
    NVENC_THROW_ERROR("Invalid encoder width and height",
                      NV_ENC_ERR_INVALID_PARAM);
  }

  if (pEncoderParams->encodeGUID != NV_ENC_CODEC_H264_GUID &&
      pEncoderParams->encodeGUID != NV_ENC_CODEC_HEVC_GUID) {
    NVENC_THROW_ERROR("Invalid codec guid", NV_ENC_ERR_INVALID_PARAM);
  }

  if (pEncoderParams->encodeGUID == NV_ENC_CODEC_H264_GUID) {
    if (m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ||
        m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) {
      NVENC_THROW_ERROR("10-bit format isn't supported by H264 encoder",
                        NV_ENC_ERR_INVALID_PARAM);
    }
  }

  // set other necessary params if not set yet
  if (pEncoderParams->encodeGUID == NV_ENC_CODEC_H264_GUID) {
    if ((m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444) &&
        (pEncoderParams->encodeConfig->encodeCodecConfig.h264Config
             .chromaFormatIDC != 3)) {
      NVENC_THROW_ERROR("Invalid ChromaFormatIDC", NV_ENC_ERR_INVALID_PARAM);
    }
  }

  if (pEncoderParams->encodeGUID == NV_ENC_CODEC_HEVC_GUID) {
    bool yuv10BitFormat =
        (m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV420_10BIT ||
         m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT);
    if (yuv10BitFormat && pEncoderParams->encodeConfig->encodeCodecConfig
                                  .hevcConfig.pixelBitDepthMinus8 != 2) {
      NVENC_THROW_ERROR("Invalid PixelBitdepth", NV_ENC_ERR_INVALID_PARAM);
    }

    if ((m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444 ||
         m_eBufferFormat == NV_ENC_BUFFER_FORMAT_YUV444_10BIT) &&
        (pEncoderParams->encodeConfig->encodeCodecConfig.hevcConfig
             .chromaFormatIDC != 3)) {
      NVENC_THROW_ERROR("Invalid ChromaFormatIDC", NV_ENC_ERR_INVALID_PARAM);
    }
  }

  memcpy(&m_initializeParams, pEncoderParams, sizeof(m_initializeParams));
  m_initializeParams.version = NV_ENC_INITIALIZE_PARAMS_VER;

  if (pEncoderParams->encodeConfig) {
    memcpy(&m_encodeConfig, pEncoderParams->encodeConfig,
           sizeof(m_encodeConfig));
    m_encodeConfig.version = NV_ENC_CONFIG_VER;
  } else {
    NV_ENC_PRESET_CONFIG presetConfig = {NV_ENC_PRESET_CONFIG_VER,
                                         {NV_ENC_CONFIG_VER}};
    m_nvenc.nvEncGetEncodePresetConfig(m_hEncoder, pEncoderParams->encodeGUID,
                                       NV_ENC_PRESET_DEFAULT_GUID,
                                       &presetConfig);
    memcpy(&m_encodeConfig, &presetConfig.presetCfg, sizeof(NV_ENC_CONFIG));
    m_encodeConfig.version = NV_ENC_CONFIG_VER;
    m_encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
    m_encodeConfig.rcParams.constQP = {28, 31, 25};
  }
  m_initializeParams.encodeConfig = &m_encodeConfig;

  NVENC_API_CALL(
      m_nvenc.nvEncInitializeEncoder(m_hEncoder, &m_initializeParams),
      m_nvenc.nvEncGetLastErrorString(m_hEncoder));

  m_bEncoderInitialized = true;
  m_nWidth = m_initializeParams.encodeWidth;
  m_nHeight = m_initializeParams.encodeHeight;
  m_nMaxEncodeWidth = m_initializeParams.maxEncodeWidth;
  m_nMaxEncodeHeight = m_initializeParams.maxEncodeHeight;

  m_nEncoderBufferSize = m_encodeConfig.frameIntervalP +
                         m_encodeConfig.rcParams.lookaheadDepth +
                         m_nExtraOutputDelay;
  m_nOutputDelay = m_nEncoderBufferSize - 1;
  m_vMappedInputBuffers.resize(m_nEncoderBufferSize, nullptr);

  if (!m_bOutputInVideoMemory) {
    m_vpCompletionEvents.resize(m_nEncoderBufferSize, nullptr);
  }

#if defined(_WIN32)
  for (int i = 0; i < m_vpCompletionEvents.size(); i++) {
    m_vpCompletionEvents[i] = CreateEvent(NULL, FALSE, FALSE, NULL);
    NV_ENC_EVENT_PARAMS eventParams = {NV_ENC_EVENT_PARAMS_VER};
    eventParams.completionEvent = m_vpCompletionEvents[i];
    m_nvenc.nvEncRegisterAsyncEvent(m_hEncoder, &eventParams);
  }
#endif

  if (m_bMotionEstimationOnly) {
    m_vMappedRefBuffers.resize(m_nEncoderBufferSize, nullptr);

    if (!m_bOutputInVideoMemory) {
      InitializeMVOutputBuffer();
    }
  } else {
    if (!m_bOutputInVideoMemory) {
      m_vBitstreamOutputBuffer.resize(m_nEncoderBufferSize, nullptr);
      InitializeBitstreamBuffer();
    }
  }

  AllocateInputBuffers(m_nEncoderBufferSize);
}

void NvEncoder::DestroyEncoder() {
  if (!m_hEncoder) {
    return;
  }

  ReleaseInputBuffers();

  DestroyHWEncoder();
}

void NvEncoder::DestroyHWEncoder() {
  if (!m_hEncoder) {
    return;
  }

#if defined(_WIN32)
  for (uint32_t i = 0; i < m_vpCompletionEvents.size(); i++) {
    if (m_vpCompletionEvents[i]) {
      NV_ENC_EVENT_PARAMS eventParams = {NV_ENC_EVENT_PARAMS_VER};
      eventParams.completionEvent = m_vpCompletionEvents[i];
      m_nvenc.nvEncUnregisterAsyncEvent(m_hEncoder, &eventParams);
      CloseHandle(m_vpCompletionEvents[i]);
    }
  }
  m_vpCompletionEvents.clear();
#endif

  if (m_bMotionEstimationOnly) {
    DestroyMVOutputBuffer();
  } else {
    DestroyBitstreamBuffer();
  }

  m_nvenc.nvEncDestroyEncoder(m_hEncoder);

  m_hEncoder = nullptr;

  m_bEncoderInitialized = false;
}

const NvEncInputFrame* NvEncoder::GetNextInputFrame() {
  int i = m_iToSend % m_nEncoderBufferSize;
  return &m_vInputFrames[i];
}

void NvEncoder::MapResources(uint32_t bfrIdx) {
  NV_ENC_MAP_INPUT_RESOURCE mapInputResource = {NV_ENC_MAP_INPUT_RESOURCE_VER};

  mapInputResource.registeredResource = m_vRegisteredResources[bfrIdx];
  NVENC_API_CALL(m_nvenc.nvEncMapInputResource(m_hEncoder, &mapInputResource),
                 m_nvenc.nvEncGetLastErrorString(m_hEncoder));
  m_vMappedInputBuffers[bfrIdx] = mapInputResource.mappedResource;

  if (m_bMotionEstimationOnly) {
    mapInputResource.registeredResource =
        m_vRegisteredResourcesForReference[bfrIdx];
    NVENC_API_CALL(m_nvenc.nvEncMapInputResource(m_hEncoder, &mapInputResource),
                   m_nvenc.nvEncGetLastErrorString(m_hEncoder));
    m_vMappedRefBuffers[bfrIdx] = mapInputResource.mappedResource;
  }
}

void NvEncoder::EncodeFrame(vector<vector<uint8_t>>& vPacket,
                            NV_ENC_PIC_PARAMS* pPicParams, bool output_delay,
                            uint32_t seiPayloadArrayCnt,
                            NV_ENC_SEI_PAYLOAD* seiPayloadArray) {
  vPacket.clear();
  if (!IsHWEncoderInitialized()) {
    NVENC_THROW_ERROR("Encoder device not found", NV_ENC_ERR_NO_ENCODE_DEVICE);
  }

  int bfrIdx = m_iToSend % m_nEncoderBufferSize;

  MapResources(bfrIdx);

  NVENCSTATUS nvStatus =
      DoEncode(m_vMappedInputBuffers[bfrIdx], m_vBitstreamOutputBuffer[bfrIdx],
               pPicParams, seiPayloadArrayCnt, seiPayloadArray);

  if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT) {
    m_iToSend++;
    GetEncodedPacket(m_vBitstreamOutputBuffer, vPacket, output_delay);
  } else {
    NVENC_THROW_ERROR("nvEncEncodePicture API failed", nvStatus);
  }
}

NVENCSTATUS
NvEncoder::DoEncode(NV_ENC_INPUT_PTR inputBuffer,
                    NV_ENC_OUTPUT_PTR outputBuffer,
                    NV_ENC_PIC_PARAMS* pPicParams, uint32_t seiPayloadArrayCnt,
                    NV_ENC_SEI_PAYLOAD* seiPayloadArray) {
  NV_ENC_PIC_PARAMS picParams = {};
  if (pPicParams) {
    picParams = *pPicParams;
  }

  if (seiPayloadArrayCnt) {
    NV_ENC_CONFIG config = {0};
    NV_ENC_INITIALIZE_PARAMS params = {0};
    params.encodeConfig = &config;
    GetInitializeParams(&params);
    bool is_h264 = (0 == memcmp(&NV_ENC_CODEC_H264_GUID, &params.encodeGUID,
                                sizeof(NV_ENC_CODEC_H264_GUID)));
    if (is_h264) {
      picParams.codecPicParams.h264PicParams.seiPayloadArrayCnt =
          seiPayloadArrayCnt;
      picParams.codecPicParams.h264PicParams.seiPayloadArray = seiPayloadArray;
    } else {
      picParams.codecPicParams.hevcPicParams.seiPayloadArrayCnt =
          seiPayloadArrayCnt;
      picParams.codecPicParams.hevcPicParams.seiPayloadArray = seiPayloadArray;
    }
  }

  picParams.version = NV_ENC_PIC_PARAMS_VER;
  picParams.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
  picParams.inputBuffer = inputBuffer;
  picParams.bufferFmt = GetPixelFormat();
  picParams.inputWidth = GetEncodeWidth();
  picParams.inputHeight = GetEncodeHeight();
  picParams.outputBitstream = outputBuffer;
  picParams.completionEvent =
      GetCompletionEvent(m_iToSend % m_nEncoderBufferSize);
  NVENCSTATUS nvStatus = m_nvenc.nvEncEncodePicture(m_hEncoder, &picParams);

  return nvStatus;
}

void NvEncoder::SendEOS() {
  NV_ENC_PIC_PARAMS picParams = {NV_ENC_PIC_PARAMS_VER};
  picParams.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
  picParams.completionEvent =
      GetCompletionEvent(m_iToSend % m_nEncoderBufferSize);
  NVENC_API_CALL(m_nvenc.nvEncEncodePicture(m_hEncoder, &picParams),
                 m_nvenc.nvEncGetLastErrorString(m_hEncoder));
}

bool NvEncoder::Reconfigure(
    const NV_ENC_RECONFIGURE_PARAMS* pReconfigureParams) {
  NVENC_API_CALL(m_nvenc.nvEncReconfigureEncoder(
                     m_hEncoder, const_cast<NV_ENC_RECONFIGURE_PARAMS*>(
                                     pReconfigureParams)),
                 m_nvenc.nvEncGetLastErrorString(m_hEncoder));

  memcpy(&m_initializeParams, &(pReconfigureParams->reInitEncodeParams),
         sizeof(m_initializeParams));
  if (pReconfigureParams->reInitEncodeParams.encodeConfig) {
    memcpy(&m_encodeConfig, pReconfigureParams->reInitEncodeParams.encodeConfig,
           sizeof(m_encodeConfig));
  }

  m_nWidth = m_initializeParams.encodeWidth;
  m_nHeight = m_initializeParams.encodeHeight;
  m_nMaxEncodeWidth = m_initializeParams.maxEncodeWidth;
  m_nMaxEncodeHeight = m_initializeParams.maxEncodeHeight;

  return true;
}

void NvEncoder::EndEncode(vector<vector<uint8_t>>& vPacket) {
  vPacket.clear();
  if (!IsHWEncoderInitialized()) {
    NVENC_THROW_ERROR("Encoder device not initialized",
                      NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
  }

  SendEOS();

  GetEncodedPacket(m_vBitstreamOutputBuffer, vPacket, false);
}

void NvEncoder::GetEncodedPacket(vector<NV_ENC_OUTPUT_PTR> const& vOutputBuffer,
                                 vector<vector<uint8_t>>& vPacket,
                                 bool bOutputDelay) {
  unsigned i = 0;
  int iEnd = bOutputDelay ? m_iToSend - m_nOutputDelay : m_iToSend;
  for (; m_iGot < iEnd; m_iGot++) {
    WaitForCompletionEvent(m_iGot % m_nEncoderBufferSize);
    NV_ENC_LOCK_BITSTREAM lockBitstreamData = {NV_ENC_LOCK_BITSTREAM_VER};
    lockBitstreamData.outputBitstream =
        vOutputBuffer[m_iGot % m_nEncoderBufferSize];
    lockBitstreamData.doNotWait = false;
    NVENC_API_CALL(m_nvenc.nvEncLockBitstream(m_hEncoder, &lockBitstreamData),
                   m_nvenc.nvEncGetLastErrorString(m_hEncoder));

    auto* pData = (uint8_t*)lockBitstreamData.bitstreamBufferPtr;
    if (vPacket.size() < i + 1) {
      vPacket.emplace_back(vector<uint8_t>());
    }
    vPacket[i].clear();
    vPacket[i].insert(vPacket[i].end(), &pData[0],
                      &pData[lockBitstreamData.bitstreamSizeInBytes]);
    i++;

    NVENC_API_CALL(m_nvenc.nvEncUnlockBitstream(
                       m_hEncoder, lockBitstreamData.outputBitstream),
                   m_nvenc.nvEncGetLastErrorString(m_hEncoder));

    if (m_vMappedInputBuffers[m_iGot % m_nEncoderBufferSize]) {
      NVENC_API_CALL(
          m_nvenc.nvEncUnmapInputResource(
              m_hEncoder, m_vMappedInputBuffers[m_iGot % m_nEncoderBufferSize]),
          m_nvenc.nvEncGetLastErrorString(m_hEncoder));
      m_vMappedInputBuffers[m_iGot % m_nEncoderBufferSize] = nullptr;
    }

    if (m_bMotionEstimationOnly &&
        m_vMappedRefBuffers[m_iGot % m_nEncoderBufferSize]) {
      NVENC_API_CALL(
          m_nvenc.nvEncUnmapInputResource(
              m_hEncoder, m_vMappedRefBuffers[m_iGot % m_nEncoderBufferSize]),
          m_nvenc.nvEncGetLastErrorString(m_hEncoder));
      m_vMappedRefBuffers[m_iGot % m_nEncoderBufferSize] = nullptr;
    }
  }
}

NV_ENC_REGISTERED_PTR
NvEncoder::RegisterResource(void* pBuffer,
                            NV_ENC_INPUT_RESOURCE_TYPE eResourceType, int width,
                            int height, int pitch,
                            NV_ENC_BUFFER_FORMAT bufferFormat,
                            NV_ENC_BUFFER_USAGE bufferUsage) {
  NV_ENC_REGISTER_RESOURCE registerResource = {NV_ENC_REGISTER_RESOURCE_VER};
  registerResource.resourceType = eResourceType;
  registerResource.resourceToRegister = pBuffer;
  registerResource.width = width;
  registerResource.height = height;
  registerResource.pitch = pitch;
  registerResource.bufferFormat = bufferFormat;
  registerResource.bufferUsage = bufferUsage;
  NVENC_API_CALL(m_nvenc.nvEncRegisterResource(m_hEncoder, &registerResource),
                 m_nvenc.nvEncGetLastErrorString(m_hEncoder));

  return registerResource.registeredResource;
}

void NvEncoder::RegisterInputResources(vector<void*> const& input_frames,
                                       NV_ENC_INPUT_RESOURCE_TYPE eResourceType,
                                       int width, int height, int pitch,
                                       NV_ENC_BUFFER_FORMAT bufferFormat,
                                       bool bReferenceFrame) {
  for (auto& inputFrame : input_frames) {
    NV_ENC_REGISTERED_PTR registeredPtr =
        RegisterResource(inputFrame, eResourceType, width, height, pitch,
                         bufferFormat, NV_ENC_INPUT_IMAGE);

    vector<uint32_t> _chromaOffsets;
    NvEncoder::GetChromaSubPlaneOffsets(bufferFormat, pitch, height,
                                        _chromaOffsets);
    NvEncInputFrame encInputFrame = {};
    encInputFrame.inputPtr = (void*)inputFrame;
    encInputFrame.chromaOffsets[0] = 0;
    encInputFrame.chromaOffsets[1] = 0;
    for (uint32_t ch = 0; ch < _chromaOffsets.size(); ch++) {
      encInputFrame.chromaOffsets[ch] = _chromaOffsets[ch];
    }
    encInputFrame.numChromaPlanes = NvEncoder::GetNumChromaPlanes(bufferFormat);
    encInputFrame.pitch = pitch;
    encInputFrame.chromaPitch = NvEncoder::GetChromaPitch(bufferFormat, pitch);
    encInputFrame.bufferFormat = bufferFormat;
    encInputFrame.resourceType = eResourceType;

    if (bReferenceFrame) {
      m_vRegisteredResourcesForReference.push_back(registeredPtr);
      m_vReferenceFrames.push_back(encInputFrame);
    } else {
      m_vRegisteredResources.push_back(registeredPtr);
      m_vInputFrames.push_back(encInputFrame);
    }
  }
}

void NvEncoder::FlushEncoder() {
  if (!m_bMotionEstimationOnly && !m_bOutputInVideoMemory) {
    // In case of error it is possible for buffers still mapped to encoder.
    // flush the encoder queue and then unmapped it if any surface is still
    // mapped
    try {
      vector<vector<uint8_t>> vPacket;
      EndEncode(vPacket);
    } catch (...) {
    }
  }
}

void NvEncoder::SetIOCudaStreams(NV_ENC_CUSTREAM_PTR inputStream,
                                 NV_ENC_CUSTREAM_PTR outputStream) {
  NVENC_API_CALL(
      m_nvenc.nvEncSetIOCudaStreams(m_hEncoder, inputStream, outputStream),
      m_nvenc.nvEncGetLastErrorString(m_hEncoder));
}

void NvEncoder::UnregisterInputResources() {
  FlushEncoder();

  if (m_bMotionEstimationOnly) {
    for (auto& mappedRefBuffer : m_vMappedRefBuffers) {
      if (mappedRefBuffer) {
        m_nvenc.nvEncUnmapInputResource(m_hEncoder, mappedRefBuffer);
      }
    }
  }
  m_vMappedRefBuffers.clear();

  for (auto& mappedInputBuffer : m_vMappedInputBuffers) {
    if (mappedInputBuffer) {
      m_nvenc.nvEncUnmapInputResource(m_hEncoder, mappedInputBuffer);
    }
  }
  m_vMappedInputBuffers.clear();

  for (auto& registeredResource : m_vRegisteredResources) {
    if (registeredResource) {
      m_nvenc.nvEncUnregisterResource(m_hEncoder, registeredResource);
    }
  }
  m_vRegisteredResources.clear();

  for (auto& registeredResourceForReference :
       m_vRegisteredResourcesForReference) {
    if (registeredResourceForReference) {
      m_nvenc.nvEncUnregisterResource(m_hEncoder,
                                      registeredResourceForReference);
    }
  }
  m_vRegisteredResourcesForReference.clear();
}

void NvEncoder::WaitForCompletionEvent(int iEvent) {
#if defined(_WIN32)
  // Check if we are in async mode. If not, don't wait for event;
  NV_ENC_CONFIG sEncodeConfig = {0};
  NV_ENC_INITIALIZE_PARAMS sInitializeParams = {0};
  sInitializeParams.encodeConfig = &sEncodeConfig;
  GetInitializeParams(&sInitializeParams);

  if (0U == sInitializeParams.enableEncodeAsync) {
    return;
  }
#ifdef DEBUG
  WaitForSingleObject(m_vpCompletionEvents[iEvent], INFINITE);
#else
  // wait for 20s which is infinite on terms of gpu time
  if (WaitForSingleObject(m_vpCompletionEvents[iEvent], 20000) == WAIT_FAILED) {
    NVENC_THROW_ERROR("Failed to encode frame", NV_ENC_ERR_GENERIC);
  }
#endif
#endif
}

uint32_t NvEncoder::GetWidthInBytes(const NV_ENC_BUFFER_FORMAT bufferFormat,
                                    const uint32_t width) {
  switch (bufferFormat) {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
  case NV_ENC_BUFFER_FORMAT_YUV444:
    return width;
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return width * 2;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return width * 4;
  default:
    NVENC_THROW_ERROR("Invalid Buffer format", NV_ENC_ERR_INVALID_PARAM);
    return 0;
  }
}

uint32_t
NvEncoder::GetNumChromaPlanes(const NV_ENC_BUFFER_FORMAT bufferFormat) {
  switch (bufferFormat) {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    return 1;
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return 2;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return 0;
  default:
    NVENC_THROW_ERROR("Invalid Buffer format", NV_ENC_ERR_INVALID_PARAM);
    return -1;
  }
}

uint32_t NvEncoder::GetChromaPitch(const NV_ENC_BUFFER_FORMAT bufferFormat,
                                   const uint32_t lumaPitch) {
  switch (bufferFormat) {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return lumaPitch;
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
    return (lumaPitch + 1) / 2;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return 0;
  default:
    NVENC_THROW_ERROR("Invalid Buffer format", NV_ENC_ERR_INVALID_PARAM);
    return -1;
  }
}

void NvEncoder::GetChromaSubPlaneOffsets(
    const NV_ENC_BUFFER_FORMAT bufferFormat, const uint32_t pitch,
    const uint32_t height, vector<uint32_t>& chromaOffsets) {
  chromaOffsets.clear();
  switch (bufferFormat) {
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    chromaOffsets.push_back(pitch * height);
    return;
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
    chromaOffsets.push_back(pitch * height);
    chromaOffsets.push_back(chromaOffsets[0] +
                            (NvEncoder::GetChromaPitch(bufferFormat, pitch) *
                             GetChromaHeight(bufferFormat, height)));
    return;
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    chromaOffsets.push_back(pitch * height);
    chromaOffsets.push_back(chromaOffsets[0] + (pitch * height));
    return;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return;
  default:
    NVENC_THROW_ERROR("Invalid Buffer format", NV_ENC_ERR_INVALID_PARAM);
    return;
  }
}

uint32_t NvEncoder::GetChromaHeight(const NV_ENC_BUFFER_FORMAT bufferFormat,
                                    const uint32_t lumaHeight) {
  switch (bufferFormat) {
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
  case NV_ENC_BUFFER_FORMAT_NV12:
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    return (lumaHeight + 1) / 2;
  case NV_ENC_BUFFER_FORMAT_YUV444:
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return lumaHeight;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return 0;
  default:
    NVENC_THROW_ERROR("Invalid Buffer format", NV_ENC_ERR_INVALID_PARAM);
    return 0;
  }
}

uint32_t
NvEncoder::GetChromaWidthInBytes(const NV_ENC_BUFFER_FORMAT bufferFormat,
                                 const uint32_t lumaWidth) {
  switch (bufferFormat) {
  case NV_ENC_BUFFER_FORMAT_YV12:
  case NV_ENC_BUFFER_FORMAT_IYUV:
    return (lumaWidth + 1) / 2;
  case NV_ENC_BUFFER_FORMAT_NV12:
    return lumaWidth;
  case NV_ENC_BUFFER_FORMAT_YUV420_10BIT:
    return 2 * lumaWidth;
  case NV_ENC_BUFFER_FORMAT_YUV444:
    return lumaWidth;
  case NV_ENC_BUFFER_FORMAT_YUV444_10BIT:
    return 2 * lumaWidth;
  case NV_ENC_BUFFER_FORMAT_ARGB:
  case NV_ENC_BUFFER_FORMAT_ARGB10:
  case NV_ENC_BUFFER_FORMAT_AYUV:
  case NV_ENC_BUFFER_FORMAT_ABGR:
  case NV_ENC_BUFFER_FORMAT_ABGR10:
    return 0;
  default:
    NVENC_THROW_ERROR("Invalid Buffer format", NV_ENC_ERR_INVALID_PARAM);
    return 0;
  }
}

int NvEncoder::GetCapabilityValue(GUID guidCodec, NV_ENC_CAPS capsToQuery) {
  if (!m_hEncoder) {
    return 0;
  }
  NV_ENC_CAPS_PARAM capsParam = {NV_ENC_CAPS_PARAM_VER};
  capsParam.capsToQuery = capsToQuery;
  int v;
  m_nvenc.nvEncGetEncodeCaps(m_hEncoder, guidCodec, &capsParam, &v);
  return v;
}

void NvEncoder::GetInitializeParams(
    NV_ENC_INITIALIZE_PARAMS* pInitializeParams) {
  if (!pInitializeParams || !pInitializeParams->encodeConfig) {
    NVENC_THROW_ERROR(
        "Both pInitializeParams and pInitializeParams->encodeConfig can't be "
        "NULL",
        NV_ENC_ERR_INVALID_PTR);
  }
  NV_ENC_CONFIG* pEncodeConfig = pInitializeParams->encodeConfig;
  *pEncodeConfig = m_encodeConfig;
  *pInitializeParams = m_initializeParams;
  pInitializeParams->encodeConfig = pEncodeConfig;
}

void NvEncoder::InitializeBitstreamBuffer() {
  for (int i = 0; i < m_nEncoderBufferSize; i++) {
    NV_ENC_CREATE_BITSTREAM_BUFFER createBitstreamBuffer = {
        NV_ENC_CREATE_BITSTREAM_BUFFER_VER};
    NVENC_API_CALL(
        m_nvenc.nvEncCreateBitstreamBuffer(m_hEncoder, &createBitstreamBuffer),
        m_nvenc.nvEncGetLastErrorString(m_hEncoder));
    m_vBitstreamOutputBuffer[i] = createBitstreamBuffer.bitstreamBuffer;
  }
}

void NvEncoder::DestroyBitstreamBuffer() {
  for (auto& bitstreamOutputBuffer : m_vBitstreamOutputBuffer) {
    if (bitstreamOutputBuffer) {
      m_nvenc.nvEncDestroyBitstreamBuffer(m_hEncoder, bitstreamOutputBuffer);
    }
  }

  m_vBitstreamOutputBuffer.clear();
}

void NvEncoder::InitializeMVOutputBuffer() {
  for (int i = 0; i < m_nEncoderBufferSize; i++) {
    NV_ENC_CREATE_MV_BUFFER createMVBuffer = {NV_ENC_CREATE_MV_BUFFER_VER};
    NVENC_API_CALL(m_nvenc.nvEncCreateMVBuffer(m_hEncoder, &createMVBuffer),
                   m_nvenc.nvEncGetLastErrorString(m_hEncoder));
    m_vMVDataOutputBuffer.push_back(createMVBuffer.mvBuffer);
  }
}

void NvEncoder::DestroyMVOutputBuffer() {
  for (auto& mvDataOutputBuffer : m_vMVDataOutputBuffer) {
    if (mvDataOutputBuffer) {
      m_nvenc.nvEncDestroyMVBuffer(m_hEncoder, mvDataOutputBuffer);
    }
  }

  m_vMVDataOutputBuffer.clear();
}

namespace VPF {
struct NvencEncodeFrame_Impl {
  using packet = vector<uint8_t>;

  NV_ENC_BUFFER_FORMAT enc_buffer_format;
  queue<packet> packetQueue;
  vector<uint8_t> lastPacket;
  Buffer* pElementaryVideo;
  NvEncoderCuda* pEncoderCuda = nullptr;
  CUstream stream = 0;
  int m_gpu_id;
  bool didEncode = false;
  bool didFlush = false;
  NV_ENC_RECONFIGURE_PARAMS recfg_params;
  NV_ENC_INITIALIZE_PARAMS& init_params;
  NV_ENC_CONFIG encodeConfig;
  std::map<NV_ENC_CAPS, int> capabilities;

  NvencEncodeFrame_Impl() = delete;
  NvencEncodeFrame_Impl(const NvencEncodeFrame_Impl& other) = delete;
  NvencEncodeFrame_Impl& operator=(const NvencEncodeFrame_Impl& other) = delete;

  uint32_t GetWidth() const { return pEncoderCuda->GetEncodeWidth(); };
  uint32_t GetHeight() const { return pEncoderCuda->GetEncodeHeight(); };
  int GetCap(NV_ENC_CAPS cap) const {
    auto it = capabilities.find(cap);
    if (it != capabilities.end()) {
      return it->second;
    }

    return -1;
  }

  NvencEncodeFrame_Impl(NV_ENC_BUFFER_FORMAT format,
                        NvEncoderClInterface& cli_iface, int gpu_id,
                        CUstream str, int32_t width, int32_t height,
                        bool verbose)
      : init_params(recfg_params.reInitEncodeParams) {
    pElementaryVideo = Buffer::Make(0U);

    stream = str;
    pEncoderCuda = new NvEncoderCuda(gpu_id, str, width, height, format);
    enc_buffer_format = format;

    init_params = {NV_ENC_INITIALIZE_PARAMS_VER};
    encodeConfig = {NV_ENC_CONFIG_VER};
    init_params.encodeConfig = &encodeConfig;

    cli_iface.SetupInitParams(init_params, false, pEncoderCuda->GetApi(),
                              pEncoderCuda->GetEncoder(), capabilities,
                              verbose);

    pEncoderCuda->CreateEncoder(&init_params);

    pEncoderCuda->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&stream,
                                   (NV_ENC_CUSTREAM_PTR)&stream);
  }

  bool Reconfigure(NvEncoderClInterface& cli_iface, bool force_idr,
                   bool reset_enc, bool verbose) {
    recfg_params.version = NV_ENC_RECONFIGURE_PARAMS_VER;
    recfg_params.resetEncoder = reset_enc;
    recfg_params.forceIDR = force_idr;

    cli_iface.SetupInitParams(init_params, true, pEncoderCuda->GetApi(),
                              pEncoderCuda->GetEncoder(), capabilities,
                              verbose);

    return pEncoderCuda->Reconfigure(&recfg_params);
  }

  ~NvencEncodeFrame_Impl() {
    pEncoderCuda->DestroyEncoder();
    delete pEncoderCuda;
    delete pElementaryVideo;
  }
};
} // namespace VPF

NvencEncodeFrame* NvencEncodeFrame::Make(int gpu_id, CUstream cuStream,
                                         NvEncoderClInterface& cli_iface,
                                         NV_ENC_BUFFER_FORMAT format,
                                         uint32_t width, uint32_t height,
                                         bool verbose) {
  return new NvencEncodeFrame(gpu_id, cuStream, cli_iface, format, width,
                              height, verbose);
}

bool VPF::NvencEncodeFrame::Reconfigure(NvEncoderClInterface& cli_iface,
                                        bool force_idr, bool reset_enc,
                                        bool verbose) {
  return pImpl->Reconfigure(cli_iface, force_idr, reset_enc, verbose);
}

auto const cuda_stream_sync = [](void* stream) {
  LibCuda::cuStreamSynchronize((CUstream)stream);
};

NvencEncodeFrame::NvencEncodeFrame(int gpu_id, CUstream cuStream,
                                   NvEncoderClInterface& cli_iface,
                                   NV_ENC_BUFFER_FORMAT format, uint32_t width,
                                   uint32_t height, bool verbose)
    :

      Task("NvencEncodeFrame", NvencEncodeFrame::numInputs,
           NvencEncodeFrame::numOutputs, cuda_stream_sync, (void*)cuStream) {
  pImpl = new NvencEncodeFrame_Impl(format, cli_iface, gpu_id, cuStream, width,
                                    height, verbose);
}

NvencEncodeFrame::~NvencEncodeFrame() { delete pImpl; };

TaskExecDetails NvencEncodeFrame::Run() {
  NvtxMark tick(GetName());
  SetOutput(nullptr, 0U);

  try {
    auto& pEncoderCuda = pImpl->pEncoderCuda;
    auto& didFlush = pImpl->didFlush;
    auto& didEncode = pImpl->didEncode;
    auto input = (Surface*)GetInput(0U);
    vector<vector<uint8_t>> encPackets;

    if (input) {
      auto& stream = pImpl->stream;
      const NvEncInputFrame* encoderInputFrame =
          pEncoderCuda->GetNextInputFrame();
      auto width = input->Width(), height = input->Height(),
           pitch = input->Pitch();

      bool is_resize_needed = (pEncoderCuda->GetEncodeWidth() != width) ||
                              (pEncoderCuda->GetEncodeHeight() != height);

      if (is_resize_needed) {
        return TaskExecDetails(
            TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
            "Input Surface is of different size. Encoder resize required");
      } else {
        NvEncoderCuda::CopyToDeviceFrame(
            stream, (void*)input->PixelPtr(), pitch,
            (CUdeviceptr)encoderInputFrame->inputPtr,
            (int32_t)encoderInputFrame->pitch, pEncoderCuda->GetEncodeWidth(),
            pEncoderCuda->GetEncodeHeight(), CU_MEMORYTYPE_DEVICE,
            encoderInputFrame->bufferFormat, encoderInputFrame->chromaOffsets,
            encoderInputFrame->numChromaPlanes);
      }

      auto pSEI = (Buffer*)GetInput(2U);
      NV_ENC_SEI_PAYLOAD payload = {0};
      if (pSEI) {
        payload.payloadSize = pSEI->GetRawMemSize();
        // Unregistered user data for H.265 and H.264 both;
        payload.payloadType = 5;
        payload.payload = pSEI->GetDataAs<uint8_t>();
      }

      auto const seiNumber = pSEI ? 1U : 0U;
      auto pPayload = pSEI ? &payload : nullptr;

      auto sync = GetInput(1U);
      if (sync) {
        pEncoderCuda->EncodeFrame(encPackets, nullptr, false, seiNumber,
                                  pPayload);
      } else {
        pEncoderCuda->EncodeFrame(encPackets, nullptr, true, seiNumber,
                                  pPayload);
      }
      didEncode = true;
    } else if (didEncode && !didFlush) {
      // No input after a while means we're flushing;
      pEncoderCuda->EndEncode(encPackets);
      didFlush = true;
    }

    /* Push encoded packets into queue;
     */
    for (auto& packet : encPackets) {
      pImpl->packetQueue.push(packet);
    }

    /* Then return least recent packet;
     */
    pImpl->lastPacket.clear();
    if (!pImpl->packetQueue.empty()) {
      pImpl->lastPacket = pImpl->packetQueue.front();
      pImpl->pElementaryVideo->Update(pImpl->lastPacket.size(),
                                      (void*)pImpl->lastPacket.data());
      pImpl->packetQueue.pop();
      SetOutput(pImpl->pElementaryVideo, 0U);
    }

    return TaskExecDetails(TaskExecStatus::TASK_EXEC_SUCCESS,
                           TaskExecInfo::SUCCESS);
  } catch (exception& e) {
    return TaskExecDetails(TaskExecStatus::TASK_EXEC_FAIL, TaskExecInfo::FAIL,
                           e.what());
  }
}

uint32_t NvencEncodeFrame::GetWidth() const { return pImpl->GetWidth(); }

uint32_t NvencEncodeFrame::GetHeight() const { return pImpl->GetHeight(); }

int NvencEncodeFrame::GetCapability(NV_ENC_CAPS cap) const {
  return pImpl->GetCap(cap);
}