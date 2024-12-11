#include "Utils.hpp"
#include <iostream>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
}

std::string AvErrorToString(int av_error_code) {
  const auto buf_size = 1024U;
  char* err_string = (char*)calloc(buf_size, sizeof(*err_string));
  if (!err_string) {
    return std::string();
  }

  if (0 != av_strerror(av_error_code, err_string, buf_size - 1)) {
    free(err_string);
    std::stringstream ss;
    ss << "Unknown error with code " << av_error_code;
    return ss.str();
  }

  std::string str(err_string);
  free(err_string);
  return str;
}

AVDictionary*
GetAvOptions(const std::map<std::string, std::string>& ffmpeg_options) {
  AVDictionary* options = nullptr;
  for (auto& pair : ffmpeg_options) {
    auto err =
        av_dict_set(&options, pair.first.c_str(), pair.second.c_str(), 0);

    if (err < 0) {
      av_dict_free(&options);
      std::stringstream ss;
      ss << "Can't set up dictionary option: " << pair.first << " "
         << pair.second << ": " << AvErrorToString(err) << "\n";
      throw std::runtime_error(ss.str());
    }
  }

  return options;
}

std::string GetFormatName(Pixel_Format fmt) {
  static const std::map<Pixel_Format, std::string> names({
      MAKE_PAIR(UNDEFINED),
      MAKE_PAIR(Y),
      MAKE_PAIR(RGB),
      MAKE_PAIR(NV12),
      MAKE_PAIR(YUV420),
      MAKE_PAIR(RGB_PLANAR),
      MAKE_PAIR(BGR),
      MAKE_PAIR(YUV444),
      MAKE_PAIR(RGB_32F),
      MAKE_PAIR(RGB_32F_PLANAR),
      MAKE_PAIR(YUV422),
      MAKE_PAIR(P10),
      MAKE_PAIR(P12),
      MAKE_PAIR(YUV444_10bit),
      MAKE_PAIR(YUV420_10bit),
      MAKE_PAIR(GRAY12),
  });

  auto it = names.find(fmt);
  if (it == names.end()) {
    return "";
  }

  return it->second;
}

std::shared_ptr<AVFrame> makeAVFrame(int width, int height, int format) {
  std::shared_ptr<AVFrame> frame(av_frame_alloc(),
                                 [](auto* p) { av_frame_free(&p); });
  frame->width = width;
  frame->height = height;
  frame->format = format;

  // Use 0 for auto-alignment according to ffmpeg doc;
  auto ret = av_frame_get_buffer(frame.get(), 0);
  if (ret < 0) {
    throw std::runtime_error("meaningful message");
  }

  return frame;
}

std::shared_ptr<AVFrame> asAVFrame(Buffer* pBuf, int width, int height,
                                   AVPixelFormat format) {
  std::shared_ptr<AVFrame> frame(av_frame_alloc(),
                                 [](auto* p) { av_frame_free(&p); });

  auto const alignment = 1U;
  auto ret = av_image_fill_arrays(frame->data, frame->linesize,
                                  pBuf->GetDataAs<uint8_t>(), format, width,
                                  height, alignment);
  if (ret < 0) {
    throw std::runtime_error("meaningful message");
  }

  return frame;
}

std::shared_ptr<Buffer> makeBufferFromAVFrame(std::shared_ptr<AVFrame> src) {
  auto const alignment = 1U;
  auto const out_size = getBufferSize(src->width, src->height,
                                      (AVPixelFormat)src->format, alignment);
  auto pBuf = std::shared_ptr<Buffer>(Buffer::MakeOwnMem(out_size));

  auto ret = av_image_copy_to_buffer(
      pBuf->GetDataAs<uint8_t>(), out_size, src->data, src->linesize,
      (AVPixelFormat)src->format, src->width, src->height, alignment);
  if (ret < 0) {
    throw std::runtime_error("meaningful message");
  }

  return pBuf;
}

std::shared_ptr<Buffer> makeBuffer(int width, int height,
                                   AVPixelFormat format) {
  return std::shared_ptr<Buffer>(
      Buffer::MakeOwnMem(getBufferSize(width, height, format)));
}

size_t getBufferSize(int width, int height, AVPixelFormat format,
                     int alignment) {
  return av_image_get_buffer_size(format, width, height, alignment);
}

static const std::vector<std::pair<Pixel_Format, AVPixelFormat>>
    formats({{UNDEFINED, AV_PIX_FMT_NONE},
             {Y, AV_PIX_FMT_GRAY8},
             {RGB, AV_PIX_FMT_RGB24},
             {NV12, AV_PIX_FMT_NV12},
             {YUV420, AV_PIX_FMT_YUV420P},
             {RGB_PLANAR, AV_PIX_FMT_NONE},
             {BGR, AV_PIX_FMT_BGR24},
             {YUV444, AV_PIX_FMT_YUV444P},
             {RGB_32F, AV_PIX_FMT_RGBF32LE},
             {RGB_32F_PLANAR, AV_PIX_FMT_NONE},
             {YUV422, AV_PIX_FMT_YUV422P},
             {P10, AV_PIX_FMT_P010},
             {P12, AV_PIX_FMT_YUV420P12},
             {YUV444_10bit, AV_PIX_FMT_YUV444P10},
             {YUV420_10bit, AV_PIX_FMT_YUV420P10},
             {GRAY12, AV_PIX_FMT_GRAY10}});

AVPixelFormat toFfmpegPixelFormat(Pixel_Format fmt) {
  for (const auto& pair : formats) {
    if (fmt == pair.first) {
      return pair.second;
    }
  }

  return AV_PIX_FMT_NONE;
}

Pixel_Format fromFfmpegPixelFormat(AVPixelFormat fmt) {
  for (const auto& pair : formats) {
    if (fmt == pair.second) {
      return pair.first;
    }
  }

  return UNDEFINED;
}

static const std::vector<std::pair<ColorSpace, AVColorSpace>>
    spaces({{BT_601, AVCOL_SPC_BT470BG},
            {BT_709, AVCOL_SPC_BT709},
            {UNSPEC, AVCOL_SPC_NB}});

AVColorSpace toFfmpegColorSpace(ColorSpace space) {
  for (auto& pair : spaces) {
    if (pair.first == space) {
      return pair.second;
    }
  }

  return AVCOL_SPC_NB;
}

ColorSpace fromFfmpegColorSpace(AVColorSpace space) {
  for (auto& pair : spaces) {
    if (pair.second == space) {
      return pair.first;
    }
  }

  return UNSPEC;
}

static const std::vector<std::pair<ColorRange, AVColorRange>>
    ranges({{MPEG, AVCOL_RANGE_MPEG},
            {JPEG, AVCOL_RANGE_JPEG},
            {UDEF, AVCOL_RANGE_UNSPECIFIED}});

AVColorRange toFfmpegColorRange(ColorRange range) {
  for (auto& pair : ranges) {
    if (pair.first == range) {
      return pair.second;
    }
  }

  return AVCOL_RANGE_UNSPECIFIED;
}

ColorRange fromFfmpegColorRange(AVColorRange range) {
  for (auto& pair : ranges) {
    if (pair.second == range) {
      return pair.first;
    }
  }

  return UDEF;
}

// Helper functions
void ThrowOnAvError(int res, const std::string& msg) {
  ThrowOnAvError(res, msg, nullptr);
}

void ThrowOnAvError(int res, const std::string& msg, AVDictionary** options) {
  if (res < 0) {
    if (options) {
      av_dict_free(options);
    }
    throw std::runtime_error(msg + ": " + AvErrorToString(res));
  }
}

static void RegisterTimeout(TimeoutHandler* handler, AVFormatContext* fmt_ctx) {
  if (!handler || !fmt_ctx) {
    throw std::runtime_error("Can not register timeout. Null pointer given.");
  }

  fmt_ctx->interrupt_callback.opaque = (void*)handler;
  fmt_ctx->interrupt_callback.callback = &TimeoutHandler::Check;
}

TimeoutHandler::TimeoutHandler(unsigned long timeout_ms,
                               AVFormatContext* fmt_ctx) {
  m_timeout = std::chrono::milliseconds(timeout_ms);
  RegisterTimeout(this, fmt_ctx);
  Reset();
}

static std::chrono::milliseconds fromString(const char* val) {
  try {
    return std::chrono::milliseconds(std::stoul(val));
  } catch (std::exception& e) {
    std::cerr << "Error parsing timeout value: " << e.what() << "\n";
    std::cerr << "Using default timeout value: "
              << TimeoutHandler::GetDefaultTimeout();
    std::cerr << std::endl;

    return std::chrono::milliseconds(TimeoutHandler::GetDefaultTimeout());
  }
}

TimeoutHandler::TimeoutHandler(const char* timeout_str,
                               AVFormatContext* fmt_ctx) {
  m_timeout = fromString(timeout_str);
  RegisterTimeout(this, fmt_ctx);
  Reset();
}

TimeoutHandler::TimeoutHandler(AVDictionary** dict, AVFormatContext* fmt_ctx) {
  m_timeout = std::chrono::milliseconds(TimeoutHandler::GetDefaultTimeout());
  auto entry = av_dict_get(*dict, "timeout", nullptr, 0);

  if (entry) {
    m_timeout = fromString(entry->value);
  }

  // Delete "timeout" and "stimeout" entries.
  const char* tbd[] = {"timeout", "stimeout"};
  for (auto& key : tbd) {
    auto entry = av_dict_get(*dict, key, nullptr, 0);
    if (entry) {
      ThrowOnAvError(av_dict_set(dict, key, nullptr, 0),
                     "Failed to reset timeout", dict);
    }
  }

  RegisterTimeout(this, fmt_ctx);
  Reset();
}

bool TimeoutHandler::IsTimeout() const {
  auto delay = std::chrono::steady_clock::now() - m_last_time;
  auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(delay);
  return diff > m_timeout;
}

int TimeoutHandler::Check(void* self) {
  return self && static_cast<TimeoutHandler*>(self)->IsTimeout();
}

void TimeoutHandler::SetDefaultTimeout(unsigned long new_default_timeout) {
  std::lock_guard<std::mutex> lock(s_lock);
  s_default_timeout = new_default_timeout;
}

unsigned long TimeoutHandler::GetDefaultTimeout() {
  std::lock_guard<std::mutex> lock(s_lock);
  return s_default_timeout;
}

void TimeoutHandler::Reset() { m_last_time = std::chrono::steady_clock::now(); }

unsigned long TimeoutHandler::s_default_timeout = 3000U;
std::mutex TimeoutHandler::s_lock;