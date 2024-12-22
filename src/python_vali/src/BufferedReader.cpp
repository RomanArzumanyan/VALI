/*
 * Copyright 2024 VisionLabs LLC
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

namespace py = pybind11;

BufferedReader::BufferedReader(py::object obj) : m_obj(obj) {
  py::gil_scoped_acquire acq;

  // Do this outside try-catch block because read attribute is mandatory.
  m_obj.attr("read");

  // But it's doable without seek.
  try {
    m_obj.attr("seek");
  } catch (...) {
    m_is_seekable = false;
  }
}

int BufferedReader::read(void* self, uint8_t* buf, int buf_size) {
  auto me = static_cast<BufferedReader*>(self);
  try {
    py::gil_scoped_acquire acq;

    if (!me || !buf || buf_size <= 0 || buf_size > me->m_buffer_size) {
      std::cerr << __FUNCTION__ << "Invalid argument given";
      return AVERROR_UNKNOWN;
    }

    /* Get read method and run it. It return bytes so we have to memcpy from
     * bytes to actual buffer
     */
    auto read_func =
        py::reinterpret_borrow<py::function>(me->m_obj.attr("read"));
    py::buffer_info info(py::buffer(read_func(buf_size)).request());

    auto const num_bytes = info.shape[0];
    if (num_bytes) {
      memcpy((void*)buf, info.ptr, buf_size);
      return num_bytes;
    } else {
      return AVERROR_EOF;
    }
  } catch (std::exception& e) {
    std::cerr << e.what();
    return AVERROR_UNKNOWN;
  }
}

int64_t BufferedReader::seek(void* self, int64_t offset, int whence) {
  auto me = static_cast<BufferedReader*>(self);
  try {
    py::gil_scoped_acquire acq;

    if (!me) {
      std::cerr << __FUNCTION__ << "Invalid argument given";
      return AVERROR_UNKNOWN;
    }

    // Handle special whence parameter values.
    if (whence & AVSEEK_SIZE) {
      return me->m_buffer_size;
    } else if (whence & AVSEEK_FORCE) {
      std::cerr << "AVSEEK_FORCE isn't supported as whence parameter value";
      return AVERROR_UNKNOWN;
    }

    auto seek_func =
        py::reinterpret_borrow<py::function>(me->m_obj.attr("seek"));
    return seek_func(offset, whence).cast<int64_t>();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    me->m_is_seekable = false;
    me->m_io_ctx_ptr->seek = 0x0;
    return AVERROR_UNKNOWN;
  }
}

std::shared_ptr<AVIOContext>
BufferedReader::GetAVIOContext(size_t buffer_size) {
  if (m_io_ctx_ptr) {
    return m_io_ctx_ptr;
  }

  m_buffer_size = buffer_size;
  auto buf = static_cast<unsigned char*>(av_malloc(m_buffer_size));
  if (!buf) {
    throw std::bad_alloc();
  }

  /* TODO (r.arzumanyan): try to fix the seek issue when VALI takes input from
   * pipe. Given BufferedReader object in that case has seek attribute but it
   * throws an exception and cripples the pipe upon the first call.
   *
   * Need to find out smarter way to check if seek operation can be used.
   */
  AVIOContext* io_ctx =
      avio_alloc_context(buf, m_buffer_size, 0, static_cast<void*>(this),
                         BufferedReader::read, nullptr, nullptr);

  if (!io_ctx) {
    av_free(buf);
    throw std::bad_alloc();
  }

  m_io_ctx_ptr = std::shared_ptr<AVIOContext>(io_ctx, [](void* p) {
    AVIOContext* p_ctx = static_cast<AVIOContext*>(p);
    avio_context_free(&p_ctx);
  });

  return m_io_ctx_ptr;
}