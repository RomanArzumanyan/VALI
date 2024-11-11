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

BufferedRandom::BufferedRandom(py::object obj) : m_obj(obj) {}

int BufferedRandom::read(void* self, uint8_t* buf, int buf_size) {
  auto me = static_cast<BufferedRandom*>(self);
  if (!me || !buf || buf_size <= 0) {
    return 1;
  }

  /* Get read method and run it. It return bytes so we have to memcpy from
   * bytes to actual buffer
   */
  auto read_func = py::reinterpret_borrow<py::function>(me->m_obj.attr("read"));
  py::buffer_info info(py::buffer(read_func(buf_size)).request());

  memcpy((void*)buf, info.ptr, buf_size);
  return info.shape[0];
}

int BufferedRandom::write(void* self, const uint8_t* buf, int buf_size) {
  auto me = static_cast<BufferedRandom*>(self);
  if (!me || !buf || buf_size <= 0) {
    return 1;
  }

  auto write_func =
      py::reinterpret_borrow<py::function>(me->m_obj.attr("write"));
  
  std::vector<uint8_t> bytes(buf, buf + buf_size);
  return write_func(bytes).cast<int>();
}

int64_t BufferedRandom::seek(void* self, int64_t offset, int whence) {
  auto me = static_cast<BufferedRandom*>(self);
  if (!me) {
    return 1;
  }

  auto seek_func = py::reinterpret_borrow<py::function>(me->m_obj.attr("seek"));
  return seek_func(offset, whence).cast<int64_t>();
}