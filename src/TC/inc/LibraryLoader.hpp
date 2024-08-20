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

#include "tc_dlopen.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

// These macro definitions are used to put together library names and versions.
#ifndef XCONCAT
#define XCONCAT(A, B, C) A##B##C
#endif

#ifndef XSTR
#define XSTR(A) XSTRINGIFY(A)
#endif

#ifndef XSTRINGIFY
#define XSTRINGIFY(A) #A
#endif

class LibraryLoader {
public:
  LibraryLoader(const char* filename);
  ~LibraryLoader();

  TC_LIB getHandle() const { return m_hModule; }
  const std::string& getFilename() const { return m_filename; };

private:
  const std::string m_filename;
  TC_LIB m_hModule;
};

template <std::shared_ptr<LibraryLoader> (*LoadLibrary)(), typename Return,
          typename... Args>
class LoadableFunction {
public:
  LoadableFunction(const char* name) : m_name(name) {
    auto pLoader = LoadLibrary();
    m_filename = pLoader->getFilename();
    if (pLoader->getHandle()) {
      m_fptr = reinterpret_cast<decltype(m_fptr)>(
          tc_dlsym(pLoader->getHandle(), m_name.c_str()));
    }
  }

  Return operator()(Args... args) {
    if (m_fptr) {
      return (*m_fptr)(args...);
    }

    if (m_filename.empty()) {
      throw std::runtime_error(m_name + " unavailable, because library " +
                               m_filename + " was not found");
    } else {
      throw std::runtime_error(m_name + " not found in " + m_filename);
    }
  }

private:
  const std::string m_name;
  std::string m_filename;
  Return (*m_fptr)(Args...){};
};
