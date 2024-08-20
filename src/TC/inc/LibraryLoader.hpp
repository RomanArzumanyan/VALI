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

  TC_LIB getHandle() const { return hModule; }
  const std::string& getFilename() const { return filename; };

private:
  const std::string filename;
  TC_LIB hModule;
};

template <std::shared_ptr<LibraryLoader> (*LoadLibrary)(), typename Return,
          typename... Args>
class LoadableFunction {
public:
  LoadableFunction(const char* name) : name(name) {
    auto pLoader = LoadLibrary();
    if (pLoader->getHandle()) {
      filename = pLoader->getFilename();
      f = reinterpret_cast<decltype(f)>(tc_dlsym(pLoader->getHandle(), name));
    }
  }
  Return operator()(Args... args) {
    if (f)
      return (*f)(args...);
    printf("Not found\n");

    if (filename.empty())
      throw std::runtime_error(name + " unavailable, because library \"" +
                               filename + "\" not found");
    else
      throw std::runtime_error(name + " not found in \"" + filename + '"');
  }

private:
  const std::string name;
  std::string filename;
  Return (*f)(Args...){};
};

#define WITH_NAME(x)                                                           \
  x { #x }
