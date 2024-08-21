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

#include "LibraryLoader.hpp"
#include <sstream>

LibraryLoader::LibraryLoader(const char* filename) : m_filename(filename) {
  m_hModule = tc_dlopen(filename);
}

LibraryLoader::~LibraryLoader() {
  if (m_hModule) {
    tc_dlclose(m_hModule);
  }
}

const char* LibraryLoader::makeFilename(const char* libName, int libVersion,
                                        const char* libFileExt) {
  std::stringstream name;
  name << libName << libVersion << libFileExt;
  return name.str().c_str();
}
