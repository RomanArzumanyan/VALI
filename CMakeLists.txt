#
# Copyright 2019 NVIDIA Corporation
# Copyright 2021 Kognia Sports Intelligence
# Copyright 2023 Roman Arzumanyan
# Copyright 2024 VisionLabs LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required(VERSION 3.21)

project(python_vali)

set(CMAKE_CXX_STANDARD 17)

set(
    PROJECT_ROOT_DIR 
    "${CMAKE_CURRENT_SOURCE_DIR}" 
    CACHE 
    INTERNAL 
    "project_root")

add_subdirectory(src)