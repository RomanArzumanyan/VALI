/*
 * Copyright 2022 NVIDIA Corporation
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

#include "CudaUtils.hpp"
#include "VALI.hpp"
#include "dlpack.h"
#include <map>
#include <sstream>

using namespace std;
using namespace VPF;
using namespace chrono;

namespace py = pybind11;
using namespace pybind11::literals;

string ToString(Pixel_Format fmt) {
  static map<Pixel_Format, string> fmt_names = {
      {Y, "Y"},
      {RGB, "RGB"},
      {NV12, "NV12"},
      {YUV420, "YUV420"},
      {RGB_PLANAR, "RGB_PLANAR"},
      {BGR, "BGR"},
      {YUV444, "YUV444"},
      {RGB_32F, "RGB_32F"},
      {RGB_32F_PLANAR, "RGB_32F_PLANAR"},
      {YUV422, "YUV422"},
      {P10, "P10"},
      {P12, "P12"},
  };

  auto it = fmt_names.find(fmt);
  if (fmt_names.end() != it) {
    return it->second;
  } else {
    return string("UNDEFINED");
  }
};

string ToString(SurfacePlane& self, int space = 0) {
  stringstream spacer;
  for (int i = 0; i < space; i++) {
    spacer << " ";
  }

  stringstream ss;
  ss << spacer.str() << "Owns mem:  " << self.OwnMemory() << "\n";
  ss << spacer.str() << "Width:     " << self.Width() << "\n";
  ss << spacer.str() << "Height:    " << self.Height() << "\n";
  ss << spacer.str() << "Pitch:     " << self.Pitch() << "\n";
  ss << spacer.str() << "Elem size: " << self.ElemSize() << "\n";
  ss << spacer.str() << "Cuda ctx:  " << self.Context() << "\n";
  ss << spacer.str() << "CUDA ptr:  " << self.GpuMem() << "\n";

  return ss.str();
}

string ToString(Surface& self) {
  stringstream ss;
  ss << "Width:            " << self.Width() << "\n";
  ss << "Height:           " << self.Height() << "\n";
  ss << "Format:           " << ToString(self.PixelFormat()) << "\n";
  ss << "Pitch:            " << self.Pitch() << "\n";
  ss << "Elem size(bytes): " << self.ElemSize() << "\n";

  for (int i = 0; i < self.NumPlanes(); i++) {
    ss << "Plane " << i << "\n";
    ss << ToString(self.GetSurfacePlane(i), 2) << "\n";
  }

  return ss.str();
}

static void dlpack_capsule_deleter(PyObject* self) {
  if (PyCapsule_IsValid(self, "used_dltensor")) {
    return;
  }

  PyObject *type, *value, *traceback;
  PyErr_Fetch(&type, &value, &traceback);

  DLManagedTensor* managed =
      (DLManagedTensor*)PyCapsule_GetPointer(self, "dltensor");
  if (managed == NULL) {
    PyErr_WriteUnraisable(self);
    goto done;
  }

  if (managed->deleter) {
    managed->deleter(managed);
    assert(!PyErr_Occurred());
  }

done:
  PyErr_Restore(type, value, traceback);
}

void Init_PySurface(py::module& m) {
  py::class_<SurfacePlane, shared_ptr<SurfacePlane>>(
      m, "SurfacePlane",
      "Continious 2D chunk of memory stored in vRAM which represents single "
      "plane / channel of video frame. It supports DLPack specification.")
      .def_property_readonly("Width", &SurfacePlane::Width,
                             R"pbdoc(
         Get the width of the surface plane in pixels.

         :return: Width of the plane in pixels
         :rtype: int
     )pbdoc")
      .def_property_readonly("Height", &SurfacePlane::Height,
                             R"pbdoc(
         Get the height of the surface plane in pixels.

         :return: Height of the plane in pixels
         :rtype: int
     )pbdoc")
      .def_property_readonly("Pitch", &SurfacePlane::Pitch,
                             R"pbdoc(
         Get the pitch (stride) of the surface plane in bytes.

         The pitch represents the number of bytes between the start of consecutive rows
         in the surface plane. This may be larger than the width * element size due to
         memory alignment requirements.

         :return: Pitch of the plane in bytes
         :rtype: int
     )pbdoc")
      .def_property_readonly("ElemSize", &SurfacePlane::ElemSize,
                             R"pbdoc(
         Get the size of each element in the surface plane in bytes.

         :return: Size of each element in bytes
         :rtype: int
     )pbdoc")
      .def_property_readonly("HostFrameSize", &SurfacePlane::HostMemSize,
                             R"pbdoc(
         Get the amount of host memory needed to store this surface plane.

         This is the total size in bytes required to store the plane's data
         in host memory, taking into account the pitch and height.

         :return: Required host memory size in bytes
         :rtype: int
     )pbdoc")
      .def_property_readonly("GpuMem", &SurfacePlane::GpuMem,
                             R"pbdoc(
         Get the CUDA device pointer to the surface plane data.

         :return: CUDA device pointer to the plane's data
         :rtype: int
     )pbdoc")
      .def(
          "__dlpack_device__",
          [](shared_ptr<SurfacePlane> self) {
            if (self->FromDLPack()) {
              throw(std::runtime_error(
                  "Cant get __dlpack_device__ attribute from "
                  "Surface created from DLPack."));
            }

            return std::make_tuple(self->DLPackCtx().DeviceType(),
                                   self->DeviceId());
          },
          R"pbdoc(
         Get DLPack device information.

         Returns a tuple containing the device type and device ID.
         This method cannot be called on surfaces created from DLPack.

         :return: Tuple of (device_type, device_id)
         :rtype: tuple[int, int]
         :raises RuntimeError: If called on a surface created from DLPack
     )pbdoc")
      .def(
          "__dlpack__",
          [](shared_ptr<SurfacePlane> self, int stream) {
            auto dlmt = self->ToDLPack();
            return py::capsule(dlmt, "dltensor", dlpack_capsule_deleter);
          },
          py::arg("stream") = 0,
          R"pbdoc(
         Get DLPack capsule for the surface plane.

         Creates a DLPack capsule containing the surface plane data.
         The capsule will be automatically cleaned up when no longer needed.

         :param stream: CUDA stream to use for the operation (default: 0)
         :type stream: int
         :return: DLPack capsule containing the surface data
         :rtype: capsule
     )pbdoc")
      .def_property_readonly(
          "__cuda_array_interface__",
          [](shared_ptr<SurfacePlane> self) {
            CudaArrayInterfaceDescriptor cai;
            self->ToCAI(cai);

            return py::dict(
                "shape"_a = py::make_tuple(cai.m_shape[0], cai.m_shape[1],
                                           cai.m_shape[2]),
                "typestr"_a = cai.m_typestr,
                "data"_a = py::make_tuple(cai.m_ptr, cai.m_read_only),
                "version"_a = cai.m_version,
                "strides"_a = py::make_tuple(cai.m_strides[0], cai.m_strides[1],
                                             cai.m_strides[2]),
                "stream"_a = size_t(cai.m_stream));
          },
          R"pbdoc(
         Get CUDA Array Interface descriptor.

         Returns a dictionary containing the CUDA Array Interface (CAI) descriptor
         for the surface plane, which can be used to create numpy arrays or other
         CUDA-compatible data structures.

         :return: Dictionary containing CAI descriptor
         :rtype: dict
     )pbdoc")
      .def("__repr__",
           [](shared_ptr<SurfacePlane> self) { return ToString(*self.get()); });

  py::class_<Surface, shared_ptr<Surface>>(
      m, "Surface", "Image stored in vRAM. Consists of 1+ SurfacePlane(s).")
      .def_property_readonly(
          "Width", [](Surface& self) { return self.Width(0); },
          R"pbdoc(
         Get the width of the first plane in pixels.

         :return: Width of the first plane in pixels
         :rtype: int
     )pbdoc")
      .def_property_readonly(
          "Height", [](Surface& self) { return self.Height(0); },
          R"pbdoc(
         Get the height of the first plane in pixels.

         :return: Height of the first plane in pixels
         :rtype: int
     )pbdoc")
      .def_property_readonly(
          "Pitch", [](Surface& self) { return self.Pitch(0); },
          R"pbdoc(
         Get the pitch (stride) of the first plane in bytes.

         The pitch represents the number of bytes between the start of consecutive rows
         in the first plane. This may be larger than the width * element size due to
         memory alignment requirements.

         :return: Pitch of the first plane in bytes
         :rtype: int
     )pbdoc")
      .def_property_readonly("Format", &Surface::PixelFormat,
                             R"pbdoc(
         Get the pixel format of the surface.

         :return: Pixel format of the surface
         :rtype: Pixel_Format
     )pbdoc")
      .def_property_readonly("IsEmpty", &Surface::Empty,
                             R"pbdoc(
         Check if the surface has allocated memory.

         :return: True if the surface has no allocated memory, False otherwise
         :rtype: bool
     )pbdoc")
      .def_property_readonly("NumPlanes", &Surface::NumPlanes,
                             R"pbdoc(
         Get the number of planes in the surface.

         Different pixel formats may have different numbers of planes.
         For example, RGB has 1 plane while YUV420 has 3 planes.

         :return: Number of planes in the surface
         :rtype: int
     )pbdoc")
      .def_property_readonly("HostSize", &Surface::HostMemSize,
                             R"pbdoc(
         Get the total amount of host memory needed for device-to-host copy.

         This is the total size in bytes required to store all planes of the surface
         in host memory, taking into account the pitch and height of each plane.

         :return: Required host memory size in bytes
         :rtype: int
     )pbdoc")
      .def_property_readonly("IsOwnMemory", &Surface::OwnMemory,
                             R"pbdoc(
         Check if the surface owns its memory.

         :return: True if the surface owns its memory, False if it only references
             memory owned by another object
         :rtype: bool
     )pbdoc")
      .def_property_readonly("Shape", &Surface::Shape,
                             R"pbdoc(
         Get the numpy-like shape of the surface.

         For multi-plane formats (like YUV420), returns the total memory size.
         The shape represents the dimensions of the surface data.

         :return: Tuple containing the dimensions of the surface
         :rtype: tuple
     )pbdoc")
      .def("Clone", &Surface::Clone, py::return_value_policy::take_ownership,
           R"pbdoc(
         Create a deep copy of the surface.

         Allocates new CUDA memory and copies all surface data.
         The returned object is managed by the Python interpreter.

         :return: A new surface containing a copy of the data
         :rtype: Surface
     )pbdoc")
      .def_static(
          "Make",
          [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
             int gpuID) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(format, newWidth, newHeight,
                              CudaResMgr::Instance().GetCtx(gpuID)));
            return pNewSurf;
          },
          py::arg("format"), py::arg("width"), py::arg("height"),
          py::arg("gpu_id"), py::return_value_policy::take_ownership,
          R"pbdoc(
         Create a new surface with the specified parameters.

         Allocates a new surface with the given pixel format and dimensions
         on the specified GPU. The surface will be managed by the Python interpreter.

         :param format: Pixel format for the new surface
         :type format: Pixel_Format
         :param width: Width of the surface in pixels
         :type width: int
         :param height: Height of the surface in pixels
         :type height: int
         :param gpu_id: ID of the GPU to allocate memory on
         :type gpu_id: int
         :return: New surface with allocated memory
         :rtype: Surface
         :raises RuntimeError: If memory allocation fails
     )pbdoc")
      .def_static(
          "Make",
          [](Pixel_Format format, uint32_t newWidth, uint32_t newHeight,
             size_t ctx) {
            auto pNewSurf = shared_ptr<Surface>(
                Surface::Make(format, newWidth, newHeight, (CUcontext)ctx));
            return pNewSurf;
          },
          py::arg("format"), py::arg("width"), py::arg("height"),
          py::arg("context"), py::return_value_policy::take_ownership,
          R"pbdoc(
        Constructor method.

        :param format: target pixel format
        :param width: width in pixels
        :param height: height in pixels
        :param context: CUDA contet to use
    )pbdoc")
      .def(
          "__dlpack_device__",
          [](Surface& self) {
            if (self.NumPlanes() > 1U) {
              throw(
                  std::runtime_error("Surface has multiple planes. Use DLPack "
                                     "methods for particular plane instead."));
            }

            auto plane = self.GetSurfacePlane(0U);
            if (plane.FromDLPack()) {
              throw(std::runtime_error(
                  "Cant get __dlpack_device__ attribute from "
                  "SurfacePlane already created from DLPack."));
            }

            return std::make_tuple(plane.DLPackCtx().DeviceType(),
                                   plane.DeviceId());
          },
          R"pbdoc(
        DLPack: get device information.
    )pbdoc")
      .def(
          "__dlpack__",
          [](Surface& self, int stream) {
            if (self.NumPlanes() > 1U) {
              throw(
                  std::runtime_error("Surface has multiple planes. Use DLPack "
                                     "methods for particular plane instead."));
            }

            auto dlmt = self.ToDLPack();
            return py::capsule(dlmt, "dltensor", dlpack_capsule_deleter);
          },
          py::arg("stream") = 0,
          R"pbdoc(
        DLPack: get capsule.
    )pbdoc")
      .def_property_readonly(
          "__cuda_array_interface__",
          [](Surface& self) {
            if (self.NumPlanes() > 1U) {
              throw(
                  std::runtime_error("Surface has multiple planes. Use CAI "
                                     "methods for particular plane instead."));
            }

            auto plane = self.GetSurfacePlane(0U);
            CudaArrayInterfaceDescriptor cai;
            self.ToCAI(cai);

            return py::dict(
                "shape"_a = py::make_tuple(cai.m_shape[0], cai.m_shape[1],
                                           cai.m_shape[2]),
                "typestr"_a = cai.m_typestr,
                "data"_a = py::make_tuple(cai.m_ptr, cai.m_read_only),
                "version"_a = cai.m_version,
                "strides"_a = py::make_tuple(cai.m_strides[0], cai.m_strides[1],
                                             cai.m_strides[2]),
                "stream"_a = size_t(cai.m_stream));
          },
          R"pbdoc(
        CAI: export necessary dict.
    )pbdoc")
      .def_static(
          "from_dlpack",
          [](py::capsule cap, Pixel_Format fmt) {
            auto ptr = cap.ptr();
            if (!ptr) {
              throw std::runtime_error("Empty capsule.");
            }

            auto managed =
                (DLManagedTensor*)PyCapsule_GetPointer(ptr, "dltensor");
            if (!managed) {
              throw std::runtime_error("Capsule doesn't contain dltensor.");
            }

            auto surface = std::shared_ptr<Surface>(Surface::Make(fmt));
            if (!surface) {
              throw std::runtime_error("Failed to make Surface.");
            }

            auto surface_plane = SurfacePlane(*managed);
            surface->Update({&surface_plane});
            return surface;
          },
          py::arg("capsule"), py::arg("format") = Pixel_Format::RGB,
          R"pbdoc(
        DLPack: Make Surface from dlpack, don not own memory.

        :param capsule: capsule object with manager dltensor inside
        :param fmt: pixel format, by default python_vali.PixelFormat.RGB
        :return: Surface
        :rtype: python_vali.Surface
    )pbdoc")
      .def_static(
          "from_cai",
          [](py::object obj, Pixel_Format fmt) {
            if (!py::hasattr(obj, "__cuda_array_interface__")) {
              throw std::runtime_error("'__cuda_array_interface__' not found");
            }

            py::dict dict = obj.attr("__cuda_array_interface__");
            CudaArrayInterfaceDescriptor cai;

            for (auto item : dict) {
              auto key = item.first.cast<std::string>();
              if ("shape" == key) {
                const auto tup = item.second.cast<py::tuple>();
                const auto min_len = std::min(
                    CudaArrayInterfaceDescriptor::m_num_elems, py::len(tup));
                for (int i = 0; i < min_len; i++) {
                  cai.m_shape[i] = tup[i].cast<unsigned int>();
                }
              } else if ("strides" == key) {
                if (item.second.is_none()) {
                  continue;
                }
                const auto tup = item.second.cast<py::tuple>();
                const auto min_len = std::min(
                    CudaArrayInterfaceDescriptor::m_num_elems, py::len(tup));
                for (int i = 0; i < min_len; i++) {
                  cai.m_shape[i] = tup[i].cast<unsigned int>();
                }
              } else if ("typestr" == key) {
                cai.m_typestr = item.second.cast<std::string>();
              } else if ("data" == key) {
                const auto tup = item.second.cast<py::tuple>();
                cai.m_ptr = (CUdeviceptr)tup[0].cast<size_t>();
                cai.m_read_only = tup[1].cast<bool>();
              } else if ("stream" == key) {
                cai.m_stream = (CUstream)item.second.cast<int>();
              } else if ("version" == key) {
                auto const version = item.second.cast<int>();
                if (version != 3) {
                  throw std::runtime_error("Unsupported version " + key);
                }
              } else {
                throw std::runtime_error("Unsupported attribute " + key);
              }
            };

            std::string layout =
                SurfacePlane::CudaArrayInterfaceContext::LayoutFromFormat(
                    static_cast<int>(fmt));

            auto surface_plane = SurfacePlane(cai, layout);

            auto surface = std::shared_ptr<Surface>(Surface::Make(fmt));
            if (!surface) {
              throw std::runtime_error("Failed to make Surface.");
            }

            surface->Update({&surface_plane});
            return surface;
          },
          py::arg("dict"), py::arg("format") = Pixel_Format::RGB,
          R"pbdoc(
        DLPack: Make Surface from CAI, don not own memory.

        :param dict: dictionary which corresponds to CUDA Array Interface specs.
        :param fmt: pixel format, by default python_vali.PixelFormat.RGB
        :return: Surface
        :rtype: python_vali.Surface
    )pbdoc")
      .def_property_readonly(
          "Planes",
          [](Surface& self) {
            py::tuple planes(self.NumPlanes());
            for (int i = 0U; i < self.NumPlanes(); i++) {
              auto plane = self.GetSurfacePlane(i);
              planes[i] = py::cast(make_shared<SurfacePlane>(plane));
            }
            return planes;
          },
          R"pbdoc(
        Get SurfacePlane reference

        :param plane: SurfacePlane index
    )pbdoc")
      .def("__repr__", [](Surface& self) { return ToString(self); });
}
