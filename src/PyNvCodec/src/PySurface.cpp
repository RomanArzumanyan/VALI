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
#include "PyNvCodec.hpp"
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
        Get width in pixels
    )pbdoc")
      .def_property_readonly("Height", &SurfacePlane::Height,
                             R"pbdoc(
        Get height in pixels
    )pbdoc")
      .def_property_readonly("Pitch", &SurfacePlane::Pitch,
                             R"pbdoc(
        Get pitch in bytes
    )pbdoc")
      .def_property_readonly("ElemSize", &SurfacePlane::ElemSize,
                             R"pbdoc(
        Get element size in bytes
    )pbdoc")
      .def_property_readonly("HostFrameSize", &SurfacePlane::HostMemSize,
                             R"pbdoc(
        Get amount of host memory needed to store this SurfacePlane
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
        DLPack: get device information.
    )pbdoc")
      .def(
          "__dlpack__",
          [](shared_ptr<SurfacePlane> self, int stream) {
            auto dlmt = self->ToDLPack();
            return py::capsule(dlmt, "dltensor", dlpack_capsule_deleter);
          },
          py::arg("stream") = 0,
          R"pbdoc(
        DLPack: get capsule.
    )pbdoc")
      .def("__repr__",
           [](shared_ptr<SurfacePlane> self) { return ToString(*self.get()); });

  py::class_<Surface, shared_ptr<Surface>>(
      m, "Surface", "Image stored in vRAM. Consists of 1+ SurfacePlane(s).")
      .def_property_readonly(
          "Width", [](Surface& self) { return self.Width(0); },
          R"pbdoc(
        Width in pixels of plane 0.
    )pbdoc")
      .def_property_readonly(
          "Height", [](Surface& self) { return self.Height(0); },
          R"pbdoc(
        Height in pixels of plane 0.
    )pbdoc")
      .def_property_readonly(
          "Pitch", [](Surface& self) { return self.Pitch(0); },
          R"pbdoc(
        Pitch in bytes of plane 0.
    )pbdoc")
      .def_property_readonly("Format", &Surface::PixelFormat,
                             R"pbdoc(
        Get pixel format
    )pbdoc")
      .def_property_readonly("IsEmpty", &Surface::Empty,
                             R"pbdoc(
        Tell if Surface plane has memory allocated or it's empty inside.
    )pbdoc")
      .def_property_readonly("NumPlanes", &Surface::NumPlanes,
                             R"pbdoc(
        Number of SurfacePlanes
    )pbdoc")
      .def_property_readonly("HostSize", &Surface::HostMemSize,
                             R"pbdoc(
        Amount of memory in bytes which is needed for DtoH memcopy.
    )pbdoc")
      .def_property_readonly("IsOwnMemory", &Surface::OwnMemory,
                             R"pbdoc(
        Return True if Surface owns memory, False if it only references actual
        memory allocation but doesn't own it.
    )pbdoc")
      .def("Clone", &Surface::Clone, py::return_value_policy::take_ownership,
           R"pbdoc(
        CUDA mem alloc + deep copy.
        Object returned is manager by Python interpreter.
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
        Constructor method.

        :param format: target pixel format
        :param width: width in pixels
        :param height: height in pixels
        :param gpu_id: GPU to use
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
        :param fmt: pixel format, by default PyNvCodec.PixelFormat.RGB
        :return: Surface
        :rtype: PyNvCodec.Surface
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