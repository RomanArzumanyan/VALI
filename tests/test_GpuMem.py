#
# Copyright 2024 Yves33
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

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import sys
import os
from os.path import join, dirname

__DBG_IMG__ = False

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(cuda_path)
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)

    # Add PATH as well for minor CUDA releases
    sys_path = os.environ["PATH"]
    if sys_path:
        paths = sys_path.split(";")
        for path in paths:
            if os.path.isdir(path):
                os.add_dll_directory(path)
    else:
        print("PATH environment variable is not set.", file=sys.stderr)
        exit(1)

import python_vali as vali
import unittest
from parameterized import parameterized

import ctypes, sys
sys.path.insert(0,"../extern/dlpack/apps/numpy_dlpack/dlpack")
from dlpack import _c_str_dltensor, DLManagedTensor

ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

pixel_formats=[[ k,v[0]] for k, v in vali.PixelFormat.__entries.items() if k!='UNDEFINED' ]

def dlpackptr(plane:vali.SurfacePlane):
    pycapsule=plane.__dlpack__()
    dl_managed_tensor = ctypes.pythonapi.PyCapsule_GetPointer(pycapsule, _c_str_dltensor)
    dl_managed_tensor_ptr = ctypes.cast(dl_managed_tensor, ctypes.POINTER(DLManagedTensor))
    return dl_managed_tensor_ptr.contents.dl_tensor.data

class TestGpuMem(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    '''@parameterized.expand([
        ["nv12", vali.PixelFormat.NV12,],
        ["rgb", vali.PixelFormat.RGB,],
    ])'''
    @ parameterized.expand(pixel_formats)
    def test_gpu_mem(self, name, pix_fmt):
        surf = vali.Surface.Make(format = pix_fmt, width = 640, height = 480, gpu_id = 0)
        for idx in range(surf.NumPlanes()):
            self.assertEqual(surf.PlanePtr(idx).GpuMem(), dlpackptr(surf.PlanePtr(idx)))
    
if __name__ == "__main__":
    unittest.main()
