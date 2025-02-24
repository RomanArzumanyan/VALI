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

import python_vali as vali
import unittest
from parameterized import parameterized

pixel_formats=[[ k,v[0]] for k, v in vali.PixelFormat.__entries.items() if k!='UNDEFINED' ]

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
        for idx in range(surf.NumPlanes):
            self.assertEqual(surf.Planes[idx].GpuMem, surf.Planes[idx].__cuda_array_interface__["data"][0])
    
if __name__ == "__main__":
    unittest.main()
