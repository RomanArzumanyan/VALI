#
# Copyright 2025 Vision Labs LLC
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

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(os.path.join(cuda_path, "bin"))
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
import numpy as np
import unittest
import json
import test_common as tc
from parameterized import parameterized
from nvidia import nvimgcodec
from PIL import Image
from io import BytesIO

# We use 42 (dB) as the measure of similarity.
# If two images have PSNR higher than 42 (dB) we consider them the same.
psnr_threshold = 42.0


class TestSurfaceConverter(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_yuv420(self):
        """
        This test checks UD transform on YUV420 frame.
        """
        py_dec = vali.PyDecoder(input="data/test.mp4", opts={}, gpu_id=0)
        py_cvt = vali.PySurfaceConverter(gpu_id=0)
        py_ud = vali.PySurfaceUD(gpu_id=0)
        py_dwn = vali.PySurfaceDownloader(gpu_id=0)

        surf = [
            vali.Surface.Make(vali.PixelFormat.NV12,
                              py_dec.Width, py_dec.Height, 0),
            vali.Surface.Make(vali.PixelFormat.YUV420,
                              py_dec.Width, py_dec.Height, 0),
            vali.Surface.Make(vali.PixelFormat.YUV444, 640, 360, 0)
        ]

        success, info = py_dec.DecodeSingleSurface(surf[0])
        if not success:
            self.fail(info)

        success, info = py_cvt.Run(surf[0], surf[1])
        if not success:
            self.fail(info)

        success, info = py_ud.Run(surf[1], surf[2])
        if not success:
            self.fail(info)

        frame = np.ndarray(dtype=np.uint8, shape=surf[-1].Shape)
        success, info = py_dwn.Run(surf[-1], frame)
        if not success:
            self.fail(info)

        with open("output.yuv", "wb") as f_out:
            f_out.write(frame)


if __name__ == "__main__":
    unittest.main()
