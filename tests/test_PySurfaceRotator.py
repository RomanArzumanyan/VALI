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

    def test_unsupported_params(self):
        """
        This test checks that rotation with unsupported params will
        return proper error.
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        py_dec = vali.PyDecoder(input=gtInfo.uri, opts={}, gpu_id=0)

        py_rot = vali.PySurfaceRotator(gpu_id=0)

        surf = [
            vali.Surface.Make(
                vali.PixelFormat.NV12,
                py_dec.Width,
                py_dec.Height,
                gpu_id=0),

            vali.Surface.Make(
                vali.PixelFormat.NV12,
                py_dec.Height,
                py_dec.Width,
                gpu_id=0)
        ]

        success, details = py_dec.DecodeSingleSurface(surf[0])
        self.assertTrue(success)

        success, details = py_rot.Run(src=surf[0], dst=surf[-1], angle=90.0)
        self.assertFalse(success)
        self.assertEqual(details, vali.TaskExecInfo.NOT_SUPPORTED)

    @parameterized.expand([
        [90.0],
        [180.0],
        [270.0]
    ])
    def test_rotate(self, angle: float):
        """
        This test checks rotation.
        """
        jpeg_dec = nvimgcodec.Decoder()
        py_rot = vali.PySurfaceRotator(gpu_id=0)
        py_dwn = vali.PySurfaceDownloader(gpu_id=0)

        # Decode into GPU memory
        img = jpeg_dec.read("data/frame_0.jpg")

        # Share memory with VALI
        surf_src = vali.Surface.from_cai(img)

        # Create space for rotated Surface
        surf_dst = vali.Surface.Make(
            format=surf_src.Format,
            width=surf_src.Width if angle == 180.0 else surf_src.Height,
            height=surf_src.Height if angle == 180.0 else surf_src.Width,
            gpu_id=0)

        # Perform the rotation
        success, info = py_rot.Run(surf_src, surf_dst, angle)
        self.assertTrue(success)
        self.assertEqual(info, vali.TaskExecInfo.SUCCESS)

        # Download to RAM (necessary for comparison)
        frame = np.ndarray(dtype=np.uint8, shape=(surf_dst.Shape))
        success, info = py_dwn.Run(surf_dst, frame)
        self.assertTrue(success)
        self.assertEqual(info, vali.TaskExecInfo.SUCCESS)

        # Compare against etalon
        fname = "data/frame_0_" + str(int(angle)) + "_deg.jpg"

        psnr_score = tc.measurePSNR(np.asarray(Image.open(fname)), frame)
        self.assertGreaterEqual(psnr_score, psnr_threshold)


if __name__ == "__main__":
    unittest.main()
