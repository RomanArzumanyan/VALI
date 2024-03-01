#
# Copyright 2024 Vision Labs LLC
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

import PyNvCodec as nvc
import numpy as np
import unittest
import json
import test_common as tc

# We use 44 (dB) as the measure of similarity.
# If two images have PSNR higher than 44 (dB) we consider them the same.
psnr_threshold = 44.0


class TestFrameConverter(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_yuv420_rgb(self):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            yuvInfo = tc.GroundTruth(**gt_values["basic"])
            rgbInfo = tc.GroundTruth(**gt_values["basic_rgb"])

        ffDec = nvc.PyFfmpegDecoder(
            input=yuvInfo.uri,
            opts={})

        ffCvt = nvc.PyFrameConverter(
            ffDec.Width(),
            ffDec.Height(),
            ffDec.Format(),
            nvc.PixelFormat.RGB)

        # Use color space and range of original file.
        ccCtx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_709,
            nvc.ColorRange.MPEG)

        yuv_frame = np.ndarray(shape=(), dtype=np.uint8)
        rgb_frame = np.ndarray(shape=(), dtype=np.uint8)
        frame_size = rgbInfo.width * rgbInfo.height * 3

        with open(rgbInfo.uri, "rb") as f_in:
            for i in range(0, rgbInfo.num_frames):
                success, _ = ffDec.DecodeSingleFrame(yuv_frame)
                if not success:
                    self.fail("Fail to decode frame: " + str(_))

                success, _ = ffCvt.Execute(yuv_frame, rgb_frame, ccCtx)
                if not success:
                    self.fail("Fail to convert frame: " + str(_))

                rgb_ethalon = np.fromfile(f_in, np.uint8, frame_size)
                score = tc.measurePSNR(rgb_ethalon, rgb_frame)

                if score < psnr_threshold:
                    tc.dumpFrameToDisk(rgb_frame, "cc", rgbInfo.width,
                                       rgbInfo.height, "rgb")
                    self.fail(
                        "PSNR score is below threshold: " + str(score))

if __name__ == "__main__":
    unittest.main()
