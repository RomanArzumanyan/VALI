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


class TestSurfaceConverter(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_unsupported_params(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(
            input=gtInfo.uri,
            gpu_id=0)

        nvCvt = nvc.PySurfaceConverter(
            nvDec.Width(),
            nvDec.Height(),
            nvc.PixelFormat.NV12,
            nvc.PixelFormat.RGB,
            gpu_id=0)

        # NV12 > RGB NPP conversion doesn't support BT601 + MPEG params.
        cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_601,
            nvc.ColorRange.MPEG)

        surf_src, _ = nvDec.DecodeSingleSurface()
        if surf_src.Empty():
            self.fail("Fail to decode surface")

        surf_dst, details = nvCvt.Execute(surf_src, cc_ctx)
        self.assertTrue(surf_dst.Empty())
        self.assertEqual(details, nvc.TaskExecInfo.UNSUPPORTED_FMT_CONV_PARAMS)

    def test_nv12_rgb(self, ):
        modes = ['inplace', 'generic']
        for mode in modes:
            with self.subTest(mode):
                with open("gt_files.json") as f:
                    gt_values = json.load(f)
                    yuvInfo = tc.GroundTruth(**gt_values["basic"])
                    rgbInfo = tc.GroundTruth(**gt_values["basic_rgb"])

                nvDec = nvc.PyNvDecoder(
                    input=yuvInfo.uri,
                    gpu_id=0)

                nvCvt = nvc.PySurfaceConverter(
                    nvDec.Width(),
                    nvDec.Height(),
                    nvc.PixelFormat.NV12,
                    nvc.PixelFormat.RGB,
                    gpu_id=0)

                nvDwn = nvc.PySurfaceDownloader(
                    nvDec.Width(),
                    nvDec.Height(),
                    nvc.PixelFormat.RGB,
                    gpu_id=0)

                # Use color space and range of original file.
                cc_ctx = nvc.ColorspaceConversionContext(
                    nvc.ColorSpace.BT_709,
                    nvc.ColorRange.MPEG)

                frame_size = rgbInfo.width * rgbInfo.height * 3
                rgb_frame = np.ndarray(shape=(frame_size), dtype=np.uint8)

                with open(rgbInfo.uri, "rb") as f_in:
                    for i in range(0, rgbInfo.num_frames):
                        surf_src, _ = nvDec.DecodeSingleSurface()
                        if surf_src.Empty():
                            self.fail("Fail to decode surface")

                        if mode == 'inplace':
                            surf_dst = nvc.Surface.Make(
                                nvc.PixelFormat.RGB, nvDec.Width(), nvDec.Height(), gpu_id=0)

                            if surf_dst.Empty():
                                self.fail("Fail to make RGB surface")

                            success, details = nvCvt.Execute(
                                surf_src, surf_dst, cc_ctx)
                        elif mode == 'generic':
                            surf_dst, details = nvCvt.Execute(
                                surf_src, cc_ctx)

                        if not success:
                            self.fail("Fail to convert surface " +
                                      str(i) + ": " + str(details))

                        success = nvDwn.DownloadSingleSurface(
                            surf_dst, rgb_frame)
                        if not success:
                            self.fail("Fail to download surface")

                        rgb_ethalon = np.fromfile(f_in, np.uint8, frame_size)
                        score = tc.measurePSNR(rgb_ethalon, rgb_frame)

                        if score < psnr_threshold:
                            tc.dumpFrameToDisk(rgb_frame, "cc", rgbInfo.width,
                                               rgbInfo.height, "rgb")
                            self.fail(
                                "PSNR score is below threshold: " + str(score))


if __name__ == "__main__":
    unittest.main()
