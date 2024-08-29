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
from PIL import Image
from io import BytesIO
from parameterized import parameterized
import time

# We use 42 (dB) as the measure of similarity.
# If two images have PSNR higher than 42 (dB) we consider them the same.
psnr_threshold = 42.0


class TestJpegEncoder(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    @unittest.skip("Need to prepare load test env. more carefully.")
    def test_compress_multiple(self):
        """
        This test measures compression time of individual frames vs batch.
        Batch is expected to be faster.

        This test is skipped for now because in low load scenarios batch
        encode turns to be slightly slower. I assume that happens because
        CUDA stream isn't used much and is more tolerant to constant sync.
        """
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            gtInfo = tc.GroundTruth(**gt_values["basic"])

        nvDec = vali.PyDecoder(
            input=gtInfo.uri,
            opts={},
            gpu_id=0)

        nvCvt = vali.PySurfaceConverter(
            src_format=vali.PixelFormat.NV12,
            dst_format=vali.PixelFormat.RGB,
            gpu_id=0)

        nvJpg = vali.PyNvJpegEncoder(gpu_id=0)
        nvJpgCtx = nvJpg.Context(
            compression=100,
            pixel_format=vali.PixelFormat.RGB)

        # Make input and output Surfaces
        surf_src = vali.Surface.Make(
            vali.PixelFormat.NV12,
            nvDec.Width,
            nvDec.Height,
            gpu_id=0)

        surf_dst = vali.Surface.Make(
            vali.PixelFormat.RGB,
            nvDec.Width,
            nvDec.Height,
            gpu_id=0)

        # Compress one by one
        start = time.time()
        for i in range(0, gtInfo.num_frames):
            success, info = nvDec.DecodeSingleSurface(surf_src)
            if not success:
                self.fail("Failed to decode surface: " + str(info))

            success, info = nvCvt.Run(surf_src, surf_dst)
            if not success:
                self.fail("Failed to convert surface: " + str(info))

            buffers, info = nvJpg.Run(nvJpgCtx, [surf_dst])
            if len(buffers) != 1:
                self.fail("Failed to compress surfaces: " + str(info))
        time_single = time.time() - start

        # Compress in batch
        # Reset decoder because it has reached EOF
        nvDec = vali.PyDecoder(
            input=gtInfo.uri,
            opts={},
            gpu_id=0)

        # Pre-allocate memory to avoid overhead
        surfaces = []
        for i in range(0, gtInfo.num_frames):
            surfaces.append(vali.Surface.Make(
                vali.PixelFormat.RGB,
                nvDec.Width,
                nvDec.Height,
                gpu_id=0))

        start = time.time()
        for i in range(0, gtInfo.num_frames):
            success, info = nvDec.DecodeSingleSurface(surf_src)
            if not success:
                self.fail("Failed to decode surface: " + str(info))

            success, info = nvCvt.Run(surf_src, surfaces[i])
            if not success:
                self.fail("Failed to convert surface: " + str(info))

        buffers, info = nvJpg.Run(nvJpgCtx, surfaces)
        if len(buffers) != len(surfaces):
            self.fail("Failed to compress surfaces: " + str(info))
        time_batch = time.time() - start

        self.assertLessEqual(time_batch, time_single)

    @parameterized.expand([
        ["yuv420", vali.PixelFormat.YUV420],
        ["rgb24", vali.PixelFormat.RGB],
    ])
    def test_compress(self, case_name, dst_fmt):
        """
        This test checks compression quality for RGB format and the very fact
        of successful compression for YUV420 format.

        For RGB, PSNR between raw and compressed frames is measured.
        It has to be over 42 dB.
        """
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            nv12Info = tc.GroundTruth(**gt_values["basic_nv12"])

        nvUpl = vali.PyFrameUploader(gpu_id=0)
        nvDwn = vali.PySurfaceDownloader(gpu_id=0)
        nvCvt = vali.PySurfaceConverter(
            vali.PixelFormat.NV12, dst_fmt, gpu_id=0)
        nvJpg = vali.PyNvJpegEncoder(gpu_id=0)
        nvJpgCtx = nvJpg.Context(
            compression=100,
            pixel_format=dst_fmt)

        nv12_fin = open(nv12Info.uri, "rb")

        # Make input and output Surfaces
        surf_src = vali.Surface.Make(
            vali.PixelFormat.NV12,
            nv12Info.width,
            nv12Info.height,
            gpu_id=0)

        surf_dst = vali.Surface.Make(
            dst_fmt,
            nv12Info.width,
            nv12Info.height,
            gpu_id=0)

        # Read input and GT frames from file
        frame_src = np.fromfile(nv12_fin, np.uint8, surf_src.HostSize)
        frame_dst = np.ndarray(dtype=np.uint8, shape=(surf_dst.HostSize))

        for i in range(0, nv12Info.num_frames):
            # Upload to GPU
            success, info = nvUpl.Run(frame_src, surf_src)
            if not success:
                self.fail("Failed to upload frame: " + str(info))

            # Convert to dst pixel format
            success, info = nvCvt.Run(surf_src, surf_dst)
            if not success:
                self.fail("Failed to convert surface: " + str(info))

            # Compress
            buffers, info = nvJpg.Run(nvJpgCtx, [surf_dst])
            self.assertEqual(len(buffers), 1)
            self.assertGreater(len(buffers[0]), 0)

            # Decode with PIL and compare against etalon
            # Can do that only for RGB input for now because that's what PIL
            # supports.
            if dst_fmt == vali.PixelFormat.RGB:
                jpeg_bytes = BytesIO(np.ndarray.tobytes(buffers[0]))
                img_recon = np.asarray(Image.open(jpeg_bytes)).flatten()

                success, info = nvDwn.Run(surf_dst, frame_dst)
                if not success:
                    self.fail("Failed to download surface: " + str(info))

                score = tc.measurePSNR(img_recon, frame_dst)
                self.assertGreaterEqual(score, psnr_threshold)

        nv12_fin.close()


if __name__ == "__main__":
    unittest.main()
