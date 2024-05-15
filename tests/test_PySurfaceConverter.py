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

# We use 42 (dB) as the measure of similarity.
# If two images have PSNR higher than 42 (dB) we consider them the same.
psnr_threshold = 42.0


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

        surf_dst = nvc.Surface.Make(
            nvc.PixelFormat.RGB, surf_src.Width(), surf_src.Height(), gpu_id=0)

        surf_dst, details = nvCvt.Execute(surf_src, surf_dst, cc_ctx)
        self.assertEqual(details, nvc.TaskExecInfo.UNSUPPORTED_FMT_CONV_PARAMS)

    def test_rgb_deinterleave(self):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            rgbInfo = tc.GroundTruth(**gt_values["basic_rgb"])
            plnInfo = tc.GroundTruth(**gt_values["basic_rgb_planar"])

            nvUpl = nvc.PyFrameUploader(
                gpu_id=0)

            toPLN = nvc.PySurfaceConverter(
                rgbInfo.width,
                rgbInfo.height,
                nvc.PixelFormat.RGB,
                nvc.PixelFormat.RGB_PLANAR,
                gpu_id=0)

            nvDwn = nvc.PySurfaceDownloader(gpu_id=0)

            # Use color space and range of original file.
            cc_ctx = nvc.ColorspaceConversionContext(
                nvc.ColorSpace.BT_709,
                nvc.ColorRange.MPEG)

            f_in = open(rgbInfo.uri, "rb")
            f_gt = open(plnInfo.uri, "rb")
            for i in range(0, rgbInfo.num_frames):
                frame_size = rgbInfo.width * rgbInfo.height * 3
                # Read from ethalon RGB file
                rgb_frame = np.fromfile(file=f_in, dtype=np.uint8, count=frame_size)

                # Upload to GPU
                surf_rgb = nvc.Surface.Make(nvc.PixelFormat.RGB, rgbInfo.width, 
                                            rgbInfo.height, gpu_id=0)
                success = nvUpl.Run(rgb_frame, surf_rgb)
                if not success:
                    self.fail("Fail to upload frame.")

                # Deinterleave
                surf_pln = nvc.Surface.Make(
                    nvc.PixelFormat.RGB_PLANAR,
                    plnInfo.width,
                    plnInfo.height,
                    gpu_id=0)
                
                success, details = toPLN.Execute(surf_rgb, surf_pln, cc_ctx)
                if not success:
                    self.fail("Fail to convert RGB > RGB_PLANAR: " + details)

                # Download and save to disk
                dst_frame = np.ndarray(shape=(frame_size), dtype=np.uint8)
                success = nvDwn.Run(surf_pln, dst_frame)
                if not success:
                    self.fail("Failed to download surface.")

                # Compare against GT
                pln_frame = np.fromfile(
                    file=f_gt, dtype=np.uint8, count=frame_size)
                score = tc.measurePSNR(pln_frame, dst_frame)
                if score < psnr_threshold:
                    tc.dumpFrameToDisk(dst_frame, "cc", rgbInfo.width,
                                       rgbInfo.height, "rgb_pln_dist")
                    tc.dumpFrameToDisk(pln_frame, "cc", plnInfo.width,
                                       plnInfo.height, "rgb_pln_gt")
                    self.fail(
                        "PSNR score is below threshold: " + str(score))

            f_in.close()
            f_gt.close()

    def test_nv12_rgb(self):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            nv12Info = tc.GroundTruth(**gt_values["basic_nv12"])
            rgbInfo = tc.GroundTruth(**gt_values["basic_rgb"])

        nvUpl = nvc.PyFrameUploader(
            gpu_id=0)

        nvCvt = nvc.PySurfaceConverter(
            nv12Info.width,
            nv12Info.height,
            nvc.PixelFormat.NV12,
            nvc.PixelFormat.RGB,
            gpu_id=0)

        nvDwn = nvc.PySurfaceDownloader(gpu_id=0)

        # Use color space and range of original file.
        cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_709,
            nvc.ColorRange.MPEG)

        nv12_fin = open(nv12Info.uri, "rb")
        rgb_fin = open(rgbInfo.uri, "rb")

        for i in range(0, nv12Info.num_frames):
            # Read NV12 frame from file
            frame_src = np.fromfile(
                nv12_fin,
                np.uint8,
                int(nv12Info.width * nv12Info.height * 3 / 2))

            # Upload to GPU
            surf_src = nvc.Surface.Make(nvc.PixelFormat.NV12, nv12Info.width, 
                                            nv12Info.height, gpu_id=0)
            success = nvUpl.Run(frame_src, surf_src)
            if not success:
                self.fail("Failed to upload frame")

            # Convert to RGB
            surf_dst = nvc.Surface.Make(
                nvc.PixelFormat.RGB, surf_src.Width(), surf_src.Height(), gpu_id=0)

            success, details = nvCvt.Execute(
                surf_src, surf_dst, cc_ctx)

            if not success:
                self.fail("Fail to convert surface " +
                          str(i) + ": " + str(details))

            # Download to numpy array
            rgb_frame = np.ndarray(
                shape=(surf_dst.HostSize()), dtype=np.uint8)
            success = nvDwn.Run(
                surf_dst, rgb_frame)
            if not success:
                self.fail("Fail to download surface")

            # Read ethalon RGB frame and compare
            rgb_ethalon = np.fromfile(
                rgb_fin, np.uint8, surf_dst.HostSize())
            score = tc.measurePSNR(rgb_ethalon, rgb_frame)

            # Dump both frames to disk in case of failure
            if score < psnr_threshold:
                tc.dumpFrameToDisk(rgb_frame, "cc", rgbInfo.width,
                                   rgbInfo.height, "rgb_dist")
                tc.dumpFrameToDisk(rgb_ethalon, "cc", rgbInfo.width,
                                   rgbInfo.height, "rgb_gt")
                self.fail(
                    "PSNR score is below threshold: " + str(score))

        nv12_fin.close()
        rgb_fin.close()


if __name__ == "__main__":
    unittest.main()
