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
from parameterized import parameterized

# We use 42 (dB) as the measure of similarity.
# If two images have PSNR higher than 42 (dB) we consider them the same.
psnr_threshold = 42.0


class TestSurfaceConverter(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_unsupported_params(self):
        """
        This test checks that color conversion with unsupported params will
        return proper error.
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(
            input=gtInfo.uri, opts={}, gpu_id=0)

        nvCvt = vali.PySurfaceConverter(gpu_id=0)

        # NV12 > RGB NPP conversion doesn't support BT601 + MPEG params.
        cc_ctx = vali.ColorspaceConversionContext(
            vali.ColorSpace.BT_601,
            vali.ColorRange.MPEG)

        surf_src = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)
        success, _ = pyDec.DecodeSingleSurface(surf_src)
        if not success:
            self.fail("Fail to decode surface")

        surf_dst = vali.Surface.Make(
            vali.PixelFormat.RGB, surf_src.Width, surf_src.Height, gpu_id=0)

        surf_dst, details = nvCvt.Run(surf_src, surf_dst, cc_ctx)
        self.assertEqual(
            details, vali.TaskExecInfo.UNSUPPORTED_FMT_CONV_PARAMS)

    @parameterized.expand([
        [True],
        [False]
    ])
    def test_no_cc_ctx(self, is_async):
        """
        This test checks that color conversion can be run with default color
        conversion context parameters.

        Args:
            is_async (bool): True if launched in non-blocking mode, False otherwise.
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(
            input=gtInfo.uri, opts={}, gpu_id=0)

        nvCvt = vali.PySurfaceConverter(
            gpu_id=0,
            stream=pyDec.Stream)

        surf_src = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)
        success, _ = pyDec.DecodeSingleSurface(surf_src)
        if not success:
            self.fail("Fail to decode surface")

        surf_dst = vali.Surface.Make(
            vali.PixelFormat.RGB, surf_src.Width, surf_src.Height, gpu_id=0)

        if is_async:
            success, details, event = nvCvt.RunAsync(surf_src, surf_dst)
            event.Wait()
        else:
            success, details = nvCvt.Run(surf_src, surf_dst)
        self.assertEqual(details, vali.TaskExecInfo.SUCCESS)

    @parameterized.expand([
        [True],
        [False]
    ])
    def test_rgb_deinterleave(self, is_async):
        """
        This test checks NV12 -> YUV420 conversion

        Args:
            is_async (bool): True if launched in non-blocking mode, False otherwise.
        """
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            dst_info = tc.GroundTruth(**gt_values["basic_rgb"])
            pln_info = tc.GroundTruth(**gt_values["basic_rgb_planar"])

            nvUpl = vali.PyFrameUploader(
                gpu_id=0)

            toPLN = vali.PySurfaceConverter(gpu_id=0)

            nvDwn = vali.PySurfaceDownloader(gpu_id=0)

            # Use color space and range of original file.
            cc_ctx = vali.ColorspaceConversionContext(
                vali.ColorSpace.BT_709,
                vali.ColorRange.MPEG)

            f_in = open(dst_info.uri, "rb")
            f_gt = open(pln_info.uri, "rb")
            for i in range(0, dst_info.num_frames):
                frame_size = dst_info.width * dst_info.height * 3
                # Read from ethalon RGB file
                dist_frame = np.fromfile(
                    file=f_in, dtype=np.uint8, count=frame_size)

                # Upload to GPU
                surf_rgb = vali.Surface.Make(vali.PixelFormat.RGB, dst_info.width,
                                             dst_info.height, gpu_id=0)
                success = nvUpl.Run(dist_frame, surf_rgb)
                if not success:
                    self.fail("Fail to upload frame.")

                # Deinterleave
                surf_pln = vali.Surface.Make(
                    vali.PixelFormat.RGB_PLANAR,
                    pln_info.width,
                    pln_info.height,
                    gpu_id=0)

                if is_async:
                    success, details, _ = toPLN.RunAsync(
                        surf_rgb, surf_pln, cc_ctx, record_event=False)
                else:
                    success, details = toPLN.Run(surf_rgb, surf_pln, cc_ctx)
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
                    tc.dumpFrameToDisk(dst_frame, "cc", dst_info.width,
                                       dst_info.height, "rgb_pln_dist")
                    tc.dumpFrameToDisk(pln_frame, "cc", pln_info.width,
                                       pln_info.height, "rgb_pln_gt")
                    self.fail(
                        "PSNR score is below threshold: " + str(score))

            f_in.close()
            f_gt.close()

    @parameterized.expand([
        [True],
        [False]
    ])
    def test_nv12_rgb(self, is_async: bool):
        """
        This test checks NV12 -> RGB conversion

        Args:
            is_async (bool): True if launched in non-blocking mode, False otherwise.
        """
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            src_info = tc.GroundTruth(**gt_values["basic_nv12"])
            dst_info = tc.GroundTruth(**gt_values["basic_rgb"])

        nvUpl = vali.PyFrameUploader(gpu_id=0)

        nvCvt = vali.PySurfaceConverter(gpu_id=0)

        nvDwn = vali.PySurfaceDownloader(gpu_id=0)

        # Use color space and range of original file.
        cc_ctx = vali.ColorspaceConversionContext(
            vali.ColorSpace.BT_709,
            vali.ColorRange.MPEG)

        src_fin = open(src_info.uri, "rb")
        dst_fin = open(dst_info.uri, "rb")

        for i in range(0, src_info.num_frames):
            # Read NV12 frame from file
            frame_src = np.fromfile(
                src_fin, np.uint8, int(src_info.width * src_info.height * 3 / 2))

            # Upload to GPU
            surf_src = vali.Surface.Make(vali.PixelFormat.NV12, src_info.width,
                                         src_info.height, gpu_id=0)
            success = nvUpl.Run(frame_src, surf_src)
            if not success:
                self.fail("Failed to upload frame")

            # Convert to RGB
            surf_dst = vali.Surface.Make(
                vali.PixelFormat.RGB, surf_src.Width, surf_src.Height, gpu_id=0)

            if is_async:
                success, details, _ = nvCvt.RunAsync(
                    surf_src, surf_dst, cc_ctx, record_event=False)
            else:
                success, details = nvCvt.Run(surf_src, surf_dst, cc_ctx)

            if not success:
                self.fail("Fail to convert surface " +
                          str(i) + ": " + str(details))

            # Download to numpy array
            dist_frame = np.ndarray(
                shape=(surf_dst.HostSize), dtype=np.uint8)
            if not nvDwn.Run(surf_dst, dist_frame):
                self.fail("Fail to download surface")

            # Read ethalon RGB frame and compare
            gt_frame = np.fromfile(
                dst_fin, np.uint8, surf_dst.HostSize)
            score = tc.measurePSNR(gt_frame, dist_frame)

            # Dump both frames to disk in case of failure
            if score < psnr_threshold:
                tc.dumpFrameToDisk(dist_frame, "cc", dst_info.width,
                                   dst_info.height, "rgb_dist")
                tc.dumpFrameToDisk(gt_frame, "cc", dst_info.width,
                                   dst_info.height, "rgb_gt")
                self.fail(
                    "PSNR score is below threshold: " + str(score))

        src_fin.close()
        dst_fin.close()

    @parameterized.expand([
        [True],
        [False]
    ])
    def test_p10_nv12(self, is_async: bool):
        """
        This test checks HDR to SDR conversion (10 bit NV12 -> 8 bit NV12)

        Args:
            is_async (bool): True if launched in non-blocking mode, False otherwise.
        """
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            src_info = tc.GroundTruth(**gt_values["hevc10_p10"])
            dst_info = tc.GroundTruth(**gt_values["hevc10_nv12"])

        nvUpl = vali.PyFrameUploader(
            gpu_id=0)

        nvCvt = vali.PySurfaceConverter(gpu_id=0)

        nvDwn = vali.PySurfaceDownloader(gpu_id=0)

        # Use color space and range of original file.
        cc_ctx = vali.ColorspaceConversionContext(
            vali.ColorSpace.BT_709,
            vali.ColorRange.MPEG)

        src_fin = open(src_info.uri, "rb")
        dst_fin = open(dst_info.uri, "rb")

        for i in range(0, src_info.num_frames):
            # Read source frame from file
            frame_src = np.fromfile(
                src_fin, np.uint16, int(src_info.width * src_info.height * 3 / 2))

            # Upload to GPU
            surf_src = vali.Surface.Make(vali.PixelFormat.P10, src_info.width,
                                         src_info.height, gpu_id=0)
            success = nvUpl.Run(frame_src, surf_src)
            if not success:
                self.fail("Failed to upload frame")

            # Convert to destination format
            surf_dst = vali.Surface.Make(
                vali.PixelFormat.NV12, surf_src.Width, surf_src.Height, gpu_id=0)

            if is_async:
                success, details, _ = nvCvt.RunAsync(
                    surf_src, surf_dst, cc_ctx, record_event=False)
            else:
                success, details = nvCvt.Run(
                    surf_src, surf_dst, cc_ctx)

            if not success:
                self.fail("Fail to convert surface " +
                          str(i) + ": " + str(details))

            # Download to numpy array
            dist_frame = np.ndarray(
                shape=(surf_dst.HostSize), dtype=np.uint8)
            success = nvDwn.Run(surf_dst, dist_frame)
            if not success:
                self.fail("Fail to download surface")

            # Read ethalon frame and compare
            gt_frame = np.fromfile(dst_fin, np.uint8, surf_dst.HostSize)
            score = tc.measurePSNR(gt_frame, dist_frame)

            # Dump both frames to disk in case of failure
            if score < psnr_threshold:
                tc.dumpFrameToDisk(dist_frame, "cc", dst_info.width,
                                   dst_info.height, "dist")

                tc.dumpFrameToDisk(gt_frame, "cc", dst_info.width,
                                   dst_info.height, "gt")

                self.fail(
                    "PSNR score is below threshold: " + str(score))

        src_fin.close()
        dst_fin.close()


if __name__ == "__main__":
    unittest.main()
