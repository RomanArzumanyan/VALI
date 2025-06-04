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
        self.gpu_id = 0

    def test_unsupported_params(self):
        """
        This test checks that color conversion with unsupported params will
        return proper error.
        """
        with open("gt_files.json") as f:
            gt_info = tc.GroundTruth(**json.load(f)["basic"])

        py_dec = vali.PyDecoder(
            input=gt_info.uri,
            opts={},
            gpu_id=self.gpu_id)

        py_cvt = vali.PySurfaceConverter(gpu_id=self.gpu_id)

        # NV12 > RGB NPP conversion doesn't support BT601 + MPEG params.
        cc_ctx = vali.ColorspaceConversionContext(
            vali.ColorSpace.BT_601,
            vali.ColorRange.MPEG)

        surf_src = vali.Surface.Make(
            py_dec.Format, py_dec.Width, py_dec.Height, gpu_id=self.gpu_id)
        success, _ = py_dec.DecodeSingleSurface(surf_src)
        if not success:
            self.fail("Fail to decode surface")

        surf_dst = vali.Surface.Make(
            vali.PixelFormat.RGB, surf_src.Width, surf_src.Height, gpu_id=self.gpu_id)

        surf_dst, details = py_cvt.Run(surf_src, surf_dst, cc_ctx)
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

        gt_info = tc.gt_by_name("basic")

        py_dec = vali.PyDecoder(
            input=gt_info.uri,
            opts={},
            gpu_id=self.gpu_id)

        py_cvt = vali.PySurfaceConverter(
            gpu_id=self.gpu_id,
            stream=py_dec.Stream)

        event = vali.CudaStreamEvent(
            stream=py_dec.Stream,
            gpu_id=self.gpu_id)

        surf_src = vali.Surface.Make(
            py_dec.Format,
            py_dec.Width,
            py_dec.Height,
            gpu_id=self.gpu_id)

        surf_dst = vali.Surface.Make(
            vali.PixelFormat.RGB,
            surf_src.Width,
            surf_src.Height,
            gpu_id=self.gpu_id)

        success, _ = py_dec.DecodeSingleSurface(surf_src)
        if not success:
            self.fail("Fail to decode surface")

        success, details = py_cvt.RunAsync(
            surf_src, surf_dst) if is_async else py_cvt.Run(surf_src, surf_dst)
        self.assertEqual(details, vali.TaskExecInfo.SUCCESS)

        event.Record()
        event.Wait()

        # In case of failure exceptions will be thrown and test will be failed
        # So if we got here, async API is working fine

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
        dst_info = tc.gt_by_name("basic_rgb")
        pln_info = tc.gt_by_name("basic_rgb_planar")
        f_in = open(dst_info.uri, "rb")
        f_gt = open(pln_info.uri, "rb")

        py_upl = vali.PyFrameUploader(gpu_id=self.gpu_id)
        py_cvt = vali.PySurfaceConverter(gpu_id=self.gpu_id)
        py_dwn = vali.PySurfaceDownloader(gpu_id=self.gpu_id)

        # Use color space and range of original file.
        cc_ctx = vali.ColorspaceConversionContext(
            vali.ColorSpace.BT_709,
            vali.ColorRange.MPEG)

        for i in range(0, dst_info.num_frames):
            frame_size = dst_info.width * dst_info.height * 3
            # Read from ethalon RGB file
            dist_frame = np.fromfile(
                file=f_in, dtype=np.uint8, count=frame_size)

            # Upload to GPU
            surf_rgb = vali.Surface.Make(
                vali.PixelFormat.RGB,
                dst_info.width,
                dst_info.height,
                gpu_id=self.gpu_id)

            success = py_upl.Run(dist_frame, surf_rgb)
            if not success:
                self.fail("Fail to upload frame.")

            # Deinterleave
            surf_pln = vali.Surface.Make(
                vali.PixelFormat.RGB_PLANAR,
                pln_info.width,
                pln_info.height,
                gpu_id=self.gpu_id)

            # DtoH memcpy is blocking, no need to sync on event
            success, details = py_cvt.RunAsync(
                surf_rgb, surf_pln, cc_ctx) if is_async else py_cvt.Run(surf_rgb, surf_pln, cc_ctx)
            if not success:
                self.fail("Fail to convert RGB > RGB_PLANAR: " + details)

            # Download and save to disk
            dst_frame = np.ndarray(shape=(frame_size), dtype=np.uint8)
            success = py_dwn.Run(surf_pln, dst_frame)
            if not success:
                self.fail("Failed to download surface.")

            # Compare against GT
            pln_frame = np.fromfile(
                file=f_gt, dtype=np.uint8, count=frame_size)
            score = tc.measure_psnr(pln_frame, dst_frame)
            if score < psnr_threshold:
                tc.dump_to_disk(dst_frame, "cc", dst_info.width,
                                dst_info.height, "rgb_pln_dist")
                tc.dump_to_disk(pln_frame, "cc", pln_info.width,
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
        src_info = tc.gt_by_name("basic_nv12")
        dst_info = tc.gt_by_name("basic_rgb")

        py_upl = vali.PyFrameUploader(gpu_id=self.gpu_id)
        py_cvt = vali.PySurfaceConverter(gpu_id=self.gpu_id)
        py_dwn = vali.PySurfaceDownloader(gpu_id=self.gpu_id)

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
            surf_src = vali.Surface.Make(
                vali.PixelFormat.NV12,
                src_info.width,
                src_info.height,
                gpu_id=self.gpu_id)

            success = py_upl.Run(frame_src, surf_src)
            if not success:
                self.fail("Failed to upload frame")

            # Convert to RGB
            surf_dst = vali.Surface.Make(
                vali.PixelFormat.RGB,
                surf_src.Width,
                surf_src.Height,
                gpu_id=self.gpu_id)

            success, details = py_cvt.RunAsync(
                surf_src, surf_dst, cc_ctx) if is_async else py_cvt.Run(surf_src, surf_dst, cc_ctx)
            if not success:
                self.fail("Fail to convert surface " +
                          str(i) + ": " + str(details))

            # Download to numpy array
            dist_frame = np.ndarray(
                shape=(surf_dst.HostSize), dtype=np.uint8)
            if not py_dwn.Run(surf_dst, dist_frame):
                self.fail("Fail to download surface")

            # Read ethalon RGB frame and compare
            gt_frame = np.fromfile(
                dst_fin, np.uint8, surf_dst.HostSize)
            score = tc.measure_psnr(gt_frame, dist_frame)

            # Dump both frames to disk in case of failure
            if score < psnr_threshold:
                tc.dump_to_disk(dist_frame, "cc", dst_info.width,
                                dst_info.height, "rgb_dist")
                tc.dump_to_disk(gt_frame, "cc", dst_info.width,
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
        src_info = tc.gt_by_name("hevc10_p10")
        dst_info = tc.gt_by_name("hevc10_nv12")

        py_upl = vali.PyFrameUploader(gpu_id=self.gpu_id)
        py_cvt = vali.PySurfaceConverter(gpu_id=self.gpu_id)
        py_dwn = vali.PySurfaceDownloader(gpu_id=self.gpu_id)

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
            surf_src = vali.Surface.Make(
                vali.PixelFormat.P10,
                src_info.width,
                src_info.height,
                gpu_id=self.gpu_id)

            success = py_upl.Run(frame_src, surf_src)
            if not success:
                self.fail("Failed to upload frame")

            # Convert to destination format
            surf_dst = vali.Surface.Make(
                vali.PixelFormat.NV12,
                surf_src.Width,
                surf_src.Height,
                gpu_id=self.gpu_id)

            success, details = py_cvt.RunAsync(
                surf_src, surf_dst, cc_ctx) if is_async else py_cvt.Run(
                surf_src, surf_dst, cc_ctx)
            if not success:
                self.fail("Fail to convert surface " +
                          str(i) + ": " + str(details))

            # Download to numpy array
            dist_frame = np.ndarray(
                shape=surf_dst.Shape,
                dtype=tc.to_numpy_dtype(surf_dst))
            success, info = py_dwn.Run(surf_dst, dist_frame)
            if not success:
                self.fail(info)

            # Read ethalon frame and compare
            gt_frame = np.fromfile(
                file=dst_fin,
                dtype=tc.to_numpy_dtype(surf_dst),
                count=int(surf_dst.HostSize / surf_dst.Planes[0].ElemSize))

            gt_frame = np.reshape(gt_frame, dist_frame.shape)
            score = tc.measure_psnr(gt_frame, dist_frame)

            # Dump both frames to disk in case of failure
            if score < psnr_threshold:
                tc.dump_to_disk(dist_frame, "cc", dst_info.width,
                                dst_info.height, "dist")

                tc.dump_to_disk(gt_frame, "cc", dst_info.width,
                                dst_info.height, "gt")

                self.fail(
                    "PSNR score is below threshold: " + str(score))

        src_fin.close()
        dst_fin.close()


if __name__ == "__main__":
    unittest.main()
