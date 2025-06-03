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

# We use 42 (dB) as the measure of similarity.
# If two images have PSNR higher than 42 (dB) we consider them the same.
psnr_threshold = 42.0


class TestSurfaceUD(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

        self.target_w = 640
        self.target_h = 360

    @staticmethod
    def get_gt_name(fmt: vali.PixelFormat) -> str:
        if fmt == vali.PixelFormat.NV12 or fmt == vali.PixelFormat.YUV420:
            return 'basic'
        elif fmt == vali.PixelFormat.P10 or fmt == vali.PixelFormat.YUV420_10bit:
            return 'hevc10'

    @parameterized.expand(vali.PySurfaceUD.SupportedFormats())
    def test_cpu_decode(self, src_fmt, dst_fmt):
        """
        This test checks UD transform.
        It takes source frame from CPU-accelerated decoder.
        """
        supported_formats = [
            vali.PixelFormat.YUV420,
            vali.PixelFormat.YUV420_10bit,
        ]

        if not src_fmt in supported_formats:
            return

        gt = tc.gt_by_name(self.get_gt_name(src_fmt))

        py_dec = vali.PyDecoder(input=gt.uri, opts={}, gpu_id=-1)
        py_up = vali.PyFrameUploader(gpu_id=0)
        py_ud = vali.PySurfaceUD(gpu_id=0)
        py_dwn = vali.PySurfaceDownloader(gpu_id=0)

        surf_src = vali.Surface.Make(
            src_fmt,
            py_dec.Width,
            py_dec.Height,
            0)

        frame_src = np.ndarray(shape=surf_src.Shape,
                               dtype=tc.to_numpy_dtype(surf_src))

        success, info = py_dec.DecodeSingleFrame(frame_src)
        if not success:
            self.fail(info)

        success, info = py_up.Run(frame_src, surf_src)
        if not success:
            self.fail(info)

        surf_dst = vali.Surface.Make(
            dst_fmt,
            self.target_w,
            self.target_h,
            gpu_id=0
        )

        success, info = py_ud.Run(surf_src, surf_dst)
        if not success:
            self.fail(info)

        frame = np.ndarray(shape=surf_dst.Shape,
                           dtype=tc.to_numpy_dtype(surf_dst))
        success, info = py_dwn.Run(surf_dst, frame)
        if not success:
            self.fail(info)

        fname = str(self.target_w) + 'x' + str(self.target_h) + \
            '_' + str(src_fmt) + '_' + str(dst_fmt) + '.raw'
        fname = 'data/' + fname

        with open(fname, 'rb') as f_in:
            gt = np.fromfile(fname, dtype=frame.dtype)
            gt = np.reshape(gt, frame.shape)
            self.assertGreaterEqual(tc.measure_psnr(gt, frame), psnr_threshold)

    @parameterized.expand(vali.PySurfaceUD.SupportedFormats())
    def test_gpu_decode(self, src_fmt, dst_fmt):
        """
        This test checks UD transform.
        It takes source frame from GPU-accelerated decoder.
        """
        supported_formats = [
            vali.PixelFormat.NV12,
            vali.PixelFormat.P10,
        ]

        if not src_fmt in supported_formats:
            return

        gt = tc.gt_by_name(self.get_gt_name(src_fmt))

        py_dec = vali.PyDecoder(input=gt.uri, opts={}, gpu_id=0)
        py_ud = vali.PySurfaceUD(gpu_id=0)
        py_dwn = vali.PySurfaceDownloader(gpu_id=0)

        surf_src = vali.Surface.Make(
            py_dec.Format,
            py_dec.Width,
            py_dec.Height,
            0)

        success, info = py_dec.DecodeSingleSurface(surf_src)
        if not success:
            self.fail(info)

        surf_dst = vali.Surface.Make(
            dst_fmt,
            self.target_w,
            self.target_h,
            gpu_id=0
        )

        success, info = py_ud.Run(surf_src, surf_dst)
        if not success:
            self.fail(info)

        frame = np.ndarray(shape=surf_dst.Shape,
                           dtype=tc.to_numpy_dtype(surf_dst))
        success, info = py_dwn.Run(surf_dst, frame)
        if not success:
            self.fail(info)

        fname = str(self.target_w) + 'x' + str(self.target_h) + \
            '_' + str(src_fmt) + '_' + str(dst_fmt) + '.raw'
        fname = 'data/' + fname

        with open(fname, 'rb') as f_in:
            gt = np.fromfile(fname, dtype=frame.dtype)
            gt = np.reshape(gt, frame.shape)
            self.assertGreaterEqual(tc.measure_psnr(gt, frame), psnr_threshold)


if __name__ == "__main__":
    unittest.main()
