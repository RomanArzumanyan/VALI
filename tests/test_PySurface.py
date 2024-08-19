#
# Copyright 2021 NVIDIA Corporation
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
import torch
import logging


class TestSurface(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        self.log = logging.getLogger(__name__)

    def test_tensor_from_splane(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

            pyDec = vali.PyDecoder(
                input=gtInfo.uri, opts={}, gpu_id=0)

            nvCvt = vali.PySurfaceConverter(
                vali.PixelFormat.NV12,
                vali.PixelFormat.RGB,
                gpu_id=0)

            nvDwn = vali.PySurfaceDownloader(gpu_id=0)

            # Use color space and range of original file.
            cc_ctx = vali.ColorspaceConversionContext(
                vali.ColorSpace.BT_709,
                vali.ColorRange.MPEG)

            surf_src = vali.Surface.Make(
                pyDec.Format(),
                pyDec.Width(),
                pyDec.Height(),
                gpu_id=0)

            for i in range(0, gtInfo.num_frames):

                success, _ = pyDec.DecodeSingleSurface(surf_src)
                if not success:
                    self.fail("Failed to decode surface")

                surf_dst = vali.Surface.Make(
                    vali.PixelFormat.RGB,
                    surf_src.Width(),
                    surf_src.Height(),
                    gpu_id=0)

                success, details = nvCvt.Run(surf_src, surf_dst, cc_ctx)
                if not success:
                    self.fail("Failed to convert surface: " + details)

                src_tensor = torch.from_dlpack(surf_dst.PlanePtr())

                # Check dimensions
                self.assertEqual(len(src_tensor.shape), 2)
                self.assertEqual(src_tensor.shape[0] * src_tensor.shape[1],
                                 surf_dst.HostSize())

                # Check if sizes are equal
                rgb_frame = src_tensor.cpu().numpy().flatten()
                self.assertEqual(rgb_frame.size, surf_dst.HostSize())

                # Check if memory is bit 2 bit equal
                frame_dst = np.ndarray(
                    shape=(surf_dst.HostSize()),
                    dtype=np.uint8)

                success, details = nvDwn.Run(surf_dst, frame_dst)
                if not success:
                    self.fail("Failed to download decoded surface: " + details)
                self.assertTrue(np.array_equal(rgb_frame, frame_dst))

    def test_tensor_from_surface(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

            pyDec = vali.PyDecoder(
                input=gtInfo.uri,
                opts={},
                gpu_id=0)

            nvCvt = vali.PySurfaceConverter(
                vali.PixelFormat.NV12,
                vali.PixelFormat.RGB,
                gpu_id=0)

            nvDwn = vali.PySurfaceDownloader(gpu_id=0)

            # Use color space and range of original file.
            cc_ctx = vali.ColorspaceConversionContext(
                vali.ColorSpace.BT_709,
                vali.ColorRange.MPEG)

            surf_dec = vali.Surface.Make(
                pyDec.Format(),
                pyDec.Width(),
                pyDec.Height(),
                gpu_id=0)

            for i in range(0, gtInfo.num_frames):
                success, details = pyDec.DecodeSingleSurface(surf_dec)
                if not success:
                    self.fail("Fail to decode surface: " + details)

                surf_cvt = vali.Surface.Make(
                    vali.PixelFormat.RGB,
                    surf_dec.Width(),
                    surf_dec.Height(),
                    gpu_id=0)

                success, details = nvCvt.Run(surf_dec, surf_cvt, cc_ctx)
                if not success:
                    self.fail("Failed to convert surface: " + details)

                ten_rgb = torch.from_dlpack(surf_cvt)

                # Check dimensions
                self.assertEqual(len(ten_rgb.shape), 3)
                self.assertEqual(ten_rgb.shape[0], surf_cvt.Height())
                self.assertEqual(ten_rgb.shape[1], surf_cvt.Width())
                self.assertEqual(ten_rgb.shape[2], 3)

                # Check size in bytes
                frame_ten = ten_rgb.cpu().numpy().flatten()
                self.assertEqual(frame_ten.size, surf_cvt.HostSize())

                # Bit 2 bit memory cmp
                frame_surf = np.ndarray(
                    shape=(surf_cvt.HostSize()),
                    dtype=np.uint8)

                success, details = nvDwn.Run(surf_cvt, frame_surf)
                if not success:
                    self.fail("Failed to download decoded surface: " + details)

                if not np.array_equal(frame_ten, frame_surf):
                    self.log.error(
                        "PSNR: " + str(tc.measurePSNR(frame_ten, frame_surf)))

                    tc.dumpFrameToDisk(frame_ten, "from_tensor", surf_cvt.Width(),
                                       surf_cvt.Height(), ".rgb")
                    tc.dumpFrameToDisk(frame_surf, "from_surface", surf_cvt.Width(),
                                       surf_cvt.Height(), ".rgb")
                    self.fail("Mismatch at frame " + str(i))

    def test_surface_from_tensor(self):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            rgbInfo = tc.GroundTruth(**gt_values["basic_rgb"])

        with open(rgbInfo.uri, "rb") as f_in:
            frame_size = rgbInfo.width * rgbInfo.height * 3
            rgb_frame = np.fromfile(f_in, np.uint8, frame_size)

        tensor = torch.from_numpy(rgb_frame).to(device="cuda")
        tensor = torch.reshape(tensor, (rgbInfo.height, rgbInfo.width * 3))

        surface = vali.Surface.from_dlpack(
            torch.utils.dlpack.to_dlpack(tensor))
        if not surface or surface.Empty():
            self.fail("Failed to import Surface from dlpack")

        nvDwn = vali.PySurfaceDownloader(gpu_id=0)

        # Check dimensions
        self.assertEqual(len(tensor.shape), 2)
        self.assertEqual(tensor.shape[0] * tensor.shape[1], surface.HostSize())

        # Check if memory is bit 2 bit equal
        frame = np.ndarray(shape=(surface.HostSize()), dtype=np.uint8)
        if not nvDwn.Run(surface, frame):
            self.fail("Failed to download decoded surface")

        array = tensor.cpu().numpy().flatten()
        self.assertTrue(np.array_equal(array, frame))


if __name__ == "__main__":
    unittest.main()
