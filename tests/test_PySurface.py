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

try:
    import pycuda.driver as cuda
    import torch
except ImportError as e:
    raise unittest.SkipTest(
        f"Skipping because of insufficient dependencies: {e}")


class TestSurfacePycuda(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        cuda.init()

    def test_tensor_from_surface_plane(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

            nvDec = nvc.PyNvDecoder(
                input=gtInfo.uri,
                gpu_id=0)

            nvCvt = nvc.PySurfaceConverter(
                nvc.PixelFormat.NV12,
                nvc.PixelFormat.RGB,
                gpu_id=0)
            
            nvDwn = nvc.PySurfaceDownloader(gpu_id=0)

            # Use color space and range of original file.
            cc_ctx = nvc.ColorspaceConversionContext(
                nvc.ColorSpace.BT_709,
                nvc.ColorRange.MPEG)

            for i in range(0, gtInfo.num_frames):
                surf_src, _ = nvDec.DecodeSingleSurface()
                if surf_src.Empty():
                    self.fail("Failed to decode surface")

                surf_dst = nvc.Surface.Make(
                    nvc.PixelFormat.RGB, 
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
                frame_dst = np.ndarray(shape=(surf_dst.HostSize()), dtype=np.uint8)
                if not nvDwn.Run(surf_dst, frame_dst):
                    self.fail("Failed to download decoded surface")
                self.assertTrue(np.array_equal(rgb_frame, frame_dst))

    def test_tensor_from_surface(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

            nvDec = nvc.PyNvDecoder(
                input=gtInfo.uri,
                gpu_id=0)

            nvCvt = nvc.PySurfaceConverter(
                nvc.PixelFormat.NV12,
                nvc.PixelFormat.RGB,
                gpu_id=0)
            
            nvDwn = nvc.PySurfaceDownloader(gpu_id=0)

            # Use color space and range of original file.
            cc_ctx = nvc.ColorspaceConversionContext(
                nvc.ColorSpace.BT_709,
                nvc.ColorRange.MPEG)

            for i in range(0, gtInfo.num_frames):
                surf_src, _ = nvDec.DecodeSingleSurface()
                if surf_src.Empty():
                    self.fail("Fail to decode surface")

                surf_dst = nvc.Surface.Make(
                    nvc.PixelFormat.RGB, 
                    surf_src.Width(), 
                    surf_src.Height(), 
                    gpu_id=0)
                
                success, details = nvCvt.Run(surf_src, surf_dst, cc_ctx)
                if not success:
                    self.fail("Failed to convert surface: " + details)

                src_tensor = torch.from_dlpack(surf_dst)
                
                # Check dimensions
                self.assertEqual(len(src_tensor.shape), 3)
                self.assertEqual(src_tensor.shape[0], surf_dst.Height())
                self.assertEqual(src_tensor.shape[1], surf_dst.Width())
                self.assertEqual(src_tensor.shape[2], 3)
                
                # Check if sizes are equal
                rgb_frame = src_tensor.cpu().numpy().flatten()
                self.assertEqual(rgb_frame.size, surf_dst.HostSize())
                
                # Check if memory is bit 2 bit equal
                frame_dst = np.ndarray(shape=(surf_dst.HostSize()), dtype=np.uint8)
                if not nvDwn.Run(surf_dst, frame_dst):
                    self.fail("Failed to download decoded surface")
                self.assertTrue(np.array_equal(rgb_frame, frame_dst))                

    def test_surface_from_tensor(self):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            rgbInfo = tc.GroundTruth(**gt_values["basic_rgb"])

        with open(rgbInfo.uri, "rb") as f_in:
            frame_size = rgbInfo.width * rgbInfo.height * 3
            rgb_frame = np.fromfile(f_in, np.uint8, frame_size)

        tensor = torch.from_numpy(rgb_frame).to(device="cuda")
        tensor = torch.reshape(tensor, (rgbInfo.height, rgbInfo.width * 3))

        surface = nvc.Surface.from_dlpack(
            torch.utils.dlpack.to_dlpack(tensor))
        if not surface or surface.Empty():
            self.fail("Failed to import Surface from dlpack")

        nvDwn = nvc.PySurfaceDownloader(gpu_id=0)
        
        # Check dimensions
        self.assertEqual(len(tensor.shape), 2)
        self.assertEqual(tensor.shape[0] * tensor.shape[1], surface.HostSize())

        # Check if memory is bit 2 bit equal
        frame = np.ndarray(shape=(surface.HostSize()), dtype=np.uint8)
        if not nvDwn.Run(surface, frame):
            self.fail("Failed to download decoded surface")

        array = tensor.cpu().numpy().flatten()
        self.assertTrue(np.array_equal(array, frame))

    @unittest.skip("Known issue: unstable test")
    def test_list_append(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)

        # Decode all the surfaces and store them in the list.
        dec_surfaces = []
        while True:
            surf, _ = nvDec.DecodeSingleSurface()
            if not surf or surf.Empty():
                break
            else:
                dec_surfaces.append(surf)

        # Now compare saved surfaces against data from decoder
        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)
        nvDwn = nvc.PySurfaceDownloader(gpu_id=0)

        for surf in dec_surfaces:
            dec_frame = np.ndarray(shape=(surf.HostSize()), dtype=np.uint8)
            svd_frame = np.ndarray(shape=(surf.HostSize()), dtype=np.uint8)

            nvDwn.Run(surf, svd_frame)
            nvDec.DecodeSingleFrame(dec_frame)

            if not np.array_equal(dec_frame, svd_frame):
                tc.dumpFrameToDisk(dec_frame, "dec_frame", nvDec.Width(),
                                   nvDec.Height(), "nv12.yuv")
                tc.dumpFrameToDisk(svd_frame, "svd_frame", nvDec.Width(),
                                   nvDec.Height(), "nv12.yuv")
                self.fail("Frames are not same")


if __name__ == "__main__":
    unittest.main()
