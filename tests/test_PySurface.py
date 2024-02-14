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

    def test_pycuda_memcpy_Surface_Surface(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        gpu_id = 0
        enc_file = gtInfo.uri
        cuda_ctx = cuda.Device(gpu_id).retain_primary_context()
        cuda_ctx.push()
        cuda_str = cuda.Stream()
        cuda_ctx.pop()

        nvDec = nvc.PyNvDecoder(enc_file, cuda_ctx.handle, cuda_str.handle)
        nvDwn = nvc.PySurfaceDownloader(
            nvDec.Width(),
            nvDec.Height(),
            nvDec.Format(),
            cuda_ctx.handle,
            cuda_str.handle,
        )

        while True:
            surf_src, _ = nvDec.DecodeSingleSurface()
            if surf_src.Empty():
                break
            src_plane = surf_src.PlanePtr()

            surf_dst = nvc.Surface.Make(
                nvDec.Format(),
                nvDec.Width(),
                nvDec.Height(),
                gpu_id,
            )
            self.assertFalse(surf_dst.Empty())
            dst_plane = surf_dst.PlanePtr()

            memcpy_2d = cuda.Memcpy2D()
            memcpy_2d.width_in_bytes = src_plane.Width() * src_plane.ElemSize()
            memcpy_2d.src_pitch = src_plane.Pitch()
            memcpy_2d.dst_pitch = dst_plane.Pitch()
            memcpy_2d.width = src_plane.Width()
            memcpy_2d.height = src_plane.Height()
            memcpy_2d.set_src_device(src_plane.GpuMem())
            memcpy_2d.set_dst_device(dst_plane.GpuMem())
            memcpy_2d(cuda_str)

            frame_src = np.ndarray(shape=(0), dtype=np.uint8)
            if not nvDwn.DownloadSingleSurface(surf_src, frame_src):
                self.fail("Failed to download decoded surface")

            frame_dst = np.ndarray(shape=(0), dtype=np.uint8)
            if not nvDwn.DownloadSingleSurface(surf_dst, frame_dst):
                self.fail("Failed to download decoded surface")

            if not np.array_equal(frame_src, frame_dst):
                self.fail("Video frames are not equal")

    def test_pycuda_memcpy_Surface_Tensor(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        gpu_id = 0
        enc_file = gtInfo.uri
        cuda_ctx = cuda.Device(gpu_id).retain_primary_context()
        cuda_ctx.push()
        cuda_str = cuda.Stream()
        cuda_ctx.pop()

        nvDec = nvc.PyNvDecoder(enc_file, cuda_ctx.handle, cuda_str.handle)
        nvDwn = nvc.PySurfaceDownloader(
            nvDec.Width(),
            nvDec.Height(),
            nvDec.Format(),
            cuda_ctx.handle,
            cuda_str.handle,
        )

        while True:
            surf_src, _ = nvDec.DecodeSingleSurface()
            if surf_src.Empty():
                break
            src_plane = surf_src.PlanePtr()

            surface_tensor = torch.rand(
                src_plane.Height(),
                src_plane.Width(),
                1,
                dtype=torch.uint8,
                device=torch.device(f"cuda:{gpu_id}"),
            )
            dst_plane = surface_tensor.data_ptr()

            memcpy_2d = cuda.Memcpy2D()
            memcpy_2d.width_in_bytes = src_plane.Width() * src_plane.ElemSize()
            memcpy_2d.src_pitch = src_plane.Pitch()
            memcpy_2d.dst_pitch = nvDec.Width()
            memcpy_2d.width = src_plane.Width()
            memcpy_2d.height = src_plane.Height()
            memcpy_2d.set_src_device(src_plane.GpuMem())
            memcpy_2d.set_dst_device(dst_plane)
            memcpy_2d(cuda_str)

            frame_src = np.ndarray(shape=(0), dtype=np.uint8)
            if not nvDwn.DownloadSingleSurface(surf_src, frame_src):
                self.fail("Failed to download decoded surface")

            frame_dst = surface_tensor.to("cpu").numpy()
            frame_dst = frame_dst.reshape(
                (src_plane.Height() * src_plane.Width()))

            if not np.array_equal(frame_src, frame_dst):
                self.fail("Video frames are not equal")

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
        nvDwn = nvc.PySurfaceDownloader(nvDec.Width(), nvDec.Height(),
                                        nvDec.Format(), gpu_id=0)

        for surf in dec_surfaces:
            dec_frame = np.ndarray(shape=(0), dtype=np.uint8)
            svd_frame = np.ndarray(shape=(0), dtype=np.uint8)

            nvDwn.DownloadSingleSurface(surf, svd_frame)
            nvDec.DecodeSingleFrame(dec_frame)

            if not np.array_equal(dec_frame, svd_frame):
                tc.dumpFrameToDisk(dec_frame, "dec_frame", nvDec.Width(),
                                   nvDec.Height(), "nv12.yuv")
                tc.dumpFrameToDisk(svd_frame, "svd_frame", nvDec.Width(),
                                   nvDec.Height(), "nv12.yuv")
                self.fail("Frames are not same")

                
    def test_context_manager(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)
        surf, _ = nvDec.DecodeSingleSurface()

        with nvc.SurfaceView(surf) as surfView:
            self.assertIsNotNone(surfView)
            self.assertEqual(surfView.Width(), surf.Width())

if __name__ == "__main__":
    unittest.main()
