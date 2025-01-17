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
import torchvision
import logging
from PIL import Image
from nvidia import nvimgcodec
from io import BytesIO

# We use 42 (dB) as the measure of similarity.
# If two images have PSNR higher than 42 (dB) we consider them the same.
psnr_threshold = 42.0


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
                pyDec.Format,
                pyDec.Width,
                pyDec.Height,
                gpu_id=0)

            for i in range(0, gtInfo.num_frames):

                success, _ = pyDec.DecodeSingleSurface(surf_src)
                if not success:
                    self.fail("Failed to decode surface")

                surf_dst = vali.Surface.Make(
                    vali.PixelFormat.RGB,
                    surf_src.Width,
                    surf_src.Height,
                    gpu_id=0)

                success, details = nvCvt.Run(surf_src, surf_dst, cc_ctx)
                if not success:
                    self.fail("Failed to convert surface: " + details)

                src_tensor = torch.from_dlpack(surf_dst.Planes[0])

                # Check dimensions
                self.assertEqual(len(src_tensor.shape), 2)
                self.assertEqual(src_tensor.shape[0] * src_tensor.shape[1],
                                 surf_dst.HostSize)

                # Check if sizes are equal
                rgb_frame = src_tensor.cpu().numpy().flatten()
                self.assertEqual(rgb_frame.size, surf_dst.HostSize)

                # Check if memory is bit 2 bit equal
                frame_dst = np.ndarray(
                    shape=(surf_dst.HostSize),
                    dtype=np.uint8)

                success, details = nvDwn.Run(surf_dst, frame_dst)
                if not success:
                    self.fail("Failed to download decoded surface: " + details)
                self.assertTrue(np.array_equal(rgb_frame, frame_dst))

    def test_tensor_from_surface(self):
        """
        This test creates cuda float tensor from surface
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

            py_dec = vali.PyDecoder(
                input=gtInfo.uri,
                opts={},
                gpu_id=0)

            convs = [
                vali.PySurfaceConverter(
                    py_dec.Format, vali.PixelFormat.RGB, gpu_id=0),

                vali.PySurfaceConverter(
                    vali.PixelFormat.RGB, vali.PixelFormat.RGB_32F, gpu_id=0),

                vali.PySurfaceConverter(
                    vali.PixelFormat.RGB_32F, vali.PixelFormat.RGB_32F_PLANAR, gpu_id=0)
            ]

            surfaces = [
                vali.Surface.Make(
                    format=py_dec.Format, width=py_dec.Width, height=py_dec.Height, gpu_id=0),

                vali.Surface.Make(
                    format=vali.PixelFormat.RGB, width=py_dec.Width, height=py_dec.Height, gpu_id=0),

                vali.Surface.Make(
                    format=vali.PixelFormat.RGB_32F, width=py_dec.Width, height=py_dec.Height, gpu_id=0),

                vali.Surface.Make(format=vali.PixelFormat.RGB_32F_PLANAR,
                                  width=py_dec.Width, height=py_dec.Height, gpu_id=0)
            ]

            for i in range(0, gtInfo.num_frames):
                success, details = py_dec.DecodeSingleSurface(surfaces[0])
                if not success:
                    self.fail(f"Fail to decode surface: {details}")

                for i in range(0, len(convs)):
                    success, details = convs[i].Run(
                        src=surfaces[i], dst=surfaces[i+1])
                    if not success:
                        self.fail(f"Fail to convert surface: {details}")

                    img_tensor = torch.from_dlpack(surfaces[3])
                    img_tensor = img_tensor.clone().detach()

                    # Normalize tensor to meet the NN expectations.
                    data_transforms = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                    data_transforms(img_tensor)

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
        if not surface or surface.IsEmpty:
            self.fail("Failed to import Surface from dlpack")

        nvDwn = vali.PySurfaceDownloader(gpu_id=0)

        # Check dimensions
        self.assertEqual(len(tensor.shape), 2)
        self.assertEqual(tensor.shape[0] * tensor.shape[1], surface.HostSize)

        # Check if memory is bit 2 bit equal
        frame = np.ndarray(shape=(surface.HostSize), dtype=np.uint8)
        if not nvDwn.Run(surface, frame):
            self.fail("Failed to download decoded surface")

        array = tensor.cpu().numpy().flatten()
        self.assertTrue(np.array_equal(array, frame))

    def test_cai_export(self):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            rgbInfo = tc.GroundTruth(**gt_values["basic_rgb"])

        with open(rgbInfo.uri, "rb") as f_in:
            frame_size = rgbInfo.width * rgbInfo.height * 3
            rgb_frame = np.fromfile(f_in, np.uint8, frame_size)

        pyUpl = vali.PyFrameUploader(gpu_id=0)

        surf_rgb = vali.Surface.Make(
            vali.PixelFormat.RGB,
            rgbInfo.width,
            rgbInfo.height,
            gpu_id=0)

        nvJpg = vali.PyNvJpegEncoder(gpu_id=0)

        nvJpgCtx = nvJpg.Context(
            compression=100,
            pixel_format=vali.PixelFormat.RGB)

        success, info = pyUpl.Run(rgb_frame, surf_rgb)
        if not success:
            self.fail("Failed to upload RGB frame: " + str(info))

        # Share memory with nvimmgcodec and compress to jpeg
        encoder = nvimgcodec.Encoder()
        encoder.write("frame_nvcv.jpg", nvimgcodec.as_image(surf_rgb))

        # Do the same thing with PyNvJpegEncoder
        buffers, info = nvJpg.Run(nvJpgCtx, [surf_rgb])
        if len(buffers) != 1:
            self.fail("Failed to encode jpeg: " + str(info))

        with open("frame_vali.jpg", "wb") as fout:
            fout.write(buffers[0])

        # Measure PSNR scores between them to make sure RGB Surface was
        # exported to nvcv image correctly via __cuda_array_interface__.
        psnr_score = tc.measurePSNR(
            np.asarray(Image.open("frame_nvcv.jpg")),
            np.asarray(Image.open("frame_vali.jpg")))
        self.assertGreaterEqual(psnr_score, psnr_threshold)

        if os.path.exists("frame_nvcv.jpg"):
            os.remove("frame_nvcv.jpg")

        if os.path.exists("frame_vali.jpg"):
            os.remove("frame_vali.jpg")

    def test_cai_import(self):
        decoder = nvimgcodec.Decoder()

        with open("data/frame_0.jpg", 'rb') as in_file:
            data = in_file.read()

        nvcv_img = decoder.decode(data)
        vali_img = vali.Surface.from_cai(nvcv_img)

        nvJpg = vali.PyNvJpegEncoder(gpu_id=0)

        nvJpgCtx = nvJpg.Context(
            compression=100,
            pixel_format=vali.PixelFormat.RGB)

        buffers, info = nvJpg.Run(nvJpgCtx, [vali_img])
        if len(buffers) != 1:
            self.fail("Failed to encode jpeg: " + str(info))

        with open("frame_vali.jpg", "wb") as fout:
            fout.write(buffers[0])

        psnr_score = tc.measurePSNR(
            np.asarray(Image.open("data/frame_0.jpg")),
            np.asarray(Image.open(BytesIO(np.ndarray.tobytes(buffers[0])))))
        self.assertGreaterEqual(psnr_score, psnr_threshold)


if __name__ == "__main__":
    unittest.main()
