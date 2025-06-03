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

import python_vali as vali
import numpy as np
import unittest
import json
import test_common as tc
import torch
import logging
from PIL import Image
from nvidia import nvimgcodec
from io import BytesIO
import os

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

            nvCvt = vali.PySurfaceConverter(gpu_id=0)

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
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

            pyDec = vali.PyDecoder(
                input=gtInfo.uri,
                opts={},
                gpu_id=0)

            nvCvt = vali.PySurfaceConverter(gpu_id=0)

            nvDwn = vali.PySurfaceDownloader(gpu_id=0)

            # Use color space and range of original file.
            cc_ctx = vali.ColorspaceConversionContext(
                vali.ColorSpace.BT_709,
                vali.ColorRange.MPEG)

            surf_dec = vali.Surface.Make(
                pyDec.Format,
                pyDec.Width,
                pyDec.Height,
                gpu_id=0)

            for i in range(0, gtInfo.num_frames):
                success, details = pyDec.DecodeSingleSurface(surf_dec)
                if not success:
                    self.fail("Fail to decode surface: " + details)

                surf_cvt = vali.Surface.Make(
                    vali.PixelFormat.RGB,
                    surf_dec.Width,
                    surf_dec.Height,
                    gpu_id=0)

                success, details = nvCvt.Run(surf_dec, surf_cvt, cc_ctx)
                if not success:
                    self.fail("Failed to convert surface: " + details)

                ten_rgb = torch.from_dlpack(surf_cvt)

                # Check dimensions
                self.assertEqual(len(ten_rgb.shape), 3)
                self.assertEqual(ten_rgb.shape[0], surf_cvt.Height)
                self.assertEqual(ten_rgb.shape[1], surf_cvt.Width)
                self.assertEqual(ten_rgb.shape[2], 3)

                # Check size in bytes
                frame_ten = ten_rgb.cpu().numpy().flatten()
                self.assertEqual(frame_ten.size, surf_cvt.HostSize)

                # Bit 2 bit memory cmp
                frame_surf = np.ndarray(
                    shape=(surf_cvt.HostSize),
                    dtype=np.uint8)

                success, details = nvDwn.Run(surf_cvt, frame_surf)
                if not success:
                    self.fail("Failed to download decoded surface: " + details)

                if not np.array_equal(frame_ten, frame_surf):
                    self.log.error(
                        "PSNR: " + str(tc.measure_psnr(frame_ten, frame_surf)))

                    tc.dump_to_disk(frame_ten, "from_tensor", surf_cvt.Width,
                                       surf_cvt.Height, ".rgb")
                    tc.dump_to_disk(frame_surf, "from_surface", surf_cvt.Width,
                                       surf_cvt.Height, ".rgb")
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
        pyRes = vali.PySurfaceResizer(vali.PixelFormat.RGB, gpu_id=0)

        # At some point nvimgcodec lost support of pitched allocations
        # So we convert image to bigger one, which width will be
        # same as pitch. Sic !
        surf_rgb = vali.Surface.Make(
            vali.PixelFormat.RGB,
            rgbInfo.width,
            rgbInfo.height,
            gpu_id=0)
        
        surf_big = vali.Surface.Make(
            vali.PixelFormat.RGB,
            1024,
            560,
            gpu_id=0)
        
        nvJpg = vali.PyNvJpegEncoder(gpu_id=0)

        nvJpgCtx = nvJpg.Context(
            compression=100,
            pixel_format=vali.PixelFormat.RGB)

        success, info = pyUpl.Run(rgb_frame, surf_rgb)
        if not success:
            self.fail("Failed to upload RGB frame: " + str(info))

        success, info = pyRes.Run(surf_rgb, surf_big)
        if not success:
            self.fail(str(info))

        # Share memory with nvimmgcodec and compress to jpeg
        encoder = nvimgcodec.Encoder()
        img = nvimgcodec.as_image(surf_big)
        encoder.write("frame_nvcv.jpg", img)

        # Do the same thing with PyNvJpegEncoder
        buffers, info = nvJpg.Run(nvJpgCtx, [surf_big])
        if len(buffers) != 1:
            self.fail("Failed to encode jpeg: " + str(info))

        with open("frame_vali.jpg", "wb") as fout:
            fout.write(buffers[0])

        # Measure PSNR scores between them to make sure RGB Surface was
        # exported to nvcv image correctly via __cuda_array_interface__.
        psnr_score = tc.measure_psnr(
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

        psnr_score = tc.measure_psnr(
            np.asarray(Image.open("data/frame_0.jpg")),
            np.asarray(Image.open(BytesIO(np.ndarray.tobytes(buffers[0])))))
        self.assertGreaterEqual(psnr_score, psnr_threshold)

    def test_surface_make_all_formats(self):
        """Test Surface.Make with all supported pixel formats.
        
        This test verifies that Surface.Make can create surfaces for all
        supported pixel formats with valid dimensions.
        """
        # Test dimensions
        width = 1920
        height = 1080
        gpu_id = 0  # Use first GPU device
        
        # List of all supported pixel formats
        formats = [
            vali.PixelFormat.Y,
            vali.PixelFormat.RGB,
            vali.PixelFormat.NV12,
            vali.PixelFormat.YUV420,
            vali.PixelFormat.RGB_PLANAR,
            vali.PixelFormat.BGR,
            vali.PixelFormat.YUV444,
            vali.PixelFormat.YUV444_10bit,
            vali.PixelFormat.YUV420_10bit,
            vali.PixelFormat.RGB_32F,
            vali.PixelFormat.RGB_32F_PLANAR,
            vali.PixelFormat.YUV422,
            vali.PixelFormat.P10,
            vali.PixelFormat.P12
        ]
        
        for fmt in formats:
            # Create surface with current format
            surf = vali.Surface.Make(fmt, width, height, gpu_id)
            
            # Verify surface was created successfully
            self.assertIsNotNone(surf)
            
            # Verify surface properties
            self.assertEqual(surf.Width, width)
            self.assertEqual(surf.Height, height)
            self.assertEqual(surf.Format, fmt)
            
            # Verify surface has valid memory allocation
            self.assertGreater(surf.HostSize, 0)
            
            # Verify surface has correct number of planes based on format
            if fmt in [vali.PixelFormat.Y, vali.PixelFormat.RGB, vali.PixelFormat.BGR, 
                      vali.PixelFormat.RGB_32F, vali.PixelFormat.NV12, 
                      vali.PixelFormat.P10, vali.PixelFormat.P12, vali.PixelFormat.RGB_PLANAR,
                        vali.PixelFormat.RGB_32F_PLANAR]:
                self.assertEqual(surf.NumPlanes, 1)
            elif fmt in [vali.PixelFormat.YUV420, vali.PixelFormat.YUV420_10bit,
                        vali.PixelFormat.YUV422, vali.PixelFormat.YUV444, 
                        vali.PixelFormat.YUV444_10bit]:
                self.assertEqual(surf.NumPlanes, 3)


if __name__ == "__main__":
    unittest.main()
