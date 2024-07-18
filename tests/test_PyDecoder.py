#
# Copyright 2023 Vision Labs LLC
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
import time
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
import logging
import random
from parameterized import parameterized


class TestDecoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

        with open("gt_files.json") as f:
            self.data = json.load(f)
        self.gtInfo = tc.GroundTruth(**self.data["basic"])
        self.yuvInfo = tc.GroundTruth(**self.data["basic_yuv420"])
        self.nv12Info = tc.GroundTruth(**self.data["basic_nv12"])
        self.hbdInfo = tc.GroundTruth(**self.data["hevc10"])

        self.log = logging.getLogger(__name__)

    def test_width(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.width, pyDec.Width())

    def test_height(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.height, pyDec.Height())

    def test_color_space(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.color_space, str(pyDec.ColorSpace()))

    def test_color_range(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.color_range, str(pyDec.ColorRange()))

    def test_format(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        # The only difference between NV12 and YUV420 is chroma sampling
        # So we consider them the same.
        format = pyDec.Format()
        if pyDec.Format() == nvc.PixelFormat.YUV420:
            format = nvc.PixelFormat.NV12
        self.assertEqual(self.gtInfo.pix_fmt, str(format))

    def test_framerate(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.framerate, pyDec.Framerate())

    def test_avgframerate(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.framerate, pyDec.AvgFramerate())

    def test_timebase(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {})
        epsilon = 1e-4
        self.assertLessEqual(
            np.abs(self.gtInfo.timebase - pyDec.Timebase()), epsilon)

    def test_decode_all_frames(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {}, gpu_id=-1)
        dec_frames = 0
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = pyDec.DecodeSingleFrame(frame)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.gtInfo.num_frames, dec_frames)
        self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)

    def test_decode_all_surfaces(self):
        gpu_id = 0
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {}, gpu_id)
        dec_frames = 0
        surf = nvc.Surface.Make(
            pyDec.Format(), pyDec.Width(), pyDec.Height(), gpu_id)
        while True:
            success, details = pyDec.DecodeSingleSurface(surf)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.gtInfo.num_frames, dec_frames)
        self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)

    @unittest.skip("HBD is known to be broken on GPU: #58")
    def test_decode_high_bit_depth_gpu(self):
        gpu_id = 0
        pyDec = nvc.PyDecoder(self.hbdInfo.uri, {}, gpu_id)
        dec_frames = 0
        surf = nvc.Surface.Make(
            pyDec.Format(), pyDec.Width(), pyDec.Height(), gpu_id)
        while True:
            success, details = pyDec.DecodeSingleSurface(surf)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.hbdInfo.num_frames, dec_frames)
        self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)

    def test_decode_high_bit_depth_cpu(self):
        gpu_id = -1
        pyDec = nvc.PyDecoder(self.hbdInfo.uri, {}, gpu_id)
        dec_frames = 0
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = pyDec.DecodeSingleFrame(frame)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.hbdInfo.num_frames, dec_frames)
        self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)

    def test_check_all_frames(self):
        pyDec = nvc.PyDecoder(input=self.gtInfo.uri, opts={}, gpu_id=-1)

        dec_frames = 0
        with open(self.yuvInfo.uri, "rb") as f_in:
            while True:
                # Decode single frame from file
                frame = np.ndarray(dtype=np.uint8, shape=())
                success, details = pyDec.DecodeSingleFrame(frame)
                if not success:
                    self.log.info('decode: ' + str(details))
                    break

                # Read ethalon and compare
                frame_gt = np.fromfile(
                    file=f_in, dtype=np.uint8, count=frame.size)
                if not frame_gt.size == frame.size:
                    if dec_frames < self.yuvInfo.num_frames:
                        self.log.error(
                            "Failed to read GT video frame " + str(dec_frames))
                    break

                if not np.array_equal(frame_gt, frame):
                    self.log.error("Frames mismatch: ", dec_frames)
                    self.log.error("PSNR: ", tc.measurePSNR(frame_gt, frame))
                    self.fail()

                dec_frames += 1

        self.assertEqual(self.yuvInfo.num_frames, dec_frames)

    @unittest.skip("cuvid fail this test on runner.")
    def test_check_all_surfaces(self):
        pyDec = nvc.PyDecoder(input=self.gtInfo.uri, opts={}, gpu_id=0)
        pyDwn = nvc.PySurfaceDownloader(gpu_id=0)

        dec_frames = 0
        with open(self.nv12Info.uri, "rb") as f_in:
            while True:
                surf = nvc.Surface.Make(
                    pyDec.Format(), pyDec.Width(), pyDec.Height(), gpu_id=0)
                frame = np.ndarray(dtype=np.uint8, shape=surf.HostSize())

                # Decode single surface from file
                success, details = pyDec.DecodeSingleSurface(surf)
                if not success:
                    self.log.info('decode: ' + str(details))
                    break

                # Download decoded surface to RAM
                if not pyDwn.Run(surf, frame):
                    self.log.error("Failed to download decoded surface")
                    break

                # Read ethalon and compare
                frame_gt = np.fromfile(
                    file=f_in, dtype=np.uint8, count=frame.size)
                if not frame_gt.size == frame.size:
                    if dec_frames < self.nv12Info.num_frames:
                        self.log.error(
                            "Failed to read GT video frame " + str(dec_frames))
                    break

                if not np.array_equal(frame_gt, frame):
                    self.log.error("Mismatch at frame " + str(dec_frames))
                    self.log.error(
                        "PSNR: " + str(tc.measurePSNR(frame_gt, frame)))

                    tc.dumpFrameToDisk(frame_gt, "dec", surf.Width(),
                                       surf.Height(), "yuv_gt.yuv")
                    tc.dumpFrameToDisk(frame, "dec", surf.Width(),
                                       surf.Height(), "yuv_dist.yuv")

                    break

                dec_frames += 1

        self.assertEqual(self.nv12Info.num_frames, dec_frames)

    def test_check_decode_status(self):
        pyDec = nvc.PyDecoder(self.gtInfo.uri, {}, gpu_id=-1)
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = pyDec.DecodeSingleFrame(frame)
            if not success:
                self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)
                break
            self.assertEqual(details, nvc.TaskExecInfo.SUCCESS)

    def test_decode_single_frame_out_pkt_data(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id=-1)
        frame = np.ndarray(dtype=np.uint8, shape=())

        dec_frame = 0
        last_pts = nvc.NO_PTS
        while True:
            pdata = nvc.PacketData()
            success, _ = pyDec.DecodeSingleFrame(frame, pdata)
            if not success:
                break
            self.assertNotEqual(pdata.pts, nvc.NO_PTS)
            if 0 != dec_frame:
                self.assertGreaterEqual(pdata.pts, last_pts)
            dec_frame += 1
            last_pts = pdata.pts

    def test_seek_cpu_decoder(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id=-1)
        frame = np.ndarray(dtype=np.uint8, shape=())
        frame_gt = np.ndarray(dtype=np.uint8, shape=())

        # Seek to random frame within input video frames range
        # start_frame = random.randint(0, gtInfo.num_frames - 1)
        start_frame = random.randint(0, gtInfo.num_frames - 1)
        seek_ctx = nvc.SeekContext(seek_frame=start_frame)
        success, _ = pyDec.DecodeSingleFrame(
            frame=frame, seek_ctx=seek_ctx)
        self.assertTrue(success)

        # Now check if it's the same as via continuous decode
        # For that decoder has to be recreated
        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id=-1)
        dec_frames = 0
        while dec_frames <= start_frame:
            success, _ = pyDec.DecodeSingleFrame(frame=frame_gt)
            self.assertTrue(success, "Failed to decode frame")
            dec_frames += 1

        if not np.array_equal(frame, frame_gt):
            # Sometimes there are small differences between two frames.
            # They may be caused by different decoding results due to jumps
            # between frames.
            #
            # If PSNR is higher then 40 dB we still consider frames to be the
            # same.
            psnr_score = tc.measurePSNR(frame_gt, frame)
            self.log.warning("Mismatch at frame " + str(dec_frames))
            self.log.warning("PSNR: " + str(psnr_score))

            if psnr_score < 40:
                tc.dumpFrameToDisk(frame_gt, "dec", pyDec.Width(),
                                   pyDec.Height(), "yuv_cont.yuv")
                tc.dumpFrameToDisk(frame, "dec", pyDec.Width(),
                                   pyDec.Height(), "yuv_seek.yuv")
                self.fail(
                    "Seek frame isnt same as continuous decode frame")

    def test_seek_gpu_decoder(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id=0)
        pyDwn = nvc.PySurfaceDownloader(gpu_id=0)

        surf = nvc.Surface.Make(
            pyDec.Format(), pyDec.Width(), pyDec.Height(), gpu_id=0)
        frame = np.ndarray(dtype=np.uint8, shape=(surf.HostSize()))
        frame_gt = np.ndarray(dtype=np.uint8, shape=(surf.HostSize()))

        # Seek to random frame within input video frames range
        start_frame = random.randint(0, gtInfo.num_frames - 1)
        seek_ctx = nvc.SeekContext(seek_frame=start_frame)
        success, _ = pyDec.DecodeSingleSurface(
            surf=surf, seek_ctx=seek_ctx)
        self.assertTrue(success)
        self.assertTrue(pyDwn.Run(src=surf, dst=frame))

        # Now check if it's the same as via continuous decode
        # For that decoder has to be recreated
        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id=0)
        dec_frames = 0
        while dec_frames <= start_frame:
            success, details = pyDec.DecodeSingleSurface(surf=surf)
            self.assertTrue(success, "Failed to decode frame: " + str(details))
            dec_frames += 1

        success, details = pyDwn.Run(src=surf, dst=frame_gt)
        if not success:
            self.fail("Failed to download surface: " + str(details))

        if not np.array_equal(frame, frame_gt):
            # Sometimes there are small differences between two frames.
            # They may be caused by different decoding results due to jumps
            # between frames.
            #
            # If PSNR is higher then 40 dB we still consider frames to be the
            # same.
            psnr_score = tc.measurePSNR(frame_gt, frame)
            self.log.warning("Mismatch at frame " + str(dec_frames))
            self.log.warning("PSNR: " + str(psnr_score))

            if psnr_score < 40:
                tc.dumpFrameToDisk(frame_gt, "dec", pyDec.Width(),
                                   pyDec.Height(), "yuv_cont.yuv")
                tc.dumpFrameToDisk(frame, "dec", pyDec.Width(),
                                   pyDec.Height(), "yuv_seek.yuv")
                self.fail(
                    "Seek frame isnt same as continuous decode frame")

    def test_get_motion_vectors(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])
            pyDec = nvc.PyDecoder(
                gtInfo.uri, {"flags2": "+export_mvs"}, gpu_id=-1)

        frame = np.ndarray(shape=(0), dtype=np.uint8)

        success, _ = pyDec.DecodeSingleFrame(frame)
        self.assertTrue(success)

        # First frame shall be I frame, hence no motion vectors.
        mv = pyDec.GetMotionVectors()
        self.assertEqual(len(mv), 0)

        success, _ = pyDec.DecodeSingleFrame(frame)
        self.assertTrue(success)

        # Second frame shall be either P or B, hence motion vectors
        # shall be there.
        mv = pyDec.GetMotionVectors()
        self.assertGreater(len(mv), 0)

        # Very basic sanity check:
        # Motion scale means precision, can't be 0.
        # Usually it's 2 or 4 (half- or quater- pixel precision).
        # Source is either -1 (prediction from past) or 1 (from future).
        first_mv = mv[0]
        self.assertNotEqual(first_mv.source, 0)
        self.assertNotEqual(first_mv.motion_scale, 0)

    def test_decode_resolution_change_gpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["res_change"])

        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id=0)

        width = gtInfo.width
        height = gtInfo.height

        dec_frame = 0
        while True:
            surf = nvc.Surface.Make(pyDec.Format(), width, height, gpu_id=0)
            success, info = pyDec.DecodeSingleSurface(surf)

            if not success:
                break

            if info == nvc.TaskExecInfo.RES_CHANGE:
                width = int(width * gtInfo.res_change_factor)
                height = int(height * gtInfo.res_change_factor)

                # Upon resolution change decoder will not return decoded
                # pixels to user. Hence surface dimensions will not be same
                # to that of decoder.
                self.assertNotEqual(surf.Width(), width)
                self.assertNotEqual(surf.Height(), height)
            else:
                dec_frame += 1

            self.assertEqual(pyDec.Width(), width, str(dec_frame))
            self.assertEqual(pyDec.Height(), height, str(dec_frame))

        self.assertEqual(dec_frame, gtInfo.num_frames)

    def test_decode_resolution_change_cpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["res_change"])

        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id=-1)

        width = gtInfo.width
        height = gtInfo.height

        dec_frame = 0
        while True:
            frame = np.ndarray(shape=(0), dtype=np.uint8)
            success, info = pyDec.DecodeSingleFrame(frame)

            if not success:
                break

            if info == nvc.TaskExecInfo.RES_CHANGE:
                width = int(width * gtInfo.res_change_factor)
                height = int(height * gtInfo.res_change_factor)

                # Upon resolution change decoder will not return decoded
                # pixels to user. Hence frame size will not be same
                # to that of decoder.
                self.assertNotEqual(pyDec.HostFrameSize(), frame.size)
            else:
                dec_frame += 1

            self.assertEqual(pyDec.Width(), width, str(dec_frame))
            self.assertEqual(pyDec.Height(), height, str(dec_frame))

        self.assertEqual(dec_frame, gtInfo.num_frames)


if __name__ == "__main__":
    unittest.main()
