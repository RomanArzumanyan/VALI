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
from test_common import GroundTruth


class TestDecoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

        with open("gt_files.json") as f:
            data = json.load(f)["basic"]
        self.gtInfo = GroundTruth(**data)

    def test_width(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.width, ffDec.Width())

    def test_height(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.height, ffDec.Height())

    def test_color_space(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.color_space, str(ffDec.ColorSpace()))

    def test_color_range(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.color_range, str(ffDec.ColorRange()))

    def test_format(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        # The only difference between NV12 and YUV420 is chroma sampling
        # So we consider them the same.
        format = ffDec.Format()
        if ffDec.Format() == nvc.PixelFormat.YUV420:
            format = nvc.PixelFormat.NV12
        self.assertEqual(self.gtInfo.pix_fmt, str(format))

    def test_framerate(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.framerate, ffDec.Framerate())

    def test_avgframerate(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        self.assertEqual(self.gtInfo.framerate, ffDec.AvgFramerate())

    def test_timebase(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        epsilon = 1e-4
        self.assertLessEqual(
            np.abs(self.gtInfo.timebase - ffDec.Timebase()), epsilon)

    def test_decode_all_frames(self):
        ffDec = nvc.PyDecoder(input=self.gtInfo.uri, opts={}, gpu_id=-1)
        dec_frames = 0
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = ffDec.DecodeSingleFrame(frame)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.gtInfo.num_frames, dec_frames)
        self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)

    def test_decode_all_surfaces(self):
        gpu_id = 0
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {}, gpu_id)
        dec_frames = 0
        surf = nvc.Surface.Make(
            ffDec.Format(), ffDec.Width(), ffDec.Height(), gpu_id)
        while True:
            success, details = ffDec.DecodeSingleSurface(surf)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.gtInfo.num_frames, dec_frames)
        self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)

    def test_check_decode_status(self):
        ffDec = nvc.PyDecoder(self.gtInfo.uri, {})
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = ffDec.DecodeSingleFrame(frame)
            if not success:
                self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)
                break
            self.assertEqual(details, nvc.TaskExecInfo.SUCCESS)

    def test_decode_single_frame_out_pkt_data(self):
        with open("gt_files.json") as f:
            gtInfo = GroundTruth(**json.load(f)["basic"])

        ffDec = nvc.PyDecoder(gtInfo.uri, {})
        frame = np.ndarray(dtype=np.uint8, shape=())

        dec_frame = 0
        last_pts = nvc.NO_PTS
        while True:
            pdata = nvc.PacketData()
            success, _ = ffDec.DecodeSingleFrame(frame, pdata)
            if not success:
                break
            self.assertNotEqual(pdata.pts, nvc.NO_PTS)
            if 0 != dec_frame:
                self.assertGreaterEqual(pdata.pts, last_pts)
            dec_frame += 1
            last_pts = pdata.pts

    @unittest.skip("Disable test: very noisy output")
    def test_log_warnings(self):
        with open("gt_files.json") as f:
            gtInfo = GroundTruth(**json.load(f)["log_warnings_ffdec"])
            ffDec = nvc.PyDecoder(gtInfo.uri, {})

            self.assertEqual(ffDec.Width(), gtInfo.width)
            self.assertEqual(ffDec.Height(), gtInfo.height)
            self.assertEqual(str(ffDec.Format()), gtInfo.pix_fmt)
            self.assertEqual(ffDec.Framerate(), gtInfo.framerate)

            frame = np.ndarray(shape=(0), dtype=np.uint8)
            dec_frames = 0

            while True:
                newFrame, execInfo = ffDec.DecodeSingleFrame(frame)
                if not newFrame:
                    break
                else:
                    dec_frames += 1

            self.assertEqual(execInfo, nvc.TaskExecInfo.END_OF_STREAM)
            self.assertEqual(gtInfo.num_frames, dec_frames)

    def test_get_motion_vectors(self):
        with open("gt_files.json") as f:
            gtInfo = GroundTruth(**json.load(f)["basic"])
            ffDec = nvc.PyDecoder(gtInfo.uri, {"flags2": "+export_mvs"})

        frame = np.ndarray(shape=(0), dtype=np.uint8)

        success, _ = ffDec.DecodeSingleFrame(frame)
        self.assertTrue(success)

        # First frame shall be I frame, hence no motion vectors.
        mv = ffDec.GetMotionVectors()
        self.assertEqual(len(mv), 0)

        success, _ = ffDec.DecodeSingleFrame(frame)
        self.assertTrue(success)

        # Second frame shall be either P or B, hence motion vectors
        # shall be there.
        mv = ffDec.GetMotionVectors()
        self.assertGreater(len(mv), 0)

        # Very basic sanity check:
        # Motion scale means precision, can't be 0.
        # Usually it's 2 or 4 (half- or quater- pixel precision).
        # Source is either -1 (prediction from past) or 1 (from future).
        first_mv = mv[0]
        self.assertNotEqual(first_mv.source, 0)
        self.assertNotEqual(first_mv.motion_scale, 0)

    @unittest.skip("Need to setup RTSP server. Behaves differently on win / linux")
    def test_rtsp_nonexisting(self):
        timeout_ms = 1000
        tp = time.time()

        with self.assertRaises(RuntimeError):
            ffDec = nvc.PyDecoder(
                input="rtsp://127.0.0.1/nothing",
                opts={"timeout": str(timeout_ms)})

        tp = (time.time() - tp) * 1000
        self.assertGreaterEqual(tp, timeout_ms)


if __name__ == "__main__":
    unittest.main()
