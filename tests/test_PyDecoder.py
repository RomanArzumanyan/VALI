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
import logging
import random
from parameterized import parameterized


class TestDecoder(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

        with open("gt_files.json") as f:
            self.data = json.load(f)
        self.gtInfo = tc.GroundTruth(**self.data["basic"])
        self.yuvInfo = tc.GroundTruth(**self.data["basic_yuv420"])
        self.nv12Info = tc.GroundTruth(**self.data["basic_nv12"])
        self.hbdInfo = tc.GroundTruth(**self.data["hevc10"])
        self.p10Info = tc.GroundTruth(**self.data["hevc10_p10"])
        self.ptsInfo = tc.GroundTruth(**self.data["pts_increase_check"])
        self.rotInfo = tc.GroundTruth(**self.data["rotation_90_deg"])
        self.multiresInfo = tc.GroundTruth(**self.data["multires"])

        self.log = logging.getLogger(__name__)

    def gtByName(self, name: str) -> tc.GroundTruth:
        if name == "basic":
            return self.gtInfo
        elif name == "basic_yuv420":
            return self.yuvInfo
        elif name == "basic_nv12":
            return self.nv12Info
        elif name == "hevc10":
            return self.hbdInfo
        elif name == "hevc10_p10":
            return self.p10Info
        elif name == "pts_increase_check":
            return self.ptsInfo
        elif name == "multires":
            return self.multiresInfo
        else:
            return None

    def test_probe(self):
        """
        This test checks PyDecoder probe functionality.
        Input with 2 video tracks is used for that.
        """
        gt = self.gtByName("multires")
        info = vali.PyDecoder.Probe(gt.uri)

        # GT file has 1 audio stream which is ignored by PyDecoder
        self.assertEqual(len(info), gt.num_streams - 1)

        # Check first video stream, must be full resolution
        str_info = info[0]
        self.assertEqual(gt.width, str_info.width)
        self.assertEqual(gt.height, str_info.height)

        # Second video stream, must be 2x smaller
        str_info = info[1]
        self.assertEqual(gt.width * gt.res_change_factor, 1.0 * str_info.width)
        self.assertEqual(gt.height * gt.res_change_factor,
                         1.0 * str_info.height)

    @parameterized.expand(tc.getDevices())
    def test_preferred_width(self, device_name: str, device_id: int):
        """
        This test checks stream selection by preferred width.
        It's done by choosing a video track and decoding a single video frame
        from it.

        Args:
            device_name (str): device name
            device_id (int): gpu ID or -1 if run on CPU
        """

        gt = self.gtByName("multires")
        for info in vali.PyDecoder.Probe(gt.uri):
            py_dec = vali.PyDecoder(
                gt.uri, {"preferred_width": str(info.width)}, gpu_id=device_id)

            self.assertEqual(py_dec.Width, info.width)
            self.assertEqual(py_dec.Height, info.height)

            if py_dec.IsAccelerated:
                surf = vali.Surface.Make(
                    py_dec.Format, py_dec.Width, py_dec.Height, gpu_id=0)
                success, _ = py_dec.DecodeSingleSurface(surf)
                self.assertTrue(success)
            else:
                frame = np.ndarray(
                    dtype=np.uint8, shape=(py_dec.HostFrameSize))
                success, _ = py_dec.DecodeSingleFrame(frame)
                self.assertTrue(success)

    @parameterized.expand([
        ["avc_8bit", "basic",],
        ["hevc_10bit", "hevc10"],
    ])
    def test_width(self, case_name: str, gt_name: str):
        pyDec = vali.PyDecoder(self.gtByName(gt_name).uri, {})
        self.assertEqual(self.gtByName(gt_name).width, pyDec.Width)

    @parameterized.expand([
        ["avc_8bit", "basic",],
        ["hevc_10bit", "hevc10"],
    ])
    def test_height(self, case_name: str, gt_name: str):
        pyDec = vali.PyDecoder(self.gtByName(gt_name).uri, {})
        self.assertEqual(self.gtByName(gt_name).height, pyDec.Height)

    @parameterized.expand([
        ["avc_8bit_cpu", -1, "basic",],
        ["avc_8bit_gpu", 0, "basic",],
        ["hevc_10bit_cpu", -1, "hevc10"],
        ["hevc_10bit_gpu", 0,  "hevc10"],
    ])
    def test_format(self, case_name: str, gpu_id: int, gt_name: str):
        pyDec = vali.PyDecoder(self.gtByName(gt_name).uri, {}, gpu_id)

        # SW decoder returns frames in YUV420, HW decoder does it in NV12.
        # Chroma sampling is the only difference, so treat both as NV12.
        format = pyDec.Format
        if format == vali.PixelFormat.YUV420:
            format = vali.PixelFormat.NV12

        if case_name == 'hevc_10bit_cpu':
            # More format shenanigans
            self.assertEqual(vali.PixelFormat.YUV420_10bit, format)
        else:
            self.assertEqual(self.gtByName(gt_name).pix_fmt, str(format))

    @parameterized.expand(tc.getDevices())
    def test_level(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.level, pyDec.Level)

    @parameterized.expand(tc.getDevices())
    def test_profile(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.profile, pyDec.Profile)

    @parameterized.expand(tc.getDevices())
    def test_delay(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.delay, pyDec.Delay)

    @parameterized.expand(tc.getDevices())
    def test_gop_size(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.gop_size, pyDec.GopSize)

    @parameterized.expand(tc.getDevices())
    def test_bitrate(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.bitrate, pyDec.Bitrate)

    @parameterized.expand(tc.getDevices())
    def test_num_streams(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.num_streams, pyDec.NumStreams)

    @parameterized.expand(tc.getDevices())
    def test_video_stream_idx(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.video_stream_idx, pyDec.StreamIndex)

    @parameterized.expand(tc.getDevices())
    def test_start_time(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.start_time, pyDec.StartTime)

    @parameterized.expand(tc.getDevices())
    def test_metadata(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.rotInfo.uri, {}, gpu_id=device_id)
        expected_metadata = {
            'context': {
                'compatible_brands': 'isomiso2avc1mp41',
                'creation_time': '2024-12-31T21:00:00.000000Z',
                'encoder': 'Lavf60.16.100',
                'major_brand': 'isom',
                'minor_version': '512'
            },
            'video_stream': {
                'creation_time': '2024-12-31T21:00:00.000000Z',
                'handler_name': 'Core Media Video',
                'language': 'und',
                'vendor_id': '[0][0][0][0]'
            }
        }
        self.assertEqual(expected_metadata, pyDec.Metadata)

    @parameterized.expand(tc.getDevices())
    def test_color_space(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.color_space, str(pyDec.ColorSpace))

    @parameterized.expand(tc.getDevices())
    def test_color_range(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.color_range, str(pyDec.ColorRange))

    @parameterized.expand(tc.getDevices())
    def test_framerate(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.framerate, pyDec.Framerate)

    @parameterized.expand(tc.getDevices())
    def test_avgframerate(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        self.assertEqual(self.gtInfo.framerate, pyDec.AvgFramerate)

    @parameterized.expand(tc.getDevices())
    def test_timebase(self, device_name: str, device_id: int):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=device_id)
        epsilon = 1e-4
        self.assertLessEqual(
            np.abs(self.gtInfo.timebase - pyDec.Timebase), epsilon)

    def test_dec_frame_cpu(self):
        """
        This test checks that `DecodeSingleFrame` methods don't 
        work on GPU and return proper result.
        """
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=0)

        pkt_data = vali.PacketData()
        frame = np.ndarray(dtype=np.uint8, shape=())

        res, info = pyDec.DecodeSingleFrame(frame)
        self.assertFalse(res)
        self.assertEqual(info, vali.TaskExecInfo.FAIL)

        res, info = pyDec.DecodeSingleFrame(frame, pkt_data)
        self.assertFalse(res)
        self.assertEqual(info, vali.TaskExecInfo.FAIL)

    def test_dec_surface_cpu(self):
        """
        This test checks that `DecodeSingleSurface` methods don't
        work on CPU and return proper result and event.
        """

        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=-1)

        pkt_data = vali.PacketData()
        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)

        res, info = pyDec.DecodeSingleSurface(surf)
        self.assertFalse(res)
        self.assertEqual(info, vali.TaskExecInfo.FAIL)

        res, info = pyDec.DecodeSingleSurface(surf, pkt_data)
        self.assertFalse(res)
        self.assertEqual(info, vali.TaskExecInfo.FAIL)

        res, info, evt = pyDec.DecodeSingleSurfaceAsync(surf)
        self.assertFalse(res)
        self.assertEqual(info, vali.TaskExecInfo.FAIL)
        self.assertIsNone(evt)

        res, info, evt = pyDec.DecodeSingleSurfaceAsync(surf, pkt_data)
        self.assertFalse(res)
        self.assertEqual(info, vali.TaskExecInfo.FAIL)
        self.assertIsNone(evt)

    @parameterized.expand([
        ["from_url"],
        ["from_buf"]
    ])
    def test_decode_all_frames_cpu(self, input_type):
        buf = None

        if input_type == "from_url":
            pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=-1)
        else:
            buf = open(self.gtInfo.uri, "rb")
            pyDec = vali.PyDecoder(buf, {}, gpu_id=-1)

        dec_frames = 0
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = pyDec.DecodeSingleFrame(frame)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.gtInfo.num_frames, dec_frames)
        self.assertEqual(details, vali.TaskExecInfo.END_OF_STREAM)

        if buf is not None:
            buf.close()

    @parameterized.expand([
        ["from_url"],
        ["from_buf"]
    ])
    def test_decode_all_surfaces_gpu(self, input_type):
        buf = None
        gpu_id = 0
        if input_type == "from_url":
            pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id)
        else:
            buf = open(self.gtInfo.uri, "rb")
            pyDec = vali.PyDecoder(buf, {}, gpu_id)
        dec_frames = 0
        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id)
        while True:
            success, details = pyDec.DecodeSingleSurface(surf)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.gtInfo.num_frames, dec_frames)
        self.assertEqual(details, vali.TaskExecInfo.END_OF_STREAM)

        if buf is not None:
            buf.close()

    def test_decode_high_bit_depth_gpu(self):
        gpu_id = 0
        pyDec = vali.PyDecoder(self.hbdInfo.uri, {}, gpu_id)
        dec_frames = 0
        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id)
        while True:
            success, details = pyDec.DecodeSingleSurface(surf)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.hbdInfo.num_frames, dec_frames)
        self.assertEqual(details, vali.TaskExecInfo.END_OF_STREAM)

    def test_decode_high_bit_depth_cpu(self):
        gpu_id = -1
        pyDec = vali.PyDecoder(self.hbdInfo.uri, {}, gpu_id)
        dec_frames = 0
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = pyDec.DecodeSingleFrame(frame)
            if not success:
                break
            dec_frames += 1
        self.assertEqual(self.hbdInfo.num_frames, dec_frames)
        self.assertEqual(details, vali.TaskExecInfo.END_OF_STREAM)

    @parameterized.expand([
        ["from_url"],
        ["from_buf"]
    ])
    def test_check_all_frames_cpu(self, input_type):
        buf = None

        if input_type == "from_url":
            pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=-1)
        else:
            buf = open(self.gtInfo.uri, "rb")
            pyDec = vali.PyDecoder(buf, {}, gpu_id=-1)

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

        if buf is not None:
            buf.close()

    @parameterized.expand([
        ["basic"],
        ["pts_increase_check"],
    ])
    def test_monotonous_pts_increase_cpu(self, case_name: str):
        gtInfo = self.gtByName(case_name)

        pyDec = vali.PyDecoder(input=gtInfo.uri, opts={}, gpu_id=-1)
        frame = np.ndarray(dtype=np.uint8, shape=(pyDec.HostFrameSize))
        pktData = vali.PacketData()
        lastPts = vali.NO_PTS

        while True:
            success, info = pyDec.DecodeSingleFrame(frame, pktData)
            if not success:
                break
            self.assertGreaterEqual(pktData.pts, lastPts)
            lastPts = pktData.pts

    @parameterized.expand([
        ["basic"],
        ["pts_increase_check"],
    ])
    def test_monotonous_pts_increase_gpu(self, case_name: str):
        gtInfo = self.gtByName(case_name)

        pyDec = vali.PyDecoder(input=gtInfo.uri, opts={}, gpu_id=0)
        surf = vali.Surface.Make(pyDec.Format, pyDec.Width, pyDec.Height,
                                 gpu_id=0)
        pktData = vali.PacketData()
        lastPts = vali.NO_PTS

        while True:
            success, info = pyDec.DecodeSingleSurface(surf, pktData)
            if not success:
                break
            self.assertGreaterEqual(pktData.pts, lastPts)
            lastPts = pktData.pts

    @parameterized.expand([
        [True, "avc_8bit", "basic", "basic_nv12"],
        [False, "avc_8bit", "basic", "basic_nv12"],
        [True, "hevc_10bit", "hevc10", "hevc10_p10"],
        [False, "hevc_10bit", "hevc10", "hevc10_p10"],
    ])
    def test_check_all_surfaces_gpu(
            self,
            is_async: bool,
            case_name: str,
            gt_comp_name: str,
            gt_raw_name: str):
        """
        This test checks decoded surfaces pixel-by-pixel.

        Args:
            is_async (bool): if True, will run async non-blocking api, otherwise sync blocking api
            case_name (str): test case name
            gt_comp_name (str): ground truth information about compressed file
            gt_raw_name (str): ground truth information about raw file
        """

        gt_comp = self.gtByName(gt_comp_name)
        gt_raw = self.gtByName(gt_raw_name)

        pyDec = vali.PyDecoder(input=gt_comp.uri, opts={}, gpu_id=0)

        pyDwn = vali.PySurfaceDownloader(
            gpu_id=0) if not is_async else vali.PySurfaceDownloader(gpu_id=0, stream=pyDec.Stream)

        dec_frames = 0
        with open(gt_raw.uri, "rb") as f_in:
            while True:
                surf = vali.Surface.Make(
                    pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)
                frame = np.ndarray(dtype=np.uint8, shape=surf.HostSize)

                # Decode single surface
                if is_async:
                    success, details, _ = pyDec.DecodeSingleSurfaceAsync(
                        surf, False)
                else:
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
                    if dec_frames < gt_raw.num_frames:
                        self.log.error(
                            "Failed to read GT video frame " + str(dec_frames))
                    break

                if not np.array_equal(frame_gt, frame):
                    self.log.error("Mismatch at frame " + str(dec_frames))
                    self.log.error(
                        "PSNR: " + str(tc.measurePSNR(frame_gt, frame)))

                    tc.dumpFrameToDisk(frame_gt, "dec", surf.Width,
                                       surf.Height, "yuv_gt.yuv")
                    tc.dumpFrameToDisk(frame, "dec", surf.Width,
                                       surf.Height, "yuv_dist.yuv")

                    break

                dec_frames += 1

        self.assertEqual(self.nv12Info.num_frames, dec_frames)

    def test_check_decode_status_cpu(self):
        pyDec = vali.PyDecoder(self.gtInfo.uri, {}, gpu_id=-1)
        frame = np.ndarray(dtype=np.uint8, shape=())
        while True:
            success, details = pyDec.DecodeSingleFrame(frame)
            if not success:
                self.assertEqual(details, vali.TaskExecInfo.END_OF_STREAM)
                break
            self.assertEqual(details, vali.TaskExecInfo.SUCCESS)

    def test_decode_single_frame_out_pkt_data_cpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=-1)
        frame = np.ndarray(dtype=np.uint8, shape=())

        dec_frame = 0
        last_pts = vali.NO_PTS
        while True:
            pdata = vali.PacketData()
            success, _ = pyDec.DecodeSingleFrame(frame, pdata)
            if not success:
                break
            self.assertNotEqual(pdata.pts, vali.NO_PTS)
            if 0 != dec_frame:
                self.assertGreaterEqual(pdata.pts, last_pts)
            dec_frame += 1
            last_pts = pdata.pts

    def test_seek_cpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=-1)
        frame = np.ndarray(dtype=np.uint8, shape=())
        frame_gt = np.ndarray(dtype=np.uint8, shape=())

        # Seek to random frame within input video frames range.
        start_frame = random.randint(0, gtInfo.num_frames - 1)
        seek_ctx = vali.SeekContext(seek_frame=start_frame)
        success, _ = pyDec.DecodeSingleFrame(
            frame=frame, seek_ctx=seek_ctx)
        self.assertTrue(success)

        # Now check if it's the same as via continuous decode
        # For that decoder has to be recreated
        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=-1)
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
                tc.dumpFrameToDisk(frame_gt, "dec", pyDec.Width,
                                   pyDec.Height, "yuv_cont.yuv")
                tc.dumpFrameToDisk(frame, "dec", pyDec.Width,
                                   pyDec.Height, "yuv_seek.yuv")
                self.fail(
                    "Seek frame isnt same as continuous decode frame")

    @tc.repeat(3)
    def test_seek_backwards_gpu(self):
        """
        This test seeks to the random frame in video and saves it.
        
        Then it seeks in backward direction to the random frame and saves it
        as well.

        Two frames are expected to be different.
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=0)
        pyDwn = vali.PySurfaceDownloader(gpu_id=0)

        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)

        frames = [
            np.ndarray(dtype=np.uint8, shape=(surf.HostSize)),
            np.ndarray(dtype=np.uint8, shape=(surf.HostSize))
        ]

        # Seek to the random frame, decode, save.
        seek_frame = random.randint(
            int(gtInfo.num_frames / 2), gtInfo.num_frames - 1)

        success, details = pyDec.DecodeSingleSurface(
            surf=surf, seek_ctx=vali.SeekContext(seek_frame))
        self.assertTrue(success,
                        "Failed to decode frame " + str(seek_frame) +
                        ": " + str(details))

        success, details = pyDwn.Run(src=surf, dst=frames[0])
        if not success:
            self.fail("Failed to download surface: " + str(details))

        # Now seek back and do the same
        seek_frame = random.randint(0, seek_frame - 1)

        success, details = pyDec.DecodeSingleSurface(
            surf=surf, seek_ctx=vali.SeekContext(seek_frame))
        self.assertTrue(success,
                        "Failed to decode frame " + str(seek_frame) +
                        ": " + str(details))

        success, details = pyDwn.Run(src=surf, dst=frames[1])
        if not success:
            self.fail("Failed to download surface: " + str(details))

        # Check if frames are different (issue #89)
        self.assertFalse(np.array_equal(frames[0], frames[1]))

    def test_display_rotation(self):
        """
        This test checks display rotation sidedata.
        """
        pyDec = vali.PyDecoder(self.rotInfo.uri, {}, gpu_id=0)
        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)

        # Display rotation is bound to particular frame.
        # Hence no information shall be available at this point.
        self.assertEqual(pyDec.DisplayRotation, 361.0)

        success, info = pyDec.DecodeSingleSurface(surf)
        self.assertTrue(success)
        self.assertEqual(info, vali.TaskExecInfo.SUCCESS)

        # Now we shall get rotation info after the frame is decoded.
        self.assertEqual(pyDec.DisplayRotation, self.rotInfo.display_rotation)

    @tc.repeat(3)
    def test_seek_big_timestamp_gpu(self):
        """
        This test checks seek accuracy.
        Seek to random, but relatively big timestamp is done. It's expected
        that seek frame PTS will be within 1% tolerance of that calculated
        via frame number and duration.
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["generated"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=0)
        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)

        # Seek to random frame within second half of the video
        for i in range(0, 2):
            start_frame = random.randint(
                int(gtInfo.num_frames / 2), gtInfo.num_frames - 1)
            seek_ctx = vali.SeekContext(seek_frame=start_frame)
            packet_data = vali.PacketData()
            success, _ = pyDec.DecodeSingleSurface(
                surf, packet_data, seek_ctx)
            self.assertTrue(success)

            # This video has duration of 512 units per every frame.
            # Calculate expected timestamp. For cuvid, timestamps are
            # reconstructed by FFMpeg, so they may vary very slightly.
            #
            # Delta below 1% is considered acceptable.
            expected_pts = start_frame * 512
            self.assertLessEqual(
                abs(packet_data.pts - expected_pts) / expected_pts,
                0.01
            )

    @tc.repeat(3)
    def test_seek_gpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=0)
        pyDwn = vali.PySurfaceDownloader(gpu_id=0)

        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)
        frame = np.ndarray(dtype=np.uint8, shape=(surf.HostSize))
        frame_gt = np.ndarray(dtype=np.uint8, shape=(surf.HostSize))

        # Seek to random frame within input video frames range.
        start_frame = random.randint(0, gtInfo.num_frames - 1)
        seek_ctx = vali.SeekContext(seek_frame=start_frame)
        success, _ = pyDec.DecodeSingleSurface(
            surf=surf, seek_ctx=seek_ctx)
        self.assertTrue(success)
        self.assertTrue(pyDwn.Run(src=surf, dst=frame))

        # Now check if it's the same as via continuous decode
        # For that decoder has to be recreated
        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=0)
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
                tc.dumpFrameToDisk(frame_gt, "dec", pyDec.Width,
                                   pyDec.Height, "yuv_cont.yuv")
                tc.dumpFrameToDisk(frame, "dec", pyDec.Width,
                                   pyDec.Height, "yuv_seek.yuv")
                self.fail(
                    "Seek frame isnt same as continuous decode frame")

    def test_get_motion_vectors_cpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])
            pyDec = vali.PyDecoder(
                gtInfo.uri, {"flags2": "+export_mvs"}, gpu_id=-1)

        frame = np.ndarray(shape=(0), dtype=np.uint8)

        success, _ = pyDec.DecodeSingleFrame(frame)
        self.assertTrue(success)

        # First frame shall be I frame, hence no motion vectors.
        mv = pyDec.MotionVectors
        self.assertEqual(len(mv), 0)

        success, _ = pyDec.DecodeSingleFrame(frame)
        self.assertTrue(success)

        # Second frame shall be either P or B, hence motion vectors
        # shall be there.
        mv = pyDec.MotionVectors
        self.assertGreater(len(mv), 0)

        # Very basic sanity check:
        # Motion scale means precision, can't be 0.
        # Usually it's 2 or 4 (half- or quater- pixel precision).
        # Source is either -1 (prediction from past) or 1 (from future).
        first_mv = mv[0]
        self.assertNotEqual(first_mv.source, 0)
        self.assertNotEqual(first_mv.motion_scale, 0)

    def test_resolution_change_gpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["res_change"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=0)

        width = gtInfo.width
        height = gtInfo.height

        dec_frame = 0
        while True:
            surf = vali.Surface.Make(pyDec.Format, width, height, gpu_id=0)
            success, info = pyDec.DecodeSingleSurface(surf)

            if not success:
                break

            if info == vali.TaskExecInfo.RES_CHANGE:
                width = int(width * gtInfo.res_change_factor)
                height = int(height * gtInfo.res_change_factor)

                # Upon resolution change decoder will not return decoded
                # pixels to user. Hence surface dimensions will not be same
                # to that of decoder.
                self.assertNotEqual(surf.Width, width)
                self.assertNotEqual(surf.Height, height)
            else:
                dec_frame += 1

            self.assertEqual(pyDec.Width, width, str(dec_frame))
            self.assertEqual(pyDec.Height, height, str(dec_frame))

        self.assertEqual(dec_frame, gtInfo.num_frames)

    def test_resolution_change_cpu(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["res_change"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=-1)

        width = gtInfo.width
        height = gtInfo.height

        dec_frame = 0
        while True:
            frame = np.ndarray(shape=(0), dtype=np.uint8)
            success, info = pyDec.DecodeSingleFrame(frame)

            if not success:
                break

            if info == vali.TaskExecInfo.RES_CHANGE:
                width = int(width * gtInfo.res_change_factor)
                height = int(height * gtInfo.res_change_factor)

                # Upon resolution change decoder will not return decoded
                # pixels to user. Hence frame size will not be same
                # to that of decoder.
                self.assertNotEqual(pyDec.HostFrameSize, frame.size)
            else:
                dec_frame += 1

            self.assertEqual(pyDec.Width, width, str(dec_frame))
            self.assertEqual(pyDec.Height, height, str(dec_frame))

        self.assertEqual(dec_frame, gtInfo.num_frames)

    @parameterized.expand(tc.getDevices())
    def test_invalid_url(self, device_name: str, device_id: int):
        """
        This test checks invalid input URL. Decoder shall raise exception.

        Args:
            device_name (str): device name
            device_id (int): gpu ID or -1 if run on CPU
        """
        err_str = 'I/O error' if os.name == 'nt' else 'Input/output error'
        try:
            url = "http://www.middle.of.nowhere:8765/cam_9000"
            pyDec = vali.PyDecoder(url, {}, device_id)
        except RuntimeError as e:
            self.assertRegex(str(e), err_str)
            return

        self.fail("Test is expected to raise exception")

    @parameterized.expand(tc.getDevices())
    def test_decode_key_frames_only(self, device_name: str, device_id: int):
        """
        This test checks that only key frames are decoded in corresp.
        decode mode.

        Args:
            device_name (str): GPU / CPU name
            device_id (int): GPU id or -1 when on CPU
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["generated"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=device_id)
        pyDec.SetMode(vali.DecodeMode.KEY_FRAMES)

        num_key_frames = gtInfo.num_frames // gtInfo.gop_size
        dec_frames = 0

        if pyDec.IsAccelerated:
            surf = vali.Surface.Make(
                pyDec.Format, pyDec.Width, pyDec.Height, device_id)
        else:
            frame = np.ndarray(dtype=np.uint8, shape=pyDec.HostFrameSize)

        while True:
            success, info = pyDec.DecodeSingleSurface(
                surf) if pyDec.IsAccelerated else pyDec.DecodeSingleFrame(frame)
            if not success:
                break
            dec_frames += 1

        self.assertEqual(dec_frames, num_key_frames)

    @parameterized.expand(tc.getDevices())
    def test_seek_key_frames_only(self, device_name: str, device_id: int):
        """
        This test checks that in key frames mode decoder only jumps to actual
        key frames and doesn't decode anything else.

        Args:
            device_name (str): GPU / CPU name
            device_id (int): GPU id or -1 when on CPU
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["generated"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=device_id)
        pyDec.SetMode(vali.DecodeMode.KEY_FRAMES)

        if pyDec.IsAccelerated:
            surf = vali.Surface.Make(
                pyDec.Format, pyDec.Width, pyDec.Height, device_id)
        else:
            frame = np.ndarray(dtype=np.uint8, shape=pyDec.HostFrameSize)

        num_key_frames = gtInfo.num_frames // gtInfo.gop_size

        # Key frame we want to seek to
        rnd_key_frame = random.randint(1, num_key_frames - 2) * gtInfo.gop_size

        # Now generate some random frame number between it and next key frame
        seek_frame = random.randint(
            rnd_key_frame, rnd_key_frame+gtInfo.gop_size-1)

        # And make sure that when we seek in key frame mode, decoder
        # won't jump just to any frame number
        pkt_data = vali.PacketData()
        seek_ctx = vali.SeekContext(seek_frame)
        success, _ = pyDec.DecodeSingleSurface(
            surf, pkt_data, seek_ctx) if pyDec.IsAccelerated else pyDec.DecodeSingleFrame(frame, pkt_data, seek_ctx)

        # Decoded frame PTS shall be equal to that of our desired key frame.
        # This video has duration of 512 units per every frame.
        self.assertTrue(success)
        self.assertEqual(pkt_data.key, 1)
        self.assertEqual(pkt_data.pts, rnd_key_frame * 512)

    @parameterized.expand(tc.getDevices())
    def test_variable_frame_rate(self, device_name: str, device_id: int):
        """Test variable frame rate detection.
        
        This test verifies that the IsVFR() method correctly identifies
        videos with variable frame rates by comparing the stream's fps
        with its average fps.
        
        Args:
            device_name (str): GPU / CPU name
            device_id (int): GPU id or -1 when on CPU
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=device_id)

        # For a constant frame rate video, IsVFR should return False
        self.assertFalse(pyDec.IsVFR)

        # TODO: Add test with a VFR video file when available
        # For a variable frame rate video, IsVFR should return True
        # self.assertTrue(pyDec.IsVFR())

    @parameterized.expand(tc.getDevices())
    def test_decode_mode(self, device_name: str, device_id: int):
        """Test decode mode setting and retrieval.
        
        This test verifies that the decoder's mode can be set and retrieved
        correctly using SetMode() and GetMode() methods.
        
        Args:
            device_name (str): GPU / CPU name
            device_id (int): GPU id or -1 when on CPU
        """
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=device_id)

        # Default mode should be NORMAL
        self.assertEqual(pyDec.Mode, vali.DecodeMode.ALL_FRAMES)

        # Test setting to KEY_FRAMES mode
        pyDec.SetMode(vali.DecodeMode.KEY_FRAMES)
        self.assertEqual(pyDec.Mode, vali.DecodeMode.KEY_FRAMES)

        # Test setting back to NORMAL mode
        pyDec.SetMode(vali.DecodeMode.ALL_FRAMES)
        self.assertEqual(pyDec.Mode, vali.DecodeMode.ALL_FRAMES)

    def test_cuda_stream(self):
        """Test CUDA stream handling.
        
        This test verifies that the decoder's CUDA stream can be retrieved
        and used for synchronization. Only runs on GPU devices.
        """
        gpu_id = 0  # Use first GPU device

        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id=gpu_id)

        # Get the CUDA stream
        stream = pyDec.Stream
        self.assertIsNotNone(stream)

        # Create a surface and decode a frame
        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id)
        success, _ = pyDec.DecodeSingleSurface(surf)
        self.assertTrue(success)

        # The stream should be valid and usable for synchronization
        # Note: We can't directly test stream validity, but if we got here
        # without errors, the stream is likely valid


if __name__ == "__main__":
    unittest.main()
