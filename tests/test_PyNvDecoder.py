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
import random
import json
import time
import test_common as tc


class TestDecoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        with open("gt_files.json") as f:
            self.gtInfo = tc.GroundTruth(**json.load(f)["basic"])
        self.nvDec = nvc.PyNvDecoder(self.gtInfo.uri, 0)

    def test_width(self):
        self.assertEqual(self.gtInfo.width, self.nvDec.Width())

    def test_height(self):
        self.assertEqual(self.gtInfo.height, self.nvDec.Height())

    def test_color_space(self):
        self.assertEqual(self.gtInfo.color_space, str(self.nvDec.ColorSpace()))

    def test_color_range(self):
        self.assertEqual(self.gtInfo.color_range, str(self.nvDec.ColorRange()))

    def test_format(self):
        self.assertEqual(self.gtInfo.pix_fmt, str(self.nvDec.Format()))

    def test_framerate(self):
        self.assertEqual(self.gtInfo.framerate, self.nvDec.Framerate())

    def test_avgframerate(self):
        self.assertEqual(self.gtInfo.framerate, self.nvDec.AvgFramerate())

    def test_isvfr(self):
        self.assertEqual(self.gtInfo.is_vfr, self.nvDec.IsVFR())

    def test_framesize(self):
        frame_size = int(self.nvDec.Width() * self.nvDec.Height() * 3 / 2)
        self.assertEqual(frame_size, self.nvDec.Framesize())

    def test_timebase(self):
        epsilon = 1e-4
        self.assertLessEqual(
            np.abs(self.gtInfo.timebase - self.nvDec.Timebase()), epsilon)

    def test_lastpacketdata(self):
        try:
            pdata = nvc.PacketData()
            self.nvDec.LastPacketData(pdata)
        except:
            self.fail("Test case raised exception unexpectedly!")


class TestDecoderStandalone(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        with open("gt_files.json") as f:
            self.gtInfo = tc.GroundTruth(**json.load(f)["basic"])

    def test_decodesurfacefrompacket(self):
        nvDmx = nvc.PyFFmpegDemuxer(self.gtInfo.uri, {})
        nvDec = nvc.PyNvDecoder(
            nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), 0
        )

        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while nvDmx.DemuxSinglePacket(packet):
            surf, _ = nvDec.DecodeSurfaceFromPacket(packet)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                self.assertEqual(nvDmx.Width(), surf.Width())
                self.assertEqual(nvDmx.Height(), surf.Height())
                self.assertEqual(nvDmx.Format(), surf.Format())
                return

    def test_decodesurfacefrompacket_outpktdata(self):
        nvDmx = nvc.PyFFmpegDemuxer(self.gtInfo.uri, {})
        nvDec = nvc.PyNvDecoder(
            nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), 0
        )

        dec_frames = 0
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        out_bst_size = 0
        while nvDmx.DemuxSinglePacket(packet):
            in_pdata = nvc.PacketData()
            nvDmx.LastPacketData(in_pdata)
            out_pdata = nvc.PacketData()
            surf, _ = nvDec.DecodeSurfaceFromPacket(
                in_pdata, packet, out_pdata)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
                out_bst_size += out_pdata.bsl

        while True:
            out_pdata = nvc.PacketData()
            surf, _ = nvDec.FlushSingleSurface(out_pdata)
            if not surf.Empty():
                out_bst_size += out_pdata.bsl
            else:
                break

        self.assertNotEqual(0, out_bst_size)

    def test_decode_all_surfaces(self):
        nvDmx = nvc.PyFFmpegDemuxer(self.gtInfo.uri, {})
        nvDec = nvc.PyNvDecoder(
            nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), 0
        )

        dec_frames = 0
        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while nvDmx.DemuxSinglePacket(packet):
            surf, _ = nvDec.DecodeSurfaceFromPacket(packet)
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
        while True:
            surf, _ = nvDec.FlushSingleSurface()
            self.assertIsNotNone(surf)
            if not surf.Empty():
                dec_frames += 1
            else:
                break
        self.assertEqual(self.gtInfo.num_frames, dec_frames)

    def test_check_decode_status(self):
        nvDmx = nvc.PyFFmpegDemuxer(self.gtInfo.uri, {})
        nvDec = nvc.PyNvDecoder(
            nvDmx.Width(), nvDmx.Height(), nvDmx.Format(), nvDmx.Codec(), 0
        )

        packet = np.ndarray(shape=(0), dtype=np.uint8)
        while nvDmx.DemuxSinglePacket(packet):
            surf, _ = nvDec.DecodeSurfaceFromPacket(packet)
            self.assertIsNotNone(surf)
        while True:
            surf, details = nvDec.FlushSingleSurface()
            self.assertIsNotNone(surf)
            if surf.Empty():
                self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)
                break
            else:
                self.assertEqual(details, nvc.TaskExecInfo.SUCCESS)


class TestDecoderBuiltin(unittest.TestCase):
    def test_decode_single_surface(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)
        try:
            surf, _ = nvDec.DecodeSingleSurface()
            self.assertIsNotNone(surf)
            self.assertFalse(surf.Empty())
        except:
            self.fail("Test case raised exception unexpectedly!")

    def test_decode_single_surface_outpktdata(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)

        dec_frame = 0
        last_pts = nvc.NO_PTS
        while True:
            pdata = nvc.PacketData()
            surf, _ = nvDec.DecodeSingleSurface(pdata)
            if surf.Empty():
                break
            self.assertNotEqual(pdata.pts, nvc.NO_PTS)
            if 0 != dec_frame:
                self.assertGreaterEqual(pdata.pts, last_pts)
            dec_frame += 1
            last_pts = pdata.pts

    def test_decode_single_surface_sei(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)

        total_sei_size = 0
        while True:
            sei = np.ndarray(shape=(0), dtype=np.uint8)
            surf, _ = nvDec.DecodeSingleSurface(sei)
            if surf.Empty():
                break
            total_sei_size += sei.size
        self.assertNotEqual(0, total_sei_size)

    def test_decode_single_surface_seek(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)

        start_frame = random.randint(0, gtInfo.num_frames - 1)
        dec_frames = 1
        seek_ctx = nvc.SeekContext(seek_frame=start_frame)
        surf, _ = nvDec.DecodeSingleSurface(seek_ctx)
        self.assertNotEqual(True, surf.Empty())
        while True:
            surf, _ = nvDec.DecodeSingleSurface()
            if surf.Empty():
                break
            dec_frames += 1
        self.assertEqual(gtInfo.num_frames - start_frame, dec_frames)

    @unittest.skip("Disable test: unstable on runner (but OK on local machine)")
    def test_decode_single_surface_cmp_vs_continuous(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        for idx in range(0, gtInfo.num_frames):
            # First get reconstructed frame with seek
            seek_ctx = nvc.SeekContext(seek_frame=idx)
            frame_seek = np.ndarray(shape=(0), dtype=np.uint8)
            pdata_seek = nvc.PacketData()
            nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)
            self.assertTrue(nvDec.DecodeSingleFrame(
                frame_seek, seek_ctx, pdata_seek)[0])

            # Then get it with continuous decoding
            nvDec = nvc.PyNvDecoder(gtInfo.uri, 0)
            frame_cont = np.ndarray(shape=(0), dtype=np.uint8)
            pdata_cont = nvc.PacketData()
            for i in range(0, idx + 1):
                self.assertTrue(nvDec.DecodeSingleFrame(
                    frame_cont, pdata_cont)[0])

            # Compare frames
            if not np.array_equal(frame_seek, frame_cont):
                fail_msg = ""
                fail_msg += "Seek frame number: " + str(idx) + ".\n"
                fail_msg += "Seek frame pts:    " + str(pdata_seek.pts) + ".\n"
                fail_msg += "Cont frame pts:    " + str(pdata_cont.pts) + ".\n"
                fail_msg += "Video frames are not same\n"

                # Save to disk
                tc.dumpFrameToDisk(frame_seek, "frame_seek", nvDec.Width(),
                                   nvDec.Height(), "nv12.yuv")
                tc.dumpFrameToDisk(frame_cont, "frame_cont", nvDec.Width(),
                                   nvDec.Height(), "nv12.yuv")

                self.fail(fail_msg)

    def test_check_decode_status(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)

        while True:
            surf, details = nvDec.DecodeSingleSurface()
            self.assertIsNotNone(surf)
            if surf.Empty():
                self.assertEqual(details, nvc.TaskExecInfo.END_OF_STREAM)
                break
            else:
                self.assertEqual(details, nvc.TaskExecInfo.SUCCESS)

    def test_decode_all_surfaces(self):
        for test_case in [
            "basic",
            "basic_mpeg4",
            "hevc10"
        ]:
            with open("gt_files.json") as f:
                gtInfo = tc.GroundTruth(**json.load(f)[test_case])

            nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)

            dec_frames = 0
            while True:
                surf, _ = nvDec.DecodeSingleSurface()
                self.assertIsNotNone(surf)
                if surf.Empty():
                    break
                dec_frames += 1
            self.assertEqual(gtInfo.num_frames, dec_frames)

    def test_decode_resolution_change(self):
        with open("gt_files.json") as f:
            resChangeInfo = tc.GroundTruth(**json.load(f)["res_change"])
        nvDec = nvc.PyNvDecoder(input=resChangeInfo.uri, gpu_id=0)
        rw = int(resChangeInfo.width * resChangeInfo.res_change_factor)
        rh = int(resChangeInfo.height * resChangeInfo.res_change_factor)

        dec_frames = 0
        while True:
            surf, _ = nvDec.DecodeSingleSurface()
            self.assertIsNotNone(surf)
            if surf.Empty():
                break
            else:
                dec_frames += 1

            if dec_frames < resChangeInfo.res_change_frame:
                self.assertEqual(surf.Width(), resChangeInfo.width)
                self.assertEqual(surf.Height(), resChangeInfo.height)
            else:
                self.assertEqual(surf.Width(), rw)
                self.assertEqual(surf.Height(), rh)

    @unittest.skip("Disable test: very noisy output")
    def test_log_warnings(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["log_warnings_nvdec"])
            nvDec = nvc.PyNvDecoder(input=gtInfo.uri, gpu_id=0)

            self.assertEqual(nvDec.Width(), gtInfo.width)
            self.assertEqual(nvDec.Height(), gtInfo.height)
            self.assertEqual(str(nvDec.Format()), gtInfo.pix_fmt)
            self.assertEqual(nvDec.Framerate(), gtInfo.framerate)

            frame = np.ndarray(shape=(0), dtype=np.uint8)

            dec_frames = 0
            while True:
                newFrame, execInfo = nvDec.DecodeSingleFrame(frame)
                if not newFrame:
                    break
                else:
                    dec_frames += 1

            self.assertEqual(execInfo, nvc.TaskExecInfo.END_OF_STREAM)
            self.assertEqual(gtInfo.num_frames, dec_frames)

    @unittest.skip("Need to setup RTSP server. Behaves differently on win / linux")
    def test_rtsp_nonexisting(self):
        timeout_ms = 1000
        tp = time.time()

        with self.assertRaises(RuntimeError):
            nvDec = nvc.PyNvDecoder(
                input="rtsp://127.0.0.1/nothing",
                gpu_id=0,
                opts={"timeout": str(timeout_ms)})

        tp = (time.time() - tp) * 1000
        self.assertGreaterEqual(tp, timeout_ms)


if __name__ == "__main__":
    unittest.main()
