#
# Copyright 2024 Vision Labs LLC
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

# We use 42 (dB) as the measure of similarity.
# If two images have PSNR higher than 42 (dB) we consider them the same.
psnr_threshold = 42.0


class TestSurfaceResizer(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_resize_nv12(self):
        with open("gt_files.json") as f:
            gt_values = json.load(f)
            nv12Info = tc.GroundTruth(**gt_values["basic_nv12"])
            nv12SmallInfo = tc.GroundTruth(**gt_values["small_nv12"])

        nvUpl = vali.PyFrameUploader(gpu_id=0)
        nvRes = vali.PySurfaceResizer(vali.PixelFormat.NV12, gpu_id=0)
        nvDwn = vali.PySurfaceDownloader(gpu_id=0)

        nv12_fin = open(nv12Info.uri, "rb")
        small_nv12_fin = open(nv12SmallInfo.uri, "rb")

        for i in range(0, nv12Info.num_frames):
            # Make input and output Surfaces
            surf_src = vali.Surface.Make(
                vali.PixelFormat.NV12,
                nv12Info.width,
                nv12Info.height,
                gpu_id=0)

            surf_dst = vali.Surface.Make(
                vali.PixelFormat.NV12,
                nv12SmallInfo.width,
                nv12SmallInfo.height,
                gpu_id=0)

            # Read input and GT frames from file
            frame_nv12 = np.fromfile(nv12_fin, np.uint8, surf_src.HostSize)
            frame_gt = np.fromfile(small_nv12_fin, np.uint8, surf_dst.HostSize)

            # Upload src to GPU
            if not nvUpl.Run(frame_nv12, surf_src):
                self.fail("Failed to upload frame")

            # Resize to dst
            if not nvRes.Run(surf_src, surf_dst):
                self.fail("Fail to resize surface ")

            # Download dst to numpy array
            frame_nv12.resize(frame_gt.size)
            if not nvDwn.Run(surf_dst, frame_nv12):
                self.fail("Fail to download surface")

            # Compare resize result against GT
            score = tc.measurePSNR(frame_gt, frame_nv12)

            # Dump both frames to disk in case of failure
            if score < psnr_threshold:
                tc.dumpFrameToDisk(
                    frame_nv12,
                    "res",
                    nv12SmallInfo.width,
                    nv12SmallInfo.height,
                    "nv12_dist")

                tc.dumpFrameToDisk(
                    frame_gt,
                    "res",
                    nv12SmallInfo.width,
                    nv12SmallInfo.height,
                    "nv12_gt")

                self.fail(
                    "PSNR score is below threshold: " + str(score))

        nv12_fin.close()
        small_nv12_fin.close()


if __name__ == "__main__":
    unittest.main()
