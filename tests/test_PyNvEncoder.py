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
import test_common as tc
import json


class TestEncoderBasic(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def test_encode_all_surfaces(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])

        gpu_id = 0
        res = str(gtInfo.width) + "x" + str(gtInfo.height)
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)

        pyDec = vali.PyDecoder(gtInfo.uri, {}, gpu_id)
        nvEnc = vali.PyNvEncoder(
            {
                "preset": "P4",
                "tuning_info": "high_quality",
                "codec": "h264",
                "profile": "high",
                "s": res,
                "bitrate": "1M",
            },
            gpu_id,
        )

        frames_sent = 0
        frames_recv = 0

        surf = vali.Surface.Make(
            pyDec.Format, pyDec.Width, pyDec.Height, gpu_id=0)
        while True:
            success, _ = pyDec.DecodeSingleSurface(surf)
            if not success:
                break
            frames_sent += 1

            nvEnc.EncodeSingleSurface(surf, encFrame)
            if encFrame.size:
                frames_recv += 1

        while True:
            success = nvEnc.FlushSinglePacket(encFrame)
            if success and encFrame.size:
                frames_recv += 1
            else:
                break

        self.assertEqual(frames_sent, frames_recv)


if __name__ == "__main__":
    unittest.main()
