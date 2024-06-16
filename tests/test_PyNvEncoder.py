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

        pyDec = nvc.PyDecoder(gtInfo.uri, {}, gpu_id)
        nvEnc = nvc.PyNvEncoder(
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

        surf = nvc.Surface.Make(
            pyDec.Format(), pyDec.Width(), pyDec.Height(), gpu_id=0)
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
