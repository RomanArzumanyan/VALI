#
# Copyright 2025 Roman Arzumanyan
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

import numpy as np
import sys
from datetime import datetime
import python_vali as vali
import psutil
import test_common as tc

PROCESS = psutil.Process()

URLS = [
    "data/output.mp4",
    "data/test_multires.mkv",
    "data/pts_test_video.mkv"
]

SAMPLING_PERIOD = 500


def get_sampling_period(py_dec: vali.PyDecoder) -> int:
    if not py_dec.NumFrames:
        return py_dec.AvgFramerate

    if py_dec.NumFrames < SAMPLING_PERIOD:
        return py_dec.NumFrames // 10

    return SAMPLING_PERIOD


def print_mem_usage(handle=None):
    new_ram = PROCESS.memory_info().rss // 2**20
    if not handle:
        print(f"{datetime.now()} {new_ram}MB RAM")
    else:
        total, used, free = tc.get_gpu_mem_stats(handle)
        print(f"{datetime.now()} {new_ram}MB RAM / {used}MB vRAM")


def on_gpu(url, gpu_id):
    py_dec = vali.PyDecoder(url, {}, gpu_id)
    surf = vali.Surface.Make(
        py_dec.Format, py_dec.Width, py_dec.Height, gpu_id)
    i = 0

    print(f'--- {url}: {py_dec.Width}x{py_dec.Height} {py_dec.Bitrate}bps ---')

    with tc.nvml_session(gpu_id) as handle:
        while True:
            if i % get_sampling_period(py_dec) == 0:
                print_mem_usage(handle)

            success, _ = py_dec.DecodeSingleSurface(surf)
            if not success:
                break

            i += 1

    print(f'{i} frames decoded')


def on_cpu(url):
    py_dec = vali.PyDecoder(url, {}, gpu_id=-1)
    frame = np.ndarray(dtype=np.uint8, shape=(py_dec.HostFrameSize))
    i = 0

    print(f'--- {url}: {py_dec.Width}x{py_dec.Height} {py_dec.Bitrate}bps ---')

    while True:
        if i % get_sampling_period(py_dec) == 0:
            print_mem_usage()

        success, _ = py_dec.DecodeSingleFrame(frame)
        if not success:
            break

        i += 1

    print(f'{i} frames decoded')


# This test doesn't rely on unittest framework.
# It's done on purpose, because it's quite long and be better run
# explicitely, not via unittest discover.
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {__file__} %num_runs% %device_id% (-1 for cpu)")
        exit(1)

    for i in range(0, int(sys.argv[1])):
        device_id = int(sys.argv[2])
        if device_id >= 0:
            print(f'---- GPU {device_id} ----')
            for url in URLS:
                on_gpu(url, device_id)
        else:
            print(f'---- CPU ----')
            for url in URLS:
                on_cpu(url)
