import python_vali as vali
import numpy as np

import torch
from ultralytics import YOLO

import threading
import time
import nvtx
import queue

urls = [
    "/home/vlabs/Videos/ToS-4k-1920.mov",
    "/home/vlabs/Videos/ToS-4k-1920.mov",
    "/home/vlabs/Videos/ToS-4k-1920.mov",
    "/home/vlabs/Videos/ToS-4k-1920.mov",
    "/home/vlabs/Videos/ToS-4k-1920.mov",
    "/home/vlabs/Videos/ToS-4k-1920.mov",
    "/home/vlabs/Videos/ToS-4k-1920.mov",
    "/home/vlabs/Videos/ToS-4k-1920.mov",    
]

model = YOLO("yolo11n.pt")
model.eval()
model.to("cuda")

try:
    pyDecs = []
    for url in urls:
        pyDecs.append(vali.PyDecoder(url, opts={}, gpu_id=0))
except Exception as e:
    print(f"{repr(e)}")
    exit(1)

pyUD = vali.PySurfaceUD(gpu_id=0, stream=pyDecs[0].Stream)

target_w = 640
target_h = 480

batch_size = 4
frame_queue = queue.Queue(maxsize=batch_size)


class ReaderContext:
    def __init__(self, source_id: int, emulate_real_time: bool):
        self.source_id = source_id
        self.pyDec = pyDecs[self.source_id]
        self.emulate_real_time = emulate_real_time
        self.pkt_id = 0
        self.worker = threading.Thread(
            target=self.read_packet, name=f"reader_thread_{source_id}")

    def start(self):
        self.worker.start()

    def join(self):
        self.worker.join()

    def read_packet(self):
        @nvtx.annotate()
        def _read_pkt(pyDec: vali.PyDecoder):
            return pyDec.ReadPacket()

        while True:
            status = _read_pkt(self.pyDec)

            if status in [vali.DecodeStatus.OVER, vali.DecodeStatus.ERROR]:
                return

            self.pkt_id += 1
            if self.emulate_real_time:
                time.sleep(1.0 / float(self.pyDec.Framerate))


class PreprocessingContext:
    def __init__(self, frame_queue: queue.Queue, target_w: int, target_h: int, source_id: int):
        self.source_id = source_id
        self.frame_id = 0
        self.active = True
        self.frame_queue = frame_queue
        self.pyDec = pyDecs[self.source_id]

        self.surfaces = [
            vali.Surface.Make(
                format=self.pyDec.Format,
                width=self.pyDec.Width,
                height=self.pyDec.Height,
                gpu_id=0),

            vali.Surface.Make(
                format=vali.PixelFormat.RGB_32F_PLANAR,
                width=target_w,
                height=target_h,
                gpu_id=0)
        ]

    @nvtx.annotate()
    def decode_to_tensor(self) -> vali.DecodeStatus:
        status = self.pyDec.DecodePacketToSurfaceAsync(self.surfaces[0])
        if status != vali.DecodeStatus.SUCCESS:
            return status

        success, details = pyUD.RunAsync(self.surfaces[0], self.surfaces[-1])
        if not success:
            print(details)
            return vali.DecodeStatus.ERROR

        img_tensor = torch.from_dlpack(self.surfaces[-1]).clamp(0.0, 1.0)
        img_tensor = torch.reshape(img_tensor, [3, target_h, target_w])

        # Queue has fixed size, so if this NVTX marker is lettings larger it
        # only means that preprocessing thread is waiting for inference thread
        # to take frames from queue.
        with nvtx.annotate("PutIntoQueue"):
            self.frame_queue.put(img_tensor.clone().detach())

        self.frame_id += 1
        return vali.DecodeStatus.SUCCESS

    @staticmethod
    def process_sources(contexts: list):
        while True:
            num_active_ctx = len(contexts)

            for ctx in contexts:
                if not ctx.active:
                    num_active_ctx -= 1
                    continue

                status = ctx.decode_to_tensor()
                if status == vali.DecodeStatus.DONE or status == vali.DecodeStatus.ERROR:
                    ctx.active = False
                    print(f"{ctx.frame_id} frames processed\n")

            if num_active_ctx == 0:
                return


class InferenceContext:
    def __init__(self):
        self.stream = torch.cuda.Stream()

    @nvtx.annotate()
    def _output(self, results) -> None:
        for result in results:
            pass

    @nvtx.annotate()
    def run_inference(self, frames: list[torch.tensor]) -> tuple[list[list[np.int32]], list[str]]:
        batch = torch.stack(frames, dim=0)
        with torch.cuda.stream(self.stream):
            results = model.model(batch)
            self._output(results)


# Init readers first because it may take some time
readers = []
for source_id in range(0, len(urls)):
    readers.append(ReaderContext(source_id, emulate_real_time=False))

# Then start the threads
for reader in readers:
    reader.start()

# Init preprocessing
preproc_ctx = []
for source_id in range(0, len(readers)):
    preproc_ctx.append(PreprocessingContext(
        frame_queue, target_w, target_h, source_id))

preproc_thread = threading.Thread(
    target=PreprocessingContext.process_sources, name=f"preproc_thread", kwargs={'contexts': preproc_ctx})
preproc_thread.start()

# Run inference in main thread
inf_ctx = InferenceContext()
while True:
    try:
        # If no frames arrive within timeout, consider job done
        timeout_s = 3.0
        frames = []
        while len(frames) < batch_size:
            frames.append(frame_queue.get(timeout=timeout_s))
        inf_ctx.run_inference(frames)
    except queue.Empty:
        break

for reader in readers:
    reader.join()

preproc_thread.join()
