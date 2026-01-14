# %% [markdown]
# This sample illustrates torch segmentation. \
# To illustrate interop with cvcuda and nvimgcodec, they are used to:
# - OSD operations like bbox and label drawing.
# - JPEG compression.
#
# For memory sharing both DLPack and CAI (CUDA Array Interface) are used.

# %%
import python_vali as vali
import numpy as np

import torch
from ultralytics import YOLO

import threading
import time
import nvtx
import queue

# %%
# Every input will be processed in separate thread to avoid potential network
# packet read timeout influencing other video sources
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

# %%
coco_names = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# %% [markdown]
# Resources which are shared between threads:
#   - Model
#   - Color converter
#   - JPEG encoder
#   - Video decoders (one per input)
#

# %%
# Prepare model
model = YOLO("yolo11n.pt")
model.eval()
model.to("cuda")

# %%
# GPU-accelerated decoders
try:
    pyDecs = []
    for url in urls:
        pyDecs.append(vali.PyDecoder(url, opts={}, gpu_id=0))
except Exception as e:
    print(f"{repr(e)}")
    exit(1)

# GPU-accelerated combined resize + conversion
pyUD = vali.PySurfaceUD(gpu_id=0)

# NN expects input pictures to be of this size
target_w = 640
target_h = 480

# NN inference batch size
batch_size = 8

# Tenor queue
frame_queue = queue.Queue(maxsize=batch_size)

# %% [markdown]
# `VideoSourceContext` encapsulates compressed packets read from source

# %%


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

# %% [markdown]
# `InferenceContext` encapsulates data needed to decode video frame and prepare it for inference

# %%


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

        success, details = pyUD.Run(self.surfaces[0], self.surfaces[-1])
        if not success:
            print(details)
            return vali.DecodeStatus.ERROR

        img_tensor = torch.from_dlpack(self.surfaces[-1]).clamp(0.0, 1.0)
        img_tensor = torch.reshape(img_tensor, [3, target_h, target_w])

        self.frame_id += 1
        self.frame_queue.put(img_tensor.clone().detach())

        return vali.DecodeStatus.SUCCESS

    @staticmethod
    def process_sources(contexts: list):
        while True:
            num_active_ctx = len(contexts)

            for ctx in contexts:
                # Skip inactive context processing
                if not ctx.active:
                    num_active_ctx -= 1
                    continue

                status = ctx.decode_to_tensor()
                if status == vali.DecodeStatus.DONE or status == vali.DecodeStatus.ERROR:
                    ctx.active = False
                    print(f"{ctx.frame_id} frames processed\n")

            # If there are no active contexts left, all inputs are processed
            if num_active_ctx == 0:
                return


class InferenceContext:
    def __init__(self):
        pass

    @nvtx.annotate()
    def _output(self, results) -> None:
        for result in results:
            pass

    @nvtx.annotate()
    def run_inference(self, frames: list[torch.tensor]) -> tuple[list[list[np.int32]], list[str]]:
        batch = torch.stack(frames, dim=0)
        results = model.model(batch)
        self._output(results)


# %% [markdown]
# Create multiple reader threads


# %%
readers = []

# Init readers first because it may take some time
for source_id in range(0, len(urls)):
    readers.append(ReaderContext(source_id, emulate_real_time=False))

# then start the threads
for reader in readers:
    reader.start()

# %% [markdown]
# Run decode + inference in another thread

# %%


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
