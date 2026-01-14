import python_vali as vali

import torch
from ultralytics import YOLO

import threading
import nvtx
import queue
import time

# Input video URLs
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

# NN model
model = YOLO("yolo11s.pt")
model.eval()
model.to("cuda")

# Video decoders pool
pyDecs = []
for url in urls:
    pyDecs.append(vali.PyDecoder(url, opts={}, gpu_id=0))

# Combined conversion + rescale
pyUD = vali.PySurfaceUD(gpu_id=0, stream=pyDecs[0].Stream)

# Video frame sidth and height as model likes it
target_w = 640
target_h = 480

# Model batch size
batch_size = 4

# Queue of video frames and CUDA events for sync between decoding and inference
# CUDA streams
inf_queue = queue.Queue(maxsize=batch_size)


class ReaderContext:
    """
    This class is used to read compressed packets from video source.
    """

    def __init__(self, source_id: int):
        """
        Constructor.

        Args:
            source_id (int): video source id
        """
        self.source_id = source_id
        self.pyDec = pyDecs[self.source_id]
        self.pkt_id = 0
        self.worker = threading.Thread(
            target=self.read_packet, name=f"reader_thread_{source_id}")

    def start(self):
        """
        Start reader thread.
        """
        self.worker.start()

    def join(self):
        """
        Join reader thread.
        """
        self.worker.join()

    def read_packet(self) -> None:
        """
        Reads packet into decoder internal queue.
        """

        @nvtx.annotate()
        def _read_pkt(pyDec: vali.PyDecoder) -> vali.DecodeStatus:
            return pyDec.ReadPacket()

        while True:
            status = _read_pkt(self.pyDec)
            if status in [vali.DecodeStatus.OVER, vali.DecodeStatus.ERROR]:
                return
            self.pkt_id += 1


class PreprocessingContext:
    """
    This class is used to decode and preprocess video frames for futher inference.
    """

    def __init__(self, inf_queue: queue.Queue, source_id: int):
        """
        Constructor

        Args:
            inf_queue (queue.Queue): preprocessed frames queue
            source_id (int): video source id
        """
        self.source_id = source_id
        self.frame_id = 0
        self.active = True
        self.inf_queue = inf_queue
        self.pyDec = pyDecs[self.source_id]
        self.event = vali.CudaStreamEvent(self.pyDec.Stream, gpu_id=0)
        self.start_time = time.time()

        self.surf_dec = vali.Surface.Make(
            format=self.pyDec.Format,
            width=self.pyDec.Width,
            height=self.pyDec.Height,
            gpu_id=0)

    @nvtx.annotate()
    def decode_to_tensor(self, surf_inf: vali.Surface) -> vali.DecodeStatus:
        """
        Decode and preprocess video frame in-place.
        Outputs performance stats every 3 seconds.

        Args:
            surf_inf (vali.Surface): output surface

        Returns:
            vali.DecodeStatus: opeation status
        """
        status = self.pyDec.DecodePacketToSurfaceAsync(self.surf_dec)
        if status != vali.DecodeStatus.SUCCESS:
            return status

        success, details = pyUD.RunAsync(self.surf_dec, surf_inf)
        if not success:
            print(details)
            return vali.DecodeStatus.ERROR

        # It's safe to reuse CUDA events
        self.event.Record()

        img_tensor = torch.from_dlpack(surf_inf).clamp(0.0, 1.0)
        img_tensor = torch.reshape(img_tensor, [3, target_h, target_w])

        # Queue has fixed size, so if this NVTX marker is lettings larger it
        # only means that preprocessing thread is waiting for inference thread
        # to take frames from queue.
        with nvtx.annotate("PutIntoQueue"):
            self.inf_queue.put((img_tensor, self.event))

        self.frame_id += 1

        # Output perf stats every 3 seconds
        if self.frame_id % (3 * int(self.pyDec.Framerate)) == 0:
            fps = int(self.frame_id / (time.time() - self.start_time))
            print(f"source {self.source_id}: {fps} fps")

        return vali.DecodeStatus.SUCCESS

    @staticmethod
    def process_sources(contexts: list):
        """
        Static method which checks decoders in round-robin fashon. If there's a
        decoded frame, it's preprocessed and put into queue.

        Args:
            contexts (list): list of preprocessing contexts.
        """
        # Max memory consumption is: `batch_size` tensors processed by model +
        # `batch_size` tensors in the queue.
        # So batch_size * 2 surfaces shall be plenty.
        surf_inf = [vali.Surface.Make(
            format=vali.PixelFormat.RGB_32F_PLANAR,
            width=target_w,
            height=target_h,
            gpu_id=0) for _ in range(0, batch_size * 2)]

        while True:
            num_active_ctx = len(contexts)
            idx = 0

            for ctx in contexts:
                if not ctx.active:
                    num_active_ctx -= 1
                    continue

                status = ctx.decode_to_tensor(surf_inf[idx])
                if status == vali.DecodeStatus.DONE or status == vali.DecodeStatus.ERROR:
                    ctx.active = False
                    print(f"{ctx.frame_id} frames processed\n")

                idx = ctx.frame_id % len(surf_inf)

            if num_active_ctx == 0:
                return


class InferenceContext:
    """
    This class is used to run inference on video frames.
    """

    def __init__(self):
        self.stream = torch.cuda.Stream()

    @nvtx.annotate()
    def _output(self, results) -> None:
        """
        Post process inference results
        """
        for result in results:
            pass

    @nvtx.annotate()
    def run_inference(self, frames: list[torch.tensor]) -> None:
        """
        Run inference.

        Args:
            frames (list[torch.tensor]): list of video frames.
        """
        batch = torch.stack(frames, dim=0)
        with torch.cuda.stream(self.stream):
            results = model.model(batch)
            self._output(results)


# Init readers first because it may take some time
readers = []
for source_id in range(0, len(urls)):
    readers.append(ReaderContext(source_id))

# Then start the threads
for reader in readers:
    reader.start()

# Init preprocessing
preproc_ctx = []
for source_id in range(0, len(readers)):
    preproc_ctx.append(PreprocessingContext(inf_queue, source_id))

preproc_thread = threading.Thread(
    target=PreprocessingContext.process_sources, name=f"preproc_thread", kwargs={'contexts': preproc_ctx})
preproc_thread.start()

# Run inference in main thread
inf_ctx = InferenceContext()
while True:
    try:
        # If no frames arrive within timeout, consider job done.
        timeout_s = 3.0
        frames = []
        events = []
        while len(frames) < batch_size:
            (frame, event) = inf_queue.get(timeout=timeout_s)
            frames.append(frame)
            events.append(event)

        # Postpone the events wait as long as possible in hope majority of them
        # will be done when it's time to run inference.
        for event in events:
            event.Wait()

        inf_ctx.run_inference(frames)
    except queue.Empty:
        break

for reader in readers:
    reader.join()

preproc_thread.join()
