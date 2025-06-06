#
# Copyright 2022 NVIDIA Corporation
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

# Starting from Python 3.8 DLL search policy has changed.
# We need to add path to CUDA DLLs explicitly.
import sys
import os

import numpy as np

import torch
import torchvision
import json
import test_common as tc
import unittest

if os.name == "nt":
    # Add CUDA_PATH env variable
    cuda_path = os.environ["CUDA_PATH"]
    if cuda_path:
        os.add_dll_directory(os.path.join(cuda_path, "bin"))
    else:
        print("CUDA_PATH environment variable is not set.", file=sys.stderr)
        print("Can't set CUDA DLLs search path.", file=sys.stderr)
        exit(1)


import python_vali as vali

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


def collect_detections(boxes, classes, detections: list, frame_number) -> None:
    for i, box in enumerate(boxes):
        entry = {}
        if classes[i] in ["person", "dog"]:
            entry["frame"] = frame_number
            entry["label"] = classes[i]
            entry["bbox"] = np.ndarray.tolist(box)
            detections.append(entry)


def count_detections(types: dict, detections: dict) -> dict:
    my_types = types
    for entry in detections:
        label = entry["label"]
        if label in types.keys():
            my_types[label] += 1

    return my_types


class TestTorchSegmentation(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)
        # Init resnet
        self.model = torchvision.models.detection.ssd300_vgg16(
            weights=torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1)
        self.model.eval()
        self.model.to("cuda")
        self.gpu_id = 0

    def run_inference_on_video(self, input_video: str):
        # Init HW decoder
        py_dec = vali.PyDecoder(
            input=input_video,
            opts={},
            gpu_id=self.gpu_id)

        surfaces = [
            vali.Surface.Make(
                format=py_dec.Format,
                width=py_dec.Width,
                height=py_dec.Height,
                gpu_id=self.gpu_id),

            vali.Surface.Make(
                format=vali.PixelFormat.RGB,
                width=py_dec.Width,
                height=py_dec.Height,
                gpu_id=self.gpu_id),

            vali.Surface.Make(
                format=vali.PixelFormat.RGB_32F,
                width=py_dec.Width,
                height=py_dec.Height,
                gpu_id=self.gpu_id),

            vali.Surface.Make(
                format=vali.PixelFormat.RGB_32F_PLANAR,
                width=py_dec.Width,
                height=py_dec.Height,
                gpu_id=self.gpu_id)
        ]

        py_cvt = vali.PySurfaceConverter(
            gpu_id=self.gpu_id,
            stream=py_dec.Stream)

        event = vali.CudaStreamEvent(
            stream=py_dec.Stream,
            gpu_id=self.gpu_id)

        # Decoding cycle + inference on video frames.
        detections = []
        frame_number = 0
        while True:
            # Decode
            success, _ = py_dec.DecodeSingleSurfaceAsync(surfaces[0])
            if not success:
                break

            # Go through color conversion chain
            for i in range(0, len(surfaces) - 1):
                success, _ = py_cvt.RunAsync(
                    src=surfaces[i], dst=surfaces[i+1])
                if not success:
                    break

            event.Record()
            event.Wait()

            # Export to PyTorch tensor.
            # Please note that from_dlpack doesn't copy anything so we have to do
            # that manually. Otherwise, torch will use memory owned by VALI.
            img_tensor = torch.from_dlpack(surfaces[-1])
            img_tensor = img_tensor.clone().detach()

            # Normalize tensor to meet the NN expectations.
            img_tensor = torch.divide(img_tensor, 255.0)
            data_transforms = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            surface_tensor = data_transforms(img_tensor)
            input_batch = surface_tensor.unsqueeze(0).to("cuda")

            # Run inference.
            with torch.no_grad():
                outputs = self.model(input_batch)

            # Collect segmentation results.
            pred_classes = [coco_names[i]
                            for i in outputs[0]["labels"].cpu().numpy()]
            pred_scores = outputs[0]["scores"].detach().cpu().numpy()
            pred_bboxes = outputs[0]["boxes"].detach().cpu().numpy()
            boxes = pred_bboxes[pred_scores >= 0.5].astype(np.int32)

            collect_detections(boxes, pred_classes, detections, frame_number)
            frame_number += 1

        # Compare against GT. For simplicity, compare histograms only.
        types = {"person": 0, "dog": 0}
        hist = count_detections(types, detections)

        with open("detections.json", "r") as f_in:
            gt_detections = json.load(f_in)["detections"]
            gt_hist = count_detections(types, gt_detections)

            for key in types.keys():
                self.assertEqual(hist[key], gt_hist[key])

    def test_inference(self):
        with open("gt_files.json") as f:
            gtInfo = tc.GroundTruth(**json.load(f)["basic"])
            self.run_inference_on_video(input_video=gtInfo.uri)


if __name__ == "__main__":
    unittest.main()
