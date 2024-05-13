#
# Copyright 2022 NVIDIA Corporation
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


class TestSurfacePycuda(unittest.TestCase):
    def __init__(self, methodName):
        super().__init__(methodName=methodName)

    def run_inference_on_video(self, gpu_id: int, input_video: str):
        # Init resnet
        model = torchvision.models.detection.ssd300_vgg16(
            weights=torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1)
        model.eval()
        model.to("cuda")

        # Init HW decoder
        nvDec = nvc.PyNvDecoder(input_video, gpu_id)

        # NN expects images to be 3 channel planar RGB.
        # No requirements for input image resolution, it will be rescaled internally.
        target_w, target_h = nvDec.Width(), nvDec.Height()

        # Converter from NV12 which is Nvdec native pixel fomat.
        to_rgb = nvc.PySurfaceConverter(
            target_w, target_h, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, gpu_id
        )

        # Converter from RGB to planar RGB because that's the way
        # pytorch likes to store the data in it's tensors.
        to_pln = nvc.PySurfaceConverter(
            target_w, target_h, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, gpu_id
        )

        # Use bt709 and jpeg just for illustration purposes.
        cc_ctx = nvc.ColorspaceConversionContext(
            nvc.ColorSpace.BT_709, nvc.ColorRange.JPEG)

        # Decoding cycle + inference on video frames.
        detections = []
        frame_number = 0
        while True:
            # Decode 1 compressed video frame to CUDA memory.
            surf_nv12, _ = nvDec.DecodeSingleSurface()
            if surf_nv12.Empty():
                break

            # Convert NV12 > RGB.
            surg_rgb, _ = to_rgb.Execute(surf_nv12, cc_ctx)
            if surg_rgb.Empty():
                print("Can not convert nv12 -> rgb")
                break

            # Convert RGB > planar RGB.
            surf_pln, _ = to_pln.Execute(surg_rgb, cc_ctx)
            if surf_pln.Empty():
                print("Can not convert rgb -> rgb planar")
                break

            # Export to PyTorch tensor.
            # Please note that from_dlpack doesn't copy anything so we have to do
            # that manually. Otherwise, torch will use memory owned by VALI.
            img_tensor = torch.from_dlpack(surf_pln)
            img_tensor = img_tensor.clone().detach()
            img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)

            # Normalize tensor to meet the NN expectations.
            img_tensor = torch.divide(img_tensor, 255.0)
            data_transforms = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            surface_tensor = data_transforms(img_tensor)
            input_batch = surface_tensor.unsqueeze(0).to("cuda")

            # Run inference.
            with torch.no_grad():
                outputs = model(input_batch)

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
            self.run_inference_on_video(gpu_id=0, input_video=gtInfo.uri)


if __name__ == "__main__":
    unittest.main()
