from pydantic import BaseModel
from typing import Optional
import numpy as np
from math import log10, sqrt
import unittest


class GroundTruth(BaseModel):
    uri: str
    width: int
    height: int
    res_change_factor: Optional[float] = None
    is_vfr: Optional[bool] = None
    pix_fmt: str
    framerate: Optional[float] = None
    num_frames: int
    res_change_frame: Optional[int] = None
    broken_frame: Optional[int] = None
    timebase: Optional[float] = None
    color_space: Optional[str] = None
    color_range: Optional[str] = None
    len_s: Optional[float] = None
    level: Optional[int] = None,
    profile: Optional[int] = None,
    delay: Optional[float] = None,
    gop_size: Optional[int] = None,
    bitrate: Optional[int] = None,
    num_streams: Optional[int] = None,
    video_stream_idx: Optional[int] = None,
    start_time: Optional[float] = None


def repeat(times):
    def repeatHelper(f):
        def callHelper(*args):
            for i in range(0, times):
                f(*args)

        return callHelper

    return repeatHelper


def dumpFrameToDisk(
        frame: np.ndarray,
        prefix: str,
        width: int,
        height: int,
        extension: str) -> None:
    """
    Saves numpy array with raw frame to disk.
    Filename is $(prefix)_$(width)x$(height).$(extension).

    Parameters
    ----------
    frame:      Numpy array with pixels
    prefix:     Filename prefix
    width:      Frame width in pixels
    height:     Frame height in pixels
    extension:  File extension. Will be treated as string, not as file format
    """

    fname = prefix + '_'
    fname += str(width) + 'x' + str(height) + '_' + extension

    with open(fname, 'wb') as fout:
        fout.write(frame)


def measurePSNR(gt: np.ndarray, dist: np.ndarray) -> float:
    """
    Measures the distance between frames using PSNR metric.

    Parameters
    ----------
    gt:     Ground Truth picture
    dist:   Distorted picture
    """
    mse = np.mean((gt - dist) ** 2)
    if mse == 0:
        return 100.0

    max_pixel = 255.0
    return 20 * log10(max_pixel / sqrt(mse))
