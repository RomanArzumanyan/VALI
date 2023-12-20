from pydantic import BaseModel
from typing import Optional
import numpy as np


class GroundTruth(BaseModel):
    uri: str
    width: int
    height: int
    res_change_factor: Optional[float] = None
    is_vfr: Optional[bool] = None
    pix_fmt: str
    framerate: float
    num_frames: int
    res_change_frame: Optional[int] = None
    broken_frame: Optional[int] = None
    timebase: Optional[float] = None
    color_space: Optional[str] = None
    color_range: Optional[str] = None
    len_s: Optional[float] = None


def dumpFrameToDisk(
        frame: np.ndarray,
        prefix: str,
        width: int,
        height: int,
        extension: str):

    fname = prefix + '_'
    fname += str(width) + 'x' + str(height) + '_' + extension

    with open(fname, 'wb') as fout:
        fout.write(frame)
