from pydantic import BaseModel
from typing import Optional


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
