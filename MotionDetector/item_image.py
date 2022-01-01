import datetime
from dataclasses import dataclass, field
from typing import List
from numpy import ndarray, array
from cv2 import imencode, rectangle

from MotionDetector.item_contours import ContourItem
from MotionDetector.buffer_motion import MotionFlag


@dataclass
class ImageItem:
    has_data: bool = False
    timestamp: str = ""
    motion_status: MotionFlag = MotionFlag.NoMotion
    original_frame: ndarray = array([])
    resized_frame: ndarray = array([])
    binary_frame: ndarray = array([])
    masked_frame: ndarray = array([])
    eroded_frame: ndarray = array([])
    dilated_frame: ndarray = array([])
    frames: List[ndarray] = field(default_factory=list)
    contours: List[ContourItem] = field(default_factory=list)

    def __post_init__(self):
        self.timestamp: str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    @property
    def overlay_frame(self) -> ndarray:
        frame = self.original_frame
        contour_items = self.contours
        for contour_item in contour_items:
            pt1, pt2 = contour_item.roi
            rectangle(img=frame,
                      pt1=pt1,
                      pt2=pt2,
                      color=(0, 0, 255),
                      thickness=3)
        return frame

    @property
    def binary_image(self) -> bytes:
        return imencode(".jpg", self.original_frame)[1].tobytes()

    @property
    def binary_overlay(self) -> bytes:
        return imencode(".jpg", self.overlay_frame)[1].tobytes()

    @property
    def has_contours(self) -> bool:
        return bool(self.contours)

    @property
    def min_area_ratio(self) -> float:
        return min([contour_item.area_ratio for contour_item in self.contours])

    @property
    def max_area_ratio(self) -> float:
        return max([contour_item.area_ratio for contour_item in self.contours])

    @property
    def sum_area_ratio(self) -> float:
        return sum([contour_item.area_ratio for contour_item in self.contours])

    @property
    def resolution(self) -> tuple:
        height = self.original_frame.shape[0]
        width = self.original_frame.shape[1]
        return tuple((width, height))
