import logging
import cv2
from numpy import ndarray
from typing import List, Tuple

from MotionDetector.item_contours import ContourItem
from MotionDetector.item_image import ImageItem
from CustomLogger import getLogger


class ContourProcessor:
    def __init__(self,
                 min_area_ratio: float = 2.0,
                 max_area_ratio: float = 35.0,
                 scale: float = 0.3,
                 logger: logging.Logger = None):
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.scale = scale
        self.logger = logger if logger else getLogger()
        self.frame_area: int = 0

    def compute_area_ratio(self, contour: ndarray) -> bool:
        area_ratio = round((cv2.contourArea(contour) / self.frame_area) * 100, 1)
        self.logger.debug(f"Area ratio: '{area_ratio}'")
        return self.min_area_ratio < area_ratio < self.max_area_ratio

    def filter_contours(self, contours: Tuple[ndarray]) -> List[ContourItem]:
        return [ContourItem(contour, self.frame_area, self.scale)
                for contour in contours if self.compute_area_ratio(contour)]

    def find_contours(self, image_item: ImageItem) -> ImageItem:
        frame = image_item.dilated_frame
        self.frame_area = frame.shape[0] * frame.shape[1]
        contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image_item.contours = self.filter_contours(contours)
        return image_item
