from dataclasses import dataclass
from numpy import ndarray, array
import cv2


@dataclass
class ContourItem:
    contour: ndarray = array([])
    frame_area: float = 0.0
    scale: float = 0.0

    @property
    def contour_area(self):
        return cv2.contourArea(self.contour)

    @property
    def area_ratio(self):
        return round((self.contour_area / self.frame_area) * 100, 1)

    @property
    def roi(self):
        args = cv2.boundingRect(self.contour)
        x1, y1, w, h = [int(arg / self.scale) for arg in args]
        return (x1, y1), (x1 + w, y1 + h)

    @property
    def scaled_roi(self):
        x1, y1, w, h = cv2.boundingRect(self.contour)
        return (x1, y1), (x1 + w, y1 + h)
