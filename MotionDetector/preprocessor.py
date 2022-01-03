import logging
import cv2
from numpy import ndarray, array
from os.path import isfile

from MotionDetector.item_image import ImageItem
from CustomLogger import getLogger


def resize_frame(frame: ndarray, scale: float = 1.0) -> ndarray:
    height = frame.shape[0]
    width = frame.shape[1]
    size = (int(width * scale), int(height * scale))
    return cv2.resize(src=frame, dsize=size)


def get_kernel(morph_element: int, kernel_size: int) -> ndarray:
    return cv2.getStructuringElement(shape=morph_element,
                                     ksize=(2 * kernel_size + 1, 2 * kernel_size + 1),
                                     anchor=(-1, -1))


def morphological_transformation(frame: ndarray,
                                 morph_type: int,
                                 morph_element: int,
                                 kernel_size: int,
                                 iterations: int) -> ndarray:
    kernel = get_kernel(morph_element=morph_element, kernel_size=kernel_size)
    return cv2.morphologyEx(src=frame, op=morph_type, kernel=kernel, iterations=iterations)


class PreProcessor:
    def __init__(self,
                 scale: float = 1.0,
                 sub_threshold: int = 25,
                 mask_path: str = None,
                 logger: logging.Logger = None):
        self.scale = scale
        self.mask_path = mask_path
        self.mask = self._load_mask() if mask_path else array([])
        self.logger = logger if logger else getLogger()
        self.subtractor = cv2.createBackgroundSubtractorMOG2(varThreshold=sub_threshold, detectShadows=False)

    def _subtract_background(self, frame: ndarray) -> ndarray:
        return self.subtractor.apply(frame)

    def _load_mask(self) -> ndarray:
        if not isfile(self.mask_path):
            raise FileNotFoundError(f"Mask '{self.mask_path}' not found.")
        mask = cv2.imread(self.mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # threshType:  0=Binary, 1=Binary_inv, 2=Trunc, 3=ToZero, 4=ToZero_inv
        return cv2.threshold(src=mask, thresh=128, maxval=255, type=0)[1]

    def _mask_image(self, frame: ndarray) -> ndarray:
        if frame.shape != self.mask.shape:
            raise RuntimeError(f"Shape of mask '{self.mask.shape}' does not fit to frame shape '{frame.shape}'")
        return cv2.bitwise_and(frame, self.mask)

    def preprocess_image(self, frame: ndarray) -> ImageItem:
        resized_frame = resize_frame(frame, self.scale)
        binary_frame = self._subtract_background(resized_frame)
        masked_frame = self._mask_image(binary_frame) if self.mask.any() else binary_frame
        eroded_frame = morphological_transformation(frame=masked_frame, morph_type=cv2.MORPH_ERODE,
                                                    morph_element=cv2.MORPH_CROSS, kernel_size=1, iterations=3)
        dilated_frame = morphological_transformation(frame=eroded_frame, morph_type=cv2.MORPH_DILATE,
                                                     morph_element=cv2.MORPH_ELLIPSE, kernel_size=2, iterations=2)
        return ImageItem(has_data=True,
                         original_frame=frame,
                         resized_frame=resized_frame,
                         binary_frame=binary_frame,
                         masked_frame=masked_frame,
                         eroded_frame=eroded_frame,
                         dilated_frame=dilated_frame)
