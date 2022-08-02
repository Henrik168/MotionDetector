import logging
import queue
import threading

import Eventhandler
from MotionDetector.buffer_frame import FrameBuffer
from MotionDetector.item_image import ImageItem
from MotionDetector.capture import StreamReader
from MotionDetector.preprocessor import PreProcessor
from MotionDetector.contours import ContourProcessor
from MotionDetector.buffer_motion import MotionBuffer, MotionFlag
from src.lib_path import get_path

log = logging.getLogger(__name__)


class MotionDetector(threading.Thread):
    def __init__(self,
                 url: str,
                 min_area_ratio: float = 2.0,
                 max_area_ratio: float = 10.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.exception = None

        self.url = url
        self.frame_buffer = FrameBuffer(buffer_size=30)
        self.stream_reader = StreamReader(url=url,
                                          buffer=self.frame_buffer,
                                          max_fps=1.0)
        # ToDo: Define Threshold for motion detection
        self.preprocessor = PreProcessor(scale=0.3,
                                         sub_threshold=80,
                                         mask_path=get_path("./data/mask.png"))
        self.contour_processor = ContourProcessor(min_area_ratio=min_area_ratio,
                                                  max_area_ratio=max_area_ratio)
        self.motion_buffer = MotionBuffer(buffer_size=2)

        self.motion_start_handler = Eventhandler.ThreadHandler()
        self.motion_end_handler = Eventhandler.ThreadHandler()

        self.output_queue = queue.Queue()

    def __del__(self):
        self.stream_reader.disconnect()

    def get_frame(self):
        return self.stream_reader.get_frame()

    def motion_trigger(self) -> ImageItem:
        if self.output_queue.empty():
            return ImageItem()
        image_item = self.output_queue.get()
        self.output_queue.task_done()
        return image_item

    def run(self) -> None:
        # Start threaded Handler
        self.motion_start_handler.start()
        self.motion_end_handler.start()

        while True:
            try:
                frame = self.stream_reader.capture()
                image_item = self.preprocessor.preprocess_image(frame)
                self.contour_processor.find_contours(image_item)

                image_item.motion_status = self.motion_buffer.get_motion(image_item.has_contours)
                if image_item.motion_status == MotionFlag.MotionStart:
                    image_item.frames = self.frame_buffer.get_frames()
                    self.motion_start_handler.fire_event(image_item)

                elif image_item.motion_status == MotionFlag.MotionEnd:
                    image_item.frames = self.frame_buffer.get_frames()
                    self.motion_end_handler.fire_event(image_item)

            except KeyboardInterrupt as e:
                self.exception = e
                log.exception(e)
                break
            except Exception as e:
                self.exception = e
                log.exception(e)
                break

    def get_exception(self) -> Exception:
        return self.exception
