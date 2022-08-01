import time
from time import sleep
from cv2 import VideoCapture
from numpy import ndarray
import logging

from MotionDetector.buffer_frame import FrameBuffer

log = logging.getLogger(__name__)


class StreamReader:
    def __init__(self,
                 url: str,
                 buffer: FrameBuffer,
                 max_fps: float = 0.5):
        self.url = url
        self.buffer = buffer
        self.max_fps = max_fps
        self.last_capture = time.time()

        self.cap = VideoCapture()

        self.connect()

    def connect(self) -> None:
        self.cap.open(self.url)
        if not self.cap.isOpened():
            self.reconnect()
        self.capture()

    def reconnect(self, max_sec: int = 1024) -> None:
        sec_wait = 1
        log.warning(f"Cannot connect to Videostream.")
        while True:
            self.cap.release()
            self.cap.open(self.url)
            if self.cap.isOpened():
                log.warning(f"Reconnection successful! to stream: {self.url}")
                break

            log.warning(f"Waiting for {sec_wait} seconds to reconnect.")
            sleep(sec_wait)
            sec_wait = min(max_sec, sec_wait * 2)  # limit waiting Time to max_sec
            if sec_wait == max_sec:
                raise ConnectionError(f"Unable to reconnect to {self.url}")

    def disconnect(self) -> None:
        self.cap.release()

    def get_frame(self):
        return self.buffer[-1]

    def wait(self) -> None:
        secs = max((self.last_capture + 1 / self.max_fps - time.time()), 0)
        sleep(secs)
        self.last_capture = time.time()

    def capture(self) -> ndarray:
        self.wait()
        ret, frame = self.cap.read()
        if not ret:
            self.reconnect()
            ret, frame = self.cap.read()
        self.buffer.add_frame(frame.copy())
        return frame
