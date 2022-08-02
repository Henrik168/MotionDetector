import threading
import queue
import time
import logging

log = logging.getLogger(__name__)


class BasicHandler:
    def __init__(self):
        self.subscribers = []

    def fire_event(self, *args) -> None:
        for fn in self.subscribers:
            log.debug(f"Fire Event: Function {fn} with args {args}")
            fn(*args)

    def subscribe(self, fn) -> None:
        log.debug(f"Add Function {fn} to subscribers")
        self.subscribers.append(fn)


class ThreadHandler(threading.Thread):
    def __init__(self):
        self.subscribers = []
        self.input_queue = queue.Queue()

        threading.Thread.__init__(self)
        self.daemon = True
        self.exception = None

    def fire_event(self, *args) -> None:
        log.debug(f"Put Event with args {args} to Queue.")
        self.input_queue.put(args)

    def subscribe(self, fn) -> None:
        log.debug(f"Add Function {fn} to subscribers")
        self.subscribers.append(fn)

    def run(self) -> None:
        log.debug("Start Eventhandler Thread")
        while True:
            if self.input_queue.empty():
                time.sleep(1)
                log.debug("No data in Queue.")
                continue
            args = self.input_queue.get()
            log.debug(f"Got {args} from Queue.")
            for fn in self.subscribers:
                fn(*args)
