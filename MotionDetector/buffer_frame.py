from dataclasses import dataclass, field


@dataclass
class FrameBuffer:
    buffer_size: int
    buffer: list = field(default_factory=list)

    def __post_init__(self):
        if self.buffer_size < 1:
            raise ValueError("Buffer size must be 1 or greater.")

    def __getitem__(self, item):
        return self.buffer[item]

    def get_frames(self):
        return self.buffer

    def buffer_full(self) -> bool:
        return len(self.buffer) == self.buffer_size

    def add_frame(self, frame) -> None:
        if self.buffer_full():
            self.buffer.pop(0)
        self.buffer.append(frame)
