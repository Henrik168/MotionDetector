from dataclasses import dataclass, field
from enum import Enum, auto


class MotionFlag(Enum):
    MotionStart = auto()
    MotionEnd = auto()
    MotionOngoing = auto()
    NoMotion = auto()


@dataclass
class MotionBuffer:
    buffer_size: int
    motion_flag: bool = False
    buffer: list = field(default_factory=list)

    def __post_init__(self):
        if self.buffer_size < 1:
            raise ValueError("Buffer size must be 1 or greater.")

    def buffer_full(self) -> bool:
        return len(self.buffer) == self.buffer_size

    def add_motion(self, value: bool) -> None:
        if self.buffer_full():
            self.buffer.pop(0)
        self.buffer.append(value)

    def get_motion(self, value: bool) -> MotionFlag:
        self.add_motion(value)
        if all(self.buffer) and not self.motion_flag:
            self.motion_flag = True
            return MotionFlag.MotionStart
        elif not any(self.buffer) and self.motion_flag:
            self.motion_flag = False
            return MotionFlag.MotionEnd
        elif any(self.buffer) and self.motion_flag:
            return MotionFlag.MotionOngoing
        else:
            return MotionFlag.NoMotion
