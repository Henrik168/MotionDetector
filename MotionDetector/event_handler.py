class BasicHandler:
    def __init__(self):
        self.subscribers = []

    def fire_event(self, *args) -> None:
        for fn in self.subscribers:
            fn(*args)

    def subscribe(self, fn) -> None:
        self.subscribers.append(fn)


