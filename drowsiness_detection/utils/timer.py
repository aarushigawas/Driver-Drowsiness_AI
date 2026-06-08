import time


class Stopwatch:
    def __init__(self) -> None:
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.time()

    def reset(self) -> None:
        self._start_time = None

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def is_running(self) -> bool:
        return self._start_time is not None
