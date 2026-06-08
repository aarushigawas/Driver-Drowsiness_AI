from collections import deque
from typing import Deque, Dict

from drowsiness_detection.config.thresholds import Thresholds


class NoddingDetector:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.history: Deque[tuple[float, float]] = deque()

    def update(self, phi: float, timestamp: float) -> Dict[str, object]:
        self.history.append((timestamp, phi))
        while self.history and timestamp - self.history[0][0] > self.thresholds.nodding_window_seconds:
            self.history.popleft()

        nodding = self._detect_nods()
        return {
            "nodding": nodding,
            "phi": phi,
        }

    def _detect_nods(self) -> bool:
        if len(self.history) < 5:
            return False

        values = [value for _, value in self.history]
        minima = 0
        for index in range(1, len(values) - 1):
            previous_value = values[index - 1]
            current_value = values[index]
            next_value = values[index + 1]
            if (
                current_value < previous_value
                and current_value < next_value
                and (previous_value - current_value) >= self.thresholds.nodding_min_drop
            ):
                minima += 1

        return minima >= self.thresholds.nodding_min_cycles
