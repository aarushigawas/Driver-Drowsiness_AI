from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from drowsiness_detection.config.thresholds import Thresholds


class ShoulderStabilityDetector:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.history: Deque[tuple[float, float]] = deque()

    def update(self, left_shoulder_y: float, right_shoulder_y: float, timestamp: float) -> Dict[str, object]:
        if left_shoulder_y is None or right_shoulder_y is None:
            return {"imbalance": 0.0, "std": 0.0, "unstable": False}

        imbalance = abs(left_shoulder_y - right_shoulder_y)
        self.history.append((timestamp, imbalance))
        while self.history and timestamp - self.history[0][0] > self.thresholds.shoulder_stability_window_seconds:
            self.history.popleft()

        values = [value for _, value in self.history]
        std = float(np.std(values)) if values else 0.0
        unstable = (
            imbalance >= self.thresholds.shoulder_imbalance_threshold
            or std >= self.thresholds.shoulder_stability_std_threshold
        )

        return {
            "imbalance": imbalance,
            "std": std,
            "unstable": unstable,
        }
