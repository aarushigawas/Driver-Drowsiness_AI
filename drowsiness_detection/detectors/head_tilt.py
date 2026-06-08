import time
from collections import deque
from typing import Deque, Dict, Optional

from drowsiness_detection.config.thresholds import Thresholds
from drowsiness_detection.utils.math_utils import signed_angle_difference


class HeadTiltDetector:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.baseline_samples: Deque[float] = deque(maxlen=thresholds.history_size)
        self.baseline_angle: Optional[float] = None
        self.start_timestamp: Optional[float] = None

    def update(self, angle: float, timestamp: float) -> Dict[str, object]:
        if self.start_timestamp is None:
            self.start_timestamp = timestamp

        elapsed = timestamp - self.start_timestamp
        if self.baseline_angle is None:
            self.baseline_samples.append(angle)
            if elapsed >= self.thresholds.baseline_seconds and len(self.baseline_samples) >= self.thresholds.min_baseline_samples:
                self.baseline_angle = float(sum(self.baseline_samples) / len(self.baseline_samples))

            return {
                "baseline_ready": False,
                "baseline_angle": None,
                "delta_angle": 0.0,
                "abnormal": False,
            }

        delta_angle = signed_angle_difference(angle, self.baseline_angle)
        return {
            "baseline_ready": True,
            "baseline_angle": self.baseline_angle,
            "delta_angle": delta_angle,
            "abnormal": abs(delta_angle) >= self.thresholds.delta_angle_threshold,
        }
