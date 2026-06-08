from typing import Dict, Optional

from drowsiness_detection.config.thresholds import Thresholds


class TiltDurationDetector:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.abnormal_start: Optional[float] = None

    def update(self, delta_angle: float, timestamp: float) -> Dict[str, object]:
        abnormal = abs(delta_angle) >= self.thresholds.delta_angle_threshold
        if abnormal:
            if self.abnormal_start is None:
                self.abnormal_start = timestamp
            duration = timestamp - self.abnormal_start
        else:
            duration = 0.0
            self.abnormal_start = None

        return {
            "duration": duration,
            "sustained": duration >= self.thresholds.tilt_duration_seconds,
            "abnormal": abnormal,
        }
