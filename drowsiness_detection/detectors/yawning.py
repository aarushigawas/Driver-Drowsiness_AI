from collections import deque
from typing import Deque, Dict, Optional, Tuple
import math

from drowsiness_detection.config.thresholds import Thresholds


class YawningDetector:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.history: Deque[tuple[float, float]] = deque(maxlen=120)  # 4 seconds at 30fps

    def update(self, mouth_open_ratio: Optional[float], timestamp: float) -> Dict[str, object]:
        if mouth_open_ratio is not None:
            self.history.append((timestamp, mouth_open_ratio))
        
        # Keep only recent history
        while self.history and timestamp - self.history[0][0] > 4.0:
            self.history.popleft()

        yawning = self._detect_yawning()
        return {
            "yawning": yawning,
            "mouth_ratio": mouth_open_ratio,
        }

    def _detect_yawning(self) -> bool:
        if len(self.history) < 10:  # Need some data
            return False

        recent_ratios = [ratio for _, ratio in self.history]
        max_ratio = max(recent_ratios)
        avg_ratio = sum(recent_ratios) / len(recent_ratios)

        return (
            max_ratio > self.thresholds.yawn_peak_threshold
            and avg_ratio > self.thresholds.yawn_avg_threshold
        )