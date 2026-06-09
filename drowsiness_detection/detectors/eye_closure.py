from collections import deque
from typing import Deque, Dict, Optional

from drowsiness_detection.config.thresholds import Thresholds
from drowsiness_detection.utils.timer import Stopwatch


class EyeClosureDetector:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.closure_timer = Stopwatch()
        self.is_closed = False

    def update(self, eye_open_ratio: Optional[float], timestamp: float) -> Dict[str, object]:
        if eye_open_ratio is None:
            return {
                "eyes_closed": False,
                "closure_duration": 0.0,
                "eye_ratio": None,
            }
        
        # Eyes are considered closed if opening ratio is very small
        currently_closed = eye_open_ratio < 0.1  # Threshold for closed eyes
        
        if currently_closed and not self.is_closed:
            # Eyes just closed
            self.closure_timer.start()
            self.is_closed = True
        elif not currently_closed and self.is_closed:
            # Eyes just opened
            self.closure_timer.stop()
            self.is_closed = False
        
        closure_duration = self.closure_timer.elapsed() if self.is_closed else 0.0
        
        # Sustained closure indicates drowsiness
        sustained_closed = closure_duration >= 2.0  # 2 seconds of closed eyes
        
        return {
            "eyes_closed": sustained_closed,
            "closure_duration": closure_duration,
            "eye_ratio": eye_open_ratio,
        }