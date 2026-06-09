from typing import Dict
import time

from drowsiness_detection.config.thresholds import Thresholds


class DrowsinessDecision:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.last_drowsy_time = 0.0

    def evaluate(
        self,
        delta_angle: float,
        tilt_sustained: bool,
        nodding_detected: bool,
        shoulder_unstable: bool,
        yawning_detected: bool = False,
        eyes_closed: bool = False,
    ) -> Dict[str, object]:
        reasons = []
        if abs(delta_angle) >= self.thresholds.delta_angle_threshold:
            reasons.append(f"Tilt deviation Δθ={delta_angle:.1f}°")

        if tilt_sustained:
            reasons.append("Tilt has been abnormal for a sustained period")

        if nodding_detected:
            reasons.append("Nodding pattern detected")

        if shoulder_unstable:
            reasons.append("Shoulder posture is unstable")

        if yawning_detected:
            reasons.append("Yawning detected")

        if eyes_closed:
            reasons.append("Eyes closed for extended period")

        drowsy = tilt_sustained and (nodding_detected or shoulder_unstable or yawning_detected or eyes_closed)
        if drowsy:
            self.last_drowsy_time = time.time()
        
        persistent_drowsy = drowsy or (time.time() - self.last_drowsy_time < 2.0)
        
        if not persistent_drowsy:
            if tilt_sustained:
                reasons.append("Alert: supporting motion not strong enough")
            else:
                reasons.append("Alert: no sustained abnormal posture")

        return {
            "drowsy": persistent_drowsy,
            "reasons": reasons,
        }
