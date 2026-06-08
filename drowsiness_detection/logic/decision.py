from typing import Dict
import time

from drowsiness_detection.config.thresholds import Thresholds


class DrowsinessDecision:
    def __init__(self, thresholds: Thresholds):
        self.thresholds = thresholds
        self.last_drowsy_time = 0.0
        self.confirm_start_time: float | None = None
        self.recovery_start_time: float | None = None

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
            reasons.append("Tilt sustained")

        if nodding_detected:
            reasons.append("Nodding pattern detected")

        if shoulder_unstable:
            reasons.append("Shoulder posture is unstable")

        if yawning_detected:
            reasons.append("Yawning detected")

        if eyes_closed:
            reasons.append("Eyes closed for extended period")

        drowsy_candidate = tilt_sustained and (nodding_detected or yawning_detected or eyes_closed)
        current_time = time.time()

        if drowsy_candidate:
            if self.confirm_start_time is None:
                self.confirm_start_time = current_time
            confirmation_duration = current_time - self.confirm_start_time
        else:
            confirmation_duration = 0.0
            self.confirm_start_time = None

        confirmed_drowsy = drowsy_candidate and confirmation_duration >= self.thresholds.drowsiness_confirmation_seconds
        if confirmed_drowsy:
            self.last_drowsy_time = current_time
            self.recovery_start_time = None

        if not confirmed_drowsy and self.last_drowsy_time > 0:
            if self.recovery_start_time is None:
                self.recovery_start_time = current_time
            recovery_duration = current_time - self.recovery_start_time
        else:
            recovery_duration = 0.0

        persistent_drowsy = confirmed_drowsy or (current_time - self.last_drowsy_time < self.thresholds.drowsy_release_seconds)

        if not persistent_drowsy:
            if drowsy_candidate:
                reasons.append("Alert: waiting for steady confirmation")
            else:
                reasons.append("Alert: no confirmed drowsy behavior")

        return {
            "drowsy": persistent_drowsy,
            "confirmed": confirmed_drowsy,
            "confirmation_duration": confirmation_duration,
            "recovery_duration": recovery_duration,
            "reasons": reasons,
        }
