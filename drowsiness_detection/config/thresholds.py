from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    baseline_seconds: float = 5.0
    delta_angle_threshold: float = 8.0
    tilt_duration_seconds: float = 1.5
    nodding_window_seconds: float = 4.0
    nodding_min_cycles: int = 2
    nodding_min_drop: float = 0.02
    shoulder_imbalance_threshold: float = 0.03
    shoulder_stability_window_seconds: float = 4.0
    shoulder_stability_std_threshold: float = 0.012
    history_size: int = 120
    min_baseline_samples: int = 15
