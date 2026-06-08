import math
from typing import Tuple

Point = Tuple[float, float]


def midpoint(point_a: Point, point_b: Point) -> Point:
    return (int((point_a[0] + point_b[0]) / 2), int((point_a[1] + point_b[1]) / 2))


def angle_between_points(origin: Point, target: Point) -> float:
    dx = target[0] - origin[0]
    dy = target[1] - origin[1]
    return math.degrees(math.atan2(dy, dx))


def signed_angle_difference(angle: float, reference: float) -> float:
    raw_delta = (angle - reference + 180.0) % 360.0 - 180.0
    return raw_delta


def normalized_vertical_displacement(nose: Point, shoulder_center: Point, image_height: int) -> float:
    if image_height == 0:
        return 0.0
    return (nose[1] - shoulder_center[1]) / float(image_height)
