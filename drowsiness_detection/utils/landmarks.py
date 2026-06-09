from collections.abc import Sequence
from typing import Dict, Optional, Tuple

NOSE_INDEX = 0
LEFT_SHOULDER_INDEX = 11
RIGHT_SHOULDER_INDEX = 12


def _extract_landmarks(landmark_data):
    if landmark_data is None:
        return None

    if hasattr(landmark_data, "landmark"):
        return list(landmark_data.landmark)

    if isinstance(landmark_data, Sequence):
        if len(landmark_data) == 0:
            return None

        first_item = landmark_data[0]
        if hasattr(first_item, "x"):
            return list(landmark_data)

        if hasattr(first_item, "landmark"):
            return list(first_item.landmark)

        if isinstance(first_item, Sequence):
            return list(first_item)

    return None


def _to_pixel(point, width: int, height: int) -> Tuple[int, int]:
    return int(point.x * width), int(point.y * height)


def extract_pose_points(result, image_width: int, image_height: int) -> Optional[Dict[str, Tuple[int, int]]]:
    landmarks = getattr(result, "pose_landmarks", None)
    landmarks = _extract_landmarks(landmarks)
    if not landmarks or len(landmarks) <= RIGHT_SHOULDER_INDEX:
        return None

    try:
        return {
            "nose": _to_pixel(landmarks[NOSE_INDEX], image_width, image_height),
            "left_shoulder": _to_pixel(landmarks[LEFT_SHOULDER_INDEX], image_width, image_height),
            "right_shoulder": _to_pixel(landmarks[RIGHT_SHOULDER_INDEX], image_width, image_height),
        }
    except AttributeError:
        return None


def extract_face_features(result, image_width: int, image_height: int) -> Optional[Dict[str, float]]:
    face_landmarks = getattr(result, "face_landmarks", None)
    if not face_landmarks or len(face_landmarks) == 0:
        return None
    
    # For Tasks API, face_landmarks is list of FaceLandmark
    if hasattr(face_landmarks[0], "landmark"):
        landmarks = face_landmarks[0].landmark
    else:
        landmarks = face_landmarks[0]
    
    if len(landmarks) < 400:  # Need enough landmarks
        return None
    
    try:
        # Mouth opening: distance between upper lip (landmark 13) and lower lip (landmark 14)
        upper_lip = _to_pixel(landmarks[13], image_width, image_height)
        lower_lip = _to_pixel(landmarks[14], image_width, image_height)
        mouth_open = abs(upper_lip[1] - lower_lip[1]) / image_height  # normalized
        
        # Left eye opening: distance between upper (159) and lower (145) eyelid
        left_upper = _to_pixel(landmarks[159], image_width, image_height)
        left_lower = _to_pixel(landmarks[145], image_width, image_height)
        left_eye_open = abs(left_upper[1] - left_lower[1]) / image_height
        
        # Right eye opening: distance between upper (386) and lower (374) eyelid  
        right_upper = _to_pixel(landmarks[386], image_width, image_height)
        right_lower = _to_pixel(landmarks[374], image_width, image_height)
        right_eye_open = abs(right_upper[1] - right_lower[1]) / image_height
        
        # Average eye opening
        eye_open_ratio = (left_eye_open + right_eye_open) / 2.0
        
        return {
            "mouth_open_ratio": mouth_open,
            "eye_open_ratio": eye_open_ratio,
        }
    except (IndexError, AttributeError):
        return None
