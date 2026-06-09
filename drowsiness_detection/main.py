import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from drowsiness_detection.config.thresholds import Thresholds
from drowsiness_detection.detectors.eye_closure import EyeClosureDetector
from drowsiness_detection.detectors.head_tilt import HeadTiltDetector
from drowsiness_detection.detectors.nodding import NoddingDetector
from drowsiness_detection.detectors.shoulder_stability import ShoulderStabilityDetector
from drowsiness_detection.detectors.tilt_duration import TiltDurationDetector
from drowsiness_detection.detectors.yawning import YawningDetector
from drowsiness_detection.logic.decision import DrowsinessDecision
from drowsiness_detection.utils.landmarks import extract_face_features, extract_pose_points
from drowsiness_detection.utils.math_utils import (
    angle_between_points,
    midpoint,
    normalized_vertical_displacement,
    signed_angle_difference,
)

MODEL_NAME = "pose_landmarker_lite.task"
FACE_MODEL_NAME = "face_landmarker.task"


def create_mp_image(rgb_frame):
    try:
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    except Exception:
        return mp.Image.create_from_array(rgb_frame)


def get_model_path() -> Path:
    root = Path(__file__).resolve().parent
    model_dir = root / "models"
    expected_path = model_dir / MODEL_NAME

    if expected_path.exists():
        return expected_path

    candidates = list(model_dir.glob("pose_*task"))
    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple pose models found in {model_dir}: {[p.name for p in candidates]}. "
            f"Rename the correct model to {MODEL_NAME} or remove extra files."
        )

    raise FileNotFoundError(
        f"Pose landmarker model not found in {model_dir}. "
        f"Expected {MODEL_NAME}."
    )


def build_landmarker(model_path: str):
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE,
    )
    return PoseLandmarker.create_from_options(options)


def build_face_landmarker(model_path: str):
    base_options = BaseOptions(model_asset_path=model_path)
    FaceLandmarker = vision.FaceLandmarker
    FaceLandmarkerOptions = vision.FaceLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1
    )
    return FaceLandmarker.create_from_options(options)


def overlay_status(frame, status_lines):
    for index, line in enumerate(status_lines):
        y = 20 + index * 26
        color = (0, 255, 0)
        if "DROWSY" in line or "unstable" in line.lower():
            color = (0, 0, 255)
        elif "not detected" in line.lower() or "no pose" in line.lower() or "pose_landmarks" in line.lower():
            color = (0, 255, 255)
        elif index == 0:
            color = (0, 255, 0)
        else:
            color = (255, 255, 255)

        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def main():
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / MODEL_NAME
    face_model_path = models_dir / FACE_MODEL_NAME
    
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Pose landmarker model path issue: {model_path}. "
            "Check that the file exists, is readable, and is not empty."
        )
    
    if not face_model_path.exists() or face_model_path.stat().st_size == 0:
        print(f"Warning: Face landmarker model not found at {face_model_path}. Face detection features will be disabled.")
        face_model_path = None

    print(f"Using pose model: {model_path}")
    if face_model_path:
        print(f"Using face model: {face_model_path}")
    
    thresholds = Thresholds()
    tilt_detector = HeadTiltDetector(thresholds)
    tilt_duration_detector = TiltDurationDetector(thresholds)
    nodding_detector = NoddingDetector(thresholds)
    shoulder_detector = ShoulderStabilityDetector(thresholds)
    yawning_detector = YawningDetector(thresholds)
    eye_closure_detector = EyeClosureDetector(thresholds)
    decision_engine = DrowsinessDecision(thresholds)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions and connection.")

    with build_landmarker(str(model_path)) as landmarker:
        face_landmarker = None
        if face_model_path:
            face_landmarker = build_face_landmarker(str(face_model_path))
        
        try:
            print("Starting drowsiness detection. Press 'q' to quit.")

            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_height, image_width = frame.shape[:2]
                mp_image = create_mp_image(frame_rgb)

                result = landmarker.detect(mp_image)
                face_result = face_landmarker.detect(mp_image) if face_landmarker else None
                pose_landmarks = getattr(result, "pose_landmarks", None)
                pose_points = extract_pose_points(result, image_width, image_height)
                face_features = extract_face_features(face_result, image_width, image_height) if face_result else None
                status_lines = ["Drowsiness detection active"]
                if pose_landmarks is None:
                    status_lines.append("Pose not detected")
                    status_lines.append("No pose_landmarks returned by model")
                elif pose_points is None:
                    count = 0
                    pose_count = 0
                    landmark_count = 0
                    try:
                        pose_count = len(pose_landmarks)
                        if pose_count:
                            first_pose = pose_landmarks[0]
                            landmark_count = len(first_pose) if hasattr(first_pose, '__len__') else 0
                    except Exception:
                        try:
                            landmark_count = len(pose_landmarks.landmark)
                        except Exception:
                            landmark_count = 0
                    status_lines.append("Pose landmarks visible, extraction failed")
                    if pose_count:
                        status_lines.append(f"Detected poses: {pose_count}, landmarks in first pose: {landmark_count}")
                    else:
                        status_lines.append(f"Landmark count: {landmark_count}")
                else:
                    nose = pose_points["nose"]
                    left_shoulder = pose_points["left_shoulder"]
                    right_shoulder = pose_points["right_shoulder"]
                    shoulder_center = midpoint(left_shoulder, right_shoulder)
                    angle = angle_between_points(shoulder_center, nose)
                    delta_theta = signed_angle_difference(angle, tilt_detector.baseline_angle or angle)
                    phi = normalized_vertical_displacement(nose, shoulder_center, image_height)

                    tilt_output = tilt_detector.update(angle, time.time())
                    duration_output = tilt_duration_detector.update(tilt_output["delta_angle"], time.time())
                    nodding_output = nodding_detector.update(phi, time.time())
                    shoulder_output = shoulder_detector.update(
                        left_shoulder[1], right_shoulder[1], time.time()
                    )
                    yawning_output = yawning_detector.update(face_features.get("mouth_open_ratio") if face_features else None, time.time())
                    eye_output = eye_closure_detector.update(face_features.get("eye_open_ratio") if face_features else None, time.time())
                    decision = decision_engine.evaluate(
                        tilt_output["delta_angle"],
                        duration_output["sustained"],
                        nodding_output["nodding"],
                        shoulder_output["unstable"],
                        yawning_detected=yawning_output["yawning"],
                        eyes_closed=eye_output["eyes_closed"],
                    )

                    status_lines.append(
                        f"Baseline ready: {tilt_output['baseline_ready']} | "
                        f"Δθ={tilt_output['delta_angle']:.1f}°"
                    )
                    status_lines.append(
                        f"Sustained tilt: {duration_output['duration']:.1f}s / "
                        f"{thresholds.tilt_duration_seconds:.1f}s"
                    )
                    status_lines.append(
                        f"Nodding signal: {nodding_output['nodding']} | "
                        f"φ={phi:.3f}"
                    )
                    status_lines.append(
                        f"Shoulder unstable: {shoulder_output['unstable']} | "
                        f"imbalance={shoulder_output['imbalance']:.3f}"
                    )
                    status_lines.append(
                        f"Yawning: {yawning_output['yawning']} | "
                        f"mouth={yawning_output['mouth_ratio']:.3f}" if yawning_output['mouth_ratio'] is not None else "Yawning: No face detected"
                    )
                    status_lines.append(
                        f"Eyes closed: {eye_output['eyes_closed']} | "
                        f"duration={eye_output['closure_duration']:.1f}s | "
                        f"ratio={eye_output['eye_ratio']:.3f}" if eye_output['eye_ratio'] is not None else "Eyes: No face detected"
                    )
                    status_lines.append(
                        "DROWSY" if decision["drowsy"] else "Alert"
                    )
                    status_lines.extend(decision["reasons"])

                    cv2.circle(frame, nose, 10, (0, 255, 0), -1)
                    cv2.circle(frame, left_shoulder, 8, (255, 0, 0), -1)
                    cv2.circle(frame, right_shoulder, 8, (255, 0, 0), -1)
                    cv2.line(frame, left_shoulder, right_shoulder, (255, 255, 0), 3)
                    cv2.line(frame, shoulder_center, nose, (0, 255, 255), 3)
                    cv2.putText(frame, 'N', (nose[0] + 10, nose[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, 'L', (left_shoulder[0] + 10, left_shoulder[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, 'R', (right_shoulder[0] + 10, right_shoulder[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                overlay_status(frame, status_lines)
                cv2.imshow("Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            if face_landmarker:
                face_landmarker.close()


if __name__ == "__main__":
    main()
