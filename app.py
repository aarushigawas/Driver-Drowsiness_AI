import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import os
import urllib.request

# Download model if not present
model_path = 'pose_landmarker.task'
model_url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

if not os.path.exists(model_path):
    st.info("Downloading model... Please wait.")
    urllib.request.urlretrieve(model_url, model_path)
    st.success("Model downloaded!")

# Setup MediaPipe
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

st.title("Pose Detection with MediaPipe")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                st.success(f"Pose detected! Found {len(result.pose_landmarks)} poses.")

                # Draw landmarks
                annotated_image = image.copy()
                for pose in result.pose_landmarks:
                    for landmark in pose:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

                st.image(annotated_image, caption='Pose Landmarks', use_column_width=True)
            else:
                st.warning("No pose detected in the image.")
    except Exception as e:
        st.error(f"Error in pose detection: {e}")
else:
    st.info("Upload an image to detect poses.")