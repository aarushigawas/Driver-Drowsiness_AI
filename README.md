# Drowsiness Detection Using AI-Based Motion Analysis

## Overview

This project is a real-time drowsiness detection system that uses computer vision and human motion analysis to identify signs of driver fatigue. Instead of relying on eye-closure detection, the system analyzes head posture, movement patterns, and body stability using MediaPipe pose landmarks extracted from a webcam feed.

The objective is to detect behavioral indicators of drowsiness such as prolonged head tilting, repeated nodding motions, and unstable posture. Multiple signals are combined before making a drowsiness decision, reducing false positives and improving reliability.

---

## Features

- Real-time webcam-based monitoring
- AI-powered pose landmark extraction using MediaPipe
- Dynamic baseline calibration for individual users
- Head tilt angle (θ) analysis
- Vertical head movement (φ) analysis
- Tilt duration tracking
- Nodding pattern detection
- Shoulder stability analysis
- Multi-condition drowsiness decision logic
- Modular and extensible architecture

---

## Project Approach

Traditional systems often classify a person as drowsy when a fixed head angle threshold is exceeded. However, head posture naturally varies between individuals.

This project instead follows a behavioral approach:

1. Learn the user's normal posture.
2. Measure deviations from that posture.
3. Track how long abnormal posture persists.
4. Detect repeated movement patterns.
5. Combine multiple indicators before classifying drowsiness.

This makes the system adaptive rather than dependent on fixed thresholds.

---

## Motion Analysis Parameters

### 1. Head Tilt Angle (θ)

The angle between the nose position and the midpoint of the shoulders is calculated.

Instead of using an absolute angle threshold, the system computes:

Δθ = θ − θ₀

Where:

- θ = Current head angle
- θ₀ = Baseline head angle measured during calibration

This allows the system to adapt to different natural postures.

---

### 2. Head Movement (φ)

Vertical head movement is tracked relative to the shoulders.

This value helps detect:

- Head dropping forward
- Micro-sleep behavior
- Repeated nodding patterns

---

### 3. Tilt Duration

The system tracks how long the head remains in an abnormal position.

Short movements are ignored.

Sustained deviations are treated as stronger drowsiness indicators.

---

### 4. Nodding Detection

Recent head movements are stored in a history buffer.

The system identifies repeated:

- Downward movement
- Recovery movement
- Downward movement again

This oscillatory behavior is a common sign of fatigue.

---

### 5. Shoulder Stability

The vertical positions of the left and right shoulders are monitored.

Indicators include:

- Uneven shoulders
- Drooping posture
- Excessive instability

These signals support the final decision process.

---

## Final Decision Logic

The driver is classified as drowsy only when multiple conditions are satisfied.

Example logic:

DROWSY if:

- Significant head deviation
- Sustained abnormal posture
- Nodding pattern OR shoulder instability

This reduces false alarms caused by normal movement.

---

## Project Structure
