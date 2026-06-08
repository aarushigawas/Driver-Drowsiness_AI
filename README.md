# Drowsiness Detection Project

## What this project does
This project detects driver drowsiness using real-time webcam input. It analyzes pose and face landmarks to identify:
- sustained head tilt
- nodding patterns
- shoulder instability
- yawning
- eye closure

The app then combines these signals to decide whether the person is drowsy and, when confirmed, sounds an alarm until the person returns to a non-drowsy state.

## How it works
1. **Pose tracking** with MediaPipe Pose Landmarker
   - detects nose and shoulder positions
   - computes head tilt and shoulder balance

2. **Face feature extraction** with MediaPipe Face Landmarker
   - estimates mouth opening for yawning detection
   - estimates eye opening for eye closure detection

3. **Behavior detection modules**
   - `HeadTiltDetector` establishes a baseline head angle over 5 seconds
   - `TiltDurationDetector` tracks how long tilt remains abnormal
   - `NoddingDetector` looks for repeated downward motion in the head position
   - `YawningDetector` checks mouth opening history for yawns
   - `EyeClosureDetector` checks whether eyes remain closed beyond a threshold
   - `ShoulderStabilityDetector` tracks shoulder height imbalance over time

4. **Decision engine**
   - only flags drowsiness after steady evidence builds for 5 seconds
   - confirms drowsiness when tilt is sustained and one or more supporting signals are present
   - keeps the alarm active until the person recovers

## Run instructions
```powershell
cd "C:\Users\Asus\Documents\SVNIT\2nd yr\4th sem\AI\AI PROJECT"
venv\Scripts\activate
python -m drowsiness_detection.main
```

## What the new line means
The display now shows whether the drowsy condition has been "steady for 5s" before it becomes confirmed. This helps distinguish momentary head motion from sustained drowsiness.

## Why this is not reinforcement learning (RL)
This system is a rule-based detection pipeline, not RL. Key differences:
- RL learns by trial and error from rewards/penalties.
- This project uses fixed thresholds, timers, and signal combinations.
- No agent is training itself from experience.
- It applies handcrafted rules to pose and face features rather than learning a policy.

## Files of interest
- `drowsiness_detection/main.py` — main webcam loop and display
- `drowsiness_detection/logic/decision.py` — drowsiness confirmation logic
- `drowsiness_detection/detectors/` — individual behavior detectors
- `drowsiness_detection/utils/landmarks.py` — landmark extraction helpers
- `drowsiness_detection/config/thresholds.py` — thresholds and timing settings
