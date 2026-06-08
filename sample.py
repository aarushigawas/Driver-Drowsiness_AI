import cv2
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            # Tip of index finger = 8
            # Base of index finger = 6
            if landmarks[8].y < landmarks[6].y:
                finger_count = 1
            else:
                finger_count = 0

            # Draw hand
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display result
    cv2.putText(frame, f"Output: {finger_count}",
                (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3)

    cv2.imshow("Finger Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()