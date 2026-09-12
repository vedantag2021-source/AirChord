"""
Phase 3: Finger State Detection
------------------------------
Goal: Determine which fingers are extended (up) vs curled (down),
based on hand landmark positions, and display the result live.

Requires: assets/models/hand_landmarker.task
Press 'q' to quit the window.
"""

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

MODEL_PATH = "assets/models/hand_landmarker.task"

# Landmark index reference:
# 0 = wrist
# Thumb: 1,2,3,4 (4 = tip)
# Index: 5,6,7,8 (8 = tip)
# Middle: 9,10,11,12 (12 = tip)
# Ring: 13,14,15,16 (16 = tip)
# Pinky: 17,18,19,20 (20 = tip)

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]  # the joint just below each tip
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


def get_finger_states(hand_landmarks, handedness_label):
    """
    Returns a list of 5 values (1 = extended, 0 = curled),
    in order: [Thumb, Index, Middle, Ring, Pinky]
    """
    states = []

    # --- Thumb (special case: moves sideways, not up/down) ---
    # Compare X coordinates instead of Y.
    # For a right hand shown mirrored (as our flipped webcam feed is),
    # an extended thumb's tip X will be further from the palm than the joint.
    thumb_tip_x = hand_landmarks[FINGER_TIPS[0]].x
    thumb_pip_x = hand_landmarks[FINGER_PIPS[0]].x

    if handedness_label == "Right":
        states.append(1 if thumb_tip_x > thumb_pip_x else 0)
    else:
        states.append(1 if thumb_tip_x < thumb_pip_x else 0)

    # --- Other four fingers (compare Y coordinates) ---
    # In image coordinates, Y increases downward.
    # So an extended finger's tip Y will be SMALLER (higher up) than its PIP joint Y.
    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_y = hand_landmarks[tip_idx].y
        pip_y = hand_landmarks[pip_idx].y
        states.append(1 if tip_y < pip_y else 0)

    return states


def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)


def draw_finger_states(frame, states):
    """Display finger up/down status as text in the top-left corner."""
    y_offset = 30
    for name, state in zip(FINGER_NAMES, states):
        text = f"{name}: {'UP' if state == 1 else 'DOWN'}"
        color = (0, 255, 0) if state == 1 else (0, 0, 255)
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)
        y_offset += 30

    total_up = sum(states)
    cv2.putText(frame, f"Fingers up: {total_up}", (10, y_offset + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Finger state detection running. Press 'q' to quit.")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)

            # handedness tells us if MediaPipe thinks this is a "Left" or "Right" hand
            # (note: since our video is mirrored, this label actually matches
            # what the viewer visually perceives correctly)
            handedness_label = result.handedness[0][0].category_name

            states = get_finger_states(result.hand_landmarks[0], handedness_label)
            draw_finger_states(frame, states)

        cv2.imshow("AirChord - Finger State Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()