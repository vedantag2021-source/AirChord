"""
Phase 4: Gesture to Chord Mapping
------------------------------
Goal: Map a finger-state pattern to a specific guitar chord name.

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

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# --- Our gesture vocabulary ---
# Each key is a tuple of 5 values (Thumb, Index, Middle, Ring, Pinky)
# 1 = extended, 0 = curled. Each maps to a chord name.
GESTURE_TO_CHORD = {
    (0, 0, 0, 0, 0): "G",    # Fist
    (0, 1, 0, 0, 0): "C",    # Index only
    (0, 1, 1, 0, 0): "D",    # Index + Middle
    (0, 1, 1, 1, 0): "Em",   # Index + Middle + Ring
    (1, 1, 1, 1, 1): "A",    # Open palm
}


def get_finger_states(hand_landmarks, handedness_label):
    states = []

    thumb_tip_x = hand_landmarks[FINGER_TIPS[0]].x
    thumb_pip_x = hand_landmarks[FINGER_PIPS[0]].x

    if handedness_label == "Right":
        states.append(1 if thumb_tip_x > thumb_pip_x else 0)
    else:
        states.append(1 if thumb_tip_x < thumb_pip_x else 0)

    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_y = hand_landmarks[tip_idx].y
        pip_y = hand_landmarks[pip_idx].y
        states.append(1 if tip_y < pip_y else 0)

    return states


def map_gesture_to_chord(finger_states):
    """
    Takes a finger-state list (e.g. [0,1,1,0,0]) and returns the matching
    chord name, or None if the pattern isn't in our vocabulary.
    """
    key = tuple(finger_states)
    return GESTURE_TO_CHORD.get(key, None)


def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)


def draw_chord_display(frame, chord_name):
    """Show the currently detected chord in large text at the top."""
    display_text = chord_name if chord_name else "No Chord"
    color = (0, 255, 0) if chord_name else (0, 0, 255)
    cv2.putText(frame, display_text, (frame.shape[1] // 2 - 80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)


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

    print("Gesture-to-chord mapping running. Press 'q' to quit.")
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

        chord_name = None

        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)
            handedness_label = result.handedness[0][0].category_name
            states = get_finger_states(result.hand_landmarks[0], handedness_label)
            chord_name = map_gesture_to_chord(states)

        draw_chord_display(frame, chord_name)

        cv2.imshow("AirChord - Gesture to Chord", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()