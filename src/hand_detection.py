"""
Phase 2: Hand Landmark Detection (MediaPipe Tasks API)
--------------------------------------------------------
Goal: Detect a hand in the webcam feed and draw its 21 landmarks in real time.
Uses the modern MediaPipe Tasks API (mediapipe >= 1.0.0), which replaced
the older mp.solutions.hands API.

Requires: assets/models/hand_landmarker.task
Press 'q' to quit the window.
"""

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# The 21 hand landmarks are connected in a fixed skeleton pattern.
# This list defines which landmark indices connect to which, so we can
# draw the "skeleton" lines ourselves (the new API doesn't include a
# built-in drawing helper like the old mp.solutions.drawing_utils did).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),     # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),   # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),   # Pinky
    (5, 9), (9, 13), (13, 17)                # Palm connections
]

MODEL_PATH = "assets/models/hand_landmarker.task"


def draw_landmarks(frame, hand_landmarks_list):
    """Manually draw landmark dots and connecting lines on the frame."""
    h, w, _ = frame.shape

    for hand_landmarks in hand_landmarks_list:
        # Convert normalized (0-1) coordinates to actual pixel coordinates
        points = []
        for lm in hand_landmarks:
            x_px = int(lm.x * w)
            y_px = int(lm.y * h)
            points.append((x_px, y_px))

        # Draw connecting lines first (so dots appear on top)
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        # Draw a dot for each landmark
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)


def main():
    # --- Set up the HandLandmarker with the downloaded model ---
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

    print("Hand detection running. Press 'q' to quit.")

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        # Convert BGR (OpenCV) to RGB, then wrap it as a MediaPipe Image object
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # VIDEO mode requires a timestamp (in milliseconds) for each frame,
        # so the model can track motion consistently between frames.
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # result.hand_landmarks is a list of hands, each a list of 21 landmarks
        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)

        cv2.imshow("AirChord - Hand Landmark Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()