"""
Phase 7 - Step 1: Gesture Data Collection
------------------------------------------
Goal: Record labeled hand-landmark samples for training a gesture classifier.

How to use:
- Hold your hand in a gesture pose.
- Press the number key matching that gesture (see LABELS below) to save
  one sample of the CURRENT hand shape with that label.
- Repeat many times per gesture, varying angle/distance/hand slightly
  each time, for a more robust dataset (aim for 50-100+ samples per gesture).
- Press 's' to save all collected data to a CSV file.
- Press 'q' to quit without saving further.

Requires: assets/models/hand_landmarker.task
"""

import cv2
import time
import csv
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "assets/models/hand_landmarker.task"
OUTPUT_CSV = "data/gesture_data.csv"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# Map keyboard keys to gesture labels. Feel free to add more gestures here --
# this is one advantage of the ML approach over rigid rule-based logic.
LABELS = {
    ord('1'): "G",
    ord('2'): "C",
    ord('3'): "D",
    ord('4'): "Em",
    ord('5'): "A",
}


def extract_features(hand_landmarks):
    """
    Converts 21 raw (x, y, z) landmarks into a normalized feature vector.

    Normalization steps:
    1. Subtract the wrist position from every landmark, so the features
       describe hand SHAPE regardless of where the hand is in the frame.
    2. Divide by hand size (wrist-to-middle-knuckle distance), so the
       features are the same whether the hand is close to or far from
       the camera.

    Returns a flat list of 63 numbers (21 landmarks x 3 coordinates).
    """
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_size = ((wrist.x - middle_mcp.x) ** 2 +
                 (wrist.y - middle_mcp.y) ** 2) ** 0.5
    if hand_size == 0:
        hand_size = 1e-6  # avoid division by zero

    features = []
    for lm in hand_landmarks:
        features.append((lm.x - wrist.x) / hand_size)
        features.append((lm.y - wrist.y) / hand_size)
        features.append((lm.z - wrist.z) / hand_size)

    return features


def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)


def draw_instructions(frame, sample_counts):
    y = 25
    cv2.putText(frame, "Press 1-5 to label a sample | 's' to save | 'q' to quit",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y += 30
    for key_code, label in LABELS.items():
        key_char = chr(key_code)
        count = sample_counts.get(label, 0)
        cv2.putText(frame, f"[{key_char}] {label}: {count} samples",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 25


def save_to_csv(rows):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_CSV)

    with open(OUTPUT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Header: 63 feature columns + 1 label column
            header = [f"f{i}" for i in range(63)] + ["label"]
            writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved {len(rows)} samples to {OUTPUT_CSV}")


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

    print("Data collection running.")
    start_time = time.time()

    collected_rows = []
    sample_counts = {label: 0 for label in LABELS.values()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - start_time) * 1000)

        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        current_landmarks = None
        if result.hand_landmarks:
            draw_landmarks(frame, result.hand_landmarks)
            current_landmarks = result.hand_landmarks[0]

        draw_instructions(frame, sample_counts)
        cv2.imshow("AirChord - Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # --- Debug: log every keypress detected, regardless of what it does ---
        if key != 255:
            print(f"[DEBUG] Raw key detected: code={key}, char={chr(key) if 32 <= key <= 126 else 'N/A'}")

        if key == ord('q'):
            break
        elif key == ord('s'):
            if collected_rows:
                save_to_csv(collected_rows)
                collected_rows = []
        elif key in LABELS:
            if current_landmarks is not None:
                label = LABELS[key]
                features = extract_features(current_landmarks)
                collected_rows.append(features + [label])
                sample_counts[label] += 1
                print(f"Captured sample for '{label}' (total this session: {sample_counts[label]})")
            else:
                print("No hand detected -- sample not captured.")

    # Save any remaining unsaved samples on exit
    if collected_rows:
        save_to_csv(collected_rows)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()