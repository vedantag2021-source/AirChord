"""
Phase 5: Audio Playback
------------------------------
Goal: Play the correct chord sample when a gesture is recognized,
only triggering playback when the detected chord actually CHANGES
(not every single frame).

Requires:
- assets/models/hand_landmarker.task
- assets/audio/G.wav, C.wav, D.wav, Em.wav, A.wav

Press 'q' to quit the window.
"""

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pygame

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

MODEL_PATH = "assets/models/hand_landmarker.task"
AUDIO_FOLDER = "assets/audio"

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]

GESTURE_TO_CHORD = {
    (0, 0, 0, 0, 0): "G",
    (0, 1, 0, 0, 0): "C",
    (0, 1, 1, 0, 0): "D",
    (0, 1, 1, 1, 0): "Em",
    (1, 1, 1, 1, 1): "A",
}


def load_chord_sounds():
    """
    Loads each chord's .wav file into a pygame Sound object once at startup,
    so we're not reading from disk every frame (which would cause lag).
    Returns a dict like {"G": <Sound object>, "C": <Sound object>, ...}
    """
    sounds = {}
    for chord_name in set(GESTURE_TO_CHORD.values()):
        file_path = f"{AUDIO_FOLDER}/{chord_name}.wav"
        try:
            sounds[chord_name] = pygame.mixer.Sound(file_path)
        except pygame.error as e:
            print(f"Warning: Could not load {file_path} ({e})")
    return sounds


def euclidean_distance(point_a, point_b):
    """Straight-line distance between two landmarks (in normalized 0-1 coords)."""
    return ((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2) ** 0.5


def get_finger_states(hand_landmarks):
    """
    Returns a list of 5 values (1 = extended, 0 = curled): [Thumb, Index, Middle, Ring, Pinky]

    This version is hand-agnostic (works for both left and right hands)
    and does NOT rely on MediaPipe's handedness classification, which
    can be unreliable. Instead, the thumb is checked geometrically:
    if it's spread away from the palm, it's "extended," regardless of
    which hand or orientation is being used.
    """
    states = []

    # --- Thumb: distance-based, orientation independent ---
    # Reference "hand size" = distance from wrist to middle-finger base.
    # This lets us judge "far" vs "close" relative to THIS hand's size,
    # so it works whether the hand is close to or far from the camera.
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_size = euclidean_distance(wrist, middle_mcp)

    thumb_tip = hand_landmarks[4]
    pinky_mcp = hand_landmarks[17]
    thumb_spread = euclidean_distance(thumb_tip, pinky_mcp)

    # If the thumb tip is far from the pinky's base (relative to hand size),
    # the thumb is spread out / extended. This threshold (0.6) was chosen
    # empirically -- tweak it slightly if thumb detection feels off for you.
    states.append(1 if thumb_spread > 0.6 * hand_size else 0)

    # --- Other four fingers: same as before (orientation doesn't matter here) ---
    for tip_idx, pip_idx in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_y = hand_landmarks[tip_idx].y
        pip_y = hand_landmarks[pip_idx].y
        states.append(1 if tip_y < pip_y else 0)

    return states


def map_gesture_to_chord(finger_states):
    return GESTURE_TO_CHORD.get(tuple(finger_states), None)


def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)


def draw_chord_display(frame, chord_name):
    display_text = chord_name if chord_name else "No Chord"
    color = (0, 255, 0) if chord_name else (0, 0, 255)
    cv2.putText(frame, display_text, (frame.shape[1] // 2 - 80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 4)


def main():
    # Initialize pygame's audio mixer before anything else audio-related
    pygame.mixer.init()
    chord_sounds = load_chord_sounds()

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

    print("Audio playback running. Press 'q' to quit.")
    start_time = time.time()

    # Tracks the last chord that was played, so we don't replay the
    # same chord over and over every frame while the gesture is held.
    last_played_chord = None

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
            states = get_finger_states(result.hand_landmarks[0])
            chord_name = map_gesture_to_chord(states)

        # --- Only trigger playback when the chord CHANGES ---
        if chord_name != last_played_chord:
            if chord_name is not None and chord_name in chord_sounds:
                chord_sounds[chord_name].play()
            last_played_chord = chord_name

        draw_chord_display(frame, chord_name)
        cv2.imshow("AirChord", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()