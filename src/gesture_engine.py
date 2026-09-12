"""
gesture_engine.py
------------------------------
Core AirChord logic, separated from any UI. Handles:
- Hand landmark detection (MediaPipe)
- Finger state / gesture -> chord mapping (rule-based)
- Stability buffering
- Audio playback

This module is UI-agnostic on purpose: both a plain OpenCV window
(main.py) and the Tkinter app (app_ui.py) can use the same engine
without duplicating logic.
"""

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import pygame

MODEL_PATH = "assets/models/hand_landmarker.task"
AUDIO_FOLDER = "assets/audio"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [2, 6, 10, 14, 18]

GESTURE_TO_CHORD = {
    (0, 0, 0, 0, 0): "G",
    (0, 1, 0, 0, 0): "C",
    (0, 1, 1, 0, 0): "D",
    (0, 1, 1, 1, 0): "Em",
    (1, 1, 1, 1, 1): "A",
}

STABILITY_THRESHOLD = 6


def euclidean_distance(point_a, point_b):
    return ((point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2) ** 0.5


def get_finger_states(hand_landmarks):
    """Hand-agnostic finger up/down detection. See Phase 6 notes for details."""
    states = []

    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]
    hand_size = euclidean_distance(wrist, middle_mcp)

    thumb_tip = hand_landmarks[4]
    pinky_mcp = hand_landmarks[17]
    thumb_spread = euclidean_distance(thumb_tip, pinky_mcp)
    states.append(1 if thumb_spread > 0.6 * hand_size else 0)

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


class GestureEngine:
    """
    Wraps hand detection + gesture mapping + stability buffering + audio
    into one object with a simple process_frame() method, so any UI
    (CLI window, Tkinter app, etc.) can drive it the same way.
    """

    def __init__(self):
        pygame.mixer.init()
        self.chord_sounds = self._load_chord_sounds()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.start_time = time.time()
        self.last_played_chord = None
        self.candidate_chord = None
        self.candidate_streak = 0

    def _load_chord_sounds(self):
        sounds = {}
        for chord_name in set(GESTURE_TO_CHORD.values()):
            file_path = f"{AUDIO_FOLDER}/{chord_name}.wav"
            try:
                sounds[chord_name] = pygame.mixer.Sound(file_path)
            except pygame.error as e:
                print(f"Warning: Could not load {file_path} ({e})")
        return sounds

    def process_frame(self, frame):
        """
        Takes one BGR webcam frame (already flipped, if desired, by the
        caller). Draws landmarks onto it in-place, and returns:
        (confirmed_chord, hand_detected, chord_just_changed)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self.start_time) * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        hand_detected = bool(result.hand_landmarks)
        raw_chord = None

        if hand_detected:
            draw_landmarks(frame, result.hand_landmarks)
            states = get_finger_states(result.hand_landmarks[0])
            raw_chord = map_gesture_to_chord(states)

        if raw_chord == self.candidate_chord:
            self.candidate_streak += 1
        else:
            self.candidate_chord = raw_chord
            self.candidate_streak = 1

        confirmed_chord = (
            self.candidate_chord
            if self.candidate_streak >= STABILITY_THRESHOLD
            else self.last_played_chord
        )

        chord_just_changed = confirmed_chord != self.last_played_chord

        if chord_just_changed:
            if confirmed_chord is not None and confirmed_chord in self.chord_sounds:
                self.chord_sounds[confirmed_chord].play()
            self.last_played_chord = confirmed_chord

        return confirmed_chord, hand_detected, chord_just_changed

    def close(self):
        pygame.mixer.quit()