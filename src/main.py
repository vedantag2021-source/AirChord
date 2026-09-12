"""
Phase 6: Stability & Polish
------------------------------
Goal: Make gesture detection feel stable and instrument-like instead of
jittery, and improve on-screen readability.

Key additions over Phase 5:
- A gesture must be held for several consecutive frames before it's
  "confirmed" and triggers audio -- this filters out one-frame flickers
  caused by natural hand shake or momentary misdetection.
- Cleaner on-screen display with a background box behind the chord text.
- Explicit "No hand detected" message when the hand leaves the frame.

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

# How many consecutive frames a gesture must be held before we "confirm"
# it and trigger audio. Higher = more stable but slightly less responsive.
# Lower = snappier but more prone to false triggers from hand jitter.
STABILITY_THRESHOLD = 6


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


def draw_chord_display(frame, chord_name, hand_detected):
    """
    Draws a readable status box at the top of the frame.
    Shows the current chord, or 'No Chord' if a hand is visible but
    the gesture isn't recognized, or 'No Hand Detected' if there's
    no hand in view at all.
    """
    if not hand_detected:
        display_text = "No Hand Detected"
        color = (0, 165, 255)  # orange
    elif chord_name:
        display_text = chord_name
        color = (0, 255, 0)  # green
    else:
        display_text = "No Chord"
        color = (0, 0, 255)  # red

    # Draw a solid dark background box so text stays readable
    # regardless of what's behind it in the webcam feed.
    box_width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (box_width, 80), (30, 30, 30), -1)

    # Center the text horizontally using its actual rendered width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    thickness = 3
    text_size = cv2.getTextSize(display_text, font, font_scale, thickness)[0]
    text_x = (box_width - text_size[0]) // 2

    cv2.putText(frame, display_text, (text_x, 55), font, font_scale, color, thickness)


def main():
    # Initialize pygame's audio mixer before anything else audio-related
    pygame.mixer.init()
    chord_sounds = load_chord_sounds()
    print(f"[DEBUG] Successfully loaded sounds: {list(chord_sounds.keys())}")
    print(f"[DEBUG] Mixer initialized: {pygame.mixer.get_init()}")

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

    # Tracks the last CONFIRMED chord that was played (audio only
    # triggers when this changes).
    last_played_chord = None

    # --- Stability tracking ---
    # candidate_chord: what the current raw detection says right now
    # candidate_streak: how many consecutive frames it's been the same
    candidate_chord = None
    candidate_streak = 0

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

        hand_detected = bool(result.hand_landmarks)
        raw_chord = None

        if hand_detected:
            draw_landmarks(frame, result.hand_landmarks)
            states = get_finger_states(result.hand_landmarks[0])
            raw_chord = map_gesture_to_chord(states)

        # --- Update the stability streak ---
        if raw_chord == candidate_chord:
            candidate_streak += 1
        else:
            candidate_chord = raw_chord
            candidate_streak = 1

        # Only treat the gesture as "confirmed" once it's been stable
        # for STABILITY_THRESHOLD consecutive frames.
        confirmed_chord = candidate_chord if candidate_streak >= STABILITY_THRESHOLD else last_played_chord

        # --- Only trigger playback when the CONFIRMED chord changes ---
        if confirmed_chord != last_played_chord:
            if confirmed_chord is not None and confirmed_chord in chord_sounds:
                print(f"[DEBUG] Playing chord: {confirmed_chord}")
                play_result = chord_sounds[confirmed_chord].play()
                print(f"[DEBUG] play() returned channel: {play_result}")
            elif confirmed_chord is not None:
                print(f"[DEBUG] Chord '{confirmed_chord}' recognized but NOT in loaded sounds: {list(chord_sounds.keys())}")
            last_played_chord = confirmed_chord

        draw_chord_display(frame, confirmed_chord, hand_detected)
        cv2.imshow("AirChord", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()