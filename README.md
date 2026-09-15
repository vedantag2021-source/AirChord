# AirChord 🎸

**AirChord** watches your hand through a webcam and plays the matching guitar chord in real time — turning empty air into a playable instrument. Built with computer vision (MediaPipe + OpenCV), rule-based gesture recognition, a responsive desktop UI, and genuinely AI-generated chord audio.

**➡️ [Download the latest Windows build](https://github.com/vedantag2021-source/AirChord/releases/latest/download/AirChord.exe)** — no Python or setup required, just download and run.

> **Note:** since this is an independently built app (not code-signed), Windows may show a "Windows protected your PC" SmartScreen warning on first launch. This is expected for unsigned indie apps — click **More info → Run anyway** to proceed.

## Features

- 🖐️ Real-time hand tracking via MediaPipe's HandLandmarker
- 🎸 Five gesture-to-chord mappings: **G, C, D, Em, A**
- 🔊 Instant audio playback, triggered only on confirmed gesture changes
- 🤖 **AI-generated chord audio** — sounds are generated from text descriptions via ElevenLabs' Sound Generation API, not pre-recorded samples
- 🎛️ **User-customizable sound** — change the audio style (e.g., "warm acoustic", "gritty electric") and duration directly from the app's Settings panel, and regenerate all chord sounds on demand
- 📊 Live chord diagrams showing the actual finger positions for the detected chord
- 🖥️ Responsive Tkinter UI — layout adapts to window size, video scales to fit
- 📝 Running chord history log
- ✋ Hand-agnostic detection — works with either left or right hand

## Gesture Guide

| Gesture | Fingers (Thumb, Index, Middle, Ring, Pinky) | Chord |
|---|---|---|
| Fist | Down, Down, Down, Down, Down | G |
| Index only | Down, Up, Down, Down, Down | C |
| Index + Middle | Down, Up, Up, Down, Down | D |
| Index + Middle + Ring | Down, Up, Up, Up, Down | Em |
| Open palm | Up, Up, Up, Up, Up | A |

## Tech Stack

| Purpose | Tool |
|---|---|
| Language | Python 3.11 |
| Hand detection | MediaPipe Tasks API (HandLandmarker) |
| Video capture/processing | OpenCV |
| Audio playback | Pygame |
| Desktop UI | Tkinter + Pillow |
| AI audio generation | ElevenLabs Sound Generation API |

## Project Structure

```
AirChord/
├── assets/
│   ├── audio/              # Chord sound samples (G.wav, C.wav, etc.)
│   └── models/             # MediaPipe hand_landmarker.task model file
├── src/
│   ├── gesture_engine.py       # Core detection + gesture mapping + audio logic
│   ├── chord_diagrams.py       # Fretboard diagram rendering
│   ├── ai_audio_generator.py   # AI chord-audio generation (style + duration configurable)
│   ├── app_ui.py               # Main application (responsive Tkinter UI + Settings panel)
│   └── main.py                 # CLI version (plain OpenCV window, no UI shell)
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/AirChord.git
   cd AirChord
   ```

2. **Create and activate a virtual environment** (Python 3.11 recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the MediaPipe hand-tracking model**
   ```bash
   mkdir -p assets/models
   curl -L -o assets/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

5. **Add chord audio samples** to `assets/audio/`: `G.wav`, `C.wav`, `D.wav`, `Em.wav`, `A.wav`

   You have two options:
   - **AI-generated (recommended)** — get a free API key from [elevenlabs.io](https://elevenlabs.io) (Developers → API Keys), then either run `python src/generate_ai_chord_audio.py` once, or open the app and use the **Settings panel** to generate/customize sounds directly (style description + duration), entering your key there.
   - **Placeholder tones** — no API key needed: `python src/generate_placeholder_audio.py`

## Usage

**Full application (recommended):**
```bash
python src/app_ui.py
```
Click **Start**, hold up a gesture from the table above, and the matching chord will play. Click **Stop** to release the camera.

**Lightweight CLI version:**
```bash
python src/main.py
```
Press `q` to quit.

## AI-Generated Audio

Chord sounds aren't pre-recorded samples — they're generated from text descriptions using ElevenLabs' Sound Generation API. Each chord's identity (e.g., "G major chord") is combined with a user-chosen style description (e.g., "warm acoustic, fingerpicked") to build a prompt, which the model turns into audio.

This can be customized two ways:
- **From the command line:** edit `CHORD_BASE_DESCRIPTIONS` or run `generate_all_chords()` directly in `src/ai_audio_generator.py`
- **From the app itself:** open the Settings panel in `app_ui.py`, enter a style description and duration, and click **Regenerate Chord Sounds** — new audio is generated live and swapped into the running app without a restart

**Security note:** when entering your ElevenLabs API key in the app's Settings panel, it's held only in memory for that session and is never written to disk or committed to the repository. Never hardcode an API key directly into source code that gets pushed to GitHub.

## How It Works

1. **Hand detection** — MediaPipe's HandLandmarker locates 21 landmark points on your hand every frame.
2. **Finger-state detection** — for each finger, the tip position is compared to a lower joint to determine up/down. The thumb uses a hand-agnostic, distance-based check so it works with either hand.
3. **Gesture mapping** — the resulting finger-state pattern (e.g., `[0,1,1,0,0]`) is looked up in a gesture-to-chord dictionary.
4. **Stability buffering** — a gesture must hold steady for several consecutive frames before it's "confirmed," filtering out hand-jitter false triggers.
5. **Audio playback** — the matching chord sample plays only when the confirmed chord actually changes.

## Roadmap / Build Log

- [x] Phase 0: Project setup
- [x] Phase 1: Webcam capture
- [x] Phase 2: Hand landmark detection
- [x] Phase 3: Finger-state detection
- [x] Phase 4: Gesture-to-chord mapping
- [x] Phase 5: Audio playback
- [x] Phase 6: Stability & polish
- [x] Phase 7: Responsive UI, chord diagrams, documentation

