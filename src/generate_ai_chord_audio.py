"""
generate_ai_chord_audio.py
------------------------------
Generates realistic guitar chord audio using ElevenLabs' text-to-audio
Sound Generation API (genuinely AI-generated, not synthesized sine waves).

Setup required:
1. Get a free API key from https://elevenlabs.io (Profile -> API Keys)
2. Set it as an environment variable named ELEVENLABS_API_KEY
3. pip install requests pydub
4. Install ffmpeg (needed by pydub to convert mp3 -> wav)

Run once:
    python generate_ai_chord_audio.py
Output: assets/audio/G.wav, C.wav, D.wav, Em.wav, A.wav (overwrites existing)
"""

import os
import requests
from pydub import AudioSegment
import io

API_URL = "https://api.elevenlabs.io/v1/sound-generation"
OUTPUT_FOLDER = "assets/audio"

# Text prompts describing each chord's desired sound. Since this is a
# generative model, wording matters -- feel free to tweak these
# descriptions to steer the tone/style if results don't sound right.
CHORD_PROMPTS = {
    "G":  "A single clean strum of a G major chord on an acoustic guitar, warm and bright, natural string decay, no background noise,make the duration of sound upto 5 seconds",
    "C":  "A single clean strum of a C major chord on an acoustic guitar, bright and clear, natural string decay, no background noise,make the duration of sound upto 5 seconds",
    "D":  "A single clean strum of a D major chord on an acoustic guitar, warm and resonant, natural string decay, no background noise,make the duration of sound upto 5 seconds",
    "Em": "A single clean strum of an E minor chord on an acoustic guitar, mellow and slightly somber, natural string decay, no background noise,make the duration of sound upto 5 seconds",
    "A":  "A single clean strum of an A major chord on an acoustic guitar, full and rich, natural string decay, no background noise,make the duration of sound upto 5 seconds",
}

DURATION_SECONDS = 2.0


def generate_chord_audio(prompt, api_key):
    """
    Calls the ElevenLabs Sound Generation API with a text prompt and
    returns the raw MP3 audio bytes.
    """
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": prompt,
        "duration_seconds": DURATION_SECONDS,
        # prompt_influence controls how strictly the model follows the
        # text description vs. adding its own creative variation.
        # Higher = more literal/predictable, lower = more variety.
        "prompt_influence": 0.4,
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(
            f"API request failed ({response.status_code}): {response.text}"
        )

    return response.content  # raw MP3 bytes


def mp3_bytes_to_wav_file(mp3_bytes, output_path):
    """Converts in-memory MP3 bytes to a .wav file on disk using pydub."""
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    audio.export(output_path, format="wav")


def main():
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: ELEVENLABS_API_KEY environment variable not set.")
        print("Set it and restart your terminal, then try again.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for chord_name, prompt in CHORD_PROMPTS.items():
        print(f"Generating '{chord_name}'...")
        try:
            mp3_bytes = generate_chord_audio(prompt, api_key)
            output_path = os.path.join(OUTPUT_FOLDER, f"{chord_name}.wav")
            mp3_bytes_to_wav_file(mp3_bytes, output_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Failed to generate '{chord_name}': {e}")

    print("\nDone. Replace/keep these files in assets/audio/ -- no code")
    print("changes needed elsewhere, since gesture_engine.py just plays")
    print("whatever .wav files are already there.")


if __name__ == "__main__":
    main()