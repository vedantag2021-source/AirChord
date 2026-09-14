"""
ai_audio_generator.py
------------------------------
Reusable AI chord-audio generation logic, parameterized by style
description and duration, so it can be triggered from a UI (with
user-chosen settings) rather than only run as a fixed one-off script.
"""

import os
import io
import requests
from pydub import AudioSegment

API_URL = "https://api.elevenlabs.io/v1/sound-generation"
OUTPUT_FOLDER = "assets/audio"

# The core quality/character of each chord, combined with the user's
# chosen style description to build the final prompt sent to the API.
CHORD_BASE_DESCRIPTIONS = {
    "G":  "G major chord",
    "C":  "C major chord",
    "D":  "D major chord",
    "Em": "E minor chord",
    "A":  "A major chord",
}


def build_prompt(chord_name, style_description):
    """
    Combines the chord's identity with a user-provided style description
    into a single text prompt for the audio generation model.

    Example: chord_name="G", style_description="warm acoustic, fingerpicked"
    -> "A single clean strum of a G major chord, warm acoustic,
        fingerpicked, natural string decay, no background noise"
    """
    base = CHORD_BASE_DESCRIPTIONS[chord_name]
    return (
        f"A single clean strum of a {base}, {style_description}, "
        f"natural string decay, no background noise"
    )


def generate_chord_audio_bytes(prompt, api_key, duration_seconds):
    """Calls the ElevenLabs API and returns raw MP3 bytes."""
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": prompt,
        "duration_seconds": duration_seconds,
        "prompt_influence": 0.4,
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(f"API request failed ({response.status_code}): {response.text}")

    return response.content


def mp3_bytes_to_wav_file(mp3_bytes, output_path):
    audio = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    audio.export(output_path, format="wav")


def generate_all_chords(style_description, duration_seconds, api_key, progress_callback=None):
    """
    Generates all 5 chord sounds using the given style + duration.

    progress_callback, if provided, is called as:
        progress_callback(chord_name, status)
    where status is one of: "generating", "done", "error"
    (kept as plain, fixed strings so a UI can match on them exactly,
    rather than parsing free-form error text).

    Returns a dict like {"G": True, "C": True, "D": False, ...} where
    True = generated successfully, False = failed for that chord.
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    results = {}

    for chord_name in CHORD_BASE_DESCRIPTIONS:
        if progress_callback:
            progress_callback(chord_name, "generating")

        try:
            prompt = build_prompt(chord_name, style_description)
            mp3_bytes = generate_chord_audio_bytes(prompt, api_key, duration_seconds)
            output_path = os.path.join(OUTPUT_FOLDER, f"{chord_name}.wav")
            mp3_bytes_to_wav_file(mp3_bytes, output_path)

            results[chord_name] = True
            if progress_callback:
                progress_callback(chord_name, "done")

        except Exception as e:
            print(f"Error generating '{chord_name}': {e}")
            results[chord_name] = False
            if progress_callback:
                progress_callback(chord_name, "error")

    return results