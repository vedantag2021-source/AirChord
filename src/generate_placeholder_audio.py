"""
generate_placeholder_audio.py
------------------------------
Generates simple synthesized placeholder chord sounds (G, C, D, Em, A)
as .wav files, so AirChord has working audio right away.

These are NOT real guitar recordings -- they're synthesized sine-wave
chords with a pluck-like envelope, meant as functional placeholders.
Swap these out with real guitar samples later for a more authentic sound.

Run this once:
    python generate_placeholder_audio.py
It will create files in assets/audio/
"""

import numpy as np
import wave
import os
import struct

SAMPLE_RATE = 44100
DURATION = 1.5  # seconds per chord
OUTPUT_FOLDER = "assets/audio"

# Chord definitions: each chord is a list of note frequencies (Hz)
# These are simplified open-chord note approximations.
CHORDS = {
    "G": [196.00, 246.94, 293.66, 392.00],   # G3, B3, D4, G4
    "C": [261.63, 329.63, 392.00],           # C4, E4, G4
    "D": [293.66, 369.99, 440.00],           # D4, F#4, A4
    "Em": [329.63, 392.00, 493.88],          # E4, G4, B4
    "A": [440.00, 554.37, 659.25],           # A4, C#5, E5
}


def generate_chord_wave(frequencies, duration=DURATION, sample_rate=SAMPLE_RATE):
    """
    Combines multiple sine waves (one per note) into a single chord sound,
    with a slight stagger between notes (to mimic a strum) and an
    exponential decay envelope (to mimic a plucked/strummed string fading out).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = np.zeros_like(t)

    for i, freq in enumerate(frequencies):
        # Slight delay per note to simulate a strum across strings
        delay_samples = int(i * 0.015 * sample_rate)
        note_wave = np.sin(2 * np.pi * freq * t)

        # Exponential decay envelope so each note fades out naturally
        envelope = np.exp(-3 * t)

        note_wave = note_wave * envelope

        # Shift the note's start by delay_samples (strum effect)
        shifted = np.zeros_like(t)
        if delay_samples < len(t):
            shifted[delay_samples:] = note_wave[:len(t) - delay_samples]

        waveform += shifted

    # Normalize to avoid clipping when notes overlap
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.8

    return waveform


def save_wave(filename, waveform, sample_rate=SAMPLE_RATE):
    """Writes a NumPy float waveform array to a 16-bit mono .wav file."""
    # Convert float samples (-1.0 to 1.0) into 16-bit PCM integers
    int_samples = np.int16(waveform * 32767)

    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 2 bytes = 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    for chord_name, frequencies in CHORDS.items():
        waveform = generate_chord_wave(frequencies)
        output_path = os.path.join(OUTPUT_FOLDER, f"{chord_name}.wav")
        save_wave(output_path, waveform)
        print(f"Generated: {output_path}")

    print("\nDone! These are placeholder tones -- replace with real guitar")
    print("samples later for authentic sound, using the same filenames.")


if __name__ == "__main__":
    main()