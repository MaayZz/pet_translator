"""Generate synthetic WAV files that simulate pet sounds for testing the full pipeline.

Each file has controlled acoustic properties that mimic real pet vocalizations
so the professor can verify the classification pipeline end-to-end.

Usage:
    python3 generate_test_samples.py

Output:
    test_audio/dog/bark_01.wav       (broadband noise, harmonic stack)
    test_audio/dog/bark_02.wav
    test_audio/dog/growl_01.wav      (low-frequency sawtooth)
    test_audio/dog/growl_02.wav
    test_audio/dog/grunt_01.wav      (short low burst)
    test_audio/dog/grunt_02.wav
    test_audio/cat/brushing_01.wav   (high narrowband, harmonic)
    test_audio/cat/brushing_02.wav
    test_audio/cat/food_01.wav       (sustained mid-frequency)
    test_audio/cat/food_02.wav
    test_audio/cat/isolation_01.wav  (wavering, descending)
    test_audio/cat/isolation_02.wav
"""

import math
import struct
import os
from pathlib import Path

SR = 16000
AMPLITUDE = 0.5

OUT_DIR = Path(__file__).parent


def write_wav(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n * 2))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, SR, SR * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n * 2))
        for s in samples:
            s = max(-1.0, min(1.0, s))
            f.write(struct.pack("<h", int(s * 32767)))


def sine(freq, t):
    return math.sin(2 * math.pi * freq * t)


# --- Dog: Bark (broadband noise + harmonic stack, 0.3-0.8s) ---
def gen_bark(duration=0.6, pitch_shift=0.0):
    n = int(SR * duration)
    buf = [0.0] * n
    for i in range(n):
        t = i / SR
        envelope = max(0, 1 - t / duration)  # linear decay
        # Noise burst
        noise = (hash((i, 0)) % 2000 - 1000) / 1000.0
        # Harmonic stack
        harm = 0.0
        for h in range(1, 7):
            harm += (1 / h) * sine((400 + pitch_shift) * h, t)
        buf[i] = envelope * (0.4 * noise + 0.6 * harm)
    return buf


# --- Dog: Growl (low freq sawtooth, 0.5-1.5s) ---
def gen_growl(duration=1.0):
    n = int(SR * duration)
    buf = [0.0] * n
    for i in range(n):
        t = i / SR
        env = max(0, 1 - t / duration) * 0.7 + 0.3
        # Low sawtooth
        saw = 2 * ((100 * t) % 1.0) - 1
        # Sub-harmonics
        sub = 0.5 * sine(50, t) + 0.3 * sine(75, t)
        buf[i] = env * (0.6 * saw + 0.4 * sub)
    return buf


# --- Dog: Grunt (short low burst, 0.15-0.3s) ---
def gen_grunt(duration=0.2):
    n = int(SR * duration)
    buf = [0.0] * n
    for i in range(n):
        t = i / SR
        env = math.exp(-t * 15)  # fast exponential decay
        low = 0.5 * sine(120, t) + 0.3 * sine(180, t)
        buf[i] = env * low
    return buf


# --- Cat: Brushing (high narrowband harmonic, 1.5-3.0s) ---
def gen_brushing(duration=2.0):
    n = int(SR * duration)
    buf = [0.0] * n
    for i in range(n):
        t = i / SR
        env = 0.5 + 0.5 * math.sin(math.pi * t / duration)  # smooth attack/decay
        # High harmonics, slight vibrato
        vibrato = 20 * math.sin(2 * math.pi * 5 * t)
        harm = 0.0
        for h in range(1, 5):
            harm += (1 / h) * sine((600 + vibrato) * h, t)
        buf[i] = env * harm
    return buf


# --- Cat: Food (sustained mid-freq, 1.0-2.0s) ---
def gen_food(duration=1.5):
    n = int(SR * duration)
    buf = [0.0] * n
    for i in range(n):
        t = i / SR
        env = 0.3 + 0.7 * max(0, 1 - t / duration)
        freq = 450 + 100 * math.sin(2 * math.pi * 3 * t)  # frequency modulation
        buf[i] = env * (0.5 * sine(freq, t) + 0.3 * sine(freq * 1.5, t))
    return buf


# --- Cat: Isolation (wavering descending, 1.5-3.0s) ---
def gen_isolation(duration=2.5):
    n = int(SR * duration)
    buf = [0.0] * n
    for i in range(n):
        t = i / SR
        env = 0.4 + 0.6 * max(0, 1 - t / duration)
        # Descending + wavering
        base_freq = 500 - 200 * (t / duration)
        wobble = 30 * math.sin(2 * math.pi * 4 * t)
        harm = 0.0
        for h in range(1, 4):
            harm += (1 / h) * sine((base_freq + wobble) * h, t)
        buf[i] = env * harm
    return buf


def main():
    samples = [
        ("dog", "bark_01", gen_bark(0.5, pitch_shift=0)),
        ("dog", "bark_02", gen_bark(0.7, pitch_shift=50)),
        ("dog", "growl_01", gen_growl(0.8)),
        ("dog", "growl_02", gen_growl(1.2)),
        ("dog", "grunt_01", gen_grunt(0.15)),
        ("dog", "grunt_02", gen_grunt(0.25)),
        ("cat", "brushing_01", gen_brushing(2.0)),
        ("cat", "brushing_02", gen_brushing(1.8)),
        ("cat", "food_01", gen_food(1.2)),
        ("cat", "food_02", gen_food(1.8)),
        ("cat", "isolation_01", gen_isolation(2.0)),
        ("cat", "isolation_02", gen_isolation(2.8)),
    ]
    for animal, name, buf in samples:
        path = OUT_DIR / animal / f"{name}.wav"
        write_wav(path, buf)
        print(f"Wrote {path}  ({len(buf) / SR:.2f}s)")

    print(f"\n{len(samples)} synthetic test files generated in {OUT_DIR}")


if __name__ == "__main__":
    main()
