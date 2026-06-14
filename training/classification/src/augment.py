"""Per-clip audio + spectrogram augmentation for the MobileNetV2 pipeline.

Every function here takes an explicit `np.random.Generator` and only ever
looks at the ONE clip it is given - no statistic is ever computed across
clips, so calling these functions on a fold's training clips cannot leak any
information from other clips (own or validation). This is the property
`augment_cv.py` relies on for its anti-leakage argument.

TECHNIQUES (and why these magnitudes)
--------------------------------------
- Additive Gaussian noise, scaled to 3-4% of the clip's own peak amplitude:
  a mild, label-preserving "recording noise" augmentation, standard in SER
  (speech/sound emotion recognition) literature. Deliberately NEUTRAL noise
  (no "emotional" sounds added).
- Pitch shift, +/-2 semitones max: the literature on SER augmentation finds
  that moderate pitch shifts (a couple of semitones) preserve perceived
  emotion/vocalisation type, while larger shifts start to change it.
- Time stretch, +/-5% (rate in [0.95, 1.05]): kept at the conservative end of
  the "<=10%, ideally <=5%" range recommended to avoid distorting the
  vocalisation's temporal envelope (e.g. turning a bark into a growl).
- SpecAugment (one frequency mask + one time mask, ~12.5% of each axis),
  applied AFTER the log-mel extraction, filled with the spectrogram's own
  mean so the masked region doesn't introduce an out-of-distribution value.

ORDER OF OPERATIONS
--------------------
pitch_shift -> time_stretch -> add_gaussian_noise -> fix_length -> log-mel ->
spec_augment. Pitch/time changes are applied to the clean signal first, then
noise is added to the final-duration waveform (representing an independent
noise floor, not something that should itself be pitch/time-shifted).
fix_length is applied AFTER time_stretch because time_stretch changes the
number of samples.
"""

from __future__ import annotations

import librosa
import numpy as np

from preprocess import extract_logmel, fix_length

NOISE_FACTOR_RANGE = (0.03, 0.04)
PITCH_SHIFT_RANGE_SEMITONES = (-2.0, 2.0)
TIME_STRETCH_RANGE = (0.95, 1.05)
SPEC_FREQ_MASK_FRAC = 0.125
SPEC_TIME_MASK_FRAC = 0.125


def add_gaussian_noise(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add neutral Gaussian noise scaled to 3-4% of this clip's own peak amplitude."""
    factor = rng.uniform(*NOISE_FACTOR_RANGE)
    peak = float(np.max(np.abs(y))) + 1e-8
    noise = rng.normal(0.0, 1.0, size=y.shape).astype(np.float32)
    return (y + factor * peak * noise).astype(np.float32)


def pitch_shift(y: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Shift pitch by a random amount in +/-2 semitones."""
    n_steps = rng.uniform(*PITCH_SHIFT_RANGE_SEMITONES)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps).astype(np.float32)


def time_stretch(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Stretch/compress time by a random rate in [0.95, 1.05] (<=5%)."""
    rate = rng.uniform(*TIME_STRETCH_RANGE)
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def spec_augment(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One frequency mask + one time mask (~12.5% of each axis), filled with
    the spectrogram's own mean."""
    spec = spec.copy()
    n_mels, n_frames = spec.shape
    fill = float(spec.mean())

    f_width = max(1, int(round(SPEC_FREQ_MASK_FRAC * n_mels)))
    f0 = int(rng.integers(0, n_mels - f_width + 1))
    spec[f0 : f0 + f_width, :] = fill

    t_width = max(1, int(round(SPEC_TIME_MASK_FRAC * n_frames)))
    t0 = int(rng.integers(0, n_frames - t_width + 1))
    spec[:, t0 : t0 + t_width] = fill

    return spec


def augment_clip(y: np.ndarray, sr: int, target_len: int, rng: np.random.Generator) -> np.ndarray:
    """Produce ONE augmented log-mel spectrogram from ONE raw waveform.

    Same label as the original (only the audio/spectrogram is transformed).
    Draws all random magnitudes from `rng`, so calling this repeatedly with
    the same rng (in a deterministic order) gives reproducible, but distinct,
    variants.
    """
    y_aug = pitch_shift(y, sr=sr, rng=rng)
    y_aug = time_stretch(y_aug, rng=rng)
    y_aug = add_gaussian_noise(y_aug, rng=rng)
    y_aug = fix_length(y_aug, target_len)
    spec = extract_logmel(y_aug)
    return spec_augment(spec, rng=rng)
