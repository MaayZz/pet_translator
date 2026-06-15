"""Production inference module (dog + cat).

This is the stable interface the rest of the app (LLM module, frontend) is
meant to call - see reports/model_interface.md for the full integration doc.

    from predict import predict
    result = predict("some_clip.wav", animal="cat")
    # result = {
    #     "animal": "cat",
    #     "label": "isolation",          # or "uncertain" if confidence < threshold
    #     "confidence": 0.71,            # max class probability
    #     "probabilities": {"brushing": 0.12, "food": 0.17, "isolation": 0.71},
    #     "threshold": 0.5,
    # }

MODEL
------
Both animals use the same frozen MobileNetV2 (ImageNet, pooling="avg") feature
extractor followed by the small dense head (`tl_common.build_head`) - see
reports/production_model_summary.md for why this combination was selected as
the production model for both animals. The two per-animal heads are trained by
src/train_production.py and loaded here from
models/production_<animal>_mobilenet_head.keras (gitignored - run
train_production.py once to generate them).

PREPROCESSING (must match training exactly)
----------------------------------------------
For each clip: resample to 16 kHz mono -> center pad/crop to the animal's
fixed duration (`preprocess.fix_length`, 4s dog / 2s cat) -> log-mel
spectrogram (`preprocess.extract_logmel`, n_mels=64) -> normalize with the
TRAIN-set (mean, std) from data/processed/<animal>/norm_stats.json (copied
into models/production_<animal>_meta.json by train_production.py) ->
per-sample min-max image conversion + MobileNetV2 preprocessing
(`mobilenet_transfer.spectrograms_to_images`). This is exactly the pipeline
used to produce data/processed/<animal>/*_X.npy and to train the production
head, so the frozen backbone sees the same kind of input at inference as it
did during training/evaluation.

CONFIDENCE THRESHOLD
----------------------
`predict()` takes an optional `threshold` (default: `default_threshold` from
the model's meta file, 0.50). If the top class probability is below
`threshold`, `label` is set to `"uncertain"` instead of a class name - the
app should then show a generic message rather than asserting a specific
emotion/behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

from mobilenet_transfer import build_backbone, spectrograms_to_images
from preprocess import extract_logmel, fix_length
from tl_common import DATA_RAW, MODELS_DIR

ANIMALS = ("dog", "cat")

_backbone: tf.keras.Model | None = None
_heads: dict[str, tf.keras.Model] = {}
_meta: dict[str, dict] = {}


def _load_meta(animal: str) -> dict:
    if animal not in _meta:
        meta_path = MODELS_DIR / f"production_{animal}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found - run `python src/train_production.py` "
                f"first to generate the production models."
            )
        with open(meta_path) as fh:
            _meta[animal] = json.load(fh)
    return _meta[animal]


def _load_backbone() -> tf.keras.Model:
    global _backbone
    if _backbone is None:
        _backbone = build_backbone()
    return _backbone


def _load_head(animal: str) -> tf.keras.Model:
    if animal not in _heads:
        head_path = MODELS_DIR / f"production_{animal}_mobilenet_head.keras"
        _heads[animal] = tf.keras.models.load_model(head_path)
    return _heads[animal]


def predict(audio_path: str | Path, animal: str, threshold: float | None = None) -> dict:
    """Classify one audio clip for the given animal.

    Args:
        audio_path: path to a .wav (or any format librosa can read).
        animal: "dog" or "cat".
        threshold: confidence threshold for "uncertain" (defaults to the
            model's `default_threshold`, 0.50).

    Returns:
        {"animal", "label", "confidence", "probabilities", "threshold"} -
        see the module docstring for the exact format.

    Raises:
        ValueError: if `animal` is not "dog"/"cat", or the audio file cannot
            be read.
    """
    if animal not in ANIMALS:
        raise ValueError(f"Unknown animal {animal!r}: expected one of {ANIMALS}")

    meta = _load_meta(animal)
    if threshold is None:
        threshold = meta["default_threshold"]

    try:
        waveform, _ = librosa.load(str(audio_path), sr=meta["sample_rate"], mono=True)
    except Exception as exc:
        raise ValueError(f"Could not read audio file {audio_path!r}: {exc}") from exc

    target_len = int(round(meta["duration_s"] * meta["sample_rate"]))
    waveform = fix_length(waveform, target_len)
    logmel = extract_logmel(waveform)
    logmel = (logmel - meta["logmel_norm_mean"]) / meta["logmel_norm_std"]

    images = spectrograms_to_images(logmel[np.newaxis, ...])
    features = _load_backbone().predict(images, verbose=0)
    probs = _load_head(animal).predict(features, verbose=0)[0]

    classes = meta["classes"]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])
    label = classes[top_idx] if confidence >= threshold else "uncertain"

    return {
        "animal": animal,
        "label": label,
        "confidence": confidence,
        "probabilities": {name: float(p) for name, p in zip(classes, probs)},
        "threshold": threshold,
    }


if __name__ == "__main__":
    examples = [
        ("dog", DATA_RAW / "dog" / "bark" / "dog_1.wav"),
        ("cat", DATA_RAW / "cat" / "brushing" / "B_ANI01_MC_FN_SIM01_101.wav"),
    ]
    for animal, path in examples:
        result = predict(path, animal)
        print(f"\n{animal} <- {path.relative_to(DATA_RAW)}")
        print(result)
