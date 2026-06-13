"""Approach A - YAMNet (pretrained on AudioSet) as a frozen audio embedding extractor.

Usage:
    python src/yamnet_transfer.py --animal dog
    python src/yamnet_transfer.py --animal cat
    python src/yamnet_transfer.py --animal all   (default)

For each animal:
  1. I reload the RAW audio for every file listed in
     reports/<animal>_split_manifest.csv (same train/val/test split as
     everywhere else), resample it to 16 kHz mono, and apply the exact same
     centered pad/crop to a fixed duration (4s for dog, 2s for cat) as
     src/preprocess.py - so YAMNet sees "the same clips" as my other models.
  2. I run each clip through YAMNet (downloaded once from TF Hub, frozen -
     no fine-tuning) and get a sequence of 1024-dim embeddings (one per
     ~0.96s analysis window). I mean-pool over time to get a single
     1024-dim embedding per clip.
  3. On top of these frozen embeddings, I train a small dense classification
     head (Dense(64, relu) -> Dropout -> Dense(n_classes, softmax)), with
     class_weight="balanced" and early stopping on the validation loss.
  4. I evaluate the trained head on the test set (accuracy, macro-F1,
     per-class report, confusion matrix).

YAMNet expects mono float32 audio at 16 kHz with values in [-1, 1], which is
exactly what librosa.load(..., sr=16000, mono=True) gives me, and my fixed
durations (4s / 2s) are well above YAMNet's minimum input length (~0.975s),
so no extra adaptation was needed there.

Hyperparameters here are deliberately simple defaults (Adam, lr=1e-3, up to
50 epochs with early stopping, batch_size=8) - no tuning was done, as
requested for this first run.
"""

from __future__ import annotations

import argparse
import time

import librosa
import numpy as np
import tensorflow_hub as hub

from preprocess import fix_length
from tl_common import (
    CONFIGS,
    DATA_RAW,
    MODELS_DIR,
    REPORTS_DIR,
    SAMPLE_RATE,
    evaluate_and_plot,
    label_names,
    load_manifest,
    train_head,
)

YAMNET_URL = "https://tfhub.dev/google/yamnet/1"


def extract_embeddings(yamnet, df, animal: str) -> np.ndarray:
    target_len = int(round(CONFIGS[animal]["duration_s"] * SAMPLE_RATE))
    embeddings = []
    for rel_path in df["path"]:
        y, _ = librosa.load(DATA_RAW / rel_path, sr=SAMPLE_RATE, mono=True)
        y = fix_length(y, target_len)
        _, emb, _ = yamnet(y)
        embeddings.append(emb.numpy().mean(axis=0))
    return np.stack(embeddings).astype(np.float32)


def process_animal(animal: str, yamnet) -> dict:
    cfg = CONFIGS[animal]
    df = load_manifest(animal)
    label_to_idx = {name: i for i, name in enumerate(cfg["classes"])}
    names = label_names(animal)

    print(f"  Extracting YAMNet embeddings for {len(df)} {animal} files...")
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        X = extract_embeddings(yamnet, sub, animal)
        y = sub["label"].map(label_to_idx).to_numpy(dtype=np.int64)
        splits[split] = (X, y)
        print(f"    {split}: {X.shape}")

    start = time.time()
    model, history = train_head(
        splits["train"][0],
        splits["train"][1],
        splits["val"][0],
        splits["val"][1],
        len(cfg["classes"]),
    )
    elapsed = time.time() - start

    y_pred = model.predict(splits["test"][0], verbose=0).argmax(axis=1)
    result = evaluate_and_plot(
        splits["test"][1],
        y_pred,
        names,
        f"YAMNet - {animal} - confusion matrix (test)",
        REPORTS_DIR / f"yamnet_{animal}_confusion_matrix.png",
    )
    result["elapsed_s"] = elapsed
    result["epochs_run"] = len(history.history["loss"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"yamnet_{animal}_head.keras")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading YAMNet from TF Hub (frozen, no fine-tuning)...")
    yamnet = hub.load(YAMNET_URL)

    for animal in animals:
        print(f"\n=== YAMNet - {animal.upper()} ===")
        result = process_animal(animal, yamnet)
        print(f"Trained for {result['epochs_run']} epochs in {result['elapsed_s']:.1f}s")
        print(
            f"Test accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}"
        )
        print("\nPer-class report (test set):")
        print(result["report"])
        print(f"Confusion matrix saved to: {result['fig_path']}")


if __name__ == "__main__":
    main()
