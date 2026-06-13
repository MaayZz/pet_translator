"""Cross-validation evaluation of the YAMNet and MobileNetV2 transfer-learning
heads, for dog and cat.

Usage:
    python src/cross_validation.py --animal dog
    python src/cross_validation.py --animal cat
    python src/cross_validation.py --animal all   (default)

WHY CROSS-VALIDATION
---------------------
My first transfer-learning run evaluated everything on a single, small test
set (17 dog clips, 67 cat clips). A single split gives one number, and that
number has a lot of variance - especially for dog, where 17 test clips means
each misclassified file moves accuracy by ~6 percentage points. CV instead
trains and evaluates the SAME approach on several different train/validation
splits and reports the mean and standard deviation, which is a much more
honest picture of how reliable a given score actually is.

This run does NOT change any hyperparameter: same head architecture
(Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)), same Adam
1e-3, same class_weight="balanced", same early stopping, same seed=42
(reset before every fold via tl_common.train_head). The ONLY thing that
changes versus the first run is the evaluation protocol.

FOLD STRATEGY PER ANIMAL
-------------------------
- CAT (440 clips, 21 individual cats, classes brushing/food/isolation):
  I use StratifiedGroupKFold with the cat's individual ID (extracted from the
  filename, e.g. "ANI01") as the GROUP. This guarantees that every clip from
  a given cat ends up in exactly one fold - no cat is ever split across
  train and validation. Plain StratifiedKFold (no group) would let the model
  see some clips from a cat in training and other clips from the SAME cat in
  validation, which would let it partly "recognise the cat" rather than the
  sound class - a real leakage risk given how individual-specific a cat's
  vocalisations/sounds can be.

  I chose k=4 folds. With only 21 cats, k=5 would leave very few cats (and
  very few "food" clips - the smallest class, 92 total) in some validation
  folds. I checked k=3 and k=4 explicitly: both give 0 group violations and
  every fold keeps a reasonable mix of all 3 classes (k=4's smallest
  per-fold "food" count is 19). k=4 gives me one more independent estimate
  than k=3 while keeping validation folds of a decent size (88-121 clips,
  i.e. bigger than my original 67-clip test set), so I went with k=4.

- DOG (113 clips, classes bark/growl/grunt): I use plain StratifiedKFold
  (k=5), stratified by class label at the file level. 113 files / 5 folds
  gives ~22-23 validation clips per fold - already larger than the original
  17-clip test set.

  HONEST LIMITATION: the shivarao dog dataset does not provide any
  speaker/individual ID, so unlike for cat I CANNOT build a group-aware
  split - I have no way to know if two files come from the same dog. If the
  dataset does contain repeated recordings from the same animal, some of
  those could land in both the train and validation part of a fold, which
  would make the dog CV scores slightly optimistic (the model could partly
  recognise an individual dog's voice rather than the vocalisation type).
  Stratified CV is the best I can do with the information available; I'm
  flagging this so the numbers below aren't over-interpreted.

ANTI-LEAKAGE CHECKLIST
------------------------
A. Group leakage (cat): verified and printed per fold below - 0 cat_id
   shared between a fold's train and validation set (StratifiedGroupKFold
   guarantees this, but I check it explicitly anyway).

B. Dog speaker leakage: not preventable with this dataset (see above),
   documented as a known limitation.

C. Normalisation fit on train-of-fold only: I confirm here that NEITHER
   approach actually uses any cross-sample normalisation statistic that
   could leak between a fold's train and validation set:
     - YAMNet: each clip is passed through the frozen YAMNet model
       independently (raw 16 kHz waveform -> embeddings, mean-pooled over
       time). No mean/std/min/max is ever computed across clips.
     - MobileNetV2: each log-mel spectrogram is rescaled to [0, 1] with a
       PER-SAMPLE min-max (src/mobilenet_transfer.spectrograms_to_images) -
       this only uses that sample's own min/max, never anything computed
       across other samples. The GLOBAL (mean, std) normalisation used in
       src/preprocess.py is therefore irrelevant here: for any monotonically
       increasing affine map x -> a*x + b (a>0, which is exactly what
       (x - mean) / std is, since std > 0), per-sample min-max gives the
       EXACT SAME result with or without that map (the min and max are
       transformed by the same a, b and cancel out in the min-max formula).
       So I extract RAW log-mel spectrograms directly from audio for this
       script (no normalisation step at all) - the resulting MobileNetV2
       features are identical to what they would be under any global
       normalisation, fold-specific or not.
   The only statistic that genuinely depends on the fold's train labels is
   class_weight="balanced" (via tl_common.class_weight_dict), which is
   already computed from y_train only inside tl_common.train_head - i.e.
   correctly per-fold.

   Net result: per-clip feature extraction (YAMNet embeddings, MobileNetV2
   features) does not depend on the fold at all, so I compute it ONCE for
   all clips of an animal and simply index into it per fold. This is both
   correct (no leakage - see above) and efficient (no repeated YAMNet/
   MobileNetV2 inference across folds).

OUTPUT
-------
- reports/cv_scores.csv: one row per (animal, approach, fold) with n_train,
  n_val, epochs trained, accuracy, macro-F1.
- Printed mean +/- std table per (animal, approach), to compare against the
  single-split numbers from reports/transfer_learning_summary.md.
"""

from __future__ import annotations

import argparse
import time

import librosa
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from mobilenet_transfer import build_backbone, spectrograms_to_images
from preprocess import extract_logmel, fix_length
from tl_common import CONFIGS, DATA_RAW, REPORTS_DIR, SAMPLE_RATE, SEED, load_manifest, train_head
from yamnet_transfer import YAMNET_URL, extract_embeddings

N_FOLDS = {"dog": 5, "cat": 4}


def extract_logmel_batch(df: pd.DataFrame, animal: str) -> np.ndarray:
    """Raw (non-normalised) log-mel spectrograms, same recipe as preprocess.py
    (fix_length + extract_logmel) - see module docstring for why no
    normalisation is needed here."""
    target_len = int(round(CONFIGS[animal]["duration_s"] * SAMPLE_RATE))
    feats = []
    for rel_path in df["path"]:
        y, _ = librosa.load(DATA_RAW / rel_path, sr=SAMPLE_RATE, mono=True)
        y = fix_length(y, target_len)
        feats.append(extract_logmel(y))
    return np.stack(feats).astype(np.float32)


def make_folds(animal: str, df: pd.DataFrame):
    if animal == "cat":
        cv = StratifiedGroupKFold(n_splits=N_FOLDS["cat"], shuffle=True, random_state=SEED)
        return list(cv.split(df, df["label"], groups=df["cat_id"]))
    cv = StratifiedKFold(n_splits=N_FOLDS["dog"], shuffle=True, random_state=SEED)
    return list(cv.split(df, df["label"]))


def process_animal(animal: str, yamnet, backbone) -> list[dict]:
    cfg = CONFIGS[animal]
    df = load_manifest(animal).reset_index(drop=True)
    label_to_idx = {name: i for i, name in enumerate(cfg["classes"])}
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)
    n_classes = len(cfg["classes"])

    print(f"\n=== {animal.upper()}: precomputing features for {len(df)} files ===")

    t0 = time.time()
    yamnet_X = extract_embeddings(yamnet, df, animal)
    print(f"  YAMNet embeddings: {yamnet_X.shape} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    logmel = extract_logmel_batch(df, animal)
    mobilenet_X = backbone.predict(spectrograms_to_images(logmel), verbose=0)
    print(f"  MobileNetV2 features: {mobilenet_X.shape} ({time.time() - t0:.1f}s)")

    folds = make_folds(animal, df)
    strategy = "StratifiedGroupKFold (group=cat_id)" if animal == "cat" else "StratifiedKFold"
    print(f"  {len(folds)}-fold CV ({strategy})")

    rows = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        n_train_groups = n_val_groups = violations = None
        if animal == "cat":
            train_cats = set(df.loc[train_idx, "cat_id"])
            val_cats = set(df.loc[val_idx, "cat_id"])
            n_train_groups, n_val_groups = len(train_cats), len(val_cats)
            violations = len(train_cats & val_cats)
            print(
                f"  Fold {fold_i}: n_train={len(train_idx)} n_val={len(val_idx)} "
                f"train_cats={n_train_groups} val_cats={n_val_groups} "
                f"group_violations={violations}"
            )
            assert violations == 0, f"cat_id leakage in fold {fold_i}"

        for approach, X in [("yamnet", yamnet_X), ("mobilenet", mobilenet_X)]:
            start = time.time()
            model, history = train_head(
                X[train_idx], y[train_idx], X[val_idx], y[val_idx], n_classes
            )
            elapsed = time.time() - start

            y_pred = model.predict(X[val_idx], verbose=0).argmax(axis=1)
            acc = accuracy_score(y[val_idx], y_pred)
            f1 = f1_score(y[val_idx], y_pred, average="macro", zero_division=0)

            rows.append(
                {
                    "animal": animal,
                    "approach": approach,
                    "fold": fold_i,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_train_groups": n_train_groups,
                    "n_val_groups": n_val_groups,
                    "group_violations": violations,
                    "epochs": len(history.history["loss"]),
                    "elapsed_s": elapsed,
                    "accuracy": acc,
                    "macro_f1": f1,
                }
            )
            print(
                f"    fold {fold_i} {approach:10s}: epochs={len(history.history['loss']):2d} "
                f"acc={acc:.4f} macro_f1={f1:.4f} ({elapsed:.1f}s)"
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("Loading YAMNet from TF Hub (frozen, no fine-tuning)...")
    yamnet = hub.load(YAMNET_URL)
    print("Loading MobileNetV2 (ImageNet weights, frozen backbone)...")
    backbone = build_backbone()

    all_rows: list[dict] = []
    for animal in animals:
        all_rows.extend(process_animal(animal, yamnet, backbone))

    df_scores = pd.DataFrame(all_rows)
    csv_path = REPORTS_DIR / "cv_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    print(f"\nPer-fold scores saved to: {csv_path}")

    print("\n=== Mean +/- std across folds ===")
    summary = df_scores.groupby(["animal", "approach"])[["accuracy", "macro_f1"]].agg(
        ["mean", "std"]
    )
    print(summary)

    total_elapsed = time.time() - t_start
    print(f"\nTotal CV wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
