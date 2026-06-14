"""Cross-validation comparison of MobileNetV2 (frozen backbone) WITHOUT vs
WITH data augmentation, for dog and cat.

Usage:
    python src/augment_cv.py --animal dog
    python src/augment_cv.py --animal cat
    python src/augment_cv.py --animal all   (default)

GOAL
----
cross_validation.py and tune_head.py established that head-only tuning is
mostly exhausted (cat "food" F1 stuck around 0.36, CV macro-F1 0.5223). This
script tests whether AUDIO + SPECTROGRAM AUGMENTATION of the minority classes
helps, while keeping the backbone frozen and the head at its DEFAULT
hyperparameters (dense_units=64, dropout=0.3, l2=0, lr=1e-3) - so any
difference vs the repere is attributable to augmentation alone.

This script extends cross_validation.py rather than duplicating it: it
reuses `make_folds` and `extract_logmel_batch` from cross_validation.py,
`build_backbone`/`spectrograms_to_images` from mobilenet_transfer.py, and
`train_head`/`evaluate_and_plot`/etc. from tl_common.py. Only MobileNetV2 is
evaluated here (the single retained backbone) - YAMNet is out of scope.

============================================================================
ANTI-LEAKAGE - HOW EACH OF THE 4 PITFALLS IS HANDLED
============================================================================

PITFALL 1 - augment before splitting (the worst one)
------------------------------------------------------
For each fold, augmentation happens INSIDE the CV loop, AFTER `make_folds`
has produced that fold's (train_idx, val_idx):
  - `augment_fold_train` is called ONLY on `train_idx` clips.
  - The fold's validation set is `mobilenet_X_all[val_idx]` - the ORIGINAL,
    never-augmented features, exactly as in cross_validation.py.
  - The MobileNetV2 features for the augmented variants are computed fresh,
    per fold, from that fold's augmented spectrograms only - they are never
    added to `mobilenet_X_all` (the shared, fold-independent array used for
    originals) and never reused across folds.
No augmented variant of any clip can ever appear in a validation fold,
because augmented variants are only ever generated from that fold's
`train_idx`.

PITFALL 2 - individual leakage (cat)
--------------------------------------
Every augmented variant inherits the `cat_id` of the original clip it was
derived from (it's the SAME cat's clip, just transformed). Since the variant
is only generated from `train_idx`, its `cat_id` is already a member of
`train_cats = {cat_id of train_idx}`, which `make_folds`'s
StratifiedGroupKFold already guarantees is disjoint from `val_cats`. For
every fold (cat only) this script explicitly computes
`(train_cats | aug_cats) & val_cats` and prints `group_violations=0` - see
the per-fold output and `reports/augmentation_cv_scores.csv`.

For DOG, there is still no speaker ID (documented limitation from
preprocess.py / cross_validation.py) - this isn't introduced or worsened by
augmentation, it's the same pre-existing limitation.

PITFALL 3 - shared normalisation statistic
---------------------------------------------
- Feature extraction is unchanged: `extract_logmel_batch` produces RAW
  (non-normalised) log-mel spectrograms, and `spectrograms_to_images` rescales
  each spectrogram with a PER-SAMPLE min-max - both fold-independent, as
  argued in cross_validation.py's docstring.
- The augmentation functions in `augment.py` also only ever use PER-CLIP
  statistics: `add_gaussian_noise` scales noise by THIS clip's own peak
  amplitude, and `spec_augment` fills masked regions with THIS spectrogram's
  own mean. No statistic is ever computed across clips, original or
  augmented, train or validation.
- The only statistic that depends on the fold's labels is
  `class_weight="balanced"` (via `tl_common.class_weight_dict`), which is
  computed from `y_train` AFTER augmentation (i.e. the realised, more
  balanced label distribution) - still per-fold, still train-only.

PITFALL 4 - test touched more than once / used to choose
-------------------------------------------------------------
The comparison (without vs with augmentation, both animals, CV mean+/-std,
"food" F1 for cat) is evaluated ENTIRELY on CV validation folds - the test
set (`data/processed/<animal>/test_*.npy`, or equivalently
`df[df.split=="test"]`) is never touched in `run_cv_comparison`. It is
touched exactly once, in `final_eval`, with the ONE augmentation config used
throughout this script (no search over configs) - see that function's
docstring.
"""

from __future__ import annotations

import argparse
import time

import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from augment import augment_clip
from cross_validation import extract_logmel_batch, make_folds
from mobilenet_transfer import build_backbone, spectrograms_to_images
from preprocess import fix_length
from tl_common import (
    CONFIGS,
    DATA_RAW,
    MODELS_DIR,
    REPORTS_DIR,
    SAMPLE_RATE,
    SEED,
    evaluate_and_plot,
    label_names,
    load_manifest,
    train_head,
)

# Number of EXTRA augmented variants generated per clip, by animal and class.
# CAT (priority, per consigne): "food" (92 clips) is the weakest class and
# gets the most help (x2 extra -> x3 total), "brushing" (127) gets x1 extra
# (x2 total), "isolation" (221, already the majority) gets none. This is a
# soft rebalancing toward roughly-equal per-fold counts, not exact parity.
# DOG: the class imbalance is much milder (46/33/34 - ratio ~1.4 vs cat's
# ~2.4), so instead of targeted rebalancing I apply a UNIFORM x2 (n_aug=1 for
# every class) - a general augmentation/regularisation pass that doesn't
# change the (already mild) relative imbalance.
AUGMENT_FACTORS = {
    "dog": {"bark": 1, "growl": 1, "grunt": 1},
    "cat": {"isolation": 0, "brushing": 1, "food": 2},
}


def load_waveforms(df: pd.DataFrame, animal: str) -> list[np.ndarray]:
    """Fixed-length raw waveforms (post fix_length, pre log-mel) for every
    clip in `df`, in row order - the input augmentation needs."""
    target_len = int(round(CONFIGS[animal]["duration_s"] * SAMPLE_RATE))
    out = []
    for rel_path in df["path"]:
        y, _ = librosa.load(DATA_RAW / rel_path, sr=SAMPLE_RATE, mono=True)
        out.append(fix_length(y, target_len))
    return out


def augment_fold_train(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    y: np.ndarray,
    waveforms_all: list[np.ndarray],
    animal: str,
    target_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Generate augmented log-mel spectrograms for `train_idx` clips only,
    per AUGMENT_FACTORS. Returns (specs, labels, source_indices) - all three
    have the same length (sum of n_aug over train_idx)."""
    factors = AUGMENT_FACTORS[animal]
    rng = np.random.default_rng(seed)

    specs, labels, src_idx = [], [], []
    for idx in train_idx:
        n_aug = factors[df.loc[idx, "label"]]
        for _ in range(n_aug):
            specs.append(augment_clip(waveforms_all[idx], SAMPLE_RATE, target_len, rng))
            labels.append(y[idx])
            src_idx.append(int(idx))

    return np.stack(specs).astype(np.float32), np.array(labels, dtype=np.int64), src_idx


def run_cv_comparison(animal: str, backbone, rows: list[dict]) -> None:
    cfg = CONFIGS[animal]
    n_classes = len(cfg["classes"])
    food_idx = cfg["classes"].index("food") if "food" in cfg["classes"] else None
    target_len = int(round(cfg["duration_s"] * SAMPLE_RATE))

    df = load_manifest(animal).reset_index(drop=True)
    label_to_idx = {name: i for i, name in enumerate(cfg["classes"])}
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)

    print(f"\n=== {animal.upper()}: precomputing features for {len(df)} files ===")
    t0 = time.time()
    logmel_all = extract_logmel_batch(df, animal)
    mobilenet_X_all = backbone.predict(spectrograms_to_images(logmel_all), verbose=0)
    print(f"  MobileNetV2 features (originals): {mobilenet_X_all.shape} ({time.time() - t0:.1f}s)")

    t0 = time.time()
    waveforms_all = load_waveforms(df, animal)
    print(f"  raw waveforms loaded for augmentation ({time.time() - t0:.1f}s)")

    folds = make_folds(animal, df)
    strategy = "StratifiedGroupKFold (group=cat_id)" if animal == "cat" else "StratifiedKFold"
    print(f"  {len(folds)}-fold CV ({strategy})")

    for fold_i, (train_idx, val_idx) in enumerate(folds):
        n_train_groups = n_val_groups = violations = None
        if animal == "cat":
            train_cats = set(df.loc[train_idx, "cat_id"])
            val_cats = set(df.loc[val_idx, "cat_id"])
            n_train_groups, n_val_groups = len(train_cats), len(val_cats)
            violations = len(train_cats & val_cats)
            assert violations == 0, f"cat_id leakage in fold {fold_i}"

        X_train, X_val = mobilenet_X_all[train_idx], mobilenet_X_all[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # --- WITHOUT augmentation (re-verification of the repere) ---
        start = time.time()
        model, history = train_head(X_train, y_train, X_val, y_val, n_classes)
        elapsed = time.time() - start
        y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        food_f1 = (
            f1_score(y_val, y_pred, average=None, zero_division=0, labels=range(n_classes))[food_idx]
            if food_idx is not None
            else None
        )
        rows.append(
            {
                "animal": animal,
                "augmentation": "none",
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
                "food_f1": food_f1,
            }
        )
        print(
            f"  Fold {fold_i} [no aug ] n_train={len(train_idx):3d} n_val={len(val_idx):3d} "
            f"acc={acc:.4f} macro_f1={f1:.4f} ({elapsed:.1f}s)"
        )

        # --- WITH augmentation (train-of-fold only) ---
        aug_specs, aug_y, aug_src_idx = augment_fold_train(
            df, train_idx, y, waveforms_all, animal, target_len, seed=SEED + fold_i
        )
        aug_X = backbone.predict(spectrograms_to_images(aug_specs), verbose=0)

        X_train_aug = np.concatenate([X_train, aug_X], axis=0)
        y_train_aug = np.concatenate([y_train, aug_y], axis=0)

        if animal == "cat":
            aug_cats = set(df.loc[aug_src_idx, "cat_id"])
            violations_aug = len((train_cats | aug_cats) & val_cats)
            assert violations_aug == 0, f"cat_id leakage after augmentation in fold {fold_i}"

        start = time.time()
        model, history = train_head(X_train_aug, y_train_aug, X_val, y_val, n_classes)
        elapsed = time.time() - start
        y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        food_f1 = (
            f1_score(y_val, y_pred, average=None, zero_division=0, labels=range(n_classes))[food_idx]
            if food_idx is not None
            else None
        )
        rows.append(
            {
                "animal": animal,
                "augmentation": "audio+specaugment",
                "fold": fold_i,
                "n_train": len(train_idx) + len(aug_src_idx),
                "n_val": len(val_idx),
                "n_train_groups": n_train_groups,
                "n_val_groups": n_val_groups,
                "group_violations": violations_aug if animal == "cat" else None,
                "epochs": len(history.history["loss"]),
                "elapsed_s": elapsed,
                "accuracy": acc,
                "macro_f1": f1,
                "food_f1": food_f1,
            }
        )
        print(
            f"  Fold {fold_i} [+ aug  ] n_train={len(train_idx) + len(aug_src_idx):3d} "
            f"(+{len(aug_src_idx)}) n_val={len(val_idx):3d} "
            f"acc={acc:.4f} macro_f1={f1:.4f} ({elapsed:.1f}s)"
        )


def final_eval(animal: str, backbone) -> dict:
    """ONE-SHOT final evaluation: train on the (augmented) train split of
    data/processed (the 70/15/15 split from preprocess.py, NOT the CV
    folds), early-stop on the ORIGINAL (non-augmented) val split, and
    evaluate ONCE on the ORIGINAL (non-augmented) test split.

    This is the single time this script touches the test set. The
    augmentation config applied to "train" is the SAME AUGMENT_FACTORS used
    throughout this script - there is no search over configs, so there is
    nothing to "select" using the test score.
    """
    cfg = CONFIGS[animal]
    names = label_names(animal)
    n_classes = len(cfg["classes"])
    target_len = int(round(cfg["duration_s"] * SAMPLE_RATE))

    df = load_manifest(animal).reset_index(drop=True)
    label_to_idx = {name: i for i, name in enumerate(cfg["classes"])}
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)

    logmel_all = extract_logmel_batch(df, animal)
    waveforms_all = load_waveforms(df, animal)

    splits_idx = {s: df.index[df["split"] == s].to_numpy() for s in ["train", "val", "test"]}

    aug_specs, aug_y, _ = augment_fold_train(
        df, splits_idx["train"], y, waveforms_all, animal, target_len, seed=SEED + 1000
    )

    X_train_orig = backbone.predict(spectrograms_to_images(logmel_all[splits_idx["train"]]), verbose=0)
    X_aug = backbone.predict(spectrograms_to_images(aug_specs), verbose=0)
    X_train = np.concatenate([X_train_orig, X_aug], axis=0)
    y_train = np.concatenate([y[splits_idx["train"]], aug_y], axis=0)

    X_val = backbone.predict(spectrograms_to_images(logmel_all[splits_idx["val"]]), verbose=0)
    y_val = y[splits_idx["val"]]
    X_test = backbone.predict(spectrograms_to_images(logmel_all[splits_idx["test"]]), verbose=0)
    y_test = y[splits_idx["test"]]

    print(
        f"  train: {len(splits_idx['train'])} original + {len(aug_y)} augmented "
        f"= {len(y_train)} | val: {len(y_val)} (original) | test: {len(y_test)} (original)"
    )

    model, history = train_head(X_train, y_train, X_val, y_val, n_classes)

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    result = evaluate_and_plot(
        y_test,
        y_pred,
        names,
        f"MobileNetV2 + augmentation - {animal} - confusion matrix (test)",
        REPORTS_DIR / f"mobilenet_aug_{animal}_confusion_matrix.png",
    )
    result["epochs_run"] = len(history.history["loss"])
    result["n_train"] = len(y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"mobilenet_aug_{animal}_head.keras")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("Loading MobileNetV2 (ImageNet weights, frozen backbone)...")
    backbone = build_backbone()

    rows: list[dict] = []
    for animal in animals:
        run_cv_comparison(animal, backbone, rows)

    df_scores = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "augmentation_cv_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    print(f"\nPer-fold scores saved to: {csv_path}")

    print("\n=== Mean +/- std across folds (without vs with augmentation) ===")
    summary = df_scores.groupby(["animal", "augmentation"])[["accuracy", "macro_f1", "food_f1"]].agg(
        ["mean", "std"]
    )
    print(summary)

    print("\n=== Final ONE-SHOT test evaluation (with augmentation, default head config) ===")
    for animal in animals:
        print(f"\n{animal.upper()}:")
        result = final_eval(animal, backbone)
        print(
            f"  TEST accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f} "
            f"(epochs={result['epochs_run']}, n_train={result['n_train']})"
        )
        print(result["report"])
        print(f"  Confusion matrix saved to: {result['fig_path']}")

    total_elapsed = time.time() - t_start
    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
