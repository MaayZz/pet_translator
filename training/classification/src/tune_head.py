"""Hyperparameter tuning for the MobileNetV2 classification head (frozen
backbone), evaluated via the SAME cross-validation protocol as
cross_validation.py.

Usage:
    python src/tune_head.py --animal dog
    python src/tune_head.py --animal cat
    python src/tune_head.py --animal all   (default)

SCOPE
-----
MobileNetV2 is the single backbone retained for both animals (decision made
upstream of this script). Only the small dense head on top of the frozen
MobileNetV2 features is tuned here - no backbone fine-tuning, no data
augmentation, no change to the train/val/test splits.

SEARCH SPACE
------------
- dense_units (hidden layer width): {32, 64, 128}
- dropout: {0.2, 0.3, 0.5}
- l2 (kernel regularizer on both Dense layers): {0, 1e-4, 1e-3}
- learning rate: {1e-3, 3e-4}

A full cartesian product would be 3*3*3*2 = 54 configs per animal, each
evaluated over 4-5 CV folds -> 200-450 head trainings - more than needed and
slow to repeat on CPU. Instead I use a SEQUENTIAL/TARGETED search:

  Stage 1 - "reg_grid" (dropout x l2, 9 configs, dense_units=64, lr=1e-3):
  I start here because the consigne's main concern is overfitting on the
  small "food" class, and dropout/L2 are the two regularization knobs that
  most directly affect that.

  Stage 2 - "capacity" (dense_units in {32, 128}, 2 configs): refines around
  the best (dropout, l2) from stage 1. dense_units=64 is already covered.

  Stage 3 - "lr" (lr=3e-4, 1 config): refines around the best
  (dense_units, dropout, l2) from stage 2. lr=1e-3 is already covered.

This is exactly 12 configs per animal (9+2+1) - every value of every
hyperparameter is tested at least once, vs 54 for the full grid (~4.5x
fewer configs), while putting most of the budget on the regularization axis
most relevant to the "food" problem.

SELECTION CRITERION (anti-cheating)
------------------------------------
Every config is scored ONLY by its mean macro-F1 across the CV folds from
cross_validation.make_folds(animal, df) - the EXACT SAME folds
(StratifiedGroupKFold k=4 on cat_id for cat, StratifiedKFold k=5 for dog,
seed=42) as cross_validation.py. As a sanity check, the (64, 0.3, 0, 1e-3)
config in stage 1 is the untuned default - its CV mean/std should reproduce
the cross_validation.py numbers (dog 0.8244+/-0.1114, cat 0.5223+/-0.1338)
exactly, since features/folds/seed are identical.

The held-out TEST set (data/processed/<animal>/test_{X,y}.npy) is touched
exactly ONCE, at the very end, with the single winning config per animal -
never during the search.

FEATURES
--------
MobileNetV2 features for the CV search are computed once per animal from raw
log-mel spectrograms (same as cross_validation.py: no cross-sample
normalisation statistic is used anywhere, so these features are
fold-independent and leakage-free - see cross_validation.py's docstring for
the full argument). The final train/val/test evaluation reuses the
preprocessed, normalised data/processed/<animal>/{split}_X.npy arrays (same
as mobilenet_transfer.py).

OUTPUT
------
- reports/head_tuning_scores.csv: one row per (animal, stage, config) with
  mean/std accuracy, macro-F1, and (for cat) "food" F1 across CV folds.
- reports/mobilenet_tuned_{dog,cat}_confusion_matrix.png: final test-set
  confusion matrix for the winning config.
- models/mobilenet_{dog,cat}_head_tuned.keras (gitignored).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from cross_validation import extract_logmel_batch, make_folds
from mobilenet_transfer import build_backbone, spectrograms_to_images
from tl_common import (
    CONFIGS,
    DATA_PROCESSED,
    MODELS_DIR,
    REPORTS_DIR,
    evaluate_and_plot,
    label_names,
    load_manifest,
    train_head,
)

DEFAULT = {"dense_units": 64, "dropout": 0.3, "l2": 0.0, "lr": 1e-3}
DROPOUT_GRID = [0.2, 0.3, 0.5]
L2_GRID = [0.0, 1e-4, 1e-3]
DENSE_GRID = [32, 64, 128]
LR_GRID = [1e-3, 3e-4]


def cv_evaluate(X, y, folds, n_classes, config, food_idx=None):
    """Mean/std accuracy, macro-F1 (and, for cat, "food" F1) across CV folds."""
    accs, f1s, foods = [], [], []
    for train_idx, val_idx in folds:
        model, history = train_head(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            n_classes,
            dense_units=config["dense_units"],
            dropout=config["dropout"],
            l2=config["l2"],
            lr=config["lr"],
        )
        y_pred = model.predict(X[val_idx], verbose=0).argmax(axis=1)
        accs.append(accuracy_score(y[val_idx], y_pred))
        f1s.append(f1_score(y[val_idx], y_pred, average="macro", zero_division=0))
        if food_idx is not None:
            per_class = f1_score(
                y[val_idx], y_pred, average=None, zero_division=0, labels=list(range(n_classes))
            )
            foods.append(per_class[food_idx])

    out = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=1)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s, ddof=1)),
    }
    if food_idx is not None:
        out["food_f1_mean"] = float(np.mean(foods))
        out["food_f1_std"] = float(np.std(foods, ddof=1))
    return out


def search(animal, X, y, folds, n_classes, food_idx, rows) -> dict:
    def run(stage, config):
        res = cv_evaluate(X, y, folds, n_classes, config, food_idx)
        row = {"animal": animal, "stage": stage, **config, **res}
        rows.append(row)
        food_str = f" food_f1={res['food_f1_mean']:.4f}" if food_idx is not None else ""
        print(
            f"  [{stage:8s}] dense={config['dense_units']:3d} dropout={config['dropout']:.1f} "
            f"l2={config['l2']:.0e} lr={config['lr']:.0e} -> "
            f"acc={res['acc_mean']:.4f}+/-{res['acc_std']:.4f} "
            f"macro_f1={res['f1_mean']:.4f}+/-{res['f1_std']:.4f}{food_str}"
        )
        return row

    best = None

    # Stage 1: dropout x l2 (dense_units, lr = defaults)
    for dropout in DROPOUT_GRID:
        for l2 in L2_GRID:
            config = {**DEFAULT, "dropout": dropout, "l2": l2}
            row = run("reg_grid", config)
            if best is None or row["f1_mean"] > best["f1_mean"]:
                best = row

    # Stage 2: dense_units around best regularization
    base = {k: best[k] for k in ("dense_units", "dropout", "l2", "lr")}
    for dense_units in DENSE_GRID:
        if dense_units == base["dense_units"]:
            continue
        config = {**base, "dense_units": dense_units}
        row = run("capacity", config)
        if row["f1_mean"] > best["f1_mean"]:
            best = row

    # Stage 3: learning rate around the best (dense_units, dropout, l2)
    base = {k: best[k] for k in ("dense_units", "dropout", "l2", "lr")}
    for lr in LR_GRID:
        if lr == base["lr"]:
            continue
        config = {**base, "lr": lr}
        row = run("lr", config)
        if row["f1_mean"] > best["f1_mean"]:
            best = row

    return best


def process_animal(animal: str, backbone, rows: list[dict]) -> tuple[dict, dict]:
    cfg = CONFIGS[animal]
    n_classes = len(cfg["classes"])
    food_idx = cfg["classes"].index("food") if "food" in cfg["classes"] else None

    df = load_manifest(animal).reset_index(drop=True)
    label_to_idx = {name: i for i, name in enumerate(cfg["classes"])}
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)

    print(f"\n=== {animal.upper()}: precomputing MobileNetV2 features for {len(df)} files ===")
    t0 = time.time()
    logmel = extract_logmel_batch(df, animal)
    X = backbone.predict(spectrograms_to_images(logmel), verbose=0)
    print(f"  features: {X.shape} ({time.time() - t0:.1f}s)")

    folds = make_folds(animal, df)
    strategy = "StratifiedGroupKFold (group=cat_id)" if animal == "cat" else "StratifiedKFold"
    print(f"  {len(folds)}-fold CV ({strategy}), {12} configs to evaluate")

    best = search(animal, X, y, folds, n_classes, food_idx, rows)

    baseline = next(
        r
        for r in rows
        if r["animal"] == animal
        and r["dense_units"] == DEFAULT["dense_units"]
        and r["dropout"] == DEFAULT["dropout"]
        and r["l2"] == DEFAULT["l2"]
        and r["lr"] == DEFAULT["lr"]
    )

    print(
        f"  BEST: dense={best['dense_units']} dropout={best['dropout']} "
        f"l2={best['l2']:.0e} lr={best['lr']:.0e} "
        f"-> macro_f1={best['f1_mean']:.4f}+/-{best['f1_std']:.4f} "
        f"(default was {baseline['f1_mean']:.4f}+/-{baseline['f1_std']:.4f})"
    )

    return best, baseline


def final_eval(animal: str, backbone, best_config: dict) -> dict:
    """Train once on train (early-stopped on val) with the winning config,
    evaluate ONCE on the original held-out test set."""
    cfg = CONFIGS[animal]
    names = label_names(animal)
    n_classes = len(cfg["classes"])

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        Xs = np.load(DATA_PROCESSED / animal / f"{split}_X.npy")
        ys = np.load(DATA_PROCESSED / animal / f"{split}_y.npy")
        feats = backbone.predict(spectrograms_to_images(Xs), verbose=0)
        splits[split] = (feats, ys)

    model, history = train_head(
        splits["train"][0],
        splits["train"][1],
        splits["val"][0],
        splits["val"][1],
        n_classes,
        dense_units=best_config["dense_units"],
        dropout=best_config["dropout"],
        l2=best_config["l2"],
        lr=best_config["lr"],
    )

    y_pred = model.predict(splits["test"][0], verbose=0).argmax(axis=1)
    result = evaluate_and_plot(
        splits["test"][1],
        y_pred,
        names,
        f"MobileNetV2 (tuned head) - {animal} - confusion matrix (test)",
        REPORTS_DIR / f"mobilenet_tuned_{animal}_confusion_matrix.png",
    )
    result["epochs_run"] = len(history.history["loss"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"mobilenet_{animal}_head_tuned.keras")

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
    bests: dict[str, dict] = {}
    baselines: dict[str, dict] = {}
    for animal in animals:
        best, baseline = process_animal(animal, backbone, rows)
        bests[animal] = best
        baselines[animal] = baseline

    df_scores = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "head_tuning_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    print(f"\nTuning scores saved to: {csv_path}")

    print("\n=== Final test-set evaluation (best config per animal, ONE shot) ===")
    for animal in animals:
        best = bests[animal]
        baseline = baselines[animal]
        print(
            f"\n{animal.upper()}: best config = dense_units={best['dense_units']}, "
            f"dropout={best['dropout']}, l2={best['l2']:.0e}, lr={best['lr']:.0e}"
        )
        print(
            f"  CV macro-F1: best={best['f1_mean']:.4f}+/-{best['f1_std']:.4f} "
            f"vs default={baseline['f1_mean']:.4f}+/-{baseline['f1_std']:.4f}"
        )
        if animal == "cat":
            print(
                f"  CV food F1: best={best['food_f1_mean']:.4f}+/-{best['food_f1_std']:.4f} "
                f"vs default={baseline['food_f1_mean']:.4f}+/-{baseline['food_f1_std']:.4f}"
            )

        result = final_eval(animal, backbone, best)
        print(
            f"  TEST accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f} "
            f"(epochs={result['epochs_run']})"
        )
        print(result["report"])
        print(f"  Confusion matrix saved to: {result['fig_path']}")

    total_elapsed = time.time() - t_start
    print(f"\nTotal tuning + final eval wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
