"""Baseline models for the DOG and CAT classification tasks.

Usage:
    python src/baseline.py --animal dog
    python src/baseline.py --animal cat
    python src/baseline.py --animal all   (default)

For each animal this script:
  1. Loads the preprocessed log-mel features from data/processed/<animal>/
     (produced earlier by src/preprocess.py): train/val/test splits, each
     shaped (n_samples, n_mels, n_frames), already normalized with
     train-only mean/std.
  2. FLOOR BASELINE: a DummyClassifier(strategy="most_frequent") fit on the
     train labels, evaluated on the test set. This is the absolute floor any
     real model must beat.
  3. MODEL BASELINE: a Logistic Regression on the FLATTENED log-mel
     spectrograms (one feature vector per sample):
       - StandardScaler fit on train only, applied to train/val/test
         (no leakage).
       - class_weight="balanced" to account for the class imbalance.
       - A small grid search over the regularization strength C, where the
         candidate is selected using macro-F1 on the validation set (this is
         the "validate on val if needed" step).
     The selected model is then evaluated once on the test set.
  4. Reports accuracy, macro-F1, a per-class precision/recall/f1 report, and
     saves a confusion matrix figure to
     reports/<animal>_confusion_matrix.png.
  5. Prints an explicit floor vs. baseline comparison (accuracy and
     macro-F1).

WHY LOGISTIC REGRESSION AND NOT A SMALL CNN HERE
A baseline needs to be a STABLE, REPRODUCIBLE reference point. With so few
training samples (79 for dog, ~300 for cat) and high-dimensional flattened
inputs, an unregularized CNN would likely overfit and give noisy, unstable
results - a poor reference for comparing against the final model. The CNN
will be considered later as an OPTION for the main model (with proper
regularization), not as the baseline. The full reasoning is written up in
reports/baseline_summary.md.

Everything here is seeded (SEED=42) and runs on CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

SEED = 42
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]


def load_split(animal: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a split and flatten the (n_mels, n_frames) features to vectors."""
    X = np.load(DATA_PROCESSED / animal / f"{split}_X.npy")
    y = np.load(DATA_PROCESSED / animal / f"{split}_y.npy")
    return X.reshape(len(X), -1), y


def load_label_names(animal: str) -> list[str]:
    with open(DATA_PROCESSED / animal / "label_encoding.json") as fh:
        mapping = json.load(fh)
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def run_floor_baseline(y_train: np.ndarray, y_test: np.ndarray) -> dict:
    """Always-predict-the-majority-class baseline (fit on train, scored on test)."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_pred = dummy.predict(np.zeros((len(y_test), 1)))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }


def tune_logreg(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
) -> tuple[LogisticRegression, float]:
    """Try a few values of C, keep the one with the best macro-F1 on val."""
    best_c, best_score, best_model = None, -1.0, None
    for c in C_GRID:
        model = LogisticRegression(
            C=c,
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )
        model.fit(X_train, y_train)
        score = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
        if score > best_score:
            best_c, best_score, best_model = c, score, model
    return best_model, best_c


def process_animal(animal: str) -> dict:
    X_train, y_train = load_split(animal, "train")
    X_val, y_val = load_split(animal, "val")
    X_test, y_test = load_split(animal, "test")
    label_names = load_label_names(animal)

    # Standardization fit on train only, applied to val/test (no leakage).
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    floor = run_floor_baseline(y_train, y_test)

    model, best_c = tune_logreg(X_train_s, y_train, X_val_s, y_val)
    y_pred = model.predict(X_test_s)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_test, y_pred, target_names=label_names, zero_division=0
    )

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=label_names, ax=ax, cmap="Blues"
    )
    ax.set_title(f"{animal.capitalize()} baseline - confusion matrix (test)")
    fig.tight_layout()
    fig_path = REPORTS_DIR / f"{animal}_confusion_matrix.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "floor": floor,
        "best_c": best_c,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "fig_path": fig_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for animal in animals:
        print(f"\n=== {animal.upper()} ===")
        result = process_animal(animal)

        print(
            f"Floor (majority class)  -> accuracy={result['floor']['accuracy']:.4f}, "
            f"macro-F1={result['floor']['macro_f1']:.4f}"
        )
        print(
            f"Logistic regression     -> accuracy={result['accuracy']:.4f}, "
            f"macro-F1={result['macro_f1']:.4f}  (selected C={result['best_c']})"
        )
        beats_acc = result["accuracy"] > result["floor"]["accuracy"]
        beats_f1 = result["macro_f1"] > result["floor"]["macro_f1"]
        print(f"Baseline beats floor on accuracy : {beats_acc}")
        print(f"Baseline beats floor on macro-F1 : {beats_f1}")
        print("\nPer-class report (test set):")
        print(result["report"])
        print(f"Confusion matrix saved to: {result['fig_path']}")


if __name__ == "__main__":
    main()
