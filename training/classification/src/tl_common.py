"""Shared helpers for the transfer-learning experiments (YAMNet & MobileNetV2).

Both src/yamnet_transfer.py and src/mobilenet_transfer.py import from here so
that the two approaches share the same paths, label encoding, class-weight
computation, and evaluation/plotting code - the only thing that differs
between them is how the input features are produced (raw audio -> YAMNet
embeddings, vs. precomputed log-mel spectrograms -> MobileNetV2 features).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

SEED = 42
SAMPLE_RATE = 16000

# Same fixed durations as src/preprocess.py (4s for dog, 2s for cat).
CONFIGS = {
    "dog": {"classes": ["bark", "growl", "grunt"], "duration_s": 4.0},
    "cat": {"classes": ["brushing", "food", "isolation"], "duration_s": 2.0},
}


def load_manifest(animal: str) -> pd.DataFrame:
    """Read the split manifest produced by src/preprocess.py (path,label,split,[cat_id])."""
    return pd.read_csv(REPORTS_DIR / f"{animal}_split_manifest.csv")


def load_label_encoding(animal: str) -> dict[str, int]:
    with open(DATA_PROCESSED / animal / "label_encoding.json") as fh:
        return json.load(fh)


def label_names(animal: str) -> list[str]:
    """Class names in the same order as the integer labels (0, 1, 2, ...)."""
    mapping = load_label_encoding(animal)
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def class_weight_dict(y_train: np.ndarray) -> dict[int, float]:
    """Balanced class weights for Keras' class_weight=, computed on train labels only."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_head(input_dim: int, n_classes: int) -> tf.keras.Model:
    """Small dense classification head shared by every transfer-learning approach."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )


def train_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Build + train the shared head with the same fixed defaults everywhere:
    Adam(1e-3), up to 50 epochs, batch_size=8, class_weight="balanced" (from
    y_train only), early stopping on val_loss (patience=5, restore best
    weights). Resets the global seed so every call is reproducible.
    """
    tf.keras.utils.set_random_seed(SEED)
    model = build_head(X_train.shape[1], n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=8,
        verbose=0,
        class_weight=class_weight_dict(y_train),
        callbacks=[early_stop],
    )
    return model, history


def evaluate_and_plot(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    names: list[str],
    title: str,
    fig_path: Path,
) -> dict:
    """Compute accuracy/macro-F1/per-class report and save a confusion matrix figure."""
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=names, zero_division=0)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=names, ax=ax, cmap="Blues"
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "report": report,
        "fig_path": fig_path,
    }
