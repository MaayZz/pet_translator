"""Approach B - MobileNetV2 (pretrained on ImageNet) on log-mel spectrograms
treated as images.

Usage:
    python src/mobilenet_transfer.py --animal dog
    python src/mobilenet_transfer.py --animal cat
    python src/mobilenet_transfer.py --animal all   (default)

For each animal:
  1. I reuse the log-mel features I already computed in
     data/processed/<animal>/{train,val,test}_X.npy (shape (n, 64, n_frames),
     already normalized with train-only mean/std - same data the baseline
     and YAMNet approaches are compared against).
  2. To make these look like an image for MobileNetV2:
       - I rescale each spectrogram INDEPENDENTLY to [0, 1] with a per-sample
         min-max (this uses only that sample's own values, so it doesn't
         introduce any train/val/test leakage).
       - I duplicate the single channel to 3 channels and resize the
         (64, n_frames) array to (96, 96) - the smallest input size
         MobileNetV2 accepts.
       - I scale to [0, 255] and apply Keras' standard
         mobilenet_v2.preprocess_input (which maps to [-1, 1]), exactly as
         if this were a normal RGB image.
  3. I run these "images" through MobileNetV2 (ImageNet weights, frozen,
     pooling="avg") to get a 1280-dim feature vector per clip - I never
     fine-tune the backbone.
  4. On top of these frozen features, I train the SAME small dense head as
     for YAMNet (Dense(64, relu) -> Dropout -> Dense(n_classes, softmax)),
     with class_weight="balanced" and early stopping on the validation loss.
  5. I evaluate the trained head on the test set (accuracy, macro-F1,
     per-class report, confusion matrix).

I want to be upfront that mapping dB-scale log-mel values to "pixel
intensities" via per-sample min-max is a simplification - there is no
canonical "correct" way to turn a spectrogram into an RGB image for a
backbone trained on natural photos. I picked the simplest reasonable option
and documented it in reports/transfer_learning_summary.md; this is something
I would revisit during the optimization phase.

Hyperparameters here are deliberately simple defaults (Adam, lr=1e-3, up to
50 epochs with early stopping, batch_size=8) - no tuning was done, as
requested for this first run.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import tensorflow as tf

from tl_common import (
    CONFIGS,
    DATA_PROCESSED,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    class_weight_dict,
    evaluate_and_plot,
    label_names,
)

IMG_SIZE = 96


def spectrograms_to_images(X: np.ndarray) -> np.ndarray:
    """(n, n_mels, n_frames) -> (n, IMG_SIZE, IMG_SIZE, 3), MobileNetV2-ready."""
    n = X.shape[0]
    flat = X.reshape(n, -1)
    mins = flat.min(axis=1, keepdims=True)
    maxs = flat.max(axis=1, keepdims=True)
    scaled = ((flat - mins) / (maxs - mins + 1e-8)).reshape(X.shape)

    imgs = np.repeat(scaled[..., np.newaxis], 3, axis=-1).astype(np.float32)
    imgs = tf.image.resize(imgs, (IMG_SIZE, IMG_SIZE)).numpy()
    imgs = imgs * 255.0
    return tf.keras.applications.mobilenet_v2.preprocess_input(imgs)


def build_backbone() -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False
    return base


def build_head(input_dim: int, n_classes: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )


def process_animal(animal: str, backbone: tf.keras.Model) -> dict:
    cfg = CONFIGS[animal]
    names = label_names(animal)

    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in ["train", "val", "test"]:
        X = np.load(DATA_PROCESSED / animal / f"{split}_X.npy")
        y = np.load(DATA_PROCESSED / animal / f"{split}_y.npy")
        imgs = spectrograms_to_images(X)
        feats = backbone.predict(imgs, verbose=0)
        splits[split] = (feats, y)
        print(f"    {split}: {feats.shape}")

    tf.keras.utils.set_random_seed(SEED)
    model = build_head(splits["train"][0].shape[1], len(cfg["classes"]))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    start = time.time()
    history = model.fit(
        splits["train"][0],
        splits["train"][1],
        validation_data=splits["val"],
        epochs=50,
        batch_size=8,
        verbose=0,
        class_weight=class_weight_dict(splits["train"][1]),
        callbacks=[early_stop],
    )
    elapsed = time.time() - start

    y_pred = model.predict(splits["test"][0], verbose=0).argmax(axis=1)
    result = evaluate_and_plot(
        splits["test"][1],
        y_pred,
        names,
        f"MobileNetV2 - {animal} - confusion matrix (test)",
        REPORTS_DIR / f"mobilenet_{animal}_confusion_matrix.png",
    )
    result["elapsed_s"] = elapsed
    result["epochs_run"] = len(history.history["loss"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODELS_DIR / f"mobilenet_{animal}_head.keras")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MobileNetV2 (ImageNet weights, frozen backbone)...")
    backbone = build_backbone()

    for animal in animals:
        print(f"\n=== MobileNetV2 - {animal.upper()} ===")
        result = process_animal(animal, backbone)
        print(f"Trained for {result['epochs_run']} epochs in {result['elapsed_s']:.1f}s")
        print(
            f"Test accuracy={result['accuracy']:.4f}, macro-F1={result['macro_f1']:.4f}"
        )
        print("\nPer-class report (test set):")
        print(result["report"])
        print(f"Confusion matrix saved to: {result['fig_path']}")


if __name__ == "__main__":
    main()
