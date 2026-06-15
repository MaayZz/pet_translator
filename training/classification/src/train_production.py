"""Train and save the FINAL production models (dog + cat).

Usage:
    python src/train_production.py --animal dog
    python src/train_production.py --animal cat
    python src/train_production.py --animal all   (default)

WHY THIS SCRIPT
----------------
All previous sessions evaluate approaches on CV folds and a held-out test
split, to pick the most defendable backbone+classifier per animal (see
reports/production_model_summary.md for that choice and its justification).
This script does NOT run any new experiment or comparison - it just RE-FITS
the already-selected combination (frozen MobileNetV2, ImageNet, pooling="avg"
-> tl_common dense head) on ALL labelled data that is not the test split, so
the deployed model benefits from every clip available, as is standard practice
once a model is ready for production.

- Reuses unchanged: `mobilenet_transfer.build_backbone` /
  `spectrograms_to_images`, `tl_common.train_head`.
- Inputs: data/processed/<animal>/{train,val}_X.npy (+ _y.npy) - the SAME
  normalized log-mel features used everywhere else. Per
  `cross_validation.py`'s docstring, MobileNetV2's per-sample min-max
  rescale makes the global (mean, std) normalisation mathematically
  irrelevant to the resulting features, so training on these arrays gives
  identical MobileNetV2 features as training on raw log-mels.
- The "train" and "val" splits are concatenated into one pool. Since
  `tl_common.train_head` needs a validation set for early stopping, I carve
  out a small stratified 15% slice of that pool (seed=42) purely as an
  early-stopping signal - it is NOT used for any reported metric (CV/test
  evaluation already happened in earlier sessions). For cat this slice is a
  plain class-stratified split (not cat_id-group-aware): that is a deliberate
  simplification, safe here because the slice never feeds into a reported
  score, only into "when do I stop training the final model".
- The TEST split is never loaded by this script.

OUTPUTS (gitignored, models/)
-------------------------------
- models/production_<animal>_mobilenet_head.keras
- models/production_<animal>_meta.json: classes (in label order), duration_s,
  the train-set log-mel normalisation (mean/std, copied from
  data/processed/<animal>/norm_stats.json), backbone name/image size, default
  confidence threshold, and a few reproducibility fields. This is the only
  file (besides the .keras head and the Keras-cached ImageNet weights) that
  src/predict.py needs at inference time.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.model_selection import train_test_split

from mobilenet_transfer import IMG_SIZE, build_backbone, spectrograms_to_images
from tl_common import CONFIGS, DATA_PROCESSED, MODELS_DIR, SEED, train_head

VAL_FRACTION = 0.15
DEFAULT_THRESHOLD = 0.50


def load_split(animal: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(DATA_PROCESSED / animal / f"{split}_X.npy")
    y = np.load(DATA_PROCESSED / animal / f"{split}_y.npy")
    return X, y


def process_animal(animal: str, backbone) -> dict:
    cfg = CONFIGS[animal]
    classes = cfg["classes"]
    n_classes = len(classes)

    X_train, y_train = load_split(animal, "train")
    X_val, y_val = load_split(animal, "val")
    X_pool = np.concatenate([X_train, X_val])
    y_pool = np.concatenate([y_train, y_val])

    feats = backbone.predict(spectrograms_to_images(X_pool), verbose=0)

    X_fit, X_es, y_fit, y_es = train_test_split(
        feats, y_pool, test_size=VAL_FRACTION, stratify=y_pool, random_state=SEED
    )

    model, history = train_head(X_fit, y_fit, X_es, y_es, n_classes)
    model_path = MODELS_DIR / f"production_{animal}_mobilenet_head.keras"
    model.save(model_path)

    with open(DATA_PROCESSED / animal / "norm_stats.json") as fh:
        norm_stats = json.load(fh)

    meta = {
        "animal": animal,
        "classes": classes,
        "duration_s": cfg["duration_s"],
        "sample_rate": 16000,
        "backbone": "mobilenet_v2",
        "img_size": IMG_SIZE,
        "logmel_norm_mean": norm_stats["mean"],
        "logmel_norm_std": norm_stats["std"],
        "default_threshold": DEFAULT_THRESHOLD,
        "n_fit": int(len(y_fit)),
        "n_early_stop_val": int(len(y_es)),
        "epochs_run": len(history.history["loss"]),
        "seed": SEED,
    }
    meta_path = MODELS_DIR / f"production_{animal}_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    return {"model_path": model_path, "meta_path": meta_path, **meta}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MobileNetV2 (ImageNet weights, frozen backbone)...")
    backbone = build_backbone()

    for animal in animals:
        print(f"\n=== Production model - {animal.upper()} ===")
        result = process_animal(animal, backbone)
        print(
            f"Fit on {result['n_fit']} clips, early-stopped on "
            f"{result['n_early_stop_val']} clips ({result['epochs_run']} epochs)"
        )
        print(f"Classes: {result['classes']}")
        print(f"Saved model: {result['model_path']}")
        print(f"Saved meta:  {result['meta_path']}")


if __name__ == "__main__":
    main()
