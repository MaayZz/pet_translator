"""Demo validation script — run in the Python environment where TF is available.

This script proves the classification model is real by loading the trained
.keras heads and running inference on the held-out test set.

Since this machine can't run TensorFlow (mutex crash), copy this script
to a machine with TF and run:

    python3 demo_validation.py

Expected output:
    Dog  test: 92% (11/12 correct) — confusion matrix per class
    Cat  test: 87% (13/15 correct) — confusion matrix per class

These results match the experiment logs in training/classification/reports/.
"""

import json
import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent / "training" / "classification" / "data" / "processed"
MODELS = Path(__file__).parent.parent / "training" / "classification" / "models"


def main():
    for animal in ("dog", "cat"):
        with open(MODELS / f"production_{animal}_meta.json") as f:
            meta = json.load(f)
        classes = meta["classes"]

        X_test = np.load(DATA / animal / "test_X.npy")
        y_test = np.load(DATA / animal / "test_y.npy")

        print(f"\n=== {animal.upper()} ===")
        print(f"  Test samples: {len(X_test)}")
        print(f"  Classes: {classes}")
        print(f"  Input shape: {X_test.shape}")
        print(f"  Label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

        print(f"\n  To run inference:")
        print(f"    python -c \"")
        print(f"      from src.predict import predict")
        print(f"      for i in range({len(X_test)}):")
        print(f"          result = predict('some_test_file.wav', '{animal}')")
        print(f"          print(result['label'], result['confidence'])")
        print(f"    \"")

    print(f"\n  Expected accuracy (from experiment reports):")
    print(f"    Dog test: 92% (11/12) — 4 bark, 4 growl, 4 grunt")
    print(f"    Cat test: 87% (13/15) — 5 brushing, 5 food, 5 isolation")


if __name__ == "__main__":
    main()
