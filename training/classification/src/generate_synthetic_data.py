"""Generate synthetic processed data for training when raw audio is unavailable."""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"

CONFIGS = {
    "dog": {"classes": ["bark", "growl", "grunt"], "shape": (64, 126), "n_samples": 80},
    "cat": {"classes": ["brushing", "food", "isolation"], "shape": (64, 63), "n_samples": 100},
}

SEED = 42

def generate(animal: str) -> None:
    cfg = CONFIGS[animal]
    classes = cfg["classes"]
    n_classes = len(classes)
    shape = cfg["shape"]
    n = cfg["n_samples"]

    rng = np.random.default_rng(SEED)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    n_test = n - n_train - n_val

    for split_name, split_size in [("train", n_train), ("val", n_val), ("test", n_test)]:
        X = rng.normal(loc=-10, scale=8, size=(split_size, *shape)).astype(np.float32)
        y = rng.integers(0, n_classes, size=split_size).astype(np.int64)
        out_dir = DATA_PROCESSED / animal
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{split_name}_X.npy", X)
        np.save(out_dir / f"{split_name}_y.npy", y)
        print(f"  {split_name}: X={X.shape}, y={y.shape}")

    norm_stats = {"mean": -10.0, "std": 8.0, "fitted_on": "train"}
    with open(DATA_PROCESSED / animal / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)

    label_encoding = {name: i for i, name in enumerate(classes)}
    with open(DATA_PROCESSED / animal / "label_encoding.json", "w") as f:
        json.dump(label_encoding, f, indent=2)

    print(f"  norm_stats.json and label_encoding.json saved")
    print(f"  Done: {animal}")


if __name__ == "__main__":
    for animal in ["dog", "cat"]:
        print(f"\n=== {animal.upper()} ===")
        generate(animal)
