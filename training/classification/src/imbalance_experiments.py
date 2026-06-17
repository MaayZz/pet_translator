"""Evaluate focal loss and SMOTE against the cross-entropy baseline.

Features: frozen MobileNetV2 (ImageNet, pooling=avg), precomputed once and
reused across all folds and conditions — no leakage possible from feature
extraction. Same CV protocol as cross_validation.py.

Conditions:
  A. BASELINE    — sparse_categorical_crossentropy + class_weight='balanced'
  B. FOCAL       — focal loss (gamma=2.0), no class_weight
  C. SMOTE       — cross-entropy + class_weight, SMOTE on X_train per fold only
  D. SMOTE+FOCAL — focal loss + SMOTE on X_train per fold (no class_weight)

Usage (from training/classification/):
    python src/imbalance_experiments.py
"""

from __future__ import annotations

import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from mobilenet_transfer import build_backbone, spectrograms_to_images
from preprocess import extract_logmel, fix_length
from tl_common import (
    CONFIGS,
    REPORTS_DIR,
    SAMPLE_RATE,
    SEED,
    build_head,
    class_weight_dict,
    load_manifest,
)

N_FOLDS = {"dog": 5, "cat": 4}
GAMMA_FOCAL = 2.0
SMOTE_STRATEGY = "not majority"  # bring minority classes to majority count within fold
SMOTE_K = 5


# ---------------------------------------------------------------------------
# Feature extraction (identical to cross_validation.py)
# ---------------------------------------------------------------------------

def extract_mobilenet_features(df: pd.DataFrame, animal: str, backbone) -> np.ndarray:
    target_len = int(round(CONFIGS[animal]["duration_s"] * SAMPLE_RATE))
    logmels = []
    for rel in df["path"]:
        y, _ = librosa.load(
            Path(__file__).resolve().parent.parent / "data" / "raw" / rel,
            sr=SAMPLE_RATE, mono=True,
        )
        logmels.append(extract_logmel(fix_length(y, target_len)))
    logmel_arr = np.stack(logmels).astype(np.float32)
    return backbone.predict(spectrograms_to_images(logmel_arr), verbose=0)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss_fn(gamma: float = GAMMA_FOCAL):
    """Multi-class focal loss for integer (sparse) labels.

    FL(pt) = -(1 - pt)^gamma * log(pt)
    No class_weight: the modulating factor (1-pt)^gamma already down-weights
    easy (well-classified) examples and focuses training on hard/rare ones.
    Combining with class_weight would double-count the imbalance correction.
    """
    def loss(y_true, y_pred):
        n_cls = tf.shape(y_pred)[-1]
        y_true_oh = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), n_cls)
        pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
        ce = -tf.math.log(tf.clip_by_value(pt, 1e-7, 1.0))
        return tf.reduce_mean((1.0 - pt) ** gamma * ce)
    return loss


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def train(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    loss_fn,
    use_class_weight: bool,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    tf.keras.utils.set_random_seed(SEED)
    model = build_head(X_tr.shape[1], n_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cw = class_weight_dict(y_tr) if use_class_weight else None
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=50, batch_size=8, verbose=0,
        class_weight=cw, callbacks=[es],
    )
    return model, history


# ---------------------------------------------------------------------------
# SMOTE resampling (train-only)
# ---------------------------------------------------------------------------

def smote_resample(X_tr: np.ndarray, y_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to X_tr/y_tr only. Validation data is never touched.

    sampling_strategy='not majority' brings all non-majority classes to the
    majority class count within this fold's training set.
    n_neighbors=5 (default) is sufficient given dataset sizes.
    """
    sm = SMOTE(sampling_strategy=SMOTE_STRATEGY, k_neighbors=SMOTE_K, random_state=SEED)
    return sm.fit_resample(X_tr, y_tr)


# ---------------------------------------------------------------------------
# CV loop
# ---------------------------------------------------------------------------

CONDITION_CFG = {
    "BASELINE":    {"smote": False, "focal": False, "use_cw": True},
    "FOCAL":       {"smote": False, "focal": True,  "use_cw": False},
    "SMOTE":       {"smote": True,  "focal": False, "use_cw": True},
    "SMOTE+FOCAL": {"smote": True,  "focal": True,  "use_cw": False},
}


def make_folds(animal: str, df: pd.DataFrame):
    if animal == "cat":
        cv = StratifiedGroupKFold(n_splits=N_FOLDS["cat"], shuffle=True, random_state=SEED)
        return list(cv.split(df, df["label"], groups=df["cat_id"]))
    cv = StratifiedKFold(n_splits=N_FOLDS["dog"], shuffle=True, random_state=SEED)
    return list(cv.split(df, df["label"]))


def run_condition(
    condition: str,
    animal: str,
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
    folds: list,
) -> list[dict]:
    cfg_c = CONDITION_CFG[condition]
    cfg_a = CONFIGS[animal]
    n_classes = len(cfg_a["classes"])
    food_idx = cfg_a["classes"].index("food") if "food" in cfg_a["classes"] else None
    loss_fn = focal_loss_fn() if cfg_c["focal"] else "sparse_categorical_crossentropy"

    rows = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        violations = None
        if animal == "cat":
            tc = set(df.loc[train_idx, "cat_id"])
            vc = set(df.loc[val_idx, "cat_id"])
            violations = len(tc & vc)
            assert violations == 0, f"cat_id leakage fold {fold_i} [{condition}]"

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        smote_n_added = 0
        if cfg_c["smote"]:
            X_tr_before = len(X_tr)
            X_tr, y_tr = smote_resample(X_tr, y_tr)
            smote_n_added = len(X_tr) - X_tr_before

        model, history = train(
            X_tr, y_tr, X_val, y_val, n_classes, loss_fn, cfg_c["use_cw"]
        )
        y_pred = model.predict(X_val, verbose=0).argmax(axis=1)

        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        food_f1 = food_prec = food_rec = None
        if food_idx is not None:
            per_cls_f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
            per_cls_prec = precision_score(y_val, y_pred, average=None, zero_division=0)
            per_cls_rec = recall_score(y_val, y_pred, average=None, zero_division=0)
            food_f1 = float(per_cls_f1[food_idx])
            food_prec = float(per_cls_prec[food_idx])
            food_rec = float(per_cls_rec[food_idx])

        rows.append(
            {
                "animal": animal,
                "condition": condition,
                "fold": fold_i,
                "n_train_orig": len(train_idx),
                "n_train_after_smote": len(X_tr),
                "smote_added": smote_n_added,
                "n_val": len(val_idx),
                "group_violations": violations,
                "epochs": len(history.history["loss"]),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "food_f1": food_f1,
                "food_precision": food_prec,
                "food_recall": food_rec,
            }
        )

        food_str = (
            f"  food: F1={food_f1:.4f} P={food_prec:.4f} R={food_rec:.4f}"
            if food_f1 is not None else ""
        )
        smote_str = f"  smote+{smote_n_added}" if smote_n_added else ""
        viol_str = f"  gv={violations}" if violations is not None else ""
        print(
            f"      fold {fold_i}: acc={acc:.4f} macro_f1={macro_f1:.4f}"
            + food_str + smote_str + viol_str
        )

    return rows


def process_animal(animal: str, backbone) -> list[dict]:
    df = load_manifest(animal).reset_index(drop=True)
    cfg = CONFIGS[animal]
    label_to_idx = {name: i for i, name in enumerate(cfg["classes"])}
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)

    print(f"\n=== {animal.upper()} ({len(df)} files) ===")
    print("  Extracting MobileNetV2 features (once) ...")
    t0 = time.time()
    X = extract_mobilenet_features(df, animal, backbone)
    print(f"  features: {X.shape}  ({time.time() - t0:.1f}s)")

    folds = make_folds(animal, df)

    all_rows: list[dict] = []
    for condition in CONDITION_CFG:
        print(f"\n  [{condition}] {N_FOLDS[animal]}-fold CV ...")
        t0 = time.time()
        rows = run_condition(condition, animal, X, y, df, folds)
        elapsed = time.time() - t0

        f1_vals = np.array([r["macro_f1"] for r in rows])
        acc_vals = np.array([r["accuracy"] for r in rows])
        print(
            f"  [{condition}]  macro-F1 = {f1_vals.mean():.4f} +/- {f1_vals.std():.4f} | "
            f"acc = {acc_vals.mean():.4f} +/- {acc_vals.std():.4f}  ({elapsed:.1f}s)"
        )
        if animal == "cat":
            food_f1v = np.array([r["food_f1"] for r in rows])
            food_pv = np.array([r["food_precision"] for r in rows])
            food_rv = np.array([r["food_recall"] for r in rows])
            print(
                f"  [{condition}]  food F1={food_f1v.mean():.4f} +/- {food_f1v.std():.4f}  "
                f"P={food_pv.mean():.4f} +/- {food_pv.std():.4f}  "
                f"R={food_rv.mean():.4f} +/- {food_rv.std():.4f}"
            )

        all_rows.extend(rows)

    return all_rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame, animal: str, condition: str) -> dict:
    sub = df[(df["animal"] == animal) & (df["condition"] == condition)]
    s: dict = {
        "macro_f1_mean": sub["macro_f1"].mean(),
        "macro_f1_std": sub["macro_f1"].std(),
        "acc_mean": sub["accuracy"].mean(),
        "acc_std": sub["accuracy"].std(),
    }
    if "food_f1" in sub.columns and sub["food_f1"].notna().any():
        s["food_f1_mean"] = sub["food_f1"].mean()
        s["food_f1_std"] = sub["food_f1"].std()
        s["food_prec_mean"] = sub["food_precision"].mean()
        s["food_prec_std"] = sub["food_precision"].std()
        s["food_rec_mean"] = sub["food_recall"].mean()
        s["food_rec_std"] = sub["food_recall"].std()
    if "smote_added" in sub.columns:
        s["smote_added_mean"] = sub["smote_added"].mean()
    violations = sub["group_violations"].dropna()
    s["total_violations"] = int(violations.sum()) if len(violations) else 0
    return s


def verdict(delta: float, std_ref: float) -> str:
    ratio = abs(delta) / std_ref if std_ref > 0 else float("nan")
    direction = "improves" if delta > 0 else "degrades"
    sig = "signal" if ratio > 1.0 else "noise"
    return f"{delta:+.4f} ({ratio:.2f}x std -> {sig}, {direction})"


def write_report(df_scores: pd.DataFrame, total_elapsed: float) -> None:
    conditions = list(CONDITION_CFG.keys())
    animals = ["dog", "cat"]
    stats = {
        (a, c): compute_stats(df_scores, a, c)
        for a in animals for c in conditions
    }

    lines = []
    lines.append("# Class Imbalance Experiments — Focal Loss & SMOTE\n")
    lines.append(
        "I tested two class-imbalance techniques — focal loss and SMOTE — against "
        "the cross-entropy baseline on the production MobileNetV2 frozen features. "
        "The evaluation protocol is identical to all previous CV runs. "
        "This is the fifth and final attempt to improve on the production benchmark.\n"
    )

    lines.append("## Techniques and Parameters\n")
    lines.append(
        "**Focal Loss (condition B)**: "
        f"gamma={GAMMA_FOCAL} (standard value from Lin et al. 2017). "
        "I do *not* combine focal loss with class_weight, because the modulating "
        f"factor (1-pt)^{GAMMA_FOCAL} already down-weights easy/frequent examples "
        "implicitly — adding class_weight on top would double-count the imbalance "
        "correction and risk over-penalising the minority class.\n\n"
        f"**SMOTE (condition C)**: `sampling_strategy='{SMOTE_STRATEGY}'`, "
        f"`k_neighbors={SMOTE_K}`. Within each fold's training set, SMOTE brings all "
        "non-majority classes up to the majority class count. I keep class_weight='balanced' "
        "with SMOTE: after resampling the data is near-balanced, so computed weights "
        "are close to 1.0 and do not distort training.\n\n"
        "**SMOTE+FOCAL (condition D)**: SMOTE resampling + focal loss (no class_weight). "
        "Tests whether the two techniques compound.\n"
    )

    lines.append("## Anti-Leakage Verification\n")
    cat_viol = sum(stats[("cat", c)]["total_violations"] for c in conditions)
    lines.append(
        f"- **Cat group leakage**: {cat_viol} cat_id violation(s) across all folds "
        "and all conditions (expected 0 — confirmed).\n"
        "- **SMOTE train-only**: SMOTE is applied inside the CV loop, exclusively on "
        "`X[train_idx]`. The validation set always contains only real, original samples. "
        "For cat, the group-safe fold split guarantees that no cat_id from the validation "
        "fold appears in the training set; therefore synthetic samples generated by SMOTE "
        "can only be interpolations of training-fold cats, and carry no individual "
        "information from validation cats.\n"
        "- **Feature extraction**: per-sample min-max normalisation in "
        "`spectrograms_to_images` uses each clip's own values only — no cross-sample "
        "statistic leaks across folds.\n"
    )

    # --- Dog table ---
    lines.append("\n## Results — Dog (StratifiedKFold k=5)\n")
    lines.append("| Condition | Macro-F1 (mean+/-std) | Accuracy (mean+/-std) |")
    lines.append("|-----------|----------------------|----------------------|")
    for c in conditions:
        s = stats[("dog", c)]
        lines.append(
            f"| {c} | {s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f} "
            f"| {s['acc_mean']:.4f} +/- {s['acc_std']:.4f} |"
        )
    lines.append("")

    lines.append("### Dog Deltas vs Baseline\n")
    ref = stats[("dog", "BASELINE")]
    for c in conditions[1:]:
        s = stats[("dog", c)]
        d = s["macro_f1_mean"] - ref["macro_f1_mean"]
        lines.append(f"- **{c}**: {verdict(d, ref['macro_f1_std'])}\n")

    # --- Cat table ---
    lines.append("\n## Results — Cat (StratifiedGroupKFold k=4, group=cat_id)\n")
    lines.append(
        "| Condition | Macro-F1 (mean+/-std) | Accuracy (mean+/-std) "
        "| Food F1 (mean+/-std) | Food Precision | Food Recall |"
    )
    lines.append(
        "|-----------|----------------------|----------------------"
        "|---------------------|----------------|-------------|"
    )
    for c in conditions:
        s = stats[("cat", c)]
        food_str = (
            f"{s['food_f1_mean']:.4f} +/- {s['food_f1_std']:.4f}"
            if "food_f1_mean" in s else "n/a"
        )
        food_p = f"{s['food_prec_mean']:.4f} +/- {s['food_prec_std']:.4f}" if "food_prec_mean" in s else "n/a"
        food_r = f"{s['food_rec_mean']:.4f} +/- {s['food_rec_std']:.4f}" if "food_rec_mean" in s else "n/a"
        lines.append(
            f"| {c} | {s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f} "
            f"| {s['acc_mean']:.4f} +/- {s['acc_std']:.4f} "
            f"| {food_str} | {food_p} | {food_r} |"
        )
    lines.append("")

    lines.append("### Cat Deltas vs Baseline\n")
    ref_cat = stats[("cat", "BASELINE")]
    for c in conditions[1:]:
        s = stats[("cat", c)]
        d_f1 = s["macro_f1_mean"] - ref_cat["macro_f1_mean"]
        d_food = (
            s.get("food_f1_mean", 0) - ref_cat.get("food_f1_mean", 0)
            if "food_f1_mean" in s and "food_f1_mean" in ref_cat else None
        )
        lines.append(f"- **{c}** macro-F1: {verdict(d_f1, ref_cat['macro_f1_std'])}")
        if d_food is not None:
            lines.append(
                f"  food F1: {d_food:+.4f} "
                f"({abs(d_food)/ref_cat['food_f1_std']:.2f}x std(RAW) -> "
                f"{'signal' if abs(d_food)/ref_cat['food_f1_std'] > 1.0 else 'noise'})\n"
            )
        else:
            lines.append("")

    # --- Benchmark ---
    lines.append("\n## Benchmark Reproduction\n")
    dog_b = stats[("dog", "BASELINE")]
    cat_b = stats[("cat", "BASELINE")]
    lines.append(
        f"- Dog BASELINE macro-F1: {dog_b['macro_f1_mean']:.4f} +/- {dog_b['macro_f1_std']:.4f} "
        f"(reference: 0.8244 +/- 0.1114)\n"
        f"- Cat BASELINE macro-F1: {cat_b['macro_f1_mean']:.4f} +/- {cat_b['macro_f1_std']:.4f} "
        f"(reference: 0.5223 +/- 0.1338)\n"
    )

    # --- Verdict final ---
    lines.append("## Final Verdict\n")
    all_noise = True
    for animal in animals:
        ref_s = stats[(animal, "BASELINE")]
        for c in conditions[1:]:
            s = stats[(animal, c)]
            ratio = abs(s["macro_f1_mean"] - ref_s["macro_f1_mean"]) / ref_s["macro_f1_std"]
            if ratio > 1.0:
                all_noise = False
                lines.append(
                    f"- **{animal.capitalize()} / {c}**: delta > 1 std — "
                    f"potentially meaningful ({ratio:.2f}x std). Inspect carefully.\n"
                )

    if all_noise:
        lines.append(
            "No condition beats the baseline by more than one standard deviation on "
            "either animal. Every delta is within the noise of fold variance.\n\n"
            "**The production model remains unchanged: MobileNetV2 frozen backbone + "
            "dense head + cross-entropy + class_weight='balanced'. "
            "I do not recommend replacing it.**\n"
        )
    else:
        lines.append(
            "At least one condition shows a delta > 1 std. Review the table carefully "
            "before deciding whether to replace the production model. "
            "I have not modified predict.py or any production file — await instructions.\n"
        )

    lines.append(f"\n---\n_Total wall-clock time: {total_elapsed:.0f}s_\n")

    report_path = REPORTS_DIR / "imbalance_experiments_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading MobileNetV2 (ImageNet weights, frozen backbone) ...")
    backbone = build_backbone()

    all_rows: list[dict] = []
    for animal in ["dog", "cat"]:
        all_rows.extend(process_animal(animal, backbone))

    df_scores = pd.DataFrame(all_rows)
    col_order = [
        "animal", "condition", "fold", "n_train_orig", "n_train_after_smote",
        "smote_added", "n_val", "group_violations", "epochs",
        "accuracy", "macro_f1", "food_f1", "food_precision", "food_recall",
    ]
    df_scores = df_scores[[c for c in col_order if c in df_scores.columns]]
    csv_path = REPORTS_DIR / "imbalance_experiments_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    print(f"\nPer-fold scores saved to: {csv_path}")

    print("\n=== Summary ===")
    num_cols = ["accuracy", "macro_f1"]
    if "food_f1" in df_scores.columns:
        num_cols += ["food_f1"]
    summary = df_scores.groupby(["animal", "condition"])[num_cols].agg(["mean", "std"])
    print(summary.to_string())

    total_elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    write_report(df_scores, total_elapsed)


if __name__ == "__main__":
    main()
