"""Approach C - AST (Audio Spectrogram Transformer, AudioSet-pretrained) as a
frozen AUDIO embedding extractor (dog + cat).

Usage (run on Google Colab, GPU T4 - see reports/ast_colab_cells.md):
    python src/ast_transfer.py --animal dog
    python src/ast_transfer.py --animal cat
    python src/ast_transfer.py --animal all   (default)

>>> THIS SCRIPT IS NOT MEANT TO RUN ON MY LOCAL MACHINE <<<
AST is too heavy for CPU-only local inference at a useful speed. It uses
torch.cuda if available and otherwise falls back to CPU without crashing, but
it is written, reviewed and committed here, then executed on Colab.

WHY AST
--------
Three independent levers on top of the frozen MobileNetV2 (ImageNet, image
backbone) features all plateaued at the same ceiling: head hyperparameter
tuning (tune_head.py), data augmentation (augment_cv.py), and classifier
family (compare_classifiers.py - logreg/SVM linear+RBF/LDA/dense head, none
beats the dense head, all within ~1 std of dog 0.8244 / cat 0.5223 macro-F1).
That convergence pointed at the FEATURES themselves as the bottleneck, not the
classifier on top. AST is an audio-NATIVE transformer pretrained (and
fine-tuned) on AudioSet, so its features were learned on sounds rather than
natural photographs - the hypothesis is that they separate vocalisation
classes (especially cat "food", the weakest class so far) better, without the
risk of fine-tuning a backbone.

CHECKPOINT
-----------
AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593" (HuggingFace,
transformers). This is the AST-Base checkpoint from the original AST paper
(Gong et al., 2021), fine-tuned on the full AudioSet (527 classes, mAP
0.4593 - the headline number in the model name and the highest-scoring AST
checkpoint released by the authors). I picked it because: (1) it is the
de-facto reference AST checkpoint used in the HuggingFace docs and most
downstream work, so it is well-documented and unlikely to have integration
surprises; (2) AudioSet's 527 classes include a wide range of animal sounds,
so its frozen representations should transfer better to dog/cat vocalisations
than an ImageNet-pretrained CNN; (3) AST-Base (12 layers, hidden_size=768) is
small enough to run comfortably on a single Colab T4 for ~550 short clips.

EMBEDDING EXTRACTION
----------------------
For each clip, I reuse preprocess.fix_length to center pad/crop the RAW 16 kHz
waveform to the SAME fixed duration as every other approach (4s dog / 2s cat),
so AST "sees" the same clips as YAMNet/MobileNetV2. I then run the AST
feature extractor (ASTFeatureExtractor, log-mel fbank with AudioSet-wide
mean/std + pad/truncate to a fixed number of frames - see
feature_extractor_info() below and the generated summary for the exact
numbers) followed by the frozen AST encoder (torch.no_grad(), model.eval(),
requires_grad=False everywhere), and mean-pool ASTModel's last_hidden_state
over the token sequence (CLS + distillation + patch tokens) to get ONE
hidden_size-dim (768) embedding per clip - the same "mean-pool over time/
tokens" idea as YAMNet's mean-pool over its 1024-dim frame embeddings.

ANTI-LEAKAGE (same guarantees as cross_validation.py / compare_classifiers.py)
---------------------------------------------------------------------------------
1. SAME CV folds/seed: cross_validation.make_folds is reused unchanged -
   StratifiedGroupKFold(k=4, group=cat_id) for cat, StratifiedKFold(k=5) for
   dog, both seed=42 (tl_common.SEED). Cat group violations are checked and
   asserted == 0 per fold, exactly as before.
2. AST embeddings are FOLD-INDEPENDENT: each clip is encoded independently by
   the frozen AST model (no batch statistics, no cross-sample normalisation -
   the feature extractor's mean/std are fixed AudioSet-wide constants baked
   into the pretrained checkpoint, not computed from my data). So I extract
   embeddings ONCE per animal, for ALL clips, and index into them per fold -
   identical argument and identical efficiency benefit as for YAMNet/
   MobileNetV2 in cross_validation.py.
3. The dense head (tl_common.train_head, unchanged) computes class_weight
   from y_train only. The logreg classifier is an sklearn
   Pipeline(StandardScaler, LogisticRegression); every fold calls
   sklearn.base.clone() to get a fresh, unfitted pipeline, fit on
   X[train_idx] only - StandardScaler.mean_/scale_ never see validation rows,
   exactly as in compare_classifiers.py.
4. TEST set touched ONCE at the end, with the CV-selected winner
   (dense_head vs logreg, chosen on mean CV macro-F1 only).

OUTPUTS (written by THIS script when it runs on Colab)
----------------------------------------------------------
- reports/ast_cv_scores.csv: one row per (animal, classifier, fold).
- reports/ast_summary.md: full write-up (English, 1st person) - checkpoint
  justification, anti-leakage section, AST vs MobileNet vs YAMNet comparison
  table, "food" focus, signal-vs-noise discussion, final test scores, and a
  conclusion on whether AST unblocks the problem or fine-tuning is next.
- reports/ast_<animal>_<classifier>_confusion_matrix.png
- models/ast_<animal>_<classifier>.{keras,joblib} (gitignored)
"""

from __future__ import annotations

import argparse
import time

import joblib
import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import ASTFeatureExtractor, ASTModel

from cross_validation import make_folds
from preprocess import fix_length
from tl_common import (
    CONFIGS,
    DATA_RAW,
    MODELS_DIR,
    REPORTS_DIR,
    SAMPLE_RATE,
    SEED,
    evaluate_and_plot,
    load_manifest,
    train_head,
)

AST_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_BATCH_SIZE = 16

# Reference numbers from earlier sessions (reports/cross_validation_summary.md
# and reports/classifier_comparison_summary.md), used for the explicit
# AST-vs-MobileNet-vs-YAMNet comparison. YAMNet's per-class "food" F1 was never
# computed in those runs (only accuracy/macro-F1), hence the None.
REFERENCE_RESULTS = {
    "dog": {
        "yamnet": {"accuracy": (0.7972, 0.0771), "macro_f1": (0.7986, 0.0740), "food_f1": None},
        "mobilenet": {"accuracy": (0.8229, 0.1109), "macro_f1": (0.8244, 0.1114), "food_f1": None},
    },
    "cat": {
        "yamnet": {"accuracy": (0.4056, 0.0913), "macro_f1": (0.3565, 0.0728), "food_f1": None},
        "mobilenet": {"accuracy": (0.5603, 0.1431), "macro_f1": (0.5223, 0.1338), "food_f1": (0.3646, 0.1896)},
    },
}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ast(device: torch.device) -> tuple[ASTFeatureExtractor, ASTModel]:
    """Load the frozen AST feature extractor + encoder, moved to `device`."""
    feature_extractor = ASTFeatureExtractor.from_pretrained(AST_CHECKPOINT)
    assert feature_extractor.sampling_rate == SAMPLE_RATE, (
        f"AST feature extractor expects {feature_extractor.sampling_rate} Hz audio, "
        f"but my pipeline uses {SAMPLE_RATE} Hz"
    )
    model = ASTModel.from_pretrained(AST_CHECKPOINT)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return feature_extractor, model


def feature_extractor_info(feature_extractor: ASTFeatureExtractor) -> dict:
    """Key fields of the pretrained feature extractor, for the summary report."""
    return {
        "sampling_rate": feature_extractor.sampling_rate,
        "num_mel_bins": feature_extractor.num_mel_bins,
        "max_length": feature_extractor.max_length,
        "do_normalize": feature_extractor.do_normalize,
        "mean": feature_extractor.mean,
        "std": feature_extractor.std,
    }


def extract_ast_embeddings(
    feature_extractor: ASTFeatureExtractor,
    model: ASTModel,
    df: pd.DataFrame,
    animal: str,
    device: torch.device,
) -> np.ndarray:
    """One hidden_size-dim (768) embedding per clip.

    Raw 16 kHz waveform -> preprocess.fix_length (same 4s dog / 2s cat
    centered pad/crop as every other approach) -> AST feature extractor
    (log-mel fbank, AudioSet-wide mean/std + pad/truncate to max_length
    frames) -> frozen AST encoder (torch.no_grad()) -> mean-pool
    last_hidden_state over the token sequence.
    """
    target_len = int(round(CONFIGS[animal]["duration_s"] * SAMPLE_RATE))
    waveforms = []
    for rel_path in df["path"]:
        y, _ = librosa.load(DATA_RAW / rel_path, sr=SAMPLE_RATE, mono=True)
        waveforms.append(fix_length(y, target_len))

    embeddings = []
    for i in range(0, len(waveforms), AST_BATCH_SIZE):
        batch = waveforms[i : i + AST_BATCH_SIZE]
        inputs = feature_extractor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        embeddings.append(out.last_hidden_state.mean(dim=1).cpu().numpy())
    return np.concatenate(embeddings, axis=0).astype(np.float32)


def build_logreg() -> Pipeline:
    """Same recipe as compare_classifiers.build_classifier("logreg")."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)),
        ]
    )


def eval_fold_sklearn(clf, X_train, y_train, X_val, y_val, n_classes, food_idx) -> dict:
    """Clone -> fit on fold-train -> predict on fold-val (StandardScaler fit
    fresh per fold via clone(), as in compare_classifiers.py)."""
    model = clone(clf)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
    food_f1 = np.nan
    if food_idx is not None:
        per_class = f1_score(y_val, y_pred, average=None, zero_division=0, labels=range(n_classes))
        food_f1 = per_class[food_idx]

    return {"accuracy": acc, "macro_f1": macro_f1, "food_f1": food_f1, "elapsed_s": elapsed}


def process_animal(
    animal: str,
    feature_extractor: ASTFeatureExtractor,
    model: ASTModel,
    device: torch.device,
    all_rows: list[dict],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict, list[dict]]:
    cfg = CONFIGS[animal]
    classes = cfg["classes"]
    n_classes = len(classes)
    food_idx = classes.index("food") if "food" in classes else None

    df = load_manifest(animal).reset_index(drop=True)
    label_to_idx = {name: i for i, name in enumerate(classes)}
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)

    print(f"\n=== {animal.upper()}: extracting AST embeddings for {len(df)} files ===")
    t0 = time.time()
    X = extract_ast_embeddings(feature_extractor, model, df, animal, device)
    extraction_time = time.time() - t0
    print(f"  AST embeddings: {X.shape} ({extraction_time:.1f}s, device={device})")
    extraction_info = {"shape": X.shape, "time_s": extraction_time, "device": str(device)}

    folds = make_folds(animal, df)
    strategy = "StratifiedGroupKFold (k=4, group=cat_id)" if animal == "cat" else "StratifiedKFold (k=5)"
    print(f"  {len(folds)}-fold CV ({strategy})")

    fold_info: list[dict] = []
    for fold_i, (train_idx, val_idx) in enumerate(folds):
        violations = None
        if animal == "cat":
            train_cats = set(df.loc[train_idx, "cat_id"])
            val_cats = set(df.loc[val_idx, "cat_id"])
            violations = len(train_cats & val_cats)
            print(
                f"  Fold {fold_i}: n_train={len(train_idx)} n_val={len(val_idx)} "
                f"train_cats={len(train_cats)} val_cats={len(val_cats)} group_violations={violations}"
            )
            assert violations == 0, f"cat_id leakage in fold {fold_i}"
            fold_info.append(
                {
                    "fold": fold_i,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "train_cats": len(train_cats),
                    "val_cats": len(val_cats),
                    "group_violations": violations,
                }
            )
        else:
            print(f"  Fold {fold_i}: n_train={len(train_idx)} n_val={len(val_idx)}")
            fold_info.append({"fold": fold_i, "n_train": len(train_idx), "n_val": len(val_idx)})

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # --- dense head (same architecture/defaults as every other approach) ---
        start = time.time()
        head_model, history = train_head(X_train, y_train, X_val, y_val, n_classes)
        elapsed = time.time() - start
        y_pred = head_model.predict(X_val, verbose=0).argmax(axis=1)
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        food_f1 = np.nan
        if food_idx is not None:
            per_class = f1_score(y_val, y_pred, average=None, zero_division=0, labels=range(n_classes))
            food_f1 = per_class[food_idx]
        all_rows.append(
            {
                "animal": animal,
                "classifier": "dense_head",
                "fold": fold_i,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "group_violations": violations,
                "epochs": len(history.history["loss"]),
                "accuracy": acc,
                "macro_f1": macro_f1,
                "food_f1": food_f1,
                "elapsed_s": elapsed,
            }
        )
        food_str = f" food_f1={food_f1:.4f}" if food_idx is not None else ""
        print(
            f"    dense_head: epochs={len(history.history['loss']):2d} "
            f"acc={acc:.4f} macro_f1={macro_f1:.4f}{food_str} ({elapsed:.1f}s)"
        )

        # --- logreg (Pipeline(StandardScaler, LogisticRegression), as in compare_classifiers.py) ---
        clf = build_logreg()
        res = eval_fold_sklearn(clf, X_train, y_train, X_val, y_val, n_classes, food_idx)
        all_rows.append(
            {
                "animal": animal,
                "classifier": "logreg",
                "fold": fold_i,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "group_violations": violations,
                "epochs": np.nan,
                **res,
            }
        )
        food_str = f" food_f1={res['food_f1']:.4f}" if food_idx is not None else ""
        print(
            f"    logreg    : acc={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f}"
            f"{food_str} ({res['elapsed_s']:.2f}s)"
        )

    return df, X, y, extraction_info, fold_info


def final_eval(animal: str, best_name: str, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    """Test set touched ONCE, with the CV-selected winner (dense_head vs logreg)."""
    n_classes = len(classes)
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    X_test, y_test = X[test_mask], y[test_mask]

    if best_name == "dense_head":
        model, history = train_head(X[train_mask], y[train_mask], X[val_mask], y[val_mask], n_classes)
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        model_path = MODELS_DIR / f"ast_{animal}_dense_head.keras"
        model.save(model_path)
        fit_info = f"train={int(train_mask.sum())} clips, early-stopped using val={int(val_mask.sum())} clips"
    else:
        trainval_mask = train_mask | val_mask
        clf = build_logreg()
        clf.fit(X[trainval_mask], y[trainval_mask])
        y_pred = clf.predict(X_test)
        model_path = MODELS_DIR / f"ast_{animal}_logreg.joblib"
        joblib.dump(clf, model_path)
        fit_info = f"train+val={int(trainval_mask.sum())} clips (no early stopping needed)"

    fig_path = REPORTS_DIR / f"ast_{animal}_{best_name}_confusion_matrix.png"
    result = evaluate_and_plot(
        y_test, y_pred, classes, f"AST - {animal} - confusion matrix (test, {best_name})", fig_path
    )
    result["model_path"] = model_path
    result["fit_info"] = fit_info
    result["n_test"] = int(test_mask.sum())

    food_idx = classes.index("food") if "food" in classes else None
    if food_idx is not None:
        per_class = f1_score(y_test, y_pred, average=None, zero_division=0, labels=range(n_classes))
        result["food_f1"] = per_class[food_idx]
    return result


# ---------------------------------------------------------------------------
# Summary report (reports/ast_summary.md) - generated at runtime on Colab.
# ---------------------------------------------------------------------------


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def compare_to_reference(ast_mean: float, ast_std: float, ref: tuple[float, float] | None) -> str:
    if ref is None:
        return "no reference value available"
    ref_mean, ref_std = ref
    delta = ast_mean - ref_mean
    combined_std = max(ast_std, ref_std)
    if abs(delta) < combined_std:
        verdict = "within noise (smaller than 1 std)"
    elif delta > 0:
        verdict = "a real improvement (more than 1 std)"
    else:
        verdict = "a real regression (more than 1 std)"
    return f"delta = {delta:+.4f} vs {fmt(ref_mean, ref_std)} -> {verdict}"


def write_summary_md(
    animals: list[str],
    summary: pd.DataFrame,
    best_per_animal: dict[str, str],
    final_results: dict[str, dict],
    extraction_info: dict[str, dict],
    fold_info: dict[str, list[dict]],
    fe_info: dict,
    device: torch.device,
    embed_dim: int,
    total_elapsed: float,
) -> None:
    lines: list[str] = []
    lines.append("# AST (Audio Spectrogram Transformer) as a frozen audio embedding extractor (dog + cat)\n")

    # --- Context ---
    lines.append("## Context and rationale\n")
    lines.append(
        "Three independent levers on top of the frozen MobileNetV2 (ImageNet, image "
        "backbone) features all plateaued at the same ceiling: dense-head "
        "hyperparameter tuning (`tune_head.py`), data augmentation "
        "(`augment_cv.py`), and classifier family (`compare_classifiers.py` - "
        "logistic regression / linear SVM / RBF SVM / LDA / dense head, none of "
        "which beat the dense head, all within roughly 1 std of dog "
        "0.8244 ± 0.1114 and cat 0.5223 ± 0.1338 macro-F1). That convergence "
        "pointed at the frozen MobileNetV2 features themselves as the bottleneck, "
        "not the classifier on top of them.\n"
    )
    lines.append(
        "In this session, I tried AST (Audio Spectrogram Transformer), an "
        "audio-NATIVE transformer pretrained and fine-tuned on AudioSet, as a "
        "frozen feature extractor in place of MobileNetV2. The hypothesis: "
        "features learned on sounds (including a wide range of animal "
        "vocalisations in AudioSet's 527 classes) should separate my "
        "bark/growl/grunt and brushing/food/isolation classes better than "
        "features learned on natural photographs - especially for cat \"food\", "
        "the weakest class so far (CV F1 0.3646 ± 0.1896, by far the "
        "highest-variance number in any of my results).\n"
    )

    # --- Checkpoint ---
    lines.append("## Checkpoint choice\n")
    lines.append(f"I used `{AST_CHECKPOINT}` (HuggingFace `transformers`).\n")
    lines.append(
        "This is the AST-Base checkpoint from the original AST paper (Gong et al., "
        "2021), fine-tuned on the full AudioSet (527 classes, mAP 0.4593 - the "
        "number in the checkpoint name, and the highest-scoring AST checkpoint the "
        "authors released). I picked it because: (1) it is the reference AST "
        "checkpoint used throughout the HuggingFace docs, so it is well "
        "documented and unlikely to have integration surprises; (2) AudioSet's "
        "527 classes cover many animal sounds, so its frozen representations "
        "should transfer better to dog/cat vocalisations than ImageNet features; "
        "(3) AST-Base is small enough to extract embeddings for all ~550 clips on "
        f"a single Colab T4 in a reasonable time (this run took "
        f"{total_elapsed:.1f}s total on {device}).\n"
    )
    lines.append(
        f"AST feature extractor config (read from the loaded checkpoint, not "
        f"hardcoded): sampling_rate={fe_info['sampling_rate']} Hz, "
        f"num_mel_bins={fe_info['num_mel_bins']}, max_length={fe_info['max_length']} "
        f"frames, do_normalize={fe_info['do_normalize']} "
        f"(mean={fe_info['mean']}, std={fe_info['std']}). AST encoder output: "
        f"{embed_dim}-dim embeddings (`last_hidden_state` mean-pooled over the "
        "token sequence).\n"
    )

    # --- Method ---
    lines.append("## Method\n")
    lines.append(
        "For each clip, I reuse `preprocess.fix_length` to center pad/crop the raw "
        "16 kHz waveform to the SAME fixed duration as every other approach (4s "
        "dog / 2s cat), so AST sees \"the same clips\" as YAMNet/MobileNetV2. I run "
        "the frozen AST feature extractor + encoder (`torch.no_grad()`, "
        "`model.eval()`, `requires_grad=False`) and mean-pool "
        f"`last_hidden_state` over the token sequence to get one {embed_dim}-dim "
        "embedding per clip - the same \"mean-pool\" idea as YAMNet's mean-pool "
        "over its 1024-dim frame embeddings.\n"
    )
    lines.append(
        "On top of these frozen AST embeddings, I evaluate two lightweight "
        "classifiers, reusing existing code unchanged: `tl_common.train_head` "
        "(the dense head, Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, "
        "softmax)) and `logreg` (`Pipeline(StandardScaler, "
        "LogisticRegression(class_weight=\"balanced\", max_iter=5000))`, the "
        "same recipe as `compare_classifiers.py`). Both are evaluated with the "
        "SAME CV folds and seed as every other approach via "
        "`cross_validation.make_folds` (unchanged).\n"
    )

    # --- Anti-leakage ---
    lines.append("## Anti-leakage checklist (as required)\n")
    lines.append(
        "1. **Same CV protocol/folds/seed as `cross_validation.py`**: confirmed - "
        "`cross_validation.make_folds` is imported and used unchanged "
        "(`StratifiedGroupKFold(k=4, group=cat_id)` for cat, "
        "`StratifiedKFold(k=5)` for dog, both `seed=42` via `tl_common.SEED`).\n"
    )
    lines.append(
        "2. **AST embeddings are fold-independent**: each clip is encoded "
        "independently by the frozen AST model under `torch.no_grad()` - no "
        "batch statistics, no cross-sample normalisation. The feature "
        "extractor's `mean`/`std` (reported above) are fixed AudioSet-wide "
        "constants baked into the pretrained checkpoint, NOT computed from my "
        "data, so they cannot leak information between my train/val/test splits. "
        "I therefore extract embeddings ONCE per animal, for ALL clips, and "
        "index into them per fold - identical argument and efficiency benefit as "
        "for YAMNet/MobileNetV2 in `cross_validation.py`.\n"
    )
    if "cat" in fold_info:
        lines.append("3. **Cat group violations, printed per fold - all 0**:\n")
        lines.append("   | Fold | n_train | n_val | train_cats | val_cats | group_violations |")
        lines.append("   |---|---|---|---|---|---|")
        for f in fold_info["cat"]:
            lines.append(
                f"   | {f['fold']} | {f['n_train']} | {f['n_val']} | "
                f"{f['train_cats']} | {f['val_cats']} | **{f['group_violations']}** |"
            )
        lines.append(
            "\n   No cat individual ever appears in both the train and validation "
            "part of a fold.\n"
        )
    lines.append(
        "4. **Scaler fit per fold (Pipeline + `clone()`)**: `logreg` is built as "
        "`Pipeline([(\"scaler\", StandardScaler()), (\"clf\", "
        "LogisticRegression(...))])`. For every fold, `sklearn.base.clone()` "
        "produces a brand-new, unfitted copy, then `.fit(X[train_idx], "
        "y[train_idx])` is called - so `StandardScaler.mean_`/`scale_` are "
        "computed from that fold's training rows ONLY. `class_weight=\"balanced\"` "
        "(dense head and logreg) is likewise computed from `y[train_idx]` only.\n"
    )
    lines.append(
        "5. **Test touched once**: for each animal, the classifier with the best "
        "mean CV macro-F1 (dense_head vs logreg) is selected on CV ONLY, then "
        "evaluated exactly once on the \"test\" split.\n"
    )

    # --- Results table ---
    lines.append("## Results: AST vs MobileNet vs YAMNet (CV mean ± std)\n")
    for animal in animals:
        cfg = CONFIGS[animal]
        food_idx_present = "food" in cfg["classes"]
        title = "Dog (5-fold StratifiedKFold)" if animal == "dog" else "Cat (4-fold StratifiedGroupKFold, group=cat_id)"
        lines.append(f"### {title}\n")
        header = "| Approach | accuracy | macro-F1 |"
        sep = "|---|---|---|"
        if food_idx_present:
            header += " food F1 |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)

        for clf_name in ["dense_head", "logreg"]:
            row = summary.loc[(animal, clf_name)]
            best_marker = "**" if clf_name == best_per_animal[animal] else ""
            line = (
                f"| {best_marker}ast_{clf_name}{best_marker} | "
                f"{fmt(row['accuracy_mean'], row['accuracy_std'])} | "
                f"{best_marker}{fmt(row['macro_f1_mean'], row['macro_f1_std'])}{best_marker} |"
            )
            if food_idx_present:
                line += f" {fmt(row['food_f1_mean'], row['food_f1_std'])} |"
            lines.append(line)

        ref = REFERENCE_RESULTS[animal]
        for ref_name, ref_vals in ref.items():
            line = (
                f"| {ref_name} (repere) | "
                f"{fmt(*ref_vals['accuracy'])} | "
                f"{fmt(*ref_vals['macro_f1'])} |"
            )
            if food_idx_present:
                food = ref_vals["food_f1"]
                line += f" {fmt(*food) if food else 'n/a'} |"
            lines.append(line)
        lines.append("")

    # --- AST vs MobileNet/YAMNet comparison ---
    lines.append("## Does AST beat the MobileNet / YAMNet repere?\n")
    for animal in animals:
        best_name = best_per_animal[animal]
        row = summary.loc[(animal, best_name)]
        lines.append(f"**{animal.upper()}** (best AST classifier: `{best_name}`)\n")
        lines.append(
            f"- macro-F1 {fmt(row['macro_f1_mean'], row['macro_f1_std'])} vs "
            f"MobileNet {compare_to_reference(row['macro_f1_mean'], row['macro_f1_std'], REFERENCE_RESULTS[animal]['mobilenet']['macro_f1'])}"
        )
        lines.append(
            f"- macro-F1 vs YAMNet "
            f"{compare_to_reference(row['macro_f1_mean'], row['macro_f1_std'], REFERENCE_RESULTS[animal]['yamnet']['macro_f1'])}"
        )
        if "food" in CONFIGS[animal]["classes"]:
            lines.append(
                f"- \"food\" F1 {fmt(row['food_f1_mean'], row['food_f1_std'])} vs "
                f"MobileNet {compare_to_reference(row['food_f1_mean'], row['food_f1_std'], REFERENCE_RESULTS[animal]['mobilenet']['food_f1'])}"
            )
        lines.append("")

    # --- Signal vs noise ---
    lines.append("## Signal vs noise\n")
    lines.append(
        "The deltas above are compared against `max(ast_std, reference_std)` - "
        "if the absolute difference is smaller than that, I call it \"within "
        "noise\", regardless of its sign. With only 4-5 CV folds, a single fold "
        "flipping can move macro-F1 by several hundredths, so a difference has "
        "to clear roughly one full standard deviation before I treat it as a "
        "real signal rather than fold-to-fold variance. The cat \"food\" class in "
        "particular has the highest variance of any number in this project "
        "(MobileNet: 0.3646 ± 0.1896, i.e. the 4 fold values range roughly from "
        "0.16 to 0.66) - any AST food-F1 result should be read with that in "
        "mind.\n"
    )

    # --- Final test evaluation ---
    lines.append("## Final test evaluation (touched once)\n")
    for animal in animals:
        best_name = best_per_animal[animal]
        result = final_results[animal]
        lines.append(
            f"**{animal.upper()}** - `{best_name}` (CV-selected) - fit on "
            f"{result['fit_info']}, n_test={result['n_test']}\n"
        )
        lines.append(f"- accuracy = **{result['accuracy']:.4f}**, macro-F1 = **{result['macro_f1']:.4f}**")
        if "food_f1" in result:
            lines.append(f"- food F1 = **{result['food_f1']:.4f}**")
        lines.append("")
        lines.append("```")
        lines.append(result["report"].rstrip())
        lines.append("```\n")
        lines.append(f"Confusion matrix: `{result['fig_path'].as_posix()}`\n")

    # --- Conclusion ---
    lines.append("## Conclusion: does AST unblock the problem?\n")
    verdicts = []
    for animal in animals:
        best_name = best_per_animal[animal]
        row = summary.loc[(animal, best_name)]
        ref_mac = REFERENCE_RESULTS[animal]["mobilenet"]["macro_f1"]
        delta = row["macro_f1_mean"] - ref_mac[0]
        combined_std = max(row["macro_f1_std"], ref_mac[1])
        verdicts.append((animal, delta, combined_std))

    food_verdict = None
    if "cat" in animals:
        best_name = best_per_animal["cat"]
        row = summary.loc[("cat", best_name)]
        ref_food = REFERENCE_RESULTS["cat"]["mobilenet"]["food_f1"]
        food_delta = row["food_f1_mean"] - ref_food[0]
        food_combined_std = max(row["food_f1_std"], ref_food[1])
        food_verdict = (food_delta, food_combined_std)

    any_real_gain = any(delta > combined_std for _, delta, combined_std in verdicts)
    food_real_gain = food_verdict is not None and food_verdict[0] > food_verdict[1]
    any_real_loss = any(delta < -combined_std for _, delta, combined_std in verdicts)

    if food_real_gain:
        lines.append(
            "**Yes, at least partially.** AST's CV \"food\" F1 improves on the "
            "MobileNet repere by more than 1 std - the one weak spot every "
            "previous lever failed to move. This is the strongest evidence so "
            "far that audio-native features carry information the image-based "
            "MobileNetV2 features did not. I would build on this AST-based "
            "pipeline next (e.g. revisit the classifier choice or pooling "
            "strategy on top of AST embeddings) before considering backbone "
            "fine-tuning.\n"
        )
    elif any_real_gain and not any_real_loss:
        lines.append(
            "**Partially.** AST's CV macro-F1 improves on the MobileNet repere "
            "by more than 1 std for at least one animal, even though \"food\" "
            "itself did not move beyond noise. This is still a meaningful signal "
            "that audio-native features help in general - I would keep "
            "iterating on the AST-based pipeline (classifier choice, pooling) "
            "rather than moving to backbone fine-tuning yet.\n"
        )
    elif any_real_loss and not any_real_gain:
        lines.append(
            "**No - AST performs worse here.** Its CV macro-F1 is below the "
            "MobileNet repere by more than 1 std for at least one animal, with "
            "no compensating gain elsewhere. A likely factor: my clips (4s dog / "
            "2s cat) are short compared to AST's 10-second training window "
            f"({fe_info['max_length']} frames), so a large fraction of the AST "
            "input is the feature extractor's own padding - diluting the actual "
            "signal. Given that AST does not help and MobileNetV2 plus three "
            "independent levers (head tuning, augmentation, classifier family) "
            "all plateaued at the same ceiling, **fine-tuning the MobileNetV2 "
            "backbone remains the strongest untried lever** - or, if fine-tuning "
            "is ruled out for this project, revisiting how short clips are fed "
            "to AST (e.g. tiling/repeating to better fill its 10s window) would "
            "be the next AST-specific experiment.\n"
        )
    else:
        lines.append(
            "**No - AST lands within noise of the MobileNet repere**, for both "
            "macro-F1 and (where applicable) \"food\" F1. Combined with the fact "
            "that MobileNetV2 plus three independent levers (head tuning, "
            "augmentation, classifier family) all plateaued at the same ceiling, "
            "this is now the FOURTH architecturally different approach landing in "
            "the same range. That convergence across such different feature "
            "spaces (ImageNet CNN, AudioSet CNN via YAMNet, AudioSet "
            "Transformer) suggests the ceiling may be closer to this dataset's "
            "irreducible difficulty (small dataset, especially for cat \"food\") "
            "than to any specific frozen backbone. **Fine-tuning the backbone "
            "remains the strongest untried lever** - it is the one approach "
            "where the feature extractor itself adapts to this exact data, "
            "rather than being used as-is.\n"
        )

    # --- Reproducibility / output files ---
    lines.append("## Reproducibility and output files\n")
    lines.append(f"- `seed=42` everywhere (`tl_common.SEED`), AST inference on {device}.")
    lines.append("- Same CV folds/seed as every other approach (`cross_validation.make_folds`, unchanged).")
    for animal in animals:
        info = extraction_info[animal]
        lines.append(
            f"- {animal}: AST embeddings shape {info['shape']}, extracted in "
            f"{info['time_s']:.1f}s on {info['device']}."
        )
    lines.append(f"- Total wall-clock time: {total_elapsed:.1f}s.")
    lines.append("- `reports/ast_cv_scores.csv` - one row per (animal, classifier, fold).")
    for animal in animals:
        best_name = best_per_animal[animal]
        lines.append(f"- `reports/ast_{animal}_{best_name}_confusion_matrix.png`")
        ext = "keras" if best_name == "dense_head" else "joblib"
        lines.append(f"- `models/ast_{animal}_{best_name}.{ext}` (gitignored)")

    (REPORTS_DIR / "ast_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSummary written to: {REPORTS_DIR / 'ast_summary.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    t_start = time.time()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading AST ({AST_CHECKPOINT}, frozen)...")
    feature_extractor, model = load_ast(device)
    fe_info = feature_extractor_info(feature_extractor)
    print(f"  feature extractor: {fe_info}")

    all_rows: list[dict] = []
    data_per_animal: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    extraction_info: dict[str, dict] = {}
    fold_info: dict[str, list[dict]] = {}
    embed_dim = None

    for animal in animals:
        df, X, y, ext_info, f_info = process_animal(animal, feature_extractor, model, device, all_rows)
        data_per_animal[animal] = (df, X, y)
        extraction_info[animal] = ext_info
        fold_info[animal] = f_info
        embed_dim = X.shape[1]

    df_scores = pd.DataFrame(all_rows)
    csv_path = REPORTS_DIR / "ast_cv_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    print(f"\nPer-fold CV scores saved to: {csv_path}")

    pd.set_option("display.width", 200)
    print("\n=== Mean +/- std across CV folds ===")
    summary = df_scores.groupby(["animal", "classifier"]).agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        food_f1_mean=("food_f1", "mean"),
        food_f1_std=("food_f1", "std"),
    )
    print(summary)

    print("\n=== Best AST classifier per animal (selected on CV macro-F1) ===")
    best_per_animal: dict[str, str] = {}
    for animal in animals:
        candidates = summary.loc[animal]
        best_name = candidates["macro_f1_mean"].idxmax()
        best_per_animal[animal] = best_name
        print(f"  {animal}: {best_name}, CV macro_f1={candidates.loc[best_name, 'macro_f1_mean']:.4f}")

    print("\n=== Final test evaluation (test set touched ONCE per animal) ===")
    final_results: dict[str, dict] = {}
    for animal in animals:
        df, X, y = data_per_animal[animal]
        result = final_eval(animal, best_per_animal[animal], df, X, y, CONFIGS[animal]["classes"])
        final_results[animal] = result
        print(f"\n{animal.upper()} - {best_per_animal[animal]} - fit on {result['fit_info']}, n_test={result['n_test']}")
        print(f"  accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
        if "food_f1" in result:
            print(f"  food_f1={result['food_f1']:.4f}")
        print(result["report"])
        print(f"  Confusion matrix saved to: {result['fig_path']}")
        print(f"  Model saved to: {result['model_path']}")

    total_elapsed = time.time() - t_start
    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s")

    write_summary_md(
        animals,
        summary,
        best_per_animal,
        final_results,
        extraction_info,
        fold_info,
        fe_info,
        device,
        embed_dim,
        total_elapsed,
    )


if __name__ == "__main__":
    main()
