"""Compare classical classifiers against the dense head, all trained on the
SAME frozen MobileNetV2 features (1280-dim), for dog and cat.

Usage:
    python src/compare_classifiers.py --animal dog
    python src/compare_classifiers.py --animal cat
    python src/compare_classifiers.py --animal all   (default)

WHY THIS COMPARISON
--------------------
Every approach tried so far (cross_validation.py, tune_head.py, augment_cv.py)
put a small DENSE HEAD (Dense(64, relu) -> Dropout -> Dense(n_classes,
softmax)) on top of the frozen MobileNetV2 features. Two levers (head
hyperparameters, data augmentation) both gave gains smaller than the CV
standard deviation - i.e. within noise.

Our actual regime is 1280-dim features with very few training samples per
fold (~79-91 for dog, ~310-352 for cat). That is exactly the "many features,
few samples" regime where regularized linear/margin classifiers (logistic
regression, linear SVM, LDA) are often reported to generalize at least as
well as a small neural net, and sometimes better. This script tests that
hypothesis directly: same features, same folds, same seed, only the
classifier on top changes.

PROTOCOL REUSE (no duplication)
---------------------------------
- cross_validation.extract_logmel_batch: raw (non-normalised) log-mel
  spectrograms, fold-independent (see that module's docstring for why no
  cross-sample normalisation is involved).
- mobilenet_transfer.build_backbone / spectrograms_to_images: frozen
  MobileNetV2 (ImageNet, pooling="avg"), per-sample min-max rescale -> 1280-dim
  features. Computed ONCE per animal, for ALL clips, then indexed per fold -
  exactly as in cross_validation.py.
- cross_validation.make_folds: StratifiedGroupKFold(k=4, group=cat_id) for
  cat, StratifiedKFold(k=5) for dog, both seed=42 (via tl_common.SEED).
- tl_common.train_head: the dense head, UNCHANGED defaults - this is run
  first in every fold to re-verify it reproduces the existing CV repere
  (dog ~0.8244, cat ~0.5223) before any new classifier is even considered.

CLASSIFIERS COMPARED (all on the same 1280-dim MobileNet features)
---------------------------------------------------------------------
- dummy_floor : DummyClassifier(strategy="most_frequent") - sanity floor.
- logreg      : LogisticRegression(class_weight="balanced", max_iter=5000).
- svm_linear  : SVC(kernel="linear", class_weight="balanced").
- svm_rbf     : SVC(kernel="rbf", class_weight="balanced"), with a small
                C x gamma grid (RBF_C_GRID x RBF_GAMMA_GRID) explored via the
                SAME CV folds; the (C, gamma) with the best MEAN CV macro-F1
                is reported as "svm_rbf" (selection on CV only, never test).
- lda         : LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").
                "auto" shrinkage is required here: with 1280 features and a
                few hundred samples, the plain (unshrunk) within-class
                covariance matrix would be singular. LDA has no class_weight
                parameter in scikit-learn - this is the one classifier here
                that does NOT get explicit class-balancing (documented as a
                known difference in the summary report).
- dense_head  : tl_common.train_head, unchanged - the existing reference.

ANTI-LEAKAGE: SCALER FIT PER FOLD
-------------------------------------
logreg / svm_linear / svm_rbf / lda are each an sklearn
Pipeline(StandardScaler(), classifier). For every fold, sklearn.base.clone()
produces a FRESH, UNFITTED copy of that pipeline, and .fit(X[train_idx],
y[train_idx]) is called - so StandardScaler's mean_/scale_ are computed from
that fold's training rows ONLY. .predict(X[val_idx]) then only APPLIES the
already-fitted transform (no re-fitting on validation data). The MobileNet
features themselves are fold-independent (see cross_validation.py docstring),
so the only fold-dependent statistic anywhere in this script is this
per-fold StandardScaler fit (plus class_weight="balanced", computed from
y[train_idx] only, exactly as before).

For cat, every fold's train/val cat_id sets are checked for overlap
(group_violations) and asserted to be 0, exactly as in cross_validation.py.

FINAL EVALUATION (test set touched once)
--------------------------------------------
For each animal, the classifier with the BEST mean CV macro-F1 (dummy_floor
excluded - it is a floor, not a candidate) is re-fit on the manifest's
"train"+"val" rows combined (sklearn classifiers need no validation split for
early stopping) - or, if the dense head wins, on "train" with early stopping
on "val", exactly as in mobilenet_transfer.py. It is then evaluated ONCE on
the "test" split: accuracy, macro-F1, per-class report, confusion matrix PNG.
Selection of the winning classifier (and of the RBF C/gamma) is based ONLY on
CV macro-F1 - the test set plays no role in any selection.

OUTPUTS
--------
- reports/classifier_comparison_cv_scores.csv: one row per (animal,
  classifier, fold).
- reports/classifier_comparison_svm_rbf_grid.csv: one row per (animal, fold,
  C, gamma) for the full RBF grid search.
- reports/classifier_comparison_summary_table.csv: mean +/- std per (animal,
  classifier).
- reports/best_classifier_<animal>_<classifier>[_C..._gamma...]_confusion_matrix.png
- models/best_classifier_<animal>_<classifier>.{keras,joblib} (gitignored).
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from cross_validation import extract_logmel_batch, make_folds
from mobilenet_transfer import build_backbone, spectrograms_to_images
from tl_common import CONFIGS, MODELS_DIR, REPORTS_DIR, SEED, evaluate_and_plot, load_manifest, train_head

RBF_C_GRID = [0.1, 1.0, 10.0]
RBF_GAMMA_GRID = ["scale", 0.001, 0.01]

SKLEARN_CLASSIFIER_NAMES = ["dummy_floor", "logreg", "svm_linear", "lda"]


def build_classifier(name: str, params: dict | None = None):
    """Fresh, unfitted classifier (or Pipeline) for `name`. `params` only used
    for "svm_rbf" (C, gamma)."""
    params = params or {}
    if name == "dummy_floor":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=SEED)),
            ]
        )
    if name == "svm_linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", class_weight="balanced", random_state=SEED)),
            ]
        )
    if name == "svm_rbf":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=params["C"],
                        gamma=params["gamma"],
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if name == "lda":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )
    raise ValueError(name)


def eval_fold_sklearn(clf, X_train, y_train, X_val, y_val, n_classes, food_idx) -> dict:
    """Clone -> fit on fold-train -> predict on fold-val. The clone() call is
    what guarantees the Pipeline's StandardScaler is fit fresh on this fold's
    training rows only."""
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


def process_animal(animal: str, backbone, all_rows: list[dict], rbf_grid_rows: list[dict]):
    cfg = CONFIGS[animal]
    classes = cfg["classes"]
    n_classes = len(classes)
    food_idx = classes.index("food") if "food" in classes else None

    df = load_manifest(animal).reset_index(drop=True)
    label_to_idx = {name: i for i, name in enumerate(classes)}
    y = df["label"].map(label_to_idx).to_numpy(dtype=np.int64)

    print(f"\n=== {animal.upper()}: precomputing MobileNetV2 features for {len(df)} files ===")
    t0 = time.time()
    logmel = extract_logmel_batch(df, animal)
    X = backbone.predict(spectrograms_to_images(logmel), verbose=0)
    print(f"  MobileNetV2 features: {X.shape} ({time.time() - t0:.1f}s)")

    folds = make_folds(animal, df)
    strategy = "StratifiedGroupKFold (k=4, group=cat_id)" if animal == "cat" else "StratifiedKFold (k=5)"
    print(f"  {len(folds)}-fold CV ({strategy})")

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
        else:
            print(f"  Fold {fold_i}: n_train={len(train_idx)} n_val={len(val_idx)}")

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # --- dense head (repere re-verification) ---
        start = time.time()
        model, history = train_head(X_train, y_train, X_val, y_val, n_classes)
        elapsed = time.time() - start
        y_pred = model.predict(X_val, verbose=0).argmax(axis=1)
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
                "config": "",
                "accuracy": acc,
                "macro_f1": macro_f1,
                "food_f1": food_f1,
                "elapsed_s": elapsed,
            }
        )
        food_str = f" food_f1={food_f1:.4f}" if food_idx is not None else ""
        print(
            f"    dense_head : epochs={len(history.history['loss']):2d} "
            f"acc={acc:.4f} macro_f1={macro_f1:.4f}{food_str} ({elapsed:.1f}s)"
        )

        # --- sklearn classifiers (Pipeline fit per fold via clone) ---
        for name in SKLEARN_CLASSIFIER_NAMES:
            clf = build_classifier(name)
            res = eval_fold_sklearn(clf, X_train, y_train, X_val, y_val, n_classes, food_idx)
            all_rows.append(
                {
                    "animal": animal,
                    "classifier": name,
                    "fold": fold_i,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "group_violations": violations,
                    "epochs": np.nan,
                    "config": "",
                    **res,
                }
            )
            food_str = f" food_f1={res['food_f1']:.4f}" if food_idx is not None else ""
            print(
                f"    {name:11s}: acc={res['accuracy']:.4f} macro_f1={res['macro_f1']:.4f}"
                f"{food_str} ({res['elapsed_s']:.2f}s)"
            )

        # --- SVM RBF grid (C x gamma), recorded for the grid CSV ---
        for C in RBF_C_GRID:
            for gamma in RBF_GAMMA_GRID:
                clf = build_classifier("svm_rbf", {"C": C, "gamma": gamma})
                res = eval_fold_sklearn(clf, X_train, y_train, X_val, y_val, n_classes, food_idx)
                rbf_grid_rows.append(
                    {
                        "animal": animal,
                        "fold": fold_i,
                        "n_train": len(train_idx),
                        "n_val": len(val_idx),
                        "group_violations": violations,
                        "C": C,
                        "gamma": gamma,
                        **res,
                    }
                )

    return df, X, y


def select_best_rbf(animal: str, rbf_grid_rows: list[dict]) -> dict:
    """Pick (C, gamma) with the best MEAN CV macro-F1 across folds for this
    animal - selection on CV only."""
    rbf_df = pd.DataFrame([r for r in rbf_grid_rows if r["animal"] == animal])
    pivot = rbf_df.groupby(["C", "gamma"])["macro_f1"].agg(["mean", "std"]).reset_index()
    print(f"\n  SVM RBF grid for {animal} (mean +/- std CV macro-F1):")
    for _, row in pivot.iterrows():
        print(f"    C={row['C']:<6} gamma={str(row['gamma']):<6}: {row['mean']:.4f} +/- {row['std']:.4f}")
    best = pivot.loc[pivot["mean"].idxmax()]
    print(f"  -> best: C={best['C']}, gamma={best['gamma']} (mean CV macro_f1={best['mean']:.4f})")
    return {"C": best["C"], "gamma": best["gamma"]}


def best_rbf_rows(animal: str, rbf_grid_rows: list[dict], best_C: float, best_gamma) -> list[dict]:
    rows = []
    for r in rbf_grid_rows:
        if r["animal"] == animal and r["C"] == best_C and r["gamma"] == best_gamma:
            rows.append(
                {
                    "animal": animal,
                    "classifier": "svm_rbf",
                    "fold": r["fold"],
                    "n_train": r["n_train"],
                    "n_val": r["n_val"],
                    "group_violations": r["group_violations"],
                    "epochs": np.nan,
                    "config": f"C={best_C}, gamma={best_gamma}",
                    "accuracy": r["accuracy"],
                    "macro_f1": r["macro_f1"],
                    "food_f1": r["food_f1"],
                    "elapsed_s": r["elapsed_s"],
                }
            )
    return rows


def final_eval(animal: str, best_name: str, best_params: dict | None, df: pd.DataFrame, X: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    n_classes = len(classes)
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()
    X_test, y_test = X[test_mask], y[test_mask]

    config_suffix = ""
    if best_name == "dense_head":
        model, history = train_head(X[train_mask], y[train_mask], X[val_mask], y[val_mask], n_classes)
        y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
        model_path = MODELS_DIR / f"best_classifier_{animal}_dense_head.keras"
        model.save(model_path)
        fit_info = f"train={int(train_mask.sum())} clips, early-stopped using val={int(val_mask.sum())} clips"
    else:
        trainval_mask = train_mask | val_mask
        clf = build_classifier(best_name, best_params)
        clf.fit(X[trainval_mask], y[trainval_mask])
        y_pred = clf.predict(X_test)
        if best_params:
            config_suffix = f"_C{best_params['C']}_gamma{best_params['gamma']}"
        model_path = MODELS_DIR / f"best_classifier_{animal}_{best_name}{config_suffix}.joblib"
        joblib.dump(clf, model_path)
        fit_info = f"train+val={int(trainval_mask.sum())} clips (no early stopping needed)"

    fig_path = REPORTS_DIR / f"best_classifier_{animal}_{best_name}{config_suffix}_confusion_matrix.png"
    result = evaluate_and_plot(
        y_test, y_pred, classes, f"Best classifier for {animal} ({best_name}{config_suffix}) - test", fig_path
    )
    result["model_path"] = model_path
    result["fit_info"] = fit_info
    result["n_test"] = int(test_mask.sum())

    food_idx = classes.index("food") if "food" in classes else None
    if food_idx is not None:
        per_class = f1_score(y_test, y_pred, average=None, zero_division=0, labels=range(n_classes))
        result["food_f1"] = per_class[food_idx]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    t_start = time.time()

    print("Leakage-safety: LogReg / SVM (linear & RBF) / LDA are sklearn Pipeline(StandardScaler, ")
    print("classifier). Each fold calls sklearn.base.clone() to get a fresh, UNFITTED pipeline,")
    print("then .fit(X[train_idx], y[train_idx]) - so the scaler's mean/std come from that fold's")
    print("training rows only, and val rows only go through .predict() (transform, not fit).")
    print("MobileNet features are fold-independent (see cross_validation.py docstring).")

    print("\nLoading MobileNetV2 (ImageNet weights, frozen backbone)...")
    backbone = build_backbone()

    all_rows: list[dict] = []
    rbf_grid_rows: list[dict] = []
    data_per_animal: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    best_rbf_per_animal: dict[str, dict] = {}

    for animal in animals:
        df, X, y = process_animal(animal, backbone, all_rows, rbf_grid_rows)
        data_per_animal[animal] = (df, X, y)

        best_rbf = select_best_rbf(animal, rbf_grid_rows)
        best_rbf_per_animal[animal] = best_rbf
        all_rows.extend(best_rbf_rows(animal, rbf_grid_rows, best_rbf["C"], best_rbf["gamma"]))

    df_scores = pd.DataFrame(all_rows)
    csv_path = REPORTS_DIR / "classifier_comparison_cv_scores.csv"
    df_scores.to_csv(csv_path, index=False)
    print(f"\nPer-fold CV scores saved to: {csv_path}")

    rbf_grid_df = pd.DataFrame(rbf_grid_rows)
    rbf_grid_csv = REPORTS_DIR / "classifier_comparison_svm_rbf_grid.csv"
    rbf_grid_df.to_csv(rbf_grid_csv, index=False)
    print(f"SVM RBF grid scores saved to: {rbf_grid_csv}")

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
    summary_csv = REPORTS_DIR / "classifier_comparison_summary_table.csv"
    summary.to_csv(summary_csv)
    print(f"Summary table saved to: {summary_csv}")

    print("\n=== Repere re-check (dense_head row above) ===")
    for animal in animals:
        dh = summary.loc[(animal, "dense_head")]
        print(f"  {animal}: macro_f1 = {dh['macro_f1_mean']:.4f} +/- {dh['macro_f1_std']:.4f}")

    print("\n=== Best classifier per animal (selected on CV macro-F1, dummy_floor excluded) ===")
    final_results: dict[str, tuple[str, dict | None, dict]] = {}
    for animal in animals:
        cfg = CONFIGS[animal]
        candidates = summary.loc[animal].drop(index="dummy_floor", errors="ignore")
        best_name = candidates["macro_f1_mean"].idxmax()
        best_params = best_rbf_per_animal[animal] if best_name == "svm_rbf" else None
        suffix = f" (C={best_params['C']}, gamma={best_params['gamma']})" if best_params else ""
        print(f"  {animal}: {best_name}{suffix}, CV macro_f1={candidates.loc[best_name, 'macro_f1_mean']:.4f}")

        df, X, y = data_per_animal[animal]
        result = final_eval(animal, best_name, best_params, df, X, y, cfg["classes"])
        final_results[animal] = (best_name, best_params, result)

    print("\n=== Final test evaluation (test set touched ONCE per animal) ===")
    for animal in animals:
        best_name, best_params, result = final_results[animal]
        suffix = f" (C={best_params['C']}, gamma={best_params['gamma']})" if best_params else ""
        print(f"\n{animal.upper()} - {best_name}{suffix} - fit on {result['fit_info']}, n_test={result['n_test']}")
        print(f"  accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f}")
        if "food_f1" in result:
            print(f"  food_f1={result['food_f1']:.4f}")
        print(result["report"])
        print(f"  Confusion matrix saved to: {result['fig_path']}")
        print(f"  Model saved to: {result['model_path']}")

    total_elapsed = time.time() - t_start
    print(f"\nTotal wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
