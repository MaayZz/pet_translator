# Classifier comparison on frozen MobileNetV2 features (dog + cat)

## Context and rationale

Every approach I have tried so far on top of the frozen MobileNetV2 feature
extractor (1280-dim, ImageNet-pretrained, `pooling="avg"`) used the same small
**dense head** (`Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes,
softmax)`). Two levers I tried to improve it both failed to beat the CV
standard deviation:

- head hyperparameter tuning (`tune_head.py`),
- data augmentation (`augment.py` / `augment_cv.py`, see
  `reports/augmentation_summary.md`).

My actual data regime is **1280 features with very few training samples per
fold** (~79-91 for dog, ~310-352 for cat). That is the classic "many
features, few samples" regime where regularized linear/margin classifiers
(logistic regression, linear SVM, LDA) are often reported to generalize at
least as well as - or better than - a small neural net. This session tests
that hypothesis directly: **same MobileNet features, same CV folds, same
seed**, only the classifier on top changes.

## Method (protocol reuse - no duplication)

`src/compare_classifiers.py` reuses, unchanged:

- `cross_validation.extract_logmel_batch` + `mobilenet_transfer.build_backbone`
  / `spectrograms_to_images`: frozen MobileNetV2 (ImageNet, `pooling="avg"`),
  per-sample min-max rescale -> 1280-dim features, computed **once per
  animal, for all clips**, then indexed per fold (fold-independent, as
  established in `cross_validation.py`'s docstring).
- `cross_validation.make_folds`: `StratifiedGroupKFold(k=4, group=cat_id)` for
  cat, `StratifiedKFold(k=5)` for dog, both `seed=42`.
- `tl_common.train_head`: the dense head, completely unchanged - run first in
  every fold to re-verify it reproduces the existing repere.

### Classifiers compared (all on the same 1280-dim MobileNet features)

| Classifier | Definition |
|---|---|
| `dummy_floor` | `DummyClassifier(strategy="most_frequent")` - sanity floor |
| `logreg` | `LogisticRegression(class_weight="balanced", max_iter=5000)` |
| `svm_linear` | `SVC(kernel="linear", class_weight="balanced")` |
| `svm_rbf` | `SVC(kernel="rbf", class_weight="balanced")`, best of a 3x3 `C x gamma` grid (see below) |
| `lda` | `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` |
| `dense_head` | `tl_common.train_head`, unchanged (the existing reference) |

`logreg`, `svm_linear`, `svm_rbf` and `lda` are each an sklearn
`Pipeline(StandardScaler(), classifier)`. LDA needs `shrinkage="auto"`
because with 1280 features and a few hundred samples the plain (unshrunk)
within-class covariance matrix would be singular.

**Known asymmetry**: scikit-learn's `LinearDiscriminantAnalysis` has no
`class_weight` parameter, so `lda` is the one classifier here that does not
get explicit class-balancing (it does use the training class priors, but that
is not the same as `class_weight="balanced"`). I flag this rather than hide
it - it is a small but real difference in how `lda` is set up versus the
other classifiers.

## Anti-leakage checklist (as required)

1. **Same CV protocol/folds/seed as `cross_validation.py`**: confirmed -
   `StratifiedGroupKFold(k=4, group=cat_id)` for cat, `StratifiedKFold(k=5)`
   for dog, `seed=42` via `tl_common.SEED`. The MobileNet features themselves
   are fold-independent (computed once, before any fold split - identical
   argument as in `cross_validation.py`'s docstring).

2. **Cat group violations, printed per fold - all 0**:

   | Fold | n_train | n_val | train_cats | val_cats | group_violations |
   |---|---|---|---|---|---|
   | 0 | 322 | 118 | 16 | 5 | **0** |
   | 1 | 319 | 121 | 15 | 6 | **0** |
   | 2 | 352 | 88  | 17 | 4 | **0** |
   | 3 | 327 | 113 | 15 | 6 | **0** |

   No cat individual ever appears in both the train and validation part of a
   fold.

3. **Scaler fit per fold (Pipeline + `clone()`)**: `logreg`, `svm_linear`,
   `svm_rbf` and `lda` are each built as `Pipeline([("scaler",
   StandardScaler()), ("clf", ...)])`. For **every fold**,
   `sklearn.base.clone(pipeline)` produces a brand-new, **unfitted** copy,
   then `.fit(X[train_idx], y[train_idx])` is called - so
   `StandardScaler.mean_`/`scale_` are computed from that fold's training rows
   ONLY. `.predict(X[val_idx])` only *applies* the already-fitted transform
   (no re-fitting on validation data). `class_weight="balanced"` is likewise
   computed from `y[train_idx]` only (same as the dense head's
   `class_weight_dict`).

4. **Repere re-check (dense head)** - reproduced exactly, proving the
   protocol is intact:
   - dog: **0.8244 ± 0.1114** (matches the existing repere)
   - cat: **0.5223 ± 0.1338** (matches the existing repere)

5. **Test touched once**: for each animal, the classifier selection below is
   based ONLY on CV macro-F1 (mean across folds). The "test" split is used
   exactly once, at the very end, with the winning classifier per animal.

## Full results table (CV mean ± std)

### Dog (5-fold `StratifiedKFold`)

| Classifier | accuracy | macro-F1 |
|---|---|---|
| **dense_head** | 0.8229 ± 0.1109 | **0.8244 ± 0.1114** |
| lda | 0.8047 ± 0.1150 | 0.8074 ± 0.1136 |
| logreg | 0.7960 ± 0.1198 | 0.7977 ± 0.1215 |
| svm_linear | 0.7696 ± 0.0953 | 0.7742 ± 0.0931 |
| svm_rbf (C=10, gamma=scale) | 0.7348 ± 0.1353 | 0.7307 ± 0.1383 |
| dummy_floor | 0.4071 ± 0.0178 | 0.1928 ± 0.0060 |

### Cat (4-fold `StratifiedGroupKFold`, group=cat_id)

| Classifier | accuracy | macro-F1 | food F1 |
|---|---|---|---|
| **dense_head** | 0.5603 ± 0.1431 | **0.5223 ± 0.1338** | 0.3646 ± 0.1896 |
| svm_rbf (C=1, gamma=scale) | 0.5273 ± 0.1512 | 0.4854 ± 0.1231 | 0.3548 ± 0.1340 |
| lda | 0.5260 ± 0.0923 | 0.4744 ± 0.0520 | 0.3270 ± 0.0993 |
| logreg | 0.5088 ± 0.0933 | 0.4642 ± 0.0569 | 0.3094 ± 0.1247 |
| svm_linear | 0.4943 ± 0.1169 | 0.4528 ± 0.0787 | 0.3010 ± 0.1287 |
| dummy_floor | 0.4961 ± 0.0604 | 0.2205 ± 0.0186 | 0.0000 ± 0.0000 |

Both dummy floors land far below every real classifier (dog macro-F1 0.19,
cat macro-F1 0.22, cat food F1 0.00), so the sanity check passes: every real
classifier is doing substantially better than majority-class guessing.

## SVM RBF grid search (selected on CV, not test)

3x3 grid, `C in {0.1, 1, 10}` x `gamma in {0.001, "scale", 0.01}`, each
config evaluated with the SAME folds, mean CV macro-F1 used for selection:

**Dog**

| C | gamma | mean macro-F1 |
|---|---|---|
| 0.1 | 0.001 | 0.2269 ± 0.1319 |
| 0.1 | 0.01 | 0.1670 ± 0.0198 |
| 0.1 | scale | 0.2259 ± 0.1298 |
| 1 | 0.001 | 0.7184 ± 0.1575 |
| 1 | 0.01 | 0.1928 ± 0.0060 |
| **1** | **scale** | 0.7300 ± 0.1516 |
| **10** | **0.001** | 0.7089 ± 0.1445 |
| 10 | 0.01 | 0.1928 ± 0.0060 |
| **10** | **scale** | **0.7307 ± 0.1383** (selected) |

**Cat**

| C | gamma | mean macro-F1 |
|---|---|---|
| 0.1 | 0.001 | 0.2394 ± 0.0914 |
| 0.1 | 0.01 | 0.1624 ± 0.0540 |
| 0.1 | scale | 0.2455 ± 0.0932 |
| **1** | **0.001** | 0.4709 ± 0.1187 |
| 1 | 0.01 | 0.2205 ± 0.0186 |
| **1** | **scale** | **0.4854 ± 0.1231** (selected) |
| 10 | 0.001 | 0.4353 ± 0.1051 |
| 10 | 0.01 | 0.2205 ± 0.0186 |
| 10 | scale | 0.4334 ± 0.0922 |

`gamma=0.01` collapses to near-dummy performance for both animals (the kernel
is too narrow for 1280-dim features with this little data); `gamma="scale"`
(~1/1280, i.e. close to `gamma=0.001`) consistently does much better.

## Best classifier per animal - honest signal vs noise

**For both animals, the dense head has the highest mean CV macro-F1 - exactly
the existing repere.** No classical classifier beats it.

- **Dog**: dense_head 0.8244 ± 0.1114 vs the closest competitor, LDA, at
  0.8074 ± 0.1136. Gap = **-0.017**, far smaller than either std (~0.11). LDA,
  logreg and svm_linear are all within ~1 std of the dense head; svm_rbf is
  the weakest of the five real classifiers (-0.094, still within 1 std).
  **No classifier change is a significant improvement for dog.**

- **Cat**: dense_head 0.5223 ± 0.1338 vs the closest competitor, svm_rbf
  (C=1, gamma=scale), at 0.4854 ± 0.1231. Gap = **-0.037**, again far smaller
  than either std (~0.12-0.13). LDA, logreg and svm_linear all trail further
  behind, still within 1 std of the dense head. **No classifier change is a
  significant improvement for cat either.**

## Does "food" finally improve?

**No.** dense_head's CV food F1 is 0.3646 ± 0.1896 (the highest mean, but also
by far the **highest variance** - this number swings from 0.16 to 0.62 across
the 4 folds). svm_rbf gives 0.3548 ± 0.1340 (gap = -0.010, essentially
identical, well within noise) and LDA gives 0.3270 ± 0.0993 (gap = -0.038,
within noise). All three are within roughly 1 std of each other.

The one thing worth noting honestly: **LDA and svm_rbf have a roughly 2x
smaller food-F1 standard deviation than the dense head** (0.099 / 0.134 vs
0.190). That means they are more *consistent* across folds even though their
*mean* is not higher - a minor practical observation, but it is **not** the
"food F1 improves" result I was hoping to see, and a gain this small (and in
the wrong direction on the mean) does not change the conclusion.

## Final test evaluation (touched once, dense head wins both)

Since `dense_head` has the best CV macro-F1 for both animals, it is the
classifier carried to the final test evaluation - trained on `train`
(early-stopped on `val`), as in `mobilenet_transfer.py`. These numbers are
therefore identical to the existing repere's single-split test results:

**Dog** (n_test=17): accuracy = **0.7647**, macro-F1 = **0.6887**

```
              precision    recall  f1-score   support

        bark       0.70      1.00      0.82         7
       growl       1.00      0.20      0.33         5
       grunt       0.83      1.00      0.91         5

    accuracy                           0.76        17
   macro avg       0.84      0.73      0.69        17
weighted avg       0.83      0.76      0.70        17
```

Confusion matrix: `reports/best_classifier_dog_dense_head_confusion_matrix.png`

**Cat** (n_test=67): accuracy = **0.6418**, macro-F1 = **0.5130**, food F1 = **0.2000**

```
              precision    recall  f1-score   support

    brushing       0.80      0.44      0.57        18
        food       0.33      0.14      0.20        14
   isolation       0.65      0.94      0.77        35

    accuracy                           0.64        67
   macro avg       0.59      0.51      0.51        67
weighted avg       0.62      0.64      0.60        67
```

Confusion matrix: `reports/best_classifier_cat_dense_head_confusion_matrix.png`

## Strategic conclusion: classifier or features?

I have now tried **three architecturally very different levers** on top of
the frozen MobileNetV2 features:

1. dense-head hyperparameter tuning (`tune_head.py`) - no significant gain,
2. data augmentation (`augment_cv.py`) - no significant gain (and the wrong
   direction for cat),
3. classifier family - this session: logistic regression, linear SVM, RBF
   SVM (with a small grid search), LDA, and a dummy floor - **none beats the
   dense head**, and all of them cluster within roughly 1 std of it for both
   animals.

The fact that a **neural head**, **regularized linear models**,
**margin-based kernels**, and a **generative linear model (LDA)** all land
within noise of each other (dog: ~0.73-0.82, cat: ~0.45-0.52) on the SAME
1280-dim feature vectors is itself a meaningful result: it strongly suggests
that **the ceiling is set by the frozen MobileNetV2 features themselves**,
not by which classifier sits on top of them. If the features fully separated
the classes, at least one of these five quite different decision boundaries
would be expected to exploit that better than the others - none does.

**This points toward fine-tuning the backbone** (unfreezing some MobileNetV2
layers) as the next logical lever: MobileNetV2's ImageNet features were
learned on natural photographs, and log-mel spectrograms turned into
"images" via per-sample min-max are visually quite different from that
domain. Letting at least the top conv blocks adapt to this domain is the one
lever among the "obvious" ones that has not yet been tried, and the
convergence of three independent levers to the same ceiling makes a strong
case for it.

## Reproducibility, anti-leakage summary, timing

- `seed=42` everywhere (`tl_common.SEED`), CPU only.
- 0 cat group violations across all 4 folds (table above).
- `StandardScaler` fit per fold via `Pipeline` + `sklearn.base.clone()`,
  confirmed above.
- Test set touched exactly once per animal, with the CV-selected winner.
- **Total wall-clock time: 40.2s** (both animals, full comparison + RBF grid
  + final evaluation).

## Output files

- `reports/classifier_comparison_cv_scores.csv` - one row per
  (animal, classifier, fold).
- `reports/classifier_comparison_svm_rbf_grid.csv` - one row per
  (animal, fold, C, gamma) for the full 3x3 RBF grid.
- `reports/classifier_comparison_summary_table.csv` - mean ± std per
  (animal, classifier).
- `reports/best_classifier_dog_dense_head_confusion_matrix.png`
- `reports/best_classifier_cat_dense_head_confusion_matrix.png`
- `models/best_classifier_dog_dense_head.keras` (gitignored)
- `models/best_classifier_cat_dense_head.keras` (gitignored)
