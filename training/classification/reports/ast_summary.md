# AST (Audio Spectrogram Transformer) as a frozen audio embedding extractor (dog + cat)

## Context and rationale

Three independent levers on top of the frozen MobileNetV2 (ImageNet, image backbone) features all plateaued at the same ceiling: dense-head hyperparameter tuning (`tune_head.py`), data augmentation (`augment_cv.py`), and classifier family (`compare_classifiers.py` - logistic regression / linear SVM / RBF SVM / LDA / dense head, none of which beat the dense head, all within roughly 1 std of dog 0.8244 ± 0.1114 and cat 0.5223 ± 0.1338 macro-F1). That convergence pointed at the frozen MobileNetV2 features themselves as the bottleneck, not the classifier on top of them.

In this session, I tried AST (Audio Spectrogram Transformer), an audio-NATIVE transformer pretrained and fine-tuned on AudioSet, as a frozen feature extractor in place of MobileNetV2. The hypothesis: features learned on sounds (including a wide range of animal vocalisations in AudioSet's 527 classes) should separate my bark/growl/grunt and brushing/food/isolation classes better than features learned on natural photographs - especially for cat "food", the weakest class so far (CV F1 0.3646 ± 0.1896, by far the highest-variance number in any of my results).

## Checkpoint choice

I used `MIT/ast-finetuned-audioset-10-10-0.4593` (HuggingFace `transformers`).

This is the AST-Base checkpoint from the original AST paper (Gong et al., 2021), fine-tuned on the full AudioSet (527 classes, mAP 0.4593 - the number in the checkpoint name, and the highest-scoring AST checkpoint the authors released). I picked it because: (1) it is the reference AST checkpoint used throughout the HuggingFace docs, so it is well documented and unlikely to have integration surprises; (2) AudioSet's 527 classes cover many animal sounds, so its frozen representations should transfer better to dog/cat vocalisations than ImageNet features; (3) AST-Base is small enough to extract embeddings for all ~550 clips on a single Colab T4 in a reasonable time (this run took 135.0s total on cuda).

AST feature extractor config (read from the loaded checkpoint, not hardcoded): sampling_rate=16000 Hz, num_mel_bins=128, max_length=1024 frames, do_normalize=True (mean=-4.2677393, std=4.5689974). AST encoder output: 768-dim embeddings (`last_hidden_state` mean-pooled over the token sequence).

## Method

For each clip, I reuse `preprocess.fix_length` to center pad/crop the raw 16 kHz waveform to the SAME fixed duration as every other approach (4s dog / 2s cat), so AST sees "the same clips" as YAMNet/MobileNetV2. I run the frozen AST feature extractor + encoder (`torch.no_grad()`, `model.eval()`, `requires_grad=False`) and mean-pool `last_hidden_state` over the token sequence to get one 768-dim embedding per clip - the same "mean-pool" idea as YAMNet's mean-pool over its 1024-dim frame embeddings.

On top of these frozen AST embeddings, I evaluate two lightweight classifiers, reusing existing code unchanged: `tl_common.train_head` (the dense head, Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)) and `logreg` (`Pipeline(StandardScaler, LogisticRegression(class_weight="balanced", max_iter=5000))`, the same recipe as `compare_classifiers.py`). Both are evaluated with the SAME CV folds and seed as every other approach via `cross_validation.make_folds` (unchanged).

## Anti-leakage checklist (as required)

1. **Same CV protocol/folds/seed as `cross_validation.py`**: confirmed - `cross_validation.make_folds` is imported and used unchanged (`StratifiedGroupKFold(k=4, group=cat_id)` for cat, `StratifiedKFold(k=5)` for dog, both `seed=42` via `tl_common.SEED`).

2. **AST embeddings are fold-independent**: each clip is encoded independently by the frozen AST model under `torch.no_grad()` - no batch statistics, no cross-sample normalisation. The feature extractor's `mean`/`std` (reported above) are fixed AudioSet-wide constants baked into the pretrained checkpoint, NOT computed from my data, so they cannot leak information between my train/val/test splits. I therefore extract embeddings ONCE per animal, for ALL clips, and index into them per fold - identical argument and efficiency benefit as for YAMNet/MobileNetV2 in `cross_validation.py`.

3. **Cat group violations, printed per fold - all 0**:

   | Fold | n_train | n_val | train_cats | val_cats | group_violations |
   |---|---|---|---|---|---|
   | 0 | 352 | 88 | 16 | 5 | **0** |
   | 1 | 273 | 167 | 15 | 6 | **0** |
   | 2 | 362 | 78 | 17 | 4 | **0** |
   | 3 | 333 | 107 | 15 | 6 | **0** |

   No cat individual ever appears in both the train and validation part of a fold.

4. **Scaler fit per fold (Pipeline + `clone()`)**: `logreg` is built as `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(...))])`. For every fold, `sklearn.base.clone()` produces a brand-new, unfitted copy, then `.fit(X[train_idx], y[train_idx])` is called - so `StandardScaler.mean_`/`scale_` are computed from that fold's training rows ONLY. `class_weight="balanced"` (dense head and logreg) is likewise computed from `y[train_idx]` only.

5. **Test touched once**: for each animal, the classifier with the best mean CV macro-F1 (dense_head vs logreg) is selected on CV ONLY, then evaluated exactly once on the "test" split.

## Results: AST vs MobileNet vs YAMNet (CV mean ± std)

### Dog (5-fold StratifiedKFold)

| Approach | accuracy | macro-F1 |
|---|---|---|
| **ast_dense_head** | 0.8411 ± 0.0724 | **0.8456 ± 0.0674** |
| ast_logreg | 0.8320 ± 0.0833 | 0.8377 ± 0.0787 |
| yamnet (repere) | 0.7972 ± 0.0771 | 0.7986 ± 0.0740 |
| mobilenet (repere) | 0.8229 ± 0.1109 | 0.8244 ± 0.1114 |

### Cat (4-fold StratifiedGroupKFold, group=cat_id)

| Approach | accuracy | macro-F1 | food F1 |
|---|---|---|---|
| ast_dense_head | 0.5900 ± 0.1416 | 0.4985 ± 0.1462 | 0.3652 ± 0.2082 |
| **ast_logreg** | 0.5647 ± 0.1309 | **0.5064 ± 0.0859** | 0.3072 ± 0.1190 |
| yamnet (repere) | 0.4056 ± 0.0913 | 0.3565 ± 0.0728 | n/a |
| mobilenet (repere) | 0.5603 ± 0.1431 | 0.5223 ± 0.1338 | 0.3646 ± 0.1896 |

## Does AST beat the MobileNet / YAMNet repere?

**DOG** (best AST classifier: `dense_head`)

- macro-F1 0.8456 ± 0.0674 vs MobileNet delta = +0.0212 vs 0.8244 ± 0.1114 -> within noise (smaller than 1 std)
- macro-F1 vs YAMNet delta = +0.0470 vs 0.7986 ± 0.0740 -> within noise (smaller than 1 std)

**CAT** (best AST classifier: `logreg`)

- macro-F1 0.5064 ± 0.0859 vs MobileNet delta = -0.0159 vs 0.5223 ± 0.1338 -> within noise (smaller than 1 std)
- macro-F1 vs YAMNet delta = +0.1499 vs 0.3565 ± 0.0728 -> a real improvement (more than 1 std)
- "food" F1 0.3072 ± 0.1190 vs MobileNet delta = -0.0574 vs 0.3646 ± 0.1896 -> within noise (smaller than 1 std)

## Signal vs noise

The deltas above are compared against `max(ast_std, reference_std)` - if the absolute difference is smaller than that, I call it "within noise", regardless of its sign. With only 4-5 CV folds, a single fold flipping can move macro-F1 by several hundredths, so a difference has to clear roughly one full standard deviation before I treat it as a real signal rather than fold-to-fold variance. The cat "food" class in particular has the highest variance of any number in this project (MobileNet: 0.3646 ± 0.1896, i.e. the 4 fold values range roughly from 0.16 to 0.66) - any AST food-F1 result should be read with that in mind.

## Final test evaluation (touched once)

**DOG** - `dense_head` (CV-selected) - fit on train=79 clips, early-stopped using val=17 clips, n_test=17

- accuracy = **0.8824**, macro-F1 = **0.8857**

```
              precision    recall  f1-score   support

        bark       0.86      0.86      0.86         7
       growl       0.80      0.80      0.80         5
       grunt       1.00      1.00      1.00         5

    accuracy                           0.88        17
   macro avg       0.89      0.89      0.89        17
weighted avg       0.88      0.88      0.88        17
```

Confusion matrix: `/content/pet_translator/training/classification/reports/ast_dog_dense_head_confusion_matrix.png`

**CAT** - `logreg` (CV-selected) - fit on train+val=373 clips (no early stopping needed), n_test=67

- accuracy = **0.6418**, macro-F1 = **0.5059**
- food F1 = **0.3077**

```
              precision    recall  f1-score   support

    brushing       0.56      0.28      0.37        18
        food       0.33      0.29      0.31        14
   isolation       0.74      0.97      0.84        35

    accuracy                           0.64        67
   macro avg       0.54      0.51      0.51        67
weighted avg       0.61      0.64      0.60        67
```

Confusion matrix: `/content/pet_translator/training/classification/reports/ast_cat_logreg_confusion_matrix.png`

## Conclusion: does AST unblock the problem?

**No - AST lands within noise of the MobileNet repere**, for both macro-F1 and (where applicable) "food" F1. Combined with the fact that MobileNetV2 plus three independent levers (head tuning, augmentation, classifier family) all plateaued at the same ceiling, this is now the FOURTH architecturally different approach landing in the same range. That convergence across such different feature spaces (ImageNet CNN, AudioSet CNN via YAMNet, AudioSet Transformer) suggests the ceiling may be closer to this dataset's irreducible difficulty (small dataset, especially for cat "food") than to any specific frozen backbone. **Fine-tuning the backbone remains the strongest untried lever** - it is the one approach where the feature extractor itself adapts to this exact data, rather than being used as-is.

## Reproducibility and output files

- `seed=42` everywhere (`tl_common.SEED`), AST inference on cuda.
- Same CV folds/seed as every other approach (`cross_validation.make_folds`, unchanged).
- dog: AST embeddings shape (113, 768), extracted in 19.9s on cuda.
- cat: AST embeddings shape (440, 768), extracted in 35.8s on cuda.
- Total wall-clock time: 135.0s.
- `reports/ast_cv_scores.csv` - one row per (animal, classifier, fold).
- `reports/ast_dog_dense_head_confusion_matrix.png`
- `models/ast_dog_dense_head.keras` (gitignored)
- `reports/ast_cat_logreg_confusion_matrix.png`
- `models/ast_cat_logreg.joblib` (gitignored)
