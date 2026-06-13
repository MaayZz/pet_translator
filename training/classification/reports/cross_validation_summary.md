# Cross-Validation — Reliable Scores Before Tuning

## Why cross-validation

My first transfer-learning run evaluated YAMNet and MobileNetV2 on a single, small held-out test set: 17 dog clips and 67 cat clips. A single split gives exactly one number, and that number can be misleading:

- For dog, 17 test clips means each misclassified file moves accuracy by ~6 percentage points. "82% vs 76%" between two approaches is really just "one extra file right or wrong" — not a solid basis for ranking approaches.
- For cat, the 67-clip test set is a single random draw from 21 individual cats; depending on which cats happen to land in test, the score could look better or worse than the approach's "true" average performance.

Before I do any hyperparameter optimization, I want a more honest picture: train and evaluate the *same* untuned approach on several different train/validation splits and report the mean and standard deviation. The std tells me how much I should trust the mean, and whether the rankings from the single-split run still hold.

**Nothing about the models changed in this run.** Same head (`Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)`), same `Adam(1e-3)`, same `class_weight="balanced"`, same `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`, `batch_size=8`, up to 50 epochs, `seed=42` reset before every fold. The only thing that changed is the evaluation protocol — this is implemented in the new `src/cross_validation.py`, which reuses `tl_common.train_head` (now shared by `yamnet_transfer.py`, `mobilenet_transfer.py`, and this script — I factored the head-building/compile/fit/early-stopping code that used to be duplicated in both transfer-learning scripts into `tl_common.build_head` / `tl_common.train_head`, and re-ran both single-split scripts afterwards to confirm I get the exact same numbers as before the refactor).

## Fold strategy per animal

### Cat — `StratifiedGroupKFold`, k=4, group = `cat_id` (mandatory)

The cat dataset has 440 clips coming from only **21 individual cats**, with the cat's ID recoverable from the filename (e.g. `I_ANI01_...` -> `ANI01`). If I used plain stratified k-fold without accounting for this, clips from the *same cat* could end up in both train and validation. A model could then partly learn to recognise *that cat's voice/recording conditions* rather than the sound class (brushing/food/isolation) — a real leakage risk, since vocalisations are quite individual-specific. Plain `StratifiedKFold` is therefore not acceptable here.

I used `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)` with `groups=cat_id`, which keeps every clip from a given cat in exactly one fold while still trying to balance the three classes across folds.

**Why k=4 and not k=3 or k=5**: with only 21 cats, k=5 leaves very few cats (and very few "food" clips — the smallest class, 92 clips total) in some validation folds. I checked k=3 and k=4 explicitly: both give 0 group violations and keep all 3 classes represented in every fold. k=4 gives one more independent estimate than k=3 while keeping validation folds reasonably sized (88–121 clips — already bigger than the original 67-clip test set), so I went with k=4.

### Dog — `StratifiedKFold`, k=5, stratified by class label (honest limitation)

The dog dataset (`shivarao`, 113 clips, classes bark/growl/grunt) does **not** provide any speaker/individual-dog ID. So unlike for cat, I have no way to build a group-aware split — I cannot tell whether two recordings come from the same dog.

I used `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, stratified by class label at the file level. 113 files / 5 folds gives ~22–23 validation clips per fold, already larger than the original 17-clip test set, and the smallest class ("growl", 33 clips) is still comfortably split across 5 folds.

**Honest limitation**: if the dataset contains multiple recordings of the same dog, some could land in both the train and validation part of a fold. That would let the model partly recognise an individual dog's voice rather than the vocalisation type, which would make the dog CV scores slightly optimistic. Stratified CV by class is the best I can do with the information available in this dataset — I'm flagging this so the dog numbers below aren't over-interpreted as a hard ceiling.

## Anti-leakage checklist

### A. Cat group leakage — verified per fold, 0 violations

For every cat fold I computed the set of `cat_id`s in train and in validation and checked their intersection. The script asserts `group_violations == 0` and would crash immediately if this were ever violated.

| Fold | n_train | n_val | train cats | val cats | shared cats (violations) |
|---|---|---|---|---|---|
| 0 | 322 | 118 | 16 | 5 | **0** |
| 1 | 319 | 121 | 15 | 6 | **0** |
| 2 | 352 | 88  | 17 | 4 | **0** |
| 3 | 327 | 113 | 15 | 6 | **0** |

Each row sums to 21 cats total (e.g. 16+5, 15+6, 17+4, 15+6), and the violation count is 0 in all 4 folds — no cat ever appears in both the train and validation part of the same fold.

### B. Dog speaker leakage — not preventable, documented above

As explained above, `shivarao` has no individual-dog ID, so this cannot be checked or guaranteed. This is the one open caveat on the dog CV numbers.

### C. Normalisation fit per fold — confirmed not to matter (with proof)

I confirm that **no cross-sample normalisation statistic ever leaks between a fold's train and validation set**, for either approach:

- **YAMNet**: each clip is passed through the frozen YAMNet model independently (raw 16 kHz waveform -> per-window embeddings -> mean-pooled over time to a single 1024-dim vector). No mean/std/min/max is ever computed across clips, so there is nothing to "fit per fold" in the first place.
- **MobileNetV2**: each log-mel spectrogram is rescaled to `[0, 1]` with a **per-sample min-max** (`mobilenet_transfer.spectrograms_to_images`) — using only that sample's own min and max, never anything computed across other samples. This means the *global* `(mean, std)` normalisation computed in `preprocess.py` is irrelevant here: for any monotonically increasing affine map `x -> a*x + b` (`a > 0`, exactly what `(x - mean) / std` is, since `std > 0`), per-sample min-max produces the **exact same result** with or without that map — the min and max are transformed by the same `a, b` and cancel out in the min-max formula.

So `cross_validation.py` extracts **raw** log-mel spectrograms directly from audio (no normalisation step at all), and the resulting MobileNetV2 features are identical to what they'd be under any global normalisation, fold-specific or not.

**Net result**: neither feature extraction pipeline depends on the fold at all, so I compute YAMNet embeddings and MobileNetV2 features **once per animal** and index into them per fold. This is both correct (no leakage, per the argument above) and efficient (no repeated backbone inference across folds).

The *only* statistic that genuinely depends on the fold's training labels is `class_weight="balanced"`, computed via `tl_common.class_weight_dict(y_train)` — and this is already computed from `y[train_idx]` only, inside `tl_common.train_head`, i.e. correctly per fold.

## Results — mean ± std across folds

| Animal | Approach | Folds | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|---|---|---|---|---|
| Dog | YAMNet | 5 | 0.7972 ± 0.0771 | 0.7986 ± 0.0740 |
| Dog | MobileNetV2 | 5 | 0.8229 ± 0.1109 | 0.8244 ± 0.1114 |
| Cat | YAMNet | 4 | 0.4056 ± 0.0913 | 0.3565 ± 0.0728 |
| Cat | MobileNetV2 | 4 | 0.5603 ± 0.1431 | 0.5223 ± 0.1338 |

### Per-fold detail — Dog (StratifiedKFold, k=5)

| Fold | n_train | n_val | YAMNet acc | YAMNet macro-F1 | MobileNetV2 acc | MobileNetV2 macro-F1 |
|---|---|---|---|---|---|---|
| 0 | 90 | 23 | 0.7391 | 0.7513 | 0.8696 | 0.8704 |
| 1 | 90 | 23 | 0.6957 | 0.6977 | 0.6522 | 0.6524 |
| 2 | 90 | 23 | 0.8696 | 0.8745 | 0.9565 | 0.9582 |
| 3 | 91 | 22 | 0.8182 | 0.8110 | 0.8182 | 0.8234 |
| 4 | 91 | 22 | 0.8636 | 0.8585 | 0.8182 | 0.8175 |

### Per-fold detail — Cat (StratifiedGroupKFold, k=4, group=cat_id)

| Fold | n_train | n_val | YAMNet acc | YAMNet macro-F1 | MobileNetV2 acc | MobileNetV2 macro-F1 |
|---|---|---|---|---|---|---|
| 0 | 322 | 118 | 0.2797 | 0.2605 | 0.5508 | 0.5215 |
| 1 | 319 | 121 | 0.4380 | 0.3918 | 0.6364 | 0.5657 |
| 2 | 352 | 88  | 0.4091 | 0.3445 | 0.3636 | 0.3414 |
| 3 | 327 | 113 | 0.4956 | 0.4293 | 0.6903 | 0.6605 |

All per-fold rows (plus group-violation columns for cat) are saved to `reports/cv_scores.csv`.

## Comparison to the single-split benchmark

Single-split numbers, from `reports/transfer_learning_summary.md`:

| | Dog Accuracy | Dog Macro-F1 | Cat Accuracy | Cat Macro-F1 |
|---|---|---|---|---|
| Floor (majority class) | 0.4118 | 0.1944 | 0.5224 | 0.2288 |
| YAMNet (single split) | 0.8235 | 0.8244 | 0.4925 | 0.3454 |
| MobileNetV2 (single split) | 0.7647 | 0.6887 | 0.6418 | 0.5130 |

### Dog

- **YAMNet**: CV mean (0.7972 / 0.7986) is close to, but slightly below, the single-split score (0.8235 / 0.8244) — and well within 1 std of the CV mean (std ≈ 0.077 / 0.074). The single-split test fold happened to be on the easier side, but it wasn't an outlier; YAMNet's single-split score looks **roughly confirmed**.
- **MobileNetV2**: CV mean (0.8229 / 0.8244) is *notably higher* than the single-split score (0.7647 / 0.6887) — the original 17-clip test set happened to be a relatively hard split for MobileNetV2 (fold 1 in the CV table above, acc=0.6522, is close to that single-split number, while folds 0 and 2 reach 0.87–0.96). So the single-split MobileNetV2 score was actually **pessimistic**, not optimistic. The flip side is the large std (≈0.11): MobileNetV2's dog performance varies a lot depending on which 22-23 clips end up in validation, more so than YAMNet's.
- Both approaches comfortably beat the floor (0.4118 / 0.1944) in every single fold.

### Cat

- **YAMNet**: CV mean macro-F1 (0.3565) is essentially the same as the single-split macro-F1 (0.3454) — confirmed. But CV mean **accuracy** (0.4056) is clearly *lower* than the single-split accuracy (0.4925), and is actually **below the majority-class floor's accuracy (0.5224)**. The single-split test set happened to give a higher-accuracy fold; on average across the 4 group-CV folds, YAMNet does not beat "always predict isolation" on raw accuracy, although it does still beat the floor on macro-F1 (0.3565 vs 0.2288) — which is the metric that matters here, since the floor's high accuracy comes purely from class imbalance.
- **MobileNetV2**: CV mean (0.5603 / 0.5223) is close to the single-split score (0.6418 / 0.5130) — macro-F1 in particular is essentially the same (0.5223 vs 0.5130), confirming MobileNetV2 as the best cat approach so far. It beats the floor's macro-F1 (0.2288) in **every** fold, including its weakest fold (fold 2, macro-F1=0.3414). The std (≈0.143 / 0.134) is larger than for any dog approach, reflecting how different the 4 cat folds are in size (88–121 clips) and class mix.

### Overall takeaway

The single-split run's *ranking* (MobileNetV2 best for cat on macro-F1, YAMNet/MobileNetV2 roughly comparable for dog) holds up under CV. But the exact numbers move more than I expected, especially for MobileNetV2 on dog (single split was pessimistic) and YAMNet on cat (single split was optimistic on accuracy, though not on macro-F1). The standard deviations (0.07–0.14) confirm that with this little data, a single test score should not be read to more than roughly ±10 points of precision — which is exactly why this CV step was worth doing before any tuning.

## Total run time

`src/cross_validation.py --animal all` (5 dog folds + 4 cat folds, × 2 approaches each = 18 train/eval runs, plus one-time YAMNet/MobileNetV2 loading and feature precomputation for 113 + 440 clips):

**Total wall-clock time: 68.4s**, all on CPU. Feature precomputation (YAMNet embeddings + MobileNetV2 features, done once per animal) took ~7.1s + ~3.8s for dog and ~7.0s + ~4.2s for cat; each individual fold's head training took 1.4–5.0s.

## What's NOT done yet

No hyperparameter tuning happened in this session, as requested — every fold uses the exact same fixed defaults as the first transfer-learning run. The next step is to use these CV mean ± std numbers as the baseline to beat, and explore hyperparameters (head size, learning rate, epochs/patience, batch size, and for MobileNetV2 the spectrogram-to-image conversion) for the most promising approach(es) per animal.

## Artifacts produced this session

- `src/tl_common.py`: added shared `build_head` / `train_head` helpers (refactored out of `yamnet_transfer.py` and `mobilenet_transfer.py`, behavior verified unchanged)
- `src/yamnet_transfer.py`, `src/mobilenet_transfer.py`: now call `tl_common.train_head` instead of duplicating the head/compile/fit code
- `src/cross_validation.py`: new CV evaluation script (`--animal dog|cat|all`)
- `reports/cv_scores.csv`: per-fold scores (18 rows: 5 dog folds × 2 approaches + 4 cat folds × 2 approaches), including cat group-violation checks
- `reports/cross_validation_summary.md`: this report

---

# Suggested commit message

```
Add cross-validation evaluation for dog/cat transfer-learning heads

- Factor build_head/train_head out of yamnet_transfer.py and
  mobilenet_transfer.py into tl_common.py (shared, behavior-preserving
  refactor - re-verified single-split numbers are unchanged)
- Add src/cross_validation.py: evaluates YAMNet and MobileNetV2 heads
  via cross-validation with the same untuned defaults as the first run
  - Cat: StratifiedGroupKFold (k=4, group=cat_id) - no cat split across
    train/val, 0 group violations verified and asserted per fold
  - Dog: StratifiedKFold (k=5) by class label, with documented
    limitation (shivarao has no speaker ID, so dog scores may be
    slightly optimistic)
  - Features (YAMNet embeddings, MobileNetV2 spectrogram features) are
    fold-independent (no cross-sample normalisation statistic is ever
    used), so they're precomputed once per animal and indexed per fold
- Add reports/cv_scores.csv (per-fold accuracy/macro-F1, 18 rows) and
  reports/cross_validation_summary.md (mean +/- std per (animal,
  approach), comparison to the single-split benchmark)
- Total CV wall-clock time: 68.4s on CPU
- No hyperparameter tuning in this session - that's the next step
```
