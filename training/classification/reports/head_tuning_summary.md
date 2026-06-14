# Head Hyperparameter Tuning (MobileNetV2, frozen backbone)

## Method

Following the decision to retain a single backbone (MobileNetV2, frozen, for both animals - YAMNet stays in the repo as a documented comparison but isn't tuned further), this session tunes only the small dense head on top of the frozen MobileNetV2 features.

**Anti-cheating protocol**:
- Every config is scored ONLY by its **mean macro-F1 across the CV folds** produced by `cross_validation.make_folds(animal, df)` - the *exact same* folds as `cross_validation.py` (StratifiedGroupKFold k=4 on `cat_id` for cat, StratifiedKFold k=5 for dog, `seed=42`). No new split was created.
- The held-out **test set** (`data/processed/<animal>/test_{X,y}.npy`) is touched **exactly once**, at the very end, with the single winning config per animal - never during the search.
- Features (MobileNetV2, frozen backbone) are precomputed once per animal, same as `cross_validation.py` - no cross-sample normalisation statistic is used, so this is leakage-free (see `cross_validation.py`'s docstring for the full argument).

**Sanity check**: I re-evaluated the untuned default config (`dense_units=64, dropout=0.3, l2=0, lr=1e-3`) through this exact pipeline. It reproduced the `cross_validation.py` numbers **exactly**:
- Dog: macro-F1 = 0.8244 ± 0.1114 (matches the repere)
- Cat: macro-F1 = 0.5223 ± 0.1338, food F1 = 0.3646 ± 0.1896 (matches the repere)

This confirms the tuning pipeline is consistent with the CV baseline - any difference in the tables below is purely due to the hyperparameter change, not a different protocol.

## Search space and why I didn't run the full grid

Suggested grid: dense_units {32, 64, 128} x dropout {0.2, 0.3, 0.5} x l2 {0, 1e-4, 1e-3} x lr {1e-3, 3e-4} = 54 configs per animal. Evaluated over 4-5 CV folds each, that's 200-450 head trainings per animal - more than needed and slow to repeat on CPU.

Instead I used a **sequential/targeted search** of 12 configs per animal (covering every value of every hyperparameter at least once - a ~4.5x reduction vs the full grid):

1. **Stage "reg_grid"** (9 configs): dropout x l2, with `dense_units=64, lr=1e-3` (the original defaults). I started here because the consigne's main concern is overfitting on the small "food" class, and dropout/L2 are the two regularization knobs that most directly affect that.
2. **Stage "capacity"** (2 configs): `dense_units` in {32, 128}, with the best (dropout, l2) from stage 1. `dense_units=64` is already covered by stage 1.
3. **Stage "lr"** (1 config): `lr=3e-4`, with the best (dense_units, dropout, l2) from stage 2. `lr=1e-3` is already covered.

All 24 rows (12 per animal) are in `reports/head_tuning_scores.csv`.

## Dog results (StratifiedKFold, k=5, 113 clips)

| Stage | dense_units | dropout | l2 | lr | Accuracy (mean±std) | Macro-F1 (mean±std) |
|---|---|---|---|---|---|---|
| reg_grid | 64 | 0.2 | 0 | 1e-3 | 0.8482 ± 0.0834 | 0.8476 ± 0.0867 |
| reg_grid | 64 | 0.2 | 1e-4 | 1e-3 | 0.8486 ± 0.0622 | 0.8469 ± 0.0668 |
| reg_grid | 64 | 0.2 | 1e-3 | 1e-3 | 0.8482 ± 0.0834 | 0.8476 ± 0.0867 |
| reg_grid (**default**) | 64 | 0.3 | 0 | 1e-3 | 0.8229 ± 0.1109 | 0.8244 ± 0.1114 |
| reg_grid | 64 | 0.3 | 1e-4 | 1e-3 | 0.8403 ± 0.1296 | 0.8409 ± 0.1291 |
| reg_grid | 64 | 0.3 | 1e-3 | 1e-3 | 0.8051 ± 0.1293 | 0.8071 ± 0.1279 |
| reg_grid | 64 | 0.5 | 0 | 1e-3 | 0.8047 ± 0.1146 | 0.8079 ± 0.1120 |
| reg_grid (**best**) | 64 | 0.5 | 1e-4 | 1e-3 | 0.8581 ± 0.0844 | **0.8604 ± 0.0837** |
| reg_grid | 64 | 0.5 | 1e-3 | 1e-3 | 0.8403 ± 0.1059 | 0.8429 ± 0.1049 |
| capacity | 32 | 0.5 | 1e-4 | 1e-3 | 0.8146 ± 0.1199 | 0.8179 ± 0.1147 |
| capacity | 128 | 0.5 | 1e-4 | 1e-3 | 0.8494 ± 0.1335 | 0.8530 ± 0.1294 |
| lr | 64 | 0.5 | 1e-4 | 3e-4 | 0.8399 ± 0.0923 | 0.8388 ± 0.0951 |

**Best config: `dense_units=64, dropout=0.5, l2=1e-4, lr=1e-3`** -> CV macro-F1 = **0.8604 ± 0.0837**, vs default **0.8244 ± 0.1114**.

**Is this a real gain or noise?** The improvement is +0.036 macro-F1. Both the default's std (0.1114) and the best config's std (0.0837) are larger than this gap - **so this is within the noise**, not a significant improvement. I'm not going to claim dropout=0.5 is "better" for dog based on this; it's a plausible config among several (dense=128 with the same dropout/l2 reaches 0.8530 ± 0.1294, also overlapping). What I can say is that **higher dropout (0.5) combined with light L2 (1e-4) consistently did at least as well as the default** across the configs I tried, and never did clearly worse.

### Dog - final test evaluation (ONE shot, best config)

- **Test accuracy = 0.7647, macro-F1 = 0.7500** (8 epochs)
- Compare to the untuned default's test score (from `transfer_learning_summary.md`): accuracy = 0.7647, macro-F1 = 0.6887

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bark | 0.67 | 0.86 | 0.75 | 7 |
| growl | 0.67 | 0.40 | 0.50 | 5 |
| grunt | 1.00 | 1.00 | 1.00 | 5 |

Same accuracy (12/17 right either way) but macro-F1 improved from 0.6887 to 0.7500: "grunt" goes from 4/5 to 5/5 correct, and "growl" recall improves from 1/5 to 2/5 (at the cost of "bark" recall dropping from 7/7 to 6/7). One swapped file changes the picture quite a bit on this 17-clip test set - consistent with the "within noise" conclusion above. Confusion matrix: `reports/mobilenet_tuned_dog_confusion_matrix.png`.

## Cat results (StratifiedGroupKFold, k=4, group=cat_id, 440 clips)

| Stage | dense_units | dropout | l2 | lr | Accuracy (mean±std) | Macro-F1 (mean±std) | "food" F1 (mean±std) |
|---|---|---|---|---|---|---|---|
| reg_grid | 64 | 0.2 | 0 | 1e-3 | 0.5520 ± 0.1088 | 0.5200 ± 0.0886 | 0.3525 ± 0.0994 |
| reg_grid | 64 | 0.2 | 1e-4 | 1e-3 | 0.5656 ± 0.1093 | 0.5311 ± 0.0922 | 0.3730 ± 0.1402 |
| reg_grid | 64 | 0.2 | 1e-3 | 1e-3 | 0.5562 ± 0.1116 | 0.5254 ± 0.0990 | 0.3679 ± 0.1318 |
| reg_grid (**default**) | 64 | 0.3 | 0 | 1e-3 | 0.5603 ± 0.1431 | 0.5223 ± 0.1338 | 0.3646 ± 0.1896 |
| reg_grid | 64 | 0.3 | 1e-4 | 1e-3 | 0.5679 ± 0.1334 | 0.5285 ± 0.1185 | 0.3309 ± 0.1762 |
| reg_grid (**best**) | 64 | 0.3 | 1e-3 | 1e-3 | 0.5867 ± 0.1203 | **0.5367 ± 0.1043** | 0.3672 ± 0.1497 |
| reg_grid | 64 | 0.5 | 0 | 1e-3 | 0.5502 ± 0.1607 | 0.5204 ± 0.1417 | **0.4023 ± 0.1878** |
| reg_grid | 64 | 0.5 | 1e-4 | 1e-3 | 0.5625 ± 0.1470 | 0.5223 ± 0.1316 | 0.3547 ± 0.2106 |
| reg_grid | 64 | 0.5 | 1e-3 | 1e-3 | 0.5606 ± 0.1477 | 0.5156 ± 0.1297 | 0.3104 ± 0.1760 |
| capacity | 32 | 0.3 | 1e-3 | 1e-3 | 0.5678 ± 0.1261 | 0.5310 ± 0.1019 | 0.3973 ± 0.1224 |
| capacity | 128 | 0.3 | 1e-3 | 1e-3 | 0.5465 ± 0.1341 | 0.5109 ± 0.0966 | 0.3199 ± 0.0937 |
| lr | 64 | 0.3 | 1e-3 | 3e-4 | 0.5523 ± 0.1432 | 0.5117 ± 0.1167 | 0.3648 ± 0.1870 |

**Best config: `dense_units=64, dropout=0.3, l2=1e-3, lr=1e-3`** -> CV macro-F1 = **0.5367 ± 0.1043**, vs default **0.5223 ± 0.1338**.

**Is this a real gain or noise?** The improvement is +0.0144 macro-F1 - much smaller than either std (0.1043 / 0.1338). **This is clearly within the noise** - I would not claim this config is meaningfully better than the default for cat.

### Focus on "food" (the weak class)

CV "food" F1: best config = 0.3672 ± 0.1497, vs default = 0.3646 ± 0.1896. A gain of +0.0026 - **essentially unchanged**, and far smaller than either std. **None of the 12 configs tuned here meaningfully fixes the "food" weakness** - across the whole grid, "food" F1 ranges from 0.31 to 0.40, and the std on every config is large (0.09-0.21) because "food" is the smallest class (92/440 clips) and a handful of clips flip the per-fold F1 a lot.

One observation worth recording honestly: the config with the *highest* "food" F1 in this search was `dropout=0.5, l2=0` (food F1 = 0.4023 ± 0.1878), but its overall macro-F1 (0.5204 ± 0.1417) is essentially tied with the default (0.5223 ± 0.1338) - it trades a (noisy) gain on "food" for a (noisy) loss elsewhere. Since the selection criterion is overall macro-F1 (as specified), this config wasn't picked, but it's a candidate worth re-checking if a future session specifically targets "food" with a different lever (e.g. data augmentation, which is explicitly out of scope here).

**Bottom line for cat**: head-only hyperparameter tuning (within this grid) does not move the needle, neither on overall macro-F1 nor specifically on "food". This suggests the bottleneck isn't the head's capacity/regularization - it's more likely the small "food" sample size, the frozen-backbone features themselves (MobileNetV2 on spectrogram-as-image), or both. That's consistent with being told not to pursue fine-tuning/augmentation in this session.

### Cat - final test evaluation (ONE shot, best config)

- **Test accuracy = 0.6418, macro-F1 = 0.5130** (7 epochs)
- Compare to the untuned default's test score (from `transfer_learning_summary.md`): accuracy = 0.6418, macro-F1 = 0.5130 - **identical**, including the per-class report and epoch count.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brushing | 0.80 | 0.44 | 0.57 | 18 |
| food | 0.33 | 0.14 | 0.20 | 14 |
| isolation | 0.65 | 0.94 | 0.77 | 35 |

The only difference between this config and the default is `l2=1e-3` (vs `l2=0`); on this particular train/val/test split it changed nothing about the final model's predictions. This is consistent with the CV finding above (+0.0144 macro-F1, within noise): the "food" class (F1 = 0.20 on test, same as before) remains the clear weak point. Confusion matrix: `reports/mobilenet_tuned_cat_confusion_matrix.png`.

## Total time

- Dog: 12 configs x 5-fold CV + final eval = **140.6s**
- Cat: 12 configs x 4-fold CV + final eval = **118.0s**
- **Total: 258.6s (~4.3 min)**, all on CPU.

## What's NOT done yet

As requested, this session stayed within head-hyperparameter tuning: frozen MobileNetV2 backbone (no fine-tuning), no data augmentation, same CV protocol/splits as `cross_validation.py`. The main takeaway is that **this lever is mostly exhausted** for cat (no config beats the default by more than its noise band, "food" unchanged) and gives a small, within-noise CV improvement for dog that shows up as a modest test-set macro-F1 gain (0.6887 -> 0.7500) without changing accuracy. Fine-tuning the backbone and/or augmenting the "food" class are the natural next levers, but that's for a future session.

## Artifacts produced this session

- `src/tl_common.py`: `build_head`/`train_head` now accept `dense_units`, `dropout`, `l2`, `lr` (defaults unchanged, so all earlier scripts behave identically)
- `src/tune_head.py`: new sequential head-tuning script (`--animal dog|cat|all`)
- `reports/head_tuning_scores.csv`: all 24 CV configs (12 per animal) with mean±std accuracy/macro-F1/(food F1 for cat)
- `reports/head_tuning_summary.md`: this report
- `reports/mobilenet_tuned_{dog,cat}_confusion_matrix.png`: final one-shot test confusion matrices
- `models/mobilenet_{dog,cat}_head_tuned.keras` (gitignored)

---

# Suggested commit message

```
Tune MobileNetV2 head hyperparameters via CV (dog + cat)

- Extend tl_common.build_head/train_head with dense_units, dropout, l2,
  lr parameters (defaults unchanged - existing scripts behave identically)
- Add src/tune_head.py: sequential/targeted head hyperparameter search
  (dropout x l2 grid, then dense_units, then lr - 12 configs per animal
  vs 54 for the full cartesian grid), scored by mean CV macro-F1 on the
  exact same folds/features as cross_validation.py
  - Sanity check: default config reproduces the CV repere exactly
    (dog 0.8244+/-0.1114, cat 0.5223+/-0.1338)
  - Best dog config (dense=64, dropout=0.5, l2=1e-4): CV macro-F1
    0.8604+/-0.0837 (gain within noise); one-shot test macro-F1
    0.6887 -> 0.7500 (same accuracy 0.7647)
  - Best cat config (dense=64, dropout=0.3, l2=1e-3): CV macro-F1
    0.5367+/-0.1043 (gain within noise); one-shot test result identical
    to the untuned default (0.6418/0.5130) - "food" F1 still 0.20
- Add reports/head_tuning_scores.csv (24 configs) and
  reports/head_tuning_summary.md (method, full results, honest
  noise-vs-signal discussion, final one-shot test scores)
- Add reports/mobilenet_tuned_{dog,cat}_confusion_matrix.png
- Total tuning time: 258.6s (~4.3 min) on CPU
- Head-only tuning is mostly exhausted for cat (no config beats default
  beyond its noise band, "food" unchanged) - fine-tuning/augmentation
  are the next levers, not done in this session
```
