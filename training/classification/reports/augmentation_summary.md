# Data Augmentation for MobileNetV2 (Frozen Backbone) - Dog & Cat

## What I tested

The previous session (head hyperparameter tuning) concluded that the head
isn't the bottleneck - for cat especially, the "food" class stayed stuck
around F1 ~0.36 in CV no matter how I regularised the head. This session
tests a different lever: **data augmentation**, applied only to the
training data, targeted at the weakest classes (cat's "food" and
"brushing"). The backbone stays **frozen** (no fine-tuning) and the head
uses the **same default hyperparameters** as the original CV repere
(`dense_units=64, dropout=0.3, l2=0, lr=1e-3`), so any change in score is
attributable to augmentation alone.

I implemented this in two new files:
- `src/augment.py`: pure, per-clip, seeded augmentation functions (no file
  I/O, no cross-clip statistics).
- `src/augment_cv.py`: extends `cross_validation.py`'s protocol (same
  `make_folds`, same `extract_logmel_batch`, same `train_head`) to compare
  "without" vs "with" augmentation, fold by fold, then runs the one-shot
  final test evaluation.

## Augmentation techniques and magnitudes

On the raw audio (each clip processed independently with `librosa`):

- **Pitch shift**: random shift in **+/-2 semitones**. This is the
  conservative end of the range commonly used in speech/sound emotion
  recognition (SER) augmentation - moderate pitch shifts are reported to
  preserve the perceived emotion/vocalisation type, while larger shifts
  start to change it.
- **Time stretch**: random rate in **[0.95, 1.05]** (<=5%). I stayed at the
  conservative end of the "<=10%, ideally <=5%" guidance to avoid distorting
  a clip's temporal envelope enough to turn, say, a bark into a growl.
- **Additive Gaussian noise**: scaled to **3-4% of the clip's own peak
  amplitude** (per-clip, so no cross-sample statistic). This is plain
  "neutral" noise - I deliberately did not add any "emotional" noise
  (crying, laughing, etc.).

Order: pitch shift -> time stretch -> add noise -> `fix_length` (centered
pad/crop back to the fixed duration, exactly as in `preprocess.py`) -> log-mel
extraction. I add the noise *after* the pitch/time transforms and *after*
`fix_length`, so it represents an independent noise floor on the
final-duration waveform rather than something that itself gets stretched or
shifted.

On the resulting log-mel spectrogram:

- **SpecAugment**: one frequency mask and one time mask, each covering
  ~12.5% of that axis, filled with the spectrogram's own mean (so the masked
  region doesn't introduce an out-of-distribution value).

Every augmented variant keeps the same label as its original - I only ever
transform the audio/spectrogram, never the label.

## Targeted strategy: per-class augmentation factors

`AUGMENT_FACTORS` in `src/augment_cv.py` sets, per animal and class, how many
EXTRA augmented variants are generated per original clip (`n_aug`):

| Animal | Class | Total clips | n_aug | Effective multiplier |
|---|---|---|---|---|
| Cat | isolation | 221 | 0 | x1 |
| Cat | brushing | 127 | 1 | x2 |
| Cat | food | 92 | 2 | x3 |
| Dog | bark | 46 | 1 | x2 |
| Dog | growl | 33 | 1 | x2 |
| Dog | grunt | 34 | 1 | x2 |

**Cat (priority class, per the task)**: I gave "food" (the smallest and
weakest class) the most augmentation (x3), "brushing" (the second-smallest)
x2, and left "isolation" (already the majority, 221 clips) untouched. This is
a *soft* rebalancing toward more comparable per-fold counts (e.g. fold 0:
isolation ~166 unchanged, brushing 95->190, food 69->207) - not exact parity,
but a clear step toward it, with the most help going to the weakest class.

**Dog**: the class imbalance here (46/33/34, ratio ~1.4) is much milder than
cat's (221/92/127, ratio up to ~2.4). Rather than invent a targeted
rebalancing for an imbalance this small, I applied a **uniform x2** (every
class gets exactly one augmented variant per clip) - a general
augmentation/regularisation pass that doesn't change the relative class
proportions. Priority for the *targeted* strategy went to cat, as specified.

**On `class_weight="balanced"`**: I kept it as-is - it's computed
automatically inside `tl_common.train_head` from `y_train` **after**
augmentation. For dog, since augmentation is uniform across classes, the
post-augmentation class proportions are identical to the pre-augmentation
ones, so the balanced weights are unchanged. For cat, the post-augmentation
train set is more balanced than before, so the balanced weights
automatically become *milder* - augmentation does most of the rebalancing
work, and `class_weight` only corrects whatever imbalance remains. I didn't
need to add any extra logic for this: it falls out of computing the weights
from the realised post-augmentation labels, which `train_head` already does.

## Anti-leakage: how each of the 4 pitfalls is handled

**Pitfall 1 - augmenting before the split.** I did NOT generate augmented
data for the whole dataset and then split. In `augment_cv.py`, for each CV
fold, `augment_fold_train` is called only on that fold's `train_idx` (after
`make_folds` has already produced the split). The fold's validation set is
`mobilenet_X_all[val_idx]` - the original, never-augmented features, exactly
as in `cross_validation.py`. MobileNetV2 features for the augmented variants
are computed fresh, per fold, from that fold's augmented spectrograms only,
and are never added to the shared `mobilenet_X_all` array or reused by
another fold.

**Pitfall 2 - individual leakage (cat).** Every augmented variant inherits
the `cat_id` of the original clip it was derived from. Since variants are
only generated from `train_idx`, their `cat_id`s are a subset of
`train_cats`, which `StratifiedGroupKFold` already guarantees is disjoint
from `val_cats`. I computed `(train_cats | aug_cats) & val_cats` explicitly
for every cat fold and asserted it equals 0:

| Fold | train cats | val cats | group violations (after augmentation) |
|---|---|---|---|
| 0 | 16 | 5 | **0** |
| 1 | 15 | 6 | **0** |
| 2 | 17 | 4 | **0** |
| 3 | 15 | 6 | **0** |

For dog, the lack of a speaker ID remains a documented, pre-existing
limitation (from `preprocess.py`/`cross_validation.py`) - augmentation
doesn't introduce or worsen it, since augmented dog clips simply have no
group information to begin with, same as the originals.

**Pitfall 3 - shared normalisation statistic.** Feature extraction is
unchanged: `extract_logmel_batch` produces raw (non-normalised) log-mel
spectrograms, and `spectrograms_to_images` rescales each spectrogram with a
**per-sample** min-max - both fold-independent (see
`cross_validation.py`'s docstring for the full argument). The augmentation
functions in `augment.py` only ever use **per-clip** statistics:
`add_gaussian_noise` scales noise by *that clip's own* peak amplitude, and
`spec_augment` fills masked regions with *that spectrogram's own* mean. No
statistic is ever computed across clips - original or augmented, train or
validation. The only label-dependent statistic, `class_weight="balanced"`,
is computed from `y_train` (post-augmentation) inside `train_head`, i.e.
per-fold and train-only, as before.

**Pitfall 4 - test set touched more than once / used to choose.** The
without-vs-with comparison below is entirely a CV-validation-set comparison;
`run_cv_comparison` never touches `df[df.split=="test"]`. The test set is
touched exactly once, in `final_eval`, using the single augmentation config
defined above (`AUGMENT_FACTORS`) - there was no search over augmentation
configs, so there was nothing to select using the test score.

## CV results: without vs with augmentation

First, the **re-verification** (no augmentation, same folds/features/seed as
`cross_validation.py`):

| Animal | Metric | Repere (previous session) | This session (re-run) |
|---|---|---|---|
| Dog | Macro-F1 | 0.8244 +/- 0.1114 | **0.8244 +/- 0.1114** |
| Cat | Macro-F1 | 0.5223 +/- 0.1338 | **0.5223 +/- 0.1338** |
| Cat | "food" F1 | ~0.3646 +/- 0.1896 | **0.3646 +/- 0.1896** |

Exact match (to 4 decimals) - the protocol, features, folds and seed are
intact, so the "with augmentation" numbers below are comparable on a level
playing field.

### Without vs with augmentation

| Animal | Augmentation | Accuracy (mean +/- std) | Macro-F1 (mean +/- std) | "food" F1 (mean +/- std) |
|---|---|---|---|---|
| Dog | none | 0.8229 +/- 0.1109 | 0.8244 +/- 0.1114 | - |
| Dog | audio + SpecAugment | 0.8395 +/- 0.1353 | **0.8433 +/- 0.1319** | - |
| Cat | none | 0.5603 +/- 0.1431 | 0.5223 +/- 0.1338 | 0.3646 +/- 0.1896 |
| Cat | audio + SpecAugment | 0.5580 +/- 0.1651 | **0.4932 +/- 0.1553** | **0.2874 +/- 0.1777** |

### Per-fold detail - Dog (StratifiedKFold, k=5)

| Fold | n_train (none -> +aug) | Macro-F1 (none) | Macro-F1 (+aug) |
|---|---|---|---|
| 0 | 90 -> 180 (+90) | 0.8704 | 0.9568 |
| 1 | 90 -> 180 (+90) | 0.6524 | 0.7040 |
| 2 | 90 -> 180 (+90) | 0.9582 | 1.0000 |
| 3 | 91 -> 182 (+91) | 0.8234 | 0.8234 |
| 4 | 91 -> 182 (+91) | 0.8175 | 0.7321 |

### Per-fold detail - Cat (StratifiedGroupKFold, k=4, group=cat_id)

| Fold | n_train (none -> +aug) | Macro-F1 (none) | Macro-F1 (+aug) | "food" F1 (none) | "food" F1 (+aug) |
|---|---|---|---|---|---|
| 0 | 322 -> 554 (+232) | 0.5215 | 0.4244 | 0.3396 | 0.2903 |
| 1 | 319 -> 548 (+229) | 0.5657 | 0.6275 | 0.3396 | 0.3158 |
| 2 | 352 -> 592 (+240) | 0.3414 | 0.3064 | 0.1600 | 0.0556 |
| 3 | 327 -> 559 (+232) | 0.6605 | 0.6145 | 0.6190 | 0.4878 |

All rows are saved in `reports/augmentation_cv_scores.csv`.

## Honest takeaway: signal vs noise

**Dog**: macro-F1 moved from 0.8244 +/- 0.1114 to 0.8433 +/- 0.1319, a change
of **+0.0189**. This is smaller than either standard deviation (0.111 /
0.132) - **within the noise**. Fold 0 and fold 2 improved a lot (+0.086,
+0.042, fold 2 even reaching a perfect 1.0), fold 4 got clearly worse
(-0.085), and folds 1/3 were flat-to-slightly-better. I can't claim this
augmentation helps dog; at best it's "not clearly harmful, possibly a very
mild positive trend that this sample size can't confirm."

**Cat**: macro-F1 moved from 0.5223 +/- 0.1338 to 0.4932 +/- 0.1553, a change
of **-0.0291**, and "food" F1 moved from 0.3646 +/- 0.1896 to 0.2874 +/-
0.1777, a change of **-0.0772**. Both changes are smaller in magnitude than
their standard deviations, so I can't call either one "significant" either -
but both point in the **wrong direction** for the class I specifically
targeted with the most augmentation (food, x3). Fold 1 actually improved
(macro-F1 +0.060, food F1 -0.024), but folds 0, 2 and 3 all got worse on
"food" - fold 2's "food" F1 dropped from 0.16 to 0.06 (it was already the
weakest fold).

**Conclusion**: based on CV - the metric that's supposed to drive this
decision - **this augmentation config does not demonstrate a real
improvement for either animal**. Dog shows a small, statistically
inconclusive positive trend; cat shows a small, statistically inconclusive
negative trend, concentrated on the very class ("food") I was trying to
help. I don't have a definitive explanation for the cat/food result, but a
few honest hypotheses:

1. **"food" CV folds are tiny** (14-23 clips), so a couple of flipped
   predictions swing the per-fold F1 by 0.1-0.2 - both the baseline and
   augmented numbers already have very large stds (0.19/0.18), so a -0.077
   mean shift could easily flip sign with a different seed or fold count.
2. **The backbone is frozen and was trained on natural images.** Pitch
   shift / time stretch / SpecAugment all change what the spectrogram
   "looks like" as an image. For a backbone that's never fine-tuned on
   spectrograms, these transformed "images" might land in a less
   discriminative region of MobileNetV2's 1280-d feature space than the
   originals - effectively adding noisy features rather than informative
   ones, and "food" (the class with the fewest original anchors, x3
   augmented) would be hit hardest by this.
3. **Cat clips are short** (63 frames). A 12.5% time mask removes ~8 of
   those 63 frames - a larger fraction of an already-short clip's context
   than the same 12.5% removes from a dog clip's 126 frames.

## Final one-shot test evaluation

Using the same augmentation config (`AUGMENT_FACTORS`), default head
hyperparameters, trained on the augmented `train` split (early-stopped on the
original, non-augmented `val` split), evaluated **once** on the original,
non-augmented `test` split:

| Animal | n_train (orig + aug) | Test accuracy | Test macro-F1 | vs. default (no aug, from `transfer_learning_summary.md`) |
|---|---|---|---|---|
| Dog | 79 + 79 = 158 | 0.7059 | 0.6983 | acc 0.7647 -> 0.7059, macro-F1 0.6887 -> 0.6983 |
| Cat | 301 + 212 = 513 | 0.6567 | 0.5433 | acc 0.6418 -> 0.6567, macro-F1 0.5130 -> 0.5433 |

Cat per-class report (test):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brushing | 0.80 | 0.44 | 0.57 | 18 |
| food | 0.38 | 0.21 | 0.27 | 14 |
| isolation | 0.67 | 0.94 | 0.79 | 35 |

Dog per-class report (test):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bark | 0.60 | 0.86 | 0.71 | 7 |
| growl | 0.67 | 0.40 | 0.50 | 5 |
| grunt | 1.00 | 0.80 | 0.89 | 5 |

On this single test split, cat's "food" F1 actually went **up** (0.20 ->
0.27) and macro-F1 improved (0.5130 -> 0.5433), while dog's accuracy dropped
(0.7647 -> 0.7059) but macro-F1 ticked up slightly (0.6887 -> 0.6983).

**I'm flagging this explicitly so it isn't over-read**: these single-split
numbers point in a *more favourable* direction for cat than the CV numbers
do - but the test sets are tiny (17 dog clips, 67 cat clips, only 14 of them
"food"), even noisier than the CV folds, and per the anti-leakage rules I
used them only once and not to choose anything. **The CV result is the one
I trust for the "does augmentation help?" question, and it says "not shown
to help, possibly slightly hurts food."** The test result is reported
because the task asks for it, not because it overturns the CV conclusion.

## Total time

`src/augment_cv.py --animal all`: **108.2s** total, on CPU - well within the
"a few minutes" budget (feature precomputation: 4.8s dog + 3.9s cat; raw
waveform loading: 0.1s dog + 0.3s cat; 10 dog fold-runs + 8 cat fold-runs +
2 final evaluations).

## What's NOT done

As instructed, this session stayed at the augmentation level: frozen
MobileNetV2 backbone, default head hyperparameters, no fine-tuning. The
augmentation config tested here (pitch shift +/-2 semitones, time stretch
+/-5%, 3-4% Gaussian noise, one SpecAugment freq+time mask, with the
per-class factors above) **did not show a CV-validated improvement** for
either animal. If a future session revisits augmentation, candidates worth
trying based on the hypotheses above: lighter SpecAugment (smaller mask
fractions, especially for cat's short 63-frame clips), fewer combined
audio transforms per variant (e.g. noise-only or pitch-only variants), or
revisiting augmentation together with backbone fine-tuning (a frozen
ImageNet backbone may simply not "see" these spectrogram transforms as
useful variation).

## Artifacts produced this session

- `src/augment.py`: seeded, per-clip audio (pitch shift, time stretch,
  Gaussian noise) + spectrogram (SpecAugment) augmentation functions.
- `src/augment_cv.py`: without-vs-with-augmentation CV comparison (extends
  `cross_validation.py`'s protocol) + one-shot final test evaluation.
- `reports/augmentation_cv_scores.csv`: per-fold scores, both animals, both
  augmentation settings (18 rows), including cat group-violation columns.
- `reports/augmentation_summary.md`: this report.
- `reports/mobilenet_aug_{dog,cat}_confusion_matrix.png`: final one-shot test
  confusion matrices.
- `models/mobilenet_aug_{dog,cat}_head.keras` (gitignored).

---

# Suggested commit message

```
Add data augmentation for MobileNetV2 heads (dog + cat) - CV comparison

- Add src/augment.py: seeded, per-clip audio augmentation (pitch shift
  +/-2 semitones, time stretch +/-5%, 3-4% Gaussian noise) + SpecAugment
  (one freq + one time mask, ~12.5% each), all using only per-clip
  statistics (no cross-sample leakage)
- Add src/augment_cv.py: extends cross_validation.py's protocol -
  augmentation happens INSIDE the CV loop, applied ONLY to each fold's
  train_idx; validation stays 100% original
  - Per-class factors: cat food x3 / brushing x2 / isolation x1
    (targeted at the weakest classes), dog uniform x2 (mild imbalance,
    general regularisation)
  - Sanity check: no-augmentation re-run reproduces the CV repere exactly
    (dog 0.8244+/-0.1114, cat 0.5223+/-0.1338, food 0.3646+/-0.1896)
  - Cat group-violation check after augmentation: 0/4 folds, for all folds
- Results (CV macro-F1, mean+/-std):
  - Dog: 0.8244+/-0.1114 -> 0.8433+/-0.1319 (+0.019, within noise)
  - Cat: 0.5223+/-0.1338 -> 0.4932+/-0.1553 (-0.029, within noise, wrong
    direction); "food" F1: 0.3646+/-0.1896 -> 0.2874+/-0.1777 (-0.077)
  - Honest conclusion: this augmentation config is NOT shown to help
    either animal in CV (the metric that should drive the decision)
- One-shot final test (touched once, config applied regardless of the CV
  verdict, for completeness): dog acc 0.7647->0.7059, macro-F1
  0.6887->0.6983; cat acc 0.6418->0.6567, macro-F1 0.5130->0.5433, food F1
  0.20->0.27 - more favourable than CV, but on tiny test sets (17/67
  clips), not used to override the CV conclusion
- Add reports/augmentation_cv_scores.csv, reports/augmentation_summary.md,
  reports/mobilenet_aug_{dog,cat}_confusion_matrix.png
- Total wall-clock time: 108.2s on CPU
- No fine-tuning in this session, as instructed
```
