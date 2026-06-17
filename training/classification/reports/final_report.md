# Pet Translator - Audio Classification: Final Technical Report

This is a consolidated, end-to-end report of my work on the audio classification
part of the "Pet Translator" project. It pulls together everything from the
step-by-step reports already produced in `training/classification/reports/`
into a single narrative for grading. No new experiments, code runs, or
retraining happened while writing this report - every number below is taken
as-is from an earlier report, and I point to the source report wherever it is
useful to dig deeper.

I have tried to be as honest as possible about what worked, what didn't, and
where the numbers should not be over-interpreted. Where a result is "within
noise" (smaller than the relevant standard deviation), I say so explicitly,
and I do not present it as an improvement.

*Update: steps 8 and 9 were added after the original version to incorporate
two subsequent experiments (denoising comparison and class-imbalance
techniques). Every number in those steps is taken directly from their
respective sub-reports (`denoise_comparison_summary.md` and
`imbalance_experiments_summary.md`).*

## 1. Problem and objective

"Pet Translator" takes a short audio clip of a pet and turns it into a
human-readable message about what the pet might be "saying" or feeling. My
part of the project is the audio classification step: given a clip and the
animal it came from, predict a label that the downstream LLM module (student
3) and the app (student 4) can turn into that message.

The task is split into **two completely independent 3-class models**, one per
animal. The app asks the user which animal the clip belongs to, and routes the
clip to the corresponding model - there is no automatic animal detection.

| Animal | Classes | What the label describes |
|---|---|---|
| Dog | `bark`, `growl`, `grunt` | The *type* of vocalization |
| Cat | `brushing`, `food`, `isolation` | The behavioral *context* the cat is reacting to |

Looking at these two label sets side by side, there is an obvious asymmetry:
the dog classes describe the **acoustic type** of sound (a bark vs a growl vs
a grunt could in principle happen in many different situations), while the cat
classes describe the **situation** the cat is vocalizing in (being brushed,
waiting for food, or left alone) rather than an acoustically-defined "type" of
meow.

I want to be upfront that this asymmetry is **not a design choice I made** -
it is inherited directly from how the two source datasets were labeled by
their original authors:

- The dog dataset (`shivarao`) organizes its 113 clips by sound type
  (bark/growl/grunt), independent of any situational context.
- The cat dataset (CatMeows, Pirrone et al., 2020) was built specifically to
  study **context-dependent** meow production: its three classes come from the
  filename prefixes `B`/`F`/`I` (brushing / waiting-for-food / isolation in an
  unfamiliar environment), i.e. the *recording situation*, not an
  acoustically-defined meow category.

Practically, this means the two production models answer slightly different
questions: the dog model says *what kind of sound* the dog made, while the cat
model says *what situation* probably produced the meow. This has a direct
consequence for the LLM/app integration (see section 8) - the generated text
for dog should talk about the sound itself ("your dog is barking/growling"),
while for cat it can talk about the inferred situation ("this sounds like a
food-related meow"). I flag this so the team doesn't assume both models speak
the same "language" of labels.

## 2. Data

### Dog - `shivarao` dataset (113 clips)

| Class | Total | Train | Val | Test |
|---|---|---|---|---|
| bark | 46 | 32 | 7 | 7 |
| growl | 33 | 23 | 5 | 5 |
| grunt | 34 | 24 | 5 | 5 |
| **Total** | **113** | **79 (69.9%)** | **17 (15.0%)** | **17 (15.0%)** |

The class balance is mild (ratio bark/growl ≈ 1.4), and the split is a plain
class-stratified split (`preprocessing_summary.md`). The class proportions
stay almost identical across train/val/test (e.g. bark stays ~40.5-41.2%
everywhere).

**Known limitation**: the `shivarao` dataset does not provide any
speaker/individual-dog ID. I cannot tell whether two clips come from the same
dog, so I cannot build a group-aware split for dog the way I do for cat (see
section 3). This is documented and re-flagged in every report that uses the
dog cross-validation numbers - it is the one open caveat on the dog results.

### Cat - CatMeows dataset (Pirrone et al., 2020) - 440 clips, 21 cats

| Class | Total | Share |
|---|---|---|
| isolation | 221 | 50.2% |
| brushing | 127 | 28.9% |
| food | 92 | 20.9% |

This is a real imbalance: the largest class (`isolation`) has **2.40x** as
many clips as the smallest (`food`). `food` being both the smallest class and,
as it turns out, the hardest to classify, is the single biggest theme running
through this whole report (see sections 4-6).

Every clip's filename encodes which of the 21 individual cats it came from
(e.g. `I_ANI01_...` -> cat `ANI01`). The cat split (`preprocessing_summary.md`)
is **group-aware at the individual-cat level**:

| Split | Clips | Share | Cats |
|---|---|---|---|
| Train | 301 | 68.4% | 15 |
| Val | 72 | 16.4% | 3 |
| Test | 67 | 15.2% | 3 |

This assignment of 21 cats to 3 splits was found via a random search (5000
trials, seed=42) that picked, among all assignments that **never split a
single cat across two sets**, the one that best matched both the 70/15/15
sample-count target and the global class distribution. The resulting
imbalance across splits is small (within ~3 points of the global distribution
for every class).

Both source files (raw `.wav`s) and processed `.npy` feature arrays are
**gitignored** and were never committed - this report and the rest of
`reports/` are the durable record of what was measured.

A side note on the cat data itself, from `cat_eda_summary.md`: the original
CatMeows paper is generally cited with much shorter clip durations
(~0.3-0.4s), but the actual `.wav` files I have average ~1.8s (median 1.81s).
My best guess is that the published duration refers to the meow sound itself,
while these files include a margin of quiet before/after it. I did not chase
this further since it doesn't affect the pipeline (both durations get
pad/cropped to a fixed length anyway), but I flag it as a discrepancy I
noticed and didn't fully resolve. All 440 cat clips were verified non-corrupt
and non-silent (`cat_eda_summary.md`).

## 3. Methodology and anti-leakage guardrails

This section is, I think, the strongest part of this project - not because
any single number is impressive, but because the **evaluation protocol itself
is rigorous enough that I can trust the numbers it produces**, including the
disappointing ones.

### Why not a CNN trained from scratch

Before touching any model, I built a baseline (`baseline_summary.md`):
flatten each log-mel spectrogram into a single vector (8,064 values for dog,
4,032 for cat) and fit a `LogisticRegression(class_weight="balanced")`, with
`C` selected on the validation set from `{0.0001, 0.001, 0.01, 0.1, 1, 10}`.

The selected value was the **strongest regularization tested, `C=0.0001`**,
for *both* animals. I checked this wasn't just an artifact of the grid's
smallest value: with less regularization (`C=0.01` and above), the model
reaches a **training** macro-F1 of essentially 1.0 - i.e. it perfectly
memorizes the training set - while its **validation** macro-F1 stays flat or
drops. With 8,064 features and 79 dog training samples (or 4,032 features and
301 cat training samples), this is exactly the overfitting signature I would
expect.

This result is the concrete reason I never trained a CNN from scratch on this
data: if a *linear* model with thousands of features needs this much
regularization to avoid memorizing 79-301 examples, an unregularized CNN with
many more parameters would almost certainly do the same, and its results would
depend heavily on initialization and training length - unstable and hard to
compare against. **Frozen transfer learning** is the natural answer: borrow
features learned on a much larger dataset (ImageNet for MobileNetV2, AudioSet
for YAMNet and AST), freeze them completely, and train only a tiny head
(`Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)`, a few hundred
parameters) on top.

### Cross-validation, not a single train/val/test split

My first transfer-learning run (`transfer_learning_summary.md`) used a single
split: 17 dog test clips and 67 cat test clips. At that size, **each
individual misclassified file moves accuracy by ~6 points for dog**, and the
67-clip cat test set is a single draw from 21 individual cats - the score
could look better or worse purely depending on which cats happened to land in
test. Before doing any tuning, I therefore moved to cross-validation
(`cross_validation_summary.md`) to get a mean ± std per approach, which tells
me how much a given number can be trusted.

**Cat: `StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)`,
`groups=cat_id` (mandatory).** Cat vocalizations are known to be quite
individual-specific. If clips from the same cat could land in both train and
validation, a model could partly learn to recognize *that cat's voice* rather
than the sound class - a real leakage risk. I checked k=3 and k=4 explicitly;
both give 0 group violations and keep all 3 classes represented in every fold.
I picked k=4 because it gives one more independent estimate than k=3 while
keeping validation folds reasonably sized (88-121 clips, already larger than
the original 67-clip test set).

For every cat fold, the script computes the set of `cat_id`s in train and in
validation, checks their intersection, and **asserts it is empty** - the run
would crash if this were ever violated:

| Fold | n_train | n_val | train cats | val cats | violations |
|---|---|---|---|---|---|
| 0 | 322 | 118 | 16 | 5 | **0** |
| 1 | 319 | 121 | 15 | 6 | **0** |
| 2 | 352 | 88 | 17 | 4 | **0** |
| 3 | 327 | 113 | 15 | 6 | **0** |

This 0-violations check was re-run and re-verified in every later session that
touches the cat data (head tuning, augmentation, classifier comparison, AST) -
it's not a one-off check, it's part of the protocol.

**Dog: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, by class
label (honest limitation).** The `shivarao` dataset provides no
speaker/individual-dog ID, so a group-aware split is not possible for dog. 113
clips / 5 folds gives ~22-23 validation clips per fold (larger than the
original 17-clip test set), and even the smallest class (`growl`, 33 clips)
is comfortably split across folds. I'm flagging this explicitly: **if the
dataset contains multiple recordings of the same dog, the dog CV numbers could
be slightly optimistic** - this is the one place in the protocol where I
cannot rule out leakage, simply because the information needed to rule it out
doesn't exist in the dataset.

### Per-fold / normalization correctness

`preprocess.py` computes a global `(mean, std)` from the **training set only**
and uses it to normalize all log-mel spectrograms before they're saved to
`data/processed/`. For MobileNetV2, the spectrogram-to-image conversion then
applies a **per-sample min-max rescale to [0, 1]** before resizing and feeding
the frozen backbone. I worked out (and `cross_validation_summary.md` documents
this) that for any monotonically increasing affine transform `x -> a*x + b`
with `a > 0` - which is exactly what `(x - mean) / std` is, since `std > 0` -
**per-sample min-max produces the exact same result with or without that
transform**: the min and max get shifted/scaled by the same `a, b` and cancel
out in the min-max formula. So the global normalization is mathematically a
no-op for MobileNetV2 features, which means these features are
**fold-independent**: they can be computed once per animal (not per fold) and
indexed into per fold, with zero risk of a global statistic leaking between a
fold's train and validation sets. For YAMNet and AST, no cross-sample
statistic is ever computed at all (YAMNet processes each clip independently;
AST's normalization constants are fixed AudioSet-wide values baked into the
pretrained checkpoint, not derived from my data). The only statistic that
genuinely depends on a fold's training labels, `class_weight="balanced"`, is
always computed from `y[train_idx]` only, inside `tl_common.train_head`.

### Test touched once

In every session, model/configuration selection is based **exclusively on
mean CV macro-F1**. The held-out test split (`data/processed/<animal>/test_*`)
is evaluated **exactly once per session**, at the very end, using whichever
configuration won on CV - never used to choose between alternatives. This is
why, in section 4 below, some "final test" numbers move around session to
session even when the CV story doesn't change much: they're a single honest
draw from a small test set (17 dog / 67 cat clips), not a number that was
searched over.

## 4. Chronology of experiments

Each step below reuses the protocol from section 3 (frozen backbone, CV
selection, test touched once) unless noted otherwise. For each step I give the
goal, the headline numbers, and my conclusion on whether the result is a real
signal or within noise.

### Step 1 - Baseline (logistic regression on flattened spectrograms)

**Goal**: establish a floor (majority-class `DummyClassifier`) and a simple,
deterministic reference point before touching any "real" model.

| | Dog accuracy | Dog macro-F1 | Cat accuracy | Cat macro-F1 |
|---|---|---|---|---|
| Floor (majority class) | 0.4118 | 0.1944 | 0.5224 | 0.2288 |
| Logistic regression (C=0.0001) | 0.8235 | 0.8190 | 0.4328 | 0.3333 |

For cat, the baseline's `food` F1 was **0.00** - it never predicted `food`
correctly at all on the 14 `food` test clips.

**Conclusion**: for dog, even a linear baseline already does reasonably well
(0.82 accuracy/macro-F1) on a 17-clip test set - encouraging, but I was
cautious about reading too much into it given the test size. For cat, the
baseline beats the floor on macro-F1 but not on raw accuracy, and completely
fails on `food`. This set the bar: any "real" model should beat 0.33 macro-F1
for cat, and `food` recognition was already, at this very first step, the
clear weak point.

### Step 2 - Transfer learning, first run (YAMNet vs MobileNetV2, single split)

**Goal**: get both frozen-backbone pipelines (YAMNet/AudioSet,
MobileNetV2/ImageNet "spectrogram-as-image") working end to end with
untuned defaults, and get a first comparison point. No tuning yet.

| | Dog accuracy | Dog macro-F1 | Cat accuracy | Cat macro-F1 |
|---|---|---|---|---|
| Floor | 0.4118 | 0.1944 | 0.5224 | 0.2288 |
| Logistic regression (baseline) | 0.8235 | 0.8190 | 0.4328 | 0.3333 |
| YAMNet + dense head | 0.8235 | 0.8244 | 0.4925 | 0.3454 |
| MobileNetV2 + dense head | 0.7647 | 0.6887 | 0.6418 | 0.5130 |

**Conclusion**: for dog, YAMNet and the logistic baseline were essentially
tied (both 14/17 correct); MobileNetV2 was behind both, but still clearly
above the floor. For cat, MobileNetV2 was the clear leader on macro-F1
(0.5130) and the only approach beating the floor on **both** accuracy and
macro-F1. I explicitly flagged that, with test sets this small (17/67 clips),
none of these rankings should be considered final - which is exactly why the
next step was cross-validation rather than tuning.

Both runs together took under two minutes on CPU (~61s YAMNet, ~33s
MobileNetV2, including one-time model downloads), confirming CPU is
sufficient at this scale.

### Step 3 - Cross-validation (establishing the protocol from section 3)

**Goal**: replace the single noisy test score with a mean ± std across CV
folds, using the *exact same* untuned configuration as step 2 - nothing about
the models changed, only the evaluation protocol.

| Animal | Approach | Folds | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
|---|---|---|---|---|
| Dog | YAMNet | 5 | 0.7972 ± 0.0771 | 0.7986 ± 0.0740 |
| Dog | MobileNetV2 | 5 | 0.8229 ± 0.1109 | 0.8244 ± 0.1114 |
| Cat | YAMNet | 4 | 0.4056 ± 0.0913 | 0.3565 ± 0.0728 |
| Cat | MobileNetV2 | 4 | 0.5603 ± 0.1431 | 0.5223 ± 0.1338 |

**Conclusion**: the cat ranking from step 2 held up - MobileNetV2 clearly beats
YAMNet on macro-F1 (0.5223 vs 0.3565, a gap larger than either std). For dog,
the picture is murkier: MobileNetV2's CV mean (0.8244) is actually *higher*
than its single-split score (0.6887) - the original 17-clip test happened to
be a hard split for MobileNetV2 - while YAMNet's CV mean (0.7986) is close to
its single-split score (0.8244). Both dog approaches comfortably beat the floor
in every fold. The standard deviations (0.07-0.14) confirmed that, with this
little data, a single test score shouldn't be read to more than roughly ±10
points of precision. From here on, **MobileNetV2 + dense head (0.8244 ± 0.1114
dog, 0.5223 ± 0.1338 cat) became the reference ("repere") that every later
session tries to beat**; YAMNet stayed in the repo as a documented comparison
point but wasn't tuned further.

Total CV run time: 68.4s on CPU for both animals (18 train/eval runs total).

### Step 4 - Head hyperparameter tuning

**Goal**: with the backbone now fixed (frozen MobileNetV2), tune the small
dense head (`dense_units`, `dropout`, `l2`, `lr`). A sequential/targeted search
of 12 configs per animal (vs. 54 for the full grid) was run, scored purely by
mean CV macro-F1 on the *same* folds as step 3.

| Animal | Metric | Default | Best found | Gap | Verdict |
|---|---|---|---|---|---|
| Dog | CV macro-F1 | 0.8244 ± 0.1114 | 0.8604 ± 0.0837 (dropout=0.5, l2=1e-4) | +0.036 | within noise |
| Cat | CV macro-F1 | 0.5223 ± 0.1338 | 0.5367 ± 0.1043 (dropout=0.3, l2=1e-3) | +0.0144 | within noise |
| Cat | `food` F1 | 0.3646 ± 0.1896 | 0.3672 ± 0.1497 | +0.0026 | essentially unchanged |

In both cases the gap is smaller than either configuration's standard
deviation, so neither "best" config is a statistically meaningful improvement.

The one-shot final test run with the best dog config did move the test
macro-F1 from 0.6887 to 0.7500 (same accuracy, 0.7647) - `grunt` recall went
from 4/5 to 5/5 and `growl` recall from 1/5 to 2/5, at the cost of one `bark`.
One swapped file changes the picture quite a bit on a 17-clip test set, which
is consistent with the "within noise" CV conclusion. For cat, the best config's
one-shot test result was **identical** to the default (accuracy 0.6418,
macro-F1 0.5130, `food` F1 0.20) - the only hyperparameter that differed
(`l2=1e-3` vs `0`) changed nothing about this particular split's predictions.

**Conclusion**: head-only tuning is mostly exhausted, especially for cat -
`food` F1 ranged from 0.31 to 0.40 across all 12 configs with large
per-config std (0.09-0.21), and nothing in this search moved the needle beyond
noise. Total time: 258.6s (~4.3 min) on CPU for both animals.

### Step 5 - Data augmentation

**Goal**: the head-tuning conclusion pointed away from the head as the
bottleneck, so this step tries a different lever - data augmentation, targeted
at the weakest classes. Backbone stays frozen, head reverts to the **default**
hyperparameters (so any change is attributable to augmentation alone).

Techniques: pitch shift (±2 semitones), time stretch (±5%), additive Gaussian
noise (3-4% of peak amplitude), and one SpecAugment frequency + time mask
(~12.5% each), applied with per-class multipliers - cat `food` x3 / `brushing`
x2 / `isolation` x1 (targeted rebalancing toward the weakest class), dog
uniform x2 (mild imbalance, general regularization). Augmentation is applied
**inside the CV loop, only to each fold's training indices** - validation
stays 100% original, and augmented cat clips inherit their source cat's ID so
the group-leakage check (0 violations, re-verified after augmentation) still
holds.

| Animal | Metric | Without augmentation | With augmentation | Delta | Verdict |
|---|---|---|---|---|---|
| Dog | CV macro-F1 | 0.8244 ± 0.1114 | 0.8433 ± 0.1319 | +0.0189 | within noise |
| Cat | CV macro-F1 | 0.5223 ± 0.1338 | 0.4932 ± 0.1553 | -0.0291 | within noise, wrong direction |
| Cat | `food` F1 | 0.3646 ± 0.1896 | 0.2874 ± 0.1777 | -0.0772 | within noise, wrong direction |

The "without augmentation" re-run reproduced the step-3 CV repere to 4
decimals, confirming the protocol was intact and the comparison is
apples-to-apples.

The one-shot final test results were, on this particular small split, more
favorable: dog accuracy 0.7647→0.7059 / macro-F1 0.6887→0.6983; cat accuracy
0.6418→0.6567 / macro-F1 0.5130→0.5433, `food` F1 0.20→0.27. I'm flagging this
explicitly so it isn't over-read: the test sets are tiny (17/67 clips, only 14
`food`), even noisier than the CV folds, and per the protocol they were not
used to choose anything. **The CV result is the one I trust for "does
augmentation help?", and it says no** - for cat specifically, the change went
in the *wrong* direction for exactly the class (`food`) I targeted with the
most augmentation (x3).

**Conclusion**: this augmentation configuration does not demonstrate a real
improvement for either animal in CV. My best honest guess for *why* cat/`food`
got worse: the `food` CV folds are tiny (14-23 clips, std already 0.19), the
MobileNetV2 backbone is frozen and trained on natural photos - pitch/time/
SpecAugment transforms change what the spectrogram "looks like" as an image in
ways a never-fine-tuned backbone may not represent usefully - and cat clips are
short (63 frames), so a 12.5% time mask removes a larger fraction of an
already-short clip's context than for dog's 126-frame clips. Total time:
108.2s on CPU for both animals.

### Step 6 - Classifier family comparison

**Goal**: with 1280-dim MobileNetV2 features and very few training samples per
fold (~79-91 dog, ~310-352 cat), this is the classic "many features, few
samples" regime where regularized linear/margin classifiers are often reported
to match or beat a small neural net. This session swaps the classifier on top
of the *same* frozen MobileNetV2 features, same CV folds, same seed.

Classifiers compared: `dummy_floor`, `logreg`, `svm_linear`, `svm_rbf`
(3x3 `C x gamma` grid, best selected on CV), `lda` (shrinkage="auto"), and
`dense_head` (the existing reference, unchanged).

**Dog (5-fold)**

| Classifier | Accuracy | Macro-F1 |
|---|---|---|
| **dense_head** | 0.8229 ± 0.1109 | **0.8244 ± 0.1114** |
| lda | 0.8047 ± 0.1150 | 0.8074 ± 0.1136 |
| logreg | 0.7960 ± 0.1198 | 0.7977 ± 0.1215 |
| svm_linear | 0.7696 ± 0.0953 | 0.7742 ± 0.0931 |
| svm_rbf (C=10, gamma=scale) | 0.7348 ± 0.1353 | 0.7307 ± 0.1383 |
| dummy_floor | 0.4071 ± 0.0178 | 0.1928 ± 0.0060 |

**Cat (4-fold, group=cat_id)**

| Classifier | Accuracy | Macro-F1 | `food` F1 |
|---|---|---|---|
| **dense_head** | 0.5603 ± 0.1431 | **0.5223 ± 0.1338** | 0.3646 ± 0.1896 |
| svm_rbf (C=1, gamma=scale) | 0.5273 ± 0.1512 | 0.4854 ± 0.1231 | 0.3548 ± 0.1340 |
| lda | 0.5260 ± 0.0923 | 0.4744 ± 0.0520 | 0.3270 ± 0.0993 |
| logreg | 0.5088 ± 0.0933 | 0.4642 ± 0.0569 | 0.3094 ± 0.1247 |
| svm_linear | 0.4943 ± 0.1169 | 0.4528 ± 0.0787 | 0.3010 ± 0.1287 |
| dummy_floor | 0.4961 ± 0.0604 | 0.2205 ± 0.0186 | 0.0000 ± 0.0000 |

**Conclusion**: `dense_head` has the best mean CV macro-F1 for both animals,
but every gap to the runner-up (dog: -0.017 vs LDA; cat: -0.037 vs `svm_rbf`)
is far smaller than either standard deviation - no classifier change is a
significant improvement for either animal. For `food` specifically, LDA and
`svm_rbf` have roughly **half the standard deviation** of `dense_head`
(0.099/0.134 vs 0.190) - more *consistent* across folds, but not higher on
average, so this is a minor observation, not a fix. The fact that a neural
head, regularized linear models, margin-based kernels, and a generative linear
model (LDA) all land within noise of each other on the *same* feature vectors
is itself informative: it points at the **frozen MobileNetV2 features
themselves** as the ceiling, not the classifier sitting on top of them - which
is exactly the motivation for step 7. Total time: 40.2s on CPU for both
animals (full comparison + RBF grid + final evaluation).

### Step 7 - AST (Audio Spectrogram Transformer) as an alternative frozen backbone

**Goal**: steps 4-6 all pointed at the frozen MobileNetV2 features as the
ceiling. This step replaces the **feature extractor itself** with AST
(`MIT/ast-finetuned-audioset-10-10-0.4593`, an audio-native transformer
pretrained and fine-tuned on the full AudioSet, 527 classes including many
animal sounds) - the hypothesis being that audio-native features, especially
on a wide range of animal vocalizations, might separate the classes better
than ImageNet features, especially for cat `food`.

Each clip is pad/cropped to the same fixed duration as every other approach
(4s dog / 2s cat), passed through the frozen AST encoder
(`torch.no_grad()`), and mean-pooled over the token sequence to a 768-dim
embedding. Two lightweight classifiers were evaluated on top, with the *same*
CV folds/seed as every other approach: `dense_head` and `logreg`.

**Dog (5-fold)**

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| **ast_dense_head** | 0.8411 ± 0.0724 | **0.8456 ± 0.0674** |
| ast_logreg | 0.8320 ± 0.0833 | 0.8377 ± 0.0787 |
| yamnet (repere) | 0.7972 ± 0.0771 | 0.7986 ± 0.0740 |
| mobilenet (repere) | 0.8229 ± 0.1109 | 0.8244 ± 0.1114 |

**Cat (4-fold, group=cat_id)**

| Approach | Accuracy | Macro-F1 | `food` F1 |
|---|---|---|---|
| ast_dense_head | 0.5900 ± 0.1416 | 0.4985 ± 0.1462 | 0.3652 ± 0.2082 |
| **ast_logreg** | 0.5647 ± 0.1309 | **0.5064 ± 0.0859** | 0.3072 ± 0.1190 |
| yamnet (repere) | 0.4056 ± 0.0913 | 0.3565 ± 0.0728 | n/a |
| mobilenet (repere) | 0.5603 ± 0.1431 | 0.5223 ± 0.1338 | 0.3646 ± 0.1896 |

Comparing the best AST classifier per animal to the MobileNet repere:

- Dog: AST `dense_head` 0.8456 ± 0.0674 vs MobileNet 0.8244 ± 0.1114 -> delta
  +0.0212, **within noise**.
- Cat: AST `logreg` 0.5064 ± 0.0859 vs MobileNet 0.5223 ± 0.1338 -> delta
  -0.0159, **within noise**. `food` F1: AST `logreg` 0.3072 ± 0.1190 vs
  MobileNet 0.3646 ± 0.1896 -> delta -0.0574, **within noise**. (Against
  YAMNet, AST cat macro-F1 *is* a real improvement: +0.1499, more than 1 std.)

Final one-shot test (touched once, CV-selected classifier per animal):

- **Dog**, `ast_dense_head`: accuracy = 0.8824, macro-F1 = 0.8857 -
  `bark` F1 0.86, `growl` F1 0.80, **`grunt` F1 1.00**.
- **Cat**, `ast_logreg`: accuracy = 0.6418, macro-F1 = 0.5059, **`food` F1 =
  0.3077** - `brushing` F1 0.37, **`isolation` F1 0.84**.

**Conclusion**: AST lands within noise of the MobileNet repere for both
animals (and for `food` specifically). Combined with steps 4-6, this is now
the **fourth architecturally different lever** - head hyperparameters, data
augmentation, classifier family, and now the entire feature-extraction
backbone (ImageNet CNN -> AudioSet CNN -> AudioSet Transformer) - landing in
the same range. AST inference itself ran on a Colab GPU (135.0s total for
~550 clips); `src/ast_transfer.py` documents that AST is too heavy for
CPU-only local inference at a useful speed, which becomes relevant for the
production choice in section 8.

### Step 8 - Raw vs. denoised audio comparison

**Goal**: a teammate built an audio-cleaning pipeline and produced a denoised
version of every clip, with identical filenames and folder structure
(`dataset_nettoye/data/clean/`). I evaluated whether training on this denoised
audio changes any CV metric - same model, same CV protocol, same splits, same
seed; only the audio source changes.

All 113 dog and 440 cat manifest files were verified present in both the raw
and clean sources before running. The RAW condition (`dataset_nettoye/data/raw/`)
reproduced the step-3 repères exactly (dog 0.8244, cat 0.5223), confirming
the protocol was intact before reading the CLEAN numbers.

| Animal | Condition | Macro-F1 (CV) | Delta | Verdict |
|---|---|---|---|---|
| Dog | RAW | 0.8244 ± 0.0997 | — | — |
| Dog | CLEAN | 0.8354 ± 0.0972 | +0.0110 | within noise (0.11× std) |
| Cat | RAW | 0.5223 ± 0.1158 | — | — |
| Cat | CLEAN | 0.4694 ± 0.0811 | -0.0529 | within noise (0.46× std) |

Cat `food` F1: RAW 0.3646, CLEAN 0.3172, delta -0.0474, within noise.

**Conclusion**: denoising has no significant effect in either direction for
either animal. This eliminates audio noise/quality as an explanation for the
cat `food` plateau. Total time: 89s on CPU.

### Step 9 - Class-imbalance techniques: focal loss and SMOTE

**Goal**: target the cat `food` class imbalance (92/440 clips, 20.9%) directly,
with two techniques not yet tried: focal loss (replaces cross-entropy) and SMOTE
(over-samples the minority class in feature space, train fold only). Four
conditions were compared on the same frozen MobileNetV2 features, same CV folds:

- **BASELINE**: cross-entropy + `class_weight='balanced'` (reference)
- **FOCAL**: focal loss (`gamma=2.0`), no `class_weight` (focal loss handles
  imbalance via its modulating factor; combining with `class_weight` would
  double-count the correction)
- **SMOTE**: cross-entropy + `class_weight`, SMOTE applied **on `X[train_idx]`
  of each fold only** — validation always contains only real, original clips.
  For cat, the group-safe split guarantees synthetic samples interpolate only
  among training-fold cats, not validation cats.
- **SMOTE+FOCAL**: SMOTE + focal loss (no `class_weight`)

0 group violations across all folds and all four conditions. SMOTE added
~18-21 synthetic samples per dog fold and ~146-203 per cat fold.

**Dog** (all deltas < 0.11× std — all noise):

| Condition | Macro-F1 (CV mean ± std) | Delta |
|---|---|---|
| BASELINE | 0.8244 ± 0.1114 | — |
| FOCAL | 0.8340 ± 0.1019 | +0.0096 (noise) |
| SMOTE | 0.8345 ± 0.1145 | +0.0101 (noise) |
| SMOTE+FOCAL | 0.8318 ± 0.0935 | +0.0074 (noise) |

**Cat**:

| Condition | Macro-F1 (CV) | `food` F1 | `food` Precision | `food` Recall |
|---|---|---|---|---|
| BASELINE | 0.5223 ± 0.1338 | 0.3646 | 0.3648 | 0.3751 |
| FOCAL | 0.4967 ± 0.1160 | 0.3168 | 0.3380 | 0.2991 |
| SMOTE | 0.4718 ± 0.1239 | 0.3561 | 0.3128 | 0.4688 |
| SMOTE+FOCAL | 0.4773 ± 0.1585 | 0.3777 | 0.3179 | 0.5021 |

All macro-F1 deltas for cat are negative and within noise (largest: -0.0505,
0.38× std). The precision/recall breakdown on `food` tells the real story:
**SMOTE raises `food` recall from 0.38 to 0.47-0.50, but simultaneously lowers
`food` precision from 0.36 to 0.31-0.32, leaving F1 essentially unchanged.**
The model generates more `food` predictions, catching more true positives, but
also more false positives — because the MobileNetV2 features are not distinctive
enough to make those extra predictions reliably precise. Focal loss alone also
fails to improve `food` (F1 0.3168 vs 0.3646 baseline, within noise).

**Conclusion**: this is the fifth independent lever showing no significant
improvement. The production model is unchanged. Total time: 132s on CPU.

## 5. Central diagnosis: the bottleneck is the data, not the model

Section 4 walks through five independent levers, each targeting a different
part of the pipeline:

1. **Head hyperparameters** (dropout, L2, width, learning rate) - step 4
2. **Training data itself**, via augmentation targeted at the weakest classes
   - step 5
3. **The classifier family** on top of the frozen features (neural net, linear/
   margin/generative classifiers) - step 6
4. **The feature-extraction backbone itself** (ImageNet CNN -> AudioSet CNN ->
   AudioSet Transformer) - step 7
5. **Class-imbalance handling** (focal loss and SMOTE on features, train fold
   only) - step 9

Additionally, step 8 tested an orthogonal hypothesis: that the cat `food`
plateau might be partly explained by audio noise quality. Training on a
teammate's denoised version of the same clips (same CV splits, same seed)
produced deltas of +0.0110 for dog and -0.0529 for cat — both within noise.
This eliminates audio quality as an explanation alongside the five levers above.

None of these five levers, each evaluated under the same CV protocol with the
same seed and folds, produced a change larger than its own standard deviation.
For cat's `food` class in particular - the single most-tested number in this
whole project - the result is essentially the same story every time: roughly
0.31-0.40 mean F1, with a standard deviation (0.09-0.21) often *larger* than
the differences between approaches.

The precision/recall breakdown from step 9's SMOTE results provides a concrete
illustration of *why* the `food` plateau is so stubborn. SMOTE raised `food`
recall from 0.38 to 0.47-0.50 (the model issues more `food` predictions and
catches more true positives), but simultaneously lowered `food` precision from
0.36 to 0.31-0.32 (it also predicts `food` for more non-food clips). The F1
stayed flat. This is the classic over-sampling failure mode when the underlying
features lack discriminative power: generating more synthetic `food` examples
teaches the model to predict `food` more often, but the MobileNetV2 features
are not distinctive enough to make those additional predictions reliably
correct.

I read this as an **elimination-based conclusion**: if the limiting factor
were the head's capacity/regularization, step 4 should have shown a real gain
for at least one config. If it were the amount or diversity of training
examples the model sees, step 5's targeted 3x augmentation of `food` should
have helped `food` specifically - instead it moved in the wrong direction. If
it were the classifier's inductive bias (neural vs. linear vs. margin vs.
generative), step 6 should have shown one family pulling ahead - instead all
five land within ~1 std of each other. If it were the choice of pretrained
feature space (natural images vs. general audio events), step 7's switch to an
audio-native transformer should have shown a clear jump - instead it's within
noise of MobileNetV2. If it were class imbalance per se, focal loss or SMOTE
(step 9) should have improved `food` F1 - instead SMOTE only redistributed
errors between precision and recall without net gain. If it were audio noise,
step 8's denoised audio should have helped - it did not.

Five very different levers, plus a denoising test — six "no significant change"
results — with `food` (92 clips spread across 21 cats, ~4.4 clips/cat on
average) being the hardest and noisiest number throughout. The most consistent
explanation across all of this is that **the bottleneck is the dataset itself**
- specifically its size and the resulting per-fold variance - rather than
anything about the model, the classifier, the backbone, or the signal quality
sitting on top of it. This doesn't mean the ceiling is *permanently* fixed (see
section 9 for what I think could actually move it), but none of the levers I
had access to did.

## 6. Honest final results

The table below reports **cross-validated macro-F1** (mean ± std) for every
approach where a CV number exists. The logistic-regression baseline was only
ever evaluated on the single train/val/test split (step 1) - I did not find a
CV number for it in any report, so I report its single-split test score
separately rather than inventing a CV figure for it.

| Approach | Dog macro-F1 (CV) | Cat macro-F1 (CV) | Cat `food` F1 (CV) |
|---|---|---|---|
| Floor (`dummy_floor`) | 0.1928 ± 0.0060 | 0.2205 ± 0.0186 | 0.0000 ± 0.0000 |
| Baseline (logreg) - single-split test only | 0.8190 (test) | 0.3333 (test) | 0.00 (test) |
| YAMNet + dense head | 0.7986 ± 0.0740 | 0.3565 ± 0.0728 | n/a |
| MobileNetV2 + dense head (reference) | 0.8244 ± 0.1114 | 0.5223 ± 0.1338 | 0.3646 ± 0.1896 |
| AST + best classifier | 0.8456 ± 0.0674 | 0.5064 ± 0.0859 | 0.3072 ± 0.1190 |

**Dog**: solid across the board. CV macro-F1 sits in the **0.82-0.85** range
for both frozen-CNN approaches (MobileNetV2 and AST), comfortably above the
floor (0.19) and the YAMNet repere (0.80). On the final AST test evaluation,
`grunt` reaches **F1 = 1.00** (perfect), with `bark` at 0.86 and `growl` at
0.80. I'd call the dog model genuinely usable, with the one caveat that the CV
numbers may be very slightly optimistic due to the missing dog-speaker-ID
issue (section 3).

**Cat**: a mixed picture by class. `isolation` is reliably recognized (test
F1 = 0.84 on the AST run, 0.77 on the MobileNetV2 run). `brushing` is
reasonable (test F1 0.37-0.57 depending on the run). `food` is **weak**
everywhere: CV F1 in the 0.31-0.40 range across four architecturally different
approaches (steps 4-7), and 0.20-0.31 on the one-shot test evaluations. Overall
cat macro-F1 (CV) sits around **0.50-0.52** for MobileNetV2/AST, both clearly
above the 0.22 floor but well below the dog numbers.

## 7. Comparison to the literature

I want to be careful here, because **I have not done a systematic literature
review as part of this project** - the only external reference that appears in
my reports is the CatMeows dataset paper itself (Pirrone et al., 2020), cited
as the source of the data, not as a benchmark to compare against. So rather
than quoting specific numbers from other studies (which I don't have recorded
anywhere and don't want to invent), I'll lay out the structural reasons I'd
expect published numbers on similar tasks to sometimes look higher than mine,
and why I don't think that makes my numbers "wrong".

First, **individual-level leakage is a very easy mistake to make and an easy
one not to notice**. A plain stratified k-fold on the CatMeows data, without
grouping by `cat_id`, would very likely report a higher cat macro-F1 than my
0.5223 ± 0.1338 - because the model could partly learn to recognize individual
cats' voices rather than the brushing/food/isolation context, and with only 21
cats, that's a real and exploitable signal. I built `StratifiedGroupKFold` by
`cat_id` specifically to avoid this (section 3), and I'd expect that to cost
some apparent performance compared to a non-grouped evaluation - by design, not
by accident.

Second, **dataset size and privacy**. Many published results on related
audio-emotion/context tasks use larger and/or private datasets, which both
gives more data per individual and allows for from-scratch architectures
(larger CNNs, full transformers) that would simply overfit on my 79-301
training samples (section 3's baseline already demonstrates this overfitting
risk at a much smaller parameter count).

Third, and more generally: a stricter, individual-aware, CV-based evaluation
with small datasets will tend to look more "modest" than a single train/test
split on a larger or non-grouped dataset, even when the underlying modeling
approach is similar or identical. **I can't prove that any specific published
result suffers from individual-level leakage or a less strict evaluation** -
that would require re-analyzing their splits, which I haven't done. But it is
a documented, common failure mode in this kind of small, individual-centric
audio dataset, and I think it's a more likely explanation for any gap than "my
models are worse". My numbers come from a protocol designed to not let a model
cheat by recognizing individuals - that's a deliberate trade-off, and I'd
rather report the honest, possibly-lower number than a higher one I can't fully
trust.

## 8. Production model and integration

Once the diagnosis in section 5 was clear - five levers plus a denoising test,
all plateauing at the same level - the next step was not more optimization, but
**picking one combination per animal, freezing it, and wrapping it in a stable
interface** for the LLM module and the app (`production_model_summary.md`,
`model_interface.md`).

### Backbone choice: MobileNetV2 + dense head, for both animals

The two candidates with CV numbers were MobileNetV2 + dense head (the
reference from steps 3-6) and AST + its best classifier (step 7):

| Animal | Approach | CV macro-F1 |
|---|---|---|
| Dog | MobileNetV2 + dense_head | 0.8244 ± 0.1114 |
| Dog | AST + dense_head | 0.8456 ± 0.0674 |
| Cat | MobileNetV2 + dense_head | 0.5223 ± 0.1338 |
| Cat | AST + logreg | 0.5064 ± 0.0859 |

Both deltas (dog +0.0212, cat -0.0159) are within noise - **neither backbone is
a clear winner on CV macro-F1 alone**. With a tie this close, I chose
**MobileNetV2 (frozen, ImageNet, `pooling="avg"`) + the small dense head
(`Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)`) for both
animals**, for three reasons: (1) the tie is within noise either way, so there
is no accuracy cost to picking the simpler option; (2) AST is explicitly
documented in `src/ast_transfer.py` as too heavy for CPU-only local inference
at a useful speed (it was run on a Colab GPU), whereas this project's
deployment target is CPU-only; (3) a single shared backbone means the app loads
one MobileNetV2 instance plus two small `.keras` heads, with one dependency set
(`tensorflow`/Keras) rather than also needing `torch`+`transformers` for AST.

This is exactly the "first/reference" combination from `mobilenet_transfer.py`
- nothing new was trained for this choice, only re-fit on more data, as
described next.

### Final training: train+val combined

`src/train_production.py` re-fits this combination on **train+val combined**
(96 clips for dog, 373 for cat) rather than train-only, since the held-out test
set has already done its job across steps 1-7 and isn't touched again. Because
`tl_common.train_head` needs a validation set for early stopping, a stratified
15% slice of the train+val pool (seed=42) is carved out purely as an
early-stopping signal - it never produces a reported metric. For cat this slice
is class-stratified but not `cat_id`-group-aware; I flagged this as a
deliberate, low-risk simplification, since it only decides *when to stop
training the final model*.

| Animal | Fit on | Early-stop val | Epochs run |
|---|---|---|---|
| Dog | 81 clips | 15 clips | 13 |
| Cat | 317 clips | 56 clips | 10 |

Outputs (`models/`, gitignored): `production_{dog,cat}_mobilenet_head.keras`
and `production_{dog,cat}_meta.json` (class lists, fixed duration, log-mel
normalization mean/std, image size, default threshold, seed - everything
`predict.py` needs besides the backbone's cached ImageNet weights).

### The `predict()` interface

`src/predict.py` exposes a single function:

```python
predict(audio_path, animal, threshold=None) -> dict
```

- `animal` is `"dog"` or `"cat"` (anything else raises `ValueError`).
- `threshold` defaults to `0.50` - if the model's top probability is below it,
  `label` becomes `"uncertain"` instead of a real class name, while
  `probabilities` still reports the full distribution.

Return format:

```python
{
    "animal": "cat",
    "label": "isolation",          # one of the class names, or "uncertain"
    "confidence": 0.71,            # max(probabilities.values())
    "probabilities": {"brushing": 0.12, "food": 0.17, "isolation": 0.71},
    "threshold": 0.5,
}
```

Two real examples from `production_model_summary.md`:

```python
>>> predict("data/raw/dog/bark/dog_1.wav", "dog")
{'animal': 'dog', 'label': 'bark', 'confidence': 0.9923,
 'probabilities': {'bark': 0.9923, 'growl': 0.0062, 'grunt': 0.0015},
 'threshold': 0.5}

>>> predict("data/raw/cat/brushing/B_ANI01_MC_FN_SIM01_101.wav", "cat")
{'animal': 'cat', 'label': 'uncertain', 'confidence': 0.4832,
 'probabilities': {'brushing': 0.4832, 'food': 0.2423, 'isolation': 0.2745},
 'threshold': 0.5}
```

The second example is a real demonstration of the threshold mechanism: the
model's top guess (`brushing`, 0.48) falls just below the 0.50 threshold, so
`predict()` honestly returns `"uncertain"` rather than guessing.

### Preprocessing identity between training and inference

`predict()` reuses, **unchanged**, the same functions used to build
`data/processed/`: `preprocess.fix_length` (centered pad/crop to 4s dog / 2s
cat), `preprocess.extract_logmel` (log-mel, `n_mels=64`, `power_to_db` with
`ref=1.0`), normalization with the train-set `(mean, std)` copied into
`production_<animal>_meta.json`, and `mobilenet_transfer.spectrograms_to_images`
(per-sample min-max -> 3 channels -> resize to 96x96 ->
`mobilenet_v2.preprocess_input`). Steps 1-3 and 5-6 are imported and called
unchanged - there is no parallel reimplementation that could drift out of sync.
As established in section 3, the normalization step is mathematically a no-op
for MobileNetV2, but it's kept for literal fidelity to `preprocess.py`.

### Guidance for the LLM/app integration

`model_interface.md` is the integration doc for students 3 and 4. The key
points: use `probabilities`, not just `label`, since the full distribution
carries information about how torn the model is between classes (useful for
hedging messages); treat `"uncertain"` as a signal to show a generic, friendly
fallback rather than picking the top class anyway; and, given section 6's
results, **for cat - especially around `food`/`brushing` - keep the generated
tone playful and a bit cautious/hedging rather than confidently assertive**,
since the threshold and probabilities exist precisely so the app doesn't have
to pretend more certainty than the model actually has. Dog messages can be
reasonably confident across all three classes.

## 9. Limits and future directions

**The data ceiling on `food` is the main open problem.** Section 5's
elimination argument now covers six independent tests (head tuning,
augmentation, classifier family, backbone, class-imbalance techniques, and a
denoised-audio comparison) — none moved `food` F1 meaningfully with the
current 92-clip, 21-cat dataset. The two things I'd expect to actually help
are (a) more `food` clips, ideally from more individual cats (more clips from
the *same* 21 cats would help less, since the group-aware split would still
only have ~15-17 cats' worth of `food` examples in any training fold), and (b)
a fundamentally different representation.

**Fine-tuning (unfreezing part of a backbone) is now the only "obvious" lever
not yet tried.** Everything in this report used **frozen** backbones; letting
at least the top layers of MobileNetV2 or AST adapt to log-mel
spectrograms/animal audio specifically — rather than natural photos or generic
AudioSet content — is the remaining candidate. For AST specifically,
**LoRA/PEFT-style lightweight fine-tuning** is worth investigating as a way to
adapt the transformer without the instability/overfitting risk of full
fine-tuning on ~440 clips - but I want to flag this as an **unproven avenue**,
not a guaranteed win; it's entirely possible it would also land within the
noise band that everything else has landed in.

**A larger dataset, or a stricter dog split, would tighten the error bars.**
The dog numbers carry one open caveat (no speaker ID, section 3) that a future
dataset with individual-dog IDs would resolve, the same way `cat_id` already
does for cat. More cats (or more clips per existing cat, though with
diminishing returns per the group-CV argument above) would shrink the large
`food`-class standard deviations (0.09-0.21) that make it hard to tell signal
from noise in the first place.

## 10. Conclusion

Over this project I built, and honestly evaluated, a frozen-transfer-learning
pipeline for two independent 3-class audio classifiers (dog: bark/growl/grunt;
cat: brushing/food/isolation), on top of two small, imbalanced, individual-
centric datasets (113 dog clips, no speaker ID; 440 cat clips from 21
individually-identified cats).

What I think this project demonstrates most strongly is the **evaluation
protocol itself**: a `StratifiedGroupKFold` by individual cat with an asserted
zero-violation guarantee, a documented and honestly-flagged limitation on the
dog side where that guarantee isn't possible, a proof that the normalization
step doesn't leak information across folds, and a discipline of touching the
test set exactly once per session and judging every change against its
standard deviation rather than its raw delta. That discipline is what let me
say, with some confidence, that six very different interventions — head
tuning, data augmentation, classifier family, an entirely different audio
backbone, class-imbalance techniques (focal loss and SMOTE), and a denoised-
audio comparison — all returned "no significant change" at the same level — a
conclusion I would not trust if any of those results had come from a single
noisy train/test split.

The **dog model is solid** (CV macro-F1 ~0.82-0.85, `grunt` essentially
perfect) and ready to use. The **cat model is honestly limited**: `isolation`
and `brushing` are usable, but `food` (CV F1 ~0.31-0.40, test F1 ~0.20-0.31) is
not, and I've shown - by elimination across six independent tests - that this
looks like a property of the 92-clip `food` class itself, not of any model
choice I made. Rather than hide that, the production interface (`predict()`,
section 8) surfaces the full probability distribution and an `"uncertain"`
label specifically so the app and LLM can be honest about it too.

What I learned, beyond the specific numbers: with very little data, the
**evaluation protocol matters more than the model architecture** - I spent far
more effort getting the CV/leakage story right than on any single model
change, and that effort is exactly what let me trust (and report honestly) a
result I didn't want to find (`food` not improving no matter what I tried). I
also learned to read standard deviations as a first-class part of every result,
not an afterthought - several "improvements" in this report (dog head tuning,
dog augmentation, AST for dog) would have looked like real wins if I'd only
looked at the mean.

---

