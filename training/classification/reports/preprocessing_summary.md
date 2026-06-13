# Data Preprocessing Pipeline — My Approach (dog & cat)

I wrote a single pipeline, `src/preprocess.py`, that I can run as `python src/preprocess.py --animal {dog,cat,all}`. The same code handles both animals — only the configuration (durations, classes, split strategy) changes. I fixed a global seed of **42** everywhere I needed randomness (the splits and the group-search for the cat split), and everything runs on CPU, since that's all I have available.

## 1. Shared parameters

| Parameter | Value | Why I chose it |
|---|---|---|
| Sample rate | **16,000 Hz**, mono | I think this is enough for animal vocalizations, since most of their useful energy sits below 8 kHz, and it lets me share one resampling step across both pipelines |
| Feature | Log-mel spectrogram | This is the standard input format for a CNN like MobileNetV2 |
| `n_mels` | 64 | A reasonable size for a CNN input, without being excessive given how small my dataset is |
| `n_fft` | 1024 (= 64 ms at 16 kHz) | Gives good frequency resolution for vocalizations |
| `hop_length` | 512 (= 32 ms, 50% overlap) | A standard trade-off between time resolution and data size |
| `power_to_db` | `ref=1.0` (default — **not** `ref=np.max`) | I wanted an **absolute, comparable** dB scale across files, which is necessary for a global mean/std normalization to make sense. With `ref=np.max`, each spectrogram would be rescaled relative to its own peak, which would destroy the absolute level information before I even get to normalize it. |

## 2. Fixed duration and my padding/cropping strategy

| Animal | Target duration | Samples (at 16kHz) | Spectrogram shape |
|---|---|---|---|
| Dog | ~4 s | 64,000 | **(64, 126)** |
| Cat | ~2 s | 32,000 | **(64, 63)** |

- **If a clip is longer** than the target, I keep the **center** portion (a centered crop) — not the beginning, not the end.
- **If a clip is shorter**, I apply **centered zero-padding** (silence split roughly evenly before and after the audio).

**Why centered, instead of always padding/cropping at the end?** I was worried that always padding or cropping on the same side would introduce a systematic positional bias — the network could end up learning "the useful signal always starts at index 0" instead of learning the actual spectral content. Centering avoids that shortcut, without requiring any voice-activity detection, which felt out of scope for this step.

## 3. Normalization

I compute a single **global scalar** `(mean, std)` pair, calculated over every value (across all files, mel bins, and time frames) in the **train split only**, and then apply `(x - mean) / std` to train, val, and test.

| Animal | mean (train) | std (train) |
|---|---|---|
| Dog | -36.6671 | 20.8748 |
| Cat | -53.2483 | 19.2837 |

I save these values to `data/processed/<animal>/norm_stats.json`. **I never compute any statistics on val or test** — in the code, `mean` and `std` are derived only from `features["train"]`, and that's the only place those two numbers come from before I reuse them to normalize all three splits.

## 4. Dog split — stratified by class (113 files)

| Class | train | val | test | Total |
|---|---|---|---|---|
| bark | 32 | 7 | 7 | 46 |
| growl | 23 | 5 | 5 | 33 |
| grunt | 24 | 5 | 5 | 34 |
| **Total** | **79 (69.9%)** | **17 (15.0%)** | **17 (15.0%)** | 113 |

The class proportions stay almost identical across the three splits (e.g. bark is ~40.5-41.2% everywhere), so the stratified split worked well at this size.

⚠️ **A limitation I want to be upfront about**: the shivarao dataset doesn't provide any individual identifier (i.e. I don't know which dog produced which sound). Because of that, my split is done at the file level, with **no guarantee that the same dog doesn't appear in both train and test**. This means the dog model's metrics could end up **slightly optimistic** — the model might partially "recognize" a specific dog rather than purely generalizing the vocalization type. I want to be clear that this isn't a bug in my code, it's a constraint of the source dataset, and I plan to mention it explicitly in my final report.

## 5. Cat split — by individual (group split, 440 files, 21 cats)

**Proof that there's no leakage**: for each of the 21 `cat_id` values, I checked how many different splits it appears in, and the answer is always **1** (`df.groupby("cat_id")["split"].nunique()` → maximum value is 1, so **0 violations**).

The 21 cats break down as: **15 in train, 3 in val, 3 in test**.

| Class | train | val | test | Total |
|---|---|---|---|---|
| brushing | 86 | 23 | 18 | 127 |
| food | 63 | 15 | 14 | 92 |
| isolation | 152 | 34 | 35 | 221 |
| **Total** | **301 (68.4%)** | **72 (16.4%)** | **67 (15.2%)** | 440 |

Class proportions, global vs. per split:

| Class | Global | train | val | test |
|---|---|---|---|---|
| brushing | 28.9% | 28.6% | 31.9% | 26.9% |
| food | 20.9% | 20.9% | 20.8% | 20.9% |
| isolation | 50.2% | 50.5% | 47.2% | 52.2% |

**Resulting imbalance**: very small — all differences are within about 3 percentage points of the global distribution. With only 21 groups to distribute across 3 splits, I couldn't take this result for granted going in. I got there by running a random search (5000 trials, seed=42) over possible ways to assign the 21 cats to the 3 splits, and keeping the assignment that best matched both the 70/15/15 sample-count target and the global class distribution — all while **never splitting a single cat across two sets**. If the best possible assignment had required breaking that group constraint, I would have kept the resulting imbalance and reported it honestly instead, but that turned out not to be necessary here.

## 6. Checks I ran

- **Final shapes** (normalized features, ready to feed into a model):
  - `data/processed/dog/{train,val,test}_X.npy` → `(79,64,126)`, `(17,64,126)`, `(17,64,126)`
  - `data/processed/cat/{train,val,test}_X.npy` → `(301,64,63)`, `(72,64,63)`, `(67,64,63)`
  - Each sample is a `(64, n_frames)` array (n_mels × time frames). For MobileNetV2, which expects an RGB image, I'll need to duplicate/adapt this single channel into 3 channels and resize it — I'm leaving that for the modeling phase.
- **Train-only normalization**: confirmed (section 3).
- **No cat leakage across splits**: confirmed (section 5), 0 violations across 21 cats.

## 7. What I generated

```
data/processed/dog/   train_X.npy train_y.npy val_X.npy val_y.npy test_X.npy test_y.npy
                       norm_stats.json  label_encoding.json    (gitignored)
data/processed/cat/   (same structure)                          (gitignored)

reports/dog_split_manifest.csv   (path, label, split)            -> versioned
reports/cat_split_manifest.csv   (path, label, cat_id, split)    -> versioned
reports/split_class_counts.csv   (recap of class x split counts, dog+cat) -> versioned
reports/preprocessing_summary.md (this file)                      -> versioned
```

`label_encoding.json`: for `dog` it's `{"bark":0,"growl":1,"grunt":2}`, and for `cat` it's `{"brushing":0,"food":1,"isolation":2}` — both in alphabetical order of the class names.

---

# Suggested commit message

```
Add data preprocessing pipeline (resampling, fixed-length, log-mel, split)

- Add src/preprocess.py: shared pipeline parameterized per animal
  (16kHz mono, log-mel n_mels=64/n_fft=1024/hop=512, centered pad/crop
  to 4s for dog / 2s for cat, train-only mean/std normalization)
- Dog: stratified file-level split (no individual ID available in
  shivarao); Cat: group split by cat_id (anti-leakage, 0 violations)
- Add reports/{dog,cat}_split_manifest.csv, split_class_counts.csv,
  preprocessing_summary.md
- data/processed/ stays gitignored (.npy not committed)
```
