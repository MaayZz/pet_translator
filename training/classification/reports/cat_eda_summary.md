# CatMeows Dataset Inventory — My Findings

I inspected `nouveau_data/cat/` (483 `.wav` files in total) using a small script I wrote, `src/cat_inventory.py`. It scans the dataset and saves per-file metadata to `reports/cat_inventory.csv`. Based on the file count, the number of distinct cats, and the naming convention, I'm confident this is the published **CatMeows dataset** (Pirrone et al., 2020): 21 cats, 440 vocalizations, 3 behavioral contexts.

## A. Structure

I found three subfolders, and — unlike what I expected — the classes are **not** organized into per-class subfolders:

| Folder | Files | What I believe it contains |
|---|---|---|
| `dataset/` | 440 | The main set: one isolated vocalization per file |
| `other_vocalizations/` | 13 | Vocalizations that don't fit the standard pattern (e.g. purring) |
| `sequences/` | 30 | Sequences of several vocalizations concatenated together |

Instead, I found that the **context is encoded directly in the filename**, through a prefix, following this convention:

```
<Context>_<CatID><Session>_<Breed>_<Sex><Age>_<OwnerID><Session>_<Number>.wav
```

| Prefix | Context | Example from `dataset/` |
|---|---|---|
| `B` | Brushing | `B_ANI01_MC_FN_SIM01_101.wav` |
| `F` | Waiting for food | `F_BAC01_MC_MN_SIM01_101.wav` |
| `I` | Isolation in an unfamiliar environment | `I_ANI01_MC_FN_SIM01_101.wav` |

**My conclusion for part A**: I can recover the 3 classes (contexts) with 100% reliability just by reading the first character before the underscore in the filename. All 440 files in `dataset/` have a `B`, `F`, or `I` prefix — none fell into an "unknown" bucket. I implemented this as a simple `filename.split("_")[0]` lookup in `parse_filename()` inside `src/cat_inventory.py`.

I also found 21 distinct cat IDs in `dataset/` (e.g. `ANI01`, `BAC01`, `CAN01`...), which matches the official description of the dataset. I'm noting this now because it will matter later: when I build the train/val/test split, I'll want to make sure the same cat doesn't end up in two different sets.

## B. Counts per class (dataset/, n=440)

| Context | Count |
|---|---|
| isolation | 221 |
| brushing | 127 |
| food | 92 |
| **Total** | **440** |

The **majority/minority ratio is 2.40** (isolation vs. food). This is **more imbalanced** than the dog dataset (ratio 1.52), so I'll need to keep this in mind later — either with class weights during training or by augmenting the `food` class a bit more.

`other_vocalizations/` (13 files) and `sequences/` (30 files) follow the same prefix convention, but I deliberately **left them out** of the 440 above. I explain why in part D.

## C. Audio properties

- **Format**: 100% `.wav`
- **Channels**: 100% mono
- **Sample rate**: **uniformly 8000 Hz across all 483 files** (compared to 5 different sample rates in the dog dataset), so I won't need to resample anything within this dataset.
- **Quality**: 0 corrupted files, 0 zero-duration files, 0 fully silent files, 0 duplicates (checked via MD5 hash) — overall the dataset looks clean.

### Durations (seconds)

| Folder | n | mean | median | std | min | max |
|---|---|---|---|---|---|---|
| `dataset/` | 440 | 1.83 | 1.81 | 0.36 | 1.09 | 4.00 |
| `other_vocalizations/` | 13 | 1.49 | 1.41 | 0.33 | 1.16 | 2.01 |
| `sequences/` | 30 | 12.76 | 11.48 | 7.30 | 3.84 | 29.99 |

By context (`dataset/` only):

| Context | n | mean | median | min | max |
|---|---|---|---|---|---|
| brushing | 127 | 1.85 | 1.81 | 1.11 | 4.00 |
| food | 92 | 1.64 | 1.61 | 1.09 | 2.30 |
| isolation | 221 | 1.90 | 1.87 | 1.22 | 2.93 |

⚠️ **Something I need to flag honestly**: I originally expected the clips to be very short, around 0.3-0.4 seconds on average. **That's not what I measured** — the clips in `dataset/` average around **1.8 seconds** (median 1.81s, 75th percentile ~1.98s). My best guess is that the 0.3-0.4s figure refers to the duration of the *meow sound itself* as reported in the original paper, while the `.wav` files I have probably include a bit of quiet margin before and after the actual vocalization. I'd want to listen to a sample to confirm this if it becomes important, but for now, for pipeline purposes, **the real file duration is ~1.8-2s, not 0.3-0.4s**.

## D. My conclusions

**1. How many usable classes do I have, and with what counts?**
I can use 3 classes directly from `dataset/`: `isolation` (221), `brushing` (127), and `food` (92), for a total of 440.

**2. How are the classes identified?**
Through the **filename prefix** (B/F/I), not through subfolders. This convention is 100% reliable across `dataset/`.

**3. Are the durations very short, and what does that mean for choosing a fixed clip length later?**
Not as extreme as I expected (~1.8s rather than ~0.3-0.4s), but still noticeably shorter than the dog dataset (~5s). For the pipeline I'll build next, I think:
- A fixed input length around **2 seconds** would cover roughly 75% of the clips without losing information — I'd pad shorter clips and crop the one 4.0s outlier.
- Since the sample rate is already uniform at 8000 Hz, I won't need to resample within this dataset (although I may still need to match it to the dog pipeline's sample rate later — that's a separate decision since the two models are independent).
- This difference in clip length and sample rate compared to the dog dataset reinforces my decision to keep **two separate preprocessing pipelines** for dog and cat.

**4. Is the dataset ready to be reorganized into `data/raw/cat/<class>/`?**
- **Yes, for `dataset/`**: I can route all 440 files to `data/raw/cat/{brushing,food,isolation}/` unambiguously, based on the filename prefix. There are no corrupted, silent, or duplicate files to exclude.
- **I still need to make a separate decision about**:
  - `sequences/` (30 files, durations ranging from 3.8 to 30s): these are concatenations of several calls, so they're not directly comparable to the individual clips. I think they could be useful later for robustness testing or augmentation, but I wouldn't mix them as-is into `data/raw/cat/<class>/`.
  - `other_vocalizations/` (13 files): vocalizations outside the standard pattern, probably excluded from the original CatMeows benchmark for a reason. I'd rather keep them aside and documented, not included in the main training set by default.

I didn't reorganize any files during this session — this was inspection only, as planned.

---

# Suggested commit message

```
Add CatMeows dataset inventory (inspection only)

- Add src/cat_inventory.py to scan nouveau_data/cat/ and decode the
  B/F/I filename prefix (brushing/food/isolation contexts)
- Add reports/cat_inventory.csv (per-file metadata) and
  reports/cat_eda_summary.md (findings + recommendations)
- No files moved or reorganized; no training started
```
