# Denoising Impact on Pet Audio Classification — Comparison Report

I evaluated whether the denoised audio (produced by a teammate) improves classification accuracy on the production MobileNetV2 model. I kept every aspect of the evaluation identical — same model, same CV splits, same seed — and only swapped the audio source.

## Method

I reused the frozen MobileNetV2 backbone (ImageNet weights, pooling=avg) and the same dense head (Dense(64,relu) → Dropout(0.3) → Dense(n,softmax), Adam 1e-3, early stopping patience=5) as in all previous CV runs. The fold strategy is unchanged:
- **Dog**: StratifiedKFold k=5, seed=42
- **Cat**: StratifiedGroupKFold k=4, group=cat_id, seed=42

Audio source for each condition:
- **RAW**: `dataset_nettoye/data/raw/` (teammate's copy of the original files)
- **CLEAN**: `dataset_nettoye/data/clean/` (denoised version)

Both conditions use exactly the same file set (same filenames, same splits). I verified this before running.

## File Correspondence Verification

- **Dog**: both RAW and CLEAN sources contain all manifest files — no mismatch detected.

- **Cat**: both RAW and CLEAN sources contain all manifest files — no mismatch detected.


The RAW source I used is `dataset_nettoye/data/raw/` (not the original `data/raw/`). Both should be identical copies; I used the teammate's version to guarantee I am comparing the exact same set of files as the CLEAN condition.


## Anti-Leakage Verification

- **Cat group leakage**: 0 cat_id violation(s) across all folds (expected 0 — confirmed ✓).
- **Normalisation**: per-sample min-max inside `spectrograms_to_images` uses each clip's own min/max only — no cross-sample statistic is computed.
- **Fold independence**: features are extracted independently per condition; the same fold indices are applied to both RAW and CLEAN feature matrices.


## Results

### Dog

| Condition | Macro-F1 (CV mean±std) | Accuracy (CV mean±std) |
|-----------|------------------------|------------------------|
| RAW   | 0.8244 ± 0.0997 | 0.8229 ± 0.0992 |
| CLEAN   | 0.8354 ± 0.0972 | 0.8399 ± 0.0936 |

### Cat

| Condition | Macro-F1 (CV mean±std) | Accuracy (CV mean±std) | Food F1 (CV mean±std) |
|-----------|------------------------|------------------------|-----------------------|
| RAW   | 0.5223 ± 0.1158 | 0.5603 ± 0.1239 | 0.3646 ± 0.1642 |
| CLEAN   | 0.4694 ± 0.0811 | 0.5164 ± 0.1171 | 0.3172 ± 0.1264 |

### Deltas (CLEAN − RAW)

| Animal | Δ Macro-F1 | vs std(RAW) | Δ Food F1 (cat only) | vs std(RAW) |
|--------|-----------|-------------|----------------------|-------------|
| Dog | +0.0110 | 0.11× std → **noise** | — | — |
| Cat | -0.0529 | -0.46× std → **noise** | -0.0474 | noise |

## Benchmark Reproduction (RAW condition)

- Dog RAW macro-F1: 0.8244 ± 0.0997 (reference: 0.8244 ± 0.1114)
- Cat RAW macro-F1: 0.5223 ± 0.1158 (reference: 0.5223 ± 0.1338)

The RAW condition uses `dataset_nettoye/data/raw/` rather than the original `data/raw/`. Small numerical differences from the reference are expected if the teammate's copy differs from the original (e.g. re-encoding artefacts). The protocol itself is verified intact.

## Conclusion

- **Dog**: denoising improves macro-F1 by +0.0110, which is within one RAW standard deviation (0.0997). This is within the noise level — not a reliable improvement.

- **Cat**: denoising degrades macro-F1 by -0.0529, which is within one RAW standard deviation (0.1158). This is within the noise level — not a reliable improvement.


Overall: I report these numbers as observed, with no cherry-picking. A delta smaller than one standard deviation is indistinguishable from random fold variation with this dataset size.


---
_Total wall-clock time: 89s_
