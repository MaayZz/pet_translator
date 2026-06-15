# Production model: freezing a backbone + classifier per animal, with a stable interface

## Goal of this session

Up to now, every session was about *exploring*: trying backbones (YAMNet,
MobileNetV2, AST), tuning the head, augmenting data, comparing classifier
families. All of that converged on the same conclusion (see
`reports/ast_summary.md` and `reports/classifier_comparison_summary.md`): for
both animals, every architecturally different lever lands in the same range,
and for cat the `food` class stays weak (F1 ~0.30-0.37) no matter what I
change on the model side - a data limitation (92/440 clips), not a modeling
one.

This session is different: **no more optimization or fine-tuning**. The goal
is to pick ONE production combination per animal from what already exists,
freeze it, and wrap it in a clean, stable `predict()` interface that the LLM
module (student 3) and the app (student 4) can build on. Reliability and
integration clarity are what matter here, not squeezing out another point of
macro-F1.

## Backbone + classifier choice

### The candidates

Two frozen backbones exist in the repo, both evaluated with the same CV
protocol (`StratifiedKFold(k=5)` for dog, `StratifiedGroupKFold(k=4,
group=cat_id)` for cat, seed=42):

| Animal | Approach | CV macro-F1 |
|---|---|---|
| Dog | MobileNetV2 + dense_head | 0.8244 ± 0.1114 |
| Dog | AST + dense_head | **0.8456 ± 0.0674** |
| Cat | MobileNetV2 + dense_head | **0.5223 ± 0.1338** |
| Cat | AST + logreg | 0.5064 ± 0.0859 |

(Full tables in `reports/ast_summary.md`.)

### Why MobileNetV2 + dense_head, for BOTH animals

1. **Both ties are within noise.** AST is +0.0212 above MobileNetV2 for dog,
   and -0.0159 below it for cat - both deltas are smaller than the relevant
   standard deviation, so neither backbone is a clear winner on CV macro-F1
   alone. With a tie within noise, the brief asks me to prioritize
   **simplicity and stability**, and to prefer a single backbone for both
   animals if defensible.

2. **AST is explicitly documented as unsuitable for this deployment.** From
   `src/ast_transfer.py`'s own docstring: *"AST is too heavy for CPU-only
   local inference at a useful speed [...] it is written, reviewed and
   committed here, then executed on Colab."* This project's constraint for
   the production model is CPU-only, fast inference (no GPU assumed for the
   app). AST also pulls in `torch` + `transformers` as extra runtime
   dependencies for whoever integrates `predict()`. MobileNetV2 only needs
   `tensorflow`/Keras, which the rest of the project already uses.

3. **One backbone simplifies the app/LLM integration.** With a single frozen
   MobileNetV2 instance shared by both animals' heads, `predict.py` loads one
   backbone and two small `.keras` heads (a few hundred KB each). Student 4's
   app doesn't need to branch its model-loading logic per animal, and there's
   only one set of dependencies to install.

Given the tie is within noise either way, and MobileNetV2 is the strictly
easier, faster, and more dependency-light choice for a CPU-only deployed app,
**MobileNetV2 (frozen, ImageNet, `pooling="avg"`) + the small dense head
(`Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)`) is the
production combination for both dog and cat.** This is exactly the
"first/reference" approach from `mobilenet_transfer.py` - nothing new was
trained, only re-fit on more data (see below).

## How the production models were trained

`src/train_production.py` re-fits this same combination on **train+val
combined** (96 clips for dog, 373 for cat) instead of train-only, since
that's standard practice once a model is ready to ship - the held-out test
set has already done its job (CV + single-split evaluation in earlier
sessions) and isn't touched here.

`tl_common.train_head` needs a validation set for early stopping, so I carve
out a small stratified 15% slice of the train+val pool (seed=42) purely as an
early-stopping signal - it's not used for any reported metric. For cat this
slice is a plain class-stratified split, not `cat_id`-group-aware; I'm
flagging this as a deliberate simplification, safe here because this slice
only decides *when to stop training the final model*, it never produces a
number that gets reported or compared.

Results of this final fit:

| Animal | Fit on | Early-stop val | Epochs |
|---|---|---|---|
| Dog | 81 clips | 15 clips | 13 |
| Cat | 317 clips | 56 clips | 10 |

Outputs (gitignored, in `models/`):
- `production_dog_mobilenet_head.keras`, `production_cat_mobilenet_head.keras`
- `production_dog_meta.json`, `production_cat_meta.json` (classes, fixed
  duration, log-mel normalisation mean/std, default threshold, image size,
  seed - everything `predict.py` needs besides the two files above and the
  Keras-cached ImageNet weights).

## Preprocessing: inference vs training

`predict.py` runs each clip through exactly the steps `preprocess.py` and
`mobilenet_transfer.py` use to build `data/processed/<animal>/*_X.npy`:

1. `librosa.load(..., sr=16000, mono=True)`
2. `preprocess.fix_length` - centered pad/crop to the animal's fixed duration
   (4s dog / 2s cat)
3. `preprocess.extract_logmel` - log-mel spectrogram, `n_mels=64`,
   `power_to_db` with `ref=1.0` (absolute dB scale)
4. Normalize with the TRAIN-set `(mean, std)` from
   `data/processed/<animal>/norm_stats.json` (copied into
   `production_<animal>_meta.json`)
5. `mobilenet_transfer.spectrograms_to_images` - per-sample min-max -> 3
   channels -> resize to 96x96 -> `mobilenet_v2.preprocess_input`
6. Frozen MobileNetV2 -> 1280-dim features -> production dense head ->
   softmax probabilities

Steps 1-3 and 5-6 are imported and called unchanged from the existing
`preprocess`/`mobilenet_transfer` modules - there is no reimplementation to
drift out of sync. Step 4 is, as noted in `reports/cross_validation_summary.md`,
mathematically a no-op for MobileNetV2 (the per-sample min-max in step 5
cancels any global affine transform), but I kept it so the pipeline mirrors
`preprocess.py` literally, in case the normalisation step ever becomes
load-bearing for a future backbone.

## Example `predict()` output

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

The cat example is a nice real demonstration of the threshold mechanism: the
model's top guess (`brushing`, 0.48) is below the default 0.50 threshold, so
`predict()` honestly returns `"uncertain"` instead of guessing - exactly the
behaviour `reports/model_interface.md` asks the app to handle.

## Honest recap of limits (unchanged from earlier sessions)

- **Dog**: solid, cross-validated macro-F1 ~0.82-0.85, `grunt` essentially
  perfect.
- **Cat**: `isolation` reliable (F1 ~0.84), `brushing` reasonable, but `food`
  stays weak (F1 ~0.30-0.37, 92/440 clips). Four independent levers (head
  tuning, augmentation, five classifier families, and the AST audio backbone)
  all plateaued at the same ceiling - this is a demonstrated data limitation,
  not something this session claims to fix. The confidence threshold and the
  full `probabilities` dict exist so the app can stay honest about this
  (`reports/model_interface.md` has the messaging guidance for student 4).

## What the interface provides

See `reports/model_interface.md` for the full integration doc:
`predict(audio_path, animal) -> dict` with `label`/`confidence`/
`probabilities`/`threshold`, the class lists per animal, the `"uncertain"`
behaviour, and a recommendation to use `probabilities` (not just `label`) for
generating nuanced messages.

## Reproducibility

- `seed=42` everywhere (`tl_common.SEED`), CPU only.
- `python src/train_production.py --animal all` takes well under a minute on
  CPU (one MobileNetV2 forward pass over ~400 clips total + two small head
  fits).
- No `.wav` files, and no `.keras`/`.joblib` model files, are committed -
  `training/classification/models/` and `training/classification/data/` are
  both gitignored.

## Suggested commit message

```
Add production inference pipeline (predict.py) for dog/cat classification

- Add src/train_production.py: re-fits the frozen MobileNetV2 + dense head
  (the existing reference approach) on train+val combined for both animals,
  saving models/production_<animal>_mobilenet_head.keras and
  production_<animal>_meta.json (gitignored)
- Add src/predict.py: predict(audio_path, animal, threshold=0.5) -> dict with
  label/confidence/probabilities/threshold, including an "uncertain" label
  below the confidence threshold. Reuses preprocess.py/mobilenet_transfer.py
  unchanged so inference preprocessing matches training exactly.
- Add reports/model_interface.md: integration doc for the LLM/app teammates
  (call signature, return format, class lists, threshold behaviour, messaging
  guidance for cat's weaker classes)
- Add reports/production_model_summary.md: justification for picking
  MobileNetV2 (not AST) as the single production backbone for both animals -
  both are within noise of each other on CV macro-F1, and MobileNetV2 is the
  CPU-friendly, dependency-light choice (AST is documented as Colab/GPU-only)
```
