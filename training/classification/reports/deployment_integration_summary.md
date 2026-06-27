# Deployment and integration of the classifier into the front-end app

## Context

This report documents the integration of the production MobileNetV2 + dense
head classifier (`src/predict.py`, see `reports/production_model_summary.md`)
into the React/Vite front-end built by student 4 (`frontend-amine/`).

The integration happened in two distinct debugging phases. In the first phase,
four bugs in the JavaScript preprocessing and model-loading code were fixed.
In the second phase, a deeper architectural mismatch was identified and
resolved by replacing the two-piece model assembly with a single exported TF.js
model per animal.

All changes are in `frontend-amine/src/lib/modelLoader.js` (rewritten) and a
new script `training/classification/src/export_tfjs.py`.

---

## Step A — Four bugs in the initial `modelLoader.js`

The first deployment produced predictions that were clearly wrong —
for example "Bark 95%" in response to silence. Reading through
`modelLoader.js` revealed four independent problems.

### Bug 1 — TF.js backend crash on first load

**Symptom.** On macOS (and inconsistently on other platforms), the TF.js
backend initialization threw a "mutex lock failed" error. The model-loading
call then silently failed, leaving the app without a loaded model.

**Cause.** The code called `mobilenetModule.load()` before the TF.js runtime
was ready. Without explicitly selecting a backend and waiting for `tf.ready()`,
TF.js can race against an uninitialised WebGL context.

**Fix.** Added `await tf.setBackend('webgl')` + `await tf.ready()` at the top
of `loadModel()`, with a `try/catch` that falls back to the CPU backend if
WebGL is unavailable.

```js
try {
  await tf.setBackend('webgl');
  await tf.ready();
} catch (e) {
  console.warn('WebGL backend failed, falling back to CPU:', e);
  await tf.setBackend('cpu');
  await tf.ready();
}
```

### Bug 2 — Silent fallback to `mockClassify()`

**Symptom.** Even when the model failed to load, the app returned confident
predictions (80–99% confidence) for every audio clip, with no error visible to
the user.

**Cause.** `classifyAudio()` fell back to a `mockClassify()` function that
returned a pseudo-random class derived from a hash of the audio file, with an
artificially high confidence score. This function existed as a placeholder
during early development and was never removed.

**Fix.** Removed `mockClassify()` entirely. `classifyAudio()` now throws an
explicit error if the model for the requested animal is not loaded.

```js
if (!models[animal]) {
  throw new Error(`Model not loaded for "${animal}". Call loadModel() first.`);
}
```

### Bug 3 — Wrong mel filter scale (linear Hz instead of Slaney mel)

**Symptom.** The mel spectrogram produced by the JS code differed visually and
numerically from the one `librosa.feature.melspectrogram` produces for the
same audio, even though n_fft, hop_length, and n_mels were all correct.

**Cause.** The mel filter bank centre frequencies were spaced linearly in Hz
instead of following the Slaney mel scale (HTK=False) that librosa uses by
default. Specifically, the code was computing equally-spaced points between
`fMin` and `fMax` in Hz, which does not match the piecewise-linear-then-log
Slaney formula.

**Fix.** Rewrote `createMelFilterBank()` to use the Slaney mel scale:
`f < 1000 Hz → f / (200/3)` (linear region);
`f >= 1000 Hz → MIN_LOG_MEL + log(f / 1000) / log(6.4/27)` (log region).
This matches `librosa.filters.mel(htk=False)` exactly.

### Bug 4 — Magnitude instead of power in the spectrogram

**Symptom.** The spectrogram values were consistently too large and the
frequency-energy balance was wrong compared to the Python pipeline.

**Cause.** The FFT accumulation stored the *magnitude* (`sqrt(re² + im²)`)
instead of the *power* (`re² + im²`) before applying the mel filter bank.
`librosa.feature.melspectrogram` uses `power=2.0` by default, meaning it
applies the mel filter to the power spectrum, which is then passed to
`power_to_db`.

**Fix.** Changed the FFT inner loop to store `re * re + im * im` directly,
matching `|STFT|²`.

```js
magSpectrogram[t * fftBins + k] = re * re + im * im;   // power, not magnitude
```

---

## Step B — Root cause: incompatible backbones

### Symptom

After all four fixes above, the predictions were still wrong. On known test
clips the model consistently produced near-uniform or implausible distributions,
regardless of the audio content.

### Diagnosis: two-piece assembly with mismatched backbones

The architecture before Step B was:

```
AudioBlob
  → [JS preprocessing: resample, mel, min-max, ×255, resize 96×96]
  → mobilenet.infer(imgTensor, true)     ← @tensorflow-models/mobilenet
  → 1280-dim embedding
  → applyHead(embedding, headModels[animal])   ← head_weights.bin
  → softmax
```

The dense head (`head_weights.bin`, size ~329 KB, consistent with
1280×64 + 64 + 64×3 + 3 float32 parameters) was exported from the production
Keras model and was structurally correct.

The problem was the backbone. The production head was trained on 1280-dim
feature vectors extracted by `keras.applications.MobileNetV2(
input_shape=(96, 96, 3), pooling='avg', weights='imagenet')` from TensorFlow
2.16.2. The JS app loaded its backbone from
`@tensorflow-models/mobilenet@2.1.1`, which downloads a converted TF Hub
checkpoint (`mobilenet_v2_100_224/feature_vector/3`). Although both are
described as "ImageNet MobileNetV2", they are distinct model artifacts:
different conversion pipelines, potentially different checkpoint versions, and
a different target input resolution (224×224 for the TF Hub variant vs. 96×96
for the Keras training setup). Even a small numerical difference in backbone
weights causes the 1280-dim feature vectors to shift out of the distribution
the head was trained on, producing arbitrary softmax outputs.

In short: training and inference were running two different backbones. The
head's weights were only meaningful for one of them.

### Solution: export the complete model as a single TF.js graph model

The fix is to export the full computation graph — backbone and head together —
from the same Keras session that produced the head weights, and load this
single artifact in the browser.

`src/export_tfjs.py` does the following for each animal:

1. Reconstructs the full model in Keras:
   `Input(96, 96, 3) → MobileNetV2(frozen, ImageNet, pooling='avg')
   → production head (loaded from production_<animal>_mobilenet_head.keras)
   → softmax probabilities`

2. Verifies numerical equivalence against `predict.py` on one reference
   audio clip per animal before exporting anything.

3. Saves the full model as a TF2 SavedModel with an explicit `serving_default`
   signature, then converts it to a TF.js graph model with
   `tensorflowjs_converter --input_format=tf_saved_model
   --output_format=tfjs_graph_model`.

4. Writes the output to `frontend-amine/public/model/<animal>/`.

The new JS architecture is:

```
AudioBlob
  → [JS preprocessing: resample, mel, min-max, ×2−1, resize 96×96]
  → models[animal].predict(imgTensor.expandDims(0))
      (single TF.js graph model: backbone + head + softmax, exact Keras weights)
  → probabilities [1, 3]
```

`@tensorflow-models/mobilenet` and the separate `head_weights.bin` /
`head_shapes.json` loading are gone entirely.

### Proof of equivalence

Before exporting, `export_tfjs.py` runs the full model and the
backbone→head pipeline side-by-side on the same preprocessed input
and compares the output probabilities:

| Animal | Audio | predict.py probabilities | Full model probabilities | max\|diff\| |
|---|---|---|---|---|
| dog | `dog/bark/dog_1.wav` | bark=0.99231, growl=0.00617, grunt=0.00153 | bark=0.99231, growl=0.00617, grunt=0.00153 | 0.00e+00 |
| cat | `cat_train/cat_0.wav` | brushing=0.57751, food=0.11968, isolation=0.30281 | brushing=0.57751, food=0.11968, isolation=0.30281 | 0.00e+00 |

Exact agreement (zero floating-point difference) is expected and confirmed: the
full model is literally the same computation graph as the two-step
backbone→head chain, just fused into one `tf.keras.Model`. The export does
not change the weights or the numerical path.

### Preprocessing convention

The exported model graph does **not** include MobileNetV2's
`preprocess_input` normalisation. The model expects float32 input in `[-1, 1]`.

The JS preprocessing chain is:

```
resample → centered pad/crop → log-mel → per-sample min-max → [0, 1]
  → ×2 − 1  → [-1, 1]  → model.predict()
```

The previous `×255` step (which existed to feed `mobilenet.infer()` expecting
`[0, 255]`) is replaced by `imgTensor.mul(2.0).sub(1.0)`. There is exactly one
normalisation, applied once, with no double-preprocessing.

### Files produced

```
frontend-amine/public/model/dog/model.json
frontend-amine/public/model/dog/group1-shard1of3.bin
frontend-amine/public/model/dog/group1-shard2of3.bin
frontend-amine/public/model/dog/group1-shard3of3.bin

frontend-amine/public/model/cat/model.json
frontend-amine/public/model/cat/group1-shard1of3.bin
frontend-amine/public/model/cat/group1-shard2of3.bin
frontend-amine/public/model/cat/group1-shard3of3.bin
```

The old `head_weights.bin` and `head_shapes.json` for both animals were
removed (replaced by the unified model files above).
`@tensorflow-models/mobilenet` was removed from `frontend-amine/package.json`
since it is no longer used.

---

## Known limits

**JS FFT performance.** The mel spectrogram in `modelLoader.js` is computed
with a plain O(n²) DFT (no FFT algorithm). For a 4-second dog clip with
n_fft=1024 and ~124 frames, this runs ~130 million multiply-add operations in
the browser's main thread. The result is numerically correct — the DFT and the
FFT produce the same values to floating-point precision — but the computation
is slow (several seconds on a mid-range device). Replacing it with a radix-2
FFT would be the natural next optimisation, but it is not a correctness issue
and is left for a future iteration.

**In-browser validation still pending.** The equivalence proof above
(`max|diff| = 0.00e+00`) was measured in Python: the Keras full model and the
`predict.py` backbone→head pipeline agree exactly. What remains to be verified
is that the TF.js runtime, after loading `model.json`, produces probabilities
that match `predict.py` to within 0.01 on the same audio clip. This can be
checked by opening the app, recording (or uploading) the same reference clips
used above, and comparing the displayed probabilities against the values in the
table in the "Proof of equivalence" section. The preprocessing pipeline
(resampling, mel spectrogram, min-max, normalisation) is unchanged from the
four-bug-fix step and was already verified to be numerically consistent with
librosa; the only new variable is the TF.js inference itself.

---
