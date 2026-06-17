# Prompt for Amine's AI — Pet Translator Audio Preprocessing

Copy-paste this prompt as-is to your coding AI.

---

## Context

You are working on the **Pet Translator** frontend (React/Vite, `frontend-amine/`).
The audio classification is done by a frozen **MobileNetV2** model exported to
TF.js format. Two model files exist, one per animal:

- Dog model: `/model/dog/model.json` → 3 classes: `["bark", "growl", "grunt"]`
- Cat model: `/model/cat/model.json` → 3 classes: `["brushing", "food", "isolation"]`

**Critical**: the model does NOT accept raw audio. It expects a preprocessed
tensor of shape `[1, 96, 96, 3]` (float32, values in `[-1, 1]`) produced by a
specific log-mel spectrogram pipeline. If the preprocessing is wrong, the
predictions will be garbage even with the correct model weights.

The file to rewrite is `src/lib/modelLoader.js`. Its current implementation has
wrong parameters (128 mel bins, 128×128 resize, wrong dB formula, wrong classes,
1-channel input). Replace it entirely with the correct pipeline described below.

---

## Exact Preprocessing Pipeline (must match Python exactly)

Process the steps **in this exact order**.

### Step 1 — Decode and resample to 16 000 Hz mono

```js
const audioCtx = new AudioContext({ sampleRate: 16000 });
const arrayBuffer = await audioBlob.arrayBuffer();
const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
// Take channel 0 (mono). If stereo, average the two channels.
let samples = audioBuffer.getChannelData(0);
// Resample from audioBuffer.sampleRate to 16000 Hz using linear interpolation:
samples = resampleLinear(samples, audioBuffer.sampleRate, 16000);
```

Linear interpolation resampler:
```js
function resampleLinear(audio, fromRate, toRate) {
  if (fromRate === toRate) return audio;
  const ratio = toRate / fromRate;
  const out = new Float32Array(Math.round(audio.length * ratio));
  for (let i = 0; i < out.length; i++) {
    const pos = i / ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    out[i] = idx + 1 < audio.length
      ? audio[idx] * (1 - frac) + audio[idx + 1] * frac
      : audio[idx];
  }
  return out;
}
```

### Step 2 — Centered pad / crop to fixed duration

| Animal | Duration | Samples at 16 000 Hz |
|--------|----------|----------------------|
| dog    | 4.0 s    | **64 000**           |
| cat    | 2.0 s    | **32 000**           |

```js
function fixLength(audio, targetLen) {
  if (audio.length >= targetLen) {
    // center crop
    const start = Math.floor((audio.length - targetLen) / 2);
    return audio.slice(start, start + targetLen);
  }
  // center pad with zeros
  const out = new Float32Array(targetLen); // already zero-filled
  const padLeft = Math.floor((targetLen - audio.length) / 2);
  out.set(audio, padLeft);
  return out;
}
```

### Step 3 — Log-mel spectrogram

Parameters (must match exactly):

| Parameter   | Value  |
|-------------|--------|
| `n_fft`     | 1024   |
| `hop_length`| 512    |
| `n_mels`    | **64** |
| `sample_rate` | 16000 |
| `fmin`      | 0.0    |
| `fmax`      | 8000.0 (= sr / 2) |
| Window      | **Hann** |
| dB formula  | `10 * log10(power + 1e-10)` with `ref = 1.0` |

Output shape: `(64, n_frames)` where `n_frames ≈ 126` for dog (4 s), `≈ 63` for cat (2 s).

**Recommended**: use [`essentia.js`](https://mtg.github.io/essentia.js/) or a
mel-spectrogram library that matches librosa's Slaney/O'Shaughnessy mel filter
bank. If implementing from scratch, the mel filter bank must use the same formula
as `librosa.filters.mel(sr=16000, n_fft=1024, n_mels=64, fmin=0, fmax=8000,
htk=False)` (Slaney formula, which is librosa's default).

dB conversion (equivalent to `librosa.power_to_db(mel, ref=1.0)`):
```js
// mel is a Float32Array of power values (linear scale)
const logMel = mel.map(v => 10 * Math.log10(v + 1e-10));
```

> The global normalization step (`(x - mean) / std`) applied in the Python
> `predict()` is **mathematically a no-op** for the per-sample min-max that
> follows (step 4). Skip it in JS — you get identical results without it.

### Step 4 — Per-sample min-max rescaling → [0, 1]

```js
// spectrogram: Float32Array of length 64 * n_frames (row-major)
const flatSpec = new Float32Array(spectrogram); // copy
let minVal = Infinity, maxVal = -Infinity;
for (const v of flatSpec) {
  if (v < minVal) minVal = v;
  if (v > maxVal) maxVal = v;
}
const range = maxVal - minVal + 1e-8;
const scaled = flatSpec.map(v => (v - minVal) / range);
// scaled: values in [0, 1], same shape (64, n_frames)
```

### Step 5 — Duplicate to 3 channels

```js
// Build a [n_mels, n_frames, 3] array
const n_mels = 64, n_frames = scaled.length / n_mels;
const rgb = new Float32Array(n_mels * n_frames * 3);
for (let i = 0; i < n_mels * n_frames; i++) {
  rgb[i * 3]     = scaled[i];
  rgb[i * 3 + 1] = scaled[i];
  rgb[i * 3 + 2] = scaled[i];
}
```

### Step 6 — Resize to 96 × 96 (bilinear)

```js
const tf = await import('@tensorflow/tfjs');
// tensor shape: [n_mels, n_frames, 3]
let imgTensor = tf.tensor3d(rgb, [n_mels, n_frames, 3]);
// add batch dim → [1, n_mels, n_frames, 3], resize → [1, 96, 96, 3]
imgTensor = tf.image.resizeBilinear(imgTensor.expandDims(0), [96, 96]);
```

### Step 7 — Scale to [0, 255] then MobileNetV2 preprocess → [-1, 1]

```js
// scale to [0, 255]
imgTensor = imgTensor.mul(255.0);
// MobileNetV2 preprocess_input: x / 127.5 - 1  →  [-1, 1]
imgTensor = imgTensor.div(127.5).sub(1.0);
// Final shape: [1, 96, 96, 3], values in [-1, 1]
```

---

## Model Loading

Load both models at startup and cache them:

```js
const MODELS = {};
const CLASSES = {
  dog: ['bark', 'growl', 'grunt'],
  cat: ['brushing', 'food', 'isolation'],
};
const THRESHOLD = 0.50;

export async function loadModel() {
  const tf = await import('@tensorflow/tfjs');
  for (const animal of ['dog', 'cat']) {
    try {
      MODELS[animal] = await tf.loadLayersModel(`/model/${animal}/model.json`);
      console.log(`${animal} model loaded`);
    } catch (e) {
      console.warn(`${animal} model not found, will use mock`);
    }
  }
}
```

---

## Inference and Output Format

```js
export async function classifyAudio(audioBlob, animal) {
  // animal must be "dog" or "cat"
  if (!['dog', 'cat'].includes(animal)) throw new Error(`Unknown animal: ${animal}`);

  if (!MODELS[animal]) {
    // mock fallback
    const classes = CLASSES[animal];
    const idx = Math.floor(Math.random() * classes.length);
    return {
      animal,
      label: classes[idx],
      confidence: Math.round((0.75 + Math.random() * 0.2) * 100) / 100,
      probabilities: Object.fromEntries(classes.map((c, i) => [c, i === idx ? 0.9 : 0.05])),
      threshold: THRESHOLD,
    };
  }

  // --- run the preprocessing pipeline (steps 1-7) ---
  const inputTensor = await preprocessAudio(audioBlob, animal); // implement above

  // --- inference ---
  const tf = await import('@tensorflow/tfjs');
  const output = MODELS[animal].predict(inputTensor);
  const probs = await output.data();
  tf.dispose([inputTensor, output]);

  // --- format output ---
  const classes = CLASSES[animal];
  const topIdx = probs.indexOf(Math.max(...probs));
  const confidence = Math.round(probs[topIdx] * 10000) / 10000;
  const label = confidence >= THRESHOLD ? classes[topIdx] : 'uncertain';

  return {
    animal,
    label,
    confidence,
    probabilities: Object.fromEntries(classes.map((c, i) => [c, Math.round(probs[i] * 10000) / 10000])),
    threshold: THRESHOLD,
  };
}
```

Return value example (dog, bark clip):
```json
{
  "animal": "dog",
  "label": "bark",
  "confidence": 0.9923,
  "probabilities": { "bark": 0.9923, "growl": 0.0062, "grunt": 0.0015 },
  "threshold": 0.5
}
```

If `confidence < 0.50`, `label` is `"uncertain"` — the app should show a generic
fallback message, not guess. `probabilities` is always populated regardless.

---

## Mandatory Validation Test

After implementing the pipeline, run this check before considering it done:

1. Pick any `.wav` file from the dataset (e.g. `data/raw/dog/bark/dog_1.wav`).
2. Run the Python pipeline: `python -c "from src.predict import predict; import json; print(json.dumps(predict('data/raw/dog/bark/dog_1.wav', 'dog'), indent=2))"` from `training/classification/`.
3. In the browser, load the same file as a Blob and call `classifyAudio(blob, 'dog')`.
4. **Compare the `probabilities` values**: every class probability must match the
   Python output to within **±0.01**. If they don't match, the preprocessing has
   a bug.

The most common mistakes to check first:

| Bug | Symptom |
|-----|---------|
| Pad at end instead of centered | Probabilities off, especially for long clips |
| `Math.log` instead of `10 * log10` | Completely wrong scale |
| Resize to 128×128 instead of 96×96 | Shape mismatch or wrong features |
| Missing Hann window | FFT magnitudes slightly off |
| Missing `/ 127.5 - 1` step | Model sees [0,1] instead of [-1,1] |
| 1-channel instead of 3-channel input | Shape mismatch |
| Wrong `n_mels` (128 instead of 64) | Wrong spectrogram shape |

---

## Summary of True Parameter Values

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 000 Hz |
| Dog duration | 4.0 s → 64 000 samples |
| Cat duration | 2.0 s → 32 000 samples |
| n_fft | 1024 |
| hop_length | 512 |
| n_mels | **64** |
| fmin / fmax | 0 / 8000 Hz |
| Window | Hann |
| dB formula | `10 * log10(power + 1e-10)` |
| Global norm | skip (no-op for per-sample min-max) |
| Resize target | **96 × 96** |
| Channels | **3** (R=G=B=mel value) |
| MobileNetV2 preprocess | `pixel / 127.5 - 1` → [-1, 1] |
| Model input shape | [1, 96, 96, 3] |
| Confidence threshold | 0.50 |
| Dog classes (in order) | bark, growl, grunt |
| Cat classes (in order) | brushing, food, isolation |
