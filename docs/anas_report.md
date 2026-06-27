# Audio Classification for Pet Translator — Technical Report
**Anas ISARTI — Student 2 — ML01 Project**

---

## 1. Problem and Objective

My role in the Pet Translator project was to build the audio classification module: given a short audio clip and the animal it came from, output a label that the LLM module and the frontend app can use to generate a natural-language message.

The task is split into **two independent 3-class classifiers**, one per animal:

| Animal | Classes | What the label describes |
|---|---|---|
| Dog | `bark`, `growl`, `grunt` | Acoustic type of vocalization |
| Cat | `brushing`, `food`, `isolation` | Behavioral context of the meow |

This asymmetry is inherited from the datasets: the dog dataset labels sounds by acoustic type, while the cat dataset labels the recording situation. The two models therefore answer slightly different questions — the dog model says *what kind of sound*, the cat model says *what situation* — and the LLM should reflect this in how it phrases its output.

---

## 2. Data

### Dog — ShivaRao dataset (113 clips)

| Class | Clips | Train | Val | Test |
|---|---|---|---|---|
| bark | 46 | 32 | 7 | 7 |
| growl | 33 | 23 | 5 | 5 |
| grunt | 34 | 24 | 5 | 5 |
| **Total** | **113** | **79** | **17** | **17** |

Mild class imbalance (bark/growl ratio ≈ 1.4). **No individual dog ID** is provided in the dataset — this is the main open caveat on the dog evaluation (see §10).

### Cat — CatMeows dataset, Ludovico et al. 2021 \[[1]\] (440 clips, 21 cats)

| Class | Clips | Share |
|---|---|---|
| isolation | 221 | 50.2% |
| brushing | 127 | 28.9% |
| food | 92 | **20.9%** |

Real imbalance: `isolation` has 2.4× as many clips as `food`. The split is group-aware by individual cat ID — 15 cats in train, 3 in val, 3 in test — so no cat ever appears in two splits simultaneously. This is fundamental to a trustworthy evaluation (§3).

---

## 3. Methodology

### Why not a CNN trained from scratch

Before any model, I built a logistic regression baseline on flattened log-mel spectrograms to quantify the overfitting risk. With less regularization (`C ≥ 0.01`), training macro-F1 reaches **1.00** while validation stays flat for both animals. With 79–301 training samples and 4 032–8 064 features, a CNN from scratch would face the same memorization problem but with far more parameters. **Frozen transfer learning** is the natural answer: import features from a large pretrained model, freeze them, and train only a small head on top.

### Cross-validation and anti-leakage protocol

A 17-clip dog test set or 67-clip cat test set is so small that one misclassified file shifts the score by 6 points. I therefore used cross-validation as the primary evaluation signal, with the test set touched **exactly once per session** to report the final number.

**Cat — `StratifiedGroupKFold(n_splits=4, groups=cat_id)`.**
Cat vocalizations are known to be individual-specific \[[2]\]. Without grouping by individual cat, a model could partly learn to recognize that cat's voice rather than the behavioral context — a real leakage risk in a 21-cat dataset. I chose k=4 because it gives more independent estimates than k=3 while keeping val folds ≥ 88 clips. An assertion checked every session verifies **zero group violations** across all folds.

| Fold | n_train | n_val | train cats | val cats | violations |
|---|---|---|---|---|---|
| 0 | 322 | 118 | 16 | 5 | **0** |
| 1 | 319 | 121 | 15 | 6 | **0** |
| 2 | 352 | 88 | 17 | 4 | **0** |
| 3 | 327 | 113 | 15 | 6 | **0** |

**Dog — `StratifiedKFold(n_splits=5)`.**
No speaker ID exists in the ShivaRao dataset, so a group-aware split is not possible. I flag this openly: the dog CV numbers may be slightly optimistic if multiple clips from the same dog appear in both train and validation folds.

**Reference metric:** macro-F1 (unweighted average across classes, sensitive to the weakest class).

---

## 4. Chronology of Experiments

All sessions reuse the same folds and seed (42).

### Step 1 — Baseline (logistic regression)

| | Dog accuracy | Dog macro-F1 | Cat accuracy | Cat macro-F1 |
|---|---|---|---|---|
| Floor (dummy) | 0.412 | 0.194 | 0.522 | 0.229 |
| LogReg (C=0.0001) | **0.824** | **0.819** | 0.433 | 0.333 |

Cat `food` F1 = **0.00** on test — the baseline never predicted `food` correctly. This immediately flagged `food` as the central problem.

### Step 2 — Transfer learning, first run (YAMNet vs MobileNetV2, single split)

| | Dog macro-F1 | Cat macro-F1 |
|---|---|---|
| YAMNet + dense head | 0.824 | 0.345 |
| MobileNetV2 + dense head | 0.689 | **0.513** |

MobileNetV2 was the clear winner for cat. The single dog test split was hard for MobileNetV2 (17 clips — high variance). Both backbones beat the floor for both animals.

### Step 3 — Cross-validation (establishing the reference)

Same untuned configuration as step 2, now evaluated with full CV protocol.

| Animal | Approach | Accuracy (CV) | Macro-F1 (CV) |
|---|---|---|---|
| Dog | YAMNet | 0.797 ± 0.077 | 0.799 ± 0.074 |
| Dog | **MobileNetV2** | 0.823 ± 0.111 | **0.824 ± 0.111** |
| Cat | YAMNet | 0.406 ± 0.091 | 0.357 ± 0.073 |
| Cat | **MobileNetV2** | 0.560 ± 0.143 | **0.522 ± 0.134** |

From here on, **MobileNetV2 + dense head** (0.824 dog, 0.522 cat) becomes the reference that every subsequent session tries to beat.

### Step 4 — Head hyperparameter tuning

12 configurations per animal (dropout, L2, width, lr), scored by CV macro-F1.

| Animal | Metric | Default | Best | Delta | Verdict |
|---|---|---|---|---|---|
| Dog | CV macro-F1 | 0.824 ± 0.111 | 0.860 ± 0.084 | +0.036 | within noise |
| Cat | CV macro-F1 | 0.522 ± 0.134 | 0.537 ± 0.104 | +0.014 | within noise |
| Cat | `food` F1 | 0.365 ± 0.190 | 0.367 ± 0.150 | +0.003 | essentially unchanged |

No statistically meaningful improvement for either animal. Head-only tuning is exhausted.

### Step 5 — Data augmentation

Pitch shift, time stretch, Gaussian noise, SpecAugment masks. Applied inside CV folds, train indices only. Cat `food` was oversampled at 3×, dog uniform 2×. Group-leakage check remained at 0 violations.

| Animal | Metric | Without | With | Delta | Verdict |
|---|---|---|---|---|---|
| Dog | CV macro-F1 | 0.824 ± 0.111 | 0.843 ± 0.132 | +0.019 | within noise |
| Cat | CV macro-F1 | 0.522 ± 0.134 | 0.493 ± 0.155 | −0.029 | **wrong direction**, within noise |
| Cat | `food` F1 | 0.365 ± 0.190 | 0.287 ± 0.178 | −0.077 | wrong direction |

Augmentation made cat/food *worse* in CV. My best explanation: the frozen MobileNetV2 backbone, never exposed to audio, represents pitch/time/mask transforms in ways that decrease rather than increase class separability — especially for cat's short (63-frame) clips where a 12.5% time mask removes a significant fraction of content.

### Step 6 — Classifier family comparison

Same frozen MobileNetV2 features, same CV folds, five classifier families: dense head, SVM-linear, SVM-RBF, LDA, logistic regression.

**Dog (5-fold):**

| Classifier | Macro-F1 (CV) |
|---|---|
| dense_head | **0.824 ± 0.111** |
| lda | 0.807 ± 0.114 |
| logreg | 0.798 ± 0.122 |
| svm_linear | 0.774 ± 0.093 |

**Cat (4-fold):**

| Classifier | Macro-F1 (CV) | `food` F1 |
|---|---|---|
| dense_head | **0.522 ± 0.134** | 0.365 ± 0.190 |
| svm_rbf (C=1) | 0.485 ± 0.123 | 0.355 ± 0.134 |
| lda | 0.474 ± 0.052 | 0.327 ± 0.099 |
| logreg | 0.464 ± 0.057 | 0.309 ± 0.125 |

All classifiers land within one standard deviation of each other. The fact that neural net, linear, margin-based, and generative models all converge at the same level points at the **frozen MobileNetV2 features themselves** as the ceiling, not the classifier sitting on top.

### Step 7 — AST (Audio Spectrogram Transformer) backbone

Replaced MobileNetV2 features (ImageNet, 1280-dim) with an audio-native AST backbone (`MIT/ast-finetuned-audioset-10-10-0.4593`, 768-dim), same CV protocol.

| Animal | Approach | Macro-F1 (CV) | `food` F1 |
|---|---|---|---|
| Dog | AST + dense_head | 0.846 ± 0.067 | — |
| Dog | MobileNetV2 (ref) | 0.824 ± 0.111 | — |
| Cat | AST + logreg | 0.506 ± 0.086 | 0.307 ± 0.119 |
| Cat | MobileNetV2 (ref) | 0.522 ± 0.134 | 0.365 ± 0.190 |

Dog delta: +0.021 (within noise). Cat delta: −0.016 (within noise). Even an audio-native transformer pretrained on 527 AudioSet classes — including many animal sounds — doesn't move the needle for cat/food. **This is the fourth independent lever landing in the same band.**

### Step 8 — Denoising comparison

Ran the same CV protocol on a teammate's denoised version of all clips (same filenames, same splits, same seed). RAW run reproduced the step-3 reference exactly before reading the CLEAN numbers.

| Animal | Condition | Macro-F1 (CV) | Delta |
|---|---|---|---|
| Dog | RAW | 0.824 ± 0.100 | — |
| Dog | CLEAN | 0.835 ± 0.097 | +0.011 (within noise) |
| Cat | RAW | 0.522 ± 0.116 | — |
| Cat | CLEAN | 0.469 ± 0.081 | −0.053 (within noise) |

Audio noise is not the explanation for the cat/food ceiling.

### Step 9 — Focal loss and SMOTE

Four conditions on the same MobileNetV2 features and CV folds: baseline (cross-entropy + class_weight), focal loss (gamma=2.0), SMOTE, SMOTE+focal. SMOTE applied on train indices only; zero group violations maintained.

| Condition | Cat macro-F1 (CV) | `food` F1 | `food` Recall | `food` Precision |
|---|---|---|---|---|
| BASELINE | 0.522 ± 0.134 | 0.365 | 0.375 | 0.365 |
| FOCAL | 0.497 ± 0.116 | 0.317 | 0.299 | 0.338 |
| SMOTE | 0.472 ± 0.124 | 0.356 | **0.469** | 0.313 |
| SMOTE+FOCAL | 0.477 ± 0.159 | 0.378 | **0.502** | 0.318 |

SMOTE raises `food` recall from 0.38 to 0.47–0.50, but simultaneously drops precision from 0.37 to 0.31–0.32 — the model predicts `food` more often but also more incorrectly. F1 is unchanged. The features are simply not distinctive enough to make those additional `food` predictions reliable.

---

## 5. Central Diagnosis: The Data Ceiling

Five independent levers — head tuning, augmentation, classifier family, backbone, class-imbalance techniques — plus a denoising test: **six "no significant change" results**, all within noise of each other at every step.

This is an elimination argument. If the bottleneck were the head's capacity, tuning would have helped. If it were data diversity, augmentation would have helped food specifically — instead it went the wrong direction. If it were the classifier's inductive bias, one family would have pulled ahead. If it were the feature space, AST (trained on audio including animal sounds) would have shown a jump. If it were class imbalance per se, focal loss or SMOTE would have improved food F1 — instead they only redistributed errors between precision and recall. If it were audio quality, denoised audio would have helped.

The most consistent explanation: **the bottleneck is the dataset size**, specifically the 92-clip `food` class (roughly 4 clips per individual cat in any training fold). The model generates more `food` predictions when pushed (SMOTE) but cannot make them reliable because the MobileNetV2 features lack the discriminative power to separate this small, varied class from the others.

---

## 6. Comparison to the Literature

The original classification study on the CatMeows data, Ntalampiras et al. (2019) \[[3]\], reports **95.94% overall accuracy** with a DAG-HMM classifier using 10-fold cross-validation (100% for `food`, 95.24% for `brushing`, 92.59% for `isolation`). My best cat macro-F1 is 0.52.

The gap deserves an honest explanation rather than simply concluding that one approach is better. I can identify at least two structural reasons:

**1. Individual-level leakage risk.** With 440 clips from 21 cats (~21 clips/cat), a standard 10-fold CV splits approximately 2 cats per fold. Clips from the same cat will appear in both train and validation with high probability. Cat vocalizations are known to be individual-specific \[[2]\] — a model can partly learn to recognize *that cat's voice* rather than the behavioral context. This inflates apparent performance without the model having learned anything about `brushing` vs `food` vs `isolation` in a generalizable sense. My `StratifiedGroupKFold(group=cat_id)` is designed specifically to prevent this: no cat ever appears in both train and validation in any fold. By design, this costs some apparent performance — the number is lower and more trustworthy.

The problem of individual leakage in speaker-centric audio datasets is well-documented in speech emotion recognition, where "randomly splitting datasets can cause the training data to over-fit to one particular actor, leading to bias of the model — random splitting is considered a form of data leakage for this task" \[[4]\]. The same risk applies to cat vocalization classification, where individual identity is a stronger signal than behavioral context for a small 21-cat dataset.

**2. Handcrafted acoustic features vs frozen image CNN.** Ntalampiras et al. used HMMs with acoustic features specifically designed for speech and audio. My approach treats spectrograms as images and uses frozen ImageNet features — a deliberate choice given the dataset size (§3), but one that may not capture the temporal dynamics of cat meows as well as HMM-based models. A frozen AudioSet transformer (step 7) was still within noise of MobileNetV2, suggesting that audio-native features also reach the same ceiling on this dataset with my evaluation protocol.

I want to be clear: I cannot confirm that Ntalampiras et al.'s study suffers from individual-level leakage without re-analyzing their splits, which I have not done. But it is a documented and common failure mode for small, individual-centric audio datasets, and it is the most plausible structural explanation for the large gap between their numbers and mine under a group-aware protocol.

---

## 7. Final Results

| Approach | Dog macro-F1 (CV) | Cat macro-F1 (CV) | Cat `food` F1 (CV) |
|---|---|---|---|
| Floor (dummy) | 0.193 ± 0.006 | 0.221 ± 0.019 | 0.000 |
| Logistic regression (single-split) | 0.819 (test) | 0.333 (test) | 0.00 (test) |
| YAMNet + dense head | 0.799 ± 0.074 | 0.357 ± 0.073 | n/a |
| **MobileNetV2 + dense head (production)** | **0.824 ± 0.111** | **0.522 ± 0.134** | **0.365 ± 0.190** |
| AST + best classifier | 0.846 ± 0.067 | 0.506 ± 0.086 | 0.307 ± 0.119 |

**Dog**: solid across all three classes. CV macro-F1 0.82–0.85. On the final AST test evaluation (one-shot), `grunt` F1 = 1.00, `bark` 0.86, `growl` 0.80. The dog model is genuinely usable.

**Cat**: `isolation` reliable (test F1 ≈ 0.84), `brushing` reasonable, `food` persistently weak (CV F1 0.31–0.40, test F1 0.20–0.31). This is not a bug to fix later — it is a demonstrated property of the 92-clip `food` class under six independent experiments.

---

## 8. Production Model

### Choice: MobileNetV2 + dense head for both animals

Both MobileNetV2 and AST tie on CV macro-F1 (delta: +0.021 dog, −0.016 cat — both within noise). With a tie this close, I chose MobileNetV2 for three reasons: (1) it runs entirely on CPU (AST is documented as Colab/GPU-only); (2) it requires only `tensorflow`/Keras, not also `torch`+`transformers`; (3) one shared backbone for both animals simplifies the app's loading logic.

**Architecture:** `Input(96, 96, 3) → MobileNetV2(frozen, ImageNet, pooling="avg") → Dense(64, relu) → Dropout(0.3) → Dense(3, softmax)`

### Final training

Re-fit on train+val combined (96 dog clips, 373 cat clips), with a stratified 15% early-stopping slice carved out of the pool (not used for any reported metric).

| Animal | Fit on | Early-stop val | Epochs |
|---|---|---|---|
| Dog | 81 clips | 15 clips | 13 |
| Cat | 317 clips | 56 clips | 10 |

### Interface (`predict.py`)

```python
predict(audio_path, animal, threshold=0.5) -> dict
```

Return format:
```python
{
    "animal": "cat",
    "label": "isolation",       # class name or "uncertain" if max(prob) < threshold
    "confidence": 0.71,
    "probabilities": {"brushing": 0.12, "food": 0.17, "isolation": 0.71},
    "threshold": 0.5,
}
```

The `"uncertain"` label fires when the model's top probability is below `threshold` (default 0.50). Real example: a brushing clip returned `label="uncertain"` with `confidence=0.4832` — the model was genuinely torn between brushing (0.48) and the other classes and correctly refused to guess. The app should show a generic fallback in this case rather than picking the top class anyway.

**Recommendation to the app/LLM team:** use `probabilities`, not just `label`. A high-confidence isolation prediction (`conf=0.95`) warrants a direct message; a low-confidence brushing prediction with food close behind (`conf=0.45`, food `0.40`) warrants a hedged one. For cat, especially around `food`/`brushing`, the generated tone should be playful and cautious, not assertive.

---

## 9. Integration and Deployment Phase

### 9.1 Handoff documentation

Before the frontend integration started, I produced two documents for my teammate Amine (Student 4, frontend):

- **`reports/model_interface.md`** — the full integration spec: call signature of `predict()`, exact return format, class lists per animal, the `"uncertain"` threshold behavior, and messaging guidance for the weak cat classes.
- **`reports/amine_preprocessing_prompt.md`** — a copy-paste prompt for his AI assistant with the exact JS preprocessing pipeline (every parameter, step by step, including the Slaney mel scale formula, the power spectrum vs magnitude distinction, and the exact dB formula) and a validation test requiring that JS probabilities match Python to within ±0.01 on the same reference clip.

### 9.2 Four bugs in the initial `modelLoader.js`

After the first deployment, the app returned confident predictions on silence and random noise. Reading the JS code revealed four independent problems:

**Bug 1 — TF.js backend crash.** On macOS, `mobilenet.load()` was called before the TF.js runtime was initialized. This caused a silent `mutex lock failed` failure, leaving the app without a loaded model and falling back to the mock (see Bug 2).
*Fix:* added `await tf.setBackend('webgl'); await tf.ready()` at the top of `loadModel()`, with a `try/catch` that falls back to the CPU backend.

**Bug 2 — `mockClassify()` returning false predictions.** A placeholder function derived the predicted class from a simple hash of the raw audio bytes, with hardcoded confidence between 80–99%. It had no knowledge of audio content — silence and a real bark received identical predictions if their file hashes landed on the same index.
*Fix:* removed `mockClassify()` entirely. `classifyAudio()` now throws an explicit error if the model for that animal is not loaded.

**Bug 3 — Mel filter scale: linear Hz instead of Slaney mel.** The mel filter bank spaced center frequencies linearly in Hz rather than following the piecewise-linear-then-log Slaney formula that librosa uses by default (`htk=False`). This produced a mel spectrogram that looked different from the Python output even with correct `n_fft`, `hop_length`, and `n_mels`.
*Fix:* rewrote `createMelFilterBank()` to implement the Slaney formula: `f < 1000 Hz → f / (200/3)` (linear); `f ≥ 1000 Hz → MIN_LOG_MEL + log(f/1000) / log(6.4/27)` (logarithmic).

**Bug 4 — Magnitude instead of power in the FFT.** The inner FFT loop stored `sqrt(re² + im²)` (magnitude) instead of `re² + im²` (power) before applying the mel filter bank. librosa uses `power=2.0` by default — it filters the power spectrum, not the magnitude spectrum.
*Fix:* changed the accumulation to `re * re + im * im`.

### 9.3 Root cause: incompatible backbones

After all four fixes, predictions were still wrong. Diagnosis: the app was assembling a two-piece model at inference time:

```
AudioBlob
  → [JS preprocessing] → mel spectrogram → [0,1] × 255 → 96×96 tensor
  → @tensorflow-models/mobilenet@2.1.1  ← backbone (TF Hub checkpoint)
  → 1280-dim embedding
  → head_weights.bin                    ← dense head loaded separately
  → softmax
```

The dense head (`head_weights.bin`) was exported from the production Keras model — its weights were correct. The problem was the backbone. `@tensorflow-models/mobilenet@2.1.1` loads a TF Hub checkpoint (`mobilenet_v2_100_224/feature_vector/3`) — a different model artifact than `keras.applications.MobileNetV2(input_shape=(96, 96, 3), weights="imagenet")` that the production head was trained on. Although both are described as "ImageNet MobileNetV2", they come from different conversion pipelines and possibly different checkpoint versions, with a different target resolution (224×224 for the Hub variant vs 96×96 for training). The 1280-dim feature vectors produced by the Hub backbone are out of distribution for the head — which results in arbitrary softmax outputs.

**Solution:** export the complete model — backbone and head fused together — from the exact same Keras session that produced the head weights.

`src/export_tfjs.py` does this for each animal:
1. Reconstructs `Input(96,96,3) → MobileNetV2(frozen,ImageNet) → production head` in Keras.
2. Verifies numerical equivalence against `predict.py` on one reference audio clip before exporting anything.
3. Saves as a TF2 SavedModel with an explicit `serving_default` signature, then converts via `tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model`.
4. Writes the output to `frontend-amine/public/model/<animal>/`.

**Equivalence result:**

| Animal | Reference audio | predict.py top class | max\|diff\| |
|---|---|---|---|
| dog | `dog/bark/dog_1.wav` | bark (0.9923) | **0.00e+00** |
| cat | `cat_train/cat_0.wav` | brushing (0.5775) | **0.00e+00** |

Exact agreement (zero floating-point difference) is expected: the full model is the same computation as the two-step backbone→head chain, just fused into one `tf.keras.Model`.

`modelLoader.js` was rewritten to use `tf.loadGraphModel('/model/${animal}/model.json')` and call `models[animal].predict(imgTensor.expandDims(0))` directly. The preprocessing scaling changed from `×255` (for `mobilenet.infer()`) to `×2 − 1` (the unified model expects float32 in [−1, 1]). `@tensorflow-models/mobilenet` and the separate `head_weights.bin`/`head_shapes.json` loading are gone.

---

## 10. Limits and Future Work

- **The food ceiling is the main open problem.** Six independent tests show this is a data issue: 92 clips from 21 cats (~4.4 food clips/cat in training) is not enough. Adding more clips from *new* individual cats would help more than adding clips from the same 21 (because the group-aware CV would give the new cats to validation/test, providing genuinely novel evaluation points).

- **No dog speaker ID.** The ShivaRao dataset provides no individual-dog ID, so I cannot rule out that clips from the same dog appear in both train and validation. The dog CV numbers may be slightly optimistic as a result.

- **Out-of-distribution inputs.** The models were trained on clean pet vocalizations. Human speech, silence, and ambient sounds produce arbitrary softmax outputs. The 0.50 confidence threshold catches some of these cases (returning `"uncertain"`), but a proper silence/VAD gate at the frontend would be the right fix.

- **Fine-tuning (unfreezing) is the one obvious lever not yet tried.** Everything in this project used fully frozen backbones. Lightweight fine-tuning — LoRA or top-layer unfreezing — on MobileNetV2 or AST might adapt the features more closely to audio spectrograms. Given the data size, this is not a guaranteed win, but it is the remaining candidate worth trying.

---

## 11. Conclusion

Over this project I built, honestly evaluated, and deployed two frozen-transfer-learning audio classifiers — one for dogs (bark/growl/grunt) and one for cats (brushing/food/isolation) — on top of small, individually-centric datasets (113 dog clips, 440 cat clips from 21 identified cats).

The dog model is solid (CV macro-F1 ~0.82–0.85) and was successfully deployed to the browser. The cat model is mixed: `isolation` is reliable, `brushing` is reasonable, but `food` (CV F1 ~0.31–0.40) hit a ceiling that six independent experiments could not move. I demonstrated by elimination that this is a property of the 92-clip `food` class, not of any modeling choice I made, and the production interface explicitly surfaces this uncertainty through the full probability distribution and the `"uncertain"` label.

What I learned most from this project: with very little data, the evaluation protocol matters more than the model architecture. The `StratifiedGroupKFold` by individual cat, the zero-violations assertion, the discipline of touching the test set exactly once — these are what made it possible to say, with confidence, that six very different approaches all returned "no significant change" at the same level. A looser evaluation would have produced higher-looking numbers that I could not have trusted.

---

## Sources

\[1\] Ludovico, L.A., Ntalampiras, S., et al. "CatMeows: A Publicly-Available Dataset of Cat Vocalizations." *MultiMedia Modeling*, Springer, 2021. [link.springer.com/chapter/10.1007/978-3-030-67835-7_20](https://link.springer.com/chapter/10.1007/978-3-030-67835-7_20) — also available on Zenodo: [zenodo.org/records/4008297](https://zenodo.org/records/4008297)

\[2\] Ntalampiras, S., Ludovico, L.A., et al. "Automatic Classification of Cat Vocalizations Emitted in Different Contexts." *Animals*, MDPI, 2019. [mdpi.com/2076-2615/9/8/543](https://www.mdpi.com/2076-2615/9/8/543) — PMC version: [pmc.ncbi.nlm.nih.gov/articles/PMC6719916/](https://pmc.ncbi.nlm.nih.gov/articles/PMC6719916/)

\[3\] Same as \[2\]: Ntalampiras et al. 2019, cited here for the reported classification results (95.94% accuracy DAG-HMM, 10-fold CV).

\[4\] Chowdhury, A. et al. "A Case Study on the Independence of Speech Emotion Recognition in Bangla and English." *arXiv*, 2021. [arxiv.org/pdf/2111.10776](https://arxiv.org/pdf/2111.10776) — cited for the observation that random splitting without speaker grouping is a documented form of data leakage in speaker-centric audio datasets.

\[5\] TensorFlow. "Transfer learning with YAMNet for environmental sound classification." TF Core Tutorials. [tensorflow.org/tutorials/audio/transfer_learning_audio](https://www.tensorflow.org/tutorials/audio/transfer_learning_audio) — cited for the standard practice of freezing pretrained audio backbones on small datasets.
