# Transfer Learning — First Run (Dog & Cat)

## Why transfer learning, and why now

In the baseline session I already flagged the core problem: my datasets are tiny (79 dog / 301 cat training clips). I explicitly avoided a from-scratch CNN for the baseline because, with this few samples, it would very likely overfit and give unstable, hard-to-reproduce numbers.

Transfer learning is the standard answer to "I don't have enough data to learn good features from scratch": I take a model that already learned general-purpose features from a *much* larger dataset, freeze it, and only train a small classification head on top. The head has very few parameters, so it's much less prone to overfitting on 79-301 samples than a full CNN would be.

For this session I tried two pretrained backbones, both **frozen** (no fine-tuning):

- **Approach A — YAMNet**: pretrained on AudioSet (audio).
- **Approach B — MobileNetV2**: pretrained on ImageNet (images), repurposed on spectrograms.

The goal of this run was explicitly **not** to optimize anything: get both approaches working end-to-end with reasonable defaults, get a first score, and compare them to each other and to the baseline. Hyperparameter tuning is for a later session, on GPU.

## Shared setup

Both approaches reuse the exact same train/val/test split as the baseline (`reports/{animal}_split_manifest.csv` for YAMNet's raw-audio path, `data/processed/{animal}/{split}_{X,y}.npy` for MobileNetV2), so the comparison with the baseline is apples-to-apples.

For both approaches and both animals I used the same fixed defaults, chosen without any tuning:

- Head architecture: `Dense(64, relu) -> Dropout(0.3) -> Dense(n_classes, softmax)`
- Optimizer: Adam, learning rate `1e-3`
- Loss: sparse categorical cross-entropy
- `class_weight="balanced"` (computed from the train labels) to address the class imbalance
- Up to 50 epochs, with `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`
- `batch_size=8`
- `seed=42` everywhere (`tf.keras.utils.set_random_seed(42)`)

Everything ran on CPU (`tf.config.list_physical_devices('GPU')` returns `[]` on this machine), as required.

## Approach A — YAMNet (frozen audio embeddings)

YAMNet is a model pretrained on AudioSet (millions of labeled YouTube audio clips, 521 sound event classes — including several animal sounds). It takes mono float32 audio at 16 kHz in `[-1, 1]` and returns, among other things, a sequence of 1024-dim embeddings (one per ~0.96s analysis window).

For each clip I:
1. Reload the raw audio listed in the split manifest, resample to 16 kHz mono with `librosa.load`.
2. Apply the same centered pad/crop to a fixed duration as `preprocess.py` (4s for dog, 2s for cat) — so YAMNet sees "the same clips" as everything else. Both durations are comfortably above YAMNet's ~0.975s minimum input length, so no special handling was needed.
3. Run the clip through YAMNet (loaded once via `hub.load`, frozen) and **mean-pool** the embeddings over time to get one 1024-dim vector per clip.
4. Train the shared head on these 1024-dim embeddings.

### Results — Dog (test set, 17 samples)

- Embedding shapes: train (79, 1024), val (17, 1024), test (17, 1024)
- Trained for 18 epochs in 3.6s
- **Test accuracy = 0.8235, macro-F1 = 0.8244**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bark | 0.86 | 0.86 | 0.86 | 7 |
| growl | 0.67 | 0.80 | 0.73 | 5 |
| grunt | 1.00 | 0.80 | 0.89 | 5 |

Confusion matrix: `reports/yamnet_dog_confusion_matrix.png`. 14/17 correct: 1 bark misread as growl, 1 growl misread as bark, 1 grunt misread as growl.

### Results — Cat (test set, 67 samples)

- Embedding shapes: train (301, 1024), val (72, 1024), test (67, 1024)
- Trained for 12 epochs in 3.7s
- **Test accuracy = 0.4925, macro-F1 = 0.3454**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brushing | 0.31 | 0.22 | 0.26 | 18 |
| food | 0.14 | 0.07 | 0.10 | 14 |
| isolation | 0.60 | 0.80 | 0.68 | 35 |

Confusion matrix: `reports/yamnet_cat_confusion_matrix.png`. The model leans heavily towards "isolation": 9/18 brushing clips and 10/14 food clips get predicted as isolation. Only 33/67 correct overall.

## Approach B — MobileNetV2 (frozen, spectrograms as "images")

MobileNetV2 is pretrained on ImageNet — natural photos, nothing audio-related. To use it I have to convert my log-mel spectrograms into something that looks like an RGB image:

1. Start from the same normalized log-mel features used by the baseline (`data/processed/<animal>/{split}_X.npy`, shape `(n, 64, n_frames)`).
2. **Per-sample min-max scaling to [0, 1]** — each spectrogram is rescaled using only its own min/max, so there's no cross-sample or cross-split leakage.
3. Duplicate the single channel to 3 identical channels and `tf.image.resize` to `96x96` (the smallest input size MobileNetV2 accepts).
4. Scale to `[0, 255]` and apply `mobilenet_v2.preprocess_input` (maps to `[-1, 1]`), exactly as if this were a normal photo.
5. Run through MobileNetV2 (`include_top=False, pooling="avg", weights="imagenet"`, frozen) to get a 1280-dim feature vector per clip.
6. Train the shared head on these 1280-dim features.

I want to flag upfront that this "spectrogram-as-image" mapping is a simplification — there's no canonical correct way to turn a dB-scale time-frequency representation into RGB pixel intensities for a backbone trained on photos. I picked the simplest reasonable option (per-sample min-max + channel triplication + resize) and it's something I'd revisit during optimization (e.g. per-channel encodings of different feature representations, different resize strategies, etc.).

### Results — Dog (test set, 17 samples)

- Feature shapes: train (79, 1280), val (17, 1280), test (17, 1280)
- Trained for 7 epochs in 2.7s
- **Test accuracy = 0.7647, macro-F1 = 0.6887**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bark | 0.70 | 1.00 | 0.82 | 7 |
| growl | 1.00 | 0.20 | 0.33 | 5 |
| grunt | 0.83 | 1.00 | 0.91 | 5 |

Confusion matrix: `reports/mobilenet_dog_confusion_matrix.png`. 13/17 correct: bark and grunt are perfectly recalled, but growl collapses — only 1/5 growl clips are recognized as growl, the other 4 go to bark (3) or grunt (1).

### Results — Cat (test set, 67 samples)

- Feature shapes: train (301, 1280), val (72, 1280), test (67, 1280)
- Trained for 7 epochs in 2.1s
- **Test accuracy = 0.6418, macro-F1 = 0.5130**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| brushing | 0.80 | 0.44 | 0.57 | 18 |
| food | 0.33 | 0.14 | 0.20 | 14 |
| isolation | 0.65 | 0.94 | 0.77 | 35 |

Confusion matrix: `reports/mobilenet_cat_confusion_matrix.png`. 43/67 correct. Like YAMNet, this model is also biased towards "isolation" (7/18 brushing and 11/14 food clips end up there), but it still picks up brushing (8/18 correct) and isolation (33/35 correct) much better than YAMNet did.

## Comparison table — all four "models"

| | Dog Accuracy | Dog Macro-F1 | Cat Accuracy | Cat Macro-F1 |
|---|---|---|---|---|
| Floor (majority class) | 0.4118 | 0.1944 | 0.5224 | 0.2288 |
| Logistic regression (baseline) | 0.8235 | 0.8190 | 0.4328 | 0.3333 |
| YAMNet + dense head | 0.8235 | 0.8244 | 0.4925 | 0.3454 |
| MobileNetV2 + dense head | 0.7647 | 0.6887 | 0.6418 | 0.5130 |

### Dog

YAMNet and the logistic regression baseline are essentially **tied**: both get 14/17 test clips right (accuracy 0.8235), and YAMNet's macro-F1 (0.8244) is only marginally above the baseline's (0.8190) — well within the noise of a single misclassified file on a 17-sample test set. Both clearly beat the floor (0.4118 / 0.1944). MobileNetV2 (0.7647 / 0.6887) is behind both but still well above the floor.

**Honest caveat**: with only 17 test files, each misclassification moves accuracy by ~6 percentage points. So "82% vs 76%" (one extra correct file) is a real but small difference — I wouldn't over-interpret the exact ranking here yet.

### Cat

Here macro-F1 is the metric that matters (as discussed in the baseline report, "isolation" is over half the test set, so accuracy alone is misleading — the floor already gets 0.5224 accuracy by always predicting "isolation", but its macro-F1 is only 0.2288).

On macro-F1: **MobileNetV2 (0.5130) > YAMNet (0.3454) > logistic baseline (0.3333) > floor (0.2288)**. MobileNetV2 is the clear leader, and it's also the *only* approach of the four that beats the floor on **both** accuracy (0.6418 vs 0.5224) and macro-F1.

## Which approach looks most promising per animal (to confirm after optimization)

- **Dog**: YAMNet and the logistic baseline are tied for the lead, with MobileNetV2 behind. Given the tiny test set, I don't think this ranking is solid yet — it could easily flip with different hyperparameters or even a different random seed.
- **Cat**: MobileNetV2 currently looks like the most promising direction by a real margin on macro-F1. That said, I find it a bit counter-intuitive that the image-pretrained backbone (with a fairly crude spectrogram-to-image conversion) beats the audio-pretrained one here — it could be that YAMNet's 1024-dim embeddings need more than 12 epochs / a bigger head to make full use of the 301 cat training samples, or that the spectrogram-as-image features happen to capture useful time-frequency texture for this dataset. I want to dig into this during the optimization phase before drawing a firm conclusion.

In both cases, **none of these rankings are final** — they're first runs with untuned defaults, and the next session is specifically about exploring hyperparameters (and possibly architectural choices) for whichever approach(es) look worth pursuing.

## Training time — is CPU enough?

Everything above ran on CPU. Approximate wall-clock times (both animals together, including one-time model downloads):

- **YAMNet**: ~61s total. Most of this (~48s) is the one-time YAMNet download/load from TF Hub plus extracting embeddings for the 113 dog clips; extracting embeddings for the 440 cat clips and training both heads added ~13s. The actual head training itself is tiny: 18 epochs in 3.6s (dog), 12 epochs in 3.7s (cat).
- **MobileNetV2**: ~33s total. ~27s for the one-time ImageNet weight download (~9.4 MB) plus feature extraction for the 113 dog clips; cat features + both head trainings added ~6s. Head training: 7 epochs in 2.7s (dog), 7 epochs in 2.1s (cat).

So for this scale (a few hundred clips, frozen backbones, small heads), **CPU is more than sufficient** — both approaches together finish in under two minutes, well within the "a few minutes per model" budget. GPU will become relevant for the optimization phase if that involves a real hyperparameter search (many runs) or unfreezing/fine-tuning the backbones, neither of which I did here.

## What's NOT done yet

- **No hyperparameter optimization**: the head architecture, learning rate, batch size, number of epochs/patience, and (for MobileNetV2) the spectrogram-to-image conversion are all untuned defaults, chosen for being "reasonable", not "best".
- **No fine-tuning**: both backbones (YAMNet, MobileNetV2) stayed fully frozen — only the small heads were trained.
- This was a first run to confirm both pipelines work end-to-end and to get a first comparison point. The next session is about exploring hyperparameters (and possibly unfreezing parts of a backbone) for the most promising approach(es), likely on GPU.

## Artifacts produced this session

- `src/tl_common.py`, `src/yamnet_transfer.py`, `src/mobilenet_transfer.py`
- `reports/yamnet_dog_confusion_matrix.png`, `reports/yamnet_cat_confusion_matrix.png`
- `reports/mobilenet_dog_confusion_matrix.png`, `reports/mobilenet_cat_confusion_matrix.png`
- `models/yamnet_{dog,cat}_head.keras`, `models/mobilenet_{dog,cat}_head.keras` (gitignored, not versioned)

---

# Suggested commit message

```
Add first transfer-learning runs (YAMNet + MobileNetV2) for dog and cat

- Add src/tl_common.py: shared helpers (config, split loading, class
  weights, evaluation + confusion matrix plotting) reused by both
  transfer-learning approaches
- Add src/yamnet_transfer.py: Approach A - frozen YAMNet (AudioSet)
  embeddings + small dense head per animal
- Add src/mobilenet_transfer.py: Approach B - frozen MobileNetV2
  (ImageNet) on spectrograms converted to "images" + small dense head
  per animal
- Both approaches use fixed, untuned defaults (Adam 1e-3, dense head,
  class_weight="balanced", early stopping, seed=42), run on CPU in
  well under a minute total per approach
- Save reports/{yamnet,mobilenet}_{dog,cat}_confusion_matrix.png and
  reports/transfer_learning_summary.md comparing floor / logistic
  baseline / YAMNet / MobileNetV2 on accuracy + macro-F1 for both
  animals
- Add tensorflow, tensorflow-hub (and setuptools<81 pin) to
  requirements.txt
- No hyperparameter optimization or backbone fine-tuning in this
  session - that's the next step
```
