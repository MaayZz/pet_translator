# Model interface for the app/LLM integration

This is the integration doc for the classification part of the project. If
you are working on the LLM module or the app/frontend, this is what you need
to know to call the model and interpret what it returns. The model code lives
in `training/classification/src/predict.py`; the reasoning behind which
backbone/classifier I picked is in `reports/production_model_summary.md`.

## How to call it

```python
from predict import predict

result = predict("path/to/clip.wav", animal="cat")
# or, to override the default confidence threshold:
result = predict("path/to/clip.wav", animal="cat", threshold=0.6)
```

- `audio_path`: path to an audio file (anything `librosa.load` can read - in
  practice this means `.wav`).
- `animal`: `"dog"` or `"cat"` - the app should ask the user which animal the
  clip is from and pass that here. Anything else raises `ValueError`.
- `threshold` (optional): confidence threshold below which the result becomes
  `"uncertain"` (see below). Defaults to `0.50` if not given.

Before calling `predict()` for the first time, run
`python src/train_production.py` once (CPU, a few seconds) to generate the
production model files under `models/` (gitignored, not committed).

## Exact return format

```python
{
    "animal": "cat",
    "label": "isolation",          # one of the class names below, or "uncertain"
    "confidence": 0.71,            # float in [0, 1] - the highest class probability
    "probabilities": {"brushing": 0.12, "food": 0.17, "isolation": 0.71},
    "threshold": 0.5,
}
```

- `probabilities` always contains ALL classes for that animal (see below),
  even when `label == "uncertain"`.
- `confidence` is just `max(probabilities.values())`.
- `threshold` echoes back the threshold that was actually used (so the app
  can display it if useful).

## Classes per animal

### Dog (`animal="dog"`)

| Class | Meaning |
|---|---|
| `bark` | barking |
| `growl` | growling |
| `grunt` | grunting |

### Cat (`animal="cat"`)

| Class | Meaning |
|---|---|
| `brushing` | sounds made while being brushed/petted |
| `food` | food-related vocalisations (e.g. asking for food) |
| `isolation` | distress/isolation calls |

## The "uncertain" label

If the model's top probability is below `threshold` (default `0.50`),
`label` is set to `"uncertain"` instead of one of the real class names. This
is meant to happen for clips the model genuinely isn't sure about - the app
should show a generic, friendly fallback message in this case ("I'm not sure
what your pet is trying to say right now!") rather than picking the top class
anyway.

## Recommendation: use `probabilities`, not just `label`

Please don't just branch on `label` and ignore the rest. The class
probabilities carry information that's useful for generating a more nuanced
message, especially for cat:

- If `label == "isolation"` with `confidence = 0.95`, the message can be
  fairly direct ("your cat seems to want company").
- If `label == "brushing"` with `confidence = 0.45` and `food` close behind at
  `0.40`, the model is genuinely torn between two classes - a message that
  hedges between both ("might be enjoying some attention, or asking for food")
  is more honest than committing to one.
- When `label == "uncertain"`, you can still glance at `probabilities` if you
  want a softer hint ("possibly related to food?") instead of a fully generic
  message - but treat it as a hint, not a result.

In short: `label`/`confidence` are a convenient summary, but `probabilities`
is the real signal - use it to make the generated message reflect how sure (or
unsure) the model actually is.

## Known limits - please reflect these in the generated messages

- **Dog**: the model is solid across all three classes (cross-validated
  macro-F1 ~0.82-0.85, with `grunt` essentially perfect). Messages for dog can
  be reasonably confident.
- **Cat**: `isolation` is reliable (F1 ~0.84) and `brushing` is reasonable,
  but `food` is weak (F1 ~0.30-0.37). This isn't a bug to be fixed later - I
  tried four different things to improve it (head tuning, data augmentation,
  five different classifiers, and a completely different audio backbone) and
  all of them plateaued at the same ceiling, which points at the "food" class
  itself being too small and varied (92/440 clips) rather than at the model.
  **For cat, especially around `food`/`brushing`, keep the generated tone
  playful and a bit cautious/hedging rather than confidently assertive** - the
  threshold + probabilities are there precisely so the app doesn't have to
  pretend more certainty than the model actually has.

## Errors

- `predict(audio_path, "bird")` -> `ValueError` (only `"dog"`/`"cat"` are
  valid).
- `predict("missing.wav", "dog")` -> `ValueError` with a message naming the
  file, if the audio can't be read.

Both are regular Python exceptions - catch `ValueError` around the call if you
want to show a friendly error in the UI instead of a stack trace.
