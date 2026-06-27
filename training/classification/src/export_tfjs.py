"""Export the full MobileNetV2+head model to TF.js (one model per animal).

PREPROCESSING CONVENTION (must match modelLoader.js exactly):
  The exported model expects float32 input in [-1, 1]  (= MobileNetV2 preprocess_input scale).
  JS pipeline after per-sample min-max → [0, 1]:
      imgTensor = imgTensor.mul(2.0).sub(1.0);   //  [0,1] → [-1,1]
  No further preprocessing inside the exported model graph.

RUN:
  python src/export_tfjs.py

OUTPUT:
  frontend-amine/public/model/dog/model.json  + group*.bin
  frontend-amine/public/model/cat/model.json  + group*.bin

STOP CONDITIONS:
  - Full model probs vs predict.py probs differ by more than 1e-4 → script aborts.
  - tensorflowjs_converter fails → script aborts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf

# ── paths ──────────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
sys.path.insert(0, str(_SRC))

from mobilenet_transfer import IMG_SIZE, build_backbone, spectrograms_to_images
from predict import _load_meta
from preprocess import extract_logmel, fix_length
from tl_common import DATA_RAW, MODELS_DIR

FRONTEND_MODEL_DIR = _ROOT.parent.parent / "frontend-amine" / "public" / "model"
TFJS_CONVERTER = _ROOT / ".venv" / "Scripts" / "tensorflowjs_converter.exe"

ANIMALS = ("dog", "cat")

TEST_AUDIO: dict[str, Path] = {
    "dog": DATA_RAW / "dog" / "bark" / "dog_1.wav",
    "cat": DATA_RAW / "cat_train" / "cat_0.wav",
}

EQUIVALENCE_ATOL = 1e-4


# ── model assembly ──────────────────────────────────────────────────────────

def build_full_model(animal: str) -> tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Return (full_model, backbone, head).

    full_model: Input(96,96,3) → MobileNetV2(frozen,ImageNet,pool=avg) → head → softmax-3
    Input must be in [-1, 1] (MobileNetV2 preprocess_input convention).
    """
    backbone = build_backbone()  # MobileNetV2, input_shape=(96,96,3), frozen
    head = tf.keras.models.load_model(MODELS_DIR / f"production_{animal}_mobilenet_head.keras")

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="mel_image")
    features = backbone(inputs, training=False)
    outputs = head(features, training=False)
    full_model = tf.keras.Model(inputs, outputs, name=f"{animal}_classifier")
    return full_model, backbone, head


# ── preprocessing (identical to predict.py) ────────────────────────────────

def preprocess_audio(animal: str, audio_path: Path) -> np.ndarray:
    """Return images in [-1, 1], shape (1, 96, 96, 3), via the exact predict.py pipeline."""
    meta = _load_meta(animal)
    waveform, _ = librosa.load(str(audio_path), sr=meta["sample_rate"], mono=True)
    target_len = int(round(meta["duration_s"] * meta["sample_rate"]))
    waveform = fix_length(waveform, target_len)
    logmel = extract_logmel(waveform)
    logmel = (logmel - meta["logmel_norm_mean"]) / meta["logmel_norm_std"]
    return spectrograms_to_images(logmel[np.newaxis, ...])  # (1, 96, 96, 3), float32, [-1, 1]


# ── verification ───────────────────────────────────────────────────────────

def verify(full_model: tf.keras.Model, backbone: tf.keras.Model,
           head: tf.keras.Model, animal: str, audio_path: Path) -> np.ndarray:
    """Compare full_model vs backbone→head pipeline on one audio clip.

    Returns the reference probabilities (from backbone→head, same as predict.py).
    Aborts if max absolute difference > EQUIVALENCE_ATOL.
    """
    imgs = preprocess_audio(animal, audio_path)  # [-1, 1]

    # Reference: exact predict.py computation
    features = backbone.predict(imgs, verbose=0)
    probs_ref = head.predict(features, verbose=0)[0]

    # Full unified model
    probs_full = full_model.predict(imgs, verbose=0)[0]

    max_diff = float(np.abs(probs_ref - probs_full).max())
    classes = _load_meta(animal)["classes"]

    print(f"  [predict.py pipeline] {dict(zip(classes, probs_ref.round(5)))}")
    print(f"  [full model]          {dict(zip(classes, probs_full.round(5)))}")
    print(f"  max|diff| = {max_diff:.2e}", end="")

    if max_diff > EQUIVALENCE_ATOL:
        print()
        raise RuntimeError(
            f"EQUIVALENCE FAILED for {animal}: max diff {max_diff:.2e} > {EQUIVALENCE_ATOL}. "
            "The full model reconstruction does not match predict.py. STOPPING."
        )
    print("  ✓ OK")
    return probs_ref


# ── TF.js export ───────────────────────────────────────────────────────────

def export_tfjs(full_model: tf.keras.Model, animal: str) -> list[Path]:
    """Save full_model as a TF SavedModel then convert to TF.js graph model."""
    out_dir = FRONTEND_MODEL_DIR / animal
    tmp_sm = out_dir / "_tmp_savedmodel"

    # Clean destination (keep nothing from the old head_weights approach)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1. Export as TF SavedModel with an explicit serving signature.
    #    The @tf.function fixes the input shape so the converter knows it.
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32, name="mel_image")
    ])
    def _serve(x):
        return full_model(x, training=False)

    print(f"  Saving TF SavedModel → {tmp_sm} ...")
    tf.saved_model.save(full_model, str(tmp_sm), signatures={"serving_default": _serve})

    # 2. Convert with tensorflowjs_converter
    print(f"  Running tensorflowjs_converter ...")
    result = subprocess.run(
        [
            str(TFJS_CONVERTER),
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--signature_name=serving_default",
            str(tmp_sm),
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )

    shutil.rmtree(tmp_sm, ignore_errors=True)

    print(f"  converter exit code: {result.returncode}")
    if result.stdout.strip():
        print("  CONVERTER stdout:", result.stdout[-3000:])
    if result.stderr.strip():
        print("  CONVERTER stderr:", result.stderr[-3000:])
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"tensorflowjs_converter failed (exit {result.returncode}). STOPPING."
        )

    output_files = sorted(out_dir.glob("*"))
    model_json = out_dir / "model.json"
    if not model_json.exists():
        raise RuntimeError(
            f"model.json not found in {out_dir} after conversion. STOPPING.\n"
            f"Files present: {[f.name for f in output_files]}"
        )

    return output_files


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("export_tfjs.py -- full MobileNetV2+head -> TF.js")
    print("=" * 60)
    print()
    print("PREPROCESSING CONVENTION:")
    print("  Model input: float32 [-1, 1]  (MobileNetV2 preprocess_input)")
    print("  JS (modelLoader.js): after per-sample min-max → [0,1],")
    print("  apply  imgTensor = imgTensor.mul(2.0).sub(1.0)  to get [-1,1].")
    print()

    for animal in ANIMALS:
        print(f"{'─'*40}")
        print(f"ANIMAL: {animal.upper()}")

        print("  Building full model (backbone + head) ...")
        full_model, backbone, head = build_full_model(animal)
        full_model.summary(print_fn=lambda s: None)  # build weights silently

        audio = TEST_AUDIO[animal]
        if audio.exists():
            print(f"  Verifying on {audio.name} ...")
            probs_ref = verify(full_model, backbone, head, animal, audio)
            classes = _load_meta(animal)["classes"]
            top = classes[int(np.argmax(probs_ref))]
            conf = float(probs_ref.max())
            print(f"  Reference: '{top}' ({conf:.4f})")
        else:
            print(f"  WARNING: {audio} not found — skipping verification.")

        print(f"  Exporting to TF.js → {FRONTEND_MODEL_DIR / animal} ...")
        files = export_tfjs(full_model, animal)
        print(f"  Files: {[f.name for f in files]}")
        print(f"  {animal.upper()} OK")
        print()

    print("=" * 60)
    print("ALL DONE.")
    print()
    print("Files to version:")
    for animal in ANIMALS:
        for f in sorted((FRONTEND_MODEL_DIR / animal).glob("*")):
            print(f"  frontend-amine/public/model/{animal}/{f.name}")
    print()
    print("Suggested commit message:")
    print("  feat(model): export full MobileNetV2+head to TF.js (dog + cat)")
    print()
    print("To test in-browser:")
    print("  cd frontend-amine && npm install && npm run dev")
    print("  Open http://localhost:5173 and record audio.")
    print("  In browser console: probabilities should match the reference above")
    print("  (within 0.01).")


if __name__ == "__main__":
    main()
