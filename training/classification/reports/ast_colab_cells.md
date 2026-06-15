# Colab notebook - AST transfer learning (copy-paste cells)

I wrote `src/ast_transfer.py` to run on Colab (GPU T4), not on my local
machine - AST is too heavy for a useful CPU-only run there. Below is the
exact sequence of cells I use on Colab, in order. Each fenced code block is
one cell (Python unless marked `%%bash` / starting with `!`, which are shell
commands inside a Python cell, as usual in Colab).

## 1. Check the GPU

```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

!nvidia-smi
```

If `CUDA available` is `False`, go to Runtime > Change runtime type > select
a GPU (T4). `src/ast_transfer.py` falls back to CPU without crashing, but it
will be much slower.

## 2. Clone my repo

```python
!git clone https://github.com/MaayZz/pet_translator.git
%cd pet_translator/training/classification
```

If I need a specific branch (not the default one), I add:

```python
!git checkout <branch-name>
```

## 3. Install dependencies

`requirements.txt` uses `>=` constraints, so this will NOT downgrade the
torch/CUDA stack Colab already has - it only installs what is missing
(`transformers`, `librosa`, `soundfile`, `tensorflow-hub`, ...).

```python
!pip install -q -r requirements.txt
```

## 4. Provide the audio data

The `.wav` files are gitignored, so they are NOT in the clone - only
`reports/{dog,cat}_split_manifest.csv` (which IS versioned) came with the
repo. `src/ast_transfer.py` needs `data/raw/dog/<class>/*.wav` and
`data/raw/cat/<class>/*.wav` to exist, with the SAME filenames as in the
manifests (113 dog clips, 440 cat clips).

I pick ONE of the two options below.

### Option A - upload a zip

I zip my local `data/raw/` folder (so the zip's root contains a `dog/` folder
and a `cat/` folder, e.g. `dog/bark/dog_0.wav`, `cat/brushing/B_ANI01_..._101.wav`,
...), then run:

```python
from google.colab import files

uploaded = files.upload()  # pick my data_raw.zip in the dialog
```

```python
import zipfile

zip_name = next(iter(uploaded))  # name of the file I just uploaded
with zipfile.ZipFile(zip_name) as zf:
    zf.extractall("data/raw")
```

Expected result: `data/raw/dog/{bark,growl,grunt}/*.wav` and
`data/raw/cat/{brushing,food,isolation}/*.wav`.

### Option B - mount Google Drive

I upload `data/raw/dog/` and `data/raw/cat/` to my Google Drive beforehand
(e.g. under `MyDrive/pet_translator_data/`), then:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```python
import shutil
from pathlib import Path

DRIVE_DATA = Path("/content/drive/MyDrive/pet_translator_data")  # adjust to my actual path

Path("data/raw").mkdir(parents=True, exist_ok=True)
for animal in ["dog", "cat"]:
    shutil.copytree(DRIVE_DATA / animal, f"data/raw/{animal}", dirs_exist_ok=True)
```

## 5. Sanity-check the data is in place

```python
from pathlib import Path

for animal, n_expected in [("dog", 113), ("cat", 440)]:
    n = len(list(Path(f"data/raw/{animal}").rglob("*.wav")))
    print(f"{animal}: {n} .wav files found (expected {n_expected})")
```

## 6. Run the AST transfer-learning script

This extracts AST embeddings once per animal, runs the CV evaluation
(`reports/ast_cv_scores.csv`), the final test evaluation (confusion
matrices), and writes `reports/ast_summary.md`.

```python
!python src/ast_transfer.py --animal all
```

## 7. Download the results

```python
import shutil
from pathlib import Path
from google.colab import files

out_dir = Path("/content/ast_outputs")
out_dir.mkdir(exist_ok=True)

for f in Path("reports").glob("ast_*"):
    shutil.copy(f, out_dir / f.name)
for f in Path("models").glob("ast_*"):
    shutil.copy(f, out_dir / f.name)

shutil.make_archive("/content/ast_outputs", "zip", out_dir)
files.download("/content/ast_outputs.zip")
```

I then unzip `ast_outputs.zip` into `training/classification/reports/` and
`training/classification/models/` on my local machine before committing
(I commit everything myself - the assistant does not commit/push).
