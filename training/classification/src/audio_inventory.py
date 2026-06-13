"""Construit un inventaire (DataFrame) du dataset audio brut.

Chaque ligne = un fichier audio, avec ses métadonnées (classe, split,
durée, sample rate, canaux, etc.) et des indicateurs de qualité
(corrompu, silence total, hash pour détection de doublons).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

# Seuil sous lequel l'amplitude max d'un clip est considérée comme "silence total"
SILENCE_AMPLITUDE_THRESHOLD = 1e-4


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """Sépare un nom de dossier en (classe, split).

    ex: "dog_bark_train" -> ("dog_bark", "train")
        "cat_test"       -> ("cat", "test")
    """
    for split in ("train", "test"):
        suffix = f"_{split}"
        if folder_name.endswith(suffix):
            return folder_name[: -len(suffix)], split
    return folder_name, "unknown"


def file_md5(path: Path, chunk_size: int = 65536) -> str:
    """Hash MD5 du contenu binaire du fichier (utilisé pour repérer les doublons)."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(data_dir: Path) -> pd.DataFrame:
    """Parcourt data_dir/<classe>_<split>/*.wav et retourne un DataFrame d'inventaire."""
    rows: list[dict] = []

    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label, split = parse_folder_name(folder.name)

        for f in sorted(p for p in folder.iterdir() if p.is_file()):
            row = {
                "path": str(f.relative_to(data_dir)),
                "label": label,
                "split": split,
                "folder": folder.name,
                "extension": f.suffix.lower().lstrip("."),
                "md5": file_md5(f),
            }

            try:
                info = sf.info(f)
                data, sr = sf.read(f, dtype="float32", always_2d=False)
                duration = info.frames / info.samplerate if info.samplerate else 0.0
                max_amplitude = float(np.max(np.abs(data))) if data.size else 0.0
                rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0

                row.update(
                    {
                        "sample_rate": info.samplerate,
                        "channels": info.channels,
                        "n_frames": info.frames,
                        "duration_s": duration,
                        "max_amplitude": max_amplitude,
                        "rms": rms,
                        "is_silent": max_amplitude < SILENCE_AMPLITUDE_THRESHOLD,
                        "is_corrupted": False,
                        "error": None,
                    }
                )
            except Exception as exc:  # fichier illisible / corrompu
                row.update(
                    {
                        "sample_rate": None,
                        "channels": None,
                        "n_frames": None,
                        "duration_s": None,
                        "max_amplitude": None,
                        "rms": None,
                        "is_silent": None,
                        "is_corrupted": True,
                        "error": str(exc),
                    }
                )

            rows.append(row)

    return pd.DataFrame(rows)
