"""Inventaire du nouveau dataset chat (CatMeows) — nouveau_data/cat/.

Usage:
    python src/cat_inventory.py

Parcourt nouveau_data/cat/{dataset,other_vocalizations,sequences}/*.wav,
extrait le contexte (B/F/I -> brushing/food/isolation) depuis le nom de
fichier selon la convention CatMeows, calcule les métadonnées audio (durée,
sample rate, canaux, etc.), affiche un résumé et sauvegarde
reports/cat_inventory.csv.

Inspection uniquement : ne déplace ni ne modifie aucun fichier du dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from audio_inventory import SILENCE_AMPLITUDE_THRESHOLD, file_md5

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CAT_DIR = REPO_ROOT / "nouveau_data" / "cat"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Convention CatMeows : préfixe de contexte au début du nom de fichier
CONTEXT_MAP = {"B": "brushing", "F": "food", "I": "isolation"}

SUBFOLDERS = ["dataset", "other_vocalizations", "sequences"]


def parse_filename(name: str) -> dict:
    """Extrait le contexte (B/F/I) et le flag "séquence" depuis le nom de fichier."""
    stem = Path(name).stem
    prefix = stem.split("_", 1)[0]
    return {
        "context_code": prefix,
        "context": CONTEXT_MAP.get(prefix, "unknown"),
        "is_sequence": "SEQ" in stem,
    }


def build_cat_inventory() -> pd.DataFrame:
    """Parcourt nouveau_data/cat/{dataset,other_vocalizations,sequences}/*.wav."""
    rows: list[dict] = []

    for folder_name in SUBFOLDERS:
        folder = CAT_DIR / folder_name
        if not folder.exists():
            continue

        for f in sorted(p for p in folder.iterdir() if p.is_file()):
            row = {
                "path": f"{folder_name}/{f.name}",
                "folder": folder_name,
                "extension": f.suffix.lower().lstrip("."),
                "md5": file_md5(f),
                **parse_filename(f.name),
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


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scan de {CAT_DIR} ...")
    df = build_cat_inventory()
    df.to_csv(REPORTS_DIR / "cat_inventory.csv", index=False)
    print(f"Inventaire sauvegardé : {REPORTS_DIR / 'cat_inventory.csv'}")

    print("\n=== A. Structure : fichiers par sous-dossier ===")
    print(df.groupby("folder").size().to_string())

    print("\n=== A. Contexte détecté via le préfixe du nom de fichier (tous dossiers) ===")
    print(df.groupby(["folder", "context_code"]).size().unstack(fill_value=0).to_string())

    print("\n=== B. Effectifs par contexte (dataset/ uniquement) ===")
    main_set = df[df["folder"] == "dataset"]
    counts = main_set["context"].value_counts()
    print(counts.to_string())
    ratio = counts.max() / counts.min()
    print(f"\nClasse majoritaire : {counts.idxmax()} ({counts.max()})")
    print(f"Classe minoritaire : {counts.idxmin()} ({counts.min()})")
    print(f"Ratio majoritaire / minoritaire : {ratio:.2f}")

    valid = df[~df["is_corrupted"]].copy()

    print("\n=== C. Sample rates ===")
    print(valid["sample_rate"].value_counts().to_string())

    print("\n=== C. Canaux (1=mono, 2=stéréo) ===")
    print(valid["channels"].value_counts().to_string())

    print("\n=== C. Formats de fichiers ===")
    print(df["extension"].value_counts().to_string())

    print("\n=== C. Durées (s) par sous-dossier ===")
    print(valid.groupby("folder")["duration_s"].describe().to_string())

    print("\n=== C. Durées (s) par contexte (dataset/ uniquement) ===")
    main_valid = valid[valid["folder"] == "dataset"]
    print(main_valid.groupby("context")["duration_s"].describe().to_string())

    print("\n=== C. Fichiers potentiellement problématiques ===")
    report_corrupted(df)
    report_zero_duration(valid)
    report_silent(valid)
    report_duplicates(df)


def report_corrupted(df: pd.DataFrame) -> None:
    corrupted = df[df["is_corrupted"]]
    print(f"- Corrompus / illisibles : {len(corrupted)}")
    for _, row in corrupted.iterrows():
        print(f"    {row['path']} -> {row['error']}")


def report_zero_duration(valid: pd.DataFrame) -> None:
    zero = valid[valid["duration_s"] <= 0]
    print(f"- Durée nulle : {len(zero)}")
    for path in zero["path"]:
        print(f"    {path}")


def report_silent(valid: pd.DataFrame) -> None:
    silent = valid[valid["is_silent"]]
    print(f"- Silence total (amplitude max < seuil) : {len(silent)}")
    for path in silent["path"]:
        print(f"    {path}")


def report_duplicates(df: pd.DataFrame) -> None:
    dupes = df[df.duplicated("md5", keep=False)].sort_values("md5")
    n_groups = dupes["md5"].nunique()
    print(f"- Doublons (contenu identique) : {len(dupes)} fichiers dans {n_groups} groupe(s)")
    for md5_value, group in dupes.groupby("md5"):
        paths = ", ".join(group["path"])
        print(f"    [{md5_value[:8]}] {paths}")


if __name__ == "__main__":
    main()
