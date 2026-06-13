"""Exploration des données (EDA) du dataset audio brut.

Usage:
    python src/eda.py

Affiche dans la console les statistiques demandées (effectifs par classe,
déséquilibre, durées, sample rates, canaux, formats, fichiers problématiques)
et sauvegarde dans reports/ :
    - file_inventory.csv      : inventaire complet (1 ligne par fichier)
    - class_distribution.png  : bar chart des effectifs par classe
    - duration_histogram.png  : histogramme des durées
    - mel_spectrograms.png    : 2-3 exemples de mel-spectrogrammes par classe
"""

from __future__ import annotations

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from audio_inventory import build_inventory

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

N_SPECTROGRAMS_PER_CLASS = 3


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Scan de {DATA_DIR} ...")
    df = build_inventory(DATA_DIR)
    df.to_csv(REPORTS_DIR / "file_inventory.csv", index=False)
    print(f"Inventaire sauvegardé : {REPORTS_DIR / 'file_inventory.csv'}")

    print("\n=== 1. Nombre total de fichiers audio ===")
    print(len(df))

    print("\n=== 2. Classes et effectifs ===")
    counts = df["label"].value_counts()
    print(counts.to_string())
    print("\nRépartition train/test par classe :")
    print(df.groupby(["label", "split"]).size().unstack(fill_value=0).to_string())

    print("\n=== 3. Déséquilibre entre classes ===")
    ratio = counts.max() / counts.min()
    print(f"Classe majoritaire : {counts.idxmax()} ({counts.max()} exemples)")
    print(f"Classe minoritaire : {counts.idxmin()} ({counts.min()} exemples)")
    print(f"Ratio majoritaire / minoritaire : {ratio:.2f}")

    valid = df[~df["is_corrupted"]].copy()

    print("\n=== 4. Distribution des durées (secondes) ===")
    durations = valid["duration_s"]
    print(durations.describe().to_string())

    print("\n=== 5. Sample rates ===")
    print(valid["sample_rate"].value_counts().to_string())

    print("\n=== 6. Canaux (1=mono, 2=stéréo) ===")
    print(valid["channels"].value_counts().to_string())

    print("\n=== 7. Formats de fichiers ===")
    print(df["extension"].value_counts().to_string())

    print("\n=== 8. Fichiers potentiellement problématiques ===")
    report_corrupted(df)
    report_zero_duration(valid)
    report_silent(valid)
    report_duplicates(df)

    print("\n=== 9. Génération des figures ===")
    plot_class_distribution(counts)
    plot_duration_histogram(durations)
    plot_mel_spectrograms(valid)
    print(f"Figures sauvegardées dans {REPORTS_DIR}")


def report_corrupted(df) -> None:
    corrupted = df[df["is_corrupted"]]
    print(f"- Corrompus / illisibles : {len(corrupted)}")
    for _, row in corrupted.iterrows():
        print(f"    {row['path']} -> {row['error']}")


def report_zero_duration(valid) -> None:
    zero = valid[valid["duration_s"] <= 0]
    print(f"- Durée nulle : {len(zero)}")
    for path in zero["path"]:
        print(f"    {path}")


def report_silent(valid) -> None:
    silent = valid[valid["is_silent"]]
    print(f"- Silence total (amplitude max < seuil) : {len(silent)}")
    for path in silent["path"]:
        print(f"    {path}")


def report_duplicates(df) -> None:
    dupes = df[df.duplicated("md5", keep=False)].sort_values("md5")
    n_groups = dupes["md5"].nunique()
    print(f"- Doublons (contenu identique) : {len(dupes)} fichiers dans {n_groups} groupe(s)")
    for md5_value, group in dupes.groupby("md5"):
        paths = ", ".join(group["path"])
        print(f"    [{md5_value[:8]}] {paths}")


def plot_class_distribution(counts) -> None:
    plt.figure(figsize=(7, 5))
    counts.sort_values(ascending=False).plot(kind="bar", color="darkorange", edgecolor="black")
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.title("Effectifs par classe (train + test)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png", dpi=150)
    plt.close()


def plot_duration_histogram(durations) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(durations.dropna(), bins=30, color="steelblue", edgecolor="black")
    plt.xlabel("Durée (s)")
    plt.ylabel("Nombre de fichiers")
    plt.title("Distribution des durées des clips audio")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "duration_histogram.png", dpi=150)
    plt.close()


def plot_mel_spectrograms(valid, n_per_class: int = N_SPECTROGRAMS_PER_CLASS) -> None:
    labels = sorted(valid["label"].unique())
    fig, axes = plt.subplots(
        len(labels), n_per_class, figsize=(4 * n_per_class, 3 * len(labels)), squeeze=False
    )

    rng = np.random.default_rng(42)

    for i, label in enumerate(labels):
        subset = valid[valid["label"] == label]
        n_pick = min(n_per_class, len(subset))
        picked = subset.sample(n=n_pick, random_state=42) if n_pick else subset

        for j in range(n_per_class):
            ax = axes[i, j]
            if j < n_pick:
                row = picked.iloc[j]
                y, sr = librosa.load(DATA_DIR / row["path"], sr=None)
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
                ax.set_title(f"{label} — {row['path']}", fontsize=8)
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "mel_spectrograms.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
