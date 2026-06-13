"""Pipeline de préparation des données pour les modèles CHIEN et CHAT.

Usage:
    python src/preprocess.py --animal dog
    python src/preprocess.py --animal cat
    python src/preprocess.py --animal all   (par défaut)

Étapes (identiques pour les deux animaux, seule la config change) :
  1. Liste les fichiers de data/raw/<animal>/<classe>/*.wav
  2. Split train/val/test (stratégie différente par animal, voir plus bas)
  3. Pour chaque fichier : resampling 16 kHz mono, durée fixe (pad/crop centré),
     extraction d'un log-mel spectrogramme
  4. Normalisation (mean/std) calculée sur le TRAIN uniquement, appliquée
     ensuite à train/val/test
  5. Sauvegarde :
       - data/processed/<animal>/{train,val,test}_X.npy (features normalisées)
       - data/processed/<animal>/{train,val,test}_y.npy (labels entiers)
       - data/processed/<animal>/norm_stats.json (mean/std du train)
       - data/processed/<animal>/label_encoding.json
       - reports/<animal>_split_manifest.csv (path, label, split, [cat_id])
       - reports/split_class_counts.csv (récap effectifs classe x split)

CHOIX TECHNIQUES (CPU only, reproductible avec SEED=42)
--------------------------------------------------------
- Resampling 16 kHz mono : fréquence standard pour la voix/les vocalisations
  animales (couvre jusqu'à 8 kHz, largement suffisant), et commune aux deux
  modèles pour simplifier le pipeline partagé.

- Log-mel spectrogramme, n_mels=64, n_fft=1024, hop_length=512 :
    * n_fft=1024 à 16 kHz = fenêtres de 64 ms : bonne résolution fréquentielle
      pour des vocalisations (qui ont l'essentiel de leur énergie < 4 kHz).
    * hop_length=512 (50% overlap, 32 ms) : compromis standard
      résolution temporelle / taille des données.
    * n_mels=64 : taille raisonnable pour un input CNN (MobileNetV2), pas
      excessif vu la petite taille du dataset.
    * power_to_db SANS ref=np.max (donc ref=1.0 par défaut) : on veut une
      échelle de dB ABSOLUE et comparable entre fichiers, condition
      nécessaire pour qu'une normalisation globale (mean/std sur le train)
      ait du sens. Avec ref=np.max, chaque spectrogramme serait recalé sur
      son propre pic (0 dB = max du fichier), ce qui détruirait
      l'information de niveau absolu avant même la normalisation.

- Durée fixe + stratégie de pad/crop CENTRÉE :
    * CHIEN : 4s (64000 échantillons à 16kHz) -> spectrogramme (64, 126)
    * CHAT  : 2s (32000 échantillons à 16kHz) -> spectrogramme (64, 63)
    * Si le clip est plus long que la cible : on garde la portion CENTRALE
      (crop centré), pas le début ni la fin.
    * Si le clip est plus court : zero-padding CENTRÉ (moitié avant, moitié
      après), pas seulement en fin.
    Raison : un padding/crop toujours du même côté (ex: toujours en fin)
    introduit un biais positionnel systématique — le réseau pourrait
    apprendre "le signal utile commence toujours à l'indice 0" plutôt que
    d'apprendre le contenu spectral. Centrer évite ce raccourci, sans
    nécessiter de détection d'activité vocale (VAD).

- Normalisation : un seul couple (mean, std) GLOBAL (scalaire), calculé sur
  l'ensemble des valeurs (tous fichiers, tous mel-bins, tous frames) du split
  TRAIN uniquement, puis appliqué à train/val/test : (x - mean) / std.
  Sauvegardé dans norm_stats.json pour pouvoir reproduire le même
  prétraitement sur de nouvelles données (ex: inférence).

SPLIT TRAIN/VAL/TEST (70/15/15)
--------------------------------
- CHIEN (113 fichiers, classes bark/growl/grunt) : split STRATIFIÉ par classe
  au niveau fichier (train_test_split, stratify=label, random_state=42).
  LIMITE CONNUE : le dataset shivarao ne fournit pas d'identifiant d'individu
  (quel chien a produit quel son). On ne peut donc PAS garantir qu'un même
  chien n'apparaisse pas à la fois en train et en test. Les métriques chien
  pourraient être légèrement optimistes (le modèle pourrait en partie
  "reconnaître" un chien plutôt que généraliser le type de vocalisation).
  C'est une limite documentée, pas un bug.

- CHAT (440 fichiers, 21 chats, classes brushing/food/isolation) : split PAR
  INDIVIDU (group split) — tous les clips d'un même cat_id vont dans le MÊME
  split. On cherche, parmi plusieurs assignations aléatoires des 21 chats aux
  3 splits (random_state=42), celle qui s'approche le mieux à la fois des
  proportions 70/15/15 ET de la distribution de classes globale, SANS jamais
  violer la contrainte de groupe. Le déséquilibre résiduel est rapporté
  honnêtement (voir reports/preprocessing_summary.md).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"

SEED = 42
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 64
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test

CONFIGS = {
    "dog": {
        "classes": ["bark", "growl", "grunt"],
        "duration_s": 4.0,
        "group_split": False,
    },
    "cat": {
        "classes": ["brushing", "food", "isolation"],
        "duration_s": 2.0,
        "group_split": True,
    },
}


def list_files(animal: str) -> pd.DataFrame:
    """Liste data/raw/<animal>/<classe>/*.wav avec label (et cat_id pour le chat)."""
    cfg = CONFIGS[animal]
    rows: list[dict] = []
    for label in cfg["classes"]:
        folder = DATA_RAW / animal / label
        for f in sorted(folder.glob("*.wav")):
            row = {"path": f"{animal}/{label}/{f.name}", "label": label}
            if cfg["group_split"]:
                row["cat_id"] = f.stem.split("_")[1]
            rows.append(row)
    return pd.DataFrame(rows)


def dog_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split stratifié par classe, au niveau fichier (pas d'ID individu disponible)."""
    train_val, test = train_test_split(
        df, test_size=SPLIT_RATIOS[2], stratify=df["label"], random_state=SEED
    )
    val_size = SPLIT_RATIOS[1] / (SPLIT_RATIOS[0] + SPLIT_RATIOS[1])
    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["label"], random_state=SEED
    )

    df = df.copy()
    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


def cat_group_split(df: pd.DataFrame, n_trials: int = 5000) -> pd.DataFrame:
    """Split par individu (cat_id) : recherche aléatoire de la meilleure assignation
    des 21 chats aux 3 splits, en minimisant l'écart aux proportions 70/15/15 ET
    à la distribution de classes globale. Reproductible (random_state=SEED).
    """
    cat_sizes = df.groupby("cat_id").size()
    cat_ids = cat_sizes.index.to_numpy()
    total = len(df)
    labels = sorted(df["label"].unique())
    overall_props = df["label"].value_counts(normalize=True)
    cat_label_counts = df.groupby(["cat_id", "label"]).size().unstack(fill_value=0)

    targets_n = {
        "train": SPLIT_RATIOS[0] * total,
        "val": SPLIT_RATIOS[1] * total,
        "test": SPLIT_RATIOS[2] * total,
    }

    rng = np.random.default_rng(SEED)
    best_score = None
    best_assignment: dict[str, str] | None = None

    for _ in range(n_trials):
        order = rng.permutation(cat_ids)
        running_n = {"train": 0, "val": 0, "test": 0}
        running_label = {s: {label: 0 for label in labels} for s in running_n}
        assignment: dict[str, str] = {}

        for cid in order:
            size = int(cat_sizes[cid])
            remaining = {s: targets_n[s] - running_n[s] for s in running_n}
            split = max(remaining, key=remaining.get)
            assignment[cid] = split
            running_n[split] += size
            for label in labels:
                running_label[split][label] += int(cat_label_counts.loc[cid, label])

        size_score = sum(
            ((running_n[s] / total) - r) ** 2 for s, r in zip(["train", "val", "test"], SPLIT_RATIOS)
        )
        label_score = 0.0
        for s in ["train", "val", "test"]:
            n_s = running_n[s]
            if n_s == 0:
                continue
            for label in labels:
                label_score += (running_label[s][label] / n_s - overall_props[label]) ** 2

        score = size_score + label_score
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment

    df = df.copy()
    df["split"] = df["cat_id"].map(best_assignment)
    return df


def fix_length(y: np.ndarray, target_len: int) -> np.ndarray:
    """Pad/crop CENTRÉ à target_len échantillons (voir docstring du module)."""
    if len(y) >= target_len:
        start = (len(y) - target_len) // 2
        return y[start : start + target_len]
    pad_total = target_len - len(y)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(y, (pad_left, pad_right), mode="constant")


def extract_logmel(y: np.ndarray) -> np.ndarray:
    """Log-mel spectrogramme en dB absolus (ref=1.0, voir docstring du module)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    return librosa.power_to_db(mel).astype(np.float32)


def process_animal(animal: str) -> dict:
    cfg = CONFIGS[animal]
    df = list_files(animal)

    df = cat_group_split(df) if cfg["group_split"] else dog_split(df)

    manifest_path = REPORTS_DIR / f"{animal}_split_manifest.csv"
    df.to_csv(manifest_path, index=False)

    target_len = int(round(cfg["duration_s"] * SAMPLE_RATE))
    label_to_idx = {label: i for i, label in enumerate(cfg["classes"])}

    features: dict[str, np.ndarray] = {}
    labels_out: dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        feats = []
        for rel_path in sub["path"]:
            y, _ = librosa.load(DATA_RAW / rel_path, sr=SAMPLE_RATE, mono=True)
            y = fix_length(y, target_len)
            feats.append(extract_logmel(y))
        features[split] = np.stack(feats).astype(np.float32)
        labels_out[split] = sub["label"].map(label_to_idx).to_numpy(dtype=np.int64)

    # Normalisation : statistiques calculées UNIQUEMENT sur le train.
    mean = float(features["train"].mean())
    std = float(features["train"].std())

    out_dir = DATA_PROCESSED / animal
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        normalized = (features[split] - mean) / std
        np.save(out_dir / f"{split}_X.npy", normalized)
        np.save(out_dir / f"{split}_y.npy", labels_out[split])

    with open(out_dir / "norm_stats.json", "w") as fh:
        json.dump({"mean": mean, "std": std, "fitted_on": "train"}, fh, indent=2)
    with open(out_dir / "label_encoding.json", "w") as fh:
        json.dump(label_to_idx, fh, indent=2)

    return {
        "df": df,
        "shapes": {s: features[s].shape for s in features},
        "mean": mean,
        "std": std,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=["dog", "cat", "all"], default="all")
    args = parser.parse_args()
    animals = ["dog", "cat"] if args.animal == "all" else [args.animal]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    recap_rows = []
    for animal in animals:
        print(f"\n=== {animal.upper()} ===")
        result = process_animal(animal)
        df = result["df"]

        print("Distribution classe x split :")
        pivot = df.groupby(["label", "split"]).size().unstack(fill_value=0)
        print(pivot.to_string())

        for split, shape in result["shapes"].items():
            print(f"Shape {split}_X : {shape}")

        print(
            f"Normalisation (fit sur train uniquement) -> "
            f"mean={result['mean']:.4f}, std={result['std']:.4f}"
        )

        if CONFIGS[animal]["group_split"]:
            per_cat_splits = df.groupby("cat_id")["split"].nunique()
            n_violations = int((per_cat_splits > 1).sum())
            print(
                f"Vérif anti-fuite (cat_id dans 1 seul split) : "
                f"{n_violations} violation(s) sur {len(per_cat_splits)} chats"
            )

        counts = pivot.reset_index().melt(id_vars="label", var_name="split", value_name="n")
        counts.insert(0, "animal", animal)
        recap_rows.append(counts)

    recap = pd.concat(recap_rows, ignore_index=True)
    recap_path = REPORTS_DIR / "split_class_counts.csv"
    recap.to_csv(recap_path, index=False)
    print(f"\nRécap effectifs classe x split sauvegardé : {recap_path}")


if __name__ == "__main__":
    main()
