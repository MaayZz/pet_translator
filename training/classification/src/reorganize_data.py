"""Réorganise les données brutes vers une structure propre par classe.

Usage:
    python src/reorganize_data.py

Construit, par COPIE (les sources d'origine ne sont jamais modifiées) :

  data/raw/dog/bark/    <- dog_bark_train/  + dog_bark_test/
  data/raw/dog/growl/   <- dog_growl_train/ + dog_growl_test/
  data/raw/dog/grunt/   <- dog_grunt_train/ + dog_grunt_test/
  data/raw/cat/brushing/  <- nouveau_data/cat/dataset/B_*.wav
  data/raw/cat/food/      <- nouveau_data/cat/dataset/F_*.wav
  data/raw/cat/isolation/ <- nouveau_data/cat/dataset/I_*.wav

Pour le chien, train et test sont fusionnés volontairement (le split
train/test sera refait de façon contrôlée à l'étape suivante). En cas de
collision de nom entre train et test, le fichier de test est renommé avec
le préfixe "test_" pour éviter d'écraser celui de train.

Pour le chat, le contexte (brushing/food/isolation) est lu depuis le préfixe
B/F/I du nom de fichier (convention CatMeows), et le nom d'origine est
préservé (il contient l'ID du chat, utile pour un futur split par individu).
Les fichiers avec un préfixe inattendu (ni B/F/I) ne sont pas copiés et sont
listés en sortie. Les sous-dossiers sequences/ et other_vocalizations/ sont
ignorés.
"""

from __future__ import annotations

import shutil
from pathlib import Path

CLASSIFICATION_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CLASSIFICATION_ROOT.parent.parent
DATA_RAW = CLASSIFICATION_ROOT / "data" / "raw"
CAT_SOURCE = REPO_ROOT / "nouveau_data" / "cat" / "dataset"

# (dossiers sources train/test, dossier cible)
DOG_GROUPS = [
    (["dog_bark_train", "dog_bark_test"], "dog/bark"),
    (["dog_growl_train", "dog_growl_test"], "dog/growl"),
    (["dog_grunt_train", "dog_grunt_test"], "dog/grunt"),
]

CAT_PREFIX_TO_LABEL = {"B": "brushing", "F": "food", "I": "isolation"}


def copy_dog_classes() -> None:
    for source_names, target_rel in DOG_GROUPS:
        target_dir = DATA_RAW / target_rel
        target_dir.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        for source_name in source_names:
            source_dir = DATA_RAW / source_name
            is_test = source_name.endswith("_test")

            for f in sorted(source_dir.glob("*.wav")):
                dest_name = f.name
                if dest_name.lower() in seen and is_test:
                    dest_name = f"test_{f.name}"
                    print(f"Collision détectée : {f} -> renommé en {dest_name}")
                seen.add(dest_name.lower())
                shutil.copy2(f, target_dir / dest_name)

        n_copied = len(list(target_dir.glob("*.wav")))
        print(f"{target_rel}: {n_copied} fichiers")


def copy_cat_classes() -> None:
    targets = {label: DATA_RAW / "cat" / label for label in CAT_PREFIX_TO_LABEL.values()}
    for target_dir in targets.values():
        target_dir.mkdir(parents=True, exist_ok=True)

    unexpected: list[Path] = []
    counts = {label: 0 for label in CAT_PREFIX_TO_LABEL.values()}

    for f in sorted(CAT_SOURCE.glob("*.wav")):
        prefix = f.stem.split("_", 1)[0]
        label = CAT_PREFIX_TO_LABEL.get(prefix)
        if label is None:
            unexpected.append(f)
            continue
        shutil.copy2(f, targets[label] / f.name)
        counts[label] += 1

    for label, count in counts.items():
        print(f"cat/{label}: {count} fichiers")

    if unexpected:
        print(f"\nFichiers avec préfixe inattendu (non copiés) : {len(unexpected)}")
        for f in unexpected:
            print(f"    {f}")


if __name__ == "__main__":
    print("=== Chien (fusion train + test) ===")
    copy_dog_classes()
    print("\n=== Chat (répartition par préfixe B/F/I) ===")
    copy_cat_classes()
