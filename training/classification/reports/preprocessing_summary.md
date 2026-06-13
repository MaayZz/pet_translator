# Synthèse — Pipeline de préparation des données (dog & cat)

Code : `src/preprocess.py` (`python src/preprocess.py --animal {dog,cat,all}`),
seed globale fixe = **42** (split + group search), tout tourne en CPU.

## 1. Paramètres communs

| Paramètre | Valeur | Justification |
|---|---|---|
| Sample rate | **16 000 Hz**, mono | Suffisant pour des vocalisations animales (énergie utile < 8 kHz), commun aux deux pipelines |
| Feature | Log-mel spectrogramme | Format standard en entrée d'un CNN type MobileNetV2 |
| `n_mels` | 64 | Taille raisonnable pour un CNN, sans excès vu la petite taille du dataset |
| `n_fft` | 1024 (= 64 ms à 16kHz) | Bonne résolution fréquentielle pour des vocalisations |
| `hop_length` | 512 (= 32 ms, overlap 50%) | Compromis standard résolution temporelle / volume de données |
| `power_to_db` | `ref=1.0` (défaut, **pas** `ref=np.max`) | Échelle dB **absolue et comparable** entre fichiers — condition nécessaire pour qu'une normalisation globale (mean/std) ait un sens. Avec `ref=np.max`, chaque spectrogramme serait recalé sur son propre pic, détruisant l'info de niveau absolu. |

## 2. Durée fixe + stratégie pad/crop

| Animal | Durée cible | Échantillons (16kHz) | Shape spectrogramme |
|---|---|---|---|
| Chien | ~4 s | 64 000 | **(64, 126)** |
| Chat | ~2 s | 32 000 | **(64, 63)** |

- **Si le clip est plus long** que la cible → on garde la portion **centrale** (crop centré), pas le début ni la fin.
- **Si le clip est plus court** → **zero-padding centré** (silence réparti moitié avant / moitié après).

**Pourquoi centré et pas "en fin" ?** Un pad/crop toujours du même côté introduirait un biais positionnel systématique : le réseau pourrait apprendre "le signal utile commence à l'indice 0" plutôt que le contenu spectral lui-même. Centrer évite ce raccourci, sans recourir à une détection d'activité vocale (VAD), hors scope ici.

## 3. Normalisation

Un couple `(mean, std)` **scalaire global**, calculé sur **toutes les valeurs du split TRAIN uniquement** (tous fichiers × tous mel-bins × tous frames), puis appliqué à train/val/test : `(x - mean) / std`.

| Animal | mean (train) | std (train) |
|---|---|---|
| Chien | -36.6671 | 20.8748 |
| Chat | -53.2483 | 19.2837 |

Sauvegardé dans `data/processed/<animal>/norm_stats.json`. **Aucune statistique n'est calculée sur val/test** — confirmé par le code : `mean`/`std` sont dérivés de `features["train"]` uniquement, puis réutilisés pour normaliser les 3 splits.

## 4. Split CHIEN — stratifié par classe (113 fichiers)

| Classe | train | val | test | Total |
|---|---|---|---|---|
| bark | 32 | 7 | 7 | 46 |
| growl | 23 | 5 | 5 | 33 |
| grunt | 24 | 5 | 5 | 34 |
| **Total** | **79 (69.9%)** | **17 (15.0%)** | **17 (15.0%)** | 113 |

Proportions de classes quasi identiques entre les 3 splits (ex: bark ≈ 40.5-41.2% partout) — le split stratifié fonctionne bien sur ces effectifs.

⚠️ **Limite documentée** : le dataset shivarao ne fournit **aucun identifiant d'individu** (quel chien a produit quel son). Le split est donc fait au niveau fichier, **sans garantie qu'un même chien ne se retrouve pas à la fois en train et en test**. Les métriques du modèle chien pourraient donc être **légèrement optimistes** (risque que le modèle "reconnaisse" partiellement un chien plutôt que de généraliser le type de vocalisation). Ce n'est pas un bug — c'est une contrainte du dataset source, à mentionner dans le rapport final.

## 5. Split CHAT — par individu (group split, 440 fichiers, 21 chats)

**Preuve anti-fuite** : pour chacun des 21 `cat_id`, le nombre de splits différents dans lesquels il apparaît est **1** (vérifié programmatiquement : `df.groupby("cat_id")["split"].nunique()` → max = 1, **0 violation**).

Répartition des 21 chats : **15 en train, 3 en val, 3 en test**.

| Classe | train | val | test | Total |
|---|---|---|---|---|
| brushing | 86 | 23 | 18 | 127 |
| food | 63 | 15 | 14 | 92 |
| isolation | 152 | 34 | 35 | 221 |
| **Total** | **301 (68.4%)** | **72 (16.4%)** | **67 (15.2%)** | 440 |

Comparaison des proportions de classes (global vs par split) :

| Classe | Global | train | val | test |
|---|---|---|---|---|
| brushing | 28.9% | 28.6% | 31.9% | 26.9% |
| food | 20.9% | 20.9% | 20.8% | 20.9% |
| isolation | 50.2% | 50.5% | 47.2% | 52.2% |

**Déséquilibre résultant** : très faible (écarts ≤ 3 points de pourcentage). Avec seulement 21 groupes pour 3 splits, ce résultat n'était pas garanti d'avance — il vient d'une recherche aléatoire (5000 tirages, seed=42) parmi les assignations possibles des 21 chats aux 3 splits, en choisissant celle qui respecte à la fois les proportions 70/15/15 (en nombre d'échantillons) ET la distribution de classes globale, **sans jamais diviser un même chat entre deux splits**. Si une assignation strictement meilleure avait nécessité de casser cette contrainte, on aurait gardé le déséquilibre — ce n'est heureusement pas le cas ici.

## 6. Vérifications

- **Shapes finales** (features normalisées, prêtes pour un modèle) :
  - `data/processed/dog/{train,val,test}_X.npy` → `(79,64,126)`, `(17,64,126)`, `(17,64,126)`
  - `data/processed/cat/{train,val,test}_X.npy` → `(301,64,63)`, `(72,64,63)`, `(67,64,63)`
  - Un échantillon = un tableau `(64, n_frames)` = (n_mels, n_frames temporels). Pour un MobileNetV2 (entrée image RGB), il faudra dupliquer/adapter ce canal unique en 3 canaux et redimensionner — étape laissée à la phase modélisation.
- **Normalisation fit train-only** : confirmé (section 3).
- **Anti-fuite chat** : confirmé (section 5), 0 violation sur 21 chats.

## 7. Sorties générées

```
data/processed/dog/   train_X.npy train_y.npy val_X.npy val_y.npy test_X.npy test_y.npy
                       norm_stats.json  label_encoding.json    (gitignoré)
data/processed/cat/   (idem)                                    (gitignoré)

reports/dog_split_manifest.csv   (path, label, split)            -> versionné
reports/cat_split_manifest.csv   (path, label, cat_id, split)    -> versionné
reports/split_class_counts.csv   (récap classe x split, dog+cat) -> versionné
reports/preprocessing_summary.md (ce fichier)                    -> versionné
```

`label_encoding.json` : `dog` = `{"bark":0,"growl":1,"grunt":2}`, `cat` = `{"brushing":0,"food":1,"isolation":2}` (ordre alphabétique des classes).

---

# Message de commit suggéré

```
Add data preprocessing pipeline (resampling, fixed-length, log-mel, split)

- Add src/preprocess.py: shared pipeline parameterized per animal
  (16kHz mono, log-mel n_mels=64/n_fft=1024/hop=512, centered pad/crop
  to 4s for dog / 2s for cat, train-only mean/std normalization)
- Dog: stratified file-level split (no individual ID available in
  shivarao); Cat: group split by cat_id (anti-leakage, 0 violations)
- Add reports/{dog,cat}_split_manifest.csv, split_class_counts.csv,
  preprocessing_summary.md
- data/processed/ stays gitignored (.npy not committed)
```
