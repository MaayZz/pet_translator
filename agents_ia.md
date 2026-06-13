# Agents IA — Pet Translation Device 🐾

## Contexte & Objectif

Projet ML01 — Sujet 4 : Fun & Experimental.  
Dispositif attaché au collier d'un animal (chat/chien) qui capture les sons, les classifie, et les "traduit" en texte via un LLM fine-tuné.

**Inspiration :** PettiChat (萌小译) — produit réel chinois basé sur Qwen, 94.6% précision revendiquée.

---

## Stack Finale

### Audio Pipeline (Python)

| Composant | Technologie |
|-----------|-------------|
| Feature extraction | `librosa` (MFCC, Mel-spectrogram) |
| Noise reduction | `noisereduce` |
| VAD | `webrtcvad` |
| Datasets | ESC-50, AudioSet (subset), custom recordings |

### Classification (Python → TensorFlow.js)

| Composant | Technologie |
|-----------|-------------|
| Framework | TensorFlow / Keras |
| Modèle | MobileNetV2 fine-tuné sur spectrogrammes |
| Catégories | Hunger, Pain, Play, Attention, Fear, Content |
| Format déploiement | `.tflite` → TensorFlow.js (browser) |
| Taille | ~3-5 MB (INT8 quantized) |
| Latence browser | ~20-50ms |

### LLM (Python → Fine-tuning LoRA)

| Composant | Technologie |
|-----------|-------------|
| Modèle | Llama 3.2 1B (ou 3B si GPU dispo) |
| Fine-tuning | LoRA via `unsloth` + `peft` + `transformers` |
| Dataset | Synthétique : 5000+ paires {intention + contexte + personnalité → phrase} |
| Format déploiement | GGUF (Q4_K_M) via `llama.cpp` |
| Taille | ~700 MB (1B Q4) / ~2 GB (3B Q4) |
| Latence | ~1-5s (sur serveur FastAPI) |

### Backend (Python)

| Composant | Technologie |
|-----------|-------------|
| Serveur API | FastAPI |
| LLM inference | `llama.cpp` Python bindings |
| Endpoints | POST `/translate`, GET `/history` |

### Frontend (Web)

| Composant | Technologie |
|-----------|-------------|
| Framework | React + Vite |
| ML browser | TensorFlow.js (MobileNetV2) |
| Audio capture | MediaRecorder API (navigateur) |
| UI | Chat style iMessage |
| Styling | CSS simple ou Tailwind |

---

## Pipeline Complet

```
Animal → [MEMS Microphone → Collar Device (concept)]
    ↓
Navigateur (Frontend)
    ├── MediaRecorder → capture audio (WAV)
    ├── TensorFlow.js → MobileNetV2 → catégorie + confiance
    └── fetch → POST /translate {category, confidence, history}
            ↓
Backend (FastAPI + llama.cpp)
    ├── Reçoit l'intention classifiée
    ├── Construit le prompt avec personnalité + historique
    ├── Llama 3.2 1B (LoRA fine-tuned) → génère phrase
    └── Retourne {text, emotion, timestamp}
            ↓
Frontend → Affiche dans UI Chat
    └── Stockage localStorage / SQLite (historique)
```

### Exemple de Flux

```
1. Chien aboie
2. Browser capture → TensorFlow.js → "jeu" (confidence 0.92)
3. POST /translate → {category: "play", pet: "dog", personality: "excited"}
4. Llama génère → "OUIIII ! Une balle ! J'adore ! Va la chercher !"
5. Chat UI affiche le message
```

---

## Structure du Dépôt

```
pet_translator/
├── frontend/                # React + Vite (interface web)
│   ├── public/
│   │   └── model/          # TFJS model files (MobileNetV2)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatUI.jsx       # Interface de chat
│   │   │   ├── AudioRecorder.jsx # Capture micro
│   │   │   └── PetSelector.jsx  # Chat/Chien + personnalité
│   │   ├── lib/
│   │   │   ├── modelLoader.js   # TFJS model loading
│   │   │   └── api.js           # FastAPI client
│   │   └── App.jsx
│   └── package.json
├── backend/                 # FastAPI (serveur LLM)
│   ├── models/             # GGUF files (Llama)
│   ├── audio/              # Pipeline audio Python
│   │   ├── preprocess.py   # librosa, denoise, VAD
│   │   └── features.py     # MFCC extraction
│   ├── llm/
│   │   ├── generator.py    # llama.cpp inference
│   │   └── prompt.py       # Prompt templates
│   ├── main.py             # FastAPI app
│   └── requirements.txt
├── training/               # Entraînement des modèles
│   ├── classification/
│   │   ├── train.py        # MobileNetV2 fine-tuning
│   │   ├── convert.py      # .h5 → .tflite → TFJS
│   │   └── data/           # Datasets audio
│   └── llm/
│       ├── train_lora.py   # unsloth LoRA fine-tuning
│       ├── dataset.py      # Génération dataset synthétique
│       ├── convert.py      # HF → GGUF
│       └── data/
│           └── synthetic_dataset.json
├── docs/
│   ├── brief_projet.md
│   ├── repartition_taches.md
│   └── tech_stack.md       # Stack détaillée
└── README.md
```

---

## Répartition Équipe (4 membres)

### Membre 1 — Traitement Audio & Pipeline
- `backend/audio/` : preprocessing, VAD, feature extraction
- Datasets : ESC-50, AudioSet, recordings
- `training/classification/data/` : préparation des données audio

### Membre 2 — Classification (MobileNetV2)
- `training/classification/train.py` : fine-tuning
- `training/classification/convert.py` : TFLite → TFJS
- `frontend/public/model/` : déploiement du modèle dans le browser
- `frontend/src/lib/modelLoader.js` : intégration TFJS

### Membre 3 — LLM (Llama + LoRA)
- `training/llm/dataset.py` : génération dataset synthétique
- `training/llm/train_lora.py` : LoRA fine-tuning avec unsloth
- `training/llm/convert.py` : conversion GGUF
- `backend/llm/` : inference serveur + prompt engineering
- Personnalités : chat hautain / chien excité / chat timide

### Membre 4 — Frontend Web & Intégration
- `frontend/` : React + Vite, UI Chat
- `frontend/src/components/` : AudioRecorder, ChatUI, PetSelector
- `backend/main.py` : FastAPI serveur
- Intégration complète du pipeline (audio → classification → LLM → UI)
- Démo vidéo + documentation

---

## Expériences Clés à Documenter

| # | Expérience | Protocole |
|---|-----------|-----------|
| 1 | Edge vs Cloud | Comparer TFJS (browser) vs serveur lourd pour classification |
| 2 | Avant/Après LoRA | Phrases générées par base Llama vs fine-tuned |
| 3 | Faux positifs | Taux de classification erronée sur bruits ambiants |
| 4 | Personnalités | Même intention → prompts différents → styles de phrases |
| 5 | Quantization | FP32 vs INT8 → précision vs taille |
| 6 | Latence | Temps total de la boucle : capture → traduction → affichage |

---

## Dépendances

### Python
```
librosa>=0.10.0
noisereduce>=3.0.0
webrtcvad>=2.0.10
tensorflow>=2.15.0
transformers>=4.40.0
peft>=0.10.0
unsloth>=2024.5
llama-cpp-python>=0.2.0
fastapi>=0.110.0
uvicorn>=0.28.0
numpy>=1.24.0
```

### Frontend
```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "@tensorflow/tfjs": "^4.17.0",
    "@tensorflow/tfjs-model": "^4.17.0"
  },
  "devDependencies": {
    "vite": "^5.2.0",
    "@vitejs/plugin-react": "^4.2.0"
  }
}
```

---

## Livrables

1. **Code source** — dépôt GitHub avec suivi des contributions
2. **Vidéo 10 min** — ludique + technique, en anglais
3. **Démo fonctionnelle** — interface web avec pipeline complet
4. **Rapport** — documentation des expériences et trade-offs
