# Task Distribution (Team of 4 Students)

This document details the specific responsibilities for each team member regarding the **Pet Translation Device** project, precisely aligned with the technical stack outlined in `agents_ia.md`.

## Student 1: Mohamed MELLOUK - Audio Signal Processing and Pipeline
**Role:** Responsible for the capture, cleaning, and extraction of raw audio features.
*   **Target Directories:** `backend/audio/`, `training/classification/data/`
*   **Technologies:** Python, `librosa`, `noisereduce`, `webrtcvad`
*   **Tasks:**
    *   **Data Collection:** Gather and prepare audio datasets (ESC-50, subset of AudioSet, and custom recordings).
    *   **Preprocessing Pipeline:** Implement ambient noise reduction (`noisereduce`) and Voice Activity Detection (`webrtcvad`).
    *   **Feature Extraction:** Convert raw audio waves into Mel-spectrograms and MFCCs using `librosa` to feed the classification model.

## Student 2: Anas ISARTI - Classification Models (MobileNetV2 to TFJS) & Scientific Evaluation
**Role:** Responsible for the analytical "brain" that translates raw audio features into an emotional category directly in the browser, and scientifically proving data limitations.
*   **Target Directories:** `training/classification/`, `frontend/public/model/`, `frontend/src/lib/`
*   **Technologies:** TensorFlow/Keras, TensorFlow.js, MobileNetV2, SMOTE
*   **Tasks:**
    *   **Model Fine-Tuning:** Fine-tune a frozen MobileNetV2 architecture with a small trained head to avoid overfitting on tiny datasets.
    *   **Scientific Evaluation:** Implement group-aware cross-validation (split by individual cat) to prevent data leakage. Test 5 independent levers (AST backbone, focal loss, etc.) to establish the "Data Ceiling".
    *   **Browser Integration:** Export the full model directly to TensorFlow.js (without .tflite) and verify output is mathematically identical to Python (diff=0).

## Student 3: Abir ISLAM - LLM Fine-Tuning & Inference (Llama 3.2 + LoRA)
**Role:** Responsible for generating human-like "translations" with customized pet personalities.
*   **Target Directories:** `training/llm/`, `backend/llm/`
*   **Technologies:** Llama 3.2 (1B/3B), `unsloth`, `llama.cpp` (Python bindings)
*   **Tasks:**
    *   **Dataset Generation:** Create a synthetic dataset (`synthetic_dataset.json`) of 5000+ pairs mapping {intent + context + personality} to natural language sentences.
    *   **Model Fine-Tuning:** Execute LoRA fine-tuning on Llama 3.2 using `unsloth` and `peft`.
    *   **Deployment & Prompting:** Convert the fine-tuned model to GGUF (Q4_K_M) format for efficient serving via `llama.cpp`. Handle prompt engineering to establish personalities (e.g., haughty cat, excited dog).

## Student 4: Amine KHALIL - Frontend Web, UI/UX & Full Integration
**Role:** Responsible for user experience, API routing, and the final end-to-end communication of the pipeline.
*   **Target Directories:** `frontend/`, `backend/main.py`
*   **Technologies:** React + Vite, FastAPI, MediaRecorder API, Tailwind/CSS
*   **Tasks:**
    *   **Frontend UI & UX:** Build a modern React application featuring a dark/light theme, three-column layout, probability bars, and a 3-second perception delay. Handle silent audio via energy-based VAD (RMS < 0.015).
    *   **Pet Selector & State:** Implement the pet selector where personality is handled entirely server-side per species.
    *   **Edge Integration & Deployment:** Convert Keras model to GraphModel, resolve TFJS preprocessing bugs (mel scale, filter norm, power vs magnitude), and enforce explicit WebGL backend loading for macOS compatibility.
    *   **Real-Time Feedback & Fallback:** Compute class probabilities in-browser via TFJS and display them alongside the remote LLM translation. Include a mock LLM fallback to ensure demo reliability.
